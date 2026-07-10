# Phase 3a Task 3 — Sharded Oracle labeling design

**Status:** Proposed (supersedes the radius-1→3 CLI-arg plan in `docs/Phase 3a.md` §5)
**Date:** 2026-07-08

## 0. Why this design

The original Task 3 plan ("re-run `generate_gnn_training_data.py --num-subgraphs 4000
--subgraph-radius 3`") is empirically impossible on the dense 5002-ep DialogSum corpus:

| radius | nodes | edges | JSON | one-call labelable? |
|--------|-------|-------|------|---------------------|
| 1 | 9 | 12 | 1.4 KB | ✅ (but cluster task empty — 1 episode/subgraph) |
| 2 | 4,944 | 22,998 | 1.9 MB | ❌ output blows past `oracle_max_tokens=32768` |
| 3 | 10,680 | 45,702 | 4.0 MB | ❌ ≈ the whole connected graph |

The bottleneck is **one Oracle call's output budget**, not the GNN (GAT over 10K nodes ×
45K edges is a small graph for PyG on a GPU). So we **decouple subgraph size from Oracle
call size**: keep the full radius-3 subgraph for what the GNN trains on, and shard the
*labeling* across many calls so each call's output fits in 32K. The GNN then trains on
real full subgraphs with complete labels — no windowing, no sampling artifact.

This is the design that matches the **lifelong-growth deployment fact**: production
starts sparse and densifies over ~5 years. Batching trains on the *real* extraction at
every scale (early small full subgraph → late large full subgraph); train/deploy
distributions match by construction. Node-budgeting windows away structure; a sparse
training corpus trains on a regime the system grows out of; batching keeps everything
real.

## 1. The shard unit

A **shard** = `(subgraph, subset_to_label, context_window)`:

- `subset_to_label`: the nodes (or edges, or pairs) THIS call must score — ≤
  `shard_size` items (default **500 nodes** for salience; see per-head strategy below).
- `context_window`: the prompt content sent with the shard. **Local context**, not the
  full 4 MB:
  - the center node (id + hydrated content if episode),
  - the shard's nodes (id, type, hydrated episode content: summary/entities/topics/tones/
    decisions/timestamp),
  - the **induced edges** among shard nodes + edges between shard nodes and the center,
  - a one-line **global summary**: `{total_nodes, total_edges, top_shared_entities,
    top_shared_topics}` so the Oracle has global awareness without 4 MB.
- Instruction: "score ONLY these N nodes/pairs by salience relative to the center; do not
  invent nodes outside this set."

Local context is correct for the per-node/per-edge heads because their judgments are
*local* (a node's salience-to-center is determined by its neighborhood, not the whole
graph). The global summary line handles the one head that wants global awareness
(anomaly's ISOLATED_CLUSTER) — but see §3: most anomaly types are structural and don't
need the Oracle at all.

## 2. Per-head strategy (4 Oracle-labelable + 1 self-supervised)

### Salience — per-node, sharded
- Shard the subgraph's nodes into chunks of `shard_size` (default 500). ~22 calls per
  radius-3 subgraph.
- Each call returns `{"node_scores": {id: {"salience": x}}, ...}` for the shard's nodes.
- **Edge scores computed in code** from node scores (edge salience = mean of endpoint node
  salience) — a reasonable structural proxy that halves the shard count. Documented; the
  head trains on node MSE anyway (`SalienceHead.loss` is per-node).
- **Recombination:** merge all shards' `node_scores` into one dict; build `edge_scores`
  in code. Matches the existing `salience_labels.jsonl` schema (`{"node_scores": {...},
  "edge_scores": {...}}`) so `train_gnn.py` is unchanged.

### Anomaly — injection-based, no Oracle for the head; 9 rule-detectable types

The DialogSum corpus is anomaly-free by construction (the encoder always writes well-formed
edges/timestamps/follows, and distinct-conversation summaries don't contradict or
duplicate). The PoC's only "positives" were a constant `madeBy` schema artifact (`madeBy`
is declared in the ontology but never written → 100% of decisions are "orphan"). So the
corpus is **all-negative for every meaningful anomaly type** → a multi-label BCE head can't
be trained on it (collapses to "predict 0") and can't be F1-evaluated (no positives).

The standard fix for anomaly detection on clean data: **corrupt clean subgraphs to create
positive examples.** Each corruption is injected in code; the structural detector labels it
deterministically — so labels are exact and **zero Oracle calls** are needed for the head.
The head learns "what does each corruption look like structurally"; at deploy the rule
detector is the ground-truth backstop, the head is a cheap pre-filter that flags
candidates for the rules (and for the Bonsai decider — see §2.5).

The refined 9-type taxonomy (drops the `madeBy`-artifact orphan and the vague
"contradiction"; concretizes the rest to the real lifelong-memory failure modes):

| # | type | real-world cause | inject (code) | rule detector (deploy) | label on |
|---|------|------------------|---------------|------------------------|----------|
| 1 | CONTRADICTORY_STATE | facts change over time | plant 2 live `state` on one `E:` | group live `state` by entity, >1 distinct value | entity node |
| 2 | DUPLICATE_EPISODE | re-import / cross-device sync | clone an `ep_` (+content) | pairwise sim > threshold (token-overlap / embedding cosine) | ep node |
| 3 | DUPLICATE_DECISION | re-decide / re-ingest | clone a `D:` | pairwise sim > threshold | D node |
| 4 | ORPHAN_DECISION | partial ingest (encoder crash) | delete `has_decision` from a `D:` | `D:` degree 0 on link preds | D node |
| 5 | DETACHED_EPISODE | partial ingest | strip an `ep_`'s link edges | `ep_` degree 0 on link preds | ep node |
| 6 | BROKEN_FOLLOWS | edits / long history | rewire/delete a `follows` | target-exists + single-parent in follows DAG + temporal order | ep node |
| 7 | TYPE_VIOLATION | ontology drift over years | insert a wrong-kind edge | pred declared domain/range vs endpoint kinds | both endpoints |
| 8 | ISOLATED_CLUSTER | separate life-domain (often legitimate) | detach a component | connected components (excluding seeded `subClassOf`) | all nodes in component |
| 9 | STALE_ABSTRACTION | consolidator re-ingest (dogfood 3a's own writes) | point `M:` `abstracts` at a dead ep | `abstracts` target exists | M node |

New modules: `src/gnn/anomaly_rules.py` (the pure, unit-testable rule detectors — also the
deploy backstop) and `src/gnn/anomaly_injector.py` (corrupts a clean subgraph and emits the
exact labels the rule detector would produce on the corruption). The injector + rules
together form a closed loop: inject → rule-detect → label, no Oracle, run over the full
10K-node subgraph in code. The head trains on these labels (with the partial-label mask of
§7); at deploy, the rule detector runs as ground truth and the head pre-filters.

**IDENTITY_DRIFT is a rule-flag, not a head label.** "One node name, two different
referents" (E:Alice a person at ep_50, E:Alice a project at ep_5000) is genuinely semantic —
no rule can decide it, and the only clean signal (type-level `subClassOf` incompatibility) is
too rare to train on, while the naive "disjoint topic neighborhoods" heuristic over-fires on
every legitimately multifaceted entity (a person who is a coder AND a parent). So
IDENTITY_DRIFT is emitted as a **flag-for-review** by `anomaly_rules.py` (the disjoint-
neighborhood heuristic, deliberately over-firing) and routed to the Bonsai decider (§2.5),
NOT trained as a head label. A real user's graph will produce genuine cases occasionally;
those get Bonsai's retrieve-then-decide treatment, not an auto-fix.

### Cluster (DiffPool) — self-supervised, optional weak Oracle supervision
- **Primary:** self-supervised. `DiffPoolHead.loss` (per-node entropy + cluster-link
  preservation + cluster-balance) trains from graph structure — no Oracle labels needed.
  The cluster-balance term is the anti-collapse fix: without it the loss's global minimum
  is the trivial "every node -> one cluster" solution (entropy 0, link -log(1)=0), so the
  val metric is clusters-used (higher = not collapsed), not per-node entropy (which
  rewarded collapse). This is how DiffPool is normally trained.
- **Optional weak supervision:** one Oracle call over an **episode-only context** (just
  the episode nodes + their shared entities/topics/timestamps, not all 10K nodes — fits
  in one call) using `gnn_cluster_prompt`. This gives a weak "which episodes should
  cluster" signal to seed the assignment. Gated by `--oracle-cluster-supervision` (off by
  default; the risk register already listed topic-co-occurrence as the cluster fallback).
- No sharding needed — episode-only context is small.

### 2.5 anomaly_decision_pairs — Oracle distillation data for Bonsai (Task 3 extension)

The GNN anomaly head + rule detectors handle the **structural** anomalies. The
**semantic** decision — "given a flagged anomaly, what should the system DO about it?" —
is Bonsai's job (the small, local, always-on, $0 model), and Bonsai needs app-specific
training to make those decisions in Hippo's action vocabulary. This is the **existing
Phase-1d Bonsai-training pattern** (`data/training/bonsai/query_planning_pairs.jsonl` /
`relation_extraction_pairs.jsonl`: Oracle generates training pairs, Bonsai fine-tuned on
them) applied to one new pair type: **anomaly-decision pairs**.

A pair record: `{flagged_entity, retrieved_context, anomaly_type, decision, action,
reasoning}` where `decision ∈ {fix, ask_user, dismiss}`, `action` is the Hippo-specific
operation (split the node / supersede the state / re-link / ask a clarifying question /
dismiss as legitimate), and `reasoning` is the Oracle's chain.

**Critical grounding — retrieve-then-prompt.** The Oracle's demonstration must be made
from the *same context Bonsai will have at deploy* — so before the Oracle call, pull what's
known about the flagged entity from the DB/STM (its episodes, states, topics, neighbors)
and bake that retrieval into the prompt. Otherwise the Oracle demonstrates a decision Bonsai
can't reproduce. The retrieval step is small, reusable, and lives alongside the consolidator's
existing verifier hook.

**This is Oracle-generated *training data*, not a gatekeeper.** The Oracle is the teacher
that demonstrates decisions on injected anomalies (we know the ground truth — we planted the
drift — so the Oracle's "correct" decision is checkable); Bonsai is the student fine-tuned on
those demonstrations; the Oracle is not in the deploy loop. Cost: more Oracle calls (one
per flagged candidate, with retrieval context), $0 and cached, generated as part of Task 3.

**Staged Bonsai path (the non-over-engineered way to "prepare Bonsai"):**
1. **3a core** — GNN anomaly head (9 types) + IDENTITY_DRIFT rule-flag in `anomaly_rules.py`.
   The flagging exists; nothing decides yet.
2. **Best-effort baseline** — Bonsai **zero-shot** on the flags via the existing `verifier`
   hook in `consolidate.py` (the decider pattern is already wired there). An 8B model with a
   good action-space prompt + retrieval may be "good enough for now" — and this *measures*
   how good before any fine-tuning investment.
3. **Gated refinement** — generate `anomaly_decision_pairs` now (cheap, in Task 3, cached),
   and run the Bonsai LoRA fine-tune **only if** the zero-shot baseline underperforms on
   injected-drift decisions. If zero-shot is fine, the GPU fine-tune run is saved; if not,
   the data + the evidence are both ready. The fine-tune is a separate gated step (its own
   GPU run + eval), NOT bundled into the 3a cold-start commit.

### Link-pred — candidate-pair shards
- Generate candidate non-edges (sampled same-kind pairs, per `consolidate.
  _sample_candidate_pairs` logic) capped at `max_candidate_pairs` (default 500).
- Shard the candidate pairs into chunks of `shard_size` (default 500 → ~1 call).
- Oracle returns `predicted_edges` + `negative_edges` for the shard.
- Recombination: concat. Matches `link_prediction_labels.jsonl` schema.

### Ontology — entity/topic pair shards
- Candidate `subClassOf` pairs = entity∪topic pairs in the subgraph (small — entities +
  topics are a fraction of 10K nodes). Cap at `max_candidate_pairs` (default 500).
- ~1–2 Oracle calls (the existing `gnn_ontology_prompt` over the candidate pairs + the
  `SEED_ONTOLOGY` context).
- Recombination: concat `suggested_edges` + `misclassified`.

### Cluster (DiffPool) — self-supervised, optional weak Oracle supervision
- **Primary:** self-supervised. `DiffPoolHead.loss` (per-node entropy + cluster-link
  preservation + cluster-balance) trains from graph structure — no Oracle labels needed.
  The cluster-balance term is the anti-collapse fix: without it the loss's global minimum
  is the trivial "every node -> one cluster" solution (entropy 0, link -log(1)=0), so the
  val metric is clusters-used (higher = not collapsed), not per-node entropy (which
  rewarded collapse). This is how DiffPool is normally trained.
- **Optional weak supervision:** one Oracle call over an **episode-only context** (just
  the episode nodes + their shared entities/topics/timestamps, not all 10K nodes — fits
  in one call) using `gnn_cluster_prompt`. This gives a weak "which episodes should
  cluster" signal to seed the assignment. Gated by `--oracle-cluster-supervision` (off by
  default; the risk register already listed topic-co-occurrence as the cluster fallback).
- No sharding needed — episode-only context is small.

## 3. Per-subgraph Oracle call budget (radius 3, ~10K nodes)

| head | calls/subgraph | notes |
|------|----------------|-------|
| salience | ~22 | 500-node shards; edge scores computed in code from endpoints |
| anomaly (head) | **0** | injection + rule-detect in code; no Oracle for the head |
| anomaly_decision_pairs (Bonsai data) | ~1–3 | one Oracle call per flagged candidate w/ retrieval context |
| link-pred | ~1–2 | 500 candidate pairs/shard |
| ontology | ~1–2 | entity/topic pairs (small) |
| cluster | 0–1 | self-supervised; optional 1 episode-only call |
| **total** | **~25–30** | anomaly head training is Oracle-free; Oracle calls are for salience + link + ontology + Bonsai distillation data |

**Number of subgraphs drops at radius 3.** "4000 subgraphs" was a radius-1 target (9
nodes each, highly local). At radius 3 each subgraph ≈ the whole 5002-ep graph, so 4000
would be ~4000 near-identical copies. For full-graph subgraphs, **~300 distinct centers**
gives full coverage (each subgraph already spans most of the graph). Default
`--num-subgraphs 300` at radius 3; tunable.

**Wall-clock:** ~300 subgraphs × ~29 calls ≈ 8.7K calls. At `--oracle-max-workers 8` and
~3 s/call (reasoning model) → ~3.4K s ≈ **~1 hour**. At ~8 s/call → ~3 hours. $0 (Ollama
cloud credits). The risk is Ollama-cloud throughput at high concurrency — see §6 probe.

## 4. Recombination + schema preservation

Each subgraph produces ONE record per task in the existing JSONL schema (so `train_gnn.py`
and `validators.py` are unchanged):

```jsonl
{"subgraph_id": "ep_000001", "labels": {"node_scores": {...}, "edge_scores": {...}}, "cost": ...}
```

`labels` is the recombined merge of all shard outputs (salience), the structural+semantic
merge (anomaly), or the concat (link/ontology). The recombination pass runs after
`run_batches` per task, grouping shard records by `subgraph_id`. Shards that fail/parse-
error are logged and skipped (partial labels are still usable — mask unlabeled nodes in
the training loss; see §7).

## 5. Reusing the existing machinery

`run_batches(oracle, items, build_prompt, to_record, ...)` already drives concurrency,
checkpointing, and the on-disk prompt-cache. Sharding fits as **`item = shard`**:

- `build_prompt(shard, idx)` renders the local-context shard prompt.
- `to_record(shard, result, idx)` returns a **shard record** tagged with
  `(subgraph_id, shard_idx)`.
- After `run_batches`, a `recombine(records)` pass groups shard records by `subgraph_id`
  and merges into the final per-subgraph JSONL record.
- The on-disk Oracle cache (`.oracle_cache.json`, keyed by prompt hash) makes resumes
  free — an interrupted run re-sends no completed shard.
- Checkpoint granularity stays per-task (one `{task}_checkpoint.json` holding shard
  records); `--resume` skips completed shards.

New module: `src/gnn/sharded_labeling.py` — shard construction, local-context prompt
builders, recombination, the structural anomaly detector, candidate-pair samplers. The
generator script (`scripts/generate_gnn_training_data.py`) gets a `--sharded` mode (on
by default when radius ≥ 2 or subgraph node count > `shard_threshold`, default 200) that
routes through `sharded_labeling.py`; radius 1 small subgraphs keep the old one-call path.

## 6. Concurrency probe (before the full run)

A 5-minute probe before committing to 8.7K calls: send 32 concurrent `gnn_salience_prompt`
shard calls to the local Ollama endpoint at `--oracle-max-workers {1,4,8,16}` and measure
throughput + error rate. Confirms the cloud backend sustains concurrency before the
~1-hour run; picks the highest workers that doesn't error/throttle. Logged to
`data/pod_runs/phase3a/concurrency_probe.json`.

## 7. Partial-label handling + training-loss masking

Not every node will get a salience label (shard parse failures, truncated shards). The
training loss must **mask unlabeled nodes** rather than treat them as salience=0:

- `SalienceHead.loss` gains an optional `mask` arg; only labeled nodes contribute to MSE.
- The loader emits a per-node `salience_label` (NaN where unlabeled); the trainer masks.
- Same pattern for anomaly (per-node label vector, mask unlabeled), link-pred (only
  labeled pairs), ontology (only labeled pairs).
- A subgraph with <`min_labeled_fraction` (default 0.5) of labeled nodes is dropped from
  training (logged), so the model never trains on near-empty labels.

This is the honest way to handle partial labels — no silent 0-fill that would teach the
head "low salience = everything I couldn't label."

## 8. Acceptance criteria

- One radius-3 subgraph labeled via sharding: ≤ ~30 Oracle calls, salience/link/ontology
  labels present (salience covers ≥80% of nodes; link-pred has pos+neg).
- Anomaly labels produced by **injection + rule detection, zero Oracle**: the injector
  corrupts a clean subgraph; `anomaly_rules.py` labels each corruption exactly; the 9 types
  are all represented where applicable; IDENTITY_DRIFT emitted as a review-flag (not a head
  label).
- `anomaly_decision_pairs.jsonl` validates (Bonsai pair shape): each pair carries the
  retrieved context + a fix/ask/dismiss decision + Hippo action + reasoning.
- `validators.validate_gnn` passes on the sharded output (GNN label schema unchanged).
- A 300-subgraph run completes in ≤ ~3 hours at the probed concurrency; produces
  `quality_report.json` with per-head label counts + Oracle call counts.
- `train_gnn.py` (Task 4a) trains on the sharded labels + injected-anomaly labels with masked
  losses and converges.
- `anomaly_rules.py` unit-tested against synthetic orphans/gaps/type-violations/isolated
  clusters/contradictory-state/duplicates/stale-abstraction; the injector round-trips
  (inject → rule-detect recovers exactly what was planted).
- Bonsai zero-shot baseline on injected IDENTITY_DRIFT flags measured (decision accuracy
  vs planted ground truth) — the gate for whether the LoRA fine-tune runs.

## 9. What this does NOT do

- Not OGB pretraining — that's a separate pod-only lever (ADR 008); sharded labels are
  the direct-training path. OGB-transfer can still run later and be compared.
- Not SEAL subgraph features for link-pred — GAE dot-product cold start (ADR 008); SEAL
  is a later quality lever.
- Not full DiffPool with successive pooled layers — the simplified cold-start version
  (ADR 008).
- Not a one-shot 4000-subgraph run — 300 full-graph subgraphs at radius 3 gives full
  coverage with far less redundancy.
- Not a Bonsai fine-tune bundled into 3a — `anomaly_decision_pairs` data is generated in
  Task 3, but the LoRA fine-tune is a gated step after the zero-shot Bonsai baseline is
  measured. The Oracle is the teacher that prepares Bonsai; it is not a deploy-time
  gatekeeper.
- Not IDENTITY_DRIFT as a learned head label — it's a rule-flag for Bonsai review, because
  its only clean detector (type-level incompatibility) is too rare to train on and its naive
  detector (disjoint neighborhoods) over-fires on legitimately multifaceted entities.