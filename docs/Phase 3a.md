# Phase 3a: GNN Consolidator — Implementation Plan

**Ponder Engine · Architecture v2.0 · July 2026**

> **This is a design document, not an implementation.** It mirrors the §0 alignment-pass
> pattern of `docs/Phase 2b.md` / `docs/Phase 2c.md`: a frank corrections table measured
> against the **real** `src/` codebase, then the corrected tasks. No 3a code, GPU run, or
> budget is spent by writing this doc. The architectural choices marked *recommended* are
> defaults from the alignment pass — the user may adjust them before implementation begins.

> **Terminology:** "Phase 3a" here is the body of work that `docs/Ponder Engine Phases.md`
> §339-380 calls "Phase 3a: GNN Consolidator." This doc standardizes on that name.

---

## 0. Alignment Notes — doc vs. the real codebase

The `Ponder Engine Phases.md` Phase 3a spec is high-level. This section records every
correction needed to ground it in the real codebase, the way `docs/Phase 2b.md` §0 does.
Implementation follows the **Reality** column.

| Spec / draft said | Reality (this doc corrects to) |
|---|---|
| "5 task-specific heads, trained on Oracle-labeled memory graphs" (build the label pipeline in 3a) | **Phase 1d already shipped the full label pipeline.** `scripts/generate_gnn_training_data.py` + the 5 `gnn_*_prompt` functions in `src/training/prompts.py:21-137` + `src/training/validators.py` produce `salience_labels.jsonl`, `cluster_labels.jsonl`, `link_prediction_labels.jsonl`, `anomaly_labels.jsonl`, `ontology_labels.jsonl`, validated against a 3-subgraph proof-of-concept (commit 6bcb491). So 3a is **regenerate-at-scale + extend + consume**, not build-from-scratch. |
| "Oracle-generated salience labels + link prediction examples" (assume plentiful training data) | **Data on disk is ~0.075% of target.** `data/training/gnn/` has 3 labeled subgraphs (radius=1) per task vs the `docs/Phase 1d.md` target of **4,000+ subgraphs × 5 tasks**. `cluster_labels.jsonl` is **all empty** (zero positive examples at radius 1). Re-run the generator at `--num-subgraphs 4000 --subgraph-radius 3` (the extractor `OracleLabelingPipeline.extract_subgraph` already defaults to radius 3 — `src/training/oracle_labeling.py:112`; the PoC just passed `--subgraph-radius 1`). The validator at `scripts/validate_training_data.py` already enforces ≥1000/task. |
| "Link prediction (GAE/SEAL): discover implicit edges" (positive edges only) | **`link_prediction_labels.jsonl` is positive-only.** `gnn_link_prediction_prompt` (`prompts.py:72`) asks only for edges that "SHOULD exist but are not in the graph." SEAL/GAE training needs **explicit negative edges** (non-edges that should remain non-edges). Extend the prompt to also emit `negative_edges`; add random-vs-hard negative sampling in the loader (Task 3). |
| "GNN backbone: GAT pre-trained on OGB benchmarks" (assume OGB is wired) | **No `torch_geometric`, `dgl`, or `ogb` is installed** anywhere in `src/`, `scripts/`, or `pyproject.toml`. The only ML library present is `torch` (transitive via `gliner2[local]`). 3a must add `torch_geometric` + `ogb`. **Decision required** (presented, not assumed — see §1.3): (A) pretrain a GAT on OGB (e.g. `ogbn-arxiv`) then transfer to the memory graph [spec's intent, *recommended*], (B) load a published pretrained GAT checkpoint, or (C) train directly on the Oracle-labeled memory graphs [cheap fallback if (A) underperforms on the small, typed graph]. |
| "Abstracted memories stored in HBTrie; linked to source episodes via `abstracts` edges; source episodes marked abstracted" (assume the storage exists) | **Entirely greenfield.** There is no `abstracts` edge, no `SemanticMemory` node kind, and no consolidator code in `src/`. The `supersedes` predicate is **declared** in the ontology (`src/memory/ontology.py:79`, domain/range `Episode`, comment "reconsolidation") but is **never written** by any code. `Episode.consolidation_window_start` exists (`src/memory/episode.py:50`, `Optional[str] = None`) and is written at `src/memory/store.py:110-111` only `if episode.consolidation_window_start:` — i.e. **never set** by anyone today. The DiffPool head's output (semantic memories) has nowhere to land yet. Storage is a 3a deliverable (Task 5). |
| "GNN scores all nodes/edges → detects clusters → abstracts → predicts → detects anomalies → refines ontology → prunes low-salience (archives, never deletes)" (assume a consolidation loop exists) | **No scheduler or consolidation loop exists.** `consolidate.py` + a nightly entrypoint are 3a deliverables (Task 6). The "archive, never delete" rule needs an archive subtree in WaveDB (greenfield but simple). The loop must be **dry-run by default** with an explicit `--apply` gate — it mutates the live graph. |
| "Bonsai verifications → training examples for GNN anomaly detector" (>70% of predicted edges validated, line 583) | The 1d Oracle client (`src/training/oracle_labeling.py`) already talks to the local Bonsai 8B server. Reuse it for the consolidation loop's edge/anomaly verification (Task 6). No new LLM client. |
| "Salience scoring (GAT): learned structural importance" (assume a clean salience slot) | **The slot is real and pre-flagged for replacement.** `src/retrieval/graph_traversal.py:389-393` and `:418-419` score retrieval with a **heuristic mention-count** salience on **entities only** (`_W_ENTITY * (0.5 + 0.5 * salience)` via `store.get_entity_salience`), with comments that literally say "Phase 3 GNN salience replaces this heuristic." Per-episode `Episode.salience` (default 0.5, **never computed**) lives at `content/ep/{eid}/salience` — the placeholder the GAT head fills. So the GAT head has a concrete code site it supersedes, and the 1c heuristic is a **cold-start prior** (a weak-supervision feature) until Oracle labels are regenerated. |
| "EXPAND frequency feeds GNN salience" (inherited from Phase 2c §15 handoff) | **BLOCKER — the signal is not durable.** `src/subconscious/presentation_gate.py:167-168`: `outcome_buffer` and `override_buffer` are in-memory `ReplayBuffer`s (deque-backed), **not persisted**, and `PonderOrchestrator.record_outcome` is **never auto-invoked by `query()`** — the caller must call it, and nothing in the live path does. So the EXPAND-frequency / unused-primary / user-satisfaction signal the GAT head is supposed to learn from does not durably exist across sessions. 3a must resolve this (Task 7): either (a) add a persistence layer for the buffers + auto-invoke `record_outcome` [*recommended*], or (b) scope the first GNN slice to Oracle salience labels + the 1c heuristic prior only and defer EXPAND-frequency to 3a.1. **Presented as a decision, not a silent assumption.** |
| "Training hardware: RTX 4090, Vast.ai spot, 48 hours, ~$13" (lines 859-860) | **The user's actual GPU path is RunPod dashboard pods**, not Vast.ai. Per memory `runpod-api-pods-dead-on-arrival`: the RunPod MCP `create-pod` SSH proxy is broken (container launches, sshd listens, but the public SSH port stays "Connection refused"); the user creates pods via the **dashboard** and pastes the ssh string. The ~$13 / 48h / RTX 4090 budget is the **anchor**; the provider is RunPod. The disk-wipe rule applies (memory `runpod-community-pod-disk-wipe`): **SCP checkpoints local before stopping the pod** — container disk is ephemeral across stop/start on both community and secure cloud. |
| "GNN (~200M params)" (cost table, line 885) | The 200M figure is spec-side. Size the model to the **real** architecture (GAT backbone + 5 heads; layer count tuned to a graph of ~10⁴-10⁵ nodes). A 200M GNN may be oversized for the memory graph; reconcile in Task 2 and record the actual param count. |
| (Implicit) graph nodes have vector features to feed the GNN | **They do not.** WaveDB graph nodes are typed keys (`ep_NNNNNN`, `S:NNNN`, `U:{user}`, `E:{entity}`, `T:{topic}`, `A:{tone}`, `D:{decision}`) with no inherent vector features. 3a must define the **node-feature pipeline** (Task 1): episodes → 384-dim embedder vector (the same `Embedder` Protocol the subconscious package uses); entity/topic/tone → type-onehot + optional embedding; user/decision → onehot. Documented explicitly, not assumed. |
| (Implicit) a graph loader reads WaveDB → tensors | **None exists.** `OracleLabelingPipeline.extract_subgraph` (`oracle_labeling.py:112`) does BFS and emits a JSON `{center, radius, nodes, edges}` — reuse its BFS, extend it (or wrap it) to emit `torch_geometric.data.Data` with `edge_index`, `x`, `node_kind`, `edge_attr` (predicate onehot). One loader, used by both training and the nightly loop. |
| (None) — two GNN-label prompt libraries that disagree | **Schema mismatch to clean up.** `src/training/oracle_labeling.py:56-69` defines `ORACLE_GNN_LABELING_PROMPT` — a **single-label** schema ("relevance / salience / should_recall per node") with a comment "Not invoked in 1b." The generator actually uses the **5-prompt** library in `prompts.py:21-137` (per-task schemas: `node_scores`/`edge_scores`, `clusters`, `predicted_edges`, `anomalies`, `suggested_edges`). The former is **dead and contradictory**. Task 3 removes it (or reconciles it) so there is one source of truth. |
| Predicates use camelCase (ontology registry style) | **The graph uses snake_case predicates** despite `src/memory/ontology.py` property names: `has_entity, in_episode, has_topic, has_tone, has_decision, follows, state, validity_start, has_session, has_episode, in_session, at_time, started_at, follows_session, ended_at, subClassOf` + open-ended Bonsai relations. The loader and all head code must key on the **snake_case** graph predicates, not the registry's camelCase. |
| The conversational + code ontology is a doc artifact | **It is materialized as real `subClassOf` triples in the graph.** `src/memory/store.py:_seed_ontology` (`:51`, gated by the `content/system/ontology_seeded` marker at `:62`) writes every `(child, subClassOf, parent)` pair from the merged `SEED_ONTOLOGY` (`ontology.py`) at first DB init — the conversational taxonomy unioned with `docs/Code_Ontology.md`. So the **ontology-refinement head has a real `subClassOf` edge set to refine**, not a paper ontology. |

### What does **not** exist yet (greenfield for 3a)

`src/gnn/` (the whole package), the `abstracts` / `supersedes` edge writers, the archive
subtree, the consolidation loop, the nightly scheduler, the PyG graph loader, the
node-feature pipeline, the GAT model + 5 heads, `scripts/train_gnn.py`,
`scripts/run_consolidation.py`, `docs/adr/008` + `009`. Confirmed by grep: no
`torch_geometric`/`dgl`/`ogb`/`GAT`/`DiffPool`/`GAE`/`SEAL` symbols in `src/` or `scripts/`.

### What **already** exists (reused, not rebuilt)

- The 5-prompt Oracle label library + generator + validator (Phase 1d, commit 6bcb491).
- `OracleLabelingPipeline.extract_subgraph` BFS (radius-configurable, defaults to 3).
- The local Bonsai 8B Oracle client (`src/training/oracle_labeling.py`) — reused for
  consolidation-loop verification.
- The 1c heuristic entity salience + `store.get_entity_salience` / `write_entity_salience_batch`
  — cold-start prior for the GAT head.
- The materialized `subClassOf` ontology triples — substrate for the ontology-refinement head.
- The `WorkingMemoryState` / `JGSSnapshot` format (`src/subconscious/state_serializer.py`)
  — the GNN reads WM state to prioritize consolidating what's "in awareness."
- `scripts/train_backbone.py` (the 2a pod entrypoint) — the template `scripts/train_gnn.py`
  mirrors (float32, checkpoint-to-`data/pod_runs/`, SCP-before-stop).

---

## 1. Overview

Phase 3a deploys the **GNN Consolidator**: a static, stateless graph neural network with five
task-specific heads, trained on Oracle-labeled memory graphs, run in a nightly "dream-state"
loop that abstracts semantic memories, predicts and verifies implicit edges, flags structural
anomalies, refines the ontology, and prunes low-salience edges (archives, never deletes).

| Subsystem | What it does | Why it matters |
|---|---|---|
| **Graph loader** | WaveDB SPO/POS graph → PyG `Data` (edge_index + node features), BFS radius-3 subgraphs | The GNN eats tensors, not triples; one loader serves training + the nightly loop |
| **GNN model + 5 heads** | GAT backbone + salience / diffpool / linkpred / anomaly / ontology heads | Learned structural importance replaces the 1c mention-count heuristic |
| **Regenerated Oracle labels** | 4,000+ subgraphs × 5 tasks, radius 3, **with negative edges** for link prediction | The 1d PoC (3 subgraphs) is a schema proof, not training data |
| **Semantic-memory storage** | `abstracts` + `supersedes` edges, abstracted-episode marking, archive subtree | The DiffPool head's output needs somewhere to land |
| **Consolidation loop** | Nightly: score → cluster → abstract → predict → verify → anomaly → ontology → prune | The "dream state" — memory compaction overnight |
| **EXPAND-frequency durability** | Persist the 2c outcome/override buffers so they survive sessions | The salience signal 2c §15 promised but never made durable |

### 1.1 Prerequisites (status)

- [x] Phase 1d — 5 label prompts + generator + validator + 3-subgraph PoC (shipped, 6bcb491)
- [x] Phase 1c — heuristic entity salience (cold-start prior for the GAT head)
- [x] Phase 2c — `WorkingMemoryState` / `JGSSnapshot` format stable + serializable
- [x] 2a pod entrypoint pattern (`scripts/train_backbone.py`) to mirror
- [ ] GNN deps (`torch_geometric`, `ogb`) — **added by 3a** (Task 8)
- [ ] Oracle labels at scale (4,000 subgraphs, radius 3, +negatives) — **regenerated by 3a** (Task 3)
- [ ] EXPAND-frequency durability — **BLOCKER, resolved by 3a** (Task 7)

### 1.2 What 3a does NOT do (honest scope)

- **No temporal-continuity GNN.** Per the spec's note (`Ponder Engine Phases.md` §378): the
  initial GNN is a **stateless function**. SSM-augmented instances with temporal memory of
  past consolidation decisions are added **only after failure modes are observed** (cluster
  flapping, prediction miscalibration, false-positive anomalies, ontology oscillation). 3a
  ships the stateless GNN; temporal continuity is a later, failure-driven addition — not
  premature optimization.
- **No cross-document deduplication** in the first slice (spec §373-376). It depends on
  document-section features not yet defined; deferred to a 3a.x once the core loop is stable.
- **No replacement of the 1c heuristic at runtime until the GAT head validates.** The heuristic
  stays the live retrieval scorer until the GAT head's val MAE meets threshold (Task 4); the
  switch is a separate, flagged cutover.

### 1.3 Decisions presented to the user (recommended defaults, adjustable)

1. **GAT pretraining strategy** — *recommended*: (A) pretrain GAT on OGB (`ogbn-arxiv`) then
   transfer to the memory graph, with (C) direct-training-on-Oracle-labels as the fallback if
   (A) underperforms on the small, typed graph. (B) load-a-published-checkpoint is a middle
   option if OGB pretraining is too slow.
2. **EXPAND-frequency blocker resolution** — *recommended*: (a) persist the 2c
   `outcome_buffer`/`override_buffer` to the store (per-user, cross-session, mirroring the
   `save_jgs_state` pattern in `src/memory/store.py`) **and** auto-invoke `record_outcome` in
   `PonderOrchestrator.query`. (b) scope-cut is the alternative if durability lands late.
3. **Node-feature pipeline** — *recommended*: episodes use the 384-dim embedder vector;
   entity/topic/tone use type-onehot ∪ optional embedding; user/decision use onehot. The
   feature dim is the max across kinds, zero-padded (or a per-kind projection MLP).
4. **Archive semantics** — *recommended*: pruned edges move to an `archive/` subtree
   (`archive/edge/{...}`, `archive/ep/{eid}/...`), recoverable, excluded from default queries;
   `abstracted` episodes stay retrievable but are excluded from default queries (spec §371).

---

## 2. File Map

All paths keyed to the real `src/` layout. `[MODIFY]` = existing file changed. `NEW` = created
by 3a.

```plaintext
src/gnn/                              # NEW package
├── __init__.py
├── graph_loader.py                   # NEW — WaveDB graph → PyG Data; BFS radius-3; node features
├── features.py                       # NEW — node-feature pipeline (per node-kind → vector)
├── model.py                          # NEW — GAT backbone (OGB-pretrained option) + head registry
├── heads.py                          # NEW — SalienceHead, DiffPoolHead, LinkPredHead,
│                                     #         AnomalyHead, OntologyHead (losses + metrics)
├── train.py                          # NEW — per-head training loop (CPU-dev + pod-ready)
├── consolidate.py                    # NEW — nightly dream-state loop (dry-run default + --apply)
└── semantic_memory.py                # NEW — write/read abstracts + supersedes; mark abstracted
src/memory/
├── store.py                          # [MODIFY] — abstracts/supersedes edge ops; archive subtree;
│                                     #   persist presentation_gate buffers (Task 7 option a)
├── ontology.py                       # [MODIFY] — accept GNN-proposed subClassOf (Bonsai-gated)
└── episode.py                        # [MODIFY] if needed — abstracted flag / consolidation fields
src/subconscious/
└── presentation_gate.py              # [MODIFY] — (Task 7 option a) persist + restore buffers
src/training/
├── prompts.py                        # [MODIFY] — gnn_link_prediction_prompt emits negative_edges
└── oracle_labeling.py                # [MODIFY] — remove dead ORACLE_GNN_LABELING_PROMPT (:56-69)
scripts/
├── generate_gnn_training_data.py     # [MODIFY] — default --subgraph-radius 3; negative-edge emit
├── train_gnn.py                      # NEW — pod entrypoint (mirrors scripts/train_backbone.py)
└── run_consolidation.py              # NEW — nightly-loop entrypoint (--dry-run default, --apply)
src/
└── config.py                         # [MODIFY] — Phase3aConfig dataclass
pyproject.toml                        # [MODIFY] — add torch_geometric, ogb
tests/
├── test_graph_loader.py              # NEW
├── test_gnn_features.py              # NEW
├── test_gnn_model.py                 # NEW
├── test_gnn_heads.py                 # NEW
├── test_semantic_memory.py           # NEW
├── test_consolidate.py               # NEW
└── integration/test_phase3a_pipeline.py  # NEW
docs/adr/
├── 008-gnn-consolidation-design.md   # NEW
└── 009-semantic-memory-storage.md    # NEW
```

---

## 3. Task 1 — Graph loader (WaveDB → PyG)

**Files:** `src/gnn/graph_loader.py`, `src/gnn/features.py` (NEW)

### 3.1 Design

Reuse `OracleLabelingPipeline.extract_subgraph` (`src/training/oracle_labeling.py:112`) as the
BFS core — it already walks node-to-node edges over `_NODE_PREDICATES`, normalizes edge
orientation, and dedups. Wrap it (or extend it) to emit `torch_geometric.data.Data`:

```python
@dataclass
class GraphSlice:
    data: Data            # edge_index [2, E], x [N, F], node_kind [N], edge_attr [E, P]
    node_ids: list[str]   # positional → node index
    center_id: str
    radius: int

class WaveDBGraphLoader:
    def __init__(self, store, embedder: Optional[Embedder] = None,
                 feature_fn: Optional[NodeFeatureFn] = None): ...
    def load_subgraph(self, center_id: str, radius: int = 3) -> GraphSlice: ...
    def load_all(self, max_nodes: Optional[int] = None) -> GraphSlice: ...  # full graph
```

Node features (`features.py`) per node-kind, keyed on the real id prefixes:

| Node kind | Prefix | Feature |
|---|---|---|
| episode | `ep_` | 384-dim embedder vector over `summary` (the same `Embedder.encode` the subconscious package uses); if no embedder, a bag-of-topics hash |
| entity | `E:` | type-onehot ∪ optional 384-dim embedding of the entity string |
| topic | `T:` | type-onehot ∪ optional embedding |
| tone | `A:` | type-onehot (small fixed vocabulary) |
| decision | `D:` | type-onehot |
| session/user | `S:` / `U:` | type-onehot |

Feature dim `F` = max across kinds; shorter vectors zero-padded, **or** a per-kind projection
MLP into a shared `F` (decision in §1.3 — projection is cleaner, onehot+pad is simpler).

`edge_attr` = predicate onehot over the snake_case predicate vocabulary + `subClassOf` + the
open Bonsai-relation bucket (hashed into K slots).

### 3.2 Acceptance criteria

- [ ] `load_subgraph(center, radius=1)` on a tmp_path store reproduces the exact node/edge set
      `extract_subgraph` returns (round-trip parity).
- [ ] `edge_index` is undirected-symmetric (each stored triple contributes both directions, or
      the model uses `edge_attr` to distinguish — pick one, document it).
- [ ] `x` shape `[N, F]`, no NaNs; `node_kind` indices in range.
- [ ] Loads 100 radius-3 subgraphs (CPU) in < 2s (excluding embedder calls; embedder is the
      bottleneck — cache episode embeddings).
- [ ] No dependency on the SSM stack (this is `torch_geometric`, not `mamba_ssm`).

**Test file:** `tests/test_graph_loader.py`, `tests/test_gnn_features.py`

---

## 4. Task 2 — GNN model + 5 heads

**Files:** `src/gnn/model.py`, `src/gnn/heads.py` (NEW)

### 4.1 Backbone

GAT backbone (multi-head attention over the graph), configurable hidden dim and layer count.
Per §1.3 decision 1, the backbone is **pretrained on OGB then transferred** (recommended):
load `ogbn-arxiv`, train GAT to convergence, then fine-tune on the memory graph. The
direct-train fallback (C) skips OGB and trains on Oracle labels from scratch.

### 4.2 Heads (5, per spec §347-352)

| Head | Architecture | Loss | Label source | Metric |
|---|---|---|---|---|
| Salience | GAT regression over node + edge scores | MSE | `salience_labels.jsonl` (`node_scores`/`edge_scores`, `prompts.py:21-46`) | val MAE |
| Subgraph summarization | DiffPool (cluster assignment + pooling) | cluster assignment CE + coherence | `cluster_labels.jsonl` (`clusters` + `abstracted_summary` + `coherence_score`) | cluster purity / coherence |
| Link prediction | GAE / SEAL (SEAL uses subgraph features around the candidate edge) | BCE over **pos + neg** edges | `link_prediction_labels.jsonl` (+ **negative_edges** added in Task 3) | val AUC |
| Anomaly detection | multi-label classification over 6 types (ORPHAN_DECISION, MISSING_TEMPORAL, CONTRADICTION, TYPE_VIOLATION, ISOLATED_CLUSTER, DUPLICATE_DECISION) | multi-label BCE | `anomaly_labels.jsonl` (`gnn_anomaly_prompt`, `prompts.py:94-105`) | per-class F1 (macro) |
| Ontology refinement | BCE over proposed `subClassOf` edges | BCE | `ontology_labels.jsonl` (`suggested_edges` + `misclassified`) | val accuracy |

### 4.3 Honest scope (per §1.2)

The model is **stateless** in 3a. No SSM-augmented temporal instance — that comes only after
failure modes are observed (spec §378). The heads are trained **per-head** (5 separate
training runs, shared backbone) in the first slice; joint multi-task training is a later
optimization once per-head baselines validate.

### 4.4 Acceptance criteria

- [ ] Forward pass on a `GraphSlice` produces all 5 head outputs with documented shapes.
- [ ] Each head's loss trains to a non-trivial baseline on a tiny synthetic graph (sanity).
- [ ] `model.parameters()` count recorded (reconcile against the spec's ~200M figure).
- [ ] Backbone is excludable/freezable for head-only fine-tuning.

**Test file:** `tests/test_gnn_model.py`, `tests/test_gnn_heads.py`

---

## 5. Task 3 — Regenerate Oracle labels at scale

**Files:** `scripts/generate_gnn_training_data.py` [MODIFY], `src/training/prompts.py` [MODIFY],
`src/training/oracle_labeling.py` [MODIFY]

### 5.1 Changes

1. **Default `--subgraph-radius` to 3** in the generator (the extractor already supports it;
   the PoC passed `1`). Target `--num-subgraphs 4000`.
2. **Extend `gnn_link_prediction_prompt`** (`prompts.py:72`) to also emit `negative_edges` —
   non-edges that should remain non-edges — so SEAL/GAE have both classes. Add a
   random-negative sampler in the loader as a cheap supplement (hard negatives via BFS
   same-hop sampling — decision documented in the ADR).
3. **Remove the dead `ORACLE_GNN_LABELING_PROMPT`** in `oracle_labeling.py:56-69` (single-label,
   "Not invoked in 1b", contradicts the 5-prompt library the generator actually uses). One
   source of truth: `prompts.py:21-137`.
4. Run the generator (local Bonsai 8B Oracle, `src/training/oracle_labeling.py` client). The
   validator (`scripts/validate_training_data.py`) already enforces ≥1000/task.

### 5.2 Acceptance criteria

- [ ] ≥4,000 subgraphs/task on disk (or a documented smaller count if Bonsai throughput
      bounds it — no silent truncation; `log()` the count).
- [ ] `link_prediction_labels.jsonl` has both `predicted_edges` (positive) and
      `negative_edges`.
- [ ] `cluster_labels.jsonl` has **non-empty** `clusters` at radius 3 (radius 1 yielded zero
      positives). If still sparse, document and add weak supervision from topic-co-occurrence
      (Risk R2).
- [ ] The dead `ORACLE_GNN_LABELING_PROMPT` is gone; `grep ORACLE_GNN_LABELING_PROMPT src/` is empty.

---

## 6. Task 4 — Train (pod, RTX 4090 / RunPod dashboard)

**Files:** `src/gnn/train.py` (NEW, library), `scripts/train_gnn.py` (NEW, entrypoint)

### 6.1 Design

`scripts/train_gnn.py` mirrors `scripts/train_backbone.py` (the 2a pod entrypoint): argparse
config, float32, checkpoint best-val to `data/pod_runs/phase3a/{head}.pt`, log train/val
metrics. Per-head training (5 runs, shared backbone checkpoint loaded once).

**float32, not bf16.** The 2a bf16/autocast dtype-mix bug is still open (memory
`hippo-phase-2a-status`, `mamba3-cuda-build-fails`). GNN training is **independent of the SSM
bf16 path** (this is `torch_geometric`, not `mamba_ssm`), but the lesson stands: train
float32 first, attempt bf16/amp only after a clean float32 baseline.

### 6.2 Pod ops (per memory)

- Create the pod via the **RunPod dashboard** (MCP `create-pod` SSH proxy is broken —
  `runpod-api-pods-dead-on-arrival`). Paste the ssh string.
- Training needs only the regenerated labels + `src/` + `scripts/train_gnn.py` (no DB on the
  pod — labels are JSONL).
- **SCP checkpoints local before stopping the pod** — container disk is ephemeral across
  stop/start (`runpod-community-pod-disk-wipe`). Verify the checkpoint loads locally before
  terminating.

### 6.3 Acceptance criteria

- [ ] Each head meets a val threshold: salience MAE < 0.15 (calibrate against the regenerated
      Oracle label distribution — set the bar after inspecting label variance, not before);
      linkpred AUC > 0.80; anomaly macro-F1 > 0.60 (heavy class imbalance — "no anomaly"
      dominates); ontology accuracy > 0.75; diffpool coherence > 0.70.
- [ ] Checkpoints SCP'd local to `data/pod_runs/phase3a/` (gitignored) + verified to load.
- [ ] Training log records per-head param count, final train/val metrics, wall-clock.
- [ ] If OGB pretraining (§1.3 decision 1A) underperforms direct training (1C), the fallback
      is run and the better checkpoint is kept; the decision is recorded in the ADR.

---

## 7. Task 5 — Semantic-memory storage (greenfield)

**Files:** `src/gnn/semantic_memory.py` (NEW), `src/memory/store.py` [MODIFY],
`src/memory/ontology.py` [MODIFY]

### 7.1 Design

`semantic_memory.py` + store ops for the DiffPool head's output:

- **`abstracts` edge** (`SemanticMemory` node → source episodes): a new node kind `M:{hash}`
  (semantic memory) with `abstracts` edges to each source `ep_NNNNNN`. The semantic-memory
  node carries the abstracted summary text + the cluster's coherence score.
- **`supersedes` edge** (new abstract supersedes a stale one): the predicate is already
  declared (`ontology.py:79`) — wire the writer. Reconsolidation: when a newer abstract
  covers the same source set, write `M:new supersedes M:old`.
- **`Episode.consolidation_window_start`**: finally **set** it (`episode.py:50` field,
  `store.py:110-111` writer) when an episode is first abstracted — the timestamp of the
  consolidation pass that absorbed it.
- **`abstracted` flag**: mark source episodes "abstracted" (still retrievable via explicit
  lookup, **excluded from default queries** — spec §371). Implement as a `content/ep/{eid}/abstracted`
  marker; default-query traversal skips abstracted episodes.
- **Archive subtree**: pruned low-salience edges move to `archive/edge/{...}` (and pruned
  episodes to `archive/ep/{eid}/...`), recoverable, **never deleted** (spec §366). Default
  queries exclude `archive/`.

### 7.2 Acceptance criteria

- [ ] Round-trip: write a semantic memory + `abstracts` edges → read back → source episodes
      resolve; summary + coherence preserved.
- [ ] An abstracted episode is **excluded** from a default `retrieve()` but **included** when
      queried explicitly by id.
- [ ] An archived edge is recoverable from `archive/` and excluded from default traversal.
- [ ] `supersedes` written on reconsolidation; the older abstract is still readable (not
      deleted).
- [ ] `consolidation_window_start` is set on the first abstraction (no longer always-None).

**Test file:** `tests/test_semantic_memory.py`

---

## 8. Task 6 — Consolidation loop (nightly dream-state)

**Files:** `src/gnn/consolidate.py` (NEW), `scripts/run_consolidation.py` (NEW)

### 8.1 Loop (per spec §360-366)

```
for each node/edge in the graph (or a WM-prioritized subset — see §8.2):
  1. score salience (GAT head) → prune low-salience edges (archive, never delete)
  2. detect clusters (DiffPool head) → abstract semantic memories (write abstracts edges)
  3. predict missing edges (GAE/SEAL head)
       → auto-accept high-confidence (> accept_threshold)
       → propose medium-confidence to Bonsai (reuse src/training/oracle_labeling.py client)
  4. detect anomalies (Anomaly head) → wake Bonsai for verification on flagged subgraphs
  5. refine ontology (Ontology head) → suggest new subClassOf (Bonsai-gated before write)
emit a ConsolidationReport {edges_proposed, edges_accepted, anomalies, abstracts, pruned}
```

**Dry-run by default.** `scripts/run_consolidation.py --dry-run` produces the report without
mutating the graph; `--apply` writes. Bonsai verification gates every medium-confidence
accept (target >70% validated, `Ponder Engine Phases.md` line 583) and every ontology
addition.

### 8.2 WM-prioritized consolidation

The GNN reads `WorkingMemoryState` (`JGSSnapshot`, `src/subconscious/state_serializer.py`) to
prioritize consolidating what's currently "in awareness" — episodes/topics the user has
recently touched get scored first. The WM metadata keys actually written are `last_query_type`
and `active_domains` (NOT `recent_topics` — minor 2c doc drift noted in §0; the code is the
source of truth).

### 8.3 Acceptance criteria

- [ ] `--dry-run` on a tmp_path store produces a `ConsolidationReport` with zero graph
      mutations (verify via before/after key-set diff).
- [ ] `--apply` writes only the auto-accepted + Bonsai-validated edges/abstracts; rejected
      proposals are logged, not written.
- [ ] No edge is ever deleted — pruned edges appear in `archive/`.
- [ ] The loop is resumable (a crash mid-pass doesn't leave the graph half-mutated — batched
      `expand_triple` ops, atomic per cluster).

**Test file:** `tests/test_consolidate.py`, `tests/integration/test_phase3a_pipeline.py`

---

## 9. Task 7 — Resolve the EXPAND-frequency blocker (from 2c)

**Files:** `src/subconscious/presentation_gate.py` [MODIFY], `src/memory/store.py` [MODIFY],
`src/orchestrator.py` [MODIFY]

### 9.1 The blocker (restated)

Phase 2c §15 handoff promises "EXPAND frequency feeds GNN salience." But
`presentation_gate.py:167-168`: `outcome_buffer` and `override_buffer` are in-memory
`ReplayBuffer`s, and `PonderOrchestrator.record_outcome` is never auto-invoked by `query()`.
The signal does not survive a session restart and is not even collected in the live path.

### 9.2 Resolution (recommended: option a)

- **Persist the buffers** to the store, per-user cross-session, mirroring the `save_jgs_state`
  pattern (`src/memory/store.py`): `save_presentation_outcomes(user_id, records)` /
  `load_presentation_outcomes(user_id)`. Append-only JSONL blob under
  `content/system/user/{user_id}/presentation_outcomes`.
- **Auto-invoke `record_outcome`** in `PonderOrchestrator.query` using the run's measured
  `expand_count` (the `ExpandHandler.expand_count` already tracked per query). The other two
  `PresentationOutcome` fields are **not faked**: `unused_primary_count` is not directly
  measured today (we don't observe which primary chunks the model actually attended to) — it
  stays caller-supplied, defaulting to 0 with a "not measured" log entry until a proxy is
  defined (e.g. primary chunks whose topics never appear in the response); `user_satisfaction`
  stays caller-supplied (absent → 0.0 + "missing" log). The **durable, honestly-measured**
  signal is `expand_count` — that is the EXPAND-frequency feature the GAT head consumes.
- **GAT head feature column**: the persisted per-episode EXPAND-frequency becomes a node
  feature (or a salience-label prior) in the GAT training set.

### 9.3 Alternative (option b — scope cut)

If durability lands late, scope the first GNN slice to **Oracle salience labels + the 1c
heuristic mention-count prior only**, and defer EXPAND-frequency to 3a.1. The doc records
which path was taken.

### 9.4 Acceptance criteria (option a)

- [ ] After `query()` runs, the outcome buffer is persisted to the store.
- [ ] A **restarted** orchestrator (new process, same user_id) loads prior outcome counts
      (non-zero) — the cross-session property.
- [ ] `record_outcome` is invoked exactly once per `query()` (not on unsupported pathways).
- [ ] `user_satisfaction` is never faked — absent → 0.0 + a "missing" log entry.
- [ ] The GAT training set includes an EXPAND-frequency feature column sourced from the
      persisted records.

---

## 10. Task 8 — Configuration + dependencies

**Files:** `pyproject.toml` [MODIFY], `src/config.py` [MODIFY]

### 10.1 Dependencies

Add to `pyproject.toml`:
- `torch_geometric` (PyG — the graph-ML layer; pure-Python on top of `torch`).
- `ogb` (OGB benchmarks for the GAT pretraining, §1.3 decision 1A).
- (No `dgl` — pick PyG, not both.)

Note: **3a does NOT depend on the SSM stack.** `mamba_ssm` / Mamba3 build still fails
(memory `mamba3-cuda-build-fails`); that's irrelevant here — the GNN is `torch_geometric`,
not `mamba_ssm`. Record this in the ADR so nobody blocks 3a on the Mamba3 build.

### 10.2 Config

```python
@dataclass
class Phase3aConfig:
    gnn: GNNConfig = GNNConfig(hidden_dim=128, num_heads=4, num_layers=3)  # reconcile ~200M
    consolidation: ConsolidationConfig = ConsolidationConfig(
        accept_threshold=0.85, bonsai_propose_threshold=0.60,
        dry_run_default=True, prune_salience_below=0.15)
    archive: ArchiveConfig = ArchiveConfig(subtree="archive/")
    labels: LabelGenConfig = LabelGenConfig(num_subgraphs=4000, subgraph_radius=3)
```

---

## 11. Task 9 — Integration tests + ADRs

### 11.1 Tests

Offline suite, no GPU, no Bonsai (stub the Oracle client for the consolidation loop), no SSM:
`tests/test_graph_loader.py`, `test_gnn_features.py`, `test_gnn_model.py`, `test_gnn_heads.py`,
`test_semantic_memory.py`, `test_consolidate.py`, `integration/test_phase3a_pipeline.py`.

| Test | Description | Expected |
|---|---|---|
| `test_loader_roundtrips_extract_subgraph` | load_subgraph vs extract_subgraph on tmp_path | identical node/edge sets |
| `test_features_no_nans` | features for all node kinds | `x` finite, dims correct |
| `test_model_forward_all_heads` | forward pass on a tiny GraphSlice | 5 head outputs, correct shapes |
| `test_linkpred_has_negatives` | loader emits pos+neg | both classes present |
| `test_semantic_memory_round_trip` | write abstracts → read → resolve sources | sources resolve, summary preserved |
| `test_abstracted_excluded_from_default_query` | abstracted episode | excluded by default, explicit-lookup works |
| `test_archive_not_delete` | prune an edge | recoverable from archive/ |
| `test_consolidation_dry_run_no_mutation` | --dry-run | zero key-set diff |
| `test_consolidation_apply_gated_by_bonsai` | --apply with stubbed Bonsai rejects | only accepted edges written |
| `test_outcome_buffer_persisted_across_restart` | Task 7 option a | restarted orchestrator sees prior counts |

### 11.2 ADRs

- **`docs/adr/008-gnn-consolidation-design.md`** — why **stateless-first** (spec §378: temporal
  continuity only after failure modes); why **OGB-pretrain-then-transfer** with direct-train
  fallback (§1.3 decision 1); why **per-head training first** (joint multi-task later); why
  **archive-not-delete** (spec §366 — reversible consolidation); the negative-edge sampling
  choice (random vs hard).
- **`docs/adr/009-semantic-memory-storage.md`** — the `abstracts` / `supersedes` / `M:` node
  schema; `abstracted`-vs-default-query semantics (spec §371);
  `consolidation_window_start` finally set; the archive subtree layout.

---

## 12. Cost & hardware (corrected)

| Item | Spec | Reality / this doc |
|---|---|---|
| Hardware | RTX 4090, Vast.ai spot | RTX 4090, **RunPod dashboard pod** (MCP create-pod SSH broken — `runpod-api-pods-dead-on-arrival`) |
| Impl time | 16h, $4.32 | same anchor |
| Train time | 32h, $8.64 | same anchor; 5 per-head runs may push this — record actual |
| Total | ~$13 / 48h | anchor; provider = RunPod |
| Checkpoint safety | (unspecified) | **SCP local before pod stop** (`runpod-community-pod-disk-wipe`) |
| Params | ~200M (line 885) | reconciled to real layer counts (Task 2) |

3a adds **one training cost** (the GNN) — unlike 2c (which was runtime-only). The Oracle
label regeneration (Task 3) uses the **local Bonsai 8B** (no cloud spend, $0 tracked — same
as 1d). A note is added to `Ponder Engine Phases.md` Cost Summary: provider reality + the
1d-pipeline-pre-shipped note.

---

## 13. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **EXPAND-frequency not durable (BLOCKER)** | High | GAT head loses a designed signal | Task 7 option a (persist + auto-record); fallback option b (scope cut to Oracle + heuristic prior) |
| `cluster_labels` empty at radius 1 → radius-3 regen still sparse | Medium | DiffPool head undertrains | Weak supervision from topic-co-occurrence; lower cluster-purity threshold for first slice; document if non-empty count is still small |
| Link-pred negative sampling choice affects AUC | Medium | Suboptimal linkpred | Random negatives for baseline; hard negatives (same-hop BFS) as the documented upgrade; ADR records the choice |
| OGB transfer underperforms on the small typed memory graph | Medium | Wasted pretraining | Fallback to direct training (§1.3 decision 1C); keep the better checkpoint |
| GNN ~200M oversized for ~10⁴-10⁵ nodes | Medium | Slow train, overfit | Size to the real graph (Task 2); record actual param count |
| Nightly loop mutates the live graph | High | Data corruption / unrecoverable state | Dry-run default + `--apply` gate; atomic per-cluster `expand_triple` batches; archive-not-delete (reversible) |
| Bonsai verification throughput bounds Task 3 + Task 6 | Medium | Label regen / consolidation slow | Batch Bonsai calls; rate-limit; `log()` the count if below target (no silent truncation) |
| 2a bf16/autocast bug still open | Low (for 3a) | None — GNN is float32, independent of SSM bf16 path | Train float32; do not block 3a on the Mamba3 build |
| Predicate casing confusion (snake_case graph vs camelCase registry) | Low | Loader keys on wrong predicate | §0 records it; loader keys on snake_case; test asserts the predicate vocabulary |
| Pod disk wipe loses checkpoints | Medium | Lost training run | SCP local before stop; verify checkpoint loads locally before terminating pod |

---

## 14. Definition of Done (implementation, after this doc is approved)

- [ ] All unit + integration tests pass (Task 9).
- [ ] Graph loader round-trips `extract_subgraph`; features finite; model forward pass clean.
- [ ] Oracle labels regenerated at ≥4,000 subgraphs/task, radius 3, with link-pred negatives;
      dead `ORACLE_GNN_LABELING_PROMPT` removed.
- [ ] GNN trained float32; per-head val metrics meet thresholds; checkpoints SCP'd local +
      verified.
- [ ] Semantic-memory storage round-trips; abstracted episodes excluded from default queries;
      archive recoverable; `supersedes` + `consolidation_window_start` finally written.
- [ ] Consolidation loop dry-run produces a report with zero mutations; `--apply` gated by
      Bonsai; no edge ever deleted.
- [ ] EXPAND-frequency blocker resolved (Task 7 option a or b — documented which).
- [ ] `torch_geometric` + `ogb` added; `Phase3aConfig` dataclass in place.
- [ ] ADRs 008 + 009 written.
- [ ] **No regression** on 1b/1c/2a/2b/2c suites (the 1c heuristic stays the live scorer until
      the GAT cutover, which is a separate flagged change — not in this DoD).
- [ ] **de-wonk clean**: no untrained/dead params, no faked `user_satisfaction`, no silent
      truncation of label counts, no TODO left in scope, no stubbed head.

---

## 15. Next Phase Handoff

- **Phase 3b — Forgetting System** (`Ponder Engine Phases.md` §384): retrieval-weighted
  persistence — consumes the GNN salience scores 3a produces to drive retention/pruning
  policy. The archive subtree (Task 5) is the 3b substrate.
- **Temporal-continuity GNN** (spec §378): once 3a's stateless GNN has run nightly and failure
  modes are observed (cluster flapping, miscalibration, anomaly false positives, ontology
  oscillation), SSM-augmented instances with memory of past consolidation decisions are
  added. **Not premature** — built only for failure modes that actually occur.
- **Cross-document deduplication** (spec §373-376): deferred from 3a; lands once document-
  section features are defined.
- **Learned Presentation Gate** (from 2c §15): the persisted outcome buffer (Task 7 option a)
  is the training-data seed for the deferred learned gate — 3a's durability fix unblocks it.