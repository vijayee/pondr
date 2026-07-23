# The Ponder Engine consolidation pipeline

Date: 2026-07-23. Status: code-grounded map of the consolidation pipeline as
it actually runs today (Phases 1c–3c). All file:line references are the file's
own `cat -n` numbering — verified against current code by three read-only
sweeps of `src/gnn/`, `src/memory/`, `src/encoding/`, `src/ingestion/`,
`src/runtime/`, and `src/orchestrator.py`.

## The headline reframe

The "consolidation pipeline" is **not one running process** — it is three
separate execution contexts that share the WaveDB store as their contract:

| Context | When it runs | What it does | Where |
|---|---|---|---|
| **Ingest** | On every `ingest_document` / live turn | Write content KV + graph edges + edge sidecars | `src/ingestion/pipeline.py`, `src/memory/store.py`, `src/orchestrator.py::_persist_exchange` |
| **Dream-pass consolidation** | Manual/offline (`run_consolidation.py --apply`) | GNN 5-head pass + forgetting + contradiction + ontology | `src/gnn/consolidate.py` |
| **Query** | On every `query()` | Retrieve + synthesize; live-encode the exchange | `src/orchestrator.py:494-922` |

The contract between them is the WaveDB store: **ingest** writes SPO triples
+ `content/ep/` + `content/doc/` KV + edge sidecars; **consolidation** reads
the graph + sidecars via the loader and writes back through
`SemanticMemoryWriter` / `store.supersede_assertion` / `edge_meta_put_op`;
**query** reads the graph for retrieval and writes only the *current*
exchange's edges.

**Critical fact:** the GNN consolidator is offline-only and is **not wired
into `build_ponder` or the query path**. `src/runtime.py` has zero
`consolidat|gnn|GNN|dream` matches; `query()` does not call `Consolidator.run`.

---

## 1. Ingest — the write contract

Two ingest paths, both funnelling through `HippocampalStore`.

### 1a. Document path — `store.encode_document` (`src/memory/store.py:1645-1750`)

- **Hot metadata keys** under `content/doc/{d}/` — `source_type`,
  `source_path`, `title`, `doc_kind`, `entities`, `topics`, `relations`,
  `citations`, `resolved_citations`, `state_assertions`, `section_ids`, etc.
  (`:1574-1605`).
- **Graph edges** (`_document_graph_ops`, `:1465-1514`):
  `(doc, instanceOf, Document)`, `(doc, has_entity, E:x)` + reverse
  `appears_in_doc`, `has_topic`, `has_section`/`child_of` (the doc tree),
  section-level `has_entity`/`has_topic`, `cites` edges, Bonsai relation
  triples, and entity-state assertion edges `(E:entity, state, value)`
  carrying `asserted_by = section_id or doc_id` provenance (`:1516-1534`).
- **Citations**: `Document.citations` (`src/memory/document.py:124`) is
  populated by parsers; `encode_document` resolves them to doc_ids via
  title/URL match when `citation_resolution_enabled` (`:1709-1715`) and emits
  `(doc_id, "cites", resolved_target)` edges (`:1510-1512`). `cites` is a
  snake_case hash-tail predicate — GNN-invisible, checkpoint-safe.

### 1b. Episode path — two flavors

- **Synchronous** (`encode_messages` → `store.encode_episode`): one fused
  content+edges write.
- **Async stub-then-fill** (default now that `async_distill_enabled=True`):
  - **Main thread** (in `query()`'s `_persist_exchange`,
    `src/orchestrator.py:989-1001`): `encode_messages_stub` →
    `store.encode_episode_content` (stub: content KV + summary-embedding
    upsert, **no extraction, graph-thin**) → `distill_worker.enqueue(...)`
    → return immediately. The turn is vector-retrievable but graph-invisible.
  - **Background `DistillWorker`** (`src/encoding/distill_worker.py:93-124`):
    sets `pause_gate`, runs `encode_messages_fill` (GLiNER + Bonsai relations
    + deterministic state-assertion normalizer, in-memory) then
    `store.encode_episode_edges` (graph edges only). On failure logs
    `[distill-fail]` and the stub stays (vector-retrievable, graph-thin).

**Edge writes the fill produces** (`store._edge_ops`,
`src/memory/store.py:241-315`): `has_entity`/`in_episode`, `instanceOf`
(when a seed class is assigned), `has_topic`, `has_tone`, `has_decision`,
each Bonsai relation triple, **`(E:entity, state, value)` assertion edges**
with `asserted_by=eid`/`asserted_at` sidecar (D1, `:282-291`, `:1064-1093`
— RMW-merges, never revives a tombstone), **`(eid, cited_from, doc_id)`**
when the episode text mentions a doc title (D5, `:302-311`), and `follows`.

### 1c. Doc-kind tagging (Phase 3c) — ingest-time, not dream-pass

`build_doc_kind_tagger` (`src/ingestion/doc_kind.py:299-367`) builds the
**2-head ensemble** `EnsembleBackboneDocKindTagger` on the shared frozen
19.5M backbone:

- `pen0` — `data/training/doc_kind_head_attn_ce0/best.pt` (pure CE,
  snap-strong)
- `pen2` — `data/training/doc_kind_head_attn_ce2/best.pt` (dec-strong)

Logit-averaged over 5 classes:
`point_in_time_snapshot / decision_update / plan / reference / other`
(`src/subconscious/doc_kind_head.py:68-74`). Falls back to single-head →
`BonsaiDocKindTagger` (wraps `BonsaiDecider.classify_doc_kind`) → `None`.
Invoked at `src/ingestion/pipeline.py:271-277`; the result lands on
`doc.doc_kind`. **`build_ponder` does not construct this** — it is an
ingest-CLI concern (`scripts/ingest_document.py:124-139`).

### 1d. The 10-pass isolated Bonsai extractor

`src/encoding/bonsai_relations.py`. `ISOLATION_CLASSES` is 10
single-predicate passes (`has_state, decides, expresses, questions,
suggests, explains, concerns, involves, contradicts, follows_up_on`,
`:80-115`). `extract_isolated` (`:303-336`) loops all 10, calling
`pause_gate()` before each HTTP call, building a per-class prompt,
force-normalizing the predicate, degrading a failed pass to empty. Cost
~22.8s/doc — only viable because `async_distill_enabled` moves it to the
background worker.

The `foreground_busy` `threading.Event` (`distill_worker.py:66`) is set at
`query()` entry (`orchestrator.py:531-532`) and cleared on every return path
(`:919-920`); the worker blocks on it before each fill, so extraction never
races the foreground.

---

## 2. The dream-pass — GNN consolidation (`src/gnn/consolidate.py`)

### Trigger: manual only

```
python scripts/run_consolidation.py --apply \
  --checkpoint data/pod_runs/phase3a/all_fixed.pt --decide
```

No per-ingest, per-query, or scheduled trigger exists — the "nightly" in the
module docstring is aspirational. `build_ponder` has zero
`consolidat|gnn|GNN|dream` matches; `query()` does not call
`Consolidator.run`.

### `Consolidator.run` (`src/gnn/consolidate.py:159-323`), per center episode

1. `WaveDBGraphLoader.load(center)` → homogeneous PyG `Data` (BFS radius-3,
   bidirectional, `x[N,384]` + `edge_attr[E,32]` predicate onehot +
   `node_kind`/`node_depth`/`center_idx`). Not `HeteroData`.
2. `model(data)` → 5 head outputs.
3. `_step_cluster` → `_step_predict` → `_step_anomaly_bounded` →
   `_step_ontology` → `_step_prune` → `_step_forget`.
4. After all centers: global `_step_ontology_decay` then `_step_deep_archive`.
5. **One `_apply` phase** (`:722-1099`) is the *only* mutator — skipped under
   `--dry-run`; an untrained model refuses to apply unless `--force-untrained`
   (random salience would prune ~every edge).

### The 5 heads (`src/gnn/heads.py`)

> **Memory-correction:** the 5 heads are **salience / diffpool / linkpred /
> anomaly / ontology**. There is no "topic" or "tone" head — those are node
> *kinds* the GAT encodes structurally.

1. **`SalienceHead`** (`:34-60`): per-node logit → sigmoid → `[0,1]`.
   Consumed by `_step_prune` (hard-prune edges where both endpoints <
   `prune_salience_below`) and `_step_forget` (structural factor in
   `utility_score = 0.4·access + 0.6·structural`). Per-entity salience
   persisted to `content/entity/{bare}/structural_salience` for the retrieval
   hot path (`consolidate.py:811-812`).
2. **`DiffPoolHead`** (`:64-145`): soft cluster assignment; `_step_cluster`
   proposes one abstract `M:NNNN` per cluster with ≥2 episodes; `_apply`
   writes it via `SemanticMemoryWriter.create_abstract`. Cold-start DiffPool
   (entropy + link-preservation + balance losses), not full dense-pooling.
3. **`LinkPredHead`** (`:149-191`): GAE dot-product on sampled same-kind
   non-edge pairs; `_step_predict` auto-accepts ≥ `accept_threshold`,
   Bonsai-verifies in the band below.
4. **`AnomalyHead`** (`:205-273`): 9-type multi-label
   (`contradictory_state, duplicate_episode, duplicate_decision,
   orphan_decision, detached_episode, broken_follows, type_violation,
   isolated_cluster, stale_abstraction`). Re-runs on a **bounded radius-2 +
   fanout-cap-64 subgraph** to match its training distribution. Labels are
   **Oracle-FREE** (`anomaly_injector` + `anomaly_rules`), not from an LLM
   prompt.
5. **`OntologyHead`** (`:277-344`): two-encoder pair classifier (episode
   `node_emb` × taxonomy `class_emb` over the live class DAG); scores
   entity↔class; `_apply` writes `instanceOf` (Bonsai-gated) and may
   `create_class`.

### `BonsaiDecider` (`src/gnn/bonsai_decider.py:94`)

HTTP client to the 8B at `localhost:8080`. Consolidation-time methods: `gist`
(abstracts), `verify_typing` (ontology), `decide_anomaly` (identity_drift),
`decide_contradiction`. Only `classify_doc_kind` runs outside consolidation
(at ingest). It is **not** on the query path.

### Checkpoint

`data/pod_runs/phase3a/all_fixed.pt` — assembled by
`scripts/assemble_gnn_checkpoint.py` from `all.pt` (provides
`input_proj`/`layers`/`salience`/`linkpred`) + per-head retrains
(`diffpool_retrain/cluster.pt`, `anomaly_retrain/anomaly.pt`,
`ontology_trained.pt` which provides `ontology.*` + `taxonomy.*`).
Strict-loads into `GNNModel(hidden_dim=128, num_heads=4, num_layers=3,
predicate_vocab_size=32, num_clusters=16)`. `linkpred` is taken from the base
`all.pt` (never retrained — no `linkpred_retrain/` dir).

---

## 3. Forgetting (Phase 3b) — embedded in the same dream-pass

Forgetting is **not a separate pass** — it is steps 6–8 of `Consolidator.run`
plus the retrieval-time boost. All training-free (consumes the trained
salience head; retrain is the deferred 3b-P3 lever).

- **A1 Deep-archive** (shipped): soft-archive stamps `archived_at`; a global
  sweep physically removes edges older than `deep_archive_days` (365) via
  `archive_edge(..., remove_from_graph=True)` (recoverable `archive/edge/...`
  JSON). `consolidate.py:1436-1491`, apply `:868-883`.
- **A2 Anomaly-resolver** (deferred/dormant): would re-derive `(old_ep,
  new_ep)` from a `contradictory_state` flag and `supersede_episode`. **No
  production path writes entity-state assertion edges from episodes** — only
  the injector does — so the episode-supersede path always returns `None`.
  3c unblocked it at the **fact** level via `store.supersede_assertion`
  (`:1135-1164`), driven by the Bonsai adjudicator (`consolidate.py:1014-1099`).
- **A3 instanceOf + class-decay reassignment** (shipped, dormant): emits
  `instanceOf`, rewrites typing when a class is deprecated. Dormant because
  no discovered classes exist yet.
- **A4 Ontology-decay** (shipped, no-op): stamps `content/class/{c}/last_seen`;
  deprecates discovered classes older than `ontology_decay_days` (30). Seed
  classes are never eligible → no-op today.

**Ordering rule:** 3a `_step_prune` records first; 3b `_step_forget` skips
edges already in `report["pruned"]` (3a hard-prune takes precedence). Both
mutate in the single `_apply` phase.

**Retrieval-time boost** is the only forgetting activity outside the
dream-pass: `graph_traversal.py:930` applies `apply_retrieval_boost` per
matched edge on the hot path (gated on `forgetting_enabled`), updating
access-frequency/recency sidecars that feed `utility_score`.

---

## 4. Phase 3c — citation + contradiction (+ doc-kind + async-distill)

### Citation (ingest-time)

`cites` doc→doc edges + `cited_from` episode→doc provenance, both gated on
`citation_resolution_enabled` (see §1a, §1b).

### Contradiction (dream-pass-only)

In `_apply` (`consolidate.py:1014-1099`):

1. The anomaly head flags `contradictory_state` (distinct live `state` values
   for one entity).
2. `_gather_entity_context` (`:1154-1252`) reads each `(E:entity, state,
   value)` out-edge + its sidecar, resolves `asserted_by` →
   `source_path`/`doc_kind` (strips `_sec_NNN` to doc_id, one `get_sync`
   each).
3. `BonsaiDecider.decide_contradiction` (`bonsai_decider.py:218-266`) runs
   **two deterministic pre-HTTP guards** (`_deterministic_non_conflict`,
   `:376-487`):
   - **Equal values → `dismiss` + `no_action`** (defense-in-depth; the
     detector only flags distinct values).
   - **Both sources `point_in_time_snapshot` (semantic) OR both carry a
     calendar-month filename prefix (fallback) → `ask_user` + `no_action`**
     — complementary temporal, not a supersession; non-mutating. A
     `decision_update` or absent doc_kind with no month prefix falls through
     to the LLM.
4. Outcomes:
   - `fix` + `supersede_assertion` + `forgetting_enabled` + old≠new + both
     non-None → **the one auto-applied write** (fact-level tombstone: old
     edge sidecar `state="superseded"`, `superseded_by=new`,
     `validity_end=when`; edge NOT deleted — MVCC; `is_edge_current` returns
     False).
   - Any other `fix` → demoted to `ask_user` (record-only).
   - `ask_user`/`dismiss` → record-only.
   - No episode-level `supersede_episode` on this path (that is the 3b
     no-decider cold-start path, `consolidate.py:833-840`).

**doc-kind** (§1c) and **async-distill** (§1b) round out 3c.

---

## 5. Runtime wiring — what `build_ponder` actually constructs

`build_ponder` (`src/runtime.py:51-307`) builds, in order:

1. `HippocampalStore(db_path or config.db_path)` — `:161`.
2. Frozen Phase 2a **backbone** (`:166`).
3. Trained Phase 2b **retrieval gate** on that backbone (`:167`).
4. bge-small embedder.
5. `BonsaiQueryPlanner`.
6. `HippocampalRetriever` (with the gate) — `:176-182`.
7. `DocumentRetriever` if the store has docs — `:189-191` (Phase 1c).
8. `ModeAGenerator` (the 8B) — `:193`.
9. `HippocampalEncoder` (live-encode) — `:201-206`.
10. Optional STRM heads (relevance / context-builder / graduation-proxy /
    graduation / recoverability / latent-dynamics / salience-thresholds,
    all default-off) — `:218-286`.
11. `PonderOrchestrator` — `:288-306`, which internally constructs
    `WorkingMemory`, `SSMChunker`, `PresentationGate`, `ExpandHandler`,
    `ChunkedContextFormatter`, and the **`DistillWorker`** at
    `orchestrator.py:285-287` (only when `encoder is not None and
    async_distill_enabled`).

**Not constructed in `build_ponder`:** the **GNN Consolidator** (no
`Consolidator` import; runs only from `run_consolidation.py`) and the
**doc-kind tagger** (ingest-CLI only).

---

## 6. The query path — `PonderOrchestrator.query` (`orchestrator.py:494-922`)

`foreground_busy.set` → embed prompt + update WM → (optional) STRM salience
hook → compress prompt → **retrieval-gate route** (`retrieve_with_routing`,
`:607-642`): the gate picks a `pathway`; unsupported ones
(`ssm_direct`/`process_exec`/`tool_plan`) early-return → inject retrieved
episodes into WM as gist steps → Presentation Gate chunking + end-state plan
→ chunk → `_synthesize` (self-chat tool loop or one-shot 8B call) →
`dispatch_end_state` (only `synthesize` calls the LLM) → record presentation
outcome → **live-encode the exchange** (`_persist_exchange`, `:914-915`:
stub-then-enqueue or sync) → `foreground_busy.clear`.

**A query never triggers consolidation.** The only query-time graph writes
are (a) the live-encode of the *current* exchange and (b) `record_feedback`
(retrieval-boost). Contradiction detection and citation resolution are
**not** on the query path — they are dream-pass and ingest actions
respectively. There is no per-sentence citation-binding step in `query()`;
retrieved episodes are surfaced as `result["retrieved_episodes"]` and fed to
the LLM as context, with `record_feedback` rating them post-hoc.

---

## 7. Deferred pieces (tracked in `docs/phase-1c-3c-followups.md`)

- **3a-P1** OGB pretrain-then-transfer — loud-fail stub at
  `src/gnn/train.py:219`.
- **3a-P5** link_prediction re-label/retrain — 4200 skipped endpoints, val
  AUC=1.0 suspicious.
- **3b-P2** ontology discovered-class promotion (Bonsai-gated) — hooks ship,
  promoter deferred; `_step_ontology_decay` is a no-op today.
- **3b-P3** salience-head retrain.
- **3c-P1** Bonsai contradiction LoRA fine-tune — deterministic guards hold
  the line until it lands.
- **1d-1/1d-2/1d-3/1d-4/1d-6** training-data at scale (validate-slices ran,
  full runs deferred — Oracle budget/pod).

---

## 8. End-to-end summary

The consolidation pipeline as it **actually runs**:

```
ingest writes  →  offline run_consolidation.py reads+mutates  →  query reads (writes only the current turn)
```

- The **GNN** is the offline dream-pass engine (5 heads, BFS subgraphs,
  DiffPool, single `_apply` mutator).
- **Phase 3b forgetting** and **Phase 3c contradiction** are steps *inside*
  that same offline pass (training-free A1/A3/A4; A2 dormant-unblocked-by-3c).
- **Doc-kind tagging** and **async-distill** are ingest-time.
- **Citation** is ingest-time.
- **Nothing in the live query path touches the GNN.**