# ADR 008: GNN Consolidator — stateless GAT + 5 heads, OGB-pretrain-then-transfer, archive-not-delete

**Status:** Accepted (Phase 3a)
**Date:** 2026-07-07

## Context

Phase 3a deploys a "dream-state" consolidator: a GNN that scores the memory
graph, abstracts semantic memories, predicts/verifies edges, flags anomalies,
refines ontology, and prunes low-salience edges — run nightly (or on demand).
The spec (`docs/Ponder Engine Phases.md` §339-380) describes "5 task-specific
heads, trained on Oracle-labeled memory graphs," a "static GNN," and
"abstracts edges; source episodes marked abstracted."

The §0 alignment pass (`docs/Phase 3a.md`) surfaced the real starting point:

- **Phase 1d pre-shipped the label pipeline** — 5 `gnn_*_prompt` functions
  (`src/training/prompts.py:21-137`) + `scripts/generate_gnn_training_data.py`
  + validators, with a 3-subgraph proof-of-concept on disk (~0.075% of the
  4,000+ target). So 3a is **regenerate-at-scale + build the model + loop**, not
  build-from-scratch.
- **Deps were greenfield** — no `torch_geometric`/`ogb`. **3a does NOT depend on
  the SSM/mamba_ssm stack** (PyG, not mamba) — the 2a Mamba3-cuda build failure
  is irrelevant.
- **link-prediction labels were positive-only** — SEAL/GAE need negatives.
- **salience target is real + pre-flagged** — `graph_traversal.py:389-430`'s
  heuristic mention-count is the cold-start prior the GAT head supersedes.
- **EXPAND-frequency salience was not durable** (inherited 2c §15 blocker) —
  see ADR 009 / Task 7.

## Decision

### Stateless GAT backbone first

The GNN is **static and stateless** per the spec's temporal-continuity note
(§378): no recurrent state, no per-instance memory. Temporal SSM-augmented
instances come only after failure modes are observed. This keeps the cold start
simple (a GAT is enough to beat the heuristic mention-count prior) and avoids
coupling 3a to the still-flaky SSM stack.

### 5 heads, per-head losses

`src/gnn/heads.py` — SalienceHead (MSE regression), DiffPoolHead (simplified
DiffPool: cluster assignment + per-node entropy + cluster-link preservation +
cluster-balance; the balance term is the anti-collapse fix — without it the
global min of entropy+link is the trivial "every node -> one cluster" solution,
loss 0 regardless of input), LinkPredHead (GAE dot-product; SEAL subgraph
features deferred), AnomalyHead (9-type multi-label pos-weighted BCE; the
pos-weight counters severe per-type node imbalance so BCE doesn't collapse to
"predict no anomaly"), OntologyHead (two-encoder `subClassOf` pair classifier,
BCE). Per-head training (Task 4) optimizes one
loss at a time; the consolidation loop (Task 6) calls whichever heads it needs.

### OGB-pretrain-then-transfer, with direct-train fallback

The GAT is pretrained on `ogbn-arxiv` (OGB), then transferred to the Oracle-
labeled memory graph. Rationale: the memory graph is small and typed (~10⁴-10⁵
nodes); a GAT pretrained on a large citation graph converges faster and
generalizes better than cold-starting on ~4,000 subgraphs. The cheap fallback
is direct training on the Oracle labels if OGB transfer underperforms
(`GNNConfig.ogb_pretrain` gates this; `ogb` is GPU/pod-only, in the `[gnn]`
extra). **OGB is not exercised on the dev machine** — it's a lazy, pod-only
path.

### Node features: per-kind, parameter-free loader + model-side projection

Graph nodes have no inherent vectors. `src/gnn/features.py` builds raw
features per node-kind (episode → 384-dim embedder vec; entity → type-onehot +
1c heuristic salience; others → type-onehot), packed into a 384-wide tensor.
The **per-kind projection MLP** lives in the model (`InputProjection`), selected
by a `node_kind` index — so the loader is parameter-free (testable + reusable by
the consolidation loop) and learnable parameters stay in the model.

### Archive-not-delete

Pruned low-salience edges are COPIED to an `archive/edge/...` subtree (JSON
record + reason + timestamp) and then removed from the live graph, in one
atomic `batch_sync`. Archive is recoverable (`read_archived_edge`) and never
deleted (spec §371). This is reversible forgetting, not destruction.

## Consequences

- The consolidation loop is **dry-run by default** (`ConsolidationConfig.
  dry_run_default = True`); `--apply` mutates. An **untrained model's apply is
  refused** without `--force-untrained` (random salience would prune ~every
  edge — destructive).
- The loop runs end-to-end on an untrained model (CPU dev) and produces a
  shape-correct report; the `trained` flag in the report is honest about which.
- Per-head training (Task 4) is a separate, pod/RunPod gate (~$13/48h RTX 4090,
  float32 — independent of the open 2a bf16 bug). See `docs/Phase 3a.md` §12.
- SEAL subgraph features for link-prediction are a documented later lever; the
  cold start is GAE dot-product. The link-prediction prompt now emits
  `negative_edges` (Task 3) so the head has both classes.

## References

- `docs/Phase 3a.md` — §0 alignment table + 9 tasks
- `src/gnn/{features,graph_loader,model,heads,consolidate}.py`
- ADR 009 — semantic-memory storage (the `abstracts`/`supersedes` schema)
- `hippo-phase-3a-status` memory