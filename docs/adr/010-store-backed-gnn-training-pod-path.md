# ADR 010: Store-backed GNN training pod path (copy the compact corpus DB to the pod)

**Status:** Accepted (Phase 3a Task 4a)
**Date:** 2026-07-08

## Context

The Phase 3a spec (`docs/Ponder Engine Phases.md` §6, "no DB on the pod — labels
are JSONL") envisioned the GNN trainer consuming only the regenerated label
files on the pod, with no WaveDB store present. Task 4a's trainer
(`src/gnn/train.py` + `scripts/train_gnn.py`) cannot honor that line with the
data flow Task 3 actually produces, because three things the trainer does all
require a live store:

1. **Subgraph reproduction for the anomaly head.** The label record carries
   `(subgraph_id, seed, types)` and the trainer REBUILDS the corrupted subgraph
   deterministically via `OracleLabelingPipeline.extract_subgraph` →
   `anomaly_rules.enrich_subgraph` → `anomaly_injector.inject_anomalies` →
   `graph_loader.data_from_subgraph`. `extract_subgraph` is a store BFS; there is
   no persisted subgraph structure (the generator serializes only the labels +
   the injection keys — deliberately, to avoid a 10K-node corrupted graph per
   JSONL record).
2. **Real 384-dim node features.** `NodeFeatureBuilder.feature_for` reads the
   backfilled episode embedding at `content/ep/{eid}/embedding` and the entity
   salience from the store. The hash-embedding stub is a dev-only fallback, not
   a training feature.
3. **The clean-head subgraphs** (salience / link / cluster / ontology) train on
   the SAME clean subgraph the labels were generated on, loaded fresh via
   `WaveDBGraphLoader` (the same BFS). This is the zero-train/serve-skew
   guarantee — the loader and the label generator walk the identical subgraph
   for a given `(center, radius)` by construction.

The generator persists no tensors and no subgraph structure, so a
tensor-persistence layer (the spec's implied "Path B") would be a new
end-to-end subsystem: serialize per-subgraph `Data` + corrupted `Data` + aligned
`node_labels`, ship them to the pod, and guarantee the loader and the
label-generation snapshot stayed byte-identical across that pipeline. That is
real engineering with its own skew surface.

## Decision

**Copy the compact corpus DB to the pod (Path A), open it read-only during
training.** By user decision (2026-07-08): the compact corpus DB is ~83 MB
(WaveDB 0.1.14 compaction, 4805 MB → 83 MB), SCP'd in seconds, read-only on the
pod (no corruption risk), and it preserves the loader's same-walk zero-skew
guarantee literally — the trainer walks the same store the generator walked.

This departs from the spec's "no DB on the pod" line. The departure is
recorded here; the spec is not amended in place.

### Provenance rule

The **exact** DB used for the generation run (Task 3, #123) is snapshotted and
kept under `data/` (gitignored — `data/pod_runs/`, `data/compact_corpus.db`,
etc.) and **re-SCP'd to any future training pod**. A training run is only
correct against the generation-snapshot DB; a re-compacted or re-encoded DB is a
different graph and would silently skew the labels from the structure. The
checkpoint sidecar `{head}.pt.meta.json` records `radius` + the config for
audit, but the DB itself is the source of truth and must be preserved
out-of-band.

## Consequences

- **Pod setup:** `scp` the compact corpus DB + the regenerated label dir to the
  pod; `python scripts/train_gnn.py --db <db> --labels <dir> --head all
  --device cuda --epochs 50` (Task 4b, #125). No tensor pipeline to build.
- **Zero skew:** the trainer and the generator share the BFS, so a training
  example and its labels are over the same node/edge set by construction. No
  alignment table is needed.
- **Read-only discipline:** the trainer opens the store for reads only (it never
  calls a mutator); the DB on the pod is the generation snapshot, untouched. The
  one in-memory mutation (`inject_anomalies`) is on a `deepcopy` of the
  extracted subgraph, never on the store.
- **Future Path B not precluded:** if the DB grows past cheap-SCP size (the
  lifelong-growth deployment fact is that production starts sparse and densifies
  over ~5 yr), a tensor-persistence layer can be added later without changing
  the trainer's loop — only `_build_inputs` would swap its source. The store
  path is the cold-start baseline; Path B is a future optimization, not a
  present requirement.

## `--head` joint/per-head training topology

Resolved with the user (2026-07-08): `--head {all,salience,link_prediction,
ontology,cluster,anomaly}`.

- `--head all` — one joint multi-task run: per step, sum every head loss that
  has usable labels for this subgraph (heads with no labels this step are
  skipped, not zeroed). Saves `all.pt` + one self-contained `{head}.pt` per
  head (all carry the same full `state_dict` — consolidation inference loads by
  head name). The cheap CPU-dev default.
- `--head <one>` — train that head only (`cluster` trains the diffpool head;
  the user-facing name matches the `cluster_labels.jsonl` stem). With
  `--backbone-checkpoint`, load a full `state_dict` (`strict=True`), FREEZE the
  GAT backbone (`input_proj` + `layers`), and refine just the head on the shared
  features (mirrors the 2b frozen-backbone gate pattern).

Pod path: `--head all` to prime the backbone, then optionally 5× `--head <one>
--backbone-checkpoint .../all.pt` to refine.

## Checkpoint format

A RAW `model.state_dict()` (strict-loadable), matching
`scripts/run_consolidation.py:_load_model`, which does `torch.load` +
`load_state_dict(state)` on a bare state_dict. Metadata (step, per-head val
metrics, config, wall-clock, skipped-endpoint counts) is written to a sidecar
`{head}.pt.meta.json` — NOT wrapped into the `.pt`, because the consolidation
loader expects a bare state_dict and a wrapped `{"model": ...}` dict would break
it.