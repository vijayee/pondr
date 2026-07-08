# ADR 009: Semantic-memory storage — `abstracts`/`supersedes`, abstracted-vs-default-query, durable EXPAND-frequency

**Status:** Accepted (Phase 3a)
**Date:** 2026-07-07

## Context

The Phase 3a consolidation loop's DiffPool head clusters related episodes and
abstracts them into a **semantic memory** — a single node carrying the cluster's
gist. The spec (`docs/Ponder Engine Phases.md` §339-380) calls for "`abstracts`
edges; source episodes marked abstracted," and §371 says abstracted episodes
"still retrievable, not in default queries." Two gaps blocked this:

1. **Semantic-memory storage was entirely greenfield.** No `abstracts` edge, no
   semantic-memory node kind, no consolidator code. The `supersedes` predicate
   was declared in `src/memory/ontology.py:79` but **never written**.
   `Episode.consolidation_window_start` (`episode.py:50`) was written at
   `store.py:110-111` only if set → **never set**.
2. **EXPAND-frequency salience was not durable** (the inherited 2c §15
   blocker): `presentation_gate.py:167-168` `outcome_buffer`/`override_buffer`
   were in-memory `deque`s, never persisted, and `PonderOrchestrator.
   record_outcome` was never auto-invoked by `query()`. The salience signal the
   GAT head is supposed to learn from did not durably exist.

## Decision

### `M:` node kind + `abstracts` edges

A new node-key prefix `M:NNNN` (semantic Memory) — consistent with the graph's
id-prefix-typing convention (`E:`/`T:`/`A:`/`D:`/`S:`/`U:`/`ep_`). A semantic
memory is created by `SemanticMemoryWriter.create_abstract(source_episode_ids,
summary, ...)`, which in ONE atomic `batch_sync` writes:
- the `M:` node's content under `content/mem/{mid}/{summary,text,ts,
  abstracted_from,embedding?}`,
- `(M:NNNN, abstracts, ep_...)` per source episode,
- `content/ep/{eid}/abstracted = 1` per source,
- `content/ep/{eid}/consolidation_window_start = <ts>` per source (the field
  that existed but was never set),
- an optional `(M:new, supersedes, M:old)` edge.

### `supersedes` edges (the predicate that was declared but never written)

`SemanticMemoryWriter.supersede(new, old)` / the `supersedes=` kwarg on
`create_abstract` write the `supersedes` triple the ontology already declared.
A fresh abstraction that replaces a stale one points at it; the stale one is not
deleted (archive-not-delete, ADR 008).

### Abstracted-vs-default-query semantics

`HippocampalStore.default_episode_ids(include_abstracted=False)` scans
`content/ep/` and filters out abstracted episodes. The retrieval layer's two
enumeration sites (`vector_search._all_episode_ids`,
`graph_traversal._get_all_episode_ids`) delegate to it, so default queries
exclude abstracted sources uniformly. `include_abstracted=True` opts in
(explicit "show everything including abstractions"). Abstracted episodes' content
is untouched — they remain retrievable by id; only the default candidate set
excludes them (spec §371).

### Edge archive (prune, never delete)

`SemanticMemoryWriter.archive_edge(s, p, o, reason=...)` copies the triple to
`archive/edge/{s}/{p}/{o}` (JSON: subject/predicate/object/reason/archived_at,
hashing any `/`-bearing component) and deletes the live triple via
`graph.expand_triple(..., delete=True)` — in one atomic batch. `read_archived_edge`
recovers it. See ADR 008 for the archive-not-delete rationale.

### Durable EXPAND-frequency (resolving the 2c §15 blocker)

`PresentationGate.serialize_buffers()` / `load_buffers()` JSON-snapshot the two
`ReplayBuffer`s (records are already JSON-safe dicts). `HippocampalStore.
save_presentation_outcomes` / `load_presentation_outcomes` persist them under
`content/system/user/{user_id}/presentation_outcomes/state`, mirroring
`save_jgs_state`. `PonderOrchestrator.query()` now **auto-invokes
`record_outcome`** with the **measured** `expand_count` (from
`expand_handler.expand_count`, reset per query) — so the buffer populates
without a caller remembering to call it. `save_outcomes`/`load_outcomes` flush/
restore; the orchestrator auto-loads on init (when a store + user_id are set).

**`unused_primary_count` and `user_satisfaction` are NOT faked.** We don't
observe which primary chunks the model attended to, and no satisfaction rating
is collected in `query()`. They stay 0 (caller-supplied via `record_outcome`
when available); `expand_count` is the real, durable, measured signal. This is
the honest split — the doc overclaim that `unused_primary_count` is "derivable
from the ChunkedContext" was corrected in the §0 pass.

## Consequences

- A restarted orchestrator now sees prior outcome counts (the GAT head's
  training set can include an EXPAND-frequency feature column).
- Default retrieval automatically excludes abstracted episodes; existing tests
  are unaffected (no episode was abstracted before 3a, so the filter is a
  no-op on pre-3a corpora).
- `M:` nodes are reachable via the graph's `abstracts`/`supersedes` edges but
  are NOT in the loader's `_NODE_PREDICATES` BFS set — the GNN does not need to
  read M nodes as input (the DiffPool head PRODUCES the clusters that become M
  nodes; it doesn't consume them). This is intentional, not an omission.
- The save TRIGGER policy for outcomes mirrors `save_session` — the caller
  decides when to flush; `query()` only auto-records into the in-memory buffer.

## References

- `docs/Phase 3a.md` — §0 alignment (the greenfield storage + 2c blocker rows)
- `src/gnn/semantic_memory.py`, `src/memory/store.py` (new helpers),
  `src/subconscious/presentation_gate.py` (buffer serialize/load),
  `src/orchestrator.py` (auto-record + save/load_outcomes)
- ADR 008 — GNN consolidation design
- `hippo-phase-3a-status`, `hippo-phase-2c-status` memory