# Phase 4: Citation + Contradiction Detection

Date: 2026-07-15
Status: Implemented (deterministic path + Bonsai adjudication wired; full
suite green; live Bonsai dogfood PASS).

## 0. Alignment with the design chat

The Ponder Engine coding chat (`docs/The _Ponder_Engine_Coding_Chat.json`)
names conflict-aware cognition as a differentiator:

- **Msg 14** -- `contradicts` / `supersedes` typed edges, "Disturbance
  Detector + reconsolidation" for conflict resolution, citation /
  cross-document contradiction.
- **Msg 24** -- the fact-level tombstone model
  (`fact/old { superseded_by: fact/new, superseded_at, superseded_by_commit }`)
  + the `fact -> supported_by -> turn` provenance edge.

Phase 3b shipped the **Forgetting System** including the
`contradictory_state` anomaly resolver, but it was **wired-but-dormant in
production (A2, deferred)** for two reasons:

1. No production path wrote entity `(E:entity, state, literal)` edges (only
   the training injector + episode-level `(eid, "state", ...)`).
2. No value -> episode provenance (the resolver used a timestamp heuristic).

Phase 4 is the entity-state-assertion extraction path that closes A2, plus
citation as the provenance backbone. The outcome: Ponder detects when a new
statement contradicts an existing memory ("Policy A said X, Policy B updated
it to Y"), tombstones the old fact at the FACT level (not the whole episode),
and cites the source.

## 1. Design decisions (locked)

### D1. Assertion = `(E:entity, state, literal)` edge + edge-sidecar provenance

Reuse the existing `state` predicate (the resolver + detector already key on
it). Provenance (the chat's `supported_by` analog) is stored on the **edge
sidecar** (`content/edge/{s}/{p}/{o}`, `edge_meta.py`) as `asserted_by`
(episode/doc/section id) + `asserted_at` (timestamp). The sidecar is keyed
per-(s,p,o) so each distinct value carries its own provenance. No new node
type, no schema overhaul.

Assertion edges are **ENTITY-owned** (subject = `E:entity`), so they are
**PUT-ONLY** -- a doc update/delete does NOT retract a shared fact (a doc
updating X -> Y leaving the old edge IS the contradiction to catch). The
sidecar is RMW-merged (`_assertion_edge_ops`), so re-assertion preserves
forgetting state and NEVER revives a tombstone.

### D2. Fact-level tombstone (chat intent), NOT episode-supersede

The chat's tombstone is at the FACT level. Retiring a whole episode (the 3b
`supersede_episode` path) is too coarse -- it would tombstone the episode's
other still-valid facts. Contradiction resolution writes the edge sidecar
`state="superseded"` + `superseded_by` + `superseded_at` on the old
`(E:entity, state, oldValue)` edge (`store.supersede_assertion`). The edge is
NOT deleted (MVCC -- stays retrievable via include_inactive); `is_edge_current`
returns False for it. The anomaly subgraph load (`enrich_subgraph` state
branch) threads edge-currentness, so the tombstoned old value drops out of the
live-values set and the detector goes quiet -- the contradiction is
*resolved*, not just recorded.

The existing episode-level `supersede_episode` path is LEFT INTACT for the
`identity_drift` `fix` path and the public `reconsolidate` API (no regression;
guarded by `decider_active`). No persistent `contradicts` edge is written --
the contradiction is ephemeral (detect -> adjudicate -> tombstone), avoiding
the ontology dict-merge silent-overwrite issue.

### D3. Two detectors, one adjudicator (mirroring `identity_drift`)

- **Deterministic**: `_detect_contradictory_state` (anomaly_rules.py) -- ships
  unchanged; once assertions are written it fires in production.
- **Bonsai semantic adjudication**: `BonsaiDecider.decide_contradiction`
  (mirrors `decide_anomaly`) takes the flag + the conflicting values WITH
  provenance (gathered by an extended `_gather_entity_context`) and returns
  `{decision, action, reasoning}`. Conservative dispatch: `fix` auto-applies
  ONLY when `action` contains `supersede_assertion` AND `forgetting_enabled`
  AND `decider_active` -> edge tombstone (D2). Any other `fix` -> `ask_user`
  (record-only). Every decision is recorded in
  `report["contradictions_resolved"]` -- never a silent mutation.

### D4. A2 resolver provenance fix

`_resolve_contradictory_state` now reads each value's edge-sidecar
`asserted_by` / `asserted_at` to select old/new from real provenance. When two
values carry an episode-id `asserted_by` + `asserted_at`, it returns that pair
directly (no heuristic). **Falls back to the timestamp heuristic when sidecar
provenance is absent** (injector-planted edges, no sidecar) -- the 3b path is
byte-identical.

### D5. Citation (three pieces, all hash-tail predicates, GNN-invisible)

- **doc -> doc `cites` resolution**: `Document.citations` literals resolve to
  Document node ids via `find_document_by_title_or_url` (title / URL match);
  unresolved literals kept verbatim. `resolved_citations` is PERSISTED so
  update/delete emit symmetric deletes for exactly the edges written.
- **email provenance**: `in_reply_to` / `references` edges from the parser's
  metadata maps (Message-ID -> section id). PUT gated on
  `citation_resolution_enabled`; DELETE always emits (clean-up on a flag flip).
- **episode -> doc `cited_from`**: best-effort `(eid, cited_from, doc_id)` when
  an episode's text references a known doc title; no match -> no edge. The
  assertion sidecar `asserted_by` (which may be a doc/section) IS the primary
  `supported_by` link.

`state` / `cites` / `in_reply_to` / `references` / `realized_as` / `cited_from`
/ `appears_in_section` / `instanceOf` are all hash-tail predicates -- kept OUT
of `KNOWN_PREDICATES` (graph_loader) and `_NODE_PREDICATES` (oracle_labeling):
checkpoint-safe, GNN-invisible until retrain.

### D6. Cold-start byte-identical (load-bearing)

Every new write is gated: `assertion_extraction_enabled` (default True --
inert on corpora with no state patterns), `citation_resolution_enabled`
(default True -- unresolved literals stay as-is), `contradiction` adjudication
gated on `decider is not None` AND `decider_active` AND `forgetting_enabled`
(no HTTP, no mutation when cold). A plain corpus with no state assertions ->
zero `state` edges -> detector never fires -> no tombstones -> identical to
today.

### D7. Scope of deferred items (revised)

Two of three formerly-deferred items are IN this slice; the third stays
deferred.

- **IN: EnterpriseRAG-Bench eval as offline pytest fixtures (D8).** Vendored
  subset of the bench's "Conflicting Info" near-duplicate doc pairs +
  `expected_doc_ids` citation ground truth. Asserts contradiction-detection
  recall + cite-resolve rate on the **deterministic path only** (no server).
- **IN: `.mbox` single-file thread parsing (D9).** `mailbox.mbox` yields
  stdlib `Message` objects; the thread core `parse_messages` needs NO changes
  -- two wiring touches only (ext routing + `EmailParser.parse`).
- **OUT (deferred, noted here not TODO'd):** the Bonsai LoRA fine-tune on
  contradiction decision pairs (Bonsai is zero-shot this slice, like
  identity_drift); the full EnterpriseRAG-Bench LLM-judge harness
  (correctness x completeness, three-judge consensus) -- we take its labels,
  not its scorer.

### D8. EnterpriseRAG-Bench eval = offline pytest fixtures, deterministic only

`tests/fixtures/enterpriserag/pairs.json` (5 pairs, committed) +
`tests/test_enterpriserag_eval.py` encodes each pair, runs the deterministic
detector (no Bonsai), and asserts (a) contradiction recall on the `catchable`
pairs at the deterministic-normalizer ceiling (a paraphrased-only conflict is
honestly counted as a miss), (b) citation resolve-rate on `expected_doc_ids`.
No HTTP, no model, fully reproducible.

### D9. `.mbox` = feed `mailbox.mbox` into the existing thread core

`EmailParser.parse` detects a `.mbox` suffix and feeds
`list(mailbox.mbox(source_path))` into `self.parse_messages`. The dir-of-`.eml`
and single-`.eml` branches are byte-identical; the thread core is untouched.

## 2. File map

- `src/encoding/assertion_extractor.py` (NEW) -- deterministic normalizer.
- `src/encoding/bonsai_relations.py` -- `has_state(Entity, Value)` relation
  (subject is an ENTITY, not a person -- `decides` keeps person-decisions).
- `src/encoding/encoder.py` -- builds `episode.state_assertions` (deterministic
  UNION Bonsai `has_state`, dedup; try/except -> []).
- `src/memory/episode.py` -- `state_assertions` field.
- `src/memory/document.py` -- `resolved_citations` + `state_assertions` fields.
- `src/memory/store.py` -- `_assertion_edge_ops`; `encode_episode` assertion +
  `cited_from` blocks; `_document_graph_ops` cites + assertion + email
  provenance; `encode_document` resolves + persists `resolved_citations`;
  `find_document_by_title_or_url`; `supersede_assertion` (fact-level tombstone).
- `src/memory/edge_meta.py` / `forgetting.py` -- `default_meta` gains
  `asserted_by` / `asserted_at` / `superseded_by` / `superseded_at`.
- `src/memory/ontology.py` -- `EmailMessage` class; `cited_from` predicate.
- `src/gnn/anomaly_rules.py` -- `enrich_subgraph` state branch filters
  superseded edges (`is_edge_current`).
- `src/gnn/consolidate.py` -- `contradictions_resolved` report field; the
  no-decider `contradictory_state` block guarded by `decider_active`; the new
  decider adjudication loop; `_gather_entity_context` extended with
  `state_values` + provenance; `_resolve_contradictory_state` provenance fix.
- `src/gnn/bonsai_decider.py` -- `decide_contradiction`.
- `src/training/prompts.py` -- `bonsai_contradiction_decision_prompt`.
- `src/config.py` -- `assertion_extraction_enabled`,
  `citation_resolution_enabled`, `contradiction_resolve_threshold`.
- `scripts/run_consolidation.py` -- `--assertions/--no-assertions`,
  `--citation-resolution/--no-citation-resolution`,
  `--contradiction-resolve-threshold` (+ stdout utf-8 guard).
- `src/ingestion/parsers.py` -- `.mbox` -> `"email"`.
- `src/ingestion/email_parser.py` -- `.mbox` -> `mailbox.mbox` -> core.
- `src/ingestion/pipeline.py` -- per-section + doc-level `state_assertions`.
- `tests/test_assertion_extraction.py`, `tests/test_contradiction.py`,
  `tests/test_citation_resolution.py`, `tests/test_mbox_ingestion.py`,
  `tests/test_enterpriserag_eval.py` (NEW) + `tests/fixtures/enterpriserag/`.

## 3. Verification

1-5. The 5 new test files: 61 tests, all green (deterministic normalizer +
  Bonsai merge; detector fires, tombstone resolves, FakeDecider all three
  branches, provenance fix + fallback, edge-currentness; cites resolution +
  email provenance + cited_from; .mbox thread parsing end-to-end; bench
  recall + citation resolve-rate).
6. Full suite: **849 passed, 4 skipped** (788 prior + 61 new). The 3b
   `identity_drift` `supersede_episode` path + the public `reconsolidate` API
   are unchanged; the 3b forgetting tests still pass (new sidecar keys are
   optional/additive).
7. Live dogfood: the 8B Bonsai `decide_contradiction` (localhost:8080/v1)
   returns `{'decision': 'fix', 'action': 'supersede_assertion', ...}` on a
   realistic flag + state_values with provenance -- the conservative-safe
   action the dispatcher auto-applies, with coherent reasoning. The HTTP +
   prompt + parse path works live end-to-end.
8. de-wonk (CLAUDE.md) -- see the gate below; passed.

## 4. De-wonk gate (CLAUDE.md)

- **No persistent `contradicts` edge** -- grep confirms no code writes a
  `contradicts` graph triple (the Bonsai relation prompt still *lists* it for
  the model, but a `contradicts` relation is NOT routed to a graph edge).
- **Edge-currentness thread** -- `test_tombstone_resolves_contradiction_detector_goes_quiet`
  confirms the detector goes quiet post-tombstone end-to-end.
- **Cold-start byte-identical** -- guarded by tests; 3b + cold-start tests
  unchanged.
- **Provenance fallback** -- `test_resolver_fallback_for_no_sidecar_edges` +
  the 3b anomaly tests (injector-planted, no sidecar) pass unchanged.
- **Conservative adjudication** -- only `fix` + `supersede_assertion` +
  `forgetting_enabled` + `decider_active` mutates; everything else record-only.
  The `--no-bonsai` escape hatch gates the contradiction loop too (fixed during
  de-wonk). No fabricated Bonsai decision (live + FakeDecider None-path honest
  record-only).
- **No TODO/stub** -- grep clean across the changed surface.
- **ASCII-only** -- CLI stdout reconfigured to utf-8 before printing the
  report (Bonsai reasoning may be non-ASCII); prompt + messages ASCII-only.
- **`.mbox` additive** -- dir-of-`.eml` + single-`.eml` byte-identical
  (`test_mbox_parse_matches_parse_messages`); thread core untouched.
- **Eval honesty** -- deterministic-only recall threshold is the normalizer
  ceiling; pair 5 (paraphrased-only) is an honest miss, asserted as such.
  Fixed committed fixtures (no network at test time); source + license in the
  fixture README. We do NOT claim the full bench's LLM-judge metrics.

## 5. Risk register

- **Deterministic ceiling**: paraphrased conflicts are Bonsai's job. The
  eval honestly documents this; the Bonsai path is wired and live-verified
  but the LoRA fine-tune (deferred) would sharpen it.
- **Best-effort citation resolution**: `find_document_by_title_or_url` is
  O(N_docs) per encode + per citation; fine for a small corpus, a future
  title-index can replace the scan for large corpora with many docs AND
  episodes.
- **RMW sidecar**: `_assertion_edge_ops` / `supersede_assertion` are
  read-modify-write (single-threaded consolidation -> consistent); a future
  concurrent-writer path would need batching.

## 6. Handoff / Definition of Done

A2 (the dormant anomaly-resolver) is **UNBLOCKED**: the production writer of
entity `state` edges exists (deterministic + Bonsai), value -> episode
provenance rides on the edge sidecar, the fact-level tombstone resolves the
contradiction, and citation is the provenance backbone. The fact-level
tombstone refines the prior episode-level path (which stays for
`identity_drift` + `reconsolidate`). `.mbox` parsing and the offline
EnterpriseRAG-Bench eval are in-slice. The Bonsai LoRA fine-tune on
contradiction decision pairs + the full bench LLM-judge harness are deferred
(Section D7).