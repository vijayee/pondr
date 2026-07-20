# Phase 3c: Citation + Contradiction Detection

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

Phase 3c is the entity-state-assertion extraction path that closes A2, plus
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

## 7. Zero-shot Bonsai eval + the LoRA fine-tune decision (the D7 gate, closed)

D7 deferred the Bonsai LoRA fine-tune "by analogy" to `identity_drift`
(Bonsai zero-shot this slice). That deferral was supposed to hinge on a
judgement of how well Bonsai performs **without** fine-tuning -- a judgement
that had never actually been made. This section records that judgement from a
zero-shot eval run on 2026-07-15, and supersedes the by-analogy deferral with
an evidence-based one: **the fine-tune is warranted** on both axes.

### 7.1 Method

Fixture: `tests/fixtures/enterpriserag/semantic_pairs.json` -- 16 pairs, up
from the 5-pair deterministic-only `pairs.json`: **4 field-control** (F1-F4,
`key: value` / `key is value` patterns the deterministic normalizer catches),
**9 paraphrased** (P5-P13, mid-sentence claims the normalizer misses -- the
discriminating set), **3 negatives** (N14 complementary-temporal, N15
same-value-no-conflict, N16 different-entity-same-value). Harness:
`scripts/_probe_bonsai_zeroshot_eval.py` (probe, not committed; result in
`scripts/_scratch/bonsai_zeroshot_eval_result.json`). Three measurements:

1. **Extraction catch** -- does a route flag a collision (shared normalized
   entity + different values across old/new)? `det` (deterministic
   normalizer), `bonsai_strict` (Bonsai relations filtered to `has_state`/
   `state` -- exactly what the production `extract_state_assertions` lifts),
   `bonsai_relaxed` (Bonsai relations with ANY predicate; subject=entity,
   object=value -- the latent capability ignoring schema adherence).
2. **Adjudication** -- `decide_contradiction` is independent of the extraction
   schema (it takes a flag + `state_values` + provenance), so we adjudicate
   ALL 13 conflicts from ground-truth values, not just the Bonsai-caught
   subset. Correct = `decision=="fix"` AND `action` contains
   `supersede_assertion`.
3. **Negatives** -- feed each negative's values as a flag; a `fix`+
   `supersede_assertion` is a false auto-tombstone.

### 7.2 Results

| Measurement                                  | Result          |
|----------------------------------------------|-----------------|
| Deterministic catch (recall)                 | 4/13 (30.77%)   |
| Bonsai strict has_state catch (recall)        | **0/13 (0%)**   |
| Bonsai relaxed any-predicate catch            | 5/13 (38.46%)   |
| Schema-adherence gap (relaxed - strict)      | +38.46%         |
| Bonsai strict false-positives on negatives    | 0/3             |
| Bonsai relaxed false-positives on negatives   | 1/3 (N14)       |
| Adjudication correct (fix + supersede_assertion) | 12/13 (92.31%) |
| Adjudication returned-None (failure)         | 0/13            |
| Negatives false auto-fix (raw rubber-stamp)   | 3/3             |

### 7.3 Findings

**Finding 1 -- Bonsai zero-shot does NOT follow the `has_state` schema.**
The relation prompt lists `has_state(Entity, Value)`, but Bonsai 8B zero-shot
ignores it and emits freeform predicates (`is`, `uses`, `decides`,
`is going to be`, `runs on`). Strict has_state catch is 0/13. Because the
production encoder (`extract_state_assertions`) only lifts `has_state`/
`state` predicates from `episode.relations`, the shipped Bonsai assertion
arm contributes **nothing** in production -- Bonsai-enabled encode is
byte-identical to deterministic-only encode today. The semantic capability IS
latent (relaxed catch 5/13, +2 paraphrased over deterministic: P5, P13), but
it is unrealized because of schema non-adherence, not capability absence, and
it is noisy (1/3 relaxed false-positive on N14).

**Finding 2 -- Bonsai zero-shot adjudicates with high recall but is a
rubber-stamp on non-conflicts.** On the 13 ground-truth conflicts it returns
`fix + supersede_assertion` on 12/13 (P8 conservatively `ask_user`). But fed
a `contradictory_state` flag + two values, it auto-tombstoned 3/3 negatives.
This is a precision failure, not a recall one: it does not discriminate
non-conflicts. The realistic production-reachable negative is **N14**
(complementary-temporal: shared entity + different values, e.g. "database is
MySQL for prod / Postgres for staging" -- NOT a real conflict, but the
deterministic detector WOULD flag it, since it keys on shared entity +
different value). N15 (same value) and N16 (different entity) are not
detector-reachable, so they are a decider-robustness probe rather than a
production risk. The honest production risk is therefore **1/1 realistic
false-tombstone** (N14), with 3/3 as the raw rubber-stamp probe.

**Finding 3 -- the deterministic route IS leaving opportunities on the
table.** Deterministic catch is 4/13 (the 4 field-controls); it misses all 9
paraphrased conflicts. Bonsai-relaxed catches 2 of those (P5, P13). The
deterministic-only path is the honest ceiling it was documented to be, and
the paraphrased conflicts are exactly the opportunity Bonsai is supposed to
claim.

### 7.4 Decision (the D7 gate, now evidence-based)

**The Bonsai LoRA fine-tune is WARRANTED -- on both axes -- and the D7
"defer by analogy" deferral is superseded by this evidence.**

- **Extraction axis**: zero-shot Bonsai contributes 0 in production (schema
  non-adherence). The fine-tune teaches the `has_state(Entity, Value)` schema
  with consistent entity/value naming, unlocking the paraphrased-conflict
  catch the deterministic route cannot reach.
- **Adjudication axis (the stronger driver)**: zero-shot Bonsai rubber-stamps
  non-conflicts, including the realistic N14 complementary-temporal case ->
  a silent false fact-tombstone. This is the worst failure mode (a correct
  fact quietly marked superseded), and it is NOT cheaply filterable -- it
  requires the model to actually discriminate, which is the fine-tune's job.

A cheaper partial alternative exists for the extraction axis alone: a
**relaxed production filter** that accepts `is`/`uses`/`runs on`/
`is going to be` as state-like predicates would close part of the
schema-adherence gap without a fine-tune. It is NOT adopted as the path
forward because (a) it inherits the 1/3 relaxed false-positive rate, and (b)
it does nothing for the adjudication precision gap, which is the binding
problem. It remains available as a stopgap if the fine-tune is delayed.

### 7.5 Fine-tune scope (next slice, not this one)

Train Bonsai (LoRA on the 8B gguf, or the in-process path per
[[hippo-bonsai-local-server]] option A) on **entity-centered decision pairs**
regenerated with the 3b `identity_drift` extraction: (a) extraction pairs
that force the `has_state(Entity, Value)` predicate + canonical entity
naming on paraphrased assertions (the P5-P13 shape); (b) adjudication pairs
that include N14-style complementary-temporal / N15 same-value / N16
different-entity negatives so the decider learns to `dismiss`/`ask_user`
rather than rubber-stamp. The eval harness here
(`_probe_bonsai_zeroshot_eval.py`) is the before/after gate: a successful
fine-tune lifts strict has_state catch off 0 and drives the negative
false-fix rate to 0 while holding adjudication recall >= 12/13. The full
EnterpriseRAG-Bench LLM-judge harness (D7 deferred item 2) remains out of
scope; this 16-pair harness is the cheaper, sufficient gate.

### 7.6 27B zero-shot probe (the de-risk before the fine-tune, 2026-07-15)

Before committing to the LoRA fine-tune and to using the **27B as the
training-data generator**, the 8B's bigger sibling `Ternary-Bonsai-27B-Q2_0`
(6.83 GB Q2_0, served the same way on the 5080 via PTX JIT) was run through the
identical 16-pair harness to separate **capacity-bound** failure (8B too small;
27B does it) from **task-bound** (hard even at 27B). Server config matched the
8B run: `--reasoning-budget 0 --reasoning-format none
--chat-template-kwargs {"enable_thinking": false}`, `-c 16384`, `--top-p 0.85`
(client sends `temperature=0.1`, the one sampling param it controls). The
harness's `_parse_relations` (fence-strip + carve-outermost + salvage) parsed
both models identically, so the comparison is apples-to-apples.

Side-by-side (8B -> 27B):

| Metric | 8B | 27B |
|---|---|---|
| Deterministic catch (recall) | 4/13 | 4/13 (unchanged) |
| Bonsai strict `has_state` catch | **0/13** | **4/13** (F2, F3, F4, P7) |
| Bonsai relaxed any-predicate catch | 5/13 | 5/13 |
| schema-adherence gap (relaxed - strict) | +38.5% | +7.7% |
| strict FP on negatives | 0/3 | 1/3 (N14) |
| adjudication correct (conflicts, fix+supersede) | 12/13 (92%) | 9/13 (69%) |
| negatives false-fix (false tombstone) | **3/3** | **1/3** (N15, N16 dismissed) |

**Verdict: capacity-bound, not task-bound.** Three takeaways:

1. **The 27B emits `has_state` natively** (strict catch 0 -> 4, including the
   paraphrased P7 shape the 8B never reached). The 8B's schema defiance is a
   capacity shortfall, not a family/task limit. Fine-tuning the 8B toward
   27B behavior is therefore plausible, and the 27B is a viable
   `has_state`-emitting generator for Stage B's training pairs.

2. **The 27B discriminates non-conflicts** (negative false-fix 3/3 -> 1/3):
   it correctly `dismiss`es N15 (same value) and N16 (different entity), where
   the 8B rubber-stamped all three. So 27B-as-adjudication-label-source is
   viable.

3. **Two caveats survive at 27B**, and both shape the fine-tune:
   - **Adjudication recall dropped** (12/13 -> 9/13): the 27B is more
     conservative -- it `ask_user`s on 4 real conflicts (F1, P6, P9, P13)
     instead of auto-fixing. The fine-tune must keep the 8B's conflict-recall
     (12/13), not inherit the 27B's over-caution.
   - **N14 (complementary-temporal) still false-fixes at 27B** (the prod MySQL /
     staging Postgres shape). The 27B over-extracts complementary scopes as a
     `has_state` collision. So N14-shape negatives stay **load-bearing** in the
     fine-tune data, and 27B-as-judge cannot be trusted for that shape --
     planted/structural labels (the "no Oracle-as-judge" decision) carry it.

Net: the de-risk clears the fine-tune. Stage B uses the 27B as the generator
(emits `has_state`, discriminates N15/N16), targets the 8B to learn the
`has_state` schema + canonical naming while *keeping* its 12/13 conflict
recall, and leans on planted N14/N15/N16 negatives to drive the false-tombstone
rate to 0. Trainer + serving notes (the ternary merge-into-ternary is dead per
`ternative`; runtime-LoRA on the ternary Q2_0 base is the serve path; the Prism
fork's `llama-finetune.exe` is full-finetune-only so PEFT on the dense Qwen3-8B
is the trainer) are in the plan at `.claude/plans/mellow-jumping-token.md`.

### 7.7 Three-way comparison: 8B vs 27B vs DeepSeek v4 flash (2026-07-15)

After the 27B probe, the same 16-pair harness was pointed at DeepSeek v4 flash
through the local Ollama server (`BONSAI_EVAL_ENDPOINT=http://localhost:11434/v1`,
`BONSAI_EVAL_MODEL=deepseek-v4-flash:cloud`; the probe was made
endpoint/model-agnostic via env vars -- no production code change). Result
file `scripts/_scratch/bonsai_zeroshot_eval_result_dsflash.json` (probe not
committed).

| Metric | 8B | 27B | DeepSeek v4 flash |
|---|---|---|---|
| Deterministic catch | 4/13 | 4/13 | 4/13 |
| strict `has_state` catch | 0/13 | 4/13 | 0/13 |
| relaxed any-predicate catch | 5/13 | 5/13 | 0/13 |
| adjudication correct (fix+supersede) | 12/13 | 9/13 | **13/13** |
| negatives false-fix | 3/3 | 1/3 | 1/3 |

Two findings:

1. **DeepSeek's 0/13 extraction is an interface artifact, not a capability
   gap.** The production extractor sends `response_format: {"type":
   "json_object"}`; under that constraint DeepSeek v4 flash (via the Ollama
   cloud proxy) returns **empty content** (`finish_reason: stop`, zero tokens)
   on the decision-framed bodies (F1-F4, P5-P13), while it DOES extract
   status-framed text (N14 -> `has_state build green`). Without the
   `json_object` constraint it produces valid JSON including `has_state`
   (P5 -> `the team has_state MySQL`). So its latent extraction is fine and
   schema-adherent; the 0/13 is the `json_object`-constraint + this
   model/proxy misbehaving on decision-framed inputs. Through the production
   interface as-is, though, it is a no-op extractor.

2. **DeepSeek is the strongest adjudicator** (13/13, best of the three; 8B 12/13,
   27B 9/13) and, like the 27B, discriminates the easy negatives (dismisses
   N15/N16; only N14 false-fixes). The adjudication prompt also uses
   `json_object` but the model satisfies it -- the single decision object is an
   easier constrained target than a list of relations.

Implications for Stage B: DeepSeek v4 flash (already the preferred Oracle per
[[deepseek-flash-over-pro]]) is the strongest **adjudication-label** source
(13/13 + non-conflict discrimination), reinforcing its use there. As an
**extraction-shape** generator the 27B is preferable because it emits `has_state`
**through the actual production interface** (4/13 strict), where DeepSeek is
hobbled by `json_object`. N14 false-fixes across all three models (8B 3/3, 27B
1/3, DeepSeek 1/3 -- always the complementary-temporal case) -> the planted
N14-shape negatives stay load-bearing regardless of generator.

### 7.8 Dense-base probe: local qwen3:8b (2026-07-15)

Stage B's open decision was "verify Bonsai's dense base == Qwen3-8B-Instruct
(Prism post-training-quant only, or quant+extra-train)?" -- i.e. is the 8B's
zero-shot `has_state` non-adherence a Bonsai artifact or a Qwen3-8B base trait?
A locally-installed Ollama `qwen3:8b` (Q4_K_M, 8.2B -- the dense Qwen3-8B that
is the PEFT candidate) answers it directly. Ran via the same 16-pair harness
(probe `scripts/_probe_qwen3_zeroshot_thinkoff.py`, not committed; result
`scripts/_scratch/bonsai_zeroshot_eval_result_qwen3_8b_thinkoff.json`).

**Interface caveat (material).** The production extractor POSTs to
`/v1/chat/completions` with `response_format: {json_object}` and does NOT pass
Ollama's `think` flag; `qwen3:8b` defaults to **thinking-on**, and with
`max_tokens=768` the reasoning eats the whole budget -> `finish_reason=length`,
**empty content** (reproduced deterministically on P7: think-on 3/3 empty,
think-off 3/3 valid). `/v1` ignores a `think:false` passthrough (verified), so
driving `/api/chat` with `think:false` is the only way to get the fair number.
The 8B/27B Bonsai runs were thinking-off (`--reasoning-budget 0`), so
think-off is the like-for-like interface. Through the production `/v1`
interface as-is, qwen3:8b is flaky (first run 1/13 strict, 12/32 extraction
calls empty) -- an interface artifact, not a capability ceiling (P5 and P11
emit valid `has_state` JSON when they don't overrun). Think-off results:

| Metric | Bonsai 8B (ternary) | qwen3:8b (dense) | Bonsai 27B | DeepSeek flash |
|---|---|---|---|---|
| Deterministic catch | 4/13 | 4/13 | 4/13 | 4/13 |
| strict `has_state` catch | 0/13 | **0/13** | 4/13 | 0/13 (artifact) |
| relaxed any-predicate catch | 5/13 | 3/13 | 5/13 | 0/13 (artifact) |
| adjudication correct (fix+supersede) | 12/13 | **13/13** | 9/13 | 13/13 |
| adjudication None-fail | 0/13 | 0/13 | 0/13 | 0/13 | (omitted from prior table) |
| negatives false-fix | 3/3 | **3/3** | 1/3 | 1/3 |

**Two findings:**

1. **Dense qwen3:8b == Bonsai 8B ternary on the `has_state` schema (both
   0/13 strict).** The schema non-adherence is a **Qwen3-8B base trait**, not a
   Bonsai/ternary artifact -- Bonsai is quant-only (not quant+extra-train) on
   this task. This **resolves the Stage B open decision**: the PEFT base can be
   upstream Qwen3-8B-Instruct; there is no hidden Bonsai post-train to recover.
   (Minor divergence: dense qwen3 relaxed 3/13 vs Bonsai 8B relaxed 5/13 --
   Bonsai's post-train slightly *helps* latent extraction -- but both are 0
   strict, so it does not change the fine-tune-justified verdict.)

2. **Dense qwen3:8b is the most aggressive rubber-stamp adjudicator** (13/13
   decided, 13/13 correct on conflicts, but 3/3 negatives false-fix -- it
   fixes unconditionally, never `ask_user`/`dismiss`). Perfect conflict recall,
   zero non-conflict discrimination -- the same rubber-stamp shape as the
   ternary 8B (12/13 + 3/3) but more consistent. This confirms the binding
   driver is **adjudication precision, not capacity**: you cannot get
   non-conflict discrimination from the 8B at any quantization (dense or
   ternary both rubber-stamp); only 27B/DeepSeek discriminate, and only
   marginally (1/3). -> the fine-tune's planted-negative (N14/N15/N16) data is
   the load-bearing ingredient for the decider, exactly as planned.

Stage B implication: PEFT on upstream Qwen3-8B-Instruct is the correct base;
the fine-tune teaches a schema the 8B genuinely lacks (dense == ternary == 0)
AND a categorical non-conflict call it rubber-stamps at every quantization.
Note for the runtime: the production extractor would need `think:false` (or a
larger `max_tokens`) to serve qwen3:8b-class models reliably over Ollama --
moot for Stage B since the serve target is the ternary Bonsai via llama-server
(thinking already off), not Ollama qwen3.
### 7.9 Isolated 10-pass extractor + async-distill (the serve-time enabler, 2026-07-16)

The 8B `has_state` schema gap (7.2-7.8) has a zero-shot fix that predates the
fine-tune: **one focused single-predicate pass per class**, merged. The V1
merged prompt makes `has_state` race `decides`/`concerns`/`involves` for the
"at most 6" salience slots and loses; isolation removes the race -- one pass
per class, no competing predicates, no salience cap. Probed
(`scripts/_scratch/_probe_isolate_classes.py`, uncommitted) on the ternary 8B:

| metric | V1 merged | isolated 10-pass |
|---|---|---|
| strict `has_state` catch | 0/13 | **11/13** |
| classes that emit | 7/10 | **10/10** |
| negative false-fix (FP) | 3/3 | **0/3** |
| cost / doc | ~2.3 s (1 call) | **~22.8 s** (10 calls) |

So isolation closes the schema gap **zero-shot** -- the shipped Bonsai
assertion arm goes from a no-op to live without any fine-tune. The catch: 10
HTTP round-trips to the one 8B on `:8080` is ~22.8 s/doc, which is unacceptable
on the synchronous `query()` path (the user would wait 22 s for every reply).

**Async-distill (the architectural answer).** The 22 s extraction is moved off
the synchronous path so the response returns immediately:

```
main thread (query):  id = store.next_episode_id()          # counter is main-thread-only
                      ep  = encoder.encode_messages_stub(messages, id, ...)   # no extract, no store
                      store.encode_episode_content(id, ep)  # stub: content + vector index
                      encoder.last_episode_id = id          # follows chain
                      enqueue(worker, id, ep)  ->  return response
worker (DistillWorker): ep  = encoder.encode_messages_fill(ep, id)   # GLiNER + 10-pass Bonsai
                        store.encode_episode_edges(id, ep)           # fill: graph edges
```

Design (`/.claude/plans/async-distill-stub.md`):

- **Stub-then-fill.** `encode_episode` is split into `_content_ops` + `_edge_ops`
  (byte-identical: the sync path merges them into one `batch_sync`). The stub
  writes content + the vector index (immediately retrievable by embedding,
  graph-invisible); the worker fills the graph edges. Content-before-edges
  preserves the atomicity invariant -- never edges without content.
- **Pre-allocated id.** `next_episode_id()` is called on the main thread; the
  worker never touches the persisted counter, `last_episode_id`, or `session_id`
  (sidesteps the RMW race the counter's docstring warns about).
- **Single-worker FIFO** (`queue.Queue` + one daemon thread) so encodes never
  run concurrently -- avoids encoder-state races and keeps WaveDB writes
  serialized. A user who out-types 22 s/turn just queues; the queue drains; the
  user never blocks.
- **Foreground-priority yielding.** `query()` sets `foreground_busy` for the
  response-build duration; the worker installs a `pause_gate` on the encoder
  (`_extract`) + its Bonsai extractor (each isolated per-class call) and blocks
  while busy -- extraction runs only in the GAPS between turns, so a fast user
  never contends with the background for the one 8B. Residual TOCTOU: at most
  one in-flight extraction call (~2.3 s) can contend with a query that lands
  just after a gate check (the deliberate trade -- full mutual exclusion would
  make the foreground block on the worker's in-flight call).
- **Failure semantics.** A worker exception (Bonsai HTTP 500, GLiNER hiccup,
  store write error) is logged; the episode keeps its stub (content + embedding,
  no edges) -- vector-retrievable, just graph-thin. The queue survives; the next
  turn still encodes. Best-effort memory, not transactional (mirrors
  `_persist_exchange`'s "never lose the response").
- **Teardown.** `orch.drain(timeout)` stops accepting work, finishes in-flight +
  queued encodes, joins the thread; `serve_ponder.py` calls it in `finally`
  before `store.close()` so queued stubs get their edges while WaveDB is still
  writable. Idempotent + force-clears `foreground_busy` so a blocked worker can
  exit.

**Gating (both default ON as of the Phase 1c-3c hardening; overridable via
`serve_ponder --no-async-distill` / `--no-bonsai-isolation` or env):**
- `config.bonsai_isolation_extraction` -- the 10-pass isolated extractor (vs the
  V1 single-pass). Only viable behind async (22.8 s/doc).
- `config.async_distill_enabled` -- the background worker (vs the fused sync
  `encode_messages` path). The orchestrator constructs the `DistillWorker` in
  `__init__` only when this is on AND an encoder is wired.

Both flags off = the synchronous V1 path, byte-identical to pre-async. The
flags are independent but isolation-without-async would block the response
(`serve_ponder` warns and refuses to compose them silently). The defaults are
pinned by `tests/test_defaults.py`; the sync-path invariants (immediate
`follows` edge; foreground encode failure prevents persistence) are pinned by
`tests/test_orchestrator_persist.py`, which selects the sync path explicitly
now that the default is async. `gliner_timing` (the per-stage extraction
stderr log) also flipped default ON in the same hardening pass.

**Tests.** 15 new offline tests: store split lossless + graph-invisibility
(`test_store_stub_fill.py`, 4); encoder stub/fill contract -- fill never touches
the counter / `last_episode_id`, stub+fill matches fused
(`test_encoder_stub_fill.py`, 5); worker failure-isolation / drain /
foreground-priority yield / thread-isolation (`test_distill_worker.py`, 6);
isolated extractor dispatch + per-class normalization + degrade
(`test_bonsai_relations.py`, +3); end-to-end async stub-then-fill through the
orchestrator (`test_orchestrator_persist.py::test_async_distill_stub_then_fill_end_to_end`).
The V1 single-pass path is asserted byte-identical (one HTTP call, the merged
prompt, no predicate normalization).

**Relation to the fine-tune (Stage B).** Async-distill is the serve-time enabler
for the isolated extractor, NOT the fine-tune. The fine-tune (7.5) targets
adjudication precision (the binding driver, capacity-invariant at 8B per 7.8)
and is served as a runtime LoRA adapter (`llama-server --lora`, never merged
into ternary -- 7.5/ternative). The isolated extractor + async-distill ship
independently and are orthogonal to the adapter: they fix `has_state` extraction
zero-shot; the adapter fixes the decider's non-conflict discrimination.
