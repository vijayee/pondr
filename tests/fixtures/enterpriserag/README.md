# EnterpriseRAG-Bench -- "Conflicting Info" offline eval fixtures (Phase 3c D8)

This directory vendors a SMALL, fixed, COMMITTED subset of the
[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
"Conflicting Info" question category, distilled into the fields the
deterministic Phase 3c path consumes. It is loaded by
`tests/test_enterpriserag_eval.py` at test time -- **no network, no model, no
server**.

## What this is

EnterpriseRAG-Bench's "Conflicting Info" category is built from
near-duplicate document pairs where **a newer document supersedes facts from
an older one**. That structure is direct ground truth for the Phase 3c
contradiction -> fact-level-tombstone path, and the per-question
`expected_doc_ids` is citation ground truth.

We distill each pair into:
- `old_doc` / `new_doc`: `{title, source_path, body}` -- the body is crafted so
  the **deterministic normalizer** (`src/encoding/assertion_extractor.py`)
  extracts an explicit `entity -> value` field assertion from each (the shape
  Jira/Linear/Confluence status fields and config snippets take).
- `conflicting_entity`: the normalized entity key that MUST collide across the
  two docs (so `(E:entity, state, V_old)` + `(E:entity, state, V_new)` are
  both written -> the `_detect_contradictory_state` detector fires).
- `expected_doc_id`: the newer (authoritative) doc's id in a fresh store
  (`doc_000002`) -- citation ground truth.
- `catchable`: whether the deterministic normalizer can see the conflict.
  Pair 5 is `catchable=false` (a paraphrased-only conflict with no field
  shape) -- an **honest deterministic miss**, the documented ceiling.

## What this is NOT

- **Not the full bench.** 5 pairs, not the bench's 20 "Conflicting Info"
  questions.
- **Not the bench's scorer.** We do NOT run the bench's LLM-judge harness
  (correctness x completeness, three-judge consensus). We take its
  *labels/structure*, not its metrics.
- **Not the Bonsai path.** The Bonsai semantic adjudicator
  (`BonsaiDecider.decide_contradiction`) is exercised separately in
  `tests/test_contradiction.py` (via `FakeDecider`) and in the live dogfood
  step. This fixture is the **deterministic-only** ceiling.

## Source + license

- Upstream: `onyx-dot-app/EnterpriseRAG-Bench` (MIT-licensed).
- The pair *structure* (near-duplicate, newer-supersedes-older) and the
  `expected_doc_ids` citation concept are taken from the bench; the specific
  titles/bodies here are distilled fixtures crafted for the deterministic
  normalizer, not verbatim bench documents.

## Honest ceiling

The deterministic recall threshold in the test is set to the
deterministic-normalizer ceiling and documented in the test docstring. A
paraphrased-only conflict the normalizer cannot see is honestly counted as a
miss, not papered over. See `pairs.json` `caveat` + pair 5 `miss_reason`.