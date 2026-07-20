# Path (b): Bonsai Independence Plan (DEFERRED)

Date: 2026-07-18
Status: **DEFERRED plan, not active work.** Written so the work is
reproducible if/when it is taken up. Decision context (user, 2026-07-18):
*do not optimize pieces before assessing what the total package can even do
via the roadmap.* This doc captures the surface, the strategy, and the
verification probe so a future implementer does not re-derive them.

Goal: remove the runtime dependency on the GPU-served Bonsai 8B
(`localhost:8080/v1`, `llama-server`) so the runtime is **CPU-capable and
endpoint-optional**. This is the second of two levers for CUDA-independent
deploy (the first being "serve on `ReferenceSSM`, not Mamba3" — see
`docs/training-guide.md` §7).

---

## 0. Strategic note (why this is deferred)

Path (b) optimizes a subsystem (the contradiction/ingest path) before the
total system's capability is assessed against the roadmap. The right
sequence is: (1) walk the roadmap, (2) measure what the whole package does
end-to-end, (3) *then* decide whether Bonsai-independence is the right
optimization to spend on. This plan is recorded so the analysis is not
lost. **Do not start implementation from this doc without that upstream
check.**

The one exception worth doing regardless of strategy: the **safety wrap in
§3.1** — making "Bonsai unreachable" a safe, tested runtime state. That is a
correctness fix, not an optimization, and it de-risks every roadmap demo
today. Everything else is deferred.

---

## 1. The live Bonsai HTTP surface today

Seven live roles call `config.bonsai_endpoint` (`http://localhost:8080/v1`,
`src/config.py:61`) at runtime. An eighth (`classify_doc_kind`) is already
replaced by the local DocKindHead and kept only as third-tier fallback.

### 1.1 Consolidation dream pass — `BonsaiDecider` (`src/gnn/bonsai_decider.py`)

All five roles POST through `_post_json` (`src/gnn/bonsai_decider.py:300-336`),
which returns `None` on any HTTP/parse failure (never raises).

| Role | Method | Call site | Notes |
|---|---|---|---|
| Adjudication | `decide_contradiction` (218-266) | `src/gnn/consolidate.py:1030` | 3-way fix/ask_user/dismiss. Deterministic guards run first (see §2.1). On `fix`+`supersede_assertion`+`old_value!=new_value`: `store.supersede_assertion(...)` at `consolidate.py:1080` (the FACT tombstone). |
| Gist | `gist` (140-158) | `consolidate.py:738` | Free-text abstract synthesis from source episodes. |
| Ontology promotion | `verify_typing` (160-188) | `consolidate.py:902` | "Is this entity really class C?" adjudication. |
| Anomaly disposition | `decide_anomaly` (190-216) | `consolidate.py:956` | identity_drift fix/ask_user/dismiss. |
| Doc-kind tag | `classify_doc_kind` (268-296) | (ingest) | **REPLACED** by `DocKindHead`; Bonsai is fallback only. |

Wiring: `scripts/run_consolidation.py:90-101` (`_build_decider`); `--no-bonsai`
sets `cfg.bonsai_decider_enabled=False` (`run_consolidation.py:231-279`).

### 1.2 Ingest / live-encode — `BonsaiRelationExtractor` (`src/encoding/bonsai_relations.py`)

`_post` (`src/encoding/bonsai_relations.py:255-295`) **raises `RuntimeError`**
on failure (unlike the decider's `None`). `extract` (233-253) dispatches on
`config.bonsai_isolation_extraction` (`src/config.py:162`, default `False`):
single-pass `_extract_single` (1 HTTP call, `BONSAI_RELATION_PROMPT`, cap 6)
or 10-pass `extract_isolated` (303-336, one call per class in
`ISOLATION_CLASSES` 80-115).

Call sites:
- `src/encoding/encoder.py:128-142` (`_extract_relations` — catches
  `RuntimeError` → `[]`; called by `encode_messages` 236-237 and
  `encode_messages_fill` 290-291).
- `src/encoding/distill_worker.py:100-117` (async-distill background worker).
- `src/ingestion/pipeline.py` (doc-level extract — **wrapped** try/except →
  `[]` as of the Phase 1c-3c hardening; see §3.1).

Extracted `has_state` triples merge with the deterministic normalizer in
`extract_state_assertions` (`src/encoding/assertion_extractor.py:229-275`)
and are written by `store.record_state_assertion`
(`src/memory/store.py:1006-1032`).

### 1.3 Query path — `src/runtime.py`

- `ModeAGenerator` (`src/runtime.py:118`, `_complete` at
  `src/generation/mode_a.py:209-252`) — the conversational answer LLM. Raises
  `RuntimeError` on failure; the orchestrator catches it and returns `""`
  (`src/orchestrator.py:394-396, 419-421`). `mode_a` is injectable for tests.
- `BonsaiQueryPlanner` (`src/runtime.py:109`, `src/retrieval/query_planner.py:265-336`)
  — `plan()` (295-305) catches `RuntimeError` and falls back to
  `plan_rule_based`. **Already self-degrades.**

---

## 2. What is already local / deterministic

### 2.1 Deterministic guards before the adjudication HTTP call

`_deterministic_non_conflict` (`src/gnn/bonsai_decider.py:376-487`):
- **Guard 1 (equal values → dismiss)**, 436-447: `len(set(vals))==1` →
  `{"decision":"dismiss","action":"no_action"}`.
- **Guard 2 (complementary temporal → ask_user)**, 449-486: both sources
  `point_in_time_snapshot` (semantic, via `doc_kind`) OR both paths carry a
  month-name prefix → non-mutating `ask_user`.

These are correct-by-construction and skip the HTTP call when they fire. The
remaining LLM-only adjudication case is "different_entity / real conflict"
discrimination.

### 2.2 Deterministic extraction normalizer

`extract_state_assertions` (`src/encoding/assertion_extractor.py:229-275`) is
pure-regex, no IO: scans `key: value`, `key = value`, `key is [now] value`,
change-verb patterns, and lifts `has_state` triples from Bonsai's `relations`
list (263-273). It runs **even when Bonsai returns `[]`**, so extraction
degrades to a deterministic ceiling (structured field-style state claims).
This is the honest cold-start documented in `docs/Phase 3c.md` D8.

### 2.3 The no-decider episode-supersede path (pre-3c, still present)

When `decider_active` is False, `src/gnn/consolidate.py:814-840` falls through
to the OLD 3b resolver `_resolve_contradictory_state` (833) → coarse
whole-episode supersede via `SemanticMemoryWriter.supersede_episode`
(no fact-level tombstone, no value→episode provenance). This is what runs
under `--no-bonsai` today.

### 2.4 The anomaly detector is structural, not adjudication

`_detect_contradictory_state` (`src/gnn/anomaly_rules.py:249-262`) is a pure
rule (count distinct live `state` literals on one entity; >1 → flag). The
Phase 3a GNN anomaly head supplies the `score` the
`contradiction_resolve_threshold` gate checks — it does **not** adjudicate
real-vs-non-real; that is Bonsai's job.

---

## 3. Graceful-degradation reality + the one safety fix to do now

The runtime **already survives Bonsai being unreachable**:
- Extraction (encode path) → `[]` → deterministic normalizer still fires.
- Decider → `None` → record-only, no mutation.
- Query planner → rule-based fallback.
- ModeA → `RuntimeError` caught → empty answer, process stays up.

So path (b) is about **quality** without the GPU LLM and about not *shipping*
`:8080` as a requirement — not about preventing crashes.

### 3.1 Safety fix (do regardless of strategy) — the one hard-failure window — DONE

`src/ingestion/pipeline.py` previously called `relation_extractor.extract(...)`
**without a try/except**, while the encode path (`encoder.py:128-142`) did
catch. A Bonsai that is up at construct time but drops mid-ingest raised
`RuntimeError` and propagated.

**DONE (Phase 1c-3c hardening):** mirrored `encoder.py:128-142` — the ingest
`extract` call is wrapped, degrades to `[]` on any `Exception` (logs
`[bonsai-fail]` to stderr), and the deterministic normalizer still runs. This
matters more now that `bonsai_isolation_extraction` defaults ON (ingest hits
Bonsai 10x/doc, ~22.8 s/doc) — a transient server error or unparseable JSON
no longer drops the whole ingest. "Bonsai optional/unreachable" is now a safe
runtime state.

---

## 4. Replacement strategy per role

A 19.5M JGS head on the frozen backbone can do **narrow classification**
(the DocKindHead precedent, §6) but **cannot do free-text generation**. Roles
split accordingly:

| Role | Strategy | Effort | Labels |
|---|---|---|---|
| `decide_contradiction` | JGS specialist (`common_sense_resolver`, declared) + expand deterministic guards | Medium | **Already exist** (`contradiction_pairs.jsonl`, structural) |
| `verify_typing` | Deterministic graph-structure check, or JGS head | Low | derivable from graph |
| `decide_anomaly` | JGS 3-way head (like adjudication) | Low–Medium | needs labeling |
| `gist` | **Deterministic salience-pull** using the existing GNN salience head (top-k salient sentences) — not a generator | Low | none (reuses trained head) |
| `has_state` extract | **New extraction head** (pointer/BILOU over backbone outputs) + `build_assertion_extractor` policy | High | distill from Bonsai 10-pass / Oracle-label |
| `ModeAGenerator` | **Decouple the endpoint** — endpoint-agnostic (remote/cloud LLM or CPU `llama.cpp` model); stays an LLM | Low | n/a |
| `BonsaiQueryPlanner` | Make rule-based the default; Bonsai optional | Trivial | n/a |

### 4.1 The eval-driven priority (important)

The EnterpriseRAG-Bench harness found the **decider is sound at scale**
(200/200, 0 false-fix) but **extraction is the bottleneck** (~50% recall;
most "collisions" are spurious). The headline finding: *invest in
extraction, not the decider.* So:
- The **decider replacement is the most shovel-ready** (labels exist, narrow
  3-way) but gives the **marginal** quality gain.
- The **extractor replacement is the hard one** (new head architecture,
  labeling cost) but gives the **real** quality gain and removes the most
  expensive runtime call (22.8s/doc isolated, or 1 call/doc at every
  ingest/encode single-pass).

Do not let the shovel-ready decider displace the higher-value extractor in
the sequencing.

### 4.2 What a small JGS head categorically cannot do

Free-text generation (`gist` as a synthesizer, `ModeA` as an answer writer)
cannot be replaced by a 19.5M head in embedding space. Those roles need
either a **deterministic reformulation** (gist → salience-pull, reusing the
trained salience head) or a **decoupled LLM endpoint** (ModeA →
endpoint-agnostic). Path (b) is "remove the *local GPU-served* dependency,"
not "remove all LLMs."

---

## 5. Phased sequence (if/when implemented)

Ordered so each phase leaves the runtime runnable and Bonsai-optional-ish.

1. **Safety + Bonsai-optional switch (do first, regardless).** §3.1 wrap;
   add `--no-bonsai`/`bonsai_enabled=False` runtime config that constructs
   nothing Bonsai; verify the suite runs with Bonsai unreachable. *Low.*
2. **Local adjudicator.** Train `common_sense_resolver` on
   `data/training/bonsai/contradiction_pairs.jsonl` (adjudication half);
   expand deterministic guards (esp. the different-entity case where values
   are clearly disjoint); ship `build_contradiction_decider` (prefer head →
   Bonsai → guards). *Medium, shovel-ready — labels exist.*
3. **Deterministic siblings.** `verify_typing` via graph-structure; `gist`
   via the existing salience head; `decide_anomaly` as a 3-way head. *Low–Medium.*
4. **Decouple generation endpoint.** Make ModeA endpoint-agnostic. *Low.*
5. **Local extractor.** Extraction head + distilled/Oracle labels +
   `build_assertion_extractor`; deterministic normalizer stays as the floor.
   *High, highest payoff — the real bottleneck.*

After 1–4 the runtime is Bonsai-optional for everything except extraction
quality; after 5 it is fully independent.

---

## 6. The DocKindHead precedent (replicate this pattern)

The concrete "replace a Bonsai role with a local JGS specialist" pattern,
from the shipped doc-kind replacement:

- **Instance config** — declare an `INSTANCE_CONFIGS` entry
  (`src/subconscious/configs.py:96-100` for `doc_kind`, `lora_rank=4`,
  gate placeholder).
- **Head** — subclass `JGSInstance` (`src/subconscious/doc_kind_head.py:51-181`):
  `forward` steps the SSM over embeddings, pools per-section step outputs
  (mean or additive attention, `forward` at 115), applies a linear head;
  `classify` (183) is the no-grad serve entrypoint.
- **Tagger policy** — `build_doc_kind_tagger`
  (`src/ingestion/doc_kind.py:299-367`): prefer ensemble → single head →
  `BonsaiDocKindTagger` → `None`. Ship `EnsembleBackboneDocKindTagger`
  (240-296, logit-averaged; `DEFAULT_DOC_KIND_ENSEMBLE_PATHS` 67-70).
- **Call site** — `scripts/ingest_document.py:118-131` builds the tagger;
  `args.doc_kind_ensemble` defaults to the ensemble paths (191-192) so
  production serves the local head; Bonsai is third-tier fallback.

For a future adjudicator/extractor: declare the instance config, subclass
`JGSInstance` with the right head (classification for adjudicator; pointer/
BILOU for extractor), train against the labels Bonsai is currently producing
(or the structural pairs for adjudication), ship a `build_X` policy that
prefers the head → Bonsai → deterministic floor → `None`.

### 6.1 Declared-but-unwired instances that can absorb roles

`src/subconscious/configs.py:73-131`:
- `common_sense_resolver` (116-120) — natural home for the adjudicator.
  lora_rank=6, context_features `(ambiguity_magnitude, context_coherence,
  historical_frequency)`. No implementation file exists.
- `uncertainty_detector` (101-105), `disturbance_detector` (121-125) —
  candidates for `decide_anomaly`. Declared only.

---

## 7. Verification probe (when implemented)

Mirror the doc-kind `ensemble_serve_gate` probe pattern (uncommitted scratch,
`scripts/_scratch/`): run the full suite with `bonsai_endpoint` unreachable
and assert:
- Contradiction handling produces correct decisions via local adjudicator +
  guards (no `:8080` call).
- Extraction runs at the deterministic ceiling (structured `key: value`
  claims become `has_state` edges; no HTTP).
- Generation runs via the decoupled endpoint (or documented empty-answer
  fallback).
- No `requests.post` to `localhost:8080` from any runtime path.

Gate: the Bonsai-unreachable run matches the Bonsai-present run on the
deterministic cases and degrades gracefully (documented ceiling) on the
LLM-only cases.

---

## 8. Out of scope / decision triggers

- **Not in scope:** the GNN (GAT, independent of Bonsai); the Oracle
  (DeepSeek via Ollama — teacher only, already CPU-capable); the JGS
  backbone (ReferenceSSM, already CPU-capable).
- **Decision triggers for implementing this plan** (revisit after the
  roadmap assessment): (a) a deploy target that has no GPU; (b) the
  per-ingest/per-encode Bonsai latency becomes a measured user-facing
  bottleneck; (c) the Bonsai Qwen3-8B model's hosting cost/rate-limits
  become a constraint; (d) extraction quality is demonstrated to cap a
  roadmap capability that the total-package assessment cares about.
- **If implemented, also update:** `docs/training-guide.md` (add the
  adjudicator/extractor trainer entries), `docs/jgs-the-new-primitive.md`
  (add the new instances to the shipped list), and the relevant memory
  entries.

---

## 9. Pointers

- Live surface: `src/gnn/bonsai_decider.py`, `src/encoding/bonsai_relations.py`,
  `src/gnn/consolidate.py`, `src/runtime.py`, `src/generation/mode_a.py`,
  `src/retrieval/query_planner.py`.
- Deterministic floor: `src/encoding/assertion_extractor.py`,
  `src/gnn/bonsai_decider.py:_deterministic_non_conflict`,
  `src/gnn/anomaly_rules.py`.
- Precedent: `src/subconscious/doc_kind_head.py`, `src/ingestion/doc_kind.py`,
  `scripts/ingest_document.py`.
- Labels (adjudication, ready): `data/training/bonsai/contradiction_pairs.jsonl`
  from `scripts/generate_contradiction_training_data.py`.
- Training: `docs/training-guide.md` (Bonsai QLoRA §6; instance trainers §3–4).
- Memory: `pondr-bonsai-contradiction-guards-shipped`,
  `pondr-erag-bench-judge-harness` (the "extraction is the bottleneck"
  finding), `pondr-doc-kind-tagging-shipped`, `hippo-phase-3c-status`.