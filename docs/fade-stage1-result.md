# Stage-1 fade -- build result (task #34, DONE 2026-07-26)

> The build record for the Stage-1 fade module. Design: `docs/fade-architecture.md`.
> Gating probes: `scripts/probe_verbatim_reach.py` (#31), `scripts/probe_vector_carry.py`
> (#32). Result memory: `pondr-fade-stage1-result`. Committed `a058e29` on main (NOT
> pushed).

## What shipped

`src/subconscious/fade.py` (443 lines) + `tests/test_fade.py` (17 tests, all green).
Isolated: no orchestrator/runtime/serve changes.

### The pieces

- **`VectorCarrySSM` (SSM-A)** -- the 384-d bge-space EWMA vector channel (the fade
  leg): `state_p = decay*state_{p-1} + write_gate*bge(chunk_p)`, `state_0=0`. `query()`
  L2-normalizes in float64 (probe #32's control fix for underflow at extreme
  `decay**N`). Structured to upgrade to a trained `SelectiveSSM` via a `step()`
  subclass (Stage-2 follow-on).
- **`BlurbStore`** -- external (out-of-state) numpy cosine store of
  `(bge(chunk), blurb_text)` keyed by `anchor_id`. Blurbs are RETRIEVED by the faded
  vector, not decoded from the state (the 3x-disproven path).
- **The free cosine router** -- `e(i,t) = 1 - cos(state_t, bge(anchor_i))`. Because
  SSM-A's state is a blend of bge vectors in the anchor's own 384-d space, cosine is
  BOTH the retrieval score AND the recoverability signal. No ridge decoder, no
  trained head, no training. The Sec-6.1 recovery-decoder recipe (`e = ||D(state) -
  anchor||^2`, AUC 0.81) realized directly by cosine -- the payoff of the bge-space
  channel decision (Option 2). Probe #32 showed this cosine tracks the fade: 1.0
  (N=0) -> 0.95 (N=1) -> ~0.8 plateau (tip-of-tongue floor).
- **4 regimes** (per-anchor, dispatched in `recall_anchor`):
  1. VERBATIM -- `anchor in ring OR cos >= cos_ring` -> exact blurb text. The ring
     (`deque[int]`) gives true verbatim independent of the state's fade.
  2. FILL -- `regime2_enabled AND cos >= cos_gist` -> labeled FILL. Default OFF
     (deprioritized per probe #31). Without the Stage-2 `CrossSlotTransformerZHead`,
     degrades to the same retrieve+expand R3 does (honest-degraded, not a stub).
  3. GIST -- `cos >= cos_gist` -> faded state retrieves a (possibly sibling) blurb
     -> SSM-B expands it.
  4. FORGOTTEN -- `cos < cos_gist` -> `"[forgotten]"`, no confabulation (the graceful
     tip-of-tongue floor).
- **`TokenLMVoice` / `load_token_lm_voice` / `bge_embedder`** -- real production
  seams (trained `SSMLanguageModel.generate`; frozen bge-small-en-v1.5). `Embedder`
  / `Voice` are injected `Protocol`s so the unit test runs CPU-only with synthetic
  vectors + a test-double voice.
- **`FadeMemory`** -- `ingest` / `recall_anchor` / `recall(query)` / `reset`.
- **`FadeConfig`** defaults from probe #32: `decay=0.99, cos_ring=0.95,
  cos_gist=0.30, ring_capacity=32, blurb_chars=600, expand_tokens=64,
  regime2_enabled=False`.

## Test status

- `tests/test_fade.py`: 17 pass (CPU, self-contained, no bge/torch). Covers SSM step
  math / reset / zero-state safety / dim rejection; BlurbStore retrieve / lookups /
  empty; the free router (recent > older); R1 immediate (cos~1.0, no voice); regime
  sweep (R1->R3->R4 + voice for R3); R4 no-confabulation; R2 off->R3, R2
  on->intercepts; ring eviction -> cos_ring verbatim; query recall routing; reset;
  unknown anchor.
- `test_subconscious` + `test_working_memory` + `test_recoverability_head_wiring`:
  40 pass, no regression.

## De-wonk (CLAUDE.md): 2 rounds, clean

Round 1: R2 returned the anchor's EXACT blurb as "fill" (misleading -- R2 is a fuzzy
Transformer reconstruction). Fixed: R2 shares `_retrieve_and_expand` with R3
(honest degraded behavior, REGIME_FILL label). Round 2: unused `import math`
removed. No CRITICAL/HIGH/MEDIUM; no TODO/FIXME; no production stubs; no disabled
code.

## NEXT

- **#35 (Stage 2):** `CrossSlotTransformerZHead` (KEPT from the STRM burn) as the
  Regime-2 readout + JEPA-fade (recency-as-prediction-horizon). Deprioritized (thin
  per probe #31).
- **#36 (eval):** 4-regime fade eval on Bible OEB-US chapter-as-session (via
  bible-api.com, NOT copyrighted CJB).
- Upgrade SSM-A from EWMA to a trained `SelectiveSSM` (selective gating should extend
  the exact window beyond N=0; the ring covers the ring window regardless).
- Wire `FadeMemory` into `serve_ponder` (gated, now unblocked by #34).

## Constraints honored

ERAG public only; no onyx, no private transcripts. bge-small is a frozen open model
(SSM-A's vector; no teacher LLM). The token-LM is the frozen voice (SSM-B). Commit on
main per `commit-at-will` (no Co-Author, ASCII, no push unless asked, never commit
untracked data/scratch). De-wonk at completion (CLAUDE.md). A FAIL's ckpt is not
uploaded; this is a PASS-equivalent build (no training, no ckpt to upload).