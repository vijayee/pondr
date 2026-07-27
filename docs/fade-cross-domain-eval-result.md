# Cross-domain fade eval -- result (task #37, DONE 2026-07-26)

> Closes the honest gap from #36 (`docs/fade-bible-eval-result.md`): R4 (forgotten)
> was validated only on the SYNTHETIC embedder in the unit tests, never on real bge
> on a real session. This eval exercises R4 on real bge via a cross-domain session.
> Stage-1 fade module: `src/subconscious/fade.py` (`docs/fade-stage1-result.md`).
> Design: `docs/fade-architecture.md`. Result memory: `pondr-fade-cross-domain-eval-result`.
> Committed `bc22aff` on main (NOT pushed).

## What ran

`scripts/eval_fade_cross_domain.py` ingests a few Bible verses (domain A -- john 1)
as probe anchors, then streams many ERAG technical docs (domain B -- confluence/
runbooks) past them, probing at an erag-step grid `[0,1,2,4,8,16,32,64,128,192,256]`.
The SSM-A state drifts to domain B; the Bible anchors' `cos(state, anchor)` drops
with the fade. R4 triggers when it drops below `cos_gist`. Real bge-small-en-v1.5;
passthrough voice; R2 off (#36 decision). CPU, ~10s.

- 8 Bible anchors (john 1), 300 ERAG chunks, `decay=0.99, cos_ring=0.95,
  cos_gist=0.40, ring_capacity=32`.

## 4 gates -- all PASS

1. **R4 triggers on REAL bge** -- >=1 Bible anchor reaches R4 at some erag-step. PASS.
   (The key gate #36 could not exercise.)
2. **R4 no-confabulation** -- every R4 recall content == "[forgotten]", voice not
   called. PASS.
3. **graceful** -- >=1 anchor transitions R1 -> R3 -> R4 in order (R3 before R4). PASS.
4. **R1 exact (sanity)** -- step-0 anchors R1 with content == verse text. PASS.

**VERDICT: PASS.** Regimes observed: R1-verbatim, R3-gist, R4-forgotten. Cross-domain
cos floor 0.369.

## THE KEY CALIBRATION FINDING (load-bearing)

**Real bge-small has a HIGH cosine floor** -- NOT the ~0.1 one might expect (bge-small
lives in a narrow cone): same-domain ~0.6, cross-domain (Bible <-> technical runbook)
~0.37 (measured here). The probe-#32 default `cos_gist=0.30` was calibrated on the
SYNTHETIC test embedder, whose cross-doc floor is ~0.01 (`_StubEmbedder` with
`within_doc=0.9` -> cross-doc `(1-0.9)**2 = 0.01`). So 0.30 sat BELOW real bge's
cross-domain floor (~0.37) and **never reached R4 on real bge** -- the unit test's
R4 did not transfer to real bge (the synthetic-vs-real gap). At `cos_gist=0.30` this
eval FAILS (cross-domain floor 0.369 > 0.30); at `0.40` it PASSES.

**Re-calibration: `cos_gist` 0.30 -> 0.40**, sitting BETWEEN the two real floors:
same-domain (~0.6) -> R3 (fuzzy gist), cross-domain (~0.37) -> R4 (forgotten). The
router threshold is DOMAIN-FLOOR-DEPENDENT and must be calibrated on real bge, not
the synthetic embedder. Applied to `FadeConfig.cos_gist` (the production default;
`src/subconscious/fade.py`). The unit tests OVERRIDE `cos_gist` (0.15-0.20, calibrated
for their synthetic ~0.01 floor), so they are UNAFFECTED -- 17 fade + 40 neighbor
tests still pass. The Bible eval (#36) `--cos-gist` default also aligned to 0.40; its
verdict is UNCHANGED (Bible floor ~0.6 > 0.40, still R1+R3).

## What this proves end-to-end

- **R4 (forgotten) works on real bge** on a real cross-domain session: the fade
  EMERGES from state compression (the state drifting to domain B), and the free
  cosine router (`e = 1 - cos(state, anchor)`, `docs/fade-stage1-result.md`) drops
  Bible anchors below the threshold -> R4 with content "[forgotten]", no
  confabulation. The full 4-regime dispatch (R1, R3, R4; R2 off) is now validated on
  real bge, not just synthetic.
- The router's same-domain/cross-domain separation is real and tunable via
  `cos_gist`: same-topic -> fuzzy gist (R3), different-topic -> forgotten (R4). This
  is the architecture's intended behavior (`docs/fade-architecture.md`) on a real
  session.
- The fade is graceful: R1 (ring, verbatim) -> R3 (faded-but-retrievable, sibling
  blurb) -> R4 (forgotten) as the state drifts across the domain boundary. No cliff.

## De-wonk (CLAUDE.md): 3 rounds, clean

Round 1: docstring overclaim fixed (it predicted cross-domain cos ~0.1-0.3; the
actual measured floor is ~0.37 -- the overclaim implied 0.30 would work, which it
doesn't); `PassthroughVoice` rename (was `_PassthroughVoice` in the bible eval,
imported across modules as a `_`-prefixed name -- now a public shared eval utility).
Round 2: stale `cos_gist=0.30` reference in the docstring made threshold-agnostic
(Bible floor ~0.6 is above both 0.30 and 0.40). Round 3: docstring-only, no new
issues. No stubs in production (passthrough voice is a documented eval choice; the
embedder, Bible fetch, ERAG load, and `FadeMemory` dispatch under test are all
real); no TODO/FIXME; no disabled code.

## Also committed this session

`fd7898d` -- `docs/fade-stage1-result.md` (the #34 docs note, created last session but
omitted from `a058e29`; tracked now).

## NEXT

- **#35 (Stage 2, deprioritized):** `CrossSlotTransformerZHead` (KEPT from the STRM
  burn) as the Regime-2 readout + JEPA-fade (recency-as-prediction-horizon). Thin per
  probe #31.
- **Wire `FadeMemory` into `serve_ponder`** (gated, now unblocked by #34 + #36 + this
  eval). Use `cos_gist=0.40` (the real-bge-calibrated default).
- Upgrade SSM-A from EWMA to a trained `SelectiveSSM` (selective gating should extend
  the exact window beyond N=0; the ring covers the ring window regardless).
- A multi-domain-session eval (3+ domains) to confirm the router separates each pair
  at one threshold (bge-small's narrow-cone floor may make a single global cos_gist
  tight; per-domain or adaptive thresholds are a possible follow-on).

## Constraints honored

ERAG/Bible public only (OEB-US via bible-api.com, NOT copyrighted CJB; ERAG public
corpus read locally from a gitignored parquet -- NOT committed, never re-distributed);
no onyx, no private transcripts. bge-small is a frozen open model. Passthrough voice =
no trained ckpt loaded. Commit on main per `commit-at-will` (no Co-Author, ASCII, no
push unless asked, never commit untracked data/scratch -- `data/probe/fade_cross_domain*/`
output is gitignored, NOT committed; the ERAG parquet under `scripts/_scratch/` is
gitignored, NOT committed). De-wonk at completion (CLAUDE.md). No FAIL ckpt (this is a
PASS eval, no training).