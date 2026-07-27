# Fade serve integration -- Phase A result (task #38, DONE 2026-07-27)

> Wires the validated fade architecture into the live serve path -- the ship
> milestone. The fade reaches `serve_ponder`. Stage-1 module: #34
> (`src/subconscious/fade.py`, `docs/fade-stage1-result.md`). Bible eval: #36
> (`docs/fade-bible-eval-result.md`). Cross-domain R4-on-real-bge + the
> `cos_gist` 0.30 -> 0.40 re-calibration: #37 (`docs/fade-cross-domain-eval-result.md`).
> Design: `docs/fade-architecture.md`. Committed `f486030` on main (NOT pushed).

## What shipped

`FadeMemory` is now wired into `serve_ponder` behind a gated, phased rollout
that mirrors the STRM template. **Phase A scope: observability + ingest, NO
behavior change to the user-facing response.** `FadeMemory` ingests each
(user, assistant) exchange per turn (building the SSM-A fade state over a
real session) and on each query runs `recall`, surfacing the routed recalls
as `result["fade_recalls"]` for observation. The recalls are NOT fed into the
retrieval/presentation/LLM-context flow -- the user-facing response is
byte-identical whether the flag is on or off. Phase B (feed recalls into
context) is a separate, later plan gated on Phase A observations.

The point of Phase A is to validate the free cosine router on REAL serve
traffic (the calibration was Bible/ERAG; a real conversation is the real
test) and to observe which regimes actually fire on a real session before
committing the fade into the LLM context.

## Gating

Gating = presence of the `FadeMemory` instance (mirrors the existing
`encoder: Optional[HippocampalEncoder] = None` DI pattern). No separate
config flag. `fade_memory=None` kwarg into `PonderOrchestrator.__init__`;
`None` = not wired = byte-identical to pre-fade. Every read gated with
`if self._fade is not None`. Best-effort: any failure in recall/ingest is
swallowed (the turn proceeds unchanged), mirroring `_run_salience_hook`.

## Design decisions (reuse-first)

- **Voice is optional** (`Optional[Voice] = None` in `FadeMemory.__init__`).
  When `None`, Regime 3 returns the retrieved blurb verbatim (the built-in
  passthrough -- no separate stub voice class). This lets Phase A run WITHOUT
  loading the token-LM: the token-LM checkpoint is on HF
  (`vijayee/pondr-models/token_lm/`), not local (`data/token_lm/` was deleted
  after upload). The full voice leg is a follow-on, gated on
  `--fade-memory-voice-path` + a local ckpt.
- **Embedder reuse: YES -- the one bge instance.** The single bge embedder
  built at `runtime.py` (`embedder = build_embedder(embedder_source)`) is
  passed straight into `FadeMemory(cfg, embedder, voice, dim=384)`. Do NOT
  call `fade.bge_embedder()` (it would load a second `SentenceTransformer`).
- **Ingest granularity: one chunk per exchange (joined user+assistant).** The
  fade advances one step per turn. With `decay=0.99` that is ~8-16 TURNS of
  gist window. Tunable via `--fade-memory-decay`.
- **In-process, per-orchestrator-instance state.** `FadeMemory` state
  accumulates across the REPL session. No persistence across restarts (a
  Phase B/C concern). `reset()` is not wired into session loading for Phase A.
- **`REGIME_NAME` moved into `fade.py`** next to the `REGIME_*` constants, so
  the orchestrator doesn't reach into `eval_fade_bible.py`. (The eval
  scripts' own `REGIME_NAME` uses `"R1-verbatim"` with the regime-number
  prefix for their table display; `fade.py`'s uses plain `"verbatim"` for the
  serve observability line -- intentionally different, not a duplication.)

## Seams (src/orchestrator.py)

1. **RECALL seam** -- after the salience-fired merge, before the WM-inject:
   `fade_recalls: list = []`; if `self._fade is not None`, run
   `self._fade.recall(user_prompt, top_k=self._fade_top_k)` and collect
   `{anchor_id, regime, regime_name, cos, content, blurb}` per recall.
   Swallowed on any failure. The recall runs BEFORE `_persist_exchange`
   (which ingests), so on query N the recall sees anchors from queries
   1..N-1 -- the correct order (the fade state reflects PAST exchanges).
2. **RESULT-AUGMENT seam** -- `if self._fade is not None:
   result["fade_recalls"] = fade_recalls`. Key ABSENT when off = byte-identical.
3. **INGEST seam** -- in `_persist_exchange`, at the top of the `try`, BEFORE
   the `encoder is None` early-return: hoisted `response = result.get(...)`
   + `has_response` check, then `if self._fade is not None and has_response:
   self._fade.ingest(f"User: {user_prompt}\nAssistant: {response}")`. Runs
   even when no HippocampalEncoder is wired (the fade is independent of the
   episode store). Swallowed on any failure.

## CLI (scripts/serve_ponder.py)

`--fade-memory` (store_true, default False), `--fade-memory-voice-path`,
`--fade-memory-tokenizer-path`, `--fade-memory-top-k` (5),
`--fade-memory-decay` (0.99), `--fade-memory-cos-gist` (0.40),
`--fade-memory-ring-capacity` (32), `--fade-memory-expand-tokens` (64),
`--fade-debug` (store_true, False).

- EXPERIMENTAL NOTE to stderr when `--fade-memory` (observability-only this
  phase; the voice leg requires `--fade-memory-voice-path` + a local ckpt).
- `--fade-memory-voice-path` without `--fade-memory-tokenizer-path` -> hard
  ERROR + return 1 (the voice leg needs its tokenizer).
- `--fade-memory-voice-path` pointing at a missing file -> hard ERROR +
  return 1 (matches the STRM ckpt-missing pattern; the alternative -- warn +
  run passthrough -- would say one thing and crash in `load_token_lm_voice`
  on the other, so it was made a hard error in de-wonk round 1).
- `[load] fade_memory=...` line mirrors the STRM load line.
- `--fade-debug` prints `[fade] {res.get('fade_recalls', [])}` to stderr after
  `_print_result(res)` in BOTH the one-shot and REPL paths. This is the Phase
  A observability mechanism. `fade_recalls` does NOT leak into
  `conversation_history` -- the REPL history is built from explicit
  `{"role","content"}` dicts, not the result wholesale.

## Tests

- `tests/test_fade.py`: `test_voice_none_passes_blurb_verbatim` -- the
  passthrough contract (`voice=None` -> Regime 3 returns the retrieved blurb
  verbatim, no `[expanded]`). Existing tests pass `_StubVoice()` and are
  unaffected by the now-optional voice. 18 fade tests pass.
- `tests/test_fade_serve_integration.py` (NEW, 5 tests):
  - `test_flag_off_no_fade_recalls`: `fade_memory=None` -> `"fade_recalls"`
    ABSENT (byte-identical).
  - `test_flag_on_surfaces_fade_recalls`: wired -> `fade_recalls` is a list;
    the first exchange is ingested (`len(fade.blurbs) == 1`).
  - `test_flag_on_response_byte_identical_to_off`: the REAL proof -- compares
    `mode_a.calls` (the messages passed to the LLM), not just the stub reply
    (which would be trivially identical since the stub ignores its messages).
    Uses separate db subdirs so the two runs have identical, non-leaking
    store state. The LLM saw the SAME messages with the fade on as off.
  - `test_fade_ingest_failure_does_not_break_query`: a `FadeMemory` whose
    `ingest` raises -> the query still returns a response (swallowed).
  - `test_fade_recall_failure_does_not_break_query`: a `FadeMemory` whose
    `recall` raises -> the query still returns a response and `fade_recalls`
    is present-and-empty (the recall seam swallows, the list stays `[]`, the
    result-augment seam still fires because `self._fade is not None`).

## De-wonk (CLAUDE.md): 2 rounds, clean

Round 1:
- Fixed a would-be-broken `--fade-memory-voice-path` missing-file warning:
  the warning said "FadeMemory will run WITHOUT the voice leg" but the path
  was still passed to `build_ponder`, so `load_token_lm_voice` would crash.
  Made it a hard ERROR + return 1 (matches the STRM ckpt-missing pattern).
- Removed dead attributes (`cfg`, `blurbs`, `ring`, `ssm_a`, `voice`) from
  the `_BrokenIngest`/`_BrokenRecall` test stubs -- the orchestrator only
  calls `.ingest()`/`.recall()` and fetches `REGIME_NAME` from the module.
- Fixed a wrong docstring: `test_fade_recall_failure_does_not_break_query`
  said `fade_recalls` is ABSENT, but it's present-and-empty (the result-
  augment seam fires whenever `self._fade is not None`).
- Strengthened `test_flag_on_response_byte_identical_to_off` to compare
  `mode_a.calls` (the LLM context) instead of just the stub reply, with
  separate db subdirs so store state doesn't leak between the two runs.
- Removed an unused `import pytest` (dead code).

Round 2: removed dead `import pytest`; made the first query in
`test_flag_on_surfaces_fade_recalls` a definite synthesize trigger ("Why...")
so the test isn't accidentally relying on a borderline end-state dispatch
to produce a response (the ingest only runs on a non-empty response).

No stubs/TODOs/disabled code in production; test doubles clearly marked. The
`voice=None` passthrough is real (built into `FadeMemory`, not a stub class
left in production); the embedder reuse is real (shared single instance); the
gating is real (`None` = off, not `if False`); the best-effort try/except
mirrors the existing salience pattern.

## Verification

- 23 fade tests pass (`tests/test_fade.py` + `tests/test_fade_serve_integration.py`).
- 41 regression tests pass (`test_orchestrator.py`, `test_runtime.py`,
  `test_serve_ponder_live.py`, `test_orchestrator_persist.py`,
  `test_orchestrator_forgetting.py`) -- 1 skip = no Bonsai server (expected).
- CLI smoke: `--help` shows the new `--fade-memory*` + `--fade-debug` flags;
  `--fade-memory --fade-debug --query "..."` runs without error and prints
  `[fade] [...]` to stderr; the same query without `--fade-memory` runs
  byte-identically (no `[fade]` line).
- REPL smoke (manual, the real Phase A validation): `--fade-memory --fade-debug`
  in REPL mode; ask related questions then a question about the first; observe
  `[fade]` showing `regime_name` transitioning over the session. This is the
  gate Phase A exists for (deferred to a live Bonsai session).

## NEXT

- **REPL validation on real serve traffic** (the gate Phase A exists for):
  run `--fade-memory --fade-debug` against a live Bonsai and observe which
  regimes fire on a real conversation (the cross-domain-calibrated
  `cos_gist=0.40` exercised on a real conversation, not Bible/ERAG). This
  validates the router on the real surface it was built for.
- **Phase B -- feed fade recalls into context** (its own plan): promote
  `fade_recalls` into the retrieval/presentation flow (append as synthetic
  `kind="fade"` episodes or a dedicated context block) so the fade actually
  reaches the user-facing response. Gated on Phase A showing the router
  behaves on real serve traffic.
- **The voice leg** -- load the token-LM via `--fade-memory-voice-path` once
  the ckpt is local (fetch from HF), so R3 expands the blurb via continuation
  instead of returning it verbatim. Independent of Phase B.
- **Upgrade SSM-A from EWMA to a trained `SelectiveSSM`** -- selective gating
  should extend the exact window beyond N=0; the ring covers the ring window
  regardless.
- **Persist fade state across process restarts** (WaveDB-backed blurb store +
  state) -- a Phase B/C concern; Phase A is in-process per-session.
- **`reset()` wiring** into session load / user switch -- deferred.

## Constraints honored

`src/` edits appropriate (extend `fade.py` minimally + thread a new optional
DI through `runtime`/`orchestrator`/`serve_ponder`, mirroring the STRM
template). No onyx, no private transcripts. bge-small is a frozen open model
(reused, not reloaded). The token-LM ckpt (HF, private `vijayee/pondr-models`)
is NOT required for Phase A (`voice=None` passthrough); loading it is a
follow-on. `cos_gist=0.40` (the real-bge-calibrated default from #37).
Isolated by construction: `fade_memory=None` -> byte-identical to pre-fade.
Commit on main per `commit-at-will` (no Co-Author, ASCII, `-m`, no push, never
commit untracked data/scratch). De-wonk at completion (CLAUDE.md). No FAIL
ckpt (this is a wiring task, no training).