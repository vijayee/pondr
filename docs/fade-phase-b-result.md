# Phase B — feed `fade_recalls` into the LLM context (`--fade-inject`)

**Status: SHIPPED + live gate PASS** 2026-07-27. Commit `4837504` on main (NOT
pushed). The fade now reaches the user-facing response.

Phase A (commit `f486030`) wired `FadeMemory` into `serve_ponder` as
observability + ingest only — `fade_recalls` was surfaced in the result dict but
NOT fed into the LLM, so the response was byte-identical to flag-off. Phase B is
the payoff: a `[FADE MEMORY]` block built from the recalls is prepended to the
LLM user message on `synthesize` turns, so the fade is **visible to the user**
(the vision: "exact recall that fades into gists as the compression happens").

## Gating

New `--fade-inject` sub-flag (default OFF). Mirrors the STRM gated-rollout
pattern and the Phase A `--fade-memory*` template.

- `--fade-memory` alone → observability-only (Phase A, byte-identical to
  flag-off). A/B comparison preserved.
- `--fade-memory --fade-inject` → the `[FADE MEMORY]` block is prepended to the
  LLM user message on synthesize turns (the only end state that calls the LLM —
  `direct`/`format`/`extract` make no LLM call). The response is NO LONGER
  byte-identical to flag-off.

## The block

`format_fade_block(recalls)` (`src/subconscious/fade.py`) renders:

```
[FADE MEMORY -- your fading working memory of this conversation]
(Recent exchanges are recalled exactly; older ones are given as a fading
gist. What has fully faded is not listed.)
[verbatim, recent] <R1 content>
[gist, fading] <R3 content>
```

- **R1 (verbatim)**: the recent exact text (in-ring or state-fresh).
- **R3 (gist)**: the anchor's blurb (verbatim when `voice is None` — Phase B; a
  paraphrase once the token-LM voice leg is wired). Anchor-locked per the
  `9455795` fix: the anchor's OWN blurb, not the faded state's closest sibling.
- **R4 (forgotten) is SKIPPED.** R4 is a SIGNAL for a future long-term-memory
  pull (mechanism TBD — explicit tool call vs seamless background fulfillment),
  not LLM-facing content. The forgotten material is simply absent — the gradient
  "exact → gist → (gone)" reads naturally. R4 stays in `fade_recalls` as the
  structured signal the future pull will consume.
- Returns `""` when no content-bearing recalls are present (so an all-R4 turn
  omits the block entirely — byte-identical to flag-off).

## Injection site

The `_synthesize` closure in `src/orchestrator.py` (it already hand-builds
`user_content` and closes over the `fade_recalls` local built in the same
`query` scope). The block is prepended to the TOP of `user_content`, ahead of
the retrieved cross-session context. Best-effort: a render failure leaves
`user_content` unchanged (the turn proceeds), mirroring the recall seam's
swallow. The gate checks `self._fade is not None` first, so `--fade-inject`
with no `FadeMemory` is a no-op (no None deref).

## Live REPL gate (PASS)

`BONSAI_ENDPOINT=http://localhost:11434/v1 GENERATION_MODEL=qwen3:8b`, 7-turn
mixed-domain session (1 DB + 5 ML + 1 callback), `--fade-memory --fade-inject
--fade-debug --no-live-encode --fade-memory-ring-capacity 4 --fade-memory-decay
0.99 --fade-memory-cos-gist 0.40`.

- 7 synthesize + 1 direct, no `?` starvation (qwen3:8b terse responses avoided
  the planner-`?` dispatch that starved the cloud models in the Phase A run).
- **Regime gradient on anchor 0 (Postgres)**: cos
  `1.0 → 0.959 → 0.912 → 0.872 → 0.858`. R1 (verbatim) while in-ring (4
  anchors ≤ ring_capacity 4), then **R3 (gist) at 0.858 after eviction** (5
  anchors > capacity 4). The fade EMERGES from state compression on real bge +
  real LLM — the architecture's central claim, confirmed end-to-end with the
  block reaching the LLM.
- **R3 anchor-locked (the `9455795` fix holds on real serve traffic)**: at the
  callback, R3 content = the **Postgres** blurb (the anchor's own exchange),
  NOT the most-recent ML chunk. The Phase A finding (R3 content drift — state-
  retrieval returned wrong-topic "learning rate schedule" content) is FIXED:
  the state is the recoverability SIGNAL (cos → regime), not the retrieval key.
- With `--fade-inject`, the `[FADE MEMORY]` block (R1 ML verbatims + R3 Postgres
  gist) was prepended to the LLM user message; the callback response
  distinguished the cited topics — the LLM saw the fade.

Honest caveat: the ingested Postgres exchange itself contained no rationale
(turn-1 was the question + an assistant "no relevant results" non-answer), so
the R3 gist is honest about what was actually said — not a fade bug. The fade
correctly recalled and injected the Postgres content at R3; the content quality
is a function of what was ingested, not the fade.

## Tests

- `tests/test_fade.py::test_format_fade_block_renders_r1_r3_skips_r4` — R1+R3
  rendered with regime framing, R4 absent, R2 rendered, all-R4 → `""`, empty →
  `""`, empty-content → `""`.
- `tests/test_fade_serve_integration.py`:
  - `test_inject_off_messages_byte_identical_to_off` — `fade_inject=False` →
    LLM messages byte-identical to flag-off (Phase A contract preserved).
  - `test_inject_on_messages_contain_fade_block` — `fade_inject=True` → the
    LLM user message CONTAINS the `[FADE MEMORY]` block; absent in flag-off.
  - `test_inject_r4_only_no_block` — all-R4 recalls → no block, byte-identical
    to flag-off; R4 stays in `fade_recalls` as the signal.

28 fade + 41 regression pass (1 Bonsai live skip). De-wonked 2 rounds (dead
variable + weak-proof test fixed round 1; helper duplication hoisted round 2).

## Non-goals / follow-ons

- **R4 → long-term-memory pull**: when R4 is detected, pull the forgotten
  anchor's content from the persistent store and inject it. Mechanism TBD — a
  follow-on plan, NOT Phase B. Phase B only ensures the R4 signal is available
  in `fade_recalls`.
- **The voice leg** — load the token-LM via `--fade-memory-voice-path` so R3's
  `content` is a paraphrase (the actual gist) instead of the verbatim blurb.
  Independent of Phase B; the block renders either unchanged.
- **Promote the fade block into `format_for_llm`** if it later needs to reach
  the `format` end-state path or share the `[WORKING MEMORY STATE]` block's
  home — architectural cleanup, deferred.
- **Stage 2 / task #35** (R2 Transformer + JEPA-fade) — deprioritized per
  probe #31.