# Fade serve validation -- Phase A REPL gate result (2026-07-27)

> The gate Phase A exists for: validate the free cosine router on REAL serve
> traffic (a real conversation, not the Bible/ERAG calibration). Wiring + the
> observability-only contract: `docs/fade-serve-integration-result.md`
> (task #38, committed `f486030`). Stage-1 module: `docs/fade-stage1-result.md`.
> Bible eval: `docs/fade-bible-eval-result.md`. Cross-domain R4 + the
> `cos_gist` 0.30 -> 0.40 re-calibration: `docs/fade-cross-domain-eval-result.md`.

## Verdict: GATE PASS (with one Phase B design input surfaced)

The free cosine router fires correctly on real serve traffic. Run against a
live Ollama `qwen3:8b` endpoint (`BONSAI_ENDPOINT=http://localhost:11434/v1`,
`GENERATION_MODEL=qwen3:8b`) over a 10-turn mixed-domain conversation (3 DB
turns + 6 ML turns + 1 callback), `--fade-memory --fade-debug` produced real
fade regimes on real bge embeddings and real LLM responses:

- **R1 verbatim fires** and returns the anchor's OWN exchange content (full
  Postgres-WAL text) while the anchor is in the ring -- no confabulation.
- **R3 gist fires** once the anchor is evicted from the ring but still
  recoverable (cos >= cos_gist).
- **Graceful monotonic fade**: anchor 0 (Postgres) cos trajectory
  `1.0 -> 1.0 -> 0.866 -> 0.808 -> 0.774 -> 0.756` (R1 verbatim across the
  in-ring turns, then R3 gist at the callback). The fade EMERGES from state
  compression on a real session -- the architecture's central claim,
  confirmed on real bge, not just the synthetic embedder.
- **Ingest is correctly gated**: only turns that produced a non-empty
  response ingested; empty-response turns (qwen3:8b "thinking" mode emitting
  only thinking tokens) were skipped with no crash -- the best-effort
  swallow held.
- **No crash, no regression**: the user-facing response was untouched (Phase A
  is observability + ingest only; `fade_recalls` is NOT fed into the LLM
  context).

This is the surface the router was built for. The gate it existed to pass is
passed.

## The finding: R3 content drift (a Phase B design input, NOT a Phase A bug)

Phase A is observability-only -- the drift does NOT affect the user-facing
response (byte-identical to flag-off). But it MUST be resolved before Phase B
feeds `fade_recalls` into the LLM context, so it is recorded here as the
single blocking design input for Phase B.

**Symptom.** At the callback ("why did I pick Postgres?"), anchor 0 (Postgres)
was recalled (query-relevance) and routed to **R3 gist, cos 0.756** -- but
the R3 `content` and `blurb` were the **"learning rate schedule"** text (the
most-recent ML anchor), NOT anchor 0's Postgres blurb. Hard evidence from the
`[fade]` line:

```
{'anchor_id': 0, 'regime': 3, 'regime_name': 'gist', 'cos': 0.7562,
 'content': "User: What learning rate schedule works best for transformer pretraining?\nAssistant: ...linear warmup followed by inverse square root decay...",
 'blurb':  "User: What learning rate schedule works best..."}
```

The recalled anchor is Postgres (the query picked it); the regime says gist;
the *content* is the wrong topic.

**Root cause** (`src/subconscious/fade.py`, `_retrieve_and_expand`):
Regime 3 retrieves `self.blurbs.retrieve(self.ssm_a.query(), k=1)` -- the
**faded state's** closest blurb. The state is an EWMA dominated by the most-
recent chunk (weight 1.0), so once cross-domain turns have entered, the state's
closest blurb is the most-recent (cross-domain) content, not a sibling of the
recalled anchor. The architecture's "the faded state retrieves a sibling of
the anchor" (probe #32) only holds **within a domain**: the Bible eval (#36,
same-domain) retrieved valid same-domain gist siblings. Across a mixed-domain
conversation -- the common serve case -- the state drifts and retrieves
wrong-topic content. This was not caught by #36 (same-domain) or #37 (R4-
focused); the serve REPL surfaced it.

**The asymmetry that makes it visible.** R1 (in-ring) returns the ring's
stored exchange (the anchor's OWN content) directly -- so R1 is anchor-locked
and correct. R3 (evicted) goes through `_retrieve_and_expand`, which is
state-retrieved -- so R3 is NOT anchor-locked and drifts. Anchor 0 returned
correct Postgres content at R1 for turns 2-9, then wrong-topic content at R3
on turn 10.

**Proposed fix (for Phase B sign-off -- this is a design change to R3
semantics, not a mechanical patch).** R3 means "vector still retrievable" --
the anchor's address survives. Retrieve the anchor's OWN blurb
(`self.blurbs.text(anchor_id)`) and let SSM-B expand it; the "fade" is
expressed through the SSM-B expansion (paraphrase/continuation = the faded
voice) and the regime label (lower confidence), NOT through retrieving a
state-closest sibling. The state's role becomes purely the recoverability
SIGNAL (cos -> regime), not the retrieval KEY. With `voice=None` (Phase A),
anchor-locked R3 returns the anchor's full blurb (same text as R1, labeled
gist/faded) -- honest: "I recall this, but less confidently." With the voice
loaded (the Phase B voice leg), SSM-B turns it into a paraphrase (the actual
gist). The one-line change:

```python
def _retrieve_and_expand(self, anchor_id, cos_i, regime):
    blurb = self.blurbs.text(anchor_id)          # anchor-locked, not state-retrieved
    if blurb is None:
        return Recall(anchor_id, REGIME_FORGOTTEN, cos_i, content="[forgotten]")
    if self.voice is None:
        return Recall(anchor_id, regime, cos_i, content=blurb, blurb=blurb)
    expanded = self.voice.expand(blurb, self.cfg.expand_tokens)
    return Recall(anchor_id, regime, cos_i, content=expanded, blurb=blurb)
```

This is the single input Phase B needs before `fade_recalls` can be promoted
into the LLM context. It is deferred to a user sign-off because it revises the
architecture's R3 retrieval semantics ("faded vector retrieves blurb" ->
"faded vector's cos determines the regime; the anchor's address retrieves the
blurb").

## R4 did not fire on this run (expected, consistent with #36/#37)

cos for anchor 0 bottomed at 0.756 -- well above `cos_gist=0.40`, so R4
(forgotten) never triggered. Only ~5 ML anchors ingested (qwen3:8b produced
empty responses on ~4 turns), and real bge-small's high same-domain floor
(~0.6) plus the single DB anchor kept the DB component of the state
recoverable. R4 on real bge was already validated in #37 (the cross-domain
eval, Bible + ERAG) with enough cross-domain push; this run did not have
enough cross-domain mass to reach it, which is consistent, not a regression.

## Cloud-model attempts (the user asked to switch off the slow local model)

Three cloud models were probed. Result:

- `deepseek-v4-flash:cloud` -- currently unresponsive (read timeout at 60s in
  a direct probe; 120s timeout in the serve run). Not used.
- `glm-5.2:cloud` -- fast (2s) and capable in a one-shot probe, BUT in the
  serve REPL only the first turn synthesized; turns 2-12 routed to
  `[end-state] ?` (the PresentationGate returned no `end_state`), so no
  response -> no ingest -> the fade starved (1 anchor). The trigger is the
  model's polished, conversation-closing turn-1 response ("Let me know if
  you'd like to discuss anything...") fooling the existing planner's
  end-state heuristic into treating the conversation as resolved.
- `kimi-k2.6:cloud` -- same: 1 synthesize + 11 `?`. Same planner-`?` dispatch
  after a complete-sounding turn 1.

The `?` dispatch is **existing orchestrator behavior** (`scripts/serve_ponder.py`
prints `[end-state] {end_state_name}` where `end_state_name = getattr(end_state_plan, "end_state", "?")` -- `?` means the PresentationGate returned a plan
with no `end_state`), NOT a fade issue. It affects only cloud models with a
polite-closing response style; qwen3:8b's terser responses avoided it and
produced the 6-synthesize + 1-direct run that passed the gate. The cloud
attempts did not invalidate the fade -- they exposed a planner/end-state
interaction worth noting for Phase B (the planner can suppress synthesis
across a whole session on response style), but the fade itself is validated
by the qwen run.

## What this unblocks / does not unblock

- **Unblocked**: Phase A is validated. The router behaves on real serve
  traffic. The observability + ingest contract holds; the user-facing
  response is byte-identical to flag-off.
- **NOT unblocked (Phase B prerequisite)**: the R3 content-drift fix above.
  Before `fade_recalls` are fed into the LLM context, R3 retrieval must be
  anchor-locked so a "why did I pick Postgres?" callback returns the Postgres
  gist, not the learning-rate-schedule text.
- **Deprioritized**: Stage 2 / task #35 (the Regime-2 Transformer + JEPA-fade
  fill-holes) remains deprioritized per probe #31 (R2 THIN: the token-embed
  cluster is too tight for a fill regime to add value).
- **Independent follow-ons**: the voice leg (load the token-LM via
  `--fade-memory-voice-path` so R3 expands instead of returning verbatim),
  upgrading SSM-A from EWMA to a trained `SelectiveSSM`, persisting fade state
  across restarts, `reset()` wiring.

## Repro

```bash
BONSAI_ENDPOINT=http://localhost:11434/v1 GENERATION_MODEL=qwen3:8b \
  python scripts/serve_ponder.py --fade-memory --fade-debug --no-live-encode \
  --fade-memory-ring-capacity 4 --fade-memory-decay 0.99 \
  --fade-memory-cos-gist 0.40 --db <tmp-db>
# ask 2-3 DB questions, 5-6 cross-domain ML questions, then "why did I pick
# Postgres?"; watch [fade] show regime_name verbatim -> gist with the cos
# declining monotonically, and the R3 content drifting to the most-recent
# ML blurb.
```