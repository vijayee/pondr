# Gist-objective retrain -- verdict: FAIL (the state still carries no gist content)

> Companion to `docs/gist-readout-probe.md` (the frozen-encoder probe that FAILED
> with discrimination margin 0.000). Written for a FRESH session.
> Code: `src/subconscious/gist_readout.py`, `scripts/train_gist_readout.py`,
> `scripts/eval_gist_readout.py`, `scripts/gen_gist_cache.py`. Result memory:
> `pondr-gist-retrain-result`.

## 1. What this was

The probe (`ded522b`) proved the frozen token-LM encoder's state is
**continuation-shaped, not gist-shaped**: with the encoder frozen and a trained
state-seeded decoder, the likelihood-swap **discrimination margin was 0.000 nats**
-- swapping the encoder state between two docs changed the gist likelihood by
nothing. Post-mortem Sec 6 ([[pondr-strm-burned-postmortem]]) said the fix is to
**unfreeze the encoder and change the objective to gist recovery**, so gradient
pressure reshapes the continuation-state into a gist-shaped state. This retrain
was that prescription, executed exactly:

- **Encoder unfrozen**, two optim groups (decoder at lr, encoder at lr x 0.1),
  phase-1 decoder-only warmup (300 steps) then thaw.
- **Variable-length qualitative summary target** (deepseek-flash teacher,
  length-agnostic prompt, no one-sentence constraint; user directive), 7820
  train pairs / 200 val, 2500 steps, bf16 on the 5080.
- **Collapse watchdog**: encoder continuation val ppl on held-out ERAG chunks
  (reused `train_token_lm.eval_perplexity`), tracked each log step + end.
- **Gate**: the same judge-free likelihood swap (discrimination margin must rise
  >0, swap fidelity >= 0.8), run with `--target gist` so the swap test recovers
  the deepseek SUMMARY (the trained target), not the title.

## 2. The result -- FAIL on BOTH axes (gate + watchdog)

| Metric | Probe (frozen) | Retrain (end-to-end) | Bar | |
|---|---|---|---|---|
| **Discrimination margin** | 0.000 nats | **-0.003 nats** | >0 | **FAIL** (the decisive number, unchanged) |
| **Likelihood swap fidelity** | 4/30 = 0.133 | **4/25 = 0.160** | 0.8 | FAIL |
| State distinctness \|\|sA-sB\|\| | 4.298 | 8.571 | (diag) | states MORE doc-distinct |
| **State-vs-zero NLL gap** | +0.050 | **-0.037** | >0 | **FAIL** (decoder IGNORES the state) |
| Encoder continuation val ppl | 175 (frozen) | **175 -> 1419 (8.1x)** | ~2-3x | **FAIL** (encoder COLLAPSED) |
| Gist val ppl | 348.9 (title) | 301.98 (gist) | -- | decoder learned the target |
| Compression / Fluency | 1.00 / 1.00 | 1.00 / 1.00 | 0.6 | pass |

The decisive number -- the **discrimination margin is -0.003 nats**, i.e.
**literally unchanged from the probe's 0.000**. Swapping the encoder state
between two docs still changes the gist likelihood by NOTHING. End-to-end
training on the summary objective did NOT shape gist-discriminating content into
the state.

Worse than the probe on two axes:
- **The decoder learned to IGNORE the state entirely.** The state-vs-zero NLL
  gap went from +0.050 (probe: decoder barely used the state) to **-0.037**
  (retrain: the ZERO state is slightly BETTER than the real state). The decoder
  actively does better by ignoring the seeded state and falling back on its own
  LM prior.
- **The encoder's language prior was destroyed.** The collapse watchdog went
  from 175.00 (frozen baseline) to **1419.56 -- an 8.1x blowup**, far past the
  2-3x collapse flag. The differential-LR (0.1) + 300-step warmup mitigation
  was insufficient.

The decoded free gists confirm it: they are **degenerate stereotyped
summary-fragments** ("Summary: ... The decisions: ... Redwood ... latency ...
customer-level ..."), doc-agnostic, not faithful. The decoder learned the
marginal STYLE of the summary distribution and emits it regardless of which doc
was encoded.

## 3. The failure mode (pinned by the diagnostics)

The state IS doc-distinct (`||sA-sB|| = 8.571`, larger than the probe), the
encoder DID move under gradient, but the state's content is STILL not
gist-recoverable (margin ~= 0). And the decoder IGNORES the state
(state-vs-zero gap negative). The encoder collapsed WITHOUT gaining gist
content.

**Why:** the summary-CE objective lets the decoder SHORTCUT. Most tokens in a
qualitative summary are GENERIC summary-style structure ("The document
describes...", "Key decisions include...", "Redwood Inference...") -- the
doc-SPECIFIC content is a minority of the tokens. The decoder minimizes CE by
learning the marginal summary distribution from BOS + its own prior and ignoring
the state; the doc-specific tokens contribute little to the loss, so the state
gets weak/noisy gradient. The encoder, trained at 0.1x LR through a state
projection the decoder has learned to down-weight, drifts and loses its
continuation prior WITHOUT getting a clear gist-shaping signal. The decoder
shortcuts around the state; the encoder collapses without gaining gist.

This is a deeper version of the probe's failure: **unfreezing did not help
because the OBJECTIVE does not FORCE the state to carry gist content.** The
decoder can minimize the summary-CE loss while ignoring the state entirely.

## 4. What this rules in / rules out

- **Ruled OUT:** the plain summary-CE objective (doc -> state -> summary, end-
  to-end) shapes gist-recoverable content into the SSM state. Sec 6's
  prescription ("unfreeze + change the objective to summary") is REFUTED for
  this objective. The state does not carry gist content even after end-to-end
  training; the decoder shortcuts to the marginal summary distribution and
  ignores the state. This is a clean, decisive negative -- the foundation the
  fade was to be built on does NOT hold with this objective.
- **Ruled IN (the diagnosis points at the fix):** the objective must FORCE the
  state to be the source of doc-specific information, so the decoder CANNOT
  shortcut to a generic prior. Three directions (each its own gated step):
  1. **An objective whose loss is dominated by doc-specific tokens, not generic
     summary structure.** A reconstruction objective (doc -> state -> reconstruct
     doc-IDs -- the recoverability signal that DID work, AUC 0.81) forces the
     state to carry doc content because reconstruction cannot be done from a
     generic prior. This is content/identity recovery, not gist -- but it is the
     one objective that has actually put doc-specific content into a state.
  2. **An information bottleneck that prevents the decoder from learning a
     doc-agnostic prior.** E.g. a state-only channel with the decoder's own LM
     prior frozen/regularized, or a JEPA-style latent-prediction objective
     (predict the latent of the gist, not the tokens -- lossy latent prediction
     is intrinsically gist-shaped and has no generic-token prior to shortcut
     to). This connects directly to the Stage-2 JEPA-fade design.
  3. **A query-conditioned objective from the start** -- forces the state to
     carry query-relevant content (the salience framing, [[pondr-strm-llm-salience-result]]).
- **The encoder-collapse is a separate, second failure** that compounds the
  first: even the weak gradient that did reach the encoder destroyed its
  language prior. Any retrain attempt needs an LM-prior auxiliary loss (or
  partial-layer freeze) to preserve "text as state" -- but that is moot while
  the objective still lets the decoder shortcut.

## 5. Reusable harness validated (again)

`train_gist_readout.py` / `eval_gist_readout.py` + the likelihood-swap gate
worked end-to-end for the retrain: the `--train-encoder` path unfroze the
encoder, the collapse watchdog tracked the prior (and caught the collapse),
the loader restored the retrained encoder from the gist ckpt, and `--target
gist` made the swap test recover the trained target. The gate gave a clean,
decisive, judge-free verdict. The harness is the right tool for the next
attempt too.

## 6. Verdict + the fork

**FAIL.** The plain summary-CE objective does not shape gist-recoverable content
into the SSM state; the decoder shortcuts to the marginal summary distribution
and ignores the state, and the encoder collapses without gaining gist. The
Stage-2 fade is NOT built on this foundation -- the foundation is wrong for this
objective.

Per the plan's branch logic (both the "margin stays ~0" and "encoder collapsed"
branches hit), this is a decision point for the user. The next attempt should
use an objective that FORCES the state to carry doc-specific content
(reconstruction, JEPA-latent, or query-conditioned), with an LM-prior auxiliary
to prevent collapse. The ckpt is NOT uploaded to HF (failed retrain, per
policy -- the probe's wasn't either).

## Constraints honored

ERAG public only; no onyx; deepseek-flash for the teacher. `src/` edits
appropriate (the retrain path + the eval `--target` fix + the parallel
cache-fill tool). Committed on main per commit-at-will (NO Co-Author, ASCII
only, `git commit -F`, NOT pushed): `23605ef` (retrain path), `29da10e`
(eval --target fix), `3fb2abd` (gen_gist_cache). `data/gist_retrain/` ckpts
NOT uploaded to HF (FAIL).