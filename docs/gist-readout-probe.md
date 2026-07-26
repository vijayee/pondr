# Gist-readout probe -- the §3.3 content probe done RIGHT (verdict: FAIL)

> Companion to `docs/STRM-postmortem.md` §3.3 and §6. Written for a FRESH session.
> Code: `src/subconscious/gist_readout.py`, `scripts/train_gist_readout.py`,
> `scripts/eval_gist_readout.py`, `tests/test_gist_readout.py`. Result memory:
> `pondr-gist-readout-probe-result`.

## 1. What this is and why it was the needle-mover

The original design (Hippocampal v2 PDF p.6-7, p.12) specified **Mode B**: the SSM
state alone generates the response -- ingest text into recurrent state, decode
**gist / summaries / relevant facts as text** an LLM can process. The shipped
implementation retreated to Mode A (state compresses; the LLM receives text via an
ID-pointer; the state is never decoded). The 2026-07-25 §3.3 content probe then
**FAILED** to recover any doc-specific content from the bge backbone's state --
because that backbone was trained on a next-**embedding identity** objective, so its
state was identity-shaped, not content-shaped (post-mortem §6: **state shape is set
by the OBJECTIVE, not the backbone**).

The token-LM (`3b11aeb`, memory `pondr-token-lm-ssm-result`) then proved the owned
`SelectiveSSM` architecture and "language in/out that makes sense" -- but on a
next-token **CE (continuation)** objective, so its state is continuation-shaped: it
can continue a doc, not summarize one.

**The load-bearing open question this probe answers:** does a content objective
shape **gist-recoverable content** into the SSM state? The §3.3 probe ran on the
identity backbone and failed; it had never been run on the token-LM backbone (a
content objective). Per the post-mortem's own methodology (§3 item 1: *"probe the
EXACT property you will serve -- not a proxy; frozen backbone, no retrain"*), we
freeze the trained token-LM as an **encoder**, train **only a small new decoder**
that reads the encoder's final recurrent state and generates a gist, and gate on
whether the decoded gist follows the state on held-out docs with the §3.3 swap
control. This is §3.3 done right -- on a content-objective backbone, as a *trained*
readout.

## 2. Architecture (state-seeded decoder; encoder frozen for real)

```
encoder = SSMLanguageModel          # FROZEN, loaded strict from the token-LM ckpt
doc_ids -> encoder.forward -> enc_states   # per-layer [b, d_state, d_model_enc]
state_proj[i] : Linear(d_model_enc, d_model_dec)   # per decoder layer, TRAINABLE
enc_states[-n_dec:] -> projected -> dec_states     # seed the decoder's initial state
decoder = SSMLanguageModel(small)   # TRAINABLE; fresh, NOT tied to the encoder
dec_states + BOS -> decoder.step (autoregressive) -> gist token ids
```

The decoder reads **`states`** (seeded via a per-layer projection), NOT the encoder
block output `x`. The encoder's own `lm_head` reads `x` (the continuation path --
that is why the token-LM continues instead of summarizing). The only doc-specific
signal available to the decoder at generate time is the encoder's final recurrent
**state**, so if the decoder produces a doc-specific gist the content MUST have come
through the state -- the property under test. A swap of states between two docs must
therefore swap the decoded gist (`swap-follows-state`); §3.3 failed exactly here
(swap ~= main).

Encoder frozen for real: `requires_grad=False` on every encoder param, asserted at
load. Trainable surface = decoder + per-layer `state_proj` only (1.03M params).

## 3. The gate (the proof -- isolated, no orchestrator)

Run on **fully held-out docs** (unseen by the decoder trainer AND beyond the
encoder's train+val slice; `eval_gist_readout.py` offset 60_000):

1. **Likelihood swap (PRIMARY, judge-free).** For each held-out pair (A, B):
   `NLL(gist_A | state_A)` vs `NLL(gist_A | state_B)`. The state carries
   doc-specific content iff `state_A` makes `gist_A` more likely than `state_B`
   does. `ok_a = nll_a_a < nll_a_b`; pair passes iff `ok_a and ok_b`. Bar 0.8.
   - **Discrimination margin** = `(nll_a_b - nll_a_a) + (nll_b_a - nll_b_b)`. The
     decisive number: 0.0 nats means swapping states changes the gist likelihood
     by NOTHING -- the state carries no gist-discriminating content.
   - **State-vs-zero gap** = `nll_a_z - nll_a_a` (z = zero state). How much the
     decoder uses the state at all.
2. **Compression** -- gist shorter than the doc (bar 0.6).
3. **Fluency** -- readable text (bar 0.6).
4. **State distinctness** -- `||sA - sB||` L2 mean over layers (diagnostic: are the
   encoder states even doc-distinct, or collapsed?).
5. **LLM swap judge (SECONDARY, `--use-judge` only)** -- pairwise DeepSeek-flash
   judge on free generation; reported, does NOT veto. Free-generation faithfulness
   conflates "state has no content" with "decoder too weak to render it"; the
   likelihood swap separates them, which is why it is primary.

The likelihood swap is the load-bearing gate: it is deterministic and judge-free,
and it isolates the property under test (state carries doc-specific content) from
decoder rendering quality.

## 4. Verdict: FAIL (reproduces §3.3 with a TRAINED decoder, on a content objective)

Trained decoder: `--target title` (ERAG content -> title, free supervised long->short
signal), 8000 train docs / 200 val, 1500 steps, val ppl 348.9. Gate on 30 held-out
swap pairs + 60 held-out docs:

| Metric | Value | Bar | |
|---|---|---|---|
| **Likelihood swap fidelity** | **4/30 = 0.133** | 0.8 | **FAIL** |
| **Discrimination margin** | **0.000 nats** | >0 | **FAIL** (the decisive number) |
| Compression | 1.00 | 0.6 | pass |
| Fluency | 1.00 | 0.6 | pass |
| State distinctness `||sA-sB||` | 4.298 | (diag) | states ARE doc-distinct |
| State-vs-zero NLL gap | 0.050 | (diag) | decoder barely uses the state |
| `state_proj` W norm | 6.2 | (diag) | projection healthy, not collapsed |

**The decisive result:** discrimination margin = **0.000 nats**. Swapping the
encoder state between two docs changes the gist likelihood by literally nothing --
`NLL(gist_A | state_A) == NLL(gist_A | state_B)`. The earlier observation that "free
gists differ per doc" was sampling noise from a state-agnostic distribution, not
content read out of the state. This is the same failure §3.3 found on the identity
backbone (`perm ~= corpus-mean ~= main`), now reproduced on the token-LM backbone
**with a trained decoder**.

**The failure mode (pinned by the three diagnostics):**
- `||sA - sB|| = 4.298` -- the encoder states ARE doc-distinct, NOT collapsed. The
  state is not empty.
- `state_proj` W norm = 6.2 -- the projection is alive and healthy, NOT collapsed.
  The decoder was not prevented from reading the state.
- `state-vs-zero NLL gap = 0.050` -- the decoder barely uses the state. It learned to
  mostly ignore the seeded state and fall back on its own LM prior.

**Interpretation:** the projection is alive and the state is doc-distinct, but the
state's content does NOT help predict the gist. The doc-distinct content carried in
a token-CE state is **continuation-shaped** (it helps predict the doc's *next
tokens*), not **gist-shaped** (it does not help predict a one-sentence summary of
the doc). The decoder correctly down-weighted a state that genuinely lacks gist
info -- this is not a decoder-capacity failure, it is a state-content failure.

This **confirms post-mortem §6 from a new angle**: the objective sets what the state
carries. A continuation objective carries continuation-content; that is not
gist-recoverable content. §6 said "to get a content-shaped state, train a
content/reconstruction objective"; this probe shows that a *continuation* objective
is NOT that content objective -- the distinction matters.

## 5. What this rules in / rules out

- **Ruled OUT:** decoding a faithful gist from a frozen continuation-objective
  state. Mode B is not viable on the token-LM backbone as-is. The token-LM remains
  what it proved itself to be: a continuation LM ("language in/out that makes
  sense"), not a summarizer.
- **Ruled IN (the §6 prescription, now with evidence):** the state must be SHAPED
  by a gist-recovery objective. The failure mode ("state is doc-distinct but not
  gist-shaped") tells us the fix is NOT "bigger decoder" or "fix the projection" --
  those are healthy. The fix is to **unfreeze the encoder and change the objective
  to gist recovery** (encoder-decoder end-to-end on doc -> gist), so the
  continuation-shaped state is reshaped into a gist-shaped state.
- **The reusable harness is validated:** `train_gist_readout.py` / `eval_gist_readout.py`
  + the likelihood-swap gate are the right tools for the retrain's gate too. The
  swap control + discrimination margin are the judge-free test for "did the state
  learn to carry gist content." Reuse them verbatim on the retrained model.
- **Negative control (deferred):** the text2x identity encoder (Ashes-of-STRM) is a
  `JGSBackbone` with a different state interface and needs a separate adapter. The
  core probe failing on the content-objective backbone already makes the point §6
  needed; the control is a follow-on, not a blocker.

## 6. The prescribed next step (gated -- confirm before starting)

**The gist-objective retrain:** unfreeze the encoder, train end-to-end on
(doc -> gist) -- the decoder already exists; the change is (a) thaw the encoder,
(b) put the encoder in the optim, (c) keep the likelihood-swap gate. The state then
gets gradient pressure to carry gist-discriminating content (the discrimination
margin must rise above 0). This is the plan's FAIL branch and the direct §6
prescription. It is a bigger build (the encoder trains), so it is a separate gated
step -- confirm before starting.

Non-goals until that passes: orchestrator integration, query-conditioned decode,
multi-doc streamed ingestion, the EXPAND confidence signal. All gated on a
state that demonstrably carries gist-recoverable content.