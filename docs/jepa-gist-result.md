# JEPA-latent gist retrain -- verdict: FAIL (the state carries no generalizable gist)

> Companion to `docs/gist-retrain-result.md` (the summary-CE retrain that FAILED
> with discrimination margin -0.003 and an 8.1x encoder collapse). Written for a
> FRESH session. Code: `src/subconscious/jepa_gist.py`, `scripts/train_jepa_gist.py`,
> `scripts/eval_jepa_gist.py`. Result memory: `pondr-jepa-gist-result`.

## 1. What this was

The summary-CE retrain ([[pondr-gist-retrain-result]]) proved the plain summary
objective does NOT shape gist-recoverable content into the SSM state: the decoder
shortcuts to the marginal summary distribution and ignores the state. Post-mortem
Sec 6 ([[pondr-strm-burned-postmortem]]) said the fix is an objective that
**FORCES the state to carry doc-specific content**. The user picked the pivot
(option 2): a **JEPA-latent** objective -- predict the bge *latent* of the gist
from the SSM state, not the gist *tokens*. Lossy latent-space prediction is
intrinsically gist-shaped and has no generic-token prior to shortcut to; the
contrastive negatives make doc-specificity the explicit objective. This is the
same objective family as the planned Stage-2 JEPA-fade, so Stage 1 stops being a
throwaway and becomes the first half of Stage 2.

- **Encoder: the token-LM `SSMLanguageModel`** (d_model=256, 6 layers, d_state=16,
  val ppl 177) -- the right substrate for the fade (a TOKEN stream; verbatim-recall
  of recent tokens -> gist of older tokens). NOT the bge-space `JGSBackbone`.
- **Anti-collapse fix the summary run lacked:** an **LM-prior auxiliary** (next-token
  CE on the doc, weight 0.1) -- directly penalizes the continuation-prior blowup
  that went 8.1x last time. PLUS differential-LR (0.1x) + 300-step warmup-thaw.
- **Target (frozen):** `bge-small-en-v1.5` encodes the deepseek-flash gist text
  (7844 cached gists) -> 384-d L2-normalized. bge is the frozen referee.
- **Gate (judge-free, bge referee):** discrimination margin, swap fidelity,
  state-vs-zero gap, ||sA-sB||, top-1/top-3 retrieval, collapse watchdog. 60
  held-out docs (offset 60000), 25 swap pairs.

## 2. The result -- FAIL (the state carries no generalizable gist)

This retrain ran TWICE. The first run revealed the reused loss was DEGENERATE; the
second run fixed the loss and produced the real verdict.

| Metric | Degenerate run (contrastive) | InfoNCE run (the verdict) | Bar | |
|---|---|---|---|---|
| **Loss** | `jepa_contrastive_loss` (reused) | `jepa_infonce_loss` (bounded) | -- | the fix |
| Train latent loss | ~-4.6 (chasing anti-correlation) | 2.75 -> 2.42 (random log(17)=2.83) | -- | overfit train |
| **Val latent cosine** | **-0.8398** (anti-correlated) | **0.0674** (~random; noise floor 0.05) | >0 | **FAIL (decisive)** |
| Val latent loss | -4.500 (degenerate) | 2.781 | -- | -- |
| Discrimination margin | (not gated) | **0.0090** | >0 | noise (technically >0) |
| **Swap fidelity** | (not gated) | **0.200** | 0.8 | **FAIL** (~chance) |
| State-vs-zero gap | (not gated) | 0.0924 | >0 | marginal pass |
| State distinctness \|\|sA-sB\|\| | (not gated) | 5.384 | (diag) | states ARE doc-distinct |
| **Retrieval top-1 / top-3** | (not gated) | **0/60 = 0.000 / 1/60 = 0.017** | -- | **below chance** (random top-3 ~0.05) |
| Encoder continuation val ppl | 175.00 -> 175.23 (1.00x) | 175.00 -> 183.25 (1.05x) | <2-3x | OK (no collapse -- LM-prior held) |
| **VERDICT** | (loss was wrong) | **FAIL** | -- | -- |

The decisive number is **val latent cosine = 0.0674**: on held-out docs the
predicted latent is **uncorrelated** with the true bge gist latent (random cosine
in 384-d has std ~1/sqrt(384) = 0.051, so 0.067 is within the noise floor). The
nearest-neighbor samples confirm cluster-mean collapse: for every doc, the
predicted latent retrieves the same cluster-central gists (Kimberly Park /
Elliot Strauss -- the cluster-mean gist), and each doc's OWN gist ranks mid-pack
(rank 29, 48, 3, 23, 56...). `pred_cos_to_own` hovers around 0 (0.06, 0.02, 0.12,
0.08, **-0.10**).

## 3. The failure mode (pinned by the diagnostics)

Two distinct failures, one per run:

### 3a. The first run: the reused loss was DEGENERATE (a loss bug, now fixed)

The reused `jepa_contrastive_loss` (from `training/jepa_loss.py`, built for the
JEPA backbone pretrain) is degenerate for a TIGHT target cluster. Its negative
term `logsumexp(cos(pred, negatives)/temp)` is unbounded below (~`-1/temp +
log(N)` = -7.2 for temp=0.1, N=16), while its positive term `-cos(pred, actual)`
is bounded [-1, 1]. bge gist latents of similar ERAG docs cluster tightly (cos
~0.7), so the loss is LOWER at `pred = -actual` (anti-correlate with the whole
cluster, total ~-3.2) than at `pred = actual` (total ~+8.8). The predictor
correctly finds the degenerate optimum -> **val_latent_cos = -0.8398**. The
contrastive negatives INTENDED as the anti-shortcut instead CAUSED the
degeneracy.

The anti-collapse LM-prior auxiliary did its job here: the encoder did NOT
collapse (175.0 -> 175.2, ratio 1.001 -- the summary run went 8.1x). The loss
itself was the bug. **Fix:** `jepa_infonce_loss` puts the positive in the SAME
cross-entropy denominator as the negatives, so the optimum is `pred = actual`
(bounded by construction). Pinned by a regression-guard test
(`test_infonce_prefers_actual_over_anticorrelation`): on a tight cluster the OLD
loss is degenerate (anti < actual) AND InfoNCE is correct (actual < mean < anti),
robust across dim 64/128/384. Committed in `9c14210`.

### 3b. The second run (correct loss): the state carries no GENERALIZABLE gist

With the loss fixed, the InfoNCE train loss dropped 2.75 -> 2.42 -- the predictor
learned to map TRAINING states to their gist latents. But **val latent cosine =
0.067 (random)**. The predictor OVERFIT the training docs; on held-out docs it
falls back to the cluster mean. The state carries TRAINING-doc-specific signal
but NOT a generalizable gist signal.

The state IS doc-distinct (`||sA-sB|| = 5.384` -- larger than the probe's 4.298),
and the encoder did NOT collapse (175.0 -> 183.3, ratio 1.05 -- the LM-prior
auxiliary held where the summary run's 8.1x blew up). So the failure is NOT
collapse and NOT a non-distinct state. The failure is that the doc-distinctness
is **doc-IDENTITY, not gist** -- it does not align with gist content and does
not generalize to held-out docs. This is the same finding as Phase 0b
([[pondr-strm-phase0b-gate-no-go]]: "the JEPA backbone encodes doc-identity
query-orthogonal") and the gist-readout probe ([[pondr-gist-readout-probe-result]]:
state = continuation + recoverability, not content), confirmed from a new angle:
even a content-forcing contrastive latent objective does not reshape the
token-LM state to carry generalizable gist content.

## 4. What this rules in / rules out

- **Ruled OUT:** a content-forcing contrastive latent-prediction objective
  reshapes the token-LM encoder state to carry GENERALIZABLE gist content. Sec
  6's load-bearing bet -- "a content-forcing objective reshapes the state" -- is
  REFUTED for this objective on this encoder: the predictor overfits the
  training mapping, but the held-out state carries no gist signal (val cos 0.067,
  retrieval 0/60). The state is doc-distinct but the distinctness is doc-identity,
  not gist.
- **Ruled IN (both fixes worked, and are reusable):** (a) the bounded InfoNCE
  loss is the correct contrastive formulation for a tight target cluster (the
  reused contrastive loss is degenerate there); (b) the LM-prior auxiliary +
  differential-LR + warmup-thaw PREVENTED the encoder collapse that sank the
  summary run (1.05x vs 8.1x). Both carry forward to any next attempt.
- **The diagnosis points at the fork (per the plan's "margin stays ~0" branch):**
  - **(a) The token-LM encoder is too sticky to reshape** (the continuation-prior
    pretraining shape dominates; gradient at 0.1x LR through the predictor cannot
    bend it to gist). Quick check: run the SAME InfoNCE objective on the
    `JGSBackbone` (bge-space, 384-d -- the target and the state live in the same
    space, no 256->384 projection). If THAT puts generalizable gist in the state,
    the fade must be rebuilt on the bge-backbone (gist-only, no verbatim -- a
    lesser Stage 2, but a working one).
  - **(b) If neither works**, the state genuinely cannot carry gist under a
    PREDICTION objective. Pivot to **reconstruction** (option 1: doc -> state ->
    reconstruct doc-IDs; the recoverability signal that DID work, AUC 0.81) or
    **query-conditioned** (option 3: the salience framing,
    [[pondr-strm-llm-salience-result]]).

## 5. Reusable harness validated (again)

`train_jepa_gist.py` / `eval_jepa_gist.py` + the latent-space swap gate worked
end-to-end: the `--train-encoder` warmup-thaw path unfroze the encoder, the
collapse watchdog tracked the prior (and confirmed NO collapse), the `--loss`
flag isolated the loss bug (the degenerate run reproduced, the InfoNCE run
fixed it), the loader restored the retrained encoder from the JEPA ckpt, and the
judge-free bge-referee gate gave a clean decisive verdict. The harness is the
right tool for the next attempt (the JGSBackbone quick check) -- swap the
encoder in `JEPAGistModel`, reuse the rest.

## 6. Verdict + the fork

**FAIL.** With the loss fixed (InfoNCE) and collapse prevented (LM-prior), the
JEPA-latent objective still does NOT shape generalizable gist content into the
token-LM state: val latent cosine 0.067 (random), retrieval 0/60, swap fidelity
0.2 (~chance). The predictor overfits the training mapping; the held-out state
is doc-distinct but doc-identity-shaped, not gist-shaped. The Stage-2 fade is
NOT built on this foundation -- the foundation does not hold for this objective
on this encoder.

Per the plan's branch logic (the "margin stays ~0" branch), this is a decision
point for the user. The two sub-branches:
- **(a)** Quick check on the `JGSBackbone` (bge-space, 384-d) -- same InfoNCE
  objective, same gate. If it works, rebuild the fade on the bge-backbone
  (gist-only, no verbatim).
- **(b)** Pivot to reconstruction (option 1, AUC 0.81 worked) or query-conditioned
  (option 3).

The ckpt is NOT uploaded to HF (failed retrain, per policy). Stage 2 remains
GATED on a Stage-1 objective that actually puts generalizable gist in the state.

## Constraints honored

ERAG public only; no onyx; deepseek-flash for the teacher gist text; bge-small is
a frozen open model (the latent target). `src/` edits appropriate (a NEW isolated
module + NEW trainer/eval + the InfoNCE loss; no orchestrator/runtime/serve
changes). Committed on main per commit-at-will (NO Co-Author, ASCII only, NOT
pushed): `9c14210` (InfoNCE loss + `--loss` flag + regression-guard test).
`data/jepa_gist_infonce/` ckpts NOT uploaded to HF (FAIL). The degenerate
contrastive run's artifacts are preserved in `data/jepa_gist/` for the record.