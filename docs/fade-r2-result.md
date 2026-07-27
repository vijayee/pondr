# R2 fill-holes readout — VERDICT FAIL (honest negative)

Task #35 / Stage 2-3. The "recovery of the degraded SSM retrieval using JEPA
and the Transformer" — Regime 2 of the fade. **Outcome: FAIL. R2 is NOT wired.
R3 + R4 stand unchanged.** This is the honest-negative result the staged plan
(`twinkly-roaming-mango.md`) was built to produce if the gate failed; the
four-metric gate failed on all four, and the failure is structural, not a
tuning artifact.

## What was tried

`src/subconscious/fill_holes_readout.py` — `FillHolesReadout`, a 2-layer
Transformer cross-attention readout (mirrors `CrossSlotTransformerZHead`: learned
[CLS] + pos emb + `nn.TransformerEncoder`, 4 heads, FFN 512, d_model 384, 2.4M
params). It reads **memory** (the degraded SSM-A state `[CLS]` + the recent ring
slots) and reconstructs the anchor's bge **address** (384-d, L2-normalized) —
not content. Trained with `jepa_infonce_loss` (JEPA-fade InfoNCE, `jepa_gist.py`),
AdamW lr=1e-3 wd=0.01, no LM-prior anti-collapse (SSM-A is frozen; InfoNCE itself
prevents collapse). `scripts/train_r2_readout.py` drives the REAL `FadeMemory`
(real bge-small-en-v1.5, decay=0.99, ring=32, cos_gist=0.40) on a cross-domain
stream (Bible OEB-US John 1 anchors + ERAG technical docs) — the only stream that
reaches the R2/R4 band (cos < cos_gist), per the cross-domain eval.

R2 sits BELOW R3 (cos < cos_gist), reads MEMORY (state + ring) NOT the RECORD
(the full blurb store), and recalls the anchor's OWN blurb anchor-locked (the
9455795 invariant — no sibling drift). Train/eval split by ANCHOR (held-out
Bible verses), the stricter test than the plan's "held-out ERAG docs" (which
would let the readout memorize all anchor bges and pass trivially).

## The gate (all four failed)

On held-out eval anchors' R2-band triples (cos_raw < 0.40), with 24 train / 24
eval anchors, 384 ERAG, 20 epochs:

| metric | value | threshold | result |
|---|---|---|---|
| raw-state top-1 | 0.000 | — | baseline |
| ring-only closed-form top-1 | 0.000 | — | baseline |
| readout top-1 (eval band) | 0.000 | >= raw+0.15 | **FAIL** |
| cos delta (recovered - raw) | -0.138 | >= +0.10 | **FAIL** |
| vs ring-only closed-form | 0.000 | >= ring+0.05 | **FAIL** |
| neg-retrieval (sibling hit rate) | 1.000 | readout >= 5x neg | **FAIL** |
| content (9455795, own-blurb rate) | 0.000 | >= 0.15 | **FAIL** |

The readout's recovered vector retrieves a cross-domain sibling 100% of the time
(`neg-retrieval = 1.000`) and is FURTHER from the anchor than the raw state
(cos delta negative). It learned nothing useful on held-out anchors.

## Why it fails — the overfitting-vs-impossibility diagnostic

The train-vs-eval split was load-bearing. Two runs:

- **8 train anchors:** readout top-1 = **0.539 on TRAIN**, 0.000 on eval, cos
  delta +0.008. The readout MEMORIZED the 8 train anchors' bges (a fragile nudge
  to cos ~0.40, barely above the 0.389 cross-domain floor — NOT the cos ~1.0 a
  real "subtract the centroid" recovery would give). It did not generalize.
- **24 train anchors:** readout top-1 = **0.000 on TRAIN AND eval**, cos delta
  -0.109 / -0.138. With too many anchors to memorize, the readout learned NO
  recovery at all — and made things worse (negative cos delta).

The general recovery operation is **not learnable** from the ring at R2-band
depth. The 8-anchor "0.539" was a memorization artifact, not real recovery (the
marginal cos ~0.40 exposes it: a true recovery would hit cos ~1.0, as the
oracle closed-form does).

## The structural reason (the Catch-22)

The lag-bin diagnostic (`sd_ring_frac` — fraction of triples whose ring still
carries a same-domain Bible anchor) is **0.000 in every lag bin**, including the
shallowest evicted lag (33-64). By the time any anchor is evicted from the
recency ring, the ring has already turned over to all-ERAG. So at EVERY evicted
lag, the readout's inputs (state + ring) carry **zero anchor-domain signal**:

- The state is cross-domain-dominated (the anchor's contribution is
  `decay**N * bge(anchor)`, ~0.02% of the state magnitude at N~400 — buried
  under the interferer mass).
- The ring (the only memory context beyond the state) is a RECENCY window, and
  recency is anti-correlated with fade depth: the deeper the fade, the further
  the ring is from the anchor's domain.

This is the Catch-22 of the fade: **the R2 band (cos < cos_gist) is reachable
only at deep lag, where the recency ring has lost the anchor's domain.** R2's
band exists only where its ring can't help. No readout architecture can recover
signal that isn't in its inputs — this is information-theoretic, not a
model-capacity or training issue (more capacity / epochs would not help; the
24-anchor run proves the general operation is unlearnable, not merely
underfit).

## The closed-form insight (what distinguishes this from "signal is gone")

The address **IS** in the SSM-A state. `scripts/probe_r2_band.py` (Stage 0/1)
showed the FULL closed-form un-fade — `recovered = (state - Σ_j decay**(T-j)·wg·bge_j) / decay**N`
subtracting ALL interferers (the RECORD) — is **exact: top-1 = 1.000, cos = 1.0**.
So the signal is present, just attenuated + interfered. This distinguishes R2-on-
SSM-A from probe #31's token-LM finding (where the signal was genuinely gone):
here the signal is present but **not separable from the interferers using memory
alone**. The full closed-form works because it uses the RECORD (the exact
interferer set); memory (state + ring) lacks the evicted interferers, and the
ring can't reconstruct them cross-domain.

The ring-only closed-form (the linear baseline, subtracting only the ring
interferers) also fails (top-1 = 0.000): the evicted interferers NEWER than the
anchor are AMPLIFIED by the division (`decay**(anchor-j)` with `j > anchor` ->
`(1/decay)**(j-anchor) > 1`) and swamp the signal. The Transformer was meant to
approximate those missing evicted interferers nonlinearly from the ring; it
cannot, because the ring is compositionally devoid of the anchor's domain in the
band.

This matches the plan's honest-negative **criterion #3 (cross-domain floor)**:
at cos < cos_gist the state is cross-domain-dominated (floor ~0.37,
`docs/fade-cross-domain-eval-result.md`), and recovery from memory is not
significantly above the raw state. The earlier-successful Bible eval
(`docs/fade-bible-eval-result.md`) and cross-domain eval (`#37`) stand — R1
(verbatim) and R3 (anchor-locked gist) and R4 (forgotten) are unchanged; only
the R2 recovery layer is declined.

## What is kept / what is not

- **Kept (unused, the tried-and-failed artifact):** `src/subconscious/fill_holes_readout.py`
  and `scripts/train_r2_readout.py`. The readout is real (not a stub); it is
  simply not wired (Stage 4 cancelled by the FAIL gate). They remain as the
  record of what was tried and as a starting point if a future substrate
  revisits R2.
- **NOT uploaded:** no checkpoint (the gate FAILED; a FAIL's ckpt is not uploaded,
  per policy). No checkpoint was saved.
- **Unchanged:** `fade.py` R2 stub stays as-is (default-off `regime2_enabled`).
  R3 + R4 are the production fade path. No `--fade-r2-readout-path` flag, no
  wiring, no R2 tests.

## Follow-ons (NOT this plan)

The negative is structural to the **recency ring** as the memory context. Ideas
that would change the substrate (each a separate fork, not assumed viable):

- **A domain-stratified ring** that retains same-domain anchors past the recency
  window — would give the ring anchor-domain signal in the band. A different
  memory architecture, not the plan's R2.
- **A larger `ring_capacity`** — retains same-domain anchors longer, but at the
  cost of more verbatim (R1) / less fade; a whole-fade config change.
- **A trained SelectiveSSM for SSM-A** (selective gating: write only important
  chunks, preserve others) — could extend the exact window and make the above-R3
  band non-empty (the original architecture's placement, vacuous under the EWMA
  defaults per Stage 0). The `VectorCarrySSM` is structured to upgrade to this.

None are assumed to work; each would need its own gate. The honest result of
THIS plan: R2 (read the recency ring) does not recover degraded SSM-A retrieval
at fade depth, because the recency ring has lost the anchor's domain there.

## Reproduce

```
python scripts/train_r2_readout.py --n-anchors 16 --n-erag 384 --epochs 15 --seed 0   # 8-anchor (overfitting)
python scripts/train_r2_readout.py --n-anchors 48 --n-erag 384 --epochs 20 --seed 0   # 24-anchor (impossibility)
```

Both deterministic; the first shows train top-1=0.539 / eval 0.000 (memorization),
the second shows train 0.000 / eval 0.000 (the definitive negative). Writes
`data/probe/r2_readout/run_summary.json` (untracked data output, not committed).
```