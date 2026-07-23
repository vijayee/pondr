# STRM Readout -- Journey, Failures, and the From-Scratch Solid Approach

Date: 2026-07-23
Status: Stage 3 PASS (first readout to clear the 6-seed live gate on BOTH
buckets). Nothing wired into the live engine yet; integration gated on a
validation step (see Sec.5).

This document is the postmortem + forward recipe for the STRM readout path
(the work tracked under Phases 0a/0b -> 1 -> 1a-1e -> 1f -> 1f-5/6 -> 1f-7
Stages 0-3). It records **what was tried, what failed and why, and the approach
that would have avoided the pitfalls from the start**. It is the complement to
`docs/training-guide.md` (which covers the *trained components* -- JGS backbone,
RetrievalGate, DocKindHead, GNN, Bonsai -- and the Mamba3 swap) and
`docs/STRM-architecture-proposal.md` (the design). It synthesizes the
per-experiment memory files in the project memory index.

The account is honest about dead ends: roughly twenty numbered experiments were
run on this path before a head cleared the robustness bar. The point of writing
them down is that the *pattern* of failure is more reusable than any single
result.

---

## 1. The goal and the substrate

**Goal.** A readout head that, from the live `WorkingMemory` ring state, locates
which ring slots are query-relevant (the "locate query-relevant memory in the
live ring" primitive from the STRM architecture). The acceptance gate is a
**live, per-source z_logit selectivity gap median >= 2.0**, measured in TWO
buckets -- `retrieved_text` AND `retrieved_code` -- each cleared in
**>= 4 of 6 seeds**. The 6-seed bar is deliberate: 3-seed passes proved to be
variance luck (see Sec.2.8 / pitfall #4).

**Substrate.**
- A frozen 19.5M `ReferenceSSM` backbone: `d_model=384`, `n_layers=4`,
  `d_state=16`. Originally pre-trained with `L_relevance + L_trajectory` on
  **mean-pool `z_k [384]`** over **ERAG-Bench** -- a routing/relevance
  objective. This is the load-bearing fact the whole journey turns on.
- The ring slot state the readout consumes is `flat_last [6144]` = the **last
  SSM layer's** 16 channels x 384, flattened. This is a *different* and richer
  representation than the mean-pool `z_k [384]` the backbone was actually
  trained on.
- Ring slot: `(y_t, source_id, text, pinned, h, h_pre, u, slot_type)`.
- Doc kinds in the ring: `conv` (0), `text-doc` (1), `code-doc` (2).
- The training/eval corpus is the user's private conversational ring (~52
  sessions, 939 mixed-ring records). No session content is reproduced anywhere
  in code or docs; only aggregate counts.

**The core mismatch, stated up front because everything follows from it.** The
backbone was trained to make **mean-pool `z_k [384]`** query-doc-relevant on
**ERAG**. The readout consumes **`flat_last [6144]`** on the **real serve
ring**. Neither the representation nor the distribution was what the readout
needed. Every readout-only experiment below banged into this fact before it was
named; Stage 3 finally named it and fixed it.

---

## 2. The journey (what was tried, what failed, why)

### 2.1 Phase 0a/0b -- is the signal in the state at all? (GO)

- **0a** asked where the recoverability signal lives; it lives in the
  **readout**, not the raw backbone -- a learned readout must mix channels
  (probe AUC 0.810 over the free k-baseline 0.732).
- **0b** showed a **linear** next-state map clears the surprise gate and *beats*
  an EMA JEPA predictor. JEPA's anti-collapse machinery solves a problem that
  cannot occur with a frozen backbone. Ships linear. (This is the "do the cheap
  closed-form baseline first" lesson, applied cleanly once.)
- Both gates PASSED. The design assumption (SSM forgetting is predictable enough
  to read off) was de-risked. Lesson: **the cheap probe gates the expensive
  build, and the cheap version of the probe beats the fancy one.**

### 2.2 Phase B / early probes -- train-go, serve-fail (the OOD finding)

- The head trained well and passed within-corpus gates, then **failed on live
  serve data**. Root: **train/serve out-of-distribution**. The training traces
  were not the live serve distribution.
- Probes 1-3 (context-coverage ON-vs-OFF, threshold sweep, cost parity) found
  headroom in coverage but a **THETA bottleneck** (shipped `theta=p30`), and
  that the 2a head **saturates at serve** -- a train/dist problem, not a head
  problem.
- Task #33 onward: `--identity-instance` (train and serve on the same instance
  identity) and held-out live sessions. Lesson: **the true gate is live serve,
  not within-corpus. Within-corpus verdicts inverted on live repeatedly.**

### 2.3 Phase 1 -- from-scratch backbone (`L_relevance + L_trajectory`)

- A fresh `ReferenceSSM` trained from scratch with joint relevance + trajectory
  losses. P1 cheap probe (5K docs, ~300 steps) gated the full run; the full run
  produced `backbone_v2_full.pt` (the backbone the rest of the journey used).
- The probe-first discipline worked here. But note what the objective was:
  **mean-pool relevance on ERAG**, not `flat_last` on the serve ring. The
  mismatch in Sec.1 was baked in at this point and not noticed for ~15 more
  experiments.

### 2.4 Phase 1a-1e -- mixed ring, slot types, regularization, held-out

- 1a/1b: flag-gated `text`/`source_id`/`slot_type` on conversation slots + a
  slot-type embedding and learnable temperature in the cross-slot Transformer
  head.
- 1c: the mixed-ring trace generator (conversation + retrieved-doc + slot
  types) -- the data substrate for everything after.
- 1d: regularization (dropout, label smoothing, cosine, ensemble).
- 1e: clean held-out full-ring acceptance test. This is where the **conv ring
  Transformer beat bilinear** within-corpus -- a result later **inverted** on
  live (1f-4).
- Lesson: **within-corpus wins are not live wins.** This kept happening.

### 2.5 Phase 1f -- production doc corpus (incl. code)

- Built the production-matched doc corpus including **code docs** and the
  `generate_onyx_doc_ring_traces.py` generator.
- 1f-4 acceptance: **BOTH heads FAIL 0/3 on live** (`--doc-store`). Bilinear
  0.0 (docs mis-rank), transformer 0.994. The within-corpus verdict inverted on
  live *again*. This is the moment the path became "diagnose the live failure
  properly" rather than "try another head."

### 2.6 Phase 1f-5/1f-6 -- margin loss, doc-kind split, code summarization

- 1f-5: reconsulted DeepSeek (#1). Tried **margin loss** on the transformer
  (failed 0/3 live) but **bilinear cleared the text-doc gate 2/3**. Code-docs
  mis-ranked. Diagnosis: code-doc root cause is in the **SSM state h / backbone**,
  not the embedding handle.
- 1f-6: **prose-summary embed** for code docs (CodeSectionSummarizer + threaded
  `embed_text`). This **lifted text 7.6x** (3.969 -> 30.16, 2/3) and flipped
  retrieved 0/3 -> 2/3 -- but **code-docs got WORSE** (-0.801 -> -4.94).
  Key finding: summarizing code-doc embeddings helps text and hurts code; the
  code problem is downstream of the embedding, in the backbone state.
- Lesson: **the code-bucket failure is a representation problem, not a data-
  formatting problem.** Formatting (summarization) moved text but not code.

### 2.7 Phase 1f-7 Stage 0-1 -- code-only diagnostic, per-kind readout, MoE

- **Stage 0 (code-only diagnostic):** bilinear on **CODE-GOLD-ONLY** cleared
  held-out z_logit 6.76 (2/3). This **ruled out a doc-kind artifact** and
  pinned the root cause to **query-relevance-in-h** -- the signal IS in the
  state for code when you ask a code-only question. Important: this is the
  experiment that said "representation has the signal, the cross-kind readout
  is the problem."
- **Stage 1 (per-kind readout):** shared body 6144->128 + per-kind
  Linear 128->384. FAILED: ret_code +0.825 (0/3) AND ret_text **regressed**
  30.16 -> 0.558. Root: shared body conv-majority + kind-head over-regularizes.
- **Stage 1 redesign (MoE):** N independent 6144->128->384 readouts, NO shared
  body. **FAILED WORSE (0/3; ceiling 0.000).** Structural lesson, the most
  important architectural finding of the journey:

  > **The selectivity gate is FUNDAMENTALLY CROSS-KIND. Any per-kind
  > decomposition makes cross-head logits ill-defined. ONLY a single SHARED
  > readout gives a comparable cross-kind logit space.**

  This is why per-kind bodies (Stage 2 #6) and MoE both collapsed: you cannot
  compare a logit from a code-head to a logit from a text-head. The gate
  *requires* one shared logit space.

### 2.8 Phase 1f-7 Stage 2 #1-#6 -- the readout-only ladder (all fail at 6-seed)

All of these kept the backbone FROZEN and tried to fix the readout. They are the
exhaustion of the readout-only path.

| # | What | Result | Why it failed |
|---|------|--------|---------------|
| #1 | Class-balanced SAMPLER (inverse-freq, replacement) on shared readout | 6-seed FAIL, but **s2 first seed EVER to pass both buckets** | Proved the shared readout CAN pass both; instability was the blocker. Extreme variance (s0/s1 inverted). |
| #2 | Per-kind **LOSS** weighting (uniform sampling) | COLLAPSE under AdamW (train_loss stuck ~2.495) | **Key lesson: with AdamW, class-balanced SAMPLING is clean (scale-1.0 steps); class-balanced LOSS weighting is NOT.** |
| #3 | Code hard-negative mining | FAIL: ret_text +31.9 (2/3) BUT ret_code -14.2 (0/3) | Loss-structure change cannot fix a representation weakness. 2 de-wonk bugs found (neg_mask device; empty-filler grad-safe zero). |
| #4 | No-replacement inverse-freq sampler (weighted shuffle) | FAIL (ret_code 1/3) but closest-yet (code median -4.26 -> +0.72) | 3.6x code upweight starved s1. Tempered by #5. |
| #5 | sqrt-inverse-freq no-replacement sampler | **3-seed PASS (variance luck) -> 6-seed FAIL** (ret_text 3/6, ret_code 2/6) | The ceiling. True both-buckets rate ~2/6 = 33%. The 3-seed PASS drew the 2 good seeds first. DeepSeek's consult #2 warning ("if true rate ~1/3, 6 seeds = ~2/6 = FAIL") was exactly right. |
| #6 | Per-kind BODIES + shared head (DeepSeek ladder step 3) | COLLAPSED MAXIMALLY on s0 (#5's best seed, +23/+17 -> 0.000) | Same structural lesson as MoE: per-kind decomposition breaks the cross-kind logit space. Arch-caused; routing verified correct. |

**End of readout-only path.** #5 (2/6) was the ceiling. Every data/loss/arch
fix on the frozen backbone was exhausted at the robustness bar. The shared
body collapses to a flat or code-weak direction for 2/3 of seeds; per-kind
decompositions collapse maximally. DeepSeek consult #3 verdict: **REPRESENTATION
problem, not readout-arch** -- `flat_last [6144]` is kind-isolated.

### 2.9 Phase 1f-7 Stage 3 -- the backbone fine-tune (PASS) + the fidelity fix

DeepSeek #3 said: lightly fine-tune the backbone so `flat_last` carries a common
cross-kind query-doc relevance subspace. The cheap diagnostic:

- **8 epochs, ALL 19.5M params, lr 1e-5** (1/10 the from-scratch base), AdamW
  wd 0.01 cosine, **margin loss (m=2.5, hard-neg) on `flat_last`** over the
  onyx doc ring, via a **THROWAWAY slim readout** (`Linear(6144,384)` +
  `cos(z,q)/T=0.05`, **discarded** after). Low capacity FORCES the backbone to
  produce a query-relevant `flat_last`; a powerful readout could compensate for
  a weak one and hide the real problem. The readout is NOT saved; only
  `backbone.state_dict()` is written.
- **Replay: pre-state replay, truncated-BPTT-depth-1.** Each kept slot's
  pre-step WM state (`slots_pre_state`) + exact step-input (`slots_step_input`)
  are captured on the slot; the fine-tune seeds `states = slots_pre_state[k]`
  (DETACHED) and re-steps ONLY `slots_step_input[k]` WITH grad through
  `layer.step` -> reproduces `slots_h_raw` within fp16 epsilon AND backprops
  into the shared backbone (`W_A/W_B`). No cross-slot BPTT, no memory blowup.
- **THE FIDELITY FIX (the key correctness piece, and a general pitfall).** The
  first replay attempt re-embedded `slot.text` at trace-build to get the
  step-input. This DIVERGED from the actually-stepped vector for ~20% of
  retrieved slots (the code-doc slots): retrieved code docs are injected **by
  MEANING** -- the orchestrator steps `embed(embed_text or summary)`, but
  `slot.text` stores the summary string. Re-embedding the summary is not the
  same vector as was stepped. Replay fidelity broke (max-abs-diff 0.2421 >
  atol 0.15 -- the fine-tune would have trained on bogus state). **Fix: capture
  the EXACT step-input `u` on the slot (post-pin, pre-SSM, fp32) and replay
  from it** -> fidelity 0.0002 (essentially exact). Loss then dropped smoothly
  2.51 -> 0.46 across 8 epochs (NOT stuck at 2.49 -- the per_kind_bodies
  collapse signature).

**RESULT (6-seed live serve gate, fine-tuned backbone, then retrained the
UNCHANGED #5 shared-body readout on the fine-tuned backbone's regenerated
traces):**

```
retrieved_text: median +10.26, 4/6 pass (need 4) -> PASS
retrieved_code: median +8.25,  6/6 pass (need 4) -> PASS
per-seed: s0 code 10.44 / text -0.24
          s1  8.09 / 28.00
          s2  8.42 /  1.27
          s3  6.62 / 10.05
          s4  5.69 / 14.44
          s5 10.10 / 10.46
within-corpus held-out: bilinear z_logit median 3.02, 3/3 pass
```

- **First readout EVER to clear BOTH buckets at the 6-seed bar.** `ret_code`
  went 2/6 -> 6/6. The unchanged #5 readout cleared 4/6 on BOTH once the
  backbone's `flat_last` carried cross-kind relevance -- clean causal
  confirmation that the problem was the representation.
- The 4/6 text (s0 text -0.24, s2 text 1.27) is a **yellow flag, not red**:
  code is 6/6, text is just at the bar. The 939-record ring skews code-heavy,
  so text relevance is less reinforced. Coherent, not a hidden red flag.
- `backbone_v2_full_finetuned.pt` is a NEW file; `backbone_v2_full.pt` is
  untouched and remains the live default. **Nothing is wired into the live
  engine.** Commit `57d28c8`.

---

## 3. The pitfalls, ranked by pain caused vs cheapness to design out upfront

(DeepSeek consult #4 synthesis, confirmed against the experiment record.)

1. **Backbone objective mismatch (HIGHEST pain, CHEAPEST to fix upfront).**
   The backbone was trained on mean-pool `z_k [384]` / ERAG; the readout
   consumes `flat_last [6144]` / serve ring. ~15 experiments burned on
   readout-only fixes before this was named. **Design out: include a
   `flat_last` query-doc relevance loss on the real serve ring, jointly with
   the routing objective, from day 1.** No separate fine-tune needed.

2. **Injection-by-meaning fidelity trap (HIGH pain, TRIVIAL to design out).**
   Re-embedding `slot.text` instead of capturing the exact step-input `u`
   caused a 20% silent fidelity break that would have ruined any replay
   training. **Design out: the trace format captures `slots_step_input` (the
   exact step-input embedding) by construction.** One-line schema change, zero
   cost upfront.

3. **Per-kind decomposition / MoE (HIGH pain, AVOIDABLE by design).** The
   selectivity gate requires a single cross-kind logit space; per-kind heads,
   per-kind bodies, and MoE all collapse because cross-head logits are
   incomparable. **Design out: exactly ONE shared readout head. Never split by
   kind.** Two collapsed experiments (MoE, per_kind_bodies) cost real time.

4. **3-seed variance-luck trap (MODERATE pain, EASY to avoid).** Multiple
   3-seed passes gave false confidence (#5 was the textbook case: 3-seed PASS
   -> 6-seed FAIL, true rate ~33%). **Design out: the robustness bar is 6
   seeds from the start; no decision on <6 seeds.** DeepSeek warned about this
   in consult #2 and was exactly right.

5. **Class-balanced LOSS weighting vs SAMPLING (MODERATE pain, EASY to avoid).**
   Per-kind loss weighting collapsed under AdamW (stuck at the 2.49 margin
   ceiling); class-balanced no-replacement sampling was clean. **Design out:
   always class-balanced no-replacement sampling for imbalanced relevance data;
   never loss weighting with AdamW for this problem.**

6. **Within-corpus vs live serve inversion (MODERATE pain, EASY to avoid).**
   Within-corpus wins inverted on live repeatedly (1e conv-ring transformer,
   1f-4 both heads). **Design out: the acceptance gate is live serve from the
   first real experiment; within-corpus is a smoke check only.**

7. **Catastrophic-forgetting risk on a shared-backbone fine-tune (OPEN, must be
   measured).** The Stage 3 fine-tune moved the shared `W_A/W_B` that every
   other head reads through mean-pool. The fine-tune targeted an untrained
   path, but it still modified shared parameters. **Design out (for the
   from-scratch recipe): the joint multi-task objective keeps the original
   objectives while adding the `flat_last` loss, so forgetting is structurally
   prevented. For the current fine-tune: must be measured (Sec.5).**

---

## 4. The from-scratch solid approach

If we were building the STRM readout from scratch, knowing what we know now:

1. **Train the backbone with a joint multi-task loss from day 1:**
   - the original ERAG routing / mean-pool relevance objective (preserves what
     2a/2b/DocKindHead need), AND
   - a `flat_last` query-doc relevance loss (margin or InfoNCE) **on the real
     serve ring**, so the representation the readout consumes is directly
     optimized for it.
   This single decision designs out pitfall #1 (the ~15-experiment readout-only
   detour) and pitfall #7 (forgetting, structurally).

2. **Capture the exact step-input `u` (`slots_step_input`) in the trace format
   by construction.** Designs out pitfall #2 (the 20% fidelity break) for free.
   The trace stores per-kept-slot `slots_pre_state` + `slots_step_input`; any
   replay-based training seeds from `slots_pre_state` (detached) and re-steps
   `slots_step_input` with grad (truncated-BPTT-depth-1). This is already the
   shape shipped in Stage 3 (`WorkingMemory.capture_pre_state`,
   `RingSlot.u`, `generate_onyx_doc_ring_traces --capture-pre-state`).

3. **Train a single shared readout** (cosine-similarity with learned
   temperature, or the `CompositeZHead` bilinear) on the frozen backbone's
   `flat_last`. Never per-kind heads, never per-kind bodies, never MoE.
   Designs out pitfall #3.

4. **Class-balanced no-replacement sampling** (sqrt-inverse-freq weights) for
   the imbalanced conv/text/code relevance data; never loss weighting with
   AdamW. Designs out pitfall #5. (This is the #5 sampler, kept.)

5. **6-seed robustness bar from the first real experiment; no decision on <6
   seeds.** Designs out pitfall #4. (The gate script already takes `--seeds`.)

6. **Live serve as the acceptance gate; within-corpus as a smoke check only.**
   Designs out pitfall #6.

7. **Margin-ranking loss (m=2.5, hard-negative) as the tight surrogate for the
   z_logit gate** -- not InfoNCE (which hurt bias-invariance/transfer in task
   #44) and not a loss-structure change (which cannot fix a representation
   weakness, per #3).

8. **If a fine-tune is ever needed instead of a from-scratch retrain:**
   throwaway slim readout (low capacity, forces the backbone to do the work,
   discarded), pre-state replay (truncated-BPTT-depth-1), low LR (1e-5), few
   epochs (8), fidelity gate at epoch 0 (abort if the seed does not reproduce
   the trace). This is the Stage 3 recipe, and it works -- but the from-scratch
   joint objective (step 1) is strictly better because it needs no fine-tune
   and structurally prevents forgetting.

The single highest-leverage decision is #1. Everything else is hygiene that
matters but is secondary to "train the representation the readout actually
consumes, on the distribution it actually serves."

---

## 5. The Stage 3 result and what is open

**What is done.** Stage 3 PASSES the 6-seed live gate on both buckets. The
fine-tuned backbone (`backbone_v2_full_finetuned.pt`) and the #5 readout
protocol are the first working STRM readout. The original backbone is
preserved; no existing functionality changed. Commit `57d28c8` (8 files,
+1232); memory `pondr-strm-phase1f7-stage3-flatlast-finetune-result`.

**What is open (integration is a separate decision; do NOT auto-proceed).**
DeepSeek consult #4 gave a clear ship-vs-continue protocol:

1. **Load-bearing validation (the single gating step before any wiring):**
   re-run the original validation suite -- **2a retrieval gate (val 0.826),
   DocKindHead ensemble (ship gate), ERAG routing** -- on the fine-tuned
   backbone. The fine-tune moved shared `W_A/W_B`; this measures whether
   catastrophic forgetting occurred.

2. **If original metrics are preserved (no degradation): SHIP.** Wire the
   fine-tuned backbone as the **single** backbone, and serve the STRM readout
   with an **ensemble of the 4 passing seeds (s1, s3, s4, s5)** -- average
   their logits before the gate. (s0/s2 are unreliable on text; including them
   injects noise.) All heads benefit from the improved `flat_last`.

3. **If original metrics degrade: DO NOT SHIP.** The fine-tune caused
   forgetting. The correct next action is the **joint multi-task fine-tune (or
   from-scratch retrain) from Sec.4** -- add the `flat_last` relevance loss while
   keeping the original objectives. A temporary two-backbone split is
   acceptable ONLY as a short-lived stopgap while that retrain runs.

4. **Do NOT ship a permanent two-backbone split.** It is a maintenance trap
   (two 19.5M copies, drift, every change duplicated). The only durable
   outcomes are "one backbone, fine-tuned" (case 2) or "one backbone,
   retrained joint" (case 3).

**Fragility notes (DeepSeek #4).** Overfit to 939 records is mitigated by low
LR / 8 epochs and the within-corpus held-out pass, but text-bucket weakness is
a symptom of it -- test on OOD sessions before full trust. The throwaway
readout choice is acceptable (the core direction is robust to it). The
truncated-BPTT-depth-1 gradient is a biased estimate of the true multi-step
gradient, but for a slot-level relevance objective the dominant gradient flows
through the step that produced that slot's state; the smooth loss drop and the
strong result confirm the bias is not harmful in practice. The only fragility
that could block shipping is forgetting, which Sec.5 step 1 measures.

**No further readout-only experiments. No new architectures.** The path is
clear: validate, then either ship (ensemble of s1/s3/s4/s5) or retrain with
the joint objective from Sec.4.

---

## 6. Pointers

- Design: `docs/STRM-architecture-proposal.md`, `docs/STRM-implementation-plan.md`,
  `docs/strm-transformer-relocator-plan.md`.
- Trained-components reproduction + Mamba3 swap: `docs/training-guide.md`.
- The Stage 3 code: `scripts/finetune_backbone_flatlast_margin.py` (the
  fine-tune), `scripts/generate_onyx_doc_ring_traces.py --capture-pre-state`
  (the trace with `slots_pre_state` + `slots_step_input`),
  `src/subconscious/working_memory.py` (`capture_pre_state`, `RingSlot.u`),
  `scripts/probe_head_to_head_onyx.py` (the #5 shared-body readout + sampler +
  `margin_ranking_loss` + `_drop_self_slot`), `scripts/probe_strm_selectivity_real.py`
  (the live serve gate, `--backbone`).
- Per-experiment detail: the project memory index
  (`pondr-strm-phase1f7-stage3-flatlast-finetune-result` and the chain of
  `...SUPERSEDED...` memories back through Phase 0a/0b). Each memory is one
  experiment with its result, root cause, and commit.
- DeepSeek consults #1-#4 (diagnosis = pro, `deepseek-v4-pro:cloud` via local
  Ollama at `127.0.0.1:11434`; Oracle labeling = flash). Consult #4 (the
  results review + from-scratch recipe) is the basis of Sec.3-Sec.5.