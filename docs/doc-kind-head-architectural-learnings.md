# DocKindHead: Architectural Learnings for the JGSBackbone+Head Primitive

Date: 2026-07-17
Status: Living document. The JGSBackbone + downstream classifier head is a
**new primitive**; this captures the best-practices we are establishing for
its use, derived from the DocKindHead work (the first real downstream job for
the trained SSM, Phase 3c Sec 7.11 deferred step).

The DocKindHead is NOT yet shipped (fails the strict gate by one snapshot
example on its best single-head variant). The learnings below are valid
regardless of the ship decision -- they are about the *primitive*, not just
this head.

---

## 1. The primitive

A trained `JGSBackbone` (19.5M-param state-space model, Phase 2a, frozen at
serve) plus a small downstream classifier head that is a `JGSInstance` on top
of it. The head owns its instance params (input/output projections + LoRA,
state_lora, decomposed gate) + a classifier; the backbone is held via
`object.__setattr__` so `head.state_dict()` EXCLUDES the ~19.5M backbone
params -> lean checkpoints, and an `AdamW(head.parameters(), ...)` optimizer
naturally leaves the backbone alone. Two heads ship on this primitive today:
`RetrievalGate` (Phase 2b, val 0.826) and `DocKindHead` (this work).

The open question we are answering empirically: **how do you build a head on
this primitive well?** This document is the accumulating answer.

---

## 2. What the doc-kind task is

5-class, **mutually exclusive** single-label classification: a doc is exactly
one of `{point_in_time_snapshot, decision_update, plan, reference, other}`.
The label is consumed by the contradiction guard: `snapshot` -> ask_user
(complementary, non-mutating); `decision_update` -> supersede (bypasses the
guard). So `snapshot -> decision_update` confusion is **unsafe** (wrong-
supersede); the reverse is merely conservative (extra ask_user). This
asymmetry is load-bearing throughout the design.

Ship gate (strict, decided up front): `unsafe_cell <= 1` AND
`snapshot_recall >= 0.70` AND `decision_update_recall >= 0.70` AND
`val_acc >= 0.55` AND snapshot_recall Wilson-CI95 lower bound >= 0.50.

---

## 3. Established best-practices (use for future heads on this primitive)

### 3.1 Pool the step OUTPUT, not the raw recurrent state
The first design pooled the raw recurrent state (`state.mean(dim=1)`,
`[1,16,384] -> [1,384]`, mirroring `DecomposedGate._pool`). It **mode-collapsed**
on real bge-small embeddings of similar enterprise prose (val nailed at 0.25
for 40 epochs, loss barely below random `ln(5)=1.61`) -- the frozen raw state
was not linearly separable for subtle distinctions. Pooling the per-section
**step output** (the learned `output_proj` readout, `[1,256]` each -- the same
signal `RetrievalGate` classifies on) IS separable. **Lesson: pool the
backbone's learned readout, not its hidden state.**

### 3.2 Mean-pool has a blind spot; attention-over-sections fixes it (root cause #3)
`torch.stack(outputs, dim=1).mean(dim=1)` averages every section's step output
with **equal weight**, discarding WHICH section carries the discriminative
signal. For doc-kind, the date that distinguishes "state AS OF T" (snapshot)
from "decision MADE ON T" (decision_update) lives in one section; mean-pool
dilutes it across N sections of boilerplate. Replacing the mean with a
learned **additive attention** readout (`attn_key=Linear(d,64)`,
`attn_query=Parameter(zeros(64))`, `softmax((keys*query).sum(-1))` over
sections, weighted sum) **more than doubled** `decision_update` recall
(0.29 -> 0.647 on clean labels) and, for the first time, balanced both guard
classes near the gate simultaneously. **Lesson: when the discriminative signal
is section-local, mean-pool is the bottleneck, not the frozen backbone -- use
attention over the per-section step outputs.**

### 3.3 Clean A/B via zeros-init
The attention `attn_query` is initialized to ZEROS, so softmax over equal
scores is uniform and attention == mean-pool EXACTLY at init. This gives a
clean A/B starting point with no random-init luck dependency: the arch change
is justified if attention-on beats attention-off, and the comparison starts
from an identical baseline. An exact-equality test pins this. **Lesson:
init a new readout so it reproduces the baseline at init, then let it learn to
diverge -- makes the A/B unambiguous.**

### 3.4 Re-inject what the pool discards (Phase 4 temporal feature)
A 6-dim pure-regex doc-level feature `[has_any_explicit_date, has_as_of_phrase,
has_decision_phrase, n_explicit_dates_norm, first_date_in_heading,
has_plan_phrase]` concatenated with the pooled embedding (first Linear widens
`256+k -> 128`). No `today` reference -> train/serve-invariant. This FIXED the
snapshot class (snap 0.55 -> 0.75, unsafe 1 -> 0, CI clears) -- a doc-level
hint re-injects the date/framing signal the mean discards. Orthogonal to
attention (feat adds a doc-level signal; attention finds the section). **Lesson:
a cheap hand-engineered doc-level feature can break a class-specific ceiling
the pool can't see; combine with attention, don't substitute.**

### 3.5 Severity loss for asymmetric confusion -- with the sign-trap fix
For an unsafe confusion direction (snap -> dec), add `penalty * p(dec)` when
truth is snapshot (NOT `penalty * -logp[dec]` -- that is BACKWARDS: `-logp[dec]`
is LARGE when `p(dec)` is SMALL, i.e. it rewards the model for being correct and
vanishes when wrong; the fix is `penalty * exp(logp[dec])` = `penalty * p(dec)`).
The reverse (dec -> snap) stays on base CE (it's the safe/conservative
direction). **Lesson: when a confusion direction is unsafe, penalize the
probability of the wrong class directly, and check the sign by hand -- the naive
log-form is inverted.**

BUT: the severity loss is a **crutch for mean-pool's blindness**. With attention
doing the separation, the penalty becomes load-bearing for the unsafe
direction (without it, the head opens `unsafe_cell` to 2-3 to lift dec) but
over-suppresses dec when high (caps dec recall). See section 4.

### 3.6 Gate-aware checkpoint selection -- tuple, not scalar
Checkpoint the epoch that best satisfies the SHIP GATE, NOT the best-val_acc
epoch. A lower-acc epoch can be far more gate-safe; best-val_acc once had
`unsafe=2` while a later epoch had `unsafe=0`. The score is a tuple
`(safe, min(snap_recall, dec_recall), acc)` compared lexicographically -- safe
first, then the binding guard class, then acc. NOT a weighted scalar: the gate
is a conjunction, and a scalar would let a high-acc unsafe epoch beat a safe
one. **Lesson: select checkpoints on the gate tuple, lexicographically; never
on a single accuracy scalar.**

### 3.7 Small-n honesty: Wilson CI on the binding guard class
A point recall estimate on n=17-20 val examples is nearly meaningless. The
gate requires the snapshot_recall Wilson-CI95 lower bound >= 0.50 (GBrain
lesson). At n=17, snap=12/17=0.706 gives CI lower 0.469 (fails); snap=13/17=
0.765 gives 0.527 (clears) -- ONE example swings the bar by 0.06. **Lesson:
gate on the CI lower bound, not the point estimate, for any guard class with
small val n; design the bar knowing the n.**

### 3.8 Class-weight cap + gradient accumulation for per-doc SGD
Each doc is an independent SSM forward (per-doc state, variable-length section
sequence) so the forward can't be batched the way `RetrievalGate` batches
single-step queries. Uncapped inverse-frequency class weights gave a 3-example
class weight 11.2x and destabilized per-doc SGD (mode-collapse). CAP weights at
3.0. Use gradient accumulation (`accum_steps=16`, effective mini-batch) so
per-doc SGD mode-collapse doesn't happen. **Lesson: for variable-length per-
instance forwards, cap class weights and accumulate gradients to fake a mini-
batch.**

### 3.9 Label discipline (the corpus is the ceiling if the labels lie)
The doc-kind labels came from a DeepSeek-flash teacher with a
confidence>=0.7 + abstain gate. A 3-teacher panel audit (flash + glm + gemma
majority) later found flash **over-assigns decision_update confidently** (0.85-
0.90 on support threads) -- the confidence gate did NOT catch it, and it
contaminated BOTH train and val labels (and the synthetic blind-verify). 58/331
(17.5%) of flash labels were overruled. The head was partly vindicated by the
relabel (~0.08 of "misses" were flash mislabels) but a real arch ceiling
remained. **Lesson: a teacher's confidence does not guarantee label quality;
audit with an independent teacher before trusting labels as ground truth;
relabel (majority panel) before concluding an arch is ceiling-bound.**

---

## 4. The structural snap/dec coupling (the open finding)

Across three penalty brackets (unsafe_penalty = 0, 1.0, 2.0; attention + temporal
feature on; clean 261-train / 76-val), the single 5-way head can achieve
**snap=13/17 (snapshot CI clears 0.50) OR dec>=12/17 (dec clears 0.70) at a
safe (unsafe<=1) epoch -- but never both simultaneously.**

| unsafe_penalty | best safe epoch | snap | dec | unsafe | acc | snap CI_lo | gate miss |
|---|---|---|---|---|---|---|---|
| 0 (pure CE) | ep25 | 13/17=0.765 | 11/17=0.647 | 1 | 0.566 | 0.527 | dec only |
| 1.0 | ep40 | 12/17=0.706 | 13/17=0.765 | 0 | 0.513 | 0.469 | CI + acc |
| 2.0 | ep20 | 12/17=0.706 | 12/17=0.706 | 1 | 0.553 | 0.469 | CI only |
| 5.0 (v5) | ep21 | 11/17=0.647 | 11/17=0.647 | 1 | 0.500 | 0.41 | snap+dec+acc+CI |

This is a **fixed separability budget**: the 5-way softmax trades the two
date-stamped classes against each other. Penalty tuning is exhausted (3
brackets, converged). The best single head (pen=2.0, ep20) clears 4/5 gate
criteria -- both guard classes 0.706, unsafe 1, acc 0.553 -- and misses only
the snapshot Wilson CI lower bound (0.469 vs 0.50, one snapshot example).

Mechanism (confusion at pen=2.0 ep20): `snap [13,1,2,0,1]`,
`dec [1,11,4,0,1]`, `plan [2,12,14,0,1]`. The remaining dec error is dec ->
**plan** (4 of 17), not dec -> snap (1, controlled by the severity loss) and
not dec -> other (1). The plan/dec boundary is the muddy one at the safe
operating point.

---

## 5. The multi-gate hypothesis (best-practice to TEST, deferred)

The coupling above is the **cost of mutual exclusion** in a single 5-way
softmax. Hypothesis (to test empirically, deferred): **multi-gate (multiple
specialized per-criterion heads) helps when the target classes/criteria are
NOT mutually exclusive; a single softmax is the natural fit when they ARE
(mutually-exclusive single-label).** Doc-kind IS mutually exclusive, so the
single head is the architecturally-correct default; a multi-gate would relax
the softmax competition that produces the snap/dec tradeoff.

Deferred experiments to test the hypothesis (run LATER, not now):
1. **Ensemble the penalty heads** (pen=0/1/2/5 best.pt, logit-average or
   majority vote). Cheap (no retrain). Hope: pen=0 contributes snap=13/17
   (CI clear), pen=2 contributes dec=12/17, the vote lands BOTH. Serve cost 3x
   forward unless distilled back to 1 head.
2. **Specialized binary snap-vs-dec gate** (sigmoid, trained ONLY on snap+dec
   examples, excluding plan/other/ref). Gives the hard boundary dedicated
   capacity instead of competing in a 5-way softmax. More build (new head +
   training pipeline + serve cascade + tests + guard wiring), 1x serve.
   Highest expected ceiling.
3. **Cascade**: 5-way head for the easy classes (plan/reference/other) + binary
   resolver only for the snap/dec boundary.

If a multi-gate experiment breaks the coupling net-positive, that is a
best-practice for non-mutually-exclusive (or forced-tradeoff) criteria on this
primitive. If it trades one problem for another, the single softmax is
confirmed correct for mutually-exclusive cases. Either outcome is a
best-practice learned from data.

---

## 6. Single-head levers tried and their outcomes

| lever | hypothesis | outcome |
|---|---|---|
| Step-output pool (vs raw state) | separable readout | FIXED mode-collapse; base for all else |
| Temporal feature (Phase 4) | re-inject date signal the mean discards | FIXED snapshot (0.55->0.75, unsafe 1->0, CI clears) |
| Attention-over-sections (Phase 5) | find the date-bearing section | DOUBLED dec (0.29->0.647); balanced both guard classes; NOT a capacity ceiling |
| Severity loss (unsafe_penalty) | suppress the unsafe snap->dec direction | crutch for mean-pool; load-bearing for unsafe but over-suppresses dec with attention |
| Cleaner panel labels (v4) | remove teacher noise | partly confounded v3; snap up, dec DOWN (severity + coupling) |
| More flash data (v3, 268 synth) | more dec examples | DECISIVE NEGATIVE: dec pinned 0.33 at 95 AND 199 train decisions (flash-contaminated) |
| Penalty bracket (0/1/2) | thread snap+dec | EXHAUSTED: structural coupling; gets snap=13 OR dec>=12 at safe, never both |
| Lower lr at pen=2.0 | escape overfit, find a flat min with both | NEGATIVE: lr 1.5e-4 (half) at pen=2.0 -> best ep35 snap=12/17=0.706, dec=7/17=0.412, acc=0.368 (WORSE than lr=3e-4); no safe epoch has both guard classes >=0.70. Slower convergence landed on a worse minimum. |
| More epochs | more training | RULED OUT: overfits after ~20-40ep, gate never improves |
| Unfreeze backbone | representation is the ceiling | NOT TRIED; risks RetrievalGate (shared backbone, val 0.826) |
| Targeted snap/dec boundary data | hard negatives at the frontier | NOT TRIED |

---

## 7. Reproducibility

- Architecture + wiring: `src/subconscious/doc_kind_head.py`,
  `src/subconscious/training/doc_kind_training.py`,
  `src/subconscious/training/routing_training.py`,
  `src/ingestion/doc_kind.py`, `scripts/train_doc_kind_head.py`.
- Tests: `tests/test_doc_kind_head.py` (39 tests) + `tests/test_contradiction.py`
  (29 tests). All green.
- Attention arch committed: `a116930`.
- Training data: `data/training/doc_kind_head/pairs_clean_{train,val}.jsonl` (261
  train / 76 val, 3-teacher panel majority labels, gitignored).
- Retrain artifacts: `data/training/doc_kind_head_attn*` (gitignored).
- Cold-start invariant: the canonical `data/training/doc_kind_head/best.pt` is
  kept ABSENT during experiments so `build_doc_kind_tagger` falls back to Bonsai.
  It is restored to the canonical path ONLY after the gate is met.

See memory: `pondr-doc-kind-backbone-head-shipped`, `jgs-head-multi-gate-best-
practice-hypothesis`.