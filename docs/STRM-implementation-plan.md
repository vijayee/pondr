# STRM — Implementation & Training Plan

**Companion to `docs/STRM-architecture-proposal.md`. v0.2 (draft for review).**

This is a phased build plan, not a spec. Every phase has a concrete deliverable,
the file it touches, the convention it reuses, and a stop condition. Two phases
(0a, 0b) are **go/no-go gates** — both PASSED 2026-07-19 (0a AUC 0.810; 0b GO,
ship LINEAR), and Phase 1 shipped (be77e35), so Phases 2-6 are now the live work.

## How to read this

- **Reuse, don't reinvent.** STRM heads copy existing trainers. The mapping
  (verified against the tree):

  | STRM piece | Copy this file | Notes |
  |---|---|---|
  | Relevance head (supervised classifier) | `src/subconscious/training/doc_kind_training.py` + `src/subconscious/doc_kind_head.py` | CE on frozen backbone, class-weighted, `_gate_score`-style best.pt |
  | Recoverability head (supervised, classifier or regression) | `doc_kind_training.py` (classifier); no regression template exists — borrow dataclass + CLI shell, swap CE for MSE | labels come from the Phase 0a probe decoder |
  | Graduation head (supervised) | same as Recoverability | labels from replay (the long pole) |
  | Latent-dynamics head (linear, not JEPA) | `doc_kind_training.py` shell + a `Linear` head (or closed-form ridge) | v1 ships `ẑ_{t+1}=Az_t+b` per the 0b probe (linear beat EMA JEPA on surprise-AUC, frozen backbone can't collapse). No EMA/stop-grad/anti-collapse in v1; JEPA reserved for a future generative rollout |
  | Context-builder Transformer | net-new arch (no Transformer in `src/subconscious/`) | register like `INSTANCE_CONFIGS["doc_kind"]` (`configs.py:96-100`) |

- **Conventions to follow (checklist):**
  - Arch config in `src/subconscious/configs.py` (`INSTANCE_CONFIGS` entry —
    `doc_kind` at `configs.py:96-100` is the template); training hyperparams in a
    separate `*TrainingConfig` dataclass in the trainer module.
  - CLI trainer script in `scripts/`: argparse mirrors the dataclass;
    `sys.path.insert` for `src.*`; seed-based train/val split (or `--train`/`--val`
    for fixed real val); `load_backbone(...)` frozen; `build_embedder(...)`; call
    `train_*_supervised(...)`; `final.pt` + DONE line. Templates:
    `scripts/train_doc_kind_head.py`, `scripts/train_retrieval_gate.py`.
  - Checkpoint shape: `{"head": state_dict, "labels": [...], "val_accuracy": float,
    "epoch": int, "feat_dim": int, "attention": bool}` → `best.pt` + `final.pt`.
    Extra fields read BEFORE constructing the head on load (`routing_training.py:182-218`).
    `_gate_score` lives in `train_log.json`, NOT the checkpoint.
  - Backbone loaded ONCE and shared — `load_backbone(path, BackboneConfig(),
    device)`; re-freeze belt-and-suspenders in the trainer.
  - STRM head trainers use fp32 (the head-trainer convention;
    `_resolve_dtype` warns on a non-fp32 request, `routing_training.py:58-69` —
    backbone pretrain is the separate bf16 path, `configs.py:156`, not relevant
    here); `device="auto"`; `lr=3e-4`, `weight_decay=0.01`, `epochs=20` as
    starting points.
  - Eval inline in the trainer (not a separate script); Wilson 95% CI
    (`_wilson_ci95`, `doc_kind_training.py:217-231`), per-class recall, confusion,
    top-2; ship gate = `all(checks.values())` bool dict; checkpoint selection via
    `_gate_score` lexicographic tuple. (No t-dist CI anywhere in the tree — use
    Wilson.)
  - Serve: a head is on by default when its checkpoint exists, à la
    `--doc-kind-ensemble` defaulting on (`ingest_document.py:191-196`). STRM's
    context-builder plugs into `build_ponder` (`runtime.py:51-157`) and the
    orchestrator seam at `orchestrator.py:323` (`PresentationGate.plan`).

- **Never break the shipped orchestrator.** Every phase ships behind a flag
  defaulting OFF until its gate passes. The heuristic `PresentationGate` stays as
  the fallback.

---

## Phase 0a — Recoverability probe (GO / NO-GO GATE 1) — ✅ DONE 2026-07-19, GO

**Result:** probe `P(state_t, u_i)` AUC val = **0.810** (gate 0.75), beats the free
`k`-baseline (0.732). Decay curve grows monotonically with `k` (val `e` 0.188 →
0.464 at `k=1→8`). State carries lag-independent "which anchor was forgotten"
info, not just older=more-forgotten. **Decision: GO** — the recoverability head
(Phase 2b) is viable; its labels are decoder `D`'s `e(i,t)` (free, no Oracle).
Probe used **ridge regression** (closed-form), not an MLP — an MLP decoder
overfit train (2M params / 334 pairs) and collapsed the train-error variance so
the probe trained on an uninformative target (train AUC 0.50); ridge fixed it
(train AUC 0.866). State rep = per-`d_state`-channel mean per layer (1536-dim
"pooled"); the full 4096-dim state was too slow on this LAPACK. Probe script
(`scripts/_probe_recoverability.py`) is probe-only / not committed; traces under
`data/probe/` are gitignored.

**Question:** is SSM forgetting predictable enough that a probe can estimate
recoverability from `state_t`? If no, the salience mechanism is not viable and
the plan stops (or shrinks to fixed-interval refresh).

**No retraining.** Uses the already-trained `WorkingMemory` backbone
(`data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt`).

**Deliverables:**
1. `scripts/probe_recoverability.py` — runs real activity streams through
   `WorkingMemory`, logging `(u_1..u_T, state_t@each step)` to
   `data/probe/recoverability/traces.pt`.
2. A **recovery decoder** `D(state_t) → û_i` (small MLP) trained to reconstruct a
   past input `u_i` from a later state. Reconstruction error `e(i, t)` is the
   ground-truth forgetting signal. (This is `forget(t) = D(g(z_{t+k}), z_t)` from
   the proposal.)
3. A **lightweight probe** `P(state_t, anchor_i) → ê(i, t)` predicting the error
   without doing recovery.
4. Metric: probe AUC, plus a per-`k` decay curve.

**Data:** real conversation traces from the existing corpus (DialogSum / the
ingested episode store). No Oracle calls.

**Stop condition / gate:**
- AUC ≥ ~0.75 (calibrate against the decay curve) → GO; the recoverability head
  (Phase 2b) is viable and its labels are now generated.
- AUC poor AND discretization suspected → try a Mamba2 backend swap (Mamba2, not
  Mamba3 — `mamba3-cuda` build fails here, `step()` is `NotImplementedError`).
- AUC poor with no obvious fix → STOP; simplify STRM to fixed-interval refresh,
  drop salience + recoverability + graduation heads.

**De-wonk note:** the decoder `D` and probe `P` are probe-only artifacts; they are
NOT shipped. `scripts/probe_*` is a probe script — do not commit it (per
commit-at-will: never commit untracked probe/scratch). The **output** (the AUC
number + the generated label set) is what survives into Phase 2b.

---

## Phase 0b — Latent-dynamics gate (GO / NO-GO GATE 2, two steps) — ✅ DONE 2026-07-19, GO (ship LINEAR)

**Result (step 1 — linear baseline):** linear `z_{t+1}=Az_t+b` (ridge) R² =
**0.297** over constant-mean (gate 0.15) — dynamics ARE learnable. **GOTCHA
caught by de-wonk:** at the 1536-dim "pooled" state the linear fit was
underdetermined (N=957 transitions < D=1537; ridge shrinks to the mean) and
gave R²=-0.088 (false NO-GO). At the 384-dim "last layer, mean over `d_state`"
rep (N > D, what the JEPA predictor operates on) linear R²=0.297 (GO). The probe
gained a `--state-rep` flag so the comparison is fair.

**Result (step 2 — EMA JEPA vs linear surprise):** the EMA JEPA predictor `g`
trained with `jepa_contrastive_loss` (cosine + logsumexp negatives + 0.1·MSE)
scored surprise-AUC = **0.565** in its *native cosine distance* — *underperforms*
the linear predictor's L2-residual surprise-AUC = **0.7625** (gate 0.70). Latent
variance stayed bounded (no collapse). Measuring `g`'s surprise in L2 (a
cross-objective mistake, since `g` is cosine-trained) was an early
de-wonk catch — the meaningful comparison is `g`-cosine vs linear-L2, and linear
still wins.

**Decision: GO, ship the latent-dynamics head LINEAR.** A closed-form linear map
clears the surprise gate and beats the JEPA predictor; the backbone is frozen so
collapse cannot occur, and the JEPA EMA/stop-grad/anti-collapse machinery solves
a non-problem in v1 (see proposal §6.2 for the four reasons). Phase 2c ships
`ẑ_{t+1}=Az_t+b`, not JEPA. JEPA is reserved for a future generative rollout
(predict `k>1` for imagination) — a different feature, past v1.

---

**Question (two parts, for reference):** (i) does the WM recurrent state `z_t`
have *learnable transition dynamics at all*, and (ii) if so, can a
properly-instrumented EMA predictor avoid collapse on it? If (i) is no, drop the
latent-dynamics head before paying for any EMA machinery. If (ii) is no, drop it;
the three supervised heads still stand. *(Both answered GO above; step 2's
"avoid collapse" question is moot once the linear baseline already beats JEPA.)*

**Step 1 — linear baseline (cheap, run first).** On the Phase 0a traces, fit a
linear `z_{t+1} ≈ Az_t + b` (least squares) and compare one-step prediction MSE
to a constant-mean baseline (`ẑ_{t+1} = mean(z)`). **Gate:** linear must beat
mean by a clear margin. If a linear map can't beat mean, the latent has no
learnable dynamics — no nonlinear JEPA predictor rescues it, so DROP the
latent-dynamics head here and skip step 2. This is the cheapest possible test of
the deeper risk and it runs in minutes on logged traces.

**Step 2 — EMA collapse check (only if step 1 passes). Reuses existing JEPA
machinery** — this is not net-new:
- `_update_ema(target, online, decay)` (`pretrain.py:148-151`).
- Target = `copy.deepcopy(backbone)` with `requires_grad_(False)` (`pretrain.py:182-184`).
- `jepa_loss.step_loss` / `jepa_contrastive_loss` (`jepa_loss.py:24-63`) — the
  batch-negatives logsumexp is the anti-collapse term.

**Deliverables (step 2):**
1. `scripts/_probe_latent_dynamics.py` — on the Phase 0a traces, train a predictor
   `g(z_t) → ẑ_{t+1}` with the EMA target + stop-grad + the `jepa_loss` collapse
   penalty. `k=1` (per proposal §7).
2. Metrics: prediction MSE, latent variance/covariance over training (collapse
   detector — variance → 0 means collapse), and a "surprise" head AUC on
   held-out anomalous steps (if reconstructable).

**Stop condition / gate (step 2):**
- Latent variance stays bounded AND surprise-AUC > chance → GO; the
  latent-dynamics head (Phase 2c) is viable.
- Collapses despite the anti-collapse term → either (a) strengthen the penalty
  (more negatives, higher temperature, add a VICReg-style variance/covariance
  regularizer) and retry once, or (b) DROP the latent-dynamics head. The three
  supervised heads do not depend on it.

**De-wonk note:** `A`, `g` are probe-only; not shipped. Probe scripts not
committed. (Step 2 is retained in the plan as the de-risk record; its outcome —
JEPA loses to linear — is what makes Phase 2c a linear head.)

---

## Phase 1 — Ring buffer + state read-out plumbing (no heads yet) — ✅ DONE 2026-07-19 (be77e35)

**Shipped:** `WorkingMemory` ring buffer of `(y_t, source_id, text)` provenance
slots + `state_tensors()` live read-out + `ring_buffer()` accessor. 14 new tests
in `tests/test_working_memory_ring.py`. `K=0` default verified byte-identical to
Phase 2c (state evolution bit-identical `K=0` vs `K>0`, ring is observation-only
post-step/post-decay). **Key correction vs this plan's draft:** the slot vector
is the step *output* `[1, output_dim=256]` for `working_memory`, NOT a hardcoded
`384` — the buffer is dimension-agnostic, stores whatever `step` emits.
`InstanceConfig.ring_capacity` added; `retrieved_sources` length-mismatch guard.

**Goal:** expose `y_t` history and `state_t` from `WorkingMemory` so heads can
read them, without changing any shipped behavior.

**Deliverables:**
1. Extend `WorkingMemory` (`src/subconscious/working_memory.py`) with a ring
   buffer `self.ring: deque` of recent step outputs, where each slot is a triple
   `(y_t, source_id, text)` — the vector **plus provenance** back to the event/
   episode that produced it (required by the Phase 3 context-builder to map a
   selected slot back to text; shipping a vector-only buffer now would force a
   Phase 3 redesign). Capacity `K` configurable, default OFF / `K=0` so shipped
   behavior is byte-identical. Add `ring_buffer()` read-only accessor and a
   `state_tensors()` accessor returning the live per-layer `[1, 16, 384]` state
   (distinct from `snapshot()`, which detaches+clones for serialization — the heads
   need a live, on-device view).
2. Config: add `ring_capacity` to the `working_memory` entry in `INSTANCE_CONFIGS`
   (`configs.py`), default 0.
3. Tests (`tests/test_working_memory_ring.py`): with `K=0`, existing WM tests
   pass unchanged; with `K>0`, the buffer holds the last `K` slots (with
   provenance) and pops FIFO.

**Stop condition:** all existing WM tests green; ring tests green; `K=0` path
byte-identical to today (verified by re-running the Phase 2c suite).

**De-wonk note:** `state_tensors()` must NOT detach — heads train against the
live graph. But it must NOT alias in a way that lets a caller corrupt the state
out from under the SSM; document the contract (read-only for training, do not
write). `snapshot()` stays the serialization path.

---

## Phase 2 — The four heads, one at a time

Order is by dependency + label availability, easiest first. Each head is its own
sub-phase with its own gate. All train on the frozen backbone, all produce
`best.pt`/`final.pt`/`train_log.json`, none ship until their own gate passes.

### 2a — Relevance head (supervised classifier)

**Why first:** the context-builder (Phase 3) needs it, and the label pipeline is
the cheapest to fix.

**Label pipeline (must build first):** `record_feedback` reduces 1-5 to a
compounded boost multiplier and **does not persist the raw rating**
(`store.py:707-749`). Add a raw-rating JSONL tap: in
`store.record_feedback` (or just before the boost write), append
`{unit_id, rating, query, slot_index, timestamp}` to
`data/training/strm_relevance/feedback.jsonl` when a flag
`strm_relevance_logging` is on (default off). Until enough real labels accumulate,
generate synthetic pairs à la `scripts/generate_jepa_training_data.py` (query →
relevant slots) using the existing `OracleClient` (`src/training/oracle_labeling.py`)
and `prompts.py` if needed.

**Deliverables:**
1. `src/subconscious/relevance_head.py` — `RelevanceHead(JGSInstance)` modeled on
  `DocKindHead` (`doc_kind_head.py:51-209`): one `nn.Sequential` head over the
   pooled ring-buffer slot, sigmoid output per slot. Register in
   `INSTANCE_CONFIGS["strm_relevance"]` (copy the `doc_kind` entry).
2. `src/subconscious/training/relevance_training.py` — copy
   `doc_kind_training.py`; swap the 5-class CE for per-slot BCE; keep
   `_gate_score`-style best.pt selection (define a relevance gate: e.g. top-3
   recall ≥ threshold on held-out queries).
3. `scripts/train_relevance_head.py` — copy `scripts/train_doc_kind_head.py`.
4. Eval: per-query top-3 / top-5 recall against the feedback JSONL + synthetic
   val; Wilson CI on the small real-label slice.

**Stop condition / gate:** relevance gate passes on the val set (top-3 recall ≥
~0.6 as a starting bar — calibrate). Ship behind `--strm-relevance-head PATH`,
default off.

### 2b — Recoverability head (supervised; classifier or regression)

**Labels:** already generated by Phase 0a (the `e(i, t)` reconstruction errors →
binarize at θ for classifier, or use raw for regression). No new labeling.

**Deliverables:**
1. `src/subconscious/recoverability_head.py` — reads `state_t` directly via its
   own projection head (NOT `y_t` — `y_t` is the lossy `W_C` read-out,
   `ssm.py:105-106`). This is the architectural reason it's a separate head, not
   a relevance-head variant.
2. `src/subconscious/training/recoverability_training.py` — copy
   `doc_kind_training.py` for the classifier form; if regression, borrow the
   dataclass + CLI shell and swap CE for `F.mse_loss` (no existing regression
   template — `gate_training.py:37-40` is framework-only, the only MSE precedent).
3. `scripts/train_recoverability_head.py`.
4. Eval: AUC (matching the Phase 0a metric, on a held-out trace split) + Wilson CI.

**Stop condition / gate:** held-out AUC ≥ the Phase 0a bar. Ship behind
`--strm-recoverability-head PATH`, default off.

### 2c — Latent-dynamics head (LINEAR in v1 — the cheap one, per Phase 0b)

**Phase 0b passed and decided the shape:** ship the linear predictor
`ẑ_{t+1} = A z_t + b`, NOT JEPA. The 0b probe showed linear surprise-AUC (L2
residual) = 0.7625 beats the EMA JEPA predictor's cosine surprise-AUC 0.565, and
the frozen backbone means collapse cannot occur — so the EMA / stop-grad /
anti-collapse machinery is not built in v1 (see proposal §6.2 for the four
reasons). This is now the *cheapest* of the four heads, not the hard one.

**State rep (from 0b):** use the **last-layer, mean-over-`d_state`** projection
`z_t ∈ [384]` (N > D, the rep on which linear R²=0.297 was measured). The 1536-dim
"pooled" rep was underdetermined (N < D) — do not use it for the linear head.

**Deliverables:**
1. `src/subconscious/latent_dynamics_head.py` — a `Linear(state_dim,
   state_dim)` predictor `g(z_t) → ẑ_{t+1}` (plus bias). Models on `DocKindHead`'s
   shell, not on `pretrain.py`. No EMA, no stop-grad, no negatives.
2. `src/subconscious/training/latent_dynamics_training.py` — copy
   `doc_kind_training.py`; swap the head + loss for `F.mse_loss(g(z_t), z_{t+1})`
   on `(z_t, z_{t+1})` consecutive-step pairs from the Phase 0a traces (free,
   self-supervised labels — the next state IS the label). Standard ridge-style
   MSE; closed-form ridge is also acceptable for v1 (the probe already used it).
   Keep `_gate_score`-style best.pt selection.
3. `scripts/train_latent_dynamics_head.py`.
4. Eval: prediction MSE (vs the 0b linear R²=0.297 / MSE 0.00017 baseline) +
   surprise-AUC on held-out mismatched next-states (vs the 0b linear
   surprise-AUC 0.7625). No collapse watch needed (linear cannot collapse).

**Stop condition / gate:** surprise-AUC ≥ the 0b bar (≥ 0.70) on held-out
transitions; prediction MSE ≤ the 0b linear baseline. Ship behind
`--strm-latent-dynamics-head PATH`, default off.

**De-wonk note:** do NOT drag EMA / stop-grad / `jepa_loss` negatives into this
head in v1 — the 0b probe measured that they make it *worse* on the surprise
signal, and the frozen backbone means there is nothing to collapse. The honest
upgrade path if R²=0.30 binds is a light **MSE-trained MLP** (non-linear, same L2
objective) — NOT JEPA. JEPA only re-enters for a future generative rollout
(predict `k>1` for imagination), a separate feature past v1; if that is ever
built, this is the head it reuses, and *then* the EMA/anti-collapse machinery
earns its place.

### 2d — Graduation (v1 proxy first, v2 replay head — the long pole)

**v1 — relevance-lifetime proxy (ship first, no replay).** Graduation score = the
slot's integrated relevance over its lifetime in the buffer (`∫ r_i dt`). A fact
that stayed relevant for many steps is probably worth keeping. This breaks the
circular dependency for free: "would have been needed later" is approximated by
"was relevant for a long time," which needs no replay and no downstream pipeline.
- Deliverable: a small module computing `∫ r_i dt` per slot from the relevance
  head's `r_i` stream; threshold → graduate. No training, no checkpoint.
- Ship behind `--strm-graduation-proxy`, default off. This is the heuristic
  baseline the v2 head has to beat.

**v2 — replay-supervised head (the long pole).** Replay logged streams, mark
which compressed-out facts were later needed (i.e., would have triggered a
salience recall or a consumer `search_memory`/`expand`), train the head to
predict that from `(state_t, slot_content, llm_signal)`. This is the true
credit-assignment signal but depends on the whole downstream pipeline
(relevance → context-builder → consumer behavior), so the labels are noisy and
slow. Start the replay logger in Phase 2a so labels accumulate while the other
heads train.

**Deliverables (v2):**
1. `src/subconscious/graduation_head.py` — reads `state_t` + slot content + the
   `llm_signal`-derived importance input (`forgetting.py:58-64` already defines
   the signal vocabulary — reuse it, do not invent a new one).
2. `src/subconscious/training/graduation_training.py` — copy `doc_kind_training.py`
   (classifier: graduate / don't) or the regression shell (priority score).
3. `scripts/train_graduation_head.py`.
4. A replay-label generator `scripts/generate_graduation_labels.py` consuming the
   replay log.
5. Eval: "would-have-been-needed" recall on held-out replays + Wilson CI;
   **must beat the v1 proxy** to be worth shipping.

**Stop condition / gate (v2):** v2 graduation gate passes AND v2 beats v1 proxy
on the held-out replay eval. Ship v2 behind `--strm-graduation-head PATH`, default
off, keeping v1 as fallback. Feeds the existing consolidator (`consolidate.py`)
as prioritized input — the consolidator's LTM-internal logic is unchanged.

**De-wonk note:** do not block the "remembering" feature on the long pole — v1
ships it. If v2 labels are too sparse to beat v1, keep v1 and defer v2.

---

## Phase 3 — Context-builder Transformer (the learned PresentationGate)

**Goal:** replace the heuristic `PresentationGate.plan` at `orchestrator.py:323`
with the learned builder, behind a flag, heuristic as fallback.

**This is the seam the codebase already designates.** `PresentationGate`'s
`ReplayBuffer` (`presentation_gate.py:178-179`) is "for the deferred learned gate"
(`presentation_gate.py:168`), and `record_outcome` (`presentation_gate.py:250-263`)
is already auto-firing after every query (`orchestrator.py:455-462`) and persisted
via `store.save_presentation_outcomes` (`orchestrator.py:897-898`). **No
read-back for training exists today** — Phase 3 builds the trainer that consumes
`serialize_buffers()` (`presentation_gate.py:368-380`).

**Deliverables:**
1. `src/subconscious/context_builder.py` — the Transformer **selector/reranker**
   (attention over the `K` ring-buffer slots with the relevance head's `r` as
   bias, per proposal §4.3). **The consumer-facing output is discrete selected
   text, not the continuous `ctx` vector** — `attn` produces selection weights;
   hard top-m picks the slots; each slot's provenance (`source_id`) maps back to
   source episode text; the output is the bounded text of the selected episodes,
   same shape as `PresentationGate`'s primary/compressed split. Soft attention is
   the training-time differentiable surrogate; hard top-m is the serve path.
   Emits a `PresentationPlan`-like object (smallest seam: keep `SSMChunker.chunk`
   at `orchestrator.py:336` and `format_for_llm` working).
2. **Ring buffer carries provenance** — each slot is `(y_t, source_id, text)`, not
   just `y_t` (Phase 1 must store this). Without it the selector can't map back to
   text. This is a Phase 1 / Phase 3 contract; do not let Phase 1 ship a
   vector-only buffer that Phase 3 then can't use.
3. Config: `INSTANCE_CONFIGS["strm_context_builder"]`.
4. `src/subconscious/training/context_builder_training.py` — train on context
   quality: does the block let a downstream consumer answer correctly? Reuse the
   Phase 3c / ERAG-Bench eval labels. Consume the `PresentationGate` override
   buffer (caller overrode the heuristic) as seed supervision.
5. `scripts/train_context_builder.py`.
6. Wiring: `build_ponder` (`runtime.py:51-157`) loads the builder when its
   checkpoint exists; the orchestrator at `orchestrator.py:323` calls the builder
   instead of `presentation_gate.plan` when `--strm-context-builder` is on,
   falling back to the heuristic otherwise.

**Stop condition / gate:** on the ERAG-Bench / Phase 3c factual-accuracy eval, the
learned builder ≥ the heuristic `PresentationGate` at equal token budget. Ship
default-off until that holds; then flip the default à la `--doc-kind-ensemble`.

**De-wonk note:** start at the smallest seam (replace `plan` only, emit a
`PresentationPlan`). Do NOT replace `SSMChunker.chunk` or `format_for_llm` in
this phase — those are deeper seams only needed if the plan-enum interface can't
express what the builder wants to say. Replacing all three at once is the kind of
"weird" that de-wonk catches.

---

## Phase 4 — Salience trigger + state-conditioned self-triggered retrieval

**Goal:** make retrieval self-triggered by the salience signal (the internal,
pre-emptive `search_memory`), instead of only externally by the prompt.

**Deliverables:**
1. In the orchestrator (or a small `src/subconscious/salience.py`), compute
   `salience(anchor) = recoverability < θ ∧ relevance > φ` per anchor, with the
   surprise-gate from the latent-dynamics head (high surprise → suppress, per
   proposal §5 step 9). Thresholds `θ`, `φ` from the Phase 2b/2a eval data.
2. On salience, emit an LTM pointer → call the existing retriever with a
   **state-conditioned query** (the current `z_t` projection, not just the prompt
   embedding — `retrieval_gate.py:126-146` today embeds the prompt only). This is
   the new retrieval-query shape; reuse `retrieve_with_plan`.
3. Re-inject the returned episode via `WorkingMemory.inject(emb)` with the
   **pin tag** (a learned token-type embedding added to `u_{t+1}`, same dim as
   the input so `d_model` is unchanged, per proposal §4.5) so `W_A` retains it
   over the next `K` steps.
4. Freshness watermark `Δ` (proposal §5): for anchors younger than Thread 2's
   lag, **do not silently suppress the pointer** — emit a typed **stale-uncertain**
   signal to the consumer ("I may know this but have not finished ingesting it")
   alongside the suppressed pointer, so the consumer can wait / re-ask / proceed
   with a stated gap rather than being lied to by omission. This is an
   interaction-design deliverable (a new signal type in the consumer contract),
   not just a threshold.

**Stop condition / gate:** end-to-end — on a long-horizon eval, STRM+recall
answers more factual questions correctly than fixed-interval refresh at equal
recall budget/latency. This is the **ship-deciding experiment** for the whole
primitive.

**De-wonk note:** the pin tag is a token-type *embedding*, not an extra input
feature (an extra feature would change `d_model` and break `W_A`'s
`Linear(384→16)`). Keep `d_model=384`.

---

## Phase 5 — Two-thread continuous ingestion

**Goal:** point the existing ingestion pipeline at the live activity feed so LTM
is warm and recalls are cheap reads; define the freshness watermark `Δ`.

**Deliverables:**
1. An "activity ingest" adapter that feeds the live activity stream into the
   existing TEXT+MD/PDF/Code/DOCX/Web/email parsers (`src/ingestion/`) — verify
   the parsers do something sensible on event-stream input (open question from
   proposal §7; resolve here).
2. Thread 2 loop: continuous, eventually-consistent ingestion + the existing
   consolidation dream pass (`consolidate.py`) + receiving STRM's graduation
   writes and surprise events.
3. Freshness watermark: expose Thread 2's lag `Δ` so Phase 4 can read it.

**Stop condition:** a recall after salience is a cheap read (single retrieval,
no consolidation round-trip); `Δ` measured and bounded.

**De-wonk note:** keep Thread 1 (STM) and Thread 2 (Ponder) sharing only the
input tap, not state (proposal §4.6). Do not let Thread 2 read the WM state
tensors — communicate via the three message types (recall pointer, graduation
write, surprise event).

---

## Phase 6 — Joint fine-tune (Stage B, optional)

**Only if Stage A (frozen backbone, Phases 2-3) plateaus.** Unfreeze the backbone
and jointly train SSM + heads + builder with the combined objective.

**Risk:** catastrophic forgetting of the pretrained representations the
2a/2b/doc-kind work relies on. Use a low LR on the backbone, possibly LoRA on the
SSM rather than full unfreeze. Gate: does joint training beat Stage A on the
Phase 4 ship-deciding eval by enough to justify the risk? If not, stay frozen.

**De-wonk note:** if Stage A already meets the ship-deciding bar, skip Phase 6
entirely. Do not joint-train for its own sake.

---

## Training plan — consolidated

**Data generation (per head):**
- Relevance: raw-rating JSONL tap (Phase 2a) + synthetic via `OracleClient` until
  real labels accumulate.
- Recoverability: Phase 0a decoder errors (free, no Oracle).
- Latent-dynamics: Phase 0a traces (free, self-supervised).
- Graduation: replay labels (long pole; start the replay logger early).
- Context-builder: `PresentationGate` override buffer + Phase 3c / ERAG-Bench
  factual-accuracy labels.

**Objectives:**
- Relevance, Recoverability, Graduation: supervised (CE or MSE), class-weighted,
  with a `_gate_score`-style lexicographic best.pt selection and a bool-dict ship
  gate. Copy `doc_kind_training.py`.
- Latent-dynamics: **linear MSE** — `‖A z_t + b − z_{t+1}‖²` (ridge / closed-form
  or a `Linear` head trained with `F.mse_loss`). Copy `doc_kind_training.py`'s
  shell, NOT `pretrain.py`. JEPA / EMA / anti-collapse is NOT used in v1 (the 0b
  probe showed linear beats JEPA on surprise-AUC and the frozen backbone can't
  collapse); reserved for a future generative rollout.
- Context-builder: context-quality loss (downstream answer correctness under a
  token budget).

**Eval harness (reuse, don't build new):**
- Inline per-class scorecard in each trainer (Wilson 95% CI, top-2, confusion,
  per-class recall) — copy `doc_kind_training.py:217-306`.
- Ship gates as `all(checks.values())` bool dicts; checkpoint selection via
  `_gate_score` tuples.
- The whole-primitive ship-deciding eval (Phase 4): STRM+recall vs fixed-interval
  RAG at equal budget/latency on long-horizon factual accuracy.

**Checkpoint / serve (reuse):**
- `best.pt` + `final.pt` + `train_log.json` per head, under
  `data/training/strm_<head>/`.
- `build_ponder` loads each head when its checkpoint exists; CLI flags
  `--strm-*` default OFF, flipping ON à la `--doc-kind-ensemble` once a head's
  gate passes.
- No HF upload from the trainers (matches existing convention; the private HF
  backup is a separate manual step if wanted).

---

## Risks & decision gates (summary)

| Gate | Phase | Decision |
|---|---|---|
| Recoverability probe AUC | 0a | ✅ GO (AUC 0.810, beats k-baseline 0.732) — build salience; recoverability head labels free from decoder `e(i,t)` |
| Linear dynamics beats mean | 0b step 1 | ✅ GO (R²=0.297 at 384-dim "last" rep; 1536-dim was a false NO-GO from N<D — de-wonk caught) |
| Linear surprise-AUC vs EMA JEPA | 0b step 2 | ✅ GO — linear L2-residual surprise-AUC 0.7625 beats EMA JEPA cosine 0.565; no collapse. **Ship LINEAR (not JEPA); EMA/anti-collapse machinery deferred to a future generative rollout.** |
| Relevance gate | 2a | Ship relevance head |
| Recoverability gate | 2b | Ship recoverability head |
| Latent-dynamics surprise-AUC (linear) | 2c | Ship latent-dynamics head (linear map) |
| Graduation v1 proxy ships (no gate) | 2d v1 | Ship the heuristic `∫r·dt` proxy |
| Graduation v2 beats v1 | 2d v2 | Ship the learned head; else keep v1 |
| Context-builder ≥ heuristic | 3 | Flip `--strm-context-builder` default on |
| STRM+recall vs fixed-interval RAG | 4 | Ship the primitive (the real decision) |
| Stage B beats Stage A | 6 | Joint-train; else stay frozen |

---

## What this plan is not

- Not a commitment to build all six phases. Phases 0a/0b can kill the lower half
  (they ran 2026-07-19 — both GO, see above).
- Not a spec for a new training framework. It copies `doc_kind_training.py` for
  every head (including the latent-dynamics head, which is linear in v1 —
  `pretrain.py` is no longer a template now that JEPA is deferred); the only
  net-new arch is the context-builder Transformer.
- Not a redesign of Ponder's LTM machinery. Consolidation, anomaly, forgetting,
  retrieval are reused as-is; STRM only adds the STM-side read-out and the wiring.
- Not a reason to touch the shipped orchestrator's default path. Everything ships
  behind a flag; the heuristic `PresentationGate` stays as fallback.

---

## First move (done 2026-07-19)

Phase 0a (recoverability probe) and Phase 0b (latent-dynamics gate) ran together
— they share the same logged `state_t` traces, so the trace-logging script was
built once and both probes ran on it. Both are probe-only (no shipped code, no
committed probe scripts), both needed no retraining, and both produced the
go/no-go numbers: **0a GO (AUC 0.810), 0b GO (ship LINEAR, surprise-AUC 0.7625).**
Phase 1 (ring buffer + read-out) shipped (be77e35). **Next move: Phase 2** — the
four heads. Cheapest first: the recoverability head (labels free from 0a) and
the latent-dynamics head (now linear — a closed-form ridge fit, no training
loop), then relevance (needs the `record_feedback` raw-rating tap), then
graduation v1 proxy.