# Consultation: get the STRM cross-slot Transformer (Head B) to robustly clear the LIVE serve gate

You are advising the architect of the Ponder Engine ("STRM" = Short-Term Read-Only Memory). You previously (task #45) diagnosed why a pointwise bilinear relevance head saturates on serve data and recommended a cross-slot Transformer (Head B), which won a head-to-head A/B. We ran your recommended acceptance test (task #46); Head B cleared the gate on the training-distribution ring but does NOT robustly clear it on the LIVE serve ring. We need your mechanistic diagnosis + a ranked plan to reach a robust pass. Reason mechanistically; be concrete; rank by expected leverage.

## The system (facts)

**Backbone.** A from-scratch ReferenceSSM, 19.5M params, d_model=384, n_layers=4, d_state=16, fp32. The SSM step is `new_state = g*b + (1-g)*state`, `g=sigmoid(W_A(x))`, `b=W_B(x)`. Ingestion is ONE step per token/doc from a ZEROED state (identity-instance direct-SSM path). `slot.h` = 4 per-layer tensors each `[1, 16, 384]`; the relevance representation is `flat_last = h[-1].reshape(-1)` = `[6144]` (16 d_state × 384 d_model). Frozen at serve.

**Head B (cross-slot Transformer, the task #45 winner).** `StateReadout` MLP-128 `[6144 -> 384]` -> per-slot `z_i [K,384]`; + learned positional embedding `[max_pos=64, 384]`; + learned `[CLS]` token; + `query_proj Linear(384,384)`; the query is projected and ADDED to the [CLS] token, then PREPENDED to the K slot tokens; `nn.TransformerEncoder(2 layers, 4 heads, dim_feedforward=512, norm_first=True, GELU, batch_first=True)`; a per-slot `Linear(384,1)` logit head reads each slot's encoder output (the [CLS] output is discarded). Single-record interface (one query, K slots, no batching/padding). ~2.98M params. The mechanism you identified: each slot's logit attends to the query AND every other slot -> a RELATIVE score (sim-to-query attenuated by the candidate pool), which escapes the pointwise margin bound.

**Head A (the baseline it beat).** Same `StateReadout` MLP-128 -> `ZRelevanceHead` pointwise bilinear `score = proj_z(z_i)·proj_q(q)/sqrt(P) + bias` (bias = a single scalar broadcast to all slots). The ONLY difference from Head B is the scoring: pointwise bilinear vs cross-slot attention.

**Training (identical for A and B).** Contrastive InfoNCE, T=1.0, 120 epochs, AdamW lr 1e-3, wd 0.01, grad-accum 4. Session-LEVEL 20% held-out per seed (entire conversations unseen). 1012 serve traces from 76 real Onyx chat sessions (the user's private chats). Each trace = at a user turn, the RING = that session's PRIOR user/assistant messages (non-system, non-empty), ingested one-step-per-message from zeroed state. Labels: gold = the prior msg most bge-similar to the query (top-1-cos); continuous bge cosine per slot. 3 seeds (different session-holdout splits).

**Eval metric (the gate).** Per-source z_logit selectivity gap: group (turn, slot) by source_id; for sources with >=3 occurrences, probe = max-cos occurrence, fillers = the rest; `gap = probe_logit - mean(filler_logits)`; median over eligible sources. Gate = median >= 2.0 (z_logit, pre-sigmoid). z_r (sigmoid gap) saturates sub-0.2 for both heads (sigmoid compression of the logit margin on topically-close fillers) -> z_logit (2.0) is the decisive gate.

## The results

**Task #45 head-to-head — held-out PRIOR-MESSAGE ring (the training distribution):**
- Head B (transformer): 2.614 / 1.243 / 2.858 -> median 2.614, **2/3 ROBUST PASS**.
- Head A (bilinear): -1.828 / 1.434 / 0.200 -> median 0.200, 0/3 FAIL.
- (Head B is the FIRST state-trajectory arch to clear the gate; tasks #38-#44 all failed: mean-pool, flat BCE, flat BCE + 55x lmsys, flat contrastive, all 0/3.)

**Task #46 acceptance test — LIVE serve ring (the full orchestrator replaying real transcripts):**
- Head B (transformer): +2.087 / -2.508 / +1.814 -> **1/3, NOT ROBUST**.
- Head A (bilinear): -0.225 / -0.433 / +1.180 -> 0/3 FAIL.
- 89 eligible sources, 115 turn records, both transcripts. Head B beats Head A (higher medians; s0 clears decisively; s2 mean +2.4 with 44/89 slots clearing) but does NOT robustly clear the 2.0 bar. High seed variance: range -2.5 to +2.1.

## VERIFIED facts about the live ring (this changes the diagnosis)

1. **The 3 live transcripts are IN the 76 training sessions.** The live probe replays `docs/*.json` (chat_session_ids 682afdd9, 69e17901, ed3b3157); all three are among the 76 sessions Head B trained on. So the live gate is NOT a clean held-out test — s0's pass is partly in-sample memorization. BUT s1 and s2 ALSO fail on these same in-sample sessions, so the ring-composition shift dominates even on SEEN data.

2. **The live ring is a MIX of two slot types; Head B trained on only one.** The full orchestrator, at each turn, runs the HippocampalRetriever (graph traversal over the user's INGESTED DOCUMENT CORPUS) and INJECTS each retrieved episode into WorkingMemory via `working_memory.inject(emb, source_id, text, pin)` -> `step` -> captures `.h`. So the live ring = (a) in-distribution CONVERSATION-message slots [the session's prior msgs, which ARE in the 76] + (b) OOD RETRIEVED-DOC-EPISODE slots [external ingested docs: different bge embeddings, different SSM state trajectories, possibly pin-tagged]. Head B trained on rings of ONLY (a). At serve the candidate pool Head B's cross-slot attention attends over is a mix of (a)+(b). The retrieved-doc slots are content the head never saw.

3. **Seed variance / instability.** s0 +2.087 (pass), s1 -2.508 (collapse), s2 +1.814 (near-miss: 44/89 slots clear, mean +2.4, but median below 2.0). The s1 collapse smells like overfitting/optimization instability; the signal is real (s0 + s2 mean) but the training is not stable.

## Three competing hypotheses for the 1/3 live

- **H1 (in-sample confound):** the live transcripts are in training; s0's pass is memorization; the true held-out number is ~0/3. Fix = hold the 3 transcripts out of training, OR eval on genuinely held-out sessions.
- **H2 (ring-composition OOD):** the live ring mixes conversation + retrieved-doc slots; Head B trained on conversation-only rings; the OOD retrieved-doc slots pollute the cross-slot attention's candidate pool (the relative scoring now attenuates against foreign slots it never learned to rank). Fix = train on live-ring-composition traces (include retrieved-doc slots in training rings), OR add a slot-type embedding so the head can distinguish the two slot types.
- **H3 (seed variance / overfit):** s1's -2.508 collapse is optimization instability; the signal is there but training is unstable. Fix = regularization, more data, longer training, lower lr, dropout, ensemble, etc.

## What we need from you

1. **Mechanistic diagnosis.** Given Head B robustly clears the prior-message ring (2/3) but drops to 1/3 on the live ring, and the live transcripts are in-sample — which of H1/H2/H3 (or which combination) is the DOMINANT lever? Reason from the cross-slot attention mechanism: how should retrieved-doc slots (OOD content, different state statistics) affect the per-slot logits and the per-source gap? Is the in-sample confound (H1) likely inflating s0, or is the ring-composition OOD (H2) strong enough to dominate even on seen sessions (which the s1/s2 in-sample failures suggest)?

2. **A ranked, concrete plan to reach a robust >=2/3 live pass.** Lead with CHEAP DIAGNOSTIC probes that disambiguate H1/H2/H3 before committing to expensive retraining (we have a RunPod L4 pod; CUDA available; the 76 sessions + the live orchestrator are the data). For each step: what to do, why, expected effect, cost. Distinguish "diagnose" from "fix" steps.

3. **The ring-composition fix specifically.** Should we train Head B on live-ring-composition traces (rings that include retrieved-doc slots)? If yes: the live orchestrator's retrieval is expensive to replay at scale — propose how to generate these traces cheaply (e.g., reuse the orchestrator's captured retrieved episodes; or approximate the retrieved-doc slot distribution; or sample). Is there a CHEAPER architectural fix — e.g., a slot-type embedding (conversation vs retrieved-doc) so Head B can condition on slot type, analogous to the pin-tag the orchestrator already uses for salience-fired episodes? Would that alone close the gap, or is retraining on the mixed ring necessary?

4. **The in-sample confound + clean experimental design.** The acceptance test currently replays 3 in-training transcripts. Propose the cleanest held-out live eval: (a) hold the 3 transcripts out of training and retrain, then live-eval; (b) build a per-seed held-out live eval that replays each seed's held-out 20% of sessions through the orchestrator; (c) something else. What is the minimal change to make the live gate a genuine held-out test?

5. **Arch changes.** Is the 2-layer/4-head/FFN-512 cross-slot Transformer adequate, or does it need changes to be robust? Consider: depth/heads, a temperature on the logit head (the s1 collapse + the z_r sigmoid compression suggest the logits may be poorly scaled), a slot-type embedding, dropout, layer-norm placement, the [CLS]-vs-per-slot-head design. The s1 collapse specifically — what would you add?

6. **Abandon criteria.** What result on the diagnostic probes would make you conclude the state-trajectory-locator path is architecturally blocked (abandon), vs. just needs the fixes above?

Be specific and mechanistic. Prefer the cheapest fix that addresses the dominant lever. We have NOT wired anything into the live engine (binding constraint: don't break existing functionality); Head B is local-only until it robustly clears the live gate.