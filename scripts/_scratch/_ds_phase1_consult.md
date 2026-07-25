# Consultation: Phase 1 retrain of the STRM cross-slot Transformer (Head B) COLLAPSED held-out — why, and how to fix

You are advising the architect of the Ponder Engine ("STRM" = Short-Term Read-Only Memory). You previously (task #45/#46) diagnosed why a pointwise bilinear relevance head saturates on serve data, recommended the cross-slot Transformer (Head B), which won a head-to-head A/B (2.614 median, 2/3 PASS on the held-out prior-message ring), then FAILED the live serve gate (1/3). We then ran a held-out diagnostic (D0.4a/D0.4b): the live-gate failure is H2 (ring-composition OOD) confirmed — Head B trained on conversation-only rings is OOD at serve where the ring mixes conversation + retrieved-doc episodes. Your prescribed fix was: retrain Head B on the FULL production ring (conversation + retrieved episodes) with a slot-type embedding + regularization. We implemented exactly that (Phase 1). The held-out result COLLAPSED. We need your mechanistic diagnosis + a ranked plan.

## The system (facts, unchanged where noted)

**Backbone.** Frozen ReferenceSSM, 19.5M params, d_model=384, n_layers=4, d_state=16, fp32. Ingestion is ONE step per token/doc from a ZEROED state (identity-instance direct-SSM). `slot.h` = 4 per-layer tensors each `[1,16,384]`; the relevance repr is `flat_last = h[-1].reshape(-1)` = `[6144]`. Frozen at serve.

**Head B (cross-slot Transformer).** `StateReadout` MLP-128 `[6144->384]` -> per-slot `z_i [K,384]`; + learned positional emb `[64,384]`; + learned `[CLS]`; + `query_proj Linear(384,384)` (query added to [CLS], prepended to K slot tokens); `nn.TransformerEncoder(2 layers, 4 heads, FFN=512, norm_first, GELU)`; per-slot `Linear(384,1)`. **Phase 1b added:** `nn.Embedding(n_slot_types=2, 384)` summed into `z` (slot_type 0=conversation, 1=retrieved), + learnable temperature `score = logit_head(slot_out)/softplus(logit_temp)` (init ~1.0). ~2.98M params. Single-record interface (one query, K slots).

**Head A (baseline).** Same `StateReadout` -> pointwise bilinear `ZRelevanceHead`. No slot-type embedding. The pointwise (absolute-sim) scorer that cannot make the 2.0 RELATIVE margin on topically-close fillers (your task #45 diagnosis).

**Eval gate (unchanged).** Per-source z_logit selectivity gap: group (turn,slot) by source_id; sources with >=3 occurrences; probe = max-cos occurrence, fillers = rest; `gap = probe_logit - mean(filler_logits)`; median over eligible sources. **Gate = median >= 2.0 (z_logit, pre-sigmoid).** z_r (sigmoid gap) saturates sub-0.2 for both heads -> z_logit is decisive.

## The Phase 1 retrain (your prescribed fix, implemented)

**Mixed-ring training traces (NEW, Phase 1c).** 939 records from 51 Onyx sessions (76 raw; 23 skipped <3 turns; 2 held-out live-eval sessions excluded). Per session, per user-turn i (i>=1):
- `orch.query(u_i)` runs the FULL orchestrator: (a) adds the conversation message slot via `working_memory.update` (slot_type=0, `source_id = "{session_id}#msg{turn_counter}"`, text = the user message `u_i`); (b) the HippocampalRetriever retrieves prior episodes from the persistent store + injects each via `working_memory.inject` (slot_type=1, `source_id = "{session_id}__ep{j:04d}"`, text = the episode summary).
- The store is seeded + grown with episodes built from the SAME conversation message-pairs: `build_episode("{session_id}__ep{j:04d}", u_j, a_j, ...)` — i.e. each prior user/assistant PAIR `u_j + a_j` is encoded as an episode.
- So at turn i, the ring mixes: type-0 slots = prior user messages `u_0..u_{i-1}` (text = `u_j`), AND type-1 slots = retrieved prior episodes (text = `u_j + a_j` pair-summary).
- `strm_ring_text=True` so conversation slots carry text + survive the scorer's text filter.
- **Gold = top-1-cos argmax over the FULL mixed ring.** `labels[argmax(cos_vals)] = 1.0`; all others = 0. (Contrastive InfoNCE, hard one-hot gold.)

**The critical structural fact.** A prior user message `u_j` (j < i) can appear in ring(i) as BOTH:
- a type-0 conversation slot `#msg{j}` (text = `u_j`), added when query j ran, AND
- a type-1 retrieved episode `__ep{j}` (text = `u_j + a_j`), if the retriever surfaced it.
The two slots have NEAR-IDENTICAL bge cosine to a query similar to `u_j`. The argmax picks ONE (marginally higher cos); the OTHER near-tied slot is treated as a FILLER (negative). So the contrastive loss is trained to SUPPRESS a near-identical-content slot — a contradictory/noisy target. The tie-breaking depends on whether the episode text `u_j+a_j` or the message text `u_j` has higher cos to the query — a training-specific accident.

**Training (Phase 1d regularization stack).** Contrastive InfoNCE, T=1.0, 200 epochs, AdamW lr 5e-4, wd 0.01, grad-accum 4. **dropout 0.1** on the per-slot readout `z_i` (before the encoder). **label smoothing 0.05** (soft-target InfoNCE: gold slots get (1-eps)/n_gold, fillers get eps/n_fill, instead of hard one-hot). **cosine lr schedule** over 200 epochs. 3 seeds (0,1,2), session-LEVEL 20% held-out per seed (entire conversations unseen). **select-ckpt = final** (epoch 199, NOT the valq-gate-selected best.pt — a prior de-wonk found best.pt has high selection variance from a small valq draw). 3-seed logit-avg ensemble at eval.

## The results (the collapse)

**Task #45 head-to-head — held-out PRIOR-MESSAGE ring (conversation-only, NO slot_types, NO regularization, select-ckpt best):**
- Head B (transformer): 2.614 / 1.243 / 2.858 -> median 2.614, **2/3 ROBUST PASS**.
- Head A (bilinear): -1.828 / 1.434 / 0.200 -> median 0.200, 0/3 FAIL.
- 1012 traces, same 76 Onyx sessions, ring = prior user/assistant messages only (ONE slot type).

**Phase 1 retrain — held-out MIXED ring (conv + retrieved episodes, slot_types=2, dropout 0.1, label_smoothing 0.05, cosine, select-ckpt final):**
- Head B (transformer) seed-0: held-out z_logit = **0.014** (n_ge_2.0 = 6/260), z_r = 0.0001, ALL-TURNS z_logit = 0.035. **FAIL.** train_top3 plateaued ~0.955 (label smoothing prevented full memorization; the model learned the TRAIN rings).
- Head A (bilinear) seed-0: held-out z_logit = -0.047, FAIL.
- Seeds 1-2 still running; ensemble pending. (I will send the full 3-seed + ensemble numbers when the run finishes, but seed-0 is already a dramatic collapse from 2.614.)

So on the SAME 76 Onyx sessions, merely by changing the ring from "prior messages only" to "prior messages + retrieved prior-conv-turn episodes (mixed, slot_types)" + the regularization stack + select-ckpt final, Head B's held-out z_logit went from 2.614 (PASS) to 0.014 (FAIL). The cross-slot advantage over Head A essentially VANISHED (0.014 vs -0.047).

## What we need from you

1. **Mechanistic diagnosis: why did the mixed-ring retrain collapse Head B's held-out generalization?** Is the gold-ambiguity (overlapping type-0 message / type-1 episode content with near-tied cosine, single-argmax gold) the root cause — the contrastive loss getting contradictory targets on near-duplicate slots? Or is it the regularization stack (dropout/label-smoothing/cosine/select-ckpt-final) interacting badly? Or the slot-type embedding polluting the cross-slot attention? Or something else (e.g. the ring now has MORE slots of mixed types, harder; or the retrieved episodes are OOD-ish because their state trajectory differs)? Rank the candidate causes by likelihood + explainability.

2. **Is the mixed-ring trace even well-posed?** The production ring at serve mixes conversation messages + retrieved DOCS (external ingested documents). Our Phase 1c mixed-ring instead mixes conversation messages + retrieved PRIOR-CONVERSATION-TURNS (no standalone docs yet — the user decided to finish the conversation-pair mechanism gate first, then add real docs in a v2). Is the conversation-pair mixed ring a sensible PROXY for the production mixed ring, or is it a degenerate/ill-posed training distribution (the two slot types are content-overlapping, unlike production where docs are genuinely distinct from messages)? If ill-posed, does that explain the collapse (the head can't learn a consistent ranking when the two types overlap)?

3. **A ranked, concrete plan to reach a robust held-out pass** on the mixed ring (and then the live gate). For each lever: what to change, why it should work mechanistically, and how to verify. Consider at least: (a) fixing the gold assignment (e.g. gold = top-1-cos WITHIN a slot type, or de-duplicate overlapping slots, or multi-gold / top-k-positive InfoNCE so near-tied slots aren't contradictory negatives); (b) the slot-type embedding (helpful or harmful here?); (c) the regularization stack (is label-smoothing + dropout + cosine + final-ckpt the right call, or did it hurt vs task #45's plain best-ckpt?); (d) whether to skip the conversation-pair mixed ring and go straight to the production doc corpus (the v2 with real ingested docs, where the two slot types are genuinely distinct); (e) any architectural change.

Be concrete and mechanistic. Rank by expected leverage. We can run experiments — tell us the cheapest decisive ones first. We are NOT wiring Head B into the live engine until it robustly clears the held-out live gate; the frozen backbone + 5 downstream heads + 2b gate stay byte-identical.