## Mechanistic diagnosis

**Dominant lever: H2 (ring-composition OOD) with H3 (seed variance) as a downstream symptom. H1 (in‑sample confound) inflates s0’s pass but is not the root cause.**

### Why H2 dominates
Head B’s cross‑slot attention computes a **relative** score: each slot’s logit is a function of the query *and every other slot in the ring*.  
During training the ring contained only **conversation‑message slots** (in‑distribution state trajectories).  
At serve the live orchestrator injects **retrieved‑document‑episode slots** – content the head never saw, with different BGE embeddings, different SSM state statistics, and possibly pin‑tagged.  

These OOD slots pollute the candidate pool in two ways:

1. **Attention distortion** – The OOD slots can attract high attention from the `[CLS]` token or from the correct conversation slots, pulling the relative baseline away from the true relevant message.  
2. **Logit inflation** – The per‑slot logit head may assign spuriously high scores to OOD slots, shrinking the gap between the probe and the filler mean.

Because the live transcripts are *in‑sample* (the 3 sessions are among the 76 training sessions), the fact that s1 and s2 **fail on these same sessions** proves that the ring‑composition shift alone is strong enough to break the model. The model never learned to handle foreign slots, so even on memorised conversations the cross‑slot attention is disrupted.

### Why H1 is secondary
s0’s pass (+2.087) is likely inflated by memorisation of the exact conversation rings, but the collapse of s1 (−2.508) and the near‑miss of s2 (+1.814) on the *same* in‑sample sessions show that the OOD shift is the primary failure mode. If H1 were the whole story, all three seeds would pass on in‑sample data; they don’t.

### Why H3 is a consequence
The high seed variance (range −2.5 to +2.1) is a classic sign of **brittle generalisation under distribution shift**. The model’s decision boundaries are not smooth in the region of OOD slots; small differences in initialisation or optimisation trajectory lead to large performance swings. Fixing the OOD root cause will naturally reduce variance.

---

## Ranked plan to reach a robust ≥2/3 live pass

### Phase 0 – Cheap diagnostic probes (disambiguate H1/H2/H3, cost: <1 hour on L4)

| Step | What to do | Why | Expected effect | Cost |
|------|------------|-----|-----------------|------|
| **D0.1** | **Ablation: remove all retrieved‑doc slots** from the live ring and re‑evaluate Head B on the 3 live sessions. | If performance recovers to training‑distribution levels (≥2.0 median), H2 is confirmed. | Gap should jump back to ~2.6 for all seeds. | Trivial – just filter slots by source type before feeding to Head B. |
| **D0.2** | **Attention‑pattern analysis** on the live ring. For each head, compute the mean attention from `[CLS]` to conversation slots vs. retrieved‑doc slots, and from conversation slots to retrieved‑doc slots. | Quantify how much OOD slots steal attention. | Expect retrieved‑doc slots to receive disproportionately high attention, especially in failing seeds. | A few forward passes + hook. |
| **D0.3** | **Per‑slot logit distribution** by slot type. Plot histogram of logits for conversation slots vs. retrieved‑doc slots. | Check if retrieved‑doc slots are getting high logits, compressing the gap. | Likely a long tail of high logits on OOD slots. | Same forward passes. |
| **D0.4** | **Evaluate Head B on the prior‑message ring for the 3 live sessions** (i.e., the training‑distribution ring, but restricted to those 3 sessions). | If s1/s2 still underperform here, there is a separate instability/overfitting issue beyond OOD. | If they pass, H3 is purely OOD‑driven; if they fail, we need regularisation. | Trivial. |

**Decision gate after D0:**  
- If D0.1 recovers performance → H2 confirmed, proceed to Phase 1.  
- If D0.4 shows s1/s2 still fail on clean conversation rings → H3 is independent; add regularisation (Phase 1b).  
- If D0.1 does NOT recover performance → something else is broken (abandon criteria triggered).

---

### Phase 1 – Cheapest high‑leverage fix: slot‑type embedding + mixed‑ring training

**Rationale:** The model must learn to treat conversation slots and retrieved‑doc slots differently. The cheapest architectural change is a **learned slot‑type embedding** (2 types: `conversation` and `retrieved_doc`) added to each slot’s `z_i` before the Transformer. This gives the attention mechanism a clear signal to condition on. However, the model still needs to see mixed rings during training to learn how to use that signal. So we must generate training rings that include OOD slots.

#### 1a. Cheap mixed‑ring generation (no live orchestrator replay at scale)

We can approximate the retrieved‑doc slot distribution without running the full orchestrator:

- **Option A (synthetic OOD slots):** For each training ring, sample *k* random document embeddings from a held‑out pool of ingested documents (or even random noise vectors with similar statistics) and inject them as “retrieved‑doc” slots with a dummy source_id. The model will learn to ignore slots that are not conversation messages.  
- **Option B (replay captured episodes):** The live orchestrator already captured the retrieved episodes for the 3 live sessions. We can reuse those exact episodes as OOD slots for all training rings (but careful: this may leak live‑session content into training if we later evaluate on those sessions – we’ll hold them out).  
- **Option C (offline retrieval approximation):** Run a cheap nearest‑neighbour lookup over the document corpus using the query embedding to fetch top‑k docs, and inject those. This is closer to the real distribution but more expensive.

**Recommendation:** Start with **Option A** (synthetic OOD slots) because it’s instantaneous and tests the hypothesis that the model just needs to learn to *ignore* foreign slots. If that fails, escalate to Option C.

#### 1b. Training recipe adjustments (address H3 seed variance)

- Add **dropout=0.1** to the Transformer encoder and the `StateReadout` MLP.  
- Add a **learnable temperature** parameter to the per‑slot logit head: `logit = Linear(384,1)(x) / temp`, with `temp` initialised at 1.0 and constrained >0.1. This prevents logit explosion on OOD slots.  
- Use **label smoothing (0.05)** in the InfoNCE loss to reduce overconfidence.  
- Train for more epochs (e.g., 200) with a cosine schedule; the OOD slots act as a natural regulariser.

#### 1c. Clean held‑out evaluation (address H1)

- **Hold out the 3 live sessions entirely** from training. Use the remaining 73 sessions to generate mixed rings.  
- For each seed, create a **per‑seed held‑out live eval** by replaying the held‑out 20% of sessions through the orchestrator (or a cheap approximation) to obtain live rings. This makes the live gate a genuine held‑out test.  
- If full orchestrator replay is too expensive, use the same synthetic OOD slot injection for evaluation rings, but ensure the held‑out sessions’ conversation slots are truly unseen.

**Expected outcome:** With slot‑type embedding + mixed‑ring training + regularisation, Head B should robustly clear the live gate (≥2/3 seeds) because the model will learn to down‑weight OOD slots and preserve the relative scoring on conversation slots.

---

### Phase 2 – If Phase 1 fails, architectural upgrades

| Change | Rationale | Cost |
|--------|-----------|------|
| **Increase Transformer depth to 4 layers, heads to 8** | More capacity to learn complex slot‑type interactions. | Moderate retraining. |
| **Slot‑type‑specific projection** before the Transformer: `z_i' = MLP_type(z_i)` for conversation vs. retrieved‑doc. | Allows the model to map OOD slots into a subspace where they can be safely ignored. | Small parameter increase. |
| **Explicit slot‑type gating** in the logit head: `logit = gate(slot_type) * Linear(z)`, where `gate` is a learned scalar per type. | Hard‑wires the ability to zero out OOD slots. | Minimal. |
| **Ensemble of 3 seeds** at inference (average logits). | Reduces variance; if single‑seed instability remains, ensemble can push median over 2.0. | 3× inference cost, acceptable. |

---

## The ring‑composition fix in detail

**Yes, we must train Head B on rings that include retrieved‑doc slots.** The cross‑slot attention mechanism is inherently relative; it cannot learn to ignore a slot type it has never seen. A slot‑type embedding alone will not suffice without training on mixed rings, because the model has no experience of how to use that embedding to modulate attention.

**Cheapest path to generate these traces:**

1. **Synthetic OOD injection (cost: zero).** For each training ring of K conversation slots, add M synthetic “retrieved‑doc” slots. The synthetic slots can be:
   - Random vectors drawn from a Gaussian with mean/variance matching the `StateReadout` output distribution of real conversation slots.
   - Or actual `StateReadout` outputs from a held‑out set of ingested documents (pre‑computed once).  
   Assign them a distinct `slot_type` embedding and a dummy `source_id`. Train with these mixed rings. This teaches the model that there exists a class of slots that are *never* the target and should be ignored.

2. **If synthetic slots are insufficient**, replay the live orchestrator’s retrieval step offline for the training sessions. Since we only have 76 sessions, we can run the orchestrator once per session (with the same document corpus) and cache the retrieved episodes. This is a one‑time cost of a few hours on a CPU, perfectly feasible. Then construct training rings by concatenating the conversation slots with the retrieved episodes for that turn.

**Would a slot‑type embedding alone close the gap?**  
No, because the model has never seen a retrieved‑doc slot during training. Even with a type embedding, the attention mechanism has no learned behaviour for that type; it would treat them as an unknown, likely with erratic attention. The type embedding is necessary but not sufficient – it must be paired with mixed‑ring training.

---

## Clean experimental design for the live gate

**Minimal change to make the live gate a genuine held‑out test:**

- **Hold out the 3 live sessions (682afdd9, 69e17901, ed3b3157) from all training.**  
- Train Head B (with slot‑type embedding + mixed rings) on the remaining 73 sessions.  
- For evaluation, replay the 3 held‑out sessions through the live orchestrator exactly as in the acceptance test. This yields a clean, unseen live ring.  
- To get per‑seed statistics, repeat the hold‑out process for each of the 3 original seeds (each seed already has a different 20% session holdout; we can simply ensure the 3 live sessions fall into the holdout set for every seed).  

If we want a larger held‑out live set, we can extend this to all sessions: for each seed, take its 20% held‑out sessions, replay them through the orchestrator, and compute the median gap. That gives a robust estimate of live performance.

---

## Architectural changes for robustness

The 2‑layer/4‑head Transformer is adequate in capacity, but the following additions will directly address the observed failure modes:

1. **Slot‑type embedding** (2 types, dim=384) added to `z_i`.  
2. **Learned temperature** in the logit head: `logit = Linear(384,1)(x) / softplus(temp)`. Initialise `temp` such that `softplus(temp) ≈ 1.0`. This prevents the logit head from producing extreme values on OOD slots, which was likely causing the s1 collapse.  
3. **Dropout=0.1** in the Transformer and the `StateReadout` MLP.  
4. **Pre‑norm (norm_first=True) is already used** – keep it.  
5. **The `[CLS]` token design is sound**; the per‑slot head reading each encoder output is correct. No need to change.

These changes are cheap (a few hundred extra parameters) and directly target the OOD logit inflation and instability.

---

## Abandon criteria

**Abandon the state‑trajectory‑locator path if, after Phase 1 (slot‑type embedding + mixed‑ring training + regularisation + clean held‑out eval), the model still fails to achieve median gap ≥2.0 on ≥2/3 seeds.**  

Specifically, if the diagnostic probes in D0.1 show that removing OOD slots *does not* recover performance, or if after training on realistic mixed rings the model cannot learn to separate conversation slots from retrieved‑doc slots (i.e., the gap remains <2.0), then the cross‑slot attention mechanism is fundamentally unable to handle heterogeneous candidate pools. In that case, we would need to abandon the relative‑scoring approach and consider a different architecture (e.g., a two‑stage pipeline that first filters out OOD slots, or a hybrid that uses pointwise scoring for OOD slots and cross‑slot attention only for conversation slots). However, given the clear success on conversation‑only rings, this is unlikely; the OOD problem is solvable with the right training data and a type signal.