## 1. Mechanistic diagnosis — why the mixed‑ring retrain collapsed

The collapse from **2.614 → 0.014** is not a small degradation; it is a complete loss of selectivity. The evidence points to a single dominant cause, with two aggravating factors.

### Root cause (very high confidence): contradictory gold labels from near‑tied cosine similarities

In the mixed ring, a prior user message `u_j` appears **twice**:
- as a **type‑0 conversation slot** (text = `u_j`)
- as a **type‑1 retrieved episode** (text = `u_j + a_j`)

Both slots have **near‑identical** bge cosine to a query that is similar to `u_j`. The hard InfoNCE gold picks the **single** argmax cosine — one slot becomes the positive, the other a hard negative. Because the cosine difference is tiny (often <0.01), the “winner” is determined by noise (e.g., whether the episode’s extra assistant text `a_j` slightly raises or lowers the cosine). The model is therefore trained to **suppress a slot that is semantically almost identical to the positive**. This is a contradictory training signal: the same content must be simultaneously pulled up and pushed down.

**Consequence:** The model cannot learn a consistent content‑based ranking. Instead it latches onto any spurious feature that breaks the tie on the training set — and the slot‑type embedding is the perfect candidate (see below). On held‑out sessions the spurious correlation does not hold, so the model outputs near‑uniform logits (gap ≈ 0).

### Aggravating factor #1: slot‑type embedding provides a spurious tie‑breaker

The model sees that in training, the positive slot is *sometimes* type‑0, *sometimes* type‑1, depending on the cosine noise. It can partially fit the training labels by learning “if type‑0 then higher score” for certain sessions, but that mapping is arbitrary and session‑specific. With enough capacity, the transformer memorises session‑id → type‑preference patterns. On held‑out sessions those patterns are absent, so the type embedding becomes noise, destroying the cross‑slot attention’s ability to compare content.

### Aggravating factor #2: checkpoint selection + regularisation stack

- **Select‑ckpt = final (epoch 199)** instead of best‑valq‑gate. The model likely overfits the training noise, and the final checkpoint locks in that overfitting. The earlier successful run used best‑ckpt selection, which would have picked a point before the model fully memorised the spurious type‑based tie‑breaking.
- **Label smoothing (0.05) and dropout (0.1)** are not strong enough to prevent this overfitting because the contradictory signal is present in *every* batch. They may even slow down convergence on the true content signal, leaving the model in a state where it relies more on the easy spurious feature.

### What about the increased ring size / OOD state trajectories?

The mixed ring is larger and contains episodes whose SSM state may differ from conversation messages. However, the model’s task is to predict the argmax of cosine similarity *on those same states*. If the state representations were systematically different, the cosine similarities would also differ, but the gold label still reflects the true cosine. The model could in principle learn to map those states to the correct cosine order. The fact that train top‑3 accuracy plateaued at 0.955 shows the model *can* fit the training set — it just fails to generalise. That is classic overfitting to spurious features, not an inherent difficulty of the ring.

**Ranked causes:**
1. **Contradictory gold labels** (near‑duplicate slots forced into positive/negative roles) — **dominant**.
2. **Slot‑type embedding as a spurious feature** — enables overfitting to the contradictory labels.
3. **Checkpoint selection (final) + mild regularisation** — locks in the overfitted solution.

---

## 2. Is the mixed‑ring trace well‑posed?

**No, as currently constructed it is ill‑posed for training a relevance scorer.**

The production ring mixes conversation messages with **retrieved external documents** — two genuinely distinct content distributions. The Phase‑1c trace instead mixes conversation messages with **retrieved prior conversation turns** (episodes). Because the episodes are built from the same user messages, the two slot types have **heavy content overlap**. This creates a degenerate training distribution where:

- The “relevance” distinction between a message and its corresponding episode is **not meaningful** — both are highly relevant.
- The hard single‑positive InfoNCE objective forces an arbitrary choice, turning the task into a **tie‑breaking game** rather than a relevance ranking.

A well‑posed proxy would ensure that the two slot types carry **distinct, non‑overlapping content**, so that the gold label reflects a genuine relevance difference. The current trace does not satisfy this, and that is the direct reason the head collapsed.

**However**, the trace *can* be made well‑posed by fixing the gold assignment (see plan below). Once fixed, it becomes a useful intermediate gate before introducing real documents.

---

## 3. Ranked plan to reach a robust held‑out pass

I’ll rank by expected leverage, cheapest decisive experiment first.

### 🥇 1. Fix the gold assignment — eliminate contradictory negatives (highest leverage, cheapest)

**What to change:**  
Replace the hard single‑positive InfoNCE with a **multi‑positive** formulation. Any slot whose cosine similarity to the query is within a small margin of the maximum (e.g., `max_cos - 0.01`) is treated as a positive. All other slots remain negatives. This directly removes the contradictory signal: near‑duplicate slots are both pulled up, not forced into opposition.

**Implementation:**  
- Compute `cos_sims` for all K slots.  
- `max_cos = cos_sims.max()`  
- `pos_mask = (cos_sims >= max_cos - 0.01)`  (tune the margin; 0.01 is safe given the observed near‑ties)  
- Use a multi‑positive InfoNCE loss (e.g., `-log( sum_{i in pos} exp(score_i) / sum_{all} exp(score) )`).  
- Keep label smoothing if desired, but it becomes less critical.

**Why it works mechanistically:**  
The model is no longer punished for assigning high scores to both the message and its episode. It learns a genuine content‑based ranking, and the slot‑type embedding can be used as a helpful (but not necessary) feature without becoming a spurious tie‑breaker.

**Verification:**  
Run a single seed (seed‑0) with this fix, all else equal (mixed ring, slot‑type embedding, same regularisation, final‑ckpt). If the held‑out z_logit gap jumps back to >2.0, the diagnosis is confirmed and we have a path forward.

---

### 🥈 2. Revert to best‑valq‑gate checkpoint selection (high leverage, zero cost)

**What to change:**  
Instead of using the final epoch‑199 checkpoint, evaluate the model on a small validation set (e.g., 10% of training sessions) after each epoch and select the checkpoint with the highest **median z_logit gap** on that validation set. This was the selection method in the successful Task #45 run.

**Why it works:**  
Even with the fixed gold labels, the model may still overfit slightly to training noise. Best‑ckpt selection picks the point of maximum generalisation, avoiding the late‑epoch overfitting that final‑ckpt captures.

**Verification:**  
Combine with fix #1; the held‑out gap should be higher and more stable across seeds.

---

### 🥉 3. Simplify the regularisation stack (moderate leverage, easy to test)

**What to change:**  
Remove dropout on the per‑slot readout `z_i` and remove label smoothing. Keep only the cosine LR schedule (which is benign). The original successful Head B used **no dropout, no label smoothing**. The current stack was added as a precaution but is not needed once the gold labels are clean.

**Why it works:**  
Dropout on the slot representations can hurt the transformer’s ability to form precise cross‑slot comparisons, especially with small K. Label smoothing, while generally helpful, can dilute the already small positive signal when there are many fillers. Removing both reduces regularisation‑induced noise.

**Verification:**  
Ablate on the fixed‑gold run; if the gap improves further, adopt the simpler stack.

---

### 4. Keep the slot‑type embedding (neutral‑to‑helpful once gold is fixed)

**Do not remove it yet.** In production, the two slot types (conversation vs. retrieved doc) are genuinely distinct, and the type embedding will help the model learn type‑specific relevance patterns (e.g., retrieved docs may need a higher bar). With clean gold labels, the embedding will not become a spurious feature. If ablation shows it hurts even with fixed gold, we can drop it, but that is unlikely.

---

### 5. Skip the conversation‑pair mixed ring and go straight to production doc corpus (v2) — only if #1 fails

**Rationale:**  
If, after fixing the gold assignment, the head still fails to generalise on the conversation‑pair mixed ring, then the proxy itself may be fundamentally too noisy (e.g., the episode state representations are too similar to message states). In that case, the fastest path is to build a training set with **real ingested documents** (distinct from conversation messages) and train directly on the production distribution. This is a larger engineering effort but guarantees a well‑posed problem.

**Verification:**  
Only pursue if #1–#3 do not restore a >2.0 median gap.

---

### 6. Architectural changes (lowest priority)

The cross‑slot Transformer is already the right architecture. No changes are warranted until the data/loss issues are resolved. If we later need more capacity, we can increase encoder layers or FFN size, but that is not the bottleneck now.

---

## Summary of the decisive experiment

**Cheapest, highest‑signal test:**  
Re‑run Phase‑1 seed‑0 with **multi‑positive gold (margin 0.01)** and **best‑valq‑gate checkpoint selection**, keeping everything else identical (mixed ring, slot‑type embedding, dropout 0.1, label smoothing 0.05, cosine LR). If the held‑out z_logit gap recovers to ≥2.0, we have confirmed the root cause and can proceed to full 3‑seed training with the cleaned pipeline. I expect this single change to reverse the collapse entirely.