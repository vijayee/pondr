# STRM: Restore the Transformer as the salience relevance-locator

Source: design-drift finding from `docs/copilot-activity-history.csv` (original
architecture chat) vs the shipped STRM Phase 3+4 implementation. This doc is the
proposed rewire + known holes, written for review (DeepSeek v4 pro hole-punch)
before any code change.

## The drift (what we built vs the original design)

**Original design** (`docs/copilot-activity-history.csv` lines 2658-2819, esp.
2670-2750):

- SSM produces a state trajectory `x_1 .. x_T`.
- JEPA computes a per-state relevance prior `r_t = g(z_t)` -- the "where to look."
- A **small Transformer attends over the SSM states** with the JEPA prior as an
  attention bias: `alpha_t = softmax(QK^T + r_t)`. keys/values = projections of
  **SSM states**; query = the current input.
- The Transformer compresses the attended states into the LLM's context vector.
  LTM retrieval is a **separate, earlier** nearest-neighbor step that feeds extra
  episodes in alongside the SSM states.
- "JEPA tells the Transformer where to look in the SSM; the Transformer tells the
  LLM what it means."

**What shipped** (STRM Phase 4 + Phase 3 ContextBuilder):

- The "SSM state trajectory" became a **fixed ring of the last K=16 output
  readouts `y_t`** (eviction, not indexing). Only the latest recurrent state is
  kept; the ring is the available "SSM short-term memory" proxy.
- The "predict where/what is relevant" job was given to the **2a RelevanceHead**:
  `r_i = sigmoid(bilinear(doc_emb, query_emb) + yt_sidepath(y_t) + bias)` -- a
  bilinear head reading **one `y_t`** + a doc embedding, NOT a transformer
  attending over state internals.
- The JEPA-equivalent heads (2b recoverability = forgetting, 2c latent-dynamics =
  surprise) feed the **salience trigger** (when to retrieve), NOT the
  transformer's attention.
- The actual Transformer (ContextBuilder, Phase 3, **default-off**) was demoted
  to a **post-retrieval reranker**: it attends over ring slots + retrieved
  episodes with `lambda_r * r_i` as the bias, and picks top-m for presentation.
  It does NOT locate relevance for the salience decision.
- The salience gate uses the 2a `r_i` directly:
  `salient = rec_i < theta AND r_i > phi AND surprise < surprise_cap`.

**The saturation failure (Probes 3 / 4a):** the 2a bilinear `r_i` saturates at
serve (`r_i` ~0.999, selectivity gap ~0) because a bilinear head reading one
`y_t` + a doc embedding cannot do the "where in the SSM is the relevant context"
job the original design assigned to a transformer attending over state internals.
The `yt_sidepath` (a 2-layer MLP on one `y_t`, query-independent) was a stand-in
for "connect to state internals" that collapsed to a constant ~-8.5 offset.
**Retraining the bilinear head (the current plan) polishes the wrong component.**

## The proposed rewire

Restore the Transformer to the relevance-locating role.

1. **Move the salience gate's relevance term from the 2a bilinear `r_i` to the
   ContextBuilder's query-conditioned per-slot score `s_i`.**
   `salient = rec_i < theta AND s_i > phi AND surprise < surprise_cap` (was
   `r_i > phi`). `s_i` = `ContextBuilder.logits(slots_y, slots_doc_emb,
   query_emb, r)` -- already a per-slot pre-sigmoid score (`predict` is the
   discrete top-m consumer; `logits` is the continuous score the gate needs).

2. **Restore the JEPA-prior-as-attention-bias design.** Replace (or augment) the
   ContextBuilder's additive bias `lambda_r * r_i` with the latent-dynamics
   priors: `bias = lambda_rec * rec_i + lambda_surp * surprise_i` (the 2b/2c
   heads). This is the original `alpha_t = softmax(QK^T + r_t)` where `r_t` is the
   JEPA prior -- keys/values stay `y_t` (the SSM state internals), query stays the
   current input.

3. **Run the ContextBuilder at salience-decision time (every turn), not only
   post-retrieval.** K=16, d_head=128, 4 heads -- the original design argued the
   context transformer is "shockingly tiny" (1-5% of the LLM); the per-turn cost
   is small but must clear the cost-parity gate.

4. **Retire or demote the 2a RelevanceHead.** Options: (a) retire it entirely
   (the transformer subsumes the relevance job); (b) keep it as a cheap
   pre-filter that shortlists ring slots before the transformer scores them.
   Default to (a) for the first rewire; revisit if cost is a problem.

5. **Retrain the ContextBuilder for the relevance-locator role on
   serve-distribution traces.** The existing checkpoint was trained to select
   top-m for PRESENTATION (gold = the heuristic PresentationGate's selection).
   The relevance-locator role needs different labels: gold = "this ring slot's
   content is relevant to the current query" (the probe-vs-filler signal from
   Probe 4a, or LLM-judge labels). Slot text = real assistant responses, queries
   = real user turns, hard negatives = other turns' user text.

6. **Keep the LTM retrieval mechanism as-is** (nearest-neighbor on the salient
   anchor's `doc_emb`). The transformer locates WHICH ring slot is the salience
   anchor; the anchor's `doc_emb` is the probe for LTM vector search. This
   matches the original design where LTM retrieval is a separate nearest-neighbor
   step. The transformer does NOT predict what to retrieve from LTM -- only where
   in the ring the relevant context currently lives, which drives the retrieval
   probe.

7. **The ring-as-trajectory-proxy stays** (only the latest recurrent state is
   kept; the ring of `y_t` is the available "SSM short-term memory"). Accept the
   eviction constraint: older relevant context that fell out of the ring is
   recovered by the salience re-inject (pin) -- the transformer locates a
   relevant slot, salience retrieves its neighbors from LTM and re-injects them
   as pinned slots. This closes the loop: locate -> retrieve -> re-inject ->
   re-attend.

## What this changes in the code (wiring sketch)

- `orchestrator.py:_run_salience_hook` (~:554): replace the `r_i` computation
  (`score_ring_slots`) with a `ContextBuilder.logits` call over the current ring;
  the gate uses `s_i` instead of `r_i`. The `rec_i` (2b) and `surprise` (2c)
  computations stay (they feed both the gate's other terms AND the transformer's
  attention bias).
- `context_builder.py`: extend the bias term from `lambda_r * r` to
  `lambda_rec * rec + lambda_surp * surprise` (+ optionally `lambda_r * r`). New
  inputs `rec[T]` and `surprise[T]` alongside `r[T]`. `_coerce` + `logits`
  signatures gain two optional bias-vector args.
- The 2a RelevanceHead (`relevance_head.py`, `relevance_score.py`): becomes
  inert/retired for the salience path. Keep the module + loader for the
  ContextBuilder's optional `lambda_r * r` term if we keep option (b).
- Training: new label generator for the relevance-locator objective
  (probe-vs-filler or LLM-judge on real transcripts); retrain ContextBuilder;
  ship to HF `vijayee/pondr-models:strm_context_builder/` if the gate passes.
- Gate: re-run Probe 4a capturing `s_i` (transformer score) instead of `r_i`, on
  real Onyx transcripts. Selectivity gap >= 0.2 in >= 3/4 runs = the transformer
  discriminates where the bilinear head didn't. Then re-run the cost-parity gate
  (`eval_strm_cost_parity.py`) with the transformer as the relevance signal.

## Known holes (to review)

- **H1 Label change:** the presentation-label checkpoint can't be rewired; needs
  a relevance-label retrain. Label source is the open question (cosine
  probe-vs-filler = weak proxy; LLM-judge = costly).
- **H2 Scope:** the transformer can only locate relevance among CURRENT ring
  slots; it cannot identify a relevant LTM doc that is NOT in the ring. LTM
  retrieval stays nearest-neighbor on the anchor's `doc_emb`. Is that acceptable,
  or does the design need the transformer to also score LTM candidates directly
  (attending over retrieved candidates, not just the ring)?
- **H3 Cost:** per-turn transformer attention vs the cheap bilinear; must clear
  cost-parity.
- **H4 rec_i / surprise stability:** Probe 3 found `rec_i` unstable run-to-run.
  If those feed the transformer's attention bias, the "where to look" is noisy.
  Is the 2b head reliable enough to be the JEPA prior?
- **H5 Integration weight:** this makes the ContextBuilder run every turn at
  salience time (currently default-off, post-retrieval only). Bigger integration
  than a flag flip.
- **H6 Eviction / trajectory bound:** the ring eviction means "where in the SSM"
  is bounded to the last K slots; the original attended over the full trajectory.
  The re-inject loop is the compensation, but it's reactive (only fires when a
  still-present slot is salient). A relevant slot that was evicted with no
  salient successor is gone until something else surfaces it. Is the re-inject
  loop sufficient, or do we need a longer ring / a trajectory buffer?