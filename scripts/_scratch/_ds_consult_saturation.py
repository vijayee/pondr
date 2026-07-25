"""One-off DeepSeek consult on the 2a salience saturation blocker. UNTRACKED
scratch (per commit-at-will). Not committed; run-only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.oracle_labeling import OracleClient, OracleConfig

PROMPT = r"""
You are advising a senior ML engineer on a stuck retrieval-salience mechanism. I have a precise diagnosis; I need a ranked plan + levers I may be missing + a go/no-go on the cheapest structural fix. Think hard; be concrete; no hand-waving.

# System (Pondr STRM short-term read-only memory)
A conversational memory engine. Working Memory = a fixed-capacity ring of recent turns (each slot has a 256-d SSM step readout y_t + the raw 384-d bge-small doc embedding + optional provenance source_id + text). A LONG-TERM store holds older episodes. At each turn the engine can PROACTIVELY recall an old episode into context before the user asks. The salience trigger decides which ring anchor to proactively recall.

The ship-deciding gate: on a long-horizon eval, STRM (relevance-gated proactive recall) answers more factual questions correctly than FIXED-INTERVAL refresh at equal recall budget/latency, in >=2/3 seeds. Tier1 = offline coverage (does salience surface the RIGHT fact more than fixed round-robin at equal cost). Tier2 = live factual accuracy (8B synth + judge panel).

# The salience trigger (3-term AND, per ring anchor)
  salient = (rec_i < theta) AND (r_i > phi) AND (surprise_i < surprise_cap)
- r_i : 2a RELEVANCE head, r_i in [0,1], query-conditioned.
- rec_i: 2b RECOVERABILITY head (negated forgetting; low = forgotten = salient).
- surprise_i: 2c LATENT-DYNAMICS head, ONE scalar per turn (transition surprise; high = suppress).
- theta/phi/surprise_cap are percentiles of each head's VAL score distribution, persisted in a sidecar.
Shipped thresholds: theta=-0.2643 (p30 of 2b val, n=620), phi=0.0709 (p70 of 2a val, n=1438), surprise_cap=5.55e-05 (p80 of 2c val, n=237). Note surprise_cap is TINY.

# The 2a relevance head architecture
  score = (proj(doc_emb) . proj(query_emb)) / sqrt(P) + yt_sidepath(y_t) + bias
  r_i   = sigmoid(score)
- proj: 384->128 SHARED projection (doc and query both bge-small in the same 384-d space; learned cosine; proj~I recovers bge cosine).
- yt_sidepath: Linear(256,64)->GELU->Linear(64,1) reading the SSM step readout y_t. Docstring ASSUMES this "learns near-zero on the frozen backbone" (y_t carries no relevance signal there) so the shared-proj term carries the signal.
- Trained supervised BCE on ERAG-Bench traces (gold doc slot = positive, sampled non-gold = negative). Gate = per-query top-3 recall + Wilson CI LB. NOTE: raw bge cosine ALONE clears this training gate at 1.00.

# The stuck result (the reason I am asking)
Phase 4 ship eval, 3 seeds, REAL shipped thresholds: STRM cost=0 / hit=0 on EVERY seed. Tier1 coverage 0/3, Tier2 accuracy 0/3 (skipped because coverage failed). HOLD.
Selectivity diagnostic (seed 0, 6 facts, 40-turn horizon, ring-cap 8):
  probe_r_i  ~0.9999 (saturated high)
  mean_filler_r_i ~0.66-0.91 (fillers ALSO high)
  discriminates=false, min probe-minus-filler gap 0.09
So r_i does not discriminate probe from filler -> the r_i>phi term is always satisfied (phi=0.07) AND cannot target the right turn. hit=0 means one of (rec_i<theta) or (surprise_i<surprise_cap) is over-suppressing every turn (cost=0 -> no retrieval fired at all).

The permissive upper-bound smoke (theta=+inf, phi=-inf, surprise_cap=+inf) gives STRM 1.0 > FIXED 0.0 > OFF 0.0 -- so the WIRING is sound; the MECHANISM (the heads) is the blocker. CAVEAT: in the synthetic eval the seeded fact is the ONLY provenance-bearing ring slot, so fire-on-any-scored-anchor trivially surfaces the right fact; this does NOT test discrimination. Real serve has many provenance-bearing recalled episodes.

# Prior diagnosis (convergent across 3 probes -- well-established, not speculation)
1. PROBE 4a ablate-yt (zero the yt_sidepath final layer) on REAL onyx serve transcripts (115 turns, 89 sources >=3 turns):
   - shipped r_i: median ~0.999, 87% >=0.99, selectivity gap median +0.001, 0/89 >= 0.2. FAIL.
   - ablated (pure bilinear) LOGIT: gap median +0.69, mean +0.77, 18/89 >= 2.0. The bilinear term DOES carry real-but-weak signal on real serve; the sigmoid r_i is saturated-useless.
   - the shipped yt_sidepath learned a LARGE ~-8.5 offset (NOT near-zero as the docstring assumes) to center the sigmoid in training; at serve the WM-state y_t is OOD -> the sidepath drowns the bilinear term and pins r_i high/erratic.
2. The shipped ContextBuilder transformer (s_i) is WORSE than raw bilinear logit because it consumes the saturated sigmoid r_i as its additive bias.
3. Held-out z_logit on 114 serve turns is NOISY/overfit: 2.5M params on ~91 train turns = 30x overparameterized; 3-seed held-out z_logit median 0.04-1.64 (below 2.0 gate); in-sample ceiling 2.27-3.67 passes (overfit).
4. rec_i is run-to-run unstable (same fact same turn varies depending on WM state trajectory, which diverges when prior salience injects pinned episodes).
5. Train/serve OOD has been the long-running blocker across ~15 prior experiment phases; every head retrain on training data fails to transfer to serve.

# Fixes already enumerated but NOT executed
A. Retrain a PURE-BILINEAR 2a head (drop yt_sidepath) so it MUST learn LOW bilinear for irrelevant (doc,query) pairs + use the bias to center the sigmoid. Gate on the raw bilinear LOGIT (or de-saturate via margin/temperature loss), NOT the sigmoid r_i. Ideally on SERVE-distribution traces (replay transcripts -> slot text = assistant responses, queries = user turns, HARD-NEGATIVE queries = other turns user text).
B. Get MORE onyx serve transcripts (have 76 sessions locally, 114 turns used so far) to fix small-data overfit.
C. Heavy regularization: much smaller readout, weight decay, dropout (2.5M params on 91 records is absurd).

# Cheap structural fix I want your go/no-go on
Since raw bge cosine clears the training relevance gate at 1.00 and the learned head is just learned-cosine that went OOD via the yt_sidepath: REPLACE the r_i term in the salience AND with a FIXED bge-cosine(query_emb, slot_doc_emb) (no learned params, never OOD). Optionally add a margin so only cos > some serve-calibrated percentile passes. This kills the 2a head entirely for salience; the learned head stays available for the context-builder. The salience AND becomes (rec_i < theta) AND (cos_i > cos_phi) AND (surprise_i < surprise_cap). Does this lose real signal vs the bilinear logit (gap +0.69, 18/89>=2.0)? Is the learned proj recovering anything bge cosine lacks at serve?

# Questions
1. Rank A / B / C / cheap-cosine-fix / any-other by expected value toward clearing the gate. What order, what stops-gate per step?
2. Is the cheap bge-cosine fix sound, or does the learned proj carry serve-relevant signal that fixed cosine lacks? Justify.
3. The cost=0/hit=0 says one of rec_i<theta or surprise_i<surprise_cap NEVER holds. Given surprise_cap=5.55e-05 is suspiciously tiny, is the surprise term likely the sole blocker, making the r_i saturation a secondary issue? How would I cheaply tell which term is killing it before retraining anything?
4. Is there a fundamentally cheaper redesign of the salience trigger (e.g. drop the learned heads entirely; use cos + age + a surprise guard from the turn embedding directly) that would clear the gate without another retrain round? The permissive smoke says fire-on-forgotten-provenance-bearing-slots works; the only missing piece is TARGETING which provenance slot. cos supplies targeting for free.
5. Where is the highest-risk assumption in my plan, and what is the fastest way to falsify it cheaply?

Be decisive. I want a plan I can execute, not a literature review.
"""

cfg = OracleConfig(model="deepseek-v4-flash:cloud", endpoint="http://localhost:11434/v1",
                   temperature=0.0, max_tokens=4096, think=False)
client = OracleClient(cfg)
res = client.generate(PROMPT, response_format="text")
out = getattr(res, "response", None) or getattr(res, "text", None) or str(res)
Path("/tmp/ds_consult_out.txt").write_text(out, encoding="utf-8")
print(out)