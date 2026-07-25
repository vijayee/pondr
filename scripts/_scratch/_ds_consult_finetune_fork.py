import json, time, sys, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. You have advised this project before (consults #6 and "
    "#7). Be concrete, rank actions by expected leverage, give a crisp "
    "ship-vs-continue verdict, and state the single highest-leverage next step. "
    "Do not hedge with disclaimers. When you predict an outcome, state the number "
    "you expect. When your prior prediction was wrong, say so explicitly and "
    "recalibrate. Do not recommend a path you already gave a low probability and "
    "do not repeat a fix that just failed without explaining what is different."
)

user_msg = r"""STRM consult #8 -- your fine-tune recommendation ALSO FAILED; both CE fix
attempts are now HOLD. Recalibrate and rank the next forks.

RECAP. The STRM salience task = "which prior turn in the recent 16-turn ring does
this anaphoric user query refer back to." Real-onyx ship gate, 17 hand-authored
non-circular gold pairs on 2 held-out onyx sessions. SHIP iff target_top1>=2/3,
target_in_top3>=2/3, competitor_beats<=1/3, median_breadth<=3.

YOUR CONSULT #7 recommended fine-tuning cross-encoder/ms-marco-MiniLM-L-6-v2 on
~80-100 NEW gold pairs mined from the 51 TRAINED onyx sessions (non-circular vs
the 17 held-out TEST set). Binary/regression head, 3 ep, lr 2e-5. You predicted
target_top1=0.71, target_in_top3=0.88, competitor_beats<0.2 -> gate clears,
P(clear)=0.80. Your P=0.85 zero-shot prediction (#6) had already missed (got
0.294); #7 said fine-tune would fix it via in-domain distribution.

WHAT I RAN (your #7 recipe exactly):
- Mined 123 verified in-domain gold pairs from the 51 TRAINED sessions (workflow
  = local-LLM-proposes + human-verifies; 147 proposals, 24 rejected: 2 invalid
  diffuse/forced, 18 from one over-weighted session capped for balance, 4
  out-of-window/redundant, 1 diffuse). Circularity guard honored: the 17 held-
  out were NEVER touched by training.
- 445 hard negatives = top-4 by bge-cosine among in-window non-target turns
  (age>=3, cos>0.4 floor). 568 train rows.
- Fine-tuned ms-marco-MiniLM-L-6-v2 (num_labels=1, MSE), 3 ep, lr 2e-5, batch
  16, warmup 10%, seed 7, CUDA+AMP. Saved + re-evaluated on the 17 held-out with
  rank-then-budget (byte-identical to the zero-shot probe).

HELD-OUT GATE (fine-tuned CE):
- target_top1_rate      = 0.2941 (5/17)   -- you predicted 0.71
- target_in_top3_rate   = 0.5882 (10/17)  -- you predicted 0.88
- competitor_beats_rate = 0.7059 (12/17)  -- you predicted <0.2
- median_breadth        = 3.0  (PASS by construction)
- VERDICT: HOLD, ship=false.

FINE-TUNED == ZERO-SHOT, to 4 decimals. The zero-shot ms-marco baseline was
0.2941/0.5882/0.7059/3.0. The fine-tune added ZERO held-out discrimination. The
SAME 5 pairs win top1 in both. Your #7 fine-tune prediction (P=0.80 clear) is
FALSIFIED.

THE CORRECTED DIAGNOSIS (this is the important update -- it is NOT "pairwise is
impossible" and NOT a training bug; it is a GENERALIZATION gap). I ran a
confirmatory probe: scored 8 TRAINING positives vs their top-4 cosine hard
negatives on the FINE-TUNED CE. Result: pos>neg pair-win 23/28 = 0.821; 5/8
positives beat ALL their negatives; scores clearly differentiated (positives
often >0, negatives <0). So the CE DID learn a separation signal ON the trained
sessions -- it fits the training data. But that signal transfers to ZERO
improvement on the 2 held-out sessions. The learned signal is TOPIC/SESSION-
STYLE-BOUND, not a general anaphora-resolution capability. The 2 held-out
sessions ("Benchmarking Ponder Engine", "Hippocampal Indexing Architecture" --
self-referential meta-engine register) are topically distant from the 51 trained
project talks (Poseidon, OFFS, JEPA, SNARKs, ABE, etc.). Off-distribution on
held-out, the fine-tune signal is useless there so the base model's topical-
relevance ranking dominates -> identical to zero-shot.

INTERPRETATION. Pairwise (query, candidate) scoring necessarily conflates the
anaphoric-pointer signal with topic-match, because the pair text carries topic.
Fine-tuning on in-domain pairs teaches the CE topic-correlates of the referent,
which do not transfer to a held-out topic register. The distinguishing signal
(the anaphoric marker's discourse role in the broader window) is NOT in the
(query, candidate) pair -- it is in the WINDOW CONTEXT the pair sits in. This is
why both zero-shot AND fine-tuned CE cap at 0.294 top1.

THE FALLBACK IS EXHAUSTED BY DIAGNOSIS. Your #7 fallback was "if top1<0.65:
expand to 150 pairs + hard-neg mining." I am already at 123 pairs + cosine
hard-neg mining. The gap is generalization, not data quantity -- more pairs
from the same 51 trained sessions will not reach the held-out register. The 2
held-out sessions / 17 pairs is also a small, possibly pathologically-hard test
set (the meta-engine self-referential register is unique).

WHAT IS DURABLE. rank-then-budget bounds breadth to 3 by construction on ANY
ranker -> the breadth=12 axis is GONE for good. target_in_top3=0.588 is the
recall metric (the referent IS recalled in 10/17 shortlist-3 even though it is
not rank-1 in most).

THE FORKS TO RANK (give P(clear), the target_top1 and target_in_top3 numbers you
expect on the 17 held-out, effort, and the specific model/approach for each):

A. LLM-AS-SALIENCE. A small local LLM (I have deepseek running locally on
   :11434; also a 470 NPU laptop) sees the FULL recent window + the query and
   PICKS the referent slot (multiple-choice over the ~16 window turns, or free
   generate the quoted referent then match). This directly attacks the
   diagnosis: the anaphoric signal is in the window context, not the pair. Per-
   query cost (but onyx is single-user, low QPS). Predict top1/in_top3 and
   whether a small local LLM can even do anaphora over a 16-turn window. What
   prompt framing (multiple-choice vs quote-and-match) and which local model
   size is the floor?

B. SEQUENCE-TO-CHOICE / multiple-choice trainable model. Encodes (window, query)
   -> a choice distribution over the ~16 slots, NOT pairwise (query, candidate).
   Same "see the window" fix as A but trainable, no per-query LLM cost. Train
   on the 123 gold + hard-neg. Architectures to consider: a small encoder-decoder
   that cross-attends query to each slot in context, or a long-context encoder
   with a span/choice head. Predict top1/in_top3 and effort. Can it generalize
   where pairwise CE could not, given the SAME 123-pair training set and the
   SAME 17-pair held-out topic gap -- i.e. does seeing the window actually let it
   learn a topic-invariant anaphora signal, or will it also overfit topic?

C. REVISIT THE LEARNED STRM HEADS. The original learned path (three heads: 2a
   relevance / 2b / 2c) went ALL-THREE-OOD at serve in the Phase 4 eval (STRM
   cost=0/hit=0 every seed). Retrain those heads on the 123 new gold + the
   frozen text2x backbone (a solid from-scratch ReferenceSSM backbone that
   clears a separate retrieval gate at 0.86). Different objective from the
   original. Predict top1/in_top3 and effort. Given the heads were OOD on their
   OLD objective, what makes you think a retrain on anaphora-gold would
   generalize where pairwise CE did not -- is the SSM state trajectory a
   topic-invariant feature or also topic-bound?

D. HOLD + PRODUCTIZE THE PARTIAL WIN. Keep salience opt-in, ship a shortlist-3
   product surface on recall@3=0.588 (the referent is recalled in 10/17, 2 short
   of 0.667). If the product can surface 3 candidates instead of 1, the gate is
   recall@3 not top1, and the cosine+age breadth-3 mechanism is already
   shippable as a recall layer even though it cannot single out the referent.
   Is 0.588 close enough to ship as-is, or is the 2-pair gap a hard blocker?

ALSO ANSWER:
1. Is the 2-session / 17-pair held-out set too small / pathologically hard to
   trust as a ship gate at all? Should I author gold on a THIRD held-out session
   (split one trained session out as a second generalization set) before paying
   any fork -- i.e. is the test-set problem upstream of the ranker problem?
2. Rank the four forks by P(clear) x (1/effort), recalibrated against BOTH the
   zero-shot miss (#6) and the fine-tune miss (#7). State the single highest-
   leverage next step and the number you expect.
3. Ship-vs-continue overall verdict on the CE reranker path (both phases HOLD).

CONSTRAINTS (unchanged): salience stays OPT-IN until a fix clears the real-onyx
gate AND the user calls it; onyx is PRIVATE (never uploaded unsanitized);
prefer cheap / no-retrain first unless you can justify the cost; budget=3
fixed; don't break existing functionality. The 17 held-out pairs are the test
set -- any trainable fork (B, C) must use the 123 trained-session gold only.
"""

payload = {
    "model": "deepseek-v4-pro:cloud",
    "messages": [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ],
    "stream": False,
    "options": {"temperature": 0.3},
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    URL, data=data, headers={"Content-Type": "application/json"}
)

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\ollama_response_finetune_fork.txt"

start = time.time()
last_err = None
for attempt in range(1, 4):
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - start
            obj = json.loads(body)
            content = obj.get("message", {}).get("content", "")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("=== OLLAMA_CALL_OK ===\n")
                f.write(f"elapsed_seconds={elapsed:.2f}\n")
                f.write(f"attempt={attempt}\n")
                f.write(f"model={obj.get('model','?')}\n")
                f.write(f"eval_count={obj.get('eval_count','?')}\n")
                f.write("=== CONTENT ===\n")
                f.write(content)
            sys.stdout.buffer.write(b"OK\n")
            sys.stdout.buffer.write(f"elapsed_seconds={elapsed:.2f}\n".encode())
            sys.stdout.buffer.write(f"attempt={attempt}\n".encode())
            sys.stdout.buffer.write(f"eval_count={obj.get('eval_count','?')}\n".encode())
            sys.stdout.buffer.write(f"response_file={out_path}\n".encode())
            sys.exit(0)
    except urllib.error.URLError as e:
        last_err = f"URLError: {e}"
        sys.stdout.buffer.write(f"attempt {attempt} failed: {last_err}\n".encode())
        time.sleep(2)
    except Exception as e:
        last_err = f"{type(e).__name__}: {e}"
        sys.stdout.buffer.write(f"attempt {attempt} failed: {last_err}\n".encode())
        time.sleep(2)

sys.stdout.buffer.write(f"=== OLLAMA_CALL_FAILED ===\nlast_error={last_err}\n".encode())
sys.exit(1)