import json, time, sys, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. You have advised this project before. Be concrete, "
    "rank actions by expected leverage, give a crisp ship-vs-continue verdict, "
    "and state the single highest-leverage next step. Do not hedge with "
    "disclaimers. When you predict an outcome, state the number you expect. "
    "When your prior prediction was wrong, say so explicitly and recalibrate."
)

user_msg = r"""STRM consult #7 -- your zero-shot cross-encoder recommendation FAILED the
gate; recalibrate and give the next fix.

RECAP (your consult #6). The cosine+age salience mode HOLDs the real-onyx gate
(17 hand-authored non-circular gold back-reference pairs on 2 held-out onyx
sessions; the task = "which prior USER turn does this anaphoric query refer back
to"). Cosine numbers: target_top1=0.235, competitor_beats=0.765, median_breadth
=12 (budget=3). You diagnosed RANKER + REPRESENTATION (not threshold), and
ranked fix #1 = zero-shot cross-encoder reranker (cross-encoder/ms-marco-
MiniLM-L-6-v2) + rank-then-budget: bge_cos pre-filter cos>0.4, CE re-score,
take top-3. You predicted target_top1=0.76 (13/17), competitor_beats=0.12,
breadth=3 -> gate clears, P(clear)=0.85. Fallback: if zero-shot < 0.70, fine-tune
the CE on the gold pairs augmented to ~200.

WHAT I RAN (your recipe exactly, on the same 17 gold pairs + 16-turn window):
- target_in_candidates_rate = 1.000  (every target survives the cos>0.4 filter)
- target_top1_rate          = 0.294  (5/17)   -- you predicted 0.76
- target_in_top3_rate       = 0.588  (10/17)  -- the recall metric
- competitor_beats_rate     = 0.706  (12/17)  -- you predicted 0.12
- median_breadth            = 3.0    (by construction -> PASS)
- nofilter_target_top1_rate = 0.294  (CE rank over ALL window slots, no cos floor
  -- identical to filtered; the cos>0.4 pre-filter does not change the rank at
  all, because in the long session all 16 window slots clear cos>0.4 anyway)

So your P=0.85 prediction was WRONG. Zero-shot CE is a WASH on ranking
(0.235 -> 0.294) -- it reshuffles, it does not reliably capture anaphora.

PER-PAIR PATTERN (the key evidence -- CE fixes some coreference cases cosine
missed but BREAKS others):
- FIXED by CE: q8->t3 (cos rank 3 -> CE rank 1, "coding agent workflow" ->
  "work seamlessly with coding agents"); q37->t21 (cos rank 8 -> CE rank 2,
  "storing the entire AST of the code as graph data" -> "code ontology should
  be more exhaustive" -- the case cosine got most wrong); q21->t7 (cos 8 ->
  CE 1, "code ontology more exhaustive" -> "ontology extraction from
  Documents/Emails"); q53->t40 (cos 1 -> CE 1, both win).
- BROKEN by CE: q57->t53 (cos rank 5 -> CE rank 16, DEAD LAST -- "next version
  = procedural memory generating powerhouse with transferrable domain
  knowledge" -> "graph ontology around skills / living and breathing and
  editable"); q36->t32 (cos 7 -> CE 10); q58->t52 (cos 6 -> CE 8); q19->t14
  (cos 4 -> CE 8); q54->t48 (cos 3 -> CE 7).

INTERPRETATION. MS MARCO is trained on web QUERY -> relevant PASSAGE, not
conversational anaphora -> the specific earlier turn it refers back to. These
queries are long rambling multi-topic user monologues, not web search queries.
The CE has the right ARCHITECTURE (bi-encoder cross-attention captures
paraphrase) but the wrong TRAINING DISTRIBUTION. It fixes the cases that happen
to look query->passage-like and breaks the ones that don't.

THE STRUCTURAL WIN THAT DID HOLD. rank-then-budget bounds breadth to 3 by
construction -> the breadth=12 axis (the worst) is GONE. Two of three gate axes
now pass or nearly pass (breadth=3 PASS; target_in_top3=0.588 close to 0.667).
The REMAINING problem is PURELY the ranker: get target_top1 from 0.294 to >=
0.667. The threshold problem is solved; the ranker problem is not.

THE CIRCULARITY CATCH ON YOUR FALLBACK. Your fallback was "fine-tune the CE on
the 17 gold pairs." But the 17 pairs ARE the held-out TEST set -- fine-tuning on
them is circular (it would test on train). To fine-tune non-circularly I must
mine NEW gold from the 51 TRAINED onyx sessions (which the model has never seen
as gold-labeled either), then test on these 17 held-out. That is real labeling
effort (~50-100 hand-authored pairs). The 17-pair test set is too small to
split into train/test.

QUESTIONS:
1. Given zero-shot MS MARCO underperforms (distribution mismatch, not
   architecture): is the highest-leverage next move (a) fine-tune the CE on
   ~50-100 NEW gold pairs mined from the trained sessions (non-circular, real
   labeling effort, attacks the exact ranker failure with the right
   architecture) -- and if so, how many pairs, what augmentation, and what
   number do you predict on the 17 held-out; (b) swap to a zero-shot model
   whose pretraining is closer to conversational anaphora / coreference / NLI
   (name specific models -- e.g. a NLI cross-encoder, a conversational reranker,
   an instruction-tuned retriever) and predict the number, BEFORE paying
   labeling effort; (c) go straight to Step 4 pure-bilinear 2a retrain on the
   frozen backbone (your prior P=0.50) -- but you previously said the frozen
   routing backbone encodes procedural not semantic features, so can it capture
   the AST->ontology anaphora link that even MS MARCO misses; (d) joint backbone
   fine-tune (#7, your P=0.90, high effort). Rank by P(clear) x effort AGAIN,
   recalibrated against this miss.
2. Is there a cheaper zero-shot signal I have not tried that is closer to the
   conversational-anaphora distribution than MS MARCO -- specifically, would an
   NLI/entailment cross-encoder (which asks "does the query ENTAIL / refer back
   to this turn") or an instruction-tuned retrieval model ("which prior turn
   does this refer to") likely beat 0.294 without any training? Name the models
   and predict the number. This could be a cheap intermediate test before
   paying labeling effort.
3. The target_in_top3=0.588 (recall) is closer to the gate than target_top1=
   0.294 (precision-rank-1). Is the product-relevant gate actually target_in_
   top3 >= 2/3 (the referent is RECALLED, even if not #1), not target_top1?
   If so, a model that lifts target_in_top3 from 0.588 to >= 0.667 is enough
   to ship -- a much weaker bar. Does that change your ranking?
4. Ship-vs-continue verdict on the rank-then-budget + zero-shot-CE path as-is,
   and the SINGLE highest-leverage next step (with the specific model/data and
   the number you expect on the 17 held-out).

CONSTRAINTS (unchanged): salience stays OPT-IN until a fix clears the real-onyx
gate AND the user calls it; onyx is PRIVATE (never uploaded unsanitized);
prefer cheap / no-retrain first; budget=3 fixed; don't break existing
functionality. The 17 held-out pairs are the test set -- any fine-tune must use
NEW gold from the 51 trained sessions.
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

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\ollama_response_ce_reconsult.txt"

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