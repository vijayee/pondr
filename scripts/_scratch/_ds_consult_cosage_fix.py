import json, time, sys, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. You have advised this project before. Be concrete, "
    "rank actions by expected leverage, give a crisp ship-vs-continue verdict, "
    "and state the single highest-leverage next step. Do not hedge with "
    "disclaimers. When you predict an outcome, state the number you expect."
)

user_msg = r"""STRM (State-Trajectory Ring Memory) consult #6 -- the salience discrimination fix.
You advised this project before; your "#5 falsification" prediction was exactly
right and is now CONFIRMED. I need the fix.

SYSTEM RECAP (for continuity). STRM salience decides which slots in a live ring
to recall into context at query time. The ring holds two slot types: (0)
CONVERSATION slots = the raw USER text of recent turns (orchestrator.py:642), and
(1) RECALLED-EPISODE slots = 200-char assistant summaries of older episodes
(orchestrator.py:758). The salience DECISION gates recall to a budget of 3
slots/turn. Backbone: frozen 19.5M ReferenceSSM (d_model=384, 4L, d_state=16),
trained ONLY on a routing objective; its flat_last readout [6144 = 16 channels x
384] encodes PROCEDURAL features (turn position, recency, slot type, retrieval
scores), NOT deep query-document semantics. Embedder: frozen bge-small-en-v1.5
(384-dim), used as `bge_cos(query_text, slot_text)` with NO extra normalization.

THE JOURNEY TO THE CURRENT FAILURE:
1. Learned three-head AND salience: `salient = (rec_i<theta) & (r_i>phi) &
   (surprise_i<cap)`, three heads (2a relevance / 2b recoverability / 2c
   surprise) trained on frozen backbone. ALL THREE are OOD-fragile at serve:
   2c surprise blocks 100% of turns (cap 100-280x too low); 2b rec_i never
   crosses theta (and is run-to-run unstable); 2a r_i saturates at +1.000 (no
   discrimination). -> Phase 4 ship eval HOLD. (You diagnosed this in consult #5
   and your "last resort = drop all learned heads, use cosine+age" was forced
   to be the first move.)
2. Cosine+age fallback (the fix under test): `salient = (bge_cos(query,
   slot_text) > cos_phi) AND (age >= age_threshold)`, both frozen/OOD-immune,
   cos_phi=0.6, age_threshold=3. No retrain. SYNTHETIC ship gate PASSED
   (coverage 3/3, accuracy 2/3; STRM acc=1.000 vs FIXED 0.000/0.167 at equal
   budget). But synthetic fillers are topically-DISTINCT (weather/recipes) --
   easy cosine competitors.
3. YOUR #5 FALSIFICATION (now CONFIRMED): you predicted that at REAL serve the
   ring is full of topically-CLOSE recent conversation turns, so
   `bge_cos(query, recent_turn)` exceeds cos_phi for many slots -> the trigger
   fires indiscriminately -> cost explosion + non-selective recall. The real
   ship decision = a real-onyx held-out test; I just ran it.

THE REAL-ONYX HELD-OUT RESULT (HOLD). 2 held-out onyx chat sessions (UNSEEN,
held out of the trace generator): 24-turn "Benchmarking Ponder Engine" and
226-turn "Hippocampal Indexing Architecture". I hand-authored 17 gold
back-reference pairs (a human picked the earlier USER turn each query refers
back to -- NON-CIRCULAR, not cosine), all in-window (age 3-15). The probe
reproduces the trigger's signal EXACTLY: pure bge_cos(query_user_text,
prior_user_text) over the frozen bge embedder, over a 16-turn recent window.
No orchestrator, no heads, no retriever, no LLM -- zero confounds.

Aggregate (cos_phi=0.6, age_threshold=3, window=16, budget=3, n=17/17):
  target_top1_rate      = 0.235   (gate needs >= 0.667)  -- target wins cosine in only 4/17
  fires_on_target_rate  = 1.000   (gate needs >= 0.667)  -- target always clears cos_phi
  competitor_beats_rate = 0.765   (gate needs <= 0.333)  -- a competitor beats/ties in 13/17
  median_breadth        = 12.0    (gate needs <= 3)      -- ~12 of 16 slots fire per query
  p90_breadth           = 15.0
  median_gap            = -0.040  (target_cos - max_competitor_cos)
  p10_gap               = -0.097
  median_target_cos     = 0.727   -- the target IS relevant (>> cos_phi 0.6); the
                                   problem is NOT target relevance, it is breadth + competitors.

The 4 wins (target_rank=1): gaps +0.162, +0.020, +0.021, +0.029. The other 13:
a recent SAME-THREAD turn beats the gold referent on cosine.

GOLD SOUNDNESS VERIFIED (the losses are real discrimination failures, not label
errors). Spot-checked 3 losing pairs:
- q37 "storing the entire AST of the code as graph data for an agent writing code"
  -> gold target u21 "the code ontology should be more exhaustive even for just a
  seed" (the conceptual referent: code-structure-as-graph/ontology), cos 0.669,
  rank 8. Loses to u29/u30 "hardware specs to run the fully trained solution"
  (cos 0.714) and u23 "graphql feature of wavedb" (cos 0.704) -- topically WRONG
  turns winning on lexical overlap ("graph"/"trained").
- q8 "understand the workflow of a coding agent / ponder designed around chatbot
  behavior" -> gold target u3 "work seamlessly with coding agents / Git storage of
  choice for code", cos 0.727, rank 3. Loses to u00 "benchmark the ponder engine /
  not exactly a rag or chatbot" (cos 0.767) -- shares "ponder/chatbot" vocab.
- q52 "procedural memory / a memory related workflow" -> gold target u40 "agent
  skill / meta-skills in writing processes", cos 0.721, rank 6. Loses to turns
  sharing the generic word "memory" (cos up to 0.819).

THE DEEPER FINDING (why no threshold fixes this). target_cos is always 0.63-0.82
(always > cos_phi=0.6 -- the target is always "relevant enough"), but so is
everything else in a single-thread conversation. cos_phi=0.6 cannot discriminate,
AND no threshold can both fire on the target and not on competitors, because
competitors are typically AS OR MORE cosine-similar than the referent. The
discrimination actually needed is "which prior turn does THIS query refer back to
CONCEPTUALLY" (anaphora/coreference over a single thread) -- NOT "which is most
lexically similar." Frozen bge-cosine is lexically myopic and cannot do this.

AVAILABLE LEVERS (your prior rankings, for reference):
- Step 4 "pure-bilinear 2a retrain": drop the yt_sidepath OOD offset + sigmoid,
  hard-negative mining, serve-distribution training, gate-on-logit. Prior probe
  (probe-4a) showed the RAW bilinear LOGIT (before sigmoid+yt_sidepath) has a
  selectivity gap +0.69 with 18/89 turns >= 2.0 -- i.e. a TRAINED relevance
  signal EXISTS in the backbone states but the shipped head's sigmoid+yt_sidepath
  drown it. So a trained head MIGHT beat lexical overlap where frozen cosine
  can't. BUT your prior diagnosis (#1-#7) said the frozen routing-trained backbone
  encodes PROCEDURAL not semantic features -> a trained head on FROZEN backbone
  may still lack the conceptual-reference signal (the u37->u21 "AST"->"ontology"
  link) that bge misses.
- Query-conditioned low-rank projection of h_raw (FiLM/hypernetwork, your prior #2).
- Train on LIVE-distribution traces (your prior #3).
- Joint backbone fine-tune (your prior #7): the ONLY way to inject genuine
  query-relevance signal into the states; major effort, may degrade routing.
- A small fine-tuned CROSS-ENCODER reranker over the cosine candidates (e.g.
  rerank the ~12 slots that clear cos_phi down to the budget-3, trained on the
  gold pairs). Cheap, attacks the EXACT failure (breadth + rank), no backbone
  change. Not previously ranked.
- A threshold-free formulation: replace the AND gate with a rank-then-budget
  (always recall the top-K=3 by some score, no fire threshold) so breadth is
  bounded by construction and the only question is whether the scorer ranks the
  target #1. Not previously ranked.

CONSTRAINTS (unchanged):
- Don't break existing functionality; salience stays OPT-IN until a fix clears
  the real-onyx gate AND the user calls it. The user's principle: "no point
  integrating until it serves its purpose well."
- The 2 held-out onyx sessions are PRIVATE; nothing uploaded unsanitized.
- Prefer cheap / no-retrain first; the backbone fine-tune (#7) is the expensive
  last resort.
- Budget = 3 slots/turn is fixed (cost constraint). Any fix must both RANK the
  target #1 AND bound breadth to ~3 (the current failure is breadth 12 + rank
  not-#1; either alone is a HOLD).

QUESTIONS:
1. Given the real-onyx HOLD: is the failure best framed as (a) a THRESHOLD problem
   (no cos_phi works because competitors >= target cosine) -- you already
   predicted this; (b) a RANKER problem (need a scorer that ranks the conceptual
   referent above lexical-overlap competitors); (c) a REPRESENTATION problem
   (bge-small + frozen routing-backbone both lack the anaphora signal); or some
   combination? Rank the root causes by leverage.
2. Ranked fixes to reach a SHIPPABLE real-onyx gate (target_top1>=2/3,
   competitor_beats<=1/3, breadth<=3), by expected probability-of-clearing x
   effort. Specifically evaluate: (a) the cross-encoder reranker over cosine
   candidates (attacks the exact breadth+rank failure, cheap, no backbone change)
   -- is this the highest-leverage cheap move, and what model/training data; (b)
   the threshold-free rank-then-budget formulation (bounds breadth by
   construction, reduces the problem to ranking only) -- does this change the
   gate math in our favor, and is it product-viable given cost; (c) Step 4
   pure-bilinear 2a retrain on the frozen backbone -- given your prior diagnosis
   that the backbone encodes procedural not semantic features, can a trained head
   on FROZEN backbone actually capture the u37->u21 conceptual-reference signal
   that bge misses, or will it also be lexically myopic; (d) joint backbone
   fine-tune (#7) -- is it now clearly necessary, or only if (a)/(b)/(c) fail.
3. Is the budget=3 + breadth=12 finding telling us something structural -- i.e.
   that in a single coherent conversation thread, "recall the topically-relevant
   prior turns" is the wrong unit, and the right unit is "recall the SPECIFIC
   referent this anaphoric query points back to" (a much smaller, pointer-like
   recall)? If so, does that change the architecture (e.g. an explicit
   coreference/pointer head rather than a relevance scorer)?
4. Ship-vs-continue verdict on cosine+age as-is, and the SINGLE highest-leverage
   next step (with the specific model/data/loss if it involves training, and the
   number you expect on the real-onyx gate).
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

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\ollama_response_cosage_fix.txt"

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