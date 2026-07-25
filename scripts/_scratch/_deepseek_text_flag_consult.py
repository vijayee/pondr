import json, time, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. You have advised four times before; your diagnoses "
    "have been exactly right. Be concrete, rank actions by expected leverage, "
    "give a crisp ship-vs-continue verdict, and state the single highest-"
 "leverage next step. Do not hedge with disclaimers."
)

user_msg = r"""STRM consult #5 -- the text yellow flag. Consult #4 reviewed the Stage 3 PASS
(retrieved_text 4/6 median +10.26, retrieved_code 6/6 median +8.25 -- first readout
to clear the 6-seed live gate on BOTH buckets). You called the 4/6 text a yellow
flag, not red, and guessed the 939-record fine-tune ring "skews code-heavy." I
measured the actual balance and refined the diagnosis; I want a concrete plan to
push text 4/6 -> 5/6 or 6/6 WITHOUT breaking code (6/6) or conv (5/6).

THE ACTUAL GOLD-KIND BALANCE (I was wrong about "code-heavy"):
  conv-gold:  562 records (60.2%)   <- the majority
  text-gold:  229 records (24.5%)
  code-gold:  143 records (15.3%)   <- the minority
So the ring is CONV-heavy, not code-heavy. conv and text-docs are both PROSE;
code-docs are structurally distinct (1f-6 showed summarizing code-doc embeddings
lifts text 7.6x but hurts code; the code signal is in the SSM state h).

THE SAMPLER MECHANICS (load-bearing for what levers even work):
  Stage 3 fine-tune uses the #5 sampler = WeightedRandomSampler(replacement=False,
  weights=sqrt(1/count), num_samples=len(records)=934). With replacement=False
  and num_samples = the full count, this is a WEIGHTED SHUFFLE: every record is
  seen EXACTLY ONCE per epoch; the weights only change ORDER. So per epoch the
  backbone gets 562 conv-gold gradient steps, 229 text-gold, 143 code-gold --
  fixed by the raw counts, NOT by the sqrt weights. "Upweight text in the
  sampler" does nothing for step count here; to give text more gradient I must
  DUPLICATE text records (expand the dataset) or switch to replacement=True.

PER-SEED RESULT (6-seed live serve gate, fine-tuned backbone, #5 shared readout):
            conv    text    code
  s0       15.37   -0.24   10.44   <- text FAIL; conv-strong
  s1        0.00   28.00    8.09   <- conv FAIL; text-strong
  s2       17.34    1.27    8.42   <- text FAIL (near); conv-strong
  s3       17.43   10.05    6.62
  s4       14.30   14.44    5.68
  s5        2.50   10.46   10.10   <- conv near-fail; text-strong
  bucket:   conv 5/6, text 4/6, code 6/6. gate=2.0, need 4/6.
PATTERN: the two text-FAILing seeds (s0, s2) are CONV-STRONG; the two conv-FAILing
seeds (s1, s5) are TEXT-STRONG. This is a conv/text TRADEOFF across seeds -- a
seed that locks onto the conv direction tends to conflate text-docs with conv
filler (both prose), and vice versa. Code is 6/6 with slack (every seed 5.7-10.4,
median 8.25) because it is structurally distinct AND gets the most relative
emphasis (smallest count -> largest sqrt weight, though as noted that only
reorders). The text bucket is the hard one: prose-vs-prose.

THE FINE-TUNE SETUP (for context): 8 epochs, ALL 19.5M backbone params, lr 1e-5,
AdamW wd 0.01 cosine, margin-ranking loss (m=2.5, hard-neg), pre-state replay
(truncated-BPTT-depth-1), throwaway Linear(6144,384) readout (discarded). 934
records. The original backbone_v2_full.pt is preserved; the fine-tune is a NEW
file, not wired into the live engine.

PRIOR LESSONS THAT CONSTRAIN THE LEVERS (all verified by experiment):
  #2: per-kind LOSS WEIGHTING collapses under AdamW (stuck train_loss ~2.495).
      Class-balanced SAMPLING is clean; loss weighting is not.
  #3: loss-STRUCTURE changes (e.g. code hard-neg mining) CANNOT fix a
      representation weakness -- ret_code went -14.2 (0/3) when I tried.
  #1: replacement=True class-balanced sampler had EXTREME variance (s0/s1
      inverted) -> 6-seed FAIL.
  MoE/per-kind-bodies: per-kind decomposition COLLAPSES (the selectivity gate
      is fundamentally cross-kind; cross-head logits ill-defined). Only a single
      shared readout works. So NO per-kind heads/bodies.
  #4 (your prior): the deep fix is a from-scratch JOINT multi-task backbone
      (routing + flat_last relevance loss on the real serve ring from day 1).

QUESTIONS:
1. CHEAP LEVERS, RANKED. Given conv-heavy + prose-collision + the no-replacement
   step-count mechanic, rank these to push text 4/6 -> 5/6 or 6/6 without
   breaking code/conv:
   (a) DUPLICATE text-gold records (e.g. 2x -> ~458 text vs 562 conv vs 143 code)
       keeping replacement=False over the expanded set -- stable, AdamW-clean.
       What ratio? Does duplicating conv-down (subsample conv-gold) help more,
       and is subsampling safe given conv is 5/6?
   (b) ASYMMETRIC MARGIN: larger margin for text-gold records (force the text
       direction harder away from its prose-conv hard negatives). Is this
   AdamW-safe, or does the #2 loss-weighting-collapse lesson rule it out?
       (It is a per-record margin, not a per-record loss weight -- different
       mechanism. Your call.)
   (c) TEXT-HARD-NEG: for text-gold records, force the hardest negative to be a
       CONV slot (prose), so the text direction is pushed away from prose-conv
       specifically. (Mirror of #3 which forced code-hard-neg and failed -- but
       that failed because it was a representation fix via loss; this targets
       the prose collision directly.)
   (d) A short TEXT-EMPHASIS SECOND PHASE: take the Stage 3 fine-tuned backbone
       and fine-tune a few more epochs on a text-oversampled slice. Risk:
       forget code (currently 6/6). Worth it?
   (e) MORE SEEDS + ENSEMBLE: the conv/text tradeoff across seeds looks
       variance-limited. Is the right answer "train 6 more seeds, ensemble the
       text-passing ones" rather than a data/loss fix? (Serving already ensembles
       s1/s3/s4/s5, which are 4/4 on text -- so at SERVE the text flag is
       already green. The 4/6 is a single-seed-robustness statement.)

2. BIAS OR VARIANCE? Is the conv/text tradeoff across seeds (s0/s2 conv-strong/
   text-weak, s1/s5 the reverse) a sign the fine-tune is VARIANCE-limited (-> more
   seeds + ensemble, no data fix) or BIAS-limited (-> text needs more gradient,
   data fix)? How do I tell from what I have -- is there a cheap diagnostic
   (e.g. per-seed train-top3 by gold-kind, or the text-gold train loss vs
   code-gold train loss per seed)?

3. LOSS-SIDE REALITY CHECK. The #2/#3 lessons say loss-side fixes are dangerous
   under AdamW / can't fix representation. Is there ANY loss-side lever that is
   AdamW-safe for the text flag (asymmetric margin, text-hard-neg, a text-only
   auxiliary term), or should I treat the text flag as purely a DATA/SEEDS
   problem and never touch the loss? Be direct.

4. DEEP FIX NOW? Your consult #4 said: if the forgetting-validation degrades,
   do the joint multi-task retrain; if preserved, ship the ensemble. The text
   flag + the open forgetting question both point at "retrain with the right
   objective." Is the cheap text-oversample fine-tune worth trying FIRST (it's
   ~1 GPU-hour: re-fine-tune from the ORIGINAL backbone on text-duplicated data,
   regen traces, retrain #5 6-seed, re-gate), or is it a distraction from the
   joint retrain that would fix text structurally? Rank: (i) cheap text-oversample
   fine-tune first, (ii) skip to the joint multi-task retrain, (iii) ship the
   ensemble as-is (text already 4/4 at serve) and stop.

5. Single highest-leverage next step, and ship-vs-continue. If a cheap text-
   oversample fine-tune is the right first move, give the exact recipe (duplication
   ratio, whether to subsample conv, epochs, whether to start from the original
   or the already-fine-tuned backbone). If "ship the ensemble, stop" is the
   answer, say so directly.
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

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_text_flag_consult_response.txt"
raw_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_text_flag_consult_raw.json"

start = time.time()
last_err = None
for attempt in range(1, 4):
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - start
            obj = json.loads(body)
            content = obj.get("message", {}).get("content", "")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("=== DEEPSEEK_CALL_OK ===\n")
                f.write(f"elapsed_seconds={elapsed:.2f}\n")
                f.write(f"attempt={attempt}\n")
                f.write(f"model=deepseek-v4-pro:cloud\n")
                f.write("=" * 70 + "\n\n")
                f.write(content)
            print(f"OK elapsed={elapsed:.2f}s attempt={attempt}")
            # avoid cp1252 console crash on non-ascii
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            print(content)
            break
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        last_err = e
        print(f"attempt {attempt} failed: {e}", flush=True)
        time.sleep(5)
else:
    print(f"ALL ATTEMPTS FAILED: {last_err}")
    raise SystemExit(1)