import json, time, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. Be concrete, rank actions by expected leverage, give "
    "a crisp ship-vs-continue verdict, and state the single highest-leverage next "
    "step. Do not hedge with disclaimers."
)

user_msg = r"""STRM (State-Trajectory Ring Memory) follow-up consult. Frozen 19.5M ReferenceSSM
backbone (d_model=384, 4 layers, d_state=16) trained ONLY on a routing objective.
flat_last readout [6144 = last layer 16 channels x 384] encodes PROCEDURAL features
(turn position, recency, slot type, retrieval scores), NOT deep query-doc semantics.
A CompositeZHead = StateReadout mlp128 [6144->128->384] + ZRelevanceHead bilinear
scores each ring slot vs the query. The live ring holds (conversation-message slots +
retrieved-document slots of 3 kinds: conv / text-doc / code-doc). Training data: 934
records from 50 Onyx sessions replayed against a 72-doc corpus (58 code .py sections,
14 text .md, plus conv). Gold = the top-cos prior slot after dropping the cos~1.0
self-slot. The LIVE GATE = per-source z_logit selectivity gap (top-relevant-slot logit
minus mean filler logit, per source_id, median over eligible turns) >= 2.0 in >= 2/3
seeds. PASS criterion = retrieved_text >= 2.0 AND retrieved_code >= 2.0, each in >= 2/3.
Optimizer = AdamW (lr 3e-4 default, wd 0.01, cosine schedule, 120 epochs, dropout 0.3,
margin-loss m=2.5 hard-negative, --select-ckpt final). Gold-kind counts: conv=562 /
text=229 / code=143 (conv is 60% majority).

PRIOR STATE: 1f-6 (SINGLE SHARED readout, all data) hit retrieved_TEXT 30.16 (2/3 PASS)
but retrieved_CODE -4.94 (0/3, MIS-RANKS) -- code docs are the drag. I diagnosed this as
the conv-MAJORITY-dominated shared body (562 conv vs 229 text vs 143 code gold): the
shared 6144->128 body is pulled toward conv, so the code query direction is weak.

TWO THINGS I LEARNED THAT CONSTRAIN THE SOLUTION:

(A) STRUCTURAL (the MoE redesign FAIL): the selectivity gate is FUNDAMENTALLY CROSS-KIND.
Each probe's gold is one kind, but its FILLERS span conv+text+code (the ring is mixed).
The gate measures the gold-vs-filler logit gap per probe; the fillers are cross-kind. ANY
per-kind decomposition (I tried TWO: (i) shared body 6144->128 + per-kind Linear 128->384
with --kind-head-wd 0.05; (ii) MoE = N independent full 6144->128->384 readouts per kind,
NO shared body, by-gold train routing) uses per-SLOT serve routing -> the gold goes to its
kind's readout, the CROSS-KIND fillers go to DIFFERENT readouts -> cross-head logit
comparison is ILL-DEFINED (independent readouts have independent scales/biases). Both
failed: (i) ret_text 0.558 (1/3) / ret_code 0.825 (0/3); (ii) ret_text 0.595 (0/3) /
ret_code 0.138 (0/3), conv collapsed to exactly 0.0, all-turns ceiling 0.000. KEY LESSON:
ONLY a SINGLE SHARED readout produces a comparable cross-kind logit space. 1f-6 had that.
So the solution MUST keep a shared readout (or a shared final projection head on per-kind
bodies -- untested). Per-kind decomposition is the wrong tool; it breaks the cross-kind
comparability the gate requires.

(B) ADAPTIVE-OPTIMIZER (the Stage 2 FAIL): the problem narrowed to "lift code in a single
shared readout without breaking text 30.16." I tried TWO class-balanced rebalancing
variants on the 1f-6 shared readout (n_doc_kinds=0):

  #1 SAMPLER (--class-balanced-gold): torch WeightedRandomSampler over train records
     weighted by 1/count(gold_doc_kind), WITH replacement, 621 samples/epoch (code
     upweighted ~3.6x vs conv, text ~2.4x; each kind ~207 samples/epoch). Each STEP's
     loss is UNWEIGHTED (scale 1.0); only the record MIX changes.
     RESULT: LIVE GATE FAIL 1/3, BUT seed 2 is the FIRST seed EVER to pass BOTH buckets
     simultaneously -- ret_text +5.641 AND ret_code +4.435 (1f-6 never had both; it had
     text 30.16 / code -4.94). EXTREME variance: s0 ret_text -12.455 / ret_code -19.385
     (INVERTED), s1 -3.358 / -4.259 (inverted), s2 +5.641 / +4.435 (BOTH PASS).
     3-seed medians ret_text -3.358 (1/3), ret_code -4.259 (1/3). Text REGRESSED
     30.16 -> -3.36: downweighting conv (60% of data) starved the text direction, and
     replacement-sampling added per-epoch noise. But s2 PROVES a shared readout CAN pass
     both buckets with a balanced code gradient -- the blocker is VARIANCE (1/3), not the
     approach. In-sample the sampler trained NORMALLY (top3 ~0.65, train_loss -> 0.43).

  #2 LOSS-WEIGHT (--per-kind-loss-weight): per-record loss weight = 1/count(gold_doc_kind)
     NORMALIZED to mean 1.0 (so the effective lr/loss scale + cosine schedule unchanged;
     only the per-record gradient RATIO shifts: code ~3.6x conv). UNIFORM sampling
     preserved (every record seen once/epoch, NO replacement -- the hypothesized stable
     variant of #1).
     RESULT: COLLAPSES under AdamW. ALL 3 seeds / BOTH archs (bilinear + transformer)
     stuck at the margin-loss ceiling (train_loss ~2.495, top3 ~0.27) from ~epoch 5,
     NEVER recovers; r_pos -> 0.03 (degenerate anti-correlated ranking). WORSE than #1
     (which at least trained normally in-sample). ROOT: per-record loss SCALING fights
     AdamW's adaptive moments -- the running first/second moments MIX gradients scaled by
     different per-record weights (0.57x conv to 2.0x code), and the eps + bias-correction
     terms break the per-step scale invariance AdamW normally provides -> degenerate
     fixed point. KEY LESSON: with an adaptive optimizer, class-balanced SAMPLING is
     clean (each step's loss stays at scale 1.0; only the record mix changes) but
     class-balanced LOSS weighting is NOT.

So: a shared readout CAN pass both buckets (s2 of #1 proved it). The blockers are
(1) code-lift (1f-6 code -4.94) and (2) the high variance of the sampler (1/3 pass).
Loss-weighting is refuted for AdamW. Per-kind decomposition is refuted (cross-kind gate).

QUESTIONS:
1. Variance-taming for the SAMPLER (#1, the only approach that has passed both buckets).
   It passed 1/3 with extreme variance (s2 +5.6/+4.4, s0/s1 inverted). Concrete ways to
   stabilize it without losing the s2 win:
   (a) LESS AGGRESSIVE upweighting -- e.g. upweight code only 2x (not 3.6x), or upweight
       code+text to match conv WITHOUT full balancing (square-root inverse freq, or a
       temperature on the weights). Does milder rebalancing reduce variance while keeping
       code-lift? What's the risk it also reduces the code-lift that made s2 pass?
   (b) NO REPLACEMENT (a WeightedRandomSampler without replacement = a weighted shuffle;
       each record seen exactly once/epoch in a weighted ORDER). This removes the
       replacement-sampling noise (some records unseen/epoch, minority records ~2x/epoch)
       while keeping the per-epoch balance. Does it stabilize? Is it still "sampling" in
       the AdamW-clean sense (per-step loss scale 1.0)?
   (c) MORE SEEDS + majority vote -- if the true pass-rate is ~1/3, running 6 seeds gives
       ~2 passing; but the PASS criterion is ">= 2.0 in >= 2/3 seeds", so 6 seeds at 1/3
       gives ~2/6 = FAIL. Only helps if the true rate is > 1/3. Is 1/3 the real rate or
       did s0/s1 land badly? Would you expect 1/3 or higher with milder rebalancing?
   (d) SEED ENSEMBLE (logit-average the 3 sampler ckpts at serve). But s0/s1 are strongly
       INVERTED (negative) -- would their inversion drag down s2's pass, or would averaging
       cancel the inversion and recover a stable pass? Is this worth trying given the
       inverted seeds?
   Rank (a)-(d) by expected leverage to get a stable >=2/3 pass on BOTH buckets.

2. Hard-negative mining for code (DeepSeek #3, untested): for code-gold records, force
   CODE fillers as the hard negatives in the margin hinge (currently hard-negative = the
   hardest filler overall, which is usually a conv/text filler). This is AdamW-clean
   (works via the loss STRUCTURE -- which negatives are in the hinge -- not gradient
   weighting). Does this target the code-lift problem specifically? Expected effect on
   ret_code without breaking ret_text 30.16? Rank vs the variance-taming options.

3. Per-kind BODIES + SHARED HEAD (untested fallback arch): each kind gets its own
   6144->128 body (escapes the conv-majority-pulled shared body -- the real root cause)
   feeding ONE shared 128->384 head (shared logit space = cross-kind COMPARABLE, the
   lesson (A) proved is required). This is NOT a per-kind decomposition of the HEAD (which
   (A) refuted) -- the shared head keeps the cross-kind logit space; only the BODY
   specializes. Does this actually preserve cross-kind comparability (the bodies feed a
   shared head, so logits are in one space)? Or do the per-kind bodies produce
   incompatible 128-dim representations that the shared head can't reconcile? Rank vs the
   data/loss fixes. Is this the right architectural answer to "shared body pulled by conv"?

4. Stage 2 backbone retrain (joint fine-tune so h carries deeper query-doc semantics).
   Stage 0 proved the code signal IS in h (code-only shared readout -> held-out 6.76).
   Is the readout/training-data fix (questions 1-3) sufficient, or is there a ceiling
   the procedural h can't break? When would you escalate to Stage 2? Rank.

5. Ship-vs-continue: 1f-6 text 30.16 (2/3) is a strong signal; #1 s2 passing both is
   stronger. Is a partial ship (text-only, >= 2.0) viable while code is fixed, or is code
   a core requirement that blocks any ship? Rank the paths by effort vs probability of a
   FULL robust pass on text AND code.

6. Single highest-leverage next step.
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

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_stage2_reconsult_response.txt"
raw_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_stage2_reconsult_raw.json"

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
            print(content)
            break
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        last_err = e
        print(f"attempt {attempt} failed: {e}", flush=True)
        time.sleep(5)
else:
    print(f"ALL ATTEMPTS FAILED: {last_err}")
    raise SystemExit(1)