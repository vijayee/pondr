import json, time, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. You have advised this project three times before and "
    "your diagnoses have been exactly right each time. Be concrete, rank actions "
    "by expected leverage, give a crisp ship-vs-continue verdict, and state the "
    "single highest-leverage next step. Do not hedge with disclaimers."
)

user_msg = r"""STRM (State-Trajectory Ring Memory) consult #4 -- RESULTS from Stage 3, the
fine-tune you diagnosed in consult #3. Your representation-not-readout diagnosis
was correct; I need your read on the result + the from-the-start solid approach.

RECAP OF THE ARC (so the from-scratch question is grounded):
- Substrate: a frozen 19.5M ReferenceSSM backbone (d_model=384, 4 layers,
  d_state=16), originally trained with L_relevance + L_trajectory on MEAN-POOL
  z_k [384] over ERAG-Bench (a routing/relevance objective), NOT on flat_last
  [6144] over the real onyx serve ring.
- Goal: a readout head that, from the live WM ring state, LOCATES which ring
  slots are query-relevant. Gate = live per-source z_logit selectivity gap
  median >= 2.0, in TWO buckets (retrieved_text AND retrieved_code), each in
  >= 4/6 seeds (the 6-seed robustness bar; 3-seed passes proved to be variance
  luck -- consult #2's warning).
- The ring slot state consumed by the readout is flat_last [6144] = the last
  SSM layer's 16 channels x 384, flattened.

WHAT I TRIED AND FAILED (the readout-only path, all on the FROZEN backbone):
  Stage 2 #1 class-balanced SAMPLER (inverse-freq, replacement) -> 6-seed FAIL,
    but s2 was the first seed EVER to pass both buckets -> proved the shared
    readout CAN pass both; instability was the blocker.
  Stage 2 #2 per-kind LOSS weighting -> COLLAPSES under AdamW (stuck train_loss
    ~2.495). KEY LESSON: with AdamW, class-balanced SAMPLING is clean (scale-1.0
    steps); class-balanced LOSS weighting is not.
  Stage 2 #3 code hard-neg mining -> FAIL (ret_text +31.9 2/3 BUT ret_code
    -14.2 0/3). Loss-structure change can't fix a representation weakness.
  Stage 2 #4 no-replacement inverse-freq sampler -> FAIL (ret_code 1/3) but
    closest-yet pre-#5 (code median -4.26 -> +0.72).
  Stage 2 #5 sqrt-inverse-freq no-replacement sampler -> 3-seed PASS (variance
    luck) then 6-seed FAIL (ret_text 3/6, ret_code 2/6). The ceiling.
  Stage 2 #6 per-kind BODIES + shared head (your ladder step 3, the arch
    fallback) -> COLLAPSED MAXIMALLY on s0 (#5's best seed, +23/+17 -> 0.000).
    Structural lesson: the selectivity gate is FUNDAMENTALLY CROSS-KIND; any
    per-kind decomposition makes cross-head logits ill-defined. ONLY a single
    SHARED readout gives a comparable cross-kind logit space. (An MoE redesign
    also FAILED 0/3, ceiling 0.000, same root.)
  => Readout-only path EXHAUSTED. Your consult #3 verdict: REPRESENTATION
    problem, not readout-arch -- flat_last [6144] is kind-isolated.

STAGE 3 (what you advised): cheap fine-tune of the backbone. 8 epochs, ALL
19.5M params, lr 1e-5 (1/10 from-scratch base), AdamW wd 0.01 cosine, margin
loss (m=2.5, hard-neg) on flat_last over the onyx doc ring via a THROWAWAY slim
readout (Linear(6144,384) + cos(z,q)/T=0.05, DISCARDED after -- low capacity
FORCES the backbone to produce a query-relevant flat_last, can't compensate).
Replay strategy: pre-state replay, truncated-BPTT-depth-1. Each kept slot's
pre-step WM state (slots_pre_state) + EXACT step-input embedding
(slots_step_input) are captured on the slot; the fine-tune seeds
states = slots_pre_state[k] (DETACHED) and re-steps ONLY slots_step_input[k]
WITH grad through layer.step -> reproduces slots_h_raw within fp16 epsilon AND
backprops into the shared backbone (W_A/W_B). No cross-slot BPTT.

THE FIDELITY FIX (the key correctness piece, and a general pitfall): retrieved
CODE docs are injected "by MEANING" -- the orchestrator steps
embed(embed_text or summary), but slot.text stores the summary string. My first
replay attempt re-embedded slot.text at trace-build, which DIVERGED from the
actually-stepped vector for ~20% of retrieved slots (the code-doc slots),
breaking replay fidelity (max-abs-diff 0.2421 > atol 0.15 -- the fine-tune
would have trained on bogus state). FIX: capture the EXACT step-input u on the
slot (post-pin, pre-SSM, fp32) and replay from it -> fidelity 0.0002
(essentially exact). Loss then dropped smoothly 2.51 -> 0.46 across 8 epochs
(NOT stuck at 2.49, the per_kind_bodies collapse signature).

STAGE 3 RESULT (6-seed live serve gate, fine-tuned backbone, THEN retrained the
UNCHANGED #5 shared-body readout (sqrt-freq sampler) on the fine-tuned
backbone's regenerated traces):
  retrieved_text: median +10.26, 4/6 pass (need 4) -> PASS
  retrieved_code: median +8.25,  6/6 pass (need 4) -> PASS
  per-seed: s0 code 10.44 / text -0.24 ; s1 8.09 / 28.00 ; s2 8.42 / 1.27 ;
            s3 6.62 / 10.05 ; s4 5.69 / 14.44 ; s5 10.10 / 10.46.
  EVERY seed clears code (6/6). Only s0 (text -0.24) and s2 (text 1.27, just
  under) miss -- both miss on TEXT, not code. ret_code went 2/6 -> 6/6.
  within-corpus held-out: bilinear z_logit median 3.02, 3/3 pass.
This is the FIRST readout to clear BOTH buckets at the 6-seed robustness bar.

STATE / CONSTRAINTS:
- backbone_v2_full_finetuned.pt is a NEW file; the original backbone_v2_full.pt
  is UNTOUCHED and remains the live default. Nothing is wired into the live
  engine yet (integration is a separate decision).
- The fine-tune trained on 939 records (the onyx doc ring; the user's private
  conversational data, ~52 sessions). Small dataset.
- The fine-tuned backbone has NOT been re-validated on the broader objectives
  the original backbone serves (the 2a retrieval gate val 0.826, the DocKindHead
  ensemble, the ERAG routing objective). Catastrophic-forgetting risk is OPEN.
- The original backbone's flat_last was never directly relevance-trained (it was
  trained on mean-pool z_k [384] / ERAG), so the fine-tune targeted a genuinely
  untrained path -- but it did move the shared backbone params (W_A/W_B).

QUESTIONS:
1. READ ON THE RESULT. Does this cleanly confirm the representation-not-readout
   diagnosis? Is the 4/6 text (s0 text -0.24, s2 text 1.27 -- both TEXT misses,
   code is 6/6) an acceptable robust pass, or a yellow flag that the text
   direction is still weakly unstable? Is "every seed clears code, 4/6 clear
   text" a coherent story or a red flag hiding in a green result?

2. THE FROM-THE-START SOLID APPROACH. Given the whole arc -- readout-only
   exhaustion, kind-isolation, the cross-kind-logit requirement, the
   fidelity/injection-by-meaning pitfall, AdamW-sampling-vs-loss-weighting, the
   3-seed-variance-luck trap -- if you were building this from scratch, what is
   the RIGHT approach that avoids every pitfall we hit? Concretely: should the
   backbone be trained FROM THE START with a flat_last query-doc relevance
   objective on the real serve ring (joint with routing), instead of mean-pool
   z_k / ERAG then fine-tuned? Should the readout be the only thing trained, or
   should relevance be a backbone objective from day 1? Where does the
   injection-by-meaning fidelity trap get designed out (capture step-input u by
   construction in the trace format)? Rank the pitfalls by how much pain they
   caused vs how cheaply they'd be designed out upfront.

3. INTEGRATION. The head now serves its purpose for the first time. To wire it
   into the live engine: (a) which seed(s) to serve -- single best seed, the
   4/6 majority, or a logit-ensemble of the passing seeds (s1/s3/s4/s5)? (b) Do
   I need to re-validate / re-train the 2a retrieval gate, DocKindHead, and
   ERAG routing on the fine-tuned backbone before wiring, or is the original
   backbone preserved-enough that I serve the fine-tuned backbone ONLY for the
   STRM readout path and keep the original for everything else (two-backbone
   split)? Is a two-backbone split sane, or a maintenance trap?

4. FRAGILITY. Is the fine-tune fragile in any way I should check before
   trusting the PASS: overfit to 939 records (it moved 19.5M params on a tiny
   set); dependence on the throwaway readout's low capacity (would a different
   throwaway readout change the resulting backbone); the truncated-BPTT-depth-1
   approximation (is single-step replay grad a good estimate of the true
   multi-step grad, or a bias I should quantify); catastrophic forgetting of
   the routing objective (flat_last changed, but so did the shared W_A/W_B that
   every other head reads through mean-pool)?

5. Single highest-leverage next step, and ship-vs-continue. If the answer is
   "wire it (with caveats)", say which seed/ensemble and whether to
   two-backbone-split. If "re-validate before wiring", say which validation is
   load-bearing. If "the from-scratch retrain is now warranted because you know
   the right objective", say so directly.
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

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_stage3_results_consult_response.txt"
raw_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_stage3_results_consult_raw.json"

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