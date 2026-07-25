import json, time, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. Be concrete, rank actions by expected leverage, give "
    "a crisp ship-vs-continue verdict, and state the single highest-leverage next "
    "step. Do not hedge with disclaimers."
)

user_msg = r"""STRM (State-Trajectory Ring Memory) follow-up consult #2. You advised this project
before; your prediction was exactly right and I need the next decision.

RECAP (your prior ranking): you gave a ship path -- (1) hard-neg mining -> (2)
no-replacement balanced sampling -> (3) per-kind bodies + shared head (architectural
fallback) -- and ranked variance-tamer sub-options: (b) no-replacement highest >
(a) milder sqrt-inverse-freq upweighting medium > (c) more seeds lowest > (d) seed
ensemble low. You EXPLICITLY warned about (c): "if the true pass-rate is ~1/3, 6
seeds gives ~2/6 = FAIL. Only helps if the true rate is >1/3."

WHAT I RAN AND THE RESULT. I implemented your (a): sqrt-inverse-freq no-replacement
sampler (same weighted shuffle as #4 but weights = sqrt(1/count) instead of 1/count
-> milder 1.89x code upweight vs #4's 3.6x). On the 1f-6 SHARED readout (n_doc_kinds=0,
MLP-128 6144->128->384, 934k params), the ONLY arch with a comparable cross-kind
logit space (per-kind decomposition was refuted: the gate is fundamentally CROSS-KIND
-> cross-head logits ill-defined; an MoE redesign FAILed 0/3 with all-turns ceiling
0.000). AdamW-clean (per-step loss scale 1.0; weighted shuffle changes record ORDER,
not loss scale). Trained 3 seeds first (the original gate = z_logit selectivity gap
median >= 2.0 in >= 2/3 seeds, PASS = retrieved_text >= 2.0 in >= 2/3 AND
retrieved_code >= 2.0 in >= 2/3):

3-SEED RESULT (looked like a PASS):
  s0: ret_text +23.20 PASS / ret_code +17.25 PASS  (both pass)
  s1: ret_text  +9.76 PASS / ret_code  +3.69 PASS  (both pass)
  s2: ret_text -0.49 FAIL / ret_code -0.36 FAIL    (flat)
  3-seed medians: ret_text +9.76 (2/3 PASS), ret_code +3.69 (2/3 PASS) -> PASS.
  TWO seeds (s0, s1) passed BOTH buckets simultaneously -- the first time ANY approach
  cleared the gate. s0/s1 are strong; s2 is the lone flat failure. I called it a robust
  pass and asked the user how to proceed. The user chose "run 6 seeds to de-risk"
  (your (c), which you'd ranked lowest and warned about).

6-SEED DE-RISK RESULT (REVERSES the PASS -- your (c) warning was exactly right):
  s0: ret_text +23.20 PASS / ret_code +17.25 PASS  (reproduced deterministically)
  s1: ret_text  +9.76 PASS / ret_code  +3.69 PASS  (reproduced deterministically)
  s2: ret_text -0.49 FAIL / ret_code -0.36 FAIL    (reproduced flat)
  s3: ret_text -1.88 FAIL / ret_code -1.22 FAIL    (NEW flat seed)
  s4: ret_text +0.44 FAIL / ret_code -0.03 FAIL    (NEW near-flat)
  s5: ret_text +29.61 PASS / ret_code +1.44 FAIL   (NEW, code just under 2.0)
  6-seed verdict (need 4/6): ret_text 3/6 FAIL (median +5.10), ret_code 2/6 FAIL
  (median +0.705). -> FAIL.
  TRUE both-buckets pass-rate = 2/6 = ~33%. The 3-seed PASS was variance luck: it
  happened to draw the 2 good seeds (s0, s1) in its first 3. 3 NEW seeds (s3, s4, s5)
  all fail -- s3/s4 go flat (the same failure class as s2), s5 passes text strongly
  (+29.6) but code lands +1.44 (just under 2.0). The shared body collapses to a
  flat or code-weak direction for 4 of 6 seeds.

INTERPRETATION. The sqrt upweight hits a sweet spot for SOME seeds (s0 strongly, s1,
  s5-text) but the shared 6144->128 body is STILL the bottleneck -- it collapses to a
  flat/code-weak direction for the majority. #5 operates DOWNSTREAM of the projection
  (changes record ORDER, not the representation). The data/loss fixes (#1 sampler,
  #2 loss-weight, #3 code-hard-neg, #4 no-replacement, #5 sqrt-freq) have ALL now
  failed the 6-seed robustness bar. Your ladder steps 1-2 are done; the architectural
  fallback (step 3) is the remaining evidence-consistent fix.

CONSTRAINTS (unchanged):
- The readout MUST keep a single shared final projection (or a shared 128->384 head on
  per-kind bodies) -- per-kind decomposition of the HEAD is refuted (cross-kind gate).
- AdamW optimizer; class-balanced SAMPLING is clean, class-balanced LOSS weighting is
  not (collapses). The weighted shuffle (replacement=False) is AdamW-clean.
- The frozen 19.5M ReferenceSSM backbone (d_model=384, 4 layers, d_state=16) was
  trained on a routing objective; flat_last readout [6144 = 16 channels x 384] encodes
  PROCEDURAL features, not deep query-doc semantics. Stage 0 proved the code signal
  IS in h (code-only shared readout -> held-out z_logit 6.76).

QUESTIONS:
1. Per-kind BODIES + SHARED HEAD (your ladder step 3): each kind gets its own
   6144->128 body (escapes the conv-majority pull that flattens 4/6 seeds), all
   feeding ONE shared 128->384 head (shared logit space = cross-kind comparable, the
   lesson the MoE FAIL proved is required). Is this the MOST ROBUST fix for the
   shared-body-collapses-for-2/3-of-seeds failure? Concretely: will 3 independent
   bodies trained jointly via the shared head reliably encode a coherent cross-kind
   code direction (vs the shared body's 2/6 rate), or will the bodies produce
   incompatible 128-dim representations the shared head can't reconcile (your earlier
   stated risk)? How do I train it (joint from scratch? warm-start the bodies from
   the shared 1f-6 body? freeze bodies then train head, or joint)? What's the
   expected 6-seed pass-rate?

2. 2-SEED LOGIT-ENSEMBLE (s0+s1): both passing seeds reproduce DETERMINISTICALLY
   (same numbers across the 3-seed and 6-seed runs), so they're real wins, not lucky
   draws. Logit-averaging s0+s1 at serve: does this give a robust >=2/3 pass on live,
   or is 2 seeds too thin a base to generalize to live traffic? Is there any value in
   combining this with the per-kind-bodies path (e.g. ensemble the per-kind-bodies
   seeds)? Rank vs question 1.

3. Stage 2 backbone retrain (joint fine-tune so h carries deeper query-doc semantics):
   you ranked this lowest priority earlier. Given the 6-seed result (the shared body
   collapses for 2/3 of seeds -- is that a representation weakness in the procedural
   h, or a readout-body weakness that per-kind bodies fixes?), does the priority
   change? Or is per-kind bodies still the right fix (the code signal IS in h per
   Stage 0, so the representation is fine; the readout body is the bottleneck)?

4. Is there an option I'm missing? (e.g. a different readout arch -- a residual or
   gating body; more data; a different loss; regularization to reduce the collapse
   variance; a seed-selection rule at serve.)

5. Single highest-leverage next step, and ship-vs-continue. If per-kind bodies +
   shared head is the most robust solution, say so directly -- I will proceed to
   implement it. If something cheaper is more leveraged, rank it above.
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

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_stage2_reconsult_6seed_response.txt"
raw_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_stage2_reconsult_6seed_raw.json"

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