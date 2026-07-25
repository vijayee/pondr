import json, time, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. Be concrete, rank actions by expected leverage, and "
    "give a crisp ship-vs-continue verdict. Do not hedge with disclaimers."
)

user_msg = r"""Advising on STRM (State-Trajectory Ring Memory): a frozen 19.5M ReferenceSSM
backbone (d_model=384, 4 layers, d_state=16) trained ONLY on a routing objective.
flat_last readout [6144 = last layer 16 channels x 384] encodes PROCEDURAL
features (turn position, recency, slot type, retrieval scores), NOT deep
query-document semantics. A CompositeZHead = StateReadout mlp128 [6144->128->384]
+ ZRelevanceHead bilinear scores each ring slot against the query. The live ring
holds (conversation-message slots + retrieved-document slots of 3 kinds: conv /
text-doc / code-doc). Training data: 934 records from 50 Onyx sessions replayed
against a 72-doc corpus (58 code .py sections, 14 text .md, plus conv). Gold =
the top-cos prior slot after dropping the cos~1.0 self-slot. The LIVE GATE =
per-source z_logit selectivity gap (top-relevant-slot logit minus mean filler
logit, per source_id, median over eligible turns) >= 2.0 in >= 2/3 seeds.
Buckets: full / conv / retrieved / retrieved_text / retrieved_code. PASS criterion
= retrieved_text >= 2.0 AND retrieved_code >= 2.0, each in >= 2/3.

PRIOR STATE (continuity): the doc-kind split was the key revelation. 1f-6 (SINGLE
SHARED readout, all data, margin loss m=2.5 hard-neg) hit:
  retrieved_TEXT 30.16 (2/3 PASS) -- bilinear CAN locate text-doc relevance live;
  retrieved_CODE -4.94 (0/3, MIS-RANKS) -- code docs are the drag.
I diagnosed the code drag as the conv-MAJORITY-dominated shared body (gold counts
conv=562 / text=229 / code=143): the shared 6144->128 body is pulled toward conv,
so the code query direction is weak. Stage 1 tried to fix this with a per-doc-kind
readout (shared body 6144->128 + per-kind Linear 128->384, --kind-head-wd 0.05):
LIVE FAIL, ret_text 0.558 (1/3) + ret_code 0.825 (0/3) -- WORSE on text, barely
lifted code. Stage 0 (code-gold-only shared readout, no conv competition) hit
held-out 6.76 -> the code signal IS in h.

NEW EXPERIMENT (the MoE redesign, your prior "option B" -- MoE on non-overlapping
per-kind data): per_kind_full = N INDEPENDENT full readouts
Sequential(Linear(6144,128),ReLU,Linear(128,384)) per kind, NO shared body.
Train routes by-GOLD-kind (ALL slots of a record -> the GOLD's readout -> the
margin loss is WITHIN one readout = well-defined, gradient flows into one readout
-> each readout trains ONLY on its kind's gold = non-overlapping data, mirroring
Stage 0's code-only win per kind). Serve routes per-SLOT (each slot -> its own
kind's readout, the existing per-slot machinery). Rationale was: the Stage 1
shared-body FAIL was conv-competition in the shared body; MoE removes the shared
body so each kind trains in isolation.

NEW RESULT (MoE, 3 seeds, same 1f-6 flags + --n-doc-kinds 3 --per-kind-data-isolation):
LIVE GATE:
  seed |  full  |  conv  | retrieved | ret_text | ret_code
   s0  |  0.000 |  0.000 |  -0.330   |  0.595   | -13.543  (INVERTED)
   s1  |  0.042 |  0.000 |   0.191   |  0.214   |  0.191
   s2  |  0.212 |  0.000 |   0.355   |  0.708   |  0.138
  -> ret_text 0.595 (0/3); ret_code 0.138 (0/3); conv 0.000 (0/3 = constant logits,
     no separation); ret_code s0 INVERTED -13.5.
TRAIN held-out FULL bucket (per-slot eval routing): s0 -10.5 / s1 -0.06 / s2 +0.21.
ALL-TURNS CEILING (in-sample upper bound): s0 0.000 / s1 0.000 / s2 0.306 = EVEN
IN-SAMPLE the head produces NO separation under per-slot routing.

STRUCTURAL FINDING (the diagnosis): the selectivity gate is FUNDAMENTALLY
CROSS-KIND. Each probe's gold is one kind, but its FILLERS span conv+text+code
(the ring is mixed). The gate measures the gold-vs-filler logit gap PER PROBE,
and the fillers are cross-kind. ANY per-kind decomposition (Stage 1 shared-body
OR MoE) uses per-SLOT serve routing -> the gold goes to its kind's readout, but
the CROSS-KIND fillers go to DIFFERENT readouts -> cross-head logit comparison is
ILL-DEFINED (independent readouts have independent scales/biases; the ranking
one readout learns does not transfer across heads). Concretely: the conv readout
trains on conv-gold records where ALL slots (incl code/text fillers) route to the
conv readout, so it learns "score code/text fillers LOW"; but at serve those
code/text fillers route to the CODE/TEXT readout (trained to score code-GOLD
HIGH) -> those fillers get HIGH logits -> beat the conv gold -> conv collapses to
exactly 0.0, and code inverts (code gold and code fillers both score high via the
code readout -> no within-head gap; cross-kind fillers add noise). The all-turns
0.000 ceiling confirms: per-slot routing cannot separate gold from cross-kind
fillers, period.

KEY LESSON: ONLY a SINGLE SHARED readout produces a comparable cross-kind logit
space. 1f-6 had that (text 30.16 PASS / code -4.94 FAIL). 1f-6's SOLE problem is
code underperformance, caused by the conv-majority-dominated shared body (562 conv
vs 229 text vs 143 code gold). Per-kind decomposition is the WRONG tool -- it
breaks the cross-kind comparability the gate requires. The problem NARROWS to:
"lift code in a single shared readout without breaking text (30.16)."

I see the evidence-consistent path as: keep the 1f-6 shared readout arch, fix the
conv-majority root cause via CLASS-BALANCED training (upweight the 143 code-gold +
229 text-gold records, or downweight the 562 conv-gold, OR per-kind loss weighting
by inverse kind frequency) so code-gold gets enough gradient -- WITHOUT per-kind
decomposition, preserving the cross-kind logit space and text 30.16. This is a
training-data/loss change on the existing 1f-6 arch, cheap (retrain 3 seeds + gate).

QUESTIONS:
1. Is my structural diagnosis correct -- that per-kind decomposition is
fundamentally incompatible with a cross-kind selectivity gate, and that ONLY a
single shared readout can produce a comparable cross-kind logit space? Or is
there a per-kind arch variant that PRESERVES cross-kind comparability (e.g. a
shared final projection head on top of per-kind bodies, or a shared logit-scale
normalization across per-kind readouts) that I'm missing and should try instead?
2. Ranked next experiments to reach a SHIPPABLE state (full live 2.0 robust pass
on the production mixed ring, text AND code). Specifically evaluate:
   (a) shared readout + class-balanced sampling (upweight code/text gold,
       downweight conv gold) -- does this lift code toward 2.0 without breaking
       text 30.16? What's the expected effect size, and the risk of breaking text?
   (b) shared readout + per-kind LOSS weighting (weight each record's margin loss
       by inverse kind frequency) -- same goal, cheaper; is it better/worse than (a)?
   (c) shared readout + hard-negative mining WITHIN code-gold records specifically
       (force code fillers as the hard negatives) -- targeted code lift?
   (d) Stage 2 backbone retrain (joint fine-tune so h carries deeper query-doc
       semantics, not just procedural features) -- Stage 0 says signal IS in h
       (code-only 6.76), so is this necessary, or is the readout/training-data fix
       enough? When would you escalate to Stage 2?
   (e) a per-kind GATE as partial-ship (ship text-doc >= 2.0, lower/none for code +
       conv) -- product-viable, or a blocker?
3. Ship-vs-continue verdict: is 1f-6's text 30.16 (2/3) enough to ship ANYTHING
   partial now, or is the code -4.94 + conv-flat a blocker? Rank the paths by
   expected effort vs probability of a FULL live 2.0 robust pass on text+code.
4. Single highest-leverage next step.
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

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_moe_response.txt"
raw_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_moe_raw.json"

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