import json, time, urllib.request, urllib.error

URL = "http://127.0.0.1:11434/api/chat"

system_msg = (
    "You are a senior ML research engineer advising on a state-trajectory ring "
    "memory (STRM) project. Be concrete, rank actions by expected leverage, give "
    "a crisp ship-vs-continue verdict, and state the single highest-leverage next "
    "step. Do not hedge with disclaimers."
)

user_msg = r"""STRM (State-Trajectory Ring Memory) follow-up consult #3. You advised this project
twice; your #6 recommendation (per-kind bodies + shared head) was the architectural
fix and it FAILED maximally. The readout-only path is now exhausted. I need the
next decision.

RECAP OF YOUR LADDER (consult #1/#2):
  1. hard-neg mining           -> done (#3)
  2. no-replacement balanced sampling -> done (#4 sqrt-freq variant #5)
  3. per-kind bodies + shared head (architectural fallback) -> done (#6, THIS RUN)
Variance-tamers: (a) sqrt-inverse-freq (done, #5), (b) no-replacement (done, #4),
  (c) more seeds (done, 6-seed #5), (d) seed ensemble (pending).
You predicted per-kind bodies + shared head: ">=5/6 pass-rate. The risk of
incompatible body representations is low -- the shared head is a linear+MLP map
that can align distinct 128-dim subspaces, and the cross-kind loss signal will
force alignment."

WHAT I RAN. per_kind_bodies: N INDEPENDENT 6144->128 ReLU bodies (one per kind:
conv=0/text=1/code=2) feeding ONE SHARED 128->384 head, PER-SLOT routing (each
slot -> its own kind body -> shared head, NOT MoE by-gold). Trained jointly from
scratch with the #5 sqrt-inverse-freq no-replacement sampler (AdamW-clean, per-step
loss scale 1.0; weighted shuffle changes record ORDER not loss scale). margin loss
m=2.5 hard-negative, --select-ckpt final, cosine 120 epochs, dropout 0.3. The same
frozen 19.5M ReferenceSSM backbone (d_model=384, 4 layers, d_state=16, routing-
trained), flat_last 6144 = 16 channels x 384.

RESULT (per_kind_bodies seed 0 -- a TOTAL collapse, NOT a flat seed):
  TRAIN: epoch 0 top3=0.774 GO -> epoch 1 top3=0.265 no-go -> frozen top3=0.271
    from epoch 1 through 77+, train_loss pinned at 2.4919 (the margin ceiling),
    r_pos decayed 0.454 -> 0.000 (gold slot indistinguishable from fillers).
    Deterministic-looking: frozen from epoch 1 onward. This is the EXACT signature
    of #2 (per-kind LOSS weighting collapse: top3~0.27, r_pos->0.03->0) -- but #2
    used per-kind LOSS weighting and I am NOT (I use the AdamW-clean sampler only).
    So the collapse is ARCHITECTURAL, not loss-scale.
  LIVE GATE (seed 0, held-out): full=+0.000 conv=+0.000 retrieved=-0.177
    ret_text=-0.080 ret_code=-0.177. ALL buckets FAIL (0/1). Maximal -- the head
    outputs near-constant on serve.

THE DECISIVE DETAIL. Seed 0 was #5's BEST seed. #5 (shared body, n_doc_kinds=0,
NO per-kind decomposition, sqrt-freq sampler) 6-seed result:
  s0: ret_text +23.20 PASS / ret_code +17.25 PASS  (best of 6)
  s1: ret_text  +9.76 PASS / ret_code  +3.69 PASS
  s2/s3/s4: flat FAIL
  s5: ret_text +29.61 PASS / ret_code +1.44 FAIL (code just under)
  -> 2/6 both-buckets PASS (s0, s1), both reproduce DETERMINISTICALLY.
SAME seed 0, SAME sampler, SAME backbone/data/loss -- the ONLY change is the
readout arch (undivided shared body -> per-kind bodies + shared head). +23.20/+17.25
-> 0.000/0.000. The architecture change CAUSED the collapse; seed 0 is not a bad
seed.

THE PATTERN -- ALL 3 PER-KIND DECOMPOSITIONS NOW FAIL:
  - Stage 1: shared 6144->128 body + per-kind HEADS (kind_heads.{k} 128->384) +
    kind_head_wd 0.05: LIVE GATE FAIL. ret_code lifted -4.94->+0.825 BUT ret_text
    REGRESSED 30.16->0.558 -> net worse. Diagnosis: shared body conv-majority
    dominated + kind_head_wd over-regularized minority heads.
  - Stage 1 redesign: MoE per_kind_full (N INDEPENDENT FULL readouts 6144->384,
    by-gold routing so margin loss is WITHIN one readout's logit space): FAIL 0/3,
    all-turns ceiling 0.000. Diagnosis: per-kind HEADS -> independent scales/biases
    -> ill-defined cross-head logits (the gate is fundamentally CROSS-KIND).
  - Stage 2 #6: per_kind_bodies (N bodies + ONE shared head, per-slot routing):
    maximal collapse 0.000 on the best seed (above). Diagnosis: the "incompatible
    body representations" risk you called LOW is the ACTUAL failure mode. Three
    independently-trained 6144->128 bodies produce 128-dim reps in arbitrary
    relative orientations; the shared 128->384 head, under AdamW + margin loss with
    no coordination mechanism, cannot align them -> collapses to near-constant
    (top3~1/slots, r_pos->0). The cross-kind loss signal does NOT force alignment;
    it forces the head to give up.
The ONLY arch that got any both-buckets PASS is the UNDIVIDED shared body (#5,
n_doc_kinds=0): one 6144->128 body -> one 128->384 head, every slot through the
SAME params. 2/6. Every decomposition of the body OR the head collapses or mis-ranks.

STAGE 0 (still stands): a shared readout trained on CODE-GOLD-ONLY data clears
held-out z_logit median 6.76 (2/3). The code signal IS in h. BUT it only surfaces
when the readout is trained on code-ONLY data -- any cross-kind MIXING (conv
majority 60% in the full data) collapses the code direction. The readout can find
the code signal in isolation; it cannot hold it alongside the conv/text signals in
one shared param set without being pulled flat (shared body) or producing
incompatible subspaces (per-kind bodies).

CONSTRAINTS (unchanged):
- Frozen routing-trained backbone; flat_last 6144 encodes PROCEDURAL features not
  deep query-doc semantics. Stage 0 says signal IS in h but only kind-isolated.
- AdamW optimizer; class-balanced SAMPLING clean, class-balanced LOSS weighting
  collapses. The weighted shuffle (replacement=False) is AdamW-clean.
- The gate is held-out live z_logit selectivity gap median >= 2.0; PASS needs
  retrieved_text >= 2.0 in >= 4/6 AND retrieved_code >= 2.0 in >= 4/6.
- Do not break existing functionality; nothing integrates until it serves its
  purpose. I will NOT ship anything that does not clear the 6-seed gate.

QUESTIONS:
1. DIAGNOSIS. Is the root cause that per-kind decomposition is fundamentally
   incompatible with a routing-trained PROCEDURAL backbone whose h is kind-isolated
   (signal exists per-kind but a single shared param set cannot hold all kinds)?
   In other words: is the failure NOT a readout-arch problem but a REPRESENTATION
   problem -- the backbone never learned a COMMON cross-kind relevance subspace, so
   no readout (shared OR decomposed) can produce a coherent cross-kind logit space,
   and #5's 2/6 is the ceiling of what the frozen h supports? Or do you still read
   this as a readout problem (per_kind_bodies just needs a coordination mechanism)?

2. COORDINATION MECHANISMS for per_kind_bodies (if you still think the arch is
   salvageable): (a) warm-start the N bodies from the SAME shared #5 body (so they
   start aligned, then diverge); (b) freeze bodies then train head, then unfreeze
   (staged); (c) an alignment/orthogonality regularizer on the bodies; (d) a shared
   first layer + per-kind second layer (partial decomposition). Will ANY of these
   escape the incompatible-subspace collapse, or is per-kind decomposition dead for
   this backbone? Rank or refute each.

3. NEXT-HIGHEST-LEVERAGE FIX. The data/loss ladder (#1-#5) failed 6-seed; the arch
   decompositions (heads/MoE/bodies) failed. Remaining options on the table:
   (A) 2-seed logit-ensemble of #5 s0+s1 -- the ONLY reproducible both-buckets PASS
       (deterministic, +23.20/+17.25 and +9.76/+3.69). You called 2-seed ensemble
       a "band-aid, low rank" in consult #2. Does the total collapse of every
       architectural alternative change its rank? Is 2 deterministic seeds enough
       to generalize to live traffic, or too thin? Would 3-4 seeds of #5 ensembled
       be meaningfully more robust (note s2-s5 do NOT pass both buckets)?
   (B) Stage 2 backbone retrain -- joint fine-tune so h carries a COMMON cross-kind
       query-doc relevance subspace (not just procedural routing features). You
       ranked this "distant third" in consult #2. Given EVERY readout arch now fails
       on the frozen h, does this become the PRIMARY fix? Concretely: what objective
       (add L_relevance on the cross-kind doc ring to the routing L? replace it?
       freeze-then-unfreeze?), what data (the 934-record onyx doc ring + the 52
       private onyx sessions), and what's the expected readout pass-rate after?
   (C) Something I'm missing -- a different readout arch (residual/gating body,
       mixture-of-experts with a shared gating + shared final projection, a
       per-kind NORMALIZATION layer on a shared body instead of per-kind params),
       more data, a seed-selection rule at serve, a different loss.

4. Single highest-leverage next step, and ship-vs-continue. If the backbone retrain
   is now the primary fix, say so directly and give the minimal objective+data spec.
   If the 2-seed ensemble is good enough to ship (gated on a live-traffic
   smoke), say so. If something cheaper is more leveraged, rank it above.
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

out_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_stage2_reconsult_perkindbodies_collapse_response.txt"
raw_path = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_deepseek_stage2_reconsult_perkindbodies_collapse_raw.json"

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