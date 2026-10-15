"""Reversal-tail verification screen -- the pre-registered post-harvest check.

For each REVERSAL question (spearman(cos, judge) < -0.3 in the offset-100
harvest): rebuild the cosine per candidate from the store, then re-judge the
disagreement pair under a SECOND judge model (deepseek-v4.1-flash; the
original judge was kimi-k3). Verified reversal = the verifier also rates the
judge-favored (low-cosine) candidate MORE helpful than the cosine-top
candidate (margin 0.1). Judge-FP questions fail the screen and are excluded
from the head's training corpus.

Inputs:  data/_scratch/icm/organic500_ckpt_o100.jsonl   (cand_scores, spearman)
         data/_scratch/icm/organic_o100/ledger          (episode texts)
         harness full-500 file                          (question text)
Output:  data/_scratch/icm/reversal_verification_o100.json
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

FULL500 = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness" \
          r"\datasets\longmemeval\longmemeval_interleaved_full.json"
CKPT = os.path.join("data", "_scratch", "icm", "organic500_ckpt_o100.jsonl")
DB = os.path.join("data", "_scratch", "icm", "organic_o100", "ledger")
OUT = os.path.join("data", "_scratch", "icm", "reversal_verification_o100.json")

VERIFIER = "deepseek-v4.1-flash:cloud"   # second judge (cross-family)
MARGIN = 0.1
# NOTE: kimi-k3 retired 2026-10-14 for token cost. For post-2026-10-14 tails the
# original judge is kimi-k2.7-code:cloud (probe-passing: no-answer episode 0.0,
# gold 1.0, same decoy tolerance as kimi-k3); this override must match the
# harvest run's JUDGE.


def judge_help(policy, goal_q, candidate_text) -> float:
    q = ("JUDGE: would recalling this EVIDENCE help an agent answer the GOAL? "
         'Reply ONLY JSON {"helpful": true|false, "partial": 0.0-1.0}.\n'
         f"GOAL: {goal_q}\nEVIDENCE: {candidate_text}")
    backoff = 10.0
    for attempt in range(4):
        try:
            body = {"model": policy.model,
                    "messages": [{"role": "user", "content": q}],
                    "stream": False, "options": {"temperature": 0.2, "num_ctx": 8192}}
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            out = str(json.loads(urllib.request.urlopen(req, timeout=240).read())["message"]["content"]).strip()
            rec = json.loads(out)
            v = float(rec.get("partial", 0.0))
            return max(0.0, min(1.0, v)) if rec.get("helpful") else v * 0.5
        except Exception:
            time.sleep(min(backoff, 60)); backoff *= 1.8
    return 0.0


_model = None


def _embedder(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [np.asarray(v, dtype=np.float32) for v in vecs]


def main() -> int:
    import argparse

    global MARGIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N reversal questions")
    args = ap.parse_args()
    rows = [json.loads(l) for l in io.open(CKPT, encoding="utf-8") if l.strip()]
    questions = {100 + i: rec["question"] for i, rec in
                 enumerate(json.load(io.open(FULL500, encoding="utf-8"))[100:300])}

    led = ICMLedger(DB)
    try:
        texts_by_id = {i: rec["text"] for i in led.ids("episode")
                       if (rec := led.get("episode", i))}
        print(f"[store] {len(texts_by_id)} episodes", flush=True)
    finally:
        led.close()

    verifier = DeepSeekAgentPolicy(model=VERIFIER)

    # cache candidate cosines per reversal question
    cos_cache: dict[int, dict[str, float]] = {}
    verified, rejected, skipped = [], [], []
    import os
    ORIGINAL_JUDGE = os.environ.get("ICM_ORIGINAL_JUDGE", "kimi-k2.7-code:cloud")
    results = {"verifier": VERIFIER, "original_judge": ORIGINAL_JUDGE,
               "margin": MARGIN, "verified": [], "rejected": [], "skipped": []}
    n_seen = 0

    for r in rows:
        qi = r["qi"]
        sp = r.get("spearman")
        if sp is None or sp != sp or sp >= -0.3 or qi not in questions:
            continue
        if args.limit and n_seen >= args.limit:
            break
        n_seen += 1
        cands = r.get("cand_scores") or []
        if len(cands) < 2:
            skipped.append(qi); continue
        q = questions[qi]
        if qi not in cos_cache:
            cos = {}
            mids = [cid for cid, _ in cands if cid in texts_by_id]
            if len(mids) < 2:
                skipped.append(qi); continue
            cvecs = _embedder([texts_by_id[m] for m in mids])
            qvec = _embedder([q])[0]
            for mid, cv in zip(mids, cvecs):
                cos[mid] = round(float(np.dot(cv, qvec)), 4)
            cos_cache[qi] = cos
        else:
            cos = cos_cache[qi]

        cos_ids = sorted(cos, key=lambda m: -cos[m])
        judge_ids = sorted((cid for cid, _ in cands), key=lambda m: -dict(cands).get(m, 0.0))
        cos_top, judge_top = cos_ids[0], judge_ids[0]
        if cos_top == judge_top:
            # the cosine TOP is also judge TOP: no disagreement pair (judge noise
            # on non-top candidates) -- structurally not a reversal
            skipped.append(qi); continue
        va = judge_help(verifier, q, texts_by_id.get(cos_top, ""))
        vb = judge_help(verifier, q, texts_by_id.get(judge_top, ""))
        rec_v = {"qi": qi, "spearman": sp, "cand": {"cos_top": cos_top, "vb": vb},
                 "judge_favored": {"id": judge_top, "v": va}}
        rec_v["cos"] = {"cos_top": cos[cos_top], "judge_favored": cos.get(judge_top, 0.0)}
        rec_v["orig_judge"] = {"cos_top": dict(cands).get(cos_top),
                               "judge_favored": dict(cands).get(judge_top)}
        time.sleep(2.0)
        if vb > va + MARGIN:
            verified.append(qi); results["verified"].append(rec_v)
        else:
            rejected.append(qi); results["rejected"].append(rec_v)
        print(f"[q {qi}] sp={sp} v(cos_top)={va:.2f} v(judge_fav)={vb:.2f} -> "
              f"{'VERIFIED' if vb > va + MARGIN else 'rejected'}", flush=True)
        # incremental dump so a crash loses nothing
        with io.open(OUT, "w", encoding="utf-8") as f:
            f.write(json.dumps(results | {"verified_qids": verified, "rejected_qids": rejected,
                                          "skipped_qids": skipped}, indent=2))

    print(f"\n=== VERIFICATION SCREEN ===", flush=True)
    print(f"verified:   {len(verified)}  {verified}", flush=True)
    print(f"rejected:   {len(rejected)}", flush=True)
    print(f"skipped:    {len(skipped)}  {skipped}", flush=True)
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())