"""Re-judge the UNMEASURABLE rows (judge-tie spearmans) with a second judge.

Why: kimi-k2.7-code labels are coarse — o300 had 66/200 rows with constant
judge scores (spearman NaN = unmeasurable), o100 had 16. Those rows are
tail-suppressed (some may hold undiscovered reversals). This script re-judges
each tied row's candidates with deepseek-v4.1-flash:cloud (probe-passing judge;
scores CANDIDATE EVIDENCE, not the answerer's outputs, so the same-family note
does not apply the self-judging bias it would to scoring answers; second-judge
verification of any new reversals still uses cross-family agreement) and
recomputes the labels-recovery spearman.

Output: data/_scratch/icm/rejudge_ties_o{OFFSET}.jsonl + a summary; new
reversals (sp < -0.3) become verification-screen candidates.

Run:  python scripts/icm/rejudge_ties.py --offset 100 [--offset 300 ...]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

FULL500 = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness" \
          r"\datasets\longmemeval\longmemeval_interleaved_full.json"
HARNESS_DIR = os.path.join("data", "_scratch", "icm")
JUDGE = "deepseek-v4.1-flash:cloud"
_model = None


def _judge(q: str, text: str) -> float:
    p = ("JUDGE: would recalling this EVIDENCE help an agent answer the GOAL? "
         'Reply ONLY JSON {"helpful": true|false, "partial": 0.0-1.0}.\n'
         f"GOAL: {q}\nEVIDENCE: {text}")
    body = {"model": JUDGE, "messages": [{"role": "user", "content": p}],
            "stream": False, "options": {"temperature": 0.2, "num_ctx": 8192}}
    backoff = 10.0
    for attempt in range(4):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            out = str(json.loads(urllib.request.urlopen(req, timeout=240).read())
                      ["message"]["content"]).strip()
            rec = json.loads(out)
            v = float(rec.get("partial", 0.0))
            return max(0.0, min(1.0, v)) if rec.get("helpful") else v * 0.5
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
            if attempt >= 2:
                raise
            time.sleep(2.0)
        except Exception:
            if attempt >= 2:
                raise
            time.sleep(2.0)
    raise RuntimeError("judge failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", type=str, default="100,300",
                    help="comma-separated segment offsets whose ties to re-judge")
    args = ap.parse_args()

    data = json.load(io.open(FULL500, encoding="utf-8"))
    questions = {i: rec["question"] for i, rec in enumerate(data)}

    for off_s in args.offsets.split(","):
        off = int(off_s)
        ckpt_p = os.path.join(HARNESS_DIR, f"organic500_ckpt_o{off}.jsonl")
        out_p = os.path.join(HARNESS_DIR, f"rejudge_ties_o{off}.jsonl")
        ledger_p = os.path.join(HARNESS_DIR, f"organic_o{off}", "ledger")

        ck = {}
        for l in io.open(ckpt_p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                ck[r["qi"]] = r
        # ties: spearman absent or NaN
        ties = [r for r in ck.values()
                if r.get("spearman") is None or r.get("spearman") != r.get("spearman")]
        print(f"[o{off}] {len(ties)} tie rows of {len(ck)}", flush=True)

        led = ICMLedger(ledger_p)
        try:
            texts = {i: rec["text"] for i in led.ids("episode")
                     if (rec := led.get("episode", i))}
        finally:
            led.close()

        # resume support
        done_qi = set()
        if os.path.exists(out_p):
            for l in io.open(out_p, encoding="utf-8"):
                if l.strip():
                    done_qi.add(json.loads(l)["qi"])

        n_rev = 0
        with io.open(out_p, "a", encoding="utf-8") as f:
            for r in ties:
                qi = r["qi"]
                if qi in done_qi:
                    continue
                cands = r.get("cand_scores") or []
                scored, cos, scores = [], {}, []
                for cid, _old in cands:
                    if cid in texts:
                        try:
                            j = _judge(questions[qi], texts[cid])
                        except Exception:
                            j = None
                        if j is not None:
                            scored.append(cid)
                            scores.append(j)
                            cos[cid] = j
                        time.sleep(2.0)
                # need cosine values: embedder
                spear = None
                if len(scores) >= 3:
                    from sentence_transformers import SentenceTransformer
                    global _model
                    if _model is None:
                        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                    cvecs = _model.encode([texts[c] for c in scored], batch_size=32,
                                          show_progress_bar=False,
                                          convert_to_numpy=True,
                                          normalize_embeddings=True)
                    qvec = _model.encode([questions[qi]], show_progress_bar=False,
                                         convert_to_numpy=True,
                                         normalize_embeddings=True)[0]
                    cosv = [float(np.dot(c, qvec)) for c in cvecs]
                    rh = spearmanr(scores, cosv)
                    spear = round(float(rh.statistic), 3) if hasattr(rh, "statistic") else None
                row = {"qi": qi, "judge": JUDGE, "rejudged_from": "tie",
                       "cand_scores": [[c, cos[c]] for c in scored],
                       "spearman": spear}
                f.write(json.dumps(row) + "\n")
                tag = f" REVERSAL sp={spear}" if spear is not None and spear < -0.3 else ""
                if spear is not None and spear < -0.3:
                    n_rev += 1
                print(f"[o{off} q{qi}] {len(scored)} cands sp={spear}{tag}", flush=True)
        print(f"[o{off}] done; new reversals flagged: {n_rev}", flush=True)
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())