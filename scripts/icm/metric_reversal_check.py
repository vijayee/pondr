"""Metric pre-check + upgrade -- goal-conditional payoff scoring (committed protocol).

The protocol now applies to the METRIC itself (the critique's demand):
    Q1: can the payoff metric EXPRESS a reversal (same memory, two goals,
        score flips)? -- fed by pairs (X, Y) where X's gold session is
        offered under X's goal and under Y's.
    Q2: if the metric's power is too low to see reversals, upgrade and re-check.

Results summary (2026-10-14; artifacts data/_scratch/icm/metric_reversal_precheck.json
+ metric_judge_upgrade.json):
    * the n-gram metric IS goal-conditional (its reference changes with the
      question) -- scores flip 1.0 -> 0.0 under goal change; NOT goal-blind.
      BUT its power is low: strict 6-gram grading is effectively binary
      (own-ceiling 0.36); 7/11 reversal pairs land at its floor (unmeasurable).
    * UPGRADE: a cross-model, goal-scoped LLM judge (a DIFFERENT relay model
      judges the answerer's output -- no self-judging): own-ceiling 0.85,
      8/11 directional reversals, and it DETECTS non-genuine reversal pairs
      (X's gold session legitimately answering Y -- 2 foreign-high rows,
      corroborated by n-gram).

USAGE:  python scripts/icm/metric_reversal_check.py  (runs ~66 relay calls)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

from payoff_organic_lme import ask_retry, grade, LME50, _embedder  # noqa: E402
from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402

ANSWERER = "deepseek-v4.1-flash:cloud"
JUDGE = "kimi-k3:cloud"  # a different model judges -- no self-judging bias axis

PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
         (16, 17), (18, 19), (20, 21)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=str, default=os.path.join("data", "_scratch", "icm", "organic"))
    args = ap.parse_args()
    policy_ans = DeepSeekAgentPolicy(model=ANSWERER)
    policy_j = DeepSeekAgentPolicy(model=JUDGE)
    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    ids = led.ids("episode")
    texts_by_id = {i: led.get("episode", i)["text"] for i in ids}
    qs = json.load(io.open(LME50, encoding="utf-8"))[: len(ids) // 3]

    gold_ep_of = {}
    for qi in range(len(qs)):
        gold_ep_of[qi] = [i for i in ids if i.startswith(f"lsess-{qi}-")
                          and any(s in i for s in qs[qi]["answer_session_ids"])]

    def judge(goal_question: str, answer: str) -> float:
        q = ("Judge whether the ANSWER satisfies the GOAL below. Consider substance, not wording. "
             'Reply ONLY JSON {"satisfies": true|false, "partial": 0.0-1.0}.\n'
             f"GOAL: {goal_question}\nANSWER: {answer}")
        out, _ = ask_retry(policy_j, q, None)
        try:
            rec = json.loads(out)
        except Exception:
            return 0.0
        v = float(rec.get("partial", 0.0))
        if not rec.get("satisfies"):
            v = min(v, 0.5)
        return max(0.0, min(1.0, v))

    rows = []
    try:
        for X, Y in PAIRS:
            try:
                x_ep = gold_ep_of[X][0]
            except IndexError:
                continue
            block = texts_by_id.get(x_ep)
            ans_own = ask_retry(policy_ans, qs[X]["question"], block)[0]
            ans_foreign = ask_retry(policy_ans, qs[Y]["question"], block)[0]
            rows.append({
                "X": X,
                "judge_own": judge(qs[X]["question"], ans_own),
                "judge_foreign": judge(qs[Y]["question"], ans_foreign),
                "ngram_own": grade(ans_own, qs[X]["answer"]),
                "ngram_foreign": grade(ans_foreign, qs[Y]["answer"]),
            })
            time.sleep(3.0)
    finally:
        led.close()

    genuine = [r for r in rows if r["judge_foreign"] <= r["judge_own"] - 0.15]
    fake_or_tie = [r for r in rows if abs(r["judge_own"] - r["judge_foreign"]) < 0.15]
    anti = [r for r in rows if r["judge_foreign"] > r["judge_own"] + 0.15]
    out = {
        "n_pairs": len(rows),
        "judge": {"own_ceiling_avg": round(sum(r["judge_own"] for r in rows) / max(1, len(rows)), 4),
                  "foreign_avg": round(sum(r["judge_foreign"] for r in rows) / max(1, len(rows)), 4),
                  "genuine_reversals": len(genuine), "non_genuine_or_ties": len(fake_or_tie),
                  "anti_flips_answerer_variance": len(anti)},
        "ngram": {"own_ceiling_avg": round(sum(r["ngram_own"] for r in rows) / max(1, len(rows)), 4)},
        "standing_verdict": ("metric = judge-primary (goal-scoped, cross-model), n-gram secondary; "
                             "reversal-expressibility = MEASURED, not assumed"),
        "rows": rows,
    }
    path = os.path.join("data", "_scratch", "icm", "metric_reversal_check_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())