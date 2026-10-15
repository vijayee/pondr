"""E3-head first organic verdict -- LOO training on the verified reversal tail.

THE PRE-REGISTERED PROTOCOL (offset-100 verified tail; 25 questions, 3 cands each):

  Labels:   kimi judge >= 0.5 (the cross-model payoff judge). The tail is
            exactly where labels are NOT cosine-recoverable (verified by the
            2-judge screen).

  Features: all computable at retrieval time, none encode the label:
      f1 cos                -- question-candidate cosine (the dense leg)
      f2 rel_time_overlap   -- Jaccard overlap of relative-time/ordinal tokens
                               between question and candidate
      f3 log_len            -- log candidate text length
  Head:    logistic regression on [f1,f2,f3] -> P(helpful); rank by P.
  Eval:    leave-one-out over the 25 questions. WIN = judge-favored candidate
           ranked strictly first on the held-out question.
           Cosine's own win rate is 0/25 BY CONSTRUCTION of the verified set
           (cos-top carries the lowest judge score in every verified reversal).

  The fixed-point fallback (the 8th): if the head does NOT beat the best
  TRIVIAL single-feature rule (rank by any one feature alone), the win is
  retired as "signal compact-rule decidable again" and the strong thesis
  stays gated.

Run:  python scripts/icm/train_head_loo.py   (~seconds, local)
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CORPUS = os.path.join("data", "_scratch", "icm", "icm_head_corpus_o100.json")
OUT = os.path.join("data", "_scratch", "icm", "e3_head_loo_results.json")

TIME_TOKENS = {"yesterday", "tomorrow", "today", "last", "next", "before",
               "after", "ago", "earlier", "later", "first", "second", "then",
               "week", "month", "year", "monday", "tuesday", "wednesday",
               "thursday", "friday", "saturday", "sunday", "spring", "summer",
               "fall", "autumn", "winter", "january", "february", "march",
               "april", "may", "june", "july", "august", "september",
               "october", "november", "december"}


def tokset(text: str) -> set[str]:
    return {t.strip(".,!?;:'\"()") .lower() for t in text.split()}


def features(q: str, cand: dict) -> list[float]:
    q_toks = tokset(q)
    c_toks = tokset(cand["text"])
    overlap = (q_toks & c_toks) & TIME_TOKENS
    union = (q_toks | c_toks) & TIME_TOKENS
    j = len(overlap) / max(1, len(union))
    return [cand["cos"], j, math.log1p(len(cand["text"]))]


def fit_logreg(X, y, iters=3000, lr=0.5):
    w = [0.0] * 3
    b = 0.0
    n = len(X)
    for _ in range(iters):
        gw = [0.0] * 3
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(w[k] * xi[k] for k in range(3))
            p = 1.0 / (1.0 + math.exp(-z))
            d = p - yi
            gb += d
            for k in range(3):
                gw[k] += d * xi[k]
        for k in range(3):
            gw[k] /= n
            w[k] -= lr * gw[k]
        b -= lr * gb / n
    return w, b


def predict(w, b, xi):
    z = b + sum(w[k] * xi[k] for k in range(3))
    return 1.0 / (1.0 + math.exp(-z))


def main() -> int:
    corpus = json.load(io.open(CORPUS, encoding="utf-8"))
    qs = []
    for q in corpus:
        fav = q["judge_favored_id"]
        cands = q["candidates"]
        y = [1 if c["kimi_judge"] >= 0.5 else 0 for c in cands]
        X = [features(q["question"], c) for c in cands]
        qs.append({"qi": q["qi"], "fav": fav, "cands": cands, "X": X, "y": y,
                   "label_ids": [c["id"] for c, yy in zip(cands, y) if yy]})

    def wins(rule) -> int:
        """rule(X_i) -> scores; win if judge-favored ranked strictly first."""
        w_ = 0
        for item in qs:
            scores = rule(item["X"])
            top = max(range(len(scores)), key=lambda i: scores[i])
            if item["label_ids"] and item["cands"][top]["id"] == item["label_ids"][0]:
                w_ += 1
        return w_

    n = len(qs)

    # --- LOO logistic head ---
    head_wins = 0
    log = []
    for h in range(n):
        train = [qs[i] for i in range(n) if i != h]
        Xtr = [x for t in train for x in t["X"]]
        ytr = [v for t in train for v in t["y"]]
        w, b = fit_logreg(Xtr, ytr)
        scores = [predict(w, b, x) for x in qs[h]["X"]]
        top = max(range(len(scores)), key=lambda i: scores[i])
        won = bool(qs[h]["label_ids"]) and qs[h]["cands"][top]["id"] == qs[h]["label_ids"][0]
        head_wins += won
        log.append({"qi": qs[h]["qi"], "won": won, "scores": [round(s, 3) for s in scores],
                    "fav": qs[h]["fav"], "top": qs[h]["cands"][top]["id"]})
    print(f"LOO head (cos+time+len):  {head_wins}/{n}")

    # --- trivial single-feature rules (the fixed-point fallback) ---
    cos_w = wins(lambda X: [x[0] for x in X])
    time_w = wins(lambda X: [x[1] for x in X])
    len_w = wins(lambda X: [x[2] for x in X])
    print(f"trivial: cos={cos_w}/{n}  time={time_w}/{n}  len={len_w}/{n}")
    best_trivial = max(cos_w, time_w, len_w)

    verdict = ("HEAD WINS -- beyond-cosine learned ranking earns it (strong thesis "
               "armed, first organic verdict)"
               if head_wins > best_trivial and head_wins > 0.5 * n else
               "HEAD TIES a trivial rule -- compact-rule decidable again (fixed-point, "
               "retired)" if head_wins <= best_trivial else
               "head beats cosine but under 50% -- inconclusive")
    print("VERDICT:", verdict)

    out = {"n_questions": n, "loo_head_wins": head_wins, "cosine_wins": cos_w,
           "trivial_time_wins": time_w, "trivial_len_wins": len_w,
           "best_trivial": best_trivial, "verdict": verdict, "loo_log": log}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())