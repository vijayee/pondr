"""Feature-decomposition audit of the faultiness head's PASS (train_faulty_head).

Question: WHAT carries the head's 0.814 LOO score? The pre-registered verdict
(passing all three opponents) stands or falls by its composition:

  singles (pooled):      task-cos 0.55 | plan-cos 0.55 | lexical 0.64 | loglen 0.754
  learned ablations (LOO): context-only(task+plan cos): 0.482 (!! worse than
                           the singles -- tiny per-fold overfit on correlated
                           weak features); +loglen: 0.784; +lexical: 0.799;
                           full 8-feature head: 0.814.

Reading: loglen is the workhorse; the head's +0.06 over the best trivial rule
comes from STACKING cheap signals (len + lexical + step-position + file-access
+ weak context), NOT from learning an intent representation. Also: the
round-1 ablation's 0.73 'context grain carries the signal' was contaminated
by action self-similarity (the context text CONTAINED the compared action).

Run: python scripts/icm/audit_faulty_head.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

MAPPED = os.path.join("data", "_scratch", "icm", "capture", "codetrace_mapped.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "faulty_head_audit.json")

_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l > 0.5]
    neg = [s for s, l in zip(scores, labels) if l <= 0.5]
    if not pos or not neg:
        return None
    w = sum(1 for p in pos for n in neg if p > n) + \
        0.5 * sum(1 for p in pos for n in neg if p == n)
    return w / (len(pos) * len(neg))


def tokset(v):
    return {x for x in re.findall(r"[a-z0-9]+", str(v).lower()) if len(x) > 2}


def main() -> int:
    mapped = [json.loads(l) for l in io.open(MAPPED, encoding="utf-8") if l.strip()]
    plan = {r["id"]: r["text"] for r in mapped if r["level"] == "plan"}
    task = {r["id"]: r["text"] for r in mapped if r["level"] == "goal"}
    tn, tt, pt, acts, labels, J, LEN = [], [], [], [], [], [], []
    for r in mapped:
        if r["level"] != "step":
            continue
        h = r["parent"]
        tn.append(h)
        tt.append(task.get(h, ""))
        pt.append(plan.get(h, "") or task.get(h, ""))
        acts.append(r["text"][:300])
        labels.append(1 if r.get("human_label") in ("incorrect", "unuseful") else 0)
        s1, s2 = tokset(tt[-1]), tokset(acts[-1])
        u = s1 | s2
        J.append(len(s1 & s2) / len(u) if u else 0.0)
        LEN.append(len(r["text"]))
    y = np.asarray(labels, np.float32)
    avecs = _emb(acts)

    def cos_vs(texts):
        tv = _emb(texts)
        return np.array([float(np.dot(a, t)) for a, t in zip(avecs, tv)], np.float32)

    cos_task = cos_vs(tt)
    cos_plan = cos_vs(pt)
    Jc = np.asarray(J, np.float32)
    Lc = np.log1p(np.asarray(LEN, np.float32))

    def auc(s):
        a = auroc(s, labels)
        return round(a, 3) if a is not None else None

    singles = {"task-cos": auc(cos_task), "plan-cos": auc(cos_plan),
               "trivial-lexical": auc(Jc), "trivial-loglen": auc(Lc)}

    tasks = sorted(set(tn))

    def loo(X_all):
        sc = np.full(len(y), np.nan, np.float32)
        for h in tasks:
            te = [i for i, t in enumerate(tn) if t == h]
            tr = [i for i, t in enumerate(tn) if t != h]
            if y[tr].sum() < 5 or (y[tr] == 0).sum() < 5:
                sc[te] = 0.5
                continue
            mu, sd = X_all[tr].mean(0), np.where(X_all[tr].std(0) == 0, 1, X_all[tr].std(0))
            w = np.zeros(X_all.shape[1], np.float32)
            b = 0.0
            Xz = (X_all[tr] - mu) / sd
            for _ in range(2500):
                z = Xz @ w + b
                p = 1 / (1 + np.exp(-z))
                d = (p - y[tr]) / len(tr)
                w -= 0.3 * (Xz.T @ d)
                b -= 0.3 * float(d.sum())
            sc[te] = 1 / (1 + np.exp(-((((X_all[te] - mu) / sd) @ w) + b)))
        return auc(sc)

    ablations = {
        "context-only(task+plan cos)": loo(np.stack([cos_task, cos_plan], axis=1)),
        "context+loglen": loo(np.stack([cos_task, cos_plan, Lc], axis=1)),
        "context+len+lexical": loo(np.stack([cos_task, cos_plan, Lc, Jc], axis=1)),
    }
    out = {"singles_pool": singles, "loo_ablations": ablations,
           "full_head_reference": 0.814,
           "reading": ("loglen is the workhorse (0.754 alone); the head's +0.06 "
                       "over the best trivial rule = a cheap-signal STACK, not an "
                       "intent representation; intent-context cosines alone are "
                       "~0.55 and a learned context-only model UNDERPERFORMS the "
                       "singles (tiny-fold overfit)"),
           "verdict_note": "the pre-registered PASS stands AND decomposes to cheap "
                           "features = demo #11 with a twist: the head wins, but "
                           "the win is the cheap-signal ensemble (the router) "
                           "wearing a learned hat"}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())