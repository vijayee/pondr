"""PHASE 2 REMEDIATION BATTERY, MODE 1 -- the small-n failure of the learned
head over context features (audit_faulty_head's decomposed 0.482 context-only
arm, WORSE than its own singles; textbook per-fold overfit) + a re-audit of
whether any remediated head changes the demo-#11 composition.

MODE-1 MEASURED STATE (186cb7a / faulty_head_audit.json):
  train_faulty_head.py (11,963 steps / 259 tasks / 146 faulty, LEAVE-ONE-TASK-
  OUT pooled): head 0.814 = PASS, but the audit showed the win is a cheap-
  signal stack (loglen 0.754 workhorse) and the LEARNED context-only model
  scored 0.482 -- worse than its own singles (0.593 best single cosine).
  The lens's GD had NO regularization and NO early stop: 2500 fixed iters,
  one L2=0 setting.

THE PRE-REGISTRATION (fixed BEFORE fitting):

  Universe:   the same mapped slice, human-verified labels (unchanged).
  Split:      LEAVE-ONE-TASK-OUT pooled scoring (the real split, unchanged;
              no leakage of any held-out task's rows into any fit).
  Models:     logistic GD on z-normalized-by-train-fold features, grid over
              ridge lambda in {0.01, 0.1, 1, 10} (applied to normalized
              weights each iter) x iters in {100, 500, 2500} (early stop =
              the iters axis), for TWO feature sets:
                ctx-only  = [cos(task), cos(plan), cos(stage)]   (mode-1 arm)
                full      = the original 8 features (the PASS arm)
  Opponents (must ALL be beaten on the same pooled scores, or the head
              loses): o1 best-single-cosine, o2 lexical jaccard,
              o3 log length -- AND the original unregularized heads
              (0.482 ctx-only / 0.814 stack) for direction.
  Readout:    pooled AUROC per cell, direction ALWAYS reported.
  Survival:   a remediated head earns the re-audit only if SOME cell beats
              ALL three opponents on pooled scores (beats best op by > 0.02;
              within 0.02 = TIE per the original rule). Then the audit is
              re-run on the winning cell (feature contributions) before any
              composition claim.
  If no cell survives: mode 1's overfit diagnosis stands REFINED (regularize
              + early stop did not turn context into signal) and verdict
              stays OPEN to data-cleanup / other ML / goal-drift correction.

Run:  python scripts/icm/battery_smalln_remediation.py   (local, ~20 min)
"""

from __future__ import annotations

import collections
import io
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from train_faulty_head import auroc, tokset  # noqa: E402  (same helpers)

MANIFEST = os.path.join("data", "_scratch", "codetrace", "bench_manifest.verified.jsonl")
MAPPED = os.path.join("data", "_scratch", "icm", "capture", "codetrace_mapped.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "battery_smalln_results.json")
FCACHE = os.path.join("data", "_scratch", "icm", "battery_smalln_feats.npz")

LAMBDAS = (0.01, 0.1, 1.0, 10.0)
ITERSETS = (100, 500, 2500)
LR = 0.3

_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def fit_ridge(Xz, y, lam, iters, lr=LR):
    """Logistic GD with ridge on normalized weights (per-fold shrinkage).

    Numerically stable (expit) and divergence-checked: a diverged fold gets
    flagged, never scored as a silent 0.
    """
    from scipy.special import expit

    w = np.zeros(Xz.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(iters):
        z = Xz @ w + b
        p = expit(z)
        d = (p - y) / len(y)
        w -= lr * (Xz.T @ d + lam * w)
        b -= lr * float(d.sum())
        if not np.isfinite(w).all():
            return None
    return w, b


def main() -> int:
    mapped = [json.loads(l) for l in io.open(MAPPED, encoding="utf-8") if l.strip()]
    hid_to_plan = {r["id"]: r["text"] for r in mapped if r["level"] == "plan"}
    hid_to_task = {r["id"]: r["text"] for r in mapped if r["level"] == "goal"}
    manifest = {}
    for r in (json.loads(l) for l in io.open(MANIFEST, encoding="utf-8") if l.strip()):
        manifest[r["task_name"]] = r

    def stages_of(hid):
        row = manifest.get(hid[3:])
        if row is None:
            for k in manifest:
                if hid.startswith("ct-" + k):
                    return manifest[k]
        return row or {}

    tnames, labels = [], []
    cos_task, cos_plan, cos_stage = [], [], []
    extra = {"jaccard": [], "loglen": [], "idx": [], "file": [], "logn": []}
    hid_step_total = collections.defaultdict(int)
    for r in mapped:
        if r["level"] == "step":
            hid_step_total[r["parent"]] += 1
    for r in mapped:
        if r["level"] != "step":
            continue
        hid = r["parent"]
        row = stages_of(hid)
        sc = row.get("step_count") or hid_step_total.get(hid, 1)
        batch_txt = ""
        for st in row.get("stages") or []:
            if st["start_step_id"] <= r.get("step_id", 0) <= st["end_step_id"]:
                batch_txt = f"stage {st['stage_id']} (steps {st['start_step_id']}-{st['end_step_id']})"
        tnames.append(hid)
        labels.append(1 if r.get("human_label") in ("incorrect", "unuseful") else 0)
        cos_task.append(r["text"][:300])
        cos_plan.append(hid_to_plan.get(hid, "") or hid_to_task.get(hid, ""))
        cos_stage.append(batch_txt + " of " + hid_to_task.get(hid, ""))
        tt, at = tokset(hid_to_task.get(hid, "")), tokset(r["text"])
        u = tt | at
        extra["jaccard"].append(len(tt & at) / len(u) if u else 0.0)
        extra["loglen"].append(math.log1p(len(r["text"])))
        extra["idx"].append(r.get("step_id", 0) / max(1, sc))
        extra["file"].append(1.0 if "/" in r["text"] else 0.0)
        extra["logn"].append(math.log1p(sc))
    y = np.asarray(labels, dtype=np.float64)
    print(f"[data] steps={len(y)} faulty={int(y.sum())} tasks={len(set(tnames))}",
          flush=True)

    steps = [r for r in mapped if r["level"] == "step"]
    akey = f"n={len(steps)} first={steps[0]['text'][:60]} last={steps[-1]['text'][:60]}"
    if os.path.exists(FCACHE):
        z = np.load(FCACHE, allow_pickle=True)
        if str(z["key"]) == akey:
            cos_task, cos_stage = z["ct"].astype(float), z["cs"].astype(float)
            cos_plan = z["cp"].astype(float)
            print("[cache] features loaded", flush=True)
    else:
        print("[embed] encoding context features (~13k texts)", flush=True)
        avecs = _emb([r["text"][:300] for r in mapped if r["level"] == "step"])
        tvecs = _emb(cos_task)
        pvecs = _emb(cos_plan)
        svecs = _emb(cos_stage)
        cos_task = [float(np.dot(a, t)) for a, t in zip(avecs, tvecs)]
        cos_plan = [float(np.dot(a, p)) for a, p in zip(avecs, pvecs)]
        cos_stage = [float(np.dot(a, s)) for a, s in zip(avecs, svecs)]
        os.makedirs(os.path.dirname(FCACHE), exist_ok=True)
        np.savez(FCACHE, key=akey, ct=np.asarray(cos_task), cp=np.asarray(cos_plan),
                 cs=np.asarray(cos_stage))

    cos_task = np.asarray(cos_task, dtype=np.float64)
    cos_plan = np.asarray(cos_plan, dtype=np.float64)
    cos_stage = np.asarray(cos_stage, dtype=np.float64)
    Xf = {
        "ctx_only": np.stack([cos_task, cos_plan, cos_stage], axis=1),
        "full": np.stack([
            cos_task, cos_plan, cos_stage,
            np.asarray(extra["loglen"]), np.asarray(extra["jaccard"]),
            np.asarray(extra["idx"]), np.asarray(extra["file"]),
            np.asarray(extra["logn"]),
        ], axis=1),
    }
    opponents = {
        "best-single-cosine": np.max(np.stack([cos_task, cos_plan, cos_stage]), axis=0),
        "lexical(o2)": np.asarray(extra["jaccard"]),
        "loglen(o3)": np.asarray(extra["loglen"]),
    }
    opp_scores = {k: auroc(v.tolist(), y.tolist()) for k, v in opponents.items()}
    print(f"[opponents] {opp_scores}", flush=True)
    best_op = max(v for v in opp_scores.values() if v is not None)

    tasks = sorted(set(tnames))
    task_rows = {}
    for t in tasks:
        task_rows[t] = [i for i, v in enumerate(tnames) if v == t]

    from scipy.special import expit

    def run_cell(fs_name, lam, iters):
        scores = np.zeros(len(y))
        skipped = diverged = 0
        X_all = Xf[fs_name]
        for hid, te in task_rows.items():
            tr = [i for i, t in enumerate(tnames) if t != hid]
            yt = y[tr]
            if yt.sum() < 5 or (1 - yt).sum() < 5:
                scores[te] = 0.5
                skipped += 1
                continue
            mu, sd = X_all[tr].mean(axis=0), X_all[tr].std(axis=0)
            sd = np.where(sd == 0, 1.0, sd)
            fit = fit_ridge((X_all[tr] - mu) / sd, yt, lam, iters)
            if fit is None:
                scores[te] = np.nan
                diverged += 1
                continue
            w, b = fit
            scores[te] = expit(np.asarray(((X_all[te] - mu) / sd) @ w + b))
        m = np.isfinite(scores)
        a = auroc(scores[m].tolist(), y[m].tolist())
        return a, skipped, diverged

    out = {"grid": [], "opponents": opp_scores, "original_heads":
           {"ctx_only": 0.482, "full": 0.814}}
    best_cell = None
    for fs_name in ("ctx_only", "full"):
        for lam in LAMBDAS:
            for iters in ITERSETS:
                a, sk, dv = run_cell(fs_name, lam, iters)
                cell = {"features": fs_name, "lambda": lam, "iters": iters,
                        "pooled_auc": round(a, 4) if a is not None else None,
                        "folds_skipped": sk, "folds_diverged": dv}
                out["grid"].append(cell)
                if a is not None:
                    if best_cell is None or a > best_cell[0]:
                        best_cell = (a, cell)
                print(f"[cell] {fs_name} lam={lam} iters={iters} "
                      f"auc={a if a is None else round(a, 4)}", flush=True)

    a = best_cell[0] if best_cell else None
    beats_all = a is not None and a > best_op + 0.02
    tied = a is not None and abs(a - best_op) <= 0.02
    out["best_cell"] = best_cell[1] if best_cell else None
    out["best_op"] = round(best_op, 4)
    out["verdict"] = ("REMEDIATED-PASS -- re-audit composition next" if beats_all
                      else "TIE (original rule)" if tied
                      else "NO REMEDIATED SURVIVOR -- mode-1 overfit diagnosis "
                           "stands refined; verdict stays open")
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("best_cell", "verdict")}, indent=1),
          flush=True)
    print(f"[done] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())