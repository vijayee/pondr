"""PHASE 2 E2 -- THE REMAINING LEVERS on the intent-bearing corpus.

WHY THIS EXISTS: the addendum-11 start-over ran only part of the remediation
battery on the intent data (identity-style init, early-stop checkpoints, and
two learning rates). The user's question -- "did we try all the ML levers on
the data with the signal?" -- is answered NO. This run closes the gap.

WHAT WAS ALREADY RUN ON THIS CORPUS (addendum 11, don't re-run):
  residual hinge, margin 0.1, L2 1e-4, lr {1.5, 0.15}, 600 ep, ckpt 25,
  8-fold task-disjoint CV. Result: no survivor; pooled == the init bundle
  (0.5901); val declined monotonically from the first checkpoint; the
  untrained bundle (0.5901) < embed(prompt) (0.6114).

THE G-CONFIG PRE-REGISTRATION (fixed BEFORE fitting; same corpus, same
8-fold task-disjoint CV, same opponents, same survival bar: pooled lens AUC
must beat ALL FOUR opponents by > 0.02):

  G1  MARGIN SWEEP at the engaging end: margin {0.0, 0.01, 0.05}
      (0.1 was the only value ever run; margin 0 = every inversion of a
      verified-faulty step violates = the richest signal this objective can
      have; at margin 0.1 the init census showed 96.6 percent of pairs
      already violating, so the sweep asks whether a SMALLER margin changes
      the learned direction rather than the amount of signal).
  G3  WEIGHT-DECAY SWEEP: L2 {1e-3, 1e-2} (only 1e-4 ever run).
  G4  LOW-RANK DELTA (rank 16; W_delta = U V^T, U sigma-0.03, V sigma-0.05;
      exact chain gU = dW_full @ V, gV = U^T @ dW_full; L2 on W_eff).
      NEVER RUN ANYWHERE -- the battery's item 1 named low-rank but only the
      full-rank delta was implemented.
  G2  INFONCE IN-BATCH NEGATIVES: multi-positive softmax, positives = the
      task's own non-faulty steps, candidates = own steps + K=12 steps
      sampled per task per epoch from OTHER train tasks (the real
      distribution; a sampled row colliding with an own positive is treated
      as a negative -- standard in-batch practice, stated), temperature
      {0.1, 0.3}. Residual form (raw term = bundle cosine). No early-exit
      rule for InfoNCE (stated; runs the full 600 ep).
      COVERAGE NOTE (recorded, verified 2026-10-16): with positives = own
      non-faulty steps, InfoNCE also trains on the 208 no-faulty tasks (all
      steps positive; the batch still pushes sampled negatives away), i.e.
      it saw ALL 259 tasks -- MORE data than the hinge family's 51 contrast
      tasks -- and still landed on the init bundle. The empty-task guard
      never fires (0 such tasks in this corpus).

  Hypotheses (NOT verdicts): the hinge-family levers are expected to
  reproduce the geometry-declines-from-init signature if that is a property
  of the objective rather than of margin/L2; InfoNCE and low-rank are
  genuinely open questions on this corpus. Whatever lands, direction is
  reported and the verdict stays open per the standing no-dismissal rule.

Run:  python scripts/icm/phase2_e2_intent_battery.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import phase2_e2_intent_lens as L  # noqa: E402  (audited builder + CV)

OUT = os.path.join("data", "_scratch", "icm", "phase2_e2_intent_battery_results.json")
DIM = 128


def train_lowrank(cfg, tr_tasks, val_tasks, D, log_tag, epochs=None, ckpt=None):
    """Hinge on the residual lens with a RANK-r delta: W_eff = U V^T."""
    rank = cfg.get("rank", 16)
    lr = cfg["lr"]
    margin = cfg.get("margin", L.MARGIN)
    l2 = cfg.get("l2", L.L2)
    epochs = epochs or L.EPOCHS
    ckpt = ckpt or L.CKPT
    rng = np.random.default_rng(cfg.get("seed", L.SEED))
    U = (0.03 * rng.normal(0.0, 1.0, size=(384, rank))).astype(np.float32)
    V = (0.05 * rng.normal(0.0, 1.0, size=(rank, DIM))).astype(np.float32)
    best_val, best_W, best_ep, exit_streak = -1.0, None, -1, 0
    t0 = time.time()
    for ep in range(epochs):
        W_eff = U @ V
        GW = np.zeros_like(W_eff)
        loss = 0.0
        n_pairs = 0
        for h in tr_tasks:
            d = D[h]
            if not d["faulty"]:
                continue
            xq, M = d["q"], d["M"]
            gq = np.tanh(xq @ W_eff)
            G = np.tanh(M @ W_eff)
            t_scores = G @ gq / DIM + np.asarray(d["bundle"])
            A = np.zeros(len(t_scores), dtype=np.float64)
            fl = d["faulty"]
            for f in fl:
                viol = (margin - t_scores + t_scores[f]) > 0
                viol[list(fl)] = False
                c = int(viol.sum())
                if c == 0:
                    continue
                loss += float((margin - t_scores[viol] + t_scores[f]).sum())
                n_pairs += c
                dgq = (viol @ G - c * G[f]) / DIM
                GW += np.outer(xq, dgq * (1 - gq ** 2))
                A[viol] -= 1.0
                A[f] += float(c)
            for i in np.nonzero(A)[0]:
                GW += np.outer(M[i], A[i] * gq / DIM * (1 - G[i] ** 2))
        GW += l2 * W_eff
        # exact chain through W = U V^T
        gU = GW @ V.T
        gV = U.T @ GW
        U -= lr * gU / max(1, len(tr_tasks))
        V -= lr * gV / max(1, len(tr_tasks))
        if (ep + 1) % ckpt == 0 or ep == epochs - 1:
            W_c = U @ V
            v = L.pooled_auc(val_tasks, W_c, D) if val_tasks else None
            if v is not None and v > best_val:
                best_val, best_W, best_ep = v, W_c.copy(), ep + 1
            if n_pairs and loss / n_pairs < 1e-6:
                exit_streak += 1
                if exit_streak >= 4:
                    break
            else:
                exit_streak = 0
    print(f"[{log_tag}] ep={ep + 1} val="
          f"{None if best_val < 0 else round(best_val, 3)} ({time.time() - t0:.0f}s)",
          flush=True)
    return {"W": best_W if best_W is not None else U @ V, "val_best": best_val,
            "best_epoch": best_ep, "early_exit": exit_streak >= 4}


def train_infonce(cfg, tr_tasks, val_tasks, D, log_tag, epochs=None, ckpt=None):
    """Multi-positive InfoNCE, residual form, in-batch negatives (K=12)."""
    lr = cfg["lr"]
    tau = cfg["tau"]
    l2 = cfg.get("l2", L.L2)
    K = cfg.get("k_inbatch", 12)
    epochs = epochs or L.EPOCHS
    ckpt = ckpt or L.CKPT
    rng = np.random.default_rng(cfg.get("seed", L.SEED))
    W = (0.01 * rng.normal(0.0, 1.0, size=(384, DIM))).astype(np.float32)
    all_cands = np.concatenate([D[h]["M"] for h in tr_tasks])
    n_all = all_cands.shape[0]
    best_val, best_W, best_ep = -1.0, None, -1
    t0 = time.time()
    for ep in range(epochs):
        GW = np.zeros_like(W)
        loss = 0.0
        n_q = 0
        for h in tr_tasks:
            d = D[h]
            if not len(d["M"]):
                continue
            q, M = d["q"], d["M"]
            n_own = len(M)
            Xn = all_cands[rng.integers(0, n_all, size=K)]
            Xall = np.concatenate([M, Xn], axis=0)
            raws = np.asarray(d["bundle"] + list(q @ Xn.T), dtype=np.float64)
            ispos = np.zeros(Xall.shape[0], dtype=bool)
            ispos[:n_own] = True
            ispos[list(d["faulty"])] = False        # positives = non-faulty own
            if not ispos.any():
                continue
            gq = np.tanh(q @ W)
            G = np.tanh(Xall @ W)
            s = (G @ gq) / DIM + raws
            smax = float(s.max())
            z_all = np.exp((s - smax) / tau).sum()
            z_pos = np.exp((s[ispos] - smax) / tau).sum()
            w_j = np.exp((s - smax) / tau) / z_all
            w_j[ispos] -= np.exp((s[ispos] - smax) / tau) / z_pos
            loss += float(np.log(z_all) - np.log(z_pos))
            n_q += 1
            dgq = (w_j @ G) / DIM
            GW += np.outer(q, dgq * (1 - gq ** 2))
            for j in range(Xall.shape[0]):
                if w_j[j] == 0.0:
                    continue
                GW += np.outer(Xall[j], (w_j[j] * gq / DIM) * (1 - G[j] ** 2))
        GW += l2 * W
        W -= lr * GW / max(1, n_q)
        if (ep + 1) % ckpt == 0 or ep == epochs - 1:
            v = L.pooled_auc(val_tasks, W, D) if val_tasks else None
            if v is not None and v > best_val:
                best_val, best_W, best_ep = v, W.copy(), ep + 1
    print(f"[{log_tag}] ep={ep + 1} nce={loss / max(1, n_q):.4f} val="
          f"{None if best_val < 0 else round(best_val, 3)} ({time.time() - t0:.0f}s)",
          flush=True)
    return {"W": best_W if best_W is not None else W, "val_best": best_val,
            "best_epoch": best_ep, "early_exit": False}


def main() -> int:
    t_start = time.time()
    data = L.build_data()
    results = {"opponents": {k: None if v is None else round(v, 4)
                             for k, v in data["opp_auc"].items()},
               "reference_addendum11": {"lens_lr1.5": 0.5901, "lens_lr0.15": 0.5901,
                                        "note": "pooled == the init bundle; no survivor"},
               "gaps": data["gaps"], "configs": {}}

    plans = [
        ("G1_margin0", {"lr": 1.5, "margin": 0.0}, L.train_fold),
        ("G1_margin0.01", {"lr": 1.5, "margin": 0.01}, L.train_fold),
        ("G1_margin0.05", {"lr": 1.5, "margin": 0.05}, L.train_fold),
        ("G3_wd1e-3", {"lr": 1.5, "margin": 0.1, "l2": 1e-3}, L.train_fold),
        ("G3_wd1e-2", {"lr": 1.5, "margin": 0.1, "l2": 1e-2}, L.train_fold),
        ("G4_lowrank16", {"lr": 1.5, "margin": 0.1, "rank": 16}, train_lowrank),
        ("G2_infonce_t0.1", {"lr": 1.5, "tau": 0.1, "k_inbatch": 12}, train_infonce),
        ("G2_infonce_t0.3", {"lr": 1.5, "tau": 0.3, "k_inbatch": 12}, train_infonce),
    ]
    for tag, cfg, trainer in plans:
        res = L.run_cv(cfg, data, tag, trainer=trainer)
        W_of = res.pop("W_of")
        res.pop("pairs", None)
        results["configs"][tag] = res
        print(f"[{tag}] pooled={res['pooled_auc']} BEATS_ALL={res['beats_all_opponents']}",
              flush=True)
        if res["beats_all_opponents"]:
            results.setdefault("audit", {})[tag] = L.composition_audit(
                {**res, "W_of": W_of}, data)

    results["pre_registered_bar"] = ("pooled lens AUC > ALL FOUR opponents + 0.02; "
                                     "PASS audit mandatory; ties open")
    results["survivors"] = [k for k, c in results["configs"].items()
                            if c["beats_all_opponents"]]
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
    print(f"[done] survivors={results['survivors']} -> {OUT} "
          f"({time.time() - t_start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
