"""PHASE 2 E2 battery -- F4 ADDENDUM (identity family: the cap-not-reached
window), same splits / same survival bar as the main battery + fine addendum.

WHAT PREVIOUS RUNS MEASURED:
  Identity init = the ONLY family whose delta holds or beats raw cosine OOS
  (B2 0.8266 / F1 0.8279 / F3-lr0.15 0.8316 / F3-lr0.3 0.8312 vs cosine
  0.8235, tail 2/15) -- and in F1/F3 the val-best checkpoint sat AT the
  epoch cap (130/150 of 150), i.e. val was still improving when training
  stopped. A run cut inside its improvement window cannot claim a verdict.

  F4 pre-registration (fixed BEFORE fitting; nothing else changes):
    identity init, proj-only hinge 0.1, D=384, checkpoint every 25,
    grid: lr in {0.05, 0.15, 1.0} x L2-on-(W-W0) in {1e-4, 1e-3} x 600 epochs.
    6 cells; per-cell readouts and the survival bar IDENTICAL to the main
    battery (holdout bound AUC >= cosine - 0.01; tail recovery >= 0.5;
    delta non-noop). Direction always reported; nothing survives -> the
    battery-exhausted-at-these-families verdict stands, open to more organic
    data / two-stage re-ranker / curriculum.

Run:  python scripts/icm/battery_mode2_f4.py
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
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import battery_mode2_remediation as B  # noqa: E402
from battery_mode2_remediation import (  # noqa: E402
    SEED, TRAIN_FRAC, holdout_readout, ensure_embeds, train_hinge,
)

OUT = os.path.join("data", "_scratch", "icm", "battery_mode2_f4_results.json")


def main() -> int:
    rows = B.build_rows()
    qz = ensure_embeds(rows)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    n_tr = int(len(rows) * TRAIN_FRAC)
    tr_idx, te_idx = list(order[:n_tr]), list(order[n_tr:])
    rng2 = np.random.default_rng(B.VAL_SEED)
    vord = rng2.permutation(len(tr_idx))
    n_val = int(len(tr_idx) * B.VAL_FRAC)
    val_idx = [tr_idx[i] for i in vord[:n_val]]
    tr2_idx = [tr_idx[i] for i in vord[n_val:]]
    tr_rows = [rows[i] for i in tr2_idx]
    val_rows = [rows[i] for i in val_idx]
    te_rows = [rows[i] for i in te_idx]
    print(f"[split] train {len(tr_rows)} / val {len(val_rows)} / holdout {len(te_rows)}",
          flush=True)

    cos_h = holdout_readout(te_rows, None, "raw-cosine")
    results = {"raw_cosine_holdout": cos_h, "configs": {}}
    for lr in (1.0, 0.05, 0.15):
        for l2 in (1e-4, 1e-3):
            tag = f"F4_identity_lr{lr:g}_L{l2:g}"
            cfg = {"loss": "hinge", "W": {"form": "identity", "dim": 384},
                   "margin": 0.1, "l2": l2, "lr": lr, "seed": SEED}
            t0 = time.time()
            tr = train_hinge(cfg, tr_rows, val_rows, qz, 600, tag, ckpt_every=25)
            dnorm = float(np.linalg.norm(tr["W"] - tr["W0"]))
            np.savez(os.path.join("data", "_scratch", "icm", f"battery_mode2_W_{tag}.npz"),
                     W=tr["W"].astype(np.float32))
            ro = holdout_readout(te_rows,
                                 {"W": tr["W"], "form": tr["form"], "dim": tr["dim"]},
                                 f"{tag}@bestval(ep{tr['best_epoch']})")
            cb, cc = ro["holdout_bound_auc"], ro["holdout_cosine_auc"]
            rate = ro["tail_recovered"] / ro["tail_questions"] if ro["tail_questions"] else 0.0
            surv = (cb is not None and cc is not None and cb >= cc - 0.01
                    and rate >= 0.5 and dnorm > 1e-3)
            results["configs"][tag] = {
                "val_best_auc": round(tr["val_best"], 4), "best_epoch": tr["best_epoch"],
                "delta_norm": round(dnorm, 4), "holdout_bestval": ro,
                "lr": lr, "l2": l2, "survives": bool(surv),
                "train_s": round(time.time() - t0)}
            print(f"[{tag}] SURVIVES={surv} val_best={tr['val_best']:.3f} "
                  f"@ep{tr['best_epoch']} dn={dnorm:.3f}", flush=True)

    results["pre_registered_bar"] = ("holdout mean-arm bound AUC >= cosine-0.01 AND "
                                     "tail recovery >= 0.5 AND delta_norm > 1e-3")
    results["survivors"] = [k for k, v in results["configs"].items()
                            if v.get("survives")]
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
    print(f"[done] survivors={results['survivors']} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())