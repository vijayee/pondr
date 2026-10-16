"""PHASE 2 E2 battery -- FINE-GRAINED ADDENDUM (the two engagement-mode holes
opened by the main battery's measured results, same splits / same bar).

WHAT THE MAIN BATTERY MEASURED (battery_mode2_results.json, no survivors):
  * RESIDUAL/INFONCE family was SILENT: hinge loss constant (0.089) from
    ep200 on, delta norm ~= its own sigma-0.01 init (2.22 -> 2.28 max), val
    flat exactly at raw cosine's own level (0.843), tail recovery 0/15.
    Mechanism: raw cosine already satisfies margin-{0.02..0.1} on most train
    triples -> almost no violation gradient reaches the delta. The margins
    swept were all ABOVE (or barely below) the average cosine gap.
  * PROJ-ONLY family collapsed INSIDE the checkpoint blind window: val-best
    over eps {50..1200} was always ep50 = 0.542/0.563 (vs cosine ~0.84), so
    "early stopping" never saw epochs 1-49 where the identity init's bound
    is still the base geometry.

THE F PRE-REGISTRATION (fixed BEFORE fitting; same data, same splits, the
same survival bar and readouts as the main battery):

  F1  IDENTITY-FINE : B2 unchanged in every way except checkpoint cadence:
      D=384 identity init, proj-only hinge 0.1, L2 1e-4 on (W-W0), lr 1.5,
      150 epochs, checkpoint every 2 epochs -> the pre-collapse window IS
      visible this time.
  F2  RESIDUAL-ENGAGE : residual form at margins {0, 0.005, 0.02} (margin 0
      = every cosine inversion violates -> the richest targeted-loss signal
      the residual family can have), L2 1e-4, lr 1.5, 300 epochs, checkpoint
      every 5 epochs.
  F3  IDENTITY-LR : identity init at lr {0.15, 0.3} (the lr-1.5 hinge step
      may simply be too violent near the geometry to stay in the pre-
      collapse window at any epoch), 150 epochs, checkpoint every 2.

  Survival bar (unchanged): holdout bound AUC >= cosine-0.01 AND tail
  recovery >= 0.5 AND delta non-noop. Nothing survives -> the mode-2
  negative verdict stands BATTERY-EXHAUSTED at these families and the
  remaining open remediations are (i) more organic payoff data (LoCoMo
  harvest is named in the plan), (ii) two-stage residual re-ranker over
  cosine's tail, (iii) curriculum. Nothing dismissed as impossible.

Run:  python scripts/icm/battery_mode2_fine.py
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

import battery_mode2_remediation as B  # noqa: E402  (split + data + readouts)
from battery_mode2_remediation import (  # noqa: E402
    OUT as MAIN_OUT, SEED, TRAIN_FRAC, holdout_readout, ensure_embeds,
    train_hinge,
)

OUT = os.path.join("data", "_scratch", "icm", "battery_mode2_fine_results.json")


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

    cos_h = holdout_readout(te_rows, qz, None, "raw-cosine")
    results = {"raw_cosine_holdout": cos_h, "configs": {}}

    def train_cfg(tag, cfg, epochs, ckpt_every):
        t0 = time.time()
        tr = train_hinge(cfg, tr_rows, val_rows, qz, epochs, tag, ckpt_every=ckpt_every)
        dnorm = float(np.linalg.norm(tr["W"] - tr["W0"]))
        ro = holdout_readout(te_rows, qz,
                             {"W": tr["W"], "form": tr["form"], "dim": tr["dim"]},
                             f"{tag}@bestval(ep{tr['best_epoch']})")
        res = {"val_best_auc": round(tr["val_best"], 4), "best_epoch": tr["best_epoch"],
               "delta_norm": round(dnorm, 4), "holdout_bestval": ro,
               "form": tr["form"], "dim": tr["dim"], "l2": cfg["l2"],
               "margin": cfg["margin"], "lr": cfg["lr"], "epochs": epochs,
               "train_s": round(time.time() - t0)}
        cb, cc = ro["holdout_bound_auc"], ro["holdout_cosine_auc"]
        rate = ro["tail_recovered"] / ro["tail_questions"] if ro["tail_questions"] else 0.0
        surv = (cb is not None and cc is not None and cb >= cc - 0.01
                and rate >= 0.5 and dnorm > 1e-3)
        res["survives"] = bool(surv)
        print(f"[{tag}] SURVIVES={surv}", flush=True)
        results["configs"][tag] = res
        return res

    # F1  identity, fine checkpoints (same everything as B2)
    train_cfg("F1_identity_fine",
              {"loss": "hinge", "W": {"form": "identity", "dim": 384},
               "margin": 0.1, "l2": 1e-4, "lr": 1.5, "seed": SEED},
              150, 2)

    # F2  residual at engaging margins (0 = pure inversion objective)
    for mg in (0.0, 0.005, 0.02):
        train_cfg(f"F2_residual_M{mg:g}",
                  {"loss": "hinge", "W": {"form": "residual", "dim": 128},
                   "margin": mg, "l2": 1e-4, "lr": 1.5, "seed": SEED},
                  300, 5)

    # F3  identity at gentler lr
    for lr in (0.15, 0.3):
        train_cfg(f"F3_identity_lr{lr:g}",
                  {"loss": "hinge", "W": {"form": "identity", "dim": 384},
                   "margin": 0.1, "l2": 1e-4, "lr": lr, "seed": SEED},
                  150, 2)

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