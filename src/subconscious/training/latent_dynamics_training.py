"""Closed-form training for the LatentDynamicsHead (STRM Phase 2c).

No epoch loop, no optimizer -- the head is a linear ``z_{t+1} = A z_t + b``
and its parameters are the closed-form ridge solution baked into an
``nn.Linear``. This is the "cheapest head" the Phase 0b probe earned: a
linear predictor already fits the dynamics (R^2=0.297 over the constant-mean
baseline) and its L2-residual surprise-AUC (0.7625) beats JEPA's cosine
surprise (0.565), with no collapse possible on a frozen backbone.

The fit:
  1. Load the Phase 0a traces, split chains 80/20 (no pair leakage).
  2. Project each chain's state to the "last" rep [384] (last layer, mean
     over d_state) -- the rep the 0b probe validated.
  3. Sample consecutive (z_t, z_{t+1}) transitions from the train chains.
  4. Ridge-fit ``z_{t+1} = A z_t + b`` (``strm_traces.ridge_fit``) -> weights
     that operate on RAW z, baked into ``nn.Linear(384, 384)``.
  5. Eval on the val chains: R^2 over the constant-mean baseline (gate 0.15)
     and surprise-AUC (correct vs mismatched next-state, L2 residual; gate 0.70)
     -- the two numbers the 0b probe reported.

``best.pt`` == ``final.pt`` (one fit, no epochs): both hold ``{"linear": sd,
"state_dim": 384, "r2": float, "surprise_auc": float}``. ``train_log.json``
records the fit + eval so the gate decision is auditable. The CLI is
``scripts/train_latent_dynamics_head.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..latent_dynamics_head import LatentDynamicsHead, STATE_DIM
from .strm_traces import (
    auc,
    load_traces,
    ridge_fit,
    sample_transitions,
    split_chains,
    state_rep_last,
)


@dataclass
class LatentDynamicsTrainingConfig:
    # The trace file (regenerable via scripts/generate_strm_traces.py).
    traces_path: str = "data/probe/recoverability/traces.pt"
    # Closed-form ridge penalty (on the standardized features).
    lam: float = 10.0
    # Eval gates (the Phase 0b probe thresholds).
    r2_gate: float = 0.15          # linear R^2 over the constant-mean baseline
    surprise_auc_gate: float = 0.70   # correct-vs-mismatched L2 residual
    # Split.
    val_fraction: float = 0.2
    seed: int = 0
    # IO.
    checkpoint_dir: str = "data/training/strm_latent_dynamics"
    # NB: no ``device`` field -- the fit is numpy and the checkpoint is a plain
    # nn.Linear state_dict (device-agnostic). ``load_latent_dynamics_head``
    # moves the module to the resolved device at SERVE time.


def _eval_linear(
    W: np.ndarray,
    b: np.ndarray,
    Zt_va: np.ndarray,
    Ztp1_va: np.ndarray,
) -> tuple[float, float]:
    """R^2 over the constant-mean baseline + surprise-AUC, on the val set.

    ``W``/``b`` operate on RAW z (baked). R^2 = 1 - mse_lin/mse_mean where
    ``mse_mean`` is the variance of the val z_{t+1} around its own mean (the
    "predict the mean" baseline). Surprise-AUC: the L2 residual
    ``||A z_t + b - z_{t+1}||^2`` ranks correct next-states below mismatched
    (permuted) ones -- a higher residual = more surprising = the wrong
    next-state. The permutation is seeded for reproducibility.

    Both ``Ztp1_va`` and ``W``/``b`` are 2-D here (state_dim outputs), so the
    per-row residual is ``.mean(axis=1)`` over the state dims.
    """
    pred = Zt_va @ W + b                      # [N, state_dim]  (W [D, state_dim])
    Y = Ztp1_va.astype(np.float64)
    mse_lin = float(((pred - Y) ** 2).mean())
    mse_mean = float(((Y - Y.mean(axis=0, keepdims=True)) ** 2).mean())
    r2 = 1.0 - mse_lin / mse_mean if mse_mean > 0 else float("nan")

    # surprise = per-row mean squared residual.
    resid_correct = ((pred - Y) ** 2).mean(axis=1)            # [N]
    rng = np.random.default_rng(0)
    n = len(Ztp1_va)
    Y_wrong = Y[rng.permutation(n)]                          # mismatched next-states
    resid_wrong = ((pred - Y_wrong) ** 2).mean(axis=1)        # [N]
    scores = np.concatenate([resid_correct, resid_wrong])
    labels = np.concatenate([np.zeros(n, dtype=np.int64),
                             np.ones(n, dtype=np.int64)])
    return r2, auc(scores, labels)


def fit_latent_dynamics(
    traces: list[dict],
    config: Optional[LatentDynamicsTrainingConfig] = None,
) -> dict:
    """Closed-form ridge fit of the LatentDynamicsHead + eval. Returns metrics.

    ``traces`` is the per-chain ``{inputs, states}`` list (load via
    ``load_traces`` first). The fit uses the train chains; R^2 + surprise-AUC
    are computed on the held-out val chains. The fitted ``nn.Linear`` is saved
    to ``config.checkpoint_dir`` as both ``best.pt`` and ``final.pt`` (one fit,
    no epochs -- they are identical), and a ``train_log.json`` records the fit
    + the gate decision. Returns ``{"r2", "surprise_auc", "r2_gate",
    "surprise_auc_gate", "go", "n_train", "n_val", "state_dim"}``.
    """
    cfg = config or LatentDynamicsTrainingConfig()
    n_ch = len(traces)
    if n_ch < 5:
        raise RuntimeError(
            f"need >=5 chains to fit + eval; got {n_ch}. Regenerate traces via "
            f"scripts/generate_strm_traces.py."
        )
    train_idx, val_idx = split_chains(n_ch, cfg.val_fraction, seed=cfg.seed)
    tr_traces = [traces[i] for i in train_idx]
    va_traces = [traces[i] for i in val_idx]

    # Project to the "last" rep [384] and sample consecutive transitions.
    z_tr = [state_rep_last(tr["states"]) for tr in tr_traces]
    z_va = [state_rep_last(tr["states"]) for tr in va_traces]
    Zt_tr, Ztp1_tr = sample_transitions(z_tr)
    Zt_va, Ztp1_va = sample_transitions(z_va)
    if len(Zt_tr) == 0 or len(Zt_va) == 0:
        raise RuntimeError(
            f"no transitions sampled (train={len(Zt_tr)} val={len(Zt_va)}); "
            f"chains must have >=2 steps. Regenerate traces."
        )
    state_dim = Zt_tr.shape[1]
    if state_dim != STATE_DIM:
        raise RuntimeError(
            f"unexpected state_dim={state_dim}; the head is fixed at "
            f"STATE_DIM={STATE_DIM} (last layer, mean over d_state)."
        )

    # Closed-form ridge fit -> weights operating on RAW z.
    W, b = ridge_fit(Zt_tr, Ztp1_tr, lam=cfg.lam)   # W [D, D], b [D]
    # Bake into nn.Linear: linear computes x @ W.T + b, so weight = W.T.
    head = LatentDynamicsHead(state_dim=STATE_DIM)
    with torch.no_grad():
        head.linear.weight.copy_(torch.from_numpy(np.asarray(W.T, dtype=np.float32)))
        head.linear.bias.copy_(torch.from_numpy(np.asarray(b.reshape(-1), dtype=np.float32)))
    head.eval()

    r2, surp_auc = _eval_linear(W, b, Zt_va, Ztp1_va)
    go = bool(r2 > cfg.r2_gate and (not np.isnan(surp_auc)) and surp_auc > cfg.surprise_auc_gate)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "linear": head.state_dict(),
        "state_dim": STATE_DIM,
        "r2": float(r2),
        "surprise_auc": float(surp_auc),
        "r2_gate": cfg.r2_gate,
        "surprise_auc_gate": cfg.surprise_auc_gate,
        "lam": cfg.lam,
        "n_train": len(Zt_tr),
        "n_val": len(Zt_va),
        "go": go,
    }
    # best.pt == final.pt: one closed-form fit, no epoch selection.
    torch.save(payload, ckpt_dir / "best.pt")
    torch.save(payload, ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "fit": "closed-form ridge",
            "r2": float(r2),
            "surprise_auc": float(surp_auc),
            "r2_gate": cfg.r2_gate,
            "surprise_auc_gate": cfg.surprise_auc_gate,
            "go": go,
            "lam": cfg.lam,
            "n_train_chains": len(tr_traces),
            "n_val_chains": len(va_traces),
            "n_train_transitions": len(Zt_tr),
            "n_val_transitions": len(Zt_va),
            "state_dim": state_dim,
            "config": cfg.__dict__,
        }, f, indent=2)

    print(f"  [2c latent-dynamics] closed-form ridge fit: "
          f"R^2={r2:.4f} (gate >{cfg.r2_gate})  "
          f"surprise-AUC={surp_auc:.4f} (gate >{cfg.surprise_auc_gate})  "
          f"-> {'GO' if go else 'NO-GO'}")
    print(f"    train transitions={len(Zt_tr)}  val transitions={len(Zt_va)}  "
          f"state_dim={state_dim}")
    print(f"    best.pt == final.pt at {ckpt_dir / 'best.pt'}")
    return {"r2": float(r2), "surprise_auc": float(surp_auc),
            "r2_gate": cfg.r2_gate, "surprise_auc_gate": cfg.surprise_auc_gate,
            "go": go, "n_train": len(Zt_tr), "n_val": len(Zt_va),
            "state_dim": state_dim}