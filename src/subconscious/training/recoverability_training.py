"""Closed-form training for the RecoverabilityHead (STRM Phase 2b).

No epoch loop, no optimizer -- the head is the Phase 0a probe's ridge
regressor ``P([state_t ; u_i]) -> e_hat(i,t)`` baked into an ``nn.Linear``.
The probe earned this path: ridge P scored AUC 0.810 against a per-split-
median "forgotten" label, beating the free monotonic-forgetting-in-k
baseline (0.732). Both the label-generating decoder ``D`` and ``P`` were
ridge in the probe; this trainer reproduces that exactly (the 0a probe WAS
the 2b trainer, modulo the probe being a throwaway script).

The fit:
  1. Load the Phase 0a traces, split chains 80/20 (no pair leakage).
  2. Sample (state_t, anchor u_i, lag k) triples for train + val via
     ``sample_recoverability_pairs`` with the "pooled" rep [1536].
  3. Decoder ``D = ridge(state_t, u_i)`` (train fit). Its reconstruction
     error ``e(i,t) = ||D(state_t) - u_i||^2.mean()`` is the GROUND-TRUTH
     forgetting label -- computed on BOTH train and val using the train-fit
     D (D is a fixed label generator, not a head to evaluate).
  4. Binary "forgotten" label ``y = e > median(e)`` PER SPLIT (balanced, so
     AUC has both classes regardless of train/val error-distribution skew).
  5. Head ``P = ridge([state_t ; u_i], e)`` (train fit). Bake into
     ``nn.Linear(1920, 1)``. Eval AUC on the val pairs.
  6. Baselines: ``k`` alone is the free monotonic-forgetting baseline P must
     beat (ranking by ``e`` itself is a trivial 1.0 -- the label IS the
     thresholded ``e`` -- so it is not a meaningful upper bound; omitted).
  7. GO = val AUC >= gate AND val AUC > k-baseline.

``best.pt`` == ``final.pt`` (one fit, no epochs): both hold ``{"linear": sd,
"state_dim_pooled": 1536, "anchor_dim": 384, "ridge_auc": float,
"k_auc": float, "gate_auc": float, "go": bool, ...}``. ``train_log.json``
records the decoder fit, the baselines, and the gate decision. The decoder
``D`` is train-side only and is NOT in the checkpoint (the head at serve is
just ``P``). The CLI is ``scripts/train_recoverability_head.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..recoverability_head import (
    ANCHOR_DIM,
    INPUT_DIM,
    STATE_DIM_POOLED,
    RecoverabilityHead,
)
from .strm_traces import (
    auc,
    load_traces,
    ridge_fit,
    sample_recoverability_pairs,
    split_chains,
)


@dataclass
class RecoverabilityTrainingConfig:
    # The trace file (regenerable via scripts/generate_strm_traces.py).
    traces_path: str = "data/probe/recoverability/traces.pt"
    # Max lookback horizon k = t - i for (i,t) pairs (the Phase 0a probe used 8).
    k_max: int = 8
    # Closed-form ridge penalty (on the standardized features) for D and P.
    lam: float = 10.0
    # Eval gate: P val AUC must clear this AND beat the k-baseline.
    gate_auc: float = 0.75
    # Split.
    val_fraction: float = 0.2
    seed: int = 0
    # IO.
    checkpoint_dir: str = "data/training/strm_recoverability"
    # NB: no ``device`` field -- the fit is numpy and the checkpoint is a
    # plain nn.Linear state_dict (device-agnostic). The loader moves the
    # module to the resolved device at SERVE time.


def _binary_forgotten_label(e: np.ndarray) -> np.ndarray:
    """``y = e > median(e)`` -- the per-split "forgotten" label.

    Per-split median (not a global threshold) keeps both classes present in
    each split even when the train/val error distributions have different
    scales -- the Phase 0a probe's choice. AUC needs both classes.
    """
    med = np.median(e)
    return (e > med).astype(np.int64)


def fit_recoverability(
    traces: list[dict],
    config: Optional[RecoverabilityTrainingConfig] = None,
) -> dict:
    """Closed-form ridge fit of the RecoverabilityHead + eval. Returns metrics.

    ``traces`` is the per-chain ``{inputs, states}`` list (load via
    ``load_traces`` first). The decoder ``D`` and head ``P`` are fit on the
    train chains; P's AUC (vs the per-split-median forgotten label) and the
    k-baseline are computed on the held-out val chains. The fitted ``P`` is
    saved to ``config.checkpoint_dir`` as both ``best.pt`` and ``final.pt``
    (one fit, no epochs -- identical), and a ``train_log.json`` records the
    fit + the gate decision. Returns ``{"ridge_auc", "k_auc", "gate_auc",
    "go", "n_train", "n_val", "state_dim_pooled", "anchor_dim"}``.
    """
    cfg = config or RecoverabilityTrainingConfig()
    n_ch = len(traces)
    if n_ch < 5:
        raise RuntimeError(
            f"need >=5 chains to fit + eval; got {n_ch}. Regenerate traces via "
            f"scripts/generate_strm_traces.py."
        )
    train_idx, val_idx = split_chains(n_ch, cfg.val_fraction, seed=cfg.seed)
    tr_traces = [traces[i] for i in train_idx]
    va_traces = [traces[i] for i in val_idx]

    # Sample (state_t, anchor u_i, lag k) triples with the pooled rep.
    S_tr, U_tr, K_tr = sample_recoverability_pairs(
        tr_traces, k_max=cfg.k_max, state_rep="pooled")
    S_va, U_va, K_va = sample_recoverability_pairs(
        va_traces, k_max=cfg.k_max, state_rep="pooled")
    if len(S_tr) == 0 or len(S_va) == 0:
        raise RuntimeError(
            f"no recoverability pairs sampled (train={len(S_tr)} val={len(S_va)}); "
            f"chains must have >=2 steps. Regenerate traces."
        )
    state_dim = S_tr.shape[1]
    anchor_dim = U_tr.shape[1]
    if state_dim != STATE_DIM_POOLED or anchor_dim != ANCHOR_DIM:
        raise RuntimeError(
            f"unexpected dims: state={state_dim} anchor={anchor_dim}; the head "
            f"is fixed at pooled {STATE_DIM_POOLED} + anchor {ANCHOR_DIM}."
        )

    # Stage 2: decoder D = ridge(state_t, u_i) -> ground-truth forgetting e(i,t).
    # Fit on train; compute e on BOTH train and val using the train-fit D.
    W_d, b_d = ridge_fit(S_tr, U_tr, lam=cfg.lam)     # W_d [1536,384], b_d [384]
    e_tr = ((S_tr @ W_d + b_d - U_tr) ** 2).mean(axis=1)
    e_va = ((S_va @ W_d + b_d - U_va) ** 2).mean(axis=1)
    y_tr = _binary_forgotten_label(e_tr)
    y_va = _binary_forgotten_label(e_va)

    # Stage 3: head P = ridge([state_t ; u_i], e) -> e_hat. Bake into nn.Linear.
    X_tr = np.hstack([S_tr, U_tr]).astype(np.float32)    # [N, 1920]
    X_va = np.hstack([S_va, U_va]).astype(np.float32)    # [N, 1920]
    W_p, b_p = ridge_fit(X_tr, e_tr.astype(np.float64), lam=cfg.lam)  # W_p [1920], b scalar
    ehat_va = X_va @ W_p + b_p
    ehat_tr = X_tr @ W_p + b_p
    ridge_auc_va = auc(ehat_va, y_va)
    ridge_auc_tr = auc(ehat_tr, y_tr)

    # Baseline: k alone (free monotonic-forgetting-in-k; P must beat this).
    k_auc_va = auc(K_va.astype(np.float64), y_va)

    # Bake P into nn.Linear (linear computes x @ W.T + b -> weight = W_p[None,:]).
    head = RecoverabilityHead(
        state_dim_pooled=STATE_DIM_POOLED, anchor_dim=ANCHOR_DIM)
    with torch.no_grad():
        head.linear.weight.copy_(
            torch.from_numpy(np.asarray(W_p, dtype=np.float32)).reshape(1, -1))
        head.linear.bias.copy_(
            torch.from_numpy(np.asarray([b_p], dtype=np.float32)))
    head.eval()

    go = bool((not np.isnan(ridge_auc_va))
              and ridge_auc_va >= cfg.gate_auc
              and (not np.isnan(k_auc_va))
              and ridge_auc_va > k_auc_va)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "linear": head.state_dict(),
        "state_dim_pooled": STATE_DIM_POOLED,
        "anchor_dim": ANCHOR_DIM,
        "input_dim": INPUT_DIM,
        "ridge_auc": float(ridge_auc_va),
        "k_auc": float(k_auc_va),
        "gate_auc": cfg.gate_auc,
        "lam": cfg.lam,
        "k_max": cfg.k_max,
        "n_train": len(S_tr),
        "n_val": len(S_va),
        "go": go,
    }
    # best.pt == final.pt: one closed-form fit, no epoch selection.
    torch.save(payload, ckpt_dir / "best.pt")
    torch.save(payload, ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "fit": "closed-form ridge (decoder D + head P)",
            "decoder": "ridge(state_t, u_i) -> e(i,t); train-fit, applied to val",
            "head": "ridge([state_t; u_i], e) -> e_hat; baked into nn.Linear(1920,1)",
            "ridge_auc_val": float(ridge_auc_va),
            "ridge_auc_train": float(ridge_auc_tr),
            "k_auc_val": float(k_auc_va),
            "gate_auc": cfg.gate_auc,
            "go": go,
            "lam": cfg.lam,
            "k_max": cfg.k_max,
            "n_train_chains": len(tr_traces),
            "n_val_chains": len(va_traces),
            "n_train_pairs": len(S_tr),
            "n_val_pairs": len(S_va),
            "state_dim_pooled": state_dim,
            "anchor_dim": anchor_dim,
            "e_train_mean": float(e_tr.mean()),
            "e_val_mean": float(e_va.mean()),
            "val_pos_frac": float(y_va.mean()),
            "config": cfg.__dict__,
        }, f, indent=2)

    # Decay curve: mean e vs k (the forgetting signal should grow with k).
    decay = {}
    for k in range(1, cfg.k_max + 1):
        m_va = float(e_va[K_va == k].mean()) if (K_va == k).any() else float("nan")
        decay[k] = m_va

    print(f"  [2b recoverability] closed-form ridge fit: "
          f"val AUC={ridge_auc_va:.4f} (gate >={cfg.gate_auc})  "
          f"k-baseline={k_auc_va:.4f}  -> {'GO' if go else 'NO-GO'}")
    print(f"    train pairs={len(S_tr)}  val pairs={len(S_va)}  "
          f"state_dim={state_dim}  anchor_dim={anchor_dim}")
    print(f"    decay (val mean e by k): "
          + "  ".join(f"k={k}:{decay[k]:.3f}" for k in sorted(decay)
                      if not np.isnan(decay[k])))
    print(f"    best.pt == final.pt at {ckpt_dir / 'best.pt'}")
    return {"ridge_auc": float(ridge_auc_va), "k_auc": float(k_auc_va),
            "gate_auc": cfg.gate_auc, "go": go, "n_train": len(S_tr),
            "n_val": len(S_va), "state_dim_pooled": state_dim,
            "anchor_dim": anchor_dim, "decay": decay}