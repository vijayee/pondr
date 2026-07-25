"""SCRATCH probe (never committed) -- JST Phase 0b latent-dynamics GO/NO-GO gate.

Two parts (step 1 is cheap and runs first; step 2 only if step 1 passes):
  (i)  does the WM recurrent state z_t have learnable transition dynamics at all?
       Step 1: fit a LINEAR z_{t+1} ~= A z_t + b (ridge) and compare one-step
       prediction MSE to a constant-mean baseline. If linear can't beat mean,
       the latent has no learnable dynamics -- DROP the latent-dynamics head,
       skip step 2.
  (ii) can an EMA JEPA predictor avoid collapse on it?
       Step 2: train g(z_t) -> z_{t+1} with stop-grad on the target and the
       jepa_loss batch-negatives anti-collapse term (reuses
       jepa_loss.jepa_contrastive_loss). Check latent variance stays bounded
       (collapse detector) and a surprise-AUC on mismatched next-states
       (correct vs wrong next-state ranked by prediction error).

Reuses the Phase 0a traces (data/probe/recoverability/traces.pt). z_t = the
pooled recurrent state (per-d_state mean per layer, 1536-dim) -- the same
representation Phase 0a showed carries the recoverability signal.

    python scripts/_probe_latent_dynamics.py [--state-rep pooled] [--step both]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.jepa_loss import jepa_contrastive_loss

TRACE_PATH = Path("data/probe/recoverability/traces.pt")
STATE_DIM_FULL = 4 * 16 * 384
STATE_DIM_POOL = 4 * 384


def log(msg: str) -> None:
    print(f"[probe-0b] {msg}", flush=True)


def state_repr(traces, state_rep: str):
    """Return per-chain z [T, state_dim] (fp32 numpy) for each chain.

    state_rep:
      "pooled" -> all 4 layers, mean over d_state -> [4*384]=1536
      "last"   -> last layer only, mean over d_state -> [384]  (what the JEPA
                  predictor operates on; N > D so the linear fit is determined)
      "full"   -> flattened [4,16,384]=4096
    """
    out = []
    for tr in traces:
        T = tr["states"].shape[0]
        st = tr["states"].to(torch.float32)  # [T,4,16,384]
        if state_rep == "pooled":
            z = st.mean(dim=2).reshape(T, -1).numpy()
        elif state_rep == "last":
            z = st[:, -1].mean(dim=1).numpy()  # [T,384]
        else:
            z = st.reshape(T, -1).numpy()
        out.append(z)
    return out


def transitions(zs):
    """Flatten (z_t, z_{t+1}) consecutive k=1 transitions across chains."""
    Zt, Ztp1 = [], []
    for z in zs:
        for t in range(len(z) - 1):
            Zt.append(z[t])
            Ztp1.append(z[t + 1])
    return np.asarray(Zt, dtype=np.float32), np.asarray(Ztp1, dtype=np.float32)


def auc(scores, labels):
    y = labels.astype(np.int64)
    s = scores.astype(np.float64)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ── step 1: linear baseline vs constant-mean ──

def standardize(X_tr, X_va):
    mu = X_tr.mean(0, keepdims=True)
    sd = np.where(X_tr.std(0, keepdims=True) < 1e-6, 1.0, X_tr.std(0, keepdims=True))
    return (X_tr - mu) / sd, (X_va - mu) / sd


def step1_linear(Zt_tr, Ztp1_tr, Zt_va, Ztp1_va, lam: float):
    """Fit ridge z_{t+1} = A z_t + b; return (linear_mse, mean_mse) on val."""
    # augment z with a bias column
    X_tr = np.hstack([Zt_tr, np.ones((len(Zt_tr), 1))]).astype(np.float64)
    X_va = np.hstack([Zt_va, np.ones((len(Zt_va), 1))]).astype(np.float64)
    Xs_tr, Xs_va = standardize(X_tr, X_va)
    Y_tr = Ztp1_tr.astype(np.float64)
    Y_va = Ztp1_va.astype(np.float64)
    gram = Xs_tr.T @ Xs_tr + lam * np.eye(Xs_tr.shape[1], dtype=np.float64)
    W = np.linalg.solve(gram, Xs_tr.T @ Y_tr)  # [D+1, state_dim]
    pred_lin = Xs_va @ W
    mse_lin = float(((pred_lin - Y_va) ** 2).mean())
    # constant-mean baseline: predict the train-set mean of z_{t+1}
    mean_pred = Ztp1_tr.mean(0, keepdims=True)
    mse_mean = float(((Y_va - mean_pred) ** 2).mean())
    return mse_lin, mse_mean


# ── step 2: EMA JEPA predictor + collapse + surprise-AUC ──

class Predictor(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z):
        return self.net(z)


def step2_ema(zs_tr, zs_va, dim, epochs=30, bs=128, lr=1e-3, temperature=0.1, seed=0):
    """Train g(z_t) -> z_{t+1} with stop-grad target + jepa_loss negatives.

    Returns: final train pred-MSE (L2, cross-objective -- expected high for a
    cosine-trained model), latent variance trajectory, and surprise-AUC measured
    in g's NATIVE cosine distance (1 - cos(g(z_t), z_{t+1})) on correct vs
    mismatched next-states. Mismatched next-states should rank higher (higher
    cosine distance = more surprising).
    """
    torch.manual_seed(seed)
    Zt_tr, Ztp1_tr = transitions(zs_tr)
    Zt_va, Ztp1_va = transitions(zs_va)
    g = Predictor(dim)
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    Zt_t = torch.from_numpy(Zt_tr); Ztp1_t = torch.from_numpy(Ztp1_tr)
    n = Zt_t.shape[0]
    var_traj = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            z_t = Zt_t[idx]
            z_tp1 = Ztp1_t[idx]
            pred = g(z_t)
            target = z_tp1.detach()  # stop-grad
            neg = z_tp1.detach()     # negatives: OTHER targets in the batch
            loss = jepa_contrastive_loss(pred, target, neg, temperature=temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * idx.numel()
        with torch.no_grad():
            var_traj.append(float(g(Zt_t).var(dim=0).mean()))
        if (ep + 1) % 10 == 0:
            log(f"  g epoch {ep+1}/{epochs} loss {tot/n:.4f} out-var {var_traj[-1]:.4f}")
    g.eval()
    with torch.no_grad():
        pred_va = g(torch.from_numpy(Zt_va))
        mse = float(((pred_va - torch.from_numpy(Ztp1_va)) ** 2).mean())  # cross-objective

    # surprise in g's NATIVE cosine distance (aligned with its contrastive loss)
    rng = np.random.default_rng(seed)
    n_va = len(Zt_va)
    Ztp1_wrong = Ztp1_va[rng.permutation(n_va)]
    with torch.no_grad():
        gv = g(torch.from_numpy(Zt_va))
        zc = torch.from_numpy(Ztp1_va); zw = torch.from_numpy(Ztp1_wrong)
        surp_correct = (1.0 - F.cosine_similarity(gv, zc, dim=-1)).numpy()
        surp_wrong = (1.0 - F.cosine_similarity(gv, zw, dim=-1)).numpy()
    scores = np.concatenate([surp_correct, surp_wrong])
    labels = np.concatenate([np.zeros(n_va), np.ones(n_va)])
    return mse, var_traj, auc(scores, labels)


def linear_surprise_auc(Zt_tr, Ztp1_tr, Zt_va, Ztp1_va, lam):
    """Surprise-AUC of the step-1 LINEAR predictor (L2 residual): the baseline
    the JEPA predictor must beat. surprise = ||A z_t + b - z_{t+1}||; mismatched
    next-states should have larger residuals."""
    X_tr = np.hstack([Zt_tr, np.ones((len(Zt_tr), 1))]).astype(np.float64)
    X_va = np.hstack([Zt_va, np.ones((len(Zt_va), 1))]).astype(np.float64)
    Xs_tr, Xs_va = standardize(X_tr, X_va)
    Y_tr = Ztp1_tr.astype(np.float64)
    Ztp1_va64 = Ztp1_va.astype(np.float64)
    gram = Xs_tr.T @ Xs_tr + lam * np.eye(Xs_tr.shape[1], dtype=np.float64)
    W = np.linalg.solve(gram, Xs_tr.T @ Y_tr)
    pred_va = Xs_va @ W
    rng = np.random.default_rng(0)
    n_va = len(Zt_va)
    Ztp1_wrong = Ztp1_va64[rng.permutation(n_va)]
    err_correct = ((pred_va - Ztp1_va64) ** 2).mean(axis=1)
    err_wrong = ((pred_va - Ztp1_wrong) ** 2).mean(axis=1)
    scores = np.concatenate([err_correct, err_wrong])
    labels = np.concatenate([np.zeros(n_va), np.ones(n_va)])
    return auc(scores, labels)


# ── main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-rep", choices=["pooled", "last", "full"], default="last")
    ap.add_argument("--step", choices=["1", "2", "both"], default="both")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r2-gate", type=float, default=0.15,
                    help="step-1 gate: linear R^2 over mean must exceed this")
    ap.add_argument("--surprise-auc-gate", type=float, default=0.70)
    args = ap.parse_args()
    np.random.seed(args.seed)

    log(f"loading traces -> {TRACE_PATH}")
    traces = torch.load(TRACE_PATH, weights_only=False)
    n_ch = len(traces)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_ch)
    n_va = max(1, n_ch // 5)
    va_idx = set(perm[:n_va].tolist())
    tr_idx = [i for i in range(n_ch) if i not in va_idx]

    zs_tr = state_repr([traces[i] for i in tr_idx], args.state_rep)
    zs_va = state_repr([traces[i] for i in sorted(va_idx)], args.state_rep)
    Zt_tr, Ztp1_tr = transitions(zs_tr)
    Zt_va, Ztp1_va = transitions(zs_va)
    dim = Zt_tr.shape[1]
    log(f"  chains tr/va={len(zs_tr)}/{len(zs_va)} transitions tr/va={len(Zt_tr)}/{len(Zt_va)} state_dim={dim}")

    # ── step 1 ──
    log("step 1: linear baseline z_{t+1} = A z_t + b vs constant-mean")
    t0 = time.time()
    mse_lin, mse_mean = step1_linear(Zt_tr, Ztp1_tr, Zt_va, Ztp1_va, args.lam)
    r2 = 1.0 - mse_lin / mse_mean if mse_mean > 0 else float("nan")
    log(f"  linear fit in {time.time()-t0:.1f}s")
    print(f"  val MSE linear = {mse_lin:.5f}")
    print(f"  val MSE mean    = {mse_mean:.5f}")
    print(f"  R^2 (linear over mean) = {r2:.4f}   [GATE: > {args.r2_gate}]")
    step1_go = r2 > args.r2_gate
    print(f"  step 1 decision: {'GO' if step1_go else 'NO-GO'}")

    if not step1_go:
        print("\n=== Phase 0b gate ===")
        print("  step 1 NO-GO: linear dynamics do not beat the constant-mean baseline.")
        print("  -> the latent has no learnable dynamics; DROP the latent-dynamics head.")
        print("  -> the three supervised heads (relevance/recoverability/graduation) still stand.")
        return

    if args.step == "1":
        return

    # ── step 2 ──
    print()
    log("step 2: EMA JEPA predictor g(z_t) -> z_{t+1} + collapse + surprise-AUC")
    t0 = time.time()
    mse_g, var_traj, surp_auc_g = step2_ema(zs_tr, zs_va, dim, seed=args.seed)
    log(f"  g trained in {time.time()-t0:.1f}s")
    # linear surprise-AUC baseline (the bar the JEPA predictor must clear)
    surp_auc_lin = linear_surprise_auc(Zt_tr, Ztp1_tr, Zt_va, Ztp1_va, args.lam)
    print(f"  val pred MSE (g, L2 cross-obj) = {mse_g:.5f}   (linear L2 was {mse_lin:.5f}; "
          f"high for g is expected -- cosine-trained, not L2)")
    print(f"  latent variance traj   = first {var_traj[0]:.4f} -> last {var_traj[-1]:.4f}  "
          f"({'bounded' if var_traj[-1] > 1e-4 else 'COLLAPSED'})")
    print(f"  surprise-AUC linear (L2 residual)    = {surp_auc_lin:.4f}   [baseline]")
    print(f"  surprise-AUC g (cosine, native)       = {surp_auc_g:.4f}   [GATE: > {args.surprise_auc_gate}]")
    bounded = var_traj[-1] > 1e-4
    best_surprise = max(surp_auc_lin, surp_auc_g)
    step2_go = bounded and best_surprise > args.surprise_auc_gate
    print(f"\n=== Phase 0b gate ===")
    print(f"  step 1: GO (R^2={r2:.4f}) -- linear dynamics ARE learnable")
    print(f"  step 2: {'GO' if step2_go else 'NO-GO'} "
          f"(var bounded={bounded}, best surprise-AUC={best_surprise:.4f} "
          f"[linear {surp_auc_lin:.3f} / g-cosine {surp_auc_g:.3f}])")
    if step2_go:
        if surp_auc_lin >= surp_auc_g:
            print("  -> latent-dynamics head viable; a LINEAR predictor already gives the surprise")
            print("     signal -- the JEPA/EMA anti-collapse machinery is NOT required here (linear")
            print("     cannot collapse). Ship linear (or a light MSE MLP) as the head; JEPA optional.")
        else:
            print("  -> latent-dynamics head viable; the JEPA predictor beats the linear surprise")
            print("     baseline, so the EMA machinery earns its place.")
    else:
        print("  -> no predictor clears the surprise gate despite learnable dynamics (step 1 GO).")
        print("  -> retry once with stronger anti-collapse (more negatives, higher temp, VICReg),")
        print("     or DROP the latent-dynamics head. The three supervised heads do not depend on it.")


if __name__ == "__main__":
    main()