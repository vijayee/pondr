"""SCRATCH probe (never committed) -- JST Phase 0a recoverability GO/NO-GO gate.

Question: is SSM forgetting predictable enough that a probe can estimate
recoverability of a past input ``u_i`` from a later recurrent state ``state_t``?
If yes (AUC >= ~0.75), the recoverability head (Phase 2b) is viable and its
labels are now generated. If no, JST shrinks to fixed-interval refresh and the
salience / recoverability / graduation heads are dropped.

No retraining. Uses the already-trained Phase 2a backbone
(``data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt``) frozen,
and the JEPA pre-train transition pairs
(``data/training/backbone/_after_fix2_full.jsonl``) as real input streams --
each chain is a sequence of 384-dim episode embeddings ``u_0..u_T``. No
embedder, no Oracle.

Stages (caches the trace so re-runs are fast):
  1. trace: step WorkingMemory through each chain, log (u_t, state_t) per step
     -> data/probe/recoverability/traces.pt
  2. decoder: train D(state_t) -> u_i (small MLP); e(i,t)=MSE is the
     ground-truth forgetting signal. Plot decay curve vs k=t-i.
  3. probe: train P(state_t, u_i) -> e_hat(i,t); AUC vs a binary "forgotten"
     label on held-out chains. This is the gate number.

    python scripts/_probe_recoverability.py [--max-chains 200] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.configs import BackboneConfig
from src.subconscious.training.routing_training import load_backbone
from src.subconscious.working_memory import WorkingMemory

OUT_DIR = Path("data/probe/recoverability")
TRACE_PATH = OUT_DIR / "traces.pt"
SEQ_PATH = "data/training/backbone/_after_fix2_full.jsonl"
BACKBONE_PATH = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"

STATE_DIM_FULL = 4 * 16 * 384    # flattened per-layer recurrent state [4,16,384]
STATE_DIM_POOL = 4 * 384          # per-d_state-channel MEAN per layer -> [4,384]
K_MAX = 8  # max lookback horizon k = t - i for (i,t) pairs


def log(msg: str) -> None:
    print(f"[probe-0a] {msg}", flush=True)


# ── stage 1: trace ──

def load_chains(path: str, max_chains: int) -> list[list[np.ndarray]]:
    """Forward chains as lists of 384-dim input embeddings u_0..u_T."""
    chains: dict[str, list[tuple[int, list[float]]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") != "forward":
                continue
            chains.setdefault(rec["chain_id"], []).append((rec["position"], rec["state_t"]))
    out: list[list[np.ndarray]] = []
    for cid in sorted(chains):
        steps = sorted(chains[cid], key=lambda p: p[0])
        stream = [np.asarray(s, dtype=np.float32) for _, s in steps]
        if len(stream) >= 2:
            out.append(stream)
        if len(out) >= max_chains:
            break
    return out


def build_traces(chains, backbone, device) -> list[dict]:
    """Step WorkingMemory through each chain; record (inputs, states) per chain.

    States stacked [T,4,16,384] fp16 (halves disk; upcast at train time). The
    ring is OFF -- the probe reads wm.state directly after each step.
    """
    wm = WorkingMemory(backbone, ring_capacity=0)
    traces: list[dict] = []
    t0 = time.time()
    for ci, stream in enumerate(chains):
        wm.reset()
        inputs = torch.from_numpy(np.stack(stream)).unsqueeze(1).to(device)  # [T,1,384]
        states = []
        for t in range(len(stream)):
            wm.step(inputs[t])
            st = torch.stack([s.detach().to("cpu") for s in wm.state])  # [4,1,16,384]
            states.append(st.squeeze(1))  # [4,16,384]
        states_t = torch.stack(states).to(torch.float16)  # [T,4,16,384]
        traces.append({"inputs": inputs.squeeze(1).to(torch.float32), "states": states_t})
        if (ci + 1) % 50 == 0:
            log(f"  traced {ci+1}/{len(chains)} chains ({time.time()-t0:.1f}s)")
    return traces


def save_traces(traces, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(traces, path)
    mb = path.stat().st_size / 1e6
    log(f"  saved {len(traces)} chain traces -> {path} ({mb:.1f} MB)")


def load_traces(path: Path) -> list[dict]:
    return torch.load(path, weights_only=False)


# ── (i,t) pair sampling ──

def sample_pairs(traces, k_max: int, state_rep: str, seed: int = 0):
    """For each chain, for each anchor i, pair with t=i+k for k in [1,k_max]
    (bounded by chain length). Returns arrays of (state_t [state_dim], u_i [384],
    k). ``state_rep``: "pooled" -> per-d_state-channel mean per layer [4,384]=1536;
    "full" -> flattened [4,16,384]=4096."""
    rng = np.random.default_rng(seed)
    S, U, K = [], [], []
    for cid, tr in enumerate(traces):
        T = tr["states"].shape[0]
        ins = tr["inputs"].numpy()  # [T,384] fp32
        sts = tr["states"].to(torch.float32)  # [T,4,16,384] fp16->fp32
        if state_rep == "pooled":
            sts = sts.mean(dim=2).reshape(T, -1).numpy()  # [T,1536]
        else:
            sts = sts.reshape(T, -1).numpy()  # [T,4096]
        for i in range(T):
            for k in range(1, k_max + 1):
                t = i + k
                if t >= T:
                    break
                S.append(sts[t])
                U.append(ins[i])
                K.append(k)
    S = np.asarray(S, dtype=np.float32)
    U = np.asarray(U, dtype=np.float32)
    K = np.asarray(K, dtype=np.int64)
    return S, U, K


# ── AUC (Mann-Whitney, average ranks for ties) ──

def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    y = labels.astype(np.int64)
    s = scores.astype(np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ── ridge regression (closed-form, no overfit) ──
#
# Both the recovery decoder D and the probe P are ridge regressors solved via
# augmented least squares: min ||Xw - y||^2 + lam ||w||^2  =>  solve
# [X; sqrt(lam) I] w = [y; 0]. This is well-posed when D > N (the state is
# 4096-dim and we have ~1-2k pairs) and has NO epoch/overfit tuning -- the right
# shape for a probe whose output is a single honest gate number.

def standardize(X_tr, X_va):
    mu = X_tr.mean(0, keepdims=True)
    sd = X_tr.std(0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (X_tr - mu) / sd, (X_va - mu) / sd


def ridge_fit(X_tr, Y_tr, lam: float):
    """Solve min ||Xw - y||^2 + lam||w||^2 via the normal equations + LU
    (faster than SVD-lstsq on the wide state matrix). Well-posed for D > N
    because lam*I makes the Gram positive-definite. Standardized features are
    assumed (so a single lam is meaningful)."""
    gram = X_tr.T @ X_tr
    gram += lam * np.eye(gram.shape[0], dtype=gram.dtype)
    rhs = X_tr.T @ (Y_tr if Y_tr.ndim == 2 else Y_tr.reshape(-1, 1))
    w = np.linalg.solve(gram, rhs)
    return w.flatten() if Y_tr.ndim == 1 else w


# ── stage 2: recovery decoder D(state_t) -> u_i (ridge) ──

def fit_decoder(S_tr, U_tr, S_va, U_va, lam: float):
    Ss_tr, Ss_va = standardize(S_tr.astype(np.float64), S_va.astype(np.float64))
    U_tr64 = U_tr.astype(np.float64)
    U_va64 = U_va.astype(np.float64)
    w = ridge_fit(Ss_tr, U_tr64, lam)  # [state_dim, 384]
    e_tr = ((Ss_tr @ w - U_tr64) ** 2).mean(axis=1)
    e_va = ((Ss_va @ w - U_va64) ** 2).mean(axis=1)
    return e_tr, e_va


# ── stage 3: probe P(state_t, u_i) -> e_hat (ridge) ──

def fit_probe(S_tr, U_tr, e_tr, S_va, U_va, lam: float):
    X_tr = np.hstack([S_tr, U_tr]).astype(np.float64)
    X_va = np.hstack([S_va, U_va]).astype(np.float64)
    Xs_tr, Xs_va = standardize(X_tr, X_va)
    w = ridge_fit(Xs_tr, e_tr.astype(np.float64), lam)
    return Xs_tr @ w, Xs_va @ w  # ehat_tr, ehat_va


# ── main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chains", type=int, default=200)
    ap.add_argument("--k-max", type=int, default=K_MAX)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--retrace", action="store_true", help="regenerate traces even if cached")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-auc", type=float, default=0.75)
    ap.add_argument("--lam", type=float, default=10.0, help="ridge penalty (on standardized features)")
    ap.add_argument("--state-rep", choices=["pooled", "full"], default="pooled",
                    help="state representation: pooled (per-d_state mean, 1536-dim, fast) or full (4096-dim)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # stage 1: traces
    if args.retrace or not TRACE_PATH.exists():
        log(f"stage 1: building traces from {SEQ_PATH} (max {args.max_chains} chains)")
        chains = load_chains(SEQ_PATH, args.max_chains)
        log(f"  loaded {len(chains)} forward chains (lens "
            f"min/med/max={min(len(c) for c in chains)}/{sorted(len(c) for c in chains)[len(chains)//2]}/{max(len(c) for c in chains)})")
        backbone = load_backbone(BACKBONE_PATH, config=BackboneConfig(), device=args.device)
        traces = build_traces(chains, backbone, device)
        save_traces(traces, TRACE_PATH)
    else:
        log(f"stage 1: loading cached traces -> {TRACE_PATH}")
        traces = load_traces(TRACE_PATH)

    # sample (i,t) pairs and split by chain (80/20, no leakage)
    log(f"stage 2/3: sampling (i,t) pairs with k_max={args.k_max}")
    n_ch = len(traces)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_ch)
    n_va = max(1, n_ch // 5)
    va_idx = set(perm[:n_va].tolist())
    tr_idx = [i for i in range(n_ch) if i not in va_idx]
    S_tr, U_tr, K_tr = sample_pairs([traces[i] for i in tr_idx], args.k_max, args.state_rep, seed=args.seed)
    S_va, U_va, K_va = sample_pairs([traces[i] for i in sorted(va_idx)], args.k_max, args.state_rep, seed=args.seed)
    log(f"  train pairs={len(S_tr)}  val pairs={len(S_va)}  state_rep={args.state_rep} state_dim={S_tr.shape[1]}")

    # stage 2: decoder -> ground-truth forgetting e(i,t) (ridge, no overfit)
    log("stage 2: fitting recovery decoder D(state_t) -> u_i (ridge)")
    t0 = time.time()
    e_tr, e_va = fit_decoder(S_tr, U_tr, S_va, U_va, lam=args.lam)
    log(f"  D fit in {time.time()-t0:.1f}s; train e mean={e_tr.mean():.4f} val e mean={e_va.mean():.4f}")

    # decay curve: mean e vs k (the forgetting signal must grow with k)
    print("\n  decay curve (mean reconstruction error e(i,t) by k=t-i):")
    for k in range(1, args.k_max + 1):
        m_tr = e_tr[K_tr == k].mean() if (K_tr == k).any() else float("nan")
        m_va = e_va[K_va == k].mean() if (K_va == k).any() else float("nan")
        print(f"    k={k}: train e={m_tr:.4f}  val e={m_va:.4f}  n_va={int((K_va==k).sum())}")

    # binary "forgotten" label = e above PER-SPLIT median (balanced within each
    # split, so AUC has both classes regardless of train/val error-distribution skew)
    y_tr = (e_tr > np.median(e_tr)).astype(np.int64)
    y_va = (e_va > np.median(e_va)).astype(np.int64)
    log(f"  val pos frac={y_va.mean():.3f} (per-split median threshold)")

    # stage 3: probe P -> e_hat; AUC is the gate (ridge, no overfit)
    log("stage 3: fitting probe P(state_t, u_i) -> e_hat(i,t) (ridge)")
    t0 = time.time()
    ehat_tr, ehat_va = fit_probe(S_tr, U_tr, e_tr, S_va, U_va, lam=args.lam)
    auc_va = auc(ehat_va, y_va)
    auc_tr = auc(ehat_tr, y_tr)
    log(f"  P fit in {time.time()-t0:.1f}s")

    # baselines: k alone is a free monotonic-forgetting baseline the probe must
    # beat to add information. (Ranking by e itself is a TRIVIAL 1.0 -- the label
    # IS the thresholded e -- so it is not a meaningful upper bound; omitted.)
    auc_k_va = auc(K_va.astype(np.float64), y_va)

    print("\n=== Phase 0a gate ===")
    print(f"  probe P AUC (val)      = {auc_va:.4f}   [GATE: >= {args.gate_auc}]")
    print(f"  probe P AUC (train)    = {auc_tr:.4f}")
    print(f"  free baseline: k       = {auc_k_va:.4f}   (monotonic-forgetting-in-k; P must beat this)")
    beats_k = auc_va > auc_k_va
    meets_gate = auc_va >= args.gate_auc
    decision = "GO" if (meets_gate and beats_k) else "NO-GO"
    print(f"  P beats k-baseline: {beats_k};  meets AUC gate: {meets_gate}")
    print(f"  decision: {decision}")
    if decision == "GO":
        print("  -> recoverability head (Phase 2b) viable; labels = D's e(i,t).")
    else:
        print("  -> if AUC poor with no obvious fix: shrink JST to fixed-interval refresh,")
        print("     drop salience + recoverability + graduation heads. If AUC poor AND")
        print("     discretization suspected, try a Mamba2 backend swap (not Mamba3).")


if __name__ == "__main__":
    main()