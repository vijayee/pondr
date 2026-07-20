"""Shared STRM trace + ridge utilities (Phase 2b recoverability + 2c latent-dynamics).

Both STRM read-out heads train on the Phase 0a traces -- per-chain recurrent
state ``state_t`` [T, 4, 16, 384] and the input stream ``u_t`` [T, 384] that
produced it -- so they share the same trace loading, state-representation
projections, pair sampling, ridge solver, and AUC. Centralizing them here is
the alternative to copy-pasting the probe logic into two trainers.

The two heads differ in *which* state representation and *which* labels they
consume (2c: last-layer mean-over-d_state [384], label = the next state
z_{t+1}; 2b: pooled all-layers mean-over-d_state [1536], label = the decoder's
reconstruction error e(i,t)). The projections + samplers below cover both.

These are pure-numpy fit helpers (torch is only used to load the ``.pt`` trace
file) -- the closed-form ridge is solved via the Gram matrix + LU, not SVD-
lstsq (too slow on the wide state matrix, see the Phase 0a probe), and there
is NO training loop here. The 2c head is this ridge fit baked into an
``nn.Linear``; the 2b head is a trained MLP that must beat the ridge baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── state representation projections ──
#
# The recurrent state is 4 per-layer tensors of [d_state=16, d_model=384].
# Two projections are load-bearing (validated by the Phase 0a/0b probes):
#
#   "last"   -- last layer only, mean over d_state -> [384]. N > D here, so the
#               2c ridge fit is determined (the 0b probe found the 1536-dim
#               "pooled" fit was underdetermined at N=957 < D=1537 and gave a
#               false NO-GO; the 384-dim "last" rep gave R^2=0.297 GO).
#   "pooled" -- all 4 layers, mean over d_state -> [4*384]=1536. The 0a probe
#               showed this rep carries the recoverability signal (AUC 0.810).

STATE_DIM_LAST = 384
STATE_DIM_POOLED = 4 * 384


def state_rep_last(states: torch.Tensor) -> np.ndarray:
    """Last layer, mean over d_state -> ``[T, 384]`` float32 numpy.

    ``states`` is ``[T, 4, 16, 384]`` (fp16 on disk; upcast here). Indexing the
    last layer and averaging over the d_state axis is the representation the
    Phase 0b probe found learnable (linear R^2=0.297 over the constant-mean
    baseline); the 1536-dim pooled rep was underdetermined at this N.
    """
    st = states.to(torch.float32)            # [T, 4, 16, 384]
    return st[:, -1].mean(dim=1).numpy()       # [T, 384]


def state_rep_pooled(states: torch.Tensor) -> np.ndarray:
    """All layers, mean over d_state -> ``[T, 1536]`` float32 numpy.

    The Phase 0a recoverability probe's representation (AUC 0.810). Mean over
    the d_state channel per layer then flatten the 4 layers.
    """
    st = states.to(torch.float32)             # [T, 4, 16, 384]
    T = st.shape[0]
    return st.mean(dim=2).reshape(T, -1).numpy()   # [T, 1536]


def load_traces(path) -> list[dict]:
    """Load the Phase 0a trace file -> list of per-chain ``{inputs, states}``.

    ``path`` may be a str/Path. Each chain is ``{"inputs": [T, 384] fp32,
    "states": [T, 4, 16, 384] fp16}`` (see ``scripts/generate_strm_traces.py``).
    ``weights_only=False`` because the file is the user's own probe/generator
    output (a list of plain dicts of tensors, no code) -- same contract as
    ``load_backbone``.
    """
    return torch.load(path, weights_only=False)


# ── pair sampling ──

def split_chains(n_chains: int, val_fraction: float, seed: int = 0):
    """Deterministic 80/20 chain split -> (train_idx, val_idx) as sorted lists.

    Split by CHAIN (not by pair) so no (i,t) pair leaks its own state_t into both
    train and val. Mirrors the Phase 0a/0b probes. Returns the indices into the
    trace list, not the traces themselves.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_chains)
    n_val = max(1, int(round(n_chains * val_fraction)))
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())
    return train_idx, val_idx


def sample_transitions(traces: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten consecutive (z_t, z_{t+1}) transitions across chains -> (Zt, Ztp1).

    Used by the 2c latent-dynamics head. ``z`` here is whatever state
    representation the caller already projected each chain to (the caller maps
    ``traces`` -> per-chain ``[T, state_dim]`` first, then passes those lists).
    Returns ``([N, state_dim], [N, state_dim])`` float32.
    """
    Zt, Ztp1 = [], []
    for z in traces:
        for t in range(len(z) - 1):
            Zt.append(z[t])
            Ztp1.append(z[t + 1])
    return (np.asarray(Zt, dtype=np.float32),
            np.asarray(Ztp1, dtype=np.float32))


def sample_recoverability_pairs(
    traces: list[dict],
    k_max: int,
    state_rep: str = "pooled",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (state_t, anchor u_i, lag k) triples for the 2b recoverability head.

    For each chain, for each anchor i, pair with t = i + k for k in [1, k_max]
    (bounded by chain length). ``state_t`` is the recurrent state at the later
    step; ``u_i`` is the input embedding of the earlier anchor; ``k`` is the
    lookback lag (the free monotonic-forgetting baseline the head must beat).

    The sampling is exhaustive over (i, k) -- no randomness -- so there is no
    seed. ``state_rep`` selects "pooled" (1536, the 0a probe rep) or "last".

    Returns:
        ``(S [N, state_dim], U [N, 384], K [N])`` float32/float32/int64.
    """
    S, U, K = [], [], []
    project = state_rep_pooled if state_rep == "pooled" else state_rep_last
    for tr in traces:
        T = tr["states"].shape[0]
        ins = tr["inputs"].numpy()          # [T, 384] fp32
        sts = project(tr["states"])          # [T, state_dim]
        for i in range(T):
            for k in range(1, k_max + 1):
                t = i + k
                if t >= T:
                    break
                S.append(sts[t])
                U.append(ins[i])
                K.append(k)
    return (np.asarray(S, dtype=np.float32),
            np.asarray(U, dtype=np.float32),
            np.asarray(K, dtype=np.int64))


# ── ridge regression (closed-form, baked to operate on RAW features) ──

def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge regression via the Gram matrix + LU, returning weights that
    operate on RAW (unstandardized) features.

    Solves ``min ||X w - y||^2 + lam ||w[:D]||^2`` with an unpenalized bias:
    features are internally standardized (so a single ``lam`` is meaningful
    across dimensions of differing scale), a non-standardized bias column is
    appended, the Gram matrix penalizes ONLY the feature block (not the
    intercept), and the solution is then baked back to raw-feature space so
    the returned ``(W, b)`` satisfy ``pred = X @ W + b`` on the ORIGINAL ``X``.

    This is cleaner than the Phase 0a/0b probe's ridge, which standardized the
    WHOLE augmented matrix (bias column included) -- a column of all-ones has
    zero std, so the ``< 1e-6 -> 1.0`` guard zeroed the standardized bias and
    the fit silently dropped its intercept. The probe still worked (the SSM
    state is roughly centered), but the dropped-intercept was a latent defect;
    this version keeps a real intercept and should fit at least as well.

    Args:
        X: ``[N, D]`` raw features (float64 internally for numerical stability).
        Y: ``[N]`` or ``[N, K]`` targets.
        lam: ridge penalty (on the standardized features).

    Returns:
        ``(W [D, K], b [K])`` such that ``X @ W + b`` predicts ``Y`` (``K=1``
        when ``Y`` is 1-D; the returned arrays are squeezed to ``[D]``/``[]``).
    """
    Xf = np.asarray(X, dtype=np.float64)
    yf = np.asarray(Y, dtype=np.float64)
    two_d = yf.ndim == 2
    if not two_d:
        yf = yf.reshape(-1, 1)
    N, D = Xf.shape
    mu = Xf.mean(axis=0, keepdims=True)            # [1, D]
    sd = Xf.std(axis=0, keepdims=True)             # [1, D]
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xs = (Xf - mu) / sd                            # [N, D] standardized features
    Xa = np.hstack([Xs, np.ones((N, 1))])          # [N, D+1] bias NOT standardized
    gram = Xa.T @ Xa
    # Penalize the feature block only (a ridge penalty on the intercept would
    # shrink the mean prediction -- not what we want).
    gram[:D, :D] += lam * np.eye(D, dtype=np.float64)
    rhs = Xa.T @ yf                                # [D+1, K]
    Wstd = np.linalg.solve(gram, rhs)             # [D+1, K]
    A = Wstd[:D]                                   # [D, K] on standardized features
    b_std = Wstd[D]                                # [K]
    # Bake standardization back to raw features:
    #   pred = Xs @ A + b_std = (X - mu)/sd @ A + b_std = X @ (A/sd) + (b_std - (mu/sd)@A)
    W_raw = A / sd.T                              # [D, K]  (sd is [1,D] -> [D,1] broadcast)
    b_raw = b_std - ((mu / sd).reshape(-1)) @ A   # [K]
    if not two_d:
        # squeeze to ([D], scalar) so a 1-D target yields a 1-D weight vector
        # and a 0-D bias (matches the caller's `X @ W + b` with b a scalar).
        return W_raw.reshape(-1), b_raw.reshape(-1)[0]
    return W_raw, b_raw


# ── AUC (Mann-Whitney, average ranks for ties) ──

def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic with average ranks for ties.

    Returns ``nan`` if either class is empty (no information / degenerate).
    No sklearn dependency -- this is the same implementation the Phase 0a/0b
    probes used. ``scores`` are continuous predictions (higher = more positive);
    ``labels`` are 0/1.
    """
    y = labels.astype(np.int64)
    s = scores.astype(np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # 1-indexed average rank
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))