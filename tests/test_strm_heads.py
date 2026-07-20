"""STRM Phase 2 head tests (2c latent-dynamics + the shared trace/ridge utils).

CPU-only, no backbone, no embedder, no WaveDB. The closed-form fit is
exercised on a SYNTHETIC trace whose last-layer state follows known linear
dynamics -- so the ridge fit recovers the dynamics and the gate (R^2 +
surprise-AUC) clears. This validates the fit math, the head's project/
predict/surprise, the checkpoint round-trip, and the shared sampling/AUC/
ridge helpers the 2b trainer will also use.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.subconscious.latent_dynamics_head import (
    LatentDynamicsHead,
    STATE_DIM,
    load_latent_dynamics_head,
)
from src.subconscious.training.latent_dynamics_training import (
    LatentDynamicsTrainingConfig,
    fit_latent_dynamics,
)
from src.subconscious.training.strm_traces import (
    auc,
    ridge_fit,
    sample_recoverability_pairs,
    sample_transitions,
    split_chains,
    state_rep_last,
    state_rep_pooled,
)


# ── synthetic trace with known linear last-layer dynamics ──

def _synthetic_traces(n_chains=12, length=20, active=16, seed=0):
    """Chains whose LAST-layer mean-over-d_state follows z_{t+1}=A z_t + b.

    The head only reads the last layer's mean over d_state (state_rep_last),
    so we set the last layer's 16 channels ALL equal to z_t (mean = z_t
    exactly) and the other 3 layers to zero. z lives in the first ``active``
    dims (the rest are zero -- zero-variance dims that the ridge
    standardization guard collapses, so the fit operates on ``active`` dims
    with N >> active). This makes the closed-form recovery well-posed at
    small N (the real 384-dim, all-active fit needs N > 384).

    Each active dim gets its OWN contraction ``a_d`` and bias ``b_d`` so the
    dims stay statistically independent: with a single shared a/b every dim
    converges to the same fixed point via the same trajectory, producing a
    common mode that correlates the dims (~0.83 cross-correlation) and makes
    the ridge shrink the diagonal (0.9 -> 0.76) even though the per-dim OLS
    slope is exactly 0.9. Per-dim a/b removes the common mode and the ridge
    recovers the dynamics cleanly.
    """
    rng = np.random.default_rng(seed)
    traces = []
    a = np.zeros(STATE_DIM, dtype=np.float32)
    b = np.zeros(STATE_DIM, dtype=np.float32)
    a[:active] = 0.5 + 0.03 * np.arange(active)      # 0.50 .. 0.95 (|a|<1, stable)
    b[:active] = 0.05 + 0.01 * np.arange(active)     # distinct per-dim fixed points
    for _ in range(n_chains):
        z = np.zeros(STATE_DIM, dtype=np.float32)
        z[:active] = rng.standard_normal(active).astype(np.float32)
        states = []
        inputs = []
        for _ in range(length):
            # last layer: broadcast z over the 16 d_state channels -> mean = z
            last = np.tile(z, (16, 1))                       # [16, 384]
            layer = np.zeros((4, 16, STATE_DIM), dtype=np.float32)
            layer[-1] = last
            states.append(layer)
            inputs.append(z.copy())                            # u_t = z_t (arbitrary)
            z = a * z + b                                      # known per-dim linear dynamics
        states_t = torch.from_numpy(np.stack(states))        # [T,4,16,384]
        inputs_t = torch.from_numpy(np.stack(inputs))        # [T,384]
        traces.append({"inputs": inputs_t, "states": states_t})
    return traces


# ── state-rep projections ──

def test_state_rep_last_is_last_layer_mean_over_d_state():
    traces = _synthetic_traces(n_chains=1, length=5, active=8, seed=1)
    z = state_rep_last(traces[0]["states"])                  # [T, 384]
    assert z.shape == (5, STATE_DIM)
    # the last layer's 16 channels are all z_t, so the mean over d_state is z_t.
    # only the first 8 dims are nonzero.
    assert np.allclose(z[:, 8:], 0.0, atol=1e-6)
    assert np.any(z[:, :8] != 0.0)


def test_state_rep_pooled_is_all_layers_mean_over_d_state():
    traces = _synthetic_traces(n_chains=1, length=4, active=8, seed=2)
    z = state_rep_pooled(traces[0]["states"])                # [T, 1536]
    assert z.shape == (4, 4 * STATE_DIM)
    # state_rep_pooled CONCATENATES the 4 per-layer means (mean over d_state),
    # it does NOT average across layers. 3 of 4 layers are zero, so the first
    # three 384-dim blocks are zero and the LAST block == the last-layer mean
    # == z_t exactly (all 16 d_state channels equal z_t).
    assert np.allclose(z[:, :3 * STATE_DIM], 0.0, atol=1e-6)
    last_block = z[:, -STATE_DIM:]
    expected = state_rep_last(traces[0]["states"])
    assert np.allclose(last_block, expected, atol=1e-6)


# ── split + sampling ──

def test_split_chains_disjoint_and_covered():
    tr, va = split_chains(20, 0.2, seed=0)
    assert set(tr).isdisjoint(set(va))
    assert sorted(tr + va) == list(range(20))
    assert len(va) == 4


def test_sample_transitions_consecutive():
    traces = _synthetic_traces(n_chains=2, length=5, active=8, seed=3)
    z = [state_rep_last(t["states"]) for t in traces]
    zt, ztp1 = sample_transitions(z)
    assert zt.shape == (2 * 4, STATE_DIM)        # (length-1) per chain
    assert ztp1.shape == zt.shape
    # z_{t+1} = a_d * z_t + b_d on the active dims -> check the relation holds.
    a = 0.5 + 0.03 * np.arange(8)
    b = 0.05 + 0.01 * np.arange(8)
    assert np.allclose(ztp1[:, :8], a * zt[:, :8] + b, atol=1e-5)


def test_sample_recoverability_pairs_shapes_and_lag():
    traces = _synthetic_traces(n_chains=2, length=6, active=8, seed=4)
    S, U, K = sample_recoverability_pairs(traces, k_max=3, state_rep="pooled")
    assert S.shape[1] == 4 * STATE_DIM
    assert U.shape[1] == STATE_DIM
    assert K.shape == (S.shape[0],)
    assert K.min() == 1 and K.max() <= 3


# ── ridge_fit baking ──

def test_ridge_fit_recovers_linear_on_raw_features():
    # Y = X @ W_true + b_true, N > D, low lam -> near-exact recovery.
    rng = np.random.default_rng(0)
    N, D, K = 200, 8, 3
    X = rng.standard_normal((N, D)).astype(np.float64)
    W_true = rng.standard_normal((D, K))
    b_true = rng.standard_normal(K)
    Y = X @ W_true + b_true
    W, b = ridge_fit(X, Y, lam=1e-3)
    assert W.shape == (D, K)
    assert b.shape == (K,)
    # operates on RAW X (standardization baked back in):
    pred = X @ W + b
    assert np.allclose(pred, Y, atol=1e-3)
    assert np.allclose(W, W_true, atol=1e-2)
    assert np.allclose(b, b_true, atol=1e-2)


def test_ridge_fit_1d_target_squeezed():
    rng = np.random.default_rng(1)
    N, D = 100, 4
    X = rng.standard_normal((N, D)).astype(np.float64)
    w_true = rng.standard_normal(D)
    b_true = 0.5
    y = X @ w_true + b_true
    W, b = ridge_fit(X, y, lam=1e-3)
    assert W.shape == (D,)
    assert b.shape == ()
    assert np.allclose(X @ W + b, y, atol=1e-3)


# ── AUC ──

def test_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1])
    assert auc(scores, labels) == 1.0


def test_auc_ties_get_average_ranks():
    # one negative (0.0, rank1) and one positive (1.0, rank4) are clear; the
    # middle two tie at 0.5 (one neg, one pos) and share average ranks 2.5/2.5.
    # AUC = (sum_pos_ranks - n_pos*(n_pos+1)/2)/(n_pos*n_neg)
    #     = ((2.5 + 4) - 2*3/2)/(2*2) = 3.5/4 = 0.875.
    scores = np.array([0.0, 0.5, 0.5, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert auc(scores, labels) == 0.875


def test_auc_empty_class_is_nan():
    assert math.isnan(auc(np.array([0.1, 0.2]), np.array([0, 0])))


# ── LatentDynamicsHead module ──

def test_head_project_predict_surprise_shapes():
    head = LatentDynamicsHead()
    # state_tensors: list of 4 per-layer [1,16,384] tensors (the live WM shape).
    state_tensors = [torch.zeros(1, 16, STATE_DIM) for _ in range(4)]
    state_tensors[-1] = torch.randn(1, 16, STATE_DIM)
    z = head.project(state_tensors)                           # [1, 384]
    assert z.shape == (1, STATE_DIM)
    pred = head.predict(z)                                     # [1, 384]
    assert pred.shape == (1, STATE_DIM)
    z_next = torch.randn(1, STATE_DIM)
    s = head.surprise(z, z_next)                               # [1]
    assert s.shape == (1,)
    assert s.item() >= 0.0


def test_head_project_rejects_bad_ndim():
    head = LatentDynamicsHead()
    with pytest.raises(ValueError, match="unsupported ndim"):
        head.project([torch.randn(STATE_DIM)])                # 1-D, not 2/3


# ── closed-form fit + checkpoint round-trip ──

def test_fit_latent_dynamics_clears_gate_on_synthetic(tmp_path):
    traces = _synthetic_traces(n_chains=12, length=20, active=16, seed=7)
    cfg = LatentDynamicsTrainingConfig(
        lam=1.0, r2_gate=0.15, surprise_auc_gate=0.70,
        val_fraction=0.2, seed=0, checkpoint_dir=str(tmp_path),
    )
    result = fit_latent_dynamics(traces, cfg)
    assert result["go"] is True
    assert result["r2"] > 0.5
    assert result["surprise_auc"] > 0.9
    # best.pt == final.pt (one closed-form fit, no epoch selection)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    final = torch.load(tmp_path / "final.pt", weights_only=False)
    assert best["r2"] == final["r2"]
    assert best["surprise_auc"] == final["surprise_auc"]
    assert torch.equal(best["linear"]["linear.weight"], final["linear"]["linear.weight"])
    assert best["go"] is True


def test_fit_latent_dynamics_round_trips_through_loader(tmp_path):
    traces = _synthetic_traces(n_chains=12, length=20, active=16, seed=8)
    cfg = LatentDynamicsTrainingConfig(lam=1.0, checkpoint_dir=str(tmp_path))
    result = fit_latent_dynamics(traces, cfg)
    head = load_latent_dynamics_head(str(tmp_path / "best.pt"), device="cpu")
    # the loaded head's predict must match the fit's linear (A z_t + b) on raw z.
    z = torch.from_numpy(state_rep_last(traces[0]["states"])[0]).unsqueeze(0)
    with torch.no_grad():
        pred_loaded = head.predict(z).squeeze(0).numpy()
    # rebuild the fit's raw prediction from the saved payload's weight/bias
    payload = torch.load(tmp_path / "best.pt", weights_only=False)
    W = payload["linear"]["linear.weight"].numpy()       # [384, 384] (= A^T)
    b = payload["linear"]["linear.bias"].numpy()         # [384]
    pred_expected = (z.numpy() @ W.T + b)[0]
    assert np.allclose(pred_loaded, pred_expected, atol=1e-5)
    # and surprise on the true next state is small (the fit recovered the dynamics)
    z_next = torch.from_numpy(state_rep_last(traces[0]["states"])[1]).unsqueeze(0)
    assert head.surprise(z, z_next).item() < result["surprise_auc"]  # small residual


def test_fit_latent_dynamics_raises_on_too_few_chains(tmp_path):
    traces = _synthetic_traces(n_chains=3, length=10, active=8, seed=9)
    cfg = LatentDynamicsTrainingConfig(checkpoint_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match=">=5 chains"):
        fit_latent_dynamics(traces, cfg)


def test_load_latent_dynamics_head_state_dim_mismatch_raises(tmp_path):
    # save a 384-dim head, then hand-build a "checkpoint" claiming state_dim=64
    traces = _synthetic_traces(n_chains=8, length=15, active=16, seed=11)
    fit_latent_dynamics(traces, LatentDynamicsTrainingConfig(
        lam=1.0, checkpoint_dir=str(tmp_path)))
    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    ckpt["state_dim"] = 64                                    # lie about the dim
    torch.save(ckpt, tmp_path / "mismatch.pt")
    # A 64-dim head can't absorb a 384-dim Linear state_dict -> torch raises a
    # size-mismatch (strict=False still rejects shape mismatches).
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_latent_dynamics_head(str(tmp_path / "mismatch.pt"), device="cpu")