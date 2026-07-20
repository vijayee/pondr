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
from src.subconscious.recoverability_head import (
    ANCHOR_DIM,
    INPUT_DIM,
    STATE_DIM_POOLED,
    RecoverabilityHead,
    load_recoverability_head,
)
from src.subconscious.training.latent_dynamics_training import (
    LatentDynamicsTrainingConfig,
    fit_latent_dynamics,
)
from src.subconscious.training.recoverability_training import (
    RecoverabilityTrainingConfig,
    fit_recoverability,
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


# ── 2b recoverability head ──
#
# A synthetic whose pooled last-layer state is a LEAKY SUM of NON-NEGATIVE
# sparse anchors (state_last = c*prev + u_t, c=0.7). Forgetting is real: each
# anchor u_i is encoded with weight c^k (k=t-i) plus interference from later
# anchors, so the decoder D's reconstruction error e(i,t) grows with k. The
# within-k variation that lets P beat the k-baseline comes from ||u_i||^2:
# non-negative inputs make ||u_i||^2 monotonically rankable by a linear
# functional (a linear P with positive weights on u_i ranks ||u_i||^2), and
# the leaky-sum dynamics give a clean forgetting signal. (With signed inputs
# a linear P cannot rank the quadratic ||u_i||^2 and the gate fails -- the
# non-negativity is load-bearing, mirroring how the real 0a probe's e happens
# to be linearly-rankable over the real embedding distribution.)

def _synthetic_recoverability_traces(
    n_chains=20, length=30, active=32, c=0.7, n_anchors=5, seed=0,
):
    """Pooled last layer = leaky sum of non-negative sparse anchors; 3 layers zero."""
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(n_chains):
        us = [np.zeros(ANCHOR_DIM, dtype=np.float32) for _ in range(length)]
        anchor_pos = sorted(rng.choice(
            length, size=min(n_anchors, length), replace=False).tolist())
        for p in anchor_pos:
            u = np.zeros(ANCHOR_DIM, dtype=np.float32)
            u[:active] = rng.uniform(0.0, 1.0, active).astype(np.float32)
            us[p] = u
        state_last = np.zeros(ANCHOR_DIM, dtype=np.float32)
        states, inputs = [], []
        for t in range(length):
            state_last = c * state_last + us[t]
            layer = np.zeros((4, 16, ANCHOR_DIM), dtype=np.float32)
            # all 16 d_state channels = state_last -> pooled last block == state_last
            layer[-1] = np.tile(state_last, (16, 1))
            states.append(layer)
            inputs.append(us[t].copy())
        states_t = torch.from_numpy(np.stack(states))     # [T,4,16,384]
        inputs_t = torch.from_numpy(np.stack(inputs))     # [T,384]
        traces.append({"inputs": inputs_t, "states": states_t})
    return traces


# ── RecoverabilityHead module ──

def test_recoverability_head_project_state_shape():
    head = RecoverabilityHead()
    # 4 per-layer [1,16,384] state tensors (the live WM shape).
    state_tensors = [torch.zeros(1, 16, ANCHOR_DIM) for _ in range(4)]
    state_tensors[-1] = torch.rand(1, 16, ANCHOR_DIM)
    z = head.project_state(state_tensors)                  # [1, 1536]
    assert z.shape == (1, STATE_DIM_POOLED)
    # 3 of 4 layers zero -> first three 384-blocks zero, last block == last mean.
    assert np.allclose(z[0, :3 * ANCHOR_DIM].numpy(), 0.0, atol=1e-6)


def test_recoverability_head_predict_shape():
    head = RecoverabilityHead()
    s = torch.rand(1, STATE_DIM_POOLED)
    u = torch.rand(1, ANCHOR_DIM)
    e = head.predict(s, u)                                   # [1, 1]
    assert e.shape == (1, 1)
    # 1-D inputs broadcast -> [1, 1]
    e1 = head.predict(torch.rand(STATE_DIM_POOLED), torch.rand(ANCHOR_DIM))
    assert e1.shape == (1, 1)


def test_recoverability_head_project_rejects_bad_ndim():
    head = RecoverabilityHead()
    state_tensors = [torch.zeros(1, 16, ANCHOR_DIM) for _ in range(4)]
    state_tensors[-1] = torch.randn(ANCHOR_DIM)             # 1-D, not 2/3
    with pytest.raises(ValueError, match="unsupported ndim"):
        head.project_state(state_tensors)


def test_recoverability_head_project_rejects_wrong_count():
    head = RecoverabilityHead()
    # 3 layers, not 4
    with pytest.raises(ValueError, match="expected 4 per-layer"):
        head.project_state([torch.zeros(1, 16, ANCHOR_DIM) for _ in range(3)])


# ── closed-form fit + checkpoint round-trip ──

def test_fit_recoverability_clears_gate_on_synthetic(tmp_path):
    traces = _synthetic_recoverability_traces(
        n_chains=20, length=30, active=32, c=0.7, n_anchors=5, seed=7)
    cfg = RecoverabilityTrainingConfig(
        k_max=6, lam=10.0, gate_auc=0.75, val_fraction=0.2,
        seed=0, checkpoint_dir=str(tmp_path),
    )
    result = fit_recoverability(traces, cfg)
    assert result["go"] is True
    # P must clear the AUC gate AND beat the free k-baseline.
    assert result["ridge_auc"] > 0.75
    assert result["ridge_auc"] > result["k_auc"]
    assert result["k_auc"] < 0.6        # k alone is a weak baseline here
    # best.pt == final.pt (one closed-form fit, no epoch selection)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    final = torch.load(tmp_path / "final.pt", weights_only=False)
    assert best["ridge_auc"] == final["ridge_auc"]
    assert best["go"] is True
    assert torch.equal(best["linear"]["linear.weight"],
                       final["linear"]["linear.weight"])
    assert best["state_dim_pooled"] == STATE_DIM_POOLED
    assert best["anchor_dim"] == ANCHOR_DIM
    assert best["input_dim"] == INPUT_DIM


def test_fit_recoverability_round_trips_through_loader(tmp_path):
    traces = _synthetic_recoverability_traces(
        n_chains=20, length=30, active=32, c=0.7, n_anchors=5, seed=8)
    cfg = RecoverabilityTrainingConfig(k_max=6, lam=10.0,
                                       checkpoint_dir=str(tmp_path))
    fit_recoverability(traces, cfg)
    head = load_recoverability_head(str(tmp_path / "best.pt"), device="cpu")
    # The loaded head's predict must match the fit's ridge (X @ W + b) on raw
    # [state ; anchor].
    tr = traces[0]
    S, U, _ = sample_recoverability_pairs([tr], k_max=cfg.k_max,
                                          state_rep="pooled")
    payload = torch.load(tmp_path / "best.pt", weights_only=False)
    W = payload["linear"]["linear.weight"].numpy().reshape(-1)   # [1920]
    b = payload["linear"]["linear.bias"].numpy()[0]
    X = np.hstack([S, U]).astype(np.float32)
    expected = X @ W + b
    with torch.no_grad():
        got = head.predict(torch.from_numpy(S), torch.from_numpy(U)).squeeze(-1).numpy()
    assert np.allclose(got, expected, atol=1e-4)


def test_fit_recoverability_raises_on_too_few_chains(tmp_path):
    traces = _synthetic_recoverability_traces(n_chains=3, length=12, seed=9)
    cfg = RecoverabilityTrainingConfig(checkpoint_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match=">=5 chains"):
        fit_recoverability(traces, cfg)


def test_load_recoverability_head_dim_mismatch_raises(tmp_path):
    # save a 1920-input head, then hand-build a checkpoint claiming a smaller
    # state_dim_pooled -> the constructed Linear's in_features won't match.
    traces = _synthetic_recoverability_traces(
        n_chains=12, length=20, active=32, seed=11)
    fit_recoverability(traces, RecoverabilityTrainingConfig(
        k_max=6, lam=10.0, checkpoint_dir=str(tmp_path)))
    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    ckpt["state_dim_pooled"] = 64                            # lie about the dim
    torch.save(ckpt, tmp_path / "mismatch.pt")
    # A head built for 64+384=448 inputs can't absorb the 1920-input Linear
    # state_dict -> torch raises a size-mismatch.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_recoverability_head(str(tmp_path / "mismatch.pt"), device="cpu")


# ── 2a relevance head ──
#
# A synthetic that models the REAL relevance task: the gold slot's doc_emb is
# SIMILAR to the query_emb (high cosine), the negatives' doc_embs are
# near-orthogonal to the query. slots_y is PURE NOISE -- mirroring the real
# finding that the relevance signal lives in the raw bge doc embedding, NOT in
# the frozen routing-trained backbone's y_t readout (see relevance_head.py
# docstring + scripts/_scratch/_probe_relevance_bge_baseline). This is a
# SIMILARITY task (gold close to query), which the bilinear head learns (it can
# represent cosine) and an additive MLP cannot -- matching the real finding
# (MLP tops out at 0.60 on ERAG while bge cosine clears at 1.0). Validates the
# TRAINER MACHINERY (the fit loop, the top-3 recall + Wilson CI gate, the
# checkpoint round-trip) on a clean signal, mirroring how the 2b/2c tests use
# a clean synthetic.

from src.subconscious.relevance_head import (  # noqa: E402
    DOC_DIM as REL_DOC_DIM,
    PROJ_DIM as REL_PROJ_DIM,
    QUERY_DIM as REL_QUERY_DIM,
    SLOT_DIM as REL_SLOT_DIM,
    RelevanceHead,
    load_relevance_head,
)
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    evaluate_relevance,
    fit_relevance,
    load_relevance_traces,
)


def _synthetic_relevance_traces(n_queries=40, k=15, seed=0):
    """Gold doc_emb ~ query_emb (high cosine); negs near-orthogonal; slot_y noise.

    Models real relevance: the gold slot is the doc whose embedding is similar
    to the query. 1 gold/query. The bilinear head learns the cosine similarity
    and ranks gold in the top-3."""
    rng = np.random.default_rng(seed)
    traces = []
    for qi in range(n_queries):
        # query direction; gold doc_emb is this direction + small noise (high
        # cosine with the query); negatives are random (near-orthogonal).
        q = rng.standard_normal(REL_QUERY_DIM).astype(np.float32)
        query_emb = q
        doc_embs = []
        slots_y = []
        labels = []
        # gold slot first (the trainer shuffles; order is irrelevant to the gate)
        doc_embs.append(q + 0.3 * rng.standard_normal(REL_DOC_DIM).astype(np.float32))
        slots_y.append(0.3 * rng.standard_normal(REL_SLOT_DIM).astype(np.float32))  # noise
        labels.append(1)
        for _ in range(k - 1):
            doc_embs.append(0.3 * rng.standard_normal(REL_DOC_DIM).astype(np.float32))
            slots_y.append(0.3 * rng.standard_normal(REL_SLOT_DIM).astype(np.float32))
            labels.append(0)
        traces.append({
            "query_id": f"q{qi}",
            "question": f"question {qi}",
            "category": "basic",
            "expected_doc_ids": [f"gold_{qi}"],
            "query_emb": torch.from_numpy(query_emb),
            "slots_y": torch.from_numpy(np.stack(slots_y)).float(),    # [k, 256] noise
            "slots_doc_emb": torch.from_numpy(np.stack(doc_embs)).float(),  # [k, 384] signal
            "source_ids": [f"gold_{qi}"] + [f"neg_{qi}_{j}" for j in range(k - 1)],
            "labels": torch.tensor(labels, dtype=torch.long),        # [k]
        })
    return traces


# ── RelevanceHead module ──

def test_relevance_head_predict_shape_and_range():
    head = RelevanceHead()
    slots = torch.randn(7, REL_SLOT_DIM)
    doc = torch.randn(7, REL_DOC_DIM)
    query = torch.randn(REL_QUERY_DIM)
    r = head.predict(slots, doc, query)                          # [7, 1]
    assert r.shape == (7, 1)
    # sigmoid -> [0, 1]
    assert bool((r >= 0.0).all()) and bool((r <= 1.0).all())
    # 1-D inputs broadcast -> [1, 1]
    r1 = head.predict(torch.randn(REL_SLOT_DIM), torch.randn(REL_DOC_DIM),
                      torch.randn(REL_QUERY_DIM))
    assert r1.shape == (1, 1)


def test_relevance_head_predict_rejects_bad_dims():
    head = RelevanceHead()
    doc = torch.randn(3, REL_DOC_DIM)
    # slot dim mismatch
    with pytest.raises(ValueError, match="slot_y dim"):
        head.predict(torch.randn(3, 64), doc, torch.randn(REL_QUERY_DIM))
    # doc dim mismatch
    with pytest.raises(ValueError, match="slot_doc_emb dim"):
        head.predict(torch.randn(3, REL_SLOT_DIM), torch.randn(3, 100),
                     torch.randn(REL_QUERY_DIM))
    # query dim mismatch
    with pytest.raises(ValueError, match="query dim"):
        head.predict(torch.randn(3, REL_SLOT_DIM), doc, torch.randn(100))
    # batch mismatch (slot batch != query batch, neither is 1)
    with pytest.raises(ValueError, match="incompatible"):
        head.predict(torch.randn(3, REL_SLOT_DIM), torch.randn(3, REL_DOC_DIM),
                     torch.randn(5, REL_QUERY_DIM))


def test_relevance_head_is_query_conditioned():
    # the SAME slot scores differently against different queries (the query is
    # an input, not a parameter) -- r differs across queries for one slot.
    head = RelevanceHead()
    slot = torch.randn(REL_SLOT_DIM)
    doc = torch.randn(REL_DOC_DIM)
    q_a = torch.randn(REL_QUERY_DIM)
    q_b = torch.randn(REL_QUERY_DIM)
    ra = head.predict(slot, doc, q_a).item()
    rb = head.predict(slot, doc, q_b).item()
    assert ra != rb


# ── load_relevance_traces ──

def test_load_relevance_traces_drops_no_gold(tmp_path):
    traces = _synthetic_relevance_traces(n_queries=4, k=5, seed=1)
    # inject a no-gold record (all-zero labels) -> dropped
    traces.append({
        "query_id": "qbad", "question": "x", "category": "basic",
        "expected_doc_ids": [],
        "query_emb": torch.randn(REL_QUERY_DIM),
        "slots_y": torch.randn(5, REL_SLOT_DIM),
        "slots_doc_emb": torch.randn(5, REL_DOC_DIM),
        "source_ids": [f"n{j}" for j in range(5)],
        "labels": torch.zeros(5, dtype=torch.long),
    })
    p = tmp_path / "traces.pt"
    torch.save(traces, p)
    loaded = load_relevance_traces(str(p))
    assert len(loaded) == 4                     # the no-gold record dropped


def test_load_relevance_traces_rejects_missing_doc_emb(tmp_path):
    # a stale trace file (pre-doc_emb) is rejected with a regenerate pointer,
    # not silently fed to a head that would have no signal.
    traces = _synthetic_relevance_traces(n_queries=3, k=5, seed=2)
    for r in traces:
        del r["slots_doc_emb"]
    p = tmp_path / "traces.pt"
    torch.save(traces, p)
    with pytest.raises(RuntimeError, match="slots_doc_emb"):
        load_relevance_traces(str(p))


# ── closed-form fit + gate + checkpoint round-trip ──

def test_fit_relevance_clears_gate_on_synthetic(tmp_path):
    traces = _synthetic_relevance_traces(n_queries=60, k=15, seed=7)
    cfg = RelevanceTrainingConfig(
        epochs=20, gate_top3=0.6, gate_wilson_low=0.5,
        val_fraction=0.2, seed=0, checkpoint_dir=str(tmp_path),
        # the synthetic's similarity signal (gold doc_emb ~ query_emb) is strong
        # enough that the mild pos_weight_cap=3.0 suffices -- this exercises the
        # fit loop + gate + checkpoint machinery without the heavy 14.0 class
        # weight the moderate real signal needs (see relevance_training config).
        pos_weight_cap=3.0,
    )
    result = fit_relevance(traces, cfg)
    assert result["go"] is True
    assert result["best_pc"]["mean_top3_recall"] >= 0.6
    assert result["best_pc"]["hit_ci95"][0] > 0.5
    # best.pt written with the gate payload
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["slot_dim"] == REL_SLOT_DIM
    assert best["doc_dim"] == REL_DOC_DIM
    assert best["query_dim"] == REL_QUERY_DIM
    assert best["proj_dim"] == REL_PROJ_DIM
    assert best["go"] is True
    # final.pt written (last epoch)
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "train_log.json").exists()


def test_fit_relevance_round_trips_through_loader(tmp_path):
    traces = _synthetic_relevance_traces(n_queries=60, k=15, seed=8)
    cfg = RelevanceTrainingConfig(epochs=15, checkpoint_dir=str(tmp_path),
                                  pos_weight_cap=3.0)   # mild cap suffices on the strong synthetic
    fit_relevance(traces, cfg)
    head = load_relevance_head(str(tmp_path / "best.pt"), device="cpu")
    # the loaded head's predict matches a fresh head's predict on the same
    # state_dict (the loader rebuilds the shared-projection head + loads the
    # weights).
    rec = traces[0]
    r_loaded = head.predict(rec["slots_y"], rec["slots_doc_emb"],
                            rec["query_emb"]).squeeze(-1)
    # gold slot (index 0) should rank highest -- the loaded head retained the fit
    assert int(r_loaded.argmax().item()) == 0
    assert float(r_loaded[0].item()) > float(r_loaded[1:].max().item())


def test_fit_relevance_raises_on_too_few_queries(tmp_path):
    traces = _synthetic_relevance_traces(n_queries=2, k=5, seed=9)
    cfg = RelevanceTrainingConfig(checkpoint_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match=">=2 train queries"):
        fit_relevance(traces, cfg)


def test_load_relevance_head_dim_mismatch_raises(tmp_path):
    traces = _synthetic_relevance_traces(n_queries=20, k=10, seed=11)
    fit_relevance(traces, RelevanceTrainingConfig(
        epochs=10, checkpoint_dir=str(tmp_path)))
    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    ckpt["slot_dim"] = 64                                      # lie about the dim
    torch.save(ckpt, tmp_path / "mismatch.pt")
    # A head built with slot_dim=64 has yt_sidepath Linear(64,64), but the
    # state_dict was saved with Linear(256,64) -> torch raises a size-mismatch.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_relevance_head(str(tmp_path / "mismatch.pt"), device="cpu")