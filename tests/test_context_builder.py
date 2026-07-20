"""STRM Phase 3 Step 2: ContextBuilder module tests.

CPU-only, no backbone, no embedder, no WaveDB. Validates the builder's
mechanics: predict shape, permutation-equivariance (no positional encoding --
ERAG candidates are shuffled, so a slot-permutation must permute the top-m
selection correspondingly), the 2a-relevance bias actually driving selection
(``lambda_r`` nonzero + ``r`` monotonic -> top-m follows ``r``), the
checkpoint round-trip (``from_state_dict`` reloads a strict-matching head), and
the ``predict`` ``m``-clamping (``m > K`` does not crash).

The loss-decreases-on-synthetic and gate-passes-on-synthetic tests are added in
Step 3 (``tests/test_context_builder.py`` is extended there) once the trainer's
loss + gate functions exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.context_builder import (
    ContextBuilder,
    D_HEAD,
    DOC_DIM,
    NUM_HEADS,
    QUERY_DIM,
    SLOT_DIM,
    TOP_M,
)


def _rand_inputs(K: int = 7, seed: int = 0):
    """Build deterministic ``slots_y [K,256]``, ``slots_doc_emb [K,384]``,
    ``query_emb [384]``, ``r [K]``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    slots_y = torch.randn(K, SLOT_DIM, generator=g, dtype=torch.float32)
    slots_doc_emb = torch.randn(K, DOC_DIM, generator=g, dtype=torch.float32)
    query_emb = torch.randn(QUERY_DIM, generator=g, dtype=torch.float32)
    r = torch.rand(K, generator=g, dtype=torch.float32)
    return slots_y, slots_doc_emb, query_emb, r


# ── predict shape ────────────────────────────────────────────────────────────

def test_predict_shape_and_order():
    torch.manual_seed(0)
    b = ContextBuilder()
    Y, D, Q, r = _rand_inputs(K=8)
    idx, scores = b.predict(Y, D, Q, r, m=5)
    assert len(idx) == 5
    assert scores.shape == (5,)
    # descending order
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    # indices in range
    assert all(0 <= i < 8 for i in idx)
    assert len(set(idx)) == 5                  # no duplicates


def test_predict_m_clamped_to_K():
    """m > K must not crash; returns K indices."""
    b = ContextBuilder(top_m=20)
    Y, D, Q, r = _rand_inputs(K=3)
    idx, scores = b.predict(Y, D, Q, r, m=20)
    assert len(idx) == 3
    assert len(set(idx)) == 3


def test_predict_m_zero_returns_empty():
    b = ContextBuilder()
    Y, D, Q, r = _rand_inputs(K=4)
    idx, scores = b.predict(Y, D, Q, r, m=0)
    assert idx == []
    assert scores.shape == (0,)


def test_logits_shape():
    b = ContextBuilder()
    Y, D, Q, r = _rand_inputs(K=6)
    s = b.logits(Y, D, Q, r)
    assert s.shape == (6,)


# ── permutation equivariance ─────────────────────────────────────────────────

def test_permutation_equivariance():
    """Permuting (slots_y, slots_doc_emb, r) together permutes the scores
    identically -- there is NO positional encoding, so the builder is
    permutation-equivariant. This is required: ERAG candidates are shuffled, so
    the builder must not depend on slot order."""
    torch.manual_seed(1)
    b = ContextBuilder()
    K = 9
    Y, D, Q, r = _rand_inputs(K=K, seed=42)
    s_orig = b.logits(Y, D, Q, r)
    perm = torch.randperm(K, generator=torch.Generator().manual_seed(7))
    s_perm = b.logits(Y[perm], D[perm], Q, r[perm])
    # s_perm[i] should equal s_orig[perm[i]] (the score follows the slot)
    assert torch.allclose(s_perm, s_orig[perm], atol=1e-5), (
        f"builder is NOT permutation-equivariant:\n s_perm={s_perm}\n "
        f"s_orig[perm]={s_orig[perm]}"
    )


def test_permutation_topm_follows_slots():
    """The top-m SELECTION permutes with the slots -- if slot j was selected
    before, the slot now at the permuted position of j is selected after."""
    torch.manual_seed(2)
    b = ContextBuilder(top_m=3)
    K = 8
    Y, D, Q, r = _rand_inputs(K=K, seed=11)
    idx_orig, _ = b.predict(Y, D, Q, r, m=3)
    perm = torch.randperm(K, generator=torch.Generator().manual_seed(3))
    # inverse perm: where did original slot i land?
    inv = torch.argsort(perm)
    idx_perm, _ = b.predict(Y[perm], D[perm], Q, r[perm], m=3)
    # idx_perm should be {inv[i] for i in idx_orig} as a set
    assert set(idx_perm) == set(int(inv[i]) for i in idx_orig)


# ── r-bias active ─────────────────────────────────────────────────────────────

def test_r_bias_active_when_slot_features_zero():
    """With ``slots_y = 0`` and ``slots_doc_emb = 0``, the only per-slot signal
    is ``lambda_r * r`` (``W_k``/``W_doc``/``W_v`` produce identical vectors for
    all slots, so cross-attention outputs identical rows -> ``q . h`` is a slot-
    independent constant). With ``lambda_r`` init = 1.0 > 0, the scores are
    strictly increasing in ``r``, so ``predict`` returns slots in r-descending
    order. Asserts the 2a relevance signal is a LIVE bias, not discarded."""
    b = ContextBuilder()   # lambda_init=1.0
    K = 6
    Y = torch.zeros(K, SLOT_DIM, dtype=torch.float32)
    D = torch.zeros(K, DOC_DIM, dtype=torch.float32)
    Q = torch.randn(QUERY_DIM, dtype=torch.float32)
    r = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2, 0.8], dtype=torch.float32)
    s = b.logits(Y, D, Q, r)
    # s - bias - (q.h const) = lambda_r * r  -> s is affine increasing in r
    # i.e. s[i] > s[j] iff r[i] > r[j]
    order_r = torch.argsort(r, descending=True).tolist()
    order_s = torch.argsort(s, descending=True).tolist()
    assert order_s == order_r, (
        f"r-bias not active: r-order {order_r} != s-order {order_s} "
        f"(lambda_r={float(b.lambda_r)})"
    )
    # also via predict
    idx, _ = b.predict(Y, D, Q, r, m=K)
    assert idx == order_r


def test_r_bias_inactive_when_lambda_zero():
    """Sanity: with ``lambda_r = 0`` and zero slot features, all scores are
    equal -> top-m is arbitrary (not r-ordered). Confirms the previous test is
    specifically exercising the ``lambda_r`` path, not a coincidence."""
    b = ContextBuilder(lambda_init=0.0)
    K = 6
    Y = torch.zeros(K, SLOT_DIM, dtype=torch.float32)
    D = torch.zeros(K, DOC_DIM, dtype=torch.float32)
    Q = torch.randn(QUERY_DIM, dtype=torch.float32)
    r = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.2, 0.8], dtype=torch.float32)
    s = b.logits(Y, D, Q, r)
    # all scores equal -> r has no effect
    assert torch.allclose(s, s[0].expand_as(s), atol=1e-6)


# ── checkpoint round-trip ─────────────────────────────────────────────────────

def test_from_state_dict_roundtrip(tmp_path):
    """``from_state_dict`` reloads a strict-matching builder; ``predict`` is
    byte-identical before and after the round-trip."""
    torch.manual_seed(5)
    b = ContextBuilder()
    Y, D, Q, r = _rand_inputs(K=5, seed=99)
    idx_before, scores_before = b.predict(Y, D, Q, r, m=3)

    sd = b.state_dict()
    b2 = ContextBuilder.from_state_dict(sd)
    idx_after, scores_after = b2.predict(Y, D, Q, r, m=3)
    assert idx_after == idx_before
    assert torch.allclose(scores_before, scores_after, atol=1e-6)


def test_from_state_dict_rejects_dim_mismatch():
    """A state_dict built for a different d_head must fail strictly, not
    silently mis-wire."""
    b = ContextBuilder(d_head=128)
    sd = b.state_dict()
    # build a builder expecting d_head=64 -> shape mismatch
    with pytest.raises(RuntimeError):
        ContextBuilder.from_state_dict(sd, d_head=64)


def test_load_context_builder_roundtrip(tmp_path):
    """Full ``load_context_builder`` path: write a checkpoint dict, load it,
    predict matches the in-memory builder."""
    from src.subconscious.context_builder import load_context_builder
    torch.manual_seed(6)
    b = ContextBuilder()
    Y, D, Q, r = _rand_inputs(K=5, seed=31)
    idx_before, scores_before = b.predict(Y, D, Q, r, m=3)
    ckpt = {
        "head": b.state_dict(),
        "slot_dim": SLOT_DIM,
        "doc_dim": DOC_DIM,
        "query_dim": QUERY_DIM,
        "d_head": D_HEAD,
        "num_heads": NUM_HEADS,
        "top_m": TOP_M,
        "mean_coverage": 0.9,
        "heuristic_mean_coverage": 0.33,
        "go": True,
        "epoch": 12,
    }
    p = tmp_path / "builder.pt"
    torch.save(ckpt, p)
    b2 = load_context_builder(str(p), device="cpu")
    idx_after, scores_after = b2.predict(Y, D, Q, r, m=3)
    assert idx_after == idx_before
    assert torch.allclose(scores_before, scores_after, atol=1e-6)
    # the gate provenance fields round-tripped
    assert b2.top_m == TOP_M


# ── device coercion / broadcast ───────────────────────────────────────────────

def test_query_emb_2d_accepted():
    """``query_emb`` may be [query_dim] OR [1, query_dim]; both yield the same
    scores (the query is broadcast one row per slot)."""
    b = ContextBuilder()
    Y, D, Q, r = _rand_inputs(K=4, seed=8)
    s1 = b.logits(Y, D, Q, r)
    s2 = b.logits(Y, D, Q.unsqueeze(0), r)
    assert torch.allclose(s1, s2, atol=1e-6)


def test_misaligned_slot_rows_raise():
    b = ContextBuilder()
    Y = torch.randn(5, SLOT_DIM, dtype=torch.float32)
    D = torch.randn(4, DOC_DIM, dtype=torch.float32)
    Q = torch.randn(QUERY_DIM, dtype=torch.float32)
    r = torch.rand(5, dtype=torch.float32)
    with pytest.raises(ValueError):
        b.logits(Y, D, Q, r)


def test_r_length_mismatch_raises():
    b = ContextBuilder()
    Y, D, Q, _ = _rand_inputs(K=5, seed=2)
    r = torch.rand(4, dtype=torch.float32)
    with pytest.raises(ValueError):
        b.logits(Y, D, Q, r)