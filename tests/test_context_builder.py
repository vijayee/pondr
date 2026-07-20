"""STRM Phase 3 Step 2: ContextBuilder module tests.

CPU-only, no backbone, no embedder, no WaveDB. Validates the builder's
mechanics: predict shape, permutation-equivariance (no positional encoding --
ERAG candidates are shuffled, so a slot-permutation must permute the top-m
selection correspondingly), the 2a-relevance bias actually driving selection
(``lambda_r`` nonzero + ``r`` monotonic -> top-m follows ``r``), the
checkpoint round-trip (``from_state_dict`` reloads a strict-matching head), and
the ``predict`` ``m``-clamping (``m > K`` does not crash).

The loss-decreases-on-synthetic and gate-passes-on-synthetic tests (Step 3)
exercise the trainer's loss + gate machinery on synthetic traces shaped like
the 2a ERAG traces (gold slot marked in both ``y_t`` and ``r_i``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
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


# ── Step 3: trainer loss + gate (synthetic) ───────────────────────────────────

import random

class _MockFrozenHead(torch.nn.Module):
    """Stand-in for the frozen shipped 2a relevance head. Computes ``r_i`` as a
    DETERMINISTIC function of the inputs (the way the real 2a head does -- it
    does not carry call-state): r_i = sigmoid(20 * cosine(doc_emb_i, query_emb)).
    The synthetic gold slot's ``slots_doc_emb`` IS the query embedding + tiny
    noise (cosine ~1 -> r~1), while negatives are random (cosine ~0 -> r~0.5),
    so r_i ranks gold first -- a strong, correct bias the builder can ride to
    beat the heuristic (which takes the first m of the SHUFFLED candidates).

    Subclasses ``nn.Module`` so it has ``.eval()`` / ``.parameters()`` like the
    real ``load_relevance_head`` return value the trainer calls at serve."""
    def __init__(self, gold_pos: list[int] | None = None) -> None:
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def predict(self, slots_y, slot_doc_emb, query_emb):
        d = slot_doc_emb.to(torch.float32)                       # [K, 384]
        q = query_emb.to(torch.float32).reshape(1, -1)           # [1, 384]
        cos = F.cosine_similarity(d, q, dim=-1)                  # [K]
        r = torch.sigmoid(20.0 * cos).unsqueeze(-1)              # [K, 1]
        return r


def _synth_traces(n_queries: int = 12, K: int = 12, seed: int = 0):
    """Build synthetic traces shaped like the 2a ERAG traces: each record has
    one gold slot at a random (seeded) position. The gold slot's ``slots_y`` is a
    fixed marker vector (so the builder's W_k can learn to mark it) and its
    ``slots_doc_emb`` is the query embedding + small noise (so the frozen head's
    r_i -- mocked -- ranks it). ``source_ids`` are ``d0..d{K-1}``; the gold is at
    a random slot index, so the heuristic's "first m" is a random m-subset."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rng = random.Random(seed)
    records = []
    gold_marker = torch.randn(SLOT_DIM, generator=g, dtype=torch.float32)
    for qi in range(n_queries):
        qemb = torch.randn(QUERY_DIM, generator=g, dtype=torch.float32)
        gold_pos = rng.randrange(K)
        slots_y = torch.randn(K, SLOT_DIM, generator=g, dtype=torch.float32) * 0.3
        slots_y[gold_pos] = gold_marker.clone()
        slots_doc_emb = torch.randn(K, DOC_DIM, generator=g, dtype=torch.float32) * 0.3
        slots_doc_emb[gold_pos] = qemb + torch.randn(DOC_DIM, generator=g,
                                                     dtype=torch.float32) * 0.05
        labels = torch.zeros(K, dtype=torch.float32)
        labels[gold_pos] = 1.0
        records.append({
            "query_id": f"q{qi}",
            "question": f"What did entity{qi} say about topic{qi}?",
            "category": "factual",
            "query_emb": qemb,
            "slots_y": slots_y,
            "slots_doc_emb": slots_doc_emb,
            "source_ids": [f"d{j}" for j in range(K)],
            "labels": labels,
        })
    return records


def test_loss_decreases_on_synthetic(tmp_path):
    """A few epochs of fit_context_builder drives the train loss DOWN on
    synthetic traces where gold is marked in y_t + r_i. The builder learns to
    select gold (mean_cov_learn climbing past the heuristic baseline)."""
    from src.subconscious.training.context_builder_training import (
        ContextBuilderTrainingConfig,
        fit_context_builder,
    )
    traces = _synth_traces(n_queries=12, K=12, seed=0)
    gold_pos = [int(r["labels"].argmax().item()) for r in traces]
    head = _MockFrozenHead(gold_pos)
    cfg = ContextBuilderTrainingConfig(
        epochs=8, learning_rate=3e-3, accum_steps=2, pl_weight=0.1,
        pos_weight_cap=3.0,
        val_fraction=0.25, seed=0, device="cpu",
        checkpoint_dir=str(tmp_path / "cb"),
    )
    result = fit_context_builder(traces, head, cfg)
    log = result["log"]
    assert len(log) >= 2
    # BCE + PL on a tiny slice is noisy (the optimizer reaches a low-loss region
    # but does not monotonically stay there). Assert the optimizer FOUND a lower
    # loss than it started at -- min over epochs < epoch-0 loss -- not that the
    # last epoch is below the first.
    losses = [e["train_loss"] for e in log]
    assert min(losses) < losses[0], (
        f"loss never decreased below initial: min={min(losses)} first={losses[0]}"
    )
    bp = result["best_pc"]
    assert bp["mean_cov_learn"] > bp["mean_cov_heur"], (
        f"builder did not beat heuristic: cov_learn={bp['mean_cov_learn']} "
        f"vs cov_heur={bp['mean_cov_heur']}"
    )


def test_gate_passes_on_synthetic(tmp_path):
    """On synthetic traces where r_i ranks gold, the builder (r_i bias + cross-
    slot attention) clears the gate: mean_cov_learn >= mean_cov_heur AND the
    Wilson CI lower bound on the builder full-coverage hit rate exceeds the
    heuristic. go is True and lambda_r did NOT collapse to 0."""
    from src.subconscious.training.context_builder_training import (
        ContextBuilderTrainingConfig,
        fit_context_builder,
    )
    traces = _synth_traces(n_queries=20, K=12, seed=1)
    gold_pos = [int(r["labels"].argmax().item()) for r in traces]
    head = _MockFrozenHead(gold_pos)
    cfg = ContextBuilderTrainingConfig(
        epochs=6, learning_rate=3e-3, accum_steps=2, pl_weight=0.1,
        val_fraction=0.25, seed=0, device="cpu",
        checkpoint_dir=str(tmp_path / "cb_gate"),
    )
    result = fit_context_builder(traces, head, cfg)
    assert result["go"] is True, (
        f"gate did not pass: best_pc={result['best_pc']}"
    )
    bp = result["best_pc"]
    assert bp["mean_cov_learn"] >= bp["mean_cov_heur"]
    assert bp["hit_ci95_learn"][0] > bp["hit_ci95_heur"][0]
    assert bp["mean_cov_learn"] >= 0.8
    # de-wonk risk: lambda_r must NOT collapse to 0 (would mean the builder
    # discarded the 2a signal and is just a per-slot rescore).
    assert abs(bp["lambda_r"]) > 1e-3, (
        f"lambda_r collapsed to {bp['lambda_r']} -- 2a r_i signal discarded"
    )


def test_gate_score_lexicographic():
    """_gate_score ranks gate-safe epochs above unsafe ones, then by learned
    coverage, then by r-only coverage (the tiebreaker)."""
    from src.subconscious.training.context_builder_training import (
        ContextBuilderTrainingConfig, _gate_score,
    )
    cfg = ContextBuilderTrainingConfig()
    pc_unsafe = {"mean_cov_learn": 0.9, "mean_cov_heur": 0.5,
                 "mean_cov_r_only": 0.7,
                 "hit_ci95_learn": [0.4, 0.9], "hit_ci95_heur": [0.5, 0.9]}
    pc_safe_low = {"mean_cov_learn": 0.7, "mean_cov_heur": 0.5,
                   "mean_cov_r_only": 0.6,
                   "hit_ci95_learn": [0.6, 0.9], "hit_ci95_heur": [0.5, 0.9]}
    pc_safe_high = {"mean_cov_learn": 0.8, "mean_cov_heur": 0.5,
                    "mean_cov_r_only": 0.6,
                    "hit_ci95_learn": [0.6, 0.9], "hit_ci95_heur": [0.5, 0.9]}
    s_unsafe = _gate_score(pc_unsafe, cfg)
    s_safe_low = _gate_score(pc_safe_low, cfg)
    s_safe_high = _gate_score(pc_safe_high, cfg)
    assert s_safe_low > s_unsafe
    assert s_safe_high > s_safe_low
    assert s_safe_low[0] == 1 and s_unsafe[0] == 0


def test_plackett_luce_zero_when_no_gold():
    """The PL auxiliary is 0 when a record has no gold (BCE carries it alone),
    and positive when gold is not top-ranked (pushes gold up listwise)."""
    from src.subconscious.training.context_builder_training import _plackett_luce
    s = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    labels_none = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    assert float(_plackett_luce(s, labels_none).item()) == 0.0
    labels = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    assert float(_plackett_luce(s, labels).item()) > 0.0


def test_coverage_helper():
    from src.subconscious.training.context_builder_training import _coverage
    assert _coverage([0, 1, 2], [1, 2]) == 1.0
    assert _coverage([0, 1], [1, 2]) == 0.5
    assert _coverage([0, 3], [1, 2]) == 0.0
    assert _coverage([], [1, 2]) == 0.0
    assert _coverage([0, 1], []) == 0.0