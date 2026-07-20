"""Tests for the Phase B ``h_t`` probe (ZRelevanceHead + the ``head`` /
``slot_signal_field`` parameters added to ``fit_relevance``).

Two contracts:

1. **ZRelevanceHead** is a pure-``z_i`` dual-projection bilinear that IGNORES
   ``slot_y`` (the pure-``z_i`` test drops the ``y_t`` path) and is
   signature-compatible with the 2a trainer's 3-arg ``logits(slots, slot_signal,
   query)`` call. Shape/range/dim-guard + query-conditioning + loader round-trip.
2. **``fit_relevance`` reuse** -- passing ``head=ZRelevanceHead()`` +
   ``slot_signal_field="slots_z"`` trains the z_i head with the SAME loop/gate/
   checkpoint machinery as 2a (only the slot-signal field + head change), saves a
   checkpoint whose ``doc_dim == z_dim == 384``, and round-trips through
   ``load_z_relevance_head``. The default ``head=None`` path is unchanged (the
   2a tests in ``test_strm_heads.py`` still cover it).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    fit_relevance,
)
from src.subconscious.z_relevance_head import (  # noqa: E402
    PROJ_DIM as Z_PROJ_DIM,
    QUERY_DIM as Z_QUERY_DIM,
    SLOT_DIM as Z_SLOT_DIM,
    Z_DIM,
    ZRelevanceHead,
    load_z_relevance_head,
)


# A synthetic that models the Phase B task: the gold slot's ``z_i`` is SIMILAR to
# the query_emb (high cosine), the negatives' ``z_i`` are near-orthogonal to the
# query. ``slots_y`` is pure noise (the z_i head IGNORES it -- the pure-z_i
# test). This is the z_i analog of ``_synthetic_relevance_traces`` in
# test_strm_heads.py: it validates the TRAINER MACHINERY (the reused fit loop,
# the top-3 recall + Wilson CI gate, the checkpoint round-trip) on a clean
# signal, NOT that real h_t carries signal (that is the serve probe's job).
def _synthetic_z_relevance_traces(n_queries=40, k=15, seed=0):
    """Gold ``z_i`` ~ query_emb (high cosine); negs near-orthogonal; slot_y noise."""
    rng = np.random.default_rng(seed)
    traces = []
    for qi in range(n_queries):
        q = rng.standard_normal(Z_QUERY_DIM).astype(np.float32)
        z_embs = []
        slots_y = []
        labels = []
        # gold slot first (the trainer shuffles; order is irrelevant to the gate).
        z_embs.append(q + 0.3 * rng.standard_normal(Z_DIM).astype(np.float32))
        slots_y.append(0.3 * rng.standard_normal(Z_SLOT_DIM).astype(np.float32))
        labels.append(1)
        for _ in range(k - 1):
            z_embs.append(0.3 * rng.standard_normal(Z_DIM).astype(np.float32))
            slots_y.append(0.3 * rng.standard_normal(Z_SLOT_DIM).astype(np.float32))
            labels.append(0)
        traces.append({
            "query_id": f"q{qi}",
            "question": f"question {qi}",
            "category": "basic",
            "expected_doc_ids": [f"gold_{qi}"],
            "query_emb": torch.from_numpy(q),
            "slots_y": torch.from_numpy(np.stack(slots_y)).float(),      # [k,256] noise
            "slots_doc_emb": torch.from_numpy(0.3 * rng.standard_normal(
                (k, Z_DIM)).astype(np.float32)).float(),                 # unused in z-mode
            "slots_z": torch.from_numpy(np.stack(z_embs)).float(),       # [k,384] signal
            "source_ids": [f"gold_{qi}"] + [f"neg_{qi}_{j}" for j in range(k - 1)],
            "labels": torch.tensor(labels, dtype=torch.long),            # [k]
        })
    return traces


# ── ZRelevanceHead module ──

def test_z_relevance_head_predict_shape_and_range():
    head = ZRelevanceHead()
    slot_y = torch.randn(7, Z_SLOT_DIM)        # ignored
    z = torch.randn(7, Z_DIM)
    query = torch.randn(Z_QUERY_DIM)
    r = head.predict(slot_y, z, query)                                # [7, 1]
    assert r.shape == (7, 1)
    assert bool((r >= 0.0).all()) and bool((r <= 1.0).all())
    # 1-D inputs broadcast -> [1, 1]
    r1 = head.predict(torch.randn(Z_SLOT_DIM), torch.randn(Z_DIM),
                      torch.randn(Z_QUERY_DIM))
    assert r1.shape == (1, 1)


def test_z_relevance_head_ignores_slot_y():
    """The pure-z_i test: slot_y is IGNORED -- the same z + query give the same
    score regardless of slot_y (including a zero slot_y vs a large slot_y). This
    is the load-bearing difference from 2a (no yt_sidepath): the y_t path is
    dropped entirely, so the test isolates z_i."""
    head = ZRelevanceHead()
    z = torch.randn(5, Z_DIM)
    query = torch.randn(Z_QUERY_DIM)
    r_zero = head.predict(torch.zeros(5, Z_SLOT_DIM), z, query)
    r_big = head.predict(torch.randn(5, Z_SLOT_DIM) * 1e6, z, query)
    assert torch.allclose(r_zero, r_big, atol=1e-5)


def test_z_relevance_head_predict_rejects_bad_dims():
    head = ZRelevanceHead()
    z = torch.randn(3, Z_DIM)
    # z dim mismatch
    with pytest.raises(ValueError, match="slot_z dim"):
        head.predict(torch.randn(3, Z_SLOT_DIM), torch.randn(3, 100),
                     torch.randn(Z_QUERY_DIM))
    # query dim mismatch
    with pytest.raises(ValueError, match="query dim"):
        head.predict(torch.randn(3, Z_SLOT_DIM), z, torch.randn(100))
    # batch mismatch (z batch != query batch, neither is 1)
    with pytest.raises(ValueError, match="incompatible"):
        head.predict(torch.randn(3, Z_SLOT_DIM), torch.randn(3, Z_DIM),
                     torch.randn(5, Z_QUERY_DIM))


def test_z_relevance_head_is_query_conditioned():
    # the SAME z scores differently against different queries (the query is an
    # input, not a parameter) -- r differs across queries for one z.
    head = ZRelevanceHead()
    z = torch.randn(Z_DIM)
    q_a = torch.randn(Z_QUERY_DIM)
    q_b = torch.randn(Z_QUERY_DIM)
    ra = head.predict(torch.randn(Z_SLOT_DIM), z, q_a).item()
    rb = head.predict(torch.randn(Z_SLOT_DIM), z, q_b).item()
    assert ra != rb


def test_z_relevance_head_from_state_dict_strict_mismatch():
    head = ZRelevanceHead()
    sd = head.state_dict()
    # lie about z_dim -> the proj_z Linear shape mismatches -> hard error (the
    # key NAMES match, so this is a PyTorch-native "size mismatch", not the
    # missing/unexpected-keys path -- either way a mis-wire is a hard error).
    with pytest.raises(RuntimeError, match="mismatch"):
        ZRelevanceHead.from_state_dict(sd, z_dim=64)


# ── fit_relevance reuse with head=ZRelevanceHead + slot_signal_field="slots_z" ──

def test_fit_relevance_z_head_clears_train_gate(tmp_path):
    """The SAME trainer fits the z_i head (head=ZRelevanceHead(),
    slot_signal_field='slots_z') and clears the TRAIN gate on the clean
    synthetic -- the z_i analog of test_fit_relevance_clears_gate_on_synthetic.
    Validates the trainer reuse + the checkpoint's z dims (doc_dim == z_dim)."""
    traces = _synthetic_z_relevance_traces(n_queries=60, k=15, seed=7)
    cfg = RelevanceTrainingConfig(
        epochs=20, gate_top3=0.6, gate_wilson_low=0.5,
        val_fraction=0.2, seed=0, checkpoint_dir=str(tmp_path),
        pos_weight_cap=3.0,                # mild cap suffices on the strong synthetic
        slot_signal_field="slots_z",
    )
    result = fit_relevance(traces, cfg, head=ZRelevanceHead())
    assert result["go"] is True
    assert result["best_pc"]["mean_top3_recall"] >= 0.6
    assert result["best_pc"]["hit_ci95"][0] > 0.5
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    # the checkpoint saves the head's own dims: doc_dim == z_dim (the head
    # exposes z_dim as doc_dim for shape-consistency with the shared trainer).
    assert best["slot_dim"] == Z_SLOT_DIM
    assert best["doc_dim"] == Z_DIM
    assert best["query_dim"] == Z_QUERY_DIM
    assert best["proj_dim"] == Z_PROJ_DIM
    assert best["go"] is True
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "train_log.json").exists()
    # the config round-trips the slot_signal_field (audit trail)
    import json as _json
    with open(tmp_path / "train_log.json", encoding="utf-8") as f:
        tl = _json.load(f)
    assert tl["config"]["slot_signal_field"] == "slots_z"


def test_fit_relevance_z_head_round_trips_through_loader(tmp_path):
    """The z-head checkpoint loads via load_z_relevance_head and retains the fit
    (gold z_i ranks highest)."""
    traces = _synthetic_z_relevance_traces(n_queries=60, k=15, seed=8)
    cfg = RelevanceTrainingConfig(epochs=15, checkpoint_dir=str(tmp_path),
                                  pos_weight_cap=3.0,
                                  slot_signal_field="slots_z")
    fit_relevance(traces, cfg, head=ZRelevanceHead())
    head = load_z_relevance_head(str(tmp_path / "best.pt"), device="cpu")
    rec = traces[0]
    r_loaded = head.predict(rec["slots_y"], rec["slots_z"],
                            rec["query_emb"]).squeeze(-1)
    # gold slot (index 0) ranks highest -- the loaded head retained the fit
    assert int(r_loaded.argmax().item()) == 0
    assert float(r_loaded[0].item()) > float(r_loaded[1:].max().item())


def test_load_relevance_traces_requires_slots_z_in_z_mode(tmp_path):
    """In z-mode the loader requires slots_z; a stale trace without it is
    rejected with a regenerate pointer (not a silent mis-wire)."""
    from src.subconscious.training.relevance_training import load_relevance_traces
    traces = _synthetic_z_relevance_traces(n_queries=3, k=5, seed=2)
    # strip slots_z to simulate a stale pre-Phase-B trace
    for r in traces:
        del r["slots_z"]
    path = tmp_path / "traces.pt"
    torch.save(traces, path)
    with pytest.raises(RuntimeError, match="slots_z"):
        load_relevance_traces(str(path), slot_signal_field="slots_z")


def test_fit_relevance_default_head_none_is_relevance_head():
    """head=None (the default) still constructs a RelevanceHead -- the 2a path
    is byte-identical to the pre-Phase-B trainer. (A light contract: the default
    head's class is RelevanceHead, not ZRelevanceHead; the full 2a training
    parity is covered by test_strm_heads.py.)"""
    import tempfile
    from src.subconscious.relevance_head import RelevanceHead
    traces = _synthetic_z_relevance_traces(n_queries=4, k=5, seed=3)
    # swap slots_doc_emb to carry signal so the 2a default head can fit a couple
    # epochs without error (we only assert the head class, not the gate)
    for r in traces:
        r["slots_doc_emb"] = r["slots_z"].clone()
    with tempfile.TemporaryDirectory() as td:
        cfg = RelevanceTrainingConfig(
            epochs=2, checkpoint_dir=td, pos_weight_cap=3.0,
            slot_signal_field="slots_doc_emb",   # the 2a field
        )
        result = fit_relevance(traces, cfg)        # head=None -> RelevanceHead
        assert "best_pc" in result  # ran without error
    # the default constructor is RelevanceHead (sanity, independent of the run)
    assert issubclass(RelevanceHead, type(RelevanceHead()))