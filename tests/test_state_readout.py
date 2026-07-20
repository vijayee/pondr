"""Tests for the Phase 0b learned StateReadout + CompositeZHead.

Phase 0a ([[pondr-strm-phase0a-state-signal-readout]]) showed the SSM recurrent
state is NOT collapsed -- the flattened state varies 0.45-0.76x across docs --
but the FIXED mean-pool ``LatentDynamicsHead.project`` cancels the signal to
near-constant. Phase 0b inserts a LEARNED ``StateReadout`` before the existing
``ZRelevanceHead`` and trains both end-to-end via the reused ``fit_relevance``
with ``slot_signal_field="slots_h_raw"``.

Three contracts:

1. **StateReadout** maps ``[dim_in] -> [384]`` (Linear or 2-layer MLP), casts to
   fp32, and ``from_state_dict`` infers Linear-vs-MLP from the state_dict keys.
2. **CompositeZHead** wraps readout + z-head, exposes the
   ``slot_dim``/``doc_dim``(=dim_in)/``query_dim``/``proj_dim`` attrs the shared
   trainer's checkpoint reads, IGNORES ``slot_y`` (the pure-z_i test holds), and
   round-trips through ``load_composite_z_head``.
3. **``fit_relevance`` reuse** -- ``head=CompositeZHead`` +
   ``slot_signal_field="slots_h_raw"`` trains both modules end-to-end with the
   SAME loop/gate/checkpoint machinery as 2a / the Phase B z-head, clears the
   TRAIN gate on a clean synthetic where the gold slot's raw state is a
   recoverable linear encoding of a query-aligned doc vector, and the checkpoint
   saves ``doc_dim == dim_in`` (so the loader rebuilds the readout). The MLP
   variant clears it too.

These validate the TRAINER MACHINERY on a clean signal, NOT that real h_raw
carries query-relevance (that is GATE 0b's job on the real ERAG traces).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.subconscious.state_readout import (  # noqa: E402
    DEFAULT_DIM_IN,
    CompositeZHead,
    StateReadout,
    load_composite_z_head,
)
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    fit_relevance,
    load_relevance_traces,
)
from src.subconscious.z_relevance_head import (  # noqa: E402
    PROJ_DIM as Z_PROJ_DIM,
    QUERY_DIM as Z_QUERY_DIM,
    SLOT_DIM as Z_SLOT_DIM,
    Z_DIM,
)


# A fixed random encoding matrix shared across all traces -- models the SSM's
# (deterministic, one-step) encoding of a doc into the raw state. The gold
# slot's doc vector aligns with the query; negatives' doc vectors are random.
# h_raw = M @ doc_vec is a LINEAR, full-column-rank encoding of the 384-d doc
# vector into dim_in dims, so a Linear StateReadout can invert it -> recover the
# doc vector -> the z-head scores it against the query -> gold ranks highest.
# This validates the readout+trainer machinery; the real h_raw signal is GATE 0b.
def _synthetic_h_raw_traces(n_queries=40, k=15, dim_in=DEFAULT_DIM_IN, seed=0):
    """Gold ``h_raw`` encodes a query-aligned doc vec; negs encode random vecs."""
    rng = np.random.default_rng(seed)
    # Full-column-rank encoding: dim_in x 384 (dim_in > 384 -> overcomplete).
    M = rng.standard_normal((dim_in, Z_DIM)).astype(np.float32)
    # Per-slot encoding noise -- makes the encoding LOSSY so the readout recovers
    # the doc vector only approximately. A noiseless encoding is perfectly
    # invertible -> the readout recovers gold ~= query -> sigmoid saturates to
    # 1.0 and a chance-high negative ties, breaking the margin contract for a
    # reason that has nothing to do with the loader. The lossy encoding keeps
    # scores in a non-saturated band (r_pos ~0.8) while the gold still clears
    # the gate and ranks above the negatives.
    ENC_NOISE = 0.4
    traces = []
    for qi in range(n_queries):
        q = rng.standard_normal(Z_QUERY_DIM).astype(np.float32)
        h_raws = []
        slots_y = []
        labels = []
        # gold slot first (the trainer shuffles; order is irrelevant to the gate).
        gold_doc = q + 0.3 * rng.standard_normal(Z_DIM).astype(np.float32)
        h_raws.append(M @ gold_doc
                      + ENC_NOISE * rng.standard_normal(dim_in).astype(np.float32))
        slots_y.append(0.3 * rng.standard_normal(Z_SLOT_DIM).astype(np.float32))
        labels.append(1)
        for _ in range(k - 1):
            neg_doc = 0.3 * rng.standard_normal(Z_DIM).astype(np.float32)
            h_raws.append(M @ neg_doc
                          + ENC_NOISE * rng.standard_normal(dim_in).astype(np.float32))
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
                (k, Z_DIM)).astype(np.float32)).float(),                 # unused here
            "slots_z": torch.from_numpy(0.3 * rng.standard_normal(
                (k, Z_DIM)).astype(np.float32)).float(),                 # unused here
            "slots_h_raw": torch.from_numpy(np.stack(h_raws)).float(),   # [k, dim_in] signal
            "source_ids": [f"gold_{qi}"] + [f"neg_{qi}_{j}" for j in range(k - 1)],
            "labels": torch.tensor(labels, dtype=torch.long),            # [k]
        })
    return traces


# ── StateReadout module ──

def test_state_readout_linear_shape():
    ro = StateReadout(dim_in=DEFAULT_DIM_IN)
    assert ro.hidden is None
    x = torch.randn(7, DEFAULT_DIM_IN)
    y = ro(x)
    assert y.shape == (7, Z_DIM)
    # 1-D input broadcasts -> [1, 384]
    assert ro(torch.randn(DEFAULT_DIM_IN)).shape == (1, Z_DIM)


def test_state_readout_mlp_shape():
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=256)
    assert ro.hidden == 256
    assert ro(torch.randn(4, DEFAULT_DIM_IN)).shape == (4, Z_DIM)


def test_state_readout_casts_fp16_to_fp32():
    ro = StateReadout(dim_in=DEFAULT_DIM_IN)
    x = torch.randn(3, DEFAULT_DIM_IN, dtype=torch.float16)
    y = ro(x)
    assert y.dtype == torch.float32            # the raw state is fp16 from Phase A


def test_state_readout_from_state_dict_infers_linear():
    ro = StateReadout(dim_in=DEFAULT_DIM_IN)
    sd = ro.state_dict()
    assert "net.2.weight" not in sd           # Linear -> no MLP layer-2
    ro2 = StateReadout.from_state_dict(sd, dim_in=DEFAULT_DIM_IN)
    assert ro2.hidden is None
    # round-trips the weights
    assert torch.allclose(ro2.net.weight, ro.net.weight)


def test_state_readout_from_state_dict_infers_mlp():
    ro = StateReadout(dim_in=DEFAULT_DIM_IN, hidden=128)
    sd = ro.state_dict()
    assert "net.2.weight" in sd               # MLP -> layer-2 present
    ro2 = StateReadout.from_state_dict(sd, dim_in=DEFAULT_DIM_IN)
    assert ro2.hidden == 128


def test_state_readout_from_state_dict_strict_mismatch():
    ro = StateReadout(dim_in=DEFAULT_DIM_IN)
    sd = ro.state_dict()
    # lie about dim_in -> the Linear weight shape mismatches -> hard error
    with pytest.raises(RuntimeError, match="mismatch"):
        StateReadout.from_state_dict(sd, dim_in=999)


# ── CompositeZHead module ──

def test_composite_z_head_predict_shape_and_range():
    head = CompositeZHead(dim_in=DEFAULT_DIM_IN)
    slot_y = torch.randn(7, Z_SLOT_DIM)       # ignored
    h_raw = torch.randn(7, DEFAULT_DIM_IN)
    query = torch.randn(Z_QUERY_DIM)
    r = head.predict(slot_y, h_raw, query)
    assert r.shape == (7, 1)
    assert bool((r >= 0.0).all()) and bool((r <= 1.0).all())


def test_composite_z_head_exposes_dims():
    head = CompositeZHead(dim_in=DEFAULT_DIM_IN)
    assert head.doc_dim == DEFAULT_DIM_IN     # the readout input dim
    assert head.slot_dim == Z_SLOT_DIM
    assert head.query_dim == Z_QUERY_DIM
    assert head.proj_dim == Z_PROJ_DIM


def test_composite_z_head_ignores_slot_y():
    """The pure-z_i test holds: slot_y is IGNORED -- the same h_raw + query give
    the same score regardless of slot_y (the composite drops the y_t path)."""
    head = CompositeZHead(dim_in=DEFAULT_DIM_IN)
    h_raw = torch.randn(5, DEFAULT_DIM_IN)
    query = torch.randn(Z_QUERY_DIM)
    r_zero = head.predict(torch.zeros(5, Z_SLOT_DIM), h_raw, query)
    r_big = head.predict(torch.randn(5, Z_SLOT_DIM) * 1e6, h_raw, query)
    assert torch.allclose(r_zero, r_big, atol=1e-4)


# ── fit_relevance reuse with head=CompositeZHead + slot_signal_field="slots_h_raw" ──

def test_fit_relevance_composite_linear_clears_train_gate(tmp_path):
    """The SAME trainer fits the composite (Linear readout + z-head) with
    slot_signal_field='slots_h_raw' and clears the TRAIN gate on the clean
    synthetic. Validates the trainer reuse + the checkpoint's dims
    (doc_dim == dim_in)."""
    traces = _synthetic_h_raw_traces(n_queries=60, k=15, seed=7)
    cfg = RelevanceTrainingConfig(
        epochs=25, gate_top3=0.6, gate_wilson_low=0.5,
        val_fraction=0.2, seed=0, checkpoint_dir=str(tmp_path),
        pos_weight_cap=3.0,                # mild cap suffices on the strong synthetic
        slot_signal_field="slots_h_raw",
    )
    result = fit_relevance(traces, cfg, head=CompositeZHead(dim_in=DEFAULT_DIM_IN))
    assert result["go"] is True
    assert result["best_pc"]["mean_top3_recall"] >= 0.6
    assert result["best_pc"]["hit_ci95"][0] > 0.5
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["slot_dim"] == Z_SLOT_DIM
    assert best["doc_dim"] == DEFAULT_DIM_IN     # the readout input dim
    assert best["query_dim"] == Z_QUERY_DIM
    assert best["proj_dim"] == Z_PROJ_DIM
    assert best["go"] is True
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "train_log.json").exists()
    import json as _json
    with open(tmp_path / "train_log.json", encoding="utf-8") as f:
        tl = _json.load(f)
    assert tl["config"]["slot_signal_field"] == "slots_h_raw"


def test_fit_relevance_composite_mlp_clears_train_gate(tmp_path):
    """The MLP readout variant also trains end-to-end + clears the gate (exercises
    the 2-layer MLP path + the from_state_dict MLP inference at load time). The
    MLP has more capacity than the Linear readout so it needs a smaller hidden
    (128) + a gentle epoch count to avoid overfitting the ~48 training positives
    and collapsing to "low relevance everywhere"."""
    traces = _synthetic_h_raw_traces(n_queries=80, k=15, seed=11)
    cfg = RelevanceTrainingConfig(
        epochs=15, gate_top3=0.6, gate_wilson_low=0.5,
        val_fraction=0.2, seed=0, checkpoint_dir=str(tmp_path),
        pos_weight_cap=3.0, slot_signal_field="slots_h_raw",
    )
    result = fit_relevance(traces, cfg, head=CompositeZHead(dim_in=DEFAULT_DIM_IN,
                                                            hidden=128))
    assert result["go"] is True
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["doc_dim"] == DEFAULT_DIM_IN
    # the saved state_dict has the MLP layer-2 key -> the loader infers MLP
    assert "readout.net.2.weight" in best["head"]


def test_fit_relevance_composite_round_trips_through_loader(tmp_path):
    """The composite checkpoint loads via load_composite_z_head and retains the
    fit (gold h_raw ranks highest)."""
    traces = _synthetic_h_raw_traces(n_queries=60, k=15, seed=8)
    cfg = RelevanceTrainingConfig(epochs=15, checkpoint_dir=str(tmp_path),
                                  pos_weight_cap=3.0,
                                  slot_signal_field="slots_h_raw")
    fit_relevance(traces, cfg, head=CompositeZHead(dim_in=DEFAULT_DIM_IN))
    head = load_composite_z_head(str(tmp_path / "best.pt"), device="cpu")
    rec = traces[0]
    r_loaded = head.predict(rec["slots_y"], rec["slots_h_raw"],
                            rec["query_emb"]).squeeze(-1)
    assert int(r_loaded.argmax().item()) == 0
    assert float(r_loaded[0].item()) > float(r_loaded[1:].max().item())


def test_load_composite_z_head_infers_mlp(tmp_path):
    """The loader rebuilds the MLP arch (hidden inferred from the state_dict) --
    not a silently-collapsed Linear. A short run (the gate is NOT asserted here,
    only the arch inference) on enough queries that the MLP doesn't collapse."""
    traces = _synthetic_h_raw_traces(n_queries=40, k=12, seed=5)
    cfg = RelevanceTrainingConfig(epochs=4, checkpoint_dir=str(tmp_path),
                                  pos_weight_cap=3.0,
                                  slot_signal_field="slots_h_raw")
    fit_relevance(traces, cfg, head=CompositeZHead(dim_in=DEFAULT_DIM_IN, hidden=64))
    head = load_composite_z_head(str(tmp_path / "best.pt"), device="cpu")
    assert head.readout.hidden == 64           # MLP inferred, not Linear


def test_load_relevance_traces_requires_slots_h_raw(tmp_path):
    """In Phase 0b mode the loader requires slots_h_raw; a stale trace without it
    is rejected with a regenerate pointer (not a silent mis-wire)."""
    traces = _synthetic_h_raw_traces(n_queries=3, k=5, seed=2)
    for r in traces:
        del r["slots_h_raw"]
    path = tmp_path / "traces.pt"
    torch.save(traces, path)
    with pytest.raises(RuntimeError, match="slots_h_raw"):
        load_relevance_traces(str(path), slot_signal_field="slots_h_raw")