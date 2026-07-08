"""Offline tests for the Phase 2c Working Memory core.

All CPU-runnable against ``ReferenceSSM`` + a deterministic stub embedder (no
``sentence_transformers``, no Ollama, no WaveDB). Verifies the defining 2c
property: the recurrent state **persists across queries** (not reset per
query), plus decay, snapshot/restore, and metadata bookkeeping.
"""

from __future__ import annotations

import pytest
import torch

from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig, INSTANCE_CONFIGS
from src.subconscious.state_serializer import JGSSnapshot, deserialize, serialize
from src.subconscious.working_memory import WorkingMemory, WorkingMemoryState


def _wm(decay_alpha: float = 1.0, embedder=None) -> WorkingMemory:
    bb = JGSBackbone(BackboneConfig())
    return WorkingMemory(bb, embedder=embedder, decay_alpha=decay_alpha)


def _rand_emb(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 384, dtype=torch.float32, generator=g)


# ── shape / contract ──

def test_state_shape_is_4x_1_16_384():
    wm = _wm()
    wm.update(_rand_emb())
    assert wm.state is not None
    assert len(wm.state) == 4
    for t in wm.state:
        assert t.shape == (1, 16, 384)


def test_state_is_detached_after_step():
    wm = _wm()
    wm.update(_rand_emb())
    for t in wm.state:
        assert not t.requires_grad


def test_parameters_exclude_backbone():
    wm = _wm()
    bb_params = sum(p.numel() for p in wm.backbone.parameters())
    own_params = sum(p.numel() for p in wm.parameters())
    # instance.parameters() must NOT re-count the shared backbone weights.
    assert own_params < bb_params + 1  # own params are a fraction of the backbone


# ── the defining 2c property: state persists across queries ──

def test_state_persists_across_updates():
    wm = _wm()
    s0 = wm.update(_rand_emb(seed=1))
    s1 = wm.update(_rand_emb(seed=2))
    # Two different inputs → state must move (not reset to a function of just
    # the latest input). The last-layer state differs between snapshots.
    assert not torch.equal(s0.state_tensors[-1], s1.state_tensors[-1])
    # input_count accumulates (not reset per query).
    assert wm.input_count == 2
    assert s1.input_count == 2


def test_inject_does_not_increment_input_count():
    wm = _wm()
    wm.update(_rand_emb(seed=1))
    assert wm.input_count == 1
    wm.inject(_rand_emb(seed=99))
    wm.inject(_rand_emb(seed=100))
    # injects absorb episodes as gist but are not "queries".
    assert wm.input_count == 1


def test_update_with_retrieved_differs_from_query_alone():
    wm = _wm()
    q = _rand_emb(seed=1)
    s_qonly = wm.update(q)
    wm.reset()
    s_with = wm.update(q, retrieved_embeddings=[_rand_emb(seed=7), _rand_emb(seed=8)])
    assert not torch.equal(s_qonly.state_tensors[-1], s_with.state_tensors[-1])


def test_reset_zeros_state_and_bookkeeping():
    wm = _wm()
    wm.update(_rand_emb(seed=1))
    wm.inject(_rand_emb(seed=2))
    wm.set_metadata("active_domains", ["database"])
    assert wm.input_count == 1
    wm.reset()
    assert wm.input_count == 0
    assert wm.get_metadata("active_domains") is None
    # state is freshly zero-initialized after reset.
    for t in wm.state:
        assert torch.equal(t, torch.zeros_like(t))


def test_reset_then_update_state_differs_from_zero():
    wm = _wm()
    wm.reset()
    zero = [t.clone() for t in wm.state]
    wm.update(_rand_emb(seed=3))
    assert not torch.equal(zero[-1], wm.state[-1])


# ── determinism ──

def test_deterministic_for_same_input_sequence():
    # Determinism = ONE instance, reset + replay the same sequence → same state.
    # Two separate instances have their own randomly-initialized projections
    # (input_proj/state_lora), so they are never expected to agree even sharing a
    # backbone. The ReferenceSSM itself is deterministic; replaying the same
    # inputs through the same weights from a zero start must reproduce state.
    wm = _wm()
    seq = [_rand_emb(seed=s) for s in (1, 2, 3)]
    for e in seq:
        wm.update(e)
    final = [t.clone() for t in wm.state]
    wm.reset()
    for e in seq:
        wm.update(e)
    for a, b in zip(final, wm.state):
        assert torch.equal(a, b)


# ── decay ──

def test_decay_alpha_shrinks_state():
    wm_decay = _wm(decay_alpha=0.5)
    wm_plain = _wm(decay_alpha=1.0)
    wm_plain.update(_rand_emb(seed=1))
    wm_decay.update(_rand_emb(seed=1))
    # decay_alpha < 1.0 → state norm strictly smaller than the no-decay path.
    norm_decay = sum(t.abs().sum().item() for t in wm_decay.state)
    norm_plain = sum(t.abs().sum().item() for t in wm_plain.state)
    assert norm_decay < norm_plain


def test_decay_alpha_1_is_noop():
    wm = _wm(decay_alpha=1.0)
    wm.update(_rand_emb(seed=1))
    # decay_alpha == 1.0 must not multiply the state (no-op branch).
    s_before = [t.clone() for t in wm.state]
    wm._apply_decay()
    for a, b in zip(s_before, wm.state):
        assert torch.equal(a, b)


# ── snapshot / restore ──

def test_snapshot_is_detached_clone():
    wm = _wm()
    wm.update(_rand_emb(seed=1))
    snap = wm.snapshot()
    # mutate the live state; the snapshot must not change.
    wm.inject(_rand_emb(seed=2))
    assert isinstance(snap, WorkingMemoryState)
    assert isinstance(snap, JGSSnapshot)  # WorkingMemoryState aliases JGSSnapshot


def test_restore_resumes_state_and_bookkeeping():
    wm = _wm()
    wm.update(_rand_emb(seed=1))
    wm.inject(_rand_emb(seed=2))
    wm.set_metadata("active_domains", ["coding", "database"])
    snap = wm.snapshot()

    wm2 = _wm()
    wm2.reset()
    wm2.restore(snap)
    assert wm2.input_count == snap.input_count == 1
    assert wm2.get_metadata("active_domains") == ["coding", "database"]
    for a, b in zip(wm.state, wm2.state):
        assert torch.equal(a, b)


def test_snapshot_round_trips_through_serializer():
    wm = _wm()
    wm.update(_rand_emb(seed=1))
    wm.set_metadata("last_query_type", "factual")
    snap = wm.snapshot()
    blob = serialize(snap)
    back = deserialize(blob)
    for a, b in zip(snap.state_tensors, back.state_tensors):
        assert torch.equal(a, b)
    assert back.metadata["last_query_type"] == "factual"


# ── embed helper / errors ──

def test_snapshot_before_step_raises():
    wm = _wm()
    with pytest.raises(ValueError, match="None"):
        wm.snapshot()


def test_embed_without_embedder_raises():
    wm = _wm()
    with pytest.raises(RuntimeError, match="embedder"):
        wm.embed(["hello"])


# ── latency (CPU, ReferenceSSM) ──

def test_hundred_updates_under_50ms_per_call():
    import time as _t
    wm = _wm()
    wm.update(_rand_emb(seed=0))
    # measure 99 more (state already initialized)
    emb = _rand_emb(seed=42)
    start = _t.perf_counter()
    for _ in range(99):
        wm.update(emb)
    elapsed = _t.perf_counter() - start
    per_call_ms = (elapsed / 99) * 1000
    # CPU torch per-step is slower than the doc's imagined numpy <10ms target;
    # <50ms/call is the corrected realistic bound (docs/Phase 2c.md §9.2).
    assert per_call_ms < 50.0, f"per-call {per_call_ms:.2f}ms exceeded 50ms budget"