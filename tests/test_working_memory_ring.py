"""Phase 1 (JST) tests: the provenance-carrying ring buffer + state read-out.

All CPU-runnable against ``ReferenceSSM`` + deterministic embeddings (no
``sentence_transformers``, no Ollama, no WaveDB). Two contracts:

1. ``ring_capacity == 0`` (the default) is byte-identical to shipped Phase 2c:
   no buffer is allocated, ``ring_buffer()`` is empty, and crucially the
   recurrent state evolves identically whether K=0 or K>0 (the ring is
   observation-only — it never perturbs the SSM).
2. ``ring_capacity > 0`` retains the last K step outputs (FIFO) with their
   provenance (``source_id`` / ``text``), and ``state_tensors()`` exposes the
   live per-layer state for read-only head use.
"""

from __future__ import annotations

import pytest
import torch

from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig, InstanceConfig, INSTANCE_CONFIGS
from src.subconscious.working_memory import RingSlot, WorkingMemory


def _rand_emb(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 384, dtype=torch.float32, generator=g)


def _wm(ring_capacity: int = 0, backbone=None, decay_alpha: float = 1.0) -> WorkingMemory:
    bb = backbone if backbone is not None else JGSBackbone(BackboneConfig())
    return WorkingMemory(bb, decay_alpha=decay_alpha, ring_capacity=ring_capacity)


# ── K=0: OFF, byte-identical to Phase 2c ──

def test_default_ring_capacity_is_zero():
    wm = _wm()
    assert wm.ring_capacity == 0
    assert INSTANCE_CONFIGS["working_memory"].ring_capacity == 0


def test_k0_ring_stays_empty_after_updates():
    wm = _wm(ring_capacity=0)
    for s in range(5):
        wm.update(_rand_emb(seed=s), source_id=f"q{s}", text=f"text{s}")
    assert wm.ring_buffer() == []
    # passing provenance with K=0 is a no-op, not an error
    assert wm.ring_capacity == 0


def test_k0_state_identical_to_k_positive_state():
    """The ring is observation-only: K=0 and K>0 produce bit-identical state.

    Same shared backbone + same seeded instance params + same inputs => the only
    difference is the ring. If states match exactly, the ring append never
    perturbs the SSM (it is strictly post-step, post-decay, detach+clone only).
    """
    bb = JGSBackbone(BackboneConfig())  # shared backbone, built once
    torch.manual_seed(123)
    wm_off = WorkingMemory(bb, ring_capacity=0)
    torch.manual_seed(123)
    wm_on = WorkingMemory(bb, ring_capacity=4)
    for s in range(6):
        emb = _rand_emb(seed=s)
        wm_off.update(emb, source_id=f"q{s}", text=f"text{s}")
        wm_on.update(emb, source_id=f"q{s}", text=f"text{s}")
    assert wm_off.state is not None and wm_on.state is not None
    assert len(wm_off.state) == len(wm_on.state)
    for a, b in zip(wm_off.state, wm_on.state):
        assert a.shape == b.shape
        assert torch.equal(a, b), "ring recording perturbed the SSM state"


# ── K>0: FIFO retention with provenance ──

def test_k_positive_holds_last_k_slots():
    wm = _wm(ring_capacity=3)
    for s in range(5):
        wm.update(_rand_emb(seed=s), source_id=f"q{s}", text=f"text{s}")
    slots = wm.ring_buffer()
    assert len(slots) == 3
    # FIFO: oldest dropped, last 3 retained, oldest-first ordering
    assert [s.source_id for s in slots] == ["q2", "q3", "q4"]
    assert [s.text for s in slots] == ["text2", "text3", "text4"]


def test_k_positive_under_capacity_holds_all():
    wm = _wm(ring_capacity=5)
    for s in range(3):
        wm.update(_rand_emb(seed=s), source_id=f"q{s}", text=f"text{s}")
    slots = wm.ring_buffer()
    assert len(slots) == 3
    assert [s.source_id for s in slots] == ["q0", "q1", "q2"]


def test_slot_vector_shape_is_step_output_dim():
    """The slot vector is the step output, NOT a hardcoded 384. The working_memory
    instance has output_dim=256, so slots carry [1, 256]. The buffer is
    dimension-agnostic — it stores whatever the step emits."""
    wm = _wm(ring_capacity=2)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slots = wm.ring_buffer()
    assert len(slots) == 1
    assert slots[0].y.shape == (1, INSTANCE_CONFIGS["working_memory"].output_dim)
    assert slots[0].y.dtype == torch.float32


def test_slot_vector_is_detached_clone():
    """Slot tensors are detached (no grad leak into the frozen backbone) and
    independent clones (mutating a slot does not affect the live state)."""
    wm = _wm(ring_capacity=2)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot = wm.ring_buffer()[0]
    assert not slot.y.requires_grad
    state_before = [t.clone() for t in wm.state]
    slot.y.add_(999.0)  # mutate the slot's tensor in place
    # live state untouched
    for a, b in zip(wm.state, state_before):
        assert torch.equal(a, b)
    # and the slot stored in the ring is the same tensor we mutated (clone is
    # of the step output, not of the state)
    assert wm.ring_buffer()[0].y is slot.y


def test_inject_carries_provenance():
    wm = _wm(ring_capacity=3)
    wm.update(_rand_emb(seed=1), source_id="query", text="the query")
    wm.inject(_rand_emb(seed=2), source_id="ep-42", text="recalled episode 42")
    slots = wm.ring_buffer()
    assert len(slots) == 2
    assert slots[0].source_id == "query" and slots[0].text == "the query"
    assert slots[1].source_id == "ep-42" and slots[1].text == "recalled episode 42"


def test_retrieved_sources_parallel_list():
    wm = _wm(ring_capacity=4)
    wm.update(
        _rand_emb(seed=1),
        retrieved_embeddings=[_rand_emb(seed=2), _rand_emb(seed=3)],
        source_id="query",
        text="the query",
        retrieved_sources=[("ep-a", "text a"), ("ep-b", "text b")],
    )
    slots = wm.ring_buffer()
    assert [s.source_id for s in slots] == ["query", "ep-a", "ep-b"]
    assert [s.text for s in slots] == ["the query", "text a", "text b"]


def test_retrieved_without_sources_get_none_provenance():
    wm = _wm(ring_capacity=4)
    wm.update(
        _rand_emb(seed=1),
        retrieved_embeddings=[_rand_emb(seed=2), _rand_emb(seed=3)],
        source_id="query",
        text="the query",
    )
    slots = wm.ring_buffer()
    assert slots[0].source_id == "query"
    assert slots[1].source_id is None and slots[1].text is None
    assert slots[2].source_id is None and slots[2].text is None


def test_retrieved_sources_length_mismatch_raises():
    """A mismatched parallel list must raise, not silently drop episodes (which
    would corrupt the SSM state by skipping steps)."""
    wm = _wm(ring_capacity=4)
    with pytest.raises(ValueError, match="retrieved_sources length"):
        wm.update(
            _rand_emb(seed=1),
            retrieved_embeddings=[_rand_emb(seed=2), _rand_emb(seed=3), _rand_emb(seed=4)],
            retrieved_sources=[("ep-a", "text a")],  # too short
        )


def test_reset_clears_ring():
    wm = _wm(ring_capacity=3)
    for s in range(3):
        wm.update(_rand_emb(seed=s), source_id=f"q{s}", text=f"t{s}")
    assert len(wm.ring_buffer()) == 3
    wm.reset()
    assert wm.ring_buffer() == []
    # and it refills after reset
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    assert len(wm.ring_buffer()) == 1


# ── config-driven capacity ──

def test_capacity_from_config():
    cfg = InstanceConfig(name="working_memory", ring_capacity=2)
    bb = JGSBackbone(BackboneConfig())
    wm = WorkingMemory(bb, config=cfg)
    assert wm.ring_capacity == 2
    for s in range(4):
        wm.update(_rand_emb(seed=s), source_id=f"q{s}", text=f"t{s}")
    assert len(wm.ring_buffer()) == 2


def test_kwarg_overrides_config():
    cfg = InstanceConfig(name="working_memory", ring_capacity=2)
    bb = JGSBackbone(BackboneConfig())
    wm = WorkingMemory(bb, config=cfg, ring_capacity=5)
    assert wm.ring_capacity == 5


# ── state_tensors() read-out ──

def test_state_tensors_returns_live_state():
    wm = _wm(ring_capacity=2)
    wm.update(_rand_emb(seed=0))
    st = wm.state_tensors()
    assert st is wm.state  # live, NOT a clone (heads read the actual current state)
    assert len(st) == 4
    for t in st:
        assert t.shape == (1, 16, 384)
        assert not t.requires_grad  # detached by the SSM (no BPTT)


def test_state_tensors_raises_before_step():
    wm = _wm(ring_capacity=2)
    with pytest.raises(ValueError, match="state is None"):
        wm.state_tensors()


def test_state_tensors_distinct_from_snapshot():
    """snapshot() detaches+clones+CPU-copies for serialization; state_tensors()
    returns the live on-device list. They carry equal values but are not the
    same tensor objects."""
    wm = _wm(ring_capacity=2)
    wm.update(_rand_emb(seed=0))
    live = wm.state_tensors()
    snap = wm.snapshot()
    assert len(live) == len(snap.state_tensors)
    for a, b in zip(live, snap.state_tensors):
        assert torch.equal(a, b)  # equal values
        assert a is not b  # distinct objects (snapshot cloned)