"""Phase 1 (STRM) tests: the provenance-carrying ring buffer + state read-out.

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


# ── Phase 4 Step 3: pin tag (token-type embedding on injected u_{t+1}) ──

def test_ring_slot_pinned_default_false():
    """RingSlot.pinned defaults to False (checkpoint-safe 4th field; the ring is
    not serialized so adding a field is backward-compatible)."""
    slot = RingSlot(torch.zeros(1, 256), "ep-1", "text")
    assert slot.pinned is False
    # and explicitly True round-trips
    slot_p = RingSlot(torch.zeros(1, 256), "ep-1", "text", pinned=True)
    assert slot_p.pinned is True


def test_inject_carries_pin():
    """pin=True marks the resulting ring slot pinned=True (mirrors
    test_inject_carries_provenance). The pin flag is per-slot bookkeeping for
    the replay JSONL / a future retention surrogate; the pin ITSELF is the
    input-side embedding (PinTag), not this flag."""
    wm = _wm(ring_capacity=3)
    wm.update(_rand_emb(seed=1), source_id="query", text="the query")
    wm.inject(_rand_emb(seed=2), source_id="ep-42", text="recalled 42", pin=True)
    slots = wm.ring_buffer()
    assert len(slots) == 2
    assert slots[0].pinned is False  # the query step is never pinned
    assert slots[1].pinned is True   # the salience-fired recall is pinned


def test_pin_adds_embedding_to_input():
    """pin=True adds the pin vector to the input BEFORE the SSM step. The
    pin=True output is EXACTLY what you get by manually adding the pin vector
    to the input and stepping with pin=False (no double-add, no other side
    effect). This pins the semantic: the pin is an ADD on u_{t+1}, d_model
    stays 384 (a concat would break W_A's Linear(384 -> 16))."""
    bb = JGSBackbone(BackboneConfig())
    torch.manual_seed(321)
    wm_pin = WorkingMemory(bb, ring_capacity=4)       # owns a default-init PinTag
    torch.manual_seed(321)
    wm_manual = WorkingMemory(bb, ring_capacity=4)    # same init, same pin vector
    # the two WMs share the SAME pin vector (deterministic default init)
    assert torch.equal(wm_pin._pin_tag.pin, wm_manual._pin_tag.pin)
    pin_vec = wm_pin._pin_tag.pin.detach()
    emb = _rand_emb(seed=7)
    # pin=True path vs manually adding the pin to the input (pin=False, no add)
    out_pin, _, _ = wm_pin.step(emb, source_id="a", text="a", pin=True)
    out_manual, _, _ = wm_manual.step(emb + pin_vec, source_id="a", text="a", pin=False)
    assert torch.allclose(out_pin, out_manual, atol=1e-6)
    # and pin=True measurably changes the output vs pin=False on the same input
    torch.manual_seed(321)
    wm_nopin = WorkingMemory(bb, ring_capacity=4)
    out_nopin, _, _ = wm_nopin.step(emb, source_id="a", text="a", pin=False)
    assert not torch.allclose(out_pin, out_nopin, atol=1e-6)


def test_k0_pin_is_noop():
    """Ring OFF + pin=True is byte-identical to pin=False (the pin is gated on
    the ring: no salience fires without it, so pin is a no-op at K=0). Mirrors
    test_k0_state_identical_to_k_positive_state -- the pin must never perturb
    the SSM when the ring is off."""
    bb = JGSBackbone(BackboneConfig())
    torch.manual_seed(999)
    wm_off_pin = WorkingMemory(bb, ring_capacity=0)
    torch.manual_seed(999)
    wm_off_nopin = WorkingMemory(bb, ring_capacity=0)
    for s in range(5):
        emb = _rand_emb(seed=s)
        wm_off_pin.update(emb, source_id=f"q{s}", text=f"t{s}", pin=True)
        wm_off_nopin.update(emb, source_id=f"q{s}", text=f"t{s}", pin=False)
    assert wm_off_pin.state is not None and wm_off_nopin.state is not None
    for a, b in zip(wm_off_pin.state, wm_off_nopin.state):
        assert torch.equal(a, b), "pin=True perturbed the SSM state at K=0"
    # and the ring is empty either way (K=0), so no pinned slot is recorded
    assert wm_off_pin.ring_buffer() == []
    assert wm_off_nopin.ring_buffer() == []


def test_pin_tag_default_init_is_nonzero_and_deterministic():
    """The default-init PinTag is a faithful non-stub: non-zero (so pin=True is
    not a silent no-op) and deterministic across constructions (fixed generator,
    no Date.now/random). v1 ships this; a retention surrogate can fit it later."""
    from src.subconscious.pin_tag import PinTag, D_MODEL
    t1 = PinTag()
    t2 = PinTag()
    assert t1.pin.shape == (D_MODEL,)
    assert float(t1.pin.detach().abs().max()) > 0.0  # non-zero
    assert torch.equal(t1.pin, t2.pin)  # deterministic
    # forward adds the pin to the input (broadcasts over the batch dim)
    emb = _rand_emb(seed=11)
    out = t1(emb)
    assert torch.allclose(out, emb + t1.pin)


def test_pin_tag_loader_roundtrip(tmp_path):
    """load_pin_tag recovers a saved PinTag (the loader is wired now so a future
    retention surrogate's checkpoint loads without new plumbing). v1 has no
    trained checkpoint, so this writes a default-init one and round-trips it."""
    import torch as _torch
    from src.subconscious.pin_tag import PinTag, load_pin_tag, D_MODEL
    tag = PinTag()
    # the checkpoint shape load_pin_tag expects: {"pin": state_dict, "d_model": ...}
    ckpt = {"pin": tag.state_dict(), "d_model": D_MODEL}
    path = tmp_path / "pin.pt"
    _torch.save(ckpt, path)
    loaded = load_pin_tag(str(path), device="cpu")
    assert isinstance(loaded, PinTag)
    assert not loaded.training
    assert torch.equal(loaded.pin, tag.pin)  # round-trip preserves the vector


# ── State-trajectory rewire (Phase A): store h_t per ring slot ──

def test_ringslot_h_defaults_none():
    """RingSlot.h defaults to None so any existing positional constructor
    (RingSlot(y, sid, text) / RingSlot(y, sid, text, pinned=...)) is
    backward-compatible. The field is in-memory only (not in snapshot/
    checkpoints), so adding it is checkpoint-backward-compatible."""
    slot = RingSlot(torch.zeros(1, 256), "ep-1", "text")
    assert slot.h is None
    slot_p = RingSlot(torch.zeros(1, 256), "ep-1", "text", pinned=True)
    assert slot_p.h is None


def test_slot_h_captured_shape_and_dtype():
    """A K>0 step captures the per-layer recurrent state on the slot: a list of
    n_layers (4) fp16 [1, d_state=16, d_model=384] tensors. fp16 bounds memory;
    LatentDynamicsHead.project casts to fp32 internally so the projection is
    lossless."""
    wm = _wm(ring_capacity=4)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot = wm.ring_buffer()[0]
    assert slot.h is not None
    assert len(slot.h) == 4  # n_layers
    for t in slot.h:
        assert t.shape == (1, 16, 384)
        assert t.dtype == torch.float16
        assert not t.requires_grad  # detached (no BPTT into the frozen backbone)


def test_slot_h_matches_live_state_for_last_slot():
    """The most-recent slot's h is the state that produced it; no step has
    happened since, so casting slot.h back to fp32 equals the live
    state_tensors() (the capture is a faithful snapshot of the post-step,
    post-decay recurrent state, not a stale or perturbed copy)."""
    wm = _wm(ring_capacity=4)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot = wm.ring_buffer()[-1]
    live = wm.state_tensors()
    assert slot.h is not None and len(slot.h) == len(live)
    for h, s in zip(slot.h, live):
        # fp16 storage rounds vs the fp32 live state; assert faithful within
        # fp16 tolerance (the capture is a snapshot of the post-step/post-decay
        # state, not a stale or perturbed copy).
        assert torch.allclose(h.to(torch.float32), s.to(torch.float32), atol=1e-3)


def test_slot_h_is_independent_clone():
    """slot.h is a detached clone: later steps (which mutate self.state in
    place) do not change a previously captured slot's h. This is the same
    independence contract test_slot_vector_is_detached_clone pins for y."""
    wm = _wm(ring_capacity=4)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot0 = wm.ring_buffer()[0]
    h0_frozen = [t.clone() for t in slot0.h]
    # a subsequent step mutates the live state but must not touch slot0.h
    wm.update(_rand_emb(seed=1), source_id="q1", text="t1")
    for a, b in zip(slot0.h, h0_frozen):
        assert torch.equal(a, b), "a later step perturbed a captured slot.h"


def test_k0_does_not_construct_h():
    """K=0 never enters the ring-append block, so no h is constructed and the
    state evolution is byte-identical to Phase 2c. (The ring is empty, so this
    is also covered by test_k0_ring_stays_empty_after_updates; this test pins
    that the h-capture specifically is gated on the ring.)"""
    wm = _wm(ring_capacity=0)
    for s in range(3):
        wm.update(_rand_emb(seed=s), source_id=f"q{s}", text=f"t{s}")
    assert wm.ring_buffer() == []  # no slots -> no h anywhere


def test_slot_z_helper_projects_and_none_guards():
    """slot_z(slot, head) -> head.project(slot.h) = [1, 384] for a state-bearing
    slot, and None for an h-less slot (tests/partial constructions). Reuses
    LatentDynamicsHead.project verbatim (last layer, mean over d_state) -- the
    state-trajectory rewire's unit of attention."""
    from src.subconscious.latent_dynamics_head import LatentDynamicsHead
    from src.subconscious.working_memory import slot_z

    head = LatentDynamicsHead()  # untrained is fine for a shape/projection test
    wm = _wm(ring_capacity=4)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot = wm.ring_buffer()[0]
    z = slot_z(slot, head)
    assert z is not None
    assert z.shape == (1, 384)
    # matches projecting the live state directly (same last-layer mean)
    assert torch.allclose(z, head.project(wm.state_tensors()), atol=1e-3)
    # an h-less slot (positional constructor) -> None, not a crash
    bare = RingSlot(torch.zeros(1, 256), "ep", "text")
    assert slot_z(bare, head) is None


def test_slot_z_fp16_storage_is_lossless_for_projection():
    """project casts to fp32 internally, so storing h in fp16 vs fp32 produces
    an equal projection (within fp16 round-trip tolerance). This justifies the
    fp16 storage choice on memory grounds without a precision cost downstream."""
    from src.subconscious.latent_dynamics_head import LatentDynamicsHead
    from src.subconscious.working_memory import slot_z

    head = LatentDynamicsHead()
    wm = _wm(ring_capacity=4)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot = wm.ring_buffer()[0]
    z_fp16 = slot_z(slot, head)
    # project the same last-layer state in pure fp32
    last_fp32 = slot.h[-1].to(torch.float32)
    z_fp32 = head.project([last_fp32])
    assert torch.allclose(z_fp16, z_fp32, atol=1e-3)


# ── Phase 1a: slot_type tag (conversation vs retrieved) ──

def test_ringslot_slot_type_defaults_none():
    """RingSlot.slot_type defaults to None so any existing positional constructor
    (RingSlot(y, sid, text) / .../ RingSlot(y, sid, text, pinned=..., h=...)) is
    backward-compatible. In-memory only (not in snapshot/checkpoints)."""
    assert RingSlot(torch.zeros(1, 256), "ep-1", "text").slot_type is None
    assert RingSlot(torch.zeros(1, 256), "ep-1", "text", pinned=True).slot_type is None
    assert RingSlot(torch.zeros(1, 256), "ep-1", "text", h=None).slot_type is None
    # and explicitly set round-trips
    assert RingSlot(torch.zeros(1, 256), "ep-1", "text", slot_type=0).slot_type == 0


def test_update_tags_query_slot_conversation():
    """update's query step is tagged slot_type=0 (conversation); inject is tagged
    slot_type=1 (retrieved). The production ring mixes both, and the cross-slot
    Transformer's slot-type embedding + the live scorer's per-type gap split
    condition on this tag."""
    wm = _wm(ring_capacity=4)
    wm.update(_rand_emb(seed=1), source_id="sess#msg1", text="the prompt")
    wm.inject(_rand_emb(seed=2), source_id="sess__ep0001", text="recalled doc")
    slots = wm.ring_buffer()
    assert len(slots) == 2
    assert slots[0].slot_type == 0   # the query step -> conversation
    assert slots[1].slot_type == 1   # the injected recall -> retrieved


def test_slot_type_never_perturbs_state():
    """slot_type is in-memory metadata ONLY: passing slot_type=0/1/None to step
    produces byte-identical state evolution AND output (the tag never reaches
    the SSM). This is the binding-constraint guard for Phase 1a."""
    bb = JGSBackbone(BackboneConfig())
    torch.manual_seed(11)
    wm_a = WorkingMemory(bb, ring_capacity=4)
    torch.manual_seed(11)
    wm_b = WorkingMemory(bb, ring_capacity=4)
    torch.manual_seed(11)
    wm_c = WorkingMemory(bb, ring_capacity=4)
    emb = _rand_emb(seed=9)
    out_a, _, _ = wm_a.step(emb, source_id="s", text="t", slot_type=0)
    out_b, _, _ = wm_b.step(emb, source_id="s", text="t", slot_type=1)
    out_c, _, _ = wm_c.step(emb, source_id="s", text="t", slot_type=None)
    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert torch.allclose(out_a, out_c, atol=1e-6)
    for a, b, c in zip(wm_a.state_tensors(), wm_b.state_tensors(), wm_c.state_tensors()):
        assert torch.equal(a, b)
        assert torch.equal(a, c)