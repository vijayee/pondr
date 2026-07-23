"""Phase 1f-7 Stage 3 tests: opt-in per-kept-slot pre-state capture + replay.

The Stage 3 backbone fine-tune (``scripts/finetune_backbone_flatlast_margin.py``)
seeds each kept slot's SSM step from ``slots_pre_state [K,4,16,384]`` -- the
cumulative WM state BEFORE that slot's step -- and re-steps ONLY that slot's
input WITH grad through ``backbone.layers[i].step`` (truncated-BPTT-depth-1).
For that replay to train on the REAL serve state distribution, the captured
``h_pre`` must be exactly the state the step started from. Three contracts:

1. ``capture_pre_state=False`` (the default) is byte-identical to today:
   ``RingSlot.h_pre is None`` and the SSM state evolves identically whether the
   flag is on or off (the snapshot is a READ of ``self.state`` before
   ``super().step`` -- it never perturbs the SSM).
2. ``capture_pre_state=True`` captures ``h_pre`` on every kept slot: a list of
   ``n_layers`` (4) detached fp16 ``[1,16,384]`` tensors, independent of later
   steps (a detached clone, like ``h``).
3. **Round-trip fidelity** (the core correctness check for the replay): seed
   ``states = slot.h_pre`` (fp32), step ``slot``'s input via
   ``backbone.layers[i].step``, and the resulting last-layer state matches
   ``slot.h`` within fp16-seed-propagation tolerance. This is exactly the
   primitive ``replay_flatlast`` relies on -- if it does not hold, the
   fine-tune trains on bogus state.

Plus the trace-generator contract: ``_build_mixed_record(capture_pre_state=True)``
emits ``slots_pre_state [K,4,16,384]`` fp16 AND ``slots_step_input [K,384]`` fp32
(the EXACT per-slot step-input -- NOT ``embed(slot.text)``, since code docs are
injected by meaning) both aligned with ``slots_h_raw`` / ``source_ids``; the
default-off path emits NEITHER key. A slot missing ``h_pre`` OR ``u`` is dropped
under capture (both seed the replay).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_onyx_doc_ring_traces import _build_mixed_record  # noqa: E402
from src.subconscious.backbone import JGSBackbone  # noqa: E402
from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.latent_dynamics_head import LatentDynamicsHead  # noqa: E402
from src.subconscious.working_memory import RingSlot, WorkingMemory  # noqa: E402


def _rand_emb(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 384, dtype=torch.float32, generator=g)


def _wm(ring_capacity: int = 4, backbone=None, decay_alpha: float = 1.0,
        capture_pre_state: bool = False, identity_instance: bool = True) -> WorkingMemory:
    bb = backbone if backbone is not None else JGSBackbone(BackboneConfig())
    return WorkingMemory(bb, decay_alpha=decay_alpha,
                         ring_capacity=ring_capacity,
                         capture_pre_state=capture_pre_state,
                         identity_instance=identity_instance)


# ── M1: default-off is byte-identical ──

def test_prestate_default_off_h_pre_is_none():
    """capture_pre_state=False (default) -> every slot's h_pre is None (the
    field defaults None and the snapshot branch is skipped). Backward-compatible
    with every existing positional/keyword RingSlot constructor."""
    wm = _wm(ring_capacity=4, capture_pre_state=False)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot = wm.ring_buffer()[0]
    assert slot.h_pre is None
    # and a bare RingSlot (no capture path) also defaults None
    assert RingSlot(torch.zeros(1, 256), "ep", "text").h_pre is None


def test_prestate_default_off_state_byte_identical():
    """The pre-state snapshot is a READ of self.state only -- it never perturbs
    the SSM. K>0 + capture_pre_state=True vs False on the same shared backbone
    + same inputs produces byte-identical recurrent state (mirrors
    test_k0_state_identical_to_k_positive_state for the capture flag)."""
    bb = JGSBackbone(BackboneConfig())
    torch.manual_seed(202)
    wm_off = WorkingMemory(bb, ring_capacity=4, decay_alpha=1.0,
                           capture_pre_state=False, identity_instance=True)
    torch.manual_seed(202)
    wm_on = WorkingMemory(bb, ring_capacity=4, decay_alpha=1.0,
                          capture_pre_state=True, identity_instance=True)
    for s in range(6):
        emb = _rand_emb(seed=s)
        wm_off.update(emb, source_id=f"q{s}", text=f"t{s}")
        wm_on.update(emb, source_id=f"q{s}", text=f"t{s}")
    assert wm_off.state is not None and wm_on.state is not None
    assert len(wm_off.state) == len(wm_on.state)
    for a, b in zip(wm_off.state, wm_on.state):
        assert a.shape == b.shape
        assert torch.equal(a, b), "pre-state capture perturbed the SSM state"


# ── M1: capture_on shape / dtype / independence ──

def test_prestate_on_first_step_h_pre_is_none():
    """The snapshot is gated on ``self.state is not None`` (per the Stage 3 plan)
    -- and the recurrent state is lazy-initialized INSIDE ``super().step`` via
    ``_ensure_state``. So on the FIRST step after a reset, ``self.state`` is
    still None at snapshot time -> ``h_pre=None``. (Production impact: one slot
    per session, dropped by ``_build_mixed_record``'s capture guard -- negligible
    vs the ~K*turns kept slots.) This pins the timing so a future change to
    capture the zero pre-state on step 0 is a conscious decision, not silent."""
    wm = _wm(ring_capacity=4, capture_pre_state=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot0 = wm.ring_buffer()[0]
    assert slot0.h is not None          # the post-step state IS captured
    assert slot0.h_pre is None          # but the pre-step state is not (state was None)


def test_prestate_on_captures_h_pre_shape_dtype():
    """capture_pre_state=True -> from the SECOND step on, every kept slot's
    h_pre is a list of n_layers (4) detached fp16 [1,16,384] tensors (mirrors
    slot.h's shape/dtype). The second step's pre-state is the first step's
    post-state, which is non-None -> captured."""
    wm = _wm(ring_capacity=4, capture_pre_state=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")  # step 0: h_pre=None
    wm.update(_rand_emb(seed=1), source_id="q1", text="t1")  # step 1: h_pre=state0
    slot1 = wm.ring_buffer()[1]
    assert slot1.h_pre is not None
    assert len(slot1.h_pre) == 4  # n_layers
    for t in slot1.h_pre:
        assert t.shape == (1, 16, 384)
        assert t.dtype == torch.float16
        assert not t.requires_grad  # detached (no BPTT into the frozen backbone)


def test_prestate_h_pre_is_independent_clone():
    """slot.h_pre is a detached clone: later steps (which reassign self.state)
    do not change a previously captured slot's h_pre. Same independence contract
    as test_slot_h_is_independent_clone pins for h. Uses slot1 (h_pre non-None)."""
    wm = _wm(ring_capacity=4, capture_pre_state=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    wm.update(_rand_emb(seed=1), source_id="q1", text="t1")  # slot1.h_pre = state0
    slot1 = wm.ring_buffer()[1]
    assert slot1.h_pre is not None
    hpre1_frozen = [t.clone() for t in slot1.h_pre]
    # a subsequent step reassigns self.state but must not touch slot1.h_pre
    wm.update(_rand_emb(seed=2), source_id="q2", text="t2")
    for a, b in zip(slot1.h_pre, hpre1_frozen):
        assert torch.equal(a, b), "a later step perturbed a captured slot.h_pre"


def test_prestate_on_k0_is_noop():
    """K=0 never enters the ring-append block (and the snapshot is gated on
    ring_capacity > 0), so capture_pre_state=True at K=0 builds no h_pre and
    the state evolution is byte-identical to K=0 + capture_pre_state=False."""
    bb = JGSBackbone(BackboneConfig())
    torch.manual_seed(7)
    wm_a = WorkingMemory(bb, ring_capacity=0, capture_pre_state=True,
                         identity_instance=True)
    torch.manual_seed(7)
    wm_b = WorkingMemory(bb, ring_capacity=0, capture_pre_state=False,
                         identity_instance=True)
    for s in range(3):
        emb = _rand_emb(seed=s)
        wm_a.update(emb, source_id=f"q{s}", text=f"t{s}")
        wm_b.update(emb, source_id=f"q{s}", text=f"t{s}")
    assert wm_a.ring_buffer() == []  # no slots -> no h_pre anywhere
    assert wm_b.ring_buffer() == []
    for a, b in zip(wm_a.state, wm_b.state):
        assert torch.equal(a, b)


# ── M1: round-trip fidelity (the core replay-correctness check) ──

def test_prestate_roundtrip_fidelity_last_layer():
    """THE core check for the Stage 3 replay: seed states from slot.h_pre
    (fp16 -> fp32), step slot's input via backbone.layers[i].step (the
    identity_instance path == direct layer.step, since input_proj/state_lora
    are Identity), and the resulting last-layer state matches slot.h (the
    post-step, post-decay state; decay_alpha=1.0 so post-decay == post-step).

    Uses the SECOND step's slot: its h_pre (the first step's post-state) is
    non-None, and its h is the state after stepping that h_pre with the second
    input. This is exactly ``replay_flatlast``'s primitive. The only difference
    vs the live step is the fp16 round-trip of the SEED (h_pre is stored fp16),
    so tolerance accommodates one SSM step of fp16-seed propagation -- the same
    fidelity contract the fine-tune's epoch-0 gate checks (atol 0.15). If this
    fails, the pre-state seed does NOT reproduce the trace and the fine-tune
    would train on bogus state."""
    bb = JGSBackbone(BackboneConfig())
    wm = WorkingMemory(bb, ring_capacity=4, decay_alpha=1.0,
                       capture_pre_state=True, identity_instance=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")  # step 0
    emb1 = _rand_emb(seed=1)
    wm.update(emb1, source_id="q1", text="t1")               # step 1 -> slot1
    slot1 = wm.ring_buffer()[1]
    assert slot1.h_pre is not None and slot1.h is not None

    # Seed the replay from the captured pre-state (fp16 -> fp32, detached).
    states = [slot1.h_pre[i].to(torch.float32).detach() for i in range(4)]
    # identity_instance -> input_proj is Identity, so h = x directly (mirrors
    # replay_flatlast: h = slots_doc_emb; here the step input is emb1).
    h = emb1.to(torch.float32).detach()
    new_states = []
    for i, layer in enumerate(bb.layers):
        h, s = layer.step(h, states[i])
        new_states.append(s)
    replayed_last = new_states[-1].to(torch.float32)        # [1,16,384]
    stored_last = slot1.h[-1].to(torch.float32)             # [1,16,384]
    assert replayed_last.shape == stored_last.shape
    max_diff = (replayed_last - stored_last).abs().max().item()
    # fp16-seed propagation through one SSM step; the fine-tune's fidelity gate
    # uses atol 0.15 -- pin tighter here (0.1) since this is a controlled init.
    assert max_diff <= 0.1, (
        f"pre-state round-trip diverged: max-abs-diff {max_diff:.4f} > 0.1 -- "
        "the captured h_pre does not reproduce slot.h via layer.step")


def test_prestate_roundtrip_all_layers():
    """Round-trip fidelity holds for ALL 4 layers, not just the last -- each
    layer's new state from the seeded replay matches slot.h[i] (the per-layer
    post-step state). Pins that the seed + step reproduce the full state stack,
    not just the readout layer. Uses slot1 (h_pre non-None)."""
    bb = JGSBackbone(BackboneConfig())
    wm = WorkingMemory(bb, ring_capacity=4, decay_alpha=1.0,
                       capture_pre_state=True, identity_instance=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    emb1 = _rand_emb(seed=1)
    wm.update(emb1, source_id="q1", text="t1")
    slot1 = wm.ring_buffer()[1]
    states = [slot1.h_pre[i].to(torch.float32).detach() for i in range(4)]
    h = emb1.to(torch.float32).detach()
    new_states = []
    for i, layer in enumerate(bb.layers):
        h, s = layer.step(h, states[i])
        new_states.append(s)
    for i in range(4):
        diff = (new_states[i].to(torch.float32) - slot1.h[i].to(torch.float32)).abs().max().item()
        assert diff <= 0.1, f"layer {i} round-trip diverged: {diff:.4f}"


def test_prestate_h_pre_equals_live_state_before_step():
    """h_pre is the state the step starts from: snapshot the live state right
    BEFORE the second step, then confirm slot1.h_pre (captured inside step)
    equals that live pre-step state (fp16 round-trip). This pins the snapshot
    timing (BEFORE super().step) vs slot.h (AFTER). h_pre != h for a real step."""
    bb = JGSBackbone(BackboneConfig())
    wm = WorkingMemory(bb, ring_capacity=4, decay_alpha=1.0,
                       capture_pre_state=True, identity_instance=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")  # step 0 -> state0
    # Snapshot the live state AFTER step 0 = the state step 1 will start from.
    state0_live = [t.clone() for t in wm.state]
    wm.update(_rand_emb(seed=1), source_id="q1", text="t1")  # step 1 -> slot1
    slot1 = wm.ring_buffer()[1]
    assert slot1.h_pre is not None
    # h_pre (captured before step 1) == the live state after step 0 (fp16 round-trip).
    for i in range(4):
        assert torch.allclose(slot1.h_pre[i].to(torch.float32), state0_live[i],
                              atol=1e-3)
    # and h (post-step) differs from h_pre (pre-step) -- the step did something.
    post = slot1.h[-1].to(torch.float32)
    pre = slot1.h_pre[-1].to(torch.float32)
    assert not torch.allclose(post, pre, atol=1e-3), "step did not change the state"


# ── M1: u (exact step-input) capture ──

def test_prestate_default_off_u_is_none():
    """capture_pre_state=False (default) -> every slot's u is None (the field
    defaults None and the u-capture branch is skipped). Backward-compatible with
    every existing positional/keyword RingSlot constructor (u defaults None)."""
    wm = _wm(ring_capacity=4, capture_pre_state=False)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    assert wm.ring_buffer()[0].u is None
    assert RingSlot(torch.zeros(1, 256), "ep", "text").u is None


def test_prestate_on_first_step_u_is_none():
    """u is captured under the same ``self.state is not None`` gate as h_pre --
    so on the FIRST step after a reset (state lazy-inits inside super().step),
    u is None too. Pins the timing symmetry with h_pre."""
    wm = _wm(ring_capacity=4, capture_pre_state=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    slot0 = wm.ring_buffer()[0]
    assert slot0.u is None


def test_prestate_on_captures_u_shape_dtype():
    """capture_pre_state=True -> from the SECOND step on, every kept slot's u is
    a detached fp32 [1,384] tensor == the EXACT step-input embedding (the
    ``input_embedding`` fed to super().step, post-pin/pre-SSM). fp32 (not fp16
    like h/h_pre) because the replay re-steps it and fp16 input would inject
    avoidable error; the state seed is the dominant fp16 source."""
    wm = _wm(ring_capacity=4, capture_pre_state=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    emb1 = _rand_emb(seed=1)
    wm.update(emb1, source_id="q1", text="t1")
    slot1 = wm.ring_buffer()[1]
    assert slot1.u is not None
    assert slot1.u.shape == (1, 384)
    assert slot1.u.dtype == torch.float32
    assert not slot1.u.requires_grad  # detached clone
    # u == the exact step-input (emb1), NOT embed(text) -- the whole point of
    # capturing u (code docs are injected by meaning, so embed(slot.text) would
    # diverge; here text==stepped input so they coincide, but u is the source of
    # truth).
    assert torch.equal(slot1.u, emb1.to(torch.float32))


def test_prestate_u_is_independent_clone():
    """slot.u is a detached clone: the captured tensor does not alias the
    caller's input_embedding (a later in-place op on the input must not perturb
    a previously captured slot.u). Same independence contract as h/h_pre."""
    wm = _wm(ring_capacity=4, capture_pre_state=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    emb1 = _rand_emb(seed=1)
    wm.update(emb1, source_id="q1", text="t1")
    slot1 = wm.ring_buffer()[1]
    u_frozen = slot1.u.clone()
    # mutate the caller's input in place -- slot1.u must be unaffected
    emb1.add_(99.0)
    assert torch.equal(slot1.u, u_frozen), "in-place mutate of input leaked into slot.u"


def test_prestate_roundtrip_via_slot_u_matches_emb():
    """Round-trip fidelity via the captured ``u`` (the replay_flatlast path):
    seed states from slot.h_pre, step slot.u (not a re-embedded text) via
    layer.step, and the last-layer state matches slot.h within atol 0.1. This
    pins that ``u`` is the correct replay input -- the fix for the code-doc
    embed-text-vs-summary divergence."""
    bb = JGSBackbone(BackboneConfig())
    wm = WorkingMemory(bb, ring_capacity=4, decay_alpha=1.0,
                       capture_pre_state=True, identity_instance=True)
    wm.update(_rand_emb(seed=0), source_id="q0", text="t0")
    wm.update(_rand_emb(seed=1), source_id="q1", text="t1")  # slot1
    slot1 = wm.ring_buffer()[1]
    assert slot1.u is not None and slot1.h_pre is not None
    states = [slot1.h_pre[i].to(torch.float32).detach() for i in range(4)]
    h = slot1.u.to(torch.float32).detach()   # [1,384] -- the captured step-input
    new_states = []
    for i, layer in enumerate(bb.layers):
        h, s = layer.step(h, states[i])
        new_states.append(s)
    replayed_last = new_states[-1].to(torch.float32)
    stored_last = slot1.h[-1].to(torch.float32)
    max_diff = (replayed_last - stored_last).abs().max().item()
    assert max_diff <= 0.1, (
        f"u-seeded round-trip diverged: max-abs-diff {max_diff:.4f} > 0.1 -- "
        "slot.u does not reproduce slot.h via layer.step")


# ── M2: _build_mixed_record(capture_pre_state=...) ──

def _slot_with_pre(text: str, source_id: str, slot_type: int | None,
                   y_seed: int = 0, h_pre: bool = True, u: bool = True) -> RingSlot:
    """Build a RingSlot with BOTH h and h_pre (4-layer [1,16,384] fp16 lists)
    and a step-input ``u`` [1,384] fp32.

    h and h_pre are independent random lists here -- the shape/alignment tests
    do not require them to be related (the round-trip fidelity tests above
    cover the h_pre->h relationship via a real WM step). ``u`` is a random
    [1,384] fp32 tensor (the exact step-input the replay re-steps)."""
    g = torch.Generator().manual_seed(y_seed)
    y = torch.randn(1, 256, generator=g)
    h_list = [torch.randn(1, 16, 384, generator=g).to(torch.float16)
              for _ in range(4)]
    hpre_list = None
    if h_pre:
        hpre_list = [torch.randn(1, 16, 384, generator=g).to(torch.float16)
                     for _ in range(4)]
    u_t = torch.randn(1, 384, generator=g) if u else None
    return RingSlot(y, source_id, text, pinned=False, h=h_list,
                    h_pre=hpre_list, u=u_t, slot_type=slot_type)


def _build_inputs_prestate(n_conv=2, n_ret=3, missing_pre_idx=None,
                           missing_u_idx=None):
    """Build a (ring, doc_embs, source_ids, slot_types) bundle where every slot
    has h + h_pre + u, except optionally one slot whose h_pre is None
    (``missing_pre_idx``) and/or whose u is None (``missing_u_idx``) -- to test
    the capture_pre_state guard on either missing field."""
    ring: list[RingSlot] = []
    doc_embs: list[torch.Tensor] = []
    source_ids: list[str] = []
    slot_types: list[int | None] = []
    idx = 0
    for _ in range(n_conv):
        sid = f"sess#msg{idx}"
        has_pre = missing_pre_idx != idx
        has_u = missing_u_idx != idx
        ring.append(_slot_with_pre(f"conv text {idx}", sid, 0, y_seed=idx,
                                   h_pre=has_pre, u=has_u))
        doc_embs.append(torch.randn(384))
        source_ids.append(sid)
        slot_types.append(0)
        idx += 1
    for _ in range(n_ret):
        sid = f"sess__ep{idx:04d}"
        has_pre = missing_pre_idx != idx
        has_u = missing_u_idx != idx
        ring.append(_slot_with_pre(f"retrieved episode {idx}", sid, 1,
                                   y_seed=idx, h_pre=has_pre, u=has_u))
        doc_embs.append(torch.randn(384))
        source_ids.append(sid)
        slot_types.append(1)
        idx += 1
    return ring, doc_embs, source_ids, slot_types


def test_build_prestate_off_emits_no_slots_pre_state():
    """capture_pre_state=False (default) -> NO slots_pre_state key in the record
    (byte-identical to the pre-Stage-3 trace the #5 readout / gate consume)."""
    ring, embs, sids, sts = _build_inputs_prestate(n_conv=2, n_ret=3)
    rec = _build_mixed_record(ring, torch.randn(384), LatentDynamicsHead(),
                              embs, sids, sts, False, "q",
                              capture_pre_state=False)
    assert rec is not None
    assert "slots_pre_state" not in rec
    # shared fields are unchanged
    assert rec["slots_h_raw"].shape == (5, 4, 16, 384)
    assert rec["slots_h_raw"].dtype == torch.float16


def test_build_prestate_on_emits_aligned_slots_pre_state():
    """capture_pre_state=True -> slots_pre_state [K,4,16,384] fp16 AND
    slots_step_input [K,384] fp32, both aligned with slots_h_raw / source_ids
    (same kept order, same K)."""
    ring, embs, sids, sts = _build_inputs_prestate(n_conv=2, n_ret=3)  # K=5
    rec = _build_mixed_record(ring, torch.randn(384), LatentDynamicsHead(),
                              embs, sids, sts, False, "q",
                              capture_pre_state=True)
    assert rec is not None
    assert "slots_pre_state" in rec
    ps = rec["slots_pre_state"]
    assert ps.shape == (5, 4, 16, 384)
    assert ps.dtype == torch.float16
    assert "slots_step_input" in rec
    su = rec["slots_step_input"]
    assert su.shape == (5, 384)
    assert su.dtype == torch.float32
    # aligned K with the other per-slot stacks
    assert ps.shape[0] == rec["slots_h_raw"].shape[0]
    assert su.shape[0] == rec["slots_h_raw"].shape[0]
    assert ps.shape[0] == len(rec["source_ids"])
    assert ps.shape[0] == rec["slot_types"].shape[0]


def test_build_prestate_off_emits_no_slots_step_input():
    """capture_pre_state=False -> NO slots_step_input key (byte-identical to the
    pre-Stage-3-u trace; only the capture path stacks it)."""
    ring, embs, sids, sts = _build_inputs_prestate(n_conv=2, n_ret=3)
    rec = _build_mixed_record(ring, torch.randn(384), LatentDynamicsHead(),
                              embs, sids, sts, False, "q",
                              capture_pre_state=False)
    assert rec is not None
    assert "slots_step_input" not in rec
    assert "slots_pre_state" not in rec


def test_build_prestate_on_drops_slot_with_missing_h_pre():
    """When capture_pre_state=True, a slot whose h_pre is None is DROPPED (a
    missing pre-state would seed a bogus replay) -- and the drop keeps
    slots_pre_state / slots_step_input aligned with the surviving source_ids."""
    # slot index 0 (sess#msg0) has h_pre=None -> must be dropped under capture.
    ring, embs, sids, sts = _build_inputs_prestate(n_conv=2, n_ret=3,
                                                   missing_pre_idx=0)
    rec = _build_mixed_record(ring, torch.randn(384), LatentDynamicsHead(),
                              embs, sids, sts, False, "q",
                              capture_pre_state=True)
    assert rec is not None
    # 5 -> 4 surviving slots (sess#msg0 dropped)
    assert rec["slots_pre_state"].shape == (4, 4, 16, 384)
    assert rec["slots_step_input"].shape == (4, 384)
    assert rec["slots_h_raw"].shape == (4, 4, 16, 384)
    assert "sess#msg0" not in rec["source_ids"]
    assert len(rec["source_ids"]) == 4
    # the default-off path does NOT drop it (h_pre guard not taken) -> 5 kept
    rec_off = _build_mixed_record(ring, torch.randn(384), LatentDynamicsHead(),
                                  embs, sids, sts, False, "q",
                                  capture_pre_state=False)
    assert rec_off is not None
    assert "sess#msg0" in rec_off["source_ids"]
    assert rec_off["slots_h_raw"].shape == (5, 4, 16, 384)


def test_build_prestate_on_drops_slot_with_missing_u():
    """When capture_pre_state=True, a slot whose u is None is DROPPED (a missing
    step-input would make the replay re-step a bogus vector) -- mirrors the
    h_pre drop, and the drop keeps slots_step_input aligned with the survivors.
    The kept filter requires BOTH h_pre AND u."""
    # slot index 1 (sess#msg1) has u=None but h_pre present -> still dropped.
    ring, embs, sids, sts = _build_inputs_prestate(n_conv=2, n_ret=3,
                                                   missing_u_idx=1)
    rec = _build_mixed_record(ring, torch.randn(384), LatentDynamicsHead(),
                              embs, sids, sts, False, "q",
                              capture_pre_state=True)
    assert rec is not None
    assert rec["slots_pre_state"].shape == (4, 4, 16, 384)
    assert rec["slots_step_input"].shape == (4, 384)
    assert "sess#msg1" not in rec["source_ids"]
    # default-off does NOT drop it (u guard not taken) -> 5 kept
    rec_off = _build_mixed_record(ring, torch.randn(384), LatentDynamicsHead(),
                                  embs, sids, sts, False, "q",
                                  capture_pre_state=False)
    assert "sess#msg1" in rec_off["source_ids"]
    assert rec_off["slots_h_raw"].shape == (5, 4, 16, 384)