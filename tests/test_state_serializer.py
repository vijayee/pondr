"""Offline tests for the Phase 2c JGS state-save plumbing.

Covers the *mechanism* only: :mod:`src.subconscious.state_serializer`
(serialize/deserialize + snapshot/restore on a real ``JGSInstance``) and the
``HippocampalStore.save_jgs_state``/``load_jgs_state`` round-trip. No save-trigger
policy is exercised — that is intentionally not part of this plumbing. All CPU,
ReferenceSSM, no pod, no GLiNER/Bonsai.
"""

from __future__ import annotations

import pytest
import torch

from src.memory.store import HippocampalStore
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig, INSTANCE_CONFIGS
from src.subconscious.instance import JGSInstance
from src.subconscious.state_serializer import (
    JGSSnapshot,
    deserialize,
    restore_to_instance,
    serialize,
    snapshot_from_instance,
)


def _real_shape() -> list[torch.Size]:
    """The configured WM recurrent-state shape: 4 x [1, 16, 384]."""
    return [torch.Size([1, 16, 384])] * 4


def _random_snapshot(input_count=3, timestamp=123.0, metadata=None) -> JGSSnapshot:
    torch.manual_seed(0)
    return JGSSnapshot(
        state_tensors=[torch.randn(1, 16, 384, dtype=torch.float32) for _ in range(4)],
        input_count=input_count,
        timestamp=timestamp,
        metadata=metadata if metadata is not None else {"active_domains": ["database", "coding"]},
    )


# ── serialize / deserialize ──

def test_round_trip_is_element_exact():
    snap = _random_snapshot()
    blob = serialize(snap)
    back = deserialize(blob)
    assert len(back.state_tensors) == len(snap.state_tensors)
    for a, b in zip(snap.state_tensors, back.state_tensors):
        assert a.shape == b.shape
        assert b.dtype == torch.float32
        assert torch.equal(a, b)
    assert back.input_count == snap.input_count
    assert back.timestamp == snap.timestamp
    assert back.metadata == snap.metadata


def test_blob_is_nul_free_ascii():
    blob = serialize(_random_snapshot())
    assert "\x00" not in blob
    # base64 + JSON — no control chars beyond what JSON allows; every byte < 128.
    assert all(ord(c) < 128 for c in blob)


def test_serialize_is_deterministic():
    snap = _random_snapshot()
    assert serialize(snap) == serialize(snap)


def test_zero_state_round_trips():
    snap = JGSSnapshot([torch.zeros(1, 16, 384, dtype=torch.float32) for _ in range(4)])
    back = deserialize(serialize(snap))
    assert all(torch.equal(t, torch.zeros_like(t)) for t in back.state_tensors)


def test_metadata_with_nested_structures_round_trips():
    snap = _random_snapshot(
        metadata={"active_domains": ["a", "b"], "nested": {"x": [1, 2, 3]}, "flag": True},
    )
    back = deserialize(serialize(snap))
    assert back.metadata == snap.metadata


def test_deserialize_rejects_empty_or_nul_blob():
    with pytest.raises(ValueError):
        deserialize("")
    with pytest.raises(ValueError):
        deserialize("abc\x00def")


def test_serialize_rejects_empty_state_tensors():
    with pytest.raises(ValueError):
        serialize(JGSSnapshot(state_tensors=[]))


def test_deserialize_rejects_wrong_version_or_dtype():
    import json as _json
    snap = _random_snapshot()
    obj = _json.loads(serialize(snap))
    obj["v"] = 999
    with pytest.raises(ValueError):
        deserialize(_json.dumps(obj))
    obj = _json.loads(serialize(snap))
    obj["dtype"] = "float16"
    with pytest.raises(ValueError):
        deserialize(_json.dumps(obj))


def test_deserialize_detects_element_count_mismatch():
    snap = _random_snapshot()
    import json as _json
    obj = _json.loads(serialize(snap))
    # Lie about the shapes: claim 5 layers but the payload only has 4.
    obj["shapes"] = obj["shapes"] + [[1, 16, 384]]
    with pytest.raises(ValueError):
        deserialize(_json.dumps(obj))


def test_non_float32_tensor_is_rejected():
    snap = JGSSnapshot([torch.zeros(1, 16, 384, dtype=torch.int32) for _ in range(4)])
    with pytest.raises((TypeError, RuntimeError)):
        serialize(snap)


# ── instance snapshot / restore ──

def _make_instance() -> JGSInstance:
    bb = JGSBackbone(BackboneConfig())
    return JGSInstance(bb, INSTANCE_CONFIGS["working_memory"])


def test_snapshot_from_instance_requires_state():
    inst = _make_instance()
    assert inst.state is None
    with pytest.raises(ValueError):
        snapshot_from_instance(inst)


def test_snapshot_and_restore_preserves_state_after_stepping():
    inst = _make_instance()
    inst.reset_state(1)
    # Step once so the state is non-zero and real-shaped.
    inst.step(torch.randn(1, 384, dtype=torch.float32))
    assert inst.state is not None
    shapes = [t.shape for t in inst.state]
    assert shapes == _real_shape()

    snap = snapshot_from_instance(inst, input_count=7, timestamp=42.0,
                                  metadata={"last_query_type": "graph_retrieve"})
    # Mutating the live state must not affect the snapshot (detached clones).
    live_before = [t.clone() for t in inst.state]
    inst.step(torch.randn(1, 384, dtype=torch.float32))
    assert torch.equal(snap.state_tensors[-1], live_before[-1])

    # Restore into a fresh instance and confirm element-equality.
    fresh = _make_instance()
    restore_to_instance(fresh, snap)
    assert fresh.state is not None
    assert [t.shape for t in fresh.state] == _real_shape()
    for a, b in zip(snap.state_tensors, fresh.state):
        assert torch.equal(a, b)


def test_restore_to_instance_handles_empty_snapshot():
    inst = _make_instance()
    with pytest.raises(ValueError):
        restore_to_instance(inst, JGSSnapshot(state_tensors=[]))


# ── store round-trip ──

def test_store_save_load_round_trips(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        snap = _random_snapshot()
        blob = serialize(snap)
        store.save_jgs_state("user_alice", blob)
        loaded = store.load_jgs_state("user_alice")
        assert loaded == blob
        # The loaded blob must deserialize to the same tensors.
        back = deserialize(loaded)
        for a, b in zip(snap.state_tensors, back.state_tensors):
            assert torch.equal(a, b)
    finally:
        store.close()


def test_store_load_returns_none_for_unknown_user(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        assert store.load_jgs_state("nobody") is None
        assert store.load_jgs_state("someone", scope="other_scope") is None
    finally:
        store.close()


def test_store_scope_isolates_users_and_instances(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        snap_a = _random_snapshot(metadata={"who": "alice"})
        snap_b = _random_snapshot(metadata={"who": "bob"})
        store.save_jgs_state("alice", serialize(snap_a), scope="working_memory")
        store.save_jgs_state("alice", serialize(snap_b), scope="future_gate")
        store.save_jgs_state("bob", serialize(snap_b), scope="working_memory")

        assert deserialize(store.load_jgs_state("alice")).metadata == {"who": "alice"}
        assert deserialize(store.load_jgs_state("alice", scope="future_gate")).metadata == {"who": "bob"}
        assert deserialize(store.load_jgs_state("bob")).metadata == {"who": "bob"}
        assert store.load_jgs_state("bob", scope="future_gate") is None
    finally:
        store.close()


def test_store_persists_across_reopen(tmp_path):
    """State survives close/reopen (the per-user WM-is-cross-session property)."""
    db_path = str(tmp_path / "db")
    snap = _random_snapshot()
    blob = serialize(snap)
    store = HippocampalStore(db_path)
    try:
        store.save_jgs_state("alice", blob)
    finally:
        store.close()

    store2 = HippocampalStore(db_path)
    try:
        loaded = store2.load_jgs_state("alice")
        assert loaded == blob
        back = deserialize(loaded)
        for a, b in zip(snap.state_tensors, back.state_tensors):
            assert torch.equal(a, b)
    finally:
        store2.close()


def test_store_rejects_bad_user_id_and_scope(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        with pytest.raises(ValueError):
            store.save_jgs_state("user/with/slash", serialize(_random_snapshot()))
        with pytest.raises(ValueError):
            store.save_jgs_state("alice", serialize(_random_snapshot()), scope="a/b")
        with pytest.raises(ValueError):
            store.save_jgs_state("alice", serialize(_random_snapshot()), scope="")
        with pytest.raises(ValueError):
            store.save_jgs_state("alice", "")  # empty blob would round-trip as None
        with pytest.raises(ValueError):
            store.load_jgs_state("user\x00bad")
    finally:
        store.close()


def test_store_end_to_end_save_restore_through_instance(tmp_path):
    """Full plumbing: step an instance, snapshot, store, reopen, restore into a
    fresh instance — the restored state is element-equal to the original."""
    db_path = str(tmp_path / "db")

    inst = _make_instance()
    inst.reset_state(1)
    inst.step(torch.randn(1, 384, dtype=torch.float32))
    inst.step(torch.randn(1, 384, dtype=torch.float32))
    original = [t.clone() for t in inst.state]
    snap = snapshot_from_instance(inst, input_count=2, timestamp=99.0)

    store = HippocampalStore(db_path)
    try:
        store.save_jgs_state("alice", serialize(snap))
    finally:
        store.close()

    store2 = HippocampalStore(db_path)
    try:
        loaded = store2.load_jgs_state("alice")
        assert loaded is not None
        snap2 = deserialize(loaded)
        fresh = _make_instance()
        restore_to_instance(fresh, snap2)
        for a, b in zip(original, fresh.state):
            assert torch.equal(a, b)
    finally:
        store2.close()