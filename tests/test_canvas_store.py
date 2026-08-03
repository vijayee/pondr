"""B4: ``HippocampalStore`` canvas CRUD -- the WaveDB-persistent task-canvas store.

Mirrors the B1 scene-store tests. Pins ``encode_canvas``/``get_canvas`` round-
trip, ``delete_canvas`` (``"del"`` not ``"delete"``, no residue), ``next_canvas_id``
monotonic, ``canvas_ids_for_user`` two-user isolation, ``touch_canvas`` mutable-
field rewrite, ``set_active_canvas`` one-active-per-user, ``get_active_canvas``,
and ``reclaim_canvases`` (never active, floor 15, oldest-by-``updated_ts``).
Offline: tmp_path WaveDB store; no Bonsai, no GPU.
"""

from __future__ import annotations

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)",
                allow_module_level=True)

from src.memory.store import HippocampalStore


def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": True, "embedding_dim": 384}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


def _encode(store, cid, *, label="task", progress=0, active="1", user="alice",
            ts="2026-08-01T10:00:00", mermaid=None, node_mapping=None):
    store.encode_canvas(
        cid, mermaid=mermaid or f"%%{{progress: {progress}}}%%\nflowchart TD",
        task_label=label, progress=progress, created_ts=ts, updated_ts=ts,
        user_id=user, active=active, node_mapping=node_mapping or {})


# ── encode / get round-trip ─────────────────────────────────────────────────

def test_encode_get_canvas_roundtrip(tmp_path):
    store = _store(tmp_path)
    cid = store.next_canvas_id()
    assert cid == "canvas_000001"
    _encode(store, cid, label="build-api", progress=40, active="1",
            mermaid="%%{progress: 40}%%\nflowchart TD\n    N1[\"x\"]",
            node_mapping={"N1": "ep_001"})
    cv = store.get_canvas(cid)
    assert cv is not None
    assert cv["canvas_id"] == cid
    assert cv["task_label"] == "build-api"
    assert cv["progress"] == 40
    assert cv["user_id"] == "alice"
    assert cv["active"] is True
    assert cv["mermaid"].startswith("%%{progress: 40}")
    assert cv["node_mapping"] == {"N1": "ep_001"}
    # Graph edges present.
    assert store.db.get_sync(f"memory/spo/{cid}/instanceOf/CanvasBlock") is not None
    assert store.db.get_sync(f"memory/spo/{cid}/has_topic/T:build-api") is not None
    assert store.db.get_sync(f"memory/spo/U:alice/owns_canvas/{cid}") is not None
    store.close()


def test_get_canvas_missing_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.get_canvas("canvas_999999") is None
    store.close()


# ── next_canvas_id monotonic ─────────────────────────────────────────────────

def test_next_canvas_id_increments(tmp_path):
    store = _store(tmp_path)
    assert store.next_canvas_id() == "canvas_000001"
    assert store.next_canvas_id() == "canvas_000002"
    assert store.next_canvas_id() == "canvas_000003"
    store.close()


# ── canvas_ids_for_user two-user isolation ───────────────────────────────────

def test_canvas_ids_for_user_two_user_isolation(tmp_path):
    store = _store(tmp_path)
    a1 = store.next_canvas_id(); a2 = store.next_canvas_id(); b1 = store.next_canvas_id()
    _encode(store, a1, user="alice"); _encode(store, a2, user="alice")
    _encode(store, b1, user="bob")
    assert store.canvas_ids_for_user("alice") == {a1, a2}
    assert store.canvas_ids_for_user("bob") == {b1}
    assert store.canvas_ids_for_user("nobody") == set()
    store.close()


# ── delete_canvas (del, no residue) ──────────────────────────────────────────

def test_delete_canvas_leaves_no_residue(tmp_path):
    store = _store(tmp_path)
    cid = store.next_canvas_id()
    _encode(store, cid, label="storage", user="alice",
            mermaid="%%{progress: 0}%%\nflowchart TD")
    assert cid in store.canvas_ids_for_user("alice")
    store.delete_canvas(cid)
    assert store.get_canvas(cid) is None
    # No content keys.
    for k, _ in store.db.create_read_stream(start=f"content/canvas/{cid}/",
                                            end=f"content/canvas/{cid}/\x7f"):
        pytest.fail(f"orphan content key after delete: {k}")
    # No SPO edge residue for any edge type.
    for k, _ in store.db.create_read_stream(start=f"memory/spo/{cid}/",
                                            end=f"memory/spo/{cid}/\x7f"):
        pytest.fail(f"orphan SPO edge after delete: {k}")
    assert store.db.get_sync(f"memory/spo/U:alice/owns_canvas/{cid}") is None
    store.close()


# ── touch_canvas ────────────────────────────────────────────────────────────

def test_touch_canvas_rewrites_mutable_fields(tmp_path):
    store = _store(tmp_path)
    cid = store.next_canvas_id()
    _encode(store, cid, progress=0, mermaid="%%{progress: 0}%%\nflowchart TD")
    store.touch_canvas(cid, mermaid="%%{progress: 70}%%\nflowchart TD\n    N1[\"x\"]",
                       progress=70, node_mapping={"N1": "ep_001"})
    cv = store.get_canvas(cid)
    assert cv["progress"] == 70
    assert cv["mermaid"].startswith("%%{progress: 70}")
    assert cv["node_mapping"] == {"N1": "ep_001"}
    # updated_ts bumped; created_ts unchanged.
    assert cv["created_ts"] == "2026-08-01T10:00:00"
    assert cv["updated_ts"] != "2026-08-01T10:00:00"
    store.close()


def test_touch_canvas_missing_is_noop(tmp_path):
    store = _store(tmp_path)
    store.touch_canvas("canvas_999999", mermaid="x", progress=5)  # no raise
    assert store.get_canvas("canvas_999999") is None
    store.close()


# ── set_active_canvas / get_active_canvas (one active per user) ─────────────

def test_set_active_canvas_flips_prior(tmp_path):
    store = _store(tmp_path)
    a = store.next_canvas_id(); b = store.next_canvas_id()
    _encode(store, a, active="1")
    _encode(store, b, active="0")
    assert store.get_active_canvas("alice")["canvas_id"] == a
    store.set_active_canvas("alice", b)
    assert store.get_active_canvas("alice")["canvas_id"] == b
    assert store.get_canvas(a)["active"] is False
    assert store.get_canvas(b)["active"] is True
    store.close()


def test_set_active_canvas_none_clears(tmp_path):
    store = _store(tmp_path)
    a = store.next_canvas_id()
    _encode(store, a, active="1")
    store.set_active_canvas("alice", None)
    assert store.get_active_canvas("alice") is None
    assert store.get_canvas(a)["active"] is False
    store.close()


def test_get_active_canvas_none_when_no_active(tmp_path):
    store = _store(tmp_path)
    a = store.next_canvas_id()
    _encode(store, a, active="0")  # historical, no active
    assert store.get_active_canvas("alice") is None
    store.close()


# ── reclaim_canvases ─────────────────────────────────────────────────────────

def test_reclaim_never_deletes_active(tmp_path):
    store = _store(tmp_path)
    active = store.next_canvas_id()
    _encode(store, active, active="1", ts="2026-08-01T10:00:00")
    # Many historical canvases beyond the floor.
    for i in range(5):
        c = store.next_canvas_id()
        _encode(store, c, active="0", ts=f"2026-08-01T10:0{i}:00")
    # floor 15 > total -> nothing deleted; the active is safe regardless.
    assert store.reclaim_canvases("alice", min_keep=15) == 0
    assert store.get_canvas(active)["active"] is True
    store.close()


def test_reclaim_floor_and_oldest_by_updated_ts(tmp_path):
    store = _store(tmp_path)
    active = store.next_canvas_id()
    _encode(store, active, active="1", ts="2026-08-01T10:00:00")
    # 6 historical canvases with increasing updated_ts.
    cids = []
    for i in range(6):
        c = store.next_canvas_id()
        cids.append(c)
        _encode(store, c, active="0", ts=f"2026-08-01T10:0{i}:00")
    # floor 3 -> delete the 3 oldest (the first 3 by updated_ts), keep 3 newest.
    deleted = store.reclaim_canvases("alice", min_keep=3)
    assert deleted == 3
    remaining = store.canvas_ids_for_user("alice")
    assert active in remaining
    # The 3 oldest historical are gone; the 3 newest historical survive.
    assert cids[0] not in remaining and cids[1] not in remaining
    assert cids[2] not in remaining
    assert cids[3] in remaining and cids[4] in remaining and cids[5] in remaining
    store.close()