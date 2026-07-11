"""Tests for src/memory/edge_meta.py -- the Phase 3b per-edge sidecar."""

from __future__ import annotations

import hashlib
import json

from src.memory.episode import Episode
from src.memory.store import HippocampalStore, safe_edge_component
from src.memory.edge_meta import (
    batch_update_edge_meta,
    edge_meta_key,
    edge_meta_put_op,
    get_edge_meta,
    is_edge_current,
    set_edge_state,
    update_edge_meta,
)
from src.memory.forgetting import default_meta
from src.gnn.semantic_memory import SemanticMemoryWriter


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid, entities=None):
    store.encode_episode(Episode(
        id=eid, timestamp="t", summary=f"s {eid}", full_text=f"f {eid}",
        entities=entities or [],
    ))


# ── key hashing ──
def test_safe_edge_component_passes_plain_hashes_slash():
    assert safe_edge_component("ep_000001") == "ep_000001"
    assert safe_edge_component("has_entity") == "has_entity"
    hashed = safe_edge_component("E:Alice/Bob")
    assert hashed.startswith("h_")
    assert hashed == "h_" + hashlib.sha256("E:Alice/Bob".encode()).hexdigest()[:16]
    # NUL also triggers hashing
    assert safe_edge_component("a\x00b").startswith("h_")


def test_edge_meta_key_shape_and_hashing():
    k = edge_meta_key("ep_000001", "has_entity", "E:Alice")
    assert k == "content/edge/ep_000001/has_entity/E:Alice"
    # slash-bearing object is hashed into one segment (no extra '/')
    k2 = edge_meta_key("ep_000001", "has_entity", "E:Alice/Bob")
    parts = k2.split("/")
    assert parts[0] == "content"
    assert parts[1] == "edge"
    assert parts[2] == "ep_000001"
    assert parts[3] == "has_entity"
    assert parts[4].startswith("h_")  # the object, hashed
    assert len(parts) == 5  # exactly one segment per component (no collapse)


def test_edge_meta_key_does_not_collide_with_graph_or_archive_key():
    s, p, o = "ep_000001", "has_entity", "E:Alice/Bob"
    sidecar = edge_meta_key(s, p, o)
    archive = SemanticMemoryWriter._archive_key(s, p, o)
    # distinct namespaces, same hashing -> both hash the object but differ in prefix
    assert sidecar.startswith("content/edge/")
    assert archive.startswith("archive/edge/")
    assert sidecar != archive
    # the live graph key carries the literal slash; the sidecar key hashes it
    assert "E:Alice/Bob" not in sidecar


# ── lazy-create / read ──
def test_get_edge_meta_missing_returns_default(tmp_path):
    store = _store(tmp_path)
    meta = get_edge_meta(store, "ep_000001", "has_entity", "E:Alice")
    assert meta == default_meta()
    assert meta["state"] == "current"
    store.close()


def test_get_edge_meta_merges_over_defaults(tmp_path):
    store = _store(tmp_path)
    # write a partial/old sidecar missing newer fields
    store.db.batch_sync([{
        "type": "put",
        "key": edge_meta_key("ep_1", "has_entity", "E:A"),
        "value": json.dumps({"state": "archived", "utility_score": 0.02}),
    }])
    meta = get_edge_meta(store, "ep_1", "has_entity", "E:A")
    assert meta["state"] == "archived"
    assert meta["utility_score"] == 0.02
    # newer fields present from default_meta (graceful schema growth)
    assert meta["ltp_phase"] == "early"
    assert meta["retrieval_timestamps"] == []
    store.close()


def test_get_edge_meta_bad_json_falls_back_to_default(tmp_path):
    store = _store(tmp_path)
    store.db.batch_sync([{
        "type": "put",
        "key": edge_meta_key("ep_1", "has_entity", "E:A"),
        "value": "not json{",
    }])
    assert get_edge_meta(store, "ep_1", "has_entity", "E:A") == default_meta()
    store.close()


# ── write / roundtrip ──
def test_update_then_get_roundtrip(tmp_path):
    store = _store(tmp_path)
    meta = default_meta()
    meta["utility_decay_rate"] = 0.006
    meta["access_count"] = 7
    meta["retrieval_timestamps"] = ["2026-01-01T00:00:00"]
    update_edge_meta(store, "ep_1", "has_entity", "E:A", meta)
    back = get_edge_meta(store, "ep_1", "has_entity", "E:A")
    assert back["utility_decay_rate"] == 0.006
    assert back["access_count"] == 7
    assert back["retrieval_timestamps"] == ["2026-01-01T00:00:00"]
    store.close()


def test_edge_meta_put_op_is_batchable(tmp_path):
    store = _store(tmp_path)
    # include the put-op in a larger caller batch (no separate batch_sync)
    op = edge_meta_put_op("ep_1", "has_entity", "E:A", default_meta())
    assert op["type"] == "put"
    assert op["key"] == edge_meta_key("ep_1", "has_entity", "E:A")
    store.db.batch_sync([op])
    assert get_edge_meta(store, "ep_1", "has_entity", "E:A")["state"] == "current"
    store.close()


def test_batch_update_edge_meta_atomic(tmp_path):
    store = _store(tmp_path)
    updates = [
        ("ep_1", "has_entity", "E:A", {**default_meta(), "access_count": 1}),
        ("ep_2", "has_topic", "T:db", {**default_meta(), "access_count": 2}),
        ("ep_3", "has_entity", "E:A/B", {**default_meta(), "state": "archived"}),
    ]
    batch_update_edge_meta(store, updates)
    assert get_edge_meta(store, "ep_1", "has_entity", "E:A")["access_count"] == 1
    assert get_edge_meta(store, "ep_2", "has_topic", "T:db")["access_count"] == 2
    assert get_edge_meta(store, "ep_3", "has_entity", "E:A/B")["state"] == "archived"
    store.close()


def test_batch_update_empty_is_noop(tmp_path):
    store = _store(tmp_path)
    batch_update_edge_meta(store, [])  # must not raise
    store.close()


# ── set_edge_state ──
def test_set_edge_state_deprecates_with_validity_end(tmp_path):
    store = _store(tmp_path)
    meta = set_edge_state(
        store, "ep_1", "has_entity", "E:A", "deprecated", validity_end="2026-02-01"
    )
    assert meta["state"] == "deprecated"
    assert meta["validity_end"] == "2026-02-01"
    back = get_edge_meta(store, "ep_1", "has_entity", "E:A")
    assert back["state"] == "deprecated"
    assert back["validity_end"] == "2026-02-01"
    store.close()


def test_set_edge_state_preserves_existing_meta(tmp_path):
    store = _store(tmp_path)
    # pre-existing sidecar with access history
    m = default_meta()
    m["access_count"] = 5
    m["retrieval_timestamps"] = ["2026-01-01T00:00:00"]
    update_edge_meta(store, "ep_1", "has_entity", "E:A", m)
    set_edge_state(store, "ep_1", "has_entity", "E:A", "archived")
    back = get_edge_meta(store, "ep_1", "has_entity", "E:A")
    assert back["state"] == "archived"
    # the access history is preserved (RMW didn't clobber it)
    assert back["access_count"] == 5
    assert back["retrieval_timestamps"] == ["2026-01-01T00:00:00"]
    store.close()


# ── is_edge_current (the edge-level filter predicate) ──
def test_is_edge_current_no_sidecar_is_true(tmp_path):
    store = _store(tmp_path)
    assert is_edge_current(store, "ep_1", "has_entity", "E:A") is True
    store.close()


def test_is_edge_current_reflects_state(tmp_path):
    store = _store(tmp_path)
    set_edge_state(store, "ep_1", "has_entity", "E:A", "deprecated")
    assert is_edge_current(store, "ep_1", "has_entity", "E:A") is False
    set_edge_state(store, "ep_2", "has_topic", "T:db", "archived")
    assert is_edge_current(store, "ep_2", "has_topic", "T:db") is False
    # a current sidecar is still current
    update_edge_meta(store, "ep_3", "has_entity", "E:A", default_meta())
    assert is_edge_current(store, "ep_3", "has_entity", "E:A") is True
    store.close()


# ── store wrapper delegation ──
def test_store_wrappers_delegate(tmp_path):
    store = _store(tmp_path)
    assert store.get_edge_meta("ep_1", "has_entity", "E:A") == default_meta()
    store.update_edge_meta_batch([
        ("ep_1", "has_entity", "E:A", {**default_meta(), "access_count": 3}),
    ])
    assert store.get_edge_meta("ep_1", "has_entity", "E:A")["access_count"] == 3
    assert store.is_edge_current("ep_1", "has_entity", "E:A") is True
    store.set_edge_state("ep_1", "has_entity", "E:A", "superseded", validity_end="2026-09-01")
    assert store.is_edge_current("ep_1", "has_entity", "E:A") is False
    assert store.get_edge_meta("ep_1", "has_entity", "E:A")["validity_end"] == "2026-09-01"
    store.close()


# ── regression: the safe() extraction didn't break 3a archive_edge ──
def test_archive_key_still_works_after_extraction():
    # _archive_key is a pure static method; no store needed. It must still hash
    # slash-bearing components the same way after safe() was extracted to store.
    ak = SemanticMemoryWriter._archive_key("ep_1", "has_entity", "E:A/B")
    assert ak == "archive/edge/ep_1/has_entity/" + safe_edge_component("E:A/B")
    assert ak.startswith("archive/edge/")
    assert "h_" in ak
    # plain components are not hashed
    ak2 = SemanticMemoryWriter._archive_key("ep_1", "has_entity", "E:Alice")
    assert ak2 == "archive/edge/ep_1/has_entity/E:Alice"