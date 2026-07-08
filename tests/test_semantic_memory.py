"""Tests for semantic-memory storage (``src/gnn/semantic_memory.py`` + store ops)."""

from __future__ import annotations

from src.memory.episode import Episode
from src.memory.store import HippocampalStore, _b2s
from src.gnn.semantic_memory import SemanticMemoryWriter


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid, entities=None):
    store.encode_episode(Episode(
        id=eid, timestamp="t", summary=f"s {eid}", full_text=f"f {eid}",
        entities=entities or [],
    ))


def test_create_abstract_marks_sources_and_excludes_from_default(tmp_path):
    store = _store(tmp_path)
    for i in range(1, 4):
        _encode(store, f"ep_00000{i}", entities=["Alice"])
    w = SemanticMemoryWriter(store)
    assert store.default_episode_ids() == ["ep_000001", "ep_000002", "ep_000003"]

    mid = w.create_abstract(["ep_000001", "ep_000002"], "Alice interactions")
    assert mid.startswith("M:")
    assert store.is_abstracted("ep_000001")
    assert store.is_abstracted("ep_000002")
    assert not store.is_abstracted("ep_000003")
    # Default query excludes abstracted; opt-in includes them.
    assert store.default_episode_ids() == ["ep_000003"]
    assert store.default_episode_ids(include_abstracted=True) == [
        "ep_000001", "ep_000002", "ep_000003",
    ]
    # consolidation_window_start was set on the sources.
    assert _b2s(store.db.get_sync("content/ep/ep_000001/consolidation_window_start"))
    store.close()


def test_abstracts_edges_recoverable_via_graph(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001", entities=["Alice"])
    _encode(store, "ep_000002", entities=["Alice"])
    w = SemanticMemoryWriter(store)
    mid = w.create_abstract(["ep_000001", "ep_000002"], "Alice gist")
    assert sorted(w.abstracted_episodes(mid)) == ["ep_000001", "ep_000002"]
    ab = w.get_abstract(mid)
    assert ab["summary"] == "Alice gist"
    assert sorted(ab["sources"]) == ["ep_000001", "ep_000002"]
    store.close()


def test_supersedes_edge_written(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    w = SemanticMemoryWriter(store)
    mid1 = w.create_abstract(["ep_000001"], "v1")
    mid2 = w.create_abstract(["ep_000001"], "v2", supersedes=mid1)
    # mid2 supersedes mid1: graph edge (mid2, supersedes, mid1).
    r = store.graph.query().vertex(mid2).out("supersedes").execute_sync()
    try:
        assert mid1 in list(r.vertices)
    finally:
        r.close()
    store.close()


def test_archive_edge_removes_from_live_graph_and_is_recoverable(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001", entities=["Alice"])
    w = SemanticMemoryWriter(store)

    # Before: ep_000001 has_entity E:Alice.
    r0 = store.graph.query().vertex("ep_000001").out("has_entity").execute_sync()
    try: assert "E:Alice" in list(r0.vertices)
    finally: r0.close()

    ak = w.archive_edge("ep_000001", "has_entity", "E:Alice", reason="low salience")
    assert ak.startswith("archive/edge/")

    # After: the live triple is gone.
    r1 = store.graph.query().vertex("ep_000001").out("has_entity").execute_sync()
    try: assert list(r1.vertices) == []
    finally: r1.close()

    # The archived record is recoverable.
    rec = w.read_archived_edge(ak)
    assert rec["subject"] == "ep_000001"
    assert rec["predicate"] == "has_entity"
    assert rec["object"] == "E:Alice"
    assert rec["reason"] == "low salience"
    store.close()


def test_create_abstract_rejects_empty_inputs(tmp_path):
    store = _store(tmp_path)
    w = SemanticMemoryWriter(store)
    try:
        w.create_abstract([], "x")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        w.create_abstract(["ep_000001"], "")
        assert False, "expected ValueError"
    except ValueError:
        pass
    store.close()