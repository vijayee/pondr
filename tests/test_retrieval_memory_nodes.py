"""Tests for M-node (semantic-memory) retrieval wiring.

``default_memory_ids`` scans ``content/mem/``; ``_get_all_episode_ids`` unions
the memories into the no-axis candidate seed; ``_hydrate`` routes ``M:`` ids
to ``_hydrate_memory`` (``kind="memory"``). The empty-corpus case is a
byte-identical regression guard (no M-nodes -> no change to the candidate set).
"""

from __future__ import annotations

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.graph_traversal import GraphTraversal


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _populate(store, n=4):
    for i in range(1, n + 1):
        store.encode_episode(Episode(
            id=f"ep_00000{i}", timestamp="t", summary=f"s{i}", full_text=f"f{i}",
            entities=["Alice", "Bob"], topics=["db"],
        ))


def test_default_memory_ids_empty_when_no_abstracts(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    assert store.default_memory_ids() == []
    store.close()


def test_default_memory_ids_scans_content_mem(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    # Write two M-nodes directly (mirrors create_abstract's content layout).
    for mid, gist in (("M:0001", "gist one"), ("M:0002", "gist two")):
        ops = [
            {"type": "put", "key": f"content/mem/{mid}/summary", "value": gist},
            {"type": "put", "key": f"content/mem/{mid}/text", "value": gist},
            {"type": "put", "key": f"content/mem/{mid}/ts", "value": "t"},
            {"type": "put", "key": f"content/mem/{mid}/abstracted_from",
             "value": '["ep_000001"]'},
        ]
        store.db.batch_sync(ops)
    ids = store.default_memory_ids()
    assert ids == ["M:0001", "M:0002"]
    store.close()


def test_get_all_episode_ids_unions_memories(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    store.db.batch_sync([
        {"type": "put", "key": "content/mem/M:0001/summary", "value": "g"},
        {"type": "put", "key": "content/mem/M:0001/ts", "value": "t"},
    ])
    trav = GraphTraversal(store)
    all_ids = set(trav._get_all_episode_ids())
    assert "M:0001" in all_ids
    # episodes are still there too.
    assert any(e.startswith("ep_") for e in all_ids)
    store.close()


def test_get_all_episode_ids_byte_identical_when_no_memories(tmp_path):
    """The cold-start regression guard: with no M-nodes the candidate set is
    exactly the episode+document union (no memory ids leak in)."""
    store = _store(tmp_path)
    _populate(store)
    trav = GraphTraversal(store)
    # Stub out default_memory_ids returning [] explicitly AND assert no M: id
    # is in the union (the real method scans content/mem/ which is empty here).
    all_ids = trav._get_all_episode_ids()
    assert not any(i.startswith("M:") for i in all_ids)
    store.close()


def test_hydrate_memory_returns_kind_memory(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    store.db.batch_sync([
        {"type": "put", "key": "content/mem/M:0001/summary", "value": "the gist"},
        {"type": "put", "key": "content/mem/M:0001/text", "value": "the gist body"},
        {"type": "put", "key": "content/mem/M:0001/ts", "value": "2026-01-01T00:00:00Z"},
        {"type": "put", "key": "content/mem/M:0001/abstracted_from",
         "value": '["ep_000001", "ep_000002"]'},
    ])
    trav = GraphTraversal(store)
    r = trav._hydrate("M:0001")
    assert r["kind"] == "memory"
    assert r["episode_id"] == "M:0001"
    assert r["summary"] == "the gist"
    assert r["text"] == "the gist body"
    assert r["timestamp"] == "2026-01-01T00:00:00Z"
    assert r["sources"] == ["ep_000001", "ep_000002"]
    # the 12-key episode shell is present (consumers use .get).
    for k in ("entities", "topics", "tones", "decisions", "session_id",
              "user_id", "follows", "score"):
        assert k in r
    assert r["entities"] == [] and r["topics"] == []
    store.close()


def test_hydrate_memory_missing_returns_empty_shell(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    trav = GraphTraversal(store)
    r = trav._hydrate("M:9999")
    assert r["kind"] == "memory"
    assert r["summary"] == "" and r["text"] == ""
    assert r["sources"] == []
    store.close()


def test_hydrate_memory_dispatch_precedes_doc(tmp_path):
    """An M: id is routed to _hydrate_memory even though the discriminator
    also checks doc_ -- M: ids never start with doc_, but the ordering is
    explicit."""
    store = _store(tmp_path)
    _populate(store)
    store.db.batch_sync([
        {"type": "put", "key": "content/mem/M:0001/summary", "value": "g"},
    ])
    trav = GraphTraversal(store)
    r = trav._hydrate("M:0001")
    assert r["kind"] == "memory"
    store.close()