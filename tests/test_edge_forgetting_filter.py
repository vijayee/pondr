"""Tests for the Phase 3b edge-level forgetting filter (graph_traversal.py).

The edge granularity (vs the episode-level filter in test_episode_forgetting.py):
a single deprecated ``(ep, has_entity, E:Alice)`` association excludes the
episode from an Alice query but NOT a Bob query -- the episode stays live for
its other associations. The graph triple is NOT deleted; only the per-edge
sidecar state hides it from retrieval (deprecate, don't delete).
"""

from __future__ import annotations

from src.config import config as _config
from src.memory.episode import Episode
from src.memory.store import HippocampalStore, _b2s
from src.retrieval.graph_traversal import GraphTraversal


def _store_trav(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    return store, GraphTraversal(store)


def _encode(store, eid, entities=None, topics=None):
    store.encode_episode(Episode(
        id=eid, timestamp="t", summary=f"s {eid}", full_text=f"f {eid}",
        entities=entities or [], topics=topics or [],
    ))


def _ids(results):
    return {r["episode_id"] for r in results}


# ── R3 load-bearing: one stale edge hides the episode for ONE axis only ──
def test_deprecated_alice_edge_excludes_for_alice_not_bob(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice", "Bob"])

    # deprecate ONLY the Alice association
    store.set_edge_state("ep_001", "has_entity", "E:Alice", "deprecated")

    # Alice query: the episode is hidden (its Alice edge is stale)
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == set()
    # Bob query: the episode is still live (its Bob edge is current)
    assert _ids(trav.retrieve({"entities": ["Bob"]})) == {"ep_001"}
    store.close()


def test_deprecated_edge_does_not_delete_graph_triple(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    store.set_edge_state("ep_001", "has_entity", "E:Alice", "deprecated")

    # retrieval hides it
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == set()
    # but the raw graph triple survives (deprecate, not delete): the in_episode
    # edge from E:Alice still resolves to ep_001 at the graph layer
    q = store.graph.query().vertex("E:Alice").out("in_episode")
    result = q.execute_sync()
    try:
        raw_eps = list(result.vertices)
    finally:
        result.close()
    assert "ep_001" in raw_eps
    # and the content is intact
    assert _b2s(store.db.get_sync("content/ep/ep_001/summary")) == "s ep_001"
    store.close()


# ── every non-current state excludes ──
def test_archived_edge_excludes(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    store.set_edge_state("ep_001", "has_entity", "E:Alice", "archived")
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == set()
    store.close()


def test_superseded_edge_excludes(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    store.set_edge_state("ep_001", "has_entity", "E:Alice", "superseded")
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == set()
    store.close()


def test_current_edge_state_is_a_noop(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    # explicitly writing state=current keeps it retrievable
    store.set_edge_state("ep_001", "has_entity", "E:Alice", "current")
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == {"ep_001"}
    store.close()


# ── topic axis filters independently of the entity axis ──
def test_topic_edge_filter_independent_of_entity(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"], topics=["databases"])
    # deprecate only the topic association
    store.set_edge_state("ep_001", "has_topic", "T:databases", "deprecated")
    # topic query hides it
    assert _ids(trav.retrieve({"topics": ["databases"]})) == set()
    # entity query still finds it (the Alice edge is current)
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == {"ep_001"}
    store.close()


# ── the gate: forgetting_enabled=False disables the filter entirely ──
def test_forgetting_disabled_keeps_deprecated_edge(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    store.set_edge_state("ep_001", "has_entity", "E:Alice", "deprecated")
    saved = _config.forgetting_enabled
    _config.forgetting_enabled = False
    try:
        # filter bypassed: deprecated edge is still returned
        assert _ids(trav.retrieve({"entities": ["Alice"]})) == {"ep_001"}
    finally:
        _config.forgetting_enabled = saved
    # and re-enabled hides it again
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == set()
    store.close()


# ── two granularities compose: episode-level deprecate beats every axis ──
def test_episode_level_deprecate_excludes_all_axes(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice", "Bob"])
    # episode-level: deprecate the whole episode
    store.set_episode_state("ep_001", "deprecated")
    # both axes hide it (the episode-level filter in default_episode_ids fires
    # before per-axis edge filtering even matters)
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == set()
    assert _ids(trav.retrieve({"entities": ["Bob"]})) == set()
    # historical opt-in still returns it (episode not deleted)
    assert store.default_episode_ids(include_inactive=True) == ["ep_001"]
    store.close()


# ── cold start: no sidecars -> everything current (filter is a no-op) ──
def test_no_sidecars_means_no_filtering(tmp_path):
    store, trav = _store_trav(tmp_path)
    for i in range(1, 4):
        _encode(store, f"ep_{i:03d}", entities=["Alice"])
    # nothing deprecated -> all three returned (no per-edge sidecar exists)
    assert _ids(trav.retrieve({"entities": ["Alice"]})) == {
        "ep_001", "ep_002", "ep_003",
    }
    store.close()