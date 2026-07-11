"""Tests for the Phase 3b retrieval-time forgetting boost hook (graph_traversal).

The hot-path hook: after ``retrieve`` returns its scored+limited results, the
edges that actually matched the query are strengthened via
``forgetting.apply_retrieval_boost`` (the ``on_retrieve`` persistence that makes
memories persist with use). Non-blocking: a sidecar write failure is logged and
swallowed -- retrieval never breaks on it.
"""

from __future__ import annotations

from src.config import config as _config
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.graph_traversal import GraphTraversal


def _store_trav(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    return store, GraphTraversal(store)


def _encode(store, eid, entities=None, topics=None, tones=None):
    store.encode_episode(Episode(
        id=eid, timestamp="t", summary=f"s {eid}", full_text=f"f {eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
    ))


def _ids(results):
    return {r["episode_id"] for r in results}


# ── matched edges get boosted ──
def test_retrieve_boosts_matched_entity_edge(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    results = trav.retrieve({"entities": ["Alice"]}, signal="important")
    assert _ids(results) == {"ep_001"}

    meta = store.get_edge_meta("ep_001", "has_entity", "E:Alice")
    assert meta["access_count"] == 1
    assert meta["reconsolidation_count"] == 1
    # important signal => boosted down (more persistent)
    assert meta["utility_decay_rate"] < 0.01
    assert len(meta["retrieval_timestamps"]) == 1
    store.close()


def test_retrieve_does_not_boost_unmatched_edge(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice", "Bob"])
    trav.retrieve({"entities": ["Alice"]})

    alice = store.get_edge_meta("ep_001", "has_entity", "E:Alice")
    bob = store.get_edge_meta("ep_001", "has_entity", "E:Bob")
    assert alice["access_count"] == 1          # matched -> boosted
    assert bob["access_count"] == 0            # not matched -> untouched
    assert bob["state"] == "current"           # still default
    store.close()


def test_retrieve_boosts_topic_and_tone_edges(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"], topics=["databases"], tones=["confident"])
    trav.retrieve({
        "entities": ["Alice"], "topics": ["databases"], "tones": ["confident"],
    })
    assert store.get_edge_meta("ep_001", "has_entity", "E:Alice")["access_count"] == 1
    assert store.get_edge_meta("ep_001", "has_topic", "T:databases")["access_count"] == 1
    assert store.get_edge_meta("ep_001", "has_tone", "A:confident")["access_count"] == 1
    store.close()


# ── signal threading ──
def test_signal_correction_counts_but_does_not_boost(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    trav.retrieve({"entities": ["Alice"]}, signal="correction")
    meta = store.get_edge_meta("ep_001", "has_entity", "E:Alice")
    # correction still records the retrieval (access_count + reconsolidation++)
    assert meta["access_count"] == 1
    assert meta["reconsolidation_count"] == 1
    # but no boost: decay unchanged at baseline
    assert meta["utility_decay_rate"] == 0.01
    store.close()


def test_signal_important_boosts_more_than_routine(tmp_path):
    store_a, trav_a = _store_trav(tmp_path / "a")
    _encode(store_a, "ep_001", entities=["Alice"])
    trav_a.retrieve({"entities": ["Alice"]}, signal="routine")
    routine = store_a.get_edge_meta("ep_001", "has_entity", "E:Alice")["utility_decay_rate"]
    store_a.close()

    store_b, trav_b = _store_trav(tmp_path / "b")
    _encode(store_b, "ep_001", entities=["Alice"])
    trav_b.retrieve({"entities": ["Alice"]}, signal="important")
    important = store_b.get_edge_meta("ep_001", "has_entity", "E:Alice")["utility_decay_rate"]
    store_b.close()
    # important => stronger boost => lower decay (more persistent)
    assert important < routine


# ── non-blocking: write failure never breaks retrieval ──
def test_write_failure_is_non_fatal(tmp_path, monkeypatch):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated sidecar write failure")
    # the hook lazy-imports this at call time, so patching the module attr lands.
    monkeypatch.setattr("src.memory.edge_meta.batch_update_edge_meta", boom)

    # retrieve must still return results despite the write raising
    results = trav.retrieve({"entities": ["Alice"]})
    assert _ids(results) == {"ep_001"}
    # and no sidecar landed (the write failed)
    assert store.get_edge_meta("ep_001", "has_entity", "E:Alice")["access_count"] == 0
    store.close()


# ── gate: forgetting_enabled=False skips the boost entirely ──
def test_forgetting_disabled_skips_boost(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    saved = _config.forgetting_enabled
    _config.forgetting_enabled = False
    try:
        trav.retrieve({"entities": ["Alice"]}, signal="important")
    finally:
        _config.forgetting_enabled = saved
    # no sidecar written
    assert store.get_edge_meta("ep_001", "has_entity", "E:Alice")["access_count"] == 0
    store.close()


# ── no-axis query: nothing matched an edge to boost ──
def test_no_axis_query_does_not_boost(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    # no entities/topics/tones -> candidate set is all episodes, but no edge
    # matched a query axis, so nothing to boost
    trav.retrieve({})
    assert store.get_edge_meta("ep_001", "has_entity", "E:Alice")["access_count"] == 0
    store.close()


# ── repeated retrieval accumulates (the persistence mechanism) ──
def test_repeated_retrieval_accumulates_access_count(tmp_path):
    store, trav = _store_trav(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    trav.retrieve({"entities": ["Alice"]})
    trav.retrieve({"entities": ["Alice"]})
    trav.retrieve({"entities": ["Alice"]})
    meta = store.get_edge_meta("ep_001", "has_entity", "E:Alice")
    assert meta["access_count"] == 3
    assert meta["reconsolidation_count"] == 3
    # diminishing returns: 3 boosts reduce decay below a single boost would
    assert meta["utility_decay_rate"] < 0.0096  # < the 1st-boost value
    store.close()