"""Tests for R1 supersede blast-radius propagation.

Write side:``SemanticMemoryWriter._stale_propagation_ops`` marks derived nodes
(scenes citing the superseded episode, memories abstracting it) with
``stale_since`` + ``stale_of`` when ``config.stale_propagation_enabled`` is on;
skips nodes already referencing the replacement; writes NOTHING when off.

Read side:``GraphTraversal._filter_stale_derived`` rechecks ``stale_of`` at
query time (tip ∩ citations), shrinks/clears the keys on resolution, DROPS
still-stale scenes and KEEPS + annotates still-stale M-nodes.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from src.config import config as _config
from src.gnn.semantic_memory import SemanticMemoryWriter
from src.memory.episode import Episode
from src.memory.store import HippocampalStore, _b2s
from src.retrieval.graph_traversal import GraphTraversal


@contextmanager
def propagating(on):
    prev = _config.stale_propagation_enabled
    _config.stale_propagation_enabled = on
    try:
        yield
    finally:
        _config.stale_propagation_enabled = prev


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid, entities=None):
    store.encode_episode(Episode(
        id=eid, timestamp="t", summary=f"s {eid}", full_text=f"f {eid}",
        entities=entities or [],
    ))


def _scene(store, sid, source_eps, topic="db"):
    store.encode_scene(sid, body=f"## {topic}\nbody", topic=topic, heat=1.0,
                       updated_ts="t", user_id=None, source_eps=source_eps)


def _stale_of(store, prefix, node_id):
    raw = _b2s(store.db.get_sync(f"content/{prefix}/{node_id}/stale_of"))
    return set(json.loads(raw)) if raw else set()


def _stale_since(store, prefix, node_id):
    return _b2s(store.db.get_sync(f"content/{prefix}/{node_id}/stale_since"))


# ── write side ──


def test_supersede_marks_citing_scene_and_abstracting_memory(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    mid = w.create_abstract(["ep_000001"], "v1")
    with propagating(True):
        w.supersede_episode("ep_000002", "ep_000001", when="T1")
    assert _stale_of(store, "scene", "scene_0001") == set()
    _scene(store, "scene_0001", ["ep_000001"])
    # Re-run with a scene present: a new supersession marks BOTH kinds.
    _encode(store, "ep_000003")
    with propagating(True):
        w.supersede_episode("ep_000003", "ep_000001", when="T2")
    assert _stale_of(store, "scene", "scene_0001") == {"ep_000001"}
    assert _stale_since(store, "scene", "scene_0001") == "T2"
    # The memory abstracts only ep_000001, so BOTH supersessions mark it;
    # stale_of is a SET (union), stale_since is "latest mark wins".
    assert _stale_of(store, "mem", mid) == {"ep_000001"}
    assert _stale_since(store, "mem", mid) == "T2"
    store.close()


def test_supersede_skips_derived_already_citing_replacement(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    _scene(store, "scene_0001", ["ep_000001", "ep_000002"])
    with propagating(True):
        w.supersede_episode("ep_000002", "ep_000001")
    assert _stale_since(store, "scene", "scene_0001") == ""
    assert _stale_of(store, "scene", "scene_0001") == set()
    store.close()


def test_supersede_flag_off_writes_no_derived_keys(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    mid = w.create_abstract(["ep_000001"], "v1")
    _scene(store, "scene_0001", ["ep_000001"])
    with propagating(False):
        w.supersede_episode("ep_000002", "ep_000001")
    assert _stale_since(store, "scene", "scene_0001") == ""
    assert _stale_since(store, "mem", mid) == ""
    # The MVCC chain itself still lands (unaffected by the flag).
    assert _b2s(store.db.get_sync("content/ep/ep_000001/state")) == "superseded"
    store.close()


# ── read side: tip walk + resolution ──


def test_supersession_tip_walks_chain_and_breaks_at_end(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    trav = GraphTraversal(store)
    with propagating(True):
        sw = SemanticMemoryWriter(store)
        sw.supersede_episode("ep_000002", "ep_000001")
        sw.supersede_episode("ep_000003", "ep_000002")
    assert trav._supersession_tip("ep_000001") == "ep_000003"
    assert trav._supersession_tip("ep_000003") == "ep_000003"  # no chain
    assert trav._supersession_tip("ep_999999") == "ep_999999"  # unknown node
    store.close()


def test_recheck_resolves_reauthored_scene_and_clears_keys(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    _scene(store, "scene_0001", ["ep_000001"])
    with propagating(True):
        w.supersede_episode("ep_000002", "ep_000001")
        assert _stale_of(store, "scene", "scene_0001") == {"ep_000001"}
        # Scene re-authored against the tip -> the citation union now contains
        # the tip; the recheck resolves and clears BOTH keys.
        _scene(store, "scene_0001", ["ep_000002"])
        trav = GraphTraversal(store)
        candidates = {"scene_0001", "ep_000002"}
        keep, annotations = trav._filter_stale_derived(candidates)
        # Scene was still marked at recheck ENTRY (stale_of set) but resolves.
        assert keep == {"scene_0001", "ep_000002"}
        assert annotations == {}
    with propagating(False):
        assert _stale_since(store, "scene", "scene_0001") == ""
        assert _stale_of(store, "scene", "scene_0001") == set()
    store.close()


def test_recheck_partially_resolved_shrinks_stale_of(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    _encode(store, "ep_000003")
    _encode(store, "ep_000004")
    w = SemanticMemoryWriter(store)
    _scene(store, "scene_0001", ["ep_000001", "ep_000003"])
    with propagating(True):
        w.supersede_episode("ep_000002", "ep_000001")
        w.supersede_episode("ep_000004", "ep_000003")
        assert _stale_of(store, "scene", "scene_0001") == {"ep_000001",
                                                          "ep_000003"}
        # Re-author citing ONLY ep_000002: tip(ep_000001) resolves; tip of the
        # ep_000003 chain (ep_000004) does not.
        _scene(store, "scene_0001", ["ep_000001", "ep_000003", "ep_000002"])
        trav = GraphTraversal(store)
        keep, _ = trav._filter_stale_derived({"scene_0001"})
        assert keep == set()  # ep_000003 chain unresolved -> still dropped
    assert _stale_of(store, "scene", "scene_0001") == {"ep_000003"}
    # stale_since survives the shrink (a mark is still pending).
    assert _stale_since(store, "scene", "scene_0001")
    store.close()


# ── read side: retrieve integration ──


def test_retrieve_drops_still_stale_scene_keeps_unmarked(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    _scene(store, "scene_0001", ["ep_000001"])
    _scene(store, "scene_0002", ["ep_000002"])
    with propagating(True):
        w.supersede_episode("ep_000002", "ep_000001")
    plan = {"limit": 10}
    with propagating(False):
        ids = [r["episode_id"] for r in GraphTraversal(store).retrieve(plan)]
        assert "scene_0001" in ids and "scene_0002" in ids
        # Flag-off retrieve ALSO returns the unmarked ids (no recheck ran).
        with propagating(True):
            ids = [r["episode_id"]
                   for r in GraphTraversal(store).retrieve(plan)]
        assert "scene_0001" not in ids  # still-stale -> dropped
        assert "scene_0002" in ids
        assert "ep_000002" in ids  # the healthy episode survives throughout
    store.close()


def test_retrieve_keeps_and_annotates_still_stale_memory(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    # when="t": match the fixture timestamps so the scorer's recency parse
    # (real _utc_now is tz-aware, "t" parses to datetime.min) never mixes them.
    mid = w.create_abstract(["ep_000001"], "stale gist", when="t")
    _scene(store, "scene_0001", ["ep_000001"])
    with propagating(True):
        w.supersede_episode("ep_000002", "ep_000001", when="T9")
        trav = GraphTraversal(store)
        results = trav.retrieve({"limit": 10})
        mid_result = next(r for r in results if r["episode_id"] == mid)
        assert mid_result["kind"] == "memory"
        assert mid_result["stale_since"] == "T9"
        assert mid_result["stale_of"] == ["ep_000001"]
    # Flag off: the marks are still ON the node (written above), but a flag-off
    # retrieve never rechecks -> results hydrate WITHOUT the annotation keys.
    with propagating(False):
        results = GraphTraversal(store).retrieve({"limit": 10})
        mid_result = next(r for r in results if r["episode_id"] == mid)
        assert "stale_since" not in mid_result
        assert "stale_of" not in mid_result
    store.close()


def test_filter_stale_derived_ignores_plain_episodes(tmp_path):
    """Episode/document candidates pay at most the stale_since point lookup and
    are never filtered by this pass (only scene_*/M:* carry marks)."""
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    trav = GraphTraversal(store)
    keep, annotations = trav._filter_stale_derived({"ep_000001", "doc_x"})
    assert keep == {"ep_000001", "doc_x"}
    assert annotations == {}
    store.close()


@pytest.mark.parametrize("flag_on", [True, False])
def test_supersede_mvcc_chain_invariant_across_flag(tmp_path, flag_on):
    """The MVCC supersession chain (edges + state) is written regardless of the
    flag; only the derived-node marks are gated."""
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    with propagating(flag_on):
        w.supersede_episode("ep_000002", "ep_000001")
    assert _b2s(store.db.get_sync("content/ep/ep_000001/state")) == "superseded"
    assert _b2s(store.db.get_sync("content/ep/ep_000001/validity_end"))
    r = store.graph.query().vertex("ep_000001").out("superseded_by").execute_sync()
    try:
        assert list(r.vertices) == ["ep_000002"]
    finally:
        r.close()
    store.close()


# ── vector-leg parity (retriever._filter_vector_hits_stale) ──


def test_vector_leg_recheck_drops_stale_scene_and_annotates_memory(tmp_path):
    """The semantic fallback / hybrid / embed-search vector sites hydrate
    derived ids WITHOUT GraphTraversal.retrieve -- the retriever's
    ``_filter_vector_hits_stale`` helper gives them the same verdict the graph
    leg gets (drop stale scene, keep + annotate stale M-node)."""
    from src.retrieval.retriever import HippocampalRetriever

    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    mid = w.create_abstract(["ep_000001"], "stale gist", when="t")
    _scene(store, "scene_0001", ["ep_000001"])
    with propagating(True):
        w.supersede_episode("ep_000002", "ep_000001", when="T9")
        trav = GraphTraversal(store)
        retr = HippocampalRetriever(store)
        retr.traversal = trav
        hits = [("scene_0001", 0.9), (mid, 0.8), ("ep_000002", 0.7)]
        # Flag ON: stale scene dropped, M-node kept + annotated.
        kept, annos = retr._filter_vector_hits_stale(hits)
        assert [e for e, _ in kept] == [mid, "ep_000002"]
        assert set(annos) == {mid}
        d = {"episode_id": mid}
        retr._stamp_stale_annotation(d, annos)
        assert d["stale_since"] == "T9"
        assert d["stale_of"] == ["ep_000001"]
    # Flag OFF: hits pass through unchanged, no annotations (byte-identical).
    with propagating(False):
        kept, annos = retr._filter_vector_hits_stale(hits)
        assert kept == hits
        assert annos == {}
    store.close()