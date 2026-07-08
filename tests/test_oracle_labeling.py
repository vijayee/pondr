"""Offline tests for the oracle labeling infrastructure (Phase G).

No live Bonsai calls — these validate the subgraph extraction BFS (node types,
hop depth, edge orientation, radius cutoff) and the prompt rendering against a
small hand-built graph on tmp_path.
"""

from __future__ import annotations

import json

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.training.oracle_labeling import (
    OracleLabelingPipeline,
    sample_episode_centers,
)


def _ep(eid, entities=None, topics=None, tones=None, follows=None, user="alice"):
    return Episode(
        id=eid, timestamp="2026-07-03T10:00:00", summary=f"summary {eid}",
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
        follows=follows, user_id=user,
    )


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _pipe(tmp_path):
    store = _store(tmp_path)
    return store, OracleLabelingPipeline(store)


# ── extract_subgraph ──


def test_subgraph_includes_center_and_direct_neighbors(tmp_path):
    store, pipe = _pipe(tmp_path)
    store.encode_episode(_ep("ep_000001", entities=["Alice"], topics=["db"], tones=["curious"]))
    sub = pipe.extract_subgraph("ep_000001", radius=1)

    node_ids = {n["id"] for n in sub["nodes"]}
    assert "ep_000001" in node_ids
    assert "E:Alice" in node_ids
    assert "T:db" in node_ids
    assert "A:curious" in node_ids
    # Center has depth 0; direct neighbors depth 1.
    by_id = {n["id"]: n for n in sub["nodes"]}
    assert by_id["ep_000001"]["depth"] == 0
    assert by_id["E:Alice"]["depth"] == 1
    assert by_id["E:Alice"]["type"] == "entity"
    assert by_id["T:db"]["type"] == "topic"
    assert by_id["A:curious"]["type"] == "tone"
    store.close()


def test_subgraph_edge_orientation_matches_stored_triples(tmp_path):
    store, pipe = _pipe(tmp_path)
    store.encode_episode(_ep("ep_000001", entities=["Alice"]))
    sub = pipe.extract_subgraph("ep_000001", radius=1)

    # has_entity is stored as (ep, has_entity, E:Alice); in_episode as
    # (E:Alice, in_episode, ep). Both should appear with correct orientation.
    edge_set = {(e["subject"], e["predicate"], e["object"]) for e in sub["edges"]}
    assert ("ep_000001", "has_entity", "E:Alice") in edge_set
    assert ("E:Alice", "in_episode", "ep_000001") in edge_set
    store.close()


def test_subgraph_radius_cuts_two_hop_neighbors(tmp_path):
    """radius=1 excludes the second episode reached via a shared entity."""
    store, pipe = _pipe(tmp_path)
    store.encode_episode(_ep("ep_000001", entities=["Alice"]))
    store.encode_episode(_ep("ep_000002", entities=["Alice"]))  # shares Alice
    sub = pipe.extract_subgraph("ep_000001", radius=1)

    node_ids = {n["id"] for n in sub["nodes"]}
    assert "ep_000001" in node_ids
    assert "E:Alice" in node_ids
    # ep_000002 is two hops away (ep_000001 → Alice → ep_000002); radius=1 cuts it.
    assert "ep_000002" not in node_ids
    store.close()


def test_subgraph_radius_two_reaches_shared_entity_neighbor(tmp_path):
    store, pipe = _pipe(tmp_path)
    store.encode_episode(_ep("ep_000001", entities=["Alice"]))
    store.encode_episode(_ep("ep_000002", entities=["Alice"]))
    sub = pipe.extract_subgraph("ep_000001", radius=2)

    node_ids = {n["id"] for n in sub["nodes"]}
    assert "ep_000002" in node_ids  # now reachable within 2 hops
    by_id = {n["id"]: n for n in sub["nodes"]}
    assert by_id["ep_000002"]["depth"] == 2
    store.close()


def test_subgraph_follows_chain_traversed(tmp_path):
    store, pipe = _pipe(tmp_path)
    store.encode_episode(_ep("ep_000001"))
    store.encode_episode(_ep("ep_000002", follows="ep_000001"))
    sub = pipe.extract_subgraph("ep_000001", radius=2)

    node_ids = {n["id"] for n in sub["nodes"]}
    assert "ep_000002" in node_ids
    edge_set = {(e["subject"], e["predicate"], e["object"]) for e in sub["edges"]}
    # follows is stored as (later, follows, earlier) → ep_000002 follows ep_000001.
    assert ("ep_000002", "follows", "ep_000001") in edge_set
    store.close()


def test_subgraph_edges_deduped(tmp_path):
    """A shared entity creates two paths to the same edge — no duplicates."""
    store, pipe = _pipe(tmp_path)
    store.encode_episode(_ep("ep_000001", entities=["Alice"]))
    store.encode_episode(_ep("ep_000002", entities=["Alice"]))
    sub = pipe.extract_subgraph("ep_000001", radius=2)
    # Each (s, p, o) edge appears at most once.
    triples = [(e["subject"], e["predicate"], e["object"]) for e in sub["edges"]]
    assert len(triples) == len(set(triples))
    store.close()


def test_extract_subgraph_unknown_center_returns_self_only(tmp_path):
    store, pipe = _pipe(tmp_path)
    sub = pipe.extract_subgraph("ep_nope", radius=3)
    assert [n["id"] for n in sub["nodes"]] == ["ep_nope"]
    assert sub["edges"] == []
    store.close()


# ── sampling ──
# (The two ``test_labeling_prompt_*`` tests that exercised the dead
# ``ORACLE_GNN_LABELING_PROMPT`` / ``build_labeling_prompt`` were removed in
# Phase 3a Task 3 along with those symbols. The live labeling prompts are the
# ``gnn_*`` functions in ``src/training/prompts.py``, tested in
# ``tests/test_training_prompts.py``.)


def test_sample_episode_centers_lists_all(tmp_path):
    store, pipe = _pipe(tmp_path)
    store.encode_episode(_ep("ep_000001"))
    store.encode_episode(_ep("ep_000002"))
    centers = sample_episode_centers(store)
    assert centers == ["ep_000001", "ep_000002"]
    assert sample_episode_centers(store, n=1) == ["ep_000001"]
    store.close()


def test_sample_episode_centers_skips_non_episode_keys(tmp_path):
    """The content/ep/ scan must not pick up non-episode content keys."""
    store, pipe = _pipe(tmp_path)
    store.encode_episode(_ep("ep_000001"))
    # Write a stray content/system counter (not an episode).
    store.db.put_sync("content/system/episode_counter", "5")
    centers = sample_episode_centers(store)
    assert centers == ["ep_000001"]
    store.close()