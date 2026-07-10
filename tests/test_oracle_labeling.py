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


# ── fanout cap (giant-subgraph bound) ──


def test_extract_subgraph_fanout_cap_bounds_hub(tmp_path):
    """A high-degree entity hub shared by many episodes floods a radius-2
    subgraph to ALL of them when uncapped. A fanout_cap bounds that flood: the
    hub's aggregated neighbor list is truncated, so only a few sibling episodes
    survive instead of the whole corpus. This is the giant-subgraph lever.
    """
    store, pipe = _pipe(tmp_path)
    n = 20
    for i in range(1, n + 1):
        # Distinct user per episode -> distinct session, so the ONLY shared hub
        # is the entity "Hub"; the flood is isolated to it.
        store.encode_episode(_ep(f"ep_{i:06d}", entities=["Hub"], user=f"u{i}"))

    # Uncapped radius-2: the shared entity hub reaches all n episodes.
    uncapped = pipe.extract_subgraph("ep_000001", radius=2)
    uncapped_eps = {x["id"] for x in uncapped["nodes"] if x["id"].startswith("ep_")}
    assert len(uncapped_eps) == n
    assert uncapped["fanout_cap"] is None  # uncapped records None

    # Capped: the hub's fanout truncates, so far fewer siblings survive.
    capped = pipe.extract_subgraph("ep_000001", radius=2, fanout_cap=4)
    assert capped["fanout_cap"] == 4
    capped_eps = {x["id"] for x in capped["nodes"] if x["id"].startswith("ep_")}
    assert len(capped_eps) < n            # the flood is bounded
    assert "ep_000001" in capped_eps      # the center always survives
    store.close()


def test_extract_subgraph_fanout_cap_none_reproduces_uncapped(tmp_path):
    """fanout_cap=None (the default) is byte-for-byte the uncapped path: the
    same node and edge sets as calling without the kwarg. No existing caller
    changes behavior."""
    store, pipe = _pipe(tmp_path)
    for i in range(1, 21):
        store.encode_episode(_ep(f"ep_{i:06d}", entities=["Hub"], user=f"u{i}"))

    default = pipe.extract_subgraph("ep_000001", radius=2)
    explicit_none = pipe.extract_subgraph("ep_000001", radius=2, fanout_cap=None)

    nodes_default = {x["id"] for x in default["nodes"]}
    nodes_none = {x["id"] for x in explicit_none["nodes"]}
    assert nodes_default == nodes_none
    edges_default = {(e["subject"], e["predicate"], e["object"])
                     for e in default["edges"]}
    edges_none = {(e["subject"], e["predicate"], e["object"])
                  for e in explicit_none["edges"]}
    assert edges_default == edges_none
    store.close()


def test_extract_subgraph_fanout_cap_is_deterministic(tmp_path):
    """The sorted-first-K cap is seedless + platform-independent (Python string
    sort), so two extractions of the same center+radius+cap yield the SAME node
    set. This is load-bearing: the label generator and the trainer must walk the
    same bounded subgraph or the per-node anomaly labels misalign.
    """
    store, pipe = _pipe(tmp_path)
    for i in range(1, 21):
        store.encode_episode(_ep(f"ep_{i:06d}", entities=["Hub"], user=f"u{i}"))

    a = pipe.extract_subgraph("ep_000001", radius=2, fanout_cap=4)
    b = pipe.extract_subgraph("ep_000001", radius=2, fanout_cap=4)
    nodes_a = {x["id"] for x in a["nodes"]}
    nodes_b = {x["id"] for x in b["nodes"]}
    assert nodes_a == nodes_b
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