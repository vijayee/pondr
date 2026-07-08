"""Tests for the WaveDB→PyG graph loader (``src/gnn/graph_loader.py``)."""

from __future__ import annotations

import torch

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.gnn.graph_loader import (
    WaveDBGraphLoader, data_from_subgraph, KNOWN_PREDICATES, PREDICATE_VOCAB,
    _predicate_index,
)
from src.gnn.features import FEATURE_DIM, NODE_KIND_INDEX, infer_kind


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _stub_feature_for(nid):
    """Parameter-free feature: onehot only (no store reads needed for loader tests)."""
    import torch
    k = NODE_KIND_INDEX[infer_kind(nid)]
    v = torch.zeros(FEATURE_DIM)
    v[k] = 1.0
    return k, v.to(torch.float32)


def test_predicate_index_maps_known_and_hashes_unknown():
    for i, p in enumerate(KNOWN_PREDICATES):
        assert _predicate_index(p) == i
    # Unknown predicates hash into the tail without colliding with known slots.
    assert _predicate_index("related_to") >= len(KNOWN_PREDICATES)
    assert _predicate_index("related_to") < PREDICATE_VOCAB


def test_data_from_subgraph_shapes_and_orientation():
    sub = {
        "center": "ep_000001", "radius": 3,
        "nodes": [
            {"id": "ep_000001", "type": "episode", "depth": 0},
            {"id": "E:Alice", "type": "entity", "depth": 1},
            {"id": "T:db", "type": "topic", "depth": 1},
        ],
        "edges": [
            {"subject": "ep_000001", "predicate": "has_entity", "object": "E:Alice"},
            {"subject": "E:Alice", "predicate": "in_episode", "object": "ep_000001"},
            {"subject": "ep_000001", "predicate": "has_topic", "object": "T:db"},
        ],
    }
    data = data_from_subgraph(sub, _stub_feature_for)
    assert data.x.shape == (3, FEATURE_DIM)
    assert data.edge_index.shape == (2, 3)
    assert data.edge_attr.shape == (3, PREDICATE_VOCAB)
    assert data.node_kind.tolist() == [
        NODE_KIND_INDEX["episode"], NODE_KIND_INDEX["entity"], NODE_KIND_INDEX["topic"],
    ]
    assert data.node_depth.tolist() == [0, 1, 1]
    assert int(data.center_idx) == 0
    assert data.node_id == ["ep_000001", "E:Alice", "T:db"]
    # edge_attr is a onehot per row.
    assert int(data.edge_attr.sum()) == 3


def test_data_from_subgraph_empty_center_yields_one_node():
    sub = {"center": "ep_nope", "radius": 3, "nodes": [], "edges": []}
    data = data_from_subgraph(sub, _stub_feature_for)
    assert data.x.shape[0] == 1
    assert int(data.center_idx) == 0
    assert data.edge_index.shape == (2, 0)


def test_loader_round_trips_a_real_subgraph(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(Episode(id="ep_000001", timestamp="t", summary="s",
                                  full_text="f", entities=["Alice"], topics=["db"]))
    loader = WaveDBGraphLoader(store, radius=1)
    data = loader.load("ep_000001")
    assert "ep_000001" in data.node_id
    assert "E:Alice" in data.node_id
    assert "T:db" in data.node_id
    # has_entity edge is oriented ep → E:Alice.
    idx = {nid: i for i, nid in enumerate(data.node_id)}
    s, o = idx["ep_000001"], idx["E:Alice"]
    assert ((data.edge_index[0] == s) & (data.edge_index[1] == o)).any()
    store.close()


def test_loader_episode_centers_lists_episodes(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(Episode(id="ep_000001", timestamp="t", summary="s", full_text="f"))
    store.encode_episode(Episode(id="ep_000002", timestamp="t", summary="s", full_text="f"))
    loader = WaveDBGraphLoader(store)
    assert loader.episode_centers() == ["ep_000001", "ep_000002"]
    assert loader.episode_centers(limit=1) == ["ep_000001"]
    store.close()