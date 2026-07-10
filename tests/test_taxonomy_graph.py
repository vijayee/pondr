"""Tests for ``src/gnn/taxonomy_graph.py`` (Phase 3a ontology-head rework).

The taxonomy encoder GATs over the LIVE class DAG (seed-anchored BFS over
``subClassOf``) to produce open-vocabulary class embeddings. These tests pin the
DAG enumeration: a freshly-seeded store's class DAG covers every seed class as a
node (leaves with no parents still get a row so they can be scored against) and
the ``subClassOf`` edges the store materialized are present bidirectionally.
``build_taxonomy_data`` returns a ``Data`` whose rows align to a ``name_to_row``
map covering every enumerated class.
"""

from __future__ import annotations

import pytest
import torch

from src.gnn.features import NODE_KIND_INDEX, _hash_embedding, infer_kind, NODE_KINDS
from src.gnn.graph_loader import data_from_subgraph
from src.gnn.taxonomy_graph import build_taxonomy_data, build_taxonomy_graph
from src.memory.store import HippocampalStore


def _stub_feature_for(nid):
    """Same stub as test_gnn_model: type-onehot at the kind slot + hash feature."""
    k = NODE_KIND_INDEX[infer_kind(nid)]
    v = _hash_embedding(nid)
    v[k] = 1.0
    v[len(NODE_KINDS)] = -1.0
    return k, v.to(torch.float32)


@pytest.fixture
def seeded_store(tmp_path) -> HippocampalStore:
    """A store with the seed ontology materialized (``_seed_ontology`` runs at
    init, writing the ~1448 ``child subClassOf parent`` triples). No episodes --
    the class DAG is independent of episode content."""
    store = HippocampalStore(str(tmp_path / "db"))
    yield store
    store.close()


def test_build_taxonomy_graph_covers_seed_classes(seeded_store):
    sub = build_taxonomy_graph(seeded_store)
    # The dict is in the data_from_subgraph shape.
    assert set(sub) == {"center", "radius", "nodes", "edges"}
    node_ids = {n["id"] for n in sub["nodes"]}
    # A freshly-seeded store has the full 377-class seed ontology as nodes.
    from src.memory.ontology import SEED_ONTOLOGY
    seed_classes = set(SEED_ONTOLOGY["classes"].keys())
    assert seed_classes <= node_ids            # every seed class is a node
    assert len(node_ids) == len(seed_classes)   # no non-seed classes at cold start
    # Leaves with no parents (e.g. Episode) still appear as nodes so they can be
    # scored against -- they are NOT dropped for having no subClassOf edges.
    assert "Episode" in node_ids
    assert "Person" in node_ids
    # Edges are bidirectional subClassOf (child->parent + parent->child).
    assert all(e["predicate"] == "subClassOf" for e in sub["edges"])
    edge_pairs = {(e["subject"], e["object"]) for e in sub["edges"]}
    # Person subClassOf Entity (seed: Entity subclasses include Person). The
    # store writes ``Person subClassOf Entity`` -> both orientations present.
    assert ("Person", "Entity") in edge_pairs
    assert ("Entity", "Person") in edge_pairs


def test_build_taxonomy_data_shape_and_name_to_row(seeded_store):
    data, name_to_row = build_taxonomy_data(seeded_store)
    # Every enumerated class has a row, and name_to_row aligns to it.
    assert data.x.shape[0] == len(name_to_row)
    assert set(name_to_row.keys()) == set(data.node_id)
    # Rows are contiguous indices into the encoder's [C, hidden] output.
    assert sorted(name_to_row.values()) == list(range(len(name_to_row)))
    # Class nodes are bare names -> "unknown" kind (routed through the taxonomy
    # encoder's InputProjection unknown slot).
    assert all(k == NODE_KIND_INDEX["unknown"]
               for k in data.node_kind.tolist())
    # There are real subClassOf edges for message passing to chew on (not a
    # degenerate edge-less graph).
    assert data.edge_index.shape[1] > 0


def test_build_taxonomy_data_features_are_distinct_per_class_name(seeded_store):
    """The class-node feature is name-hash + unknown-onehot, so two different
    classes get DIFFERENT initial feature rows (open-vocabulary: a new class
    name hashes to a fresh init). This guards against a regression to a
    store-feature path that yields identical onehot-only rows for every class
    (which would make the taxonomy encoder rely on structure alone)."""
    data, name_to_row = build_taxonomy_data(seeded_store)
    assert "Person" in name_to_row and "Entity" in name_to_row
    assert not torch.allclose(data.x[name_to_row["Person"]],
                              data.x[name_to_row["Entity"]])


def test_build_taxonomy_data_round_trips_through_data_from_subgraph(seeded_store):
    """The dict build_taxonomy_graph emits is consumable by data_from_subgraph
    (the same loader path the episode subgraphs use)."""
    sub = build_taxonomy_graph(seeded_store)
    data = data_from_subgraph(sub, _stub_feature_for)
    assert data.x.shape[0] == len(sub["nodes"])
    # data_from_subgraph onehots predicates; subClassOf falls in the hash-
    # randomized tail (unstable), but the taxonomy encoder ignores edge_attr
    # (edge_dim=None) so this is moot -- just confirm edge_attr exists.
    assert hasattr(data, "edge_attr")