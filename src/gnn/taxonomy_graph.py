"""Build the live class DAG for the ontology head's taxonomy encoder.

The ontology head is a TWO-encoder pair classifier: entity embeddings come from
the episode GAT backbone, class embeddings from a small GAT over the live class
DAG (``TaxonomyEncoder`` in ``model.py``). This module builds that class DAG from
the store so the encoder can produce ``[C, hidden_dim]`` class embeddings that
reflect each class's taxonomy position.

Why a DAG and not a fixed table: the ontology is a SEED that grows (vision sec
5.3). New classes arrive at runtime via discovery -> buffered -> Bonsai-
promoted, materialized as real ``subClassOf`` triples. The taxonomy encoder reads
the LIVE DAG, so a newly promoted class gets an embedding next pass via message
passing from its parent -- no fixed vocabulary, no head retrain to score a new
class. This is the open-vocabulary mechanism the user required.

Enumeration: there is no global "all vertices" / "all subClassOf" scan helper in
the graph layer (``GraphQuery`` is always scoped to a ``.vertex(id).out/in(pred)``
chain). So we anchor the BFS on the seed class set (``SEED_ONTOLOGY["classes"]``,
the in-memory source of truth before graph insertion) and follow LIVE
``subClassOf`` out/in edges to closure. This reads live edges, so Bonsai-promoted
classes (not in the seed) appear via their ``subClassOf`` edges to seed classes.
A class with NO seed-anchored path won't appear -- acceptable at cold start (the
ontology == the seed until Bonsai promotes, and promotion writes a ``subClassOf``
edge to a seed class by design). A future global ``subClassOf`` SPO-range scan is
the vision-complete path once promotions exist.

Edges are emitted BIDIRECTIONAL (child->parent + parent->child), mirroring the
episode loader (whose BFS emits both orientations so GAT message passing sees a
bidirectional graph). The taxonomy DAG has a single edge type, so the encoder
ignores predicate identity (``edge_dim=None``); the loader still builds
``edge_attr`` but it is unused.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

from .features import NODE_KIND_INDEX, _hash_embedding
from .graph_loader import data_from_subgraph

if TYPE_CHECKING:
    from ..memory.store import HippocampalStore


def _taxonomy_feature_for(nid: str) -> tuple[int, torch.Tensor]:
    """Class-node feature: deterministic-per-name hash + the ``unknown``-kind
    onehot. Class nodes are bare names with no persisted content (they are
    graph-structure vertices, not episode/entity content nodes), so their
    feature is NAME-derived, not store-derived. Distinct per name (open-vocab:
    a new class name hashes to a fresh init), then the taxonomy GAT propagates
    ``subClassOf`` structure into the embedding (vision sec 5.3).

    This is the documented dev fallback (``_hash_embedding``): a cold-start
    initial feature the GAT refines, NOT a trained class-embedding model. The
    real per-class semantic embedding is a future lever; the hash guarantees
    distinct init so the encoder isn't handed N identical rows.
    """
    k = NODE_KIND_INDEX["unknown"]
    v = _hash_embedding(nid)
    v[k] = 1.0  # stamp the unknown-kind onehot (hash zeroed the leading slots)
    return k, v.to(torch.float32)


def build_taxonomy_graph(store: "HippocampalStore") -> dict:
    """Enumerate the live class DAG (seed-anchored BFS over ``subClassOf``).

    Returns a ``data_from_subgraph``-shaped dict (``center``, ``nodes``,
    ``edges``) covering every class reachable from the seed set via live
    ``subClassOf`` edges. Node ids are bare class names (as stored); edges are
    bidirectional ``subClassOf`` (child->parent + parent->child).
    """
    from ..memory.ontology import SEED_ONTOLOGY

    graph = store.graph
    seed_classes = list(SEED_ONTOLOGY["classes"].keys())

    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], None] = {}
    visited: set[str] = set()

    # Seed nodes exist even if they have no subClassOf edges (a leaf class with
    # no parents still needs a row so it can be scored against).
    for c in seed_classes:
        nodes.setdefault(c, {"id": c, "type": "unknown", "depth": 0})

    queue: deque[tuple[str, int]] = deque((c, 0) for c in seed_classes)
    while queue:
        cls, depth = queue.popleft()
        if cls in visited:
            continue
        visited.add(cls)

        # out("subClassOf") -> parents (cls is a subclass of parent).
        r = graph.query().vertex(cls).out("subClassOf").execute_sync()
        try:
            parents = list(r.vertices)
        finally:
            r.close()
        for p in parents:
            edges[(cls, "subClassOf", p)] = None     # child -> parent
            edges[(p, "subClassOf", cls)] = None      # parent -> child (bidir)
            if p not in visited:
                nodes.setdefault(p, {"id": p, "type": "unknown", "depth": depth + 1})
                queue.append((p, depth + 1))

        # in_("subClassOf") -> children (discovered classes not in the seed).
        r = graph.query().vertex(cls).in_("subClassOf").execute_sync()
        try:
            children = list(r.vertices)
        finally:
            r.close()
        for ch in children:
            edges[(ch, "subClassOf", cls)] = None     # child -> parent
            edges[(cls, "subClassOf", ch)] = None      # parent -> child (bidir)
            if ch not in visited:
                nodes.setdefault(ch, {"id": ch, "type": "unknown", "depth": depth + 1})
                queue.append((ch, depth + 1))

    center = seed_classes[0] if seed_classes else (next(iter(nodes), "") or "")
    return {
        "center": center,
        "radius": 0,
        "nodes": list(nodes.values()),
        "edges": [{"subject": s, "predicate": p, "object": o}
                  for (s, p, o) in edges],
    }


def build_taxonomy_data(
    store: "HippocampalStore",
) -> tuple[Data, dict[str, int]]:
    """Build the taxonomy ``Data`` + a ``class name -> row`` map.

    Class nodes get ``_taxonomy_feature_for`` (name-hash + unknown-onehot):
    distinct per name, so the taxonomy encoder starts from N different rows and
    the GAT propagates ``subClassOf`` structure into each. Returns ``(Data,
    name_to_row)`` where ``name_to_row[name]`` is the row index in ``Data.node_id``
    (aligned to the encoder's ``[C, hidden_dim]`` output rows).
    """
    sub = build_taxonomy_graph(store)
    data = data_from_subgraph(sub, _taxonomy_feature_for)
    name_to_row = {nid: i for i, nid in enumerate(data.node_id)}
    return data, name_to_row


__all__ = ["build_taxonomy_graph", "build_taxonomy_data"]