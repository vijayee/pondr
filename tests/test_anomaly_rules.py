"""Tests for the structural anomaly rule detectors (``src/gnn/anomaly_rules.py``).

Each detector is a pure function of an enriched subgraph dict, so these tests
build synthetic dicts (no store) and assert each of the 9 types fires exactly
where planted and not elsewhere. The inject→rule-detect round-trip is covered
in ``test_anomaly_injector.py`` (the injector plants these same structures and
the rules recover them).
"""

from __future__ import annotations

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.gnn.semantic_memory import SemanticMemoryWriter
from src.training.oracle_labeling import OracleLabelingPipeline
from src.gnn.anomaly_rules import (
    ANOMALY_TYPES,
    ANOMALY_TYPE_INDEX,
    IDENTITY_DRIFT_FLAG,
    detect_anomalies,
    enrich_subgraph,
    flag_identity_drift,
    node_label_vectors,
)


# ── helpers ──

def _node(nid: str, **kw) -> dict:
    """A minimal node dict with a type inferred from the id prefix."""
    for prefix, typ in (
        ("ep_", "episode"), ("E:", "entity"), ("T:", "topic"),
        ("A:", "tone"), ("D:", "decision"), ("S:", "session"), ("U:", "user"),
        ("M:", "semantic_memory"),
    ):
        if nid.startswith(prefix):
            return {"id": nid, "type": typ, "depth": 0, **kw}
    return {"id": nid, "type": "unknown", "depth": 0, **kw}


def _sub(nodes: list[dict], edges: list[tuple], center: str) -> dict:
    """Build an enriched subgraph dict from (s, p, o) edge tuples."""
    return {
        "center": center,
        "nodes": nodes,
        "edges": [{"subject": s, "predicate": p, "object": o} for s, p, o in edges],
    }


def _types_for(sub: dict, nid: str) -> set[str]:
    """The set of anomaly types detected on ``nid``."""
    return {f["type"] for f in detect_anomalies(sub) if f["node"] == nid}


# ── taxonomy sanity ──

def test_anomaly_types_canonical_9():
    assert len(ANOMALY_TYPES) == 9
    # The 9 names from the spec §2 table, snake_cased, in this order.
    assert ANOMALY_TYPES == (
        "contradictory_state", "duplicate_episode", "duplicate_decision",
        "orphan_decision", "detached_episode", "broken_follows",
        "type_violation", "isolated_cluster", "stale_abstraction",
    )
    # Index is load-bearing (head output slots ↔ training labels).
    assert ANOMALY_TYPE_INDEX["stale_abstraction"] == 8
    # Identity drift is a flag, NOT a head label.
    assert IDENTITY_DRIFT_FLAG not in ANOMALY_TYPE_INDEX
    assert IDENTITY_DRIFT_FLAG == "identity_drift"


# ── 1. contradictory_state ──

def test_contradictory_state_two_distinct_live_states_on_entity():
    sub = _sub(
        [_node("E:Alice"), _node("ep_000001", summary="x")],
        [("E:Alice", "state", "alive"), ("E:Alice", "state", "dead"),
         ("ep_000001", "has_entity", "E:Alice")],
        center="ep_000001",
    )
    assert _types_for(sub, "E:Alice") == {"contradictory_state"}


def test_contradictory_state_single_state_not_flagged():
    sub = _sub(
        [_node("E:Alice"), _node("ep_000001", summary="x")],
        [("E:Alice", "state", "alive"), ("ep_000001", "has_entity", "E:Alice")],
        center="ep_000001",
    )
    assert _types_for(sub, "E:Alice") == set()


# ── 2. duplicate_episode ──

def test_duplicate_episode_near_identical_summary():
    sub = _sub(
        [_node("ep_000001", summary="Alice discussed the database schema"),
         _node("ep_000002", summary="Alice discussed the database schema"),
         _node("E:Alice")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Alice")],
        center="ep_000001",
    )
    assert _types_for(sub, "ep_000001") == {"duplicate_episode"}
    assert _types_for(sub, "ep_000002") == {"duplicate_episode"}


def test_duplicate_episode_different_summaries_not_flagged():
    # One connected component (shared topic links the two episodes) so
    # isolated_cluster doesn't fire incidentally — isolating duplicate_episode.
    sub = _sub(
        [_node("ep_000001", summary="Alice discussed the database schema"),
         _node("ep_000002", summary="Bob fixed the login bug"),
         _node("E:Alice"), _node("E:Bob"), _node("T:general")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Bob"),
         ("ep_000001", "has_topic", "T:general"),
         ("ep_000002", "has_topic", "T:general")],
        center="ep_000001",
    )
    assert _types_for(sub, "ep_000001") == set()
    assert _types_for(sub, "ep_000002") == set()


# ── 3. duplicate_decision ──

def test_duplicate_decision_near_identical_text():
    # Both decisions linked to the episode (so neither is an orphan) —
    # isolating duplicate_decision as the only anomaly on each.
    sub = _sub(
        [_node("D:0001", text="use postgres for the database"),
         _node("D:0002", text="use postgres for the database"),
         _node("ep_000001", summary="x")],
        [("ep_000001", "has_decision", "D:0001"),
         ("ep_000001", "has_decision", "D:0002")],
        center="ep_000001",
    )
    # Both decision nodes have identical text (Jaccard 1.0 ≥ 0.9).
    assert _types_for(sub, "D:0001") == {"duplicate_decision"}
    assert _types_for(sub, "D:0002") == {"duplicate_decision"}


def test_duplicate_decision_different_not_flagged():
    sub = _sub(
        [_node("D:0001", text="use postgres for the database"),
         _node("D:0002", text="rewrite the auth module from scratch"),
         _node("ep_000001", summary="x")],
        [("ep_000001", "has_decision", "D:0001"),
         ("ep_000001", "has_decision", "D:0002")],
        center="ep_000001",
    )
    assert _types_for(sub, "D:0001") == set()
    assert _types_for(sub, "D:0002") == set()


# ── 4. orphan_decision ──

def test_orphan_decision_degree_zero():
    # A D: node with NO incident link edges (the has_decision edge was deleted).
    sub = _sub(
        [_node("D:orphaned"), _node("ep_000001", summary="x")],
        [("ep_000001", "has_entity", "E:Alice")],  # no has_decision to D:
        center="ep_000001",
    )
    assert _types_for(sub, "D:orphaned") == {"orphan_decision"}


def test_orphan_decision_linked_not_flagged():
    sub = _sub(
        [_node("D:linked"), _node("ep_000001", summary="x")],
        [("ep_000001", "has_decision", "D:linked")],
        center="ep_000001",
    )
    assert _types_for(sub, "D:linked") == set()


# ── 5. detached_episode ──

def test_detached_episode_degree_zero():
    # An ep_ with no link edges.
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("ep_000002", summary="y"),
         _node("E:Alice")],
        [("ep_000002", "has_entity", "E:Alice")],
        center="ep_000002",
    )
    assert _types_for(sub, "ep_000001") == {"detached_episode"}


def test_detached_episode_linked_not_flagged():
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("E:Alice")],
        [("ep_000001", "has_entity", "E:Alice")],
        center="ep_000001",
    )
    assert _types_for(sub, "ep_000001") == set()


# ── 6. broken_follows ──

def test_broken_follows_dangling_target():
    # follows points at an id that is NOT in the node set.
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("ep_000002", summary="y")],
        [("ep_000002", "follows", "ep_999999")],  # ep_999999 absent
        center="ep_000001",
    )
    assert _types_for(sub, "ep_000002") == {"broken_follows"}


def test_broken_follows_present_target_not_flagged():
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("ep_000002", summary="y")],
        [("ep_000002", "follows", "ep_000001")],
        center="ep_000001",
    )
    assert _types_for(sub, "ep_000002") == set()


# ── 7. type_violation ──

def test_type_violation_wrong_domain():
    # has_decision domain is Episode; an Entity subject violates it.
    # (Object D: matches range Decision, so only the subject is flagged.)
    sub = _sub(
        [_node("E:Alice"), _node("D:foo"), _node("ep_000001", summary="x")],
        [("E:Alice", "has_decision", "D:foo"),
         ("ep_000001", "has_entity", "E:Alice")],
        center="ep_000001",
    )
    assert _types_for(sub, "E:Alice") == {"type_violation"}
    # D:foo matches the Decision range → not flagged for type_violation.
    assert "type_violation" not in _types_for(sub, "D:foo")


def test_type_violation_wrong_range():
    # has_topic range is Topic; an Entity object violates it.
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("E:Alice")],
        [("ep_000001", "has_topic", "E:Alice")],
        center="ep_000001",
    )
    assert _types_for(sub, "E:Alice") == {"type_violation"}


def test_type_violation_valid_edge_not_flagged():
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("T:db"), _node("E:Alice")],
        [("ep_000001", "has_topic", "T:db"),
         ("ep_000001", "has_entity", "E:Alice")],
        center="ep_000001",
    )
    assert _types_for(sub, "ep_000001") == set()
    assert _types_for(sub, "T:db") == set()
    assert _types_for(sub, "E:Alice") == set()


def test_type_violation_undeclared_predicate_not_checked():
    # 'state' is not a declared ontology property → never a type_violation,
    # even with a node object (data edges are exempt by design).
    sub = _sub(
        [_node("E:Alice"), _node("ep_000001", summary="x")],
        [("E:Alice", "state", "ep_000001")],
        center="ep_000001",
    )
    assert _types_for(sub, "E:Alice") == set()


# ── 8. isolated_cluster ──

def test_isolated_cluster_component_without_center():
    # Two disconnected components: {ep_001, E:Alice} and {ep_002, E:Bob}.
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("E:Alice"),
         _node("ep_000002", summary="y"), _node("E:Bob")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Bob")],
        center="ep_000001",
    )
    # The center's component {ep_001, E:Alice} is NOT flagged; the other is.
    assert _types_for(sub, "ep_000001") == set()
    assert _types_for(sub, "E:Alice") == set()
    assert _types_for(sub, "ep_000002") == {"isolated_cluster"}
    assert _types_for(sub, "E:Bob") == {"isolated_cluster"}


def test_isolated_cluster_subclassof_does_not_bridge():
    # subClassOf edges are seeded taxonomy; they must NOT connect clusters
    # (otherwise every entity would be one big component via the ontology).
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("E:Alice"),
         _node("ep_000002", summary="y"), _node("E:Bob")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Bob"),
         ("E:Alice", "subClassOf", "E:Bob")],  # taxonomy bridge — ignored
        center="ep_000001",
    )
    assert _types_for(sub, "ep_000002") == {"isolated_cluster"}
    assert _types_for(sub, "E:Bob") == {"isolated_cluster"}


def test_isolated_cluster_single_connected_not_flagged():
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("E:Alice"), _node("T:db")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000001", "has_topic", "T:db")],
        center="ep_000001",
    )
    assert detect_anomalies(sub) == []


# ── 9. stale_abstraction ──

def test_stale_abstraction_dead_target():
    # M:0001 abstracts a dead episode id (not in the node set).
    sub = _sub(
        [_node("M:0001"), _node("ep_000001", summary="x")],
        [("M:0001", "abstracts", "ep_999999")],  # dead target
        center="ep_000001",
    )
    assert _types_for(sub, "M:0001") == {"stale_abstraction"}


def test_stale_abstraction_live_target_not_flagged():
    sub = _sub(
        [_node("M:0001"), _node("ep_000001", summary="x")],
        [("M:0001", "abstracts", "ep_000001")],
        center="ep_000001",
    )
    assert _types_for(sub, "M:0001") == set()


# ── node_label_vectors ──

def test_node_label_vectors_aligned_to_anomaly_types():
    # One node carrying two planted anomalies → two type indices.
    sub = _sub(
        [_node("D:0001", text="ship the feature flag rollout"),
         _node("D:0002", text="ship the feature flag rollout"),
         _node("ep_000001", summary="x")],
        [("ep_000001", "has_decision", "D:0001")],  # D:0001 is linked
        # D:0001/D:0002 identical text → duplicate_decision on both;
        # D:0002 has NO link edges → orphan_decision on D:0002.
        center="ep_000001",
    )
    labels = node_label_vectors(sub)
    # Every node id is a key (unlabeled → empty list).
    assert set(labels) == {"D:0001", "D:0002", "ep_000001"}
    assert labels["ep_000001"] == []
    assert ANOMALY_TYPE_INDEX["duplicate_decision"] in labels["D:0001"]
    assert ANOMALY_TYPE_INDEX["duplicate_decision"] in labels["D:0002"]
    assert ANOMALY_TYPE_INDEX["orphan_decision"] in labels["D:0002"]
    # Sorted, no duplicates.
    assert labels["D:0002"] == sorted(set(labels["D:0002"]))


# ── IDENTITY_DRIFT flag (review-flag, NOT a head label) ──

def test_identity_drift_disjoint_topic_neighborhoods():
    # E:Alice appears in two episodes with completely disjoint topics.
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("ep_000002", summary="y"),
         _node("E:Alice"), _node("T:database"), _node("T:parenting")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Alice"),
         ("ep_000001", "has_topic", "T:database"),
         ("ep_000002", "has_topic", "T:parenting")],
        center="ep_000001",
    )
    flags = flag_identity_drift(sub)
    assert len(flags) == 1
    assert flags[0]["node"] == "E:Alice"
    assert flags[0]["type"] == IDENTITY_DRIFT_FLAG


def test_identity_drift_shared_topic_not_flagged():
    # Same entity, but the episodes share a topic → not disjoint.
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("ep_000002", summary="y"),
         _node("E:Alice"), _node("T:database"), _node("T:parenting")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Alice"),
         ("ep_000001", "has_topic", "T:database"),
         ("ep_000002", "has_topic", "T:database"),  # shared
         ("ep_000002", "has_topic", "T:parenting")],
        center="ep_000001",
    )
    assert flag_identity_drift(sub) == []


def test_identity_drift_bidirectional_edges_dedup():
    # The deploy subgraph extractor emits BOTH orientations of an entity-episode
    # link: ``has_entity`` (ep -> E:x) AND ``in_episode`` (E:x -> ep). Without
    # dedup each episode is counted twice per entity, the topic-set list holds
    # the same set twice, and a pair's Jaccard with itself is 1.0 -- so
    # "pairwise disjoint" is never true and the flag could NEVER fire on real
    # (bidirectional) data. This guards the dedup fix.
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("ep_000002", summary="y"),
         _node("E:Alice"), _node("T:database"), _node("T:parenting")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Alice"),
         # The reverse orientation the real BFS extractor emits:
         ("E:Alice", "in_episode", "ep_000001"),
         ("E:Alice", "in_episode", "ep_000002"),
         ("ep_000001", "has_topic", "T:database"),
         ("ep_000002", "has_topic", "T:parenting")],
        center="ep_000001",
    )
    flags = flag_identity_drift(sub)
    assert len(flags) == 1
    assert flags[0]["node"] == "E:Alice"


def test_identity_drift_not_in_anomaly_types():
    # The flag is routed to Bonsai, never to the head's label vector.
    sub = _sub(
        [_node("ep_000001", summary="x"), _node("ep_000002", summary="y"),
         _node("E:Alice"), _node("T:database"), _node("T:parenting")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Alice"),
         ("ep_000001", "has_topic", "T:database"),
         ("ep_000002", "has_topic", "T:parenting")],
        center="ep_000001",
    )
    labels = node_label_vectors(sub)
    # E:Alice has no head-label anomaly here (the drift is a flag, not a label).
    assert labels["E:Alice"] == []
    assert flag_identity_drift(sub)  # but it IS flagged for review


# ── enrich_subgraph (store-backed) ──

def test_enrich_subgraph_hydrates_summary_and_surfaces_abstracts_edge(tmp_path):
    """enrich_subgraph adds episode summaries (duplicate_episode signal) and
    surfaces ``abstracts`` graph edges that ``extract_subgraph`` doesn't walk —
    the two things the detectors need that the raw BFS doesn't provide."""
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        # Two episodes sharing an entity; one is abstracted by an M: memory.
        for i in (1, 2):
            store.encode_episode(Episode(
                id=f"ep_00000{i}", timestamp="t", summary=f"summary {i}",
                full_text=f"text {i}", entities=["Alice"],
            ))
        w = SemanticMemoryWriter(store)
        mid = w.create_abstract(["ep_000001"], "Alice gist")

        # Raw extract_subgraph from ep_000002 reaches Alice but NOT the M:
        # memory (abstracts isn't a traversed predicate). enrich_subgraph
        # must surface the abstracts edge from M: when M: is in the subgraph.
        pipe = OracleLabelingPipeline(store)
        raw = pipe.extract_subgraph("ep_000002", radius=3)
        raw_node_ids = {n["id"] for n in raw["nodes"]}
        assert "summary" not in next(n for n in raw["nodes"] if n["id"] == "ep_000002")
        assert mid not in raw_node_ids  # M: not reached by the BFS

        # Pull the M: node into the subgraph manually (the consolidator would
        # scope it in), then enrich.
        raw["nodes"].append({"id": mid, "type": "semantic_memory", "depth": 9})
        enriched = enrich_subgraph(store, raw)

        # Episode summary hydrated.
        ep_node = next(n for n in enriched["nodes"] if n["id"] == "ep_000002")
        assert ep_node["summary"] == "summary 2"
        # The abstracts edge (mid, abstracts, ep_000001) is now surfaced.
        abs_edges = [(e["subject"], e["predicate"], e["object"])
                     for e in enriched["edges"]
                     if e["predicate"] == "abstracts"]
        assert (mid, "abstracts", "ep_000001") in abs_edges
    finally:
        store.close()