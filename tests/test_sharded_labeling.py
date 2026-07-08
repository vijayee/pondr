"""Tests for ``src/gnn/sharded_labeling.py`` (Phase 3a Task 3 sharded labeling).

Pure-dict, no Oracle, no store: shard construction, candidate-pair samplers, the
local-context prompt builders, recombination (merge shards → per-subgraph JSONL
in the existing schemas), partial-label masking helpers, and the Oracle-free
anomaly label builder. The anomaly builder is exercised on a hand-built enriched
subgraph (the same shape ``anomaly_injector`` consumes — see ``test_anomaly_injector``).
"""

from __future__ import annotations

from src.gnn.sharded_labeling import (
    DEFAULT_MAX_CANDIDATE_PAIRS,
    DEFAULT_SHARD_SIZE,
    MIN_LABELED_FRACTION,
    build_anomaly_labels,
    build_cluster_episode_prompt,
    build_link_pred_shards,
    build_link_pred_shard_prompt,
    build_ontology_shards,
    build_ontology_shard_prompt,
    build_salience_shard_prompt,
    episode_only_context,
    global_summary,
    group_by_subgraph,
    meets_min_labeled_fraction,
    recombine_cluster,
    recombine_link_pred,
    recombine_ontology,
    recombine_salience,
    sample_link_pred_candidates,
    sample_ontology_candidates,
    salience_coverage,
    shard_nodes,
    shard_pairs,
    to_shard_record,
)
from src.gnn.anomaly_rules import ANOMALY_TYPES, ANOMALY_TYPE_INDEX


# ── helpers ──

class _Result:
    """Stand-in for ``OracleResult`` — only ``.response`` + ``.cost`` are read."""
    def __init__(self, response: dict, cost: float = 0.0) -> None:
        self.response = response
        self.cost = cost


def _node(nid: str, **kw) -> dict:
    for prefix, typ in (
        ("ep_", "episode"), ("E:", "entity"), ("T:", "topic"),
        ("A:", "tone"), ("D:", "decision"), ("S:", "session"),
        ("U:", "user"), ("M:", "semantic_memory"),
    ):
        if nid.startswith(prefix):
            return {"id": nid, "type": typ, "depth": 0, **kw}
    return {"id": nid, "type": "unknown", "depth": 0, **kw}


def _sub(nodes: list[dict], edges: list[tuple], center: str) -> dict:
    return {
        "center": center,
        "nodes": nodes,
        "edges": [{"subject": s, "predicate": p, "object": o} for s, p, o in edges],
    }


def _enriched_clean() -> dict:
    """A clean, connected, ENRICHED subgraph (summaries present) — the shape
    ``anomaly_injector.inject_anomalies`` consumes. Same fixture as
    ``test_anomaly_injector._clean_sub``."""
    return {
        "center": "ep_000001",
        "nodes": [
            {"id": "ep_000001", "type": "episode", "depth": 0,
             "summary": "Alice discussed the database schema"},
            {"id": "ep_000002", "type": "episode", "depth": 1,
             "summary": "Alice reviewed the schema migration"},
            {"id": "E:Alice", "type": "entity", "depth": 1},
            {"id": "T:database", "type": "topic", "depth": 1},
            {"id": "D:0001", "type": "decision", "depth": 1,
             "text": "use postgres for the database"},
        ],
        "edges": [
            {"subject": "ep_000001", "predicate": "has_entity", "object": "E:Alice"},
            {"subject": "ep_000001", "predicate": "has_topic", "object": "T:database"},
            {"subject": "ep_000001", "predicate": "has_decision", "object": "D:0001"},
            {"subject": "ep_000002", "predicate": "follows", "object": "ep_000001"},
            {"subject": "ep_000002", "predicate": "has_entity", "object": "E:Alice"},
        ],
    }


# ── global_summary ──

def test_global_summary_shape_and_top_shared():
    sub = _sub(
        [_node("ep_000001"), _node("ep_000002"), _node("E:Alice"), _node("T:db"),
         _node("E:Bob"), _node("T:auth")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Alice"),  # Alice in 2 episodes → top
         ("ep_000002", "has_entity", "E:Bob"),    # Bob in 1 episode
         ("ep_000001", "has_topic", "T:db"),
         ("ep_000002", "has_topic", "T:auth")],
        center="ep_000001",
    )
    g = global_summary(sub)
    assert g["total_nodes"] == 6
    assert g["total_edges"] == 5
    # Alice (degree 2) ranks above Bob (degree 1); ties break by id.
    assert g["top_shared_entities"][0] == {"id": "E:Alice", "episodes": 2}
    assert {e["id"] for e in g["top_shared_entities"]} == {"E:Alice", "E:Bob"}


# ── shard_nodes ──

def test_shard_nodes_deterministic_capped_covers_all():
    nodes = [_node(f"ep_{i:06d}") for i in range(12)]
    sub = {"center": "ep_000000", "nodes": nodes, "edges": []}
    shards = shard_nodes(sub, shard_size=5)
    assert [len(s["nodes"]) for s in shards] == [5, 5, 2]  # 3 shards
    # All nodes covered exactly once, deterministic order.
    covered = [n["id"] for s in shards for n in s["nodes"]]
    assert covered == sorted(n["id"] for n in nodes)
    # shard_idx is sequential; subgraph_id is the center.
    assert [s["shard_idx"] for s in shards] == [0, 1, 2]
    assert all(s["subgraph_id"] == "ep_000000" and s["center"] == "ep_000000"
               for s in shards)
    # Global summary + totals present.
    assert shards[0]["global"]["total_nodes"] == 12
    assert shards[0]["total_edges"] == 0


def test_shard_nodes_induced_edges_include_center():
    # Lexicographic sort puts uppercase-prefix ids ('E:', 'T:') before
    # lowercase 'ep_', so shard 0 = [E:Alice, E:Bob] + center ep_000001.
    sub = _sub(
        [_node("ep_000001"), _node("ep_000002"), _node("E:Alice"), _node("E:Bob"),
         _node("T:db")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000002", "has_entity", "E:Bob"),       # Bob's only link (cross-shard)
         ("ep_000001", "has_topic", "T:db"),
         ("E:Alice", "state", "alive")],              # data edge → excluded
        center="ep_000001",
    )
    shards = shard_nodes(sub, shard_size=2)
    for s in shards:
        s_ids = {n["id"] for n in s["nodes"]} | {s["center"]}
        for e in s["edges"]:
            # Both endpoints must be node ids in the shard ∪ center.
            assert e["subject"] in s_ids
            assert e["object"] in s_ids
    # The shard containing E:Alice has the ep_000001→E:Alice edge induced (center
    # is in every shard's induced-edge set); the data 'state' edge never appears.
    alice_shard = next(s for s in shards if any(n["id"] == "E:Alice" for n in s["nodes"]))
    induced = [(e["subject"], e["predicate"], e["object"]) for e in alice_shard["edges"]]
    assert ("ep_000001", "has_entity", "E:Alice") in induced
    assert all(e[1] != "state" for e in induced)
    # ep_000002→E:Bob is a cross-shard edge (Bob is in shard 0, ep_000002 in the
    # ep_ shard) → never induced into any single shard.
    for s in shards:
        assert ("ep_000002", "has_entity", "E:Bob") not in [
            (e["subject"], e["predicate"], e["object"]) for e in s["edges"]]


# ── candidate samplers ──

def test_sample_link_pred_candidates_same_kind_nonedge_capped():
    sub = _sub(
        [_node("E:A"), _node("E:B"), _node("T:x"), _node("E:C")],
        [("E:A", "has_entity", "T:x")],   # not same-kind, irrelevant
        center="E:A",
    )
    pairs = sample_link_pred_candidates(sub, max_candidate_pairs=2)
    # All same-kind (entity-entity) non-edge pairs; E:A–E:B before E:A–E:C (sorted).
    assert all(a.startswith("E:") and b.startswith("E:") for a, b in pairs)
    assert ("E:A", "E:B") in pairs or ("E:B", "E:A") in pairs  # sorted → (E:A, E:B)
    assert len(pairs) == 2  # capped


def test_sample_link_pred_candidates_excludes_existing_edges():
    sub = _sub(
        [_node("E:A"), _node("E:B"), _node("E:C")],
        [("E:A", "related_to", "E:B")],   # an existing same-kind edge
        center="E:A",
    )
    pairs = sample_link_pred_candidates(sub, max_candidate_pairs=10)
    assert ("E:A", "E:B") not in pairs and ("E:B", "E:A") not in pairs
    # Only E:A–E:C and E:B–E:C remain (both directions excluded for the A–B edge).
    assert set(pairs) == {("E:A", "E:C"), ("E:B", "E:C")}


def test_sample_ontology_candidates_entity_topic_pairs_capped():
    sub = _sub(
        [_node("E:A"), _node("E:B"), _node("T:x"), _node("D:1"), _node("ep_1")],
        [], center="E:A",
    )
    pairs = sample_ontology_candidates(sub, max_candidate_pairs=2)
    # Only E: and T: nodes pair up (D:/ep_ excluded); sorted; capped at 2.
    assert all((a.startswith("E:") or a.startswith("T:"))
               and (b.startswith("E:") or b.startswith("T:")) for a, b in pairs)
    assert len(pairs) == 2
    # E:A,E:B is the alphabetically-first pair.
    assert ("E:A", "E:B") == pairs[0]


# ── shard_pairs / episode_only_context ──

def test_shard_pairs_carries_local_context():
    sub = _sub([_node("E:A"), _node("E:B"), _node("E:C")], [], center="E:A")
    pairs = [("E:A", "E:B"), ("E:B", "E:C")]
    shards = shard_pairs(sub, pairs, "link_prediction", shard_size=1)
    assert len(shards) == 2
    s0 = shards[0]
    assert s0["pairs"] == [("E:A", "E:B")]
    # The shard's nodes are the union of pair endpoints, sorted.
    assert [n["id"] for n in s0["nodes"]] == ["E:A", "E:B"]
    assert s0["task"] == "link_prediction"
    assert s0["global"]["total_nodes"] == 3


def test_episode_only_context_drops_non_episodes():
    sub = _sub(
        [_node("ep_000001", summary="s1", timestamp="t1"),
         _node("ep_000002", summary="s2", timestamp="t2"),
         _node("E:Alice"), _node("T:db")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000001", "has_topic", "T:db"),
         ("ep_000002", "has_entity", "E:Alice")],
        center="ep_000001",
    )
    ctx = episode_only_context(sub)
    assert ctx["task"] == "cluster" and ctx["shard_idx"] == 0
    assert [e["id"] for e in ctx["episodes"]] == ["ep_000001", "ep_000002"]
    assert ctx["episodes"][0]["summary"] == "s1"
    # Non-episode nodes are NOT in the episodes list; Alice is shared (2 eps).
    assert "E:Alice" in ctx["shared_entities"]
    assert "T:db" in ctx["shared_topics"]


# ── prompt builders ──

def test_build_salience_shard_prompt_includes_instruction_and_nodes():
    sub = _sub([_node("ep_000001", summary="x"), _node("E:Alice")],
               [("ep_000001", "has_entity", "E:Alice")], center="ep_000001")
    shard = shard_nodes(sub, shard_size=10)[0]
    p = build_salience_shard_prompt(shard)
    # The "score ONLY these N" instruction + the shard's nodes.
    assert "Score ONLY these" in p
    assert "ep_000001" in p and "E:Alice" in p
    # The JSON return contract.
    assert '"node_scores"' in p


def test_build_link_pred_shard_prompt_includes_pairs_and_contract():
    sub = _sub([_node("E:A"), _node("E:B")], [], center="E:A")
    shard = build_link_pred_shards(sub, shard_size=10, max_candidate_pairs=5)[0]
    p = build_link_pred_shard_prompt(shard)
    assert "predicted_edges" in p and "negative_edges" in p
    assert "E:A" in p and "E:B" in p


def test_build_ontology_shard_prompt_includes_ontology():
    sub = _sub([_node("E:A"), _node("E:B")], [], center="E:A")
    shard = build_ontology_shards(sub, shard_size=10, max_candidate_pairs=5)[0]
    p = build_ontology_shard_prompt(shard, '{"properties": {}}')
    assert "CURRENT ONTOLOGY" in p
    assert '"properties": {}' in p
    assert "suggested_edges" in p and "misclassified" in p


def test_build_cluster_episode_prompt_includes_episodes():
    sub = _sub([_node("ep_000001", summary="s1")], [], center="ep_000001")
    ctx = episode_only_context(sub)
    p = build_cluster_episode_prompt(ctx)
    assert "clusters" in p and "ep_000001" in p and "s1" in p


# ── to_shard_record + group_by_subgraph ──

def test_to_shard_record_tags_ids():
    shard = {"subgraph_id": "ep_1", "shard_idx": 2, "task": "salience"}
    rec = to_shard_record(shard, _Result({"node_scores": {}}, cost=0.01), 5)
    assert rec["subgraph_id"] == "ep_1"
    assert rec["shard_idx"] == 2
    assert rec["task"] == "salience"
    assert rec["cost"] == 0.01
    assert rec["labels"] == {"node_scores": {}}


def test_group_by_subgraph_sorts_by_shard_idx():
    recs = [
        {"subgraph_id": "ep_1", "shard_idx": 2, "labels": {}, "cost": 0},
        {"subgraph_id": "ep_1", "shard_idx": 0, "labels": {}, "cost": 0},
        {"subgraph_id": "ep_2", "shard_idx": 0, "labels": {}, "cost": 0},
    ]
    grouped = group_by_subgraph(recs)
    assert [r["shard_idx"] for r in grouped["ep_1"]] == [0, 2]
    assert list(grouped) == ["ep_1", "ep_2"]


# ── recombine_salience ──

def test_recombine_salience_merges_and_computes_edge_scores():
    sub = _sub(
        [_node("ep_000001"), _node("E:Alice"), _node("E:Bob")],
        [("ep_000001", "has_entity", "E:Alice"),
         ("ep_000001", "has_entity", "E:Bob"),
         ("E:Alice", "state", "alive")],   # data edge → no edge_score
        center="ep_000001",
    )
    subgraphs_by_id = {sub["center"]: sub}
    shard_recs = [
        {"subgraph_id": "ep_000001", "shard_idx": 0, "labels": {
            "node_scores": {"ep_000001": {"salience": 0.9, "reason": "ctr"},
                            "E:Alice": {"salience": 0.8}}}, "cost": 0.1},
        {"subgraph_id": "ep_000001", "shard_idx": 1, "labels": {
            "node_scores": {"E:Bob": {"salience": 0.4}}}, "cost": 0.2},
    ]
    out = recombine_salience(shard_recs, subgraphs_by_id)
    assert len(out) == 1
    rec = out[0]
    labels = rec["labels"]
    # node_scores merged across shards.
    assert set(labels["node_scores"]) == {"ep_000001", "E:Alice", "E:Bob"}
    # edge_scores = mean of endpoints (Alice: 0.9+0.8)/2; Bob: (0.9+0.4)/2.
    es = labels["edge_scores"]
    assert abs(es["ep_000001|has_entity|E:Alice"]["salience"] - 0.85) < 1e-9
    assert abs(es["ep_000001|has_entity|E:Bob"]["salience"] - 0.65) < 1e-9
    # Data edge (literal object) has no edge_score.
    assert all("|state|" not in k for k in es)
    # Cost summed.
    assert abs(rec["cost"] - 0.3) < 1e-9


def test_recombine_salience_skips_partial_edges_and_handles_bare_numbers():
    sub = _sub(
        [_node("ep_000001"), _node("E:Alice"), _node("E:Bob")],
        [("ep_000001", "has_entity", "E:Alice"),   # both labeled
         ("ep_000001", "has_entity", "E:Bob")],    # Bob unlabeled → skip
        center="ep_000001",
    )
    out = recombine_salience(
        [{"subgraph_id": "ep_000001", "shard_idx": 0, "labels": {
            "node_scores": {"ep_000001": 0.9, "E:Alice": 0.8}}, "cost": 0}],
        {sub["center"]: sub},
    )
    es = out[0]["labels"]["edge_scores"]
    assert "ep_000001|has_entity|E:Alice" in es  # bare numbers handled
    assert abs(es["ep_000001|has_entity|E:Alice"]["salience"] - 0.85) < 1e-9
    assert "ep_000001|has_entity|E:Bob" not in es  # partial → skipped


# ── recombine link/ontology/cluster ──

def test_recombine_link_pred_concats():
    recs = [
        {"subgraph_id": "ep_1", "shard_idx": 0, "labels": {
            "predicted_edges": [{"subject": "E:A", "object": "E:B"}],
            "negative_edges": [{"subject": "E:A", "object": "T:x"}]}, "cost": 0.1},
        {"subgraph_id": "ep_1", "shard_idx": 1, "labels": {
            "predicted_edges": [{"subject": "E:B", "object": "E:C"}],
            "negative_edges": []}, "cost": 0.2},
    ]
    out = recombine_link_pred(recs)
    labels = out[0]["labels"]
    assert len(labels["predicted_edges"]) == 2
    assert len(labels["negative_edges"]) == 1
    assert abs(out[0]["cost"] - 0.3) < 1e-9


def test_recombine_ontology_concats():
    recs = [
        {"subgraph_id": "ep_1", "shard_idx": 0, "labels": {
            "suggested_edges": [{"child": "E:A", "parent": "E:B"}],
            "misclassified": [{"entity": "E:A"}]}, "cost": 0.1},
        {"subgraph_id": "ep_1", "shard_idx": 1, "labels": {
            "suggested_edges": [], "misclassified": [{"entity": "E:C"}]}, "cost": 0.2},
    ]
    out = recombine_ontology(recs)
    labels = out[0]["labels"]
    assert len(labels["suggested_edges"]) == 1
    assert len(labels["misclassified"]) == 2
    assert abs(out[0]["cost"] - 0.3) < 1e-9


def test_recombine_cluster_passthrough():
    recs = [{"subgraph_id": "ep_1", "shard_idx": 0, "labels": {
        "clusters": [{"name": "g"}]}, "cost": 0.1}]
    out = recombine_cluster(recs)
    assert out[0]["labels"]["clusters"] == [{"name": "g"}]
    assert out[0]["cost"] == 0.1


# ── partial-label masking ──

def test_salience_coverage_reports_unlabeled():
    sub = _sub([_node("ep_1"), _node("E:A"), _node("E:B")], [], center="ep_1")
    cov = salience_coverage({"ep_1": {"salience": 0.5}, "E:A": {"salience": 0.7}}, sub)
    assert cov["fraction"] == 2 / 3
    assert cov["unlabeled"] == ["E:B"]
    assert cov["labeled"] == ["E:A", "ep_1"]


def test_meets_min_labeled_fraction():
    cov_ok = {"fraction": 0.8}
    cov_low = {"fraction": 0.3}
    assert meets_min_labeled_fraction(cov_ok)
    assert not meets_min_labeled_fraction(cov_low)
    assert meets_min_labeled_fraction(cov_low, min_fraction=0.25)


def test_min_labeled_fraction_default_is_half():
    assert MIN_LABELED_FRACTION == 0.5


# ── build_anomaly_labels (Oracle-free) ──

def test_build_anomaly_labels_validator_shape_and_zero_cost():
    rec = build_anomaly_labels(_enriched_clean(), seed=0, types=None)
    assert rec["subgraph_id"] == "ep_000001"
    assert rec["cost"] == 0.0  # Oracle-free
    labels = rec["labels"]
    # The validator key is present (keeps validate_gnn unchanged).
    assert "anomalies" in labels
    # Extras the trainer + Bonsai read.
    assert "node_labels" in labels
    assert "planted" in labels
    assert "identity_drift" in labels
    # Reproducibility metadata so the trainer can re-build the corrupted graph.
    assert labels["seed"] == 0 and labels["types"] is None


def test_build_anomaly_labels_node_vectors_aligned_to_types():
    rec = build_anomaly_labels(_enriched_clean(), seed=7, types=None)
    node_labels = rec["labels"]["node_labels"]
    # Every node id in the corrupted subgraph is a key, with type indices in range.
    for nid, idxs in node_labels.items():
        assert all(0 <= i < len(ANOMALY_TYPES) for i in idxs)
        assert idxs == sorted(set(idxs))  # sorted, no dups
    # An anomaly the injector planted on this clean fixture surfaces as a label.
    assert any(idxs for idxs in node_labels.values())


def test_build_anomaly_labels_deterministic_same_seed():
    a = build_anomaly_labels(_enriched_clean(), seed=42)
    b = build_anomaly_labels(_enriched_clean(), seed=42)
    assert a["labels"]["node_labels"] == b["labels"]["node_labels"]
    assert a["labels"]["planted"] == b["labels"]["planted"]
    assert a["labels"]["anomalies"] == b["labels"]["anomalies"]


def test_build_anomaly_labels_identity_drift_is_flag_not_head_label():
    rec = build_anomaly_labels(_enriched_clean(), seed=0)
    # identity_drift is collected for Bonsai; its type string is NOT a head index.
    for flag in rec["labels"]["identity_drift"]:
        assert flag["type"] not in ANOMALY_TYPE_INDEX
    # And it never appears in the per-node head-label vectors.
    head_types = {ANOMALY_TYPES[i]
                  for idxs in rec["labels"]["node_labels"].values()
                  for i in idxs}
    assert "identity_drift" not in head_types