"""Round-trip tests for the anomaly injector (``src/gnn/anomaly_injector.py``).

The closed loop: inject a corruption → ``anomaly_rules.detect_anomalies``
recovers exactly what was planted (spec §8 acceptance). Each test builds a
clean connected subgraph, injects one type, and asserts the planted record is
recovered (recall = 1.0) with no collateral on untouched nodes (precision)
where the injection is surgical.
"""

from __future__ import annotations

from src.gnn.anomaly_injector import inject_anomalies, round_trip
from src.gnn.anomaly_rules import ANOMALY_TYPES, detect_anomalies


def _clean_sub() -> dict:
    """A clean, connected subgraph with the node kinds the injectors need:
    an episode with a summary, an entity, a decision linked to the episode,
    and a second episode that ``follows`` the first (a follows edge to rewire)."""
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


def _detected(corrupted: dict) -> set[tuple[str, str]]:
    return {(f["node"], f["type"]) for f in detect_anomalies(corrupted)}


# ── per-type recall (planted ⊆ detected) ──

def test_inject_contradictory_state_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=1, types=["contradictory_state"])
    det = _detected(corrupted)
    assert planted
    assert all((p["node"], p["type"]) in det for p in planted)


def test_inject_duplicate_episode_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=2, types=["duplicate_episode"])
    det = _detected(corrupted)
    assert len(planted) == 2  # orig + clone
    assert all((p["node"], p["type"]) in det for p in planted)


def test_inject_duplicate_decision_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=3, types=["duplicate_decision"])
    det = _detected(corrupted)
    assert len(planted) == 2
    assert all((p["node"], p["type"]) in det for p in planted)


def test_inject_orphan_decision_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=4, types=["orphan_decision"])
    det = _detected(corrupted)
    assert planted
    assert all((p["node"], p["type"]) in det for p in planted)


def test_inject_detached_episode_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=5, types=["detached_episode"])
    det = _detected(corrupted)
    assert planted
    # The detached episode is ep_000002 (the only non-center episode).
    assert planted[0]["node"] == "ep_000002"
    assert all((p["node"], p["type"]) in det for p in planted)


def test_inject_broken_follows_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=6, types=["broken_follows"])
    det = _detected(corrupted)
    assert planted
    assert planted[0]["node"] == "ep_000002"
    assert all((p["node"], p["type"]) in det for p in planted)


def test_inject_type_violation_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=7, types=["type_violation"])
    det = _detected(corrupted)
    assert planted
    assert all((p["node"], p["type"]) in det for p in planted)


def test_inject_isolated_cluster_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=8, types=["isolated_cluster"])
    det = _detected(corrupted)
    assert len(planted) == 3  # ep + 2 entities
    assert all((p["node"], p["type"]) in det for p in planted)


def test_inject_stale_abstraction_recovered():
    corrupted, planted = inject_anomalies(_clean_sub(), seed=9, types=["stale_abstraction"])
    det = _detected(corrupted)
    assert planted
    assert all((p["node"], p["type"]) in det for p in planted)


# ── full closed loop: all 9 types at once, recall == 1.0 ──

def test_round_trip_all_types_full_recall():
    """Each type injected in isolation on a fresh copy; every planted label is
    recovered by the rules (spec §8: inject → rule-detect recovers exactly
    what was planted). Isolated injection avoids cross-type interactions
    (e.g. a type_violation edge re-linking an orphaned node) that would mask a
    real injector/detector mismatch."""
    rt = round_trip(_clean_sub(), seed=0)
    # Every type was plantable on this subgraph (it has an episode, entity,
    # decision, and a follows edge).
    planted_types = {p[1] for p in rt["planted"]}
    assert planted_types == set(ANOMALY_TYPES)
    assert rt["missed"] == []
    assert rt["recall"] == 1.0


def test_round_trip_no_anomalies_on_clean_subgraph():
    """The clean subgraph (no injection) has zero anomalies — confirms the
    detectors don't false-positive on well-formed data."""
    assert detect_anomalies(_clean_sub()) == []


# ── determinism ──

def test_inject_is_deterministic_same_seed():
    """Same seed → identical corrupted subgraph + planted records."""
    a = inject_anomalies(_clean_sub(), seed=42)
    b = inject_anomalies(_clean_sub(), seed=42)
    assert a[1] == b[1]
    assert a[0] == b[0]


def test_inject_does_not_mutate_input():
    """The input subgraph is deep-copied — injection never mutates the caller's
    dict (the training pipeline reuses the clean subgraph across types)."""
    clean = _clean_sub()
    orig_edges = len(clean["edges"])
    orig_nodes = len(clean["nodes"])
    inject_anomalies(clean, seed=0)
    assert len(clean["edges"]) == orig_edges
    assert len(clean["nodes"]) == orig_nodes