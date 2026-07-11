"""Phase 3b step 8: anomaly -> reconsolidation resolver + apply hook tests.

The anomaly head flags a node as ``contradictory_state`` but its record is
``{node, type, score}`` only -- no state values, source episodes, or ordering.
The resolver (``Consolidator._resolve_contradictory_state``) re-derives what it
can from the graph: confirms >=2 distinct entity ``state`` values, finds the
entity's source episodes, and orders them by timestamp (latest = new, earliest
= old). The apply hook (``_apply``) runs the resolver only on HIGH-confidence
flags (``>= cfg.anomaly_resolve_threshold``); low-confidence stays record-only
(the head over-fires on the giant subgraph).

The encoder never writes entity ``state`` edges (only episode-lifecycle
``state``), so tests plant ``(E:X, state, literal)`` edges directly via the
graph (mirroring the anomaly injector) to exercise the resolver.
"""

from __future__ import annotations

from src.config import ConsolidationConfig, config as _config
from src.gnn import Consolidator
from src.memory.episode import Episode
from src.memory.store import HippocampalStore, _b2s


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid, entities=None, ts="2026-07-01T10:00:00Z"):
    store.encode_episode(Episode(
        id=eid, timestamp=ts, summary=f"s {eid}", full_text=f"f {eid}",
        entities=entities or [],
    ))


def _plant_state(store, entity, value):
    """Plant an entity ``state`` edge (mirrors the anomaly injector)."""
    ops = store.graph.expand_triple(entity, "state", value)
    store.db.batch_sync(ops)


def _consolidator(store, **kw):
    # Untrained model is fine -- the resolver + apply hook use only the store
    # (graph queries + get_episode), never the model. allow_untrained_apply so
    # _apply can be exercised directly.
    return Consolidator(store, dry_run=False, allow_untrained_apply=True,
                        config=ConsolidationConfig(**kw))


# ── resolver ──

def test_resolver_returns_oldest_and_newest(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_old", entities=["Alice"], ts="2026-07-01T10:00:00Z")
    _encode(store, "ep_new", entities=["Alice"], ts="2026-07-05T10:00:00Z")
    _plant_state(store, "E:Alice", "alive")
    _plant_state(store, "E:Alice", "dead")

    cons = _consolidator(store)
    pair = cons._resolve_contradictory_state("E:Alice")
    assert pair == ("ep_old", "ep_new")
    store.close()


def test_resolver_none_when_only_one_state_value(tmp_path):
    """Head over-fired: only one live state value -> no real contradiction."""
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"], ts="2026-07-01T10:00:00Z")
    _encode(store, "ep_002", entities=["Alice"], ts="2026-07-02T10:00:00Z")
    _plant_state(store, "E:Alice", "alive")  # only one value

    cons = _consolidator(store)
    assert cons._resolve_contradictory_state("E:Alice") is None
    store.close()


def test_resolver_none_when_only_one_source_episode(tmp_path):
    """Can't attribute the contradiction to >=2 assertions."""
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"], ts="2026-07-01T10:00:00Z")
    _plant_state(store, "E:Alice", "alive")
    _plant_state(store, "E:Alice", "dead")

    cons = _consolidator(store)
    assert cons._resolve_contradictory_state("E:Alice") is None
    store.close()


def test_resolver_none_for_non_entity_node(tmp_path):
    """contradictory_state is entity-scoped; non-E nodes are record-only."""
    store = _store(tmp_path)
    cons = _consolidator(store)
    assert cons._resolve_contradictory_state("ep_001") is None
    assert cons._resolve_contradictory_state("T:db") is None
    store.close()


def test_resolver_dedups_repeated_state_value(tmp_path):
    """Two edges with the SAME value are one distinct value -> None."""
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"], ts="2026-07-01T10:00:00Z")
    _encode(store, "ep_002", entities=["Alice"], ts="2026-07-02T10:00:00Z")
    _plant_state(store, "E:Alice", "alive")
    _plant_state(store, "E:Alice", "alive")  # dup

    cons = _consolidator(store)
    assert cons._resolve_contradictory_state("E:Alice") is None
    store.close()


# ── apply hook ──

def _report_with_anomaly(node, score):
    return {
        "abstracts": [], "edges_accepted": [], "pruned": [],
        "ontology_proposed": [],
        "anomalies": [
            {"node": node, "type": "contradictory_state", "score": score},
        ],
        "forgetting": {"edges_seen": 0, "boosted": 0, "archived": [], "ltp": 0,
                       "reconsolidated": []},
    }


def test_apply_reconsolidates_high_confidence_contradictory_state(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_old", entities=["Alice"], ts="2026-07-01T10:00:00Z")
    _encode(store, "ep_new", entities=["Alice"], ts="2026-07-05T10:00:00Z")
    _plant_state(store, "E:Alice", "alive")
    _plant_state(store, "E:Alice", "dead")

    cons = _consolidator(store)
    cons._forget_updates = []
    cons._forget_node_salience = {}
    report = _report_with_anomaly("E:Alice", score=0.9)
    cons._apply(report)

    rec = report["forgetting"]["reconsolidated"]
    assert len(rec) == 1
    assert rec[0] == {"entity": "E:Alice", "old": "ep_old", "new": "ep_new"}
    # the E->E supersedes chain was written.
    assert store.episode_state("ep_old") == "superseded"
    assert store.episode_validity_end("ep_old") is not None
    q = store.graph.query().vertex("ep_new").out("supersedes")
    r = q.execute_sync()
    try:
        assert list(r.vertices) == ["ep_old"]
    finally:
        r.close()
    store.close()


def test_apply_low_confidence_flag_is_record_only(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_old", entities=["Alice"], ts="2026-07-01T10:00:00Z")
    _encode(store, "ep_new", entities=["Alice"], ts="2026-07-05T10:00:00Z")
    _plant_state(store, "E:Alice", "alive")
    _plant_state(store, "E:Alice", "dead")

    cons = _consolidator(store, anomaly_resolve_threshold=0.8)
    cons._forget_updates = []
    cons._forget_node_salience = {}
    # below threshold -> record-only (no reconsolidation).
    report = _report_with_anomaly("E:Alice", score=0.5)
    cons._apply(report)
    assert report["forgetting"]["reconsolidated"] == []
    # old episode NOT superseded.
    assert store.episode_state("ep_old") == "current"
    store.close()


def test_apply_skips_non_contradictory_anomaly_types(tmp_path):
    """Only contradictory_state is resolved; other anomaly types are ignored."""
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    cons = _consolidator(store)
    cons._forget_updates = []
    cons._forget_node_salience = {}
    report = {
        "abstracts": [], "edges_accepted": [], "pruned": [],
        "ontology_proposed": [],
        "anomalies": [
            {"node": "ep_001", "type": "duplicate_episode", "score": 0.99},
        ],
        "forgetting": {"edges_seen": 0, "boosted": 0, "archived": [], "ltp": 0,
                       "reconsolidated": []},
    }
    cons._apply(report)
    assert report["forgetting"]["reconsolidated"] == []
    store.close()


def test_apply_forgetting_disabled_skips_resolver(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_old", entities=["Alice"], ts="2026-07-01T10:00:00Z")
    _encode(store, "ep_new", entities=["Alice"], ts="2026-07-05T10:00:00Z")
    _plant_state(store, "E:Alice", "alive")
    _plant_state(store, "E:Alice", "dead")

    saved = _config.forgetting_enabled
    _config.forgetting_enabled = False
    try:
        cons = _consolidator(store)
        cons._forget_updates = []
        cons._forget_node_salience = {}
        report = _report_with_anomaly("E:Alice", score=0.9)
        cons._apply(report)
    finally:
        _config.forgetting_enabled = saved
    assert report["forgetting"]["reconsolidated"] == []
    assert store.episode_state("ep_old") == "current"
    store.close()


def test_apply_resolver_returns_none_is_record_only(tmp_path):
    """High-confidence flag but resolver can't confirm (<2 state values)."""
    store = _store(tmp_path)
    _encode(store, "ep_old", entities=["Alice"], ts="2026-07-01T10:00:00Z")
    _encode(store, "ep_new", entities=["Alice"], ts="2026-07-05T10:00:00Z")
    _plant_state(store, "E:Alice", "alive")  # only one value -> resolver None

    cons = _consolidator(store)
    cons._forget_updates = []
    cons._forget_node_salience = {}
    report = _report_with_anomaly("E:Alice", score=0.9)
    cons._apply(report)
    assert report["forgetting"]["reconsolidated"] == []
    assert store.episode_state("ep_old") == "current"
    store.close()