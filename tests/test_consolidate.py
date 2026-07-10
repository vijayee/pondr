"""Tests for the dream-state consolidation loop (``src/gnn/consolidate.py``)."""

from __future__ import annotations

from src.config import ConsolidationConfig
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.gnn import Consolidator


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _populate(store, n=4):
    for i in range(1, n + 1):
        store.encode_episode(Episode(
            id=f"ep_00000{i}", timestamp="t", summary=f"s{i}", full_text=f"f{i}",
            entities=["Alice", "Bob"], topics=["db"],
        ))


def test_dry_run_does_not_mutate(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    cons = Consolidator(store, dry_run=True)
    rep = cons.run(limit=3)
    assert rep["dry_run"] is True
    assert rep["trained"] is False
    assert rep["subgraphs_scored"] == 3
    # Report has every step's field.
    for k in ("abstracts", "edges_proposed", "edges_accepted", "edges_unverified",
              "anomalies", "ontology_proposed", "pruned"):
        assert k in rep
    # No mutation: no M nodes, nothing abstracted.
    assert not any(cons.writer.get_abstract(f"M:{i:04d}") for i in range(1, 5))
    assert not any(store.is_abstracted(f"ep_00000{i}") for i in range(1, 5))
    store.close()


def test_apply_with_untrained_model_is_skipped_without_force(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    cons = Consolidator(store, dry_run=False)  # allow_untrained_apply defaults False
    rep = cons.run(limit=3)
    assert rep["dry_run"] is False
    assert "apply_skipped" in rep
    # No M nodes written.
    mems = [k.split("/")[2] for k, _ in store.db.create_read_stream(
        start="content/mem/", end="content/mem/\x7f")]
    assert mems == []
    store.close()


def test_apply_with_force_writes_abstracts(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True)
    rep = cons.run(limit=3)
    assert "apply_skipped" not in rep
    mems = sorted({k.split("/")[2] for k, _ in store.db.create_read_stream(
        start="content/mem/", end="content/mem/\x7f")})
    assert len(mems) >= 1
    assert all(m.startswith("M:") for m in mems)
    store.close()


def test_verifier_receives_medium_confidence_proposals(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    seen: list[dict] = []

    def verifier(proposal: dict) -> bool:
        seen.append(proposal)
        return True  # accept everything proposed

    cons = Consolidator(store, dry_run=True, verifier=verifier)
    rep = cons.run(limit=3)
    # The verifier is only called for proposals in the propose band; with an
    # untrained model scores are arbitrary, so just assert the counters are
    # consistent: accepted ≤ calls, and validation_rate ∈ {None or [0,1]}.
    assert rep["verifier_accepted"] <= rep["verifier_calls"]
    if rep["verifier_calls"]:
        assert 0.0 <= rep["verifier_validation_rate"] <= 1.0
    store.close()


def test_wm_prioritization_orders_resident_centers_first(tmp_path):
    store = _store(tmp_path)
    _populate(store, n=5)
    cons = Consolidator(store, dry_run=True)
    ordered = cons._wm_first(["ep_000003", "ep_000001", "ep_000004"],
                             wm_ids={"ep_000001"})
    # ep_000001 is WM-resident → first; the rest keep stable order.
    assert ordered[0] == "ep_000001"
    store.close()


def test_ontology_proposals_have_entity_class_shape(tmp_path):
    """The ontology step records ``{entity, class, confidence}`` typing proposals
    (entity->class, the two-encoder pair classifier's output) -- NOT the old
    ``{child, parent}`` shape from the single-encoder design. A low accept
    threshold forces every scored pair to be recorded so the shape is observable
    even with an untrained (random) model."""
    store = _store(tmp_path)
    _populate(store)
    # accept_threshold=0.0 -> every scored entity/class pair is recorded (the
    # untrained model's scores are arbitrary; this makes the proposals non-empty
    # so we can assert their shape, not their quality).
    cons = Consolidator(store, dry_run=True,
                        config=ConsolidationConfig(accept_threshold=0.0))
    rep = cons.run(limit=3)
    assert rep["subgraphs_scored"] == 3
    props = rep["ontology_proposed"]
    assert len(props) > 0  # entities were scored against candidate classes
    for p in props:
        # New two-encoder shape: entity (an E: node) + class (a bare class name
        # from the taxonomy DAG) + confidence.
        assert set(p) == {"entity", "class", "confidence"}
        assert p["entity"].startswith("E:")
        assert isinstance(p["class"], str)
        assert 0.0 <= p["confidence"] <= 1.0
    store.close()


def test_anomaly_step_runs_second_bounded_forward(tmp_path):
    """The anomaly step runs on a SEPARATE bounded subgraph (radius-2 + fanout-cap)
    -- the giant-subgraph fix -- while the other 4 steps stay on the radius-3
    giant. So ``loader.load`` is called TWICE per center: once with the default
    (radius-3, no cap) for cluster/link/ontology/prune, and once with the bounded
    radius+cap for anomaly. Train/serve parity: the head is served on the same
    bounded graph it trained on.
    """
    store = _store(tmp_path)
    _populate(store, n=4)
    cons = Consolidator(
        store, dry_run=True,
        config=ConsolidationConfig(
            anomaly_subgraph_radius=2, anomaly_fanout_cap=8))
    # Spy on loader.load to record the (radius, fanout_cap) of every call.
    calls: list[tuple[str, object, object]] = []
    orig_load = cons.loader.load

    def spy_load(center_id, radius=None, fanout_cap=None):
        calls.append((center_id, radius, fanout_cap))
        return orig_load(center_id, radius=radius, fanout_cap=fanout_cap)
    cons.loader.load = spy_load

    rep = cons.run(limit=2)
    assert rep["subgraphs_scored"] == 2
    # Two calls per scored center: the radius-3 giant (cluster/link/ontology/
    # prune) + the bounded anomaly forward.
    assert len(calls) == 2 * 2
    # For each center: one default-radius call (radius None -> loader default 3,
    # cap None -> uncapped) and one bounded call (radius 2, cap 8).
    for center in ("ep_000001", "ep_000002"):
        center_calls = [c for c in calls if c[0] == center]
        assert len(center_calls) == 2
        caps = {c[2] for c in center_calls}
        radii = {c[1] for c in center_calls}
        # The bounded anomaly call passes radius=2 + cap=8 explicitly.
        assert (2, 8) in {(c[1], c[2]) for c in center_calls}
        # The other call is the giant (radius None / cap None -> loader defaults).
        assert None in caps and None in radii
    store.close()


def test_anomaly_step_degenerate_guard_reuses_giant(tmp_path):
    """When anomaly_subgraph_radius >= 3 AND the cap is None (uncapped), the
    bounded subgraph IS the radius-3 giant the other steps already loaded, so
    the anomaly step reuses it -- no second ``loader.load``. This is the
    degenerate guard that preserves the prior behavior when a caller configures
    the old bound (e.g. ``--anomaly-radius 3 --anomaly-fanout-cap 0``).
    """
    store = _store(tmp_path)
    _populate(store, n=4)
    cons = Consolidator(
        store, dry_run=True,
        config=ConsolidationConfig(
            anomaly_subgraph_radius=3, anomaly_fanout_cap=None))
    calls: list[tuple] = []
    orig_load = cons.loader.load

    def spy_load(center_id, radius=None, fanout_cap=None):
        calls.append((center_id, radius, fanout_cap))
        return orig_load(center_id, radius=radius, fanout_cap=fanout_cap)
    cons.loader.load = spy_load

    rep = cons.run(limit=2)
    assert rep["subgraphs_scored"] == 2
    # ONE call per center (the giant only) -- the guard skipped the second load.
    assert len(calls) == 2
    store.close()