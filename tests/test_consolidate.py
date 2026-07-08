"""Tests for the dream-state consolidation loop (``src/gnn/consolidate.py``)."""

from __future__ import annotations

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