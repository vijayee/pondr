"""Phase 3b step 9: ontology decay tests.

Decay targets DISCOVERED classes (runtime-invented labels promoted via Bonsai --
a deferred path) whose ``last_seen`` is older than ``ontology_decay_days``. Seed
classes are NEVER eligible (the seed writes only ``subClassOf`` graph triples,
no ``content/class/`` entry), so decay is a no-op on the seed-only ontology
today; these tests plant discovered classes to exercise the mechanism.

``last_seen`` is stamped in ``_apply`` from ``report["ontology_proposed"]`` (the
class-use signal). Entity->parent reassignment is a documented no-op (no
entity->class typing edges exist yet).
"""

from __future__ import annotations

import pytest

from src.config import ConsolidationConfig, config as _config
from src.gnn import Consolidator
from src.memory.episode import Episode
from src.memory.store import HippocampalStore


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid, entities=None):
    store.encode_episode(Episode(
        id=eid, timestamp="2026-07-01T10:00:00Z", summary=f"s {eid}",
        full_text=f"f {eid}", entities=entities or [],
    ))


def _consolidator(store, **kw):
    return Consolidator(store, dry_run=False, allow_untrained_apply=True,
                        config=ConsolidationConfig(**kw))


def _apply_with(cons, report):
    cons._forget_updates = []
    cons._forget_node_salience = {}
    cons._apply(report)


def _base_report(ontology_proposed=None, anomalies=None):
    return {
        "abstracts": [], "edges_accepted": [], "pruned": [],
        "anomalies": anomalies or [],
        "ontology_proposed": ontology_proposed or [],
        "forgetting": {"edges_seen": 0, "boosted": 0, "archived": [], "ltp": 0,
                       "reconsolidated": [], "ontology_deprecated": []},
    }


# ── last_seen stamping ──

def test_apply_stamps_last_seen_for_proposed_classes(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    cons = _consolidator(store)
    report = _base_report(ontology_proposed=[
        {"entity": "E:Alice", "class": "Person", "confidence": 0.95},
        {"entity": "E:Bob", "class": "Database", "confidence": 0.9},
    ])
    _apply_with(cons, report)
    assert store.class_last_seen("Person") is not None
    assert store.class_last_seen("Database") is not None
    store.close()


def test_dry_run_does_not_stamp_last_seen(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    cons = Consolidator(store, dry_run=True, allow_untrained_apply=True)
    # Run the full pass (dry-run); ontology_proposed may be empty (untrained),
    # but either way last_seen must NOT be written (dry run = no mutation).
    rep = cons.run(limit=1)
    assert rep["dry_run"] is True
    # No class should have a last_seen entry after a dry run.
    assert store.scan_classes() == [] or all(
        store.class_last_seen(c) is None for c in store.scan_classes())
    store.close()


# ── decay ──

def test_decay_deprecates_stale_discovered_class(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    # Plant a discovered class last seen 60 days ago (well past the 30-day
    # threshold). Fix the dream clock so the test is deterministic.
    store.mark_class_discovered("OldFeature")
    store.persist_class_last_seen("OldFeature", "2026-05-01T00:00:00Z")
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2026-07-01T00:00:00Z")

    cons = _consolidator(store, ontology_decay_days=30)
    rep = cons.run(limit=1)
    assert "apply_skipped" not in rep
    assert rep["forgetting"]["ontology_deprecated"], "stale class not flagged"
    entry = rep["forgetting"]["ontology_deprecated"][0]
    assert entry["class"] == "OldFeature"
    assert store.class_state("OldFeature") == "deprecated"
    store.close()


def test_decay_skips_recently_seen_discovered_class(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    store.mark_class_discovered("FreshFeature")
    store.persist_class_last_seen("FreshFeature", "2026-06-28T00:00:00Z")  # 3d ago
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2026-07-01T00:00:00Z")

    cons = _consolidator(store, ontology_decay_days=30)
    rep = cons.run(limit=1)
    assert rep["forgetting"]["ontology_deprecated"] == []
    assert store.class_state("FreshFeature") == "current"
    store.close()


def test_decay_skips_class_without_discovered_marker(tmp_path, monkeypatch):
    """A class touched by a proposal (last_seen only) but NOT discovered is
    a seed-class proxy -- decay must leave it alone."""
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    # last_seen only, no discovered marker -> not decay-eligible.
    store.persist_class_last_seen("Person", "2020-01-01T00:00:00Z")
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2026-07-01T00:00:00Z")

    cons = _consolidator(store, ontology_decay_days=30)
    rep = cons.run(limit=1)
    assert rep["forgetting"]["ontology_deprecated"] == []
    store.close()


def test_decay_skips_already_deprecated_class(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    store.mark_class_discovered("OldFeature")
    store.persist_class_last_seen("OldFeature", "2020-01-01T00:00:00Z")
    store.set_class_state("OldFeature", "deprecated")  # already deprecated
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2026-07-01T00:00:00Z")

    cons = _consolidator(store, ontology_decay_days=30)
    rep = cons.run(limit=1)
    # Already-deprecated -> not re-flagged.
    assert rep["forgetting"]["ontology_deprecated"] == []
    store.close()


def test_decay_skips_never_seen_discovered_class(tmp_path, monkeypatch):
    """Discovered but never seen -> don't deprecate on a cold start."""
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    store.mark_class_discovered("NewFeature")  # no last_seen
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2026-07-01T00:00:00Z")

    cons = _consolidator(store, ontology_decay_days=30)
    rep = cons.run(limit=1)
    assert rep["forgetting"]["ontology_deprecated"] == []
    store.close()


def test_seed_classes_not_scanned(tmp_path):
    """Seed classes have no content/class entry -> never appear in scan_classes."""
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    # The seed ontology is loaded; "Person" is a seed class but has no
    # content/class/ entry, so scan_classes must not include it.
    assert "Person" not in store.scan_classes()
    store.close()


def test_forgetting_disabled_skips_ontology_decay(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store, "ep_001", entities=["Alice"])
    store.mark_class_discovered("OldFeature")
    store.persist_class_last_seen("OldFeature", "2020-01-01T00:00:00Z")
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2026-07-01T00:00:00Z")

    saved = _config.forgetting_enabled
    _config.forgetting_enabled = False
    try:
        cons = _consolidator(store, ontology_decay_days=30)
        rep = cons.run(limit=1)
    finally:
        _config.forgetting_enabled = saved
    assert rep["forgetting"]["ontology_deprecated"] == []
    assert store.class_state("OldFeature") == "current"
    store.close()


def test_reassign_is_noop_without_typing_edges(tmp_path):
    """Entity->parent reassignment is a documented no-op (no typing edges)."""
    store = _store(tmp_path)
    cons = _consolidator(store)
    n = cons._reassign_entities_from_deprecated_class("OldFeature", "Feature")
    assert n == 0
    store.close()