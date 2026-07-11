"""Phase 3b A1: deep-archive tier tests.

The soft tier (``state='archived'``, in-place, excluded from default queries)
always ships. The deep tier physically removes edges soft-archived more than
``deep_archive_days`` ago: the live graph edge is deleted, a recoverable
``archive/edge/...`` JSON record is written (reusing the 3a hard-prune format),
and the orphaned sidecar + consumed ``content/archived_edge/`` index entry are
deleted.

The aging index (``content/archived_edge/{s}/{p}/{o}``) is the sweep's source of
truth: its VALUE stores the original ``(s,p,o)`` + ``archived_at`` so the sweep
recovers edge identity even when a key component was hashed by
``safe_edge_component``. These tests plant index entries directly to exercise
the sweep deterministically (the dream pass's soft-archive depends on the
untrained model's salience, which is not deterministic).
"""

from __future__ import annotations

from src.config import ConsolidationConfig, config as _config
from src.gnn import Consolidator
from src.gnn.semantic_memory import SemanticMemoryWriter
from src.memory.episode import Episode
from src.memory.edge_meta import (
    archived_edge_key,
    edge_meta_key,
    edge_meta_put_op,
    record_archived_edge_op,
    scan_archived_edges,
    update_edge_meta,
)
from src.memory.forgetting import default_meta
from src.memory.store import HippocampalStore, _b2s


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid="ep_001", entities=None):
    store.encode_episode(Episode(
        id=eid, timestamp="2026-07-01T10:00:00Z", summary=f"s {eid}",
        full_text=f"f {eid}", entities=entities or ["Alice"],
    ))


def _consolidator(store, **kw):
    dry_run = kw.pop("dry_run", False)
    allow = kw.pop("allow_untrained_apply", True)
    return Consolidator(store, dry_run=dry_run, allow_untrained_apply=allow,
                        config=ConsolidationConfig(**kw))


def _plant_archived(store, s, p, o, archived_at, state="archived"):
    """Plant a soft-archived edge: live graph edge + sidecar + aging index."""
    meta = default_meta()
    meta["state"] = state
    meta["archived_at"] = archived_at
    update_edge_meta(store, s, p, o, meta)
    store.db.batch_sync([record_archived_edge_op(s, p, o, archived_at)])


def _base_report():
    return {
        "abstracts": [], "edges_accepted": [], "pruned": [],
        "anomalies": [], "ontology_proposed": [],
        "forgetting": {"edges_seen": 0, "boosted": 0, "archived": [], "ltp": 0,
                       "reconsolidated": [], "ontology_deprecated": [],
                       "deep_archived": []},
    }


def _has_graph_edge(store, s, p, o):
    r = store.graph.query().vertex(s).out(p).execute_sync()
    try:
        return o in list(r.vertices)
    finally:
        r.close()


def _index_present(store, s, p, o):
    return bool(_b2s(store.db.get_sync(archived_edge_key(s, p, o))))


def _sidecar_present(store, s, p, o):
    return bool(_b2s(store.db.get_sync(edge_meta_key(s, p, o))))


# ── soft-archive writes the aging index (the _apply path) ──

def test_apply_writes_archived_edge_index_for_soft_archived(tmp_path):
    """``_apply`` persists sidecars; for state='archived' it also writes the
    ``content/archived_edge/`` index entry so the sweep can later age the edge."""
    store = _store(tmp_path)
    _encode(store)
    cons = _consolidator(store, deep_archive_days=365)
    s, p, o = "ep_001", "has_entity", "E:Alice"
    # A forget-update with state='archived' + a stamped archived_at.
    meta = default_meta()
    meta["state"] = "archived"
    meta["archived_at"] = "2026-01-01T00:00:00Z"
    cons._forget_updates = [(s, p, o, meta)]
    cons._apply(_base_report())
    # The sidecar carries the stamped archived_at.
    assert store.get_edge_meta(s, p, o)["archived_at"] == "2026-01-01T00:00:00Z"
    # The aging index entry exists and recovers the original (s,p,o).
    scanned = list(scan_archived_edges(store))
    assert (s, p, o, "2026-01-01T00:00:00Z") in scanned
    store.close()


def test_apply_writes_no_index_for_non_archived(tmp_path):
    """A current-edge sidecar persists but writes NO index entry (no aging)."""
    store = _store(tmp_path)
    _encode(store)
    cons = _consolidator(store, deep_archive_days=365)
    s, p, o = "ep_001", "has_entity", "E:Alice"
    meta = default_meta()  # state='current', archived_at=None
    cons._forget_updates = [(s, p, o, meta)]
    cons._apply(_base_report())
    assert list(scan_archived_edges(store)) == []
    store.close()


# ── the sweep: aged past threshold -> physically removed ──

def test_sweep_archives_edges_older_than_threshold(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store)
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2027-01-01T00:00:00Z")  # 365d after archive
    cons = _consolidator(store, deep_archive_days=365)
    s, p, o = "ep_001", "has_entity", "E:Alice"
    _plant_archived(store, s, p, o, "2026-01-01T00:00:00Z")
    assert _has_graph_edge(store, s, p, o)  # edge exists before

    report = _base_report()
    cons._step_deep_archive(report)
    assert cons._deep_archive_candidates, "aged edge not flagged"
    cons._apply(report)

    # The report records the deep-archive.
    da = report["forgetting"]["deep_archived"]
    assert len(da) == 1 and da[0]["subject"] == s and da[0]["age_days"] >= 365
    # The live graph edge is gone.
    assert not _has_graph_edge(store, s, p, o)
    # A recoverable archive/edge/... JSON record was written.
    ak = SemanticMemoryWriter._archive_key(s, p, o)
    rec = cons.writer.read_archived_edge(ak)
    assert rec is not None and rec["subject"] == s and rec["object"] == o
    assert rec["archived_at"] == "2026-01-01T00:00:00Z"  # original archive time
    # The orphaned sidecar + consumed index entry are deleted.
    assert not _sidecar_present(store, s, p, o)
    assert not _index_present(store, s, p, o)
    store.close()


def test_sweep_skips_edges_under_threshold(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store)
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2026-01-11T00:00:00Z")  # 10d after archive
    cons = _consolidator(store, deep_archive_days=365)
    s, p, o = "ep_001", "has_entity", "E:Alice"
    _plant_archived(store, s, p, o, "2026-01-01T00:00:00Z")

    report = _base_report()
    cons._step_deep_archive(report)
    assert cons._deep_archive_candidates == []
    cons._apply(report)  # no candidates -> no mutation
    assert _has_graph_edge(store, s, p, o)
    assert _index_present(store, s, p, o)
    assert _sidecar_present(store, s, p, o)
    store.close()


def test_sweep_skips_archived_at_none(tmp_path, monkeypatch):
    """Legacy index entries with no ``archived_at`` (pre-A1) can't be aged ->
    skipped, NOT retroactively deep-archived (conservative)."""
    store = _store(tmp_path)
    _encode(store)
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2030-01-01T00:00:00Z")
    cons = _consolidator(store, deep_archive_days=365)
    s, p, o = "ep_001", "has_entity", "E:Alice"
    # Plant the index entry directly with archived_at=None (legacy shape).
    store.db.batch_sync([record_archived_edge_op(s, p, o, None)])
    # Also plant a sidecar so it's a realistic archived edge.
    meta = default_meta()
    meta["state"] = "archived"
    meta["archived_at"] = None
    store.db.batch_sync([edge_meta_put_op(s, p, o, meta)])

    report = _base_report()
    cons._step_deep_archive(report)
    assert cons._deep_archive_candidates == []
    assert _has_graph_edge(store, s, p, o)  # untouched
    assert _index_present(store, s, p, o)
    store.close()


# ── dry run reports candidates but does not mutate ──

def test_dry_run_reports_but_does_not_mutate(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store)
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2027-01-01T00:00:00Z")
    s, p, o = "ep_001", "has_entity", "E:Alice"
    _plant_archived(store, s, p, o, "2026-01-01T00:00:00Z")

    cons = Consolidator(store, dry_run=True, allow_untrained_apply=True,
                        config=ConsolidationConfig(deep_archive_days=365))
    rep = cons.run(limit=1)
    assert rep["dry_run"] is True
    # The sweep ran and reported the candidate.
    assert len(rep["forgetting"]["deep_archived"]) >= 1
    found = any(e["subject"] == s for e in rep["forgetting"]["deep_archived"])
    assert found, "planted aged edge not reported in dry-run"
    # But nothing was mutated: edge, index, and sidecar all intact; no archive.
    assert _has_graph_edge(store, s, p, o)
    assert _index_present(store, s, p, o)
    assert _sidecar_present(store, s, p, o)
    ak = SemanticMemoryWriter._archive_key(s, p, o)
    assert cons.writer.read_archived_edge(ak) is None
    store.close()


# ── 0 -> None sentinel: disabled ──

def test_deep_archive_disabled_when_zero(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store)
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2030-01-01T00:00:00Z")
    s, p, o = "ep_001", "has_entity", "E:Alice"
    _plant_archived(store, s, p, o, "2026-01-01T00:00:00Z")
    # 0 -> None: the sweep is disabled.
    cons = _consolidator(store, deep_archive_days=0)

    report = _base_report()
    cons._step_deep_archive(report)
    assert cons._deep_archive_candidates == []
    assert report["forgetting"]["deep_archived"] == []
    assert _has_graph_edge(store, s, p, o)
    assert _index_present(store, s, p, o)
    store.close()


# ── master gate: forgetting_enabled=False skips the sweep ──

def test_forgetting_disabled_skips_deep_archive(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _encode(store)
    monkeypatch.setattr("src.gnn.consolidate._dream_now",
                         lambda: "2030-01-01T00:00:00Z")
    s, p, o = "ep_001", "has_entity", "E:Alice"
    _plant_archived(store, s, p, o, "2026-01-01T00:00:00Z")
    saved = _config.forgetting_enabled
    _config.forgetting_enabled = False
    try:
        cons = _consolidator(store, deep_archive_days=365)
        report = _base_report()
        cons._step_deep_archive(report)
        assert cons._deep_archive_candidates == []
        assert _has_graph_edge(store, s, p, o)
    finally:
        _config.forgetting_enabled = saved
    store.close()


# ── backward-compat: batch_update_edge_meta still writes sidecars ──

def test_batch_update_edge_meta_still_writes_sidecars(tmp_path):
    """The public 2-arg ``batch_update_edge_meta`` is unchanged (the index is
    only written by the consolidator's combined batch, not by this helper)."""
    from src.memory.edge_meta import batch_update_edge_meta
    store = _store(tmp_path)
    _encode(store)
    s, p, o = "ep_001", "has_entity", "E:Alice"
    meta = default_meta()
    meta["access_count"] = 7
    batch_update_edge_meta(store, [(s, p, o, meta)])
    out = store.get_edge_meta(s, p, o)
    assert out["access_count"] == 7
    # No index entry written by the helper.
    assert list(scan_archived_edges(store)) == []
    store.close()