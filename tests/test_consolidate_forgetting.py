"""Tests for the Phase 3b consolidation dream-pass forget step (consolidate.py).

``_step_forget`` runs after ``_step_prune`` in the per-center loop: it applies
``on_dream_state`` (decay-rate drift-back + utility fade) and RECOMPOSES
``utility_score`` from ``0.4*access_frequency + 0.6*structural_salience`` (the
object node's sigmoid'd SalienceHead output), soft-archiving edges below
``utility_prune_below``. ``_apply`` persists the sidecar updates + entity
structural salience. Dry-run records but never writes; 3a-prune takes
precedence (a pruned edge is not sidecarred by 3b).
"""

from __future__ import annotations

from src.config import ConsolidationConfig, config as _config
from src.memory.episode import Episode
from src.memory.store import HippocampalStore, _b2s
from src.gnn import Consolidator


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _populate(store, n=4):
    for i in range(1, n + 1):
        store.encode_episode(Episode(
            id=f"ep_00000{i}", timestamp="t", summary=f"s{i}", full_text=f"f{i}",
            entities=["Alice", "Bob"], topics=["db"],
        ))


def _has_sidecar(store, eid, pred, obj):
    """True if a sidecar was actually written (not the lazy default)."""
    from src.memory.edge_meta import edge_meta_key
    return bool(_b2s(store.db.get_sync(edge_meta_key(eid, pred, obj))))


# The untrained model's raw salience is low, so 3a's default prune threshold
# (0.15) would hard-prune ~every edge first -- and the R5 coexistence gate then
# skips them all for 3b. To OBSERVE the dream pass (not the R5 interaction) these
# tests disable 3a pruning (threshold below any raw logit). The dedicated R5 test
# below uses the opposite extreme.
_NO_PRUNE = ConsolidationConfig(prune_salience_below=-100.0)


# ── report shape ──
def test_forgetting_section_in_report(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    cons = Consolidator(store, dry_run=True, config=_NO_PRUNE)
    rep = cons.run(limit=3)
    assert "forgetting" in rep
    f = rep["forgetting"]
    for k in ("edges_seen", "boosted", "archived", "ltp"):
        assert k in f, k
    # forward association edges (has_entity/has_topic) are present in the
    # subgraphs, so the dream pass saw some.
    assert f["edges_seen"] >= 1
    store.close()


# ── dry run records but does not mutate ──
def test_dry_run_does_not_write_sidecars(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    # Pre-boost one Alice edge via the retrieval hook so it has a sidecar.
    from src.retrieval.graph_traversal import GraphTraversal
    GraphTraversal(store).retrieve({"entities": ["Alice"]}, signal="important")
    assert _has_sidecar(store, "ep_000001", "has_entity", "E:Alice")

    cons = Consolidator(store, dry_run=True, config=_NO_PRUNE)
    rep = cons.run(limit=3)
    assert rep["dry_run"] is True
    # boosted counts edges that had retrieval history (the Alice edges do)
    assert rep["forgetting"]["boosted"] >= 1
    # but the dream pass did NOT persist: the sidecar still carries the
    # retrieval-boost state only (access_count==1, utility_score at default 0.5)
    meta = store.get_edge_meta("ep_000001", "has_entity", "E:Alice")
    assert meta["access_count"] == 1
    assert meta["utility_score"] == 0.5  # dream pass recompose not persisted
    # no structural salience persisted in dry run
    assert _b2s(store.db.get_sync("content/entity/Alice/structural_salience")) == ""
    store.close()


# ── apply writes sidecars + structural salience ──
def test_apply_persists_forget_updates_and_salience(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    from src.retrieval.graph_traversal import GraphTraversal
    GraphTraversal(store).retrieve({"entities": ["Alice"]}, signal="important")

    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True, config=_NO_PRUNE)
    rep = cons.run(limit=3)
    assert "apply_skipped" not in rep

    # The Alice edge sidecar was recomposed by the dream pass and persisted.
    meta = store.get_edge_meta("ep_000001", "has_entity", "E:Alice")
    # access_count is preserved by the dream pass (it doesn't reset retrieval)
    assert meta["access_count"] == 1
    # utility_score recomposed into [0,1]
    assert 0.0 <= meta["utility_score"] <= 1.0
    # structural salience persisted for the entity, sigmoid'd into [0,1]
    raw = _b2s(store.db.get_sync("content/entity/Alice/structural_salience"))
    assert raw, "structural_salience not persisted"
    assert 0.0 <= float(raw) <= 1.0
    store.close()


# ── R5 coexistence: a 3a-pruned edge is NOT sidecarred by 3b ──
def test_pruned_edge_is_not_sidecarred(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    # Force 3a to prune EVERY edge: set the prune threshold above any possible
    # raw-logit salience so both endpoints are always below it.
    cfg = ConsolidationConfig(prune_salience_below=10.0)
    cons = Consolidator(store, dry_run=True, config=cfg)
    rep = cons.run(limit=3)
    assert len(rep["pruned"]) >= 1  # 3a did prune
    # 3b skips every pruned edge -> nothing recorded for the forget pass
    assert rep["forgetting"]["edges_seen"] == 0
    assert cons._forget_updates == []
    store.close()


# ── master gate: forgetting_enabled=False skips the dream pass ──
def test_forgetting_disabled_skips_dream_pass(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    saved = _config.forgetting_enabled
    _config.forgetting_enabled = False
    try:
        cons = Consolidator(store, dry_run=True)
        rep = cons.run(limit=3)
    finally:
        _config.forgetting_enabled = saved
    assert rep["forgetting"]["edges_seen"] == 0
    assert rep["forgetting"]["boosted"] == 0
    assert cons._forget_updates == []
    store.close()