"""Phase 3c: fact-level contradiction detection + Bonsai adjudication + tombstone.

Exercises the D2/D3/D4 path end-to-end through ``Consolidator._apply`` (crafted
reports, like ``test_consolidate_bonsai.py``): plant two ``(E:team, state, V)``
assertion edges WITH edge-sidecar provenance, push a ``contradictory_state``
anomaly, and let a ``FakeDecider`` adjudicate. The conservative ``fix`` +
``supersede_assertion`` path tombstones the old fact at the FACT level; a
re-load excludes the tombstoned edge so the detector goes quiet (D2
resolution). ``ask_user``/``dismiss`` are record-only. The D4 provenance fix
selects old/new from ``asserted_at``; the timestamp-heuristic fallback stays
for injector-planted (no-sidecar) edges (3b back-compat). Cold-start: no
assertions -> no flags, byte-identical.
"""

from __future__ import annotations

from src.config import ConsolidationConfig, config as master_config
from src.gnn import Consolidator
from src.gnn.anomaly_rules import enrich_subgraph, _detect_contradictory_state
from src.memory.episode import Episode
from src.memory.edge_meta import update_edge_meta, default_meta
from src.memory.store import HippocampalStore


# ── fakes / helpers ───────────────────────────────────────────────────────

class FakeDecider:
    """Scripted stand-in for ``BonsaiDecider`` (no HTTP). Only the
    ``decide_contradiction`` method is exercised here."""

    def __init__(self, contradiction=None):
        self._c = contradiction or {"decision": "ask_user", "action": "ask",
                                    "reasoning": "ambiguous"}
        self.contradiction_calls = 0

    def decide_contradiction(self, flag, retrieved_context):
        self.contradiction_calls += 1
        return dict(self._c)

    # The other decider methods are not called by the contradiction loop; stub
    # them so a Consolidator that pings them (it does not here) does not crash.
    def decide_anomaly(self, flag, ctx):
        return None

    def gist(self, source_episodes):
        return None

    def verify_typing(self, entity, cls, ctx):
        return None


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _plant_assertion(store, entity_name, value, asserted_by, asserted_at):
    """Plant an entity ``(E:name, state, value)`` edge WITH sidecar provenance
    (mirrors what the Phase 3c encoder writes via ``_assertion_edge_ops``)."""
    subj = f"E:{entity_name}"
    ops = store.graph.expand_triple(subj, "state", value)
    store.db.batch_sync(ops)
    meta = default_meta()
    meta["state"] = "current"
    meta["asserted_by"] = asserted_by
    meta["asserted_at"] = asserted_at
    update_edge_meta(store, subj, "state", value, meta)


def _make_report(node, score, **overrides) -> dict:
    rep = {
        "dry_run": False, "trained": True, "subgraphs_scored": 1,
        "abstracts": [], "edges_proposed": [], "edges_accepted": [],
        "edges_unverified": [],
        "anomalies": [{"node": node, "type": "contradictory_state",
                       "score": score, "evidence": "distinct states"}],
        "ontology_proposed": [], "pruned": [],
        "verifier_calls": 0, "verifier_accepted": 0,
        "abstracts_applied": [], "ontology_applied": [],
        "ontology_rejected": [], "identity_drift_decisions": [],
        "contradictions_resolved": [],
        "score_distributions": {"ontology": [0] * 100, "linkpred": [0] * 100,
                                "salience_endpoint": [0] * 100},
        "forgetting": {"edges_seen": 0, "boosted": 0, "archived": [], "ltp": 0,
                       "reconsolidated": [], "ontology_deprecated": [],
                       "deep_archived": []},
    }
    rep.update(overrides)
    return rep


def _cons(store, decider=None, **kw):
    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True,
                        decider=decider, config=ConsolidationConfig(**kw))
    cons._forget_updates = []
    cons._forget_node_salience = {}
    return cons


# ── adjudication: fix -> fact-level tombstone (D2) ──

def test_fix_supersede_assertion_tombstones_old_fact(tmp_path):
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    dec = FakeDecider(contradiction={"decision": "fix",
                                    "action": "supersede_assertion",
                                    "reasoning": "Postgres supersedes MySQL"})
    cons = _cons(store, decider=dec)
    rep = _make_report("E:team", score=0.9)
    cons._apply(rep)

    rec = rep["contradictions_resolved"]
    assert len(rec) == 1
    assert rec[0]["decision"] == "fix"
    assert rec[0]["applied"] is True
    # old = earliest asserted_at = MySQL; new = Postgres.
    assert rec[0]["old_value"] == "MySQL"
    assert rec[0]["new_value"] == "Postgres"
    assert rec[0]["asserted_by_old"] == "ep_old"
    assert rec[0]["asserted_by_new"] == "ep_new"
    # D2: the OLD fact is tombstoned at the fact level (edge NOT deleted).
    assert store.is_edge_current("E:team", "state", "MySQL") is False
    assert store.is_edge_current("E:team", "state", "Postgres") is True
    # The episode-level state is untouched (fact-level, not episode-level).
    store.close()


def test_tombstone_resolves_contradiction_detector_goes_quiet(tmp_path):
    """D2 resolution: after the tombstone, the anomaly subgraph load excludes
    the superseded edge -> ``_detect_contradictory_state`` no longer fires."""
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    def _detect():
        sub = {"nodes": [{"id": "E:team", "type": "node", "depth": 0}],
               "edges": []}
        enriched = enrich_subgraph(store, sub)
        return _detect_contradictory_state(enriched)

    # Before: two live values -> the detector fires.
    assert _detect(), "detector should fire on two live state values"
    # Adjudicate fix -> tombstone the old fact.
    dec = FakeDecider(contradiction={"decision": "fix",
                                    "action": "supersede_assertion",
                                    "reasoning": "newer wins"})
    cons = _cons(store, decider=dec)
    cons._apply(_make_report("E:team", score=0.9))
    # After: only one LIVE value -> detector quiet (the contradiction is
    # *resolved*, not just recorded).
    assert _detect() == [], "detector should be quiet after the tombstone"
    store.close()


# ── conservative adjudication: non-supersede fix -> ask_user ──

def test_fix_with_unknown_action_becomes_ask_user(tmp_path):
    """A ``fix`` whose action is NOT ``supersede_assertion`` is conservatively
    downgraded to ``ask_user`` (record-only, no mutation)."""
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    dec = FakeDecider(contradiction={"decision": "fix",
                                    "action": "merge_entities",
                                    "reasoning": "maybe same team"})
    cons = _cons(store, decider=dec)
    rep = _make_report("E:team", score=0.9)
    cons._apply(rep)

    rec = rep["contradictions_resolved"]
    assert rec[0]["decision"] == "ask_user"  # downgraded
    assert rec[0]["applied"] is False
    # No mutation: both facts still live.
    assert store.is_edge_current("E:team", "state", "MySQL") is True
    assert store.is_edge_current("E:team", "state", "Postgres") is True
    store.close()


def test_ask_user_is_record_only(tmp_path):
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    dec = FakeDecider(contradiction={"decision": "ask_user", "action": "ask",
                                    "reasoning": "ambiguous"})
    cons = _cons(store, decider=dec)
    rep = _make_report("E:team", score=0.9)
    cons._apply(rep)

    rec = rep["contradictions_resolved"]
    assert rec[0]["decision"] == "ask_user"
    assert rec[0]["applied"] is False
    assert store.is_edge_current("E:team", "state", "MySQL") is True
    store.close()


def test_dismiss_is_record_only(tmp_path):
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    dec = FakeDecider(contradiction={"decision": "dismiss", "action": "",
                                    "reasoning": "not a real conflict"})
    cons = _cons(store, decider=dec)
    rep = _make_report("E:team", score=0.9)
    cons._apply(rep)

    rec = rep["contradictions_resolved"]
    assert rec[0]["decision"] == "dismiss"
    assert rec[0]["applied"] is False
    store.close()


# ── low-confidence -> record-only (no mutation, no decider call) ──

def test_low_confidence_is_record_only(tmp_path):
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    dec = FakeDecider()
    cons = _cons(store, decider=dec, contradiction_resolve_threshold=0.8)
    rep = _make_report("E:team", score=0.5)  # below threshold
    cons._apply(rep)

    # Below the threshold: the loop skips -> no adjudication, no record entry.
    assert rep["contradictions_resolved"] == []
    assert dec.contradiction_calls == 0
    assert store.is_edge_current("E:team", "state", "MySQL") is True
    store.close()


# ── D4 provenance fix: old/new from asserted_at ──

def test_provenance_selects_old_new_by_asserted_at(tmp_path):
    """The gather context orders the conflicting values by ``asserted_at`` so
    the tombstone lands on the EARLIEST (old) fact regardless of insertion
    order."""
    store = _store(tmp_path)
    # Plant the NEWER value FIRST (insertion order != asserted_at order).
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")

    dec = FakeDecider(contradiction={"decision": "fix",
                                    "action": "supersede_assertion",
                                    "reasoning": "older superseded"})
    cons = _cons(store, decider=dec)
    rep = _make_report("E:team", score=0.9)
    cons._apply(rep)

    rec = rep["contradictions_resolved"][0]
    # old = earliest asserted_at = MySQL (even though planted second).
    assert rec["old_value"] == "MySQL"
    assert rec["new_value"] == "Postgres"
    assert store.is_edge_current("E:team", "state", "MySQL") is False
    assert store.is_edge_current("E:team", "state", "Postgres") is True
    store.close()


# ── D4 fallback: injector-planted edges (no sidecar) -> timestamp heuristic ──

def test_resolver_fallback_for_no_sidecar_edges(tmp_path):
    """``_resolve_contradictory_state`` keeps the 3b timestamp-heuristic
    fallback for edges with NO sidecar provenance (injector-planted). The 3b
    no-decider episode-supersede path is unchanged."""
    store = _store(tmp_path)

    def _encode(eid, ts):
        store.encode_episode(Episode(id=eid, timestamp=ts,
                                    summary=f"s {eid}", full_text=f"f {eid}",
                                    entities=["team"]))
    _encode("ep_old", "2026-07-01T10:00:00Z")
    _encode("ep_new", "2026-07-05T10:00:00Z")
    # Plant state edges with NO sidecar (mirrors the anomaly injector).
    for v in ("MySQL", "Postgres"):
        store.db.batch_sync(store.graph.expand_triple("E:team", "state", v))

    cons = _cons(store)  # no decider -> the 3b path
    pair = cons._resolve_contradictory_state("E:team")
    assert pair == ("ep_old", "ep_new")  # timestamp heuristic, unchanged
    store.close()


# ── D4: episode-provenance path returns the asserting episodes ──

def test_resolver_uses_episode_provenance_when_present(tmp_path):
    """When state edges carry ``asserted_by`` = episode id + ``asserted_at``,
    the resolver returns that pair directly (no timestamp heuristic)."""
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    cons = _cons(store)
    pair = cons._resolve_contradictory_state("E:team")
    assert pair == ("ep_old", "ep_new")
    store.close()


# ── cold-start: no decider -> 3b episode-supersede path, byte-identical ──

def test_no_decider_keeps_3b_episode_supersede_path(tmp_path):
    """Without a decider, the 3b episode-level supersede path runs (the new
    fact-tombstone loop is skipped). This is the regression guard."""
    store = _store(tmp_path)

    def _encode(eid, ts):
        store.encode_episode(Episode(id=eid, timestamp=ts,
                                    summary=f"s {eid}", full_text=f"f {eid}",
                                    entities=["Alice"]))
    _encode("ep_old", "2026-07-01T10:00:00Z")
    _encode("ep_new", "2026-07-05T10:00:00Z")
    for v in ("alive", "dead"):
        store.db.batch_sync(store.graph.expand_triple("E:Alice", "state", v))

    cons = _cons(store)  # no decider
    rep = _make_report("E:Alice", score=0.9)
    cons._apply(rep)

    # 3b path: episode-level supersede (NOT fact-level tombstone).
    assert rep["forgetting"]["reconsolidated"] == [
        {"entity": "E:Alice", "old": "ep_old", "new": "ep_new"}]
    assert store.episode_state("ep_old") == "superseded"
    # The new fact-tombstone loop did not run.
    assert rep["contradictions_resolved"] == []
    store.close()


# ── cold-start: decider down -> honest record-only ──

def test_decider_returns_none_is_record_only(tmp_path):
    """A decider that returns ``None`` (Bonsai down / parse fail) is honest
    record-only -- no fabricated decision, no mutation."""
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    class NoneDecider:
        def decide_contradiction(self, flag, ctx):
            return None

        def decide_anomaly(self, flag, ctx):
            return None

        def gist(self, src):
            return None

        def verify_typing(self, e, c, ctx):
            return None

    cons = _cons(store, decider=NoneDecider())
    rep = _make_report("E:team", score=0.9)
    cons._apply(rep)

    rec = rep["contradictions_resolved"]
    assert len(rec) == 1
    assert rec[0]["decision"] is None
    assert rec[0]["applied"] is False
    assert store.is_edge_current("E:team", "state", "MySQL") is True
    store.close()


# ── cold-start: forgetting off -> no mutation ──

def test_forgetting_disabled_skips_contradiction_loop(tmp_path):
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    dec = FakeDecider(contradiction={"decision": "fix",
                                    "action": "supersede_assertion",
                                    "reasoning": "x"})
    saved = master_config.forgetting_enabled
    master_config.forgetting_enabled = False
    try:
        cons = _cons(store, decider=dec)
        rep = _make_report("E:team", score=0.9)
        cons._apply(rep)
        # Loop gated on forgetting_enabled -> skipped entirely.
        assert rep["contradictions_resolved"] == []
        assert store.is_edge_current("E:team", "state", "MySQL") is True
    finally:
        master_config.forgetting_enabled = saved
    store.close()


# ── context gather: state_values carry provenance ──

def test_gather_entity_context_carries_state_provenance(tmp_path):
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")

    cons = _cons(store)
    ctx = cons._gather_entity_context("E:team")
    values = {v["value"]: v for v in ctx["state_values"]}
    assert set(values) == {"MySQL", "Postgres"}
    assert values["MySQL"]["asserted_by"] == "ep_old"
    assert values["MySQL"]["asserted_at"] == "2026-07-01T10:00:00Z"
    assert values["Postgres"]["asserted_by"] == "ep_new"
    store.close()


def test_gather_context_excludes_tombstoned_edges(tmp_path):
    """A tombstoned (superseded) edge is NOT in the live ``state_values`` set
    (D2) -- the detector sees only current facts."""
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")
    # Tombstone MySQL manually (as a prior adjudication would).
    store.supersede_assertion("E:team", "MySQL", "ep_new",
                               "2026-07-10T00:00:00Z")

    cons = _cons(store)
    ctx = cons._gather_entity_context("E:team")
    values = {v["value"] for v in ctx["state_values"]}
    assert values == {"Postgres"}  # MySQL dropped (tombstoned)
    store.close()


# ── Phase 3c Sec 7: deterministic non-conflict guards ──
#
# The 8B (and 27B) decider capacity-boundedly rubber-stamp complementary-
# temporal pairs (jan-status green / jul-status red) into fix+supersede_assertion
# -- a silent false tombstone. Two correct-by-construction guards short-circuit
# decide_contradiction BEFORE the HTTP call: equal values -> dismiss (defense-in-
# depth; the detector only flags DISTINCT values so this is a safety net) and
# both-sources-month-named -> ask_user (non-mutating; stops the false tombstone).
# _gather_entity_context resolves each assertion's asserted_by doc/section id
# back to its source_path so the month-name signal is available in PRODUCTION
# (not just the eval harness, which passes source_path directly as asserted_by).


def _plant_doc(store, doc_id, source_path):
    """Plant the minimal hot-store doc metadata so ``document_source_path``
    resolves (the existence sentinel is ``source_type``; see ``get_document``)."""
    store.db.put_sync(f"content/doc/{doc_id}/source_type", "text")
    store.db.put_sync(f"content/doc/{doc_id}/source_path", source_path)


def test_gather_context_resolves_source_path_for_doc_provenance(tmp_path):
    """A state_value whose ``asserted_by`` is a doc/section id carries the
    resolved ``source_path`` so the complementary-temporal guard can see the
    month-named filename. Episode-id provenance -> ``source_path`` is None
    (guard falls through to the LLM)."""
    store = _store(tmp_path)
    _plant_doc(store, "doc_000001", "docs/jan-status.md")
    _plant_doc(store, "doc_000002", "docs/jul-status.md")
    # Section-id provenance: ``{doc_id}_sec_{NNN}`` -> strip ``_sec_`` tail.
    _plant_assertion(store, "dep", "green", "doc_000001_sec_002",
                     "2026-07-01T10:00:00Z")
    # Doc-id provenance (doc-level assertion): resolves directly.
    _plant_assertion(store, "dep", "red", "doc_000002",
                     "2026-07-05T10:00:00Z")

    cons = _cons(store)
    ctx = cons._gather_entity_context("E:dep")
    by_val = {v["value"]: v for v in ctx["state_values"]}
    assert by_val["green"]["source_path"] == "docs/jan-status.md"  # section id
    assert by_val["red"]["source_path"] == "docs/jul-status.md"     # doc id
    store.close()


def test_gather_context_source_path_none_for_episode_provenance(tmp_path):
    """Episode-id (``ep_...``) provenance carries no source_path -> guard falls
    through to the LLM (the existing 3b/3c episode-provenance path unchanged)."""
    store = _store(tmp_path)
    _plant_assertion(store, "team", "MySQL", "ep_old",
                     "2026-07-01T10:00:00Z")
    _plant_assertion(store, "team", "Postgres", "ep_new",
                     "2026-07-05T10:00:00Z")
    cons = _cons(store)
    ctx = cons._gather_entity_context("E:team")
    for v in ctx["state_values"]:
        assert v["source_path"] is None
    store.close()


def test_decide_contradiction_complementary_temporal_guard_no_http(tmp_path):
    """Two DIFFERENT values asserted by month-named point-in-time records ->
    ask_user, returned by the deterministic guard BEFORE any HTTP call (no
    server needed). This is the N14 false-tombstone fix."""
    from src.gnn.bonsai_decider import BonsaiDecider
    dec = BonsaiDecider()  # lazy HTTP -- never reached (guard short-circuits)
    state_values = [
        {"value": "green", "source_path": "docs/jan-status.md",
         "asserted_by": "doc_000001_sec_002", "asserted_at": "2026-07-01"},
        {"value": "red", "source_path": "docs/jul-status.md",
         "asserted_by": "doc_000002", "asserted_at": "2026-07-05"},
    ]
    flag = {"node": "E:dep", "type": "contradictory_state",
            "evidence": state_values}
    out = dec.decide_contradiction(flag, {"state_values": state_values})
    assert out is not None
    assert out["decision"] == "ask_user"
    assert out["action"] == "no_action"
    assert "complementary temporal" in out["reasoning"]


def test_decide_contradiction_equal_values_guard_no_http(tmp_path):
    """All-equal live values -> dismiss (defense-in-depth; the production
    detector only flags DISTINCT values, but a direct caller / future path
    must still not false-fix an agreeing pair)."""
    from src.gnn.bonsai_decider import BonsaiDecider
    dec = BonsaiDecider()
    state_values = [
        {"value": "Postgres", "asserted_by": "docs/plan-a.md",
         "asserted_at": "2026-07-01"},
        {"value": "Postgres", "asserted_by": "docs/plan-b.md",
         "asserted_at": "2026-07-05"},
    ]
    flag = {"node": "E:db", "type": "contradictory_state",
            "evidence": state_values}
    out = dec.decide_contradiction(flag, {"state_values": state_values})
    assert out is not None
    assert out["decision"] == "dismiss"
    assert out["action"] == "no_action"


def test_decide_contradiction_real_conflict_bypasses_guard(tmp_path):
    """A genuine same-entity value change (different values, NON-month-named
    version-suffixed decision docs) bypasses both guards so the fine-tuned
    adapter adjudicates over HTTP. Verifies the guard never false-dismisses a
    real conflict: ``_post_json`` IS reached (the LLM path runs)."""
    from src.gnn.bonsai_decider import BonsaiDecider
    dec = BonsaiDecider()
    # Monkeypatch the HTTP layer to PROVE the guard did not short-circuit: if
    # the guard fired, _post_json would never be called and the recorded prompt
    # would stay None. A real conflict must reach the LLM path.
    seen_prompt = {"p": None}

    def fake_post_json(prompt):
        seen_prompt["p"] = prompt
        return None  # simulate a miss -> decide_contradiction returns None

    dec._post_json = fake_post_json
    state_values = [
        {"value": "MySQL", "source_path": "docs/db-pick-v1.md",
         "asserted_by": "doc_000003", "asserted_at": "2026-07-01"},
        {"value": "Postgres", "source_path": "docs/db-pick-v2.md",
         "asserted_by": "doc_000004", "asserted_at": "2026-07-05"},
    ]
    flag = {"node": "E:db", "type": "contradictory_state",
            "evidence": state_values}
    out = dec.decide_contradiction(flag, {"state_values": state_values})
    assert out is None           # HTTP miss -> None (no fabricated decision)
    assert seen_prompt["p"] is not None  # the LLM path WAS reached (guard did not fire)