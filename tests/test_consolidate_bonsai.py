"""Tests for the Bonsai-in-consolidation wiring (``consolidate.py``).

A ``FakeDecider`` + a stub embedder (no live Bonsai server, no model download)
exercise the three decider actions through ``Consolidator._apply`` directly
(crafted reports) so the assertions are deterministic, not at the mercy of an
untrained GNN's scores. The cold-start paths (no decider / disabled) are
byte-identical regression guards.
"""

from __future__ import annotations

from src.config import ConsolidationConfig, config as master_config
from src.gnn import Consolidator
from src.memory.episode import Episode
from src.memory.store import HippocampalStore


# ── fakes ──────────────────────────────────────────────────────────────────

class FakeDecider:
    """A scripted stand-in for ``BonsaiDecider`` (no HTTP)."""

    def __init__(self, gist_text="A synthesized gist.",
                 typing=None, anomaly=None):
        self._gist_text = gist_text
        self._typing = typing or {"accept": True, "new_class": None,
                                  "parent": None, "reasoning": "ok"}
        self._anomaly = anomaly or {"decision": "ask_user", "action": "ask",
                                    "reasoning": "ambiguous"}
        self.gist_calls = 0
        self.typing_calls = 0
        self.anomaly_calls = 0

    def gist(self, source_episodes):
        self.gist_calls += 1
        return self._gist_text

    def verify_typing(self, entity, candidate_class, retrieved_context):
        self.typing_calls += 1
        return dict(self._typing)

    def decide_anomaly(self, flag, retrieved_context):
        self.anomaly_calls += 1
        return dict(self._anomaly)


class StubEmbedder:
    """Returns a fixed 384-dim vector (matches the store's vector layer)."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts):
        return [[0.01] * self.dim for _ in texts]


# ── helpers ────────────────────────────────────────────────────────────────

def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _populate(store, n=4):
    for i in range(1, n + 1):
        store.encode_episode(Episode(
            id=f"ep_00000{i}", timestamp=f"2026-01-0{i}",
            summary=f"summary {i}", full_text=f"full text {i}",
            entities=["Alice", "Bob"], topics=["db"],
        ))


def _make_report(**overrides) -> dict:
    """A minimal valid report shell (the keys ``_apply`` reads + writes)."""
    rep = {
        "dry_run": False, "trained": True, "subgraphs_scored": 1,
        "abstracts": [], "edges_proposed": [], "edges_accepted": [],
        "edges_unverified": [], "anomalies": [], "ontology_proposed": [],
        "pruned": [], "verifier_calls": 0, "verifier_accepted": 0,
        "abstracts_applied": [], "ontology_applied": [],
        "ontology_rejected": [], "identity_drift_decisions": [],
        "score_distributions": {"ontology": [0] * 100, "linkpred": [0] * 100,
                                "salience_endpoint": [0] * 100},
        "forgetting": {"edges_seen": 0, "boosted": 0, "archived": [], "ltp": 0,
                       "reconsolidated": [], "ontology_deprecated": [],
                       "deep_archived": []},
    }
    rep.update(overrides)
    return rep


def _cons(store, decider=None, embedder=None, **kw):
    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True,
                        decider=decider, **kw)
    if embedder is not None:
        cons._embedder_obj = embedder
    return cons


# ── abstracts ─────────────────────────────────────────────────────────────

def test_abstract_gist_embeds_and_indexes(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(gist_text="the real gist")
    cons = _cons(store, decider=dec, embedder=StubEmbedder())
    rep = _make_report(abstracts=[{"episodes": ["ep_000001", "ep_000002"]}])
    cons._apply(rep)
    assert dec.gist_calls == 1
    assert len(rep["abstracts_applied"]) == 1
    mid = rep["abstracts_applied"][0]["mid"]
    assert rep["abstracts_applied"][0]["gist"] == "the real gist"
    # The M-node was written with the gist as summary + text.
    summary = store.db.get_sync(f"content/mem/{mid}/summary")
    assert summary == b"the real gist" or summary == "the real gist"
    # An embedding was stored (the gist path) + the M-node is a memory id.
    assert mid.startswith("M:")
    assert mid in store.default_memory_ids()
    store.close()


def test_abstract_no_decider_keeps_placeholder(tmp_path):
    """Cold-start: no decider -> the placeholder abstract, byte-identical to
    today (no gist, no embedding, no abstracts_applied entry)."""
    store = _store(tmp_path)
    _populate(store)
    cons = _cons(store, decider=None)  # no embedder either -> lazy not touched
    rep = _make_report(abstracts=[{"episodes": ["ep_000001", "ep_000002"]}])
    cons._apply(rep)
    assert rep["abstracts_applied"] == []
    mems = store.default_memory_ids()
    assert len(mems) == 1
    mid = mems[0]
    summary = store.db.get_sync(f"content/mem/{mid}/summary")
    s = summary.decode() if isinstance(summary, bytes) else summary
    assert s.startswith("Abstract of")
    # No embedding key (the placeholder path writes none).
    assert store.db.get_sync(f"content/mem/{mid}/embedding") is None
    store.close()


def test_abstract_decider_returns_none_falls_back_to_placeholder(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(gist_text=None)  # Bonsai failed to produce a gist
    cons = _cons(store, decider=dec, embedder=StubEmbedder())
    rep = _make_report(abstracts=[{"episodes": ["ep_000001"]}])
    cons._apply(rep)
    assert rep["abstracts_applied"] == []
    mems = store.default_memory_ids()
    s = store.db.get_sync(f"content/mem/{mems[0]}/summary")
    s = s.decode() if isinstance(s, bytes) else s
    assert s.startswith("Abstract of")
    store.close()


def test_abstract_bonsai_disabled_is_record_only(tmp_path):
    """``bonsai_decider_enabled=False`` disables the decider EVEN when wired
    (the --no-bonsai A/B escape hatch) -> placeholder abstracts."""
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(gist_text="should not be used")
    cons = _cons(store, decider=dec, embedder=StubEmbedder(),
                 config=ConsolidationConfig(bonsai_decider_enabled=False))
    rep = _make_report(abstracts=[{"episodes": ["ep_000001"]}])
    cons._apply(rep)
    assert dec.gist_calls == 0
    assert rep["abstracts_applied"] == []
    store.close()


# ── ontology promotion ────────────────────────────────────────────────────

def test_ontology_accept_writes_instanceof(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(typing={"accept": True, "new_class": None,
                              "parent": None, "reasoning": "yes"})
    cons = _cons(store, decider=dec)
    rep = _make_report(ontology_proposed=[
        {"entity": "E:Alice", "class": "Person", "confidence": 0.95}])
    cons._apply(rep)
    assert len(rep["ontology_applied"]) == 1
    # The instanceOf edge was written.
    r = store.graph.query().vertex("E:Alice").out("instanceOf").execute_sync()
    try:
        classes = list(r.vertices)
    finally:
        r.close()
    assert "Person" in classes
    store.close()


def test_ontology_new_class_valid_parent(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(typing={"accept": True, "new_class": "DBEngineer",
                              "parent": "Person", "reasoning": "narrower"})
    cons = _cons(store, decider=dec)
    rep = _make_report(ontology_proposed=[
        {"entity": "E:Alice", "class": "Person", "confidence": 0.9}])
    cons._apply(rep)
    applied = rep["ontology_applied"]
    assert len(applied) == 1 and applied[0]["new_class"] == "DBEngineer"
    # The class was created (discovered marker + subClassOf Person).
    assert store.is_class_discovered("DBEngineer")
    assert cons._class_exists("DBEngineer")
    # The entity is typed to the NEW class.
    r = store.graph.query().vertex("E:Alice").out("instanceOf").execute_sync()
    try:
        assert "DBEngineer" in list(r.vertices)
    finally:
        r.close()
    store.close()


def test_ontology_new_class_invalid_parent_record_only(tmp_path):
    """A new class whose proposed parent does NOT exist in the seed ontology
    is NOT created (never orphan) -> record-only rejection."""
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(typing={"accept": True, "new_class": "DBEngineer",
                              "parent": "Nonexistent", "reasoning": "bad"})
    cons = _cons(store, decider=dec)
    rep = _make_report(ontology_proposed=[
        {"entity": "E:Alice", "class": "Person", "confidence": 0.9}])
    cons._apply(rep)
    assert rep["ontology_applied"] == []
    assert len(rep["ontology_rejected"]) == 1
    assert not store.is_class_discovered("DBEngineer")
    # No instanceOf edge written.
    r = store.graph.query().vertex("E:Alice").out("instanceOf").execute_sync()
    try:
        assert list(r.vertices) == []
    finally:
        r.close()
    store.close()


def test_ontology_reject_record_only(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(typing={"accept": False, "new_class": None,
                              "parent": None, "reasoning": "no"})
    cons = _cons(store, decider=dec)
    rep = _make_report(ontology_proposed=[
        {"entity": "E:Alice", "class": "Person", "confidence": 0.9}])
    cons._apply(rep)
    assert rep["ontology_applied"] == []
    assert len(rep["ontology_rejected"]) == 1
    r = store.graph.query().vertex("E:Alice").out("instanceOf").execute_sync()
    try:
        assert list(r.vertices) == []
    finally:
        r.close()
    store.close()


def test_ontology_threshold_skips_low_confidence(tmp_path):
    """Proposals below ``ontology_bonsai_threshold`` are not sent to Bonsai."""
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider()
    cons = _cons(store, decider=dec,
                 config=ConsolidationConfig(ontology_bonsai_threshold=0.95))
    rep = _make_report(ontology_proposed=[
        {"entity": "E:Alice", "class": "Person", "confidence": 0.9}])
    cons._apply(rep)
    assert dec.typing_calls == 0
    assert rep["ontology_applied"] == []
    store.close()


# ── identity_drift anomaly ────────────────────────────────────────────────

def test_identity_drift_ask_user_record_only(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(anomaly={"decision": "ask_user", "action": "ask",
                               "reasoning": "ambiguous"})
    cons = _cons(store, decider=dec)
    rep = _make_report(anomalies=[
        {"node": "E:Alice", "type": "identity_drift", "score": 1.0}])
    cons._apply(rep)
    assert len(rep["identity_drift_decisions"]) == 1
    d = rep["identity_drift_decisions"][0]
    assert d["decision"] == "ask_user"
    assert d["applied"] is False
    store.close()


def test_identity_drift_fix_conservative_to_ask_user(tmp_path):
    """A ``fix`` whose action is NOT supersede_episode is conservatively
    treated as ``ask_user`` (no silent graph mutation from an over-firing
    flag)."""
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider(anomaly={"decision": "fix", "action": "split_entity",
                               "reasoning": "drift"})
    cons = _cons(store, decider=dec)
    rep = _make_report(anomalies=[
        {"node": "E:Alice", "type": "identity_drift", "score": 1.0}])
    cons._apply(rep)
    d = rep["identity_drift_decisions"][0]
    assert d["decision"] == "ask_user"
    assert d["applied"] is False
    store.close()


def test_identity_drift_fix_supersede_applied(tmp_path):
    """The one known-safe fix: ``fix`` + ``supersede_episode`` with a real
    (old, new) pair derived by the resolver -> applied=True."""
    store = _store(tmp_path)
    # Two episodes mentioning Alice with distinct timestamps.
    store.encode_episode(Episode(id="ep_000001", timestamp="2026-01-01",
                                 summary="old state", full_text="t1",
                                 entities=["Alice"], topics=["db"]))
    store.encode_episode(Episode(id="ep_000002", timestamp="2026-01-02",
                                 summary="new state", full_text="t2",
                                 entities=["Alice"], topics=["db"]))
    # Plant two distinct live state values on E:Alice (the resolver's gate).
    store.db.batch_sync(
        store.graph.expand_triple("E:Alice", "state", "active")
        + store.graph.expand_triple("E:Alice", "state", "retired"))
    dec = FakeDecider(anomaly={"decision": "fix", "action": "supersede_episode",
                               "reasoning": "drift"})
    cons = _cons(store, decider=dec)
    rep = _make_report(anomalies=[
        {"node": "E:Alice", "type": "identity_drift", "score": 1.0}])
    cons._apply(rep)
    d = rep["identity_drift_decisions"][0]
    assert d["decision"] == "fix"
    assert d["applied"] is True
    # The old episode was superseded (state="superseded").
    st = store.db.get_sync("content/ep/ep_000001/state")
    st = st.decode() if isinstance(st, bytes) else st
    assert st == "superseded"
    store.close()


def test_identity_drift_no_decider_records_nothing(tmp_path):
    """Cold-start: with no decider the identity_drift loop is skipped (the
    flag itself never ran either), so no decision is recorded."""
    store = _store(tmp_path)
    _populate(store)
    cons = _cons(store, decider=None)
    rep = _make_report(anomalies=[
        {"node": "E:Alice", "type": "identity_drift", "score": 1.0}])
    cons._apply(rep)
    assert rep["identity_drift_decisions"] == []
    store.close()


def test_identity_drift_bonsai_disabled_records_nothing(tmp_path):
    store = _store(tmp_path)
    _populate(store)
    dec = FakeDecider()
    cons = _cons(store, decider=dec,
                 config=ConsolidationConfig(bonsai_decider_enabled=False))
    rep = _make_report(anomalies=[
        {"node": "E:Alice", "type": "identity_drift", "score": 1.0}])
    cons._apply(rep)
    assert dec.anomaly_calls == 0
    assert rep["identity_drift_decisions"] == []
    store.close()


# ── create_class helper (store-level) ──────────────────────────────────────

def test_create_class_writes_subclassof_and_discovered(tmp_path):
    store = _store(tmp_path)
    store.create_class("DBEngineer", "Person", "2026-01-01T00:00:00Z")
    assert store.is_class_discovered("DBEngineer")
    assert store.class_last_seen("DBEngineer") == "2026-01-01T00:00:00Z"
    # subClassOf edge to the parent.
    r = store.graph.query().vertex("DBEngineer").out("subClassOf").execute_sync()
    try:
        assert "Person" in list(r.vertices)
    finally:
        r.close()
    store.close()


def test_create_class_rejects_slash_in_name(tmp_path):
    store = _store(tmp_path)
    try:
        store.create_class("a/b", "Person", "t")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for '/' in name")
    store.close()


# ── identity_drift end-to-end (flag fires -> decider adjudicates) ──────────

def test_identity_drift_flag_fires_and_is_adjudicated(tmp_path):
    """The full path: a corpus where one entity's episodes have pairwise-
    disjoint topic neighborhoods makes ``flag_identity_drift`` fire, and the
    wired decider adjudicates it. Guards the radius+cap choice (radius=1 from
    an episode center would NOT reach the entity's sibling episodes, so the
    flag would never fire -- this test fails if the radius is too small)."""
    store = _store(tmp_path)
    # Alice appears in two episodes with DISJOINT topics (coding vs family).
    store.encode_episode(Episode(id="ep_000001", timestamp="2026-01-01",
                                 summary="coding work", full_text="t1",
                                 entities=["Alice"], topics=["db"]))
    store.encode_episode(Episode(id="ep_000002", timestamp="2026-01-02",
                                 summary="parenting chat", full_text="t2",
                                 entities=["Alice"], topics=["family"]))
    dec = FakeDecider(anomaly={"decision": "ask_user", "action": "ask",
                               "reasoning": "drift"})
    cons = _cons(store, decider=dec)
    rep = cons.run(limit=2)
    # The flag fired (an identity_drift anomaly was recorded) AND the decider
    # adjudicated it.
    drifts = [a for a in rep["anomalies"] if a.get("type") == "identity_drift"]
    assert drifts, "identity_drift flag did not fire (radius too small?)"
    assert dec.anomaly_calls >= 1
    assert any(d["decision"] == "ask_user"
               for d in rep["identity_drift_decisions"])
    store.close()