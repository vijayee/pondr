"""Ledger storage tests (ICM Phase 0): records, congruence stamping, graph
links, expiry, index integrity, persistence across reopen."""

import pytest

from icm.congruence import Congruence
from icm.instrument import ExpectationDecl, ICMSession, OutcomeDecl
from icm.ledger import ICMLedger, LedgerError


@pytest.fixture
def ledger(tmp_path):
    led = ICMLedger(str(tmp_path / "icm_ledger_db"))
    yield led
    led.close()


@pytest.fixture
def session(ledger):
    s = ICMSession(ledger, "sess-t1")
    s.begin()
    return s


def _decl(intent_id: str) -> ExpectationDecl:
    return ExpectationDecl(
        intent_id=intent_id,
        statement="deploy pipeline will finish green",
        branches={"desired": "green", "feared": None, "default": "stalls"},
        horizon={"kind": "manual"},
    )


def test_intent_expectation_outcome_flow(ledger, session):
    iid = session.declare_intent("deploy the service", intent_id="int-1")
    eid = session.expect(_decl(iid), exp_id="exp-1")
    session.step()
    out = session.resolve(OutcomeDecl("exp-1", "pipeline green", "desired"), "out-1")

    exp = ledger.get("expectation", "exp-1")
    assert exp["status"] == "resolved"
    assert exp["resolution"]["congruence"] == Congruence.CONFIRMED.value
    assert exp["resolution"]["outcome_id"] == "out-1"

    outcome = ledger.get("outcome", "out-1")
    assert outcome["observed_branch"] == "desired"
    assert outcome["congruence"]["congruence"] == Congruence.CONFIRMED.value

    assert out == "out-1"  # returned id matches stored record
    assert ledger.get("intent", "int-1")["aim"] == "deploy the service"


def test_double_resolution_rejected(ledger, session):
    iid = session.declare_intent("aim", "int-2")
    session.expect(_decl(iid), exp_id="exp-2")
    session.resolve(OutcomeDecl("exp-2", "first", "desired"), "out-2a")
    with pytest.raises(LedgerError):
        session.resolve(OutcomeDecl("exp-2", "second", "default"), "out-2b")


def test_resolution_after_expiry_rejected(ledger, session):
    iid = session.declare_intent("aim", "int-3")
    decl = ExpectationDecl(intent_id=iid, statement="soon", branches={"desired": "x"},
                           horizon={"kind": "step", "deadline_step": 1})
    session.expect(decl, exp_id="exp-3")
    session.step()  # step 1
    session.step()  # step 2: past deadline
    session.expire_pending()
    assert ledger.get("expectation", "exp-3")["status"] == "expired"
    with pytest.raises(LedgerError):
        session.resolve(OutcomeDecl("exp-3", "late", "desired"), "out-3")


def test_expired_has_no_congruence_label(ledger, session):
    iid = session.declare_intent("aim", "int-4")
    decl = ExpectationDecl(intent_id=iid, statement="soon", branches={"desired": "x"},
                           horizon={"kind": "step", "deadline_step": 0})
    session.expect(decl, exp_id="exp-4")
    session.step()  # past the deadline (due requires current_step > deadline_step)
    session.expire_pending()
    exp = ledger.get("expectation", "exp-4")
    assert exp["status"] == "expired"
    assert exp["resolution"] is None


def test_unknown_intent_rejected(session):
    with pytest.raises(LedgerError):
        session.expect(_decl("int-missing"), exp_id="exp-5")


def test_expectation_must_predict_something(session):
    iid = session.declare_intent("aim", "int-6")
    bad = ExpectationDecl(intent_id=iid, statement="s",
                          branches={"desired": None, "feared": None, "default": None},
                          horizon={"kind": "manual"})
    with pytest.raises(LedgerError):
        session.expect(bad, exp_id="exp-6")


def test_graph_links_out(ledger, session):
    iid = session.declare_intent("aim", "int-7")
    session.expect(_decl(iid), exp_id="exp-7")
    session.resolve(OutcomeDecl("exp-7", "observed", "desired"), "out-7")
    exps = ledger.graph_out(f"intent:int-7", "has_expectation")
    congruences = ledger.graph_out("expectation:exp-7", "has_congruence")
    assert exps == ["expectation:exp-7"]
    assert congruences == ["confirmed"]


def test_index_integrity_and_records(ledger, session):
    iid = session.declare_intent("aim", "int-8")
    session.expect(_decl(iid), exp_id="exp-8")
    eids = ledger.ids("expectation")
    assert eids == ["exp-8"]
    recs = list(ledger.records("expectation"))
    assert recs[0]["exp_id"] == "exp-8"


def test_persistence_across_reopen(tmp_path):
    db = str(tmp_path / "icm_reopen_db")
    with ICMLedger(db) as led1:
        s = ICMSession(led1, "sess-r")
        s.begin()
        iid = s.declare_intent("persist", "int-r")
        s.expect(_decl(iid), exp_id="exp-r")
        s.resolve(OutcomeDecl("exp-r", "kept", "default"), "out-r")
    with ICMLedger(db) as led2:
        exp = led2.get("expectation", "exp-r")
        assert exp is not None and exp["status"] == "resolved"
        assert exp["resolution"]["congruence"] == Congruence.CONFIRMED.value
        assert led2.ids("expectation") == ["exp-r"]
        assert [r["exp_id"] for r in led2.records("expectation")] == ["exp-r"]


def test_intent_slots_and_open_object(ledger, session):
    slots = {"slot_version": "spo-decl/1", "subject": "SELF", "attitude": "intend",
             "relation": "ingest", "object": "fixture manifest verified",
             "residue": None, "propositions": 1}
    session.declare_intent("ingest fixture", intent_id="int-s1", slots=slots)
    rec = ledger.get("intent", "int-s1")
    assert rec["slots"]["subject"] == "SELF"
    assert rec.get("open_object") is False


def test_underspecified_residue_marks_open_object(ledger, session):
    slots = {"subject": "SELF", "attitude": "maintain", "relation": "improve",
             "object": "...", "residue": "underspecified", "propositions": 2}
    session.declare_intent("make things better", intent_id="int-s2", slots=slots)
    rec = ledger.get("intent", "int-s2")
    assert rec.get("open_object") is True


def test_bad_residue_rejected(ledger, session):
    with pytest.raises(LedgerError):
        session.declare_intent("aim", intent_id="int-s3",
                               slots={"subject": "SELF", "attitude": "intend",
                                      "relation": "do", "object": "x",
                                      "residue": "without affecting production",
                                      "propositions": 1})