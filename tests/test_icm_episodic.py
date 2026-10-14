"""Episodic store tests (ICM Phase 1): capture/recall roundtrip, the
icm-forget/1 congruence-driven fidelity cascade, and close ordering.
Network-free (stub embedder)."""

import time

import pytest

from icm.episodic import EpisodicStore, FORGET_RULE_VERSION, gist_of
from icm.instrument import ICMSession, OutcomeDecl
from icm.ledger import ICMLedger, LedgerError


class StubEmbedder:
    """Deterministic 384-d hash-bag embedding: shared tokens -> shared dirs."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in t.lower().replace(":", " ").split():
                h = hash(tok) % self.dim
                vec[h] += 1.0
            n = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / n for v in vec])
        return out


@pytest.fixture
def ledger(tmp_path):
    led = ICMLedger(str(tmp_path / "ep_db"))
    yield led
    led.close()


@pytest.fixture
def store(ledger):
    return EpisodicStore(ledger, StubEmbedder(), dim=384,
                         blur_horizon_days=7.0, drop_horizon_days=30.0)


def test_gist_is_deterministic_and_compressed():
    t = ("The deployment failed because the probe endpoint returned 500 twice. "
         "Retry also failed and the checksum verify broke afterwards. "
         "Manifest was missing so verification aborted with an IO error. "
         "Follow-up analysis traced it to a stale artifact cache.")
    g1, g2 = gist_of(t), gist_of(t)
    assert g1 == g2
    assert g1.startswith("[gist] ")
    assert len(g1) < len(t)  # real compression for over-budget text


def test_capture_recall_roundtrip(store):
    eid = store.capture("s-1", "i-1", "The database migration failed with a checksum error")
    store.capture("s-1", "i-1", "Team lunch at the curry place on Friday")
    hits = store.recall("checksum error in the database migration", k=2)
    assert hits and hits[0][0]["episode_id"] == eid
    assert hits[0][1] > 0.1  # similarity, best first


def test_violated_linked_episode_stays_vivid(store, ledger):
    s = ICMSession(ledger, "s-forget")
    s.begin()
    iid = s.declare_intent("deploy")
    eid_exp = s.expect(_decl(iid), exp_id="exp-f1")
    linked = store.capture("s-forget", iid, "deployment run log with a hard failure",
                           expectation_ids=[eid_exp], created_ts=time.time() - 20 * 86400)
    s.resolve(OutcomeDecl("exp-f1", "observed failure", "desired"), "out-f1")  # predicted -> confirmed
    linked2 = store.capture("s-forget", iid, "another plain old run log",
                            expectation_ids=[eid_exp], created_ts=time.time() - 20 * 86400)
    # make the SECOND one violated
    s2 = ICMSession(ledger, "s-forget2")
    s2.begin()
    iid2 = s2.declare_intent("deploy2")
    exp2 = s2.expect(_decl2(iid2), exp_id="exp-f2")
    violated = store.capture("s-forget2", iid2, "run log of the unexpected outage",
                             expectation_ids=[exp2], created_ts=time.time() - 40 * 86400)  # past drop horizon: vivid protection test
    s2.resolve(OutcomeDecl("exp-f2", "observed unpredicted outage", "other"), "out-f2")

    summary = store.maintain(now_ts=time.time())
    rec_ok = ledger.get("episode", linked)
    rec_viol = ledger.get("episode", violated)
    rec_plain = ledger.get("episode", linked2)
    # confirmed-linked at -20d: no surprise, age rule applies -> blurred
    assert rec_ok["fidelity"] == "gist"
    assert rec_viol["fidelity"] == "full"  # violated-linked: vivid past horizons
    assert rec_plain["fidelity"] == "gist" and rec_plain["text"].startswith("[gist] ")
    assert rec_plain["text_before_blur"].startswith("another plain old run log")
    assert summary["blurred"] >= 1 and summary["kept_vivid"] >= 1


def _decl(intent_id: str):
    from icm.instrument import ExpectationDecl

    return ExpectationDecl(intent_id=intent_id, statement="deploy runs clean",
                           branches={"desired": "green"},
                           horizon={"kind": "manual"})


def _decl2(intent_id: str):
    from icm.instrument import ExpectationDecl

    return ExpectationDecl(intent_id=intent_id, statement="no surprises",
                           branches={"desired": "calm"},
                           horizon={"kind": "manual"})


def test_old_unlinked_episode_drops(store, ledger):
    old = store.capture("s-old", None, "an ancient routine log entry",
                        created_ts=time.time() - 60 * 86400)
    fresh = store.capture("s-new", None, "a fresh log entry", created_ts=time.time())
    summary = store.maintain(now_ts=time.time())
    assert ledger.get("episode", old) is None  # KV gone
    assert ledger.count("episode") == 1
    assert ledger.ids("episode") == [fresh]  # registry shrunk
    rec = ledger.get("episode", fresh)
    assert rec["fidelity"] == "full"
    assert summary["dropped"] == 1


def test_pending_links_do_not_protect(store, ledger):
    s = ICMSession(ledger, "s-pending")
    s.begin()
    iid = s.declare_intent("aim")
    exp = s.expect(_decl(iid), exp_id="exp-pending")
    store.capture("s-pending", iid, "old run awaiting resolution",
                  expectation_ids=[exp], created_ts=time.time() - 40 * 86400)
    summary = store.maintain(now_ts=time.time())
    assert summary["blurred"] == 1  # pending resolution: age rule applies


def test_counts_and_close_ordering(store, ledger):
    store.capture("s", None, "one")
    store.capture("s", None, "two")
    assert store.counts() == {"full": 2}
    ledger.close()  # vector layer registered_close -> no subtree-left-open errors


def test_delete_record_registry_integrity(ledger):
    ledger.put_session("s1")
    ledger.delete_record("session", "s1")
    assert ledger.ids("session") == []
    with pytest.raises(LedgerError):
        ledger.delete_record("session", "s1")

def test_one_pass_processes_every_old_episode(store, ledger):
    """Regression (halving bug): deletes during iteration skipped one record
    per drop; a single maintain pass must account for EVERY old episode."""
    old = time.time() - 60 * 86400
    for i in range(6):
        store.capture("s-old", None, f"routine log number {i}",
                      created_ts=old)
    summary = store.maintain(now_ts=time.time())
    assert summary["dropped"] == 6 and summary["blurred"] == 6
    assert ledger.count("episode") == 0
