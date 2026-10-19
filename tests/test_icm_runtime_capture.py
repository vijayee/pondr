"""Tests for the live-session capture adapter (icm/runtime_capture.py).

Pins the honesty rules: intents are declared, memory units are the ids the
runtime actually returned, a chat answer is never turned into a congruence
label, capture failures never lose the answer, and the variation census
surfaces the same memory under different intents.
"""

from __future__ import annotations

import pytest

from icm.intent_keys import KeyResult
from icm.ledger import ICMLedger
from icm.runtime_capture import SessionCapture, variation_index


@pytest.fixture()
def ledger(tmp_path):
    return ICMLedger(str(tmp_path / "sessionledger"))


class StubPonder:
    def __init__(self, episodes=None, result=None):
        self.calls = []
        self.episodes = episodes if episodes is not None else []
        self.result = result

    def query(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.result is not None:
            return self.result
        return {"response": "an answer",
                "retrieved_episodes": [{"episode_id": e} for e in self.episodes]}


class StubKeyer:
    def key_for(self, text, declared_qualifier=""):
        return KeyResult({"requirements": [{"subject": "SELF",
                                            "attitude": "ask",
                                            "relation": "answer",
                                            "object": text[:40]}],
                          "qualifier": "", "declared_qualifier": "",
                          "epoch_ref": {"kind": "none", "value": ""},
                          "abstained": False}, "judged")


def test_query_records_declared_intent_and_memory_units(ledger):
    ponder = StubPonder(episodes=["ep-1", "ep-2", "ep-1"])
    sc = SessionCapture(ponder, ledger, keyer=StubKeyer())
    out = sc.query("why did we choose postgres?", qualifier="design review",
                   requirements=[{"relation": "explain", "object": "the choice"}],
                   epoch=3, consumer="bonsai")

    assert out["response"] == "an answer"
    info = out["icm_capture"]
    assert info["status"] == "recorded"
    assert info["memory_units"] == ["ep-1", "ep-2"]     # de-duplicated, ordered
    assert info["outcome"] == "unmeasured"              # never a prose label

    intent = ledger.get("icapture", info["intent_id"])
    assert intent["text"] == "why did we choose postgres?"
    assert intent["prediction"] == "none"
    assert intent["qualifier"] == "design review"
    assert intent["epoch"] == 3 and intent["epoch_source"] == "declared"
    assert intent["requirements"][0]["object"] == "the choice"
    assert intent["key_status"] == "judged"

    retr = ledger.get("icapture", info["retrieval_id"])
    assert retr["memory_units"] == ["ep-1", "ep-2"]
    assert retr["intent_id"] == info["intent_id"]
    # no outcome record exists for this turn
    assert ledger.count("icapture") == 2


def test_query_passes_runtime_kwargs_through(ledger):
    ponder = StubPonder(episodes=["ep-9"])
    sc = SessionCapture(ponder, ledger)
    sc.query("q", consumer="bonsai", auto_persist=False)
    assert ponder.calls == [("q", {"consumer": "bonsai", "auto_persist": False})]


def test_empty_retrieval_is_recorded_honestly(ledger):
    ponder = StubPonder(episodes=[])
    sc = SessionCapture(ponder, ledger)
    out = sc.query("nothing was in memory")
    assert out["icm_capture"]["status"] == "recorded"
    assert out["icm_capture"]["memory_units"] == []
    retr = ledger.get("icapture", out["icm_capture"]["retrieval_id"])
    assert retr["retrieved"] == []


def test_declaration_failure_does_not_block_the_answer(ledger):
    ponder = StubPonder(episodes=["ep-1"])
    sc = SessionCapture(ponder, ledger)
    out = sc.query("bad declaration",
                   requirements=[{"relation": "no object"}])  # invalid triple
    assert out["response"] == "an answer"
    assert out["icm_capture"]["status"] == "error"
    assert out["icm_capture"]["stage"] == "declare"
    assert ponder.calls and ledger.count("icapture") == 0


def test_runtime_failure_is_not_swallowed(ledger):
    class ExplodingPonder:
        def query(self, prompt, **kwargs):
            raise RuntimeError("runtime down")

    sc = SessionCapture(ExplodingPonder(), ledger)
    with pytest.raises(RuntimeError):
        sc.query("q")


def test_variation_index_finds_the_same_memory_under_two_intents(ledger):
    ponder = StubPonder(episodes=["shared-ep"])
    sc = SessionCapture(ponder, ledger, keyer=StubKeyer())
    first = sc.query("what did we decide about storage?")
    second = sc.query("which option was rejected?")
    other = StubPonder(episodes=["solo-ep"])
    SessionCapture(other, ledger).query("a third question")
    # a turn with no retrieval must not create a unit
    SessionCapture(StubPonder(episodes=[]), ledger).query("empty turn")

    rep = variation_index(ledger)
    assert rep["memory_units"] == 2
    assert rep["with_variation"] == 1
    rows = rep["units"]["shared-ep"]
    assert {r["intent_id"] for r in rows} == {
        first["icm_capture"]["intent_id"], second["icm_capture"]["intent_id"]}
    assert all(r["key_status"] == "judged" for r in rows)


def test_variation_index_does_not_double_count_repeated_retrieval(ledger):
    ponder = StubPonder(episodes=["ep-1"])
    sc = SessionCapture(ponder, ledger)
    sc.query("first")
    sc.query("second")
    # force the same intent to be retrieved twice by reusing its id
    first_intent = next(r for r in ledger.records("icapture")
                        if r.get("level") == "goal")
    sc.capture.retrieval(first_intent["id"], "again", ["ep-1"])
    rep = variation_index(ledger)
    assert len(rep["units"]["ep-1"]) == 2   # distinct intents only
