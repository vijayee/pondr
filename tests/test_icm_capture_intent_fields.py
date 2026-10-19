"""Tests for the six-field intent record in icm/capture.py (schema v2).

Pins the laws the store spec depends on:
  - qualifier / requirements / epoch are recorded on every goal and step;
  - requirements are a LIST (a single triple is accepted, never padded);
  - epoch falls back to capture time and says so;
  - expansion (triggered_by) is only recordable after a NON-REACH outcome;
  - write-time judge keys are stored beside the intent and copied onto the
    retrieval event (the memory unit's association), and a judge failure
    never blocks a write;
  - the congruence label itself is still RULE-COMPUTED (never stated).
"""

from __future__ import annotations

import json

import pytest

from icm.capture import IntentCapture, normalize_requirements
from icm.intent_keys import KeyResult
from icm.ledger import ICMLedger


@pytest.fixture()
def ledger(tmp_path):
    return ICMLedger(str(tmp_path / "capledger"))


class StubKeyer:
    """Deterministic stand-in for IntentKeyer; records every call."""

    def __init__(self, key=None, status="judged"):
        self.key = key if key is not None else {
            "requirements": [{"subject": "SELF", "attitude": "report",
                              "relation": "report", "object": "checksum alpha"}],
            "qualifier": "", "declared_qualifier": "",
            "epoch_ref": {"kind": "ordinal", "value": "EARLIEST"},
            "abstained": False,
        }
        self.status = status
        self.calls = []

    def key_for(self, text, declared_qualifier=""):
        self.calls.append((text, declared_qualifier))
        return KeyResult(self.key, self.status)


def test_goal_records_qualifier_requirements_epoch_and_schema(ledger):
    cap = IntentCapture(ledger)
    reqs = [
        {"relation": "report", "object": "checksum alpha"},
        {"subject": "index", "attitude": "verify", "relation": "match",
         "object": "row count"},
    ]
    g = cap.goal("report the archived checksum", desired="reported",
                 qualifier="first ingest", requirements=reqs, epoch=12)
    rec = ledger.get("icapture", g)
    assert rec["schema_version"] == 2
    assert rec["qualifier"] == "first ingest"
    assert rec["requirement_count"] == 2
    assert rec["requirements"][0]["subject"] == "SELF"     # default per schema
    assert rec["requirements"][1]["attitude"] == "verify"
    assert rec["epoch"] == 12 and rec["epoch_source"] == "declared"
    assert rec["intent_key"] is None and rec["key_status"] == "disabled"


def test_epoch_defaults_to_capture_time_and_says_so(ledger):
    cap = IntentCapture(ledger)
    g = cap.goal("g", desired="x")
    rec = ledger.get("icapture", g)
    assert rec["epoch"] == rec["declared_at"]
    assert rec["epoch_source"] == "capture-time"


def test_requirement_normalisation_accepts_one_and_never_pads():
    one = normalize_requirements({"relation": "report", "object": "value"})
    assert len(one) == 1 and one[0]["subject"] == "SELF"
    with pytest.raises(ValueError):
        normalize_requirements([{"relation": "report"}])
    with pytest.raises(ValueError):
        normalize_requirements("not a list")
    assert normalize_requirements(None) == []


def test_expansion_only_after_non_reach(ledger):
    cap = IntentCapture(ledger)
    g = cap.goal("g", desired="a", feared="b")
    s = cap.step(g, "s", desired="c")
    e = cap.retrieval(s, "q", retrieved=["ep-1"])
    missed = cap.outcome(e, "feared")       # violated (predicted only c)
    assert ledger.get("icapture", missed)["label"] == "violated"

    new_goal = cap.goal("explain the miss", desired="explained",
                        triggered_by=missed)
    assert ledger.get("icapture", new_goal)["triggered_by"] == missed

    # a CONFIRMED outcome may not spawn an expansion (causal direction law)
    g2 = cap.goal("g2", desired="a")
    s2 = cap.step(g2, "s2", desired="c")
    e2 = cap.retrieval(s2, "q2", retrieved=[])
    hit = cap.outcome(e2, "desired")
    assert ledger.get("icapture", hit)["label"] == "confirmed"
    with pytest.raises(ValueError):
        cap.goal("new", desired="y", triggered_by=hit)
    with pytest.raises(ValueError):
        cap.goal("new", desired="y", triggered_by="outc-does-not-exist")


def test_write_time_key_stored_beside_intent_and_memory_unit(ledger):
    keyer = StubKeyer()
    cap = IntentCapture(ledger, keyer=keyer)
    g = cap.goal("report the earliest checksum", desired="reported",
                 qualifier="first")
    s = cap.step(g, "fetch it")
    e = cap.retrieval(s, "q", retrieved=["ep-1", "ep-2"])
    grec = ledger.get("icapture", g)
    erec = ledger.get("icapture", e)
    assert grec["key_status"] == "judged"
    assert grec["intent_key"]["epoch_ref"]["value"] == "EARLIEST"
    assert grec["key_abstained"] is False
    # the retrieval event is the memory-unit association: it carries the key of
    # the OPERATIVE intent (the step, which carries its own key)
    assert erec["intent_id"] == s
    assert erec["intent_key"] == grec["intent_key"]
    assert erec["memory_units"] == ["ep-1", "ep-2"]
    assert erec["epoch"] == ledger.get("icapture", s)["epoch"]
    assert keyer.calls[0][1] == "first"
    assert cap.key_summary()["counts"] == {"judged": 2}


def test_prediction_free_intent_cannot_produce_a_label(ledger):
    """A live-session intent declares no branch, so outcome() refuses it."""
    cap = IntentCapture(ledger)
    g = cap.intent("why did we choose postgres?", qualifier="design review")
    rec = ledger.get("icapture", g)
    assert rec["prediction"] == "none"
    assert rec["branches"] == {"desired": None, "feared": None, "default": None}
    e = cap.retrieval(g, "why postgres", retrieved=["ep-1"])
    assert ledger.get("icapture", e)["intent_id"] == g
    with pytest.raises(ValueError):
        cap.outcome(e, "desired")


def test_step_level_intent_links_to_its_goal(ledger):
    cap = IntentCapture(ledger)
    g = cap.intent("top-level aim")
    s = cap.intent("a sub-aim", level="step", parent=g)
    assert ledger.get("icapture", s)["parent"] == g
    assert ledger.get("icapture", s)["level"] == "step"
    with pytest.raises(ValueError):
        cap.intent("bad", level="step", parent="goal-missing")
    with pytest.raises(ValueError):
        cap.intent("bad", level="nonsense")


def test_judge_failure_never_blocks_a_write(ledger):
    class BoomKeyer:
        def key_for(self, text, declared_qualifier=""):
            raise RuntimeError("judge down")

    cap = IntentCapture(ledger, keyer=BoomKeyer())
    g = cap.goal("g", desired="x", requirements=[{"relation": "r", "object": "o"}])
    rec = ledger.get("icapture", g)
    assert rec["key_status"] == "error" and rec["intent_key"] is None
    assert rec["requirements"][0]["object"] == "o"     # declaration survives
    assert cap.key_summary()["errors"] == 1


def test_abstained_key_is_recorded_not_hidden(ledger):
    keyer = StubKeyer(key={"requirements": [], "qualifier": "",
                           "epoch_ref": {"kind": "none", "value": ""},
                           "abstained": True}, status="abstained")
    cap = IntentCapture(ledger, keyer=keyer)
    g = cap.goal("g", desired="x")
    rec = ledger.get("icapture", g)
    assert rec["key_status"] == "abstained"
    assert rec["key_abstained"] is True


def test_label_law_unchanged_with_new_fields(ledger):
    gap = IntentCapture(ledger)
    g = gap.goal("g", desired="a", feared="b", qualifier="latest")
    s = gap.step(g, "s", desired="c")
    e = gap.retrieval(s, "q", retrieved=["x"])
    oid = gap.outcome(e, "other")
    out = ledger.get("icapture", oid)
    assert out["label"] == "violated"
    assert out["rule_version"].startswith("icm-congruence/")
    # outcomes are unique per capture: two sessions must not overwrite
    gap2 = IntentCapture(ledger)
    g2 = gap2.goal("g2", desired="a")
    s2 = gap2.step(g2, "s2", desired="c")
    e2 = gap2.retrieval(s2, "q2", retrieved=[])
    oid2 = gap2.outcome(e2, "other")
    assert oid != oid2
    assert ledger.get("icapture", oid)["label"] == "violated"


def test_mirror_carries_the_new_fields(ledger, tmp_path):
    cap = IntentCapture(ledger, mirror_dir=str(tmp_path / "mirror"))
    g = cap.goal("g", desired="x", qualifier="q", epoch=3,
                 requirements=[{"relation": "r", "object": "o"}])
    path = cap.close()
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    goal = next(r for r in rows if r.get("id") == g)
    assert goal["qualifier"] == "q" and goal["epoch"] == 3
    assert goal["requirements"][0]["object"] == "o"
