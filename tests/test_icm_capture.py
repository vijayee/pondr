"""Tests for the hierarchical intent-capture layer (icm/capture.py).

The laws these tests pin: labels are rule-computed (never stated), declared
branches are mandatory for goals, branchless steps inherit the goal,
hierarchy links persist, and the JSONL mirror is canonical.
"""

from __future__ import annotations

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icm.capture import IntentCapture
from icm.ledger import ICMLedger


@pytest.fixture()
def ledger(tmp_path):
    return ICMLedger(str(tmp_path / "capledger"))


def test_goal_requires_a_branch(ledger):
    cap = IntentCapture(ledger)
    with pytest.raises(ValueError):
        cap.goal("no branches declared", desired=None, feared=None, default=None)


def test_full_roundtrip_and_rule_computed_label(ledger):
    cap = IntentCapture(ledger)
    g = cap.goal("measure the tail", desired="tail grows", feared="tail vanishes")
    s = cap.step(g, "run the screen", desired=">=20 verified", feared="fails")
    e = cap.retrieval(s, "verify q400", retrieved=["ep-1", "ep-2"], k=2)
    oid = cap.outcome(e, "desired")
    assert oid.startswith("outc-")
    out = ledger.get("icapture", oid)
    # the label came from the rule's evaluation, computed at observe time
    assert out["label"] == "confirmed"
    assert out["rule_version"].startswith("icm-congruence/")
    path = cap.close()
    lines = [json.loads(l) for l in io.open(path) if l.strip()]
    levels = {r["level"] for r in lines}
    assert {"goal", "step", "retrieval", "outcome"} <= levels
    # hierarchy links
    by_id = {r["id"]: r for r in lines}
    assert by_id[s]["parent"] == g
    assert by_id[e]["parent"] == s


def test_branchless_step_inherits_goal_prediction(ledger):
    cap = IntentCapture(ledger)
    g = cap.goal("build the index", desired="engine wins",
                 feared="tie", default="ns")
    s = cap.step(g, "wire the arm")            # no branches on the step
    e = cap.retrieval(s, "consult q", retrieved=["x"])
    oid = cap.outcome(e, "feared")             # inherited prediction
    out = ledger.get("icapture", oid)
    # feared was PREDICTED and observed -> CONFIRMED (prediction match, not valence)
    assert out["label"] == "confirmed"


def test_unpredicted_named_branch_observed_is_violated_not_error(ledger):
    cap = IntentCapture(ledger)
    g = cap.goal("g", desired="only this")
    s = cap.step(g, "s", desired="step thing")   # only desired predicted
    e = cap.retrieval(s, "q", retrieved=[])
    oid = cap.outcome(e, "feared")               # named but unpredicted
    out = ledger.get("icapture", oid)
    assert out["label"] == "violated"            # missed prediction, no error


def test_other_observation_is_always_violated(ledger):
    cap = IntentCapture(ledger)
    g = cap.goal("g", desired="x", feared="y", default="z")
    s = cap.step(g, "s", desired="sx", feared="fx")
    e = cap.retrieval(s, "q", retrieved=[])
    oid = cap.outcome(e, "other")                # max surprise
    out = ledger.get("icapture", oid)
    assert out["label"] == "violated"


def test_mirror_is_append_only_across_sessions(ledger, tmp_path):
    mirror = str(tmp_path / "mirror")
    cap = IntentCapture(ledger, mirror_dir=mirror)
    g = cap.goal("session one", desired="a")
    cap.step(g, "s1")
    path1 = cap.close()
    cap2 = IntentCapture(ledger, mirror_dir=mirror)
    g2 = cap2.goal("session two", desired="b")
    cap2.step(g2, "s2")
    path2 = cap2.close()
    assert path1 == path2
    texts = [json.loads(l)["text"] for l in io.open(path1) if l.strip()]
    assert "session one" in texts and "session two" in texts