"""Phase-0 gate tests: the pilot harvest passes the gate, and fails
specifically when the harvest is vacuous."""

import pytest

from icm.gate import compute_gate_metrics
from icm.ledger import ICMLedger
from icm.pilot import run_pilot


@pytest.fixture(scope="module")
def pilot_out(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("pilot") / "ledger")
    return run_pilot(db, n_sessions=40, seed=7), db


def test_clean_pilot_passes_gate(pilot_out):
    summary, db = pilot_out
    with ICMLedger(db) as led:
        gate = compute_gate_metrics(led, planted=summary["planted"])
    assert gate["verdict"] == "PASS", gate["checks"]
    g4 = gate["checks"]["g4_signal_recovery"]
    assert g4["evaluated"] and g4["correlation"] is not None


def test_all_confirmed_harvest_is_caught_as_vacuous(tmp_path):
    # stamp every stored label 'confirmed' -> violated class absent -> FAIL g2
    db = str(tmp_path / "vacuous")
    summary = run_pilot(db, n_sessions=12, seed=3)
    with ICMLedger(db) as led:
        for eid in list(led.ids("expectation")):
            exp = led.get("expectation", eid)
            if exp["status"] == "resolved":
                exp["resolution"]["congruence"] = "confirmed"
                led.put("expectation", exp)
        gate = compute_gate_metrics(led, planted=summary["planted"])
    assert gate["verdict"] == "FAIL"
    assert gate["checks"]["g2_nonvacuity"]["ok"] is False


def test_tampered_label_fails_determinism(tmp_path):
    db = str(tmp_path / "tampered")
    summary = run_pilot(db, n_sessions=12, seed=5)
    with ICMLedger(db) as led:
        eid = led.ids("expectation")[0]
        exp = led.get("expectation", eid)
        exp["resolution"]["congruence"] = ("violated"
                                           if exp["resolution"]["congruence"] == "confirmed"
                                           else "confirmed")
        exp["status"] = "resolved"
        led.put("expectation", exp)
        gate = compute_gate_metrics(led, planted=summary["planted"])
    assert gate["verdict"] == "FAIL"
    assert gate["checks"]["g3_determinism"]["ok"] is False
    assert gate["checks"]["g3_determinism"]["value"] >= 1


def test_inverted_severity_recover_fails_g4(tmp_path):
    # violated labels land ONLY in the mildest world -> recovery order inverts
    from icm.instrument import ICMSession, ExpectationDecl, OutcomeDecl
    from icm.agent_pilot import SEV_LEVEL

    db = str(tmp_path / "inverted")
    planted = {}
    with ICMLedger(db) as led:
        for si, sev in enumerate(["healthy", "healthy", "warning", "broken"]):
            sid = f"sess-{si}"
            s = ICMSession(led, sid)
            s.begin()
            iid = s.declare_intent(f"task {si}", intent_id=f"int-{si}")
            planted[iid] = sev
            branches = {"desired": "ok", "feared": None, "default": "meh"}
            for k in range(3):
                eid = s.expect(ExpectationDecl(intent_id=iid, statement=f"check {k}",
                                               branches=branches,
                                               horizon={"kind": "manual"}),
                               exp_id=f"exp-{si}-{k}")
                # healthy sessions -> all violated; severe worlds -> confirmed
                branch = "other" if sev == "healthy" else "desired"
                s.resolve(OutcomeDecl(eid, f"obs {k}", branch, "env"),
                          outcome_id=f"out-{si}-{k}")
            s.finish()
        gate = compute_gate_metrics(led, planted=planted, level_map=SEV_LEVEL)
    assert gate["verdict"] == "FAIL"
    assert gate["checks"]["g4_signal_recovery"]["ok"] is False