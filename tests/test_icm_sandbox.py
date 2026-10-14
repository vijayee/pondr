"""Sandbox pilot tests -- network-free (ScriptedPolicy), real file side effects
in tmp dirs. Proves env mechanics, observer mapping, ledger flow, and that the
vacuity guard catches a perfectly-clean world."""

import os
import sys
import time

import pytest

from icm.gate import compute_gate_metrics
from icm.ledger import ICMLedger
from icm.sandbox_env import SandboxEnv, BRANCH_FOR_EVENT, classify_run
from icm.sandbox_pilot import run_sandbox_pilot

LEVEL_MAP = {"clean": 0.0, "degraded": 0.5, "hardfail": 1.0, "other": 1.0}

sys.path.insert(0, "tests")
from test_icm_agent_pilot import ScriptedPolicy  # noqa: E402


@pytest.fixture
def ledger(tmp_path):
    from icm.ledger import ICMLedger

    led = ICMLedger(str(tmp_path / "loop_db"))
    yield led
    led.close()


def test_classify_run_mapping():
    ok = {"status": "ok"}
    err = {"status": "error", "detail": "x"}
    fatal_err = {"status": "error", "fatal": True}
    assert classify_run([ok], [ok, ok]) == "clean"
    assert classify_run([err, ok], [ok, ok]) == "degraded"
    assert classify_run([err, fatal_err], [ok, ok]) == "hardfail"
    assert classify_run([ok], [err]) == "hardfail"
    assert BRANCH_FOR_EVENT == {"clean": "desired", "degraded": "default",
                                "hardfail": "feared", "other": "other"}


def test_env_probe_rates_by_p(tmp_path):
    env0 = SandboxEnv(str(tmp_path / "e0"), seed=3, probe_fail_p=0.0)
    assert all(env0.probe_remote()["status"] == "ok" for _ in range(20))
    env1 = SandboxEnv(str(tmp_path / "e1"), seed=3, probe_fail_p=1.0)
    assert all(env1.probe_remote()["status"] == "error" for _ in range(20))


def test_env_manifest_round_trip_real_files(tmp_path):
    env = SandboxEnv(str(tmp_path / "e"), seed=5)
    d = env.materialize_fixture("fixture-a")
    assert env.write_manifest(d, "fixture-a")["status"] == "ok"
    assert env.verify_manifest(d, "fixture-a")["status"] == "ok"
    assert env.write_manifest(d, "nope")["status"] == "error"
    assert os.path.exists(os.path.join(d, "manifest.json"))
    assert len([f for f in os.listdir(d) if f.endswith(".csv")]) == 4
    env.wipe()


def test_sandbox_pilot_clean_world_is_caught_as_vacuous(tmp_path):
    db = str(tmp_path / "db")
    sandbox = str(tmp_path / "sandbox")
    summary = run_sandbox_pilot(db, ScriptedPolicy(), sandbox, n_sessions=8, seed=3,
                                probe_fail_p=0.0)
    assert all(r["event_class"] == "clean" for r in summary["session_log"])
    with ICMLedger(db) as led:
        exps = [e for e in led.records("expectation")]
        assert len(exps) == 16  # E1 + E2 per session, all resolved
        assert all(e["status"] == "resolved" for e in exps)
        planted = {e["intent_id"]: "clean" for e in exps}
        gate = compute_gate_metrics(led, planted=planted, level_map=LEVEL_MAP)
    assert gate["verdict"] == "FAIL"
    assert gate["checks"]["g2_nonvacuity"]["ok"] is False


def test_sandbox_pilot_flaky_world_harvest_flow(tmp_path):
    db = str(tmp_path / "db")
    sandbox = str(tmp_path / "sandbox")
    summary = run_sandbox_pilot(db, ScriptedPolicy(), sandbox, n_sessions=9, seed=11,
                                probe_fail_p=0.5)
    classes = {r["event_class"] for r in summary["session_log"]}
    assert classes <= {"clean", "degraded", "hardfail"}
    assert len(summary["session_log"]) == 9
    observed = {r["session"]: r["event_class"] for r in summary["session_log"]}
    with ICMLedger(db) as led:
        exps = list(led.records("expectation"))
        assert all(e["status"] == "resolved" for e in exps)
        planted = {}
        for e in exps:
            planted.setdefault(e["intent_id"], observed.get(e["session_id"], "other"))
        gate = compute_gate_metrics(led, planted=planted, level_map=LEVEL_MAP)
    assert gate["checks"]["g1_coverage"]["value"] > 0.9
    assert gate["checks"]["g4_signal_recovery"]["evaluated"]

def test_pilot_capture_loop_closes_memory(ledger, tmp_path):
    """Loop closure: episodes written by the REAL pilot (not seeded), linked
    to the session's expectations; the forget pass then treats them by their
    congruence (hardfail sessions' episodes = violated -> vivid; clean -> blur)."""
    from icm.episodic import EpisodicStore
    from test_icm_episodic import StubEmbedder

    # single substrate: the store SHARES the pilot's ledger (the violation
    # labels and episodes must live on one WaveDB for the forget pass to see them)
    real_store = EpisodicStore(ledger, StubEmbedder(), dim=384,
                               blur_horizon_days=7.0, drop_horizon_days=30.0)
    summary = run_sandbox_pilot("unused", ScriptedPolicy(),
                                str(tmp_path / "sandbox2"), n_sessions=9, seed=11,
                                probe_fail_p=0.5, store=real_store, ledger=ledger)
    # pretend episodes are 40d old (past the blur horizon) so the forget
    # pass engages on every one
    old = time.time() - 40 * 86400
    for rec in list(ledger.records("episode")):
        rec["created_ts"] = old
        ledger.put("episode", rec)
    # a fresh episode must exist for every session
    assert ledger.count("episode") == 9
    texts = [r["text"] for r in ledger.records("episode")]
    assert any("Task:" in t and "Observed:" in t for t in texts)
    # expectations from those episodes ARE the pilot's (both e1, e2 per session)
    eids = {r["episode_id"] for r in ledger.records("episode")}
    assert len(eids) == 9
    maintain = real_store.maintain(now_ts=time.time())
    hardfails = [r for r in summary["session_log"] if r["event_class"] == "hardfail"]
    if hardfails:
        # their sessions' episodes should be vivid (violated -> unpredictable observed branch)
        vivid = [(r["episode_id"], r["fidelity"]) for r in ledger.records("episode")
                 if r["fidelity"] == "full"]
        assert maintain["kept_vivid"] >= len(hardfails), (maintain, hardfails)
