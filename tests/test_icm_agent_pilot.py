"""Agent-pilot tests with a SCRIPTED policy -- network-free. Proves the
ledger flow, the E1/E2 double labeling, and the gate on real pilot code paths."""

import pytest

from icm.agent_pilot import run_agent_pilot, sample_outcome
from icm.gate import compute_gate_metrics
from icm.ledger import ICMLedger
from icm.deepseek_policy import AgentDeclaration


class ScriptedPolicy:
    """Deterministic stand-in for DeepSeekAgentPolicy: same interface."""

    def declare(self, task: str, prior_observations=None) -> AgentDeclaration:
        stateful = bool(prior_observations) and any(
            "green" in o for o in prior_observations)
        intent = "verify service health" if stateful else "run the task"
        statement = ("service will stay healthy given green probes"
                     if stateful else f"{task} will complete on the desired branch")
        return AgentDeclaration(
            intent=intent, statement=statement,
            branches={"desired": "clean", "feared": None, "default": "muddy"},
        )


def test_sample_outcome_matches_distributions():
    import random
    rng = random.Random(1)
    n = {"desired": 0}
    for _ in range(2000):
        if sample_outcome("healthy", rng) == "desired":
            n["desired"] += 1
    assert 0.80 <= n["desired"] / 2000 <= 0.90  # seeded env, ~0.85 target


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("agentpilot") / "ledger")
    summary = run_agent_pilot(db, ScriptedPolicy(), n_sessions=24, seed=11)
    return summary, db


def test_agent_pilot_harvest_flow(mock_run):
    summary, db = mock_run
    assert not summary.get("error")
    assert len(summary["planted"]) == len([s for s in range(24)])
    with ICMLedger(db) as led:
        exps = {e["exp_id"]: e for e in led.records("expectation")}
        e1 = [e for e in exps.values() if e["exp_id"].endswith("-e1")]
        e2 = [e for e in exps.values() if e["exp_id"].endswith("-e2")]
        assert e1 and e2  # both declarations harvested per session
        # E1 resolved on mid observation, E2 on the final outcome -- both stamped
        for e in e1 + e2:
            assert e["status"] == "resolved"
            assert e["resolution"]["congruence"] in ("confirmed", "violated")
        # intents carry the live (scripted) declarations: original aim, and the
        # state-informed refinement accumulated on the durable intent record
        aims = [i["aim"] for i in led.records("intent")]
        assert aims and all(a == "run the task" for a in aims)
        refined = [i for i in led.records("intent") if "verify service health" in i.get("intents", [])]
        assert refined  # at least one healthy-severity session refined its intent


def test_agent_pilot_gate_pass_with_planted_severity(mock_run):
    summary, db = mock_run
    with ICMLedger(db) as led:
        gate = compute_gate_metrics(
            led, planted={k: v[0] for k, v in summary["planted"].items()},
            level_map={"healthy": 0.0, "warning": 0.5, "broken": 1.0})
    assert gate["verdict"] == "PASS", gate["checks"]
    assert gate["checks"]["g4_signal_recovery"]["evaluated"]


def test_gate_rejects_unknown_planted_state(tmp_path):
    db = str(tmp_path / "unknown_sev")
    summary = run_agent_pilot(db, ScriptedPolicy(), n_sessions=6, seed=3)
    with ICMLedger(db) as led:
        with pytest.raises(KeyError):
            compute_gate_metrics(led, planted={k: "nonexistent" for k in summary["planted"]})