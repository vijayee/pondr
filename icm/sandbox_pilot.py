"""Sandbox pilot: expectations about the agent's REAL tool outcomes.

Per session (task = ingest a fixture, manifest it, verify):
    1. policy.declare(task)                    -> E1 (pre-run, no evidence)
    2. probe_remote() runs for real; the RESULT is shown to the agent
    3. policy.declare(task, evidence)          -> E2 (state-informed re-prediction)
    4. manifest + verify run for real
    5. classify_run(...) -> event class -> branch; BOTH E1 and E2 resolve

This is the Phase-0 bridge: predictions are about real observed tool
behavior with real stochasticity -- not planted branch labels. The env's
flaky probe (~35% seeded failure, one auto-retry) supplies genuine
uncertainty so the violated stream is non-vacuous.
"""

from __future__ import annotations

import os
import random

from icm.instrument import ICMSession, OutcomeDecl
from icm.ledger import ICMLedger
from icm.sandbox_env import SandboxEnv, BRANCH_FOR_EVENT, FIXTURES, classify_run

TASK_TEMPLATE = "Ingest dataset {name} into the sandbox, write its manifest, and verify the checksum."


def run_sandbox_pilot(db_path: str, policy, sandbox_root: str, n_sessions: int = 16,
                      seed: int = 7, probe_fail_p: float = 0.35) -> dict:
    """policy: object with .declare(task, prior_observations=None) -> AgentDeclaration."""
    rng = random.Random(seed)
    env = SandboxEnv(sandbox_root, seed=seed, probe_fail_p=probe_fail_p)
    os.makedirs(db_path, exist_ok=True)
    ledger = ICMLedger(db_path)
    session_log: list[dict] = []
    try:
        names = list(FIXTURES)
        for si in range(n_sessions):
            name = names[si % len(names)]
            task = TASK_TEMPLATE.format(name=name)
            session_id = f"sbx-{si:03d}"
            sess = ICMSession(ledger, session_id)
            sess.begin(meta={"pilot": "sandbox", "fixture": name})

            d1 = policy.declare(task)
            intent_id = f"int-{si:03d}"
            sess.declare_intent(d1.intent, intent_id=intent_id)
            sess.step()
            e1_id = sess.expect(d1.to_expectation_decl(intent_id), exp_id=f"sx-{si:03d}-e1")

            session_dir = env.materialize_fixture(name)
            sess.step()
            probe1 = env.probe_remote()
            # harness retries ONCE on a real failure -- the retry is observed
            if probe1["status"] == "error":
                probe2 = dict(env.probe_remote())
                probe2["retried"] = True
                if probe2["status"] == "error":
                    probe2["fatal"] = True  # retry also failed -> hard failure
                probe_events = [probe1, probe2]
                evidence = [f"dependency probe FAILED first try ({probe1['detail']}); "
                            + ("retry succeeded" if probe2["status"] == "ok" else "retry ALSO failed; proceeding degraded")]
            else:
                probe_events = [probe1]
                evidence = [f"dependency probe ok (latency {probe1['latency_ms']}ms)"]

            d2 = policy.declare(task, prior_observations=evidence)
            e2_id = sess.expect(d2.to_expectation_decl(intent_id), exp_id=f"sx-{si:03d}-e2")
            # intent refinement, same accumulator the agent pilot uses
            rec = ledger.get("intent", intent_id)
            intents_list = rec.setdefault("intents", [rec["aim"]])
            if d2.intent not in intents_list:
                intents_list.append(d2.intent)
            ledger.put("intent", rec)

            sess.step()
            m1 = env.write_manifest(session_dir, name)
            m2 = env.verify_manifest(session_dir, name)
            manifest_events = [m1, m2]
            event_class = classify_run(probe_events, manifest_events)
            branch = BRANCH_FOR_EVENT[event_class]
            outcome_text = (f"probe={probe_events[0]['status']}"
                            f"{'+retry:' + probe_events[1]['status'] if len(probe_events) > 1 else ''}, "
                            f"manifest={m1['status']}/{m2['status']} -> {event_class}")
            if m2.get("sha"):
                outcome_text += f", sha={m2['sha']}"

            sess.step()
            sess.resolve(OutcomeDecl(e1_id, outcome_text, branch, "env"),
                         outcome_id=f"sxo-{si:03d}-e1")
            sess.resolve(OutcomeDecl(e2_id, outcome_text, branch, "env"),
                         outcome_id=f"sxo-{si:03d}-e2")
            sess.finish()
            session_log.append({"session": session_id, "event_class": event_class,
                                "branch": branch, "probe_steps": len(probe_events)})
    finally:
        ledger.close()
        env.wipe()
    return {"sessions": n_sessions, "seed": seed, "session_log": session_log}