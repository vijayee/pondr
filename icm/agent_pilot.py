"""Real-agent Phase-0 pilot: a DeepSeek-policy agent writing expectations over
a HIDDEN world, resolved by the harness.

Per session:
    1. policy.declare(task)          -> E1 (pre-action prediction, state hidden)
    2. harness emits a mid-run observation (partial env evidence)
    3. policy.declare(task, prior)   -> E2 (state-informed re-prediction)
    4. env samples the final outcome from the planted severity
    5. both E1 and E2 resolve against the same final outcome

Both expectations get congruence labels, which yields two reportable signals:
    * harvest quality (gate), as with the synthetic pilot
    * prediction improvement with state information: violated-rate(E2) should
      sit at-or-below violated-rate(E1). This is REPORTED, not gate-blocking --
      it is a hypothesis observation, not a requirement the pilot must hit.

Expiry accounts for a slice of sessions whose check window passes unobserved.
"""

from __future__ import annotations

import os
import random

from icm.instrument import ICMSession, OutcomeDecl
from icm.ledger import ICMLedger


def _slots_of(policy, decl, task):
    """Two-call slots: second extraction AFTER the declaration is recorded's
    shape was decided -- best effort, metadata only."""
    try:
        s = policy.declare_slots(decl.intent, decl.statement, task)
    except Exception:
        return None
    return s.to_slots() if s is not None else None


SEVERITIES = ("healthy", "warning", "broken")
SEV_LEVEL = {"healthy": 0.0, "warning": 0.5, "broken": 1.0}
SEV_OUTCOME_P = {
    "healthy": {"desired": 0.85, "default": 0.15},
    "warning": {"default": 0.55, "desired": 0.20, "feared": 0.25},
    "broken": {"other": 0.60, "feared": 0.25, "default": 0.15},
}
SEV_EVIDENCE = {
    "healthy": "All pre-flight probes returned green; the service responded normally.",
    "warning": "Pre-flight probes are returning intermittent errors; one check delayed.",
    "broken": "The primary probe failed and the service did not respond.",
}

TASKS = [
    "Restart the staging deployment and confirm it is healthy.",
    "Run the project's test suite and report failures.",
    "Migrate the database to the latest schema version.",
    "Run a performance benchmark on the search service.",
    "Archive logs older than 30 days on the build machine.",
    "Check disk usage and free space if low.",
    "Send the weekly report to the team channel.",
    "Reorder inventory parts below threshold.",
    "Clean stale Docker images on the CI runner.",
    "Rotate the service's API credentials.",
    "Update the documentation index page.",
    "Verify the nightly backup completed successfully.",
]


def sample_outcome(severity: str, rng: random.Random) -> str:
    roll = rng.random()
    acc = 0.0
    for branch, p in SEV_OUTCOME_P[severity].items():
        acc += p
        if roll < acc:
            return branch
    return next(iter(SEV_OUTCOME_P[severity]))  # float-rounding guard


def run_agent_pilot(db_path: str, policy, n_sessions: int = 30, seed: int = 7,
                    expire_frac: float = 0.10) -> dict:
    """policy: any object with .declare(task, prior_observations=None) -> AgentDeclaration."""
    rng = random.Random(seed)
    os.makedirs(db_path, exist_ok=True)
    ledger = ICMLedger(db_path)
    planted: dict[str, tuple[str, str]] = {}  # intent -> (severity, task)
    try:
        for si in range(n_sessions):
            task = rng.choice(TASKS)
            severity = rng.choice(SEVERITIES)
            session_id = f"asess-{si:03d}"
            sess = ICMSession(ledger, session_id)
            sess.begin(meta={"pilot": "agent", "severity": severity})
            intent_id = f"aint-{si:03d}"
            try:
                d1 = policy.declare(task)
            except Exception as e:
                # a declaration failure is a HARVEST failure -- record it as an
                # expired session so coverage accounting reflects it, then stop
                # the run (the gate must not silently eat a dead policy)
                sess.declare_intent(f"[unresolvable] {task}", intent_id=intent_id)
                ledger.expire_pending(now_ts=1e18, current_step=10)
                sess.finish(close_status="abandoned")
                return {"sessions": n_sessions, "planted": planted, "seed": seed,
                        "error": f"policy declare failed at session {si}: {e!r}"}
            sess.declare_intent(d1.intent, intent_id=intent_id, slots=_slots_of(policy, d1, task))
            planted[intent_id] = (severity, task)

            if rng.random() < expire_frac:
                e1 = sess.expect(d1.to_expectation_decl(intent_id),
                                 exp_id=f"aexp-{si:03d}-x")
                _set_horizon(ledger, e1, {"kind": "step", "deadline_step": 1})
                sess.step()
                sess.step()
                sess.expire_pending(now_ts=1e18)
                sess.finish()
                continue

            sess.step()
            e1_id = sess.expect(d1.to_expectation_decl(intent_id), exp_id=f"aexp-{si:03d}-e1")
            sess.step()

            d2 = policy.declare(task, prior_observations=[SEV_EVIDENCE[severity]])
            e2_id = sess.expect(d2.to_expectation_decl(intent_id), exp_id=f"aexp-{si:03d}-e2")
            _record_intent_update(ledger, intent_id, d2.intent)

            final_branch = sample_outcome(severity, rng)
            outcome_text = f"{task} ended on branch: {final_branch}"
            sess.step()
            # BOTH declarations resolve against the SAME final outcome: E1 vs E2
            # then differ only in the information available at declaration time
            # (none vs mid-run evidence) -- that is the comparison's whole point.
            sess.resolve(OutcomeDecl(e1_id, outcome_text, final_branch, "env"),
                         outcome_id=f"aout-{si:03d}-e1")
            sess.resolve(OutcomeDecl(e2_id, outcome_text, final_branch, "env"),
                         outcome_id=f"aout-{si:03d}-e2")
            sess.finish()
    finally:
        ledger.close()
    return {"sessions": n_sessions, "planted": planted, "seed": seed}


def _set_horizon(ledger, exp_id: str, horizon: dict) -> None:
    exp = ledger.get("expectation", exp_id)
    exp["horizon"] = horizon
    ledger.put("expectation", exp)


def _record_intent_update(ledger, intent_id: str, new_intent: str) -> None:
    """A state-informed re-declaration may refine the WHY; accumulate on the
    durable intent record without overwriting the original aim."""
    if not new_intent:
        return
    rec = ledger.get("intent", intent_id)
    intents = rec.setdefault("intents", [rec["aim"]])
    if new_intent not in intents:
        intents.append(new_intent)
    ledger.put("intent", rec)