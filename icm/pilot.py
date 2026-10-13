"""Synthetic Phase-0 pilot: a deterministic scripted agent + environment.

The pilot's job is NOT to imitate an agent faithfully -- it is to prove the
harvest path end-to-end: intent declaration -> expectation declaration ->
outcome observation -> congruence stamp -> gate metrics, including RECOVERY of
a planted ground-truth signal (g4). Everything is seeded for reproducibility.

World design (deliberately simple):
    4 intent archetypes; each session runs one.
    Environment states per session, seeded at session creation:
        "calm"        -> outcomes land on predicted branches (confirmed)
        "partial"     -> half the steps end in an unpredicted branch ("other")
        "full"        -> nearly all steps end unpredicted
    The session state is the planted ground truth the gate must recover from
    the harvested congruence labels alone.
    A small fraction of sessions declare expectations with horizons that pass
    unobserved -> expiry (coverage accounting, no fabricated labels).

Expectation policy (the pilot's stand-in for a real agent policy): predicts
{desired, default} -- selective, not exhaustive, so "other" outcomes are
genuinely unpredicted.
"""

from __future__ import annotations

import os
import random

from icm.instrument import ExpectationDecl, ICMSession, OutcomeDecl
from icm.ledger import ICMLedger

ARCHETYPES = {
    "deploy": {"aim": "deploy the service to staging",
               "desired": "pipeline finishes green",
               "default": "pipeline stalls and requires a retry"},
    "fixtest": {"aim": "stabilize the flaky integration test",
                "desired": "test passes 20 consecutive runs",
                "default": "test still fails intermittently"},
    "campaign": {"aim": "draft the weekly email campaign",
                 "desired": "campaign scheduled for Friday 9am",
                 "default": "draft saved but unscheduled"},
    "reorder": {"aim": "reorder low inventory parts",
                "desired": "PO submitted for all low parts",
                "default": "PO partially submitted (approval missing)"},
}
WORLD_STATES = ("calm", "partial", "full")
DRIFT_OTHER_P = {"calm": 0.0, "partial": 0.5, "full": 0.9}


def run_pilot(db_path: str, n_sessions: int = 40, seed: int = 7,
              expire_frac: float = 0.15, n_expectations: int = 3) -> dict:
    rng = random.Random(seed)
    os.makedirs(db_path, exist_ok=True)
    ledger = ICMLedger(db_path)
    planted: dict[str, str] = {}
    try:
        for si in range(n_sessions):
            session_id = f"psess-{si:03d}"
            sess = ICMSession(ledger, session_id)
            sess.begin(meta={"pilot_seed": seed})
            _, tpl = rng.choice(list(ARCHETYPES.items()))
            world = rng.choice(WORLD_STATES)
            intent_id = sess.declare_intent(tpl["aim"], intent_id=f"int-{si:03d}")
            planted[intent_id] = world
            other_p = DRIFT_OTHER_P[world]
            if rng.random() < expire_frac:
                # expiry accounting: horizon passes unobserved -> no label.
                # advance the logical clock PAST the deadline first, then drain.
                sess.expect(ExpectationDecl(
                    intent_id=intent_id,
                    statement=f"{tpl['aim']} will yield: {tpl['desired']}",
                    branches={"desired": tpl["desired"], "feared": None, "default": tpl["default"]},
                    horizon={"kind": "step", "deadline_step": 1},
                ), exp_id=f"exp-{si:03d}-x")
                sess.step()
                sess.step()
                sess.expire_pending(now_ts=1e18)
                sess.finish()
                continue
            for k in range(n_expectations):
                sess.step()
                exp_id = sess.expect(ExpectationDecl(
                    intent_id=intent_id,
                    statement=f"{tpl['aim']} (check {k}) will yield: {tpl['desired']}",
                    branches={"desired": tpl["desired"], "feared": None, "default": tpl["default"]},
                    horizon={"kind": "manual"},
                ), exp_id=f"exp-{si:03d}-{k}")
                if rng.random() < other_p:
                    sess.resolve(OutcomeDecl(
                        exp_id, "an unpredicted side effect blocked the check", "other", "env"),
                        outcome_id=f"out-{si:03d}-{k}")
                else:
                    branch = "desired" if rng.random() < 0.8 else "default"
                    sess.resolve(OutcomeDecl(
                        exp_id, f"check {k} ended on branch {branch}", branch, "env"),
                        outcome_id=f"out-{si:03d}-{k}")
            sess.finish()
    finally:
        ledger.close()
    return {"sessions": n_sessions, "planted": planted, "seed": seed}