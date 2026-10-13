"""Run the Phase-0 SANDBOX pilot: live DeepSeek expectations about real,
observed tool outcomes (flaky probe + manifest verify), gated on harvest.

    python scripts/icm/run_phase0_sandbox.py [--sessions 16] [--seed 7]

DB: data/_scratch/icm/phase0sandbox/ (scratch, wiped per run); sandbox files
under data/_scratch/icm/sandbox/ (wiped by the pilot's finally).

The sandbox is one real world (no planted per-session severities), so g4's
recovery ground truth is the session's OBSERVED event class
(clean=0.0 / degraded=0.5 / hardfail=1.0) -- a monotone-violation check
against real outcomes rather than planted ones.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.gate import compute_gate_metrics  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402
from icm.sandbox_pilot import run_sandbox_pilot  # noqa: E402

LEVEL_MAP = {"clean": 0.0, "degraded": 0.5, "hardfail": 1.0, "other": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probe-fail", type=float, default=0.35)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--db-path", type=str,
                    default=os.path.join("data", "_scratch", "icm", "phase0sandbox"))
    args = ap.parse_args()

    if os.path.exists(args.db_path):
        shutil.rmtree(args.db_path)
    policy = DeepSeekAgentPolicy(model=args.model)
    sandbox_root = os.path.join("data", "_scratch", "icm", "sandbox")
    print(f"[run] sandbox pilot: {args.sessions} sessions, model={args.model}, "
          f"probe_fail_p={args.probe_fail}", flush=True)
    t0 = time.time()
    summary = run_sandbox_pilot(args.db_path, policy, sandbox_root,
                                n_sessions=args.sessions, seed=args.seed,
                                probe_fail_p=args.probe_fail)
    print(f"[run] harvested in {time.time()-t0:.0f}s", flush=True)

    event_by_session = {r["session"]: r["event_class"] for r in summary["session_log"]}
    with ICMLedger(args.db_path) as ledger:
        exps = list(ledger.records("expectation"))
        e1 = [e for e in exps if e["exp_id"].endswith("-e1") and e["status"] == "resolved"]
        e2 = [e for e in exps if e["exp_id"].endswith("-e2") and e["status"] == "resolved"]

        # planted map: intent -> OBSERVED event class (the real world it ran in)
        planted = {e["intent_id"]: event_by_session.get(e["session_id"], "other")
                   for e in exps}
        gate = compute_gate_metrics(ledger, planted=planted, level_map=LEVEL_MAP)

        def stats(recs, bkey: str) -> tuple:
            if not recs:
                return None, None, None
            v = sum(1 for e in recs if e["resolution"]["congruence"] == "violated")
            b = sum(1 for e in recs if e["branches"].get(bkey) is not None)
            w = sum(1 for e in recs
                    for bk in ("desired", "feared", "default")
                    if e["branches"].get(bk) is not None)
            return round(v / len(recs), 4), round(b / len(recs), 2), round(w / len(recs), 2)

        e1v, e1f, e1w = stats(e1, "feared")
        e2v, e2f, e2w = stats(e2, "feared")
        branch_mix = dict(Counter(e["resolution"]["observed_branch"] for e in e1))
        out = {
            "pilot": {"sessions": args.sessions, "seed": args.seed,
                      "probe_fail_p": args.probe_fail, "model": args.model},
            "event_mix": dict(Counter(r["event_class"] for r in summary["session_log"])),
            "gate": gate,
            "e1_e2_report": {
                "e1_resolved": len(e1), "e1_violated_rate": e1v,
                "e2_resolved": len(e2), "e2_violated_rate": e2v,
                "feared_bet_rate_e1": e1f, "feared_bet_rate_e2": e2f,
                "avg_branches_predicted_e1": e1w, "avg_branches_predicted_e2": e2w,
                "improved": None if (e1v is None or e2v is None) else e2v <= e1v,
            },
        }
    path = os.path.join("data", "_scratch", "icm", "phase0_sandbox_gate.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["gate"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())