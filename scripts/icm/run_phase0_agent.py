"""Run the Phase-0 AGENT pilot: DeepSeek-policy expectations over a hidden
world, harvested into the ledger, scored by the gate.

    python scripts/icm/run_phase0_agent.py [--sessions 30] [--seed 7] [--model ...]

DB lands in data/_scratch/icm/phase0agent/ (scratch; wiped per run). Costs one
relay call per declaration (~2/session) via the LOCAL Ollama endpoint.

Outputs the gate verdict plus the report-only E1->E2 prediction-improvement
observation (violated-rate with state evidence vs without).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icm.agent_pilot import SEV_LEVEL, run_agent_pilot  # noqa: E402
from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.gate import compute_gate_metrics  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402


def improvement_report(ledger) -> dict:
    """Report-only: does a state-informed re-prediction (E2) violate at or below
    the pre-action prediction (E1) for the same final outcome? Also reports the
    branch-coverage rate -- blanket hedging (predicting all three branches) makes
    the violated stream track only 'other' tail events, i.e. it carries almost
    no learnable signal."""
    e1 = e2 = e1v = e2v = 0
    b1 = b2 = 0
    for exp in ledger.records("expectation"):
        eid = exp["exp_id"]
        branches = sum(1 for b in ("desired", "feared", "default") if exp["branches"].get(b) is not None)
        if exp["status"] != "resolved":
            continue
        if eid.endswith("-e1"):
            e1 += 1
            b1 += branches
            e1v += exp["resolution"]["congruence"] == "violated"
        elif eid.endswith("-e2"):
            e2 += 1
            b2 += branches
            e2v += exp["resolution"]["congruence"] == "violated"
    return {"e1_resolved": e1, "e1_violated": e1v,
            "e2_resolved": e2, "e2_violated": e2v,
            "e1_violated_rate": round(e1v / e1, 4) if e1 else None,
            "e2_violated_rate": round(e2v / e2, 4) if e2 else None,
            "e1_avg_branches_predicted": round(b1 / e1, 2) if e1 else None,
            "e2_avg_branches_predicted": round(b2 / e2, 2) if e2 else None,
            "improved": None if not (e1 and e2) else e2v / e2 <= e1v / e1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--db-path", type=str,
                    default=os.path.join("data", "_scratch", "icm", "phase0agent"))
    args = ap.parse_args()

    if os.path.exists(args.db_path):
        shutil.rmtree(args.db_path)
    policy = DeepSeekAgentPolicy(model=args.model)
    print(f"[run] agent pilot: {args.sessions} sessions, model={args.model}", flush=True)
    t0 = time.time()
    summary = run_agent_pilot(args.db_path, policy, n_sessions=args.sessions, seed=args.seed)
    print(f"[run] harvested in {time.time()-t0:.0f}s", flush=True)
    if summary.get("error"):
        print("[FAIL] " + summary["error"])
        return 1
    with ICMLedger(args.db_path) as ledger:
        gate = compute_gate_metrics(ledger, planted={k: v[0] for k, v in summary["planted"].items()},
                                    level_map=SEV_LEVEL)
        report = improvement_report(ledger)
    out = {
        "pilot": {k: v for k, v in summary.items() if k != "planted"},
        "n_intents": len(summary["planted"]),
        "severity_mix": {w: sum(1 for s, _ in summary["planted"].values() if s == w)
                         for w in SEV_LEVEL},
        "gate": gate,
        "e1_e2_report": report,
    }
    path = os.path.join("data", "_scratch", "icm", "phase0_agent_gate.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())