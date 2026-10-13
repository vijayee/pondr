"""Run the ICM Phase-0 pilot harvest and print the gate verdict.

    python scripts/icm/run_phase0_pilot.py [--sessions 40] [--seed 7] [--db-path ...]

DB lands in data/_scratch/icm/phase0/ by default (scratch; never committed).
The gate is the plan's Phase-0 GO/NO-GO: coverage / non-vacuity / determinism /
signal recovery. A FAIL names the failed check -- it is information, not a
fallback.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

torch.set_num_threads(1)

from icm.gate import compute_gate_metrics  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402
from icm.pilot import run_pilot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--db-path", type=str,
                    default=os.path.join("data", "_scratch", "icm", "phase0"))
    args = ap.parse_args()

    # scratch db: a fresh run replaces it (phase-0 pilots are deterministic
    # per seed; keeping stale ledgers would merge sessions under explicit ids)
    import shutil

    if os.path.exists(args.db_path):
        shutil.rmtree(args.db_path)

    summary = run_pilot(args.db_path, n_sessions=args.sessions, seed=args.seed)
    with ICMLedger(args.db_path) as ledger:
        gate = compute_gate_metrics(ledger, planted=summary["planted"])

    out = {
        "pilot": {k: v for k, v in summary.items() if k != "planted"},
        "n_intents": len(summary["planted"]),
        "world_mix": {w: sum(1 for x in summary["planted"].values() if x == w)
                      for w in ("calm", "partial", "full")},
        "gate": gate,
    }
    with io.open(os.path.join("data", "_scratch", "icm", "phase0_gate.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())