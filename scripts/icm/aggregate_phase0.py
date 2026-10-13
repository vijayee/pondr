"""Aggregate the three Phase-0 pilot tiers into one corpus summary.

    python scripts/icm/aggregate_phase0.py

Reads the three *_gate.json files under data/_scratch/icm/ and writes
phase0_summary.json with per-tier gate verdicts, violated-rate tables,
the pooled label-accounting, and the E1/E2 comparison at scale. The
aggregation is derived data only -- it must not invent metrics.
"""

from __future__ import annotations

import io
import json
import os

SCRATCH = os.path.join("data", "_scratch", "icm")
TIERS = {
    "synthetic": "phase0_gate.json",
    "agent_severity": "phase0_agent_gate.json",
    "sandbox_real": "phase0_sandbox_gate.json",
}
TIER_WORLD_ORDER = {
    "synthetic": ("calm", "partial", "full"),
    "agent_severity": ("healthy", "warning", "broken"),
    "sandbox_real": ("clean", "degraded", "hardfail"),
}


def main() -> int:
    summary = {"tiers": {}, "pooled": {}}
    total_resolved = total_violated = 0
    e1_all = e1v_all = e2_all = e2v_all = 0
    for tier, fname in TIERS.items():
        path = os.path.join(SCRATCH, fname)
        if not os.path.exists(path):
            summary["tiers"][tier] = {"missing": fname}
            continue
        d = json.load(io.open(path, encoding="utf-8"))
        gate = d["gate"]
        checks = gate["checks"]
        g4 = checks.get("g4_signal_recovery", {})
        world_rates = g4.get("group_rates") or {}
        order = TIER_WORLD_ORDER.get(tier, tuple(sorted(world_rates)))
        # gate JSONs for agent/sandbox also carry the E1/E2 reports
        report = d.get("e1_e2_report") or {}
        summary["tiers"][tier] = {
            "gate_verdict": gate["verdict"],
            "coverage": checks["g1_coverage"]["value"],
            "resolved": checks["g1_coverage"].get("resolved"),
            "violated_rate": checks["g2_nonvacuity"]["value"],
            "world_rates": {w: world_rates.get(w) for w in order},
            "e1_e2_report": report,
        }
        n_res = (checks["g1_coverage"].get("resolved") or 0)
        total_resolved += n_res
        total_violated += round(checks["g2_nonvacuity"]["value"] * n_res)
        if report:
            e1_all += report.get("e1_resolved") or 0
            e1v_all += report.get("e1_violated") or 0
            e2_all += report.get("e2_resolved") or 0
            e2v_all += report.get("e2_violated") or 0
    verdicts = [t["gate_verdict"] for t in summary["tiers"].values() if "gate_verdict" in t]
    if e1_all and e2_all:
        # McNemar-flavored same-session comparison is available only inside a
        # tier's ledger; keep the pooled report to rates (report-only)
        summary["pooled"]["e1_violated_rate"] = round(e1v_all / e1_all, 4)
        summary["pooled"]["e2_violated_rate"] = round(e2v_all / e2_all, 4)
        summary["pooled"]["e1_n"] = e1_all
        summary["pooled"]["e2_n"] = e2_all
        summary["pooled"]["e2_improvement_diff"] = round(e2v_all / e2_all - e1v_all / e1_all, 4)
    summary["pooled"]["total_resolved"] = total_resolved
    summary["pooled"]["total_violated_est"] = total_violated
    summary["verdicts_all_pass"] = bool(verdicts) and all(v == "PASS" for v in verdicts)
    path = os.path.join(SCRATCH, "phase0_summary.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())