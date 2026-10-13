"""Scaffold A/B arm B: does propositional intent structure exist WITHOUT a
leading schema? Free-form declarations, then pass-2 S-P-O extraction.

    python scripts/icm/run_freeform_arm.py [--sessions 40]

Arm A (imposed declarative schema) stats are read from spo_audit.json.
This script produces the arm-B counterpart: raw prose per session, its
pass-2 extraction, residue rates, subject-SELF rate -- plus examples so a
reader can judge the prose quality directly.

No outcomes are resolved in this arm: its question is structure naturalness,
not label quality. A later hybrid arm (freeform + elicited branches) would
compare violated rates across scaffolds.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.spo_extract import extract_primary  # noqa: E402

SCRATCH = os.path.join("data", "_scratch", "icm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args()

    policy = DeepSeekAgentPolicy(model=args.model)
    # session tasks: sample from the tasks the other pilots used
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "agent_pilot", os.path.join("icm", "agent_pilot.py"))
    ap_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap_mod)
    tasks = [ap_mod.TASKS[i % len(ap_mod.TASKS)] for i in range(args.sessions)]

    prose_results = []
    extracts = []
    errors = 0
    for i, task in enumerate(tasks[:args.samples]):
        t0 = time.time()
        try:
            prose = policy.declare_freeform(task)
        except Exception as e:
            errors += 1
            prose_results.append({"session": i, "task": task, "error": f"{type(e).__name__}: {e}"})
            continue
        prose_results.append({"session": i, "task": task, "prose": prose,
                              "chars": len(prose), "latency": round(time.time() - t0, 1)})
        try:
            ex = extract_primary(policy, prose, aim='')  # the prose IS the why
            ex["session"] = i
            extracts.append(ex)
        except Exception as e:
            extracts.append({"session": i, "error": f"{type(e).__name__}: {e}"})

    full = sum(1 for e in extracts if e.get("full"))
    n = len(extracts)
    residues = Counter(e.get("residue") or "none" for e in extracts if "error" not in e)
    self_rate = sum(1 for e in extracts if e.get("subject_self")) / max(1, n)
    out = {
        "model": args.model,
        "arm_b": {
            "n_prose": len([p for p in prose_results if "prose" in p]),
            "errors": errors,
            "avg_prose_chars": round(sum(p.get("chars", 0) for p in prose_results) / max(1, len(prose_results)), 1),
            "n_extracts": n,
            "full_extraction": full,
            "full_rate": round(full / max(1, n), 3),
            "subject_self_rate": round(self_rate, 3),
            "residue": dict(residues),
            "extract_errors": sum(1 for e in extracts if "error" in e),
        },
        "extract_detail": extracts,
        "prose_detail": prose_results,
        "examples": [{"session": p["session"], "prose": p.get("prose", p.get("error")),
                      "extract": next((e for e in extracts if e.get("session") == p["session"]), None)}
                     for p in prose_results[:6]],
    }
    # arm A comparison if spo_audit.json exists
    a_path = os.path.join(SCRATCH, "spo_audit.json")
    if os.path.exists(a_path):
        a = json.load(io.open(a_path, encoding="utf-8"))["tiers"]
        out["arm_a_imposed"] = {k: {f: v.get(f) for f in
                                    ("full_rate", "subject_self_rate", "residue", "n")}
                                for k, v in a.items() if isinstance(v, dict)}
    path = os.path.join(SCRATCH, "freeform_arm.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))
    print("[examples]")
    for p in out["examples"]:
        print(f"  session {p['session']}: {str(p['prose'])[:220]}")
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())