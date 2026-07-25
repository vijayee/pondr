#!/usr/bin/env python
"""PROBE (not committed): scale guard-coverage check on the 200 adjudication
pairs in data/training/bonsai/contradiction_pairs.jsonl.

Exercises the SHIPPED deterministic guards + v6 LoRA decider at scale, WITHOUT
extraction (pairs carry pre-built state_values in their spec). This is the
binding production-soundness check the single dogfood + 16-pair fixture can only
sample: the non-real shapes (same_value / different_entity / complementary_
temporal) must NEVER false-tombstone, and the guards must short-circuit them
BEFORE the LLM (no HTTP) -- correct-by-construction.

Per pair:
  - reconstruct state_values from the spec (entity, old_value, new_value,
    old_path, new_path). different_entity uses one value (old==new).
  - call the real BonsaiDecider.decide_contradiction (guards + v6 LoRA on
    :8080). The guards run first; same_value/different_entity/complementary_
    temporal short-circuit (ask_user or dismiss, NO HTTP); real hits the LoRA.
  - correct = (real -> fix+supersede_assertion) AND (non-real -> NOT fix+
    supersede_assertion, i.e. non-mutating: ask_user OR dismiss both count).

The 59 real pairs are the v6 training set (circular for the LoRA -- noted), but
the 141 non-real pairs test the GUARDS (deterministic, training-independent):
a false-fix on any of them is a real regression.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gnn.bonsai_decider import BonsaiDecider, _deterministic_non_conflict  # noqa: E402

PAIRS = ROOT / "data" / "training" / "bonsai" / "contradiction_pairs.jsonl"


def state_values_from_spec(spec):
    ct = spec.get("conflict_type")
    old_path = spec.get("old_path") or ""
    new_path = spec.get("new_path") or ""
    if ct == "different_entity":
        # single value asserted for two different entities -> equal values
        v = spec.get("value", "")
        return [{"value": v, "asserted_by": old_path, "asserted_at": "2026-07-01",
                 "source_path": old_path},
                {"value": v, "asserted_by": new_path, "asserted_at": "2026-07-05",
                 "source_path": new_path}]
    ov = spec.get("old_value", "")
    nv = spec.get("new_value", "")
    return [{"value": ov, "asserted_by": old_path, "asserted_at": "2026-07-01",
             "source_path": old_path},
            {"value": nv, "asserted_by": new_path, "asserted_at": "2026-07-05",
             "source_path": new_path}]


def main():
    rows = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines()]
    adj = [r for r in rows if r.get("task") == "adjudication"]
    print(f"Loaded {len(adj)} adjudication pairs from {PAIRS.name}")
    dec = BonsaiDecider(timeout=90.0, max_tokens=768)
    print(f"Decider: {dec.endpoint} model={dec.model} health={dec.health_check(5.0)}\n")

    by_type = {}
    for r in adj:
        ct = r["spec"]["conflict_type"]
        svs = state_values_from_spec(r["spec"])
        # does the guard short-circuit (no HTTP)?
        guard = _deterministic_non_conflict(svs)
        guard_fired = guard is not None
        try:
            d = dec.decide_contradiction({"node": r["spec"].get("entity", "E:x"),
                                          "type": "contradictory_state"},
                                         {"state_values": svs})
        except Exception as e:  # noqa: BLE001
            d = None
        decision = d.get("decision") if d else None
        action = (d.get("action") or "") if d else ""
        is_fix = decision == "fix" and "supersede_assertion" in action
        # correctness: real -> fix ; non-real -> NOT fix (non-mutating)
        if ct == "real":
            correct = is_fix
        else:
            correct = not is_fix
        by_type.setdefault(ct, {"n": 0, "correct": 0, "guard_fired": 0,
                                "false_fix": 0, "decisions": Counter()})
        t = by_type[ct]
        t["n"] += 1
        t["correct"] += int(correct)
        t["guard_fired"] += int(guard_fired)
        if is_fix and ct != "real":
            t["false_fix"] += 1
        t["decisions"][decision or "None"] += 1

    print("=" * 70)
    print(f"{'conflict_type':22s} {'n':>3s} {'correct':>7s} {'guard':>5s} "
          f"{'false_fix':>9s} {'decisions'}")
    print("=" * 70)
    total_n = total_c = total_ff = 0
    for ct in ("real", "complementary_temporal", "same_value", "different_entity"):
        t = by_type.get(ct, {"n": 0, "correct": 0, "guard_fired": 0,
                             "false_fix": 0, "decisions": {}})
        total_n += t["n"]; total_c += t["correct"]; total_ff += t["false_fix"]
        print(f"{ct:22s} {t['n']:3d} {t['correct']:7d} "
              f"{t['guard_fired']:5d} {t['false_fix']:9d}   "
              f"{dict(t['decisions'])}")
    print("-" * 70)
    print(f"{'TOTAL':22s} {total_n:3d} {total_c:7d} {'':5s} {total_ff:9d}")
    print(f"\nOverall correct: {total_c}/{total_n} = {total_c/total_n:.1%}")
    print(f"Non-real false-fix (regression): {total_ff}  (MUST be 0)")
    n_real = by_type.get("real", {}).get("n", 0)
    print(f"Guard short-circuits (no HTTP) on non-real: "
          f"{sum(by_type[c]['guard_fired'] for c in by_type if c != 'real')}/"
          f"{sum(by_type[c]['n'] for c in by_type if c != 'real')}")


if __name__ == "__main__":
    main()