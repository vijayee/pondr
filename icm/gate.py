"""Phase-0 harvest gate: mechanical tests that congruence labels are usable.

The gate is the plan's Phase-0 GO/NO-GO, made concrete:

    g1 coverage        enough expectations actually resolve (not expire)
    g2 non-vacuity     BOTH classes present in resolved labels (the all-confirmed
                       trap -- passive runs produce nothing to learn from)
    g3 determinism     recomputing the rule on stored records reproduces every
                       stored label idempotently (twice)
    g4 signal recovery the label stream must recover planted ground truth:
                       per-intent violated rate correlates with the planted
                       world state (calm -> partial -> full drift)

Thresholds are intentionally conservative; a FAIL is information (it names the
specific failure of the harvest), never a fallback.
"""

from __future__ import annotations

import math

from icm.congruence import RULE_VERSION, Congruence, evaluate_congruence
from icm.ledger import ICMLedger

DEFAULT_THRESHOLDS = {
    "coverage_min": 0.60,          # share of expectations resolved (not expired/pending)
    "minority_share_min": 0.05,    # violated labels present (not all-confirmed)
    "majority_share_max": 0.95,    # confirmed labels not vacuous-all
    "recovery_corr_min": 0.80,     # corr(per-intent violated rate, planted drift)
}


def compute_gate_metrics(ledger: ICMLedger, planted: dict[str, str] | None = None,
                         thresholds: dict | None = None) -> dict:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    expectations = list(ledger.records("expectation"))
    intents = {r["intent_id"]: r for r in ledger.records("intent")}

    n_total = len(expectations)
    resolved = [e for e in expectations if e["status"] == "resolved"]
    expired = [e for e in expectations if e["status"] == "expired"]
    pending = [e for e in expectations if e["status"] == "pending"]
    coverage = len(resolved) / max(1, n_total)

    labels = {Congruence.CONFIRMED.value: 0, Congruence.VIOLATED.value: 0}
    for e in resolved:
        labels[e["resolution"]["congruence"]] += 1
    n_res = sum(labels.values())
    violated_share = labels[Congruence.VIOLATED.value] / max(1, n_res)

    # g2 both classes present
    nonvacuity = (labels[Congruence.CONFIRMED.value] > 0 and labels[Congruence.VIOLATED.value] > 0
                  and th["minority_share_min"] <= violated_share <= th["majority_share_max"])

    # g3 determinism: recompute from stored fields, twice
    mismatches = 0
    for e in resolved:
        out_id = e["resolution"]["outcome_id"]
        out = ledger.get("outcome", out_id)
        if out is None:
            mismatches += 1
            continue
        for _ in range(2):
            recomputed = evaluate_congruence(e["branches"], out["observed_branch"])
            if recomputed.congruence.value != e["resolution"]["congruence"] \
                    or recomputed.rule_version != e["resolution"].get("rule_version"):
                mismatches += 1
    determinism = (mismatches == 0 and n_res > 0
                   and all(e["resolution"].get("rule_version") == RULE_VERSION for e in resolved))

    # g4 signal recovery: per-intent violated rate vs planted drift level
    recovery = {"evaluated": False, "correlation": None}
    if planted is not None and n_res >= 4:
        level = {"calm": 0.0, "partial": 0.5, "full": 1.0}
        xs, ys = [], []
        for intent_id, world in planted.items():
            exps = [e for e in resolved if e["intent_id"] == intent_id]
            if not exps:
                continue
            v = sum(1 for e in exps if e["resolution"]["congruence"] == Congruence.VIOLATED.value)
            if world not in level:
                raise KeyError(f"unknown planted world state {world!r}")
            xs.append(level[world])
            ys.append(v / len(exps))
        recovery = {"evaluated": True, "n_intents": len(xs),
                    "correlation": _pearson(xs, ys)}
    corr = recovery.get("correlation")
    recovery_ok = bool(recovery["evaluated"] and corr is not None
                       and math.isfinite(corr) and corr >= th["recovery_corr_min"])

    checks = {
        "g1_coverage": {"ok": coverage >= th["coverage_min"], "value": round(coverage, 4),
                        "resolved": len(resolved), "expired": len(expired), "pending": len(pending)},
        "g2_nonvacuity": {"ok": bool(nonvacuity), "value": round(violated_share, 4),
                          "confirmed": labels[Congruence.CONFIRMED.value],
                          "violated": labels[Congruence.VIOLATED.value]},
        "g3_determinism": {"ok": bool(determinism), "value": mismatches, "n_recomputed": n_res * 2},
        "g4_signal_recovery": {**recovery, "ok": bool(recovery_ok)},
    }
    verdict = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    return {"verdict": verdict, "checks": checks, "thresholds": th, "n_intents": len(intents)}


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(a * a for a in dx)) * math.sqrt(sum(b * b for b in dy))
    if den == 0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / den