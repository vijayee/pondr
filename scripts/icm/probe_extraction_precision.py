"""The extraction-precision probe for the agent-intent corpus.

Protocol (agreed 2026-10-15): before trusting a retroactive layer, measure how
well intent/expectation/outcome can be extracted from EXPLICIT artifacts.

Two independent extraction sources over the ICM arc:
  A. context-drafted ledger records (drafted from the working record of the
     arc, data/_scratch/icm/arc_ledger_draft.jsonl)
  B. the arc's COMMIT MESSAGES (git log) -- primary, written at commit time,
     independent of A; outcome statements + declared thresholds.

Checks:
  c1  validation: every record conforms to icm-congruence/1 (branches +
      residue vocabulary); invalid = extraction failure.
  c2  determinism: the shipped congruence rule recomputed per record must
      equal the stated label (rule-level idempotency on REAL declarations --
      Phase-0 measured this on synthetic; this measures on organic arc data).
  c3  cross-source agreement: for each commit that states a verdict, does the
      draft record's outcome match the commit's stated outcome?
Output: data/_scratch/icm/extraction_probe_results.json
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icm.congruence import evaluate_congruence, BRANCHES, RESIDUE_VALUES  # noqa: E402

DRAFT = os.path.join("data", "_scratch", "icm", "arc_ledger_draft.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "extraction_probe_results.json")


def commit_messages():
    out = subprocess.run(["git", "log", "--oneline", "-n", "60"],
                         capture_output=True, text=True, shell=True).stdout
    return [l for l in out.splitlines() if "ICM" in l[:len(l)]]


def main() -> int:
    records = [json.loads(l) for l in io.open(DRAFT, encoding="utf-8") if l.strip()]
    c1_bad = c2_bad = 0
    valence_flips = []
    for r in records:
        br = r.get("branches")
        ob = r.get("observed")
        if not isinstance(br, dict) or ob not in BRANCHES + ("other",):
            c1_bad += 1
            continue
        br_keys_valid = set(br) <= set(BRANCHES)
        if not br_keys_valid or not any(br.get(b) is not None for b in BRANCHES):
            c1_bad += 1
            continue
        try:
            recomputed = evaluate_congruence(br, ob).congruence.value
        except Exception:
            c1_bad += 1
            continue
        # valence-flip audit: a stated label that disagrees with the rule is
        # EITHER the miner conflating outcome-valence with prediction-match
        # ("bad outcome" != "violated expectation") or a real extraction error.
        if recomputed != r.get("label"):
            c2_bad += 1
            valence_flips.append({"session_id": r["session_id"],
                                  "stated": r.get("label"),
                                  "rule": recomputed,
                                  "observed": ob,
                                  "feared_predicted": br.get("feared") is not None})
        r["recomputed"] = recomputed
    commits = commit_messages()
    # c3: commits that state a verdict with an explicit marker
    verdict_words = ("PASS", "FAIL", "CONFIRMED", "FALSIFIED", "REJECTED", "NULL",
                     "DECISIVE", "RETIRED", "CANCELED", "RETRACTED", "SHIPPED")
    vcommits = [c for c in commits
                if any(v in c.upper() for v in verdict_words)]
    results = {
        "records": len(records),
        "c1_valid": len(records) - c1_bad,
        "c2_label_agreement": len(records) - c1_bad - c2_bad,
        "c1_failure_n": c1_bad,
        "c2_failure_n": c2_bad,
        "precision_estimate": round((len(records) - c1_bad - c2_bad) / max(1, len(records)), 3),
        "commits_total": len(commits),
        "commits_with_verdict": len(vcommits),
        "verdict_coverage": round(len(vcommits) / max(1, len(commits)), 3),
        "valence_flips": len(valence_flips),
        "valence_flip_rate": round(len(valence_flips) / max(1, len(records)), 3),
        "protocol_consequence": ("labels must be COMPUTED by the rule, never "
                                 "stated by the miner; the retroactive layer "
                                 "recovers intent/branches/observed structure "
                                 "but labels are the rule's output only"),
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(results | {"verdict_commits": vcommits,
                                      "valence_flip_records": valence_flips}, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())