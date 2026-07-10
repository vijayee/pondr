"""Summarize a consolidation report: per-head counts + a threshold sweep.

The consolidation loop (``src/gnn/consolidate.py``) records ``score_distributions``
-- 100 binned histograms (width 0.01 over [0,1]) of the raw ontology, link-pred, and
per-edge-max-salience scores. This script turns one report into:

  (a) per-head counts AT the thresholds the run used (the ``*_proposed``/``pruned``
      list lengths), and
  (b) a threshold SWEEP from the histograms -- how many ontology/link-pred proposals
      would survive at each accept threshold, and what prune-fraction each salience
      threshold implies -- so alternatives can be compared WITHOUT re-running.

Why this works without re-running: a threshold on the SAME scored set just moves the
cutoff. ``count(score >= t)`` at a 0.01-bin boundary ``t = k/100`` is exactly the sum
of buckets ``[k..99]`` (bucket i covers ``[i/100, (i+1)/100)``). The 0.01 resolution
makes 0.05/0.15/0.85 exact (10 width-0.1 bins could not resolve these, and
``int(round(0.05*10))`` hit Python banker's rounding to 0). The ontology STRATEGY
(all/topk/rotation) changes WHICH pairs get scored, so comparing strategies needs a
re-run per strategy -- but once a strategy's report exists, its threshold sweep is
free.

Usage::

    python scripts/summarize_consolidation.py --report data/pod_runs/phase3a/consol_all.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Histograms are 100 width-0.01 buckets, so these 0.01-boundary thresholds are
# EXACT: count(>= t) = sum of buckets from floor(t*100) onward, count(< t) = sum
# before it. (With 10 width-0.1 bins, 0.05/0.15/0.85 would be unresolvable and
# int(round(0.05*10)) hit Python banker's rounding to 0 -- the 100-bin resolution
# fixes both.)
ACCEPT_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
# Salience is 99.9%-sparse, so the interesting prune range is low. Prune iff the
# per-edge MAX endpoint salience < thr.
PRUNE_SWEEP = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]


def _count_above(hist: list[int], t: float) -> int:
    """count(score >= t) from a histogram, for t at a bin boundary (width 1/n)."""
    n = len(hist)
    k = int(round(t * n))  # boundary bucket index
    return sum(hist[k:])


def _count_below(hist: list[int], t: float) -> int:
    """count(score < t) from a histogram, for t at a bin boundary (width 1/n)."""
    n = len(hist)
    k = int(round(t * n))
    return sum(hist[:k])


def _sweep_table(title: str, hist: list[int], thresholds, above: bool) -> None:
    if not hist:
        print(f"\n--- {title} (no data) ---")
        return
    total = sum(hist)
    print(f"\n--- {title} (total scored >= collect-bar: {total}) ---")
    print("  threshold | count  | fraction")
    for t in thresholds:
        n = _count_above(hist, t) if above else _count_below(hist, t)
        frac = (n / total) if total else 0.0
        rel = ">=" if above else "<"
        print(f"  {t:>5.2f}    | {n:>6d} | {frac:6.2%}   (score {rel} {t})")
    # Bin detail: the resolution (100 width-0.01 buckets) -- the full list is too
    # long to print, so show the count of NON-empty buckets as a sparsity hint.
    nonzero = sum(1 for c in hist if c)
    print(f"  ({len(hist)} width-{1/len(hist):.3f} buckets; {nonzero} non-empty)")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Summarize a consolidation report + sweep thresholds from histograms.")
    p.add_argument("--report", required=True, help="Path to a consolidation report JSON.")
    args = p.parse_args()

    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))

    print("=== top-level ===")
    for k in ["dry_run", "trained", "subgraphs_scored",
             "verifier_calls", "verifier_accepted", "verifier_validation_rate"]:
        print(f"  {k} = {rep.get(k)}")

    print("\n=== per-head counts (AT the run's configured thresholds) ===")
    for k in ["abstracts", "edges_proposed", "edges_accepted", "edges_unverified",
              "anomalies", "ontology_proposed", "pruned"]:
        print(f"  len({k}) = {len(rep.get(k, []))}")

    dist = rep.get("score_distributions")
    if not dist:
        print("\n(no score_distributions in this report -- run a current "
              "run_consolidation.py to get histograms)")
        return 0

    print("\n=== threshold sweep (from score_distributions histograms) ===")
    print("count(score >= t) -- how many proposals survive at each accept threshold:")
    _sweep_table("ontology scores", dist.get("ontology", []),
                 ACCEPT_SWEEP, above=True)
    _sweep_table("linkpred scores", dist.get("linkpred", []),
                 ACCEPT_SWEEP, above=True)

    print("\ncount(per-edge max salience < t) -- prune-fraction at each prune threshold:")
    sal = dist.get("salience_endpoint", [])
    _sweep_table("salience per-edge max", sal, PRUNE_SWEEP, above=False)
    total = sum(sal)
    if total:
        print("  (survive = 1 - prune-fraction; the run's configured prune threshold")
        print("   gave the pruned-list length shown above.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())