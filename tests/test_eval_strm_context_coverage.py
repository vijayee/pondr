"""Smoke test for the STRM Probe 1 context-coverage harness.

Asserts the report SHAPE + two mode-invariants, NOT a specific delta value (the
delta depends on the synthetic fact bank + the live heads, so pinning a number
would be brittle). Skipped when the trained backbone / 2a/2b/2c heads /
thresholds sidecar are absent or bge is not installed.

Invariants pinned:
  * permissive mode: every scored anchor is salient -> salience fires on every
    seeded fact (the fact has text -> r_i/rec_i/surprise_i all non-None -> the
    AND passes under theta=+inf/phi=-inf/surprise_cap=+inf).
  * salience MERGES fired episodes into the prompt-driven set (dedup, salience
    first) and never removes prompt-driven episodes -> coverage_on >= coverage_off
    in every mode (salience never hurts coverage, only helps or stays neutral).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_strm_context_coverage import ARTIFACTS_PRESENT, run_eval

_BGE_OK = True
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _BGE_OK = False

_SKIP = (not ARTIFACTS_PRESENT) or (not _BGE_OK)
_SKIP_REASON = ("needs the trained backbone + 2a/2b/2c heads + thresholds.json "
                "sidecar AND sentence_transformers installed")


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_permissive_report_shape_and_invariants():
    """M=2 permissive run -> well-formed report, salience fires on every fact,
    coverage_on >= coverage_off (salience only adds, never removes)."""
    report = run_eval(n_facts=2, gap=4, ring_capacity=8,
                      thresholds_mode="permissive", seed=0)
    # shape
    for key in ("n_facts", "gap", "ring_capacity", "thresholds", "seed",
                "coverage_on", "coverage_off", "coverage_delta", "rescued",
                "salience_fired_on_fact", "avg_salience_retrieval_count",
                "trials"):
        assert key in report, f"report missing {key}"
    assert report["n_facts"] == 2
    assert len(report["trials"]) == 2
    for t in report["trials"]:
        for key in ("fact_id", "summary", "query", "on", "off", "rescued"):
            assert key in t
        for cond in ("on", "off"):
            for key in ("fact_in_context", "salience_fired_on_fact",
                        "salience_retrieval_count", "n_signals", "signal_kinds"):
                assert key in t[cond]
    # invariant 1: permissive -> salience fires on every fact
    assert report["salience_fired_on_fact"] == report["n_facts"]
    # invariant 2: salience only adds -> ON coverage >= OFF coverage
    assert report["coverage_on"] >= report["coverage_off"]
    assert report["coverage_delta"] >= 0.0


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_real_thresholds_never_beat_permissive_firing():
    """The shipped thresholds are a SELECTIVE subset of permissive, so the real
    firing count is <= the permissive firing count. (The honest finding may be
    0 -- the shipped surprise_cap is very tight -- but it must never EXCEED the
    permissive upper bound.)"""
    real = run_eval(n_facts=2, gap=4, ring_capacity=8,
                    thresholds_mode="real", seed=0)
    perm = run_eval(n_facts=2, gap=4, ring_capacity=8,
                    thresholds_mode="permissive", seed=0)
    assert real["salience_fired_on_fact"] <= perm["salience_fired_on_fact"]
    # coverage_on >= coverage_off holds in real mode too (salience never hurts)
    assert real["coverage_on"] >= real["coverage_off"]