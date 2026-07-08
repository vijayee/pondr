"""Offline tests for the Phase 2c Presentation Gate (axis a: chunking).

Pure-logic — no torch, no model. Verifies the heuristic strategy selection
(direct / chunked / summary_only), determinism, the outcome ReplayBuffer, and
the chunker-cfg wiring. Axis (b) end-state tests live in test_end_state.py.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.config import Phase2cConfig
from src.subconscious.presentation_gate import (
    CHUNKED, DIRECT, SUMMARY_ONLY,
    PresentationGate, PresentationOutcome, PresentationPlan, ReplayBuffer,
)


def _gate(direct_max=3, summary_min=20, max_primary_chunks=5) -> PresentationGate:
    cfg = Phase2cConfig()
    cfg.presentation_gate.direct_max_episodes = direct_max
    cfg.presentation_gate.summary_only_min_episodes = summary_min
    g = PresentationGate(cfg)
    # Simulate the orchestrator wiring the chunker's primary-chunk cap.
    g.set_chunker_cfg(SimpleNamespace(max_primary_chunks=max_primary_chunks))
    return g


def _eps(n: int) -> list[dict]:
    return [{"episode_id": f"e{i}", "score": 1.0} for i in range(n)]


# ── strategy selection ──

def test_direct_for_small_specific_query():
    g = _gate()
    plan = g.plan("What did Alice say about Postgres?", _eps(2))
    assert plan.strategy == DIRECT
    assert plan.primary_chunk_count == 2
    assert plan.compressed_chunk_count == 0


def test_chunked_for_mid_size_specific_query():
    g = _gate()
    plan = g.plan("What have we discussed about performance?", _eps(12))
    assert plan.strategy == CHUNKED
    assert plan.primary_chunk_count == 5
    assert plan.compressed_chunk_count == 7


def test_summary_only_for_summarization_verb():
    g = _gate()
    plan = g.plan("Summarize everything about databases", _eps(8))
    assert plan.strategy == SUMMARY_ONLY
    assert plan.compressed_chunk_count == 8


def test_summary_only_for_large_episode_count():
    g = _gate()
    plan = g.plan("What did Alice say about Postgres?", _eps(30))
    assert plan.strategy == SUMMARY_ONLY


def test_direct_threshold_boundary():
    g = _gate(direct_max=3)
    # 3 episodes + specific → direct (boundary inclusive).
    assert g.plan("What did Alice say?", _eps(3)).strategy == DIRECT
    # 4 episodes → no longer direct (mid-range, specific enough) → chunked.
    assert g.plan("What did Alice say?", _eps(4)).strategy == CHUNKED


def test_empty_episodes_is_direct():
    g = _gate()
    plan = g.plan("anything", [])
    assert plan.strategy == DIRECT
    assert plan.primary_chunk_count == 0


def test_primary_chunk_count_capped_by_chunker_cfg():
    g = _gate(max_primary_chunks=3)
    plan = g.plan("What have we discussed about performance?", _eps(12))
    assert plan.strategy == CHUNKED
    assert plan.primary_chunk_count == 3
    assert plan.compressed_chunk_count == 9


def test_plan_returns_valid_strategy_vocab():
    g = _gate()
    for n in range(0, 25):
        plan = g.plan("some query here", _eps(n))
        assert plan.strategy in (DIRECT, CHUNKED, SUMMARY_ONLY)
        assert isinstance(plan, PresentationPlan)


def test_plan_deterministic_for_identical_inputs():
    g = _gate()
    q = "What did Alice say about Postgres?"
    eps = _eps(10)
    p1 = g.plan(q, eps)
    p2 = g.plan(q, eps)
    assert p1.strategy == p2.strategy
    assert p1.primary_chunk_count == p2.primary_chunk_count


def test_plan_is_fast():
    import time as _t
    g = _gate()
    eps = _eps(50)
    start = _t.perf_counter()
    for _ in range(1000):
        g.plan("query", eps)
    elapsed_ms = (_t.perf_counter() - start)
    per_call_us = (elapsed_ms / 1000) * 1e6
    # Heuristic, no model → <1ms per call (docs/Phase 2c.md §5.5). Give 1ms
    # headroom on slow CI.
    assert per_call_us < 1000.0, f"plan() took {per_call_us:.1f}us/call"


# ── outcome buffer ──

def test_record_outcome_grows_buffer():
    g = _gate()
    plan = g.plan("query", _eps(10))
    assert len(g.outcome_buffer) == 0
    g.record_outcome(plan, PresentationOutcome(expand_count=1, unused_primary_count=0))
    assert len(g.outcome_buffer) == 1
    rec = g.outcome_buffer.records[0]
    assert rec["strategy"] == plan.strategy
    assert rec["expand_count"] == 1


def test_outcome_buffer_evicts_at_capacity():
    g = _gate()
    g.outcome_buffer = ReplayBuffer(capacity=3)
    plan = g.plan("query", _eps(10))
    for i in range(5):
        g.record_outcome(plan, PresentationOutcome(expand_count=i))
    assert len(g.outcome_buffer) == 3
    # Oldest evicted; the last 3 remain.
    assert list(g.outcome_buffer.records)[-1]["expand_count"] == 4