"""Offline tests for the Phase 2c Presentation Gate (axis b: end state).

Pure-logic — no torch, no model, no Bonsai. Verifies the four end states
(direct / format / synthesize / extract), the heuristic default, caller
override → override-buffer recording, and determinism. Per chat [145]/[146]
(docs/Ponder Engine Chat Facts.md §3.2): the API is explicit, the heuristic
provides a default, and a caller override is the training signal.
"""

from __future__ import annotations

import pytest

from src.config import Phase2cConfig
from src.subconscious.presentation_gate import (
    END_DIRECT, END_EXTRACT, END_FORMAT, END_SYNTHESIZE,
    EndStatePlan, PresentationGate,
)


def _gate(direct_max=3) -> PresentationGate:
    cfg = Phase2cConfig()
    cfg.presentation_gate.direct_max_episodes = direct_max
    return PresentationGate(cfg)


def _eps(n: int) -> list[dict]:
    return [{"episode_id": f"e{i}", "score": 1.0} for i in range(n)]


# ── heuristic defaults ──

def test_extract_for_list_json_verb():
    g = _gate()
    es = g.plan_end_state("List all decisions as JSON", _eps(10))
    assert es.end_state == END_EXTRACT
    assert es.jepa_default is True


def test_extract_for_dependency_graph():
    g = _gate()
    es = g.plan_end_state("Build a dependency graph of these modules", _eps(5))
    assert es.end_state == END_EXTRACT


def test_synthesize_for_reasoning_verb():
    g = _gate()
    es = g.plan_end_state("Why did we choose Postgres over MySQL?", _eps(4))
    assert es.end_state == END_SYNTHESIZE
    assert es.model_size == "bonsai"


def test_synthesize_when_many_episodes():
    g = _gate()
    # No reasoning verb, but >direct_max episodes → synthesize.
    es = g.plan_end_state("What did Alice say about Postgres?", _eps(10))
    assert es.end_state == END_SYNTHESIZE


def test_direct_for_factual_lookup_small_set():
    g = _gate()
    es = g.plan_end_state("What did Alice say about Postgres?", _eps(2))
    assert es.end_state == END_DIRECT
    assert es.model_size is None  # no LLM call for direct


def test_default_is_synthesize_when_ambiguous():
    g = _gate()
    es = g.plan_end_state("something ambiguous", _eps(0))
    assert es.end_state == END_SYNTHESIZE


def test_plan_end_state_returns_valid_endstate():
    g = _gate()
    for n in range(0, 25):
        es = g.plan_end_state("query", _eps(n))
        assert es.end_state in (END_DIRECT, END_FORMAT, END_SYNTHESIZE, END_EXTRACT)
        assert isinstance(es, EndStatePlan)


def test_heuristic_deterministic():
    g = _gate()
    q = "What did Alice say about Postgres?"
    eps = _eps(10)
    assert g.plan_end_state(q, eps).end_state == g.plan_end_state(q, eps).end_state


# ── caller override ──

def test_caller_end_state_is_honored():
    g = _gate()
    # Heuristic would say synthesize (10 episodes); caller forces direct.
    es = g.plan_end_state("What did Alice say?", _eps(10), caller_end_state=END_DIRECT)
    assert es.end_state == END_DIRECT
    assert es.jepa_default is False
    assert "caller-specified" in es.rationale


def test_override_disagreement_records_to_buffer():
    g = _gate()
    assert len(g.override_buffer) == 0
    g.plan_end_state("What did Alice say?", _eps(10), caller_end_state=END_EXTRACT)
    assert len(g.override_buffer) == 1
    rec = g.override_buffer.records[0]
    assert rec["jepa_predicted"] == END_SYNTHESIZE
    assert rec["caller_chose"] == END_EXTRACT
    assert rec["query"] == "What did Alice say?"
    assert rec["episode_count"] == 10


def test_override_agreement_does_not_record():
    g = _gate()
    # Heuristic default for a reasoning query is synthesize; caller also says
    # synthesize → no override (no disagreement, no buffer entry).
    g.plan_end_state("Why did we choose X?", _eps(4), caller_end_state=END_SYNTHESIZE)
    assert len(g.override_buffer) == 0


def test_format_end_state_passes_format_spec():
    g = _gate()
    fs = {"consumer": "claude", "purpose": "context", "max_tokens": 4000}
    es = g.plan_end_state("query", _eps(4), caller_end_state=END_FORMAT, format_spec=fs)
    assert es.end_state == END_FORMAT
    assert es.format_spec == fs


def test_extract_end_state_passes_schema():
    g = _gate()
    schema = {"type": "list", "item_type": "decision"}
    es = g.plan_end_state("query", _eps(4),
                          caller_end_state=END_EXTRACT, extract_schema=schema)
    assert es.end_state == END_EXTRACT
    assert es.extract_schema == schema


def test_unknown_end_state_raises():
    g = _gate()
    with pytest.raises(ValueError, match="unknown end_state"):
        g.plan_end_state("query", _eps(4), caller_end_state="bogus")


# ── model_size handling ──

def test_synthesize_default_model_is_bonsai():
    g = _gate()
    es = g.plan_end_state("Why did we choose X?", _eps(4))
    assert es.end_state == END_SYNTHESIZE
    assert es.model_size == "bonsai"


def test_caller_model_size_passthrough():
    g = _gate()
    es = g.plan_end_state("Why?", _eps(4),
                          caller_end_state=END_SYNTHESIZE, model_size="70B")
    assert es.model_size == "70B"


def test_non_synthesize_end_state_has_no_model_size():
    g = _gate()
    es = g.plan_end_state("List all decisions as JSON", _eps(10))
    assert es.end_state == END_EXTRACT
    assert es.model_size is None
    es2 = g.plan_end_state("What did Alice say?", _eps(2))
    assert es2.end_state == END_DIRECT
    assert es2.model_size is None