"""Offline tests for the Phase 2c end-state dispatch (direct/format/extract/synthesize).

Verifies the core 2c contract: ``direct``/``format``/``extract`` return WITHOUT
an LLM call (no ``synthesize`` callable invoked); only ``synthesize`` calls it.
Pure-logic + a fake chunked context + stub synthesize.
"""

from __future__ import annotations

import torch

from src.retrieval.chunked_context import ChunkedContextFormatter
from src.retrieval.end_state import dispatch_end_state
from src.subconscious.presentation_gate import (
    END_DIRECT, END_EXTRACT, END_FORMAT, END_SYNTHESIZE, EndStatePlan,
)
from src.subconscious.state_serializer import JGSSnapshot


def _plan(end_state, format_spec=None, extract_schema=None, model_size=None) -> EndStatePlan:
    return EndStatePlan(
        end_state=end_state, format_spec=format_spec, extract_schema=extract_schema,
        model_size=model_size, jepa_default=True, rationale="test",
    )


def _fake_chunked(primary=None, secondary=None) -> "ChunkedContext":  # type: ignore[name-defined]
    from src.subconscious.ssm_chunker import ChunkedContext
    primary = primary if primary is not None else [
        {"episode_id": "e0", "text": "full text e0", "summary": "sum e0",
         "timestamp": "t0", "entities": ["Alice"], "topics": ["db"], "tones": [],
         "decisions": ["decide A"], "score": 1.0},
    ]
    secondary = secondary if secondary is not None else [
        {"episode_id": "e1", "text": "full text e1", "summary": "sum e1",
         "timestamp": "t1", "entities": ["Bob"], "topics": ["perf"], "tones": [],
         "decisions": ["decide B"], "score": 0.5},
    ]
    return ChunkedContext(
        primary_chunks=primary, compressed_state=None,
        chunk_map={"e0": 0, "e1": -1}, expandable_ids={"e1"},
        total_episodes=2, primary_token_count=4, compressed_episode_count=1,
        secondary_episodes=secondary,
    )


def _synthesize_calls() -> tuple[callable, list]:
    calls: list = []

    def synth(context, history):
        calls.append((context, history))
        return "LLM RESPONSE"

    return synth, calls


# ── direct ──

def test_direct_returns_episodes_no_llm():
    synth, calls = _synthesize_calls()
    eps = [{"episode_id": "e0", "summary": "sum", "text": "text",
            "entities": ["A"], "topics": ["t"], "tones": [], "decisions": [],
            "timestamp": "t0", "score": 1.0}]
    res = dispatch_end_state(
        _plan(END_DIRECT), _fake_chunked(), ChunkedContextFormatter(),
        eps, "What did Alice say?", synthesize=synth,
    )
    assert res["type"] == END_DIRECT
    assert res["episodes"][0]["episode_id"] == "e0"
    assert "response" not in res  # no LLM call
    assert calls == []


# ── format ──

def test_format_returns_context_no_llm():
    synth, calls = _synthesize_calls()
    res = dispatch_end_state(
        _plan(END_FORMAT, format_spec={"consumer": "claude", "purpose": "ctx",
                                       "max_tokens": 4000}),
        _fake_chunked(), ChunkedContextFormatter(), [], "query",
        consumer="bonsai", synthesize=synth,
    )
    assert res["type"] == END_FORMAT
    assert "context" in res and res["context"]
    assert res["consumer"] == "claude"
    assert calls == []


# ── extract ──

def test_extract_list_of_decisions_no_llm():
    synth, calls = _synthesize_calls()
    eps = [
        {"episode_id": "e0", "decisions": ["decide A", "decide B"], "entities": [],
         "topics": [], "tones": []},
        {"episode_id": "e1", "decisions": ["decide C"], "entities": [],
         "topics": [], "tones": []},
    ]
    res = dispatch_end_state(
        _plan(END_EXTRACT, extract_schema={"type": "list", "item_type": "decision"}),
        _fake_chunked(), ChunkedContextFormatter(), eps, "List all decisions as JSON",
        synthesize=synth,
    )
    assert res["type"] == END_EXTRACT
    assert res["data"] == ["decide A", "decide B", "decide C"]
    assert calls == []


def test_extract_unique_entities():
    eps = [
        {"episode_id": "e0", "entities": ["Alice", "Bob"], "decisions": [],
         "topics": [], "tones": []},
        {"episode_id": "e1", "entities": ["Bob", "Carol"], "decisions": [],
         "topics": [], "tones": []},
    ]
    res = dispatch_end_state(
        _plan(END_EXTRACT, extract_schema={"type": "list", "item_type": "entity"}),
        _fake_chunked(), ChunkedContextFormatter(), eps, "query",
    )
    assert res["data"] == ["Alice", "Bob", "Carol"]


def test_extract_graph_builds_adjacency():
    eps = [
        {"episode_id": "e0", "entities": ["Alice"], "decisions": [], "topics": [],
         "tones": [], "timestamp": "t0"},
        {"episode_id": "e1", "entities": ["Alice", "Bob"], "decisions": [],
         "topics": [], "tones": [], "timestamp": "t1"},
    ]
    res = dispatch_end_state(
        _plan(END_EXTRACT, extract_schema={"type": "graph", "item_type": "entity"}),
        _fake_chunked(), ChunkedContextFormatter(), eps, "build a graph",
    )
    g = res["data"]
    node_ids = {n["id"] for n in g["nodes"]}
    assert {"Alice", "Bob", "e0", "e1"} <= node_ids
    rels = {(e["src"], e["dst"]) for e in g["edges"]}
    assert ("Alice", "e0") in rels
    assert ("Bob", "e1") in rels


# ── synthesize ──

def test_synthesize_calls_llm_once():
    synth, calls = _synthesize_calls()
    res = dispatch_end_state(
        _plan(END_SYNTHESIZE, model_size="bonsai"),
        _fake_chunked(), ChunkedContextFormatter(), [], "Why did we choose X?",
        consumer="bonsai", synthesize=synth,
        conversation_history=[{"role": "user", "content": "hi"}],
    )
    assert res["type"] == END_SYNTHESIZE
    assert res["response"] == "LLM RESPONSE"
    assert res["model_size"] == "bonsai"
    assert len(calls) == 1  # exactly one LLM call


def test_synthesize_without_callable_raises():
    import pytest
    with pytest.raises(RuntimeError, match="synthesize"):
        dispatch_end_state(
            _plan(END_SYNTHESIZE), _fake_chunked(), ChunkedContextFormatter(),
            [], "query", synthesize=None,
        )


# ── unknown ──

def test_unknown_end_state_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown end_state"):
        dispatch_end_state(
            _plan("bogus"), _fake_chunked(), ChunkedContextFormatter(), [], "query",
        )