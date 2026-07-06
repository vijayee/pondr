"""Offline tests for the Bonsai query planner.

The rule-based planner (``plan_rule_based``) is exercised directly — it is
deterministic and needs no server. ``plan()`` is asserted to fall back to the
rule-based planner when the Bonsai server is unreachable, so retrieval degrades
gracefully offline. A live-server path is gated on an endpoint probe and
``pytest.skip``s when Bonsai is not running (the normal offline case).
"""

import pytest

from src.config import config
from src.retrieval.query_planner import BonsaiQueryPlanner


def _planner():
    return BonsaiQueryPlanner()


# ── rule-based planner (the 4 doc example questions) ──


def test_plan_simple_affect_query():
    """'What was I frustrated about?' → tones=["frustrated"], union."""
    plan = _planner().plan_rule_based("What was I frustrated about?")
    assert "frustrated" in plan["tones"]
    assert plan["entity_mode"] == "union"


def test_plan_entity_decision_query():
    """'What did Alice and I decide about the database?' → Alice + union + topic."""
    plan = _planner().plan_rule_based("What did Alice and I decide about the database?")
    assert "Alice" in plan["entities"]
    assert plan["entity_mode"] == "union"
    assert "decision_making" in plan["topics"] or "database_design" in plan["topics"]


def test_plan_temporal_query():
    """'What happened after we implemented morphisms?' → temporal_after set."""
    plan = _planner().plan_rule_based("What happened after we implemented morphisms?")
    assert plan["temporal_after"] is not None
    assert "morphism" in plan["temporal_after"].lower()


def test_plan_cross_entity_intersection():
    """'What did Alice and Bob disagree about?' → Alice+Bob, intersection."""
    plan = _planner().plan_rule_based("What did Alice and Bob disagree about?")
    assert "Alice" in plan["entities"]
    assert "Bob" in plan["entities"]
    assert plan["entity_mode"] == "intersection"


# ── absolute date-range planning (Phase 1c) ──


def test_plan_date_range_single_month():
    """'What happened in June 2025?' → date_from/date_to spanning June 2025."""
    plan = _planner().plan_rule_based("What happened in June 2025?")
    assert plan["date_from"] == "2025-06-01"
    assert plan["date_to"] == "2025-06-30"
    # Absolute range is mutually exclusive with the relative bucket filter.
    assert plan["temporal_filter"] is None


def test_plan_date_range_between_months():
    """'What happened between March and May 2025?' → March 1 .. May 31."""
    plan = _planner().plan_rule_based("What happened between March 2025 and May 2025?")
    assert plan["date_from"] == "2025-03-01"
    assert plan["date_to"] == "2025-05-31"
    assert plan["temporal_filter"] is None


def test_plan_relative_bucket_still_works():
    """'last week' still maps to temporal_filter (no absolute range present)."""
    plan = _planner().plan_rule_based("What did we discuss last week?")
    assert plan["temporal_filter"] == "last_week"
    assert plan["date_from"] is None and plan["date_to"] is None


# ── conversation context / pronoun resolution (Phase 1c) ──


def test_pronoun_resolution_with_context():
    """'he'/'it' resolve to the person/topic from recent conversation context."""
    planner = _planner()
    history = [
        {"role": "user", "content": "What did Bob say about the WAL config?"},
        {"role": "assistant", "content": "Bob said the WAL config needed better docs."},
    ]
    plan = planner.plan_rule_based("What did he suggest we do about it?", history)
    assert "Bob" in plan["entities"]          # "he" -> Bob
    assert "configuration" in plan["topics"]  # "it" -> WAL config


def test_implicit_reference_with_context():
    """'that' resolves to the topic/entity from recent context."""
    planner = _planner()
    history = [
        {"role": "user", "content": "I'm worried about the database performance."},
        {"role": "assistant", "content": "The Python async bindings are the main bottleneck."},
    ]
    plan = planner.plan_rule_based("How do we fix that?", history)
    assert "performance" in plan["topics"] or "Python" in plan["entities"]


def test_no_context_still_works():
    """Planner works without conversation history (backward compatible)."""
    planner = _planner()
    plan = planner.plan("What was I frustrated about?")
    assert "frustrated" in plan["tones"]


def test_context_not_used_without_pronoun():
    """Context entities are NOT injected when the prompt has no pronoun."""
    planner = _planner()
    history = [
        {"role": "user", "content": "What did Bob say about the WAL config?"},
    ]
    # "What did Alice say?" has its own entity (Alice) and no pronoun -> context
    # should not inject Bob.
    plan = planner.plan_rule_based("What did Alice say about databases?", history)
    assert "Bob" not in plan["entities"]
    assert "Alice" in plan["entities"]


# ── plan shape / robustness ──


def test_plan_has_canonical_keys():
    plan = _planner().plan_rule_based("anything")
    assert set(plan.keys()) == {
        "entities", "topics", "tones", "entity_mode",
        "temporal_after", "temporal_before", "temporal_filter",
        "date_from", "date_to", "limit",
    }
    assert plan["limit"] == config.default_retrieval_limit


def test_plan_falls_back_when_server_unreachable():
    """plan() must not raise when the Bonsai server is down — it returns rules."""
    planner = BonsaiQueryPlanner(endpoint="http://127.0.0.1:9/no-such-server")
    plan = planner.plan("What was I frustrated about?")
    assert "frustrated" in plan["tones"]  # fell back to rule-based


def test_plan_via_server_raises_on_failure():
    """plan_via_server surfaces errors verbatim (used by live tests)."""
    planner = BonsaiQueryPlanner(endpoint="http://127.0.0.1:9/no-such-server", timeout=1.0)
    with pytest.raises(RuntimeError):
        planner.plan_via_server("What was I frustrated about?")


# ── live Bonsai path (gated) ──


def _endpoint_up() -> bool:
    import requests
    try:
        requests.get(f"{config.bonsai_endpoint.rstrip('/')}/models", timeout=2.0)
        return True
    except Exception:
        return False


def test_plan_via_live_bonsai():
    """Live: the Bonsai server produces a well-formed plan for the affect query."""
    if not _endpoint_up():
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable")
    plan = _planner().plan_via_server("What was I frustrated about?")
    assert "tones" in plan
    assert isinstance(plan["tones"], list)