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


# ── plan shape / robustness ──


def test_plan_has_canonical_keys():
    plan = _planner().plan_rule_based("anything")
    assert set(plan.keys()) == {
        "entities", "topics", "tones", "entity_mode",
        "temporal_after", "temporal_before", "temporal_filter", "limit",
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