"""Tests for the feedback-driven salience loop (Phase 2c+).

The Ponder Engine is an *artificial subconscious*: it auto-retrieves on the
shared prompt AND exposes tools the conscious consumer LLM invokes. The
feedback loop: after a synthesizing turn the model judges which retrieved
units were useful (a 1-5 rating) -> a per-unit boost is persisted -> boost-aware
scoring reweights retrieval on the next query. Two consumers, one interface:
the external LLM via ``dispatch_tool`` (the canonical path), and Ponder's own
Bonsai self-chat via a ``record_feedback`` tool call (with a structured
fallback when Bonsai tool-calling is unsupported).

This file exercises: the boost store (clamp, cold-start 1.0), boost-aware
scoring (multiply, gated), kind-aware rerank (cap vs pure sort), the tool
surface (record_feedback / expand / search_memory / unknown / bad args never
raise), and the self-chat wiring (tool-call path + fallback path, tool_calls
never leak into the response).

Offline: tmp_path WaveDB store, stub planner/embedder/mode_a, ReferenceSSM
backbone. No GLiNER, no Bonsai server, no GPU. Mirrors ``test_orchestrator.py``.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from src.config import Phase2cConfig, config as _config
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.tools import LOOP_TOOLS, SELF_CHAT_TOOLS, TOOL_SCHEMAS, dispatch_tool


# ── stubs (mirror tests/test_orchestrator.py) ──

class _StubEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            buf = bytearray()
            h = hashlib.sha256(t.encode("utf-8")).digest()
            counter = 0
            while len(buf) < self.dim:
                buf += hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
                counter += 1
            vec = [(b / 127.5 - 1.0) for b in buf[: self.dim]]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


class _StubPlanner:
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return self._plan


class _ScriptedModeA:
    """Stub ModeAGenerator -- pops a queued ``(content, tool_calls)`` per call.

    Lets a test script the tool-call path (first call returns record_feedback
    tool_calls) and the fallback path (first call returns no tool_calls, the
    fallback's second call returns a JSON rating array). Records every call so
    tests can assert call count + that tools were offered.
    """
    def __init__(self, responses: list[tuple]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def _complete(self, messages: list[dict], tools=None, tool_choice=None) -> tuple:
        self.calls.append({"tools": tools})
        if self.responses:
            return self.responses.pop(0)
        return ("", None)


def _ep(eid, entities=None, topics=None, summary=None, text=None,
        ts="2026-07-03T10:00:00") -> Episode:
    return Episode(
        id=eid, timestamp=ts,
        summary=summary or f"summary {eid}",
        full_text=text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=[], decisions=[],
    )


def _orch(tmp_path, plan, episodes, mode_a, *, cfg=None, user_id="victor"):
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in episodes:
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                     embedder=_StubEmbedder())
    c = cfg or Phase2cConfig()
    c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=mode_a, config=c, user_id=user_id,
    )
    return orch, store


def _score_for(results, eid):
    for r in results:
        if r["episode_id"] == eid:
            return r.get("score", 0.0)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. boost store: cold-start, rating->boost, clamp
# ═══════════════════════════════════════════════════════════════════════════

def test_boost_cold_start_is_neutral(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    assert store.get_unit_boost("ep_999") == 1.0           # missing -> 1.0
    assert store.get_unit_boost("doc_nope_sec_000") == 1.0
    assert store.get_unit_boost("with/slash") == 1.0       # defensive skip
    assert store.get_unit_boost("") == 1.0
    store.close()


def test_record_feedback_rating_to_boost(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    # rating 5 -> useful 1.0 -> new = 1.0 * (0.5+1.0) = 1.5
    assert store.record_feedback([{"unit_id": "ep_a", "rating": 5}]) == 1
    assert store.get_unit_boost("ep_a") == pytest.approx(1.5)
    # rating 1 -> useful 0.0 -> new = 1.0 * (0.5+0.0) = 0.5
    assert store.record_feedback([{"unit_id": "ep_b", "rating": 1}]) == 1
    assert store.get_unit_boost("ep_b") == pytest.approx(0.5)
    # rating 3 -> useful 0.5 -> new = 1.0 * (0.5+0.5) = 1.0 (neutral)
    assert store.record_feedback([{"unit_id": "ep_c", "rating": 3}]) == 1
    assert store.get_unit_boost("ep_c") == pytest.approx(1.0)
    store.close()


def test_record_feedback_compounds_and_clamps(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    # rating 5 thrice on a fresh unit: 1.0 -> 1.5 -> 2.25 -> 3.375 (under cap).
    store.record_feedback([{"unit_id": "ep_x", "rating": 5}])
    assert store.get_unit_boost("ep_x") == pytest.approx(1.5)
    store.record_feedback([{"unit_id": "ep_x", "rating": 5}])
    assert store.get_unit_boost("ep_x") == pytest.approx(2.25)
    store.record_feedback([{"unit_id": "ep_x", "rating": 5}])
    assert store.get_unit_boost("ep_x") == pytest.approx(3.375)
    # A fourth 5 -> 3.375 * 1.5 = 5.0625 -> CLAMPED to 4.0 at write.
    store.record_feedback([{"unit_id": "ep_x", "rating": 5}])
    assert store.get_unit_boost("ep_x") == pytest.approx(4.0)
    store.close()


def test_write_unit_boost_batch_clamps_at_write(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    # A direct write of an out-of-range value is clamped to [0.25, 4.0].
    store.write_unit_boost_batch({"ep_hi": 100.0, "ep_lo": 0.001})
    assert store.get_unit_boost("ep_hi") == pytest.approx(4.0)
    assert store.get_unit_boost("ep_lo") == pytest.approx(0.25)
    store.close()


def test_record_feedback_bad_input_no_crash(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    # Junk unit_id / out-of-range rating / wrong shape never raise + never write.
    assert store.record_feedback([]) == 0
    assert store.record_feedback([{"unit_id": "bad/id", "rating": 5}]) == 0
    assert store.record_feedback([{"unit_id": "ep_ok", "rating": 9}]) == 0
    assert store.record_feedback([{"unit_id": "ep_ok", "rating": 0}]) == 0
    assert store.record_feedback([{"not_unit_id": "x", "rating": 5}]) == 0
    assert store.get_unit_boost("ep_ok") == 1.0  # nothing written
    store.close()


# ═══════════════════════════════════════════════════════════════════════════
# 2. boost-aware scoring: feedback raises the next retrieve's score
# ═══════════════════════════════════════════════════════════════════════════

def test_feedback_raises_next_retrieve_score(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", entities=["Postgres"],
                             summary="We chose Postgres"))
    trav = GraphTraversal(store)
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    before = trav.retrieve(plan)
    s_before = _score_for(before, "ep_001")
    assert s_before is not None and s_before > 0.0
    # Rate ep_001 a 5 -> boost 1.5 -> the next retrieve scores 1.5x (the boost
    # is a MULTIPLIER on the raw score, applied in _apply_unit_boost).
    assert store.record_feedback([{"unit_id": "ep_001", "rating": 5}]) == 1
    after = trav.retrieve(plan)
    s_after = _score_for(after, "ep_001")
    assert s_after == pytest.approx(s_before * 1.5, rel=1e-6)
    store.close()


def test_feedback_disabled_skips_boost_multiply(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", entities=["Postgres"],
                             summary="We chose Postgres"))
    saved = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        trav = GraphTraversal(store)
        plan = {"entities": ["Postgres"], "entity_mode": "union"}
        before = trav.retrieve(plan)
        s_before = _score_for(before, "ep_001")
        # Even with a boost written, the disabled flag skips the multiply.
        store.write_unit_boost_batch({"ep_001": 4.0})
        after = trav.retrieve(plan)
        s_after = _score_for(after, "ep_001")
        assert s_after == pytest.approx(s_before, rel=1e-6)  # unchanged
    finally:
        _config.feedback_salience_enabled = saved
    store.close()


# ═══════════════════════════════════════════════════════════════════════════
# 3. kind-aware rerank: cap vs pure sort
# ═══════════════════════════════════════════════════════════════════════════

def test_kind_aware_rerank_caps_run(tmp_path):
    """With cap=3, a wall of 5 higher-scoring sections interleaves an episode
    after 3 consecutive sections; cap=0 is pure score sort (no interleave)."""
    from src.retrieval.retriever import HippocampalRetriever as _R
    # Build 5 section results (higher score) + 2 episode results (lower score).
    secs = [{"episode_id": f"s{i}", "kind": "section", "score": 10.0 - i}
            for i in range(5)]
    eps = [{"episode_id": f"e{i}", "kind": "episode", "score": 1.0 - i * 0.1}
           for i in range(2)]
    allr = secs + eps

    saved = _config.kind_diversity_cap
    _config.kind_diversity_cap = 3
    try:
        out = _R._kind_aware_rerank(None, allr)  # static-ish: uses config only
        kinds = [r["kind"] for r in out]
        # The first 3 are sections (higher score), then the run is capped and
        # an episode is forced in before more sections.
        assert kinds[:3] == ["section", "section", "section"]
        assert kinds[3] == "episode", f"cap did not interleave: {kinds}"
        assert "episode" in kinds
    finally:
        _config.kind_diversity_cap = saved

    # cap=0 -> pure score sort (all sections first, then episodes).
    _config.kind_diversity_cap = 0
    try:
        out0 = _R._kind_aware_rerank(None, allr)
        kinds0 = [r["kind"] for r in out0]
        assert kinds0 == ["section"] * 5 + ["episode"] * 2
    finally:
        _config.kind_diversity_cap = saved


# ═══════════════════════════════════════════════════════════════════════════
# 4. tool surface: dispatch_tool never raises; schemas are OpenAI-shaped
# ═══════════════════════════════════════════════════════════════════════════

def test_tool_schemas_are_openai_shaped():
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["record_feedback", "expand", "search_memory"]
    for t in TOOL_SCHEMAS:
        assert t["type"] == "function"
        assert "description" in t["function"]
        assert "parameters" in t["function"]
        assert t["function"]["parameters"]["type"] == "object"
        # ASCII-only descriptions (cp1252-safe).
        t["function"]["description"].encode("ascii")


def test_dispatch_record_feedback_writes_boost(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    res = dispatch_tool(orch, "record_feedback",
                       {"judgments": [{"unit_id": "ep_001", "rating": 5}]})
    parsed = json.loads(res)
    assert parsed["ok"] is True and parsed["applied"] == 1
    assert store.get_unit_boost("ep_001") == pytest.approx(1.5)
    # A JSON-string arguments payload works too (the external host may pass a
    # string).
    res2 = dispatch_tool(orch, "record_feedback",
                         json.dumps({"judgments": [{"unit_id": "ep_002", "rating": 1}]}))
    assert json.loads(res2)["applied"] == 1
    store.close()


def test_dispatch_expand_returns_text(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres",
               text="User: why\nAssistant: because")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    out = dispatch_tool(orch, "expand", {"unit_id": "ep_001"})
    assert isinstance(out, str)
    assert "We chose Postgres" in out and "because" in out
    # Missing unit -> error string, no raise.
    missing = dispatch_tool(orch, "expand", {"unit_id": "ep_nope"})
    assert "error" in json.loads(missing)
    store.close()


def test_dispatch_search_memory_returns_context(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    out = dispatch_tool(orch, "search_memory",
                       {"query": "Postgres", "entities": ["Postgres"]})
    assert isinstance(out, str) and "Postgres" in out
    store.close()


def test_dispatch_never_raises_on_bad_input(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    # Unknown tool.
    assert "error" in json.loads(dispatch_tool(orch, "nope", {}))
    # Malformed JSON args string.
    assert "error" in json.loads(dispatch_tool(orch, "record_feedback", "{not json"))
    # Non-object args.
    assert "error" in json.loads(dispatch_tool(orch, "record_feedback", 42))
    # record_feedback without a judgments array.
    assert "error" in json.loads(dispatch_tool(orch, "record_feedback", {}))
    # expand without unit_id.
    assert "error" in json.loads(dispatch_tool(orch, "expand", {}))
    # search_memory without a query.
    assert "error" in json.loads(dispatch_tool(orch, "search_memory", {}))
    store.close()


# ═══════════════════════════════════════════════════════════════════════════
# 5. self-chat wiring: tool-call path (Bonsai emits record_feedback)
# ═══════════════════════════════════════════════════════════════════════════

def test_self_chat_tool_call_path_applies_feedback(tmp_path):
    """Bonsai emits a record_feedback tool call inside the tool loop -> dispatch
    applies it; the loop runs a second clean turn (no tool_calls) to finish;
    two LLM calls; tool_calls never leak into the response."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    tool_call = [{
        "id": "call_1", "type": "function",
        "function": {"name": "record_feedback",
                     "arguments": {"judgments": [{"unit_id": "ep_001", "rating": 5}]}},
    }]
    # Turn 1: the answer + record_feedback tool call. Turn 2: clean (no tools)
    # so the loop stops after applying the feedback (no fallback needed).
    mode_a = _ScriptedModeA([("THE ANSWER", tool_call), ("THE ANSWER", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    res = orch.query("Why did we choose Postgres?")
    assert res["end_state_plan"].end_state == "synthesize"
    assert res["response"] == "THE ANSWER"          # the text answer, unchanged
    assert "record_feedback" not in res["response"]  # tool_calls not concatenated
    assert res["feedback_collected"] == 1
    assert len(mode_a.calls) == 2          # tool-call turn + clean turn (no fallback)
    # The full tool surface (incl record_feedback) was offered on turn 1.
    assert mode_a.calls[0]["tools"] is TOOL_SCHEMAS
    # The boost was written.
    assert store.get_unit_boost("ep_001") == pytest.approx(1.5)
    store.close()


# ═══════════════════════════════════════════════════════════════════════════
# 6. self-chat wiring: fallback path (Bonsai emits no tool call)
# ═══════════════════════════════════════════════════════════════════════════

def test_self_chat_fallback_path_applies_feedback(tmp_path):
    """Bonsai emits no tool call -> the fallback makes one structured rating
    call -> feedback applied; two LLM calls; response unchanged."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    # Call 1 (synthesize, tools offered): no tool_calls.
    # Call 2 (fallback, no tools): a JSON array rating ep_001 a 4.
    fallback_json = '[{"unit_id":"ep_001","rating":4}]'
    mode_a = _ScriptedModeA([("THE ANSWER", None), (fallback_json, None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    res = orch.query("Why did we choose Postgres?")
    assert res["response"] == "THE ANSWER"
    assert res["feedback_collected"] == 1
    assert len(mode_a.calls) == 2                     # synthesize + fallback
    # rating 4 -> useful 0.75 -> new = 1.0 * (0.5+0.75) = 1.25
    assert store.get_unit_boost("ep_001") == pytest.approx(1.25)
    store.close()


def test_self_chat_fallback_no_rating_is_noop(tmp_path):
    """Fallback returns nothing parseable -> no feedback that turn (no-op),
    but the response is still returned."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("THE ANSWER", None), ("sorry I cannot rate", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    res = orch.query("Why did we choose Postgres?")
    assert res["response"] == "THE ANSWER"
    assert res["feedback_collected"] == 0
    assert store.get_unit_boost("ep_001") == 1.0     # no boost written
    store.close()


def test_feedback_disabled_skips_self_chat_feedback(tmp_path):
    """feedback_salience_enabled=False -> no feedback instruction, no tools
    offered, no fallback call; pure synthesize (one call). This test runs the
    NON-loop (one-shot) path (self_chat_tool_loop_enabled=False) as the A/B
    regression guard for the old ``tools is None`` assertion; the loop+disabled
    shape (LOOP_TOOLS offered) is covered by
    ``test_self_chat_loop_feedback_disabled_offers_no_record_feedback``."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("THE ANSWER", None), ("should never be used", None)])
    cfg = Phase2cConfig()
    saved_fb = _config.feedback_salience_enabled
    saved_loop = _config.self_chat_tool_loop_enabled
    _config.feedback_salience_enabled = False
    _config.self_chat_tool_loop_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a, cfg=cfg)
        res = orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved_fb
        _config.self_chat_tool_loop_enabled = saved_loop
    assert res["response"] == "THE ANSWER"
    assert res["feedback_collected"] == 0
    assert len(mode_a.calls) == 1                     # no fallback
    assert mode_a.calls[0]["tools"] is None          # tools not offered
    store.close()


# ═══════════════════════════════════════════════════════════════════════════
# 7. self-chat full agent loop (run_tool_loop wired into _synthesize)
# ═══════════════════════════════════════════════════════════════════════════

def _tool_call(name, args, cid="call_1"):
    return [{"id": cid, "type": "function",
             "function": {"name": name, "arguments": args}}]


def test_self_chat_loop_dispatches_expand_and_refeeds(tmp_path):
    """The loop lets Bonsai call ``expand`` mid-generation: turn 1 emits the
    tool call, the dispatched result is fed back, turn 2 gives the final answer
    with no tools. Two calls; expand was dispatched; the tool result appears in
    turn 2's messages. Feedback is disabled here so the structured fallback
    does NOT fire (the loop/refeed shape is the point; the fallback is
    exercised by its own test) -- the loop offers LOOP_TOOLS."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres",
               text="User: why\nAssistant: because the in-DB vector layer")]
    expand_call = _tool_call("expand", {"unit_id": "ep_001"})
    mode_a = _ScriptedModeA([("draft", expand_call), ("FINAL ANSWER", None)])
    saved = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        res = orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved
    assert res["response"] == "FINAL ANSWER"
    assert len(mode_a.calls) == 2                       # expand turn + clean turn
    assert mode_a.calls[0]["tools"] is LOOP_TOOLS       # retrieval surface offered
    # The loop surfaced its transcript + the expand dispatch is recorded.
    assert res["loop_exhausted"] is False
    names = [c["name"] for c in res["loop_collected"]]
    assert names == ["expand"]
    # The loop transcript carries the fed-back tool-role result message.
    roles = [m.get("role") for m in res["loop_tool_messages"]]
    assert "tool" in roles
    store.close()


def test_self_chat_loop_dispatches_search_memory_and_refeeds(tmp_path):
    """Same shape with ``search_memory``: the model re-searches mid-generation,
    the fresh context is fed back, then it answers. Two calls. Feedback
    disabled so the fallback does not fire (see the expand test for the
    rationale)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    search_call = _tool_call("search_memory",
                             {"query": "Postgres decision", "entities": ["Postgres"]})
    mode_a = _ScriptedModeA([("draft", search_call), ("FINAL ANSWER", None)])
    saved = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        res = orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved
    assert res["response"] == "FINAL ANSWER"
    assert len(mode_a.calls) == 2
    assert mode_a.calls[0]["tools"] is LOOP_TOOLS
    names = [c["name"] for c in res["loop_collected"]]
    assert names == ["search_memory"]
    store.close()


def test_self_chat_loop_max_iters_bound(tmp_path):
    """If the model keeps emitting tool calls, the loop stops at max_iters and
    reports ``exhausted=True`` (a truncated trajectory, not a clean stop).
    Feedback disabled so the structured fallback does not add an extra call
    after the exhausted loop -- this isolates the max_iters cap."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    search_call = _tool_call("search_memory", {"query": "x"})
    # Every turn emits a tool call -> the loop never stops cleanly.
    mode_a = _ScriptedModeA([(f"t{i}", search_call) for i in range(5)])
    saved_iters = _config.self_chat_tool_loop_max_iters
    saved_fb = _config.feedback_salience_enabled
    _config.self_chat_tool_loop_max_iters = 2
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        res = orch.query("Why did we choose Postgres?")
    finally:
        _config.self_chat_tool_loop_max_iters = saved_iters
        _config.feedback_salience_enabled = saved_fb
    assert len(mode_a.calls) == 2                       # capped at max_iters=2
    assert res["loop_exhausted"] is True
    store.close()


def test_self_chat_loop_disabled_is_byte_identical_one_shot(tmp_path):
    """self_chat_tool_loop_enabled=False -> the one-shot path: one _complete +
    _dispatch_feedback. A record_feedback tool call on the single call is still
    dispatched (the one-shot path's _dispatch_feedback handles it). One call."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    tool_call = _tool_call("record_feedback",
                           {"judgments": [{"unit_id": "ep_001", "rating": 5}]})
    mode_a = _ScriptedModeA([("THE ANSWER", tool_call)])
    saved = _config.self_chat_tool_loop_enabled
    _config.self_chat_tool_loop_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        res = orch.query("Why did we choose Postgres?")
    finally:
        _config.self_chat_tool_loop_enabled = saved
    assert res["response"] == "THE ANSWER"
    assert res["feedback_collected"] == 1
    assert len(mode_a.calls) == 1                       # one-shot, no clean turn
    # The non-loop path offers SELF_CHAT_TOOLS (record_feedback + expand).
    assert mode_a.calls[0]["tools"] is SELF_CHAT_TOOLS
    # No loop transcript keys are surfaced (byte-identical to the pre-loop result).
    assert "loop_collected" not in res
    store.close()


def test_self_chat_loop_record_feedback_counted_in_feedback_collected(tmp_path):
    """A record_feedback tool call inside the loop is counted into
    feedback_collected (filter-then-sum of the loop's collected results); no
    fallback fires because the tool path worked."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    tool_call = _tool_call("record_feedback",
                           {"judgments": [{"unit_id": "ep_001", "rating": 5}]})
    mode_a = _ScriptedModeA([("THE ANSWER", tool_call), ("THE ANSWER", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    res = orch.query("Why did we choose Postgres?")
    assert res["feedback_collected"] == 1
    assert store.get_unit_boost("ep_001") == pytest.approx(1.5)
    assert len(mode_a.calls) == 2                       # tool turn + clean turn
    # No third (fallback) call -- the record_feedback tool call applied 1.
    store.close()


def test_self_chat_loop_fallback_fires_when_no_record_feedback(tmp_path):
    """When the loop runs retrieval tools but no record_feedback, the
    filter-then-sum yields 0 and the structured fallback fires (one extra
    call). Three calls: expand turn + clean turn + fallback."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres",
               text="User: why\nAssistant: because")]
    expand_call = _tool_call("expand", {"unit_id": "ep_001"})
    fallback_json = '[{"unit_id":"ep_001","rating":4}]'
    mode_a = _ScriptedModeA([("draft", expand_call),
                             ("FINAL ANSWER", None),
                             (fallback_json, None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    res = orch.query("Why did we choose Postgres?")
    assert res["response"] == "FINAL ANSWER"
    assert res["feedback_collected"] == 1
    assert len(mode_a.calls) == 3                       # expand + clean + fallback
    # rating 4 -> useful 0.75 -> 1.0 * (0.5+0.75) = 1.25
    assert store.get_unit_boost("ep_001") == pytest.approx(1.25)
    store.close()


def test_self_chat_loop_feedback_disabled_offers_no_record_feedback(tmp_path):
    """Loop enabled + feedback disabled: the loop runs with LOOP_TOOLS (expand
    + search_memory) so record_feedback is NOT offered -> the boost side-effect
    stays behind the feedback gate even inside the loop. One call; no feedback."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("THE ANSWER", None), ("should never be used", None)])
    saved_fb = _config.feedback_salience_enabled
    saved_loop = _config.self_chat_tool_loop_enabled
    _config.feedback_salience_enabled = False
    # loop stays at its default (True) -- this is the loop+disabled shape.
    _config.self_chat_tool_loop_enabled = True
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        res = orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved_fb
        _config.self_chat_tool_loop_enabled = saved_loop
    assert res["response"] == "THE ANSWER"
    assert res["feedback_collected"] == 0
    assert len(mode_a.calls) == 1                       # clean answer, no fallback
    assert mode_a.calls[0]["tools"] is LOOP_TOOLS       # record_feedback excluded
    store.close()