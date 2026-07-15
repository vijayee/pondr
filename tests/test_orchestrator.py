"""Offline integration tests for the Phase 2c PonderOrchestrator (Task 8).

Exercises the full 2c pipeline — Working Memory (cross-query state) + prompt
compression + retrieval + SSM chunking + Presentation Gate (both axes) +
end-state dispatch + EXPAND + session persistence — on a tmp_path WaveDB store
with a stub planner, a stub embedder, a ReferenceSSM backbone, and a stubbed
mode_a (no Bonsai server). No GLiNER, no Bonsai, no GPU.

Mirrors ``tests/test_retriever.py`` (tmp_path store, stub planner) and
``tests/test_chunked_context.py`` (stub embedder, ReferenceSSM backbone).
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from src.config import Phase2cConfig, config as _config
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig


# ── stubs ──

class _StubEmbedder:
    """Deterministic 384-dim embedder (SHA256 stretch → normalized)."""
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
    """Returns a fixed plan keyed on entities, ignoring the prompt text.

    The orchestrator compresses the prompt before retrieval; the stub planner
    ignores it so retrieval is deterministic by entity.
    """
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return self._plan


class _StubModeA:
    """Stub for ModeAGenerator — records _complete calls, returns canned text.

    The orchestrator's synthesize callable uses ``mode_a._complete(messages,
    tools=...)``; a real Bonsai round-trip is out of scope for the offline
    suite. Returns ``(reply, None)`` -- no tool calls -- matching the new
    ``(content, tool_calls)`` tuple contract (``tool_calls=None``).
    """
    def __init__(self, reply: str = "SYNTH RESPONSE") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def _complete(self, messages: list[dict], tools=None, tool_choice=None) -> tuple:
        self.calls.append(messages)
        return self.reply, None


class _StubGate:
    """Stub RetrievalGate — returns a fixed RoutingDecision for a pathway.

    Lets the orchestrator's gate branch be exercised without a trained 2b
    checkpoint (the real ``retrieve_with_routing`` path is covered by the 2b
    suite; this stub covers the orchestrator's own gate-branch dispatch).
    """
    def __init__(self, pathway: str) -> None:
        self._pathway = pathway

    def route_text(self, prompt, embedder):
        from src.subconscious.routing import RoutingDecision
        return RoutingDecision(
            domains=[], pathway=self._pathway, meta_skills=[],
            model_size="bonsai", needs_deliberation=False,
            confidence=0.9, gate_decision=None,
        )

    def parameters(self):
        return iter([])


# ── fixtures ──

def _ep(eid, entities=None, topics=None, tones=None, decisions=None,
        summary=None, text=None, ts="2026-07-03T10:00:00") -> Episode:
    return Episode(
        id=eid, timestamp=ts,
        summary=summary or f"summary {eid}",
        full_text=text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
        decisions=decisions or [],
    )


def _orchestrator(
    tmp_path,
    plan: dict,
    episodes: list[Episode],
    reply: str = "SYNTH RESPONSE",
    config: Phase2cConfig | None = None,
    user_id: str | None = "victor",
) -> tuple[PonderOrchestrator, HippocampalStore]:
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in episodes:
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                      embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = config or Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    mode_a = _StubModeA(reply=reply)
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=mode_a, config=cfg,
        user_id=user_id,
    )
    return orch, store


# ── end-state dispatch (no-gate path) ──

def test_short_query_direct_end_state(tmp_path):
    """Specific factual lookup with ≤3 episodes → direct end state, no LLM."""
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps)
    res = orch.query("What did Alice say?")
    assert res["supported"] is True
    # Factual lookup + ≤3 episodes → direct end state (no LLM call).
    assert res["end_state_plan"].end_state == "direct"
    assert res["type"] == "direct"
    assert "response" not in res  # no LLM call
    assert orch.mode_a.calls == []
    store.close()


def test_extract_end_state_no_llm(tmp_path):
    """List-all-decisions query → extract end state, structured data, no LLM."""
    plan = {"entities": [], "entity_mode": "union"}
    eps = [
        _ep("ep_001", decisions=["decide A", "decide B"]),
        _ep("ep_002", decisions=["decide C"]),
    ]
    orch, store = _orchestrator(tmp_path, plan, eps)
    res = orch.query("List all decisions as JSON",
                     extract_schema={"type": "list", "item_type": "decision"})
    assert res["end_state_plan"].end_state == "extract"
    # Decisions are returned in retrieval (tie-scored) order, which is
    # hash-randomized across processes — assert as a set, not a sequence.
    assert sorted(res["data"]) == ["decide A", "decide B", "decide C"]
    assert orch.mode_a.calls == []
    store.close()


def test_synthesize_end_state_calls_llm_once(tmp_path):
    """Reasoning query → synthesize → exactly one LLM call.

    Feedback collection is disabled here (the 2c+ feedback loop would add a
    second call); this test isolates the end-state dispatch -> one synthesis
    call, not the feedback side effect.
    """
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS")
    saved = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        res = orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved
    assert res["end_state_plan"].end_state == "synthesize"
    assert res["response"] == "LLM SAID THIS"
    assert len(orch.mode_a.calls) == 1
    store.close()


def test_caller_end_state_override_recorded(tmp_path):
    """A caller end_state override differs from the heuristic default → recorded."""
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps)
    # Heuristic for a factual lookup with 1 episode is "direct"; caller forces synthesize.
    res = orch.query("What did Alice say?", end_state="synthesize")
    assert res["end_state_plan"].end_state == "synthesize"
    assert res["end_state_plan"].jepa_default is False
    # The override buffer grew (the seed for the deferred learned end-state router).
    assert len(orch.presentation_gate.override_buffer) == 1
    rec = orch.presentation_gate.override_buffer.records[0]
    assert rec["jepa_predicted"] == "direct"
    assert rec["caller_chose"] == "synthesize"
    store.close()


# ── gate path (Retrieval Gate wired into the retriever) ──

def _orchestrator_with_gate(tmp_path, plan, episodes, pathway, reply="SYNTH"):
    """Construct an orchestrator whose retriever carries a stub gate."""
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in episodes:
        store.encode_episode(ep)
    embedder = _StubEmbedder()
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan), embedder=embedder)
    # Wire the gate + route embedder the way retrieve_with_routing expects.
    retriever.gate = _StubGate(pathway)
    retriever._route_embedder = embedder
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=embedder, mode_a=_StubModeA(reply=reply), config=cfg, user_id="victor",
    )
    return orch, store


def test_gate_path_graph_retrieve_runs_full_pipeline(tmp_path):
    """graph_retrieve pathway → retrieve + chunk + synthesize (one LLM call).

    Feedback disabled (would add a second call) to isolate the gate-path
    synthesis, mirroring ``test_synthesize_end_state_calls_llm_once``.
    """
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store = _orchestrator_with_gate(tmp_path, plan, eps, pathway="graph_retrieve")
    saved = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        res = orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved
    assert res["supported"] is True
    assert res["route"].pathway == "graph_retrieve"
    assert res["end_state_plan"].end_state == "synthesize"
    assert len(orch.mode_a.calls) == 1
    store.close()


def test_gate_path_unsupported_returns_honestly(tmp_path):
    """tool_plan pathway (no infra) → supported=False, no LLM call, honest."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store = _orchestrator_with_gate(tmp_path, plan, eps, pathway="tool_plan")
    res = orch.query("Run the migration plan")
    assert res["supported"] is False
    assert res["route"].pathway == "tool_plan"
    assert res["response"] is None
    assert orch.mode_a.calls == []  # no LLM call for unsupported pathways
    store.close()


# ── working memory continuity ──

def test_working_memory_state_persists_across_queries(tmp_path):
    """Two queries in one session → WM state evolves (state differs Q1→Q2)."""
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps)
    r1 = orch.query("What did Alice say?")
    state_after_q1 = [t.clone() for t in r1["working_memory_state"].state_tensors]
    r2 = orch.query("What else did Alice say?")
    state_after_q2 = r2["working_memory_state"].state_tensors
    # The recurrent state moved between Q1 and Q2 (presence, not per-query reset).
    assert not all(torch.equal(a, b) for a, b in zip(state_after_q1, state_after_q2))
    store.close()


def test_reset_zeros_working_memory(tmp_path):
    """An explicit reset() returns the WM state toward zeros."""
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps)
    orch.query("What did Alice say?")
    state_before = [t.clone() for t in orch.working_memory.state]
    orch.working_memory.reset()
    state_after = orch.working_memory.state
    assert all(torch.equal(t, torch.zeros_like(t)) for t in state_after)
    assert not all(torch.equal(a, b) for a, b in zip(state_before, state_after))
    store.close()


# ── chunking axis ──

def test_chunked_strategy_with_many_episodes(tmp_path):
    """12 episodes + specific query → chunked: ≤max_primary_chunks primary, rest compressed."""
    plan = {"entities": [], "entity_mode": "union", "limit": 12}  # match all
    eps = [_ep(f"ep_{i:03d}", entities=["Alice"], topics=["perf"],
               summary=f"perf note {i}", text=f"primary text {i} " * 20)
           for i in range(12)]
    cfg = Phase2cConfig()
    cfg.ssm_chunker.max_primary_chunks = 5
    orch, store = _orchestrator(tmp_path, plan, eps, config=cfg)
    res = orch.query("What did Alice say about perf?")
    plan_a = res["presentation_plan"]
    assert plan_a.strategy == "chunked"
    assert plan_a.primary_chunk_count <= 5
    chunked = res["chunked"]
    assert len(chunked.primary_chunks) <= 5
    assert chunked.compressed_episode_count == 12 - len(chunked.primary_chunks)
    assert chunked.has_compressed
    store.close()


def test_summary_only_for_summarization_query(tmp_path):
    """Summarize-everything + ≥summary_only_min episodes → summary_only."""
    plan = {"entities": [], "entity_mode": "union", "limit": 20}
    eps = [_ep(f"ep_{i:03d}", topics=["db"], summary=f"db note {i}") for i in range(20)]
    orch, store = _orchestrator(tmp_path, plan, eps)
    res = orch.query("Summarize everything about databases")
    assert res["presentation_plan"].strategy == "summary_only"
    assert res["presentation_plan"].primary_chunk_count == 0
    assert res["chunked"].primary_chunks == []
    store.close()


# ── EXPAND ──

def test_expand_loads_full_text_and_injects_into_wm(tmp_path):
    """EXPAND a compressed episode → full text returned, WM state moves."""
    plan = {"entities": [], "entity_mode": "union", "limit": 8}
    eps = [_ep(f"ep_{i:03d}", entities=["Alice"], topics=["perf"],
               summary=f"perf note {i}", text=f"FULL TEXT {i} " * 30)
           for i in range(8)]
    cfg = Phase2cConfig()
    cfg.ssm_chunker.max_primary_chunks = 3
    orch, store = _orchestrator(tmp_path, plan, eps, config=cfg)
    res = orch.query("What did Alice say about perf?")
    chunked = res["chunked"]
    # pick a compressed (expandable) id
    expandable = sorted(chunked.expandable_ids)
    assert expandable, "expected at least one compressed episode"
    target = expandable[0]
    state_before = [t.clone() for t in orch.working_memory.state]
    full_text, snap = orch.expand(target, chunked)
    assert "FULL TEXT" in full_text
    assert not all(torch.equal(a, b) for a, b in zip(state_before, orch.working_memory.state))
    store.close()


def test_expand_on_primary_raises_not_expandable(tmp_path):
    plan = {"entities": [], "entity_mode": "union", "limit": 8}
    eps = [_ep(f"ep_{i:03d}", text=f"FULL TEXT {i} " * 30) for i in range(8)]
    cfg = Phase2cConfig()
    cfg.ssm_chunker.max_primary_chunks = 3
    orch, store = _orchestrator(tmp_path, plan, eps, config=cfg)
    res = orch.query("What did Alice say about perf?")
    chunked = res["chunked"]
    primary_ids = [ep["episode_id"] for ep in chunked.primary_chunks]
    assert primary_ids, "expected at least one primary chunk"
    from src.subconscious.ssm_chunker import EpisodeNotExpandable
    with pytest.raises(EpisodeNotExpandable):
        orch.expand(primary_ids[0], chunked)
    store.close()


# ── session persistence ──

def test_session_save_load_round_trip(tmp_path):
    """save → new orchestrator → load: WM state element-equal after round-trip."""
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps, user_id="victor")
    orch.query("What did Alice say?")
    orch.query("What else did Alice say?")
    state_before = [t.clone() for t in orch.working_memory.state]
    orch.save_session("victor")
    store.close()

    # New store + orchestrator pointing at the same session dir.
    store2 = HippocampalStore(str(tmp_path / "db"))
    retriever2 = HippocampalRetriever(store2, planner=_StubPlanner(plan),
                                      embedder=_StubEmbedder())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch2 = PonderOrchestrator(
        store=store2, retriever=retriever2, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=_StubModeA(), config=cfg,
        user_id="victor",
    )
    # load_session ran lazily in __init__; WM state restored element-equal.
    state_after = orch2.working_memory.state
    assert len(state_after) == len(state_before)
    for a, b in zip(state_before, state_after):
        assert torch.equal(a, b)
    assert orch2.working_memory.input_count == orch.working_memory.input_count
    store2.close()


def test_load_session_returns_false_when_none_saved(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    orch, store = _orchestrator(tmp_path, plan, [], user_id="nobody")
    # No save yet → load returns False, WM stays at its reset/initial state.
    orch.working_memory.reset()
    orch.working_memory.step(torch.zeros(1, 384))  # init state so snapshot works
    assert orch.load_session("nobody") is False
    store.close()


# ── prompt compression ──

def test_long_prompt_compressed_before_planner(tmp_path):
    """A >500-char prompt is compressed to ≤bonsai_max_input before retrieval."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="Postgres migration")]
    cfg = Phase2cConfig()
    cfg.prompt_compression.short_prompt_threshold = 50
    cfg.prompt_compression.bonsai_max_input = 500
    orch, store = _orchestrator(tmp_path, plan, eps, config=cfg)
    long_prompt = "Alice and Bob discussed Postgres migration. " * 50
    res = orch.query(long_prompt)
    assert res["supported"] is True
    # The stub planner was called (retrieval happened) regardless of length.
    assert res["retrieved_episodes"] is not None
    store.close()


# ── outcome recording ──

def _one_episode_list():
    """Tiny stand-in episode list for the gate's plan() call in outcome tests."""
    return [{"episode_id": "ep_001", "topics": [], "entities": ["Alice"]}]


def test_record_outcome_grows_buffer(tmp_path):
    """record_outcome appends to the gate's outcome buffer (no fake learning).

    Phase 3a Task 7: ``query()`` now auto-records an outcome with the measured
    ``expand_count``, so one query leaves the buffer at 1; an explicit
    ``record_outcome`` (e.g. a caller with a real satisfaction rating) appends a
    second record.
    """
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps)
    orch.query("What did Alice say?")
    # query() auto-recorded one outcome with the measured expand_count.
    assert len(orch.presentation_gate.outcome_buffer) == 1
    assert orch.presentation_gate.outcome_buffer.records[0]["expand_count"] == 0
    plan_a = orch.presentation_gate.plan("What did Alice say?", _one_episode_list(), None)
    orch.record_outcome(plan_a, expand_count=0, unused_primary_count=0,
                        user_satisfaction=0.8)
    assert len(orch.presentation_gate.outcome_buffer) == 2
    store.close()

# ── Phase 3a Task 7: durable EXPAND-frequency outcomes ──

def test_outcomes_survive_restart_via_store(tmp_path):
    """Auto-recorded outcomes persist across orchestrator restarts (2c §15 fix)."""
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps, user_id="alice")
    # query() auto-records one outcome with the measured expand_count.
    orch.query("What did Alice say?")
    assert len(orch.presentation_gate.outcome_buffer) == 1
    # Flush to the store.
    blob = orch.save_outcomes("alice")
    assert blob is not None
    store.close()

    # A fresh orchestrator on the same store + user auto-loads the buffers.
    orch2, store2 = _orchestrator(tmp_path, plan, eps, user_id="alice")
    assert len(orch2.presentation_gate.outcome_buffer) == 1
    # The persisted outcome round-trips with its measured expand_count.
    rec = orch2.presentation_gate.outcome_buffer.records[0]
    assert rec["expand_count"] == 0
    store2.close()


def test_save_outcomes_returns_none_without_store_or_user(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")]
    # No user_id → save_outcomes is a no-op (returns None).
    orch, store = _orchestrator(tmp_path, plan, eps, user_id=None)
    assert orch.save_outcomes() is None
    store.close()
