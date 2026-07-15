"""Live dogfood test for the self-chat full agent loop against the local
Bonsai server.

Skipped automatically when the endpoint (``config.bonsai_endpoint`` /
``localhost:8080/v1``) is unreachable, via the ``GET /v1/models`` probe (the
established skip guard -- mirrors ``test_bonsai_decider_live.py`` /
``test_bonsai_relations.py``). Run with the 8B Bonsai server up (see memory
``hippo-bonsai-local-server``); pre-warm to avoid PTX-JIT cold-start stalls.

This is the end-to-end check that ``run_tool_loop`` is wired into
``PonderOrchestrator._synthesize``: a synthesize query runs the multi-turn tool
loop against the real 8B, the model MAY call ``expand`` / ``search_memory`` mid-
generation (native tool-calling, confirmed by a prior probe), and the loop
transcript is surfaced on the result. The model is a small Q2 8B and is nudged
to answer directly when the context suffices, so a tool call is NOT guaranteed
-- the test asserts the loop PATH executed (transcript surfaced, response
non-empty) and that any tool it did call was a retrieval tool, not that it
necessarily called one.
"""

from __future__ import annotations

import pytest
import requests

from src.config import Phase2cConfig, config
from src.generation.mode_a import ModeAGenerator
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig


class _StubPlanner:
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return self._plan


class _StubEmbedder:
    """Deterministic embedder (mirrors tests/test_feedback_salience.py)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts):
        import hashlib

        out = []
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


def _ep(eid, entities=None, topics=None, summary=None, text=None) -> Episode:
    return Episode(
        id=eid, timestamp="2026-07-03T10:00:00",
        summary=summary or f"summary {eid}",
        full_text=text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=[], decisions=[],
    )


@pytest.fixture(scope="module")
def orch_live(tmp_path_factory):
    url = config.bonsai_endpoint.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable: {e}")

    tmp_path = tmp_path_factory.mktemp("selfchat")
    store = HippocampalStore(str(tmp_path / "db"))
    # Seed a couple episodes the model can retrieve + expand/search against.
    store.encode_episode(_ep(
        "ep_001", entities=["Postgres"], topics=["database"],
        summary="We chose Postgres for the memory store",
        text="User: why Postgres\nAssistant: the in-DB vector layer avoids a sidecar",
    ))
    store.encode_episode(_ep(
        "ep_002", entities=["WaveDB"], topics=["database"],
        summary="WaveDB beat FAISS in the benchmark",
        text="User: how did WaveDB compare\nAssistant: FLAT/COSINE was exact and fast",
    ))
    plan = {"entities": ["Postgres", "WaveDB"], "entity_mode": "union"}
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                    embedder=_StubEmbedder())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    mode_a = ModeAGenerator(retriever)
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=mode_a, config=cfg, user_id="live",
    )
    yield orch
    store.close()


def test_self_chat_loop_runs_against_live_bonsai(orch_live):
    """A synthesize query runs the tool loop against the live 8B; the loop
    transcript is surfaced; the response is non-empty; any tool the model did
    call is a retrieval tool (expand/search_memory), never record_feedback when
    feedback is disabled."""
    orch = orch_live
    # Ask something the pre-retrieved context only partly answers, so the
    # model has a reason to call search_memory / expand.
    res = orch.query("Why did we choose Postgres and how does WaveDB compare?",
                     auto_persist=False)
    assert res["end_state_plan"].end_state == "synthesize"
    # The loop path ran: the transcript keys are surfaced (loop enabled is the
    # default; the synthesize end-state is the only one that calls _synthesize).
    assert "loop_tool_messages" in res
    assert "loop_collected" in res
    assert "loop_exhausted" in res
    assert isinstance(res["loop_exhausted"], bool)
    # The model produced a non-empty answer.
    assert isinstance(res.get("response"), str) and res["response"].strip()
    # Any tool the loop dispatched must be a known retrieval/feedback tool.
    names = [c.get("name") for c in res["loop_collected"]]
    assert all(n in ("expand", "search_memory", "record_feedback") for n in names)
    # If a tool was called, the transcript carries the fed-back tool result.
    if names:
        roles = [m.get("role") for m in res["loop_tool_messages"]]
        assert "tool" in roles
    # The response never leaks raw tool-call JSON.
    assert "tool_calls" not in (res["response"] or "")


def test_self_chat_loop_feedback_disabled_still_runs(orch_live):
    """Loop + feedback disabled: the loop still runs (LOOP_TOOLS), no
    record_feedback is offered, the answer is returned. Confirms the
    feedback-disabled loop path is live-correct."""
    orch = orch_live
    saved = config.feedback_salience_enabled
    config.feedback_salience_enabled = False
    try:
        res = orch.query("Why did we choose Postgres?", auto_persist=False)
    finally:
        config.feedback_salience_enabled = saved
    assert res["end_state_plan"].end_state == "synthesize"
    assert isinstance(res.get("response"), str) and res["response"].strip()
    assert "loop_tool_messages" in res  # the loop still ran with LOOP_TOOLS
    # record_feedback was not offered, so it was never called.
    names = [c.get("name") for c in res["loop_collected"]]
    assert "record_feedback" not in names