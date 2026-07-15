"""Offline tests for the Mode A generator + a live-gated end-to-end test.

Offline tests stub ``ModeAGenerator._complete`` so the retrieve→context→LLM
wiring is exercised without the Bonsai server. The live test is gated on an
endpoint probe and ``pytest.skip``s when Bonsai is unreachable (the normal
offline case); via an SSH tunnel (``ssh -L 8080:localhost:8080`` to the pod) it
runs against the real Bonsai model.
"""

import pytest

from src.config import config
from src.generation.mode_a import ModeAGenerator
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever


class _StubPlanner:
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history: list | None = None) -> dict:
        return self._plan


def _ep(eid, entities=None, topics=None, tones=None, summary=None):
    return Episode(
        id=eid, timestamp="2026-07-03T10:00:00", summary=summary or f"summary {eid}",
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
    )


def _setup(tmp_path, plan):
    store = HippocampalStore(str(tmp_path / "db"))
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan))
    return store, retriever


# ── offline (stubbed _complete) ──


def test_generate_wires_context_and_episodes(tmp_path):
    """generate() retrieves episodes, builds context, and passes both to the LLM."""
    store, retr = _setup(tmp_path, {"tones": ["frustrated"], "entity_mode": "union"})
    store.encode_episode(_ep("ep_001", tones=["frustrated"], summary="WAL config pain"))
    store.encode_episode(_ep("ep_002", tones=["excited"], summary="great news"))
    gen = ModeAGenerator(retr)

    captured: list[list[dict]] = []

    def fake_complete(messages):
        captured.append(messages)
        return ("You were frustrated about the WAL config.", None)

    gen._complete = fake_complete  # type: ignore[method-assign]

    result = gen.generate("What was I frustrated about?")

    assert result["response"] == "You were frustrated about the WAL config."
    assert result["model"] == config.generation_model
    # Only the frustrated episode was retrieved.
    assert [e["episode_id"] for e in result["retrieved_episodes"]] == ["ep_001"]
    # Context carries the frustrated episode's summary.
    assert "WAL config pain" in result["context_used"]
    # The user message handed to the LLM contains the context + the prompt.
    user_msg = captured[0][-1]
    assert user_msg["role"] == "user"
    assert "WAL config pain" in user_msg["content"]
    assert "What was I frustrated about?" in user_msg["content"]
    # System prompt is first.
    assert captured[0][0]["role"] == "system"
    store.close()


def test_generate_includes_conversation_history(tmp_path):
    """Conversation history (last 10 turns) is included before the context user msg."""
    store, retr = _setup(tmp_path, {"tones": ["frustrated"], "entity_mode": "union"})
    store.encode_episode(_ep("ep_001", tones=["frustrated"]))
    gen = ModeAGenerator(retr)

    captured: list[list[dict]] = []
    gen._complete = lambda messages: captured.append(messages) or ("ok", None)  # type: ignore[method-assign]

    history = [{"role": "user", "content": f"turn {i}"} for i in range(15)]
    gen.generate("follow up", conversation_history=history)

    msgs = captured[0]
    # system + last 10 history + final user = 12 messages.
    assert len(msgs) == 1 + 10 + 1
    assert msgs[1]["content"] == "turn 5"  # history[-10:]
    assert msgs[-1]["content"].endswith("User: follow up")
    store.close()


def test_generate_with_no_matches_still_calls_llm(tmp_path):
    """No matching episodes → context is header-only, but the LLM is still called."""
    store, retr = _setup(tmp_path, {"tones": ["frustrated"], "entity_mode": "union"})
    store.encode_episode(_ep("ep_001", tones=["excited"]))  # no frustrated ep
    gen = ModeAGenerator(retr)

    called = []
    gen._complete = lambda messages: called.append(messages) or ("no relevant context", None)  # type: ignore[method-assign]

    result = gen.generate("What was I frustrated about?")
    assert result["retrieved_episodes"] == []
    assert called  # LLM was still invoked
    # Context is header-only (no episode chunks since nothing matched).
    assert result["context_used"].startswith("You have access to relevant past conversations.")
    assert "Summary:" not in result["context_used"]
    store.close()


def test_complete_raises_on_server_failure(tmp_path):
    """_complete surfaces server errors verbatim (used by live tests)."""
    store, retr = _setup(tmp_path, {"tones": ["frustrated"], "entity_mode": "union"})
    gen = ModeAGenerator(retr, endpoint="http://127.0.0.1:9/no-such-server", timeout=1.0)
    with pytest.raises(RuntimeError):
        gen._complete([{"role": "user", "content": "hi"}])
    store.close()


# ── live Bonsai path (gated) ──


def _endpoint_up() -> bool:
    import requests
    try:
        requests.get(f"{config.bonsai_endpoint.rstrip('/')}/models", timeout=2.0)
        return True
    except Exception:
        return False


def test_generate_via_live_bonsai(tmp_path):
    """Live: Mode A end-to-end against the Bonsai server on the pod.

    Requires the SSH tunnel (``ssh -L 8080:localhost:8080`` to the pod) or a
    local Bonsai server. Uses the real retriever (rule-based planner fallback).
    """
    if not _endpoint_up():
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable")

    store = HippocampalStore(str(tmp_path / "db"))
    # Real planner (falls back to rule-based offline / uses server when up).
    retriever = HippocampalRetriever(store)
    store.encode_episode(_ep("ep_001", tones=["frustrated"], summary="WAL config pain"))
    store.encode_episode(_ep("ep_002", tones=["frustrated"], summary="encryption key rotation"))

    gen = ModeAGenerator(retriever)
    result = gen.generate("What was I frustrated about?")

    assert isinstance(result["response"], str)
    assert result["response"].strip()  # non-empty
    # Both frustrated episodes are retrieved; order is planner-dependent (the
    # live Bonsai planner scores nondeterministically), so assert as a set.
    retrieved = {e["episode_id"] for e in result["retrieved_episodes"]}
    assert retrieved == {"ep_001", "ep_002"}, retrieved
    store.close()