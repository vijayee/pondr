"""Offline integration tests for the retriever orchestrator.

Encodes episodes directly (no GLiNER/Bonsai encoder) into a tmp_path WaveDB
store, then drives ``HippocampalRetriever`` with a stub planner that returns a
fixed plan — so the plan→traverse→load→context path is exercised
deterministically without the planner's server fallback in the loop.
"""

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever


class _StubPlanner:
    """Returns a fixed plan, ignoring the prompt (for deterministic tests)."""

    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history: list | None = None) -> dict:
        return self._plan


def _ep(eid, entities=None, topics=None, tones=None, summary=None, ts="2026-07-03T10:00:00"):
    return Episode(
        id=eid, timestamp=ts, summary=summary or f"summary {eid}",
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
    )


def _retriever(tmp_path, plan):
    store = HippocampalStore(str(tmp_path / "db"))
    return store, HippocampalRetriever(store, planner=_StubPlanner(plan))


def test_retrieve_end_to_end_by_tone(tmp_path):
    """plan(tone=frustrated) → only the frustrated episode is retrieved."""
    store, retr = _retriever(tmp_path, {"tones": ["frustrated"], "entity_mode": "union"})
    store.encode_episode(_ep("ep_001", tones=["frustrated"], summary="WAL config pain"))
    store.encode_episode(_ep("ep_002", tones=["excited"], summary="great news"))

    results = retr.retrieve("What was I frustrated about?")
    ids = [r["episode_id"] for r in results]
    assert ids == ["ep_001"]
    assert "frustrated" in results[0]["tones"]
    store.close()


def test_retrieve_with_plan_direct(tmp_path):
    """retrieve_with_plan skips the planner and traverses directly."""
    store, retr = _retriever(tmp_path, {})  # plan unused here
    store.encode_episode(_ep("ep_001", entities=["Alice"]))
    store.encode_episode(_ep("ep_002", entities=["Bob"]))

    results = retr.retrieve_with_plan({"entities": ["Alice"], "entity_mode": "union"})
    assert [r["episode_id"] for r in results] == ["ep_001"]
    store.close()


def test_build_context_string_structure(tmp_path):
    """Context string carries id, date, entities, topics, tone, summary per episode."""
    store, retr = _retriever(tmp_path, {"entities": ["Alice"], "entity_mode": "union"})
    store.encode_episode(
        _ep("ep_001", entities=["Alice"], topics=["database_design"],
            tones=["frustrated"], summary="We picked HBTrie for the index")
    )

    results = retr.retrieve("What did Alice and I decide?")
    ctx = retr.build_context_string(results)
    assert "[ep_001 | 2026-07-03T10:00:00]" in ctx
    assert "Alice" in ctx
    assert "database_design" in ctx
    assert "frustrated" in ctx
    assert "We picked HBTrie for the index" in ctx
    # Header is always present.
    assert ctx.startswith("You have access to relevant past conversations.")
    store.close()


def test_build_context_string_respects_token_cutoff(tmp_path):
    """Episodes past the token cutoff are dropped, not truncated."""
    store, retr = _retriever(tmp_path, {"tones": ["frustrated"], "entity_mode": "union"})
    for i in range(10):
        store.encode_episode(_ep(f"ep_{i:03d}", tones=["frustrated"],
                                 summary=f"frustrating incident number {i} " * 20))

    results = retr.retrieve("What was I frustrated about?")
    ctx = retr.build_context_string(results, max_tokens=50)
    # With a 50-token cutoff, only the header + at most a couple episodes fit.
    assert ctx.count("Summary:") <= 2
    store.close()


def test_semantic_fallback_is_noop_until_phase_f(tmp_path):
    """use_semantic with no vector index does not raise or change results."""
    store, retr = _retriever(tmp_path, {"tones": ["frustrated"], "entity_mode": "union"})
    store.encode_episode(_ep("ep_001", tones=["frustrated"]))

    results = retr.retrieve("What was I frustrated about?", use_semantic=True)
    assert [r["episode_id"] for r in results] == ["ep_001"]
    assert retr.vector_search is None  # Phase F will populate this
    store.close()