"""End-to-end retrieval + Mode A tests on the real 20-conv sample corpus.

Episodes are built directly from the labeled fields in
``data/sample_conversations.jsonl`` (``expected_entities`` / ``expected_topics``
/ ``expected_tones`` / ``expected_decisions``) — NO GLiNER encoder — and chained
with ``follows`` so the graph mirrors a real conversation history. The full
pipeline (rule-based query planner → graph traversal → context string) is then
exercised against the real corpus data.

The Mode A test is live-gated: it runs against the Bonsai server on the pod via
an SSH tunnel (``ssh -L 8080:localhost:8080``) and ``pytest.skip``s otherwise.
"""

import json
from pathlib import Path

import pytest

from src.config import config
from src.generation.mode_a import ModeAGenerator
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.query_planner import BonsaiQueryPlanner
from src.retrieval.retriever import HippocampalRetriever

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "sample_conversations.jsonl"


class _RulePlanner:
    """Forces the deterministic rule-based planner regardless of server state.

    The offline corpus tests below are calibrated to the rule-based plan, so they
    must not silently switch to the live Bonsai planner when an SSH tunnel is up.
    """

    def __init__(self) -> None:
        self._p = BonsaiQueryPlanner()

    def plan(self, prompt: str) -> dict:
        return self._p.plan_rule_based(prompt)


def _load_corpus_episodes() -> list[Episode]:
    """Build Episodes from the labeled sample corpus, chained with follows."""
    rows = [json.loads(l) for l in _CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    eps: list[Episode] = []
    prev: str | None = None
    for i, r in enumerate(rows):
        turns = r.get("turns") or []
        u, a = turns[-1] if turns else ("", "")
        summary = a[:200] + ("..." if len(a) > 200 else "")
        eps.append(Episode(
            id=r["id"],
            timestamp=f"2026-07-{(i % 28) + 1:02d}T10:00:00",
            summary=summary,
            full_text=f"User: {u}\nAssistant: {a}",
            entities=r.get("expected_entities") or [],
            topics=r.get("expected_topics") or [],
            tones=r.get("expected_tones") or [],
            decisions=r.get("expected_decisions") or [],
            follows=prev,
        ))
        prev = r["id"]
    return eps


def _store_with_corpus(tmp_path) -> HippocampalStore:
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in _load_corpus_episodes():
        store.encode_episode(ep)
    return store


_FRUSTRATED = {"conv_002", "conv_006", "conv_010", "conv_011", "conv_013",
               "conv_015", "conv_017", "conv_018", "conv_020"}


# ── offline: planner + traversal on the real corpus ──


def test_retrieval_frustrated_on_corpus(tmp_path):
    """'What was I frustrated about?' retrieves only frustrated episodes."""
    store = _store_with_corpus(tmp_path)
    retr = HippocampalRetriever(store, planner=_RulePlanner())

    results = retr.retrieve("What was I frustrated about?")
    ids = {r["episode_id"] for r in results}
    assert ids, "expected frustrated episodes"
    assert ids <= _FRUSTRATED, f"non-frustrated episodes returned: {ids - _FRUSTRATED}"
    assert all("frustrated" in r["tones"] for r in results)
    store.close()


def test_retrieval_alice_decide_on_corpus(tmp_path):
    """'What did Alice and I decide?' → Alice ∩ decision_making = {conv_012, conv_017}."""
    store = _store_with_corpus(tmp_path)
    retr = HippocampalRetriever(store, planner=_RulePlanner())

    results = retr.retrieve("What did Alice and I decide?")
    ids = {r["episode_id"] for r in results}
    assert ids == {"conv_012", "conv_017"}, ids
    assert all("Alice" in r["entities"] for r in results)
    store.close()


# ── live: Mode A end-to-end against Bonsai (via SSH tunnel) ──


def _endpoint_up() -> bool:
    import requests
    try:
        requests.get(f"{config.bonsai_endpoint.rstrip('/')}/models", timeout=2.0)
        return True
    except Exception:
        return False


def test_mode_a_frustrated_via_live_bonsai(tmp_path):
    """Live: Mode A over the real corpus against the Bonsai server on the pod."""
    if not _endpoint_up():
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable")

    store = _store_with_corpus(tmp_path)
    retr = HippocampalRetriever(store)
    gen = ModeAGenerator(retr)

    result = gen.generate("What was I frustrated about?")
    assert isinstance(result["response"], str)
    assert result["response"].strip(), "empty Bonsai response"
    retrieved = {e["episode_id"] for e in result["retrieved_episodes"]}
    assert retrieved <= _FRUSTRATED, f"non-frustrated retrieved: {retrieved - _FRUSTRATED}"
    # The context handed to the LLM carries the frustrated episodes.
    assert "frustrated" in result["context_used"].lower()
    store.close()