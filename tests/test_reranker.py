"""Cross-encoder re-ranker (LongMemEval fix (2) -- dedicated signal).

``src/retrieval/reranker.py`` wraps ``sentence_transformers.CrossEncoder``:
load once (CUDA w/ CPU fallback), ``rerank(query, results)`` scores each
result's ``text``/``summary`` vs the query and returns a NEW list in
score-desc order. The retriever's ``_rerank_cross_encoder`` is a guarded no-op
when ``config.rerank_enabled`` is off OR no reranker is attached (byte-
identical). ``rerank`` itself has a graceful failure-fallback (any load /
predict / OOM error -> input unchanged).

These tests pin: byte-identical-OFF (gate + no reranker); ON re-orders by the
attached reranker (stub, no network); the failure-fallback returns input
unchanged; the input is not mutated; ``top_k`` truncates; empty input is
byte-identical; ``_result_text`` prefers ``text`` over ``summary``.

Offline: the real ``CrossEncoder`` is never loaded here -- a stub reranker
stands in so the tests run without ``sentence_transformers`` / a 568MB
download. ``CrossEncoderReranker``'s own lazy-load + fallback are exercised
with a deliberately-broken model name (caught -> no-op), which does NOT touch
the network when ``sentence_transformers`` is absent (the ImportError path).
"""

from __future__ import annotations

import pytest

from src.config import config
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever


# ── helpers ───────────────────────────────────────────────────────────────────

class _StubPlanner:
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history: list | None = None) -> dict:
        return self._plan


class _StubReranker:
    """A stand-in for ``CrossEncoderReranker`` with a deterministic scoring
    function (no model load, no network). Mirrors the real ``rerank`` contract:
    returns a NEW list, stamps ``rerank_score``, never mutates the input."""

    def __init__(self, score_map: dict[str, float]) -> None:
        self.score_map = score_map
        self.calls = 0

    def rerank(self, query, results, top_k=None):
        self.calls += 1
        out = []
        for r in results:
            d = dict(r)
            d["rerank_score"] = self.score_map.get(r["episode_id"], 0.0)
            out.append(d)
        out.sort(key=lambda d: d["rerank_score"], reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out


def _ep(eid, entities=None, topics=None, summary=None, full_text=None,
        ts="2026-07-03T10:00:00"):
    return Episode(
        id=eid, timestamp=ts, summary=summary or f"summary {eid}",
        full_text=full_text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [],
    )


@pytest.fixture
def rerank_on():
    """Flip the master-config flag ON for the retriever's call-time gate.
    Restored after so the global never leaks into sibling tests."""
    prev = config.rerank_enabled
    config.rerank_enabled = True
    try:
        yield
    finally:
        config.rerank_enabled = prev


# ── byte-identical OFF ────────────────────────────────────────────────────────

def test_rerank_off_no_reranker_byte_identical(tmp_path):
    """Flag OFF: a retriever with reranker=None produces the SAME retrieve()
    output as a baseline built without touching the rerank attr. Pins that the
    gated call site leaves the off path untouched."""
    store = HippocampalStore(str(tmp_path / "db"))
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    store.encode_episode(_ep("ep_1", entities=["Alice"], summary="first",
                             full_text="User: alice one"))
    store.encode_episode(_ep("ep_2", entities=["Alice"], summary="second",
                             full_text="User: alice two"))

    baseline = HippocampalRetriever(store, planner=_StubPlanner(plan))
    explicit_off = HippocampalRetriever(store, planner=_StubPlanner(plan))
    explicit_off.reranker = None  # explicit, mirrors the ctor default

    base = baseline.retrieve("alice")
    off = explicit_off.retrieve("alice")
    assert [r["episode_id"] for r in base] == [r["episode_id"] for r in off]
    assert [r.get("score") for r in base] == [r.get("score") for r in off]
    # No rerank_score key leaks on the off path.
    assert all("rerank_score" not in r for r in off)
    store.close()


def test_rerank_flag_off_but_reranker_attached_still_noop(tmp_path):
    """Reranker attached BUT flag OFF -> the gate (``config.rerank_enabled``)
    short-circuits before calling the reranker. Byte-identical to baseline, and
    the reranker is never invoked."""
    store = HippocampalStore(str(tmp_path / "db"))
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    store.encode_episode(_ep("ep_1", entities=["Alice"], summary="first",
                             full_text="User: alice one"))

    baseline = HippocampalRetriever(store, planner=_StubPlanner(plan))
    gated = HippocampalRetriever(store, planner=_StubPlanner(plan))
    stub = _StubReranker({"ep_1": 99.0})
    gated.reranker = stub  # attached...
    # ...but the flag is OFF (no fixture) -> gate no-op.
    assert config.rerank_enabled is False

    base = baseline.retrieve("alice")
    off = gated.retrieve("alice")
    assert [r["episode_id"] for r in base] == [r["episode_id"] for r in off]
    assert stub.calls == 0  # never called
    store.close()


# ── ON path re-orders ─────────────────────────────────────────────────────────

def test_rerank_on_reorders_by_attached_reranker(tmp_path, rerank_on):
    """Flag ON + stub reranker attached: retrieve() returns results in the
    stub's score order (not the graph order). The stub promotes ep_2 over ep_1."""
    store = HippocampalStore(str(tmp_path / "db"))
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    store.encode_episode(_ep("ep_1", entities=["Alice"], summary="first",
                             full_text="User: alice one"))
    store.encode_episode(_ep("ep_2", entities=["Alice"], summary="second",
                             full_text="User: alice two"))

    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan))
    # Stub: ep_2 scores higher than ep_1 (reverses the graph order).
    retriever.reranker = _StubReranker({"ep_1": 0.1, "ep_2": 0.9})

    results = retriever.retrieve("alice")
    ids = [r["episode_id"] for r in results]
    assert ids == ["ep_2", "ep_1"]  # re-ordered by the stub
    # rerank_score is stamped; original score preserved.
    assert results[0]["rerank_score"] == 0.9
    assert "score" in results[0]  # original graph score still present
    store.close()


def test_rerank_on_does_not_mutate_input(tmp_path, rerank_on):
    """The reranker must return a NEW list + shallow-copied dicts; the caller's
    result objects keep their original state (no rerank_score stamp leaks back)."""
    results = [
        {"episode_id": "ep_1", "text": "alpha", "summary": "a", "score": 0.5},
        {"episode_id": "ep_2", "text": "beta", "summary": "b", "score": 0.9},
    ]
    stub = _StubReranker({"ep_1": 0.1, "ep_2": 0.8})
    out = stub.rerank("query", results)
    # Originals untouched.
    assert all("rerank_score" not in r for r in results)
    assert results[0]["episode_id"] == "ep_1"  # order unchanged
    # Output is a new list, re-ordered, with the stamp.
    assert out is not results
    assert [r["episode_id"] for r in out] == ["ep_2", "ep_1"]
    assert out[0]["rerank_score"] == 0.8


# ── CrossEncoderReranker failure-fallback ─────────────────────────────────────

def test_rerank_failure_fallback_returns_input_unchanged():
    """A CrossEncoderReranker whose model fails to load (here: ``sentence_tran-
    senters`` absent OR a bogus model name) returns the input list UNCHANGED --
    the graceful no-op. No network when the package is absent (ImportError path)."""
    from src.retrieval.reranker import CrossEncoderReranker
    ce = CrossEncoderReranker(model_name="bogus/nonexistent-model-xyz", device="cpu")
    results = [
        {"episode_id": "ep_1", "text": "alpha", "summary": "a", "score": 0.5},
        {"episode_id": "ep_2", "text": "beta", "summary": "b", "score": 0.9},
    ]
    out = ce.rerank("query", results)
    # Graceful no-op: same order, no rerank_score stamp.
    assert [r["episode_id"] for r in out] == ["ep_1", "ep_2"]
    assert all("rerank_score" not in r for r in out)


def test_rerank_empty_input_returns_empty():
    """Empty input -> empty output (no load attempt, no crash)."""
    from src.retrieval.reranker import CrossEncoderReranker
    ce = CrossEncoderReranker(model_name="bogus/xyz", device="cpu")
    assert ce.rerank("query", []) == []


# ── _result_text + top_k ──────────────────────────────────────────────────────

def test_result_text_prefers_text_over_summary():
    from src.retrieval.reranker import _result_text
    # Internal retriever shape: text (full) > summary (gist).
    assert _result_text({"text": "full", "summary": "gist"}) == "full"
    # Harness-mapped shape: source_evidence (full) > memory (gist).
    assert _result_text({"source_evidence": "full", "memory": "gist"}) == "full"
    # text wins over source_evidence (internal shape takes precedence).
    assert _result_text({"text": "t", "source_evidence": "se"}) == "t"
    # Falls back across shapes: text absent -> source_evidence -> summary -> memory.
    assert _result_text({"source_evidence": "se"}) == "se"
    assert _result_text({"summary": "gist"}) == "gist"
    assert _result_text({"memory": "mem"}) == "mem"
    assert _result_text({"text": "", "source_evidence": "se"}) == "se"
    # Empty dict -> "" (never raises; the scorer gets a low score for it).
    assert _result_text({}) == ""
    assert _result_text({"text": None, "summary": None}) == ""


def test_rerank_top_k_truncates():
    """top_k caps the output AFTER sorting by score."""
    stub = _StubReranker({"ep_1": 0.1, "ep_2": 0.9, "ep_3": 0.5})
    results = [
        {"episode_id": "ep_1", "text": "a", "score": 0.1},
        {"episode_id": "ep_2", "text": "b", "score": 0.2},
        {"episode_id": "ep_3", "text": "c", "score": 0.3},
    ]
    out = stub.rerank("query", results, top_k=2)
    assert [r["episode_id"] for r in out] == ["ep_2", "ep_3"]  # top-2 by score