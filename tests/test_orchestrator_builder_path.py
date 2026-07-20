"""STRM Phase 3 Step 4: context-builder serve-path tests.

Covers the ``_plan_with_context_builder`` contract + the call-site try/except
fallback to the heuristic PresentationGate. The builder branch is best-effort:
any failure (builder raises, empty ring, no matching slots, no selection)
falls back so the turn never crashes -- pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Phase2cConfig
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.orchestrator import PonderOrchestrator
from src.subconscious.presentation_gate import CHUNKED, DIRECT

from tests.test_orchestrator import _StubEmbedder, _StubModeA, _StubPlanner, _ep


# ── Mocks ────────────────────────────────────────────────────────────────────

class _MockRelHead(torch.nn.Module):
    """Frozen 2a relevance head: returns a constant r_i (shape [K,1]). The
    builder path only needs SOME r_i bias; the value doesn't matter for these
    wiring tests (the mock builder ignores it). Subclasses nn.Module so
    ``next(head.parameters()).device`` works inside score_ring_slots_*."""

    def __init__(self, r_value: float = 0.5):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self._r_value = r_value

    def predict(self, ys, ds, q):  # noqa: ANN001 - mirrors RelevanceHead.predict
        k = ys.shape[0]
        return torch.full((k, 1), self._r_value, dtype=torch.float32)


class _MockBuilder:
    """ContextBuilder stand-in. ``top_m`` is read by the orchestrator's guard
    comment (not the code -- predict gets no ``m`` arg at serve), and
    ``predict`` returns the canned indices + a scores tensor. Pass
    ``raise_on_predict`` to exercise the call-site fallback."""

    def __init__(self, idx_list, top_m=5, raise_on_predict=False):
        self.top_m = top_m
        self._idx = list(idx_list)
        self._raise = raise_on_predict

    def predict(self, slots_y, slots_doc_emb, query_emb, r, m=None):  # noqa: ANN001
        if self._raise:
            raise RuntimeError("mock builder boom")
        scores = torch.zeros(len(self._idx), dtype=torch.float32)
        return list(self._idx), scores


# ── Harness ─────────────────────────────────────────────────────────────────

def _build(tmp_path, *, ring_capacity, builder=None, rel_head=None,
            episodes=None, reply="SYNTH RESPONSE"):
    """Build an orchestrator on the stub harness with the STRM heads wired.

    Encodes ``episodes`` into the store so the stub-planner-driven retriever
    surfaces them; the orchestrator then injects each retrieved episode into
    the WM ring (provenance source_id=episode_id, text=summary), which is what
    the builder path attends over."""
    store = HippocampalStore(str(tmp_path / "db"))
    eps = episodes or [
        _ep("ep_001", entities=["Alice"], summary="Alice said use Postgres"),
    ]
    for ep in eps:
        store.encode_episode(ep)
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                    embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=_StubModeA(reply=reply),
        config=cfg, user_id="victor", ring_capacity=ring_capacity,
        relevance_head=rel_head, context_builder=builder,
    )
    return orch, store


def _inject_slots(orch, n=4):
    """Manually inject ``n`` ring slots with source_ids ep_001..ep_00N + text.

    Mirrors what orchestrator.query does at lines 379-383 (embed the summary,
    inject with source_id+text) but WITHOUT running the full query, so the
    builder-path unit test controls the ring contents + order directly."""
    for i in range(1, n + 1):
        sid = f"ep_{i:03d}"
        text = f"summary {sid}"
        emb = orch.working_memory.embed([text])[0]   # [1, 384]
        orch.working_memory.inject(emb, source_id=sid, text=text)


def _episode_dicts(n=4):
    """Retrieved-episode dicts (what the retriever hands the orchestrator)."""
    return [{"episode_id": f"ep_{i:03d}", "summary": f"summary ep_{i:03d}",
             "text": f"text ep_{i:03d}", "topics": []} for i in range(1, n + 1)]


# ── _plan_with_context_builder unit tests ───────────────────────────────────

def test_builder_reorders_episodes(tmp_path):
    """Mock builder returns slot indices [2, 0] of a 4-slot ring -> selected
    source_ids [ep_003, ep_001], ordered = [ep_003, ep_001, ep_002, ep_004],
    plan.primary_chunk_count = 2, strategy = CHUNKED (2 < 4)."""
    orch, store = _build(tmp_path, ring_capacity=16,
                        builder=_MockBuilder([2, 0], top_m=5),
                        rel_head=_MockRelHead())
    try:
        _inject_slots(orch, n=4)
        prompt_emb = orch.working_memory.embed(["What did Alice say?"])[0]
        episodes = _episode_dicts(n=4)

        plan, ordered = orch._plan_with_context_builder(
            "What did Alice say?", episodes, prompt_emb)

        ordered_ids = [e["episode_id"] for e in ordered]
        assert ordered_ids == ["ep_003", "ep_001", "ep_002", "ep_004"]
        assert plan.primary_chunk_count == 2
        assert plan.strategy == CHUNKED
        assert "context-builder" in plan.rationale
    finally:
        store.close()


def test_builder_selects_all_is_direct(tmp_path):
    """Mock builder returns ALL slot indices -> m == len(ordered) -> DIRECT."""
    orch, store = _build(tmp_path, ring_capacity=16,
                        builder=_MockBuilder([0, 1, 2], top_m=5),
                        rel_head=_MockRelHead())
    try:
        _inject_slots(orch, n=3)
        prompt_emb = orch.working_memory.embed(["q"])[0]
        episodes = _episode_dicts(n=3)

        plan, ordered = orch._plan_with_context_builder("q", episodes, prompt_emb)

        ordered_ids = [e["episode_id"] for e in ordered]
        assert ordered_ids == ["ep_001", "ep_002", "ep_003"]
        assert plan.primary_chunk_count == 3
        assert plan.strategy == DIRECT
    finally:
        store.close()


def test_empty_ring_raises(tmp_path):
    """Ring ON but empty (no injections) -> RuntimeError, caught by the call
    site's try/except -> heuristic fallback (exercised in the query test)."""
    orch, store = _build(tmp_path, ring_capacity=16,
                        builder=_MockBuilder([0], top_m=5),
                        rel_head=_MockRelHead())
    try:
        prompt_emb = orch.working_memory.embed(["q"])[0]
        episodes = _episode_dicts(n=2)
        with pytest.raises(RuntimeError, match="ring empty"):
            orch._plan_with_context_builder("q", episodes, prompt_emb)
    finally:
        store.close()


def test_no_matching_slots_raises(tmp_path):
    """Ring populated but no slot's source_id maps to a retrieved episode ->
    RuntimeError (caught at the call site -> heuristic fallback)."""
    orch, store = _build(tmp_path, ring_capacity=16,
                        builder=_MockBuilder([0], top_m=5),
                        rel_head=_MockRelHead())
    try:
        _inject_slots(orch, n=2)   # ring slots: ep_001, ep_002
        prompt_emb = orch.working_memory.embed(["q"])[0]
        # retrieved episodes have DIFFERENT ids -> no matching slots
        episodes = [{"episode_id": "ep_999", "summary": "x", "text": "x", "topics": []}]
        with pytest.raises(RuntimeError, match="no ring slots map"):
            orch._plan_with_context_builder("q", episodes, prompt_emb)
    finally:
        store.close()


def test_no_episodes_raises(tmp_path):
    """Empty episodes list -> RuntimeError (caller heuristic handles it)."""
    orch, store = _build(tmp_path, ring_capacity=16,
                        builder=_MockBuilder([0], top_m=5),
                        rel_head=_MockRelHead())
    try:
        _inject_slots(orch, n=2)
        prompt_emb = orch.working_memory.embed(["q"])[0]
        with pytest.raises(RuntimeError, match="no episodes"):
            orch._plan_with_context_builder("q", [], prompt_emb)
    finally:
        store.close()


def test_empty_selection_raises(tmp_path):
    """Builder returns an empty selection -> RuntimeError -> heuristic fallback."""
    orch, store = _build(tmp_path, ring_capacity=16,
                        builder=_MockBuilder([], top_m=5),
                        rel_head=_MockRelHead())
    try:
        _inject_slots(orch, n=2)
        prompt_emb = orch.working_memory.embed(["q"])[0]
        episodes = _episode_dicts(n=2)
        with pytest.raises(RuntimeError, match="no selection"):
            orch._plan_with_context_builder("q", episodes, prompt_emb)
    finally:
        store.close()


# ── Call-site fallback (full query) ─────────────────────────────────────────

def test_builder_fallback_on_exception(tmp_path, capsys):
    """Full query: builder.predict raises -> the call-site try/except catches it
    -> heuristic PresentationGate -> query succeeds (supported=True) with the
    heuristic plan rationale (NOT 'context-builder: ...')."""
    orch, store = _build(tmp_path, ring_capacity=16,
                        builder=_MockBuilder([0], top_m=5, raise_on_predict=True),
                        rel_head=_MockRelHead())
    try:
        res = orch.query("What did Alice say?")
        assert res["supported"] is True
        plan = res["presentation_plan"]
        assert "context-builder" not in plan.rationale
        # the stderr fallback notice was printed (best-effort, not asserted text)
        capsys.readouterr()
    finally:
        store.close()


def test_builder_path_succeeds_end_to_end(tmp_path):
    """Full query with a working builder: the builder path runs end-to-end and
    the plan rationale carries the 'context-builder:' marker."""
    orch, store = _build(tmp_path, ring_capacity=16,
                        builder=_MockBuilder([0], top_m=5),
                        rel_head=_MockRelHead())
    try:
        res = orch.query("What did Alice say?")
        assert res["supported"] is True
        plan = res["presentation_plan"]
        assert "context-builder" in plan.rationale
    finally:
        store.close()