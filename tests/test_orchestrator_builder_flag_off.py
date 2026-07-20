"""STRM Phase 3 Step 4: byte-identical flag-off regression.

The orchestrator's context-builder branch is guarded by
``if self.context_builder is not None and ring_capacity > 0 and relevance_head
is not None``; the ``else`` branch is the pre-Phase-3 plan() call verbatim. So
when the builder flag is off (``context_builder=None``), or the ring is off, or
no relevance head is wired, the serve path MUST be byte-identical to pre-Phase-3.
These tests pin that: the presentation plan + chunked context produced with the
builder wired-but-disabled equals the flag-off baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Phase2cConfig
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.orchestrator import PonderOrchestrator

from tests.test_orchestrator import _StubEmbedder, _StubModeA, _StubPlanner, _ep


def _build(tmp_path, ring_capacity, context_builder=None,
           relevance_head=None, reply="SYNTH RESPONSE"):
    """Build a tiny orchestrator with the stub harness + optional STRM heads."""
    store = HippocampalStore(str(tmp_path / "db"))
    ep = _ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")
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
        relevance_head=relevance_head, context_builder=context_builder,
    )
    return orch, store


def _capture_plan(orch):
    """Run a factual query and return the presentation plan + chunked ctx."""
    res = orch.query("What did Alice say?")
    assert res["supported"] is True
    return res["presentation_plan"], res["chunked"], res


def test_flag_off_plan_is_heuristic(tmp_path):
    """context_builder=None -> the heuristic PresentationGate plan (rationale is
    the heuristic's 'direct: ...', NOT 'context-builder: ...')."""
    orch, store = _build(tmp_path, ring_capacity=0)
    try:
        plan, _, _ = _capture_plan(orch)
        # 1 specific factual episode -> heuristic 'direct: ...'
        assert plan.strategy == "direct"
        assert "context-builder" not in plan.rationale
        assert "direct" in plan.rationale
    finally:
        store.close()


def test_builder_wired_but_ring_off_is_byte_identical_to_flag_off(tmp_path):
    """Builder wired + ring OFF -> the guard's ``else`` branch (heuristic) ->
    byte-identical plan + chunked context vs flag-off. Proves the guard: a
    builder checkpoint loaded with the ring off does NOT change the serve path."""
    # baseline: flag off, ring off
    orch_a, store_a = _build(tmp_path, ring_capacity=0, context_builder=None)
    # builder wired (a dummy object -- never reached) + ring OFF
    dummy_builder = object()   # the guard skips it (ring off); never called
    orch_b, store_b = _build(tmp_path, ring_capacity=0, context_builder=dummy_builder)
    try:
        plan_a, chunked_a, res_a = _capture_plan(orch_a)
        plan_b, chunked_b, res_b = _capture_plan(orch_b)
        assert plan_a == plan_b
        assert chunked_a == chunked_b
        # both are the heuristic plan (not the builder path)
        assert "context-builder" not in plan_a.rationale
    finally:
        store_a.close()
        store_b.close()


def test_builder_wired_but_no_relevance_head_is_byte_identical(tmp_path):
    """Builder wired + ring ON + NO relevance head -> guard's ``else`` branch
    (the builder needs r_i from the relevance head). Byte-identical plan vs
    flag-off. Proves the guard: a builder without a 2a head falls back."""
    # Need a dummy builder w/ a predict attr so the guard's `is not None` is
    # True but the relevance-head None check fails -> else branch.
    class _DummyBuilder:
        top_m = 5
        def predict(self, *a, **k):
            raise AssertionError("builder should not be reached (no relevance head)")
    orch_a, store_a = _build(tmp_path, ring_capacity=8, context_builder=None)
    orch_b, store_b = _build(tmp_path, ring_capacity=8,
                            context_builder=_DummyBuilder(),
                            relevance_head=None)
    try:
        plan_a, chunked_a, _ = _capture_plan(orch_a)
        plan_b, chunked_b, _ = _capture_plan(orch_b)
        assert plan_a == plan_b
        assert chunked_a == chunked_b
        assert "context-builder" not in plan_a.rationale
    finally:
        store_a.close()
        store_b.close()


def test_flag_off_deterministic_across_runs(tmp_path):
    """Flag-off is deterministic: two runs produce the same plan + chunked
    context (pins the byte-identical guarantee against future drift)."""
    orch_a, store_a = _build(tmp_path, ring_capacity=0)
    orch_b, store_b = _build(tmp_path, ring_capacity=0)
    try:
        plan_a, chunked_a, _ = _capture_plan(orch_a)
        plan_b, chunked_b, _ = _capture_plan(orch_b)
        assert plan_a == plan_b
        assert chunked_a == chunked_b
    finally:
        store_a.close()
        store_b.close()