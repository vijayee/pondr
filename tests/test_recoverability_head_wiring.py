"""STRM Phase 4 Step 1: 2b recoverability head + 2d v2 graduation head wiring.

Both heads ship trained this step (the 2b CLI ran a closed-form ridge fit ->
``data/training/strm_recoverability/best.pt`` GO; the 2d v2 head was trained in
Phase 2 at ``data/training/strm_graduation/best.pt``). This step wires them into
the serve path: ``build_ponder`` gains ``recoverability_head_path`` +
``graduation_head_path``, ``PonderOrchestrator`` gains the DI kwargs, and
``serve_ponder.py`` gains ``--strm-recoverability-head`` + ``--strm-graduation-head``.

The heads are ATTACHED but INERT this round: nothing reads them at serve yet
(the salience trigger that consumes them is Step 4). So the load-bearing
guarantee is flag-off byte-identical -- a head wired-but-inert must NOT change
the query result vs flag-off (pre-Phase-4). Plus loader round-trips + a sanity
prediction for each head on a synthetic state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Phase2cConfig
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.recoverability_head import (
    RecoverabilityHead, load_recoverability_head, pool_state_tensors,
)
from src.subconscious.graduation_head import (
    GraduationHeadV2, encode_llm_signal, load_graduation_head,
)

from tests.test_orchestrator import _StubEmbedder, _StubModeA, _StubPlanner, _ep

_REC_CKPT = Path("data/training/strm_recoverability/best.pt")
_GRAD_CKPT = Path("data/training/strm_graduation/best.pt")


def _build(tmp_path, *, recoverability_head=None, graduation_head=None,
           ring_capacity=0, reply="SYNTH RESPONSE"):
    """Build an orchestrator on the stub harness with the STRM heads wired."""
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
        recoverability_head=recoverability_head, graduation_head=graduation_head,
    )
    return orch, store


def _capture(orch):
    """Run a factual query and return the deterministic view of the result.

    The ``direct`` end-state path (a specific factual lookup with <=3 episodes)
    returns WITHOUT an LLM call and has NO ``response`` key; the deterministic
    serve-path outputs are ``presentation_plan`` + ``chunked`` + the retrieved
    episodes (the keys the established flag-off regression
    ``test_orchestrator_builder_flag_off`` compares). ``working_memory_state``
    carries a timestamp -> excluded from the byte-identical comparison.
    """
    res = orch.query("What did Alice say?")
    assert res["supported"] is True
    return {
        "presentation_plan": res["presentation_plan"],
        "chunked": res["chunked"],
        "retrieved_episodes": res["retrieved_episodes"],
        "supported": res["supported"],
    }


# ── loader round-trips ──

@pytest.mark.skipif(not _REC_CKPT.exists(), reason="2b checkpoint not trained")
def test_recoverability_head_loader_roundtrip():
    """load_recoverability_head -> a RecoverabilityHead on CPU, eval mode."""
    head = load_recoverability_head(str(_REC_CKPT), device="cpu")
    assert isinstance(head, RecoverabilityHead)
    assert not head.training
    # P is a single nn.Linear(1920, 1) -- the closed-form ridge baked in.
    assert head.linear.weight.shape == (1, 1920)


@pytest.mark.skipif(not _GRAD_CKPT.exists(), reason="2d v2 checkpoint not trained")
def test_graduation_head_v2_loader_roundtrip():
    """load_graduation_head -> a GraduationHeadV2 on CPU, eval mode."""
    head = load_graduation_head(str(_GRAD_CKPT), device="cpu")
    assert isinstance(head, GraduationHeadV2)
    assert not head.training


# ── head prediction sanity (sign + shape) ──

def test_recoverability_predict_shape():
    """predict(state_pooled [1,1536], anchor [1,384]) -> [1,1] scalar forgetting."""
    head = RecoverabilityHead()
    state = torch.randn(1, 1536)
    anchor = torch.randn(1, 384)
    out = head.predict(state, anchor)
    assert out.shape == (1, 1)


def test_graduation_v2_predict_shape():
    """predict(state [1,1536], slot_y [1,256], signal one-hot [1,5]) -> [1,1] in [0,1]."""
    head = GraduationHeadV2()
    state = torch.randn(1, 1536)
    slot_y = torch.randn(1, 256)
    sig = encode_llm_signal("important").unsqueeze(0)
    out = head.predict(state, slot_y, sig)
    assert out.shape == (1, 1)
    assert 0.0 <= float(out.detach()) <= 1.0


def test_pool_state_tensors_pools_four_layers():
    """pool_state_tensors([4 x (1,16,384)]) -> [1,1536] (mean over d_state, cat)."""
    state_tensors = [torch.randn(1, 16, 384) for _ in range(4)]
    pooled = pool_state_tensors(state_tensors)
    assert pooled.shape == (1, 1536)


# ── flag-off byte-identical (the load-bearing guarantee this round) ──

@pytest.mark.skipif(not _REC_CKPT.exists(), reason="2b checkpoint not trained")
def test_recoverability_head_wired_but_inert_is_byte_identical(tmp_path):
    """recoverability_head wired + attached but INERT (no trigger yet) -> the
    query result is byte-identical to flag-off. Proves the head attaches without
    changing the serve path (Phase 4's salience trigger is what reads it, Step 4)."""
    head = load_recoverability_head(str(_REC_CKPT), device="cpu")
    orch_a, store_a = _build(tmp_path, recoverability_head=None)
    orch_b, store_b = _build(tmp_path, recoverability_head=head)
    try:
        res_a = _capture(orch_a)
        res_b = _capture(orch_b)
        # The head is attached (inert) -- the plan + chunked ctx + episodes match.
        assert res_a == res_b
        assert orch_b.recoverability_head is head
        assert orch_a.recoverability_head is None
    finally:
        store_a.close()
        store_b.close()


@pytest.mark.skipif(not _GRAD_CKPT.exists(), reason="2d v2 checkpoint not trained")
def test_graduation_head_v2_wired_but_inert_is_byte_identical(tmp_path):
    """graduation_head (v2) wired + attached but INERT -> byte-identical to
    flag-off. Same guarantee as the recoverability head: the v2 head attaches
    without changing the serve path (its consumer is Phase 4's LTM promotion)."""
    head = load_graduation_head(str(_GRAD_CKPT), device="cpu")
    orch_a, store_a = _build(tmp_path, graduation_head=None)
    orch_b, store_b = _build(tmp_path, graduation_head=head)
    try:
        res_a = _capture(orch_a)
        res_b = _capture(orch_b)
        assert res_a == res_b
        assert orch_b.graduation_head is head
        assert orch_a.graduation_head is None
    finally:
        store_a.close()
        store_b.close()


def test_heads_wired_with_ring_on_still_byte_identical(tmp_path):
    """All three new heads (2b + 2d v2) wired + ring ON, but no salience trigger
    yet -> byte-identical to the same orchestrator with the heads off (ring ON
    but heads inert). The ring ON path is where the heads WILL be read; this
    pins that merely turning the ring on with the heads attached does not change
    the result before Step 4 wires the trigger."""
    # ring ON but no relevance head -> the context-builder guard's else branch
    # (heuristic) runs either way, so the only difference is the attached heads.
    orch_a, store_a = _build(tmp_path, ring_capacity=16)
    orch_b, store_b = _build(tmp_path, ring_capacity=16)
    # attach heads directly to orch_b (None-safe: the kwargs default None).
    if _REC_CKPT.exists():
        orch_b.recoverability_head = load_recoverability_head(str(_REC_CKPT), device="cpu")
    if _GRAD_CKPT.exists():
        orch_b.graduation_head = load_graduation_head(str(_GRAD_CKPT), device="cpu")
    try:
        res_a = _capture(orch_a)
        res_b = _capture(orch_b)
        assert res_a == res_b
    finally:
        store_a.close()
        store_b.close()