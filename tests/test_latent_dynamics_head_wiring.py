"""STRM Phase 4 Step 2: 2c latent-dynamics head wiring.

The 2c head ships trained this step (the CLI ran a closed-form ridge fit ->
``data/training/strm_latent_dynamics/best.pt`` GO: R^2=0.66, surprise-AUC=0.84).
This step wires it into the serve path: ``build_ponder`` gains
``latent_dynamics_head_path``, ``PonderOrchestrator`` gains the DI kwarg, and
``serve_ponder.py`` gains ``--strm-latent-dynamics-head``.

Like the 2b recoverability head, the 2c head is ATTACHED but INERT this round
(its ``surprise()`` consumer is the Phase 4 salience trigger, Step 4). So the
load-bearing guarantee is flag-off byte-identical, plus the loader round-trip +
``surprise()`` sign/shape correctness (high residual -> high surprise; self-
prediction -> ~0).
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
from src.subconscious.latent_dynamics_head import (
    LatentDynamicsHead, load_latent_dynamics_head,
)

from tests.test_orchestrator import _StubEmbedder, _StubModeA, _StubPlanner, _ep

_LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")


def _build(tmp_path, *, latent_dynamics_head=None, ring_capacity=0,
           reply="SYNTH RESPONSE"):
    """Build an orchestrator on the stub harness with the 2c head wired."""
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
        latent_dynamics_head=latent_dynamics_head,
    )
    return orch, store


def _capture(orch):
    """Run a factual query and return the deterministic view of the result.

    The ``direct`` end-state path returns WITHOUT an LLM call and has NO
    ``response`` key; the deterministic serve-path outputs are
    ``presentation_plan`` + ``chunked`` + the retrieved episodes (the keys the
    established flag-off regression compares). ``working_memory_state`` carries
    a timestamp -> excluded from the byte-identical comparison.
    """
    res = orch.query("What did Alice say?")
    assert res["supported"] is True
    return {
        "presentation_plan": res["presentation_plan"],
        "chunked": res["chunked"],
        "retrieved_episodes": res["retrieved_episodes"],
        "supported": res["supported"],
    }


# ── loader round-trip ──

@pytest.mark.skipif(not _LD_CKPT.exists(), reason="2c checkpoint not trained")
def test_latent_dynamics_head_loader_roundtrip():
    """load_latent_dynamics_head -> a LatentDynamicsHead on CPU, eval mode."""
    head = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    assert isinstance(head, LatentDynamicsHead)
    assert not head.training
    # A is a single nn.Linear(384, 384).
    assert head.linear.weight.shape == (384, 384)


# ── surprise sign + shape (the load-bearing correctness for Step 4) ──

def test_surprise_shape_is_per_row_scalar():
    """surprise(z_t [N,384], z_{t+1} [N,384]) -> [N] per-row scalar."""
    head = LatentDynamicsHead()
    z_t = torch.randn(4, 384)
    z_tp1 = torch.randn(4, 384)
    surp = head.surprise(z_t, z_tp1)
    assert surp.shape == (4,)


def test_surprise_self_prediction_is_near_zero():
    """surprise(z, predict(z)) ~ 0 -- the model predicts its own output exactly
    (the residual is the prediction error, so a self-target has none)."""
    head = LatentDynamicsHead()
    z = torch.randn(3, 384)
    surp = head.surprise(z, head.predict(z))
    assert float(surp.detach().max()) < 1e-5


def test_surprise_high_residual_is_higher_than_low():
    """The load-bearing sign for Step 4: a LARGE prediction residual -> HIGH
    surprise (an unexpected transition); a small residual -> low surprise.
    Surprise SUPPRESSES salience (high surprise -> the turn is novel, don't
    pre-empt with retrieval). This pins the sign so Step 4 wires it the right
    way around (surprise < surprise_cap, NOT surprise > cap)."""
    head = LatentDynamicsHead()
    z_t = torch.randn(64, 384)
    # low residual: target = prediction + tiny noise
    pred = head.predict(z_t)
    z_low = pred + 0.01 * torch.randn_like(pred)
    # high residual: target = random (mismatched next-state)
    z_high = torch.randn(64, 384)
    surp_low = head.surprise(z_t, z_low).mean()
    surp_high = head.surprise(z_t, z_high).mean()
    assert float(surp_high) > float(surp_low)


def test_project_last_layer_mean_over_d_state():
    """project([4 x (1,16,384)]) -> [1,384] (last layer, mean over d_state)."""
    head = LatentDynamicsHead()
    state_tensors = [torch.randn(1, 16, 384) for _ in range(4)]
    z = head.project(state_tensors)
    assert z.shape == (1, 384)


# ── flag-off byte-identical (the load-bearing guarantee this round) ──

@pytest.mark.skipif(not _LD_CKPT.exists(), reason="2c checkpoint not trained")
def test_latent_dynamics_head_wired_but_inert_is_byte_identical(tmp_path):
    """latent_dynamics_head wired + attached but INERT (no trigger yet) -> the
    query result is byte-identical to flag-off. Proves the head attaches without
    changing the serve path (its surprise() consumer is the Step 4 salience
    trigger)."""
    head = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    orch_a, store_a = _build(tmp_path, latent_dynamics_head=None)
    orch_b, store_b = _build(tmp_path, latent_dynamics_head=head)
    try:
        res_a = _capture(orch_a)
        res_b = _capture(orch_b)
        assert res_a == res_b
        assert orch_b.latent_dynamics_head is head
        assert orch_a.latent_dynamics_head is None
    finally:
        store_a.close()
        store_b.close()