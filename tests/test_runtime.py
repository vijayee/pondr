"""Tests for the runtime entrypoint (``src.runtime.build_ponder``) + ``load_retrieval_gate``.

Closes the SSM/JEPA consistency gap found 2026-07-15: the trained Phase 2a
backbone + Phase 2b RetrievalGate existed on disk but were never loaded into a
live query path -- every ``PonderOrchestrator`` construction lived in tests and
passed a FRESH ``JGSBackbone``, so the SSM/JEPA ran on untrained random weights.
These tests confirm the loaders restore the trained weights and ``build_ponder``
wires the trained gate on the frozen backbone into a live orchestrator.

Offline: stub embedder + stub mode_a, real on-disk trained checkpoints. No
Bonsai, no GLiNER. Skipped when the pod-trained checkpoints aren't local.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.runtime import DEFAULT_BACKBONE_PATH, DEFAULT_GATE_PATH, build_ponder
from src.subconscious.configs import BackboneConfig
from src.subconscious.training.routing_training import (
    load_backbone,
    load_retrieval_gate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _have(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


pytestmark = pytest.mark.skipif(
    not (_have(DEFAULT_BACKBONE_PATH) and _have(DEFAULT_GATE_PATH)),
    reason="trained checkpoints (backbone_final.pt / phase2b/best.pt) not local",
)


class _StubModeA:
    """Minimal stub -- the orchestrator stores it but never calls it here."""

    def __init__(self, reply: str = "SYNTH RESPONSE") -> None:
        self.reply = reply

    def _complete(self, messages, tools=None, tool_choice=None):
        return self.reply, None


def test_load_retrieval_gate_loads_phase2b_checkpoint():
    """load_retrieval_gate restores the trained gate on the shared frozen backbone."""
    backbone = load_backbone(DEFAULT_BACKBONE_PATH, BackboneConfig(), device="cpu")
    gate = load_retrieval_gate(DEFAULT_GATE_PATH, backbone, device="cpu")

    # Eval mode (inference).
    assert gate.training is False
    # The shared backbone is REUSED, not reloaded -- the gate's backbone IS the
    # one passed in (stored via object.__setattr__, exposed by .backbone).
    assert gate.backbone is backbone
    # Backbone frozen by load_backbone.
    assert all(not p.requires_grad for p in backbone.parameters())
    # The gate state_dict excludes the backbone (it is not a registered
    # submodule): a routing head is present, the backbone's SSM layers are not.
    sd = gate.state_dict()
    assert any(k.startswith("domain_head") for k in sd)
    assert not any(k.startswith("layers.") for k in sd)


def test_build_ponder_wires_trained_gate_and_frozen_backbone(tmp_path):
    """build_ponder constructs an orchestrator whose gate is the TRAINED gate on
    the TRAINED frozen backbone, shared with Working Memory."""
    orch = build_ponder(
        str(tmp_path / "memory_db"),
        backbone_path=DEFAULT_BACKBONE_PATH,
        gate_path=DEFAULT_GATE_PATH,
        embedder_source="stub",
        device="cpu",
        live_encode=False,
        mode_a=_StubModeA(),
    )
    try:
        gate = orch.retriever.gate
        assert gate is not None
        assert gate.training is False
        # Working Memory shares the SAME trained frozen backbone as the gate
        # (shared weights, separate states -- the JGS design intent).
        assert orch.working_memory.backbone is gate.backbone
        assert all(not p.requires_grad for p in gate.backbone.parameters())
    finally:
        orch.store.close()