"""``build_ponder(no_gate=True)`` -- bypass the stateful RetrievalGate.

The trained Phase 2b RetrievalGate is a STATEFUL SSM: on a fresh recurrent
state it routes ~every query to ``ssm_direct`` (unsupported -> no response),
which blocks all fresh-conversation serve evals ([[pondr-gate-stateful-
ssm-direct-fresh]]). ``no_gate=True`` sets ``gate = None`` so the
orchestrator's no-gate branch runs (plain ``retrieve`` + ``synthesize`` every
turn). The backbone STILL loads (used by WorkingMemory + the SSMChunker), so
this test requires the backbone checkpoint but NOT the gate checkpoint.

Offline: stub embedder + stub mode_a, real on-disk trained backbone. No Bonsai,
no GLiNER. Skipped when the pod-trained backbone checkpoint isn't local.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.runtime import DEFAULT_BACKBONE_PATH, DEFAULT_GATE_PATH, build_ponder
from src.subconscious.configs import BackboneConfig
from src.subconscious.training.routing_training import load_backbone

REPO_ROOT = Path(__file__).resolve().parent.parent


def _have(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


# no_gate still loads the backbone (WorkingMemory + SSMChunker use it); the
# gate checkpoint is NOT needed (gate=None skips the load).
pytestmark = pytest.mark.skipif(
    not _have(DEFAULT_BACKBONE_PATH),
    reason="trained backbone checkpoint (backbone_final.pt) not local",
)


class _StubModeA:
    """Minimal stub -- the orchestrator stores it but never calls it here."""

    def __init__(self, reply: str = "SYNTH RESPONSE") -> None:
        self.reply = reply

    def _complete(self, messages, tools=None, tool_choice=None):
        return self.reply, None


def test_no_gate_yields_retriever_with_none_gate(tmp_path):
    """``no_gate=True`` -> ``orch.retriever.gate is None`` (the orchestrator's
    no-gate branch will fire). The backbone still loads and is shared with
    WorkingMemory as today."""
    orch = build_ponder(
        str(tmp_path / "memory_db"),
        backbone_path=DEFAULT_BACKBONE_PATH,
        gate_path=DEFAULT_GATE_PATH,   # ignored under no_gate (gate=None)
        no_gate=True,
        embedder_source="stub",
        device="cpu",
        live_encode=False,
        mode_a=_StubModeA(),
    )
    try:
        assert orch.retriever.gate is None
        # the backbone still loaded (WorkingMemory holds it; it is frozen).
        assert orch.working_memory.backbone is not None
        assert all(not p.requires_grad
                   for p in orch.working_memory.backbone.parameters())
    finally:
        orch.store.close()


def test_no_gate_does_not_require_gate_checkpoint(tmp_path):
    """``no_gate=True`` does not even READ the gate checkpoint -- a non-existent
    gate_path is accepted (the load is skipped). This is the property that lets
    ``--no-gate`` run on a tree where only the backbone is present."""
    orch = build_ponder(
        str(tmp_path / "memory_db"),
        backbone_path=DEFAULT_BACKBONE_PATH,
        gate_path=str(tmp_path / "does_not_exist.pt"),   # never opened
        no_gate=True,
        embedder_source="stub",
        device="cpu",
        live_encode=False,
        mode_a=_StubModeA(),
    )
    try:
        assert orch.retriever.gate is None
    finally:
        orch.store.close()


def test_gate_off_by_default_loads_trained_gate(tmp_path):
    """Default (``no_gate`` unset / False) loads the trained gate as today. Pins
    the byte-identical-when-off contract. Skipped if the gate ckpt is absent
    (the default path needs it)."""
    if not _have(DEFAULT_GATE_PATH):
        pytest.skip("gate checkpoint not local")
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
        assert orch.retriever.gate is not None
        # the gate shares the frozen backbone (the existing contract).
        assert orch.working_memory.backbone is orch.retriever.gate.backbone
    finally:
        orch.store.close()