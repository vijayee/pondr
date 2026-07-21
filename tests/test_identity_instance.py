"""Phase 1 gate re-run (task #33): the ``identity_instance`` flag.

The from-scratch relevance trainer (``scripts/train_backbone_relevance.py``) drives
the shared SSM directly -- identity ``input_proj`` + zero ``state_lora`` -- and
optimizes ``z_k = mean over d_state of the last layer state``. The formal gate
(``generate_relevance_data.py``) drives the SSM through ``WorkingMemory``, which
by default uses RANDOM instance projections (``input_proj``=random LoRALinear,
``state_lora``~0 but nonzero). Measuring a backbone trained under the direct path
with the random-path gate confounds ``z_i`` with a projection the backbone never
saw. ``identity_instance=True`` makes the WM use identity projections so the gate
measures the SAME path that was trained.

These tests pin that contract on CPU against ``ReferenceSSM`` (no
sentence_transformers, no CUDA, no WaveDB):

1. ``identity_instance`` swaps ``input_proj`` + ``state_lora`` to ``nn.Identity``
   (and leaves ``output_proj``/``gate`` constructed -- they don't affect ``z_i``).
2. Default ``identity_instance=False`` is byte-identical to pre-task-#33 (the
   projections are still LoRA modules).
3. The ``z_i`` the identity-instance WM produces EQUALS the trainer's direct-SSM
   ``z_k`` on the same backbone + inputs (the whole point of the flag).
"""

from __future__ import annotations

import torch

from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.latent_dynamics_head import LatentDynamicsHead
from src.subconscious.lora import LoRALinear, StateLoRA
from src.subconscious.working_memory import WorkingMemory


def _backbone() -> JGSBackbone:
    """A fresh backbone with the trainer's default init (small PyTorch Linear
    init -- NOT randn, which blows up the recurrent state and overflows fp16)."""
    return JGSBackbone(BackboneConfig())


def _inputs(K: int = 6, seed: int = 11) -> torch.Tensor:
    """Deterministic ``[K, 384]`` doc embeddings (the SSM step inputs), roughly
    unit-norm like bge doc vectors (keeps the recurrent state bounded)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(K, 384, dtype=torch.float32, generator=g)
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def _direct_ssm_z(backbone: JGSBackbone, x_seq: torch.Tensor) -> torch.Tensor:
    """The trainer's direct-SSM path (mirrors ``train_backbone_relevance.py``
    ``step_sequence`` + ``z_from_states``): step each SSM layer directly from a
    zeroed state with identity input_proj + no state_lora, then
    ``z_k = last layer state .mean(dim=d_state)``. fp32 throughout."""
    device = x_seq.device
    dtype = x_seq.dtype
    states = [layer.init_state(1, device, dtype) for layer in backbone.layers]
    last_states = []
    for t in range(x_seq.shape[0]):
        h = x_seq[t].unsqueeze(0)            # [1, 384] -- identity input_proj
        new_states = []
        for i, layer in enumerate(backbone.layers):
            h, s = layer.step(h, states[i])  # no state_lora delta
            new_states.append(s)
        states = new_states
        last_states.append(states[-1].squeeze(0))   # [d_state, d_model]
    last_states = torch.stack(last_states)          # [K, d_state, d_model]
    return last_states.float().mean(dim=1)         # [K, d_model] (z_from_states)


# ── 1. identity_instance swaps the projections to Identity ──

def test_identity_instance_uses_identity_projections():
    wm = WorkingMemory(_backbone(), ring_capacity=4, identity_instance=True)
    assert wm.identity_instance is True
    assert isinstance(wm.input_proj, torch.nn.Identity)
    assert isinstance(wm.state_lora, torch.nn.Identity)
    # output_proj stays a real projection (it must still map 384 -> output_dim=256
    # for the ring's y_t; it does NOT affect z_i, which is projected from state).
    assert isinstance(wm.output_proj, LoRALinear)


def test_default_keeps_lora_projections():
    wm = WorkingMemory(_backbone(), ring_capacity=4)  # identity_instance=False
    assert wm.identity_instance is False
    assert isinstance(wm.input_proj, LoRALinear)
    assert isinstance(wm.state_lora, StateLoRA)
    assert isinstance(wm.output_proj, LoRALinear)


# ── 2. the z_i the gate measures under identity == the trainer's z_k ──

def test_identity_instance_z_matches_trainer_direct_ssm():
    """The decisive contract: the gate's slots_z (identity-instance WM path)
    equals the trainer's z_k (direct-SSM path) on the same backbone + inputs.

    The WM stores slot.h as fp16 (bounded memory; ``project`` casts back to
    fp32 losslessly), so the comparison tolerates fp16 rounding -- the two paths
    are otherwise bit-identical (same zeroed init, same per-step SSM, identity
    input_proj, no state_lora delta)."""
    backbone = _backbone()
    x_seq = _inputs(K=6)
    K = x_seq.shape[0]

    # trainer path
    z_k = _direct_ssm_z(backbone, x_seq)                  # [K, 384] fp32

    # gate path (identity-instance WM)
    wm = WorkingMemory(backbone, ring_capacity=K, identity_instance=True)
    wm.reset()
    for t in range(K):
        wm.step(x_seq[t].unsqueeze(0), source_id=str(t), text=str(t))
    ring = wm.ring_buffer()
    assert len(ring) == K
    ld = LatentDynamicsHead()  # untrained; project is parameter-free
    slots_z = torch.stack([
        ld.project(s.h).squeeze(0).float() for s in ring
    ])                                                     # [K, 384] fp32

    assert slots_z.shape == z_k.shape
    torch.testing.assert_close(slots_z, z_k, atol=1e-3, rtol=1e-2,
                               msg="identity-instance WM z_i must equal the "
                                   "trainer's direct-SSM z_k (fp16 storage tol)")


def test_default_instance_z_differs_from_trainer_direct_ssm():
    """Sanity: the DEFAULT (random-projection) WM path does NOT match the
    trainer's direct-SSM z_k -- that random gap is exactly the confound
    ``identity_instance`` exists to remove, so this confirms the test above is
    not trivially passing because both paths collapse to the same vector."""
    backbone = _backbone()
    x_seq = _inputs(K=6)
    K = x_seq.shape[0]
    z_k = _direct_ssm_z(backbone, x_seq)

    wm = WorkingMemory(backbone, ring_capacity=K)  # default random projections
    wm.reset()
    for t in range(K):
        wm.step(x_seq[t].unsqueeze(0), source_id=str(t), text=str(t))
    ld = LatentDynamicsHead()
    slots_z = torch.stack([
        ld.project(s.h).squeeze(0).float() for s in wm.ring_buffer()
    ])
    # they must DIFFER (the random input_proj/state_lora perturb the state).
    assert not torch.allclose(slots_z, z_k, atol=1e-3, rtol=1e-2)