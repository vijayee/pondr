"""Unit + integration tests for the Phase 2a JGS backbone/instance/gate/LoRA.

All CPU-runnable against the ``ReferenceSSM`` backend (no CUDA, no mamba_ssm).
The real ``mamba_ssm.Mamba3`` path is exercised on the pod, not here.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from src.subconscious import (
    BackboneConfig,
    BackboneTrainingConfig,
    DecomposedGate,
    GateConfig,
    GateContext,
    INSTANCE_CONFIGS,
    JGSBackbone,
    JGSInstance,
    LoRALinear,
    StateLoRA,
    freeze_base,
    make_ssm,
)
from src.subconscious.training.pretrain import pretrain_backbone, _group_chains


def test_backbone_forward_shapes():
    cfg = BackboneConfig()
    bb = JGSBackbone(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    pred, h, state = bb.forward_seq(x)
    assert pred.shape == (2, 5, cfg.pred_dim)
    assert h.shape == (2, 5, cfg.d_model)
    assert state.shape == (2, cfg.d_state, cfg.d_model)


def test_backbone_stateful_evolution():
    cfg = BackboneConfig()
    bb = JGSBackbone(cfg)
    x = torch.randn(1, 6, cfg.d_model)
    pred1, _, state1 = bb.forward_seq(x)
    pred2, _, state2 = bb.forward_seq(x)
    # Same input -> same outputs (deterministic).
    assert torch.allclose(pred1, pred2)
    # Different input -> different state.
    y = torch.randn(1, 6, cfg.d_model)
    _, _, state3 = bb.forward_seq(y)
    assert not torch.allclose(state1, state3)


def _states_differ(s1, s2) -> bool:
    return any(not torch.allclose(a, b) for a, b in zip(s1, s2))


def test_instance_independent_states():
    cfg = BackboneConfig()
    bb = JGSBackbone(cfg)
    a = JGSInstance(bb, INSTANCE_CONFIGS["retrieval_gate"])
    b = JGSInstance(bb, INSTANCE_CONFIGS["uncertainty_detector"])
    a.reset_state(1); b.reset_state(1)
    a.step(torch.randn(1, cfg.d_model))
    b.step(torch.randn(1, cfg.d_model))
    assert _states_differ(a.state, b.state)


def test_instance_param_count_modest():
    cfg = BackboneConfig()
    bb = JGSBackbone(cfg)
    inst = JGSInstance(bb, INSTANCE_CONFIGS["retrieval_gate"])
    n = sum(p.numel() for p in inst.parameters())
    # Right-sized: instance adds well under the doc's old 2.5M target.
    assert 100_000 < n < 2_000_000, n


def test_instance_backbone_not_duplicated():
    cfg = BackboneConfig()
    bb = JGSBackbone(cfg)
    a = JGSInstance(bb, INSTANCE_CONFIGS["retrieval_gate"])
    b = JGSInstance(bb, INSTANCE_CONFIGS["self_model"])
    # The shared backbone is not an owned submodule of either instance.
    assert "_backbone" not in dict(a.named_modules())
    # Instance.parameters() excludes the shared backbone.
    inst_params = sum(p.numel() for p in a.parameters())
    bb_params = sum(p.numel() for p in bb.parameters())
    # No overlap: instances don't pull in backbone params.
    assert inst_params < bb_params


def test_gate_decision_valid():
    gate = DecomposedGate(GateConfig(num_context_features=3), d_model=384, d_state=16, pred_dim=384)
    state = torch.randn(2, 16, 384)
    pred = torch.randn(2, 384)
    ctx = GateContext(features=torch.randn(2, 3))
    d = gate(state, pred, ctx)
    assert d.value_estimate.shape == (2, 1)
    assert d.cost_estimate.shape == (2, 1)
    assert 0.0 <= d.confidence <= 1.0
    assert isinstance(d.pursue, bool)


def test_gate_threshold_is_bounded():
    gate = DecomposedGate(GateConfig(num_context_features=3), d_model=384, d_state=16, pred_dim=384)
    state = torch.randn(1, 16, 384)
    pred = torch.randn(1, 384)
    low_noise = GateContext(features=torch.tensor([[0.3, 0.1, 0.1]]))
    high_noise = GateContext(features=torch.tensor([[0.3, 0.9, 0.1]]))
    th_low = gate(state, pred, low_noise).threshold.mean().item()
    th_high = gate(state, pred, high_noise).threshold.mean().item()
    # The threshold head is randomly initialized (the gate isn't trained until
    # 2b+), so we can't assert a directional noise->threshold relationship yet.
    # We only assert the threshold is bounded in [0.3*0.5, 1.7*0.5] = [0.15, 0.85]
    # for any context — the modifier sigmoid * the 0.3..1.7 span guarantees this.
    assert 0.15 <= th_low <= 0.85
    assert 0.15 <= th_high <= 0.85


def test_lora_parameter_efficiency():
    base = nn.Linear(384, 384)
    n_base = sum(p.numel() for p in base.parameters())
    lora = LoRALinear(base, rank=4)
    n_adapter = lora.A.numel() + lora.B.numel()
    assert n_adapter / n_base < 0.05  # LoRA adapter is a small fraction of base (rank-4 on 384^2 ≈ 2%)
    assert all(not p.requires_grad for p in base.parameters())  # base frozen
    assert lora.A.requires_grad and lora.B.requires_grad


def test_lora_starts_as_identity():
    base = nn.Linear(4, 4)
    base.weight.data.copy_(torch.eye(4)); base.bias.data.zero_()
    lora = LoRALinear(base, rank=2)
    x = torch.randn(3, 4)
    assert torch.allclose(lora(x), base(x), atol=1e-6)  # B is zero-init -> no delta


def test_state_lora_starts_as_identity():
    sl = StateLoRA(d_state=16, d_model=384, rank=4)
    state = torch.randn(2, 16, 384)
    out = sl(state)
    assert torch.allclose(out, state, atol=1e-6)  # B zero-init -> no delta


def test_instance_step_sequence():
    cfg = BackboneConfig()
    bb = JGSBackbone(cfg)
    inst = JGSInstance(bb, INSTANCE_CONFIGS["working_memory"])
    inst.reset_state(1)
    outs = []
    for _ in range(5):
        out, pred, dec = inst.step(torch.randn(1, cfg.d_model))
        outs.append(out)
    assert not torch.allclose(outs[0], outs[-1])
    assert all(isinstance(d.pursue, bool) for d in [dec])


def test_multiple_instances_shared_backbone():
    cfg = BackboneConfig()
    bb = JGSBackbone(cfg)
    instances = [JGSInstance(bb, INSTANCE_CONFIGS[k]) for k in
                 ("retrieval_gate", "uncertainty_detector", "self_model")]
    for inst in instances:
        inst.reset_state(1)
        inst.step(torch.randn(1, cfg.d_model))
    s = [i.state for i in instances]
    assert _states_differ(s[0], s[1])
    assert _states_differ(s[1], s[2])
    assert _states_differ(s[0], s[2])


def test_all_instance_configs_constructible():
    cfg = BackboneConfig()
    bb = JGSBackbone(cfg)
    for name, iconf in INSTANCE_CONFIGS.items():
        inst = JGSInstance(bb, iconf)
        inst.reset_state(2)
        out, pred, dec = inst.step(torch.randn(2, iconf.input_dim))
        assert out.shape == (2, iconf.output_dim), name


def test_group_chains_reconstructs_sequences():
    # 2 chains: c0 = [e0,e1,e2], c1 = [e0,e1]
    pairs = [
        {"type": "forward", "chain_id": "c0", "position": 0, "state_t": [0.0]*4, "state_t_plus_1": [1.0]*4},
        {"type": "forward", "chain_id": "c0", "position": 1, "state_t": [1.0]*4, "state_t_plus_1": [2.0]*4},
        {"type": "reverse", "chain_id": "c0", "position": 1, "state_t": [2.0]*4, "state_t_plus_1": [1.0]*4},
        {"type": "forward", "chain_id": "c1", "position": 0, "state_t": [3.0]*4, "state_t_plus_1": [4.0]*4},
    ]
    chains = _group_chains(pairs)
    assert len(chains) == 2
    c0 = [c for c in chains if c == [[0.0]*4, [1.0]*4, [2.0]*4]]
    assert len(c0) == 1


def test_pretrain_runs_and_checkpoints(tmp_path):
    # Synthetic chains.
    pairs = []
    for cid in range(6):
        embs = [torch.randn(384).tolist() for _ in range(4)]
        for i in range(3):
            pairs.append({"type": "forward", "chain_id": f"c{cid}", "position": i,
                          "state_t": embs[i], "state_t_plus_1": embs[i + 1]})
    p = tmp_path / "sequences.jsonl"
    with open(p, "w") as f:
        for r in pairs:
            f.write(json.dumps(r) + "\n")
    ckpt = tmp_path / "ckpts"
    cfg = BackboneTrainingConfig(total_steps=4, batch_size=3, warmup_steps=1,
                                 checkpoint_every=0, checkpoint_dir=str(ckpt),
                                 device="cpu", dtype="float32")
    bb = pretrain_backbone(cfg, str(p))
    assert os.path.exists(ckpt / "backbone_final.pt")