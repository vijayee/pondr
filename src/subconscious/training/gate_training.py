"""Offline gate fine-tuning from the replay buffer (framework, Phase 2b+).

Gradients flow from the outcome back through the gate, but NOT through the SSM
state (avoids BPTT across many steps). Phase 2a ships the function; it is not
called until a gate has replay data (2b+).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .replay_buffer import ReplayBuffer


def train_gate_offline(gate, replay_buffer: ReplayBuffer, optimizer, batch_size: int = 32) -> float:
    """One offline gate training step over a replay sample. Returns the loss."""
    batch = replay_buffer.sample(batch_size)
    if not batch:
        return 0.0

    total = Tensor([0.0])
    for entry in batch:
        if not entry.filled:
            continue
        decision = gate(entry.state, entry.predicted_outcome, entry.context)
        # Targets must match the gate's output device/dtype (the replay entries
        # hold plain python floats); otherwise mse_loss / BCE crash on a
        # device mismatch when the gate runs on GPU.
        dev = decision.value_estimate.device
        dt = decision.value_estimate.dtype
        actual_reward = Tensor([float(entry.outcome["reward"])]).to(dev, dt)
        actual_cost = Tensor([float(entry.outcome["effort"])]).to(dev, dt)
        optimal = float(actual_reward.item() > actual_cost.item())

        value_loss = F.mse_loss(decision.value_estimate, actual_reward)
        cost_loss = F.mse_loss(decision.cost_estimate, actual_cost)
        decision_loss = F.binary_cross_entropy(decision.excite_score.clamp(1e-6, 1 - 1e-6),
                                               torch.tensor([[optimal]], device=dev, dtype=dt))
        total = total + value_loss + cost_loss + decision_loss

    if total.requires_grad:
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    return float(total.item())