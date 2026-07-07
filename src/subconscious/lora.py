"""LoRA adapters: instance-specific low-rank modulation.

Two forms:

- ``LoRALinear`` wraps a frozen base ``nn.Linear`` and adds a trainable
  low-rank delta ``B @ A`` (initialized so the delta is zero at start, per LoRA
  convention). This is used for the instance input/output projections — base
  weights stay shared/frozen-ish, the instance learns its adapter.

- ``StateLoRA`` produces a low-rank residual added to the SSM recurrent state,
  which is how an instance modulates the shared SSM's dynamics without touching
  the shared weights. This is the hand-rolled replacement for the doc's
  fictional ``ssm.with_lora(A, B)`` (see §0.1).

Base weights are frozen via ``freeze_base`` so the optimizer only updates the
adapter parameters — the shared backbone stays shared.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def freeze_base(module: nn.Module) -> nn.Module:
    """Set ``requires_grad=False`` on all of ``module``'s parameters."""
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class LoRALinear(nn.Module):
    """``y = base(x) + scale * (x @ A) @ B`` with base frozen.

    ``A`` is down-projected with small Gaussian init, ``B`` is zero-init so the
    adapter starts as the identity (no change to base behavior at step 0).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 8.0):
        super().__init__()
        freeze_base(base)
        self.base = base
        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * (x @ self.A @ self.B)


class StateLoRA(nn.Module):
    """Low-rank residual on the recurrent state: ``new_state += scale * B(A state)``.

    Operates on a state tensor ``[batch, d_state, d_model]`` by flattening to
    ``[batch, d_state*d_model]``, applying a down->up low-rank delta, and adding
    it back. Zero-init ``B`` keeps the modulation off at start. This is the
    instance-specific modulation of shared SSM dynamics.
    """

    def __init__(self, d_state: int, d_model: int, rank: int):
        super().__init__()
        dim = d_state * d_model
        self.d_state = d_state
        self.d_model = d_model
        self.rank = rank
        self.scale = 8.0 / max(rank, 1)
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, state: Tensor) -> Tensor:
        batch = state.shape[0]
        flat = state.reshape(batch, -1)
        delta = (flat @ self.A @ self.B) * self.scale
        return state + delta.reshape(batch, self.d_state, self.d_model)