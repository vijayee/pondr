"""The shared JGS backbone: stacked SSM layers + a JEPA predictor.

The backbone holds the **shared** weights (SSM transition kernels + JEPA
predictor). It is loaded once and called by every JGS instance; all per-instance
state (recurrent state, gate, projections, LoRA) lives on the instance.

Phase 2a pre-trains this shared backbone on ``follows``-chain state-transition
pairs: given a sequence of episode embeddings, predict the next embedding at
each step (JEPA in embedding space). No instance is involved on the pre-training
path — instances (and their gates/LoRA) are Phase 2b+.

Layers are **stacked by us** (``nn.ModuleList``) rather than passed as an
``n_layers`` kwarg to the SSM, because real ``mamba_ssm.Mamba3`` has no such
kwarg (see §0.1) and stacking is backend-agnostic.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .configs import BackboneConfig
from .ssm import SSMBackend, make_ssm


class JEPAPredictor(nn.Module):
    """Predicts the next state's embedding from the SSM output (per token)."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        d = config.d_model
        layers = [nn.Linear(d, d), nn.GELU()]
        for _ in range(config.pred_layers - 1):
            layers += [nn.Linear(d, d), nn.GELU()]
        layers += [nn.Linear(d, config.pred_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq, d_model] (or [batch, d_model]) -> [..., pred_dim]
        return self.net(x)


class JGSBackbone(nn.Module):
    """Shared SSM + JEPA predictor. Stateless itself — state lives on instances."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([make_ssm(config.ssm_backend, config) for _ in range(config.n_layers)])
        self.predictor = JEPAPredictor(config)

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def d_state(self) -> int:
        return self.config.d_state

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Single-layer state (used by ``forward_seq``'s public signature)."""
        return self.layers[0].init_state(batch, device, dtype)

    def init_states(self, batch: int, device: torch.device, dtype: torch.dtype) -> list[Tensor]:
        """One recurrent state per layer (used by the per-step instance path)."""
        return [layer.init_state(batch, device, dtype) for layer in self.layers]

    def forward_seq(self, x: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """Pre-training path: predict the next embedding at each step.

        Args:
            x: episode embeddings ``[batch, seq, d_model=384]``.
            state: optional recurrent state for the FIRST layer
                ``[batch, d_state, d_model]`` (deeper layers start from zero).
        Returns:
            predictions ``[batch, seq, pred_dim]`` (predicted embedding at each
            step — caller shifts against actual next embeddings for the loss),
            final SSM output ``[batch, seq, d_model]``, last layer's final state.
        """
        h = x
        for i, layer in enumerate(self.layers):
            h, st = layer.forward(h, state if i == 0 else None)
        predictions = self.predictor(h)
        return predictions, h, st

    def step(self, x: Tensor, states: list[Tensor], instance: "object") -> tuple[Tensor, Tensor, list[Tensor]]:
        """Single recurrent step for live instance inference (Phase 2b+).

        Threads through ALL layers (each with its own recurrent state), applies
        the instance's input/output projections and the StateLoRA delta to each
        layer's state. ``states`` is a per-layer list; the instance owns it.
        """
        h = instance.input_proj(x)
        new_states: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            h, s = layer.step(h, states[i])
            s = instance.state_lora(s)
            new_states.append(s)
        predicted = self.predictor(h)
        output = instance.output_proj(h)
        return output, predicted, new_states