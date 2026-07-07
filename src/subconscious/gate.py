"""The decomposed gate: value / cost / decision heads.

Per ``docs/Phase 2a.md`` §2.4, the gate decides pursue-vs-inhibit from a value
estimate ("how good is the predicted future?"), a cost estimate ("how hard?"),
and a context vector. The gate is **instance-specific** (its own weights), in
contrast to the shared SSM+JEPA backbone.

De-wonk corrections vs the doc's §2.4 code (see §0.5):

- Param math is right-sized. The doc flattened the whole state
  (``d_model*d_state*2 = 16384``) into the first Linear → ~8.4M params, not the
  claimed ~400K, and the gate would have been ~20x over its "~1.5M" budget. We
  pool the state over its ``d_state`` channels first (``[batch, d_model]``),
  so the heads are small (~100-300K params).
- ``_threshold_modifier`` no longer creates a fresh ``nn.Linear`` inside
  ``forward`` (that one was unregistered and never trained). It is a
  registered ``threshold_head`` module.
- ``DecomposedGate`` subclasses ``nn.Module``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .configs import GateConfig


@dataclass
class GateContext:
    """Per-step context features.

    ``features`` is ``[batch, num_context_features]``. The doc's named fields
    (entity_recency, noise_level, ...) are the caller's responsibility to
    assemble into this tensor; the gate only consumes the vector.
    """

    features: Tensor

    def to_vector(self) -> Tensor:
        return self.features


@dataclass
class GateDecision:
    """Output of a gate forward pass.

    Tensors retain grad for the later gate-fine-tuning phase (2b+). ``pursue``
    is a detached python bool (a discrete decision, not differentiated).
    """

    value_estimate: Tensor
    cost_estimate: Tensor
    ratio: Tensor
    inhibit_score: Tensor
    excite_score: Tensor
    threshold: Tensor
    pursue: bool
    confidence: float


class DecomposedGate(nn.Module):
    """Value / cost / decision heads + a context encoder.

    Inputs are the recurrent state ``[batch, d_state, d_model]`` and the JEPA
    predicted-future ``[batch, pred_dim]``. The state is pooled over ``d_state``
    before being fed to the heads so the gate stays small.
    """

    def __init__(self, config: GateConfig, d_model: int = 256, d_state: int = 16, pred_dim: int = 384):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.pred_dim = pred_dim

        # Context encoder: num_context_features -> context_dim.
        self.context_encoder = nn.Sequential(
            nn.Linear(config.num_context_features, 128),
            nn.GELU(),
            nn.Linear(128, config.context_dim),
        )

        # Value head: how good is the predicted future? (no context — intrinsic)
        value_in = d_model + pred_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_in, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

        # Cost head: how hard to reach? (context-aware)
        cost_in = d_model + pred_dim + config.context_dim
        self.cost_head = nn.Sequential(
            nn.Linear(cost_in, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

        # Decision head: combine ratio, cost, context.
        self.decision_head = nn.Sequential(
            nn.Linear(2 + config.context_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 2),  # [inhibit_logit, excite_logit]
        )

        # Registered threshold module + learnable base threshold.
        self.threshold_head = nn.Linear(config.context_dim, 1)
        self.base_threshold = nn.Parameter(torch.tensor(0.5))

    def _pool(self, state: Tensor) -> Tensor:
        # [batch, d_state, d_model] -> [batch, d_model] (mean over state channels)
        return state.mean(dim=1)

    def forward(self, state: Tensor, predicted_future: Tensor, context: GateContext) -> GateDecision:
        ctx = self.context_encoder(context.to_vector())
        pooled = self._pool(state)

        v_in = torch.cat([pooled, predicted_future], dim=-1)
        value = self.value_head(v_in)                       # [batch, 1]

        c_in = torch.cat([pooled, predicted_future, ctx], dim=-1)
        cost = self.cost_head(c_in)                          # [batch, 1]

        ratio = value / (cost.abs() + 1e-8)                 # [batch, 1]

        d_in = torch.cat([ratio, cost, ctx], dim=-1)
        logits = self.decision_head(d_in)                    # [batch, 2]
        inhibit_score = torch.sigmoid(logits[:, 0:1])
        excite_score = torch.sigmoid(logits[:, 1:2])

        # Learned threshold modulation in [0.3, 1.7] * base (registered module).
        modifier = torch.sigmoid(self.threshold_head(ctx))  # [batch, 1]
        threshold = self.base_threshold * (0.3 + 1.4 * modifier)

        # The live gate contract is one decision per step (batch=1 — each
        # instance gates its own next action). For a batched call we require
        # every element to agree before pursuing, which keeps batch>1 well-
        # defined (and is a no-op at batch=1).
        pursue = bool((excite_score > inhibit_score).all().item()
                      and (excite_score.mean() > threshold.mean()).item())
        conf = float((excite_score - inhibit_score).abs().mean().item())

        return GateDecision(
            value_estimate=value,
            cost_estimate=cost,
            ratio=ratio,
            inhibit_score=inhibit_score,
            excite_score=excite_score,
            threshold=threshold,
            pursue=pursue,
            confidence=conf,
        )