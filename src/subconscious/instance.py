"""JGSInstance: one cognitive function, with its own state/gate/projections/LoRA.

The instance owns everything that specializes the shared backbone: a recurrent
state, a decomposed gate, input/output projections, and a StateLoRA delta that
modulates the shared SSM's dynamics. The backbone itself is **shared** — many
instances point at the same backbone object.

Backbone ownership: the backbone is NOT an owned submodule (else every instance
would re-register and re-move the shared weights on ``.to()``). It is stored via
``object.__setattr__`` so PyTorch does not register it, and ``instance.parameters()``
returns only instance-owned params.

Phase 2a builds and unit-tests the instance framework but does NOT train any
instance (gates/projections train in 2b+; see §0.6).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .configs import InstanceConfig
from .gate import DecomposedGate, GateContext, GateDecision
from .lora import LoRALinear, StateLoRA
from .backbone import JGSBackbone


class JGSInstance(nn.Module):
    def __init__(self, backbone: JGSBackbone, config: InstanceConfig):
        super().__init__()
        # Store the shared backbone WITHOUT registering it as a submodule.
        object.__setattr__(self, "_backbone", backbone)
        self.config = config

        # Instance-owned projections (LoRA on a frozen base).
        self.input_proj = LoRALinear(
            nn.Linear(config.input_dim, config.d_model), rank=config.lora_rank
        )
        self.output_proj = LoRALinear(
            nn.Linear(config.d_model, config.output_dim), rank=config.lora_rank
        )
        # Instance-owned modulation of the shared SSM's recurrent state.
        self.state_lora = StateLoRA(config.d_state, config.d_model, rank=config.lora_rank)

        # Instance-owned gate.
        self.gate = DecomposedGate(
            config.gate_config,
            d_model=config.d_model,
            d_state=config.d_state,
            pred_dim=config.d_model,
        )

        # Recurrent state — one tensor per SSM layer. Plain attribute (not a
        # buffer): lazy-initialized at first step so we know batch/device/dtype.
        self.state: Optional[list[Tensor]] = None

    @property
    def backbone(self) -> JGSBackbone:
        return self._backbone  # type: ignore[attr-defined]

    def reset_state(self, batch: int = 1, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        b = self._backbone  # type: ignore[attr-defined]
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        self.state = b.init_states(batch, device, dtype)

    def _ensure_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        b = self._backbone  # type: ignore[attr-defined]
        if self.state is None or self.state[0].shape[0] != batch \
                or self.state[0].device != device:
            self.state = b.init_states(batch, device, dtype)
        if self.state[0].dtype != dtype:
            self.state = [s.to(dtype) for s in self.state]

    def step(self, input_embedding: Tensor,
             context: Optional[GateContext] = None) -> tuple[Tensor, Tensor, GateDecision]:
        """One recurrent step. Returns ``(output, prediction, gate_decision)``.

        ``input_embedding`` is ``[batch, input_dim]``. If ``context`` is None a
        zero context vector is used.

        Device alignment: the shared backbone (``W_A`` and the rest of the SSM)
        is moved to its device at load time (``load_backbone``), but this
        instance's own modules (``input_proj``/``gate``/``state_lora``/
        ``output_proj``) are constructed on CPU, and the embedder feeds a CPU
        tensor. On a CUDA build that mismatches inside ``self._backbone.step``
        (``W_A`` on cuda vs ``x`` on cpu -> addmm RuntimeError). Align the input
        and the instance-owned modules to the backbone's device; the recurrent
        state then initializes on that device too (``_ensure_state`` reads
        ``input_embedding.device``). On CPU every branch is a no-op (devices
        already match), so the CPU path is byte-identical.
        """
        if input_embedding.dim() == 1:
            input_embedding = input_embedding.unsqueeze(0)
        b = self._backbone  # type: ignore[attr-defined]
        target = next(b.parameters()).device
        # Constructors leave the instance-owned modules on CPU; follow the
        # backbone (the shared weights) to its device once. ``.to`` does NOT
        # touch the backbone -- it is held via ``object.__setattr__`` so PyTorch
        # does not register it as a submodule -- and the backbone is already on
        # ``target`` (``load_backbone`` moved it).
        if next(self.parameters()).device != target:
            self.to(target)
        if input_embedding.device != target:
            input_embedding = input_embedding.to(target)
        batch = input_embedding.shape[0]
        device = input_embedding.device
        dtype = input_embedding.dtype
        self._ensure_state(batch, device, dtype)

        output, predicted, new_states = self._backbone.step(input_embedding, self.state, self)  # type: ignore[attr-defined]

        if context is None:
            ncf = self.config.gate_config.num_context_features
            context = GateContext(features=torch.zeros(batch, ncf, device=device, dtype=dtype))
        # Gate reads the current (pre-update) state — use the last layer's.
        decision = self.gate(self.state[-1], predicted, context)

        self.state = [s.detach() for s in new_states]  # no BPTT across steps
        return output, predicted, decision