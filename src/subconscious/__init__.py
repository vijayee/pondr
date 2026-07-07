"""Phase 2a: JEPA-Gated SSM backbone and instance framework.

This package implements the shared JGS backbone (SSM + JEPA predictor), the
instance base class, the decomposed gate, and the LoRA adapter framework
described in ``docs/Phase 2a.md``. See that doc's ``§0 Alignment Notes`` for
the corrections made against the original draft (real Mamba3 API, data
prerequisite, right-sized backbone, modest-GPU training).

The SSM is behind a pluggable interface (``ssm.SSMBackend``) so the same
backbone/instance/gate code runs:

- **locally on CPU** with the zero-dependency ``ReferenceSSM`` (real, trainable,
  but NOT Mamba3-faithful — a minimal selective SSM used for dev + unit tests),
- and on the training pod with the official ``mamba_ssm.Mamba3`` via
  ``Mamba3CUDABackend`` (lazy-imported, CUDA/Hopper kernels).

The per-step recurrent ``step()`` inference path is Phase 2b+; Phase 2a only
bulk-pre-trains the shared weights (see ``training.pretrain``).
"""

from .configs import (
    BackboneConfig,
    BackboneTrainingConfig,
    GateConfig,
    InstanceConfig,
    INSTANCE_CONFIGS,
)
from .gate import DecomposedGate, GateContext, GateDecision
from .instance import JGSInstance
from .backbone import JGSBackbone, JEPAPredictor
from .lora import LoRALinear, StateLoRA, freeze_base
from .ssm import SSMBackend, ReferenceSSM, make_ssm

__all__ = [
    "BackboneConfig",
    "BackboneTrainingConfig",
    "GateConfig",
    "InstanceConfig",
    "INSTANCE_CONFIGS",
    "DecomposedGate",
    "GateContext",
    "GateDecision",
    "JGSInstance",
    "JGSBackbone",
    "JEPAPredictor",
    "LoRALinear",
    "StateLoRA",
    "freeze_base",
    "SSMBackend",
    "ReferenceSSM",
    "make_ssm",
]