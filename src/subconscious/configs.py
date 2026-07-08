"""Configuration dataclasses for the JGS backbone, instances, and training.

Right-sized for the actual Phase 2a training data: a few thousand
``follows``-chain state-transition pairs (see ``docs/Phase 2a.md`` §0.2/§0.3),
NOT the doc's original 10M-example / 480M-param / 100k-step plan. The modest
config here trains in minutes-to-an-hour on a single 24GB Ampere GPU and also
runs on CPU for the dev loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackboneConfig:
    """Shared SSM + JEPA predictor architecture.

    ``d_model`` is 384 — the bge-small embedder output dim — so the SSM
    operates directly in embedding space and the JEPA predictor predicts the
    next 384-dim episode embedding literally (no projection layer on the
    pre-training path). ``n_layers`` is 4 (not 24) to fit the few-thousand-pair
    dataset without overfitting. (Couples the backbone to the embedder dim for
    2a; acceptable — bge-small is the fixed embedder.)
    """

    d_model: int = 384
    n_layers: int = 4
    d_state: int = 16

    # JEPA predictor
    pred_layers: int = 3
    pred_dim: int = 384  # == d_model == bge-small dim: predict next embedding

    # SSM backend selection: "reference" (CPU dev), "mamba3-pytorch" (faithful
    # CPU reference), "mamba3-cuda" (official mamba_ssm.Mamba3 on the pod).
    ssm_backend: str = "reference"


@dataclass
class GateConfig:
    """Decomposed-gate shape.

    ``num_context_features`` is the size of the per-instance context vector
    (e.g. entity_recency, topic_recency, query_complexity for the retrieval
    gate). ``context_dim`` is the encoded context vector fed to the cost and
    decision heads.
    """

    num_context_features: int = 3
    context_dim: int = 128


@dataclass
class InstanceConfig:
    """Per-instance configuration (one cognitive function)."""

    name: str
    input_dim: int = 384   # episode summary embedding dim (bge-small)
    output_dim: int = 256
    d_model: int = 384
    d_state: int = 16
    lora_rank: int = 4
    gate_config: GateConfig = field(default_factory=GateConfig)


# The 8 cognitive functions that will be instantiated in Phases 2b-7b. Defined
# here so Phase 2a can build + test the instance framework against all of them.
# Phase 2c declares a 9th, ``presentation_gate`` — heuristic-only in 2c (the
# learned JGS gate is deferred until outcome signals are wired live; see
# docs/Phase 2c.md §5.1). Declared now so the instance registry stays consistent
# and a future learned gate has a home; the heuristic planner does NOT use it.
INSTANCE_CONFIGS: dict[str, InstanceConfig] = {
    "retrieval_gate": InstanceConfig(
        name="retrieval_gate",
        lora_rank=4,  # fast routing
        gate_config=GateConfig(num_context_features=3),  # entity_recency, topic_recency, query_complexity
    ),
    "working_memory": InstanceConfig(
        name="working_memory",
        input_dim=384, output_dim=256,
        lora_rank=8,  # rich state
        gate_config=GateConfig(num_context_features=2),  # input_novelty, state_saturation
    ),
    "presentation_gate": InstanceConfig(
        name="presentation_gate",
        lora_rank=4,
        gate_config=GateConfig(num_context_features=2),  # placeholder for the deferred learned gate
    ),
    "uncertainty_detector": InstanceConfig(
        name="uncertainty_detector",
        lora_rank=4,
        gate_config=GateConfig(num_context_features=3),  # error_magnitude, noise_level, novelty
    ),
    "aspirational_model": InstanceConfig(
        name="aspirational_model",
        lora_rank=6,
        gate_config=GateConfig(num_context_features=3),  # goal_alignment, expected_value, urgency
    ),
    "self_model": InstanceConfig(
        name="self_model",
        lora_rank=4,
        gate_config=GateConfig(num_context_features=3),  # domain_density, fact_specificity, retrieval_confidence
    ),
    "common_sense_resolver": InstanceConfig(
        name="common_sense_resolver",
        lora_rank=6,  # flexible dynamics
        gate_config=GateConfig(num_context_features=3),  # ambiguity_magnitude, context_coherence, historical_frequency
    ),
    "disturbance_detector": InstanceConfig(
        name="disturbance_detector",
        lora_rank=4,  # fast, bursty
        gate_config=GateConfig(num_context_features=3),  # error_magnitude, noise_level, novelty
    ),
    "intuition_module": InstanceConfig(
        name="intuition_module",
        lora_rank=8,  # slow, accumulative
        gate_config=GateConfig(num_context_features=4),  # sunk_cost, novelty, recent_reward_rate, pattern_familiarity
    ),
}


@dataclass
class BackboneTrainingConfig:
    """Backbone pre-training hyperparameters.

    Right-sized for ~4,000-8,000 transition pairs: a few thousand steps, small
    batch (CPU-friendly), modest LR. The doc's 100k-step / batch-64 plan was
    for 10M examples that don't exist.
    """

    # Model
    backbone: BackboneConfig = field(default_factory=BackboneConfig)

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 3_000
    gradient_accumulation: int = 2

    # JEPA
    temperature: float = 0.1
    num_negative_samples: int = 16
    target_ema_decay: float = 0.996  # EMA decay for the target encoder

    # Optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # Hardware / dtype
    dtype: str = "bfloat16"  # resolved to torch.dtype in the training loop
    device: str = "auto"     # "auto" | "cpu" | "cuda"
    compile: bool = False    # torch.compile; off by default (reference SSM)

    # Data
    pairs_path: str = "data/training/backbone/sequences.jsonl"
    val_fraction: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "checkpoints/backbone"
    checkpoint_every: int = 500
    seed: int = 0