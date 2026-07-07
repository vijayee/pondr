"""Phase 2a training: JEPA loss, backbone pre-training loop, replay buffer."""

from .jepa_loss import jepa_contrastive_loss
from .pretrain import pretrain_backbone, BackboneDataset, load_pairs
from .replay_buffer import ReplayBuffer, ReplayEntry
from .gate_training import train_gate_offline

__all__ = [
    "jepa_contrastive_loss",
    "pretrain_backbone",
    "BackboneDataset",
    "load_pairs",
    "ReplayBuffer",
    "ReplayEntry",
    "train_gate_offline",
]