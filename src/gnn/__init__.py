"""Phase 3a — GNN Consolidator.

Static, stateless GNN (per the spec's temporal-continuity note: temporal
SSM-augmented instances come only after failure modes are observed) with 5
heads, trained on Oracle-labeled memory graphs and run in a nightly dream-state
consolidation loop. See ``docs/Phase 3a.md``.

This package does NOT depend on the SSM/mamba_ssm stack — it is torch_geometric.
"""

from .features import NodeFeatureBuilder, NODE_KINDS, NODE_KIND_INDEX, FEATURE_DIM
from .graph_loader import WaveDBGraphLoader, PREDICATE_VOCAB, KNOWN_PREDICATES
from .heads import (
    SalienceHead, DiffPoolHead, LinkPredHead, AnomalyHead, OntologyHead, ANOMALY_TYPES,
)
from .model import GNNModel, InputProjection
from .semantic_memory import SemanticMemoryWriter
from .consolidate import Consolidator

__all__ = [
    "NodeFeatureBuilder",
    "NODE_KINDS",
    "NODE_KIND_INDEX",
    "FEATURE_DIM",
    "WaveDBGraphLoader",
    "PREDICATE_VOCAB",
    "KNOWN_PREDICATES",
    "SalienceHead", "DiffPoolHead", "LinkPredHead", "AnomalyHead", "OntologyHead",
    "ANOMALY_TYPES",
    "GNNModel", "InputProjection",
    "SemanticMemoryWriter",
    "Consolidator",
]