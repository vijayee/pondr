"""GAT backbone + 5-head GNN model (Phase 3a Task 2).

Architecture:
- ``InputProjection`` — per-kind Linear mapping the loader's raw 384-dim feature
  (kind-onehot + episode embedding / entity salience) into ``hidden_dim``,
  selected by the ``node_kind`` index tensor. This is the "per-kind projection
  MLP" from the §1.3 node-feature decision, kept in the model so the loader stays
  parameter-free.
- GAT backbone — ``num_layers`` stacked ``GATConv`` layers (the first consumes
  ``edge_attr`` via ``edge_dim`` so predicate identity feeds attention). Residual
  + dropout. Output: ``[N, hidden_dim]`` node embeddings.
- 5 heads (``heads.py``) — applied on top of the node embeddings.

The model is **stateless**: no recurrent state, no per-instance memory. float32
on CPU for the dev slice; the pod training run (Task 4) keeps float32 (the 2a
bf16/autocast dtype-mix bug is still open, but the GNN is independent of the SSM
bf16 path). OGB pretraining (``GNNConfig.ogb_pretrain``) is a pod-only, lazy
import path — not exercised on the dev machine.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from .heads import (
    AnomalyHead, DiffPoolHead, LinkPredHead, OntologyHead, SalienceHead,
    ANOMALY_TYPES,
)
from .features import FEATURE_DIM, NODE_KINDS


class InputProjection(nn.Module):
    """Per-kind linear projection of raw node features into ``hidden_dim``.

    Holds one ``Linear(FEATURE_DIM, hidden_dim)`` per node kind and selects the
    row's projection by its ``node_kind`` index — so an episode's embedding and
    an entity's onehot+salience are each mapped by a kind-appropriate layer.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(FEATURE_DIM, hidden_dim) for _ in NODE_KINDS]
        )

    def forward(self, x: torch.Tensor, node_kind: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], self.projections[0].out_features, device=x.device, dtype=x.dtype)
        for k in range(len(self.projections)):
            mask = node_kind == k
            if mask.any():
                out[mask] = self.projections[k](x[mask])
        return out


class TaxonomyEncoder(nn.Module):
    """GAT over the live class DAG -> class embeddings ``[C, hidden_dim]``.

    The ontology head's class side. Class nodes are bare names (``infer_kind``
    -> ``"unknown"``); their raw feature is the type-onehot (no persisted vector),
    so the per-kind ``InputProjection`` routes them through the ``unknown`` slot.
    The GAT then propagates ``subClassOf`` structure so a class's embedding
    reflects its taxonomy position (root vs leaf, sibling of X, parent of Y).

    Open-vocabulary mechanism: a new class discovered at runtime is added to the
    DAG with a parent ``subClassOf`` edge; the taxonomy encoder produces an
    embedding for it next pass via message passing from its parent -- no fixed
    class table, no retrain of the ontology head's MLP to score a new class
    (vision sec 5.3: the ontology is a seed that grows).

    The DAG has a single edge type (``subClassOf``), so predicate identity is
    uniform across the taxonomy and carries no information: the GAT layers use
    ``edge_dim=None`` (the loader still emits ``edge_attr`` but this encoder
    ignores it -- the ``hash()``-randomized ``subClassOf`` onehot slot would
    otherwise be unstable across runs for no signal gain). The taxonomy graph is
    built bidirectional (child->parent + parent->child, mirroring the episode
    loader) so message passing reaches both ancestors and descendants.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = InputProjection(hidden_dim)
        self.layers = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim,
                heads=num_heads,
                concat=False,   # average heads -> keep hidden_dim
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_kind: torch.Tensor
    ) -> torch.Tensor:
        """Run input projection + GAT over the class DAG -> ``[C, hidden_dim]``."""
        h = self.input_proj(x, node_kind)
        for conv in self.layers:
            res = h
            h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
            if h.shape == res.shape:
                h = h + res  # residual
        return h


class GNNModel(nn.Module):
    """GAT backbone + the 5 task heads + the taxonomy encoder."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        predicate_vocab_size: int = 32,
        num_clusters: int = 16,
        num_anomaly_types: int = len(ANOMALY_TYPES),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = InputProjection(hidden_dim)

        # GAT backbone. First layer consumes edge_attr (predicate onehot).
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            self.layers.append(
                GATConv(
                    in_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,   # average heads → keep hidden_dim
                    dropout=dropout,
                    edge_dim=(predicate_vocab_size if i == 0 else None),
                    add_self_loops=True,
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        # Taxonomy encoder for the ontology head's class side (open-vocabulary:
        # operates over the live class DAG, not a fixed table).
        self.taxonomy = TaxonomyEncoder(
            hidden_dim, num_heads=num_heads, num_layers=2, dropout=dropout,
        )

        # Heads.
        self.salience = SalienceHead(hidden_dim, dropout=dropout)
        self.diffpool = DiffPoolHead(hidden_dim, num_clusters=num_clusters, dropout=dropout)
        self.linkpred = LinkPredHead(hidden_dim)
        self.anomaly = AnomalyHead(hidden_dim, num_types=num_anomaly_types, dropout=dropout)
        self.ontology = OntologyHead(hidden_dim, dropout=dropout)

    # ── encode ──
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor,
               edge_attr: torch.Tensor, node_kind: torch.Tensor) -> torch.Tensor:
        """Run input projection + GAT backbone → ``[N, hidden_dim]`` node embeddings."""
        h = self.input_proj(x, node_kind)
        for i, conv in enumerate(self.layers):
            res = h
            kwargs = {"edge_attr": edge_attr} if i == 0 and edge_attr is not None else {}
            h = conv(h, edge_index, **kwargs)
            h = self.act(h)
            h = self.dropout(h)
            if h.shape == res.shape:
                h = h + res  # residual
        return h

    def encode_taxonomy(self, tax_data) -> torch.Tensor:
        """Run the taxonomy encoder over the class DAG -> ``[C, hidden_dim]``.

        ``tax_data`` is the ``Data`` from ``taxonomy_graph.build_taxonomy_data``
        (class nodes + bidirectional ``subClassOf`` edges). Called explicitly
        by the trainer / consolidator (NOT by ``forward``) because the ontology
        head needs BOTH the episode ``node_emb`` and these class embeddings --
        a single-graph forward cannot produce both. Recomputed each optimizer
        step (with grad) so the taxonomy encoder trains alongside the head.
        """
        return self.taxonomy.encode(
            tax_data.x, tax_data.edge_index, tax_data.node_kind
        )

    def forward(self, data) -> dict[str, torch.Tensor]:
        """Run the backbone + the 4 single-encoder heads. Returns head outputs.

        ``data`` is a PyG ``Data`` (or anything with the attributes the loader
        sets: ``x``, ``edge_index``, ``edge_attr``, ``node_kind``). Heads that
        need extra per-task inputs (link-prediction's labeled edges) are called
        separately in training; here we score the graph's own edges for link-pred
        so a single forward pass yields 4 head outputs for a shape smoke-test.

        ``ontology`` is deliberately ``None`` here: the ontology head is a
        TWO-encoder pair classifier (entity embeddings from this backbone +
        class embeddings from the taxonomy encoder over the class DAG). A
        single-graph forward has no class DAG, so it cannot score ontology
        pairs. The trainer / consolidator call ``encode_taxonomy`` +
        ``model.ontology(entity_emb, class_emb, pair_index)`` explicitly.
        """
        node_emb = self.encode(
            data.x, data.edge_index, getattr(data, "edge_attr", None), data.node_kind
        )
        salience_logits = self.salience(node_emb)
        assign = self.diffpool(node_emb, data.edge_index)
        link_scores = self.linkpred(node_emb, data.edge_index)
        anomaly_logits = self.anomaly(node_emb)
        return {
            "salience": salience_logits,
            "diffpool": assign,
            "linkpred": link_scores,
            "anomaly": anomaly_logits,
            "ontology": None,
            "node_emb": node_emb,
        }

    @torch.no_grad()
    def node_embeddings(self, data) -> torch.Tensor:
        """Convenience for the consolidation loop (Task 6)."""
        return self.encode(
            data.x, data.edge_index, getattr(data, "edge_attr", None), data.node_kind
        )


__all__ = ["GNNModel", "InputProjection", "TaxonomyEncoder"]