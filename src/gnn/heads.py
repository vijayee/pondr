"""The 5 GNN task heads (Phase 3a Task 2).

Each head consumes the GAT backbone's node embeddings (``[N, hidden_dim]``) plus
the graph structure it needs. Heads are independent ``nn.Module``s so per-head
training (Task 4) can optimize one head's loss at a time, and the consolidation
loop (Task 6) can call whichever heads it needs.

Per-head loss + metric, and what's honestly trainable now vs deferred, is noted
on each head. The whole GNN is **stateless** (per the spec's §378 temporal-
continuity note): no recurrent state, no per-instance memory — temporal
SSM-augmented instances come only after failure modes are observed.

Loss shapes (match the Oracle label schemas in ``src/training/prompts.py``):
- salience: MSE on per-node ``salience`` in ``[0, 1]``.
- diffpool: simplified DiffPool — cluster-assignment entropy reg + a cluster-
  level link-preservation loss. (Full DiffPool's dense pooling is heavy; this is
  the cold-start version, documented as such.)
- linkpred: BCE on positive + negative edges (GAE dot-product; SEAL subgraph
  features are a later lever, not the cold start).
- anomaly: multi-label BCE over 6 anomaly types.
- ontology: BCE on proposed ``subClassOf`` edges (pair classifier).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Salience head (GAT regression) ──
class SalienceHead(nn.Module):
    """Per-node salience regressor. Supersedes the Phase-1c heuristic
    mention-count prior (``graph_traversal.py:389-430``). Target: the Oracle
    ``salience`` label in ``[0, 1]`` (``prompts.gnn_salience_prompt``)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """Return per-node salience logits ``[N]`` (apply sigmoid for probs)."""
        return self.net(node_emb).squeeze(-1)

    @staticmethod
    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(logits, target.float())

    @staticmethod
    def metric(logits: torch.Tensor, target: torch.Tensor) -> float:
        """Mean absolute error (lower is better)."""
        with torch.no_grad():
            return F.l1_loss(logits, target.float()).item()


# ── 2. Subgraph-summarization head (simplified DiffPool) ──
class DiffPoolHead(nn.Module):
    """Cluster-assignment head for semantic-memory abstraction.

    Cold-start DiffPool: a soft assignment ``[N, num_clusters]`` from node
    embeddings, with two losses — (a) entropy regularization that encourages
    confident assignments and (b) a cluster-link loss that preserves the
    original edge structure at the cluster level (``A_pool = Sᵀ A S`` should
    match the induced cluster adjacency). Full DiffPool links successive pooled
    graphs across layers; that is a later lever, not the cold start.
    """

    def __init__(self, hidden_dim: int, num_clusters: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.assign = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clusters),
        )

    def forward(
        self, node_emb: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return soft cluster assignment ``[N, num_clusters]`` (row-softmax)."""
        return F.softmax(self.assign(node_emb), dim=-1)

    def loss(
        self,
        assign: torch.Tensor,
        edge_index: torch.Tensor,
        entropy_weight: float = 0.1,
    ) -> torch.Tensor:
        """DiffPool auxiliary loss: assignment entropy + cluster-link preservation."""
        # (a) Entropy regularization — push rows toward confident (low-entropy)
        # assignments. ``-(p log p)`` averaged over nodes; we MINIMIZE entropy so
        # negate the standard entropy expression.
        eps = 1e-12
        ent = -(assign * (assign + eps).log()).sum(dim=-1).mean()
        # (b) Cluster-link preservation: A_pool = Sᵀ A S should be dense where
        # the original graph had edges. Use the edge endpoints' assignment rows:
        # for each edge, the cluster-pair mass is assign[s] · assign[o]ᵀ; we want
        # it high. Cross-entropy of the cluster-pair distribution vs uniform is a
        # weak proxy; instead use the simpler ``-log(assign[s] · assign[o])``
        # averaged over edges (pulls connected nodes into shared clusters).
        if edge_index.shape[1] > 0:
            s, o = edge_index[0], edge_index[1]
            pair = (assign[s] * assign[o]).sum(dim=-1)  # [E]
            link = -(pair + eps).log().mean()
        else:
            link = torch.tensor(0.0, device=assign.device)
        return link + entropy_weight * ent

    @staticmethod
    def metric(assign: torch.Tensor) -> float:
        """Mean assignment entropy (lower = more confident clusters)."""
        with torch.no_grad():
            eps = 1e-12
            return -(assign * (assign + eps).log()).sum(dim=-1).mean().item()


# ── 3. Link-prediction head (GAE) ──
class LinkPredHead(nn.Module):
    """Edge-existence scorer via GAE dot-product.

    Score(u, v) = sigmoid(node_emb[u] · node_emb[v]). Trained on positive
    (observed) + negative (sampled non-) edges (``prompts.gnn_link_prediction_prompt``
    emits ``predicted_edges`` + ``negative_edges`` after Task 3). SEAL subgraph
    features are a later quality lever, not the cold start.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # A small projection so the dot-product isn't raw backbone output.
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score the edges in ``edge_index`` → ``[E]`` in ``[0, 1]``."""
        z = self.proj(node_emb)
        if edge_index.shape[1] == 0:
            return torch.zeros(0, device=z.device)
        s, o = edge_index[0], edge_index[1]
        return torch.sigmoid((z[s] * z[o]).sum(dim=-1))

    @staticmethod
    def loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(scores, labels.float())

    @staticmethod
    def metric(scores: torch.Tensor, labels: torch.Tensor) -> float:
        """AUC-ROC (0.5 = chance). Falls back to accuracy if a single class."""
        with torch.no_grad():
            y = labels.cpu().numpy()
            p = scores.cpu().numpy()
            if len(set(y.tolist())) < 2:
                return float((p.round() == y).mean())
            try:
                from sklearn.metrics import roc_auc_score
                return float(roc_auc_score(y, p))
            except Exception:
                return float((p.round() == y).mean())


# ── 4. Anomaly-detection head (9-type multi-label) ──
# The 9 anomaly types are owned by ``anomaly_rules.py`` (the rule detectors +
# injector that produce the head's training labels — spec §2 of the sharded-
# labeling design). Importing the canonical tuple here keeps the head's output
# slots and the training labels aligned by construction. Index order is
# load-bearing: the head's ``len(ANOMALY_TYPES)`` output slots correspond to
# these in order. (The 6-type Oracle-prompt schema was superseded in Task 3 by
# the injection-based 9-type taxonomy — see ``anomaly_rules.ANOMALY_TYPES``.)
from .anomaly_rules import ANOMALY_TYPES


class AnomalyHead(nn.Module):
    """Multi-label anomaly classifier over ``ANOMALY_TYPES``.

    Target: the per-node anomaly flag vector from the injection-based labels
    (``anomaly_rules.node_label_vectors`` — spec §2; the Oracle
    ``gnn_anomaly_prompt`` is no longer the head's label source). Output
    ``[N, len(ANOMALY_TYPES)]`` sigmoid probabilities.
    """

    def __init__(self, hidden_dim: int, num_types: int = len(ANOMALY_TYPES), dropout: float = 0.1) -> None:
        super().__init__()
        self.num_types = num_types
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(node_emb))

    @staticmethod
    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(logits, target.float())

    @staticmethod
    def metric(logits: torch.Tensor, target: torch.Tensor) -> float:
        """Per-type macro F1 (0 = no positives for that type)."""
        with torch.no_grad():
            pred = (logits.cpu().numpy() > 0.5).astype(int)
            y = target.cpu().numpy().astype(int)
            f1s = []
            for t in range(y.shape[1]):
                tp = int(((pred[:, t] == 1) & (y[:, t] == 1)).sum())
                fp = int(((pred[:, t] == 1) & (y[:, t] == 0)).sum())
                fn = int(((pred[:, t] == 0) & (y[:, t] == 1)).sum())
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                f1s.append(f1)
            return float(sum(f1s) / max(1, len(f1s)))


# ── 5. Ontology-typing head (two-encoder pair classifier) ──
class OntologyHead(nn.Module):
    """Entity->class typing as a pair classifier over TWO encoders.

    ``score(entity, class) = sigmoid(MLP(concat(emb_entity, emb_class)))`` where
    ``emb_entity`` comes from the episode GAT backbone (the entity's row in the
    episode subgraph) and ``emb_class`` comes from the taxonomy encoder over the
    live class DAG (``TaxonomyEncoder`` in ``model.py``). The two endpoints live
    in DIFFERENT graphs -- an episode entity and a taxonomy class share no
    subgraph (entities have ``in_episode`` edges; classes have only class-to-
    class ``subClassOf`` edges; the entity->class membership edge is what this
    head predicts, so it cannot also be the path that feeds it). A single
    per-subgraph ``node_emb`` cannot hold both, hence the two-encoder design.

    Target: the Oracle ``ontology_labels`` -- ``suggested_edges`` (child=entity,
    parent=class) AND ``misclassified`` (entity, suggested_class) are BOTH
    entity->class typing labels, unified by ``label_tensors.ontology_target``.

    Open-vocabulary: the class side is the live class DAG, not a fixed table. A
    new class discovered at runtime is added to the DAG with a parent
    ``subClassOf`` edge; the taxonomy encoder produces an embedding for it next
    pass via message passing from its parent -- no retraining of this head's
    ``net`` is needed to score a new class (vision sec 5.3).

    Honest scope (deferred, NOT faked): (a) class->class ``subClassOf`` refinement
    -- no class->class labels exist, so this head does not propose taxonomy edges;
    (b) new-class creation is a Bonsai-gated consolidation ACTION, not a head
    output; (c) Bonsai gating on the ontology step is not wired (only link-pred
    calls the verifier today) -- consolidation records proposals but writes none.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        entity_emb: torch.Tensor,
        class_emb: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score the entity/class pairs in ``pair_index`` ``[2, P]`` -> ``[P]``.

        ``pair_index[0]`` indexes ``entity_emb`` (episode subgraph rows);
        ``pair_index[1]`` indexes ``class_emb`` (taxonomy DAG rows).
        """
        if pair_index.shape[1] == 0:
            return torch.zeros(0, device=entity_emb.device)
        e = entity_emb[pair_index[0]]
        c = class_emb[pair_index[1]]
        return torch.sigmoid(self.net(torch.cat([e, c], dim=-1)).squeeze(-1))

    @staticmethod
    def loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(scores, labels.float())

    @staticmethod
    def metric(scores: torch.Tensor, labels: torch.Tensor) -> float:
        with torch.no_grad():
            return float((scores.round() == labels.float()).float().mean().item())


__all__ = [
    "SalienceHead", "DiffPoolHead", "LinkPredHead", "AnomalyHead", "OntologyHead",
    "ANOMALY_TYPES",
]