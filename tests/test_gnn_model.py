"""Tests for the GNN model + 5 heads (``src/gnn/model.py``, ``heads.py``)."""

from __future__ import annotations

import torch

from src.gnn.graph_loader import data_from_subgraph, PREDICATE_VOCAB
from src.gnn.features import FEATURE_DIM, NODE_KIND_INDEX, infer_kind, _hash_embedding, NODE_KINDS
from src.gnn.model import GNNModel, InputProjection, TaxonomyEncoder
from src.gnn.heads import (
    SalienceHead, DiffPoolHead, LinkPredHead, AnomalyHead, OntologyHead, ANOMALY_TYPES,
)


def _stub_feature_for(nid):
    k = NODE_KIND_INDEX[infer_kind(nid)]
    v = _hash_embedding(nid)
    v[k] = 1.0
    v[len(NODE_KINDS)] = -1.0
    return k, v.to(torch.float32)


def _toy_data():
    sub = {
        "center": "ep_000001", "radius": 3,
        "nodes": [
            {"id": "ep_000001", "type": "episode", "depth": 0},
            {"id": "E:Alice", "type": "entity", "depth": 1},
            {"id": "E:Bob", "type": "entity", "depth": 1},
            {"id": "T:db", "type": "topic", "depth": 1},
            {"id": "ep_000002", "type": "episode", "depth": 2},
        ],
        "edges": [
            {"subject": "ep_000001", "predicate": "has_entity", "object": "E:Alice"},
            {"subject": "ep_000001", "predicate": "has_entity", "object": "E:Bob"},
            {"subject": "ep_000001", "predicate": "has_topic", "object": "T:db"},
            {"subject": "E:Alice", "predicate": "in_episode", "object": "ep_000001"},
            {"subject": "ep_000002", "predicate": "follows", "object": "ep_000001"},
        ],
    }
    return data_from_subgraph(sub, _stub_feature_for)


def test_input_projection_uses_per_kind_layer():
    proj = InputProjection(hidden_dim=32)
    x = torch.randn(3, FEATURE_DIM)
    node_kind = torch.tensor([0, 1, 2])  # episode, entity, topic
    out = proj(x, node_kind)
    assert out.shape == (3, 32)
    # Each kind uses a distinct linear layer, so the three rows differ in their
    # projection parameters' effect — verify gradients flow only to the layers
    # whose kind appears.
    loss = out.sum()
    loss.backward()
    grads = [p.grad is not None for p in proj.projections.parameters()]
    # 8 layers × (weight+bias) = 16 param tensors; only kinds 0,1,2 used here.
    used = sum(1 for i, p in enumerate(proj.projections) if p.weight.grad is not None)
    assert used == 3


def test_model_forward_produces_head_shapes():
    model = GNNModel(hidden_dim=64, num_heads=2, num_layers=2,
                     predicate_vocab_size=PREDICATE_VOCAB, num_clusters=4)
    model.eval()
    data = _toy_data()
    out = model(data)
    n, e = data.x.shape[0], data.edge_index.shape[1]
    assert out["salience"].shape == (n,)
    assert out["diffpool"].shape == (n, 4)
    assert out["linkpred"].shape == (e,)
    assert out["anomaly"].shape == (n, len(ANOMALY_TYPES))
    # Ontology is a two-encoder head (entity emb from this backbone + class emb
    # from the taxonomy encoder over the class DAG); a single-graph forward has
    # no class DAG, so it cannot score ontology pairs -> None (not faked).
    assert out["ontology"] is None
    assert out["node_emb"].shape == (n, 64)


def test_each_head_loss_is_differentiable():
    model = GNNModel(hidden_dim=32, num_heads=2, num_layers=2,
                     predicate_vocab_size=PREDICATE_VOCAB, num_clusters=4)
    data = _toy_data()
    out = model(data)
    n, e = data.x.shape[0], data.edge_index.shape[1]
    sal = model.salience.loss(out["salience"], torch.rand(n))
    assign = model.diffpool(out["node_emb"], data.edge_index)
    dp = model.diffpool.loss(assign, data.edge_index)
    lp = model.linkpred.loss(out["linkpred"], torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0][:e]))
    an = model.anomaly.loss(out["anomaly"], torch.zeros(n, len(ANOMALY_TYPES)))
    # Ontology: two-encoder pair classifier. Build a small class DAG Data and
    # score entity rows against class rows through the taxonomy encoder.
    tax = _toy_taxonomy_data(num_classes=3, hidden_dim=32)
    class_emb = model.encode_taxonomy(tax)
    pair_index = torch.tensor([[0, 1, 2], [0, 1, 2]])  # entity row -> class row
    ont_scores = model.ontology(out["node_emb"][:3], class_emb, pair_index)
    ont = model.ontology.loss(ont_scores, torch.tensor([1.0, 0.0, 1.0]))
    total = sal + dp + lp + an + ont
    total.backward()
    # The backbone, the heads, AND the taxonomy encoder all received gradients.
    assert sum(1 for p in model.parameters() if p.grad is not None) > 0
    assert any(p.grad is not None for p in model.taxonomy.parameters())


def test_salience_and_anomaly_heads_metric_ranges():
    sal = SalienceHead(32)
    an = AnomalyHead(32)
    node_emb = torch.randn(5, 32)
    sal_logits = sal(node_emb)
    assert 0.0 <= SalienceHead.metric(sal_logits, torch.rand(5)) <= 2.0  # L1 in [0, ~1]
    an_logits = an(node_emb)
    # macro F1 is in [0, 1]
    assert 0.0 <= AnomalyHead.metric(an_logits, torch.zeros(5, len(ANOMALY_TYPES))) <= 1.0


def test_linkpred_metric_single_class_falls_back_to_accuracy():
    scores = torch.tensor([0.6, 0.4, 0.9])
    labels = torch.tensor([1, 1, 1])  # single class
    # Should not raise and should return a float in [0, 1].
    m = LinkPredHead.metric(scores, labels)
    assert 0.0 <= m <= 1.0


def test_diffpool_assignment_is_row_softmax():
    dp = DiffPoolHead(16, num_clusters=3)
    node_emb = torch.randn(4, 16)
    assign = dp(node_emb, torch.tensor([[0, 1], [2, 3]]))
    assert assign.shape == (4, 3)
    # Each row sums to 1 (softmax).
    assert torch.allclose(assign.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_ontology_head_scores_pairs():
    """Two-encoder pair classifier: entity emb (episode rows) x class emb
    (taxonomy rows) -> per-pair typing score."""
    ont = OntologyHead(16)
    entity_emb = torch.randn(4, 16)   # episode subgraph rows
    class_emb = torch.randn(3, 16)    # taxonomy DAG rows
    # pair_index[0] -> entity row, pair_index[1] -> class row
    pairs = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
    scores = ont(entity_emb, class_emb, pairs)
    assert scores.shape == (4,)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_taxonomy_encoder_produces_class_embeddings():
    """The taxonomy encoder GATs over the class DAG -> [C, hidden] embeddings
    (open-vocabulary: a new class node gets an embedding via message passing
    from its parent edge, no fixed class table)."""
    enc = TaxonomyEncoder(hidden_dim=16, num_heads=2, num_layers=2)
    tax = _toy_taxonomy_data(num_classes=5, hidden_dim=16)
    emb = enc.encode(tax.x, tax.edge_index, tax.node_kind)
    assert emb.shape == (5, 16)
    # Message passing ran (not a no-op): two classes with different structure
    # should not produce identical embeddings.
    assert not torch.allclose(emb[0], emb[1])


def _toy_taxonomy_data(num_classes: int, hidden_dim: int):
    """A tiny class DAG: ``num_classes`` bare-name class nodes linked by a
    ``subClassOf`` chain (c0 <- c1 <- c2 ...), bidirectional, in the
    ``data_from_subgraph`` shape the taxonomy builder emits."""
    nodes = [{"id": f"Class{i}", "type": "unknown", "depth": i} for i in range(num_classes)]
    edges = []
    for i in range(num_classes - 1):
        # child Class(i+1) subClassOf parent Classi; both orientations.
        edges.append({"subject": f"Class{i+1}", "predicate": "subClassOf", "object": f"Class{i}"})
        edges.append({"subject": f"Class{i}", "predicate": "subClassOf", "object": f"Class{i+1}"})
    sub = {"center": "Class0", "radius": 0, "nodes": nodes, "edges": edges}
    return data_from_subgraph(sub, _stub_feature_for)


# ── DiffPool collapse-fix + Anomaly imbalance-fix (Phase 3a head rework) ──

def test_diffpool_metric_counts_clusters_used_and_flags_collapse():
    K = 4
    # collapsed: every row -> cluster 0
    collapsed = torch.zeros(8, K); collapsed[:, 0] = 1.0
    # spread: rows evenly across all K clusters (one-hot, confident)
    spread = torch.zeros(8, K)
    for i in range(8):
        spread[i, i % K] = 1.0
    assert DiffPoolHead.metric(collapsed) == 1.0   # 1 cluster = collapsed
    assert DiffPoolHead.metric(spread) == float(K)  # all K used = healthy


def test_diffpool_balance_term_penalizes_collapse():
    """On a 2-community graph, a community-aware assignment must beat the
    all-in-one-cluster collapse — i.e. the balance term breaks the old global-min
    collapse (which gave loss 0 regardless of input)."""
    K = 4
    # 4 nodes, two disjoint edges = two communities (0-1, 2-3).
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    # collapsed: all -> cluster 0 (one-hot)
    collapsed = torch.zeros(4, K); collapsed[:, 0] = 1.0
    # community-aware: {0,1}->0, {2,3}->1 (one-hot; connected nodes share a cluster)
    community = torch.zeros(4, K)
    community[0, 0] = community[1, 0] = 1.0
    community[2, 1] = community[3, 1] = 1.0
    dp = DiffPoolHead(16, num_clusters=K)
    loss_collapse = dp.loss(collapsed, edge_index)
    loss_community = dp.loss(community, edge_index)
    # The community assignment uses 2 balanced clusters with link=0 -> lower loss
    # than the collapse (link=0 but balance=0). Collapse is no longer the minimum.
    assert loss_community.item() < loss_collapse.item()
    # And the balance term alone is worse for collapse (0) than community (<0).
    eps = 1e-12
    bal_col = (collapsed.mean(0) * (collapsed.mean(0) + eps).log()).sum().item()
    bal_com = (community.mean(0) * (community.mean(0) + eps).log()).sum().item()
    assert bal_com < bal_col  # community marginal is more uniform -> lower (more negative)


def test_anomaly_loss_upweights_rare_positives():
    """A model that gets the rare positive wrong should pay MORE under pos-weighted
    BCE than plain BCE — that's the whole point of rebalancing."""
    T = len(ANOMALY_TYPES)
    # 1 positive, N-1 negatives for every type (extreme imbalance).
    target = torch.zeros(20, T); target[0, :] = 1.0
    # Model predicts ~0.01 everywhere (confident "no anomaly" — the collapse mode).
    pred = torch.full((20, T), 0.01, requires_grad=True)
    weighted = AnomalyHead.loss(pred, target)
    # plain BCE for comparison
    import torch.nn.functional as F
    plain = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), target)
    # The single missed positive is upweighted -> weighted loss exceeds plain.
    assert weighted.item() > plain.item()
    # And it's finite + differentiable.
    assert torch.isfinite(weighted)
    weighted.backward()  # no error


def test_anomaly_metric_excludes_zero_positive_types():
    T = len(ANOMALY_TYPES)
    # Type 0 has positives (perfectly predicted -> F1=1); type 1 has none.
    logits = torch.zeros(10, T)
    target = torch.zeros(10, T)
    target[:3, 0] = 1.0            # 3 positives on type 0
    logits[:3, 0] = 0.9            # predicted positive (correct)
    logits[3:, 0] = 0.1            # predicted negative (correct, true negatives)
    # type 1..T-1 have zero positives anywhere -> excluded from the macro.
    m = AnomalyHead.metric(logits, target)
    assert m == 1.0  # only type 0 is measurable, and it's perfectly predicted


def test_anomaly_metric_zero_when_no_type_has_positives():
    T = len(ANOMALY_TYPES)
    logits = torch.full((10, T), 0.2)
    target = torch.zeros(10, T)  # no positives at all -> nothing measurable
    assert AnomalyHead.metric(logits, target) == 0.0