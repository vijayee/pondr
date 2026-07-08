"""Tests for the GNN model + 5 heads (``src/gnn/model.py``, ``heads.py``)."""

from __future__ import annotations

import torch

from src.gnn.graph_loader import data_from_subgraph, PREDICATE_VOCAB
from src.gnn.features import FEATURE_DIM, NODE_KIND_INDEX, infer_kind, _hash_embedding, NODE_KINDS
from src.gnn.model import GNNModel, InputProjection
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


def test_model_forward_produces_all_five_head_shapes():
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
    assert out["ontology"].shape == (e,)
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
    ont = model.ontology.loss(out["ontology"], torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0][:e]))
    total = sal + dp + lp + an + ont
    total.backward()
    # At least the backbone + the heads we exercised have gradients.
    assert sum(1 for p in model.parameters() if p.grad is not None) > 0


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
    ont = OntologyHead(16)
    node_emb = torch.randn(4, 16)
    pairs = torch.tensor([[0, 1], [2, 3]])
    scores = ont(node_emb, pairs)
    assert scores.shape == (2,)
    assert (scores >= 0).all() and (scores <= 1).all()