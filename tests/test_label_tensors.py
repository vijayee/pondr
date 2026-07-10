"""Pure unit tests for ``src/gnn/label_tensors.py`` (Phase 3a Task 4a).

No store, no model -- the builders are pure ``(Data, labels) -> tensor``. The
``Data`` objects are built via ``data_from_subgraph`` with the same stub feature
function used in ``test_gnn_model.py``, plus hand-built label dicts in the
authoritative schema the regenerated label files carry. Each test asserts one
builder behavior in isolation: salience mask + values, link-pred pos/neg +
bare-name endpoint resolution + skip-and-count, ontology entity->class
unification (suggested_edges + misclassified) + seeded negatives +
class-not-in-DAG skip + class_vocab scan, anomaly scatter (and the clean-
subgraph all-zero true-negative case), and seeded train/val split.
"""

from __future__ import annotations

import random

import torch

from src.gnn.anomaly_rules import ANOMALY_TYPE_INDEX, ANOMALY_TYPES
from src.gnn.features import NODE_KIND_INDEX, NODE_KINDS, _hash_embedding, infer_kind
from src.gnn.graph_loader import data_from_subgraph
from src.gnn.label_tensors import (
    anomaly_target, class_vocab, linkpred_pairs, ontology_target, salience_target,
    split_centers,
)


def _stub_feature_for(nid):
    k = NODE_KIND_INDEX[infer_kind(nid)]
    v = _hash_embedding(nid)
    v[k] = 1.0
    v[len(NODE_KINDS)] = -1.0
    return k, v.to(torch.float32)


def _toy_data():
    """7 nodes: 2 episodes, 4 entities, 1 topic; has_entity/has_topic/in_episode/
    follows edges so endpoint resolution + same-kind negative sampling have real
    structure to work against."""
    sub = {
        "center": "ep_000001", "radius": 3,
        "nodes": [
            {"id": "ep_000001", "type": "episode", "depth": 0},
            {"id": "ep_000002", "type": "episode", "depth": 1},
            {"id": "E:Alice", "type": "entity", "depth": 1},
            {"id": "E:Bob", "type": "entity", "depth": 1},
            {"id": "E:Carol", "type": "entity", "depth": 2},
            {"id": "E:Dave", "type": "entity", "depth": 2},
            {"id": "T:db", "type": "topic", "depth": 1},
        ],
        "edges": [
            {"subject": "ep_000001", "predicate": "has_entity", "object": "E:Alice"},
            {"subject": "ep_000001", "predicate": "has_entity", "object": "E:Bob"},
            {"subject": "ep_000001", "predicate": "has_topic", "object": "T:db"},
            {"subject": "E:Alice", "predicate": "in_episode", "object": "ep_000001"},
            {"subject": "ep_000002", "predicate": "has_entity", "object": "E:Carol"},
            {"subject": "ep_000002", "predicate": "has_entity", "object": "E:Dave"},
            {"subject": "ep_000002", "predicate": "follows", "object": "ep_000001"},
        ],
    }
    return data_from_subgraph(sub, _stub_feature_for)


def _row(data, node_id) -> int:
    return data.node_id.index(node_id)


# ── salience ──

def test_salience_target_mask_and_values():
    data = _toy_data()
    labels = {"node_scores": {"ep_000001": {"salience": 0.9}, "E:Alice": 0.3}}
    target, mask = salience_target(data, labels)
    assert mask.dtype == torch.bool and target.dtype == torch.float32
    # Only the two scored nodes are labeled.
    assert mask.sum().item() == 2
    assert mask[_row(data, "ep_000001")].item() is True
    assert mask[_row(data, "E:Alice")].item() is True
    assert mask[_row(data, "E:Bob")].item() is False
    assert torch.isclose(target[_row(data, "ep_000001")], torch.tensor(0.9))
    assert torch.isclose(target[_row(data, "E:Alice")], torch.tensor(0.3))
    # Unscored nodes stay 0 in the target (but the mask excludes them from loss).
    assert target[_row(data, "E:Bob")].item() == 0.0


def test_salience_target_empty_is_all_false():
    data = _toy_data()
    target, mask = salience_target(data, {"node_scores": {}})
    assert mask.any().item() is False  # no labels -> trainer skips this head
    assert target.shape == (7,)


# ── link prediction ──

def test_linkpred_pairs_pos_and_neg_full_ids():
    data = _toy_data()
    labels = {
        "predicted_edges": [{"subject": "ep_000002", "object": "ep_000001"}],
        "negative_edges": [{"subject": "E:Alice", "object": "E:Bob"}],
    }
    pt = linkpred_pairs(data, labels, seed=0)
    assert pt.edge_index is not None and pt.skipped == 0
    assert pt.labels.tolist() == [1.0, 0.0]
    s, o = pt.edge_index[0].tolist(), pt.edge_index[1].tolist()
    assert (s[0], o[0]) == (_row(data, "ep_000002"), _row(data, "ep_000001"))
    assert (s[1], o[1]) == (_row(data, "E:Alice"), _row(data, "E:Bob"))


def test_linkpred_pairs_bare_name_resolution():
    """Endpoints may be bare names (radius-1 one-call / Oracle paraphrase)."""
    data = _toy_data()
    labels = {
        "predicted_edges": [{"subject": "Alice", "object": "Bob"}],
        "negative_edges": [{"subject": "Carol", "object": "Dave"}],
    }
    pt = linkpred_pairs(data, labels, seed=0)
    assert pt.edge_index is not None and pt.skipped == 0
    assert pt.labels.tolist() == [1.0, 0.0]
    s, o = pt.edge_index[0].tolist(), pt.edge_index[1].tolist()
    assert (s[0], o[0]) == (_row(data, "E:Alice"), _row(data, "E:Bob"))
    assert (s[1], o[1]) == (_row(data, "E:Carol"), _row(data, "E:Dave"))


def test_linkpred_pairs_skips_unresolved():
    data = _toy_data()
    # The only positive resolves to nothing (E:Ghost not in subgraph); no
    # negatives -> no fallback sampling (src is empty) -> None.
    labels = {"predicted_edges": [{"subject": "E:Ghost", "object": "E:Alice"}]}
    pt = linkpred_pairs(data, labels, seed=0)
    assert pt.edge_index is None and pt.skipped == 1


def test_linkpred_pairs_samples_negatives_when_absent():
    """Old positive-only PoC data has no negative_edges -> in-code sampling so
    BCE can't collapse to 'predict 1'."""
    data = _toy_data()
    labels = {"predicted_edges": [{"subject": "E:Alice", "object": "E:Bob"}]}
    pt = linkpred_pairs(data, labels, seed=7)
    assert pt.edge_index is not None
    lbls = pt.labels.tolist()
    assert 1.0 in lbls and 0.0 in lbls   # the positive + at least one sampled negative
    assert pt.skipped == 0


def test_linkpred_pairs_falls_back_when_all_provided_negatives_unresolvable():
    """Oracle PROVIDED negatives but every one failed endpoint resolution ->
    still sample in-code so BCE isn't left positives-only (collapse to 'predict
    1'). The fallback keys on zero USABLE negatives, not on neg_items absence."""
    data = _toy_data()
    labels = {
        "predicted_edges": [{"subject": "E:Alice", "object": "E:Bob"}],
        "negative_edges": [{"subject": "E:Ghost", "object": "E:Phantom"}],  # both absent
    }
    pt = linkpred_pairs(data, labels, seed=7)
    assert pt.edge_index is not None
    lbls = pt.labels.tolist()
    assert 1.0 in lbls and 0.0 in lbls  # positive + a sampled fallback negative
    assert pt.skipped == 1              # the one provided negative item (both endpoints absent)


# ── ontology ──

# A toy class_index: class NAME -> taxonomy-encoder row (built by the trainer
# from the live class DAG; here it's a pure lookup so ontology_target is unit-
# testable with no store).
_Toy_CLASS_INDEX = {"Person": 0, "Topic": 1, "Concept": 2}


def test_ontology_target_unifies_suggested_and_misclassified():
    """Both label kinds are entity->class typing: unify into (entity_row,
    class_row) positives + seeded (entity, other_class) negatives."""
    data = _toy_data()
    labels = {
        "suggested_edges": [{"child": "E:Alice", "parent": "Person"}],
        "misclassified": [{"entity": "E:Bob", "suggested_class": "Topic"}],
    }
    pt = ontology_target(data, labels, _Toy_CLASS_INDEX, seed=3)
    assert pt.edge_index is not None
    lbls = pt.labels.tolist()
    assert 1.0 in lbls and 0.0 in lbls  # two positives + sampled negatives
    # Row 0 indexes entity rows in data.node_id; row 1 indexes class rows.
    ent_rows = pt.edge_index[0].tolist()
    cls_rows = pt.edge_index[1].tolist()
    # First two pairs are the positives (suggested then misclassified).
    assert (ent_rows[0], cls_rows[0]) == (_row(data, "E:Alice"), 0)  # Person
    assert (ent_rows[1], cls_rows[1]) == (_row(data, "E:Bob"), 1)     # Topic
    assert lbls[0] == 1.0 and lbls[1] == 1.0
    # Negatives reference real entity rows + real class rows only.
    for er, cr in zip(ent_rows[2:], cls_rows[2:]):
        assert er in range(len(data.node_id))
        assert cr in _Toy_CLASS_INDEX.values()
    assert pt.skipped == 0


def test_ontology_target_bare_name_entity_resolution():
    """The entity side may be a bare name (Oracle paraphrase); ``_resolve``
    strips the prefix and matches against data.node_id."""
    data = _toy_data()
    labels = {"suggested_edges": [{"child": "Alice", "parent": "Person"}],
              "misclassified": []}
    pt = ontology_target(data, labels, _Toy_CLASS_INDEX, seed=0)
    assert pt.edge_index is not None
    assert pt.edge_index[0].tolist()[0] == _row(data, "E:Alice")
    assert pt.edge_index[1].tolist()[0] == 0  # Person


def test_ontology_target_skips_class_not_in_dag():
    """A class name absent from ``class_index`` (an Oracle-suggested class not
    yet Bonsai-promoted to a seed-anchored subClassOf edge) can't form a
    scoreable pair -- skipped + counted, not fabricated."""
    data = _toy_data()
    labels = {"suggested_edges": [{"child": "E:Alice", "parent": "Nonexistent"}],
              "misclassified": []}
    pt = ontology_target(data, labels, _Toy_CLASS_INDEX, seed=0)
    # No resolvable positives -> no negatives sampled (n=0) -> no pairs.
    assert pt.edge_index is None
    assert pt.skipped == 1


def test_ontology_target_empty_returns_none():
    data = _toy_data()
    pt = ontology_target(data, {"suggested_edges": [], "misclassified": []},
                         _Toy_CLASS_INDEX, seed=0)
    assert pt.edge_index is None
    assert pt.skipped == 0


def test_ontology_target_dedupes_same_entity_class_pair():
    """The same entity->class pair appearing in both kinds is ONE positive."""
    data = _toy_data()
    labels = {
        "suggested_edges": [{"child": "E:Alice", "parent": "Person"}],
        "misclassified": [{"entity": "E:Alice", "suggested_class": "Person"}],
    }
    pt = ontology_target(data, labels, _Toy_CLASS_INDEX, seed=0)
    # Exactly one positive (deduped), plus sampled negatives.
    pos = pt.labels.tolist().count(1.0)
    assert pos == 1


def test_class_vocab_scans_labels(tmp_path):
    """``class_vocab`` reads the distinct class names from both label kinds."""
    import json
    p = tmp_path / "ontology_labels.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"subgraph_id": "ep_1", "labels": {
            "suggested_edges": [{"child": "E:A", "parent": "Person"}],
            "misclassified": [{"entity": "E:B", "suggested_class": "Topic"}],
        }}) + "\n")
        f.write(json.dumps({"subgraph_id": "ep_2", "labels": {
            "suggested_edges": [{"child": "E:C", "parent": "Concept"}],
            "misclassified": [],
        }}) + "\n")
    names = class_vocab(tmp_path)
    assert names == ["Concept", "Person", "Topic"]  # sorted distinct


def test_class_vocab_missing_file_is_empty(tmp_path):
    """A missing ontology_labels.jsonl -> empty list (head never trainable)."""
    assert class_vocab(tmp_path) == []


# ── anomaly ──

def _corrupted_data():
    """3 nodes incl. an injected ``ep_000001_dup`` clone (the structural
    duplication signal the anomaly head must learn)."""
    sub = {
        "center": "ep_000001", "radius": 1,
        "nodes": [
            {"id": "ep_000001", "type": "episode", "depth": 0},
            {"id": "ep_000001_dup", "type": "episode", "depth": 0},
            {"id": "E:Alice", "type": "entity", "depth": 1},
        ],
        "edges": [
            {"subject": "ep_000001", "predicate": "has_entity", "object": "E:Alice"},
            {"subject": "ep_000001_dup", "predicate": "has_entity", "object": "E:Alice"},
        ],
    }
    return data_from_subgraph(sub, _stub_feature_for)


def test_anomaly_target_scatters():
    data = _corrupted_data()
    dup_idx = ANOMALY_TYPE_INDEX["duplicate_episode"]
    labels = {"node_labels": {"ep_000001": [dup_idx], "ep_000001_dup": [dup_idx]}}
    target = anomaly_target(data, labels)
    assert target.shape == (3, len(ANOMALY_TYPES))
    assert target.dtype == torch.float32
    # Both duplicate episodes flagged at the duplicate_episode column.
    assert target[0, dup_idx].item() == 1.0      # ep_000001
    assert target[1, dup_idx].item() == 1.0      # ep_000001_dup
    # E:Alice is a clean node -> all-zero (true negative, NOT masked).
    assert target[2, :].sum().item() == 0.0


def test_anomaly_target_clean_subgraph_all_zero():
    """A clean subgraph (no injection) is an all-zero true-negative example --
    KEPT (the head must learn to predict 0 on clean nodes), not skipped."""
    data = _corrupted_data()
    target = anomaly_target(data, {"node_labels": {}})
    assert target.shape == (3, len(ANOMALY_TYPES))
    assert target.sum().item() == 0.0


# ── split ──

def test_split_centers_seeded_disjoint_and_deterministic():
    ids = ["ep_3", "ep_1", "ep_2"]
    train1, val1 = split_centers(ids, val_fraction=0.34, seed=0)
    train2, val2 = split_centers(ids, val_fraction=0.34, seed=0)
    assert (train1, val1) == (train2, val2)          # deterministic
    assert len(val1) == 1                            # round(3*0.34)=1, >=2 -> >=1 val
    assert set(train1) | set(val1) == set(ids)       # covers all
    assert set(train1).isdisjoint(val1)              # disjoint


def test_split_centers_single_center_empty_val():
    train, val = split_centers(["ep_1"], val_fraction=0.5, seed=0)
    assert train == ["ep_1"] and val == []  # 1 center -> no val (honest, not forced)