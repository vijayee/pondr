"""Tests for the Phase 1d training-data validators.

Feeds synthetic JSONL files (good + bad) and asserts the per-dataset
``validate_*`` checks flag the right problems: parse errors, missing top-level
keys, missing nested label keys, missing files. Pure JSON-shape checks — no
Oracle / WaveDB dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training.validators import (
    RECORD_KEYS,
    validate_all,
    validate_bonsai,
    validate_code_aware,
    validate_gates,
    validate_gnn,
    validate_jepa,
    validate_file,
)


def _write(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write((rec if isinstance(rec, str) else json.dumps(rec)) + "\n")


# ── validate_file primitives ──


def test_validate_file_good_records(tmp_path):
    p = tmp_path / "a.jsonl"
    _write(p, [{"query": "q", "route": {"pathway": "ssm_direct"}},
               {"query": "q2", "route": {}}])
    res = validate_file(p, required_keys={"query", "route"})
    assert res["ok"] is True
    assert res["lines"] == 2
    assert res["parse_errors"] == []


def test_validate_file_parse_error(tmp_path):
    p = tmp_path / "a.jsonl"
    _write(p, ['{"query": "q", "route": {}}', "{not json"])
    res = validate_file(p, required_keys={"query", "route"})
    assert res["ok"] is False
    assert len(res["parse_errors"]) == 1
    assert "line 2" in res["parse_errors"][0]


def test_validate_file_missing_keys(tmp_path):
    p = tmp_path / "a.jsonl"
    _write(p, [{"query": "q"}])  # no "route"
    res = validate_file(p, required_keys={"query", "route"})
    assert res["ok"] is False
    assert res["missing_keys"] == 1


def test_validate_file_missing_file(tmp_path):
    res = validate_file(tmp_path / "nope.jsonl", required_keys={"x"})
    assert res["ok"] is False
    assert res.get("missing_file") is True
    assert res["lines"] == 0


def test_validate_file_label_keys_nested_under_label(tmp_path):
    p = tmp_path / "g.jsonl"
    _write(p, [{"input": {}, "label": {"confidence": 0.5}}])
    res = validate_file(p, required_keys={"input", "label"},
                        label_keys={"confidence"})
    assert res["ok"] is True
    assert res["label_missing"] == 0


def test_validate_file_label_keys_missing_in_nested(tmp_path):
    p = tmp_path / "g.jsonl"
    _write(p, [{"input": {}, "label": {}}])  # label missing required sub-key
    res = validate_file(p, required_keys={"input", "label"},
                        label_keys={"confidence"})
    assert res["ok"] is False
    assert res["label_missing"] == 1


# ── per-dataset validators ──


def test_validate_gnn_all_good(tmp_path):
    gnn = tmp_path / "gnn"
    _write(gnn / "salience_labels.jsonl",
           [{"subgraph_id": "ep_1", "labels": {"node_scores": []}, "cost": 0.0}])
    _write(gnn / "cluster_labels.jsonl",
           [{"subgraph_id": "ep_1", "labels": {"clusters": []}}])
    _write(gnn / "link_prediction_labels.jsonl",
           [{"subgraph_id": "ep_1", "labels": {"predicted_edges": []}}])
    _write(gnn / "anomaly_labels.jsonl",
           [{"subgraph_id": "ep_1", "labels": {"anomalies": []}}])
    _write(gnn / "ontology_labels.jsonl",
           [{"subgraph_id": "ep_1", "labels": {"suggested_edges": [], "misclassified": []}}])
    res = validate_gnn(gnn)
    assert all(res[t]["ok"] for t in res)
    # Phase 3a Task 3: link_prediction reports an optional_present count for
    # negative_edges (here 0 — the PoC data is positive-only). It must NOT gate
    # ``ok``; positive-only data still validates.
    assert res["link_prediction"]["optional_present"]["negative_edges"] == 0
    assert res["link_prediction"]["ok"] is True


def test_validate_gnn_counts_negative_edges_without_failing(tmp_path):
    """Records carrying negative_edges are counted; positive-only still ok."""
    gnn = tmp_path / "gnn"
    _write(gnn / "link_prediction_labels.jsonl", [
        {"subgraph_id": "ep_1", "labels": {"predicted_edges": [{"x": 1}], "negative_edges": [{"y": 1}]}},
        {"subgraph_id": "ep_2", "labels": {"predicted_edges": [{"x": 2}]}},  # no negatives
    ])
    res = validate_gnn(gnn)
    assert res["link_prediction"]["ok"] is True
    assert res["link_prediction"]["optional_present"]["negative_edges"] == 1  # only ep_1
    assert res["link_prediction"]["lines"] == 2


def test_validate_gnn_ontology_accepts_suggested_edges_alone(tmp_path):
    gnn = tmp_path / "gnn"
    _write(gnn / "ontology_labels.jsonl",
           [{"subgraph_id": "ep_1", "labels": {"suggested_edges": []}}])
    res = validate_gnn(gnn)
    assert res["ontology"]["ok"] is True


def test_validate_gnn_missing_file_reports_missing(tmp_path):
    res = validate_gnn(tmp_path / "gnn")
    assert all(r.get("missing_file") for r in res.values())


def test_validate_bonsai(tmp_path):
    b = tmp_path / "bonsai"
    _write(b / "query_planning_pairs.jsonl",
           [{"conversation_id": "ep_1", "conversation_text": "t",
             "training_pair": {"question": "q", "query": "g", "reasoning": "r"}}])
    _write(b / "relation_extraction_pairs.jsonl",
           [{"conversation_id": "ep_1", "conversation_text": "t", "relations": []}])
    res = validate_bonsai(b)
    assert res["query_planning"]["ok"] is True
    assert res["relation_extraction"]["ok"] is True


def test_validate_jepa(tmp_path):
    j = tmp_path / "jepa"
    _write(j / "routing_pairs.jsonl",
           [{"query": "q", "route": {"pathway": "ssm_direct"},
             "expected_pathways": ["ssm_direct"], "cost": 0.0}])
    assert validate_jepa(j)["routing"]["ok"] is True


def test_validate_gates(tmp_path):
    g = tmp_path / "gates"
    for gate in ("uncertainty_detector", "aspirational_model", "self_model"):
        _write(g / f"{gate}.jsonl",
               [{"input": {}, "label": {"some_key": 1}, "cost": 0.0}])
    res = validate_gates(g)
    assert all(r["ok"] for r in res.values())


def test_validate_code_aware(tmp_path):
    c = tmp_path / "code_aware"
    _write(c / "code_aware_examples.jsonl",
           [{"domain": "File", "label": {"conversation": "c",
            "extracted_entities": ["auth.py"]}, "cost": 0.0}])
    res = validate_code_aware(c)
    assert res["code_aware_examples"]["ok"] is True


def test_validate_code_aware_label_missing_keys(tmp_path):
    c = tmp_path / "code_aware"
    _write(c / "code_aware_examples.jsonl",
           [{"domain": "File", "label": {"conversation": "c"}}])  # no extracted_entities
    res = validate_code_aware(c)
    assert res["code_aware_examples"]["ok"] is False
    assert res["code_aware_examples"]["label_missing"] == 1


def test_validate_all_partial_run(tmp_path):
    # Only gnn + jepa present; others absent → empty dicts, not errors.
    gnn = tmp_path / "gnn"
    _write(gnn / "salience_labels.jsonl",
           [{"subgraph_id": "ep_1", "labels": {"node_scores": []}}])
    j = tmp_path / "jepa"
    _write(j / "routing_pairs.jsonl", [{"query": "q", "route": {}}])
    res = validate_all(tmp_path)
    assert res["gnn"]["salience"]["ok"] is True
    assert res["jepa"]["routing"]["ok"] is True
    assert res["bonsai"] == {}
    assert res["gates"] == {}
    assert res["code_aware"] == {}


def test_record_keys_match_generator_outputs():
    """Guard: the expected top-level keys match what the generators write."""
    # GNN record
    assert RECORD_KEYS["salience_labels"] == {"subgraph_id", "labels"}
    # Bonsai
    assert RECORD_KEYS["query_planning_pairs"] == {"conversation_id", "training_pair"}
    assert RECORD_KEYS["relation_extraction_pairs"] == {"conversation_id", "relations"}
    # JEPA / gates / code_aware
    assert RECORD_KEYS["routing_pairs"] == {"query", "route"}
    assert RECORD_KEYS["uncertainty_detector"] == {"input", "label"}
    assert RECORD_KEYS["code_aware_examples"] == {"domain", "label"}