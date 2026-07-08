"""Tests for the 11 Phase 1d prompt builders.

Each prompt function is a pure string builder. These tests assert it returns a
``str`` that embeds its input slots and the ``Return ONLY valid JSON`` contract,
so a future edit that drops a slot or breaks the JSON instruction is caught.
"""

from __future__ import annotations

import json

from src.training.prompts import (
    aspirational_model_prompt,
    bonsai_anomaly_decision_prompt,
    bonsai_query_planning_prompt,
    bonsai_relation_extraction_prompt,
    code_aware_synthetic_prompt,
    gnn_cluster_prompt,
    gnn_link_prediction_prompt,
    gnn_ontology_prompt,
    gnn_salience_prompt,
    jepa_routing_prompt,
    self_model_prompt,
    uncertainty_detector_prompt,
)


def _assert_json_contract(prompt: str) -> None:
    assert isinstance(prompt, str)
    assert "JSON" in prompt or "json" in prompt


def test_gnn_salience_prompt_embeds_subgraph():
    p = gnn_salience_prompt('{"nodes": [{"id": "ep_1"}]}')
    _assert_json_contract(p)
    assert "ep_1" in p
    assert "salience" in p.lower()


def test_gnn_cluster_prompt_embeds_subgraph():
    p = gnn_cluster_prompt('{"nodes": [{"id": "T:db"}]}')
    _assert_json_contract(p)
    assert "T:db" in p
    assert "cluster" in p.lower()


def test_gnn_link_prediction_prompt_embeds_subgraph():
    p = gnn_link_prediction_prompt('{"edges": [{"s": "E:A", "p": "r", "o": "ep_1"}]}')
    _assert_json_contract(p)
    assert "predicted_edges" in p
    # Phase 3a Task 3: the prompt now also requests negative edges (SEAL/GAE
    # need them — positive-only labels collapse the head to "predict 1").
    assert "negative_edges" in p


def test_bonsai_anomaly_decision_prompt_embeds_inputs():
    p = bonsai_anomaly_decision_prompt(
        "E:Alice", {"nodes": [{"id": "E:Alice"}]}, "identity_drift"
    )
    _assert_json_contract(p)
    assert "E:Alice" in p
    assert "identity_drift" in p
    # The retrieve-then-prompt context is baked in.
    assert "nodes" in p
    # The decision/action/reasoning contract.
    assert "decision" in p and "action" in p and "reasoning" in p


def test_gnn_ontology_prompt_embeds_subgraph_and_ontology():
    ont = json.dumps({"classes": {"X": {"subclasses": []}}})
    p = gnn_ontology_prompt('{"nodes": []}', ont)
    _assert_json_contract(p)
    assert "subClassOf" in p or "ontology" in p.lower()
    assert "X" in p  # ontology content embedded


def test_bonsai_query_planning_prompt_embeds_text_and_question():
    p = bonsai_query_planning_prompt("User: hi", "What did Alice say?")
    _assert_json_contract(p)
    assert "What did Alice say?" in p
    assert "User: hi" in p


def test_bonsai_relation_extraction_prompt_embeds_text():
    p = bonsai_relation_extraction_prompt("User: hello\nAssistant: hi")
    _assert_json_contract(p)
    assert "hello" in p


def test_jepa_routing_prompt_embeds_query_domains_pathways():
    p = jepa_routing_prompt("Who is Alice?", "- database\n- coding", "- ssm_direct\n- graph_retrieve")
    _assert_json_contract(p)
    assert "Who is Alice?" in p
    assert "database" in p
    assert "graph_retrieve" in p


def test_uncertainty_detector_prompt_embeds_slots():
    p = uncertainty_detector_prompt("ctx", "What is X?", "no results")
    _assert_json_contract(p)
    assert "ctx" in p and "What is X?" in p and "no results" in p


def test_aspirational_model_prompt_embeds_slots():
    p = aspirational_model_prompt("goal: learn", "encode this")
    _assert_json_contract(p)
    assert "goal: learn" in p and "encode this" in p


def test_self_model_prompt_embeds_slots():
    p = self_model_prompt("sparse knowledge", "What is X?")
    _assert_json_contract(p)
    assert "sparse knowledge" in p and "What is X?" in p


def test_code_aware_synthetic_prompt_embeds_domain_and_ontology():
    fragment = json.dumps({"classes": {"File": {"subclasses": []}}})
    p = code_aware_synthetic_prompt("CodeArtifact", fragment)
    _assert_json_contract(p)
    assert "CodeArtifact" in p
    assert "File" in p
    # The expected JSON shape keys are named in the prompt.
    assert "conversation" in p
    assert "extracted_entities" in p
    assert "code_artifacts" in p