"""Offline tests for the Phase 2b Retrieval Gate.

All CPU-runnable against ``ReferenceSSM`` + the deterministic ``stub`` embedder
(no ``sentence_transformers`` download, no Ollama, no WaveDB on the gate path).
The integration test uses a tmp_path WaveDB store (mirrors
``tests/test_retriever.py``) with a stub planner. The backbone-load test is
gated on the Phase 2a checkpoint existing locally so the suite runs on a fresh
clone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig, INSTANCE_CONFIGS
from src.subconscious.gate import GateContext
from src.subconscious.retrieval_gate import RetrievalGate
from src.subconscious.routing import (
    AVAILABLE_DOMAINS, META_SKILLS, MODEL_SIZES, PATHWAYS,
    RoutingDecision, RoutingOutcome,
)
from src.subconscious.training.routing_training import (
    OutcomeBasedTrainer, RetrievalGateTrainingConfig,
    build_embedder, evaluate_routing, load_backbone,
    load_routing_pairs, train_retrieval_gate_supervised,
)

BACKBONE_PATH = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"


def _gate() -> RetrievalGate:
    bb = JGSBackbone(BackboneConfig())
    return RetrievalGate(bb)


def _example(query, domains, pathway, skills=None, size="3B", delib=False):
    return {
        "query": query,
        "route": {
            "domains": domains, "pathway": pathway,
            "meta_skills": skills or [], "model_size": size,
            "needs_deliberation": delib, "confidence": 0.9, "reasoning": "test",
        },
    }


# ── contract / shape ──

def test_forward_returns_five_logit_heads_with_correct_vocab():
    gate = _gate()
    emb = torch.randn(3, 384)
    gate.reset_state(3)
    logits, gate_decision, output = gate.forward(emb)
    assert logits["domain"].shape == (3, len(AVAILABLE_DOMAINS))
    assert logits["pathway"].shape == (3, len(PATHWAYS))
    assert logits["skill"].shape == (3, len(META_SKILLS))
    assert logits["model_size"].shape == (3, len(MODEL_SIZES))
    assert logits["deliberation"].shape == (3, 1)
    assert output.shape == (3, INSTANCE_CONFIGS["retrieval_gate"].output_dim)
    assert gate_decision.confidence is not None


def test_route_returns_valid_decision():
    gate = _gate()
    dec = gate.route(torch.randn(1, 384))
    assert isinstance(dec, RoutingDecision)
    assert all(d in AVAILABLE_DOMAINS for d in dec.domains)
    assert dec.pathway in PATHWAYS
    assert all(s in META_SKILLS for s in dec.meta_skills)
    assert dec.model_size in MODEL_SIZES
    assert isinstance(dec.needs_deliberation, bool)
    assert 0.0 <= dec.confidence <= 1.0
    # An untrained gate can still produce a vacuous domain list via the
    # argmax fallback — assert the route is always actionable (>=1 domain).
    assert len(dec.domains) >= 1


def test_route_text_uses_injected_embedder():
    gate = _gate()
    embedder = build_embedder("stub")
    dec = gate.route_text("Why did we choose WaveDB over Python?", embedder)
    assert dec.pathway in PATHWAYS
    assert dec.model_size in MODEL_SIZES


def test_gate_parameters_exclude_backbone():
    bb = JGSBackbone(BackboneConfig())
    gate = RetrievalGate(bb)
    bb_params = sum(p.numel() for p in bb.parameters())
    gate_trainable = sum(p.numel() for p in gate.parameters() if p.requires_grad)
    # The backbone is stored via object.__setattr__ (not a submodule), so the
    # gate's param set must NOT include the ~19.5M backbone params.
    assert gate_trainable < bb_params
    # And every backbone param stays trainable-unless-frozen (not owned by gate).
    assert all(p.requires_grad for p in bb.parameters())


def test_backbone_frozen_after_load_backbone():
    if not Path(BACKBONE_PATH).exists():
        pytest.skip("Phase 2a backbone checkpoint not present locally")
    bb = load_backbone(BACKBONE_PATH, BackboneConfig(), device="cpu")
    assert all(not p.requires_grad for p in bb.parameters())
    assert sum(p.numel() for p in bb.parameters()) == 19_518_016


# ── supervised training ──

def test_supervised_step_runs_and_decreases_loss(tmp_path):
    gate = _gate()
    bb = gate.backbone
    for p in bb.parameters():
        p.requires_grad = False
    embedder = build_embedder("stub")
    train = [
        _example("What is WaveDB?", ["database"], "graph_retrieve", ["factual_recall"], "3B"),
        _example("Design a new sync mode combining safety and throughput",
                 ["database", "coding"], "conscious_deliberation",
                 ["creative_synthesis", "tradeoff_analysis"], "70B", delib=True),
    ] * 4  # 8 examples (2 distinct, repeated) so the gate can overfit them
    val = train[:2]
    cfg = RetrievalGateTrainingConfig(epochs=8, batch_size=4, device="cpu",
                                       checkpoint_dir=str(tmp_path / "gate"))

    losses = []

    def cb(epoch, tl, va):
        losses.append(tl)

    train_retrieval_gate_supervised(gate, bb, train, val, embedder, cfg, progress_cb=cb)
    assert len(losses) == 8
    assert all(l == l for l in losses)  # no NaN
    # Overfitting 2 distinct examples → loss drops meaningfully.
    assert losses[-1] < losses[0]
    # L12: the checkpoint must be LEAN — gate.state_dict() excludes the backbone
    # (stored via object.__setattr__, not a submodule). No "backbone.*" keys,
    # and the saved param count is far below the 19.5M backbone.
    import torch as _t
    ckpt = _t.load(tmp_path / "gate" / "best.pt", map_location="cpu", weights_only=False)
    sd = ckpt["gate"]
    assert not any("backbone" in k for k in sd)
    assert sum(v.numel() for v in sd.values()) < 19_000_000


def test_evaluate_routing_returns_score_in_unit_interval():
    gate = _gate()
    embedder = build_embedder("stub")
    val = [_example("What is WaveDB?", ["database"], "graph_retrieve")]
    emb = torch.tensor(embedder.encode([v["query"] for v in val]), dtype=torch.float32)
    score = evaluate_routing(gate, val, emb, torch.device("cpu"))
    assert 0.0 <= score <= 1.0


def test_load_routing_pairs_parses_jsonl(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(
        '{"query":"q1","route":{"domains":["database"],"pathway":"graph_retrieve",'
        '"meta_skills":["factual_recall"],"model_size":"3B","needs_deliberation":false}}\n'
        'not-json-line\n'
        '{"query":"q2","route":{"domains":["coding"],"pathway":"tool_plan",'
        '"meta_skills":[],"model_size":"8B","needs_deliberation":true}}\n'
        # malformed-schema records the Oracle occasionally emits — must DROP
        # (not silently degrade to default labels):
        '{"query":"q3","route":{"domain":["personal"],"pathway":"graph_retrieve",'
        '"model_size":"3B"}}\n'                              # "domain" not "domains"
        '{"query":"q4","route":{"domains":[],"pathway":"nonsense",'
        '"model_size":"3B"}}\n'                               # out-of-vocab pathway
        '{"query":"q5","route":{"domains":[],"pathway":"graph_retrieve",'
        '"model_size":"999B"}}\n',                           # out-of-vocab size
        encoding="utf-8",
    )
    recs = load_routing_pairs(str(p))
    assert len(recs) == 2
    assert recs[0]["query"] == "q1"
    assert recs[1]["query"] == "q2"


# ── outcome-based trainer ──

def test_outcome_trainer_noop_below_min_buffer():
    gate = _gate()
    cfg = RetrievalGateTrainingConfig(min_buffer=5, outcome_batch_size=2)
    trainer = OutcomeBasedTrainer(gate, cfg)
    # Below min_buffer → train_from_outcomes is a no-op (returns 0.0).
    for _ in range(3):
        trainer.record_outcome(
            torch.randn(1, 384), None,
            RoutingDecision(["database"], "graph_retrieve", ["factual_recall"], "3B", False, 0.5),
            RoutingOutcome(user_accepted=True),
        )
    assert trainer.train_from_outcomes() == 0.0


def test_outcome_trainer_runs_above_min_buffer():
    gate = _gate()
    cfg = RetrievalGateTrainingConfig(min_buffer=3, outcome_batch_size=4, online_lr=1e-3)
    trainer = OutcomeBasedTrainer(gate, cfg)
    for _ in range(5):
        trainer.record_outcome(
            torch.randn(1, 384), None,
            RoutingDecision(["database"], "graph_retrieve", ["factual_recall"], "3B", False, 0.5),
            RoutingOutcome(user_accepted=True),  # reward +1.0
        )
    loss = trainer.train_from_outcomes()
    assert loss == loss  # finite, not NaN


# ── integration (tmp_path WaveDB store, stub planner, untrained gate) ──

class _StubPlanner:
    def __init__(self, plan): self._plan = plan

    def plan(self, prompt, conversation_history=None):
        return self._plan


def _ep(eid, entities=None, summary=None):
    return Episode(id=eid, timestamp="2026-07-03T10:00:00",
                   summary=summary or f"summary {eid}",
                   full_text=f"User: u{eid}\nAssistant: a{eid}",
                   entities=entities or [], topics=[], tones=[])


def test_retrieve_with_routing_contract(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        store.encode_episode(_ep("ep_001", entities=["Alice"], summary="WAL config"))
        gate = _gate()
        embedder = build_embedder("stub")
        retr = HippocampalRetriever(store, planner=_StubPlanner({"entities": ["Alice"],
                                                                 "entity_mode": "union"}),
                                    retrieval_gate=gate, embedder=embedder)
        result = retr.retrieve_with_routing("What did Alice say about the WAL config?")
        assert set(result.keys()) == {"type", "route", "results", "context", "supported"}
        assert result["route"].pathway in PATHWAYS
        if result["supported"]:
            assert isinstance(result["results"], list)
            assert isinstance(result["context"], (str, type(None)))
        else:
            assert result["results"] == []
            assert result["context"] is None
    finally:
        store.close()   # always release the WaveDB handle (Windows file locks)


def test_retrieve_unchanged_without_gate(tmp_path):
    """Back-compat: retrieve() still returns list[dict] when no gate is set."""
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        store.encode_episode(_ep("ep_001", entities=["Alice"]))
        retr = HippocampalRetriever(store, planner=_StubPlanner({"entities": ["Alice"],
                                                                "entity_mode": "union"}))
        results = retr.retrieve("What did Alice say?")
        assert isinstance(results, list)
        assert results and results[0]["episode_id"] == "ep_001"
        with pytest.raises(RuntimeError):
            retr.retrieve_with_routing("anything")  # no gate configured
    finally:
        store.close()


def test_record_outcome_noop_without_gate(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        retr = HippocampalRetriever(store, planner=_StubPlanner({}))
        # No gate → record_outcome is a silent no-op, not an error.
        retr.record_outcome("q", RoutingDecision(["database"], "graph_retrieve", [], "3B", False, 0.5),
                            RoutingOutcome(user_accepted=True))
    finally:
        store.close()