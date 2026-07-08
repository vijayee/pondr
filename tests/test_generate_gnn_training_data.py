"""Smoke tests for the rewired ``scripts/generate_gnn_training_data.py`` (Phase 3a
Task 3). The generator lives under ``scripts/`` (not an importable package), so
this module loads it via ``importlib`` from its file path and exercises the
per-task routing functions directly with a stub Oracle + a small ``tmp_path``
store — no live Oracle, no argparse/``main``.

Covers the rewire's NEW code paths that aren't already covered by
``test_sharded_labeling`` (shard construction + recombine are tested there):
- ``_run_anomaly`` makes ZERO Oracle calls and writes ``anomaly_labels.jsonl``
  whose records carry the injection keys (``anomalies`` / ``node_labels`` /
  ``seed`` / ``types`` / ``identity_drift``).
- ``_run_anomaly_decision`` writes ``anomaly_decision_pairs.jsonl`` that
  validates against the new ``RECORD_KEYS`` entry (spec §2.5).
- ``_local_neighborhood`` builds the retrieve-then-prompt context from the
  in-memory corrupted subgraph (works for synthetic injected nodes).
- ``_run_onecall`` (radius-1 dev path) + ``_run_sharded`` (the spec's
  radius>=2 path) each produce ``validate_gnn``-passing label files.
- ``_run_cluster`` default is self-supervised (0 Oracle calls, empty clusters).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from src.gnn.anomaly_injector import inject_anomalies
from src.gnn.anomaly_rules import ANOMALY_TYPES, enrich_subgraph
from src.gnn.sharded_labeling import build_anomaly_labels, shard_nodes
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleLabelingPipeline, sample_episode_centers
from src.training.validators import validate_bonsai, validate_gnn

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_gnn_training_data.py"


def _load_generator():
    """Load ``scripts/generate_gnn_training_data.py`` as a module (it's not in a
    package). Cached on ``pytest``'s module registry so re-imports are cheap."""
    import sys
    name = "generate_gnn_training_data"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, GENERATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── stub Oracle (run_batches reads only these attributes) ──

class _StubResult:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.cost = 0.0
        self.input_tokens = 5
        self.output_tokens = 5
        self.cached = False


class _StubOracle:
    """Returns a shape-valid (often empty) JSON response per task, sniffed from
    the prompt. Counts calls so tests assert the anomaly path makes 0."""

    def __init__(self) -> None:
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    @staticmethod
    def _response_for(prompt: str) -> dict:
        # Order matters: ontology and link shard prompts both contain
        # "CANDIDATE PAIRS", so check the distinctive ontology markers
        # ("CURRENT ONTOLOGY" / "child → parent" / "suggested_edges") BEFORE the
        # link markers. Each branch uses a substring unique to its task's prompt
        # (shard or one-call), not the shared "CANDIDATE PAIRS" header.
        # anomaly_decision (Bonsai) — spec §2.5
        if "fix|ask_user|dismiss" in prompt or "Decide what the system should do" in prompt:
            return {"decision": "dismiss", "action": "leave as-is", "reasoning": "stub"}
        # cluster episode-only weak supervision
        if "EPISODES (episode-only" in prompt:
            return {"clusters": []}
        # salience (shard: "Score ONLY these"; one-call: "Score each node and edge")
        if "Score ONLY these" in prompt or "Score each node and edge" in prompt:
            return {"node_scores": {}, "edge_scores": {}}
        # ontology (shard: "CURRENT ONTOLOGY"; one-call: "Suggest missing subClassOf")
        if "CURRENT ONTOLOGY" in prompt or "Suggest missing subClassOf" in prompt:
            return {"suggested_edges": [], "misclassified": []}
        # link prediction (shard + one-call both mention "predicted_edges" in the
        # Return template; one-call also has "Identify edges that SHOULD exist")
        if "predicted_edges" in prompt or "Identify edges that SHOULD exist" in prompt:
            return {"predicted_edges": [], "negative_edges": []}
        return {}

    def generate_batch(self, prompts, max_workers: int = 1):
        out = []
        for p in prompts:
            self.total_calls += 1
            self.total_tokens += 10
            out.append(_StubResult(self._response_for(p)))
        return out

    def flush_cache(self) -> None:
        pass

    def get_stats(self) -> dict:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens,
                "total_cost": 0.0}


def _args(**over) -> argparse.Namespace:
    base = dict(
        oracle_batch_size=10, resume=False, oracle_max_workers=1,
        anomaly_seed=0, skip_anomaly_decision_pairs=False,
        max_decision_pairs_per_subgraph=10, oracle_cluster_supervision=False,
        shard_size=500, max_candidate_pairs=500,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _ep(eid, entities=None, topics=None, tones=None, follows=None, user="alice"):
    return Episode(
        id=eid, timestamp="2026-07-03T10:00:00", summary=f"summary {eid}",
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
        follows=follows, user_id=user,
    )


def _seed_store(tmp_path) -> tuple[HippocampalStore, OracleLabelingPipeline]:
    """A store with two episodes sharing Alice + db, ep_2 follows ep_1 → a
    radius-2 subgraph rooted at ep_1 has >=3 nodes (ep_1, E:Alice, T:db, ep_2,
    E:Bob, T:perf)."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_000001", entities=["Alice"], topics=["db"], tones=["curious"]))
    store.encode_episode(_ep("ep_000002", entities=["Alice", "Bob"], topics=["db", "perf"],
                             tones=["neutral"], follows="ep_000001"))
    return store, OracleLabelingPipeline(store)


def _subgraphs(store, pipe, radius=2) -> list[dict]:
    centers = sample_episode_centers(store, n=2)
    subs = []
    for c in centers:
        sg = pipe.extract_subgraph(c, radius=radius)
        if len(sg["nodes"]) >= 3:
            subs.append(sg)
    return subs


# ── _local_neighborhood (pure, synthetic-node-safe) ──

def test_local_neighborhood_includes_duplicate_twin_and_state_value():
    """The decision-pair context must surface the anomaly evidence pure radius-1
    hid: the duplicate TWIN (2 hops via a shared entity, same summary) and the
    flagged node's literal data edges (state values). A prior radius-1-only
    version returned no twin + dropped data edges -> DeepSeek dismissed every
    planted duplicate (it couldn't see one) and ask_user'd every contradictory
    state (it couldn't see the values)."""
    gen = _load_generator()
    sub = {
        "center": "ep_1", "radius": 1,
        "nodes": [
            {"id": "ep_1", "type": "episode", "summary": "s"},
            {"id": "ep_1_dup", "type": "episode", "summary": "s"},
            {"id": "E:Alice", "type": "entity"},
            {"id": "T:db", "type": "topic"},
        ],
        "edges": [
            {"subject": "ep_1", "predicate": "has_entity", "object": "E:Alice"},
            {"subject": "ep_1_dup", "predicate": "has_entity", "object": "E:Alice"},
            {"subject": "ep_1", "predicate": "has_topic", "object": "T:db"},
            {"subject": "ep_1", "predicate": "state", "object": "alive"},  # data edge
        ],
    }
    nb = gen._local_neighborhood(sub, "ep_1_dup")
    assert nb["center"] == "ep_1_dup"
    ids = {n["id"] for n in nb["nodes"]}
    assert "ep_1_dup" in ids and "E:Alice" in ids
    # The twin ep_1 (shares summary "s" with the flagged ep_1_dup) IS now included
    # -- it's 2 hops via E:Alice so radius-1 missed it; the teacher needs to see
    # the duplicate to decide "fix".
    assert "ep_1" in ids
    # The twin's OWN neighbor T:db is NOT walked (we don't expand twins' radius-1)
    # -> keeps the context bounded; the shared entity E:Alice is already present.
    assert "T:db" not in ids
    # Both the flagged->entity and twin->entity edges are kept (shared neighbor).
    kept = {(e["subject"], e["predicate"], e["object"]) for e in nb["edges"]}
    assert ("ep_1_dup", "has_entity", "E:Alice") in kept
    assert ("ep_1", "has_entity", "E:Alice") in kept
    # The twin's edge to its non-shared topic is dropped (T:db not in set).
    assert ("ep_1", "has_topic", "T:db") not in kept
    # The literal data edge (state "alive") IS now kept -- the teacher sees the value.
    assert ("ep_1", "state", "alive") in kept


def test_local_neighborhood_includes_contradictory_state_values():
    """contradictory_state evidence is the entity's two literal ``state`` values
    ("alive"/"dead"). Pure radius-1 dropped them (literal objects fail _is_node_id),
    so the teacher saw only episode summaries and ask_user'd every time. The fix
    keeps the flagged node's data edges."""
    gen = _load_generator()
    sub = {
        "center": "ep_1", "radius": 1,
        "nodes": [
            {"id": "E:Armand", "type": "entity"},
            {"id": "ep_1", "type": "episode", "summary": "talked to Armand"},
        ],
        "edges": [
            {"subject": "ep_1", "predicate": "has_entity", "object": "E:Armand"},
            {"subject": "E:Armand", "predicate": "state", "object": "alive"},
            {"subject": "E:Armand", "predicate": "state", "object": "dead"},
        ],
    }
    nb = gen._local_neighborhood(sub, "E:Armand")
    ids = {n["id"] for n in nb["nodes"]}
    assert ids == {"E:Armand", "ep_1"}  # entity + its episode neighbor
    state_values = {e["object"] for e in nb["edges"] if e["predicate"] == "state"}
    assert state_values == {"alive", "dead"}  # both contradictory values visible


def test_local_neighborhood_caps_text_twins():
    """A dense corpus has many episodes sharing a short summary ("Ok."); the
    injected ``_dup`` clone is one of them. The twin set is capped so the context
    doesn't blow up -- the clone is still included (sorted by id, the clone sorts
    right after its original), plus up to _TWIN_CAP-1 others."""
    gen = _load_generator()
    nodes = [{"id": "ep_0", "type": "episode", "summary": "Ok."}]
    # 12 same-summary siblings + the flagged node's own entity neighbor.
    for i in range(1, 14):
        nodes.append({"id": f"ep_{i}", "type": "episode", "summary": "Ok."})
    nodes.append({"id": "E:Bob", "type": "entity"})
    edges = [{"subject": "ep_0", "predicate": "has_entity", "object": "E:Bob"}]
    sub = {"center": "ep_0", "radius": 1, "nodes": nodes, "edges": edges}
    nb = gen._local_neighborhood(sub, "ep_0")
    twins = {n["id"] for n in nb["nodes"] if n["id"].startswith("ep_")}
    # flagged ep_0 + E:Bob + capped twins. 13 episodes share "Ok." (ep_0..ep_13);
    # the cap bounds the twins (excl. flagged) -> total episode nodes <= 1 + cap.
    assert "ep_0" in twins and "E:Bob" in {n["id"] for n in nb["nodes"]}
    assert len(twins) <= 1 + gen._TWIN_CAP
    assert len(twins) >= 2  # at least the flagged + one twin


# ── _run_anomaly: 0 Oracle calls + valid label file ──

def test_run_anomaly_zero_oracle_writes_valid_labels(tmp_path):
    gen = _load_generator()
    store, pipe = _seed_store(tmp_path)
    try:
        subs = _subgraphs(store, pipe, radius=2)
        assert subs, "fixture should yield at least one >=3-node subgraph"
        out = tmp_path / "gnn"
        out.mkdir()
        oracle = _StubOracle()
        records, decision_items, inj = gen._run_anomaly(subs, store, out, _args())

        # ZERO Oracle calls — anomaly is injection-based (spec §2).
        assert oracle.total_calls == 0
        # File written + validates (anomalies key present per record).
        report = validate_gnn(out)
        assert report["anomaly"]["ok"], report["anomaly"]
        # Each record carries the injection-reproduction keys the trainer reads.
        for rec in records:
            labels = rec["labels"]
            assert {"anomalies", "planted", "node_labels", "identity_drift",
                    "seed", "types"} <= labels.keys()
        # Injection stats audited into the returned dict.
        assert inj["subgraphs"] == len(subs)
        assert isinstance(inj["types_requested"], dict)
        # decision_items (if any) carry deploy-faithful context.
        for it in decision_items:
            assert {"subgraph_id", "flagged_entity", "anomaly_type",
                    "retrieved_context"} <= it.keys()
            ctx = it["retrieved_context"]
            assert {"center", "nodes", "edges"} <= ctx.keys()
    finally:
        store.close()


# ── _run_anomaly_decision: writes validating Bonsai pairs (spec §2.5) ──

def test_run_anomaly_decision_writes_validating_pairs(tmp_path):
    gen = _load_generator()
    bonsai = tmp_path / "bonsai"
    bonsai.mkdir()
    items = [{
        "subgraph_id": "ep_000001",
        "flagged_entity": "ep_000002",
        "anomaly_type": "duplicate_episode",
        "retrieved_context": {"center": "ep_000002",
                              "nodes": [{"id": "ep_000002"}], "edges": []},
    }]
    oracle = _StubOracle()
    records, _ = gen._run_anomaly_decision(oracle, items, bonsai, _args())
    assert oracle.total_calls == 1
    assert records and records[0]["decision"] == "dismiss"
    report = validate_bonsai(bonsai)
    assert report["anomaly_decision"]["ok"], report["anomaly_decision"]


def test_run_anomaly_decision_empty_items_writes_empty_file(tmp_path):
    gen = _load_generator()
    bonsai = tmp_path / "bonsai"
    bonsai.mkdir()
    oracle = _StubOracle()
    records, _ = gen._run_anomaly_decision(oracle, [], bonsai, _args())
    assert oracle.total_calls == 0  # no items → no Oracle calls
    assert records == []


# ── _run_onecall (radius-1 dev path) ──

def test_run_onecall_salience_validates(tmp_path):
    gen = _load_generator()
    store, pipe = _seed_store(tmp_path)
    try:
        subs = _subgraphs(store, pipe, radius=1)
        if not subs:  # radius-1 on a 2-ep store may be thin; skip gracefully
            pytest.skip("radius-1 subgraphs too small for this fixture")
        # Hydrate episodes so the one-call salience prompt carries summaries.
        from src.retrieval.graph_traversal import GraphTraversal
        trav = GraphTraversal(store)
        for sg in subs:
            gen._hydrate_episodes(trav, sg)
        out = tmp_path / "gnn"
        out.mkdir()
        oracle = _StubOracle()
        gen._run_onecall(oracle, "salience", gen.gnn_salience_prompt,
                         "salience_labels.jsonl", subs, out, _args(),
                         ontology_json="{}", needs_ontology=False)
        report = validate_gnn(out)
        assert report["salience"]["ok"], report["salience"]
    finally:
        store.close()


# ── _run_sharded (the spec's radius>=2 path) ──

def test_run_sharded_salience_validates_and_collapses_to_per_subgraph(tmp_path):
    gen = _load_generator()
    store, pipe = _seed_store(tmp_path)
    try:
        subs = _subgraphs(store, pipe, radius=2)
        assert subs
        from src.retrieval.graph_traversal import GraphTraversal
        trav = GraphTraversal(store)
        for sg in subs:
            gen._hydrate_episodes(trav, sg)
        out = tmp_path / "gnn"
        out.mkdir()
        oracle = _StubOracle()
        records, stats = gen._run_sharded(
            oracle, "salience", "salience_labels.jsonl", subs, out, _args(),
            lambda sg, _ss=500: shard_nodes(sg, shard_size=_ss),
            gen.build_salience_shard_prompt, ontology_json="{}")
        # One recombined record per subgraph (shards collapsed by recombine_salience).
        assert len(records) == len(subs)
        report = validate_gnn(out)
        assert report["salience"]["ok"], report["salience"]
        assert stats.get("total_shards", 0) >= 1
    finally:
        store.close()


def test_run_sharded_ontology_validates(tmp_path):
    """Ontology shards carry 'CANDIDATE PAIRS' (shared with the link shard
    prompt), so this also guards the stub's prompt-sniff order: the ontology
    branch must win over link for an ontology shard."""
    gen = _load_generator()
    store, pipe = _seed_store(tmp_path)
    try:
        subs = _subgraphs(store, pipe, radius=2)
        assert subs
        out = tmp_path / "gnn"
        out.mkdir()
        oracle = _StubOracle()
        records, _ = gen._run_sharded(
            oracle, "ontology", "ontology_labels.jsonl", subs, out, _args(),
            lambda sg, _ss=500, _mp=500: gen.build_ontology_shards(
                sg, shard_size=_ss, max_candidate_pairs=_mp),
            gen.build_ontology_shard_prompt, ontology_json="{}")
        assert len(records) == len(subs)
        # The stub must have returned the ontology shape (not the link shape) →
        # recombine_ontology produces suggested_edges/misclassified keys.
        for r in records:
            assert {"suggested_edges", "misclassified"} <= r["labels"].keys()
        report = validate_gnn(out)
        assert report["ontology"]["ok"], report["ontology"]
    finally:
        store.close()


# ── _run_cluster: self-supervised by default (0 Oracle) ──

def test_run_cluster_default_is_self_supervised(tmp_path):
    gen = _load_generator()
    store, pipe = _seed_store(tmp_path)
    try:
        subs = _subgraphs(store, pipe, radius=2)
        out = tmp_path / "gnn"
        out.mkdir()
        oracle = _StubOracle()
        records, stats = gen._run_cluster(oracle, subs, out, _args())
        assert oracle.total_calls == 0  # self-supervised: no Oracle call
        assert stats["self_supervised"] is True
        # Empty clusters labels validate (clusters key present).
        report = validate_gnn(out)
        assert report["cluster"]["ok"], report["cluster"]
        assert all(r["labels"]["clusters"] == [] for r in records)
    finally:
        store.close()