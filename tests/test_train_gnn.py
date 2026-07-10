"""Tiny end-to-end tests for ``src/gnn/train.py`` (Phase 3a Task 4a).

Builds a 3-episode tmp store with a follows chain + shared entities/topics,
hand-writes the 5 GNN label JSONL files in the CURRENT schema (anomaly labels
built by REPRODUCING the deploy injection: extract -> enrich -> inject -> detect,
so the file's ``node_labels`` align with the corrupted graph the trainer
rebuilds from ``(seed, types)``), then runs ``train_gnn`` for ``--head all`` and
``--head salience`` at toy size on CPU. Asserts: finite progress (steps > 0),
the right checkpoint set is written + strict-loadable (mirrors the consolidation
loader), and per-head val metrics are present (None where a head had no usable
labels -- honest, not faked). ``training_feature_for`` is exercised end-to-end
via the anomaly sub-step (an injected ``_dup`` clone reuses the origin's feature;
a synthetic ``M:`` node degrades to a onehot).
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest
import torch

from src.gnn.anomaly_injector import inject_anomalies
from src.gnn.anomaly_rules import detect_anomalies, enrich_subgraph, flag_identity_drift, node_label_vectors
from src.gnn.features import NODE_KIND_INDEX, infer_kind, training_feature_for
from src.gnn.train import GNNTrainConfig, train_gnn
from src.gnn.model import GNNModel
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.training.oracle_labeling import OracleLabelingPipeline


@pytest.fixture(autouse=True)
def _isolate_rng():
    """``train_gnn`` seeds torch + ``random`` globally (intended -- reproducible
    training). Restore both around each test so this module doesn't leak RNG
    state into later test files (e.g. ``test_consolidate``'s untrained-model
    apply test asserts a stochastic outcome and is sensitive to the global
    torch seed)."""
    torch_state = torch.get_rng_state()
    py_state = random.getstate()
    yield
    torch.set_rng_state(torch_state)
    random.setstate(py_state)


def _ep(eid, entities=None, topics=None, tones=None, follows=None, user="alice"):
    return Episode(
        id=eid, timestamp="2026-07-03T10:00:00", summary=f"summary {eid}",
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
        follows=follows, user_id=user,
    )


def _seed_store(tmp_path) -> HippocampalStore:
    """3 episodes: ep_3 -> ep_2 -> ep_1 follows chain; Alice/db shared across
    ep_1+ep_2, Bob/perf shared across ep_2+ep_3. Radius-2 subgraphs rooted at any
    episode have >=3 nodes + entity/topic endpoints to resolve."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_000001", entities=["Alice"], topics=["db"], tones=["curious"]))
    store.encode_episode(_ep("ep_000002", entities=["Alice", "Bob"], topics=["db", "perf"],
                             tones=["neutral"], follows="ep_000001"))
    store.encode_episode(_ep("ep_000003", entities=["Bob"], topics=["perf"], tones=["happy"],
                             follows="ep_000002"))
    return store


def _write_labels(store, pipe, labels_dir: Path, radius: int = 2) -> None:
    """Write the 5 GNN label files + quality_report.json in the current schema.

    Anomaly labels are built by REPRODUCING the deploy injection (the same
    extract->enrich->inject->detect path the trainer re-runs from seed+types),
    so ``node_labels`` in the file aligns with the corrupted graph the trainer
    rebuilds -- no skew between the label and the reproduced structure."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    centers = ["ep_000001", "ep_000002", "ep_000003"]
    handles = {
        "salience": open(labels_dir / "salience_labels.jsonl", "w", encoding="utf-8"),
        "link_prediction": open(labels_dir / "link_prediction_labels.jsonl", "w", encoding="utf-8"),
        "ontology": open(labels_dir / "ontology_labels.jsonl", "w", encoding="utf-8"),
        "cluster": open(labels_dir / "cluster_labels.jsonl", "w", encoding="utf-8"),
        "anomaly": open(labels_dir / "anomaly_labels.jsonl", "w", encoding="utf-8"),
    }
    try:
        for i, sid in enumerate(centers):
            handles["salience"].write(json.dumps({
                "subgraph_id": sid,
                "labels": {"node_scores": {sid: {"salience": 0.5}, "E:Alice": 0.7},
                           "edge_scores": {}},
            }) + "\n")
            # E:Alice + E:Bob are both present in every radius-2 subgraph here.
            handles["link_prediction"].write(json.dumps({
                "subgraph_id": sid,
                "labels": {
                    "predicted_edges": [{"subject": "E:Alice", "object": "E:Bob"}],
                    "negative_edges": [{"subject": "E:Bob", "object": "E:Alice"}],
                },
            }) + "\n")
            # Ontology labels are entity->class typing: ``child``/``entity`` is
            # a real E: node in the subgraph, ``parent``/``suggested_class`` is a
            # bare class NAME that exists in the seeded taxonomy (Person/Topic --
            # both in SEED_ONTOLOGY, so build_taxonomy_data enumerates them and
            # class_index resolves them to taxonomy rows).
            handles["ontology"].write(json.dumps({
                "subgraph_id": sid,
                "labels": {
                    "suggested_edges": [{"child": "E:Alice", "parent": "Person"}],
                    "misclassified": [{"entity": "E:Bob", "suggested_class": "Topic"}],
                },
            }) + "\n")
            # Cluster is self-supervised (empty labels, 0 Oracle) -> the diffpool
            # head trains on its own entropy+cluster-link objective.
            handles["cluster"].write(json.dumps({
                "subgraph_id": sid, "labels": {"clusters": []},
            }) + "\n")
            # Anomaly: reproduce the deploy injection. types=["duplicate_episode"]
            # plants a clone when an episode is present (silently skipped otherwise
            # -- the head still trains on the all-zero true-negative target).
            sub = pipe.extract_subgraph(sid, radius=radius)
            enriched = enrich_subgraph(store, copy.deepcopy(sub))
            seed = 100 + i
            types = ["duplicate_episode"]
            corrupted, planted = inject_anomalies(enriched, seed=seed, types=types)
            handles["anomaly"].write(json.dumps({
                "subgraph_id": sid,
                "labels": {
                    "anomalies": detect_anomalies(corrupted),
                    "planted": planted,
                    "node_labels": node_label_vectors(corrupted),
                    "identity_drift": flag_identity_drift(corrupted),
                    "seed": seed, "types": types,
                },
            }) + "\n")
        with open(labels_dir / "quality_report.json", "w", encoding="utf-8") as f:
            json.dump({"radius": radius, "total_subgraphs": len(centers)}, f)
    finally:
        for h in handles.values():
            h.close()


def _toy_cfg(checkpoint_dir: Path, head: str = "all") -> GNNTrainConfig:
    return GNNTrainConfig(
        hidden_dim=32, num_heads=2, num_layers=2, epochs=2, lr=1e-3,
        device="cpu", dtype="float32", val_fraction=0.34, seed=0,
        checkpoint_dir=str(checkpoint_dir), head=head,
    )


def test_train_gnn_head_all_tiny(tmp_path):
    store = _seed_store(tmp_path)
    try:
        pipe = OracleLabelingPipeline(store)
        labels_dir = tmp_path / "labels"
        ckpt_dir = tmp_path / "ckpt"
        _write_labels(store, pipe, labels_dir, radius=2)

        summary = train_gnn(_toy_cfg(ckpt_dir, head="all"), store, labels_dir)

        # The joint run walked at least one trainable step (salience always has
        # node_scores here -> a loss is always computed).
        assert summary["steps"] > 0
        # all.pt + 5 per-head .pt, all carrying the same full state_dict.
        assert len(summary["checkpoints"]) == 6
        all_pt = Path(ckpt_dir / "all.pt")
        assert all_pt.exists()
        state = torch.load(all_pt, map_location="cpu", weights_only=True)
        fresh = GNNModel(hidden_dim=32, num_heads=2, num_layers=2,
                         predicate_vocab_size=32, num_clusters=16)
        fresh.load_state_dict(state)  # strict=True (the consolidation loader's contract)
        # final_val reports all 5 tracked heads (None where a head had no val data).
        assert set(summary["final_val"]) == {"salience", "diffpool", "link_prediction",
                                              "ontology", "anomaly"}
        # Salience had real labels -> a real val metric (not None).
        assert summary["final_val"]["salience"] is not None
        # Ontology: the two-encoder path resolves E:Alice->Person + E:Bob->Topic
        # against the seeded taxonomy (Person/Topic are SEED_ONTOLOGY classes, so
        # build_taxonomy_data enumerates them) -> pairs scoreable -> a REAL val
        # metric, not None (the bug this rework fixes: the old single-encoder head
        # resolved 0/23 -> untrained).
        assert summary["final_val"]["ontology"] is not None
        assert summary["head_steps"]["ontology"] > 0
        # The taxonomy encoder weights are in the checkpoint, and a fresh
        # GNNModel strict-loads it (the consolidation loader's contract).
        assert any(k.startswith("taxonomy.") for k in state)
        fresh2 = GNNModel(hidden_dim=32, num_heads=2, num_layers=2,
                          predicate_vocab_size=32, num_clusters=16)
        fresh2.load_state_dict(torch.load(all_pt, map_location="cpu", weights_only=True))
    finally:
        store.close()


def test_train_gnn_head_ontology_only(tmp_path):
    """``--head ontology`` trains the taxonomy encoder + the ontology head (the
    cheap CPU path to a trained ontology head without re-running the GPU backbone
    -- backbone cold-starts here since no --backbone-checkpoint)."""
    store = _seed_store(tmp_path)
    try:
        pipe = OracleLabelingPipeline(store)
        labels_dir = tmp_path / "labels"
        ckpt_dir = tmp_path / "ckpt_ont"
        _write_labels(store, pipe, labels_dir, radius=2)

        summary = train_gnn(_toy_cfg(ckpt_dir, head="ontology"), store, labels_dir)

        # Single-head run writes exactly one checkpoint (ontology.pt, NOT all.pt).
        assert len(summary["checkpoints"]) == 1
        assert Path(ckpt_dir / "ontology.pt").exists()
        # The ontology head + taxonomy encoder actually trained (steps > 0).
        assert summary["head_steps"]["ontology"] > 0
        # Pairs resolved against the seeded taxonomy -> a real val metric.
        assert summary["final_val"]["ontology"] is not None
        # The taxonomy encoder weights are in ontology.pt (single-head .pt still
        # carries the full state_dict, per the checkpoint contract).
        state = torch.load(ckpt_dir / "ontology.pt", map_location="cpu", weights_only=True)
        assert any(k.startswith("taxonomy.") for k in state)
        # Only the ontology head was tracked; other heads untouched.
        assert set(summary["final_val"]) == {"ontology"}
        assert summary["head_steps"]["salience"] == 0
    finally:
        store.close()


def test_train_gnn_head_ontology_with_backbone_checkpoint(tmp_path):
    """``--head ontology --backbone-checkpoint all.pt`` loads a backbone
    checkpoint, freezes input_proj + GAT layers, and refines the taxonomy encoder
    + ontology head on top. Exercises the strict=False backbone-load path (a
    pre-taxonomy GPU all.pt would load with only taxonomy.* keys missing)."""
    store = _seed_store(tmp_path)
    try:
        pipe = OracleLabelingPipeline(store)
        labels_dir = tmp_path / "labels"
        _write_labels(store, pipe, labels_dir, radius=2)

        # First: a joint run to produce a backbone checkpoint (all.pt carries the
        # full state_dict, including the taxonomy encoder).
        all_dir = tmp_path / "ckpt_all"
        train_gnn(_toy_cfg(all_dir, head="all"), store, labels_dir)
        all_pt = all_dir / "all.pt"
        assert all_pt.exists()

        # Then: --head ontology --backbone-checkpoint all.pt. Backbone frozen; the
        # taxonomy encoder + ontology head refine on top.
        ont_dir = tmp_path / "ckpt_ont_bb"
        cfg = _toy_cfg(ont_dir, head="ontology")
        cfg.backbone_checkpoint = str(all_pt)
        summary = train_gnn(cfg, store, labels_dir)

        assert Path(ont_dir / "ontology.pt").exists()
        assert summary["head_steps"]["ontology"] > 0
        assert summary["final_val"]["ontology"] is not None
    finally:
        store.close()


def test_train_gnn_head_salience_only(tmp_path):
    store = _seed_store(tmp_path)
    try:
        pipe = OracleLabelingPipeline(store)
        labels_dir = tmp_path / "labels"
        ckpt_dir = tmp_path / "ckpt_sal"
        _write_labels(store, pipe, labels_dir, radius=2)

        summary = train_gnn(_toy_cfg(ckpt_dir, head="salience"), store, labels_dir)

        # Single-head run writes exactly one checkpoint.
        assert len(summary["checkpoints"]) == 1
        assert Path(ckpt_dir / "salience.pt").exists()
        # Salience was trained; anomaly was never touched in a salience-only run.
        assert summary["head_steps"]["salience"] > 0
        assert summary["head_steps"]["anomaly"] == 0
    finally:
        store.close()


def test_training_feature_for_dup_clone_reuses_origin(tmp_path):
    """The anomaly head trains on a corrupted subgraph. An injected ``{orig}_dup``
    clone reuses the ORIGIN's real feature (keeps the duplication signal
    structural, not a cheap feature-divergence artefact); a synthetic ``M:`` node
    (no origin to mirror) degrades to its type-onehot via feature_for."""
    store = _seed_store(tmp_path)
    try:
        ff = training_feature_for(store)
        # ep_000001 IS in the store -> its real (hash-stub) feature; the _dup clone
        # maps to the same feature (clone -> origin).
        clone_kind, clone_vec = ff("ep_000001_dup")
        orig_kind, orig_vec = ff("ep_000001")
        assert clone_kind == orig_kind
        assert torch.allclose(clone_vec, orig_vec)
        # A synthetic M: node has no origin -> feature_for handles it directly
        # (onehot at the unknown kind slot; no _dup mirroring).
        m_kind, m_vec = ff("M:0001")
        assert m_kind == NODE_KIND_INDEX[infer_kind("M:0001")]
        assert m_vec[m_kind].item() == 1.0  # type-onehot stamped
    finally:
        store.close()