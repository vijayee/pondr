"""Tests for the serve-path dream consolidation wiring.

Two surfaces:

* the APPLY-TIME EVAL GATE inside ``Consolidator.run`` -- refuses an apply that
  would write placeholder abstracts (no decider wired) or exceeds the
  ``apply_max_prunes`` / ``apply_max_abstracts`` blast-radius caps;
* the ``DreamWorker`` -- the required-trained-checkpoint serve scheduler
  (constructor refuses an absent checkpoint; ``run_once`` is dry-run by
  default; the foreground gate + drain lifecycle).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import ConsolidationConfig
from src.gnn.consolidate import Consolidator


@pytest.fixture()
def tiny_model():
    from src.gnn.model import GNNModel

    # Random weights, but a LOADED model object -- Consolidator.trained rides
    # on ``model is not None``, so a structural gate test does not need real
    # training; the worker path additionally requires a loadable checkpoint.
    return GNNModel(
        hidden_dim=128, num_heads=4, num_layers=3,
        predicate_vocab_size=32, num_clusters=16,
    )


def _store(tmp_path):
    from src.memory.episode import Episode
    from src.memory.store import HippocampalStore

    store = HippocampalStore(str(tmp_path / "db"))
    for i in range(1, 5):
        store.encode_episode(Episode(
            id=f"ep_00000{i}", timestamp="t", summary=f"s{i}",
            full_text=f"f{i}", entities=["Alice"], topics=["db"],
        ))
    return store


def _force_abstract_step(monkeypatch):
    """Patch the cluster step to PROPOSE one abstract (record-only by design)
    so gate tests exercise the apply branch without real model scores."""
    from src.gnn.consolidate import Consolidator

    def fake_step_cluster(self, data, out, report):
        # run() calls the step once per center -- propose ONE abstract total.
        if not report["abstracts"]:
            report["abstracts"].append(
                {"episodes": ["ep_000001", "ep_000002"], "score": 0.99})

    monkeypatch.setattr(Consolidator, "_step_cluster", fake_step_cluster)


class _StubDecider:
    def gist(self, source_eps):
        return "real gist"


# ── eval gate: unit (direct gate calls) ──


def test_eval_gate_refuses_abstracts_without_decider(tmp_path, tiny_model):
    store = _store(tmp_path)
    cons = Consolidator(store, model=tiny_model, dry_run=True)
    msg = cons._eval_apply_gate(
        {"abstracts": [{"episodes": ["ep_000001"]}], "pruned": []})
    assert msg is not None and "no decider" in msg and "placeholder" in msg
    store.close()


def test_eval_gate_allows_healthy_pass(tmp_path, tiny_model):
    store = _store(tmp_path)
    cons = Consolidator(store, model=tiny_model, dry_run=True,
                        decider=_StubDecider())
    msg = cons._eval_apply_gate(
        {"abstracts": [{"episodes": ["ep_000001"]}], "pruned": ["x"]})
    assert msg is None
    store.close()


def test_eval_gate_blast_radius_caps(tmp_path, tiny_model):
    store = _store(tmp_path)
    cfg = ConsolidationConfig(apply_max_prunes=4, apply_max_abstracts=1)
    cons = Consolidator(store, model=tiny_model, dry_run=True, config=cfg,
                        decider=_StubDecider())
    msg = cons._eval_apply_gate(
        {"abstracts": [{"episodes": ["ep_000001"]}], "pruned": list(range(5))})
    assert "apply_max_prunes" in msg
    msg2 = cons._eval_apply_gate(
        {"abstracts": [{"episodes": [f"ep_{i}"]} for i in range(1, 3)],
         "pruned": []})
    assert "apply_max_abstracts" in msg2
    store.close()


# ── eval gate: integration through run() ──


def test_run_refuses_gate_failed_apply(tmp_path, tiny_model, monkeypatch):
    store = _store(tmp_path)
    _force_abstract_step(monkeypatch)
    cons = Consolidator(store, model=tiny_model, dry_run=False,
                        decider=None, config=ConsolidationConfig())
    rep = cons.run(limit=1)
    assert rep["apply_skipped"].startswith("eval gate:")
    # The proposal is RECORDED but no M-node was written (no placeholder junk).
    mems = [k.split("/")[2] for k, _ in store.db.create_read_stream(
        start="content/mem/", end="content/mem/\x7f")]
    assert mems == []
    store.close()


def test_run_applies_gate_passing_pass_and_calls_decider(
        tmp_path, tiny_model, monkeypatch):
    store = _store(tmp_path)
    _force_abstract_step(monkeypatch)
    called = {}
    monkeypatch.setattr(
        "src.gnn.consolidate.Consolidator._apply",
        lambda self, report: called.setdefault("_apply", report))
    cons = Consolidator(store, model=tiny_model, dry_run=False,
                        decider=_StubDecider(), config=ConsolidationConfig())

    # Stub the lazy gist embedder (no model download in the test).
    class _StubEmb:
        def encode(self, texts):
            return [[0.1] * 384 for _ in texts]

    monkeypatch.setattr(
        Consolidator, "_embedder", property(lambda self: _StubEmb()))
    rep = cons.run(limit=1)
    assert "apply_skipped" not in rep
    assert called["_apply"] is rep
    store.close()


def test_run_no_apply_gate_escape(tmp_path, tiny_model, monkeypatch):
    """--no-apply-gate: the placeholder apply path runs even without a decider
    (a loud warning logs the junk; the store gets the placeholder M-node)."""
    from src.gnn.semantic_memory import SemanticMemoryWriter

    store = _store(tmp_path)
    _force_abstract_step(monkeypatch)
    cons = Consolidator(
        store, model=tiny_model, dry_run=False, decider=None,
        config=replace(ConsolidationConfig(), apply_gate_enabled=False))
    rep = cons.run(limit=1)
    assert "apply_skipped" not in rep
    mems = sorted({k.split("/")[2] for k, _ in store.db.create_read_stream(
        start="content/mem/", end="content/mem/\x7f")})
    # The placeholder M-node (graph-reachable only); the stream yields one
    # key per node FIELD, so count distinct ids.
    assert mems == ["M:0001"]
    assert store.is_abstracted("ep_000001")
    w = SemanticMemoryWriter(store)
    assert w.get_abstract("M:0001")["summary"].startswith("Abstract of [")
    store.close()


def test_untrained_apply_branch_unaffected_by_gate(
        tmp_path, monkeypatch):
    """The gate lives on the TRAINED branch ONLY -- the existing untrained
    + --force-untrained behavior (old tests) is byte-identical."""
    store = _store(tmp_path)
    _force_abstract_step(monkeypatch)
    called = {}
    monkeypatch.setattr(
        "src.gnn.consolidate.Consolidator._apply",
        lambda self, report: called.setdefault("yes", True))
    cons = Consolidator(store, dry_run=False, allow_untrained_apply=True)
    rep = cons.run(limit=1)
    assert rep["trained"] is False
    assert "apply_skipped" not in rep and called.get("yes") is True
    store.close()


# ── DreamWorker ──


def _saved_ckpt(tmp_path, tiny_model) -> str:
    import torch

    path = str(tmp_path / "dream.pt")
    torch.save(tiny_model.state_dict(), path)
    return path


def test_dream_worker_requires_checkpoint(tmp_path):
    from src.subconscious.dream_worker import DreamWorker

    store = _store(tmp_path)
    with pytest.raises(ValueError, match="dream checkpoint not found"):
        DreamWorker(store, checkpoint=str(tmp_path / "missing.pt"))
    store.close()


def test_dream_worker_run_once_dry_run_reports(tmp_path, tiny_model):
    from src.subconscious.dream_worker import DreamWorker

    store = _store(tmp_path)
    worker = DreamWorker(store, checkpoint=_saved_ckpt(tmp_path, tiny_model),
                         interval_s=86400.0)
    rep = worker.run_once()
    assert rep["dry_run"] is True and rep["trained"] is True
    assert worker.passes == 1 and worker.last_report is rep
    # Dry run: no mem nodes, nothing abstracted, no scheduler started.
    mems = [k.split("/")[2] for k, _ in store.db.create_read_stream(
        start="content/mem/", end="content/mem/\x7f")]
    assert mems == []
    assert worker._thread is None
    store.close()


def test_dream_worker_run_once_applies_between_turns(tmp_path, tiny_model,
                                                     monkeypatch):
    """apply=True + gate pass -> the pass mutates the store (a real trained
    checkpoint + a stub decider with a real gist -> no placeholder junk)."""
    from src.subconscious.dream_worker import DreamWorker

    class _GistDecider:
        def gist(self, source_eps):
            return "Alice and Bob discuss db"

    store = _store(tmp_path)
    monkeypatch.setattr(
        "src.gnn.consolidate.Consolidator._eval_apply_gate",
        lambda self, report: None)  # gate passes by fiat (caps fine here)
    monkeypatch.setattr(
        "src.gnn.consolidate.Consolidator._step_cluster",
        lambda self, data, out, report: report["abstracts"].append(
            {"episodes": ["ep_000001", "ep_000002"], "score": 0.99})
        if not report["abstracts"] else None)
    worker = DreamWorker(store, checkpoint=_saved_ckpt(tmp_path, tiny_model),
                         apply=True, decider=_GistDecider())
    rep = worker.run_once()
    assert rep["dry_run"] is False and "apply_skipped" not in rep
    assert rep["abstracts_applied"], "the applied abstract is the mutation"
    mems = sorted({k.split("/")[2] for k, _ in store.db.create_read_stream(
        start="content/mem/", end="content/mem/\x7f")})
    assert mems == ["M:0001"]
    assert store.is_abstracted("ep_000001")
    store.close()


def test_dream_worker_respects_foreground_and_stop(tmp_path, tiny_model):
    """Foreground busy + stop -> ``run_once`` returns None without scoring
    (the pass waits out queries, and drain wins the race)."""
    from src.subconscious.dream_worker import DreamWorker

    store = _store(tmp_path)
    worker = DreamWorker(store, checkpoint=_saved_ckpt(tmp_path, tiny_model))
    worker.foreground_busy.set()
    worker._stop.set()
    assert worker.run_once() is None
    assert worker.last_report is None and worker.passes == 0
    store.close()


def test_dream_worker_drain_joins(tmp_path, tiny_model):
    import time as _time

    from src.subconscious.dream_worker import DreamWorker

    store = _store(tmp_path)
    worker = DreamWorker(store, checkpoint=_saved_ckpt(tmp_path, tiny_model),
                         interval_s=1.0)
    worker.start()
    _time.sleep(0.1)
    assert worker._thread is not None and worker._thread.is_alive()
    assert worker.drain(timeout=2.0) is True
    assert not worker._thread.is_alive()
    # Idempotent drain.
    assert worker.drain(timeout=2.0) is True
    store.close()