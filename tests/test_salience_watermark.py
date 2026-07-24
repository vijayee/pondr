"""STRM Phase 4 Step 6: freshness watermark + stale-uncertain consumer signal.

Each salient anchor carries an ``age`` (turns since its ``source_id`` first
entered the ring at salience-scoring time). For anchors YOUNGER than the
freshness lag (``strm_salience_freshness_lag``, turns), a retrieval that
returns nothing does NOT silently suppress the pointer -- the episode may be
known but not yet fully ingested by Thread 2's async-distill worker, so the
engine emits a typed ``stale_uncertain`` signal and the formatter surfaces a
stated gap ("I may know this but have not finished ingesting it"; proposal
sec 5: don't lie by omission). An OLD anchor that got nothing back is silently
dropped (it had its chance). A retrieval that returns hits emits ``recall``.

Flag-off -> ``salience_signals`` / ``salience_gap_text`` are ABSENT from the
result dict (byte-identical to pre-Step-6).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.latent_dynamics_head import load_latent_dynamics_head
from src.subconscious.recoverability_head import load_recoverability_head
from src.subconscious.relevance_head import load_relevance_head
from src.subconscious.salience import format_salience_gap

# Reuse the Step 5 harness helpers (permissive thresholds, stub build, preinject).
from tests.test_state_conditioned_retrieval import (
    _REC_CKPT, _LD_CKPT, _REL_CKPT, _HEADS_PRESENT, _thresh, _build, _preinject,
)


def _load_heads():
    return (
        load_recoverability_head(str(_REC_CKPT), device="cpu"),
        load_latent_dynamics_head(str(_LD_CKPT), device="cpu"),
        load_relevance_head(str(_REL_CKPT), device="cpu"),
    )


def _no_prompt_driven(orch):
    """Make the prompt-driven retrieve a no-op ([]) so the ONLY ring slots are
    the pre-injected ones -- isolates the watermark to a single anchor."""
    orch.retriever.retrieve = lambda *a, **k: []
    orch.retriever.retrieve_by_embedding = lambda *a, **k: []


# ── formatter ──

def test_format_salience_gap_stale_surfaces_text():
    """stale_uncertain signals -> a stated gap; recall-only -> empty (no lie)."""
    stale = [{"kind": "stale_uncertain", "text": "Alice chose Postgres",
              "anchor_source_id": "ep_old", "r_i": 0.9, "rec_i": -0.5, "age": 0}]
    gap = format_salience_gap(stale)
    assert "I may know this but have not finished ingesting it" in gap
    assert "Alice chose Postgres" in gap
    # recall-only -> no stale -> empty (do not fabricate a gap)
    recall = [{"kind": "recall", "text": "x", "anchor_source_id": "ep",
               "r_i": 0.9, "rec_i": -0.5, "age": 0}]
    assert format_salience_gap(recall) == ""
    # empty -> empty (byte-identical flag-off)
    assert format_salience_gap([]) == ""


def test_format_salience_gap_no_text_fallback():
    """A stale_uncertain signal with no text still surfaces the bare gap."""
    gap = format_salience_gap([{"kind": "stale_uncertain", "text": None,
                                "anchor_source_id": None, "r_i": None,
                                "rec_i": None, "age": 0}])
    assert gap == "I may know this but have not finished ingesting it."


# ── orchestrator signals ──

@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_flag_off_no_signal_keys_byte_identical(tmp_path):
    """Flag off -> salience_signals / salience_gap_text ABSENT (byte-identical
    result dict to pre-Step-6)."""
    orch, store = _build(tmp_path, strm_salience=False,
                        salience_thresholds=_thresh(), ring_capacity=16)
    try:
        res = orch.query("What did Alice say?")
        assert "salience_signals" not in res
        assert "salience_gap_text" not in res
        assert "salience_retrieval_count" not in res
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_young_anchor_failed_retrieval_emits_stale_uncertain(tmp_path, monkeypatch):
    """Young anchor (age < lag) + failed retrieval -> stale_uncertain signal
    (NOT silently suppressed). Default lag=3, first armed query -> age=0."""
    rec, ld, rel = _load_heads()
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])
    _no_prompt_driven(orch)  # failed retrieval + no prompt-driven injects
    try:
        res = orch.query("What did Alice say?")
        assert res["salience_retrieval_count"] == 0
        sigs = res["salience_signals"]
        assert len(sigs) == 1
        sig = sigs[0]
        assert sig["kind"] == "stale_uncertain"
        assert sig["anchor_source_id"] == "ep_old"
        assert sig["age"] == 0
        # the gap text surfaces the stated gap
        assert "I may know this but have not finished ingesting it" in res["salience_gap_text"]
        assert "Alice chose Postgres" in res["salience_gap_text"]
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_old_anchor_failed_retrieval_silently_dropped(tmp_path, monkeypatch):
    """Old anchor (age >= lag) + failed retrieval -> NO signal (silently
    dropped; it had its chance). lag=1: query 1 -> age=0 (young, stale_uncertain),
    query 2 -> age=1 (old, dropped)."""
    rec, ld, rel = _load_heads()
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])
    _no_prompt_driven(orch)
    # lag=1 so a second armed query ages the anchor past the watermark.
    import src.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod._runtime_config, "strm_salience_freshness_lag", 1)
    try:
        res1 = orch.query("What did Alice say?")
        # query 1: age=0 < 1 -> stale_uncertain
        assert [s["kind"] for s in res1["salience_signals"]] == ["stale_uncertain"]
        res2 = orch.query("What did Alice say?")
        # query 2: age=1 >= 1 -> old + failed -> silently dropped (no signals)
        assert res2["salience_signals"] == []
        assert res2["salience_gap_text"] == ""
        assert res2["salience_retrieval_count"] == 0
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_successful_retrieval_emits_recall_signal(tmp_path, monkeypatch):
    """A salient anchor whose retrieval returns hits -> recall signal (not
    stale_uncertain)."""
    rec, ld, rel = _load_heads()
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])
    # no prompt-driven injects; salience retrieval returns ONE hit.
    orch.retriever.retrieve = lambda *a, **k: []
    orch.retriever.retrieve_by_embedding = lambda *a, **k: [
        {"episode_id": "ep_recall", "summary": "Alice chose Postgres",
         "entities": [], "topics": [], "tones": [],
         "timestamp": "2026-07-01T10:00:00", "score": 0.4}]
    try:
        res = orch.query("What did Alice say?")
        assert res["salience_retrieval_count"] == 1
        sigs = res["salience_signals"]
        assert len(sigs) == 1
        assert sigs[0]["kind"] == "recall"
        assert sigs[0]["anchor_source_id"] == "ep_old"
        # recall -> no stale gap text
        assert res["salience_gap_text"] == ""
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_signal_shape_carries_scores_and_age(tmp_path, monkeypatch):
    """Each signal dict carries the full contract: anchor_source_id, kind, text,
    r_i, rec_i, age (r_i / rec_i are the head scores, not None for a scored
    salient anchor)."""
    rec, ld, rel = _load_heads()
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])
    _no_prompt_driven(orch)
    try:
        res = orch.query("What did Alice say?")
        sig = res["salience_signals"][0]
        for key in ("anchor_source_id", "kind", "text", "r_i", "rec_i", "age"):
            assert key in sig
        assert sig["text"] == "Alice chose Postgres for the audit log"
        assert sig["r_i"] is not None
        assert sig["rec_i"] is not None
        assert isinstance(sig["age"], int)
    finally:
        store.close()


# ── STRM Phase 5 IngestionTracker: in-flight short-circuit ─────────────────


class _StubDistillWorker:
    """Stand-in for DistillWorker exposing only the read API the salience hook
    uses (``snapshot_if_inflight``). Returns a COPY of the registered snapshot
    so the hook cannot mutate the live map (mirrors the real worker). Also
    carries a ``foreground_busy`` Event because the orchestrator sets/clears it
    at query() entry/exit regardless of which object holds the worker slot."""
    def __init__(self, snapshots):
        self._snapshots = dict(snapshots)
        self.foreground_busy = threading.Event()

    def snapshot_if_inflight(self, episode_id):
        if episode_id is None or episode_id not in self._snapshots:
            return None
        return dict(self._snapshots[episode_id])


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_inflight_anchor_short_circuits_to_recall_no_vector_roundtrip(tmp_path, monkeypatch):
    """An anchor whose source_id is an episode Thread 2 is still distilling is
    served straight from the in-flight stub snapshot -- NO vector round-trip
    (retrieve_by_embedding must NOT be called) and NO age heuristic -> the
    signal is ``recall`` (the cheap read Phase 5 wants), not ``stale_uncertain``.
    """
    rec, ld, rel = _load_heads()
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])
    orch.retriever.retrieve = lambda *a, **k: []  # no prompt-driven injects
    # retrieve_by_embedding MUST NOT be called -- raise if it is.
    def _boom(*a, **k):
        raise AssertionError("retrieve_by_embedding must not be called for an in-flight anchor")
    orch.retriever.retrieve_by_embedding = _boom
    # Stub the in-flight map: ep_old is mid-distill.
    orch._distill_worker = _StubDistillWorker({
        "ep_old": {"episode_id": "ep_old", "summary": "Alice chose Postgres",
                   "text": "Alice chose Postgres for the audit log",
                   "embed_text": "Alice chose Postgres", "summary_embedding": None},
    })
    try:
        res = orch.query("What did Alice say?")
        sigs = res["salience_signals"]
        assert len(sigs) == 1
        assert sigs[0]["kind"] == "recall"  # NOT stale_uncertain
        assert sigs[0]["anchor_source_id"] == "ep_old"
        # The in-flight snapshot fired -> one recalled episode, no stale gap.
        assert res["salience_retrieval_count"] == 1
        assert res["salience_gap_text"] == ""
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_inflight_shortcut_disabled_falls_through_to_retrieve(tmp_path, monkeypatch):
    """Rollback guard: with strm_salience_inflight_shortcut=False the hook is
    byte-identical to pre-Phase-5 -- retrieve_by_embedding IS called (even
    though the anchor is in-flight) and the vector result decides the kind."""
    rec, ld, rel = _load_heads()
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])
    orch.retriever.retrieve = lambda *a, **k: []  # no prompt-driven injects
    calls: list = []
    def _track(*a, **k):
        calls.append(a)
        return [{"episode_id": "ep_recall", "summary": "Alice chose Postgres",
                 "entities": [], "topics": [], "tones": [],
                 "timestamp": "2026-07-01T10:00:00", "score": 0.4}]
    orch.retriever.retrieve_by_embedding = _track
    # ep_old is in-flight, but the shortcut is OFF -> the hook must call the vector path.
    orch._distill_worker = _StubDistillWorker({
        "ep_old": {"episode_id": "ep_old", "summary": "Alice chose Postgres",
                   "text": "x", "embed_text": "Alice chose Postgres",
                   "summary_embedding": None},
    })
    import src.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod._runtime_config, "strm_salience_inflight_shortcut", False)
    try:
        res = orch.query("What did Alice say?")
        assert calls, "retrieve_by_embedding must be called when the shortcut is disabled"
        assert res["salience_signals"][0]["kind"] == "recall"  # via the vector hit
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_no_distill_worker_falls_through_to_retrieve(tmp_path, monkeypatch):
    """When async-distill is off (no worker -> self._distill_worker is None) the
    short-circuit is inert: the hook calls retrieve_by_embedding as before
    (byte-identical to pre-Phase-5). This is the production default-off-salience
    + the async-distill-off combination."""
    rec, ld, rel = _load_heads()
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])
    orch.retriever.retrieve = lambda *a, **k: []
    calls: list = []
    orch.retriever.retrieve_by_embedding = lambda *a, **k: (calls.append(a) or [])
    assert orch._distill_worker is None  # the _build harness wires no encoder/worker
    try:
        res = orch.query("What did Alice say?")
        assert calls, "retrieve_by_embedding must be called when no worker is wired"
        # no hits + young -> stale_uncertain (the pre-Phase-5 watermark path)
        assert res["salience_signals"][0]["kind"] == "stale_uncertain"
    finally:
        store.close()