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