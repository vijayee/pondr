"""Tests for ``src/subconscious/consolidation_worker.py`` -- the background
gist-on-forgetting loop.

CPU, self-contained. The gister + fact_sink are injected seams (doubles), so no
Bonsai HTTP and no graph write. The thing under test -- ``ConsolidationWorker``
(the daemon thread + queue + foreground-busy gate + tick/drain lifecycle) and
its interaction with the REAL ``FadeMemory.consolidate`` -- is the real code.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from src.subconscious.consolidation_worker import ConsolidationWorker
from src.subconscious.fade import FadeConfig, FadeMemory, REGIME_FORGOTTEN
from src.subconscious.gister import StructuredGist

# Reuse the test_fade doubles (same doc-themed stub embedder + a fading helper).
from tests.test_fade import _StubEmbedder, _StubVoice, _fade_anchor_to_r4


# ----------------------------------------------------------- doubles
class _StubGisterByAnchor:
    """Gister that knows the anchor_id (so fail_on/raise_on can target it).
    The real gister.gist() does NOT take anchor_id; this stub stands in for the
    worker's (id, blurb, prior, count) call by tracking the id sequence."""

    def __init__(self, fail_on=None, raise_on=None) -> None:
        self.fail_on = fail_on
        self.raise_on = raise_on
        self.calls: list[tuple] = []
        self._n = 0

    def gist(self, blurb, prior_gist, count):
        self._n += 1
        return StructuredGist(
            narrative=f"gist-{self._n} of {blurb[:8]}",
            facts=[{"p": "has_state", "o": "v"}],
            state_assertions=[{"e": "a", "v": "1"}],
            consolidation_count=count,
        )


class _RecordingSink:
    """FactSink double: records write() calls."""

    def __init__(self) -> None:
        self.writes: list[tuple[int, list, list]] = []

    def write(self, anchor_id, facts, state_assertions):
        self.writes.append((anchor_id, list(facts), list(state_assertions)))


# --------------------------------------------------------------- helpers
def _mem_with_faded_anchor():
    """Build a FadeMemory + one anchor faded to R4, return (mem, aid)."""
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.20, ring_capacity=2)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0")
    _fade_anchor_to_r4(mem, aid)
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN
    return mem, aid


def _drain(worker, timeout=5.0):
    """drain() that asserts the thread joined."""
    ok = worker.drain(timeout=timeout)
    assert ok, "consolidation worker did not join in time"


# ----------------------------------------------------------------- tests
def test_tick_enqueues_and_thread_consolidates() -> None:
    # tick() enqueues fading anchors; the worker thread consolidates them when
    # the foreground gate is clear. After drain, the anchor is R1 + count=1.
    mem, aid = _mem_with_faded_anchor()
    gister = _StubGisterByAnchor()
    sink = _RecordingSink()
    worker = ConsolidationWorker(mem, gister, fact_sink=sink,
                                 epsilon=0.03, max_depth=3, max_per_tick=8)
    # Foreground busy -> tick enqueues (read-only) but the worker blocks.
    worker.foreground_busy.set()
    n = worker.tick()
    assert n >= 1
    # Give the thread a moment; it must NOT consolidate while busy.
    time.sleep(0.2)
    assert mem.consolidation_count(aid) == 0
    # Release the gate -> the worker consolidates.
    worker.foreground_busy.clear()
    _drain(worker)
    assert mem.consolidation_count(aid) == 1
    r = mem.recall_anchor(aid)
    assert r.regime != REGIME_FORGOTTEN   # rescued (R1 or R3)
    # fact_sink.write was called with the anchor's facts + state_assertions.
    assert any(w[0] == aid for w in sink.writes)
    w = [w for w in sink.writes if w[0] == aid][0]
    assert w[1] and w[2]


def test_fact_sink_not_called_when_gister_none() -> None:
    # When gist() returns None the worker skips: no consolidate, no sink write.
    mem, aid = _mem_with_faded_anchor()

    class _NoneGister:
        def gist(self, blurb, prior_gist, count):
            return None

    sink = _RecordingSink()
    worker = ConsolidationWorker(mem, _NoneGister(), fact_sink=sink,
                                 epsilon=0.03, max_depth=3, max_per_tick=8)
    worker.foreground_busy.set()
    worker.tick()
    worker.foreground_busy.clear()
    _drain(worker)
    assert mem.consolidation_count(aid) == 0
    assert sink.writes == []
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN  # still R4


def test_gister_raise_is_best_effort_skip() -> None:
    # A raising gist() must not kill the worker: the job is skipped, the queue
    # survives, drain still joins.
    mem, aid = _mem_with_faded_anchor()

    class _RaisingGister:
        def __init__(self):
            self.n = 0

        def gist(self, blurb, prior_gist, count):
            self.n += 1
            raise RuntimeError("boom")

    sink = _RecordingSink()
    worker = ConsolidationWorker(mem, _RaisingGister(), fact_sink=sink,
                                 epsilon=0.03, max_depth=3, max_per_tick=8)
    worker.foreground_busy.set()
    worker.tick()
    worker.foreground_busy.clear()
    _drain(worker)
    assert mem.consolidation_count(aid) == 0   # raise -> skip
    assert sink.writes == []
    # The thread is gone but the per-job raise was swallowed (no panic).


def test_drain_joins_thread() -> None:
    mem, _ = _mem_with_faded_anchor()
    worker = ConsolidationWorker(mem, _StubGisterByAnchor())
    assert worker.drain(timeout=5.0) is True
    assert not worker._thread.is_alive()
    # Double drain is a no-op (idempotent teardown).
    assert worker.drain(timeout=1.0) is True


def test_tick_read_only_during_foreground() -> None:
    # tick() is safe to call while foreground_busy is set (it only scans +
    # enqueues). It returns the count and does not mutate the memory.
    mem, aid = _mem_with_faded_anchor()
    worker = ConsolidationWorker(mem, _StubGisterByAnchor())
    worker.foreground_busy.set()
    before = mem.consolidation_count(aid)
    n = worker.tick()
    assert n >= 1
    # No consolidation happened during the busy window.
    assert mem.consolidation_count(aid) == before
    _drain(worker)


def test_tick_no_fading_anchors_returns_zero() -> None:
    # A fresh anchor (in ring, R1) is not fading -> tick enqueues nothing.
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=8)
    mem = FadeMemory(cfg, emb, _StubVoice())
    mem.ingest("docA:0")  # fresh, in ring
    worker = ConsolidationWorker(mem, _StubGisterByAnchor())
    assert worker.tick() == 0
    _drain(worker)


# -------------------------------------------------- validated-compaction doubles
class _ScriptedDecider:
    """Decider double with a scriptable ``verify_fidelity``.

    ``verdict`` is one of:
      - a single dict/None returned on every call,
      - a list consumed left-to-right (popped), or
      - a callable ``f(blurb, narrative) -> dict|None`` (lets a test isolate one
        anchor among the several ``_fade_anchor_to_r4`` enqueues).
    Records calls so tests assert the judge ran.
    """

    def __init__(self, verdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    def verify_fidelity(self, blurb, narrative):
        self.calls.append((blurb, narrative))
        if callable(self.verdict):
            return self.verdict(blurb, narrative)
        if isinstance(self.verdict, list):
            return self.verdict.pop(0) if self.verdict else None
        return self.verdict


def _corrupt_only(target_blurb, reason="swapped"):
    """Predicate verdict: flag only the target blurb as corrupt; everything else
    (the cross-doc ``zzN`` fade helpers) is clean so it consolidates silently and
    leaves the pending list. Isolates one deferred review for the resolve tests."""
    def _v(blurb, _narrative):
        if blurb == target_blurb:
            return {"corruption": True, "reason": reason}
        return {"corruption": False, "reason": ""}
    return _v


class _ValidateGister:
    """Gister double for the validate path: ``gist`` returns a real
    ``StructuredGist`` + exposes ``decider`` (the scripted judge) + ``shape``
    (records rebuilds for the resolve-edit assertion)."""

    def __init__(self, decider) -> None:
        self.decider = decider
        self.shape_calls: list[tuple] = []

    def gist(self, blurb, prior_gist, count):
        return StructuredGist(
            narrative=f"gist of {blurb[:8]}",
            facts=[{"p": "has_state", "o": "v"}],
            state_assertions=[],
            consolidation_count=count,
        )

    def shape(self, narrative, fact_source, count):
        self.shape_calls.append((narrative, fact_source, count))
        return StructuredGist(
            narrative=narrative, facts=[], state_assertions=[],
            consolidation_count=count,
        )


def _drain_validate(mem, decider_verdict, validate=True):
    """Drive a validate(-aware) worker through one tick + drain. ``decider_verdict``
    is handed to ``_ScriptedDecider`` (single value or list). Returns
    ``(worker, gister, decider)`` post-drain so the test can inspect state.

    Holds the foreground gate for the tick (so the worker blocks until we clear
    it), then releases + drains -- the same shape as the non-validate tests.
    """
    decider = _ScriptedDecider(decider_verdict)
    gister = _ValidateGister(decider)
    worker = ConsolidationWorker(
        mem, gister, fact_sink=_RecordingSink(),
        epsilon=0.03, max_depth=3, max_per_tick=8, validate=validate,
    )
    worker.foreground_busy.set()
    worker.tick()
    # Worker is blocked at the gate: no consolidation yet.
    time.sleep(0.2)
    worker.foreground_busy.clear()
    _drain(worker)
    return worker, gister, decider


# -- validate off is byte-identical to the unconditional path (regression) --
def test_validate_off_unconditional_accept() -> None:
    # validate=False: even a judge that WOULD flag corruption is never consulted;
    # the gist is applied unconditionally (today's behavior).
    mem, aid = _mem_with_faded_anchor()
    worker, gister, decider = _drain_validate(
        mem, {"corruption": True, "reason": "swapped"}, validate=False)
    assert mem.consolidation_count(aid) == 1          # consolidated anyway
    assert mem.recall_anchor(aid).regime != REGIME_FORGOTTEN
    assert decider.calls == []                         # judge never ran
    assert worker.pending_reviews() == []


# -- clean verdict: judge says not-corrupt -> apply silently --
def test_validate_clean_consolidates() -> None:
    mem, aid = _mem_with_faded_anchor()
    worker, _gister, decider = _drain_validate(
        mem, {"corruption": False, "reason": ""})
    assert mem.consolidation_count(aid) == 1
    assert mem.recall_anchor(aid).regime != REGIME_FORGOTTEN
    assert worker.pending_reviews() == []
    assert decider.calls                                # judge ran (>=1 call)


# -- corruption: judge flags meaning-change -> defer, no consolidate, R4 stays --
def test_validate_corrupt_defers() -> None:
    mem, aid = _mem_with_faded_anchor()
    target = mem.blurbs.text(aid)
    worker, _gister, _decider = _drain_validate(
        mem, _corrupt_only(target, "swapped entity"))
    assert mem.consolidation_count(aid) == 0          # NOT applied
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN  # still R4
    revs = worker.pending_reviews()
    assert len(revs) == 1                             # only the target deferred
    assert revs[0]["anchor_id"] == aid
    assert revs[0]["reason"] == "swapped entity"
    assert revs[0]["narrative"]                        # proposed gist present


# -- judge unavailable (HTTP/parse fail): defer, never auto-consolidate --
def test_validate_judge_none_defers_unvalidated() -> None:
    mem, aid = _mem_with_faded_anchor()
    target = mem.blurbs.text(aid)
    none_only = lambda blurb, _n: (None if blurb == target
                                  else {"corruption": False, "reason": ""})
    worker, _gister, _decider = _drain_validate(mem, none_only)
    assert mem.consolidation_count(aid) == 0
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN
    revs = worker.pending_reviews()
    assert len(revs) == 1                             # only the target held
    assert "unavailable" in revs[0]["reason"].lower()


# -- a pending anchor is not re-gisted by a later tick (skip to avoid dup) --
def test_validate_tick_skips_pending() -> None:
    mem, aid = _mem_with_faded_anchor()
    target = mem.blurbs.text(aid)
    worker, _gister, _decider = _drain_validate(
        mem, _corrupt_only(target, "x"))
    assert worker.pending_reviews()                  # target deferred this sweep
    # A second sweep: the pending target must be SKIPPED (not re-enqueued), so
    # it is never re-gisted + re-deferred into a duplicate review. Other fading
    # anchors may still enqueue this sweep and consolidate cleanly; the
    # invariant is the target is not duplicated in the pending list.
    worker.foreground_busy.set()
    worker.tick()
    worker.foreground_busy.clear()
    _drain(worker)
    revs = worker.pending_reviews()
    assert len(revs) == 1                             # not duplicated
    assert revs[0]["anchor_id"] == aid


# -- the foreground gate blocks the worker even with validate on --
def test_validate_foreground_gate_blocks_consolidation() -> None:
    mem, aid = _mem_with_faded_anchor()
    decider = _ScriptedDecider({"corruption": False, "reason": ""})
    gister = _ValidateGister(decider)
    worker = ConsolidationWorker(
        mem, gister, epsilon=0.03, max_depth=3, max_per_tick=8, validate=True)
    worker.foreground_busy.set()
    worker.tick()
    time.sleep(0.2)
    # Gate held: the worker parks in _wait_foreground; nothing consolidates and
    # nothing is deferred (it never got past the gate to the judge).
    assert mem.consolidation_count(aid) == 0
    assert worker.pending_reviews() == []
    assert decider.calls == []
    _drain(worker)


# ---- resolve() (foreground side; worker is blocked by the gate) ----
def _defer_then_block(mem, target_blurb):
    """Drive a corrupt defer of ``target_blurb`` only (the cross-doc fade
    helpers consolidate cleanly and leave the picture), then re-hold the
    foreground gate so the foreground-side ``resolve`` calls below run with the
    worker parked. Returns ``(worker, gister)`` with exactly one pending review
    (the target) at index 0."""
    worker, gister, _ = _drain_validate(mem, _corrupt_only(target_blurb, "swapped"))
    revs = worker.pending_reviews()
    assert len(revs) == 1                              # only the target deferred
    worker.foreground_busy.set()                      # park the worker
    return worker, gister


def test_resolve_accept_applies_gist() -> None:
    mem, aid = _mem_with_faded_anchor()
    worker, _gister = _defer_then_block(mem, mem.blurbs.text(aid))
    ok = worker.resolve(0, "accept")
    assert ok
    assert mem.consolidation_count(aid) == 1
    assert mem.recall_anchor(aid).regime != REGIME_FORGOTTEN  # R4 -> R1
    assert worker.pending_reviews() == []             # entry removed
    _drain(worker)


def test_resolve_keep_drops_review() -> None:
    mem, aid = _mem_with_faded_anchor()
    worker, _gister = _defer_then_block(mem, mem.blurbs.text(aid))
    ok = worker.resolve(0, "keep")
    assert ok
    assert mem.consolidation_count(aid) == 0          # NOT applied
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN  # still R4
    assert worker.pending_reviews() == []
    _drain(worker)


def test_resolve_edit_replaces_narrative() -> None:
    mem, aid = _mem_with_faded_anchor()
    worker, gister = _defer_then_block(mem, mem.blurbs.text(aid))
    edited = "corrected narrative: the entity was X, not Y."
    ok = worker.resolve(0, "edit", edited_narrative=edited)
    assert ok
    assert mem.consolidation_count(aid) == 1
    assert mem.recall_anchor(aid).regime != REGIME_FORGOTTEN
    # shape() was called with the user's edited text as both the narrative and
    # the fact source (re-extract from the corrected text, not the stale blurb).
    assert gister.shape_calls
    last = gister.shape_calls[-1]
    assert last[0] == edited and last[1] == edited
    # The applied blurb is the edited narrative.
    assert edited in mem.blurbs.text(aid)
    assert worker.pending_reviews() == []
    _drain(worker)


def test_resolve_edit_empty_is_noop() -> None:
    mem, aid = _mem_with_faded_anchor()
    worker, _gister = _defer_then_block(mem, mem.blurbs.text(aid))
    assert worker.resolve(0, "edit", edited_narrative="   ") is False
    # Nothing changed; the review is still pending, anchor still R4.
    assert mem.consolidation_count(aid) == 0
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN
    assert len(worker.pending_reviews()) == 1
    _drain(worker)


def test_resolve_stale_idx_is_false() -> None:
    mem, _aid = _mem_with_faded_anchor()
    worker, _gister = _defer_then_block(mem, mem.blurbs.text(_aid))
    assert worker.resolve(99, "accept") is False      # out of range
    assert worker.resolve(-1, "keep") is False
    assert len(worker.pending_reviews()) == 1         # untouched
    _drain(worker)


def test_resolve_unknown_action_is_false() -> None:
    mem, _aid = _mem_with_faded_anchor()
    worker, _gister = _defer_then_block(mem, mem.blurbs.text(_aid))
    assert worker.resolve(0, "discard") is False      # not a real action
    assert len(worker.pending_reviews()) == 1
    _drain(worker)