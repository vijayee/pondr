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