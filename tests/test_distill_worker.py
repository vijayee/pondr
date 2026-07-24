"""Contract tests for the background distill worker (Phase 3c async-distill).

Offline: no GLiNER, no live Bonsai. A pair of fakes (`_FakeEncoder` +
`_FakeStore`) exercise the worker's four load-bearing contracts:

  #8  failure-isolation: a fill that raises is logged + the queue survives
      (the next enqueued episode still gets its edges).
  #9  drain: finishes in-flight + queued encodes, joins the thread, returns.
  #10 foreground-priority yield: while ``foreground_busy`` is set, the worker's
      ``pause_gate`` blocks the fill (extraction does not run); clearing it
      releases the fill. Extraction runs only in the gaps between turns.
  #11 thread-isolation of the heavy work: the stub build runs on the main
      thread, the fill (extraction + edge write) runs on the worker thread --
      so the response returns before the 22 s extraction cost is paid.

The fakes mirror the real encoder's ``pause_gate`` contract: ``encode_messages_fill``
checks ``self.pause_gate`` at entry (the real encoder checks it inside ``_extract``
+ each isolated Bonsai call) so the yield behavior is exercised without GPU.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from src.encoding.distill_worker import DistillWorker


# ── fakes ──────────────────────────────────────────────────────────────────


class _FakeBonsai:
    """Stand-in for the encoder's Bonsai extractor; carries a pause_gate."""
    pause_gate = None


class _FakeEncoder:
    """Stand-in for HippocampalEncoder with the stub/fill shape + a gate.

    Records which thread ran each method so the thread-isolation contract (#11)
    is assertable, and supports ``fail_on_eid`` (#8) + ``fill_sleep`` (#9/#10)
    to exercise failure + timing without GPU.
    """

    def __init__(self, *, fail_on_eid=None, fill_sleep=0.0):
        self.pause_gate = None
        self.bonsai = _FakeBonsai()
        self.last_episode_id = None
        self.fail_on_eid = fail_on_eid
        self.fill_sleep = fill_sleep
        self.stub_calls: list[tuple[str, int]] = []   # (eid, thread_id)
        self.fill_calls: list[tuple[str, int]] = []   # (eid, thread_id)
        self.fill_started = threading.Event()
        self.fill_finished = threading.Event()

    def encode_messages_stub(self, messages, episode_id, **_kw) -> SimpleNamespace:
        self.stub_calls.append((episode_id, threading.get_ident()))
        ep = SimpleNamespace(
            id=episode_id, full_text="User: q\nAssistant: a",
            entities=[], topics=[], tones=[], decisions=[], relations=[],
            state_assertions=[], filled=False, edges_written=False,
        )
        return ep

    def encode_messages_fill(self, episode, episode_id, *, degrade_on_extract_fail=True):
        # Mirror the real encoder: yield to the foreground before the GPU step.
        # The worker installs ``pause_gate = _wait_foreground`` before calling
        # this, so a set ``foreground_busy`` blocks here.
        if self.pause_gate is not None:
            self.pause_gate()
        self.fill_started.set()
        if self.fill_sleep:
            time.sleep(self.fill_sleep)
        if self.fail_on_eid is not None and episode_id == self.fail_on_eid:
            raise RuntimeError(f"simulated fill failure for {episode_id}")
        episode.entities = ["E1"]
        episode.relations = [{"subject": "x", "predicate": "decides", "object": "y"}]
        episode.filled = True
        self.fill_calls.append((episode_id, threading.get_ident()))
        self.fill_finished.set()


class _FakeStore:
    """Stand-in for HippocampalStore: records content + edge writes per eid."""
    def __init__(self):
        self.content_writes: set[str] = set()
        self.edge_writes: set[str] = set()
        self._counter = 0
        self._lock = threading.Lock()

    def next_episode_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"E:{self._counter}"

    def encode_episode_content(self, episode_id, episode) -> None:
        self.content_writes.add(episode_id)

    def encode_episode_edges(self, episode_id, episode) -> None:
        self.edge_writes.add(episode_id)
        episode.edges_written = True


def _enqueue_ep(worker, store, encoder, eid=None):
    """Simulate the orchestrator's async persist: stub on main thread, content
    write on main thread, enqueue for the worker to fill."""
    if eid is None:
        eid = store.next_episode_id()
    ep = encoder.encode_messages_stub([], eid, origin="live")
    store.encode_episode_content(eid, ep)
    encoder.last_episode_id = eid
    worker.enqueue(ep, eid)
    return ep


# ── #8 failure-isolation ───────────────────────────────────────────────────


def test_fill_failure_is_logged_and_queue_survives():
    """A fill that raises (Bonsai HTTP 500 / GLiNER hiccup) is logged and the
    queue survives: the stub keeps the failed episode vector-retrievable, and
    the NEXT enqueued episode still gets its edges. One hiccup does not kill
    the worker thread."""
    encoder = _FakeEncoder(fail_on_eid="E:1")
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        ep1 = _enqueue_ep(worker, store, encoder, eid="E:1")
        ep2 = _enqueue_ep(worker, store, encoder, eid="E:2")
        # Drain so both fills have been attempted before asserting.
        assert worker.drain(timeout=5.0), "worker thread did not join"
        # ep1's fill raised -> no edges, but its stub content was written.
        assert "E:1" in store.content_writes
        assert ep1.edges_written is False
        assert ep1 not in (None,)
        # The queue survived: ep2 got its edges despite ep1's failure.
        assert "E:2" in store.edge_writes
        assert ep2.edges_written is True
        assert ep2.filled is True
    finally:
        worker.drain(timeout=5.0)


# ── #9 drain ───────────────────────────────────────────────────────────────


def test_drain_finishes_queued_work_and_joins_thread():
    """drain() finishes in-flight + queued encodes and joins the worker thread
    within the timeout. A slow fill queued just before drain still completes."""
    encoder = _FakeEncoder(fill_sleep=0.3)
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        ep = _enqueue_ep(worker, store, encoder, eid="E:1")
        joined = worker.drain(timeout=5.0)
        assert joined, "worker thread did not join within timeout"
        assert not worker._thread.is_alive()
        # The queued slow fill completed before the thread exited.
        assert ep.edges_written is True
        assert "E:1" in store.edge_writes
    finally:
        worker.drain(timeout=5.0)


def test_drain_is_noop_without_worker_path():
    """drain() on an already-stopped worker is a no-op that returns True
    (idempotent teardown -- safe to call twice)."""
    encoder = _FakeEncoder()
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    assert worker.drain(timeout=5.0) is True
    # Second drain must not raise and must report joined.
    assert worker.drain(timeout=5.0) is True


# ── #10 foreground-priority yield ──────────────────────────────────────────


def test_foreground_busy_blocks_fill_until_cleared():
    """While ``foreground_busy`` is set, the worker's pause_gate blocks the
    fill -- extraction does NOT run. Clearing it releases the fill. This is
    the foreground-priority contract: the 22 s extraction runs only in the
    gaps between turns, never contending with a foreground query() for the
    one 8B on :8080."""
    encoder = _FakeEncoder(fill_sleep=0.0)
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        # Hold the foreground busy -- simulates query() in flight.
        worker.foreground_busy.set()
        ep = _enqueue_ep(worker, store, encoder, eid="E:1")
        # Give the worker a moment to pick up the item + hit the gate.
        time.sleep(0.3)
        assert not ep.edges_written, (
            "fill ran while foreground_busy was set -- pause_gate did not block"
        )
        assert not encoder.fill_finished.is_set()
        # Release the foreground -- the fill should now proceed.
        worker.foreground_busy.clear()
        assert worker.drain(timeout=5.0), "worker did not join after clear"
        assert ep.edges_written is True
        assert "E:1" in store.edge_writes
    finally:
        worker.foreground_busy.clear()
        worker.drain(timeout=5.0)


def test_pause_gate_installed_and_cleared_around_fill():
    """The worker installs ``pause_gate`` on the encoder (and its Bonsai
    extractor) before the fill and clears it after, so a synchronous path
    reusing the encoder stays byte-identical (no stray blocking callable)."""
    encoder = _FakeEncoder()
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        _enqueue_ep(worker, store, encoder, eid="E:1")
        # Wait for the fill to run.
        encoder.fill_finished.wait(timeout=5.0)
        # Drain so the finally-block clears the gate.
        worker.drain(timeout=5.0)
        # After the fill + drain, the gate is cleared back to None on both the
        # encoder and its Bonsai extractor.
        assert encoder.pause_gate is None
        assert encoder.bonsai.pause_gate is None
    finally:
        worker.drain(timeout=5.0)


# ── #11 thread-isolation of the heavy work ─────────────────────────────────


def test_stub_runs_on_main_fill_runs_on_worker():
    """The stub build (cheap, ms) runs on the MAIN thread; the fill (the 22 s
    GLiNER + 10-pass Bonsai) runs on the WORKER thread. This is the core
    async-distill contract: the response returns immediately because the heavy
    work is off the synchronous path."""
    encoder = _FakeEncoder()
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    main_thread = threading.get_ident()
    try:
        _enqueue_ep(worker, store, encoder, eid="E:1")
        # The stub already ran on the main thread (synchronous, before enqueue).
        assert encoder.stub_calls, "stub was not called on the main thread"
        assert encoder.stub_calls[0][1] == main_thread, (
            "encode_messages_stub must run on the main thread"
        )
        # Wait for the worker to run the fill.
        assert encoder.fill_finished.wait(timeout=5.0), "fill did not run"
        assert encoder.fill_calls, "fill was not called on the worker thread"
        assert encoder.fill_calls[0][1] != main_thread, (
            "encode_messages_fill must run on the worker thread, not main"
        )
    finally:
        worker.drain(timeout=5.0)


# ── #12 in-flight stub snapshot (STRM Phase 5 IngestionTracker) ────────────


def _snapshot_ep(eid="E:1"):
    """An episode carrying the fill-immutable fields the snapshot reads
    (summary/full_text/summary_embedding). The real encoder's stub builds an
    Episode with these; SimpleNamespace mirrors the attribute shape."""
    return SimpleNamespace(
        id=eid, summary="Alice chose Postgres", full_text="User: q\nAssistant: a",
        summary_embedding=[0.1, 0.2, 0.3, 0.4],
    )


def test_enqueue_snapshots_inflight_fields():
    """enqueue() snapshots the fill-immutable stub fields into the in-flight
    map. snapshot_if_inflight returns a copy carrying exactly the keys the
    orchestrator merge (episode_id) + inject (embed_text/summary/text) read."""
    encoder = _FakeEncoder()
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        ep = _snapshot_ep("E:1")
        worker.enqueue(ep, "E:1")
        snap = worker.snapshot_if_inflight("E:1")
        assert snap is not None, "in-flight snapshot missing right after enqueue"
        assert snap["episode_id"] == "E:1"
        assert snap["summary"] == "Alice chose Postgres"
        assert snap["text"] == "User: q\nAssistant: a"
        assert snap["embed_text"] == "Alice chose Postgres"  # inject reads this first
        assert snap["summary_embedding"] == [0.1, 0.2, 0.3, 0.4]
    finally:
        worker.drain(timeout=5.0)


def test_snapshot_is_a_copy_not_live():
    """snapshot_if_inflight returns a COPY -- mutating it must not touch the
    live map (the foreground must never read the object the worker mid-mutates
    during fill, and the caller must not corrupt the worker's snapshot)."""
    encoder = _FakeEncoder(fill_sleep=0.0)
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        worker.foreground_busy.set()  # hold the fill so the snapshot stays live
        worker.enqueue(_snapshot_ep("E:1"), "E:1")
        snap = worker.snapshot_if_inflight("E:1")
        snap["summary"] = "MUTATED"
        snap["episode_id"] = "MUTATED"
        # Re-read: the live snapshot is untouched.
        again = worker.snapshot_if_inflight("E:1")
        assert again["summary"] == "Alice chose Postgres"
        assert again["episode_id"] == "E:1"
    finally:
        worker.foreground_busy.clear()
        worker.drain(timeout=5.0)


def test_snapshot_drained_after_fill_completes():
    """Once the worker finishes the fill, the episode is no longer in flight --
    snapshot_if_inflight returns None (queue membership was the ground-truth
    in-flight signal)."""
    encoder = _FakeEncoder(fill_sleep=0.3)
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        worker.enqueue(_snapshot_ep("E:1"), "E:1")
        # While the fill is in flight, the snapshot is present.
        encoder.fill_started.wait(timeout=5.0)
        assert worker.snapshot_if_inflight("E:1") is not None, (
            "snapshot missing while the fill is still in flight"
        )
        # After the fill completes + drain, the snapshot is gone.
        assert worker.drain(timeout=5.0), "worker thread did not join"
        assert worker.snapshot_if_inflight("E:1") is None, (
            "snapshot survived the fill -- not drained in _run's finally"
        )
    finally:
        worker.drain(timeout=5.0)


def test_snapshot_popped_on_fill_failure():
    """A failed fill still pops the in-flight snapshot (in the finally, not the
    try) -- the episode stays a vector-retrievable stub, so the salience hook's
    retrieve_by_embedding fall-through finds it; it must not stay 'in flight'
    forever (which would short-circuit retrieval on a stub that is already
    vector-retrievable)."""
    encoder = _FakeEncoder(fail_on_eid="E:1")
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        worker.enqueue(_snapshot_ep("E:1"), "E:1")
        assert worker.drain(timeout=5.0), "worker thread did not join"
        # The fill raised, but the finally still popped the snapshot.
        assert worker.snapshot_if_inflight("E:1") is None, (
            "snapshot survived a fill failure -- pop not in finally"
        )
    finally:
        worker.drain(timeout=5.0)


def test_snapshot_none_for_unknown_and_none_id():
    """snapshot_if_inflight returns None for an unknown id AND for a None id
    (an anchor with no provenance) -- the hook falls through to
    retrieve_by_embedding (byte-identical to pre-Phase-5)."""
    encoder = _FakeEncoder()
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        worker.enqueue(_snapshot_ep("E:1"), "E:1")
        assert worker.snapshot_if_inflight("E:unknown") is None
        assert worker.snapshot_if_inflight(None) is None
    finally:
        worker.drain(timeout=5.0)


def test_drain_clears_inflight_map():
    """drain() clears the in-flight map so abandoned in-flight episodes (their
    fill was dropped by the drain) do not leave stale snapshots."""
    encoder = _FakeEncoder()
    store = _FakeStore()
    worker = DistillWorker(encoder, store)
    try:
        worker.foreground_busy.set()  # hold the fill so the entries stay in flight
        worker.enqueue(_snapshot_ep("E:1"), "E:1")
        worker.enqueue(_snapshot_ep("E:2"), "E:2")
        assert worker.snapshot_if_inflight("E:1") is not None
        assert worker.snapshot_if_inflight("E:2") is not None
    finally:
        # drain clears the map even though the fills never ran.
        worker.drain(timeout=5.0)
        assert worker.snapshot_if_inflight("E:1") is None
        assert worker.snapshot_if_inflight("E:2") is None