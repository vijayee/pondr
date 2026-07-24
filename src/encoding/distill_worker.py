"""Background episode distillation worker (Phase 3c async-distill).

Moves the ~22 s extraction (GLiNER + 10-pass isolated Bonsai) off the
synchronous ``query()`` path so the response returns immediately while the
graph distillation runs in the background.

Design (see .claude/plans/async-distill-stub.md):

  main thread (query):  id = store.next_episode_id()
                        episode = encoder.encode_messages_stub(messages, id, ...)
                        store.encode_episode_content(id, episode)   # stub: content + vector
                        encoder.last_episode_id = id
                        enqueue(id, episode)  ->  return response
  worker (this class):  episode = encoder.encode_messages_fill(episode, id)  # GLiNER + Bonsai
                        store.encode_episode_edges(id, episode)              # fill: graph edges

Single-thread FIFO (``queue.Queue`` + one daemon thread) so encodes never run
concurrently -- avoids encoder-state races and keeps WaveDB writes serialized.
Turns queue if the user out-types 22 s/turn; the queue drains; the user never
blocks.

Foreground-priority yielding: the worker sets ``pause_gate`` on the encoder +
its Bonsai extractor before the fill, and the gate (``_wait_foreground``)
blocks while ``foreground_busy`` is set by ``query()``. Extraction therefore
runs only in the GAPS between turns, so a fast user never contends with the
background for the one 8B on :8080. Residual TOCTOU: the check-then-call
window means at most ONE in-flight extraction call (~2.3 s) can contend with a
foreground query that lands just after a gate check -- the deliberate trade
(full mutual exclusion would make the foreground block on the worker's
in-flight call).

Failure semantics: a worker exception (Bonsai HTTP 500, GLiNER hiccup, store
write error) is logged and the episode keeps its stub (content + embedding, no
edges) -- vector-retrievable, just graph-thin. The queue survives; the next
turn still encodes. This mirrors ``_persist_exchange``'s "never lose the
response" philosophy: memory is best-effort, not transactional.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Callable, Optional


_DISTILL_SENTINEL = object()  # signals the worker thread to exit (drain)


class DistillWorker:
    """A single-worker background FIFO that fills stub episodes' graph edges.

    Constructed lazily by the orchestrator when ``config.async_distill_enabled``
    is on AND an encoder is wired. The orchestrator owns the
    ``foreground_busy`` event lifecycle: ``set()`` at ``query()`` entry,
    ``clear()`` at return. The worker reads it via ``_wait_foreground``.
    """

    def __init__(self, encoder, store) -> None:
        self._encoder = encoder
        self._store = store
        self._q: "queue.Queue[tuple[object, str]]" = queue.Queue()
        # Set by the orchestrator while a foreground query() is building a
        # response; the worker's pause_gate blocks on this so extraction runs
        # only between turns.
        self.foreground_busy = threading.Event()  # not set by default
        # In-flight stub snapshots (STRM Phase 5 IngestionTracker). Keyed by
        # episode_id, populated at enqueue, drained in _run's finally. The
        # salience hook reads it (snapshot_if_inflight) so an anchor whose
        # source_id is an episode Thread 2 is still distilling short-circuits
        # the vector round-trip -- the cheap read Phase 5 wants. Touched from
        # three threads (enqueue on main, pop in the worker's finally, read
        # in the salience hook on main) so a real Lock, not an Event.
        self._inflight: dict[str, dict] = {}
        self._inflight_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="ponder-distill-worker", daemon=True
        )
        self._stopped = False
        self._thread.start()

    # ── foreground-priority yielding ──

    def _wait_foreground(self) -> None:
        """The pause_gate callable: block while a foreground query() is busy.

        Called before each GPU-using step (GLiNER _extract + each isolated
        Bonsai call) via ``encoder.pause_gate`` / ``bonsai.pause_gate``. Blocks
        until the foreground clears ``foreground_busy`` (with a short poll so a
        shutdown is responsive). Returns immediately when not busy.
        """
        while self.foreground_busy.is_set() and not self._stopped:
            self.foreground_busy.wait(timeout=0.5)

    # ── the worker loop ──

    def enqueue(self, episode, episode_id: str) -> None:
        """Hand a stub episode + its pre-allocated id to the worker for the
        fill (extraction + edge write).

        Also snapshots the fill-immutable stub fields into the in-flight map
        (STRM Phase 5 IngestionTracker). The fill only reads ``full_text`` and
        never writes ``summary``/``full_text``/``summary_embedding``
        (encoder.py:402-421), so these fields are stable -- but the snapshot is
        a plain dict taken here (before the worker touches the episode) so the
        foreground's salience hook never reads the live object the worker is
        mid-mutating. The snapshot keys are exactly what the orchestrator's
        merge (episode_id) + inject (embed_text/summary/text) consume -- the
        snapshot has NO graph edges, so it serves only the vector-gist
        re-inject path, not graph-traversal retrieval. Drained in _run's
        finally (success OR failure: a failed fill leaves a vector-retrievable
        stub, so the hook's retrieve_by_embedding fall-through finds it).
        """
        snap = {
            "episode_id": episode_id,
            "summary": getattr(episode, "summary", "") or "",
            "text": getattr(episode, "full_text", "") or getattr(episode, "summary", "") or "",
            # The inject loop reads ``embed_text`` first (orchestrator.py:672);
            # fall back to ``summary`` so the re-embed uses the episode's gist.
            "embed_text": getattr(episode, "summary", "") or "",
            "summary_embedding": getattr(episode, "summary_embedding", None),
        }
        with self._inflight_lock:
            self._inflight[episode_id] = snap
        self._q.put((episode, episode_id))

    def snapshot_if_inflight(self, episode_id: Optional[str]) -> Optional[dict]:
        """Return a COPY of the in-flight stub snapshot for ``episode_id``, or
        ``None`` if it is not in flight (already distilled, or never enqueued).

        Called by the orchestrator's salience hook on the main thread. Returns
        a copy so the caller cannot mutate the live snapshot the worker may
        still be reading ``full_text`` from during the fill. ``None`` for a
        ``None`` id (an anchor with no provenance) -> the hook falls through to
        ``retrieve_by_embedding`` (byte-identical to pre-Phase-5).
        """
        if episode_id is None:
            return None
        with self._inflight_lock:
            snap = self._inflight.get(episode_id)
            return dict(snap) if snap is not None else None

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _DISTILL_SENTINEL:
                self._q.task_done()
                return
            episode, episode_id = item
            try:
                # Install the foreground-priority gate on the GPU-using steps.
                # The fill calls encoder._extract (GLiNER) and bonsai.extract
                # -> extract_isolated (10 Bonsai calls); both check the gate
                # before each GPU call and block while the foreground is busy.
                self._encoder.pause_gate = self._wait_foreground
                bonsai = getattr(self._encoder, "bonsai", None)
                if bonsai is not None:
                    bonsai.pause_gate = self._wait_foreground
                try:
                    self._encoder.encode_messages_fill(episode, episode_id)
                    self._store.encode_episode_edges(episode_id, episode)
                finally:
                    # Clear the gate so any synchronous path reusing the
                    # encoder stays byte-identical (no stray blocking).
                    self._encoder.pause_gate = None
                    if bonsai is not None:
                        bonsai.pause_gate = None
            except Exception as e:  # noqa: BLE001 - never kill the queue
                # Best-effort memory: the stub (content + embedding) is already
                # stored; a fill failure leaves the episode graph-thin but
                # vector-retrievable. Log and keep draining.
                print(f"[distill-fail] {episode_id}: {e}", file=sys.stderr)
            finally:
                # Drain the in-flight snapshot (success OR failure). On
                # failure the episode stays a vector-retrievable stub, so the
                # salience hook's retrieve_by_embedding fall-through finds it;
                # on success it is fully distilled. Either way it is no longer
                # "in flight" -- queue membership was the ground-truth signal.
                with self._inflight_lock:
                    self._inflight.pop(episode_id, None)
                self._q.task_done()

    # ── teardown ──

    def drain(self, timeout: Optional[float] = 5.0) -> bool:
        """Stop accepting new work, finish in-flight + queued encodes, join the
        worker thread. Returns True if the thread joined within ``timeout``.

        Best-effort: a Ctrl-C / hard exit may lose in-flight encodes (the stub
        keeps the turn vector-retrievable). Called from the orchestrator's
        teardown hook.
        """
        if self._stopped:
            return True
        self._stopped = True
        # Drop any in-flight snapshots (teardown -- no salience will run again,
        # but do not leave stale entries pointing at episodes whose fill was
        # abandoned by the drain).
        with self._inflight_lock:
            self._inflight.clear()
        # Wake a blocked _wait_foreground so the worker can exit.
        self.foreground_busy.clear()
        self._q.put(_DISTILL_SENTINEL)
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()