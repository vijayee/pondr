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
        fill (extraction + edge write)."""
        self._q.put((episode, episode_id))

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
        # Wake a blocked _wait_foreground so the worker can exit.
        self.foreground_busy.clear()
        self._q.put(_DISTILL_SENTINEL)
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()