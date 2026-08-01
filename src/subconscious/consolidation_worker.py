"""Background consolidation worker for the gist-on-forgetting loop.

The fade's R3->R4 boundary is where a memory "reaches the point where it can't be
recalled." Instead of dropping to ``[forgotten]`` and losing it, this worker
gists the anchor IN PLACE (``FadeMemory.consolidate``): a structured gist
(narrative + extracted facts) replaces the verbatim blurb, SSM-A is re-stepped
with ``bge(narrative)``, and the anchor jumps R4 -> R1 -- recallable again, at a
compressed level. On later sweeps the gist is compressed further (gist-of-gist,
prior-baseline-merge preserving fidelity); at ``max_depth`` the anchor stays R4
(the real forgotten -> long-term-pull floor).

Modeled on ``DistillWorker`` (the only async pattern the serve path has): a
single daemon ``threading.Thread`` + ``queue.Queue`` + a ``foreground_busy``
``threading.Event`` priority gate. The orchestrator calls ``tick()`` at the tail
of each query (read-only -- it scans ``fading_anchors`` and enqueues ids); the
worker thread processes the queue BETWEEN turns (the gate blocks while
``foreground_busy`` is set). The gate is the race fix: ``consolidate`` MUTATES
``ssm_a`` + ``blurbs`` (which Seam-2 recall reads), so consolidation must never
run mid-recall -- it runs only in the gaps between queries.

Failure semantics (mirror ``DistillWorker``): a per-job exception (Bonsai HTTP
fail, parse fail) is logged and the anchor is skipped -- it stays R4 and is
retried next sweep. The queue survives; the next turn still consolidates.
Cold-start honest: a ``None`` gist (Bonsai down) is a skip, not a fabricated
gist.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Optional, Protocol

from .fade import FadeMemory
from .gister import BonsaiGister

_CONSOLIDATE_SENTINEL = object()  # signals the worker thread to exit (drain)


class FactSink(Protocol):
    """Optional consumer of a consolidated anchor's extracted facts (the
    R4 -> long-term-memory pull hook). ``write`` is best-effort; a raise is
    logged and swallowed (the consolidation itself has already succeeded)."""

    def write(self, anchor_id: int, facts: list[dict],
              state_assertions: list[dict]) -> None: ...


class ConsolidationWorker:
    """A single-worker background FIFO that gists fading fade anchors.

    Constructed by ``build_ponder`` when ``fade_consolidation`` is on. The
    orchestrator owns the ``foreground_busy`` event lifecycle (``set()`` at
    ``query()`` entry, ``clear()`` at return) -- the SAME event the
    ``DistillWorker`` uses, so both background workers yield to the foreground
    together. The worker reads it via ``_wait_foreground`` before each
    consolidation (the mutation step).
    """

    def __init__(self, fade_mem: FadeMemory, gister: BonsaiGister,
                 fact_sink: Optional[FactSink] = None,
                 epsilon: float = 0.03, max_depth: int = 3,
                 max_per_tick: int = 8) -> None:
        self._fade = fade_mem
        self._gister = gister
        self._fact_sink = fact_sink
        self.epsilon = float(epsilon)
        self.max_depth = int(max_depth)
        self.max_per_tick = int(max_per_tick)
        self._q: "queue.Queue[int]" = queue.Queue()
        # Set by the orchestrator while a foreground query() is building a
        # response; the worker's pause_gate blocks on this so consolidation
        # (which mutates ssm_a + blurbs) runs only between turns, never mid-recall.
        self.foreground_busy = threading.Event()  # not set by default
        self._thread = threading.Thread(
            target=self._run, name="ponder-consolidation-worker", daemon=True
        )
        self._stopped = False
        self._thread.start()

    # -- foreground-priority yielding (mirrors DistillWorker._wait_foreground) --

    def _wait_foreground(self) -> None:
        """Block while a foreground query() is busy. Called before each
        consolidation so the ssm_a/blurbs mutation never races Seam-2 recall."""
        while self.foreground_busy.is_set() and not self._stopped:
            self.foreground_busy.wait(timeout=0.5)

    # -- the sweep (read-only; safe during foreground) --

    def tick(self) -> int:
        """Scan for fading anchors and enqueue them for consolidation. Called by
        the orchestrator at the tail of ``query()`` (after the fade ingest). Read-
        only on the memory (``fading_anchors`` only calls ``_recoverability`` +
        reads the store), so it is safe to run while ``foreground_busy`` is set.
        Returns the number of anchors enqueued this tick."""
        try:
            ids = self._fade.fading_anchors(
                epsilon=self.epsilon, max_depth=self.max_depth,
                max_per_tick=self.max_per_tick,
            )
        except Exception as e:  # noqa: BLE001 - never break the turn
            print(f"[consolidation-tick-fail] {e}", file=sys.stderr)
            return 0
        for aid in ids:
            self._q.put(aid)
        return len(ids)

    # -- the worker loop --

    def _run(self) -> None:
        while True:
            anchor_id = self._q.get()
            if anchor_id is _CONSOLIDATE_SENTINEL:
                self._q.task_done()
                return
            try:
                # Gate the mutation: never consolidate while a foreground query
                # is reading ssm_a/blurbs. Blocks until the gap between turns.
                # NB: a job popped BEFORE drain's sentinel is always finished --
                # ``_stopped`` only breaks the wait, not an in-hand job (drain's
                # contract is "finish in-flight + queued"; the sentinel terminates
                # the loop after the real jobs drain).
                self._wait_foreground()
                blurb = self._fade.blurbs.text(anchor_id)
                if blurb is None:
                    continue  # anchor vanished (reset) -- skip
                count = self._fade.consolidation_count(anchor_id)
                prior = self._fade.prior_gist(anchor_id)
                gist = self._gister.gist(blurb, prior, count + 1)
                if gist is None:
                    # Cold-start: Bonsai down / parse fail -- skip, leave R4,
                    # retry next sweep. No fabricated gist.
                    continue
                self._fade.consolidate(anchor_id, gist)
                if self._fact_sink is not None and (gist.facts or
                                                    gist.state_assertions):
                    try:
                        self._fact_sink.write(
                            anchor_id, gist.facts, gist.state_assertions)
                    except Exception as e:  # noqa: BLE001 - best-effort sink
                        print(f"[consolidation-factsink-fail] {e}",
                              file=sys.stderr)
            except Exception as e:  # noqa: BLE001 - never kill the queue
                print(f"[consolidation-fail] anchor {anchor_id}: {e}",
                      file=sys.stderr)
            finally:
                self._q.task_done()

    # -- teardown --

    def drain(self, timeout: Optional[float] = 5.0) -> bool:
        """Stop accepting new work, finish in-flight + queued consolidations,
        join the worker thread. Returns True if the thread joined within
        ``timeout``. Called from the orchestrator's teardown hook."""
        if self._stopped:
            return True
        self._stopped = True
        # Wake a blocked _wait_foreground so the worker can exit.
        self.foreground_busy.clear()
        self._q.put(_CONSOLIDATE_SENTINEL)
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()