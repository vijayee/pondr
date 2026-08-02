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
from .gister import BonsaiGister, StructuredGist

_CONSOLIDATE_SENTINEL = object()  # signals the worker thread to exit (drain)

# Reason recorded when the fidelity judge could not return a verdict (Bonsai
# down / parse fail). The worker DEFERS in this case -- it never auto-consolidates
# an unvalidated gist -- and surfaces the same "held for review" notice so the
# user is never silently stuck on a forgotten anchor.
_REASON_JUDGE_UNAVAILABLE = "validation unavailable (fidelity judge failed)"


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
                 max_per_tick: int = 8, validate: bool = False) -> None:
        self._fade = fade_mem
        self._gister = gister
        self._fact_sink = fact_sink
        self.epsilon = float(epsilon)
        self.max_depth = int(max_depth)
        self.max_per_tick = int(max_per_tick)
        self.validate = bool(validate)
        # Deferred consolidation reviews (validated-compaction escalation). The
        # worker appends here ONLY from its own thread, in the foreground-clear
        # windows between turns; the orchestrator reads + resolves ONLY during a
        # foreground query (foreground_busy set, worker blocked in
        # ``_wait_foreground``). The gate already serializes the two sides against
        # ``ssm_a``/``blurbs`` mutation, so it serializes this list too -- no lock.
        # Each entry: ``{"anchor_id", "gist", "reason", "blurb"}`` (``gist`` is the
        # ``StructuredGist`` the judge flagged or that was held unvalidated).
        self._pending: list[dict] = []
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
        Anchors already pending a review (held for the user) are skipped so a
        deferred anchor is not re-gisted + re-deferred every sweep (wastes a
        Bonsai call + duplicates the review). Returns the number of anchors
        enqueued this tick."""
        try:
            ids = self._fade.fading_anchors(
                epsilon=self.epsilon, max_depth=self.max_depth,
                max_per_tick=self.max_per_tick,
            )
        except Exception as e:  # noqa: BLE001 - never break the turn
            print(f"[consolidation-tick-fail] {e}", file=sys.stderr)
            return 0
        pending_ids = {r["anchor_id"] for r in self._pending}
        enqueued = 0
        for aid in ids:
            if aid in pending_ids:
                continue
            self._q.put(aid)
            enqueued += 1
        return enqueued

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
                if self.validate:
                    # Validated compaction: a Bonsai fidelity judge gates the
                    # in-place write. ``corruption=False`` -> clean compression,
                    # apply silently. ``corruption=True`` -> the gist changed a
                    # fact's MEANING; defer + escalate to the user instead of
                    # overwriting the verbatim with a corrupted memory.
                    # ``None`` (judge HTTP/parse fail) -> ALSO defer; NEVER
                    # auto-consolidate an unvalidated gist (honest, not faked).
                    verdict = self._gister.decider.verify_fidelity(
                        blurb, gist.narrative)
                    if verdict is None:
                        self._defer(anchor_id, gist, blurb,
                                    _REASON_JUDGE_UNAVAILABLE)
                        continue
                    if verdict.get("corruption"):
                        self._defer(anchor_id, gist, blurb,
                                    verdict.get("reason") or "gist changed a fact")
                        continue
                    # clean -- fall through to consolidate.
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

    def _defer(self, anchor_id: int, gist: StructuredGist, blurb: str,
               reason: str) -> None:
        """Hold a candidate gist for user review instead of applying it.

        Appends ``{anchor_id, gist, reason, blurb}`` to ``self._pending``. The
        anchor is NOT consolidated (stays R4); ``tick`` skips already-pending
        anchors so it is not re-gisted next sweep. The orchestrator surfaces the
        review on the next turn and the user resolves it via ``resolve`` (keep /
        accept / edit), which runs in the foreground (worker blocked, race-free).
        """
        self._pending.append({
            "anchor_id": anchor_id,
            "gist": gist,
            "reason": reason,
            "blurb": blurb,
        })

    # -- validated-compaction escalation (foreground side; worker is blocked) --

    def pending_reviews(self) -> list[dict]:
        """Snapshot of deferred reviews for the orchestrator to surface.

        Returns shallow copies so the orchestrator cannot mutate the worker's
        internal list. Each item adds a display ``excerpt`` (the stored blurb,
        truncated) + a 1-indexed position is implied by list order. Safe to call
        during a query (foreground_busy set, worker blocked).
        """
        out = []
        for r in self._pending:
            excerpt = r.get("blurb") or ""
            if len(excerpt) > 120:
                excerpt = excerpt[:117] + "..."
            out.append({
                "anchor_id": r["anchor_id"],
                "reason": r["reason"],
                "narrative": r["gist"].narrative,
                "excerpt": excerpt,
            })
        return out

    def resolve(self, idx: int, action: str,
                edited_narrative: Optional[str] = None) -> bool:
        """Apply a user's resolution to a deferred review. Called from the
        orchestrator DURING a query (foreground_busy set -> the worker is parked
        in ``_wait_foreground``, never mid-``consolidate``), so the
        ``self._fade.consolidate`` call here cannot race the worker thread.

        ``action``:
          - ``"keep"``: drop the review; the anchor stays at its faded regime
            (R4); the user declined the compression.
          - ``"accept"``: apply the held gist as-is (the user approved it
            despite the corruption flag); the anchor jumps R4 -> R1.
          - ``"edit"``: replace the narrative with ``edited_narrative`` (the
            user corrected the corruption), re-extract facts from the corrected
            text, then consolidate; the anchor jumps R4 -> R1.

        Returns True on a successful resolution, False on a stale/invalid index
        or a missing ``edited_narrative`` for ``edit`` (no-op; the review stays).
        """
        if not isinstance(idx, int) or idx < 0 or idx >= len(self._pending):
            return False
        rev = self._pending[idx]
        aid = rev["anchor_id"]
        if action == "keep":
            del self._pending[idx]
            return True
        if action == "accept":
            self._fade.consolidate(aid, rev["gist"])
            del self._pending[idx]
            return True
        if action == "edit":
            if (not isinstance(edited_narrative, str)
                    or not edited_narrative.strip()):
                return False
            new_gist = self._gister.shape(
                edited_narrative, edited_narrative,
                rev["gist"].consolidation_count)
            self._fade.consolidate(aid, new_gist)
            del self._pending[idx]
            return True
        return False

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