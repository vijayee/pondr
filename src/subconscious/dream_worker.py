"""Serve-scheduled GNN dream-state consolidation (the nightly dream pass).

The GNN Consolidator (``src.gnn.consolidate``) was always shipped as a
nightly ``--dry-run``-default CLI loop, but the serve path NEVER constructed
it: the trained checkpoint sat on disk unwired (the audit's "trained-but-unwired"
gap #1) and the "nightly" claim had no clock behind it. This worker is the
serve-path wire-up.

Shape: a self-scheduling daemon thread with a wall-clock interval (default
nightly), mirroring the worker family (``DistillWorker`` / ``ConsolidationWorker``
/ ``SceneAuthoringWorker``) in every discipline EXCEPT scheduling -- those
workers are tick-driven per turn, this one sleeps and self-fires, so there is
no ``tick()``. The ``foreground_busy`` Event is the same priority gate: the
orchestrator sets it at query entry and clears it at return, and the pass
waits for it to clear before mutating the store (the GNN pass mutates scenes'
memory graph, M-nodes, and edge-meta -- racing a query's retrieval would be a
read-write race).

Eval gate: ``Consolidator.run()`` refuses an apply that would write placeholder
abstracts (no decider) or exceeds the ``apply_max_prunes`` / ``apply_max_abstracts``
blast-radius caps (``ConsolidationConfig.apply_gate_enabled``). The worker's
own discipline adds the other two defenses:

- a REQUIRED trained checkpoint -- the constructor refuses to load from a
  missing path (build_ponder fails loudly at serve startup, not silently with
  random salience against a live store);
- dry-run default -- ``apply=False`` is the default, so wiring the flag on
  starts with observed reports (evidence before mutation). The offline
  recall-before/after eval is ``scripts/eval_consolidation_recall.py``; run it
  on a DB copy BEFORE ever enabling apply on a real corpus.

Failure semantics (mirror the family): a per-pass exception is logged and the
thread keeps sleeping / retrying the next interval -- one crashed pass never
loses the serve loop.
"""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

from ..config import ConsolidationConfig

log = logging.getLogger(__name__)


DEFAULT_DREAM_CHECKPOINT = "data/pod_runs/phase3a/all_fixed_bounded.pt"


class DreamWorker:
    """Scheduled, foreground-gated GNN consolidation over the live store.

    Constructed by ``build_ponder`` when ``dream_consolidation`` is on (with a
    REQUIRED ``checkpoint`` -- see the module docstring), then handed to the
    orchestrator, which sets/clears ``foreground_busy`` around each query so
    ``run_once`` executes only between turns. ``start()`` launches the daemon
    loop; ``run_once()`` is the public single-pass entry (tests + a manual
    on-demand pass use it directly, bypassing the scheduler).
    """

    def __init__(
        self,
        store,
        checkpoint: str,
        interval_s: float = 86400.0,
        apply: bool = False,
        device: str = "cpu",
        config: Optional[ConsolidationConfig] = None,
        decider=None,
        verifier=None,
        limit: Optional[int] = None,
    ) -> None:
        if not Path(checkpoint).is_file():
            raise ValueError(
                f"dream checkpoint not found: {checkpoint!r} -- the serve "
                "path refuses to wire an untrained/absent GNN model (point "
                "--dream-checkpoint at a trained combined ckpt, e.g. "
                "scripts/assemble_gnn_checkpoint.py's output all_fixed.pt)")
        self.store = store
        self.apply = apply
        self.interval_s = max(interval_s, 1.0)
        self.device = device
        self.config = config
        self.decider = decider
        self.verifier = verifier
        self.limit = limit
        self.checkpoint = checkpoint
        self.model = _load_trained_model(checkpoint, device)
        # Priority gate -- the orchestrator sets/clears around each query.
        self.foreground_busy = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Last report dict (observability: the serve log prints a one-line
        # summary per pass; the full dict stays here for inspection).
        self.last_report: Optional[dict] = None
        self.passes = 0

    # ── scheduling ──

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop, name="dream-worker", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001 -- one crashed pass never
                # kills the scheduler; the next pass retries.
                log.error("dream pass failed: %s", e)
                traceback.print_exc(file=sys.stderr)

    # ── the pass ──

    def _wait_foreground(self) -> None:
        """Block until the orchestrator clears the priority gate (0.5s polls,
        so a stopping worker never hangs past its wait)."""
        while self.foreground_busy.is_set() and not self._stop.is_set():
            self._stop.wait(0.5)

    def run_once(self) -> Optional[dict]:
        """Run ONE consolidation pass between turns; return the report.

        Dry-run unless ``apply=True`` (and the consolidated eval gate then
        governs the actual writes -- see ``Consolidator._eval_apply_gate``).
        """
        self._wait_foreground()
        if self._stop.is_set():
            return None
        from ..gnn.consolidate import Consolidator

        cons = Consolidator(
            self.store, model=self.model, config=self.config,
            verifier=self.verifier, decider=self.decider,
            dry_run=not self.apply, device=self.device,
        )
        report = cons.run(limit=self.limit)
        self.last_report = report
        self.passes += 1
        log.info(
            "dream pass #%d: scored=%s abstracts=%d edges=%d/%d pruned=%d %s",
            self.passes, report["subgraphs_scored"],
            len(report["abstracts"]), len(report["edges_accepted"]),
            len(report["edges_proposed"]), len(report["pruned"]),
            report.get("apply_skipped") or ("dry-run" if report["dry_run"]
                                            else "applied"),
        )
        return report

    # ── teardown ──

    def drain(self, timeout: float = 5.0) -> bool:
        """Stop the scheduler and join the thread. Best-effort: an in-flight
        pass finishes its current store mutation before the join expires."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True


def _load_trained_model(checkpoint: str, device: str):
    """Load a trained combined GNN checkpoint (strict), eval mode.

    Mirrors ``scripts/run_consolidation.py._load_model`` -- the serve-side
    duplicate keeps ``scripts/`` out of ``src`` imports. Failure is loud
    (raises) per the required-trained-checkpoint defense.
    """
    import torch

    from ..gnn.model import GNNModel

    model = GNNModel(
        hidden_dim=128, num_heads=4, num_layers=3,
        predicate_vocab_size=32, num_clusters=16,
    )
    # weights_only=True: checkpoint is pure tensor state, never pickled code.
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model