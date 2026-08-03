"""A5 drift-gate tests for ``consolidation_worker.py`` -- the compaction-
stability drift signal inside the Phase C validate gate.

Drift is computed over (prior_gist, new_gist); the 1st consolidation pass has no
prior (``prior_gist`` is None) -> drift is skipped. These tests drive a 2nd pass
by consolidating ONCE directly (``mem.consolidate`` sets ``prior_gist``), fading
the anchor back to R4, then running the validate worker with a gister that
returns a pass-2 narrative of controlled drift vs the pass-1 prior.
"""

from __future__ import annotations

import time

from src.subconscious.consolidation_worker import ConsolidationWorker
from src.subconscious.gister import StructuredGist

from tests.test_fade import _StubEmbedder, _StubVoice, _fade_anchor_to_r4
from tests.test_consolidation import (
    _RecordingSink,
    _ScriptedDecider,
    _drain,
    _mem_with_faded_anchor,
)

_PRIOR = "alpha line\nbeta line\ngamma line"   # 3 lines (the pass-1 gist)
_HIGH_DRIFT = "delta line\nepsilon line\nzeta line"   # 0 shared -> drift 1.0
_LOW_DRIFT = "alpha line\nbeta line\ngamma line\ndelta line"   # +1 line -> ~0.14


class _FixedNarrativeGister:
    """Gister double returning a fixed narrative (the pass-2 narrative). Exposes
    ``decider`` (the scripted fidelity judge) + ``shape`` (for resolve-edit
    symmetry, unused here)."""

    def __init__(self, decider, narrative: str) -> None:
        self.decider = decider
        self.narrative = narrative
        self.shape_calls: list[tuple] = []

    def gist(self, blurb, prior_gist, count):
        return StructuredGist(
            narrative=self.narrative,
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


def _mem_with_prior(prior_narrative: str = _PRIOR):
    """Build a FadeMemory + anchor that has been consolidated ONCE (so
    ``prior_gist`` is set to ``prior_narrative``) and faded back to R4, ready for
    a 2nd validate sweep. Returns ``(mem, aid)``."""
    mem, aid = _mem_with_faded_anchor()
    mem.consolidate(aid, StructuredGist(
        narrative=prior_narrative, facts=[], state_assertions=[],
        consolidation_count=1))
    assert mem.prior_gist(aid) == prior_narrative
    assert mem.consolidation_count(aid) == 1
    _fade_anchor_to_r4(mem, aid)
    assert mem.recall_anchor(aid).regime  # faded back
    return mem, aid


def _drain_validate_drift(mem, decider_verdict, narrative, threshold,
                          validate=True):
    """Drive a validate(-aware) worker with a drift-DEFER threshold through one
    tick + drain. Returns ``(worker, gister, decider)`` post-drain."""
    decider = _ScriptedDecider(decider_verdict)
    gister = _FixedNarrativeGister(decider, narrative)
    worker = ConsolidationWorker(
        mem, gister, fact_sink=_RecordingSink(),
        epsilon=0.03, max_depth=3, max_per_tick=8, validate=validate,
        drift_defer_threshold=threshold,
    )
    worker.foreground_busy.set()
    worker.tick()
    time.sleep(0.2)
    worker.foreground_busy.clear()
    _drain(worker)
    return worker, gister, decider


# -- 1st pass: no prior -> drift skipped -> threshold does NOT defer a clean gist --
def test_drift_first_pass_no_prior_no_defer() -> None:
    mem, aid = _mem_with_faded_anchor()   # 1st pass: prior is None
    worker, _g, decider = _drain_validate_drift(
        mem, {"corruption": False, "reason": ""}, "anything", threshold=0.5)
    # Clean verdict + no prior -> drift None -> no drift-DEFER -> consolidated.
    assert mem.consolidation_count(aid) == 1
    assert worker.pending_reviews() == []
    assert decider.calls  # judge ran


# -- 2nd pass + high drift + clean + threshold set -> DEFER --
def test_drift_high_drift_clean_defers() -> None:
    mem, aid = _mem_with_prior(_PRIOR)
    worker, _g, _d = _drain_validate_drift(
        mem, {"corruption": False, "reason": ""}, _HIGH_DRIFT, threshold=0.5)
    # count stays 1 (the pass-2 gist was deferred, not applied).
    assert mem.consolidation_count(aid) == 1
    revs = worker.pending_reviews()
    assert len(revs) == 1
    assert revs[0]["anchor_id"] == aid
    assert revs[0]["reason"].startswith("high drift")
    # the drift field is surfaced on the review (A5).
    assert revs[0]["drift"] == 1.0


# -- 2nd pass + high drift + threshold None (default) -> observe-only, NO defer --
def test_drift_threshold_none_observes_only() -> None:
    mem, aid = _mem_with_prior(_PRIOR)
    worker, _g, _d = _drain_validate_drift(
        mem, {"corruption": False, "reason": ""}, _HIGH_DRIFT, threshold=None)
    # threshold None -> drift is computed but never auto-DEFERs -> consolidated.
    assert mem.consolidation_count(aid) == 2
    assert worker.pending_reviews() == []


# -- 2nd pass + low drift + clean + threshold set -> consolidate (below threshold) --
def test_drift_low_drift_clean_consolidates() -> None:
    mem, aid = _mem_with_prior(_PRIOR)
    worker, _g, _d = _drain_validate_drift(
        mem, {"corruption": False, "reason": ""}, _LOW_DRIFT, threshold=0.5)
    # drift ~0.14 < 0.5 -> no drift-DEFER -> consolidated.
    assert mem.consolidation_count(aid) == 2
    assert worker.pending_reviews() == []


# -- corruption wins over drift: corruption=True -> corruption DEFER, not drift --
def test_drift_corruption_wins_over_drift() -> None:
    mem, aid = _mem_with_prior(_PRIOR)
    worker, _g, _d = _drain_validate_drift(
        mem, {"corruption": True, "reason": "swapped entity"},
        _HIGH_DRIFT, threshold=0.5)
    # corruption is checked BEFORE drift -> the corruption reason wins.
    # (Other R4 anchors in the fixture also corruption-DEFER; isolate aid's.)
    assert mem.consolidation_count(aid) == 1   # not applied
    rev = next(r for r in worker.pending_reviews() if r["anchor_id"] == aid)
    assert rev["reason"] == "swapped entity"
    # drift is still carried on the review (computed before the verdict checks).
    assert rev["drift"] == 1.0


# -- pending_reviews carries the drift field (None when 1st pass / no prior) --
def test_pending_reviews_drift_none_on_first_pass() -> None:
    mem, aid = _mem_with_faded_anchor()
    target = mem.blurbs.text(aid)
    # Defer via corruption on the 1st pass -> drift is None (no prior).
    from tests.test_consolidation import _corrupt_only
    decider = _ScriptedDecider(_corrupt_only(target, "swapped"))
    from tests.test_consolidation import _ValidateGister
    gister = _ValidateGister(decider)
    worker = ConsolidationWorker(
        mem, gister, fact_sink=_RecordingSink(),
        epsilon=0.03, max_depth=3, max_per_tick=8, validate=True,
        drift_defer_threshold=0.5)
    worker.foreground_busy.set()
    worker.tick()
    time.sleep(0.2)
    worker.foreground_busy.clear()
    _drain(worker)
    revs = worker.pending_reviews()
    assert len(revs) == 1
    assert revs[0]["drift"] is None   # 1st pass: prior None -> drift not computed