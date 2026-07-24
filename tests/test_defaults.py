"""Regression test for the Phase 1c-3c flag-default flip.

Three built-but-default-OFF flags were flipped ON by default (the hardening
decision): ``gliner_timing``, ``async_distill_enabled``,
``bonsai_isolation_extraction``. This pins the defaults so a future revert is
caught, and documents the coupling invariant (isolation is only safe behind
async_distill -- both flip together).
"""

from __future__ import annotations

from src.config import Config


def test_gliner_timing_defaults_on():
    assert Config().gliner_timing is True, (
        "gliner_timing default flipped back off -- the stderr timing log is the "
        "default surface for the CPU-vs-CUDA extraction bottleneck"
    )


def test_async_distill_enabled_defaults_on():
    assert Config().async_distill_enabled is True, (
        "async_distill_enabled default flipped back off -- background episode "
        "distillation is the production encode path"
    )


def test_bonsai_isolation_extraction_defaults_on():
    assert Config().bonsai_isolation_extraction is True, (
        "bonsai_isolation_extraction default flipped back off -- the 10-pass "
        "per-class extractor lifts strict has_state 0 -> 11/13"
    )


def test_isolation_coupled_with_async_distill_default():
    """Both flip together: isolation does 10x Bonsai HTTP (~22.8 s/doc) and is
    only viable because async_distill moves extraction to the background worker.
    The DEFAULT config has both ON; the code warns when a user opts out of
    async_distill while leaving isolation on (see serve_ponder)."""
    cfg = Config()
    assert cfg.async_distill_enabled and cfg.bonsai_isolation_extraction, (
        "the two flags must default ON together -- isolation without async_distill "
        "blocks the response ~22.8 s/doc"
    )


def test_strm_relevance_logging_defaults_off():
    assert Config().strm_relevance_logging is False, (
        "strm_relevance_logging default flipped on -- the raw-rating JSONL tap is "
        "side-effect-only and the labels only matter once a 2a head is in training; "
        "a cold start must accumulate nothing"
    )


def test_strm_graduation_logging_defaults_off():
    assert Config().strm_graduation_logging is False, (
        "strm_graduation_logging default flipped on -- the replay JSONL tap is "
        "side-effect-only and the v2 labels only matter once a graduation training "
        "run is gated on them; a cold start must accumulate nothing"
    )


def test_strm_salience_inflight_shortcut_defaults_on():
    """STRM Phase 5 IngestionTracker: the in-flight anchor short-circuit defaults
    ON so the cheap read (snapshot straight from the DistillWorker in-flight map,
    no vector round-trip) ships by default. Only read when --strm-salience is
    armed; False -> byte-identical to the pre-Phase-5 vector path (A/B rollback)."""
    assert Config().strm_salience_inflight_shortcut is True, (
        "strm_salience_inflight_shortcut default flipped off -- the cheap-read "
        "in-flight short-circuit is the production salience-recall path"
    )