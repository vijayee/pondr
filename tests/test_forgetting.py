"""Tests for src/memory/forgetting.py -- the Phase 3b forgetting math.

These test each canonical mechanism against the chat's LITERAL code in
docs/The_Ponder_Engine_Chat.json (line 15674 for on_retrieve, line 15590 for
on_dream_state). The earlier worked example ``0.010 -> 0.0060 -> 0.0018`` is
NOT used as the gate: those numbers come from the superseded simple formula
(``boost = 0.05 * strength``) that the author replaced with the self-limiting
formula (diminishing returns). The gate is that each mechanism matches the
chat's code and composes into the stated drift-back-to-baseline behavior.
"""

from datetime import datetime, timedelta

import pytest

from src.memory.forgetting import (
    ACCESS_SATURATION,
    BASE_BOOST,
    DIMINISHING_K,
    MIN_DECAY_RATE,
    LTP_DECAY_MULTIPLIER,
    LTP_RETRIEVAL_COUNT,
    LTP_WINDOW_DAYS,
    apply_dream_state,
    apply_retrieval_boost,
    access_frequency,
    compose_utility,
    days_between,
    default_meta,
    should_archive,
)


# ── helpers ──
def ts(days: float = 0.0, hours: float = 0.0, base: str = "2026-01-01T00:00:00") -> str:
    """ISO timestamp `days`+`hours` after `base`."""
    d = datetime.fromisoformat(base) + timedelta(days=days, hours=hours)
    return d.isoformat()


def boost_sequence(n: int, step_days: float = 2.0, **kwargs) -> list[dict]:
    """Apply n retrieval boosts, each `step_days` apart (avoids saturation)."""
    m = default_meta()
    metas = [m]
    for i in range(n):
        m = apply_retrieval_boost(m, now_ts=ts(days=i * step_days), **kwargs)
        metas.append(m)
    return metas


# ── default_meta / days_between ──
def test_default_meta_has_all_canonical_fields():
    m = default_meta()
    for k in (
        "utility_score",
        "utility_decay_rate",
        "base_decay_rate",
        "state",
        "access_count",
        "reconsolidation_count",
        "ltp_phase",
        "consolidation_window_start",
        "retrieval_timestamps",
        "saturation_flags",
        "validity_end",
    ):
        assert k in m, f"missing {k}"
    assert m["utility_decay_rate"] == m["base_decay_rate"] == 0.01
    assert m["state"] == "current"
    assert m["ltp_phase"] == "early"


def test_days_between_positive_and_subday():
    assert days_between(ts(days=1), ts(0)) == pytest.approx(1.0)
    assert days_between(ts(hours=1), ts(0)) == pytest.approx(1.0 / 24.0)
    # within 24h is < 1.0 day
    assert days_between(ts(hours=23), ts(0)) < 1.0


def test_days_between_rejects_backwards_clock():
    with pytest.raises(ValueError):
        days_between(ts(0), ts(days=1))


# ── mechanism #1: diminishing-returns boost ──
def test_first_retrieval_diminishing_value():
    # chat: recons=1 => diminishing = 1/(1+0.3*1) = 0.769 => boost 0.0385
    # 0.010 -> 0.010 * (1 - 0.0385) = 0.009615 (NOT the simple 0.0095)
    m = apply_retrieval_boost(default_meta(), retrieval_strength=1.0, now_ts=ts(0))
    assert m["access_count"] == 1
    assert m["reconsolidation_count"] == 1
    assert m["utility_decay_rate"] == pytest.approx(0.009615, rel=1e-3)
    assert m["retrieval_timestamps"] == [ts(0)]
    assert m["consolidation_window_start"] == ts(0)


def test_diminishing_returns_shrinks_each_boost():
    metas = boost_sequence(20, step_days=2.0)  # 2-day spacing avoids saturation
    decays = [m["utility_decay_rate"] for m in metas]
    # monotonic decrease
    for a, b in zip(decays, decays[1:]):
        assert b < a
    # per-step reduction shrinks: reduction_1 > reduction_5 > reduction_20
    reductions = [a - b for a, b in zip(decays, decays[1:])]
    assert reductions[0] > reductions[4] > reductions[19]
    # 20th retrieval's boost is much smaller than the 1st
    assert reductions[19] < reductions[0] / 3.0


def test_retrieval_strength_scales_boost():
    weak = apply_retrieval_boost(default_meta(), retrieval_strength=0.3, now_ts=ts(0))
    strong = apply_retrieval_boost(default_meta(), retrieval_strength=1.0, now_ts=ts(0))
    # strong match reduces decay more (smaller decay = more persistent)
    assert strong["utility_decay_rate"] < weak["utility_decay_rate"]


# ── mechanism #2: saturation ──
def test_saturation_skips_boost_after_threshold_in_24h():
    m = default_meta()
    # 6 normal retrievals 1h apart (each sees <6 recent => not saturated)
    for i in range(6):
        m = apply_retrieval_boost(m, now_ts=ts(hours=i))
    assert m["saturation_flags"] == 0
    assert len(m["retrieval_timestamps"]) == 6
    decay_before_saturation = m["utility_decay_rate"]
    # 7th retrieval within 24h sees 6 recent => saturates
    m = apply_retrieval_boost(m, now_ts=ts(hours=6))
    assert m["saturation_flags"] == 1
    # no timestamp appended, no reconsolidation counted
    assert len(m["retrieval_timestamps"]) == 6
    assert m["reconsolidation_count"] == 6
    # access_count still increments (every retrieval counts)
    assert m["access_count"] == 7
    # decay bumped up slightly (less persistent), not boosted down
    assert m["utility_decay_rate"] > decay_before_saturation
    assert m["utility_decay_rate"] == pytest.approx(
        min(0.05, decay_before_saturation * 1.02)
    )


# ── mechanism #3: LLM-mediated signal ──
def test_llm_important_boosts_more_than_routine():
    imp = apply_retrieval_boost(
        default_meta(), llm_signal="important", now_ts=ts(0)
    )
    routine = apply_retrieval_boost(
        default_meta(), llm_signal="routine", now_ts=ts(0)
    )
    assert imp["utility_decay_rate"] < routine["utility_decay_rate"]


def test_llm_frustration_increases_decay():
    fr = apply_retrieval_boost(
        default_meta(), llm_signal="frustration", now_ts=ts(0)
    )
    # frustration: decay * 1.05 (more decay, less persistent), no boost
    assert fr["utility_decay_rate"] == pytest.approx(min(0.05, 0.01 * 1.05))
    # still recorded as a retrieval
    assert fr["reconsolidation_count"] == 1
    assert len(fr["retrieval_timestamps"]) == 1


def test_llm_correction_no_boost():
    cor = apply_retrieval_boost(
        default_meta(), llm_signal="correction", now_ts=ts(0)
    )
    # correction: no boost, decay unchanged (still counts as a retrieval)
    assert cor["utility_decay_rate"] == pytest.approx(0.01)
    assert cor["reconsolidation_count"] == 1


# ── LTP promotion (one-time x0.3) ──
def test_ltp_promotion_one_time():
    m = default_meta()
    # 3 retrievals spanning 16 days (>= 15-day window)
    m = apply_retrieval_boost(m, now_ts=ts(days=0))
    m = apply_retrieval_boost(m, now_ts=ts(days=10))
    assert m["ltp_phase"] == "early"  # only 2 retrievals, window 10d < 15
    m = apply_retrieval_boost(m, now_ts=ts(days=16))
    assert m["ltp_phase"] == "late"  # 3rd retrieval, window 16d >= 15
    decay_at_ltp = m["utility_decay_rate"]
    # LTP applied a x0.3 on top of the 3rd boost
    # (3rd-boost decay ~0.009069; *0.3 ~ 0.002721)
    assert decay_at_ltp < 0.005
    # 4th retrieval: LTP must NOT re-apply x0.3 (one-time promotion)
    m4 = apply_retrieval_boost(m, now_ts=ts(days=20))
    assert m4["ltp_phase"] == "late"
    # decay_4 should be decay_at_ltp * (1 - boost_4), NOT * 0.3 again.
    # If it re-applied x0.3, decay_4 would be < 0.3 * decay_at_ltp * 1.
    assert m4["utility_decay_rate"] > 0.5 * decay_at_ltp


def test_ltp_requires_window_not_just_count():
    # 3 retrievals within 5 days: count met, window NOT met => no LTP
    m = default_meta()
    for d in (0, 2, 4):
        m = apply_retrieval_boost(m, now_ts=ts(days=d))
    assert m["ltp_phase"] == "early"
    assert m["reconsolidation_count"] == 3


# ── floor ──
def test_decay_rate_never_below_floor():
    # large boost + custom high floor forces clamping
    m = default_meta()
    m["utility_decay_rate"] = 0.006
    out = apply_retrieval_boost(
        m, retrieval_strength=20.0, now_ts=ts(0), min_decay_rate=0.005
    )
    assert out["utility_decay_rate"] == 0.005  # clamped to floor


# ── on_dream_state: drift-back-to-baseline + utility decay ──
def test_dream_state_drifts_decay_back_to_baseline():
    m = default_meta()
    m["utility_decay_rate"] = 0.006  # boosted down from base 0.01
    m["base_decay_rate"] = 0.01
    m["retrieval_timestamps"] = [ts(0)]
    m["utility_score"] = 0.5
    # 30 days of disuse: boost fades, rate drifts back toward 0.01
    out = apply_dream_state(m, now_ts=ts(days=30))
    assert out["utility_decay_rate"] > 0.006  # drifted up
    assert out["utility_decay_rate"] < 0.01  # not all the way back yet
    # ~0.0098 (fade = 0.9057**30 ~ 0.051; 0.01 + (-0.004)*0.051 ~ 0.00979)
    assert out["utility_decay_rate"] == pytest.approx(0.00979, abs=2e-3)


def test_dream_state_decays_utility_score():
    m = default_meta()
    m["utility_decay_rate"] = 0.01
    m["retrieval_timestamps"] = [ts(0)]
    m["utility_score"] = 0.5
    out = apply_dream_state(m, now_ts=ts(days=30))
    assert out["utility_score"] < 0.5
    # 0.5 * (1 - 0.01)**30 ~ 0.5 * 0.740 ~ 0.370 (drift raises rate a hair first)
    assert out["utility_score"] == pytest.approx(0.37, abs=0.05)


def test_dream_state_noop_without_retrieval_history():
    m = default_meta()
    m["utility_score"] = 0.5
    out = apply_dream_state(m, now_ts=ts(days=30))
    # no last-retrieval time => nothing to decay
    assert out["utility_score"] == 0.5
    assert out["utility_decay_rate"] == 0.01


def test_dream_state_noop_when_retrieved_now():
    m = default_meta()
    m["retrieval_timestamps"] = [ts(days=30)]
    m["utility_score"] = 0.5
    out = apply_dream_state(m, now_ts=ts(days=30))  # elapsed = 0
    assert out["utility_score"] == 0.5


# ── access_frequency (recency-weighted) ──
def test_access_frequency_recency_weighting():
    # one retrieval 1 day ago vs one 30 days ago (same "now")
    now = ts(days=30)
    recent = default_meta()
    recent["retrieval_timestamps"] = [ts(days=29)]  # 1 day before now
    old = default_meta()
    old["retrieval_timestamps"] = [ts(days=0)]  # 30 days before now
    assert access_frequency(recent, now) > access_frequency(old, now)


def test_access_frequency_disuse_drops_to_zero():
    m = default_meta()
    m["retrieval_timestamps"] = [ts(days=0)]
    # immediately after retrieval: positive frequency
    assert access_frequency(m, ts(days=0)) > 0.0
    # long disuse: the single retrieval's contribution fades toward 0
    assert access_frequency(m, ts(days=365)) < 1e-3


def test_access_frequency_saturates():
    # many recent retrievals saturate the frequency to 1.0
    m = default_meta()
    m["retrieval_timestamps"] = [ts(days=i * 0.1) for i in range(50)]
    assert access_frequency(m, ts(days=5.0)) == pytest.approx(1.0, abs=1e-6)


# ── compose_utility ──
def test_compose_utility_weights_access_and_structural():
    m = default_meta()
    m["retrieval_timestamps"] = [ts(days=0)]
    # access_frequency at saturation ~1.0, structural 0.0 => 0.4
    # (use access_saturation=ACCESS_SATURATION default; one retrieval won't
    # saturate, so test the weighting directly with extreme values)
    m2 = default_meta()
    u_low_struct = compose_utility(m2, structural_salience=0.0, now_ts=ts(days=1))
    u_high_struct = compose_utility(m2, structural_salience=1.0, now_ts=ts(days=1))
    # structural dominates (0.6 weight): high-struct > low-struct
    assert u_high_struct > u_low_struct
    # with no retrieval history + structural 1.0 => 0.6 exactly
    assert u_high_struct == pytest.approx(0.6)


def test_compose_utility_rejects_out_of_range_salience():
    m = default_meta()
    with pytest.raises(ValueError):
        compose_utility(m, structural_salience=1.5, now_ts=ts(0))
    with pytest.raises(ValueError):
        compose_utility(m, structural_salience=-0.1, now_ts=ts(0))


def test_compose_utility_disuse_lowers_score():
    # same structural salience, but disuse drops access_frequency => lower score
    m = default_meta()
    m["retrieval_timestamps"] = [ts(days=0)]
    fresh = compose_utility(m, structural_salience=0.5, now_ts=ts(days=0))
    stale = compose_utility(m, structural_salience=0.5, now_ts=ts(days=60))
    assert stale < fresh


# ── should_archive ──
def test_should_archive_below_threshold_and_current():
    m = default_meta()
    m["utility_score"] = 0.05
    assert should_archive(m) is True


def test_should_archive_false_when_above_threshold():
    m = default_meta()
    m["utility_score"] = 0.5
    assert should_archive(m) is False


def test_should_archive_false_when_not_current():
    for state in ("archived", "deprecated", "superseded"):
        m = default_meta()
        m["state"] = state
        m["utility_score"] = 0.01
        assert should_archive(m) is False, state


def test_should_archive_respects_custom_threshold():
    m = default_meta()
    m["utility_score"] = 0.15
    assert should_archive(m, utility_prune_below=0.1) is False
    assert should_archive(m, utility_prune_below=0.2) is True


# ── integration: the full drift-back-to-baseline arc ──
def test_full_arc_use_then_disuse_returns_to_baseline():
    """The chat's Combined-Effect narrative: use boosts persistence, disuse
    lets the boost fade so the decay rate drifts back toward baseline."""
    m = default_meta()
    # 3 satisfied retrievals over a week
    for d in (0, 3, 7):
        m = apply_retrieval_boost(m, llm_signal="satisfied", now_ts=ts(days=d))
    decay_after_use = m["utility_decay_rate"]
    assert decay_after_use < 0.01  # boosted (more persistent)
    # 30 days of disuse: drift back toward baseline
    m = apply_dream_state(m, now_ts=ts(days=37))
    assert m["utility_decay_rate"] > decay_after_use  # faded up
    assert m["utility_decay_rate"] < 0.01  # not fully back yet at 30d
    # 1 year of disuse: essentially back to baseline
    m = apply_dream_state(m, now_ts=ts(days=365))
    assert m["utility_decay_rate"] == pytest.approx(0.01, abs=1e-3)