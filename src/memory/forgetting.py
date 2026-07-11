"""Phase 3b forgetting math -- pure functions, no store access.

Implements the Ponder Engine forgetting system as specified in
``docs/The_Ponder_Engine_Chat.json``. The unit of forgetting is the EDGE: each
edge carries a ``utility_score`` that fades with disuse and is boosted by
retrieval, composed from retrieval history + structural salience
("combined, not competing")::

    utility_score = 0.4 * access_frequency + 0.6 * structural_salience

These functions operate on a plain ``meta`` dict (the per-edge sidecar shape
persisted by ``src/memory/edge_meta.py``). They are pure -- they take ``now_ts``
as an ISO-8601 string parameter and return a NEW dict; they never touch the
store, so they are fully unit-testable in isolation.

Design-spec reconciliation (see docs/Phase 3b.md section 0 for the full note):
The chat gives two formulas. The SIMPLE one (msg at line 15590,
``boost = 0.05 * strength``) produced the worked example ``0.010 -> 0.0060 ->
0.0018`` but creates immortal memories, so the author REPLACED it with the
self-limiting formula (msg at line 15674): diminishing returns, saturation,
LLM-mediated signal, and boost decay. 3b implements the self-limiting formula;
the ``0.0060/0.0018`` numbers are simple-formula artifacts and are NOT the test
gate. The gate is that each canonical mechanism matches the chat's literal code.

The chat's ``on_retrieve`` store-and-mutates ``utility_decay_rate`` down, but its
"Combined Effect" narrative wants drift-back-to-baseline during disuse (the
rate returns toward 0.010 over weeks of disuse). Store-and-mutate alone cannot
drift back up, so ``apply_dream_state`` realizes mechanism #4 (boost decay) as a
stored-rate drift-back toward ``base_decay_rate`` -- the boost component fades at
the boost half-life, so an unused edge's decay rate returns to baseline. This
uses every piece of the chat's design and achieves the stated behavior.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Optional

# ── Canonical constants (from docs/The_Ponder_Engine_Chat.json line 15674) ──
DEFAULT_BASE_DECAY_RATE = 0.01   # 1%/day, ~70-day half-life
DEFAULT_UTILITY_SCORE = 0.5
MIN_DECAY_RATE = 0.001           # floor: never decay below 0.1%/day (immortal guard)
DECAY_RATE_CAP = 0.05            # max decay rate (saturation / frustration cap)
SATURATION_THRESHOLD = 5         # >5 retrievals/24h => saturation
SATURATION_DECAY_BUMP = 1.02     # slight decay increase on saturation
FRUSTRATION_DECAY_BUMP = 1.05    # larger decay increase on frustration signal
BASE_BOOST = 0.05                # 5% reduction per retrieval (before diminishing)
DIMINISHING_K = 0.3              # diminishing-returns curvature: 1/(1+0.3*n)
BOOST_HALF_LIFE_DAYS = 7.0       # retrieval boost half-life (drift-back + access freq)
LTP_RETRIEVAL_COUNT = 3          # late-phase LTP threshold
LTP_WINDOW_DAYS = 15             # ...across this many days
LTP_DECAY_MULTIPLIER = 0.3       # 70% reduction on LTP promotion (one-time)
ACCESS_SATURATION = 10.0         # access_frequency normalizer (calibration lever)
UTILITY_PRUNE_BELOW = 0.1        # utility_score < this + state current => archive

# LLM-mediated importance signal modifiers (msg line 15674 mechanism #3)
LLM_SIGNAL_MODIFIERS = {
    "important": 1.5,    # 50% stronger boost
    "routine": 1.0,      # normal
    "satisfied": 1.2,    # slightly stronger
    "frustration": -0.5, # reverse: increase decay
    "correction": 0.0,   # no boost -- old memory was wrong
}

VALID_STATES = ("current", "archived", "deprecated", "superseded")


def default_meta() -> dict:
    """A fresh per-edge sidecar dict with all canonical fields at defaults."""
    return {
        "utility_score": DEFAULT_UTILITY_SCORE,
        "utility_decay_rate": DEFAULT_BASE_DECAY_RATE,
        "base_decay_rate": DEFAULT_BASE_DECAY_RATE,
        "state": "current",
        "access_count": 0,
        "reconsolidation_count": 0,
        "ltp_phase": "early",
        "consolidation_window_start": None,
        "retrieval_timestamps": [],
        "saturation_flags": 0,
        "validity_end": None,
        # A1 deep-archive: when this edge was soft-archived (state='archived').
        # The deep-archive sweep ages on this to decide physical removal
        # (>deep_archive_days). None for current/never-archived edges and for
        # edges soft-archived before this field shipped (legacy -> not aged).
        "archived_at": None,
    }


def _parse(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string to datetime."""
    return datetime.fromisoformat(ts)


def days_between(later_ts: str, earlier_ts: str) -> float:
    """Non-negative elapsed days from earlier to later (float; <1.0 = within 24h).

    Raises ValueError if earlier is after later (clocks must move forward).
    """
    later = _parse(later_ts)
    earlier = _parse(earlier_ts)
    delta = (later - earlier).total_seconds()
    if delta < 0:
        raise ValueError(f"days_between: earlier {earlier_ts} after later {later_ts}")
    return delta / 86400.0


def _daily_decay_factor(boost_half_life_days: float) -> float:
    """Per-day multiplier for a quantity with the given half-life.

    For boost_half_life_days=7.0 this is 0.5**(1/7) ~= 0.9057, matching the
    chat's literal ``0.9 ** days_ago`` (which approximates a 7-day half-life).
    """
    return 0.5 ** (1.0 / boost_half_life_days)


def apply_retrieval_boost(
    meta: dict,
    retrieval_strength: float = 1.0,
    llm_signal: str = "routine",
    now_ts: Optional[str] = None,
    saturation_threshold: int = SATURATION_THRESHOLD,
    min_decay_rate: float = MIN_DECAY_RATE,
) -> dict:
    """The on_retrieve logic (msg line 15674, mechanisms #1-3 + LTP).

    Pure: returns a new meta dict. ``now_ts`` is the retrieval timestamp
    (ISO-8601). Composition order: every retrieval counts (access_count++);
    saturation (>threshold in 24h) skips the boost; otherwise the LLM signal
    modulates a diminishing-returns boost on ``utility_decay_rate``; LTP is
    promoted (one-time x0.3) when reconsolidation_count >= 3 across >= 15 days.
    """
    if now_ts is None:
        raise ValueError("apply_retrieval_boost: now_ts is required")
    m = copy.deepcopy(meta)
    # Access tracking -- every retrieval counts (chat: "Every retrieval").
    m["access_count"] = m.get("access_count", 0) + 1
    timestamps = list(m.get("retrieval_timestamps") or [])
    decay = float(m.get("utility_decay_rate", DEFAULT_BASE_DECAY_RATE))

    # ── Mechanism #2: saturation (>threshold retrievals in the last 24h) ──
    # Breaks the frustration loop: the user keeps asking because the answer
    # isn't sticking; reinforcing the memory is counterproductive.
    recent = [t for t in timestamps if days_between(now_ts, t) < 1.0]
    if len(recent) > saturation_threshold:
        decay = min(DECAY_RATE_CAP, decay * SATURATION_DECAY_BUMP)
        m["saturation_flags"] = m.get("saturation_flags", 0) + 1
        m["utility_decay_rate"] = decay
        # Skip the normal boost, timestamp append, and reconsolidation counting.
        return m

    # ── Normal path: record the retrieval + apply the boost ──
    timestamps.append(now_ts)
    m["retrieval_timestamps"] = timestamps
    new_recons = m.get("reconsolidation_count", 0) + 1
    m["reconsolidation_count"] = new_recons
    if m.get("consolidation_window_start") is None:
        m["consolidation_window_start"] = now_ts

    modifier = LLM_SIGNAL_MODIFIERS.get(llm_signal, 1.0)
    if modifier < 0:
        # ── Mechanism #3 (frustration): increase decay, no boost ──
        decay = min(DECAY_RATE_CAP, decay * FRUSTRATION_DECAY_BUMP)
    elif modifier == 0.0:
        # ── Mechanism #3 (correction): no boost -- old memory was wrong ──
        pass
    else:
        # ── Mechanism #1: diminishing-returns boost ──
        # First retrieval (recons=1): ~3.85% reduction; 5th: ~2%; 20th: ~0.5%.
        # Approaches but never reaches zero boost.
        base_boost = BASE_BOOST * retrieval_strength
        diminishing = 1.0 / (1.0 + DIMINISHING_K * new_recons)
        effective_boost = base_boost * diminishing * modifier
        decay = max(min_decay_rate, decay * (1.0 - effective_boost))

    # ── LTP promotion (one-time x0.3; chat worked example: 0.0060 -> 0.0018) ──
    # Promoted once, when 3+ retrievals span 15+ days. Subsequent retrievals
    # boost normally but do NOT re-apply x0.3 (else decay tanks to the floor).
    window_start = m.get("consolidation_window_start")
    if (
        m.get("ltp_phase") != "late"
        and new_recons >= LTP_RETRIEVAL_COUNT
        and window_start is not None
        and days_between(now_ts, window_start) >= LTP_WINDOW_DAYS
    ):
        m["ltp_phase"] = "late"
        decay = max(min_decay_rate, decay * LTP_DECAY_MULTIPLIER)

    m["utility_decay_rate"] = decay
    return m


def apply_dream_state(
    meta: dict,
    now_ts: Optional[str] = None,
    boost_half_life_days: float = BOOST_HALF_LIFE_DAYS,
    min_decay_rate: float = MIN_DECAY_RATE,
) -> dict:
    """The on_dream_state logic (msg line 15590 on_dream_state + line 15674 #4).

    Two effects, both gated on a known last-retrieval time:
      1. Boost decay (mechanism #4): the accumulated boost FADES, so the stored
         ``utility_decay_rate`` drifts back up toward ``base_decay_rate`` as the
         edge goes unused. After ~7 days the boost is half gone; after a few
         weeks the rate is back near baseline. This is the drift-back-to-
         baseline behavior the chat's "Combined Effect" narrative specifies.
      2. Utility decay (msg 15590): ``utility_score *= (1 - decay_rate)**days``
         so a disused edge's utility fades toward the archive threshold.

    Pure: returns a new meta dict.
    """
    if now_ts is None:
        raise ValueError("apply_dream_state: now_ts is required")
    m = copy.deepcopy(meta)
    timestamps = m.get("retrieval_timestamps") or []
    if not timestamps:
        return m
    last_ts = timestamps[-1]
    elapsed = days_between(now_ts, last_ts)
    if elapsed <= 0:
        return m  # nothing to decay (retrieved "now")

    daily = _daily_decay_factor(boost_half_life_days)
    fade = daily ** elapsed  # 0..1, ->0 as the edge goes long-unused

    # 1. Drift utility_decay_rate back toward base_decay_rate.
    base = float(m.get("base_decay_rate", DEFAULT_BASE_DECAY_RATE))
    decay = float(m.get("utility_decay_rate", DEFAULT_BASE_DECAY_RATE))
    decay = base + (decay - base) * fade
    m["utility_decay_rate"] = max(min_decay_rate, decay)

    # 2. Decay utility_score over the elapsed interval.
    score = float(m.get("utility_score", DEFAULT_UTILITY_SCORE))
    m["utility_score"] = max(0.0, score * (1.0 - m["utility_decay_rate"]) ** elapsed)
    return m


def access_frequency(
    meta: dict,
    now_ts: str,
    boost_half_life_days: float = BOOST_HALF_LIFE_DAYS,
    access_saturation: float = ACCESS_SATURATION,
) -> float:
    """Recency-weighted retrieval rate in [0, 1] (the composition's "access_frequency").

    The chat composes utility from "access_frequency" (a rate) -- not the raw
    cumulative ``access_count`` (which never decreases and would make
    utility_score immortal). A recency-weighted rate naturally decays during
    disuse, which is what lets ``utility_score`` fall toward the archive
    threshold when an edge stops being retrieved. Each past retrieval
    contributes ``daily_factor ** days_ago``; the sum is normalized by
    ``access_saturation`` (a calibration lever -- how many recent retrievals
    saturate the frequency to 1.0).
    """
    timestamps = meta.get("retrieval_timestamps") or []
    if not timestamps:
        return 0.0
    daily = _daily_decay_factor(boost_half_life_days)
    total = 0.0
    for t in timestamps:
        elapsed = days_between(now_ts, t)
        total += daily ** elapsed
    return min(1.0, total / access_saturation)


def compose_utility(
    meta: dict,
    structural_salience: float,
    now_ts: str,
    boost_half_life_days: float = BOOST_HALF_LIFE_DAYS,
    access_saturation: float = ACCESS_SATURATION,
) -> float:
    """The composition: ``0.4 * access_frequency + 0.6 * structural_salience``.

    Structural salience (the trained GNN SalienceHead output, sigmoid'd to
    [0,1]) is weighted higher than access, but both feed in -- "combined, not
    competing" (chat msg line 15590). ``structural_salience`` must already be in
    [0, 1]; the caller sigmoid+clips the raw head logits before passing them.
    """
    if not 0.0 <= structural_salience <= 1.0:
        raise ValueError(
            f"compose_utility: structural_salience must be in [0,1], got {structural_salience}"
        )
    af = access_frequency(meta, now_ts, boost_half_life_days, access_saturation)
    return 0.4 * af + 0.6 * structural_salience


def should_archive(meta: dict, utility_prune_below: float = UTILITY_PRUNE_BELOW) -> bool:
    """True if a current edge has decayed below the archive threshold.

    The soft-archive criterion (chat msg line 100): edges with
    ``utility_score < 0.1 and state='current'`` -> ``state='archived'``. The edge
    stays in the live graph (excluded from default queries via the edge-level
    filter) and is NOT deleted -- the deep-archive tier (>365d archived) handles
    physical removal.
    """
    return meta.get("state") == "current" and float(
        meta.get("utility_score", DEFAULT_UTILITY_SCORE)
    ) < utility_prune_below