"""Unit tests for the B3 reclamation cascade (src/retrieval/reclaim.py).

Pure-helper tests: no I/O, no LLM, no ChunkedContext. Exercises the 4-level
cascade (fastpath / mild / aggressive / emergency) + score order + the
under-budget byte-identical-OFF-equivalent guarantee. Token counts are passed
explicitly (``full_tokens`` / ``summary_tokens``) so the assertions do not
depend on ``len(text)//4`` rounding; the ``full_text`` / ``summary_text``
strings are distinct (``F``- vs ``S``-padded) so the returned rung is readable.
"""

from __future__ import annotations

from src.retrieval.reclaim import ReclaimItem, reclaim_to_budget


def _item(score, full_tokens, summary_tokens=None, truncate=None):
    """Build a ReclaimItem with explicit token counts + distinguishable texts.

    ``full_text`` is ``F``-padded, ``summary_text`` is ``S``-padded (4 chars
    per token so ``len//4`` agrees with the declared tokens). ``truncate``
    is an optional callback returning ``(text, tokens)``.
    """
    full_text = "F" * (full_tokens * 4)
    summary_text = ("S" * (summary_tokens * 4)) if summary_tokens is not None else None
    return ReclaimItem(
        score=score,
        full_text=full_text,
        full_tokens=full_tokens,
        summary_text=summary_text,
        summary_tokens=summary_tokens,
        truncate=truncate,
    )


def _trunc_factory(marker="T"):
    """A truncate callback that returns a small fixed text (so truncation
    always shrinks) and records the char cap it was called with."""
    calls: list[int] = []

    def _t(chars):
        calls.append(chars)
        s = marker * 4  # 1 token
        return s, len(s) // 4

    _t.calls = calls  # type: ignore[attr-defined]
    return _t


# 1. under-budget no-op -> all FULL, byte-identical to the off path.
def test_under_budget_returns_all_full():
    items = [_item(0.9, 100, summary_tokens=10), _item(0.5, 100, summary_tokens=10)]
    out = reclaim_to_budget(items, max_tokens=10_000)
    assert out == [it.full_text for it in items]


# 2. mild demotes the lowest-score episode to its summary.
def test_mild_demotes_lowest_score_episode_to_summary():
    high = _item(0.9, 200, summary_tokens=10)
    low = _item(0.5, 200, summary_tokens=10)
    # total full 400; max 220 -> demote low (->10): 210 <= 220 fits. high FULL.
    out = reclaim_to_budget([high, low], max_tokens=220)
    assert out == [high.full_text, low.summary_text]


# 3. mild demotes only as many as needed (one demotion, rest stay FULL).
def test_mild_demotes_only_as_many_as_needed():
    a = _item(0.9, 200, summary_tokens=10)
    b = _item(0.7, 200, summary_tokens=10)
    c = _item(0.5, 200, summary_tokens=10)
    # total 600; max 420 -> demote c (->10): 410 <= 420 fits. a, b stay FULL.
    out = reclaim_to_budget([a, b, c], max_tokens=420)
    assert out == [a.full_text, b.full_text, c.summary_text]


# 4. mild exhausts (all demoted) then aggressive drops the lowest.
def test_mild_exhausts_then_aggressive_drops():
    items = [_item(s, 200, summary_tokens=100)
             for s in (0.9, 0.8, 0.7, 0.6, 0.5)]
    # total full 1000; mild -> all summary = 500 (>300); aggressive target 270
    # -> drop 0.5,0.6 -> 300 (kept 3 = min_keep -> stop); 300<=300 no emergency.
    out = reclaim_to_budget(items, max_tokens=300)
    assert out == [it.summary_text for it in items[:3]]  # 0.9, 0.8, 0.7 kept
    assert len(out) == 3


# 5. aggressive respects min_keep (blocked before headroom; no emergency fires
#    because total stays under max).
def test_aggressive_respects_min_keep():
    items = [_item(s, 200, summary_tokens=110) for s in (0.9, 0.7, 0.5)]
    # mild -> 3 summaries = 330 (each mild step keeps total > 350 until the last
    # demote, so all three demote). max 350, target 315, min_keep 3.
    # aggressive: kept=3<=min_keep -> drop nothing (330>315 wants to drop but
    # cannot). 330<=350 -> no emergency. Nothing dropped.
    out = reclaim_to_budget(items, max_tokens=350, min_keep=3)
    assert len(out) == 3  # min_keep blocked aggressive (would drop 1 at min_keep=2)
    assert out == [it.summary_text for it in items]


# 6. aggressive drops to max_tokens*headroom (not exactly max_tokens).
def test_aggressive_drops_to_headroom_not_max():
    items = [_item(s, 200, summary_tokens=80)
             for s in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)]
    # mild -> 7 summaries = 560. max 350, target 315, min_keep 2.
    # headroom drops 4 (->240, kept 3 > min_keep 2 -> headroom stopped it).
    # max-targeting would drop only 3 (->320, kept 4). len==3 is the headroom
    # signature (max-targeting would leave 4).
    out = reclaim_to_budget(items, max_tokens=350, min_keep=2)
    assert len(out) == 3


# 7. emergency truncates a single oversized item (truncate invoked with cap).
def test_emergency_truncates_single_oversized_item():
    trunc = _trunc_factory()
    it = _item(0.5, full_tokens=1000, summary_tokens=None, truncate=trunc)
    # mild skips (no summary); aggressive min_keep-blocked (kept 1<=3);
    # emergency truncates the body with truncate_chars.
    out = reclaim_to_budget([it], max_tokens=100, truncate_chars=800)
    assert trunc.calls == [800]
    assert out == ["T" * 4]


# 8. emergency drops to floor 1 when still over after truncation.
def test_emergency_drops_to_floor_1_when_still_over():
    def trunc(chars):
        s = "T" * 400  # 100 tokens after truncation (still > max)
        return s, 100
    a = _item(0.9, 1000, summary_tokens=None, truncate=trunc)
    b = _item(0.5, 1000, summary_tokens=None, truncate=trunc)
    # mild skips; aggressive min_keep-blocked; emergency truncates both (->200)
    # still > 50 -> drop lowest (b) to floor 1.
    out = reclaim_to_budget([a, b], max_tokens=50, truncate_chars=400)
    assert len(out) == 1
    assert out == ["T" * 400]  # survivor is a (highest score), truncated


# 9. score order: lowest demoted first; ties -> later position demoted first.
def test_score_order_lowest_first_ties_by_later_position():
    a = _item(0.5, 200, summary_tokens=10)  # earlier (higher rank)
    b = _item(0.5, 200, summary_tokens=10)  # later (demotes first on tie)
    # max 210: demote ONE -> 200+10=210 fits. The later (b) demotes, a kept.
    out = reclaim_to_budget([a, b], max_tokens=210)
    assert out == [a.full_text, b.summary_text]


# 10. output preserves original input order (relevance order), not score order.
def test_output_preserves_original_input_order():
    a = _item(0.9, 100, summary_tokens=10)  # idx 0
    b = _item(0.3, 100, summary_tokens=10)  # idx 1 (lowest, demoted)
    c = _item(0.5, 100, summary_tokens=10)  # idx 2
    # max 210: total 300 -> demote b (lowest) -> 210 fits. Output in INPUT order.
    out = reclaim_to_budget([a, b, c], max_tokens=210)
    assert out == [a.full_text, b.summary_text, c.full_text]


# 11. no summary form (non-episode): mild skips; aggressive drops it.
def test_no_summary_form_skips_mild_reclaims_at_aggressive():
    items = [_item(s, 100, summary_tokens=None) for s in (0.9, 0.7, 0.5, 0.3)]
    # No summaries -> mild skips all. aggressive target 315 (max 350) drops the
    # lowest (0.3) -> 300 (kept 3 = min_keep -> stop); 300<=350 no emergency.
    out = reclaim_to_budget(items, max_tokens=350)
    assert out == [it.full_text for it in items[:3]]  # 0.3 dropped
    # all kept items still FULL (mild did not demote any -- no summary form).
    assert all(t == it.full_text for t, it in zip(out, items[:3]))


# 12. under budget -> byte-identical to the off path (all FULL, summaries
#     available but not used).
def test_under_budget_byte_identical_to_off():
    items = [_item(0.9, 100, summary_tokens=10), _item(0.5, 50, summary_tokens=5)]
    out = reclaim_to_budget(items, max_tokens=10_000)
    assert out == [it.full_text for it in items]


# edge: empty list -> empty (format_for_llm still emits the PRIMARY header).
def test_empty_items_returns_empty():
    assert reclaim_to_budget([], max_tokens=100) == []


# edge: truncate=None in emergency -> item skipped (dropped in the floor pass).
def test_emergency_skips_non_truncatable_drops_it_instead():
    a = _item(0.9, 1000, summary_tokens=None, truncate=None)  # not truncatable
    b = _item(0.5, 1000, summary_tokens=None, truncate=None)
    # mild skips; aggressive min_keep-blocked; emergency cannot truncate
    # (truncate=None) -> drop-to-floor drops b, keeps a (still full, oversized).
    out = reclaim_to_budget([a, b], max_tokens=50)
    assert len(out) == 1
    assert out == [a.full_text]  # a kept (cannot truncate, cannot drop below 1)