"""B3: active 4-level token-reclamation cascade (pure, no I/O, no LLM).

Adapted from Tencent's ``compressor.ts`` ``resolveLevel`` (fastpath / mild /
aggressive / emergency), remapped to Pondr's per-turn retrieved context budget.
The single home of the cascade logic; designed for reuse by the deferred
``build_context_string`` follow-on (the ``search_memory`` tool renderer).

Every episode carries BOTH ``summary`` (gist) AND ``text`` (full), so "replace
with summary" is a natural content-preserving reclamation step (the reverse of
B2's ``verbatim``: summary -> full under intent, full -> summary under
pressure). Sections/documents/scenes have only a body (no separate summary
form), so they have FULL + DROP (+ emergency body-truncation) but no mild rung.

The cascade fires only on actual overflow and always runs mild -> aggressive
-> emergency, re-measuring after each step, stopping at budget. Episodes arrive
ranked highest-``score`` first; the cascade demotes/drops the LOWEST-score
(the tail) first, so it is deterministic (byte-stable when the flag is on).
Under budget -> all items render at FULL unchanged (byte-identical to the off
break-on-overflow path, which would also have kept them all). Only over-budget
ON changes behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ReclaimItem:
    """One PRIMARY-band item as seen by the reclamation cascade.

    ``score`` drives demote/drop order (lowest first). ``full_text`` /
    ``full_tokens`` are the FULL rendering (the off-path default). Episodes
    also carry a SUMMARY rendering (``summary_text`` / ``summary_tokens``) ->
    the mild rung; ``None`` -> no mild rung (sections/docs/scenes). ``truncate``
    is the emergency body-truncation callback: given a char cap, returns the
    ``(truncated_text, truncated_tokens)`` rendering; ``None`` -> no truncatable
    body (emergency skips it, drops it instead). The callback re-renders with
    the SAME header/metadata (only the body is sliced) so a truncated chunk is
    still recognizable.
    """
    score: float
    full_text: str
    full_tokens: int
    summary_text: Optional[str] = None
    summary_tokens: Optional[int] = None
    truncate: Optional[Callable[[int], tuple[str, int]]] = None


def reclaim_to_budget(
    items: list[ReclaimItem],
    max_tokens: int,
    *,
    headroom: float = 0.9,
    min_keep: int = 3,
    truncate_chars: int = 800,
) -> list[str]:
    """Reclaim PRIMARY-band items to fit ``max_tokens`` via a 4-level cascade.

    Returns the kept items' current rendered texts in ORIGINAL input order
    (renders top-down by relevance; the order is never shuffled). Under budget
    -> all ``full_text`` unchanged (byte-identical to the off path). Over
    budget -> the lowest-score episodes demote FULL -> SUMMARY (mild), then the
    lowest-score items DROP toward ``max_tokens * headroom`` (aggressive, never
    below ``min_keep``), then remaining FULL bodies truncate to
    ``truncate_chars`` (emergency), then DROP to floor 1 if still over.
    """
    if not items:
        return []

    n = len(items)
    cur_text = [it.full_text for it in items]
    cur_tok = [it.full_tokens for it in items]
    # level: 0 = FULL, 1 = SUMMARY. Dropped items are tracked separately so the
    # original-order return is reconstructable.
    level = [0] * n
    dropped = [False] * n

    def total() -> int:
        return sum(cur_tok[i] for i in range(n) if not dropped[i])

    def kept() -> int:
        return sum(1 for i in range(n) if not dropped[i])

    # 1. fastpath: under budget -> render all at FULL, byte-identical to off.
    if total() <= max_tokens:
        return [it.full_text for it in items]

    # Demote/drop order: lowest score first; ties -> later original position
    # first (earlier = higher rank = preserved longer) so the cascade is
    # deterministic (byte-stable when the flag is on).
    order = sorted(range(n), key=lambda i: (items[i].score, -i))

    # 2. mild: demote EPISODE items FULL -> SUMMARY, lowest-score first, re-
    # measuring after each. Stop when under budget or no demotable episodes
    # remain. Non-episode items (summary_text is None) are unaffected -- they
    # have no summary form, so their first reclamation rung is aggressive DROP.
    for i in order:
        if total() <= max_tokens:
            break
        if dropped[i] or level[i] == 1:
            continue
        if items[i].summary_text is None:
            continue
        cur_text[i] = items[i].summary_text
        cur_tok[i] = (
            items[i].summary_tokens
            if items[i].summary_tokens is not None
            else len(items[i].summary_text) // 4
        )
        level[i] = 1

    # 3. aggressive: DROP lowest-score items (any kind) toward the headroom
    # target, never below min_keep. Re-measure after each drop. min_keep guards
    # a pathological all-oversized set so aggressive never empties context;
    # emergency then truncates the survivors' bodies.
    target = int(max_tokens * headroom)
    for i in order:
        if total() <= target or kept() <= min_keep:
            break
        if dropped[i]:
            continue
        dropped[i] = True

    # 4. emergency (only if still over max_tokens -- i.e. aggressive was
    # min_keep-stopped with total > max_tokens): truncate remaining FULL bodies
    # to truncate_chars, re-measuring after each, stopping at budget; then, if
    # still over, DROP lowest-score down to a floor of 1. SUMMARY items are
    # already minimal (their body IS the short summary) so they are skipped
    # here -- their only further rung is DROP.
    if total() > max_tokens:
        for i in order:
            if total() <= max_tokens:
                break
            if dropped[i] or level[i] != 0:
                continue
            trunc = items[i].truncate
            if trunc is None:
                continue  # no truncatable body -> DROP in the next pass
            text, tok = trunc(truncate_chars)
            cur_text[i] = text
            cur_tok[i] = tok
        for i in order:
            if total() <= max_tokens or kept() <= 1:
                break
            if dropped[i]:
                continue
            dropped[i] = True

    return [cur_text[i] for i in range(n) if not dropped[i]]