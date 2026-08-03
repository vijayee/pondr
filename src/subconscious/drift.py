"""A5: compaction-stability drift-ratio signal (Tencent-survey Phase 1 item 5).

A faithful port of Tencent's ``computeLineDriftRatio``
(``metric-tracking-l3-latency.ts:142``) -- a cheap DETERMINISTIC line-diff that
complements the 8B fidelity judge in the Phase C validated-compaction gate
([[pondr-fade-validated-compaction]]). The 8B MISSES subtle entity-attribute
swaps + value flips; a memory that rewrites heavily across consolidation passes
is thrashing -- suspicious even when the judge says clean. drift quantifies that
thrashing in [0, 1]:

* **0.0** -- identical line set (a reorder is 0, set-based not LCS).
* **~0.1** -- a light edit (one line changed in ten).
* **~0.5** -- heavy modification.
* **1.0** -- fully rewritten (no shared non-empty lines).

**The pair that matters is (prior_gist, new_gist) -- compaction STABILITY
across consolidation passes.** drift(blurb, gist.narrative) is UNINFORMATIVE:
verbatim->gist compaction always rewrites structurally (a summary is not a
paraphrase of the source), so it drifts ~1.0 BY DESIGN, flagging every 1st-pass
compaction as "thrashing". The meaningful signal is how much the NEW gist
changed vs the PRIOR gist -- a stable memory drifts little between passes; a
thrashing one rewrites heavily. The 1st pass has no prior (``prior_gist`` is
None) -> drift is skipped (no baseline).

Set-based line diff (order ignored, dup lines collapse via ``set``) -- mirrors
Tencent exactly. ``None``/empty inputs are guarded (``or ""``) so an absent
prior never crashes.
"""

from __future__ import annotations

__all__ = ["compute_line_drift_ratio"]


def compute_line_drift_ratio(before: str, after: str) -> float:
    """Compaction-stability signal in [0, 1] (Tencent ``computeLineDriftRatio``).

    Set-based line diff over the non-empty lines of ``before`` and ``after``:
    ``added`` = new lines absent from the old set, ``removed`` = old lines
    absent from the new set, ``drift = min((added + removed) / (|old| + |new|),
    1)``. Returns ``0.0`` when both sides are empty (nothing to drift from /
    to). ``None`` inputs are treated as empty (``or ""``). Order is ignored
    (a reorder is NOT drift) and duplicate lines collapse (``set`` membership),
    matching the Tencent reference.
    """
    old = [ln for ln in (before or "").split("\n") if ln.strip()]
    new = [ln for ln in (after or "").split("\n") if ln.strip()]
    old_set, new_set = set(old), set(new)
    added = sum(1 for ln in new if ln not in old_set)
    removed = sum(1 for ln in old if ln not in new_set)
    total = len(old) + len(new)
    if total == 0:
        return 0.0
    return min((added + removed) / total, 1.0)