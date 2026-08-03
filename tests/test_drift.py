"""Unit tests for ``src/subconscious/drift.py`` -- the A5 compaction-stability
``compute_line_drift_ratio`` (a faithful port of Tencent's
``computeLineDriftRatio``). Pure function, no deps."""

from __future__ import annotations

import math

from src.subconscious.drift import compute_line_drift_ratio


def test_identical_is_zero() -> None:
    assert compute_line_drift_ratio("a\nb\nc", "a\nb\nc") == 0.0


def test_fully_rewritten_is_one() -> None:
    # No shared non-empty lines -> added + removed = total -> 1.0.
    assert compute_line_drift_ratio("a\nb\nc", "x\ny\nz") == 1.0


def test_light_edit_is_small() -> None:
    # 10 lines, 1 changed: added=1, removed=1, total=20 -> 0.1.
    before = "\n".join(f"line{i}" for i in range(10))
    after = "\n".join(f"line{i}" for i in range(9)) + "\nCHANGED"
    assert math.isclose(compute_line_drift_ratio(before, after), 0.1, abs_tol=1e-9)


def test_reorder_is_zero() -> None:
    # Set-based: order is ignored, so a reorder is NOT drift.
    assert compute_line_drift_ratio("a\nb\nc", "c\nb\na") == 0.0


def test_one_side_empty() -> None:
    # before empty, after non-empty -> added=len(new), removed=0, total=len(new) -> 1.0.
    assert compute_line_drift_ratio("", "x\ny") == 1.0
    # before non-empty, after empty -> removed=len(old), total=len(old) -> 1.0.
    assert compute_line_drift_ratio("x\ny", "") == 1.0


def test_both_empty_is_zero() -> None:
    assert compute_line_drift_ratio("", "") == 0.0
    assert compute_line_drift_ratio("   \n\n\t", "  \n") == 0.0  # whitespace-only


def test_dup_lines_collapse() -> None:
    # Duplicates collapse via set: 3 dup lines vs 1 same line -> 0.0.
    assert compute_line_drift_ratio("a\na\na", "a") == 0.0


def test_clamp_to_one() -> None:
    # A case where the raw ratio would exceed 1 is impossible with the formula
    # (added+removed <= |old|+|new| always), but min() guards it. Verify a large
    # asymmetric case still lands in [0, 1].
    d = compute_line_drift_ratio("a\nb", "x\ny\nz\nw\nu\nv")
    assert 0.0 <= d <= 1.0
    # added=6, removed=2, total=8 -> 1.0 (clamped from 8/8=1.0 anyway).
    assert d == 1.0


def test_whitespace_only_lines_dropped() -> None:
    # Lines that are only whitespace are filtered (``ln.strip()``), so a blank
    # line between real content does not count as a line.
    assert compute_line_drift_ratio("a\n\nb", "a\nb") == 0.0


def test_none_inputs_handled() -> None:
    # ``None`` is guarded by ``or ""`` -> treated as empty.
    assert compute_line_drift_ratio(None, None) == 0.0
    assert compute_line_drift_ratio(None, "x\ny") == 1.0
    assert compute_line_drift_ratio("x\ny", None) == 1.0


def test_partial_overlap() -> None:
    # 2 shared, 1 removed, 1 added: before={a,b,c}, after={a,b,d}.
    # added=1 (d), removed=1 (c), total=6 -> (1+1)/6 = 2/6 = 1/3.
    d = compute_line_drift_ratio("a\nb\nc", "a\nb\nd")
    assert math.isclose(d, 1.0 / 3.0, abs_tol=1e-9)