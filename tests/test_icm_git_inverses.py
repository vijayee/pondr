"""Tests for the git inverse census rule (scripts/icm/census_git_inverses.py).

Pins the patch-overlap rule with synthetic line sets -- no git, no repo data.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "icm"))

import census_git_inverses as gi  # noqa: E402


def test_full_inverse_is_detected():
    a = ({"timeout = 500", "pool = 8"}, set())
    b = (set(), {"timeout = 500", "pool = 8"})
    assert gi.inverse_strength(a, b) == 1.0


def test_reverse_direction_is_detected():
    a = (set(), {"trace_on()", "trace_off()"})
    b = ({"trace_on()", "trace_off()"}, set())
    assert gi.inverse_strength(a, b) > 0


def test_unrelated_changes_are_not_inverses():
    a = ({"alpha = 1"}, set())
    b = (set(), {"beta = 2"})
    assert gi.inverse_strength(a, b) == 0.0


def test_single_line_overlap_is_not_enough():
    a = ({"x = 1"}, set())
    b = (set(), {"x = 1"})
    assert gi.inverse_strength(a, b) == 0.0
