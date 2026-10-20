"""Tests for the E0-lite harness (scripts/icm/run_e0_lite.py).

The spec construction must match the world's truth for every rule (the E5
dedupe bug came from exactly this seam); the naive context-only manifest must
fail every rule; the profile parser must be rule-exact.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "icm"))

import run_e0_lite as e0  # noqa: E402
from icm.layer3_env import RULES, Layer3Env  # noqa: E402


def test_spec_for_matches_the_world_for_every_rule(tmp_path):
    env = Layer3Env(str(tmp_path / "world"))
    for rule in RULES:
        sid = f"s-{rule}"
        meta = env.materialize(sid, rule)
        spec = e0.spec_for(rule, meta)
        assert env.check(sid, spec)["ok"] is True, (rule, spec, env.check(sid, spec))


def test_naive_manifest_fails_every_rule(tmp_path):
    env = Layer3Env(str(tmp_path / "world2"))
    for rule in RULES:
        sid = f"n-{rule}"
        meta = env.materialize(sid, rule)
        assert env.check(sid, e0.naive_spec(meta))["ok"] is False


def test_profile_convention_parser_is_rule_exact():
    assert e0.parse_convention("USER PROFILE: convention=header.") == "header"
    assert e0.parse_convention("convention=name_hash") == "name_hash"
    assert e0.parse_convention("no convention here") is None
    assert e0.parse_convention("convention=bogus") is None
