"""Tests for the layer-3 sandbox world (icm/layer3_env.py).

Mechanical only: each hidden rule has a correct manifest, a default wrong
manifest, and a reason that names the rule. No model calls.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icm.layer3_env import HEADER_FILE, Layer3Env, RULES, reason_keywords  # noqa: E402


def _default(meta):
    return {"files": len(meta["files"]), "rows": meta["total_lines"],
            "hash_mode": "content"}


def test_each_rule_accepts_the_correct_manifest_and_rejects_the_default(tmp_path):
    env = Layer3Env(str(tmp_path / "root"))
    for rule in RULES:
        sid = f"s-{rule}"
        meta = env.materialize(sid, rule)
        truth = env.correct(sid)
        assert env.check(sid, {"files": truth["files"], "rows": truth["rows"],
                               "hash_mode": truth["hash_mode"]})["ok"] is True
        bad = env.check(sid, _default(meta))
        assert bad["ok"] is False
        reason = bad["reason"].lower()
        if rule == "header":
            assert HEADER_FILE in bad["reason"] and "header" in reason
        elif rule == "exclude":
            assert "exclud" in reason
        elif rule == "name_hash":
            assert "name" in reason
        elif rule == "dedupe":
            assert "duplicate" in reason


def test_world_is_deterministic_per_session(tmp_path):
    env = Layer3Env(str(tmp_path / "a"), seed=3)
    env.materialize("s1", "header")
    first = env.correct("s1")
    env2 = Layer3Env(str(tmp_path / "b"), seed=3)
    env2.materialize("s1", "header")
    assert env2.correct("s1") == first


def test_invalid_specs_are_rejected_without_crashing(tmp_path):
    env = Layer3Env(str(tmp_path / "root"))
    env.materialize("s1", "exclude")
    assert env.check("s1", "nope")["ok"] is False
    assert env.check("s1", {"files": "x", "rows": 1, "hash_mode": "content"})["ok"] is False
    assert env.check("s1", {"files": 3, "rows": 5, "hash_mode": "other"})["ok"] is False


def test_reason_keywords_follow_the_reported_failure():
    assert "header" in reason_keywords("rows mismatch: part-01.csv has a header row")
    assert "exclud" in reason_keywords("files mismatch: part-03.csv is excluded")
    assert "duplicate" in reason_keywords("rows mismatch: duplicate rows count once")
    # a row-count error under the name_hash rule aligns to rows, not the rule
    assert "row" in reason_keywords("rows mismatch: expected 75, got 71")
    assert "hash" in reason_keywords("hash_mode mismatch: expected 'content+names'")
