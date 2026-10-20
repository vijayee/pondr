"""Tests for the free-form phrasing harness (scripts/icm/git_cf_freeform.py).

Synthetic text only -- no personal phrasings. Pins the task-aware key mapping,
cache behaviour, phrasing-block parsing, and the after-pool construction.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "icm"))

import git_cf_freeform as ff  # noqa: E402


class StubAsk:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_before_after_neither_mapping(tmp_path):
    t = StubAsk(["BEFORE", "AFTER", "NEITHER"])
    k = ff.FreeformKeyer(cache_path=str(tmp_path / "c.jsonl"), ask=t)
    assert k.key_for("q1").key["epoch_ref"]["value"] == "EARLIEST"
    assert k.key_for("q2").key["epoch_ref"]["value"] == "LATEST"
    r3 = k.key_for("q3")
    assert r3.key["epoch_ref"]["kind"] == "none" and r3.key["abstained"] is True
    assert len(t.prompts) == 3


def test_cache_hit_costs_no_call(tmp_path):
    t = StubAsk(["BEFORE"])
    k = ff.FreeformKeyer(cache_path=str(tmp_path / "c.jsonl"), ask=t)
    assert k.key_for("same query").status == "judged"
    assert k.key_for("same   query").status == "cached"
    assert len(t.prompts) == 1


def test_empty_response_is_error(tmp_path):
    k = ff.FreeformKeyer(cache_path=str(tmp_path / "c.jsonl"),
                         ask=StubAsk([""]))
    r = k.key_for("q")
    assert r.key is None and r.status == "error"


def test_phrasing_blocks_split_on_blank_line(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("a {file}\nb {file}\n\nc {file}\nd {file}\n", encoding="utf-8")
    before, after = ff.load_phrasings(str(p))
    assert before == ["a {file}", "b {file}"]
    assert after == ["c {file}", "d {file}"]


def test_after_pool_drops_the_post_removal_version():
    trial = {"records": [
        {"id": "b", "kind": "version", "label": "before", "type": "f"},
        {"id": "a", "kind": "version", "label": "after", "type": "f"},
        {"id": "r", "kind": "version", "label": "removed", "type": "f"},
        {"id": "d", "kind": "decoy", "label": None, "type": "g"},
    ]}
    before_pool, after_pool = ff.build_pools(trial)
    assert {x["id"] for x in before_pool} == {"b", "a", "r", "d"}
    assert {x["id"] for x in after_pool} == {"b", "a", "d"}
