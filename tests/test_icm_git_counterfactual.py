"""Tests for the git counterfactual trial builder.

Builds a throwaway git repo with a real add-then-remove pair and checks that
the discovery yields the pre-change blob as the answer, with the correct
version/decoy pool. No network, no model.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "icm"))

import git_counterfactual_test as gc  # noqa: E402


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, "-c", "user.name=t",
                        "-c", "user.email=t@t", *args],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def _write(repo, name, text):
    with open(os.path.join(repo, name), "w", encoding="utf-8") as f:
        f.write(text)


def test_discovery_builds_the_pre_change_trial(tmp_path):
    repo = str(tmp_path / "repo")
    os.makedirs(repo)
    git(repo, "init", "-q")
    base = "alpha line one\nalpha line two\nalpha line three\n"
    _write(repo, "gate.py", base)
    _write(repo, "other.py", "decoy content one\ndecoy content two\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial state")
    _write(repo, "gate.py", base + "trace_on()\ntrace_off()\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add debug tracing")
    _write(repo, "gate.py", base)
    _write(repo, "other.py", "decoy content three\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "remove debug tracing")

    trials, skipped = gc.discover([("synthetic", repo, "own")], decoys_n=2)
    gate = [t for t in trials if t["file"] == "gate.py"]
    assert len(gate) == 1, (trials, skipped)
    t = gate[0]
    assert "add debug tracing" in t["query"]
    assert t["correct"] == gc.blob_at(repo, "HEAD~1^", "gate.py")
    assert t["correct"] != t["after"]
    labels = {r["label"] for r in t["records"] if r["kind"] == "version"}
    assert {"before", "after"} <= labels
    assert t["versions"] >= 2
    assert t["pool"] >= 2
    assert any(r["kind"] == "decoy" for r in t["records"])
    # the answer carries the pre-change content, not the feature
    answer = next(r for r in t["records"] if r["id"] == t["correct"])
    assert "trace_on()" not in answer["text"]
    after = next(r for r in t["records"] if r["id"] == t["after"])
    assert "trace_on()" in after["text"]
