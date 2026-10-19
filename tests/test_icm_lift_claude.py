"""Tests for the Claude-session lift (scripts/icm/lift_claude_sessions.py).

Synthetic transcript only -- no personal data. Pins the mechanical rules:
turn segmentation, intent-source lifting for terse directives, file units from
Read/Edit/Write, edit-failure tracking, and the end-state test verdict.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "icm"))

import lift_claude_sessions as lift  # noqa: E402


def _write(tmp_path, records):
    p = tmp_path / "proj" / "sess.jsonl"
    p.parent.mkdir(parents=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(p)


def _user_text(text, ts="2026-01-01T00:00:00Z", cwd="C:/repo"):
    return {"type": "user", "cwd": cwd, "timestamp": ts,
            "message": {"content": text}}


def _assistant(blocks, ts="2026-01-01T00:00:01Z"):
    return {"type": "assistant", "timestamp": ts,
            "message": {"content": blocks}}


def _tool_use(tid, name, inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _tool_result(tid, is_error=False, tool_use_result=None):
    rec = {"type": "user", "timestamp": "2026-01-01T00:00:02Z",
           "message": {"content": [{"type": "tool_result", "tool_use_id": tid,
                                    "content": "out", "is_error": is_error}]}}
    if tool_use_result is not None:
        rec["toolUseResult"] = tool_use_result
    return rec


def test_turn_segmentation_and_mechanical_fields(tmp_path):
    long_prompt = ("Please change the retrieval gate to use the archived index "
                   "and report the checksum for dataset alpha.")
    recs = [
        _user_text(long_prompt),
        _assistant([
            {"type": "text", "text": "I will inspect and edit the gate."},
            _tool_use("t1", "Read", {"file_path": "C:/repo/src/gate.py"}),
            _tool_use("t2", "Edit", {"file_path": "C:/repo/src/gate.py"}),
            _tool_use("t3", "Bash", {"command": "python -m pytest -q"}),
        ]),
        _tool_result("t1"),
        _tool_result("t2"),
        _tool_result("t3", tool_use_result={"stdout": "ok", "stderr": "",
                                            "interrupted": False}),
    ]
    turns, events = lift.parse_file(_write(tmp_path, recs), str(tmp_path))
    assert len(turns) == 1 and len(events) == 3
    t = turns[0]
    assert t["substantive"] is True and t["intent_source"] == "user"
    assert t["files_read"] == ["src/gate.py"]       # normalised under cwd
    assert t["files_edited"] == ["src/gate.py"]
    assert t["test_ok"] is True and t["test_runs"] == 1
    assert lift.file_label(t, "src/gate.py") == "useful"
    assert lift.file_label(t, "src/other.py") == "context"


def test_failed_edit_and_failed_test_mark_attempted(tmp_path):
    recs = [
        _user_text("Rewrite the config file so the pool size becomes eight."),
        _assistant([
            _tool_use("e1", "Edit", {"file_path": "C:/repo/cfg.yaml"}),
            _tool_use("b1", "Bash", {"command": "pytest tests/ -q"}),
        ]),
        _tool_result("e1", is_error=True),
        _tool_result("b1", is_error=True),
    ]
    turns, _ = lift.parse_file(_write(tmp_path, recs), str(tmp_path))
    t = turns[0]
    assert t["edit_failed"] == ["cfg.yaml"]
    assert t["test_ok"] is False
    assert lift.file_label(t, "cfg.yaml") == "attempted"


def test_terse_directive_lifts_the_agent_plan(tmp_path):
    recs = [
        _user_text("continue"),
        _assistant([{"type": "text",
                     "text": "Revert the timeout change and restore the pin."}]),
    ]
    turns, _ = lift.parse_file(_write(tmp_path, recs), str(tmp_path))
    t = turns[0]
    assert t["substantive"] is False
    assert t["intent_source"] == "user+agent-plan"
    assert "Revert the timeout change" in t["intent_text"]


def test_tool_result_records_do_not_start_turns(tmp_path):
    recs = [
        _user_text("Make the index FLAT and rerun the retrieval smoke test."),
        _assistant([_tool_use("r1", "Read", {"file_path": "C:/repo/a.py"})]),
        _tool_result("r1"),
    ]
    turns, events = lift.parse_file(_write(tmp_path, recs), str(tmp_path))
    assert len(turns) == 1 and len(events) == 1
    assert events[0]["ok"] is True


def test_last_test_run_decides_the_end_state(tmp_path):
    recs = [
        _user_text("Fix the failing test and make sure the suite passes."),
        _assistant([
            _tool_use("b1", "Bash", {"command": "pytest -q"}),
            _tool_use("b2", "Bash", {"command": "pytest -q"}),
        ]),
        _tool_result("b1", is_error=True),
        _tool_result("b2", is_error=False),
    ]
    turns, _ = lift.parse_file(_write(tmp_path, recs), str(tmp_path))
    t = turns[0]
    assert t["test_runs"] == 2 and t["test_ok"] is True
