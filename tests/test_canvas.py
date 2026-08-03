"""B4: pure helpers in ``src/subconscious/canvas_format.py`` -- no I/O, no LLM.

Pins ``format_canvas_block`` (empty -> "", non-empty -> block, truncate),
``parse_canvas_meta`` (well-formed -> counts+progress; malformed -> defaults),
``validate_canvas`` (accept valid; reject empty/oversized/no-directive), and
``apply_replace_blocks`` (splice a known node; skip unknown; write-mode no-op).
"""

from __future__ import annotations

from src.subconscious.canvas_format import (
    apply_replace_blocks, format_canvas_block, parse_canvas_meta, validate_canvas,
)


# ── format_canvas_block ──────────────────────────────────────────────────────

def test_format_canvas_block_empty_returns_empty():
    assert format_canvas_block([]) == ""
    assert format_canvas_block(None) == ""


def test_format_canvas_block_renders_header_and_mermaid():
    cv = {"canvas_id": "canvas_000001", "task_label": "build-api",
          "progress": 40,
          "mermaid": "%%{taskGoal: build api, progress: 40}%%\nflowchart TD\n    N1[\"x\"]"}
    block = format_canvas_block([cv], max_tokens=1024)
    assert block.startswith("[TASK CANVAS")
    assert "canvas_000001" in block
    assert "build-api" in block
    assert "40%" in block
    assert "flowchart TD" in block


def test_format_canvas_block_truncates_to_budget():
    big = "%%{progress: 0}%%\nflowchart TD\n" + ("    N1[\"x\"]\n" * 500)
    cv = {"canvas_id": "canvas_000001", "task_label": "big", "progress": 0,
          "mermaid": big}
    # max_tokens=8 -> cap = 32 chars -> the block is truncated + flagged.
    block = format_canvas_block([cv], max_tokens=8)
    assert "[... canvas truncated ...]" in block
    assert len(block) <= 32 + len("\n[... canvas truncated ...]")


# ── parse_canvas_meta ───────────────────────────────────────────────────────

def test_parse_canvas_meta_reads_progress_and_counts():
    mermaid = (
        "%%{taskGoal: ship feature, progress: 60,"
        " createdTime: 2026-08-01T10:00:00, updatedTime: 2026-08-01T11:00:00}%%\n"
        "flowchart TD\n"
        "    N1[\"phase: plan<br/>status: done<br/>summary: x\"]\n"
        "    N2[\"phase: code<br/>status: doing<br/>summary: y\"]\n"
        "    N3[\"phase: test<br/>status: blocked<br/>summary: z\"]\n"
    )
    meta = parse_canvas_meta(mermaid)
    assert meta["taskGoal"] == "ship feature"
    assert meta["progress"] == 60
    assert meta["createdTime"] == "2026-08-01T10:00:00"
    assert meta["updatedTime"] == "2026-08-01T11:00:00"
    c = meta["counts"]
    assert c["done"] == 1 and c["doing"] == 1 and c["blocked"] == 1
    assert c["paused"] == 0


def test_parse_canvas_meta_malformed_returns_defaults():
    meta = parse_canvas_meta("")
    assert meta["progress"] == 0
    assert meta["taskGoal"] == ""
    assert meta["counts"] == {"done": 0, "doing": 0, "paused": 0, "blocked": 0}
    # A header with a non-numeric progress is tolerated (progress stays 0).
    meta2 = parse_canvas_meta("%%{progress: oops}%%\nflowchart TD")
    assert meta2["progress"] == 0


# ── validate_canvas ─────────────────────────────────────────────────────────

def test_validate_canvas_accepts_valid():
    ok, err = validate_canvas("%%{progress: 0}%%\nflowchart TD\n    N1[\"x\"]",
                              max_chars=4000)
    assert ok and err == ""


def test_validate_canvas_rejects_empty():
    ok, err = validate_canvas("", max_chars=4000)
    assert not ok and "empty" in err


def test_validate_canvas_rejects_oversized():
    big = "%%{progress: 0}%%\nflowchart TD\n" + ("x" * 100)
    ok, err = validate_canvas(big, max_chars=50)
    assert not ok and "exceeds" in err


def test_validate_canvas_rejects_no_directive():
    ok, err = validate_canvas("just some prose, no mermaid here", max_chars=4000)
    assert not ok and "missing" in err


# ── apply_replace_blocks ────────────────────────────────────────────────────

def test_apply_replace_blocks_splices_known_node():
    mermaid = "%%{progress: 0}%%\nflowchart TD\n    N1[\"old\"]\n    N2[\"keep\"]"
    out = apply_replace_blocks(mermaid, [{"node_id": "N1",
                                          "new_block": "    N1[\"new\"]"}])
    assert "N1[\"new\"]" in out
    assert "N2[\"keep\"]" in out
    assert "old" not in out


def test_apply_replace_blocks_skips_unknown_node():
    mermaid = "%%{progress: 0}%%\nflowchart TD\n    N1[\"x\"]"
    out = apply_replace_blocks(mermaid, [{"node_id": "N9",
                                          "new_block": "    N9[\"ghost\"]"}])
    # Unknown node id -> no-op splice (the original is returned unchanged).
    assert out == mermaid


def test_apply_replace_blocks_write_mode_noop():
    mermaid = "%%{progress: 0}%%\nflowchart TD\n    N1[\"x\"]"
    # No replace_blocks -> write-mode -> mermaid unchanged.
    assert apply_replace_blocks(mermaid, []) == mermaid
    assert apply_replace_blocks(mermaid, None) == mermaid