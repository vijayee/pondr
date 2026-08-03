"""Task-canvas rendering + Mermaid parsing for the LLM USER message (B4).

The task canvas (``--task-canvas``) is the structural short-term task-state
axis the fade cannot provide: a per-task Mermaid flowchart the LLM authors +
reads each turn, with nodes carrying ``phase``/``status`` (done|doing|paused|
blocked)/``summary``. It is the offload-stack L2 analog of Tencent's Mermaid
canvas. The active canvas's Mermaid is injected into the USER message each turn
(per-turn dynamic -- like the fade block, NOT the cache-stable scene suffix),
re-read each turn, and sits OUTSIDE ChunkedContext (prepended after) so B3's
reclaim cascade never touches it.

This module mirrors ``scene_format.py`` / ``format_fade_block`` (``fade.py``):
pure string helpers, NO LLM/HTTP/file IO. ``format_canvas_block`` returns
``""`` when there is nothing to render so the caller's no-append path stays
byte-identical to flag-off. The parse/validate/replace helpers are reusable by
``orchestrator.update_canvas`` (the Bonsai-loop authoring tool) with no I/O.
"""

from __future__ import annotations

import re

# Mermaid ``%%{...}%%`` directive header (the first line of a canvas). Holds
# taskGoal / progress / createdTime / updatedTime. Best-effort regex: tolerant
# of extra keys / whitespace; ``progress`` is an int 0-100.
_META_RE = re.compile(r"%%\s*\{(.*)\}\s*%%")

# A node definition line: ``    N1["phase: ...<br/>status: ...<br/>..."]``.
# Capture the node id (N1) up to the ``[`` so ``apply_replace_blocks`` can find
# + replace the whole line by id. Tolerant of ``flowchart TD`` / ``graph`` and
# any shape delimiter (``[``, ``(``, ``{``, ``>``).
_NODE_LINE_RE = re.compile(r'^(\s*)(N\d+)\s*[([{>]')

# Status token inside a node body: ``status: doing``. Tolerant of ``<br/>``
# separators and surrounding whitespace.
_STATUS_RE = re.compile(r"status:\s*(done|doing|paused|blocked)", re.IGNORECASE)


def parse_canvas_meta(mermaid: str) -> dict:
    """Best-effort read of a canvas's ``%%{...}%%`` header + node status counts.

    Returns ``{"taskGoal", "progress", "createdTime", "updatedTime",
    "counts": {done, doing, paused, blocked}}``. Malformed / missing header ->
    defaults (``progress`` 0, empty strings, zero counts); never raises. The
    orchestrator stores ``progress`` from here on each ``update_canvas`` write
    so the L1.5 gate + injection can surface it without re-parsing the Mermaid.
    """
    out = {"taskGoal": "", "progress": 0, "createdTime": "",
           "updatedTime": "", "counts": {"done": 0, "doing": 0,
                                         "paused": 0, "blocked": 0}}
    if not mermaid:
        return out
    m = _META_RE.search(mermaid)
    if m:
        body = m.group(1)
        # ``key: value`` pairs, comma- or newline-separated; values may be quoted.
        for pair in re.split(r"[,\n]", body):
            if ":" not in pair:
                continue
            k, _, v = pair.partition(":")
            k = k.strip().strip('"').strip("'")
            v = v.strip().strip('"').strip("'")
            if k == "taskGoal":
                out["taskGoal"] = v
            elif k == "progress":
                try:
                    out["progress"] = max(0, min(100, int(v)))
                except ValueError:
                    pass
            elif k == "createdTime":
                out["createdTime"] = v
            elif k == "updatedTime":
                out["updatedTime"] = v
    counts = out["counts"]
    for line in mermaid.splitlines():
        sm = _STATUS_RE.search(line)
        if sm:
            counts[sm.group(1).lower()] += 1
    return out


def validate_canvas(mermaid: str, *, max_chars: int) -> tuple[bool, str]:
    """Light validation of a stored Mermaid. Returns ``(ok, error)``; ``error``
    is ``""`` when ok. The canvas is structural, not a syntax-critical artifact
    the system parses for control flow, so this is a BOUND + shape check (non-
    empty, within the size cap, looks like Mermaid) -- NOT a full grammar
    parse. Best-effort; never raises."""
    if not mermaid or not mermaid.strip():
        return False, "empty mermaid"
    if len(mermaid) > max_chars:
        return False, f"mermaid exceeds {max_chars} chars"
    if "flowchart" not in mermaid and "graph" not in mermaid and "%%{" not in mermaid:
        return False, "mermaid missing a flowchart/graph directive or %% header"
    return True, ""


def apply_replace_blocks(mermaid: str, replace_blocks: list[dict]) -> str:
    """Splice ``replace_blocks`` into ``mermaid`` (the ``file_action="replace"``
    dual-mode). Each block is ``{node_id, new_block}``: the WHOLE node line whose
    id matches ``node_id`` (matched via ``_NODE_LINE_RE`` on the leading
    ``N\\d+``) is replaced by ``new_block``. Unknown node ids are skipped
    (best-effort -- never raises). ``replace_blocks`` empty / malformed ->
    ``mermaid`` unchanged (write-mode is a no-op splice).

    The splice is line-oriented: a Mermaid node is one line in the canvas
    schema (``N1["phase: ...<br/>status: ..."]``), so replacing the whole line
    preserves the rest of the flowchart verbatim. v1 does NOT parse edge
    definitions (a ``new_block`` may include its own edges on subsequent lines
    if the LLM emits them, but the splice is one-line per block).
    """
    if not replace_blocks:
        return mermaid
    if not mermaid:
        return mermaid
    lines = mermaid.splitlines()
    # Index lines by their leading node id for O(1) lookup.
    by_id: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = _NODE_LINE_RE.match(line)
        if m:
            by_id[m.group(2)] = i
    for blk in replace_blocks:
        if not isinstance(blk, dict):
            continue
        nid = blk.get("node_id")
        new_block = blk.get("new_block")
        if not isinstance(nid, str) or not isinstance(new_block, str):
            continue
        idx = by_id.get(nid)
        if idx is None:
            continue  # unknown node id -- skip (best-effort)
        lines[idx] = new_block
    return "\n".join(lines)


def format_canvas_block(canvases: list[dict], *,
                        max_tokens: int = 1024) -> str:
    """Render the active canvas as a ``[TASK CANVAS]`` block for the LLM USER
    message. Takes the ``get_canvas`` dict shape (``{canvas_id, mermaid,
    task_label, progress, ...}``). Renders a one-line meta head (label/progress)
    + the Mermaid body, then truncates the WHOLE block to ``max_tokens``
    (``len // 4``, mirroring A4 + the formatter) -- structural text, so a char
    cap is the only reclaim it needs (no FULL->SUMMARY cascade like B3).

    Returns ``""`` when no canvas is present (no empty header -> the
    orchestrator's user-message prepend is skipped entirely -> byte-identical
    when ``--task-canvas`` is off or the gate created no active canvas).
    Mirrors ``format_scene_block`` / ``format_fade_block``. Best-effort by
    construction: a malformed canvas dict yields empty fields, never a raise
    (the orchestrator seam also wraps the call in a swallow)."""
    lines: list[str] = []
    for cv in canvases or []:
        cid = cv.get("canvas_id", "")
        label = cv.get("task_label", "")
        progress = cv.get("progress", 0)
        try:
            progress = int(progress)
        except (TypeError, ValueError):
            progress = 0
        mermaid = (cv.get("mermaid") or "").rstrip()
        lines.append(f"--- Task {cid} (label: {label}, progress: {progress}%) ---")
        if mermaid:
            lines.append(mermaid)
    if not lines:
        return ""
    header = ("[TASK CANVAS -- your active task-state flowchart]\n"
              "(structural short-term memory of the task in progress; phases: "
              "done|doing|paused|blocked. You may revise it via the update_canvas "
              "tool. Treat as the current task frame.)")
    block = header + "\n" + "\n\n".join(lines)
    # Truncate to the injection budget (len//4 token estimate, mirrors A4 +
    # the formatter). Inert for small canvases (default 1024 ~ 4096 chars).
    cap = max_tokens * 4
    if cap > 0 and len(block) > cap:
        block = block[:cap].rstrip() + "\n[... canvas truncated ...]"
    return block