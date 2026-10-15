"""Mine agent session transcripts into capture-shaped records.

Source: Claude Code session JSONLs (~/.claude/projects/<proj>/<uuid>.jsonl)
records: {"type": "user"|"assistant", "message": {"content": [...]},
"timestamp": iso, "toolUseResult": ...}

The mined hierarchy (per the agreed protocol, addendum 7):
  goal   = the session's FIRST user directive (+ its text); later user
           messages that arrive after long work spans = step-level goal
           refinements (user refinement = the agent's real goal updates).
  step   = each assistant work batch: its text narration = the step's
           purpose line; branchless (the transcript rarely states expected
           branches -> prediction comes only from explicit statements).
  retrievals = tool_use blocks of retrieval-class tools (Grep/Glob/
           WebFetch/Task/Read-on-files) attached as retrieval events.
  outcomes = the tool RESULTS (not valuations): a retrieval event that
           produced non-error output = observed "desired"-completion
           ("label_basis": "completion"). Prediction-congruence labels
           ("label_basis": "congruence") are possible ONLY where the
           transcript states a pre-registration/expected outcome.

LAWS: the miner NEVER states a congruence label (the rule computes labels
only for explicitly declared branch-predictions). Completion marks are
OBSERVATIONS of the record, and every scored row carries label_basis so the
two label types can never silently mix.

Usage:
  python scripts/icm/mine_agent_sessions.py --file <session.jsonl> \
      [--out <capture.jsonl>] [--limit-lines N]
Output shape matches data/_scratch/icm/capture/session_capture.jsonl records
plus {label_basis} on outcome rows.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import uuid

RETRIEVAL_TOOLS = {"Grep", "Glob", "Task", "WebFetch", "WebSearch", "Read",
                   "ListAgents", "Agent", "Monitor"}
WORK_TOOLS_ALL = RETRIEVAL_TOOLS | {"Bash", "PowerShell", "Edit", "Write",
                                    "NotebookEdit"}


def parse_line(line: str):
    try:
        rec = json.loads(line)
    except Exception:
        return None
    if rec.get("type") not in ("user", "assistant"):
        return None
    return rec


def blocks(msg):
    c = (msg or {}).get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return c or []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--limit-lines", type=int, default=0, help="smoke: cap input lines")
    args = ap.parse_args()
    out_path = args.out or os.path.join(
        "data", "_scratch", "icm", "capture",
        os.path.basename(args.file).replace(".jsonl", "_mined.jsonl"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    goal = None
    goal_n = 0
    rows = []
    pending_tools = []   # tool_use ids awaiting results
    observed_results = {}

    def emit(rec):
        rows.append(rec)

    with io.open(args.file, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if args.limit_lines and i >= args.limit_lines:
                break
            rec = parse_line(line)
            if rec is None:
                continue
            ts = rec.get("timestamp", "")
            kind = rec["type"]
            b = blocks(rec.get("message") or {})
            texts = [bl.get("text", "") for bl in b if bl.get("type") == "text"]

            if kind == "user":
                tool_results = [bl for bl in b if bl.get("type") == "tool_result"]
                if tool_results:
                    for tr in tool_results:
                        tid = tr.get("tool_use_id")
                        c = tr.get("content")
                        ok = not (isinstance(c, dict) and c.get("is_error"))
                        observed_results[tid] = ok
                    continue
                u_text = " ".join(texts).strip()
                if not u_text or u_text.startswith("<"):
                    continue
                # user directives: first = goal; later = goal refinement (step)
                if goal is None:
                    goal_n += 1
                    goal = {"kind": "icapture", "level": "goal", "parent": None,
                            "id": f"mgoal-{os.path.basename(args.file)[:8]}-{goal_n}",
                            "text": u_text[:400], "branches": {"desired": None},
                            "declared_at": ts,
                            "label_basis": "completion (mined)"}
                    emit(goal)
                else:
                    emit({"kind": "icapture", "level": "step", "parent": goal["id"],
                          "id": f"mstep-{uuid.uuid4().hex[:8]}", "text": u_text[:400],
                          "branches": {"desired": None},
                          "declared_at": ts,
                          "label_basis": "completion (mined)"})
            elif kind == "assistant":
                for bl in b:
                    if bl.get("type") == "tool_use":
                        name = bl.get("name", "?")
                        pending_tools.append(bl.get("id"))
                        if name in RETRIEVAL_TOOLS:
                            q = ""
                            inp = bl.get("input") or {}
                            for k in ("prompt", "pattern", "query", "url", "file_path"):
                                if inp.get(k):
                                    q = str(inp[k])[:200]
                                    break
                            emit({"kind": "icapture", "level": "retrieval",
                                  "parent": "mined-step-unlinked",
                                  "id": f"mretr-{uuid.uuid4().hex[:8]}",
                                  "query": q or name, "retrieved": [name],
                                  "declared_at": ts,
                                  "tool_use_id": bl.get("id"),
                                  "label_basis": "completion (mined)"})
    # link outcomes: each retrieval event -> its tool result as a completion
    # observation (NO stated label -- the observed fact only)
    for r in rows:
        if r["level"] == "retrieval":
            tid = r.get("tool_use_id")
            ok = observed_results.get(tid)
            emit({"kind": "icapture", "level": "outcome", "parent": r["id"],
                  "id": f"moutc-{uuid.uuid4().hex[:8]}",
                  "observed": "default" if ok else "other",
                  "note": "tool-result completion observation (mined)",
                  "declared_at": r["declared_at"],
                  "label_basis": "completion (mined)"})
    with io.open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")
    levels = {}
    for r in rows:
        levels[r["level"]] = levels.get(r["level"], 0) + 1
    print(f"[mined] {len(rows)} records from {os.path.basename(args.file)}")
    print(f"  levels: {levels}")
    print(f"  -> appended {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())