"""Lift the user's Claude Code sessions into ICM intent turns + memory events.

WHY: the wall is a corpus where the SAME memory unit appears under DIFFERENT
intents. The local transcripts carry exactly that raw material, unlike chat:
  intent   = the user's directive at the start of a turn (declared, not mined
             from prose); a terse directive may be lifted with the agent's own
             first plan sentence, marked in `intent_source`;
  memory   = file paths from Read / Edit / Write / NotebookEdit tool calls;
  outcome  = the tool result (`is_error`, stdout/stderr, interrupted) and the
             structured patch -- MECHANICAL, not a judge.

WHAT THIS PASS DOES (pre-registered BEFORE running):
  1. PARSE all transcripts under ~/.claude/projects into turns and events.
  2. CENSUS: sessions, substantive turns, file units, shared units, shared
     EDITED units, verification (test) turns and their mechanical success.
  3. OPPOSITION SAMPLE: for shared edited units, pair the turns that touched
     them and label a seeded sample with the addendum-16 judge prompt
     (SAME / DIFFERENT / OPPOSED). `--relations N` (default 0) sets the size.
     This is an ESTIMATE of whether the flip shape exists; it is not a
     retrieval result.
  4. Write JSONL under data/_scratch/icm/lifted/claude/ (UNTracked -- personal
     text; never commit). Code stays tracked and contains no chat text.

BAR (census, not a claim): the corpus is worth building on only if shared
EDITED units exist in quantity AND an appreciable fraction of sampled pairs is
OPPOSED. If the opposed rate is ~0, this corpus fails the same census as
CodeTraceBench (0/96) and Onyx (0/6336), and must be reported as such.

Labels never come from the judge: `useful` is defined mechanically below
(edited with a successful or absent verification), and the judge only labels
the RELATION between two intents for the census.

Usage:
  python scripts/icm/lift_claude_sessions.py                 # census only
  python scripts/icm/lift_claude_sessions.py --relations 120 # + judge sample
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUTDIR = os.path.join("data", "_scratch", "icm", "lifted", "claude")
TURNS = os.path.join(OUTDIR, "claude_turns.jsonl")
EVENTS = os.path.join(OUTDIR, "claude_events.jsonl")
CENSUS = os.path.join(OUTDIR, "claude_census.json")
RELATIONS = os.path.join(OUTDIR, "claude_relations.jsonl")
JUDGE_CACHE = os.path.join(OUTDIR, "claude_judge_cache.jsonl")

READ_TOOLS = {"Read", "Grep", "Glob"}
EDIT_TOOLS = {"Edit", "Write", "NotebookEdit"}
TEST_RE = re.compile(
    r"(pytest|python -m pytest|npm (run )?test|pnpm test|yarn test|"
    r"cargo test|go test|make test|tox\b|unittest|ctest|dotnet test|"
    r"mvn test|gradle test)", re.I)
TERSE_RE = re.compile(r"^(continue|proceed|go on|keep going|yes|ok|okay|"
                      r"what'?s next|next|do it|go ahead|sounds good)[.! ]*$", re.I)


def parse_ts(value):
    return str(value or "")


def norm_path(path, cwd):
    p = str(path or "").replace("\\", "/").strip()
    if not p:
        return ""
    if os.path.isabs(p) and cwd:
        try:
            rel = os.path.relpath(p, cwd).replace("\\", "/")
            return rel if not rel.startswith("..") else p
        except Exception:
            return p
    return p[2:] if p.startswith("./") else p


def is_substantive(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 40:
        return False
    if TERSE_RE.match(t):
        return False
    return True


def classify_intent(user_text: str, plan: str) -> tuple[str, str]:
    """Return (intent_text, source). Terse directives borrow the agent's plan."""
    u = " ".join(str(user_text or "").split())
    p = " ".join(str(plan or "").split())
    if is_substantive(u):
        return u[:1200], "user"
    if p:
        return (u + " || agent plan: " + p)[:1200], "user+agent-plan"
    return u[:1200], "user-terse"


def parse_file(path: str, root: str) -> list[dict]:
    """One transcript -> (turns, events). See the pre-registered rules above."""
    turns: list[dict] = []
    events: list[dict] = []
    session_id = os.path.basename(path).replace(".jsonl", "")
    rel = os.path.relpath(path, root).replace("\\", "/").split("/")
    project = rel[0] if len(rel) > 1 else "unknown"
    cwd = ""
    cur = None
    plan = ""
    turn_by_id: dict[str, dict] = {}
    tool_meta: dict[str, dict] = {}   # tool_use_id -> {turn, tool, path, command}
    n_turn = 0
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rtype = rec.get("type")
            if cwd == "" and rec.get("cwd"):
                cwd = str(rec.get("cwd"))
            if rtype == "user":
                content = (rec.get("message") or {}).get("content")
                # tool results arrive as user records with a tool_result list
                if isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in content):
                    for b in content:
                        if not isinstance(b, dict) or b.get("type") != "tool_result":
                            continue
                        meta = tool_meta.get(b.get("tool_use_id"))
                        if meta is None:
                            continue
                        tur = rec.get("toolUseResult")
                        tur = tur if isinstance(tur, dict) else {}
                        ev = {
                            "session_id": session_id, "project": project,
                            "turn_id": meta["turn"], "tool": meta["tool"],
                            "path": meta["path"], "command": meta["command"],
                            "ok": not bool(b.get("is_error", False)),
                            "is_error_known": "is_error" in b,
                            "interrupted": bool(tur.get("interrupted", False)),
                            "stdout_chars": len(str(tur.get("stdout", ""))),
                            "stderr_chars": len(str(tur.get("stderr", ""))),
                            "has_patch": bool(tur.get("structuredPatch")),
                            "user_modified": bool(tur.get("userModified", False)),
                            "test_command": bool(meta.get("test_command")),
                            "ts": parse_ts(rec.get("timestamp")),
                        }
                        events.append(ev)
                        t = turn_by_id[meta["turn"]]
                        if meta["path"]:
                            if meta["tool"] in EDIT_TOOLS:
                                t["files_edited"].add(meta["path"])
                                if not ev["ok"]:
                                    t["edit_failed"].add(meta["path"])
                            elif meta["tool"] in READ_TOOLS:
                                t["files_read"].add(meta["path"])
                        if ev["test_command"]:
                            t["tests"].append(ev["ok"])
                    continue
                # a real user directive starts a turn
                text = content if isinstance(content, str) else " ".join(
                    b.get("text", "") for b in (content or [])
                    if isinstance(b, dict) and b.get("type") == "text")
                text = " ".join(str(text or "").split())
                if not text or text.startswith("<"):
                    continue
                n_turn += 1
                cur = {
                    "session_id": session_id, "project": project, "turn_id": "",
                    "n": n_turn, "user_text": text[:2000],
                    "intent_text": "", "intent_source": "",
                    "plan": "", "ts": parse_ts(rec.get("timestamp")),
                    "is_sidechain": bool(rec.get("isSidechain")),
                    "files_read": set(), "files_edited": set(),
                    "edit_failed": set(), "tests": [],
                }
                cur["turn_id"] = f"{session_id}-t{n_turn:03d}"
                turns.append(cur)
                turn_by_id[cur["turn_id"]] = cur
                plan = ""
            elif rtype == "assistant" and cur is not None:
                for b in ((rec.get("message") or {}).get("content") or []):
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text" and not plan:
                        plan = " ".join(str(b.get("text", "")).split())[:600]
                        cur["plan"] = plan
                    elif b.get("type") == "tool_use":
                        inp = b.get("input") or {}
                        name = str(b.get("name", "?"))
                        p = norm_path(inp.get("file_path") or inp.get("path"), cwd)
                        cmd = " ".join(str(inp.get("command", "")).split())[:300]
                        tool_meta[b.get("id")] = {
                            "turn": cur["turn_id"], "tool": name, "path": p,
                            "command": cmd,
                            "test_command": name in ("Bash", "PowerShell")
                            and bool(TEST_RE.search(cmd)),
                        }
    # finalise intents after the turn is complete (plan may arrive late)
    for t in turns:
        t["intent_text"], t["intent_source"] = classify_intent(
            t["user_text"], t["plan"])
        t["substantive"] = is_substantive(t["user_text"])
        t["files_read"] = sorted(t["files_read"])
        t["files_edited"] = sorted(t["files_edited"])
        t["edit_failed"] = sorted(t["edit_failed"])
        # end-state verdict: the LAST test run in the turn decides
        t["test_ok"] = (t["tests"][-1] if t["tests"] else None)
        t["test_runs"] = len(t["tests"])
    return turns, events


def file_label(turn: dict, file: str) -> str:
    """Mechanical per-(turn,file) label. Never a judge."""
    if file in turn["files_edited"]:
        if file in turn["edit_failed"] or turn["test_ok"] is False:
            return "attempted"
        return "useful"
    return "context"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--relations", type=int, default=0,
                    help="judge-label up to N shared-edit intent pairs (0=census only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-file", type=int, default=6,
                    help="cap pairs contributed by one shared file")
    args = ap.parse_args()

    import glob
    files = sorted(glob.glob(os.path.join(args.root, "**", "*.jsonl"),
                             recursive=True))
    print(f"[scan] {len(files)} transcripts under {args.root}", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    all_turns: list[dict] = []
    all_events: list[dict] = []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        try:
            turns, events = parse_file(path, args.root)
        except Exception as e:
            print(f"  [parse-fail] {os.path.basename(path)}: {e}", flush=True)
            continue
        all_turns += turns
        all_events += events
        if i % 10 == 0:
            print(f"  [{i}/{len(files)}] turns={len(all_turns)} "
                  f"events={len(all_events)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- census
    by_project: dict[str, dict] = {}
    for t in all_turns:
        p = by_project.setdefault(t["project"], {"sessions": set(), "turns": 0,
                                                 "substantive": 0, "edits": 0})
        p["sessions"].add(t["session_id"])
        p["turns"] += 1
        p["substantive"] += 1 if t["substantive"] else 0
        p["edits"] += 1 if t["files_edited"] else 0
    units: dict[str, list[dict]] = {}
    edit_units: dict[str, list[dict]] = {}
    for t in all_turns:
        for f in t["files_read"]:
            units.setdefault(f, []).append(t)
        for f in t["files_edited"]:
            units.setdefault(f, []).append(t)
            edit_units.setdefault(f, []).append(t)
    shared = {f: v for f, v in units.items() if len(v) >= 2}
    shared_edit = {f: v for f, v in edit_units.items() if len(v) >= 2}

    # pair turns over shared edited units (same turn may repeat; dedupe pairs)
    pairs: dict[tuple[str, str, str], dict] = {}
    for f, ts in shared_edit.items():
        ts = sorted(ts, key=lambda x: (x["ts"], x["turn_id"]))
        seen = 0
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                if a["turn_id"] == b["turn_id"]:
                    continue
                key = (a["turn_id"], b["turn_id"], f)
                if key in pairs:
                    continue
                pairs[key] = {"file": f, "a": a, "b": b}
                seen += 1
                if seen >= args.max_per_file:
                    break
            if seen >= args.max_per_file:
                break

    test_turns = [t for t in all_turns if t["tests"]]
    census = {
        "transcripts": len(files),
        "sessions": len({t["session_id"] for t in all_turns}),
        "turns": len(all_turns),
        "substantive_turns": sum(1 for t in all_turns if t["substantive"]),
        "sidechain_turns": sum(1 for t in all_turns if t["is_sidechain"]),
        "intent_sources": {},
        "tool_events": len(all_events),
        "tool_errors": sum(1 for e in all_events if not e["ok"]),
        "test_turns": len(test_turns),
        "test_turns_ok": sum(1 for t in test_turns if t["test_ok"]),
        "edited_file_instances": sum(len(t["files_edited"]) for t in all_turns),
        "read_file_instances": sum(len(t["files_read"]) for t in all_turns),
        "distinct_files": len(units),
        "distinct_edited_files": len(edit_units),
        "shared_files": len(shared),
        "shared_edited_files": len(shared_edit),
        "shared_edit_pairs": len(pairs),
        "by_project": {p: {"sessions": len(v["sessions"]), "turns": v["turns"],
                           "substantive": v["substantive"], "edit_turns": v["edits"]}
                       for p, v in by_project.items()},
        "elapsed_s": round(time.time() - t0),
    }
    for t in all_turns:
        census["intent_sources"][t["intent_source"]] = \
            census["intent_sources"].get(t["intent_source"], 0) + 1

    # ---- write the lifted records (untracked)
    with io.open(TURNS, "w", encoding="utf-8", newline="\n") as f:
        for t in all_turns:
            f.write(json.dumps({
                "turn_id": t["turn_id"], "session_id": t["session_id"],
                "project": t["project"], "n": t["n"],
                "intent_text": t["intent_text"], "intent_source": t["intent_source"],
                "substantive": t["substantive"], "ts": t["ts"],
                "is_sidechain": t["is_sidechain"],
                "files_read": t["files_read"], "files_edited": t["files_edited"],
                "edit_failed": t["edit_failed"],
                "test_ok": t["test_ok"], "test_runs": t["test_runs"],
                "file_labels": {x: file_label(t, x)
                                for x in sorted(set(t["files_read"]) |
                                                set(t["files_edited"]))},
            }, sort_keys=True, ensure_ascii=True) + "\n")
    with io.open(EVENTS, "w", encoding="utf-8", newline="\n") as f:
        for e in all_events:
            f.write(json.dumps(e, sort_keys=True, ensure_ascii=True) + "\n")

    # ---- optional opposition sample
    rel_report = None
    if args.relations > 0 and pairs:
        from icm.intent_keys import IntentKeyer
        rng = random.Random(args.seed)
        chosen = rng.sample(sorted(pairs.values(), key=lambda p: p["file"] + p["a"]["turn_id"]),
                            min(args.relations, len(pairs)))
        keyer = IntentKeyer(cache_path=JUDGE_CACHE)
        counts: dict[str, int] = {}
        rows = []
        for i, p in enumerate(chosen, 1):
            rel = keyer.relation_between(p["a"]["intent_text"], p["b"]["intent_text"])
            rel = rel or "unparsed"
            counts[rel] = counts.get(rel, 0) + 1
            rows.append({"file": p["file"],
                         "a_turn": p["a"]["turn_id"], "b_turn": p["b"]["turn_id"],
                         "relation": rel})
            if i % 10 == 0:
                print(f"  [judge {i}/{len(chosen)}] {json.dumps(counts)}", flush=True)
        with io.open(RELATIONS, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        rel_report = {"n": len(rows), "counts": counts,
                      "opposed_rate": round(counts.get("opposed", 0) / len(rows), 3),
                      "sample_seed": args.seed}

    census["relations"] = rel_report
    with io.open(CENSUS, "w", encoding="utf-8") as f:
        f.write(json.dumps(census, indent=2, default=str))
    print(json.dumps(census, indent=1, default=str), flush=True)
    print(f"[done] -> {OUTDIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
