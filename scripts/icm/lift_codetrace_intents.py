"""LIFT CODETRACEBENCH INTENTS -- real-intent corpus from the raw trajectories.

Every earlier use of this benchmark read the TEMPLATE goal
(`resolve {task} ({agent})`) and discarded where the intent actually lives.
This lift handles all three agent layouts in trajs3/ and emits, per run:

  INTENT (goal grain)    : the benchmark's own issue description
                           - miniswe  : messages[user] with <pr_description>
                           - openhands: first tensorblock call, user message
                                        with <issue_description>
                           - sweagent : trajectory[0].query user message
  INTENT (declared plan) : the agent's own plan tool-call (OpenHands only --
                           measured: SWE-agent and mini-SWE do not declare one)
  INTENT (step grain)    : each step's THOUGHT text
  MEMORY units           : file paths named in each step's action
  OUTCOME                : manifest `solved` (joined by artifact basename) +
                           info.exit_status / submission
  RUN identity           : instance_id, agent, model (parsed from the tar name)

OUTPUT (untracked; derived from the public benchmark):
  data/_scratch/icm/lifted/codetrace_intents.jsonl
  data/_scratch/icm/lifted/gap_report.json      coverage per field, failures
                                                 counted, shared-file units

GAP REPORT IS FIRST-CLASS: per-agent coverage, per-field counts, parse
failures, step totals, distinct files, shared-file units (the flip shape).

Run:  python scripts/icm/lift_codetrace_intents.py [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TARS = os.path.join("data", "_scratch", "codetrace", "trajs3")
MANIFEST = os.path.join("data", "_scratch", "codetrace", "bench_manifest.verified.jsonl")
OUTDIR = os.path.join("data", "_scratch", "icm", "lifted")
OUT = os.path.join(OUTDIR, "codetrace_intents.jsonl")
GAPS = os.path.join(OUTDIR, "gap_report.json")

# extension must END at a non-word boundary: without it, ".c" matched the
# first two chars of ".contrib" and produced junk units like "django.c"
FILE_PAT = re.compile(
    r"[A-Za-z0-9_./-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|cc|cpp|md|json|ya?ml|toml|cfg|ini|txt)(?![\w])"
    r"|[A-Za-z0-9_./-]+\.(?:c|h)(?![\w])")
ISSUE_PATS = (re.compile(r"<issue_description>(.*?)</issue_description>", re.S),
              re.compile(r"<pr_description>(.*?)</pr_description>", re.S))
THOUGHT = re.compile(r"THOUGHT:\s*(.*?)(?:\n\s*```|\Z)", re.S)


def text_of(content) -> str:
    """Message content may be a string or a list of {type:text} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for it in content:
            if isinstance(it, dict) and isinstance(it.get("text"), str):
                parts.append(it["text"])
            elif isinstance(it, str):
                parts.append(it)
        return "\n".join(parts)
    return ""


def extract_issue(text: str) -> str:
    for pat in ISSUE_PATS:
        m = pat.search(text or "")
        if m:
            return m.group(1).strip()[:6000]
    return ""


def files_in(text: str, cap: int = 12):
    seen = []
    for fp in FILE_PAT.findall(text or ""):
        fp = fp.strip("`'\"(),;")
        if len(fp) > 4 and fp not in seen:
            seen.append(fp)
        if len(seen) >= cap:
            break
    return seen


def parse_miniswe(d) -> dict:
    msgs = d.get("messages") or []
    issue = ""
    for m in msgs:
        issue = issue or extract_issue(text_of(m.get("content")))
    if not issue:
        for m in msgs:
            if m.get("role") == "user":
                issue = text_of(m.get("content"))[:4000]
                break
    plan, steps = "", []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = text_of(m.get("content"))
        if not plan and ("task_list" in c or '"plan"' in c):
            plan = c[:1200]
        th = THOUGHT.search(c)
        steps.append({"thought": th.group(1).strip()[:500] if th else "",
                      "command": c[:500], "files": files_in(c)})
    return {"issue": issue, "plan": plan, "steps": steps}


def parse_openhands(members, tf) -> dict:
    calls = sorted([m for m in members
                    if "tensorblock" in m.name and m.name.endswith(".json")],
                   key=lambda m: m.name)
    issue, plan, steps = "", "", []
    for idx, mem in enumerate(calls):
        try:
            data = json.loads(tf.extractfile(mem).read().decode("utf-8", "ignore"))
        except Exception:
            continue
        if idx == 0:
            for msg in (data.get("messages") or []):
                if msg.get("role") == "user":
                    issue = extract_issue(text_of(msg.get("content")))[:6000]
                    if not issue:
                        issue = text_of(msg.get("content"))[:4000]
                    break
        resp = data.get("response") or {}
        ch = (resp.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = text_of(msg.get("content"))
        args_all = " ".join(str(((tc.get("function") or {}).get("arguments") or ""))
                            for tc in (msg.get("tool_calls") or []))
        if not plan and ("task_list" in args_all or '"plan"' in args_all):
            plan = args_all[:1200]
        th = THOUGHT.search(content)
        steps.append({"thought": th.group(1).strip()[:500] if th else "",
                      "command": (content[:200] + " | " + args_all)[:500],
                      "files": files_in(content + " " + args_all)})
    return {"issue": issue, "plan": plan, "steps": steps}


def parse_sweagent(d) -> dict:
    traj = d.get("trajectory") or []
    issue = ""
    if traj:
        for msg in (traj[0].get("query") or []):
            if msg.get("role") == "user":
                raw = text_of(msg.get("content"))
                issue = extract_issue(raw) or raw[:6000]
                break
    steps = []
    for t in traj:
        act = str(t.get("action") or "")
        steps.append({"thought": str(t.get("thought") or "")[:500],
                      "command": act[:500], "files": files_in(act)})
    return {"issue": issue, "plan": "", "steps": steps}


def manifest_outcomes() -> dict:
    out = {}
    if not os.path.exists(MANIFEST):
        return out
    for l in io.open(MANIFEST, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        ap = os.path.basename(str(r.get("artifact_path") or ""))
        key = re.sub(r"\.(tar\.zst|tar|zst)$", "", ap)
        out[key] = {"solved": r.get("solved"), "task_name": r.get("task_name"),
                    "agent": r.get("agent"), "model": r.get("model")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    tars = sorted(os.listdir(TARS))[:args.limit or None]
    mout = manifest_outcomes()
    print(f"[lift] {len(tars)} tars | manifest outcomes: {len(mout)}", flush=True)
    t0 = time.time()
    gaps = {"runs": 0, "with_issue": 0, "with_plan": 0, "with_any_thought": 0,
            "with_files": 0, "with_outcome": 0, "parse_failures": 0,
            "steps_total": 0, "steps_with_thought": 0,
            "per_agent": collections.defaultdict(
                lambda: {"runs": 0, "issue": 0, "plan": 0, "thoughts": 0, "outcome": 0})}
    files_seen = collections.defaultdict(set)
    n_out = 0
    with io.open(OUT, "w", encoding="utf-8") as out:
        for i, name in enumerate(tars):
            path = os.path.join(TARS, name)
            key = re.sub(r"\.(tar\.zst|tar|zst)$", "", name)
            parsed = None
            try:
                with tarfile.open(path, "r:*") as tf:
                    members = tf.getmembers()
                    names = [m.name for m in members]
                    if any(n.endswith(".traj.json") for n in names):
                        mem = next(m for m in members if m.name.endswith(".traj.json"))
                        d = json.loads(tf.extractfile(mem).read().decode("utf-8", "ignore"))
                        parsed = parse_miniswe(d)
                    elif any("tensorblock" in n for n in names):
                        parsed = parse_openhands(members, tf)
                    elif any(n.endswith(".traj") for n in names):
                        mem = next(m for m in members if m.name.endswith(".traj"))
                        d = json.loads(tf.extractfile(mem).read().decode("utf-8", "ignore"))
                        parsed = parse_sweagent(d)
            except Exception:
                parsed = None
            if parsed is None:
                gaps["parse_failures"] += 1
                continue
            parts = key.split("-")
            agent = parts[0] if parts else "?"
            model = parts[1] if len(parts) > 1 else "?"
            mo = mout.get(key) or {}
            rec = {"run_id": key, "agent": agent, "model": model,
                   "issue": parsed["issue"], "plan": parsed["plan"],
                   "steps": parsed["steps"], "n_steps": len(parsed["steps"]),
                   "solved": mo.get("solved"), "task_name": mo.get("task_name"),
                   "manifest_agent": mo.get("agent"), "manifest_model": mo.get("model")}
            out.write(json.dumps(rec) + "\n")
            n_out += 1
            gaps["runs"] += 1
            pa = gaps["per_agent"][agent]
            pa["runs"] += 1
            gaps["with_issue"] += bool(rec["issue"]); pa["issue"] += bool(rec["issue"])
            gaps["with_plan"] += bool(rec["plan"]); pa["plan"] += bool(rec["plan"])
            any_th = any(s["thought"] for s in rec["steps"])
            gaps["with_any_thought"] += any_th; pa["thoughts"] += any_th
            gaps["with_files"] += any(s["files"] for s in rec["steps"])
            gaps["with_outcome"] += rec["solved"] is not None
            pa["outcome"] += rec["solved"] is not None
            gaps["steps_total"] += rec["n_steps"]
            gaps["steps_with_thought"] += sum(1 for s in rec["steps"] if s["thought"])
            for s in rec["steps"]:
                for fp in s["files"]:
                    files_seen[fp].add(rec["run_id"])
            if (i + 1) % 60 == 0:
                print(f"  [{i+1}/{len(tars)}] runs={n_out} ({time.time()-t0:.0f}s)", flush=True)
    shared = {f: v for f, v in files_seen.items() if len(v) >= 2}
    gaps["files_distinct"] = len(files_seen)
    gaps["shared_file_units"] = len(shared)
    gaps["shared_file_top"] = sorted(((f, len(v)) for f, v in shared.items()),
                                     key=lambda kv: -kv[1])[:10]
    gaps["per_agent"] = {k: dict(v) for k, v in gaps["per_agent"].items()}
    gaps["elapsed_s"] = round(time.time() - t0)
    with io.open(GAPS, "w", encoding="utf-8") as f:
        f.write(json.dumps(gaps, indent=2, default=str))
    print(json.dumps(gaps, indent=1, default=str), flush=True)
    print(f"[done] {n_out} lifted runs -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
