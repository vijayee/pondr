"""LIFT CODETRACEBENCH INTENTS -- the real-intent corpus from raw trajectories.

WHY: every earlier use of this benchmark read the TEMPLATE goal
(`resolve {task} ({agent})`, built from task_name) and threw away where the
actual intent lives: `messages[1]`, the user message carrying the real
issue/PR description (repro + expected behaviour). This script lifts, per
trajectory:
  INTENT (goal grain)     : the <pr_description> issue text
  INTENT (declared plan)  : the agent's own plan declaration, when present
  INTENT (step grain)     : each step's THOUGHT block
  MEMORY units            : file paths mentioned in each step's command
  OUTCOME                 : verifier fields from report.json / info
  RUN identity            : instance_id + agent + model

OUTPUT (untracked, public-source derived):
  data/_scratch/icm/lifted/codetrace_intents.jsonl   one record per run
  data/_scratch/icm/lifted/gap_report.json           coverage + gaps, per field

GAP REPORT IS FIRST-CLASS (per the standing instruction): counts of runs
with issue text / plan / thoughts / files / outcome, step totals, distinct
files, shared-file units (the flip shape), per-agent breakdown, and every
parse failure counted, never silently dropped.

NOTHING here is authored: the intent statements are the benchmark's own
issue text; the outcomes are verifier fields. No mined/personal corpora.

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
OUTDIR = os.path.join("data", "_scratch", "icm", "lifted")
OUT = os.path.join(OUTDIR, "codetrace_intents.jsonl")
GAPS = os.path.join(OUTDIR, "gap_report.json")

FILE_PAT = re.compile(
    r"[A-Za-z0-9_./-]+\.(?:py|js|ts|tsx|jsx|rs|go|java|c|h|cc|cpp|md|json|ya?ml|toml|cfg|ini|txt)")
PR_DESC = re.compile(r"<pr_description>(.*?)</pr_description>", re.S)
THOUGHT = re.compile(r"THOUGHT:\s*(.*?)(?:\n\s*```|\Z)", re.S)
PLAN_CMD = re.compile(r'\{\s*"command"\s*:\s*"plan".*?\}\s*\]\s*\}', re.S)


def parse_traj(d) -> dict:
    msgs = d.get("messages") or []
    issue = ""
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            hit = PR_DESC.search(c)
            if hit:
                issue = hit.group(1).strip()
                break
    if not issue:                     # fallback: the first user message text
        for m in msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                issue = m["content"].strip()[:4000]
                break
    plan = ""
    steps = []
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, str):
            continue
        if m.get("role") == "assistant":
            if not plan:
                pm = PLAN_CMD.search(c)
                if pm:
                    plan = pm.group(0)[:2000]
            th = THOUGHT.search(c)
            files = []
            for fp in FILE_PAT.findall(c):
                if fp not in files:
                    files.append(fp)
            steps.append({"thought": (th.group(1).strip()[:600] if th else ""),
                          "command": c[:600],
                          "files": files[:12]})
    info = d.get("info") or {}
    return {"instance_id": d.get("instance_id"),
            "issue": issue, "plan": plan, "steps": steps,
            "n_steps": len(steps),
            "exit_status": info.get("exit_status"),
            "submitted": bool(info.get("submission"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    tars = sorted(os.listdir(TARS))[:args.limit or None]
    print(f"[lift] {len(tars)} tars", flush=True)
    t0 = time.time()
    gaps = {"runs": 0, "with_issue": 0, "with_plan": 0, "with_any_thought": 0,
            "with_files": 0, "with_exit_status": 0, "with_submission": 0,
            "parse_failures": 0, "steps_total": 0, "steps_with_thought": 0,
            "files_distinct": 0, "agents": collections.Counter(),
            "models": collections.Counter()}
    files_seen = collections.defaultdict(set)     # file -> run ids (flip shape)
    n_out = 0
    with io.open(OUT, "w", encoding="utf-8") as out:
        for i, name in enumerate(tars):
            path = os.path.join(TARS, name)
            try:
                with tarfile.open(path, "r:*") as tf:
                    traj = reprj = None
                    for mem in tf.getmembers():
                        try:
                            if mem.name.endswith(".traj.json"):
                                traj = json.loads(
                                    tf.extractfile(mem).read().decode("utf-8", "ignore"))
                            elif mem.name.endswith("report.json"):
                                reprj = json.loads(
                                    tf.extractfile(mem).read().decode("utf-8", "ignore"))
                        except Exception:
                            pass    # counted by the outer handler if nothing usable
            except Exception as e:
                gaps["parse_failures"] += 1
                print(f"  [fail] {name}: {type(e).__name__}", flush=True)
                continue
            if not isinstance(traj, dict):
                gaps["parse_failures"] += 1
                continue
            rec = parse_traj(traj)
            # verifier outcome: report.json carries the instance result map
            solved = None
            if isinstance(reprj, dict) and len(reprj) == 1:
                v = next(iter(reprj.values()))
                if isinstance(v, dict):
                    solved = v.get("resolved", v.get("success"))
            rec["solved"] = solved
            # tar naming: <agentcode>-<org>__<model>-<task>-<hash>.tar.zst
            parts = name.split("-")
            rec["agent"] = parts[0] if parts else "?"
            rec["model"] = parts[1] if len(parts) > 1 else "?"
            gaps["agents"][rec["agent"]] += 1
            gaps["models"][rec["model"]] += 1
            out.write(json.dumps(rec) + "\n")
            n_out += 1
            gaps["runs"] += 1
            gaps["with_issue"] += bool(rec["issue"])
            gaps["with_plan"] += bool(rec["plan"])
            gaps["with_any_thought"] += any(s["thought"] for s in rec["steps"])
            gaps["with_files"] += any(s["files"] for s in rec["steps"])
            gaps["with_exit_status"] += rec["exit_status"] is not None
            gaps["with_submission"] += rec["submitted"]
            gaps["steps_total"] += rec["n_steps"]
            gaps["steps_with_thought"] += sum(1 for s in rec["steps"] if s["thought"])
            for s in rec["steps"]:
                for fp in s["files"]:
                    files_seen[fp].add(rec["instance_id"])
            if (i + 1) % 40 == 0:
                print(f"  [{i+1}/{len(tars)}] runs={n_out} files={len(files_seen)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    shared = {f: v for f, v in files_seen.items() if len(v) >= 2}
    gaps["files_distinct"] = len(files_seen)
    gaps["shared_file_units"] = len(shared)
    gaps["shared_file_top"] = sorted(((f, len(v)) for f, v in shared.items()),
                                     key=lambda kv: -kv[1])[:10]
    gaps["elapsed_s"] = round(time.time() - t0)
    with io.open(GAPS, "w", encoding="utf-8") as f:
        f.write(json.dumps(gaps, indent=2, default=str))
    print(json.dumps(gaps, indent=1, default=str), flush=True)
    print(f"[done] {n_out} lifted runs -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
