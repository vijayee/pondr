"""LIFT FILE LABELS -- per-(run, file) usefulness labels from the accepted patch.

WHAT THIS ADDS to the intent lift: a verifier-grounded per-FILE label, so the
integration test has a mechanical usefulness signal at the memory-unit grain
(not just run-level solved, and not the 1.2%-coverage human step labels).

LABEL SEMANTICS (rule-computed, never stated):
  useful    = file appears in the run's final patch AND the run was solved
              (the accepted fix carried this file)
  attempted = file appears in the final patch AND the run was NOT solved
              (a fix site that failed verification)
  context   = file touched by steps but absent from the patch
              (exploration / reading, no change)
Patches come from the tars (patch.diff / .patch / .pred / SWE-agent
info.submission); paths are NORMALISED (strip ./, /, /testbed/) on both the
patch side and the step side so the two join.

OUTPUT (untracked):
  data/_scratch/icm/lifted/codetrace_files.jsonl   one record per (run, file)
      {run_id, agent, task_name, solved, file, label, touched_steps,
       cmd_text (<=2000), patch_text (<=3000), patch_len}
  data/_scratch/icm/lifted/file_label_report.json  label distribution, patch
      coverage per agent, normalisation collapse counts, and the
      evaluation-ready unit counts (files with >=2 runs; mixed-outcome units).

Run:  python scripts/icm/lift_file_labels.py
"""

from __future__ import annotations

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
LIFTED = os.path.join("data", "_scratch", "icm", "lifted", "codetrace_intents.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "lifted", "codetrace_files.jsonl")
REP = os.path.join("data", "_scratch", "icm", "lifted", "file_label_report.json")

DIFF_FILE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.M)
PLUS_FILE = re.compile(r"^\+\+\+ b/(.+)$", re.M)


def norm(p: str) -> str:
    p = str(p).strip().strip("`'\"(),;")
    for pre in ("./", "/testbed/", "/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p.strip("/")


def split_diff(text: str) -> dict:
    """file -> its diff block text (capped)."""
    out = {}
    blocks = re.split(r"(?m)^(?=diff --git )", text or "")
    for b in blocks:
        m = DIFF_FILE.search(b)
        if not m:
            fm = PLUS_FILE.search(b)
            if not fm:
                continue
            f = fm.group(1)
        else:
            f = m.group(2)
        f = norm(f)
        if f and f not in out:
            out[f] = b[:3000]
    return out


def patch_of(tf, members, agent) -> str:
    names = [m.name for m in members]
    def read(mem):
        try:
            return tf.extractfile(mem).read().decode("utf-8", "ignore")
        except Exception:
            return ""
    for suffix in ("patch.diff", ".patch"):
        for m in members:
            if m.name.endswith(suffix):
                t = read(m)
                if t.strip():
                    return t
    for m in members:
        if m.name.endswith(".pred"):
            t = read(m)
            if t.strip():
                return t
    # SWE-agent stores the submission inside .traj info
    for m in members:
        if m.name.endswith(".traj"):
            try:
                d = json.loads(read(m))
                sub = (d.get("info") or {}).get("submission")
                if isinstance(sub, str) and sub.strip():
                    return sub
            except Exception:
                pass
    return ""


def main() -> int:
    recs = [json.loads(l) for l in io.open(LIFTED, encoding="utf-8") if l.strip()]
    by_run = {r["run_id"]: r for r in recs}
    print(f"[labels] {len(recs)} lifted runs", flush=True)
    tars = {re.sub(r"\.(tar\.zst|tar|zst)$", "", n): os.path.join(TARS, n)
            for n in os.listdir(TARS)}
    rep = {"runs": 0, "runs_with_patch": 0, "patch_files_total": 0,
           "labels": collections.Counter(), "per_agent": collections.defaultdict(
               lambda: {"runs": 0, "with_patch": 0, "files": 0}),
           "dropped_no_steps": 0}
    file_runs = collections.defaultdict(dict)     # file -> run_id -> label
    t0 = time.time()
    n = 0
    with io.open(OUT, "w", encoding="utf-8") as out:
        for rid, r in by_run.items():
            path = tars.get(rid)
            patch_files = {}
            if path and os.path.exists(path):
                try:
                    with tarfile.open(path, "r:*") as tf:
                        members = tf.getmembers()
                        ptxt = patch_of(tf, members, r["agent"])
                        patch_files = split_diff(ptxt)
                except Exception:
                    patch_files = {}
            # touched files (normalised) with their step indices + command text
            touched = collections.defaultdict(lambda: {"steps": [], "cmds": []})
            for si, s in enumerate(r["steps"], start=1):
                for f in s["files"]:
                    fn = norm(f)
                    if not fn:
                        continue
                    touched[fn]["steps"].append(si)
                    if s["command"]:
                        touched[fn]["cmds"].append(s["command"])
            if not touched and not patch_files:
                rep["dropped_no_steps"] += 1
                continue
            rep["runs"] += 1
            pa = rep["per_agent"][r["agent"]]
            pa["runs"] += 1
            has_patch = bool(patch_files)
            rep["runs_with_patch"] += has_patch
            pa["with_patch"] += has_patch
            rep["patch_files_total"] += len(patch_files)
            for f in sorted(set(touched) | set(patch_files)):
                in_patch = f in patch_files
                solved = r.get("solved")
                label = ("useful" if (in_patch and solved) else
                         "attempted" if in_patch else
                         "context")
                rep["labels"][label] += 1
                pa["files"] += 1
                file_runs[f][rid] = label
                rec = {"run_id": rid, "agent": r["agent"], "task_name": r.get("task_name"),
                       "solved": solved, "file": f, "label": label,
                       "touched_steps": touched.get(f, {}).get("steps", []),
                       "cmd_text": " ".join(touched.get(f, {}).get("cmds", []))[:2000],
                       "patch_text": patch_files.get(f, "")[:3000],
                       "patch_len": len(patch_files.get(f, ""))}
                out.write(json.dumps(rec) + "\n")
                n += 1
            if (n and rep["runs"] % 60 == 0):
                print(f"  [runs={rep['runs']}] files_out={n} ({time.time()-t0:.0f}s)", flush=True)
    units = {f: v for f, v in file_runs.items() if len(v) >= 2}
    mixed = {}
    for f, v in units.items():
        labels = set(v.values())
        if "useful" in labels and ("attempted" in labels or "context" in labels):
            mixed[f] = len(v)
    rep["files_distinct"] = len(file_runs)
    rep["units_multi_run"] = len(units)
    rep["units_with_useful_and_nonuseful"] = len(mixed)
    rep["top_units"] = sorted(mixed.items(), key=lambda kv: -kv[1])[:10]
    rep["per_agent"] = {k: dict(v) for k, v in rep["per_agent"].items()}
    rep["labels"] = dict(rep["labels"])
    rep["elapsed_s"] = round(time.time() - t0)
    with io.open(REP, "w", encoding="utf-8") as f:
        f.write(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str), flush=True)
    print(f"[done] {n} (run,file) records -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
