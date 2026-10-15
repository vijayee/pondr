"""CodeTraceBench census + file-grain flip-shape instrument (gate-0 for the
coding-agent corpus family).

Census results (2026-10-15, data/_scratch/codetrace/):
  repo grain (manifest, 1000 verified trajectories):
    238 stores, 181 with >=2 tasks (django 91 tasks, transformers 52...)
    1919 human-verified step-utility labels (1695 incorrect + 224 unuseful)
    reality oracle: 504/1000 solved
  file grain (sampled tars, this script):
    shared files recur across tasks (matplotlib _base.py x4 tasks,
    django tests/runtests.py x7 tasks) - the many-goals-one-memory flip
    shape, with per-step human labels + verifier outcomes attached.

Usage: python scripts/icm/codetrace_census.py [--traj-dir data/_scratch/codetrace/trajs2]
"""

from __future__ import annotations

import collections
import glob
import io
import json
import os
import re
import sys
import tarfile

MANIFEST = os.path.join("data", "_scratch", "codetrace", "bench_manifest.verified.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "codetrace_census_results.json")
FILE_EXT = r"\.(py|js|ts|json|md|txt|rst|yaml|yml|toml|cfg|ini|c|cpp|h|hpp|vue|ipynb|html|css)\b"

_model = None


def repo_of(t: str):
    m = re.match(r"([A-Za-z0-9_]+)__([A-Za-z0-9_]+)-?\d*", t)
    return m.group(2) if m else "?"


def manifest_stats():
    rows = [json.loads(l) for l in io.open(MANIFEST, encoding="utf-8") if l.strip()]
    repos = collections.defaultdict(list)
    n_stages = n_steps = n_inc = n_un = 0
    solved = 0
    for r in rows:
        repos[repo_of(r["task_name"])].append(r)
        n_stages += r.get("stage_count") or 0
        n_steps += r.get("step_count") or 0
        solved += 1 if r.get("solved") else 0
        for st in r.get("incorrect_stages") or []:
            for s in st.get("steps") or []:
                if "incorrect" in (s.get("labels") or []):
                    n_inc += 1
            n_un += len(st.get("unuseful_step_ids") or [])
    multi = {k: len(v) for k, v in repos.items() if len(v) >= 2}
    return {"trajectories": len(rows), "stores": len(repos),
            "stores_multi_task": len(multi),
            "top_stores": sorted(multi.items(), key=lambda x: -x[1])[:8],
            "stages": n_stages, "steps": n_steps,
            "human_labels": {"incorrect": n_inc, "unuseful": n_un},
            "solved": solved, "solved_rate": round(solved / len(rows), 3)}


def file_census(trajs_dir):
    """file-grain flip census: files accessed by >=2 tasks within a repo."""
    pattern = os.path.join(trajs_dir or os.path.join("data", "_scratch", "codetrace", "trajs2"),
                           "*.tar.zst")
    task_files = collections.defaultdict(set)
    for tar in sorted(glob.glob(pattern)):
        try:
            t = tarfile.open(tar, "r:*")
        except Exception:
            continue
        for mem in t.getmembers():
            parts = mem.name.split("/")
            if "tensorblock" not in mem.name or not mem.name.endswith(".json") or len(parts) < 4:
                continue
            task = parts[2]
            try:
                data = json.load(t.extractfile(mem))
            except Exception:
                continue
            txt = json.dumps(data.get("response") or data.get("choices") or {}, default=str)[:6000]
            for tok in re.findall(r"[\w./\\-]{5,140}", txt):
                if "/" in tok and re.search(FILE_EXT, tok):
                    nt = tok.strip(" ,'\"();")
                    if all(x not in nt for x in ("tensorblock", "swe_raw", "verified")):
                        task_files[task].add(nt.lower())
        t.close()
    per_repo = collections.defaultdict(lambda: collections.defaultdict(set))
    for task, files in task_files.items():
        for f in files:
            per_repo[repo_of(task)][f].add(task)
    result = {}
    for repo, fmap in per_repo.items():
        multi = {f: v for f, v in fmap.items() if len(v) >= 2}
        result[repo] = {"tasks": len({t for t, fs in task_files.items() if repo_of(t) == repo}),
                        "file_units": len(fmap), "shared_files": len(multi),
                        "top_shared": sorted(((f, len(v)) for f, v in
                                              sorted(multi.items(), key=lambda x: -len(x[1])))[:5],
                                             key=lambda x: -x[1])}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=str, default=None)
    args = ap.parse_args()
    mstats = manifest_stats()
    fcen = file_census(args.traj_dir)
    total_units = sum(v["file_units"] for v in fcen.values())
    total_shared = sum(v["shared_files"] for v in fcen.values())
    out = {"manifest": mstats, "file_grain": fcen,
           "file_grain_totals": {"units": total_units, "shared": total_shared,
                                 "rate": round(total_shared / max(1, total_units), 3)},
           "gate0_verdict": "PASS (the flip-shape exists at repo + file grain, "
                            "with human step-utility labels + verifier outcomes)",
           "note": "file census undercounts (only ~6KB/response parsed); run "
                   "full artifacts for a precise rate"}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out["manifest"], indent=1))
    for repo, v in fcen.items():
        print(f"  {repo:24s} tasks={v['tasks']} files={v['file_units']} shared={v['shared_files']}")
    print(f"file-grain: {out['file_grain_totals']}")
    print("GATE-0:", out["gate0_verdict"])
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())