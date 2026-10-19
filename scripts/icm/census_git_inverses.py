"""Git-history inverse census: real counterfactual pairs, patch-verified.

WHY: the strong thesis needs the same memory unit under two aims whose
correctness swaps. Local git history is an ORGANIC producer: the memory units
are real file snapshots, the oracle is the file content at a commit (mechanical,
no judge), and the intents are real commit messages. A later commit that
REMOVES lines an earlier commit ADDED (or the reverse) is a genuine
counterfactual pair over that file.

This pass verifies at the PATCH level (message-level opposites are noisy):
  candidate = two commits touching the same file whose messages contain an
              add/enable/pin word in one and a remove/disable/unpin/revert word
              in the other (seeded sample);
  inverse   = the later commit's removed lines overlap the earlier commit's
              added lines (or the later's added lines overlap the earlier's
              removed), at >= max(2, 50% of the smaller side) non-trivial lines.

PRE-REGISTERED (before running):
  (1) sample up to --sample candidate pairs, seeded (default 120, seed 42);
  (2) report candidates, verified inverses, rate with Wilson 95% CI, by repo;
  (3) census only -- no retrieval claim, no keys, no judge. A low verified rate
      means the organic corpus is too thin at this source and synthetic
      stress data is the fallback; a healthy rate means the real-corpus test
      is buildable from git and does NOT need an authored world.

Usage:
  python scripts/icm/census_git_inverses.py --sample 120
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUT = os.path.join("data", "_scratch", "icm", "lifted", "git_inverse_census.json")
REPOS_DEFAULT = ["Pondr", "HPMT", "liboffs", "WaveDB"]
ADD_RE = re.compile(r"\b(add|adds|added|enable|enables|introduce|pin|pins|restore|re-enable)\b", re.I)
REM_RE = re.compile(r"\b(remove|removes|removed|delete|deletes|disable|disables|drop|drops|unpin|unpins|strip|revert|reverts)\b", re.I)


def git(repo, *args, timeout=120):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                       text=True, errors="replace", timeout=timeout)
    return p.stdout


def commits_by_file(repo):
    out = git(repo, "log", "--all", "--no-merges", "--format=@@%H|%s", "--name-only")
    commits, cur = [], None
    for line in out.splitlines():
        if line.startswith("@@"):
            h, m = line[2:].split("|", 1)
            cur = {"h": h, "m": m, "files": []}
            commits.append(cur)
        elif line.strip() and cur is not None:
            cur["files"].append(line.strip())
    by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for c in commits:
        for f in c["files"]:
            by_file[f].append(c)
    return by_file


def patch_lines(repo, commit, path):
    """Added/removed non-trivial lines for one commit on one file."""
    out = git(repo, "show", commit, "--format=", "--unified=0", "--", path)
    added, removed = set(), set()
    for line in out.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            s = line[1:].strip()
        elif line.startswith("-"):
            s = line[1:].strip()
        else:
            continue
        if len(re.sub(r"\W", "", s)) >= 3:
            (added if line.startswith("+") else removed).add(s)
    return added, removed


def inverse_strength(a, b):
    """Overlap of a's changed lines with b's opposite-direction lines."""
    a_add, a_rem = a
    b_add, b_rem = b
    best = 0.0
    for x, y in ((a_add, b_rem), (a_rem, b_add)):
        if not x or not y:
            continue
        overlap = len(x & y)
        denom = max(1, min(len(x), len(y)))
        ratio = overlap / denom
        if overlap >= max(2, math.ceil(0.5 * denom)):
            best = max(best, ratio)
    return best


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/Git-projects"))
    ap.add_argument("--repos", nargs="*", default=REPOS_DEFAULT)
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-file", type=int, default=8)
    args = ap.parse_args()

    all_pairs = []
    per_repo = {}
    for repo in args.repos:
        path = os.path.join(args.root, repo)
        if not os.path.isdir(os.path.join(path, ".git")):
            continue
        by_file = commits_by_file(path)
        cand = 0
        for f, cs in by_file.items():
            local = 0
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    a, b = cs[i], cs[j]
                    if a["h"] == b["h"]:
                        continue
                    a_add = bool(ADD_RE.search(a["m"]))
                    a_rem = bool(REM_RE.search(a["m"]))
                    b_add = bool(ADD_RE.search(b["m"]))
                    b_rem = bool(REM_RE.search(b["m"]))
                    if not ((a_add and b_rem) or (a_rem and b_add)):
                        continue
                    all_pairs.append({"repo": repo, "file": f,
                                      "a": a["h"], "b": b["h"],
                                      "a_msg": a["m"][:160], "b_msg": b["m"][:160]})
                    cand += 1
                    local += 1
                    if local >= args.max_per_file:
                        break
                if local >= args.max_per_file:
                    break
        per_repo[repo] = {"files": len(by_file), "candidates": cand}
        print(f"  [{repo}] files={len(by_file)} candidates={cand}", flush=True)

    all_pairs.sort(key=lambda p: (p["repo"], p["file"], p["a"], p["b"]))
    rng = random.Random(args.seed)
    chosen = rng.sample(all_pairs, min(args.sample, len(all_pairs)))
    verified = []
    for i, p in enumerate(chosen, 1):
        try:
            a = patch_lines(os.path.join(args.root, p["repo"]), p["a"], p["file"])
            b = patch_lines(os.path.join(args.root, p["repo"]), p["b"], p["file"])
        except Exception:
            continue
        strength = inverse_strength(a, b)
        if strength > 0:
            verified.append({**{k: p[k] for k in ("repo", "file", "a", "b")},
                             "strength": round(strength, 3)})
        if i % 20 == 0:
            print(f"  [patch {i}/{len(chosen)}] verified={len(verified)}", flush=True)

    by_repo = collections.Counter(v["repo"] for v in verified)
    lo, hi = wilson(len(verified), len(chosen))
    report = {
        "repos": per_repo,
        "candidates_total": len(all_pairs),
        "sampled": len(chosen),
        "verified_inverses": len(verified),
        "verified_rate": round(len(verified) / max(1, len(chosen)), 3),
        "verified_rate_ci95": [round(lo, 3), round(hi, 3)],
        "verified_by_repo": dict(by_repo),
        "estimated_total_inverses": round(len(all_pairs) * len(verified) / max(1, len(chosen))),
        "sample_seed": args.seed,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps({**report, "verified": verified[:50]}, indent=2, default=str))
    print(json.dumps(report, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
