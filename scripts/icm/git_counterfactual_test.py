"""GIT COUNTERFACTUAL TEST -- the first organic-substrate intent-retrieval run.

CORPUS (measured, addendum 26): patch-verified inverse pairs in real git
history. A pair = an ADD commit that introduced lines and a later REMOVE
commit that took them out of the SAME file. The memory units are the real file
blobs around that change; the intent text is the real add-commit message; the
oracle is the git blob content -- no judge, no authored world.

PRE-REGISTERED DESIGN (written before the run; no fitting afterward):
  TRIAL (adversarial direction, one per pair):
    query  = "Restore the FIRST version of <file> (the state that existed
              before this change): <add commit message>"
    pool   = unique blobs among <parent of add>, <add>, <remove> + up to 4
             decoy blobs from OTHER files at the add commit;
    answer = the parent-of-add blob (the pre-change version);
    chance = 1 / pool size (dense); the artifact/type filter reduces the
             two-stage decision to the same-file versions (usually 3).
  The adversarial property: the message names the feature, the wrong version
  (post-add) contains it, and the correct pre-change version does not.

  ARMS:
    DENSE@1        top-1 cosine over the full pool (baseline)
    STAGE1@5       is the correct blob in the dense top-5? (reported)
    SEMANTIC-ONLY  type filter (type = exact file path) -> epoch re-rank over
                   the full pool, judge-built key, NO score blending
    TWO-STAGE      dense top-5 -> the same filter + epoch re-rank
  The entity filter is DISABLED (`ignore_key_entities`) because the key's
  requirement tokens name the feature, which the correct pre-change version
  legitimately lacks; the artifact is already fixed exactly by `type`.

  PRE-REGISTERED BAR:
    (1) POWER:     n >= 100 trials with a usable key
    (2) VACUITY:   DENSE@1 <= mean(chance) + 0.15 (similarity must NOT read a
                   surface tell that solves the trial)
    (3) STAGE1@5   >= 0.85 (dense's candidates contain the answer)
    (4) TWO-STAGE  >= 0.85
    (5) TWO-STAGE  >= DENSE@1 + 0.25
    (6) JUDGE       >= 90% of keys carry an ordinal epoch reference
    PASS requires all. A dense win means the construction is vacuous and is
    DISCARDED (addendum 21), never explained away.

  EPOCH DOMAIN: pool records carry commit timestamps; the ordinal re-rank is
  by commit order. `removed` may equal `before` in content (a clean revert):
  records are DEDUPED BY BLOB, keeping the earliest epoch, so the answer is
  one record.

  SCOPE: this tests the measured mechanism (judge key -> deterministic filter
  -> epoch order) on a real substrate. It does NOT settle the fully-organic
  thesis: the query template is authored, and the pool is built around the
  verified pair. Both facts are stated in the result.

Usage:
  python scripts/icm/git_counterfactual_test.py --limit 8     # mechanics smoke
  python scripts/icm/git_counterfactual_test.py               # pre-registered run
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import census_git_inverses as gi  # noqa: E402
from icm.intent_keys import IntentKeyer  # noqa: E402
from icm.intent_query import TwoStageRetriever  # noqa: E402

OUTDIR = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "lifted", "git_cf")
POOL = os.path.join(OUTDIR, "pool.jsonl")
TRIALS = os.path.join(OUTDIR, "trials.jsonl")
RESULTS = os.path.join(OUTDIR, "results.json")
CACHE = os.path.join(OUTDIR, "keys_cache.jsonl")

LOCAL = ["Pondr", "HPMT", "liboffs", "WaveDB"]
PUBLIC = ["CRABS", "SecretAgent", "poll-dancer", "EAuth", "poseidon",
          "poseidon-crdt", "liboffs_pony", "Tangerine-Id-Migrator"]
SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip",
            ".tar", ".gz", ".woff", ".ttf", ".eot", ".mp4", ".webm",
            ".min.js", ".min.css", ".lock")

_emb = [None]


def embed(texts):
    if _emb[0] is None:
        from sentence_transformers import SentenceTransformer
        _emb[0] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _emb[0].encode(list(texts), batch_size=32, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)


def run_git(repo, *args, timeout=120):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                       text=True, errors="replace", timeout=timeout)
    return p.returncode, p.stdout


def blob_at(repo, rev, path):
    rc, out = run_git(repo, "rev-parse", f"{rev}:{path}")
    return out.strip() if rc == 0 and out.strip() else None


def blob_size(repo, blob):
    rc, out = run_git(repo, "cat-file", "-s", blob)
    try:
        return int(out.strip()) if rc == 0 else -1
    except ValueError:
        return -1


_blob_cache: dict[tuple[str, str], str] = {}


def blob_text(repo, blob, cap=6000):
    key = (repo, blob)
    if key in _blob_cache:
        return _blob_cache[key][:cap]
    size = blob_size(repo, blob)
    if size < 0 or size > 2_000_000:
        _blob_cache[key] = ""
        return ""
    rc, out = run_git(repo, "cat-file", "-p", blob)
    if rc != 0 or "\x00" in out:
        out = ""
    _blob_cache[key] = out
    return out[:cap]


def commit_time(repo, rev):
    rc, out = run_git(repo, "show", "-s", "--format=%ct", rev)
    try:
        return int(out.strip()) if rc == 0 else 0
    except ValueError:
        return 0


def overlap_ok(x, y):
    if not x or not y:
        return False
    return len(x & y) >= max(2, math.ceil(0.5 * min(len(x), len(y))))


def is_text_path(path):
    low = path.lower()
    return not any(low.endswith(ext) for ext in SKIP_EXT)


def decoy_records(repo, commit, exclude, n, seen, rng):
    rc, out = run_git(repo, "ls-tree", "-r", "--name-only", commit)
    if rc != 0:
        return []
    files = [p for p in out.splitlines()
             if p and p != exclude and is_text_path(p)]
    rng.shuffle(files)
    recs = []
    for p in files:
        if len(recs) >= n:
            break
        b = blob_at(repo, commit, p)
        if not b or b in seen:
            continue
        txt = blob_text(repo, b)
        if not txt.strip():
            continue
        seen.add(b)
        recs.append({"id": b, "path": p, "type": p,
                     "epoch": commit_time(repo, commit), "text": txt,
                     "kind": "decoy"})
    return recs


def discover(repos, max_per_file=8, decoys_n=4, seed=42):
    rng = random.Random(seed)
    trials, skipped = [], {"not_ancestor": 0, "missing_blob": 0,
                           "feature_not_visible": 0, "same_before_after": 0}
    for name, repo in repos:
        by_file = gi.commits_by_file(repo)
        repo_trials = 0
        for f, cs in by_file.items():
            if not is_text_path(f):
                continue
            local = 0
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    a, b = cs[i], cs[j]
                    if a["h"] == b["h"]:
                        continue
                    a_add, a_rem = bool(gi.ADD_RE.search(a["m"])), bool(gi.REM_RE.search(a["m"]))
                    b_add, b_rem = bool(gi.ADD_RE.search(b["m"])), bool(gi.REM_RE.search(b["m"]))
                    if not ((a_add and b_rem) or (a_rem and b_add)):
                        continue
                    try:
                        pa = gi.patch_lines(repo, a["h"], f)
                        pb = gi.patch_lines(repo, b["h"], f)
                    except Exception:
                        continue
                    if overlap_ok(pa[0], pb[1]):
                        add, rem, added = a, b, pa[0]
                    elif overlap_ok(pb[0], pa[1]):
                        add, rem, added = b, a, pb[0]
                    else:
                        continue
                    rc, _ = run_git(repo, "merge-base", "--is-ancestor",
                                    add["h"], rem["h"])
                    if rc != 0:
                        skipped["not_ancestor"] += 1
                        continue
                    before = blob_at(repo, add["h"] + "^", f)
                    after = blob_at(repo, add["h"], f)
                    removed = blob_at(repo, rem["h"], f)
                    if not (before and after and removed):
                        skipped["missing_blob"] += 1
                        continue
                    if before == after:
                        skipped["same_before_after"] += 1
                        continue
                    after_text = blob_text(repo, after)
                    visible = sum(1 for line in added if line in after_text)
                    if visible < max(1, math.ceil(0.8 * len(added))):
                        skipped["feature_not_visible"] += 1
                        continue
                    seen, records = set(), []
                    for label, blob, rev in (("before", before, add["h"] + "^"),
                                             ("after", after, add["h"]),
                                             ("removed", removed, rem["h"])):
                        if blob in seen:
                            continue
                        seen.add(blob)
                        records.append({"id": blob, "path": f, "type": f,
                                        "epoch": commit_time(repo, rev),
                                        "text": blob_text(repo, blob),
                                        "kind": "version", "label": label})
                    records += decoy_records(repo, add["h"], f, decoys_n, seen, rng)
                    if len(records) < 2:
                        continue
                    query = ("Restore the FIRST version of " + f + " (the state "
                             "that existed before this change): " + add["m"][:600])
                    trials.append({
                        "trial": f"{name}:{f}:{add['h'][:8]}",
                        "repo": name, "file": f, "query": query,
                        "records": records, "correct": before, "after": after,
                        "versions": sum(1 for r in records if r["kind"] == "version"),
                        "pool": len(records),
                        "add_msg": add["m"][:300], "add": add["h"], "rem": rem["h"],
                    })
                    repo_trials += 1
                    local += 1
                    if local >= max_per_file:
                        break
                if local >= max_per_file:
                    break
        print(f"  [{name}] trials={repo_trials}", flush=True)
    return trials, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-root", default=os.path.expanduser("~/Git-projects"))
    ap.add_argument("--public-root",
                    default=os.path.join(REPO_ROOT, "data", "_scratch", "icm",
                                         "public_repos"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these repo names")
    ap.add_argument("--limit", type=int, default=0,
                    help="mechanics smoke: cap trials (0 = full run)")
    ap.add_argument("--discover-only", action="store_true",
                    help="build and persist the trial pool, then exit (no scoring)")
    ap.add_argument("--from-pool", action="store_true",
                    help="reuse the persisted trial pool instead of rediscovering")
    ap.add_argument("--decoys", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repos = []
    for name in LOCAL:
        p = os.path.join(args.local_root, name)
        if os.path.isdir(os.path.join(p, ".git")):
            repos.append((name, p))
    for name in PUBLIC:
        p = os.path.join(args.public_root, name)
        if os.path.isdir(os.path.join(p, ".git")):
            repos.append((name, p))
    if args.only:
        repos = [r for r in repos if r[0] in set(args.only)]
    print(f"[repos] {[r[0] for r in repos]}", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)

    t0 = time.time()
    if args.from_pool:
        trials = []
        for line in io.open(POOL, encoding="utf-8"):
            if line.strip():
                trials.append(json.loads(line))
        skipped = {"reused_pool": len(trials)}
        print(f"[pool] reused {len(trials)} trials from {POOL}", flush=True)
    else:
        trials, skipped = discover(repos, decoys_n=args.decoys, seed=args.seed)
        with io.open(POOL, "w", encoding="utf-8", newline="\n") as f:
            for t in trials:
                f.write(json.dumps(t, ensure_ascii=True) + "\n")
        print(f"[discover] trials={len(trials)} skipped={skipped} "
              f"({time.time()-t0:.0f}s) -> {POOL}", flush=True)
    if args.discover_only:
        print(f"[discover-only] {len(trials)} trials; not scoring")
        return 0 if trials else 1
    if not trials:
        print("NO TRIALS -- nothing to run")
        return 1
    rng = random.Random(args.seed)
    rng.shuffle(trials)
    if args.limit:
        trials = trials[: args.limit]
    print(f"[run] {len(trials)} trials", flush=True)

    keyer = IntentKeyer(cache_path=CACHE)
    retriever = TwoStageRetriever(embed, keyer=keyer)
    arms = {"dense@1": [], "semantic_only": [], "two_stage": []}
    stage1, epochs, fallbacks, prefers_after = [], {}, {}, []
    rows = []
    for i, t in enumerate(trials, 1):
        records = t["records"]
        meta = {"type": t["file"], "ignore_key_entities": True, "k": 5}
        full = retriever.query(t["query"], records, query_meta=meta)
        key = full["key"]
        e = (key or {}).get("epoch_ref", {}).get("value", "none") or "none"
        epochs[e] = epochs.get(e, 0) + 1
        fallbacks[full["fallback"] or "none"] = fallbacks.get(full["fallback"] or "none", 0) + 1
        by_id = {r["id"]: r for r in records}
        cands5 = [by_id[x] for x in full["stage1_ids"][:5] if x in by_id]
        two = retriever.query(t["query"], cands5, query_key=key, query_meta=meta)

        dense_top = full["stage1_ids"][0]
        sem_top = full["top"][0]["id"] if full["top"] else None
        two_top = two["top"][0]["id"] if two["top"] else None
        correct = t["correct"]
        arms["dense@1"].append(1.0 if dense_top == correct else 0.0)
        arms["semantic_only"].append(1.0 if sem_top == correct else 0.0)
        arms["two_stage"].append(1.0 if two_top == correct else 0.0)
        stage1.append(1.0 if correct in full["stage1_ids"][:5] else 0.0)
        prefers_after.append(1.0 if dense_top == t["after"] else 0.0)
        rows.append({"trial": t["trial"], "repo": t["repo"], "file": t["file"],
                     "correct": correct, "dense_top": dense_top,
                     "semantic_top": sem_top, "two_top": two_top,
                     "epoch_ref": e, "fallback": full["fallback"],
                     "pool": t["pool"], "versions": t["versions"],
                     "judge_status": full["key_status"]})
        if i % 10 == 0:
            print(f"  [{i}/{len(trials)}] dense={statistics.mean(arms['dense@1']):.3f} "
                  f"sem={statistics.mean(arms['semantic_only']):.3f} "
                  f"two={statistics.mean(arms['two_stage']):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    means = {k: round(statistics.mean(v), 3) for k, v in arms.items()}
    chance_mean = round(statistics.mean(1.0 / t["pool"] for t in trials), 3)
    earliest_rate = round(epochs.get("EARLIEST", 0) / len(trials), 3)
    checks = {
        "power_n_ge_100": len(trials) >= 100,
        "vacuity_dense_near_chance": means["dense@1"] <= chance_mean + 0.15,
        "stage1_recall5": round(statistics.mean(stage1), 3) >= 0.85,
        "two_stage_high": means["two_stage"] >= 0.85,
        "two_stage_beats_dense": means["two_stage"] >= means["dense@1"] + 0.25,
        "judge_ordinal_rate": earliest_rate >= 0.90,
    }
    res = {
        "trials": len(trials), "skipped": skipped,
        "pool_sizes": {"mean": round(statistics.mean(t["pool"] for t in trials), 2),
                       "versions_mean": round(statistics.mean(t["versions"] for t in trials), 2)},
        "chance_dense_mean": chance_mean,
        "arms": means,
        "stage1_recall@5": round(statistics.mean(stage1), 3),
        "dense_prefers_after": round(statistics.mean(prefers_after), 3),
        "epoch_refs": epochs, "judge_earliest_rate": earliest_rate,
        "fallbacks": fallbacks,
        "checks": checks, "pass": all(checks.values()),
        "verdict": ("PASS -- the intent key selects the pre-change version "
                    "where similarity is at chance" if all(checks.values()) else
                    "FAIL -- the failing row names the missing piece"),
        "scope": "organic substrate; authored query template and pair-centered pool",
        "elapsed_s": round(time.time() - t0),
    }
    with io.open(TRIALS, "w", encoding="utf-8", newline="\n") as f:
        for t in trials:
            f.write(json.dumps({k: t[k] for k in
                                ("trial", "repo", "file", "query", "correct",
                                 "after", "versions", "pool", "add_msg")},
                               sort_keys=True, ensure_ascii=True) + "\n")
    with io.open(RESULTS, "w", encoding="utf-8") as f:
        f.write(json.dumps({**res, "rows": rows}, indent=2, default=str))
    print(json.dumps({k: res[k] for k in
                      ("trials", "pool_sizes", "chance_dense_mean", "arms",
                       "stage1_recall@5", "dense_prefers_after", "epoch_refs",
                       "fallbacks", "checks", "pass", "verdict")},
                     indent=1, default=str), flush=True)
    print(f"[done] -> {RESULTS}", flush=True)
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
