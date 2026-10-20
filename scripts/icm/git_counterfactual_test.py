"""GIT COUNTERFACTUAL TEST v3 -- same construction, broader organic corpus.

V1 (addendum 27) failed vacuity+power and was discarded. V2 (addendum 28) --
windowed records + ancestry depth -- passed every substantive check but failed
its pre-registered power bar at n=56 (own-work corpus exhausted at 58). V3 is a
NEW corpus run, not a re-score: the construction and the bars are IDENTICAL to
v2, and the corpus now adds major upstream public repos (real developer commit
messages, real history, same mechanical git oracle). The own/upstream split is
reported as a diagnostic; the pre-registered bar applies to the pooled corpus.

V2 DESIGN (unchanged):
  RECORD TEXT = a LOCALIZED CHANGE WINDOW around the first added hunk line
    (8 lines before, 40+ after; same-sized window at the old/new hunk start for
    the before/removed versions). Decoys = their first 48 lines.
  ORDER KEY = ANCESTRY DEPTH (`git rev-list --count`), never commit time.
  TRIAL (one per verified inverse pair, adversarial): "Restore the FIRST
    version of <file> (the state that existed before this change): <add
    message>"; pool = unique blobs among before/after/removed + up to 4
    decoys; answer = the parent-of-add blob.
  ARMS: DENSE@1 (full pool), DENSE-RELEVANT@1 (first same-file record),
    SEMANTIC-ONLY (type filter -> judge epoch re-rank), TWO-STAGE (dense top-5
    -> same). No blending; entity filter disabled.
  EASY DIRECTION: reported via dense-relevant's post-add preference, not
    scored (scoring it would inflate the primary).

  PRE-REGISTERED BAR:
    (1) POWER:      n >= 100 trials
    (2) VACUITY:    DENSE-RELEVANT@1 <= 1/versions + 0.15
    (3) STAGE1@5    >= 0.85
    (4) TWO-STAGE   >= 0.85
    (5) TWO-STAGE   >= DENSE-RELEVANT@1 + 0.25
    (6) JUDGE       >= 90% of keys carry an ordinal epoch reference
    PASS requires all; a dense win discards the construction.

Usage:
  python scripts/icm/git_counterfactual_test.py --discover-only   # build pool
  python scripts/icm/git_counterfactual_test.py --from-pool       # score
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
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

OUTDIR = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "lifted", "git_cf_v3")
POOL = os.path.join(OUTDIR, "pool.jsonl")
TRIALS = os.path.join(OUTDIR, "trials.jsonl")
RESULTS = os.path.join(OUTDIR, "results.json")
CACHE = os.path.join(OUTDIR, "keys_cache.jsonl")
CHECKPOINT = os.path.join(OUTDIR, "rows.jsonl")

LOCAL = ["Pondr", "HPMT", "liboffs", "WaveDB", "OFFS", "Space-Walk"]
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
_BLOB_CACHE_MAX = 512   # bounded: entries are >=6 KB, so ~3 MB at most


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
    # cache only the window-sized head, never the full blob (v3 memory fix)
    _blob_cache[key] = out[:6000]
    if len(_blob_cache) > _BLOB_CACHE_MAX:
        _blob_cache.clear()
    return out[:cap]


_depth: dict[tuple[str, str], int] = {}


def depth(repo, rev):
    """Ancestry depth: number of commits reachable from rev. Monotone along
    ancestry, unlike commit timestamps (v1's one order failure)."""
    key = (repo, rev)
    if key not in _depth:
        rc, out = run_git(repo, "rev-list", "--count", rev)
        try:
            _depth[key] = int(out.strip()) if rc == 0 else 0
        except ValueError:
            _depth[key] = 0
    return _depth[key]


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def diff_hunks(repo, commit, path):
    """Hunks from a commit's diff on one path: old/new starts + line texts."""
    rc, out = run_git(repo, "show", commit, "--format=", "--unified=0",
                      "--", path)
    hunks, cur = [], None
    for line in out.splitlines():
        m = HUNK.match(line)
        if m:
            cur = {"old": int(m.group(1)), "new": int(m.group(3)),
                   "added": [], "removed": []}
            hunks.append(cur)
        elif cur is not None:
            if line.startswith("+") and not line.startswith("+++"):
                cur["added"].append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                cur["removed"].append(line[1:])
    return hunks


def window_text(text, start_line, before=8, after=40):
    """Localized change window (v2): the hunk start is near the window top so
    the feature is inside the embedder's first 512 tokens."""
    if not text:
        return ""
    lines = text.splitlines()
    lo = max(0, int(start_line) - 1 - before)
    hi = min(len(lines), int(start_line) - 1 + after)
    if hi <= lo:
        return ""
    return "\n".join(lines[lo:hi])


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
        txt = window_text(blob_text(repo, b), 1, before=0, after=48)
        if not txt.strip():
            continue
        seen.add(b)
        recs.append({"id": b, "path": p, "type": p,
                     "epoch": depth(repo, commit), "text": txt,
                     "kind": "decoy"})
    return recs


def discover(repos, max_per_file=8, decoys_n=4, seed=42, out_path=None):
    """Find trials. `out_path` streams each trial to disk and returns the count
    (memory-safe for large corpora); without it the trials are returned as a
    list (tests / small runs)."""
    rng = random.Random(seed)
    streamed = out_path is not None
    fh = (io.open(out_path, "w", encoding="utf-8", newline="\n")
          if streamed else None)
    trials, n_trials = [], 0
    skipped = {"not_ancestor": 0, "missing_blob": 0,
               "feature_not_visible": 0, "same_before_after": 0,
               "no_hunk": 0}
    try:
        for name, repo, source in repos:
            # bound memory per repo: caches must not accumulate across repos
            _blob_cache.clear()
            _depth.clear()
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
                        add_hunk = next((h for h in diff_hunks(repo, add["h"], f)
                                         if h["added"]), None)
                        rem_hunk = next((h for h in diff_hunks(repo, rem["h"], f)
                                         if h["removed"]), None)
                        if add_hunk is None or rem_hunk is None:
                            skipped["no_hunk"] += 1
                            continue
                        after_win = window_text(
                            blob_text(repo, after), add_hunk["new"],
                            after=max(40, len(add_hunk["added"]) + 12))
                        # visibility is checked on the TARGETED hunk's own lines;
                        # `added` spans all hunks and would falsely fail a pair
                        # whose other hunks fall outside this window.
                        hunk_added = [line.strip() for line in add_hunk["added"]
                                      if len(re.sub(r"\W", "", line.strip())) >= 3]
                        visible = sum(1 for line in hunk_added if line in after_win)
                        if hunk_added and visible < max(1, math.ceil(0.8 * len(hunk_added))):
                            skipped["feature_not_visible"] += 1
                            continue
                        widen = max(40, len(add_hunk["added"]) + 12)
                        windows = {
                            "before": window_text(blob_text(repo, before),
                                                  add_hunk["old"], after=widen),
                            "after": after_win,
                            "removed": window_text(blob_text(repo, removed),
                                                   rem_hunk["new"],
                                                   after=max(40, len(rem_hunk["removed"]) + 12)),
                        }
                        seen, records = set(), []
                        for label, blob, rev in (("before", before, add["h"] + "^"),
                                                 ("after", after, add["h"]),
                                                 ("removed", removed, rem["h"])):
                            if blob in seen:
                                continue
                            seen.add(blob)
                            records.append({"id": blob, "path": f, "type": f,
                                            "epoch": depth(repo, rev),
                                            "text": windows[label],
                                            "kind": "version", "label": label})
                        records += decoy_records(repo, add["h"], f, decoys_n, seen, rng)
                        if len(records) < 2:
                            continue
                        query = ("Restore the FIRST version of " + f + " (the state "
                                 "that existed before this change): " + add["m"][:600])
                        trial = {
                            "trial": f"{name}:{f}:{add['h'][:8]}",
                            "repo": name, "source": source, "file": f,
                            "query": query,
                            "records": records, "correct": before, "after": after,
                            "versions": sum(1 for r in records if r["kind"] == "version"),
                            "pool": len(records),
                            "add_msg": add["m"][:300], "add": add["h"], "rem": rem["h"],
                        }
                        if streamed:
                            fh.write(json.dumps(trial, ensure_ascii=True) + "\n")
                            fh.flush()
                            n_trials += 1
                        else:
                            trials.append(trial)
                        repo_trials += 1
                        local += 1
                        if local >= max_per_file:
                            break
                    if local >= max_per_file:
                        break
            print(f"  [{name}] trials={repo_trials}", flush=True)
    finally:
        if fh is not None:
            fh.close()
    return (n_trials if streamed else trials), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-root", default=os.path.expanduser("~/Git-projects"))
    ap.add_argument("--public-root",
                    default=os.path.join(REPO_ROOT, "data", "_scratch", "icm",
                                         "public_repos"))
    ap.add_argument("--upstream-root",
                    default=os.path.join(REPO_ROOT, "data", "_scratch", "icm",
                                         "upstream_repos"))
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
    ap.add_argument("--time-budget", type=int, default=0,
                    help="stop cleanly after N seconds (checkpointed; 0=no cap)")
    args = ap.parse_args()

    repos = []
    for name in LOCAL:
        p = os.path.join(args.local_root, name)
        if os.path.isdir(os.path.join(p, ".git")):
            repos.append((name, p, "own"))
    # every cloned repo under the own public root (breadth for power)
    if os.path.isdir(args.public_root):
        for name in sorted(os.listdir(args.public_root)):
            p = os.path.join(args.public_root, name)
            if os.path.isdir(os.path.join(p, ".git")):
                repos.append((name, p, "own"))
    # major upstream public repos: real intent text by their authors (v3)
    if os.path.isdir(args.upstream_root):
        for name in sorted(os.listdir(args.upstream_root)):
            p = os.path.join(args.upstream_root, name)
            if os.path.isdir(os.path.join(p, ".git")):
                repos.append((name, p, "upstream"))
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
        n, skipped = discover(repos, decoys_n=args.decoys, seed=args.seed,
                              out_path=POOL)
        print(f"[discover] trials={n} skipped={skipped} "
              f"({time.time()-t0:.0f}s) -> {POOL}", flush=True)
        if args.discover_only:
            print(f"[discover-only] {n} trials; not scoring")
            return 0 if n else 1
        trials = [json.loads(line) for line in io.open(POOL, encoding="utf-8")
                  if line.strip()]
        print(f"[pool] loaded {len(trials)} trials", flush=True)
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
    done = {}
    if os.path.exists(CHECKPOINT):
        for line in io.open(CHECKPOINT, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["trial"]] = r
            except Exception:
                continue
    print(f"[checkpoint] {len(done)} trials already scored", flush=True)
    ck = io.open(CHECKPOINT, "a", encoding="utf-8", newline="\n")
    try:
        for i, t in enumerate(trials, 1):
            if t["trial"] in done:
                continue
            records = t["records"]
            meta = {"type": t["file"], "ignore_key_entities": True, "k": 5}
            full = retriever.query(t["query"], records, query_meta=meta)
            key = full["key"]
            e = (key or {}).get("epoch_ref", {}).get("value", "none") or "none"
            by_id = {r["id"]: r for r in records}
            cands5 = [by_id[x] for x in full["stage1_ids"][:5] if x in by_id]
            two = retriever.query(t["query"], cands5, query_key=key, query_meta=meta)

            dense_top = full["stage1_ids"][0]
            same_file = [x for x in full["stage1_ids"]
                         if by_id.get(x, {}).get("type") == t["file"]]
            dense_relevant_top = same_file[0] if same_file else None
            sem_top = full["top"][0]["id"] if full["top"] else None
            two_top = two["top"][0]["id"] if two["top"] else None
            correct = t["correct"]
            row = {
                "trial": t["trial"], "repo": t["repo"],
                "source": t.get("source", "own"), "file": t["file"],
                "correct": correct, "dense_top": dense_top,
                "dense_relevant_top": dense_relevant_top,
                "semantic_top": sem_top, "two_top": two_top,
                "epoch_ref": e, "fallback": full["fallback"],
                "pool": t["pool"], "versions": t["versions"],
                "judge_status": full["key_status"],
                "stage1_ok": 1.0 if correct in full["stage1_ids"][:5] else 0.0,
                "prefers_after": 1.0 if dense_top == t["after"] else 0.0,
                "rel_prefers_after": 1.0 if dense_relevant_top == t["after"] else 0.0,
            }
            done[t["trial"]] = row
            ck.write(json.dumps(row, ensure_ascii=True) + "\n")
            ck.flush()
            if len(done) % 10 == 0:
                print(f"  [checkpoint {len(done)}/{len(trials)}] "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if args.time_budget and (time.time() - t0) >= args.time_budget:
                print(f"[budget] stopping after {len(done)} scored "
                      f"({time.time()-t0:.0f}s)", flush=True)
                break
    finally:
        ck.close()

    rows = [done[t["trial"]] for t in trials if t["trial"] in done]
    missing = len(trials) - len(rows)
    complete = missing == 0
    if missing:
        print(f"[warn] {missing} trials unscored (incomplete run)", flush=True)
    arms = {"dense@1": [], "dense_relevant@1": [], "semantic_only": [],
            "two_stage": []}
    by_source: dict[str, dict[str, list]] = {}
    stage1, epochs, fallbacks = [], {}, {}
    prefers_after, relevant_prefers_after = [], []
    for r in rows:
        vals = {
            "dense@1": 1.0 if r["dense_top"] == r["correct"] else 0.0,
            "dense_relevant@1": 1.0 if r["dense_relevant_top"] == r["correct"] else 0.0,
            "semantic_only": 1.0 if r["semantic_top"] == r["correct"] else 0.0,
            "two_stage": 1.0 if r["two_top"] == r["correct"] else 0.0,
        }
        for arm, val in vals.items():
            arms[arm].append(val)
            by_source.setdefault(r.get("source", "own"), {}).setdefault(
                arm, []).append(val)
        stage1.append(r["stage1_ok"])
        prefers_after.append(r["prefers_after"])
        relevant_prefers_after.append(r["rel_prefers_after"])
        epochs[r["epoch_ref"]] = epochs.get(r["epoch_ref"], 0) + 1
        fallbacks[r["fallback"] or "none"] = fallbacks.get(
            r["fallback"] or "none", 0) + 1

    means = {k: round(statistics.mean(v), 3) for k, v in arms.items()}
    source_means = {src: {arm: round(statistics.mean(v), 3)
                          for arm, v in d.items()}
                    for src, d in by_source.items()}
    chance_mean = round(statistics.mean(1.0 / t["pool"] for t in trials), 3)
    relevant_chance = round(statistics.mean(1.0 / t["versions"] for t in trials), 3)
    earliest_rate = round(epochs.get("EARLIEST", 0) / max(1, len(rows)), 3)
    checks = {
        "power_n_ge_100": len(trials) >= 100,
        "vacuity_dense_relevant": means["dense_relevant@1"] <= relevant_chance + 0.15,
        "stage1_recall5": round(statistics.mean(stage1), 3) >= 0.85,
        "two_stage_high": means["two_stage"] >= 0.85,
        "two_stage_beats_dense": (means["two_stage"]
                                  >= means["dense_relevant@1"] + 0.25),
        "judge_ordinal_rate": earliest_rate >= 0.90,
    }
    res = {
        "trials": len(trials), "skipped": skipped,
        "pool_sizes": {"mean": round(statistics.mean(t["pool"] for t in trials), 2),
                       "versions_mean": round(statistics.mean(t["versions"] for t in trials), 2)},
        "chance_dense_mean": chance_mean,
        "chance_relevant_mean": relevant_chance,
        "arms": means,
        "by_source": source_means,
        "stage1_recall@5": round(statistics.mean(stage1), 3),
        "dense_prefers_after": round(statistics.mean(prefers_after), 3),
        "dense_relevant_prefers_after": round(
            statistics.mean(relevant_prefers_after), 3),
        "epoch_refs": epochs, "judge_earliest_rate": earliest_rate,
        "fallbacks": fallbacks,
        "checks": checks, "complete": complete, "scored": len(rows),
        "pass": all(checks.values()) and complete,
        "verdict": ("INCOMPLETE -- resume the scoring run" if not complete else
                    "PASS -- the intent key selects the pre-change version "
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
                      ("trials", "scored", "complete", "pool_sizes",
                       "chance_dense_mean", "chance_relevant_mean", "arms",
                       "by_source", "stage1_recall@5", "dense_prefers_after",
                       "dense_relevant_prefers_after", "epoch_refs",
                       "fallbacks", "checks", "pass", "verdict")},
                     indent=1, default=str), flush=True)
    print(f"[done] -> {RESULTS}", flush=True)
    return (0 if (complete and res["pass"]) else (2 if complete else 3))


if __name__ == "__main__":
    sys.exit(main())
