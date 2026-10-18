"""INTENT KEYSPACE v2 -- MULTI-REQUIREMENT keys + FILE-CENTRIC relation labels.

WHY (addendum 19): the v1 keyspace used ONE SPO triple per intent. Real issue
texts are ~1,300 chars carrying several requirements, so one triple discarded
most of the intent and key_sim barely varied across files (struct recall@10
0.000 vs dense 0.179, a 45x-chance signal). The relation prompt also collapsed
on long texts (94/96 DIFFERENT, 0 OPPOSED) because it asked about the intents
in general rather than about the file they share.

CHANGES (both pre-registered here, before fitting):
  1. REQUIREMENTS: the judge extracts 3-6 distinct requirement triples per
     issue (not 1). keys(run) = list of triples.
  2. FILE-CENTRIC RELATIONS: for each outcome-flip unit, the judge is told the
     SHARED FILE and asked whether the two tasks require changes to that file
     for the SAME purpose / DIFFERENT purposes / OPPOSED purposes (acting on
     one defeats the other).

OUTPUT (untracked):
  data/_scratch/icm/lifted/intent_keys_multi.jsonl          run -> [triples]
  data/_scratch/icm/lifted/flip_relations_filecentric.jsonl  file-centric labels
  data/_scratch/icm/lifted/keyspace_multi_report.json

BAR (pre-registered): mean requirements per issue in [3, 6]; >= 90 percent of
intents keyed; relation labels for >= 80 percent of the ~98 outcome-flip
units. The HARNESS bar is unchanged (same file, same split, same ground
truth): hybrid recall@10 >= dense + 0.05 and hybrid MRR >= dense; struct
competitive (recall@10 >= dense); relation gap (mean key_sim SAME - mean
key_sim OPPOSED) >= 0.25. This is a NEW test; addendum 19's negative stands
for the v1 construction.

Run:  python scripts/icm/intent_keyspace_multi.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from intent_keyspace import ask, first_json  # noqa: E402  (same relay helpers)

LIFTED = os.path.join("data", "_scratch", "icm", "lifted", "codetrace_intents.jsonl")
FILES = os.path.join("data", "_scratch", "icm", "lifted", "codetrace_files.jsonl")
CACHE = os.path.join("data", "_scratch", "icm", "lifted", "keyspace_multi_cache.jsonl")
OUTK = os.path.join("data", "_scratch", "icm", "lifted", "intent_keys_multi.jsonl")
OUTR = os.path.join("data", "_scratch", "icm", "lifted", "flip_relations_filecentric.jsonl")
REP = os.path.join("data", "_scratch", "icm", "lifted", "keyspace_multi_report.json")
BATCH = 4


_cache = {}


def load_cache():
    if os.path.exists(CACHE):
        for l in io.open(CACHE, encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    _cache[r["key"]] = r.get("val")
                except Exception:
                    pass


def save_cache(k, v):
    _cache[k] = v
    with io.open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": k, "val": v}) + "\n")


def extract_multi(texts):
    head = ("For each numbered issue below, list the DISTINCT requirements the issue "
            "asks for: 3 to 6 of them. Reply with a JSON array only; each item: "
            '{"i": <issue number>, "reqs": [{"attitude": "<stance verb>", '
            '"relation": "<what is to be done>", "object": "<the concrete artifact '
            'or behaviour aimed at, few words>", "qualifier": "<state/version/time '
            'marker if any, else empty>"}]}.')
    prompt = head + "\n\n" + "\n\n".join(
        f"{i+1}. {t[:1200]}" for i, t in enumerate(texts))
    raw = ask(prompt)
    arr = first_json(raw, "[")
    out = [None] * len(texts)
    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("i", 0)) - 1
            except Exception:
                continue
            reqs = item.get("reqs") or []
            if 0 <= idx < len(out) and isinstance(reqs, list):
                cleaned = [{k: str(r.get(k, "")) for k in
                            ("attitude", "relation", "object", "qualifier")}
                           for r in reqs if isinstance(r, dict)]
                if cleaned:
                    out[idx] = cleaned
    for i, x in enumerate(out):           # per-item fallback
        if x is not None:
            continue
        p1 = ("List the 3 to 6 distinct requirements of the issue below. Reply JSON "
              'array only: [{"attitude": "...", "relation": "...", "object": "...", '
              '"qualifier": "..."}].' + "\n\nIssue:\n" + texts[i][:1200])
        a1 = first_json(ask(p1), "[")
        if isinstance(a1, list):
            cleaned = [{k: str(r.get(k, "")) for k in
                        ("attitude", "relation", "object", "qualifier")}
                       for r in a1 if isinstance(r, dict)]
            out[i] = cleaned or None
    return out


def judge_file_relation(fpath, a, b):
    prompt = ("Two coding tasks were run against the same file: " + fpath + "."
              " Task A issue: " + a[:900] + " Task B issue: " + b[:900] +
              " Do both tasks require changes to THIS FILE for the SAME purpose, "
              "for DIFFERENT unrelated purposes, or for OPPOSED purposes (acting "
              "on one would defeat the other)? Answer exactly one word: "
              "SAME, DIFFERENT or OPPOSED.")
    raw = ask(prompt).upper()
    for w in ("OPPOSED", "DIFFERENT", "SAME"):
        if w in raw:
            return w.lower()
    return None


def main() -> int:
    load_cache()
    runs = [json.loads(l) for l in io.open(LIFTED, encoding="utf-8") if l.strip()]
    with_issue = [r for r in runs if r.get("issue")]
    print(f"[keys-v2] {len(with_issue)} intents | cache {len(_cache)}", flush=True)
    t0 = time.time()
    todo = [r for r in with_issue
            if hashlib.sha1(("M|" + r["issue"]).encode()).hexdigest() not in _cache]
    print(f"  to extract: {len(todo)}", flush=True)
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        got = extract_multi([r["issue"] for r in chunk])
        for r, g in zip(chunk, got):
            save_cache(hashlib.sha1(("M|" + r["issue"]).encode()).hexdigest(), g)
        print(f"  [{min(i + BATCH, len(todo))}/{len(todo)}] ({time.time() - t0:.0f}s)",
              flush=True)
    keys = {r["run_id"]: _cache.get(hashlib.sha1(("M|" + r["issue"]).encode()).hexdigest())
            for r in with_issue}
    ok = sum(1 for v in keys.values() if v)
    req_counts = [len(v) for v in keys.values() if v]
    mean_reqs = round(sum(req_counts) / len(req_counts), 2) if req_counts else 0.0
    with io.open(OUTK, "w", encoding="utf-8") as f:
        for r in with_issue:
            f.write(json.dumps({"run_id": r["run_id"], "agent": r["agent"],
                                "task": r.get("task_name"), "keys": keys.get(r["run_id"])}) + "\n")

    files = [json.loads(l) for l in io.open(FILES, encoding="utf-8") if l.strip()]
    by_file = {}
    for rec in files:
        by_file.setdefault(rec["file"], []).append(rec)
    units = {f: v for f, v in by_file.items() if len(v) >= 2}
    issue_of = {r["run_id"]: r["issue"] for r in with_issue}
    flips = []
    for fpath, v in units.items():
        use = [x for x in v if x["label"] == "useful" and keys.get(x["run_id"])
               and issue_of.get(x["run_id"])]
        non = [x for x in v if x["label"] in ("attempted", "context")
               and issue_of.get(x["run_id"])]
        if not use or not non:
            continue
        a_rec, b_rec = use[0], non[0]
        a, b = issue_of[a_rec["run_id"]], issue_of[b_rec["run_id"]]
        ck = hashlib.sha1(("F|" + fpath + "|" + a[:300] + "|" + b[:300]).encode()).hexdigest()
        if ck in _cache:
            rel = _cache[ck]
        else:
            rel = judge_file_relation(fpath, a, b)
            save_cache(ck, rel)
        flips.append({"file": fpath, "useful_run": a_rec["run_id"],
                      "nonuseful_run": b_rec["run_id"],
                      "nonuseful_label": b_rec["label"], "relation": rel})
        print(f"  [flip] {fpath[:46]} -> {rel}", flush=True)
    with io.open(OUTR, "w", encoding="utf-8") as f:
        for row in flips:
            f.write(json.dumps(row) + "\n")

    rel_dist = {}
    for row in flips:
        rel_dist[row["relation"]] = rel_dist.get(row["relation"], 0) + 1
    rep = {"intents": len(with_issue), "intents_with_key": ok,
           "key_rate": round(ok / max(1, len(with_issue)), 3),
           "mean_requirements": mean_reqs,
           "flip_units_labeled": len(flips), "relation_distribution": rel_dist,
           "elapsed_s": round(time.time() - t0)}
    rep["bar"] = {"key_rate_ok": rep["key_rate"] >= 0.90,
                  "reqs_in_range": 3 <= mean_reqs <= 6}
    with io.open(REP, "w", encoding="utf-8") as f:
        f.write(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str), flush=True)
    print(f"[done] -> {OUTK}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
