"""INTENT KEYSPACE -- judge-built keys over the lifted real intents.

WHY (addendum 15/16): the judge separates OPPOSED from SAME intents perfectly
on user- and model-authored pairs (0.075 / 0.90 pooled), while the structural
comparator fails unmarked opposition (0.796). The architecture those results
support: the judge builds the intent key space OFFLINE (at index-build time);
queries then run against deterministic keys.

THIS PASS (over the lifted corpus, addendum 17/18):
  1. KEY EXTRACTION for every lifted issue text: batched judge calls return
     {attitude, relation, object, qualifier} per intent.
  2. FLIP RELATION LABELS: for the outcome-flip file units, the runs touching
     each file are paired (a USEFUL touch vs an ATTEMPTED/CONTEXT touch) and
     the judge labels the relation between the two intents:
     SAME / DIFFERENT / OPPOSED.

OUTPUT (untracked, public benchmark):
  data/_scratch/icm/lifted/intent_keys.jsonl        run_id -> triple
  data/_scratch/icm/lifted/flip_relations.jsonl     one row per flip pair
  data/_scratch/icm/lifted/keyspace_report.json     coverage + failures

BATCHING: 8 intents per call; per-item fallback on parse failure; cached by
text hash so reruns are free. Failures are counted, never silently dropped.

BAR (pre-registered): >= 90 percent of lifted intents receive a key; flip
relation labels produced for >= 80 percent of the multi-run units. Less is
reported as a gap (the harness then runs on the labeled subset).

Run:  python scripts/icm/intent_keyspace.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL = "deepseek-v4.1-flash:cloud"
LIFTED = os.path.join("data", "_scratch", "icm", "lifted", "codetrace_intents.jsonl")
FILES = os.path.join("data", "_scratch", "icm", "lifted", "codetrace_files.jsonl")
CACHE = os.path.join("data", "_scratch", "icm", "lifted", "keyspace_cache.jsonl")
OUTK = os.path.join("data", "_scratch", "icm", "lifted", "intent_keys.jsonl")
OUTR = os.path.join("data", "_scratch", "icm", "lifted", "flip_relations.jsonl")
REP = os.path.join("data", "_scratch", "icm", "lifted", "keyspace_report.json")
BATCH = 8


def ask(prompt: str, tries: int = 4) -> str:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.0, "num_ctx": 8192}}
    backoff = 8.0
    for _ in range(tries):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return str(json.loads(urllib.request.urlopen(req, timeout=240).read())
                       ["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
            time.sleep(3.0)
        except Exception:
            time.sleep(3.0)
    return ""


def first_json(raw: str, opener: str):
    if not raw:
        return None
    start = raw.find(opener)
    if start < 0:
        return None
    close = "]" if opener == "[" else "}"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == close:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except Exception:
                    return None
    return None


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


def save_cache(key, val):
    _cache[key] = val
    with io.open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "val": val}) + "\n")


def extract_batch(texts):
    """Return list of triples (same length as texts), None for failures."""
    head = ("For each numbered statement below, extract the intent triple. "
            "Reply with a JSON array only; each item: "
            '{"i": <number>, "attitude": "<stance verb>", "relation": "<what is '
            'to be done>", "object": "<state of affairs aimed at, few words>", '
            '"qualifier": "<state/version/time marker if any, else empty>"}.')
    prompt = head + "\n\n" + "\n".join(
        f"{i+1}. {t[:900]}" for i, t in enumerate(texts))
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
            if 0 <= idx < len(out):
                out[idx] = {k: str(item.get(k, "")) for k in
                            ("attitude", "relation", "object", "qualifier")}
    for i, x in enumerate(out):            # per-item fallback
        if x is not None:
            continue
        prompt1 = ("Extract the intent triple from the statement. Reply JSON only: "
                   '{"attitude": "...", "relation": "...", "object": "...", '
                   '"qualifier": "..."}.' + "\n\nStatement:\n" + texts[i][:900])
        obj = first_json(ask(prompt1), "{")
        if isinstance(obj, dict):
            out[i] = {k: str(obj.get(k, "")) for k in
                      ("attitude", "relation", "object", "qualifier")}
    return out


def judge_relation(a, b):
    prompt = ("Compare the two intents below, both about the same artifact. Do they "
              "require the SAME thing of it, DIFFERENT things, or OPPOSED things "
              "(acting on one would defeat the other)? Answer exactly one word: "
              "SAME, DIFFERENT or OPPOSED." + "\n\nA: " + a[:900] + "\n\nB: " + b[:900])
    raw = ask(prompt).upper()
    for w in ("OPPOSED", "DIFFERENT", "SAME"):
        if w in raw:
            return w.lower()
    return None


def main() -> int:
    load_cache()
    runs = [json.loads(l) for l in io.open(LIFTED, encoding="utf-8") if l.strip()]
    with_issue = [r for r in runs if r.get("issue")]
    print(f"[keys] {len(with_issue)} lifted intents | cache {len(_cache)}", flush=True)
    keys = {}
    todo = [r for r in with_issue
            if hashlib.sha1(r["issue"].encode()).hexdigest() not in _cache]
    print(f"  to extract: {len(todo)}", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        got = extract_batch([r["issue"] for r in chunk])
        for r, g in zip(chunk, got):
            save_cache(hashlib.sha1(r["issue"].encode()).hexdigest(), g)
            keys[r["run_id"]] = g
        print(f"  [{min(i + BATCH, len(todo))}/{len(todo)}] ({time.time() - t0:.0f}s)",
              flush=True)
    for r in with_issue:
        if r["run_id"] not in keys:
            keys[r["run_id"]] = _cache.get(hashlib.sha1(r["issue"].encode()).hexdigest())
    ok = sum(1 for v in keys.values() if v)
    with io.open(OUTK, "w", encoding="utf-8") as f:
        for r in with_issue:
            f.write(json.dumps({"run_id": r["run_id"], "agent": r["agent"],
                                "task": r.get("task_name"),
                                "key": keys.get(r["run_id"])}) + "\n")

    # ---- flip relation labels
    files = [json.loads(l) for l in io.open(FILES, encoding="utf-8") if l.strip()]
    by_file = {}
    for rec in files:
        by_file.setdefault(rec["file"], []).append(rec)
    units = {f: v for f, v in by_file.items() if len(v) >= 2}
    issue_of = {r["run_id"]: r["issue"] for r in with_issue}
    flips = []
    for fpath, v in units.items():
        use = [x for x in v if x["label"] == "useful"
               and keys.get(x["run_id"]) and issue_of.get(x["run_id"])]
        non = [x for x in v if x["label"] in ("attempted", "context")
               and issue_of.get(x["run_id"])]
        if not use or not non:
            continue
        a_rec, b_rec = use[0], non[0]
        a, b = issue_of[a_rec["run_id"]], issue_of[b_rec["run_id"]]
        ck = hashlib.sha1(("REL|" + a[:400] + "|" + b[:400]).encode()).hexdigest()
        if ck in _cache:
            rel = _cache[ck]
        else:
            rel = judge_relation(a, b)
            save_cache(ck, rel)
        flips.append({"file": fpath, "useful_run": a_rec["run_id"],
                      "nonuseful_run": b_rec["run_id"],
                      "nonuseful_label": b_rec["label"], "relation": rel})
        print(f"  [flip] {fpath[:50]} -> {rel}", flush=True)
    with io.open(OUTR, "w", encoding="utf-8") as f:
        for row in flips:
            f.write(json.dumps(row) + "\n")

    rel_dist = {}
    for row in flips:
        rel_dist[row["relation"]] = rel_dist.get(row["relation"], 0) + 1
    rep = {"intents": len(with_issue), "intents_with_key": ok,
           "key_rate": round(ok / max(1, len(with_issue)), 3),
           "units_multi_run": len(units), "flip_units_labeled": len(flips),
           "flip_rate": round(len(flips) / max(1, len(units)), 3),
           "relation_distribution": rel_dist,
           "elapsed_s": round(time.time() - t0)}
    rep["bar"] = {"key_rate_ok": rep["key_rate"] >= 0.90,
                  "flip_rate_ok": rep["flip_rate"] >= 0.80}
    with io.open(REP, "w", encoding="utf-8") as f:
        f.write(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str), flush=True)
    print(f"[done] -> {OUTK} / {OUTR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
