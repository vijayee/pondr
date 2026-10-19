"""Session-goal census over the lifted Claude turns (census only).

WHY: the turn-level census paired consecutive directives that touched the same
file; those are often iterative follow-ups ("fix that", "now do X"), so an
opposed rate near zero may be a grain artifact rather than an absence of the
flip shape. This pass pairs SESSION-level aims instead:

  unit   = (session, file): the first SUBSTANTIVE turn in a session that
           edited the file, i.e. the aim the session brought to that unit;
  pair   = two sessions that edited the same file;
  label  = the addendum-16 judge prompt (SAME / DIFFERENT / OPPOSED) on the
           two session aims.

PRE-REGISTERED (before running):
  (1) sample up to --relations pairs, seeded (default 80, seed 42);
  (2) report the relation distribution and opposed rate, stratified by path
      class (memory_note / abs_path / project_rel);
  (3) this is a CENSUS of whether the flip shape exists in the corpus. It makes
      no retrieval claim and trains nothing. An opposed rate near zero means
      this corpus fails the same census as CodeTraceBench and Onyx at this
      grain too, and must be reported that way.

Usage:
  python scripts/icm/census_session_goals.py --relations 80
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE = os.path.join("data", "_scratch", "icm", "lifted", "claude")
TURNS = os.path.join(BASE, "claude_turns.jsonl")
RELATIONS = os.path.join(BASE, "claude_session_relations.jsonl")
OUT = os.path.join(BASE, "claude_session_census.json")
JUDGE_CACHE = os.path.join(BASE, "claude_judge_cache.jsonl")


def path_class(p: str) -> str:
    q = str(p).replace("\\", "/").lower()
    if ".claude/" in q or "/memory/" in q:
        return "memory_note"
    if q.startswith("c:/") or q.startswith("/"):
        return "abs_path"
    return "project_rel"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relations", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    turns = [json.loads(line) for line in io.open(TURNS, encoding="utf-8")
             if line.strip()]
    # first substantive edit turn per (session, file)
    aims: dict[tuple[str, str], dict] = {}
    for t in sorted(turns, key=lambda x: (x["ts"] or "", x["turn_id"])):
        if not t.get("substantive"):
            continue
        for f in t.get("files_edited") or []:
            key = (t["session_id"], f)
            if key not in aims:
                aims[key] = t
    by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for (_sid, f), t in aims.items():
        by_file[f].append(t)
    pairs = []
    for f, ts in by_file.items():
        if len(ts) < 2:
            continue
        ts = sorted(ts, key=lambda x: (x["ts"] or "", x["turn_id"]))
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                if ts[i]["session_id"] != ts[j]["session_id"]:
                    pairs.append({"file": f, "a": ts[i], "b": ts[j]})
    pairs.sort(key=lambda p: (p["file"], p["a"]["turn_id"], p["b"]["turn_id"]))
    print(f"[session census] units={len(by_file)} shared={sum(1 for v in by_file.values() if len(v) >= 2)} "
          f"pairs={len(pairs)}", flush=True)

    report = {"units": len(by_file),
              "shared_units": sum(1 for v in by_file.values() if len(v) >= 2),
              "pairs": len(pairs), "relations": None}
    if args.relations > 0 and pairs:
        from icm.intent_keys import IntentKeyer
        rng = random.Random(args.seed)
        chosen = rng.sample(pairs, min(args.relations, len(pairs)))
        keyer = IntentKeyer(cache_path=JUDGE_CACHE)
        counts: dict[str, int] = {}
        by_class: dict[str, dict[str, int]] = collections.defaultdict(dict)
        rows = []
        for i, p in enumerate(chosen, 1):
            rel = keyer.relation_between(p["a"]["intent_text"],
                                         p["b"]["intent_text"]) or "unparsed"
            counts[rel] = counts.get(rel, 0) + 1
            k = path_class(p["file"])
            by_class[k][rel] = by_class[k].get(rel, 0) + 1
            rows.append({"file": p["file"], "path_class": k,
                         "a_turn": p["a"]["turn_id"], "b_turn": p["b"]["turn_id"],
                         "a_session": p["a"]["session_id"],
                         "b_session": p["b"]["session_id"], "relation": rel})
            if i % 10 == 0:
                print(f"  [judge {i}/{len(chosen)}] {json.dumps(counts)}", flush=True)
        with io.open(RELATIONS, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        report["relations"] = {
            "n": len(rows), "counts": counts,
            "opposed_rate": round(counts.get("opposed", 0) / len(rows), 3),
            "by_path_class": {k: dict(v) for k, v in by_class.items()},
            "sample_seed": args.seed,
        }
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
