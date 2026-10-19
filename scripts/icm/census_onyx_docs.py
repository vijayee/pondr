"""Onyx shared-document census (census only).

WHY: the Claude-session census found no opposed aims at two grains. Onyx is the
other personal corpus, and its retrieval trace is explicit: each assistant
message records the documents it pulled for the prompting user message. This
pass asks whether the SAME document is ever retrieved under OPPOSED aims.

  unit  = a document id/link in the Onyx retrieval trace;
  pair  = two user queries that retrieved the same document;
  label = the addendum-16 judge prompt (SAME / DIFFERENT / OPPOSED).

PRE-REGISTERED (before running):
  (1) sample up to --relations pairs, seeded (default 80, seed 42);
  (2) report the distribution, the opposed rate, and the shared-document counts;
  (3) census only -- no retrieval claim, no training. A near-zero opposed rate
      replicates the prior 0/6336 finding at this grain and must be reported.

Usage:
  python scripts/icm/census_onyx_docs.py --relations 80
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

BASE = os.path.join("data", "_scratch", "icm", "capture")
MINED = os.path.join(BASE, "onyx_mined.jsonl")
OUT = os.path.join(BASE, "onyx_doc_census.json")
RELATIONS = os.path.join(BASE, "onyx_doc_relations.jsonl")
CACHE = os.path.join(BASE, "onyx_judge_cache.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relations", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    docs: dict[str, list[str]] = collections.defaultdict(list)
    events = 0
    for line in io.open(MINED, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("level") != "retrieval":
            continue
        events += 1
        q = (r.get("query") or "").strip()
        if not q:
            continue
        for d in r.get("retrieved") or []:
            docs[str(d)].append(q)
    shared = {d: v for d, v in docs.items() if len(v) >= 2}
    pairs = []
    for d, qs in shared.items():
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                if qs[i] != qs[j]:
                    pairs.append({"doc": d, "a": qs[i], "b": qs[j]})
    pairs.sort(key=lambda p: (p["doc"], p["a"], p["b"]))
    print(f"[onyx census] events={events} docs={len(docs)} shared={len(shared)} "
          f"pairs={len(pairs)}", flush=True)
    report = {"retrieval_events": events, "documents": len(docs),
              "shared_documents": len(shared), "pairs": len(pairs),
              "relations": None}
    if args.relations > 0 and pairs:
        from icm.intent_keys import IntentKeyer
        rng = random.Random(args.seed)
        chosen = rng.sample(pairs, min(args.relations, len(pairs)))
        keyer = IntentKeyer(cache_path=CACHE)
        counts: dict[str, int] = {}
        rows = []
        for i, p in enumerate(chosen, 1):
            rel = keyer.relation_between(p["a"], p["b"]) or "unparsed"
            counts[rel] = counts.get(rel, 0) + 1
            rows.append({"doc": p["doc"], "relation": rel})
            if i % 10 == 0:
                print(f"  [judge {i}/{len(chosen)}] {json.dumps(counts)}", flush=True)
        with io.open(RELATIONS, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        report["relations"] = {
            "n": len(rows), "counts": counts,
            "opposed_rate": round(counts.get("opposed", 0) / len(rows), 3),
            "sample_seed": args.seed,
        }
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
