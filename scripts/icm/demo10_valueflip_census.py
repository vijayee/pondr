"""DEMO #10 gate-0 -- the organic value-flip CENSUS.

Pre-registered question (from the second agent's final round, adjudicated
2026-10-15): can the corpus even carry "same memory, different value under
different goals" NATURALLY? This census measures it per dataset:

    shared sessions   = a session id appearing in >= 2 questions' haystacks
    value-flip        = a shared session whose gold-status DIFFERS across the
                        questions that share it (gold for one, noise for other)

On LongMemEval (full 500): measured 0 of 940 sessions flip value (8 shared,
0 mixed) -- the haystacks are woven per-question, so organic intent-variation
is DESIGNED OUT. Any conclusion about intent-conditioned relevance from this
corpus is corpus-limited, and the phase report now carries that caveat.

Usage:
    python scripts/icm/demo10_valueflip_census.py                    # LongMemEval
    python scripts/icm/demo10_valueflip_census.py --file <LoCoMo json/ndjson>
Output: data/_scratch/icm/demo10_valueflip_census.json
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FULL500 = (r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness"
           r"\datasets\longmemeval\longmemeval_interleaved_full.json")
OUT = os.path.join("data", "_scratch", "icm", "demo10_valueflip_census.json")


def census(records, gold_of):
    """records: iterable; gold_of(rec) -> set of session ids that are gold."""
    sid_to_q = collections.defaultdict(list)
    for qi, rec in enumerate(records):
        gold = gold_of(rec)
        sids = rec.get("haystack_session_ids") or []
        for sid in sids:
            sid_to_q[sid].append((qi, sid in gold))
    shared = {s: v for s, v in sid_to_q.items() if len(v) >= 2}
    mixed = {s: v for s, v in shared.items()
             if len({g for _, g in v}) > 1}
    n_flip_pairs = sum(len(v) - 1 for v in mixed.values())
    return {"records": len(records), "sessions": len(sid_to_q),
            "shared": len(shared), "value_flips": len(mixed),
            "flip_question_pairs": n_flip_pairs,
            "examples": [{"sid": s, "uses": v} for s, v in
                         list(mixed.items())[:5]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None,
                    help="optional dataset file (LoCoMo etc.); default LongMemEval full 500")
    args = ap.parse_args()

    if args.file:
        data = json.load(io.open(args.file, encoding="utf-8"))
        if isinstance(data, dict):
            data = list(data.values())
        stats = census(data, lambda rec: set(rec.get("answer_session_ids") or
                                             rec.get("gold_session_ids") or []))
        stats["dataset"] = os.path.basename(args.file)
    else:
        data = json.load(io.open(FULL500, encoding="utf-8"))
        stats = census(data, lambda rec: set(rec["answer_session_ids"]))
        stats["dataset"] = "longmemeval_interleaved_full"
        stats["verdict_context"] = (
            "0 value-flips => this corpus cannot measure organic "
            "intent-variation; a non-zero census is the GATE-0 pass for the "
            "intent-conditioned demo #10 (then: per shared session, check "
            "whether any cheap leg separates its gold-questions from "
            "non-gold-questions; the residue = the intent-shaped signal)")
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "examples"}, indent=2))
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())