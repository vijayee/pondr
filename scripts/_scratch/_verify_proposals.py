"""Dump anaphora proposals into a readable review sheet for human verification
(Task #122). UNTRACKED scratch.

For each proposed (query, target) pair, print the USER turns spanning target..query
so the verifier can judge whether the target is the SPECIFIC earlier turn the query
points back to -- or whether a nearer turn in between is the real referent (retarget),
or the anaphora is too diffuse to label cleanly (reject).

Short spans (age <= SPAN_FULL) are shown in full; long spans show target +- context
and query +- context with an ellipsis for the middle.

Run:
  PYTHONPATH=. python scripts/_scratch/_verify_proposals.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
EP_PATH = SCRATCH / "_trained_episodes_for_labeling.json"
PROP_PATH = SCRATCH / "_trained_gold_proposals.json"
OUT_PATH = SCRATCH / "_trained_gold_review.txt"

SPAN_FULL = 8      # age <= this -> show full target..query span
SIDE = 2           # turns of context either side for long spans
TRUNC = 520


def _short(txt: str) -> str:
    t = " ".join(txt.split())
    return t[:TRUNC] + (" ..." if len(t) > TRUNC else "")


def main() -> int:
    ep = json.loads(EP_PATH.read_text(encoding="utf-8"))
    prop = json.loads(PROP_PATH.read_text(encoding="utf-8"))
    lines: list[str] = []
    pid = 0
    for sid, pairs in prop.items():
        s = ep[sid]
        ut = [t["text"] for t in s["turns"] if t["role"] == "user"]
        n = len(ut)
        lines.append("\n" + "=" * 100)
        lines.append(f"SESSION {sid[:8]} {s['name']!r}  ({n} user turns, {len(pairs)} proposals)")
        lines.append("=" * 100)
        for p in sorted(pairs, key=lambda x: (x["query"], x["target"])):
            q, t, age = p["query"], p["target"], p["age"]
            pid += 1
            lines.append(f"\n--- P{pid:03d}  q{q:02d} -> t{t:02d}  (age {age})  "
                         f"reason: {p.get('reason','')}")
            if age <= SPAN_FULL:
                span = range(t, q + 1)
                for i in span:
                    tag = "  QUERY" if i == q else ("  TARGET" if i == t else "       ")
                    lines.append(f"  u{i:02d}{tag}: {_short(ut[i])}")
            else:
                # target +- SIDE, then gap, then query +- SIDE
                head = list(range(max(0, t - SIDE), t + SIDE + 1))
                tail = list(range(max(t + SIDE + 1, q - SIDE), q + SIDE + 1))
                for i in head:
                    tag = "  TARGET" if i == t else "       "
                    if 0 <= i < n:
                        lines.append(f"  u{i:02d}{tag}: {_short(ut[i])}")
                lines.append(f"  ... ({q - t - 1 - 2*SIDE} turns between) ...")
                for i in tail:
                    tag = "  QUERY" if i == q else "       "
                    if 0 <= i < n:
                        lines.append(f"  u{i:02d}{tag}: {_short(ut[i])}")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    tot = sum(len(v) for v in prop.values())
    print(f"[done] {tot} proposals -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())