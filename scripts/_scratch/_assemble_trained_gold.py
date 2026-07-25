"""Assemble verified gold anaphora pairs into _trained_gold.json (Task #122).
UNTRACKED scratch. PRIVATE onyx -- never uploaded.

Input:
  _trained_gold_proposals.json  -- LLM proposals {sid: [pairs]}
  _trained_verdicts.json        -- human verdicts {f"P{pid:03d}": {verdict, retarget, note}}
    verdict in {"accept","reject"}. retarget = int new target index (optional,
    only for accept-with-retarget). note = one-line audit reason (the human's
    judgment -- same standard as the 17 held-out).

Output:
  _trained_gold.json  -- {window, age_threshold, pairs:[{session_id, query,
    target, age, note}]} in the SAME shape as _heldout_gold.json so the CE
    eval/fine-tune harness reuses without adaptation.

pid ordering MUST match _verify_proposals.py: iterate prop.items() in dict order,
sort each session's pairs by (query, target), pid increments 1..N across sessions.

Run:
  PYTHONPATH=. python scripts/_scratch/_assemble_trained_gold.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
PROP_PATH = SCRATCH / "_trained_gold_proposals.json"
VERDICT_PATH = SCRATCH / "_trained_verdicts.json"
OUT_PATH = SCRATCH / "_trained_gold.json"

WINDOW = 16
AGE_THRESHOLD = 3


def main() -> int:
    prop = json.loads(PROP_PATH.read_text(encoding="utf-8"))
    verdicts = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))

    pid = 0
    out_pairs: list[dict] = []
    n_accept = n_retarget = n_reject = n_missing = 0
    for sid, pairs in prop.items():
        for p in sorted(pairs, key=lambda x: (x["query"], x["target"])):
            pid += 1
            key = f"P{pid:03d}"
            v = verdicts.get(key)
            if v is None:
                n_missing += 1
                continue
            verdict = v.get("verdict", "reject")
            if verdict == "reject":
                n_reject += 1
                continue
            q = int(p["query"])
            t = int(v.get("retarget", p["target"]))
            age = q - t - 1
            if not (0 <= t < q):
                print(f"[WARN] {key} sid {sid[:8]} retarget t{t} violates t<q{q}; dropping")
                continue
            if age < AGE_THRESHOLD:
                print(f"[WARN] {key} sid {sid[:8]} age {age} < {AGE_THRESHOLD}; dropping")
                continue
            if t != int(p["target"]):
                n_retarget += 1
            else:
                n_accept += 1
            out_pairs.append({
                "session_id": sid,
                "query": q,
                "target": t,
                "age": age,
                "note": str(v.get("note", "")).strip(),
            })

    out = {"window": WINDOW, "age_threshold": AGE_THRESHOLD, "pairs": out_pairs}
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[done] {len(out_pairs)} gold pairs "
          f"(accept={n_accept} retarget={n_retarget} reject={n_reject} "
          f"missing_verdict={n_missing}) -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())