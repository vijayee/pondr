"""IMPLICIT-OPPOSITION TEST -- the non-circular capacity test.

WHY: the earlier capacity test (comparator_capacity_test.py) was authored by
the same person who wrote the comparator, and the comparator was then
iterated until it passed. Its 9/10 is weak evidence. Worse, the fixtures
STATE their opposition in lexical markers the comparator's rules detect
("archived" vs "NOW", "add" vs "remove"), so the test does not probe the real
capability: catching opposition that is NOT marked in words.

THIS TEST: pairs authored by the USER (the only non-circular author
available). The comparator is FROZEN -- see FREEZE below. The pairs file is
untracked (data/_scratch/icm/implicit_pairs.json) and its content is never
committed; only aggregate numbers and the file's hash are recorded.

WHAT TO AUTHOR (data/_scratch/icm/implicit_pairs.json):
  - 10 pairs with relation "opposite": two intents aimed at the SAME thing
    where acting on one would defeat the other, sharing nearly all of their
    words, WITHOUT obvious opposite words (no add/remove, before/after,
    old/new, archive/current, enable/disable). Real work examples you know.
  - 5 pairs with relation "same": the SAME intent twice, worded very
    differently (so the test is balanced -- otherwise "always say low" passes).
  - optional "note": anything you want on the record.

FREEZE: the comparator under test is struct_sim_v4 as imported from
comparator_capacity_test.py. Its sha256 is computed and recorded with the
results. NO change to the comparator may be made after seeing these pairs;
if it fails, that is the finding.

PRE-REGISTERED READOUTS:
  opposite_mean  (want <= 0.35)   v4 mean over your "opposite" pairs
  same_mean      (want >= 0.60)   v4 mean over your "same" pairs
  lex_opposite   (want >= 0.40)   the lexical baseline must be FOOLED here,
                                  else the pairs are lexically separable and
                                  the test is vacuous for our purpose
  emb_opposite                    reported (the embedding baseline)
  PASS = opposite_mean <= 0.35 AND same_mean >= 0.60 AND lex_opposite >= 0.40
  Predicted outcome (stated in advance): v4 FAILS implicit opposition -- it
  catches opposition stated in words, not opposition that must be derived
  from what each intent requires of the same memory.

Run:  python scripts/icm/implicit_opposition_test.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import comparator_capacity_test as C  # noqa: E402  (the FROZEN comparator)
from intent_similarity_ladder import emb_sim, jacc  # noqa: E402

PAIRS = os.path.join("data", "_scratch", "icm", "implicit_pairs.json")
OUT = os.path.join("data", "_scratch", "icm", "implicit_opposition_results.json")


def main() -> int:
    if not os.path.exists(PAIRS):
        print(f"no pairs file yet: write {PAIRS} (see the docstring for the "
              f"authoring spec: 10 'opposite' + 5 'same' pairs)")
        return 2
    data = json.load(io.open(PAIRS, encoding="utf-8"))
    items = [p for p in data.get("pairs", []) if not p.get("example")]
    good = [p for p in items if p.get("a") and p.get("b")
            and p.get("relation") in ("opposite", "same")]
    print(f"[pairs] {len(items)} non-example pairs ({len(good)} valid)", flush=True)
    for rel in ("opposite", "same"):
        n = sum(1 for p in good if p["relation"] == rel)
        print(f"  {rel}: {n}", flush=True)

    holder = [None]
    rows = []
    for p in good:
        a, b = p["a"], p["b"]
        rows.append({"relation": p["relation"], "note": p.get("note", ""),
                     "lex": round(jacc(a, b), 3),
                     "emb": round(emb_sim(a, b, holder), 3),
                     "v4": C.struct_sim_v4(a, b)})

    def mean(rel, inst):
        vals = [r[inst] for r in rows if r["relation"] == rel and r[inst] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    table = {inst: {rel: mean(rel, inst) for rel in ("opposite", "same")}
             for inst in ("lex", "emb", "v4")}
    o, s = table["v4"]["opposite"], table["v4"]["same"]
    lo = table["lex"]["opposite"]
    checks = {"opposite_low": (o is not None and o <= 0.35),
              "same_high": (s is not None and s >= 0.60),
              "lex_fooled": (lo is not None and lo >= 0.40)}
    res = {"n_pairs": len(rows), "table": table, "checks": checks,
           "pass": all(checks.values()),
           "comparator_sha256": hashlib.sha256(
               io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "comparator_capacity_test.py"), "rb").read()
           ).hexdigest()[:16],
           "pairs_sha256": hashlib.sha256(
               io.open(PAIRS, "rb").read()).hexdigest()[:16],
           "verdict": ("PASS -- the comparator handles UNMARKED opposition; the "
                       "structural substrate is real" if all(checks.values()) else
                       "FAIL -- the comparator catches opposition stated in words, "
                       "not opposition derived from what each intent requires of "
                       "the same memory (the pre-registered prediction)")}
    # ---- judge diagnostic (separate instrument; the comparator stays frozen)
    import argparse
    if "--judge" in sys.argv:
        from intent_detection_smoke import judge_same
        jvals = {"opposite": [], "same": []}
        for p_ in good:
            j = judge_same(p_["a"], p_["b"])
            if j is not None:
                jvals[p_["relation"]].append(1.0 if j else 0.0)
        # judge answers "same" -> 1.0; for OPPOSED pairs we want 0.0
        res["judge"] = {rel: (round(sum(v) / len(v), 3) if v else None)
                        for rel, v in jvals.items()}
        print("[judge]", res["judge"], flush=True)

    res["per_pair"] = [{"relation": r["relation"], "lex": r["lex"], "emb": r["emb"],
                        "v4": r["v4"]} for r in rows]   # numbers only, no texts
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("table", "checks", "pass", "verdict")},
                     indent=1, default=str), flush=True)
    print(f"[done] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
