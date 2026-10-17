"""COMPARATOR CAPACITY TEST -- can the operator represent OPPOSITION at all?

WHY (measured, addendum 13 + the two ladder runs): every instrument called
OPPOSITE intents similar -- lexical 0.471, embeddings 0.906, structural-rules
0.667 -- on pairs whose surface overlaps almost completely ("report the
archived sha" vs "report the sha that holds now"). The structural comparator
v1 had NO representation of opposition beyond a small antonym-verb list, and
none for STATE/TEMPORAL opposition (archived vs current), which is what the
decisive pairs actually are. This test asks the capability question only:
can the operator separate these at all, on cases with known answers?

THE FIX UNDER TEST (comparator v2 -- a SCHEMA EXTENSION, not more lexicon):
  (a) the object slot carries a STATE QUALIFIER, read by deterministic rules
      from the raw text: ARCHIVED {archived, original, stored, snapshot,
      recorded, stale, old, previous, existing, legacy, at ingest} vs
      CURRENT {current, fresh, now, updated, new, latest, live, re-ingested,
      recomputed, after, refreshed} (and PRIOR/NEXT for before/after).
  (b) OPPOSITION RULES, all deterministic and applied to the raw text:
      - state opposition: one side ARCHIVED, other CURRENT, on overlapping
        objects -> heavy penalty (this is the measured reversal).
      - antonym relations: declared opposite verb classes (ADD/REMOVE,
        PROTECT/REMOVE, ENABLE/DISABLE, KEEP/DELETE, INCREASE/DECREASE,
        CREATE/DESTROY, OPEN/CLOSE).
      - negation mismatch: one side carries a negation marker the other
        lacks, on overlapping objects -> penalty.
  (c) everything v1 had is retained: predicate class match, entity overlap,
      refinement bonus.
  v2 score = v1 score - 0.7*state_opp - 0.7*antonym - 0.3*negation_mismatch,
  clipped to [0,1]. All constants fixed a priori.

  NOTE: the state qualifier and the opposition rules are RULE-BASED ON RAW
  TEXT -- no model. Only the triple (attitude/relation/object) still comes
  from the extraction model, exactly as in v1, so any improvement is
  attributable to the SCHEMA, not to a better model.

FIXTURES (all constructed/authored, declared; NO mined corpora, no personal
text -- this is a capability test, not an integration test):
  R_SAME        (2)  identical strings                     want >= 0.90
  R_PARAPHRASE  (6)  same intent, different wording       want >= 0.60
  R_REFINEMENT  (6)  same aim, more specific object       want >= 0.60
  R_UNRELATED   (6)  different aims, no shared entity     want <= 0.35
  R_REV_ANTONYM (6)  same entity, opposite verb           want <= 0.35
  R_REV_STATE   (12) same memory, archived-vs-current     want <= 0.35
  R_NEGATION    (4)  same words, one negated              want <= 0.40

PRE-REGISTERED BAR (all):
  (1) v2 <= 0.35 on R_REV_ANTONYM and R_REV_STATE; <= 0.40 on R_NEGATION
  (2) v2 >= 0.60 on R_PARAPHRASE and R_REFINEMENT; >= 0.90 on R_SAME
  (3) v2 <= 0.35 on R_UNRELATED
  (4) NO REGRESSION: v2's R_PARAPHRASE and R_REFINEMENT means >= v1's - 0.10
  (5) NOT VACUOUS: lexical is FOOLED on >= 1 reversal rung (lex >= 0.40)
  PASS -> the schema extension gives the operator the capacity; the
  CodeTraceBench lift becomes the integration test.
  FAIL -> itemized; the honest conclusion is then that intent comparison
  needs semantics no structural operator supplies (a real, recordable
  negative about the approach).

Run:  python scripts/icm/comparator_capacity_test.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

from intent_similarity_ladder import (  # noqa: E402
    OPPOSITE, PRED_CLASSES, emb_sim, jacc, pred_class, struct_sim, toks, triple,
)

OUT = os.path.join("data", "_scratch", "icm", "comparator_capacity_results.json")

# ------------------------------------------------------- v2: schema pieces --
STATE_TAGS = {
    "ARCHIVED": ("archived", "original", "stored", "snapshot", "recorded", "stale",
                 "old", "previous", "existing", "legacy", "at ingest", "as ingested",
                 "historic", "earlier", "backup"),
    "CURRENT": ("current", "fresh", "now", "updated", "new", "latest", "live",
                "re-ingested", "reingested", "recomputed", "refreshed", "present",
                "today", "just", "actual"),
}
STATE_OPPOSED = (("ARCHIVED", "CURRENT"),)
NEG_MARKERS = ("do not", "don't", "not ", "never", "avoid", "without", "no longer",
               "must not", "cannot", "can't", "refuse", "skip", "ignore")
EXTRA_ANTONYMS = (("ENABLE", "DISABLE"), ("INCREASE", "DECREASE"),
                  ("CREATE", "DESTROY"), ("OPEN", "CLOSE"))


def state_tags(text: str) -> set:
    low = " " + str(text).lower() + " "
    out = set()
    for tag, words in STATE_TAGS.items():
        if any(w in low for w in words):
            out.add(tag)
    return out


def state_opposition(a: str, b: str) -> bool:
    ta, tb = state_tags(a), state_tags(b)
    if not ta or not tb or ta == tb:
        return False
    for x, y in STATE_OPPOSED:
        if (x in ta and y in tb) or (y in ta and x in tb):
            return True
    return False


def antonym_relations(a: str, b: str, t1, t2) -> bool:
    c1 = pred_class((t1["attitude"] + " " + t1["relation"]) if t1 else a) or pred_class(a)
    c2 = pred_class((t2["attitude"] + " " + t2["relation"]) if t2 else b) or pred_class(b)
    if not c1 or not c2:
        return False
    for x, y in tuple(OPPOSITE) + EXTRA_ANTONYMS:
        if (c1, c2) == (x, y) or (c1, c2) == (y, x):
            return True
    return False


def negation_mismatch(a: str, b: str) -> bool:
    la, lb = str(a).lower(), str(b).lower()
    na = any(m in la for m in NEG_MARKERS)
    nb = any(m in lb for m in NEG_MARKERS)
    return na != nb


def struct_sim_v2(a: str, b: str):
    """v1 score minus the schema-extension opposition penalties (rule-based)."""
    base = struct_sim(a, b)
    if base is None:
        return None
    t1, t2 = triple(a), triple(b)
    # opposition only counts when the OBJECTS overlap (else it is just a
    # different subject, already handled by base)
    o1 = toks(t1["object"]) if t1 else toks(a)
    o2 = toks(t2["object"]) if t2 else toks(b)
    overlap = bool(o1 and o2 and len(o1 & o2) / len(o1 | o2) >= 0.2)
    pen = 0.0
    if overlap:
        if state_opposition(a, b):
            pen += 0.7
        if antonym_relations(a, b, t1, t2):
            pen += 0.7
        if negation_mismatch(a, b):
            pen += 0.3
    return max(0.0, min(1.0, base - pen))


# ------------------------------------------------------------- the fixtures --
def fixtures():
    P = []
    # R_SAME
    P += [("R_SAME", "verify the fix passes the existing test suite",
           "verify the fix passes the existing test suite")]
    P += [("R_SAME", "add the missing index to the migrations",
           "add the missing index to the migrations")]
    # R_PARAPHRASE (same intent, different wording)
    P += [("R_PARAPHRASE", "Run the test suite and inspect the failing tests",
           "Execute the tests and look at which ones fail")]
    P += [("R_PARAPHRASE", "Locate the function that needs to change",
           "Find the code path responsible for the behaviour")]
    P += [("R_PARAPHRASE", "Remove the deprecated compatibility shim",
           "Delete the old fallback that is no longer needed")]
    P += [("R_PARAPHRASE", "Update the documentation for the new parameter",
           "Modify the docs to describe the added argument")]
    P += [("R_PARAPHRASE", "Reproduce the bug with a minimal example",
           "Write the smallest snippet that triggers the same error")]
    P += [("R_PARAPHRASE", "Bump the dependency to the fixed release",
           "Upgrade the library to the version containing the patch")]
    # R_REFINEMENT (same aim, more specific)
    P += [("R_REFINEMENT", "fix the SQL syntax error",
           "fix the SQL syntax error in Count with distinct and Case")]
    P += [("R_REFINEMENT", "add a new ModelAdmin hook",
           "add a new ModelAdmin.get_inlines hook returning self.inlines by default")]
    P += [("R_REFINEMENT", "resolve the issue", "resolve the issue about FITS card quoting")]
    P += [("R_REFINEMENT", "improve the error message", "improve the error message shown on failure")]
    P += [("R_REFINEMENT", "run the tests", "run the tests for the queryset module")]
    P += [("R_REFINEMENT", "update the docs", "update the docs for the new parameter")]
    # R_UNRELATED (different aims, no shared entity)
    P += [("R_UNRELATED", "add a database index to speed up the query",
           "write a regression test for the reported parsing bug")]
    P += [("R_UNRELATED", "delete the legacy authentication provider",
           "summarise the release notes for the next version")]
    P += [("R_UNRELATED", "refactor the logging setup",
           "increase the timeout for the CI runner")]
    P += [("R_UNRELATED", "document the public API",
           "remove the unused imports in the CLI module")]
    P += [("R_UNRELATED", "translate the error messages",
           "cache the results of the slow query")]
    P += [("R_UNRELATED", "upgrade the build tool",
           "rename the configuration flag")]
    # R_REV_ANTONYM (same entity, opposite verb)
    P += [("R_REV_ANTONYM", "add the backwards-compatibility shim to the parser",
           "remove the backwards-compatibility shim from the parser")]
    P += [("R_REV_ANTONYM", "enable the feature flag for the new loader",
           "disable the feature flag for the new loader")]
    P += [("R_REV_ANTONYM", "keep the existing cache layer in front of the database",
           "delete the existing cache layer in front of the database")]
    P += [("R_REV_ANTONYM", "increase the retry limit for the uploader",
           "decrease the retry limit for the uploader")]
    P += [("R_REV_ANTONYM", "create the migration for the new column",
           "destroy the migration for the new column")]
    P += [("R_REV_ANTONYM", "open the connection pool at start-up",
           "close the connection pool at start-up")]
    # R_REV_STATE (the measured 2x2 shape: same memory, archived vs current)
    for i in range(12):
        P += [("R_REV_STATE",
               f"Compliance check for part-0 (run {i:04d}): report the manifest sha "
               f"recorded at the original ingest.",
               f"Compliance check for part-0 (run {i:04d}): the dataset was just "
               f"re-ingested; report the manifest sha that holds NOW.")]
    # R_NEGATION (same words, one negated)
    P += [("R_NEGATION", "trust the archived manifest record when writing the report",
           "do not trust any archived manifest record when writing the report")]
    P += [("R_NEGATION", "use the cached result for the expensive query",
           "never use the cached result for the expensive query")]
    P += [("R_NEGATION", "keep the fallback path for older Python versions",
           "avoid the fallback path for older Python versions")]
    P += [("R_NEGATION", "write the summary from the stored snapshots",
           "write the summary without using the stored snapshots")]
    return P


def main() -> int:
    pairs = fixtures()
    holder = [None]
    rows = []
    t0 = time.time()
    print(f"[fixtures] {len(pairs)} pairs", flush=True)
    for i, (rung, a, b) in enumerate(pairs):
        rows.append({"rung": rung, "a": a, "b": b,
                     "lex": round(jacc(a, b), 3),
                     "emb": round(emb_sim(a, b, holder), 3),
                     "v1": struct_sim(a, b),
                     "v2": struct_sim_v2(a, b)})
        if (i + 1) % 8 == 0:
            r = rows[-1]
            print(f"  [{i+1}/{len(pairs)}] {rung} v1={r['v1']} v2={r['v2']} "
                  f"lex={r['lex']} ({time.time()-t0:.0f}s)", flush=True)

    rungs = sorted({r["rung"] for r in rows})

    def mean(inst, rung):
        vals = [r[inst] for r in rows if r["rung"] == rung and r[inst] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    table = {inst: {rg: mean(inst, rg) for rg in rungs} for inst in ("lex", "emb", "v1", "v2")}
    g = table["v2"].get
    checks = {
        "rev_antonym_low": (g("R_REV_ANTONYM") or 1) <= 0.35,
        "rev_state_low": (g("R_REV_STATE") or 1) <= 0.35,
        "negation_low": (g("R_NEGATION") or 1) <= 0.40,
        "paraphrase_high": (g("R_PARAPHRASE") or 0) >= 0.60,
        "refinement_high": (g("R_REFINEMENT") or 0) >= 0.60,
        "same_ceiling": (g("R_SAME") or 0) >= 0.90,
        "unrelated_low": (g("R_UNRELATED") or 1) <= 0.35,
        "no_regression_paraphrase": (g("R_PARAPHRASE") or 0)
        >= (table["v1"].get("R_PARAPHRASE") or 0) - 0.10,
        "no_regression_refinement": (g("R_REFINEMENT") or 0)
        >= (table["v1"].get("R_REFINEMENT") or 0) - 0.10,
        "not_vacuous": max(table["lex"].get(r, 0) or 0
                           for r in ("R_REV_ANTONYM", "R_REV_STATE", "R_NEGATION")) >= 0.40,
    }
    res = {"n_pairs": len(rows), "table": table, "checks": checks,
           "pass": all(checks.values()),
           "verdict": ("PASS -- the schema extension gives the operator the capacity; "
                       "the CodeTraceBench lift is the next (integration) test"
                       if all(checks.values()) else
                       "FAIL -- itemized below; the failing rung names the missing piece"),
           "failures": [r for r in rows
                        if (r["rung"].startswith("R_REV") and (r["v2"] or 0) > 0.35)
                        or (r["rung"] in ("R_PARAPHRASE", "R_REFINEMENT")
                            and (r["v2"] or 0) < 0.60)][:14]}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("table", "checks", "pass", "verdict")},
                     indent=1, default=str), flush=True)
    print(f"[done] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
