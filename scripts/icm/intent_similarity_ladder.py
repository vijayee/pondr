"""INTENT SIMILARITY LADDER -- the bridge: a GRADED, polarity-aware intent
comparison, tested against a known-order ladder BEFORE any payoff claim.

WHY (addendum 13, commit 969557f): the detection smoke found extraction works
but the comparison operator was string overlap -- a lexical comparator in a
structured hat (invariance 0.00). Binary same/different was also the wrong
vocabulary: the "changed intent" pairs turned out to be REFINEMENTS. Intent
relations are graded.

THE BRIDGE = three pieces:
  (1) PREDICATE CANONICALIZATION -- observed attitude/relation verbs mapped to
      a small declared class set (the vocabulary is small by measurement:
      the SPO audit saw intend/verify/reclaim/maintain/ensure dominate).
  (2) ENTITY RESOLUTION on the object slot -- file/component identity. The
      CodeTrace corpus supplies the anchors (5,147 file-access records; the
      flip table's 154 shared file paths).
  (3) POLARITY + REFINEMENT in the comparison -- `add X` vs `remove X` (same
      entity, opposite predicate) must score LOW; `fix error` vs `fix error
      in Count with CASE` must score HIGH. String overlap expresses neither.

LADDER (public data only; cloud models never see personal corpora):
  R1 identical   (ceiling) : 20 duplicated goal texts (same string, 58 exist)
  R2 paraphrase  AUTHORED  : 20 base plan-phase statements + authored
                             paraphrases of them (DECLARED as authored: the
                             benchmark cannot supply these -- its goals are
                             template `resolve {task} ({agent})`, plans exist
                             only for OpenHands, and NO task is run twice by
                             OpenHands, so run-to-run paraphrase pairs = 0.
                             Reported as a DATA GAP, not hidden.)
  R3 sub-intent  (high)    : 40 pairs: task goal <-> the agent's own declared
                             plan; plus plan RE-DECLARATION pairs (the
                             refinements addendum 13 mislabelled)
  R4 sibling     (low-mid) : 40 pairs: two different tasks on the same repo
  R5 unrelated   (low)     : 20 pairs: tasks from different repos
  R6 reversal    (LOWEST)  : 40 pairs: the SAME file touched by two different
                             tasks (flip table) -- maximum surface, different
                             aim; plus the 11 metric_reversal pairs
                             (X's gold session under X's goal vs Y's goal)

INSTRUMENTS (each returns a similarity in [0,1]; higher = more similar):
  LEX    : token jaccard of the two statements (the floor that must be beaten
           ON REVERSALS).
  EMB    : cosine of bge-small embeddings (the measured-dead route; included
           as the second baseline).
  STRUCT : extraction (deepseek-v4.1-flash -- public text only) ->
           canonical triple -> graded score:
             class_match  = 0.5 if predicate classes match else 0.0
             entity_match = 0.5 * (object token jaccard >= 0.3 ? 1 : 0)
             refinement   = +0.2 if one object's meaningful tokens are a
                            strict subset of the other's (>=2 extra tokens)
             polarity     = -0.7 if the relation pair is in the declared
                            opposite lexicon (add/remove, enable/disable,
                            keep/delete, create/destroy, open/close)
             score = clip(class + entity + refinement - polarity, 0, 1)
           (all constants fixed A PRIORI, never tuned)

READOUTS (direction always):
  mean similarity per rung per instrument; Spearman between rung index and
  per-pair similarity; and the DECISIVE reversal readout:
     STRUCT.reversal_mean < STRUCT.subintent_mean - 0.25  AND
     STRUCT.reversal_mean < LEX.reversal_mean             (less fooled by
                                                           shared surface)
PRE-REGISTERED BAR (all):
  (1) STRUCT separation: mean(R2,R3) >= 0.60 and mean(R4,R5,R6) <= 0.35
  (2) STRUCT reversal gap: sub-intent mean - reversal mean >= 0.25
  (3) STRUCT beats LEX on reversals (lower mean) and on the separation margin
  PASS -> the comparator represents polarity/refinement; only then does the
  payoff question (does intent similarity predict memory reuse) get asked.
  FAIL -> reported with itemized pairs; the failure mode names which piece
  (canonicalization, entity resolution, or polarity) is missing.
  VACUITY GUARD: if LEX already separates the ladder (R2/R3 high, R4-R6 low),
  the ladder is lexically trivial and the smoke is VACUOUS -- reported.

Run:  python scripts/icm/intent_similarity_ladder.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

MAPPED = os.path.join("data", "_scratch", "icm", "capture", "codetrace_mapped.jsonl")
FLIP = os.path.join("data", "_scratch", "codetrace", "codetrace_flip_table.json")
REVPRECHECK = os.path.join("data", "_scratch", "icm", "metric_reversal_precheck.json")
XCACHE = os.path.join("data", "_scratch", "icm", "intent_ladder_extractions.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "intent_similarity_ladder_results.json")
MODEL = "deepseek-v4.1-flash:cloud"

# ---------------------------------------------------------------- lexicons --
PRED_CLASSES = {
    "REPAIR": ("fix", "repair", "resolve", "correct", "patch", "restore", "debug"),
    "ADD": ("add", "create", "implement", "introduce", "insert", "append", "write", "build"),
    "REMOVE": ("remove", "delete", "drop", "deprecate", "delete", "erase", "uninstall"),
    "VERIFY": ("verify", "check", "confirm", "test", "validate", "ensure", "assert",
               "reproduce", "inspect", "examine"),
    "UNDERSTAND": ("understand", "read", "restate", "clarify", "explore", "analyze",
                   "investigate", "locate", "find", "search", "plan", "rewording", "reading"),
    "RUN": ("run", "execute", "invoke", "launch", "reproduce"),
    "PROTECT": ("protect", "preserve", "maintain", "keep", "guard", "avoid", "prevent"),
    "CHANGE": ("change", "modify", "update", "refactor", "rewrite", "move", "rename", "bump"),
}
OPPOSITE = (("ADD", "REMOVE"), ("PROTECT", "REMOVE"), ("ADD", "PROTECT"))
STOP = {"the", "a", "an", "of", "to", "in", "for", "and", "or", "with", "on", "by",
        "that", "this", "it", "is", "are", "be", "as", "at", "from", "into", "its",
        "their", "our", "any", "all", "new", "when", "how", "what", "not", "no"}


def ask(prompt: str, tries: int = 4) -> str:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.0, "num_ctx": 4096}}
    backoff = 8.0
    for _ in range(tries):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return str(json.loads(urllib.request.urlopen(req, timeout=180).read())
                       ["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
            time.sleep(2.0)
        except Exception:
            time.sleep(2.0)
    return ""


def toks(v):
    return {w for w in re.findall(r"[a-z0-9_]+", str(v).lower())
            if len(w) > 2 and w not in STOP}


def jacc(a, b):
    A, B = toks(a), toks(b)
    u = A | B
    return len(A & B) / len(u) if u else 0.0


def pred_class(s: str):
    """Map a statement's attitude+relation words to PRED_CLASSES; else None."""
    t = toks(s)
    hits = [cls for cls, words in PRED_CLASSES.items() if t & set(words)]
    if not hits:
        return None
    # deterministic tie-break: the class whose matched word appears first in text
    low = str(s).lower()
    hits.sort(key=lambda c: min(low.find(w) for w in PRED_CLASSES[c] if w in low))
    return hits[0]


_extract_cache = {}


def triple(text: str):
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if key in _extract_cache:
        return _extract_cache[key]
    p = ("Extract the intent triple from the statement below. Reply with JSON "
         'only, keys: "attitude" (stance verb), "relation" (what is to be done), '
         '"object" (the state of affairs aimed at, in a few words).\n\nStatement:\n'
         + text[:1200])
    raw = ask(p)
    m = re.search(r"\{.*\}", raw or "", re.S)
    t = None
    if m:
        try:
            d = json.loads(m.group(0))
            t = {k: str(d.get(k, "")) for k in ("attitude", "relation", "object")}
        except Exception:
            t = None
    _extract_cache[key] = t
    with io.open(XCACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "text": text[:400], "triple": t}) + "\n")
    return t


def struct_sim(a: str, b: str):
    """The bridge: canonicalized structural comparison with polarity/refinement."""
    t1, t2 = triple(a), triple(b)
    if not t1 or not t2:
        return None
    c1 = pred_class(t1["attitude"] + " " + t1["relation"]) or pred_class(a)
    c2 = pred_class(t2["attitude"] + " " + t2["relation"]) or pred_class(b)
    o1, o2 = toks(t1["object"]), toks(t2["object"])
    entity = 1.0 if (o1 and o2 and len(o1 & o2) / len(o1 | o2) >= 0.3) else 0.0
    cls = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    refine = 0.0
    if o1 and o2 and (o1 < o2 or o2 < o1):
        small, big = (o1, o2) if o1 < o2 else (o2, o1)
        if len(big) - len(small) >= 2:
            refine = 1.0
    pol = 0.0
    if c1 and c2 and (c1, c2) in OPPOSITE or (c2, c1) in OPPOSITE:
        pol = 1.0
    score = 0.5 * cls + 0.5 * entity + 0.2 * refine - 0.7 * pol
    return max(0.0, min(1.0, score))


def emb_sim(a: str, b: str, model_holder):
    if model_holder[0] is None:
        from sentence_transformers import SentenceTransformer
        model_holder[0] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    v = model_holder[0].encode([a[:800], b[:800]], normalize_embeddings=True,
                               convert_to_numpy=True)
    return float(v[0] @ v[1])


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 pairs per rung, pipeline check")
    args = ap.parse_args()
    if os.path.exists(XCACHE):                     # reuse prior extractions
        for l in io.open(XCACHE, encoding="utf-8"):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
                _extract_cache[r["key"]] = r.get("triple")
            except Exception:
                pass
        print(f"[cache] {len(_extract_cache)} extractions loaded", flush=True)
    mapped = [json.loads(l) for l in io.open(MAPPED, encoding="utf-8") if l.strip()]
    goals = {r["id"]: r["text"] for r in mapped if r["level"] == "goal"}
    plans = {}
    for r in mapped:
        if r["level"] == "plan":
            plans.setdefault(r["parent"], []).append(r["text"])

    def plan_prose(t):
        try:
            obj = json.loads(t.strip())
            titles = [str(x.get("title", "")) for x in obj.get("task_list", []) if isinstance(x, dict)]
            if titles:
                return " | ".join(x for x in titles if x)
        except Exception:
            pass
        return t

    pairs = []   # (rung, a, b, note)

    # R1 identical -- duplicated goal RECORDS (built from the raw rows: the
    # goals dict is keyed by id and collapses exact duplicates)
    seen = {}
    for r in mapped:
        if r["level"] == "goal":
            seen.setdefault(r["text"], []).append(r["id"])
    dups = [v for v in seen.values() if len(v) > 1]
    for v in dups[:20]:
        pairs.append(("R1_identical", goals[v[0]], goals[v[0]], v[0]))

    # R2 paraphrase -- AUTHORED (declared): paraphrases of public plan phases
    authored = [
        ("Run the test suite and inspect the failing tests",
         "Execute the tests and look at which ones fail"),
        ("Read and restate the problem before changing code",
         "Understand the reported issue and rephrase it in my own words"),
        ("Reproduce the bug with a minimal example",
         "Write the smallest snippet that triggers the same error"),
        ("Locate the function that needs to change",
         "Find the code path responsible for the behaviour"),
        ("Fix the incorrect SQL generation for the queryset",
         "Correct the SQL that the query builder produces"),
        ("Add a guard for the empty case",
         "Insert a check that handles the no-items situation"),
        ("Update the documentation for the new parameter",
         "Modify the docs to describe the added argument"),
        ("Remove the deprecated compatibility shim",
         "Delete the old fallback that is no longer needed"),
        ("Verify the fix does not break existing tests",
         "Confirm the change passes the current test suite"),
        ("Clarify the expected behaviour with the reporter",
         "Ask the user what the correct outcome should be"),
        ("Implement the migration for the new column",
         "Add the database migration introducing the column"),
        ("Improve the error message shown on failure",
         "Make the failure output clearer for the user"),
        ("Reduce the runtime of the slow query",
         "Make the expensive query execute faster"),
        ("Refactor the duplicated validation logic",
         "Extract the repeated checks into one place"),
        ("Check whether the issue reproduces on the main branch",
         "See if the problem also happens on the current default branch"),
        ("Plan the phases before editing the repository",
         "Lay out the steps of the work before touching any files"),
        ("Guard against the null value from the parser",
         "Protect the code path when the parser returns nothing"),
        ("Write a regression test for the reported case",
         "Add a test that would have caught this bug"),
        ("Bump the dependency to the fixed release",
         "Upgrade the library to the version containing the patch"),
        ("Explain the root cause in the pull request",
         "Describe why the bug happened in the PR description"),
    ]
    for a, b in authored:
        pairs.append(("R2_paraphrase_AUTHORED", a, b, "authored"))

    # R3 sub-intent -- goal <-> own plan (sampled) ; plus plan refinements
    sub = [(goals[h], plan_prose(ps[0]), h) for h, ps in sorted(plans.items())
           if h in goals and goals[h].strip()]
    for a, b, h in sub[:28]:
        pairs.append(("R3_subintent", a, b, h))
    refs = [(plan_prose(ps[0]), plan_prose(ps[1]), h) for h, ps in sorted(plans.items())
            if len(ps) >= 2]
    for a, b, h in refs[:12]:
        pairs.append(("R3_subintent", a, b, h + "#refine"))

    # R4 sibling -- two different tasks on the same repo
    repo_of = {}
    for h in goals:
        parts = h.replace("ct-", "").split("-")
        repo_of[h] = parts[0] if parts else h
    by_repo = {}
    for h in sorted(goals):
        by_repo.setdefault(repo_of[h], []).append(h)
    n = 0
    for repo, hs in sorted(by_repo.items()):
        for i in range(0, len(hs) - 1, 2):
            if n >= 40:
                break
            pairs.append(("R4_sibling", goals[hs[i]], goals[hs[i + 1]],
                          f"{repo}:{hs[i]}|{hs[i+1]}"))
            n += 1
        if n >= 40:
            break

    # R5 unrelated -- different repos
    repos = [r for r, hs in sorted(by_repo.items()) if hs]
    for i in range(0, min(40, len(repos) - 1), 2):
        a, b = by_repo[repos[i]][0], by_repo[repos[i + 1]][0]
        pairs.append(("R5_unrelated", goals[a], goals[b], f"{repos[i]}|{repos[i+1]}"))

    # R6 reversal -- same file, two different tasks (flip table)
    ft = json.load(io.open(FLIP, encoding="utf-8"))
    table = ft.get("table", {})
    n = 0
    for fpath, tasks in sorted(table.items()):
        tids = sorted(tasks)
        if len(tids) < 2 or n >= 40:
            continue
        ga = next((goals[g] for g in goals if tids[0] in g), None)
        gb = next((goals[g] for g in goals if tids[1] in g), None)
        if ga and gb:
            pairs.append(("R6_reversal_samefile", ga, gb, fpath))
            n += 1
    try:
        rp = json.load(io.open(REVPRECHECK, encoding="utf-8"))
        for row in rp.get("rows", [])[:11]:
            qa, qb = row.get("X"), row.get("Y")
            if isinstance(qa, str) and isinstance(qb, str):
                pairs.append(("R6_reversal_metric", qa, qb, "metric_precheck"))
    except Exception:
        pass

    if args.smoke:
        import collections as _c
        cap, kept = {}, []
        for r in pairs:
            cap[r[0]] = cap.get(r[0], 0) + 1
            if cap[r[0]] <= 2:
                kept.append(r)
        pairs = kept
    print(f"[ladder] {len(pairs)} pairs:", flush=True)
    import collections
    print("  ", dict(collections.Counter(p[0] for p in pairs)), flush=True)

    holder = [None]
    rows = []
    t0 = time.time()
    for i, (rung, a, b, note) in enumerate(pairs):
        r = {"rung": rung, "note": note,
             "lex": round(jacc(a, b), 3),
             "emb": round(emb_sim(a, b, holder), 3),
             "struct": struct_sim(a, b)}
        r["a"], r["b"] = a[:200], b[:200]
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] {rung} struct={r['struct']} "
                  f"lex={r['lex']} ({time.time()-t0:.0f}s)", flush=True)

    rungs = sorted({r["rung"] for r in rows})

    def mean(inst, rung):
        vals = [r[inst] for r in rows if r["rung"] == rung and r[inst] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    table = {inst: {rg: mean(inst, rg) for rg in rungs} for inst in ("lex", "emb", "struct")}
    res = {"n_pairs": len(rows), "table": table, "rungs": rungs}

    def spearman_rungs(inst):
        # per-pair: rung index (expected ascending similarity order)
        order = {rg: i for i, rg in enumerate(["R6_reversal_samefile", "R6_reversal_metric",
                                               "R5_unrelated", "R4_sibling", "R3_subintent",
                                               "R2_paraphrase_AUTHORED", "R1_identical"])}
        xs, ys = [], []
        for r in rows:
            if r[inst] is None or r["rung"] not in order:
                continue
            xs.append(order[r["rung"]]); ys.append(r[inst])
        if len(xs) < 5:
            return None
        import numpy as np
        rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
        n = len(xs)
        return round(float(1 - 6 * ((rx - ry) ** 2).sum() / (n * (n * n - 1))), 3)

    res["spearman_rung_order"] = {inst: spearman_rungs(inst) for inst in ("lex", "emb", "struct")}
    rev = [r["struct"] for r in rows if r["rung"].startswith("R6") and r["struct"] is not None]
    sub = [r["struct"] for r in rows if r["rung"] == "R3_subintent" and r["struct"] is not None]
    rev_lex = [r["lex"] for r in rows if r["rung"].startswith("R6")]
    res["reversal_readout"] = {
        "struct_reversal_mean": round(sum(rev) / len(rev), 3) if rev else None,
        "struct_subintent_mean": round(sum(sub) / len(sub), 3) if sub else None,
        "lex_reversal_mean": round(sum(rev_lex) / len(rev_lex), 3) if rev_lex else None,
    }
    hi = [v for rg in ("R2_paraphrase_AUTHORED", "R3_subintent") for v in [table["struct"][rg]] if v]
    lo = [v for rg in ("R4_sibling", "R5_unrelated", "R6_reversal_samefile", "R6_reversal_metric")
          for v in [table["struct"][rg]] if v]
    sep_ok = bool(hi and lo and min(hi) >= 0.60 and max(lo) <= 0.35)
    gap_ok = bool(res["reversal_readout"]["struct_subintent_mean"] is not None
                  and res["reversal_readout"]["struct_reversal_mean"] is not None
                  and res["reversal_readout"]["struct_subintent_mean"]
                  - res["reversal_readout"]["struct_reversal_mean"] >= 0.25)
    lex_fooled = bool(res["reversal_readout"]["lex_reversal_mean"] is not None
                      and res["reversal_readout"]["struct_reversal_mean"] is not None
                      and res["reversal_readout"]["struct_reversal_mean"]
                      < res["reversal_readout"]["lex_reversal_mean"])
    lex_sep = bool(table["lex"]["R2_paraphrase_AUTHORED"] and table["lex"]["R6_reversal_samefile"]
                   and table["lex"]["R2_paraphrase_AUTHORED"] >= 0.60
                   and table["lex"]["R6_reversal_samefile"] <= 0.35)
    res["checks"] = {"separation": sep_ok, "reversal_gap": gap_ok,
                     "beats_lex_on_reversal": lex_fooled, "vacuous": lex_sep}
    res["pass"] = bool(sep_ok and gap_ok and lex_fooled and not lex_sep)
    res["verdict"] = ("VACUOUS -- lexical already separates the ladder" if lex_sep else
                      "PASS -- structural comparator represents polarity/refinement; "
                      "the payoff question may now be asked" if res["pass"] else
                      "FAIL -- named failures attached (canonicalization / entity / polarity)")
    res["failures"] = [r for r in rows if r["rung"].startswith("R6")
                       and (r["struct"] is None or r["struct"] > 0.35)][:12]
    res["data_gap_paraphrase"] = ("CodeTraceBench cannot supply run-to-run paraphrase pairs: "
                                  "plans exist only for OpenHands, and NO task is run twice by "
                                  "OpenHands (0 shared tasks); other-model tars are 95 percent "
                                  "absent locally (9/110 Claude, 0/97 DeepSeek). R2 is AUTHORED "
                                  "and declared.")
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("table", "spearman_rung_order", "reversal_readout",
                                          "checks", "pass", "verdict")}, indent=1, default=str),
          flush=True)
    print(f"[done] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
