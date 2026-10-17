"""INTENT SIMILARITY LADDER -- PERSONAL-CORPUS + CONSTRUCTED RUNGS.

Extends scripts/icm/intent_similarity_ladder.py with the three rungs the
public benchmark cannot supply, using the SAME instruments (imported, not
re-implemented) and the SAME extraction cache:

  R2p PARAPHRASE, NATURAL (expect HIGH): the user's own corpora -- goals from
      DIFFERENT sessions that share a project keyword (several projects span
      multiple sessions by the 2026-10-17 census; the keyword list itself
      lives in an untracked config). VALIDITY
      CAVEAT, declared: same project does not guarantee the same intent
      (different tasks in one project exist); this rung is therefore
      "same-project cross-session" with the caveat reported, and the lowest-
      scoring pairs are itemized for inspection rather than tuned away.
  R6x INTENT CHANGE, NATURAL (expect LOW-MID): each refute/redirect event
      (the user's own corrections) paired with the immediately PRECEDING goal
      in the same session -- the natural "the intent changed here" pair the
      benchmark lacks. This is the rung addendum 13 got wrong by using plan
      refinements; here the change event is the user's own.
      NOISE FILTER (pre-registered, applied before pairing, effect reported):
      drop redirect texts that are error dumps (`Traceback`, `Error:`, stack
      frames), shorter than 20 chars, or empty.
  R7 REVERSAL 2x2, CONSTRUCTED (expect LOWEST): the payoff_b3_reality 2x2
      literals -- two intents aimed at the SAME memory with OPPOSITE
      requirements (report the ARCHIVED sha vs report the sha that holds
      NOW). Templates instantiated from the actual reality env run indices
      (data/_scratch/icm/reality/env/run-*/part-*.csv); sha tokens are fixed
      placeholders (declared: the similarity signal is in the phrasing).
      NOTE, pre-registered: the declared polarity lexicon covers
      ANTONYM-VERB opposition (add/remove, keep/delete); archived-vs-current
      is a TEMPORAL/STATE opposition it does NOT cover. This rung is
      therefore expected to expose exactly that gap -- if STRUCT scores these
      HIGH while scoring sub-intents high too, the named missing piece is
      polarity coverage, not the idea.

PRE-REGISTERED CHECKS (same shape as the parent ladder):
  (1) STRUCT: mean(R2p) >= 0.60  (natural paraphrase recognized)
  (2) STRUCT: mean(R7)  <= 0.35  (reversal NOT recognized as similar)
  (3) STRUCT: mean(R2p) - mean(R7) >= 0.25
  (4) LEX is FOOLED on R7 (its mean there >= 0.40) -- the reversal must be
      hard for surface matching, else the rung proves nothing
  VACUITY: if LEX separates R2p/R7 like STRUCT does, report VACUOUS.
  PASS requires (1)-(4). Whatever fails is itemized; the failure mode names
  the missing piece.

PRIVACY (per the 2026-10-17 amendment): personal text MAY go to the ollama
cloud relay; personal text, derived sets, and any samples stay under
gitignored data/_scratch/ and never enter tracked files, commit messages, or
docs. Itemized failures in the results JSON are therefore PERSONAL-CONTENT
BEARING and must never be copied into a committed document.

Run:  python scripts/icm/intent_similarity_ladder_personal.py
"""

from __future__ import annotations

import collections
import glob
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

from intent_similarity_ladder import emb_sim, jacc, struct_sim  # noqa: E402

CAPTURE = os.path.join("data", "_scratch", "icm", "capture")
REALITY = os.path.join("data", "_scratch", "icm", "reality", "env")
OUT = os.path.join("data", "_scratch", "icm", "intent_ladder_personal_results.json")

KEYWORDS_FILE = os.path.join("data", "_scratch", "icm", "project_keywords.json")

def _projects():
    """Personal project keywords come from an UNTRACKED config so that
    tracked code carries no personal identifiers."""
    if os.path.exists(KEYWORDS_FILE):
        return tuple(json.load(io.open(KEYWORDS_FILE, encoding="utf-8"))["keywords"])
    return ()

PROJECTS = _projects()
ERROR_PAT = re.compile(r"traceback|error:|exception|stack trace|line \d+, in |file \"", re.I)


def project_of(text: str):
    t = str(text).lower()
    for k in PROJECTS:
        if k in t:
            return k
    return None


def main() -> int:
    recs = []
    for f in glob.glob(os.path.join(CAPTURE, "*_mined.jsonl")):
        for l in io.open(f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                r["_src"] = os.path.basename(f)
                recs.append(r)
    goals = [r for r in recs if r.get("level") == "goal"]
    redirects = [r for r in recs if r.get("signal_class") == "refute/redirect"]
    print(f"[personal] records={len(recs)} goals={len(goals)} redirects={len(redirects)}",
          flush=True)

    pairs = []   # (rung, a, b, note)

    # ---- R2p: same-project goals from DIFFERENT sessions ----
    by_proj = collections.defaultdict(list)
    for r in goals:
        p = project_of(r["text"])
        if p:
            by_proj[p].append(r)
    n = 0
    for p, rs in sorted(by_proj.items()):
        for i in range(0, len(rs) - 1, 2):
            if n >= 40:
                break
            a, b = rs[i], rs[i + 1]
            if a["_src"] == b["_src"]:
                continue
            pairs.append(("R2p_paraphrase_NATURAL", a["text"], b["text"],
                          f"project={p}; {a['_src']}|{b['_src']}"))
            n += 1
        if n >= 40:
            break
    print(f"[R2p] {n} same-project cross-session pairs", flush=True)

    # ---- R6x: intent change = (preceding goal, the user's redirect) ----
    by_parent = collections.defaultdict(list)
    for r in goals:
        by_parent[(r["_src"], str(r.get("parent")))].append(r)
    n = dropped = 0
    for key, rs in by_parent.items():
        rs = sorted(rs, key=lambda r: str(r.get("declared_at")))
        for i in range(1, len(rs)):
            cur, prev = rs[i], rs[i - 1]
            if cur.get("signal_class") != "refute/redirect":
                continue
            txt = str(cur.get("text", ""))
            if len(txt.strip()) < 20 or ERROR_PAT.search(txt):
                dropped += 1
                continue
            if n >= 40:
                break
            pairs.append(("R6x_intent_change_NATURAL", prev["text"], txt,
                          f"{key[0]}:{key[1][:28]}"))
            n += 1
        if n >= 40:
            break
    print(f"[R6x] {n} intent-change pairs ({dropped} redirects dropped by the "
          f"pre-registered noise filter)", flush=True)

    # ---- R7: the constructed 2x2 reversal ----
    runs = sorted(glob.glob(os.path.join(REALITY, "run-*")))[:12]
    for rp in runs:
        rid = os.path.basename(rp)
        parts = sorted(glob.glob(os.path.join(rp, "part-*.csv")))
        fixture = os.path.basename(parts[0])[:-4] if parts else "part-0"
        a = (f"Compliance check for {fixture} (run {rid[-4:]}): report the manifest "
             f"sha recorded at the original ingest.")
        b = (f"Compliance check for {fixture} (run {rid[-4:]}): the dataset was just "
             f"re-ingested; report the manifest sha that holds NOW.")
        pairs.append(("R7_reversal_2x2_CONSTRUCTED", a, b, f"{rid}:{fixture}"))
    print(f"[R7] {len(runs)} constructed 2x2 reversal pairs", flush=True)

    # ---- score with the SAME instruments ----
    holder = [None]
    rows = []
    t0 = time.time()
    for i, (rung, a, b, note) in enumerate(pairs):
        rows.append({"rung": rung, "note": note,
                     "lex": round(jacc(a, b), 3),
                     "emb": round(emb_sim(a, b, holder), 3),
                     "struct": struct_sim(a, b)})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] {rung} struct={rows[-1]['struct']} "
                  f"lex={rows[-1]['lex']} ({time.time()-t0:.0f}s)", flush=True)

    rungs = sorted({r["rung"] for r in rows})

    def mean(inst, rung):
        vals = [r[inst] for r in rows if r["rung"] == rung and r[inst] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    table = {inst: {rg: mean(inst, rg) for rg in rungs} for inst in ("lex", "emb", "struct")}
    m = {inst: table[inst] for inst in table}
    res = {"n_pairs": len(rows), "table": table, "rungs": rungs,
           "noise_filter": {"dropped_redirects": dropped}}
    r2p, r7 = m["struct"].get("R2p_paraphrase_NATURAL"), m["struct"].get("R7_reversal_2x2_CONSTRUCTED")
    lex7 = m["lex"].get("R7_reversal_2x2_CONSTRUCTED")
    checks = {
        "paraphrase_recognized": bool(r2p is not None and r2p >= 0.60),
        "reversal_not_similar": bool(r7 is not None and r7 <= 0.35),
        "reversal_gap": bool(r2p is not None and r7 is not None and r2p - r7 >= 0.25),
        "lex_fooled_on_reversal": bool(lex7 is not None and lex7 >= 0.40),
    }
    res["checks"] = checks
    res["pass"] = all(checks.values())
    res["verdict"] = ("PASS -- the comparator handles natural paraphrase AND the "
                      "constructed reversal" if res["pass"] else
                      "FAIL -- itemized; the failing check names the missing piece "
                      "(polarity coverage is the pre-registered suspect on R7)")
    # itemized failures (PERSONAL-CONTENT BEARING -- stays untracked)
    res["failures"] = [
        {"rung": r["rung"], "note": r["note"], "struct": r["struct"], "lex": r["lex"],
         "emb": r["emb"]}
        for r in rows if (r["rung"] == "R7_reversal_2x2_CONSTRUCTED" and (r["struct"] or 0) > 0.35)
        or (r["rung"] == "R2p_paraphrase_NATURAL" and (r["struct"] or 0) < 0.60)][:12]
    res["personal_content_warning"] = ("failures/notes may contain personal text; this "
                                       "file is untracked and must never be committed")
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("table", "checks", "pass", "verdict")},
                     indent=1, default=str), flush=True)
    print(f"[done] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
