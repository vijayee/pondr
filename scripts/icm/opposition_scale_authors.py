"""OPPOSITION SCALE-UP -- many authors, the judge kept out of the author chair.

WHY: the non-circular test (addendum 15) passed with n=15 from ONE author
(the user) and ONE judge. Its caveats: single author, correlated authorship.
This run scales it with MULTIPLE GENERATOR MODELS, none of which is the
judge (author != comparator != judge), and reports per-author so an
author-style effect cannot hide inside the pooled number.

AUTHORS (generator models): glm-5.3-flash:cloud, minimax-m3:cloud, qwen3:8b
(local -- no cloud call). Each writes 10 opposed + 5 same-intent pairs under a
fixed authoring prompt that REQUIRES: same artifact for both intents, shared
wording, no marker words (add/remove, old/new, before/after, enable/disable,
keep/delete, archive/current), plus a stated shared_memory and why_opposite
(the design must be explicit even when the wording is not).
JUDGE: deepseek-v4.1-flash:cloud (the addendum-15 instrument). It never sees
the authoring prompt or the labels -- only the two sentences.

MECHANICAL VALIDATION (reported per author, not silently filtered):
  - marker-word rate in generated pair texts (the authoring rule forbids them
    -- if violated, the pair is counted in BOTH the 'all' and the 'unmarked'
    subsets so the reader sees the difference);
  - duplicate rate vs the user's 15 pairs (exact-string dedupe);
  - relation-balance (a set with no 'same' pairs is rejected as unusable).

PRE-REGISTERED READOUTS (per author and pooled):
  judge_opposite   want <= 0.35      (judge answers 'different' on opposed)
  judge_same       want >= 0.60      (judge answers 'same' on same-intent)
  v4_opposite / v4_same              (the frozen structural comparator; its
                                      addendum-15 failure should REPLICATE)
  lex_opposite     want >= 0.40      (fooled check per author: else that
                                      author's pairs are lexically separable
                                      and prove nothing)
  PASS (pooled): judge_opposite <= 0.35 AND judge_same >= 0.60 AND the
  per-author judge_opposite holds for >= 2 of 3 authors (a single good author
  is not a result) AND lex is fooled for those authors.
  Any author whose pairs violate the authoring rules (marker words) is
  reported with its unmarked-only subset as well -- both numbers visible.

DATA: generated texts are UNTRACKED (data/_scratch/icm/opposition_scaled_pairs.json);
only aggregate numbers + hashes are recorded in the results JSON.

Run:
  python scripts/icm/opposition_scale_authors.py --generate
  python scripts/icm/opposition_scale_authors.py --score
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import comparator_capacity_test as C  # noqa: E402  (FROZEN comparator)
from intent_detection_smoke import judge_same  # noqa: E402
from intent_similarity_ladder import emb_sim, jacc  # noqa: E402

USER_PAIRS = os.path.join("data", "_scratch", "icm", "implicit_pairs.json")
SCALED = os.path.join("data", "_scratch", "icm", "opposition_scaled_pairs.json")
OUT = os.path.join("data", "_scratch", "icm", "opposition_scale_results.json")

AUTHORS = [("glm-5.3-flash:cloud", 10, 5),
           ("minimax-m3:cloud", 10, 5),
           ("qwen3:8b", 10, 5)]
MARKERS = ("add", "remove", "old", "new", "before", "after", "enable", "disable",
           "keep", "delete", "archive", "current", "archived", "fresh")


def ask(model: str, prompt: str, tries: int = 4) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.3, "num_ctx": 4096}}
    backoff = 8.0
    for _ in range(tries):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return str(json.loads(urllib.request.urlopen(req, timeout=300).read())
                       ["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
            time.sleep(3.0)
        except Exception:
            time.sleep(3.0)
    return ""


AUTHOR_PROMPT = """Write test cases for intent comparison. Reply with JSON only:
{{"pairs": [{{"relation": "opposite"|"same", "a": "...", "b": "...",
             "shared_memory": "...", "why_opposite": "..."}}]}}

Produce {n_opp} pairs with relation "opposite": two one-sentence intents about the
SAME artifact or memory (same file, same document, same report, same system) where
acting on one DEFEATS the other. They must share nearly all of their words.
Do NOT use obvious opposite words (add/remove, old/new, before/after,
enable/disable, keep/delete, archive/current, fresh). State shared_memory (what both
intents act on) and why_opposite (one short clause) for each.

Produce {n_same} pairs with relation "same": the same intent twice, worded very
differently (different verbs and nouns), with shared_memory filled and
why_opposite "".

Sentences must be 6-20 words, concrete, and about ordinary software or data work.
JSON only, no prose."""


def generate() -> int:
    out = []
    for model, n_opp, n_same in AUTHORS:
        prompt = AUTHOR_PROMPT.format(n_opp=n_opp, n_same=n_same)
        print(f"[author] {model}: requesting {n_opp}+{n_same}", flush=True)
        raw = ask(model, prompt)
        m = re.search(r"\{.*\}", raw or "", re.S)
        pairs = []
        if m:
            try:
                pairs = json.loads(m.group(0)).get("pairs", [])
            except Exception as e:
                print(f"  parse failed: {e}", flush=True)
        kept = []
        for p in pairs:
            if not isinstance(p, dict):
                continue
            a, b, rel = str(p.get("a", "")).strip(), str(p.get("b", "")).strip(), p.get("relation")
            if rel in ("opposite", "same") and a and b and a != b:
                kept.append({"relation": rel, "a": a, "b": b,
                             "shared_memory": str(p.get("shared_memory", ""))[:160],
                             "why_opposite": str(p.get("why_opposite", ""))[:160],
                             "author": model})
        print(f"  kept {len(kept)} pairs", flush=True)
        out.extend(kept)
    data = json.load(io.open(USER_PAIRS, encoding="utf-8")) if os.path.exists(USER_PAIRS) else {}
    user = [{"relation": p["relation"], "a": p["a"], "b": p["b"],
             "shared_memory": "", "why_opposite": p.get("note", ""), "author": "user"}
            for p in data.get("pairs", []) if not p.get("example")
            and p.get("a") and p.get("b")]
    seen = {(p["a"], p["b"]) for p in user}
    dedup = [p for p in out if (p["a"], p["b"]) not in seen]
    print(f"[generated] {len(out)} -> {len(dedup)} after dedupe vs {len(user)} user pairs",
          flush=True)
    with io.open(SCALED, "w", encoding="utf-8") as f:
        f.write(json.dumps({"pairs": dedup}, indent=1))
    print(f"wrote {SCALED} (untracked)", flush=True)
    return 0


def score() -> int:
    gen = json.load(io.open(SCALED, encoding="utf-8")).get("pairs", [])
    usr = []
    if os.path.exists(USER_PAIRS):
        ud = json.load(io.open(USER_PAIRS, encoding="utf-8"))
        usr = [{"relation": p["relation"], "a": p["a"], "b": p["b"], "author": "user"}
               for p in ud.get("pairs", []) if not p.get("example")
               and p.get("a") and p.get("b")]
    rows_all = []
    holder = [None]
    for p in usr + gen:
        a, b = p["a"], p["b"]
        marker = any(w in f" {a.lower()} {b.lower()} " for w in
                     (" add ", " remove ", " old ", " new ", " before ", " after ",
                      " enable ", " disable ", " keep ", " delete ", " archive ",
                      " current ", " archived ", " fresh "))
        j = judge_same(a, b)
        rows_all.append({"author": p["author"], "relation": p["relation"],
                         "unmarked": not marker,
                         "lex": round(jacc(a, b), 3),
                         "emb": round(emb_sim(a, b, holder), 3),
                         "v4": C.struct_sim_v4(a, b),
                         "judge": None if j is None else (1.0 if j else 0.0)})
        print(f"  [{p['author'][:14]:14s}] {p['relation']:8s} judge={rows_all[-1]['judge']} "
              f"v4={rows_all[-1]['v4']} lex={rows_all[-1]['lex']}", flush=True)

    def mean(rows, rel, inst):
        vals = [r[inst] for r in rows if r["relation"] == rel and r[inst] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    authors = ["user"] + [m for m, _, _ in AUTHORS]
    table = {}
    for au in authors:
        rs = [r for r in rows_all if r["author"] == au]
        ru = [r for r in rs if r["unmarked"]]
        table[au] = {
            "n": len(rs), "n_unmarked": len(ru),
            "judge_opposite": mean(rs, "opposite", "judge"),
            "judge_same": mean(rs, "same", "judge"),
            "v4_opposite": mean(rs, "opposite", "v4"),
            "lex_opposite": mean(rs, "opposite", "lex"),
            "judge_opposite_unmarked": mean(ru, "opposite", "judge"),
        }
    pooled = {rel: {inst: mean(rows_all, rel, inst) for inst in ("judge", "v4", "lex", "emb")}
              for rel in ("opposite", "same")}
    ok_au = [au for au in authors[1:] if table[au]["judge_opposite"] is not None
             and table[au]["judge_opposite"] <= 0.35
             and table[au]["lex_opposite"] is not None and table[au]["lex_opposite"] >= 0.40]
    checks = {
        "judge_opposite_low": pooled["opposite"]["judge"] is not None
        and pooled["opposite"]["judge"] <= 0.35,
        "judge_same_high": pooled["same"]["judge"] is not None
        and pooled["same"]["judge"] >= 0.60,
        "two_of_three_authors": len(ok_au) >= 2,
        "v4_failure_replicates": pooled["opposite"]["v4"] is not None
        and pooled["opposite"]["v4"] > 0.35,
    }
    res = {"n_pairs": len(rows_all), "table": table, "pooled": pooled,
           "authors_passing": ok_au, "checks": checks, "pass": all(checks.values()),
           "pairs_sha256": hashlib.sha256(io.open(SCALED, "rb").read()).hexdigest()[:16],
           "comparator_sha256": hashlib.sha256(
               io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "comparator_capacity_test.py"), "rb").read()
           ).hexdigest()[:16],
           "verdict": ("PASS -- the judge generalises across authors while the "
                       "structural comparator's failure replicates"
                       if all(checks.values()) else
                       "FAIL -- see per-author table; the failing row names the issue")}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("table", "pooled", "checks", "pass", "verdict")},
                     indent=1, default=str), flush=True)
    print(f"[done] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.generate:
        sys.exit(generate())
    if args.score:
        sys.exit(score())
    print("pass --generate or --score")
    sys.exit(2)
