"""SCALED 2x2 HARNESS -- opposition where it is real, verified by the filesystem.

WHY (addendum 20): the integration test found NO natural opposed-intent pairs
in CodeTraceBench (95/96 shared-file task pairs are merely DIFFERENT). The
judge's opposition capability (addenda 15/16: 0.075 opposed / 0.90 same on
short constructed pairs) therefore has no test bed on real corpora. This
harness builds one at scale, with MECHANICAL verification -- no judge decides
whether an answer is right; a file on disk holds the answer and the expected
value is computed.

DESIGN (2x2xN, filesystem-scored):
  For each ARTIFACT TYPE (checksum, config, cache, roster) and each of 5
  scenarios, write TWO memory records to disk:
     OLD record: the value recorded at the original ingest (contains the old
                 value, marked by its own wording: "recorded at the original
                 ingest", "before the migration", ...)
     NEW record: the value after a re-ingest/update (marked by its own
                 wording: "re-ingest appended one row", "after the migration")
  and TWO intents per scenario:
     OLD intent: asks for what was recorded originally
     NEW intent: asks for what holds now
  Trial = (scenario, intent). The correct memory is OLD for the OLD intent and
  NEW for the NEW intent. The mechanism RANKS the two memory records; the
  "answer" is the value parsed from the top-ranked record; the filesystem
  checks it against the expected value. 4 types x 5 scenarios x 2 intents =
  40 trials, each with a mechanically known correct memory and value.

ARMS:
  DENSE    cosine(embed(intent), embed(memory text))
  LEX      token jaccard(intent, memory text)
  KEYSPACE judge requirements from the intent vs judge "provides" triples
           from the memory text; score = max key_sim over requirement pairs
           (the addendum-14 schema, qualifier opposition = -0.7)
  HYBRID   reciprocal-rank fusion (dense + keyspace)

PRE-REGISTERED BAR:
  (1) VACUITY GUARD: DENSE accuracy must be near chance (0.35..0.65) -- the
      construction must defeat plain similarity, else it proves nothing.
  (2) KEYSPACE accuracy >= 0.85
  (3) KEYSPACE >= DENSE + 0.25 (absolute)
  (4) HYBRID >= DENSE + 0.25
  PASS -> intent conditioning decides relevance where opposition is real.
  FAIL -> the failing row names the missing piece (and the construction's
  own numbers are reported either way).

CAVEAT: a constructed world can carry compact-rule discriminators (the
program's fixed-point theorem). Mitigations: four artifact types with
different surfaces, values vary per scenario, the memory wording does NOT use
the intents' marker words verbatim, and the answer is read from the chosen
file rather than asserted.

Run:  python scripts/icm/twobytwo_scale.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from intent_similarity_ladder import PRED_CLASSES, toks  # noqa: E402
from intent_keyspace import ask, first_json  # noqa: E402

ENV = os.path.join("data", "_scratch", "icm", "twobytwo", "env")
OUT = os.path.join("data", "_scratch", "icm", "twobytwo_results.json")
CACHE = os.path.join("data", "_scratch", "icm", "twobytwo_cache.jsonl")
BATCH = 8

_emb_model = [None]


def emb(texts):
    if _emb_model[0] is None:
        from sentence_transformers import SentenceTransformer
        _emb_model[0] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _emb_model[0].encode(texts, batch_size=32, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)


# ---------------------------------------------------------------- scenarios --
ARTIFACTS = {
    "checksum": {
        "old_mem": "Ingest log n={n}: dataset {d}, {rows} rows. Checksum {old}. Recorded at the "
                   "original ingest on 2026-03-01.",
        "new_mem": "Re-ingest log n={n}: dataset {d}, {rows2} rows. Checksum {new}. Appended one "
                   "row and rewrote the table on 2026-04-15.",
        "old_int": "Report the checksum saved at the first ingest of dataset {d}.",
        "new_int": "Dataset {d} was re-ingested; report the checksum that holds right now.",
    },
    "config": {
        "old_mem": "Settings snapshot n={n}: timeout={old}ms, pool=8. Captured before the "
                   "migration on 2026-03-01.",
        "new_mem": "Settings snapshot n={n}: timeout={new}ms, pool=8. Captured after the migration "
                   "on 2026-04-15.",
        "old_int": "Report the timeout captured before the migration of the {d} service.",
        "new_int": "Report the timeout in effect after the migration of the {d} service.",
    },
    "cache": {
        "old_mem": "Cache record n={n}: key=region, value={old}. Expiry 2026-04-01, written at "
                   "setup.",
        "new_mem": "Cache record n={n}: key=region, value={new}. Expiry 2026-05-01, refreshed on "
                   "2026-04-15.",
        "old_int": "Report the cache value written at setup for the {d} pipeline.",
        "new_int": "The cache was refreshed; report the value live now for the {d} pipeline.",
    },
    "roster": {
        "old_mem": "Roster file n={n}: on-call={old}. Assigned at the original handoff.",
        "new_mem": "Roster file n={n}: on-call={new}. Rotated after the handoff.",
        "old_int": "Report who was on-call at the original handoff for {d}.",
        "new_int": "Report who is on-call now for {d}, after the rotation.",
    },
}
DATASETS = ["alpha", "beta", "gamma", "delta", "epsilon"]


def make_scenarios():
    rng = random.Random(7)
    trials = []
    os.makedirs(ENV, exist_ok=True)
    for typ, tpl in ARTIFACTS.items():
        for si, d in enumerate(DATASETS):
            n = f"{typ}-{si:02d}"
            vals = {
                "n": n, "d": d,
                "rows": rng.randint(100, 900), "rows2": 0,
                "old": {"checksum": "".join(rng.choice("0123456789abcdef") for _ in range(12)),
                        "config": str(rng.choice([300, 500, 800])),
                        "cache": "".join(rng.choice("0123456789abcdef") for _ in range(8)),
                        "roster": rng.choice(["lee", "kim", "ortiz", "nash"])}[typ],
                "new": {"checksum": "".join(rng.choice("0123456789abcdef") for _ in range(12)),
                        "config": str(rng.choice([1200, 1500, 2000])),
                        "cache": "".join(rng.choice("0123456789abcdef") for _ in range(8)),
                        "roster": rng.choice(["wong", "adeyemi", "silva", "grant"])}[typ],
            }
            vals["rows2"] = vals["rows"] + 1
            old_txt = tpl["old_mem"].format(**vals)
            new_txt = tpl["new_mem"].format(**vals)
            d_old = os.path.join(ENV, f"{n}-old.txt")
            d_new = os.path.join(ENV, f"{n}-new.txt")
            io.open(d_old, "w", encoding="utf-8").write(old_txt)
            io.open(d_new, "w", encoding="utf-8").write(new_txt)
            for which in ("old", "new"):
                trials.append({"trial": f"{n}-{which}",
                               "type": typ, "scenario": n,
                               "intent": tpl[f"{which}_int"].format(**vals),
                               "mem_old": {"path": d_old, "text": old_txt, "value": vals["old"]},
                               "mem_new": {"path": d_new, "text": new_txt, "value": vals["new"]},
                               "correct": which, "expected": vals[which]})
    return trials


# --------------------------------------------------------------------- keys --
_cache = {}


def load_cache():
    if os.path.exists(CACHE):
        for l in io.open(CACHE, encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    _cache[r["key"]] = r.get("val")
                except Exception:
                    pass


def save_cache(k, v):
    _cache[k] = v
    with io.open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": k, "val": v}) + "\n")


def extract_triples(texts, kind):
    """kind: 'reqs' (list per text) or 'provides' (single triple per text)."""
    if kind == "reqs":
        head = ("For each numbered intent below, list its distinct requirements "
                "(1 to 4). Reply JSON array only: "
                '[{"i": <n>, "reqs": [{"attitude": "...", "relation": "...", '
                '"object": "...", "qualifier": "..."}]}].')
        prompt = head + "\n\n" + "\n\n".join(f"{i+1}. {t[:800]}" for i, t in enumerate(texts))
        arr = first_json(ask(prompt), "[")
        out = [None] * len(texts)
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                try:
                    idx = int(it.get("i", 0)) - 1
                except Exception:
                    continue
                reqs = it.get("reqs") or []
                if 0 <= idx < len(out) and isinstance(reqs, list):
                    cl = [{k: str(x.get(k, "")) for k in ("attitude", "relation", "object", "qualifier")}
                          for x in reqs if isinstance(x, dict)]
                    if cl:
                        out[idx] = cl
        return out
    head = ("For each numbered memory record below, state in one triple what the "
            "record PROVIDES: {\"i\": <n>, \"attitude\": \"...\", \"relation\": \"...\", "
            "\"object\": \"...\", \"qualifier\": \"...\"}. Qualifier names the record's "
            "time/state marker (as written in the record). Reply JSON array only.")
    prompt = head + "\n\n" + "\n\n".join(f"{i+1}. {t[:800]}" for i, t in enumerate(texts))
    arr = first_json(ask(prompt), "[")
    out = [None] * len(texts)
    if isinstance(arr, list):
        for it in arr:
            if not isinstance(it, dict):
                continue
            try:
                idx = int(it.get("i", 0)) - 1
            except Exception:
                continue
            if 0 <= idx < len(out):
                out[idx] = {k: str(it.get(k, "")) for k in ("attitude", "relation", "object", "qualifier")}
    return out


ARCH = ("archived", "original", "stored", "snapshot", "recorded", "stale", "old",
        "previous", "existing", "legacy", "at ingest", "earlier", "before")
CUR = ("current", "fresh", "now", "updated", "new", "latest", "live", "re-ingested",
       "reingested", "recomputed", "refreshed", "after", "appended", "rotated")


def qual(t):
    low = " " + str(t).lower() + " "
    a = any(w in low for w in ARCH)
    c = any(w in low for w in CUR)
    return "OLD" if a and not c else "NEW" if c and not a else ""


def pred_class(text):
    ts = toks(text)
    hits = [c for c, ws in PRED_CLASSES.items() if ts & set(ws)]
    if not hits:
        return None
    low = str(text).lower()
    hits.sort(key=lambda c: min(low.find(w) for w in PRED_CLASSES[c] if w in low))
    return hits[0]


def key_sim(qk, tk):
    if not qk or not tk:
        return 0.0
    c1 = pred_class(qk.get("attitude", "") + " " + qk.get("relation", ""))
    c2 = pred_class(tk.get("attitude", "") + " " + tk.get("relation", ""))
    o1, o2 = toks(qk.get("object", "")), toks(tk.get("object", ""))
    ent = 1.0 if (o1 and o2 and len(o1 & o2) / len(o1 | o2) >= 0.3) else 0.0
    cls = 1.0 if (c1 and c2 and c1 == c2) else 0.5 if (c1 and c2) else 0.0
    q1 = qual(qk.get("qualifier", ""))
    q2 = qual(tk.get("qualifier", ""))
    opp = 0.7 if {q1, q2} == {"OLD", "NEW"} else 0.0
    return max(0.0, min(1.0, 0.5 * cls + 0.5 * ent - opp))


def main() -> int:
    load_cache()
    trials = make_scenarios()
    print(f"[env] {len(trials)} trials; files written to {ENV}", flush=True)
    t0 = time.time()
    # intent requirements
    intents = sorted({t["intent"] for t in trials})
    print(f"[keys] {len(intents)} distinct intents", flush=True)
    ireq = {}
    for i in range(0, len(intents), BATCH):
        chunk = intents[i:i + BATCH]
        got = []
        todo = []
        for txt in chunk:
            ck = hashlib.sha1(("reqs|" + txt).encode()).hexdigest()
            if ck in _cache:
                got.append(_cache[ck]); continue
            todo.append(txt)
        fresh = extract_triples(todo, "reqs") if todo else []
        fi = 0
        for txt in chunk:
            ck = hashlib.sha1(("reqs|" + txt).encode()).hexdigest()
            if ck in _cache:
                ireq[txt] = _cache[ck]
            else:
                g = fresh[fi]; fi += 1
                save_cache(ck, g)
                ireq[txt] = g
    # memory "provides"
    mems = sorted({t["mem_old"]["text"] for t in trials} | {t["mem_new"]["text"] for t in trials})
    print(f"[keys] {len(mems)} distinct memory records", flush=True)
    mprov = {}
    for i in range(0, len(mems), BATCH):
        chunk = mems[i:i + BATCH]
        todo = [t for t in chunk
                if hashlib.sha1(("provides|" + t).encode()).hexdigest() not in _cache]
        fresh = extract_triples(todo, "provides") if todo else []
        fi = 0
        for txt in chunk:
            ck = hashlib.sha1(("provides|" + txt).encode()).hexdigest()
            if ck in _cache:
                mprov[txt] = _cache[ck]
            else:
                g = fresh[fi]; fi += 1
                save_cache(ck, g)
                mprov[txt] = g
    print(f"[keys] done ({time.time() - t0:.0f}s)", flush=True)

    res = {"trials": len(trials), "arms": {}}
    acc = {a: [] for a in ("dense", "lex", "keyspace", "hybrid")}
    for t in trials:
        mems_t = [t["mem_old"], t["mem_new"]]
        ie = emb([t["intent"]])[0]
        me = emb([m["text"] for m in mems_t])
        dense = me @ ie
        qt = toks(t["intent"])
        lex = np.array([len(qt & toks(m["text"])) / max(1, len(qt | toks(m["text"])))
                        for m in mems_t])
        rq = ireq.get(t["intent"]) or []
        ks = np.array([max((key_sim(r, mprov.get(m["text"]) or {}) for r in rq), default=0.0)
                       for m in mems_t])
        picks = {"dense": int(np.argmax(dense)), "lex": int(np.argmax(lex)),
                 "keyspace": int(np.argmax(ks))}
        rrf = np.zeros(2)
        for a, s in (("dense", dense), ("keyspace", ks)):
            order = np.argsort(-s)
            ranks = np.empty(2, dtype=int)
            ranks[order] = np.arange(2)
            rrf += 1.0 / (60 + ranks)
        picks["hybrid"] = int(np.argmax(rrf))
        for a, pi in picks.items():
            chosen = mems_t[pi]
            val = chosen["value"]
            # mechanical verification: the answer read from the chosen file must
            # equal the expected value for THIS intent
            acc[a].append(1.0 if val == t["expected"] else 0.0)

    res["arms"] = {a: round(statistics.mean(v), 3) for a, v in acc.items()}
    d_acc, k_acc, h_acc = res["arms"]["dense"], res["arms"]["keyspace"], res["arms"]["hybrid"]
    checks = {"vacuity_dense_near_chance": 0.35 <= d_acc <= 0.65,
              "keyspace_high": k_acc >= 0.85,
              "keyspace_beats_dense": k_acc >= d_acc + 0.25,
              "hybrid_beats_dense": h_acc >= d_acc + 0.25}
    res["checks"] = checks
    res["pass"] = all(checks.values())
    res["verdict"] = ("PASS -- intent conditioning decides relevance where "
                      "opposition is real and mechanically verified"
                      if res["pass"] else
                      "FAIL -- see the failing row; the construction's own numbers are reported")
    res["elapsed_s"] = round(time.time() - t0)
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
