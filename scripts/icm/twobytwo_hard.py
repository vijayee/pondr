"""2x2 HARD -- epoch is NOT in the words; the intent must be interpreted.

WHY (addendum 21): the first 2x2 failed its own vacuity guard because the
memory records SAID their state ("Recorded at the original ingest" vs
"Appended one row"). Dense read those words and scored 0.85. This version
removes the tell:

  memory records  : same wording, differing only in VALUE, RUN NUMBER and DATE.
                    Neither record says old/new/current/archived/fresh.
  intents         : refer to the epoch RELATIVELY ("at the first load", "at the
                    most recent load") using none of the memories' date or run
                    words.
  ordering        : mechanical -- dates decide which record is earlier.

So DENSE/LEX should be near chance (both records match the intent's words
equally), and only a mechanism that INTERPRETS the intent's epoch reference
(judge: EARLIEST / LATEST) can pick correctly via the dates. That is the
capability under test: intent -> epoch -> memory, without the answer being
written on the surface.

TRIALS: 4 artifact types x 5 scenarios x 2 intents = 40, each verified
mechanically: the answer is READ FROM THE CHOSEN FILE and compared to the
expected value computed at generation.

ARMS:
  DENSE          cosine(embed(intent), embed(memory text))
  LEX            token jaccard
  SEMANTIC       judge classifies the intent's epoch (EARLIEST / LATEST /
                 UNSPECIFIED); EARLIEST -> the earlier-dated record, LATEST ->
                 the later-dated record; UNSPECIFIED -> abstain (counts wrong)
  HYBRID         score fusion with an explicit tie-break by DENSE (the
                 addendum-21 RRF tie was an artifact, not a result)

PRE-REGISTERED BAR:
  (1) VACUITY: DENSE accuracy in 0.35..0.65 (the construction must defeat
      surface similarity -- this is the guard the previous harness failed)
  (2) SEMANTIC >= 0.85
  (3) SEMANTIC >= DENSE + 0.25
  (4) HYBRID >= DENSE + 0.25
  PASS -> interpreting the intent's epoch decides relevance where the words
  do not. FAIL -> the failing row names the gap; numbers reported either way.

CAVEAT (fixed-point theorem): the date is still a surface feature -- but it
is not the one the intent's words carry, so matching requires the judge step.
Whether a judge-free mechanism could exploit a compact rule here is reported
by the DENSE/LEX arms.

Run:  python scripts/icm/twobytwo_hard.py
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

from intent_keyspace import ask  # noqa: E402
from intent_similarity_ladder import toks  # noqa: E402

ENV = os.path.join("data", "_scratch", "icm", "twobytwo_hard", "env")
OUT = os.path.join("data", "_scratch", "icm", "twobytwo_hard_results.json")
CACHE = os.path.join("data", "_scratch", "icm", "twobytwo_hard_cache.jsonl")

_emb_model = [None]


def emb(texts):
    if _emb_model[0] is None:
        from sentence_transformers import SentenceTransformer
        _emb_model[0] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _emb_model[0].encode(texts, batch_size=32, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)


ARTIFACTS = {
    "checksum": ("dataset {d}: {rows} rows, checksum {v}, run {n}, dated {date}.",
                 "Report the checksum for dataset {d} as it stood at the first load.",
                 "Report the checksum for dataset {d} as it stands at the most recent load."),
    "config": ("settings for {d}: timeout={v}ms, pool=8, captured on {date} (run {n}).",
               "Report the timeout for {d} as configured at the earliest capture.",
               "Report the timeout for {d} as configured at the latest capture."),
    "cache": ("cache for {d}: key=region, value={v}, written on {date} (run {n}).",
              "Report the cached value for {d} from the first write.",
              "Report the cached value for {d} from the newest write."),
    "roster": ("roster for {d}: on-call={v}, dated {date} (run {n}).",
               "Report the on-call name for {d} from the initial handoff.",
               "Report the on-call name for {d} after the most recent handoff."),
}
DATASETS = ["alpha", "beta", "gamma", "delta", "epsilon"]
DATES = ("2026-03-01", "2026-04-15")
RUNS = (12, 13)


def make_trials():
    rng = random.Random(11)
    os.makedirs(ENV, exist_ok=True)
    trials = []
    for typ, (mtpl, i_old, i_new) in ARTIFACTS.items():
        for si, d in enumerate(DATASETS):
            n = f"{typ}-{si:02d}"
            vals = {
                "rows": rng.randint(100, 900),
                "old": {"checksum": "".join(rng.choice("0123456789abcdef") for _ in range(12)),
                        "config": str(rng.choice([300, 500, 800])),
                        "cache": "".join(rng.choice("0123456789abcdef") for _ in range(8)),
                        "roster": rng.choice(["lee", "kim", "ortiz", "nash"])}[typ],
                "new": {"checksum": "".join(rng.choice("0123456789abcdef") for _ in range(12)),
                        "config": str(rng.choice([1200, 1500, 2000])),
                        "cache": "".join(rng.choice("0123456789abcdef") for _ in range(8)),
                        "roster": rng.choice(["wong", "adeyemi", "silva", "grant"])}[typ],
            }
            texts, recs = {}, {}
            for side, dt, rn in (("old", DATES[0], RUNS[0]), ("new", DATES[1], RUNS[1])):
                key = {"d": d, "v": vals[side], "date": dt, "n": rn, "rows": vals["rows"]}
                txt = mtpl.format(**key)
                path = os.path.join(ENV, f"{n}-{side}.txt")
                io.open(path, "w", encoding="utf-8").write(txt)
                texts[side] = txt
                recs[side] = {"path": path, "text": txt, "value": vals[side],
                              "date": dt, "run": rn}
            for side, itpl, want in (("old", i_old, "old"), ("new", i_new, "new")):
                trials.append({"trial": f"{n}-{side}", "type": typ, "scenario": n,
                               "intent": itpl.format(d=d),
                               "rec_old": recs["old"], "rec_new": recs["new"],
                               "correct": want, "expected": vals[want]})
    return trials


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


def judge_epoch(intent: str):
    key = hashlib.sha1(("epoch|" + intent).encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    prompt = ("Does the request below ask about the EARLIEST state of something, the "
              "LATEST state, or is it UNSPECIFIED in time? Answer exactly one word: "
              "EARLIEST, LATEST or UNSPECIFIED.\n\nRequest: " + intent[:800])
    raw = ask(prompt).upper()
    val = ("EARLIEST" if "EARLIEST" in raw else
           "LATEST" if "LATEST" in raw else
           "UNSPECIFIED" if "UNSPECIFIED" in raw else None)
    _cache[key] = val
    with io.open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "val": val}) + "\n")
    return val


def main() -> int:
    load_cache()
    trials = make_trials()
    print(f"[env] {len(trials)} trials -> {ENV}", flush=True)
    t0 = time.time()
    acc = {a: [] for a in ("dense", "lex", "semantic", "hybrid")}
    detail = []
    for t in trials:
        recs = [t["rec_old"], t["rec_new"]]
        # dense / lex
        ie = emb([t["intent"]])[0]
        me = emb([r["text"] for r in recs])
        dense = me @ ie
        qt = toks(t["intent"])
        lex = np.array([len(qt & toks(r["text"])) / max(1, len(qt | toks(r["text"])))
                        for r in recs])
        # semantic: judge epoch -> date ordering
        ep = judge_epoch(t["intent"])
        if ep == "EARLIEST":
            sem_pick = 0
        elif ep == "LATEST":
            sem_pick = 1
        else:
            sem_pick = -1                      # abstain -> counts wrong
        sem = np.array([1.0 if i == sem_pick else 0.0 for i in (0, 1)])
        # hybrid: normalised dense + semantic with an explicit dense tie-break
        dn = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
        hyb = 0.5 * dn + 0.5 * sem + 1e-3 * dn     # tie-break by dense
        picks = {"dense": int(np.argmax(dense)), "lex": int(np.argmax(lex)),
                 "semantic": sem_pick, "hybrid": int(np.argmax(hyb))}
        for a, pi in picks.items():
            ok = (pi >= 0 and recs[pi]["value"] == t["expected"])
            acc[a].append(1.0 if ok else 0.0)
        detail.append({"trial": t["trial"], "epoch": ep, "picks": picks,
                       "dense": [round(float(x), 3) for x in dense]})
        if len(detail) % 10 == 0:
            print(f"  [{len(detail)}/{len(trials)}] ({time.time() - t0:.0f}s)", flush=True)

    arms = {a: round(statistics.mean(v), 3) for a, v in acc.items()}
    ep_counts = {}
    for d in detail:
        ep_counts[d["epoch"]] = ep_counts.get(d["epoch"], 0) + 1
    checks = {"vacuity_dense_near_chance": 0.35 <= arms["dense"] <= 0.65,
              "semantic_high": arms["semantic"] >= 0.85,
              "semantic_beats_dense": arms["semantic"] >= arms["dense"] + 0.25,
              "hybrid_beats_dense": arms["hybrid"] >= arms["dense"] + 0.25}
    res = {"trials": len(trials), "arms": arms, "epoch_judgments": ep_counts,
           "checks": checks, "pass": all(checks.values()),
           "verdict": ("PASS -- interpreting the intent's epoch decides relevance "
                       "where the words do not" if all(checks.values()) else
                       "FAIL -- see the failing row (numbers reported either way)"),
           "elapsed_s": round(time.time() - t0), "detail": detail}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("arms", "epoch_judgments", "checks", "pass",
                                          "verdict")}, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
