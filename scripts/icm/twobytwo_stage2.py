"""2x2 STAGE-2 -- scaled epochs (3 per artifact) and the TWO-STAGE retriever.

WHY (addendum 22): with two epochs and a fair construction, judge-read epochs
+ mechanical ordering scored 40/40 while dense sat at chance; the equal-weight
hybrid failed because fusing scores at the same level cancels the semantic
signal. The architecture the results point at is TWO-STAGE: dense generates
candidates, the intent (epoch) re-ranks them. This run tests exactly that, at
a harder scale.

CONSTRUCTION (filesystem-verified, no state words):
  4 artifact types x 5 datasets x 3 EPOCHS = 60 memory records on disk.
  Records differ ONLY in value, run number and date; no record says
  old/new/current/archived. Intents refer to the epoch relatively
  ("at the first load" / "at the middle capture" / "at the most recent load").
  Trial = (scenario, epoch) -> the correct record is the one with that epoch's
  date. 60 trials, chance = 1/3.

POOLS (the scale-up): for each trial the candidate pool = the 15 records of the
same artifact type (5 datasets x 3 epochs) + 10 seeded decoys drawn from the
other types = 25 candidates.

ARMS:
  DENSE@1        top-1 of cosine(intent, record) over the pool (baseline)
  STAGE1@5       recall@5 of the dense top-5 for the correct record (the
                 precondition of any two-stage design; reported, not a bar)
  SEMANTIC-ONLY  epoch from the judge + entity filter (records must contain the
                 intent's dataset) + pick by date rank, over the whole pool
  TWO-STAGE      dense top-5 -> entity filter -> epoch rank (fall back to
                 dense@1 if the stage-2 set is empty)
  HYBRID         semantic primary, dense as fallback (no score blending; the
                 addendum-22 lesson)

PRE-REGISTERED BAR:
  (1) VACUITY: DENSE@1 in 0.20..0.60 (chance is 0.33)
  (2) STAGE1@5 >= 0.85 (dense's candidates contain the answer often enough for
      a re-ranker to matter)
  (3) TWO-STAGE >= 0.85
  (4) TWO-STAGE >= DENSE@1 + 0.25
  PASS -> two-stage retrieval (dense candidates + intent re-rank) decides
  correctly at 3x chance while similarity alone does not.

Judge: epoch classification per (artifact type, epoch) TEMPLATE (12 calls;
epoch is dataset-independent -- pre-registered). Mechanical verification: the
chosen record's VALUE must equal the expected value for that trial.

Run:  python scripts/icm/twobytwo_stage2.py
"""

from __future__ import annotations

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

ENV = os.path.join("data", "_scratch", "icm", "twobytwo_stage2", "env")
OUT = os.path.join("data", "_scratch", "icm", "twobytwo_stage2_results.json")

_emb_model = [None]


def emb(texts):
    if _emb_model[0] is None:
        from sentence_transformers import SentenceTransformer
        _emb_model[0] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _emb_model[0].encode(texts, batch_size=32, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)


TYPES = {
    "checksum": ("dataset {d}: {rows} rows, checksum {v}, run {n}, dated {date}.",
                 "Report the checksum for dataset {d} at the first load.",
                 "Report the checksum for dataset {d} at the middle capture.",
                 "Report the checksum for dataset {d} at the most recent load."),
    "config": ("settings for {d}: timeout={v}ms, pool=8, captured on {date} (run {n}).",
               "Report the timeout for {d} from the earliest capture.",
               "Report the timeout for {d} from the middle capture.",
               "Report the timeout for {d} from the latest capture."),
    "cache": ("cache for {d}: key=region, value={v}, written on {date} (run {n}).",
              "Report the cached value for {d} from the first write.",
              "Report the cached value for {d} from the middle write.",
              "Report the cached value for {d} from the newest write."),
    "roster": ("roster for {d}: on-call={v}, dated {date} (run {n}).",
               "Report the on-call name for {d} from the initial handoff.",
               "Report the on-call name for {d} from the middle handoff.",
               "Report the on-call name for {d} after the most recent handoff."),
}
DATASETS = ["alpha", "beta", "gamma", "delta", "epsilon"]
DATES = ("2026-03-01", "2026-03-20", "2026-04-15")
RUNS = (12, 13, 14)


def make_world():
    rng = random.Random(19)
    os.makedirs(ENV, exist_ok=True)
    recs, trials = [], []
    for typ, (mtpl, *intents) in TYPES.items():
        for si, d in enumerate(DATASETS):
            vals = []
            for ei in range(3):
                v = {"checksum": "".join(rng.choice("0123456789abcdef") for _ in range(12)),
                     "config": str(rng.choice([300, 500, 800, 1200, 1500, 2000])),
                     "cache": "".join(rng.choice("0123456789abcdef") for _ in range(8)),
                     "roster": rng.choice(["lee", "kim", "ortiz", "nash", "wong", "silva"])}[typ]
                vals.append(v)
                txt = mtpl.format(d=d, v=v, date=DATES[ei], n=RUNS[ei], rows=rng.randint(100, 900))
                path = os.path.join(ENV, f"{typ}-{si:02d}-e{ei}.txt")
                io.open(path, "w", encoding="utf-8").write(txt)
                recs.append({"rec_id": f"{typ}-{si:02d}-e{ei}", "type": typ, "dataset": d,
                             "epoch": ei, "date": DATES[ei], "value": v, "text": txt})
            for ei in range(3):
                trials.append({"trial": f"{typ}-{si:02d}-e{ei}", "type": typ, "dataset": d,
                               "epoch": ei, "intent": intents[ei].format(d=d),
                               "correct_rec": f"{typ}-{si:02d}-e{ei}",
                               "expected": vals[ei]})
    return recs, trials


def epoch_of(intent_template: str):
    prompt = ("Does the request below ask about the EARLIEST state of something, a "
              "MIDDLE state, or the LATEST state? Answer exactly one word: "
              "EARLIEST, MIDDLE or LATEST.\n\nRequest: " + intent_template[:800])
    raw = ask(prompt).upper()
    return ("EARLIEST" if "EARLIEST" in raw else
            "MIDDLE" if "MIDDLE" in raw else
            "LATEST" if "LATEST" in raw else None)


def main() -> int:
    recs, trials = make_world()
    by_type = {}
    for r in recs:
        by_type.setdefault(r["type"], []).append(r)
    print(f"[env] {len(recs)} records, {len(trials)} trials", flush=True)
    t0 = time.time()

    # judge: epoch per (type, epoch) template (12 calls; dataset-independent)
    tmpl_epoch = {}
    for typ, (_mtpl, *intents) in TYPES.items():
        for ei, it in enumerate(intents):
            tmpl_epoch[(typ, ei)] = epoch_of(it)
            print(f"  [judge] {typ}/e{ei} -> {tmpl_epoch[(typ, ei)]}", flush=True)
    ep_map = {"EARLIEST": 0, "MIDDLE": 1, "LATEST": 2}

    rng = random.Random(5)
    acc = {"dense@1": [], "semantic_only": [], "two_stage": [], "hybrid": []}
    stage1 = []
    for t in trials:
        pool = list(by_type[t["type"]])
        others = [r for r in recs if r["type"] != t["type"]]
        pool += rng.sample(others, min(10, len(others)))
        assert any(r["rec_id"] == t["correct_rec"] for r in pool)
        texts = [r["text"] for r in pool]
        ie = emb([t["intent"]])[0]
        pe = emb(texts)
        dense = pe @ ie
        order = list(np.argsort(-dense))
        # stage-1 recall@5
        ranks = [i for i, ri in enumerate(order) if pool[ri]["rec_id"] == t["correct_rec"]]
        stage1.append(1.0 if ranks and ranks[0] < 5 else 0.0)
        # dense@1
        top1 = pool[order[0]]
        acc["dense@1"].append(1.0 if top1["value"] == t["expected"] else 0.0)

        def semantic_pick(cands):
            # entity filter then epoch rank; fall back to the first candidate
            # entity + artifact-type filter: decoys of the same dataset but a
            # different artifact type must not enter the epoch ranking
            ent = [r for r in cands
                   if r["type"] == t["type"] and t["dataset"] in r["text"].lower()]
            use = ent or cands
            want = ep_map.get(tmpl_epoch.get((t["type"], t["epoch"])), None)
            if want is None:
                return use[0]
            use = sorted(use, key=lambda r: r["epoch"])
            return use[min(want, len(use) - 1)]

        sem_all = semantic_pick(pool)
        acc["semantic_only"].append(1.0 if sem_all["value"] == t["expected"] else 0.0)
        stage2 = semantic_pick([pool[i] for i in order[:5]])
        acc["two_stage"].append(1.0 if stage2["value"] == t["expected"] else 0.0)
        hyb = stage2 if tmpl_epoch.get((t["type"], t["epoch"])) else top1
        acc["hybrid"].append(1.0 if hyb["value"] == t["expected"] else 0.0)
        if len(acc["dense@1"]) % 15 == 0:
            print(f"  [{len(acc['dense@1'])}/{len(trials)}] ({time.time() - t0:.0f}s)",
                  flush=True)

    arms = {k: round(statistics.mean(v), 3) for k, v in acc.items()}
    s1 = round(statistics.mean(stage1), 3)
    checks = {"vacuity_dense_near_chance": 0.20 <= arms["dense@1"] <= 0.60,
              "stage1_recall5": s1 >= 0.85,
              "two_stage_high": arms["two_stage"] >= 0.85,
              "two_stage_beats_dense": arms["two_stage"] >= arms["dense@1"] + 0.25}
    res = {"trials": len(trials), "records": len(recs), "pool": "15 same-type + 10 decoys",
           "arms": arms, "stage1_recall@5": s1, "epoch_judgments": {
               f"{k[0]}/e{k[1]}": v for k, v in tmpl_epoch.items()},
           "checks": checks, "pass": all(checks.values()),
           "verdict": ("PASS -- two-stage retrieval (dense candidates + intent "
                       "re-rank) decides correctly where similarity alone does not"
                       if all(checks.values()) else
                       "FAIL -- see the failing row (numbers reported either way)"),
           "elapsed_s": round(time.time() - t0)}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("arms", "stage1_recall@5", "checks", "pass",
                                          "verdict")}, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
