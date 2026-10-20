"""STAGE-1 SWEEP (v4) -- does widening the generator close the two-stage gap?

WHY: v3 (addendum 29) passed, but TWO-STAGE 0.909 was capped by STAGE-1
recall@5 0.909 -- dense's top-5 missed the correct pre-change version in 51 of
561 trials -- while SEMANTIC-ONLY over the full filtered pool was 1.000. The
binding constraint was the candidate generator, not the intent key. This run
tests the fix.

PRE-REGISTERED DESIGN (written before the run; no new judge calls -- the v3
key cache answers every query):
  POOL: the 561 v3 trials (same records, same oracles).
  ARMS:
    DENSE@1                 top-1 cosine over the full pool
    DENSE-RELEVANT@1        first same-file record in the dense order
    STAGE1@5/10/20          is the correct blob in the dense top-k?
    TWO-STAGE@5/10/20       dense top-k -> type filter -> epoch re-rank
    FILTER-ONLY             type filter -> epoch re-rank over the FULL pool
                            (bypass dense when the artifact is known)
  KEY: the v3 write-time key, retrieved from the cache (unchanged).
  ORDER: ancestry depth, unchanged. No score blending.

  PRE-REGISTERED BAR:
    (1) POWER:        trials >= 100
    (2) FILTER-ONLY   >= 0.95  (the bypass should close the gap)
    (3) STAGE1@10     >= 0.95  (widening recovers the missed answers)
    (4) TWO-STAGE@10  >= 0.95
    (5) TWO-STAGE@20  >= 0.95
    (6) VACUITY:      dense-relevant <= chance + 0.15 (construction unchanged)
    Report @5 for the comparison; PASS requires all.

Usage:
  python scripts/icm/git_cf_stage1.py            # all 561
  python scripts/icm/git_cf_stage1.py --limit 10 # mechanics smoke
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from icm.intent_keys import IntentKeyer  # noqa: E402
from icm.intent_query import TwoStageRetriever  # noqa: E402

POOL = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "lifted",
                    "git_cf_v3", "pool.jsonl")
CACHE = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "lifted",
                     "git_cf_v3", "keys_cache.jsonl")
OUTDIR = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "lifted",
                      "git_cf_v4_stage1")
RESULTS = os.path.join(OUTDIR, "results.json")

_emb = [None]


def embed(texts):
    if _emb[0] is None:
        from sentence_transformers import SentenceTransformer
        _emb[0] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _emb[0].encode(list(texts), batch_size=32, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    trials = [json.loads(line) for line in io.open(POOL, encoding="utf-8")
              if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(trials)
    if args.limit:
        trials = trials[: args.limit]
    print(f"[pool] {len(trials)} trials; keys from cache", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)

    keyer = IntentKeyer(cache_path=CACHE)
    retriever = TwoStageRetriever(embed, keyer=keyer)
    arms = {name: [] for name in (
        "dense@1", "dense_relevant@1", "filter_only",
        "stage1@5", "stage1@10", "stage1@20",
        "two_stage@5", "two_stage@10", "two_stage@20")}
    t0 = time.time()
    for i, t in enumerate(trials, 1):
        meta = {"type": t["file"], "ignore_key_entities": True, "k": 20}
        full = retriever.query(t["query"], t["records"], query_meta=meta)
        key = full["key"]
        by_id = {r["id"]: r for r in t["records"]}
        order = full["stage1_ids"]
        correct = t["correct"]
        same = [x for x in order if by_id.get(x, {}).get("type") == t["file"]]
        dense_rel = same[0] if same else None
        arms["dense@1"].append(1.0 if order[0] == correct else 0.0)
        arms["dense_relevant@1"].append(1.0 if dense_rel == correct else 0.0)
        arms["filter_only"].append(1.0 if full["top"] and
                                   full["top"][0]["id"] == correct else 0.0)
        for k in (5, 10, 20):
            arms[f"stage1@{k}"].append(1.0 if correct in order[:k] else 0.0)
            cands = [by_id[x] for x in order[:k] if x in by_id]
            sub = retriever.query(t["query"], cands, query_key=key,
                                  query_meta=meta)
            top = sub["top"][0]["id"] if sub["top"] else None
            arms[f"two_stage@{k}"].append(1.0 if top == correct else 0.0)
        if i % 50 == 0:
            print(f"  [{i}/{len(trials)}] ({time.time()-t0:.0f}s)", flush=True)

    means = {k: round(statistics.mean(v), 3) for k, v in arms.items()}
    chance = round(statistics.mean(1.0 / t["pool"] for t in trials), 3)
    chance_rel = round(statistics.mean(1.0 / t["versions"] for t in trials), 3)
    checks = {
        "power_ge_100": len(trials) >= 100,
        "filter_only_ge_0.95": means["filter_only"] >= 0.95,
        "stage1_at_10_ge_0.95": means["stage1@10"] >= 0.95,
        "two_stage_at_10_ge_0.95": means["two_stage@10"] >= 0.95,
        "two_stage_at_20_ge_0.95": means["two_stage@20"] >= 0.95,
        "vacuity": means["dense_relevant@1"] <= chance_rel + 0.15,
    }
    res = {"trials": len(trials), "chance_dense": chance,
           "chance_relevant": chance_rel, "arms": means, "checks": checks,
           "pass": all(checks.values()),
           "verdict": ("PASS -- widening / bypassing stage-1 closes the gap"
                       if all(checks.values()) else
                       "FAIL -- see the failing row"),
           "elapsed_s": round(time.time() - t0)}
    with io.open(RESULTS, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)
    print(f"[done] -> {RESULTS}", flush=True)
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
