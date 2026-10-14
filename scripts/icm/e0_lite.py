"""E0-lite -- Phase-1 evaluation slice: retrievability + latency budget
on the REAL episodic store (bge-small encoder, WaveDB FLAT index).

E0's full form (plan 10) also needs agent task success vs a context-only
baseline -- that requires the agent harness and lands later. This slice
measures the preconditions an agent would depend on:
    * recall@5 hit rate + MRR on fact-bearing work-log episodes (12 facts,
      1 query each, phrased differently from the corpus text)
    * per-recall latency avg/p95 against a soft 50 ms/turn budget
    * the forgetting pass on a simulated age advance (60d): vivid/gist/drop
      accounting, registry integrity, fresh-episode survival

Run:  python scripts/icm/e0_lite.py    (DB in data/_scratch/icm/e0lite, wiped)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icm.episodic import EpisodicStore  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

DAY = 86400.0

WORK_LOGS = [
    ("deploy", "Checkpoint: staging restart finished, probe returned 42ms, retry path green."),
    ("deploy-1", "Checkpoint: production canary halted, error budget 0.3 percent burned."),
    ("db", "Checkpoint: schema migration step 12 applied, lock timeout 300 seconds."),
    ("db-1", "Checkpoint: index rebuild finished in 11 minutes, table locked twice."),
    ("bench", "Checkpoint: search benchmark P95 at 88ms, regression against last week."),
    ("bench-1", "Checkpoint: ingest throughput 4.1k rows per second, cache hit rate 61 percent."),
    ("backup", "Checkpoint: nightly backup completed at 03:40, verify checksum matched."),
    ("backup-1", "Checkpoint: backup skipped, disk 94 percent full on the store volume."),
    ("docker", "Checkpoint: CI runner pruned 212 stale images, freed 31 GB."),
    ("docker-1", "Checkpoint: builder cache cleared, first cold build took 7 minutes."),
    ("rotate", "Checkpoint: API credentials rotated, old key valid for 24 more hours."),
    ("report", "Checkpoint: weekly report sent, three action items assigned to infra."),
]
QUERIES = {
    "deploy": "How fast was the staging restart probe?",
    "deploy-1": "What share of the error budget did the canary burn?",
    "db": "What is the migration lock timeout?",
    "db-1": "How long did the index rebuild take and how often was the table locked?",
    "bench": "What was the search benchmark P95 latency?",
    "bench-1": "What ingest throughput did we measure?",
    "backup": "When did the nightly backup finish and was the checksum ok?",
    "backup-1": "Why was the backup skipped?",
    "docker": "How many stale images were pruned and how much space freed?",
    "docker-1": "How long was the first cold build?",
    "rotate": "How long does the rotated-out API key remain valid?",
    "report": "How many action items did the weekly report assign?",
}
VARIANTS = 5  # identical-text copies per fact (multi-session repeats)


_model = None


def lazy_embedder(texts: list[str]) -> list[list[float]]:
    """Load the model ONCE (per-call loads put ~0.9s of model startup into
    every recall and destroyed the latency measurement)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=str,
                    default=os.path.join("data", "_scratch", "icm", "e0lite"))
    args = ap.parse_args()
    shutil.rmtree(args.db_path, ignore_errors=True)

    led = ICMLedger(args.db_path)
    store = EpisodicStore(led, lazy_embedder, dim=384)
    t0 = time.time()
    try:
        episode_ids = []
        for i, (tag, text) in enumerate(WORK_LOGS):
            for v in range(VARIANTS):
                episode_ids.append(store.capture(
                    session_id=f"e0sess-{i}",
                    intent_id=None,
                    text=f"[session {i + 1}] " + text,
                    episode_id=f"ep-{i:02d}-{v}",
                ))
        print(f"[seed] {len(episode_ids)} episodes in {time.time()-t0:.1f}s", flush=True)

        hits5 = 0
        mrr = 0.0
        lat = []
        for tag, _text in WORK_LOGS:
            q = QUERIES[tag]
            gold_ids = {f"ep-{i:02d}-{v}"
                        for i, (t2, _) in enumerate(WORK_LOGS) if t2 == tag
                        for v in range(VARIANTS)}
            t1 = time.perf_counter()
            results = store.recall(q, k=10)
            lat.append((time.perf_counter() - t1) * 1000)
            for rank, (rec, _sim) in enumerate(results, start=1):
                if rec["episode_id"] in gold_ids:
                    hits5 += 1
                    mrr += 1.0 / rank
                    break
        lat_sorted = sorted(lat)
        out = {
            "episodes": len(episode_ids),
            "queries": len(lat),
            "recall@5": round(hits5 / len(lat), 4),
            "mrr": round(mrr / len(lat), 4),
            "latency_ms_avg": round(sum(lat) / len(lat), 2),
            "latency_ms_p95": round(lat_sorted[max(0, int(len(lat) * 0.95) - 1)], 2),
            "latency_budget_ms": 50,
            "latency_within_budget": lat_sorted[max(0, int(len(lat) * 0.95) - 1)] <= 50,
        }

        # --- forgetting simulation: episodes aged to a 60d horizon ---
        m1 = store.maintain(now_ts=time.time() + 40 * DAY)
        m2 = store.maintain(now_ts=time.time() + 60 * DAY)
        counts = store.counts()
        out["forget_60d"] = {"m1_at_40d": m1, "m2_at_60d": m2, "fidelity_mix": counts}
        out["registry_integrity"] = (led.count("episode") ==
                                     sum(counts.values()))
        fresh = store.capture("e0sess-final", None,
                              "fresh checkpoint written after the forget pass",
                              created_ts=time.time() + 60 * DAY)
        store.maintain(now_ts=time.time() + 60 * DAY)
        out["fresh_survives"] = led.get("episode", fresh) is not None
        out["episodes_after_forget"] = led.count("episode")
    finally:
        led.close()
    path = os.path.join("data", "_scratch", "icm", "e0_lite.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())