"""PAYOFF BENCHMARK v0 -- the corpus the dense-leg premise needs.

The job the premise is about has never been instantiated: no record in the
repo contains a (recall candidate x downstream success) relation. This
builds it.

Design (dependent tasks; the world is the same correlated law world):
    Task A: an ingest run; its episode stores the run's facts.
    Task B (a LATER task, its own session): "verify the probe latency
           recorded for the run whose backup completed at {ts}" -- the answer
           requires A's stored fact; the agent CANNOT know it from its own
           context (the value came from the env dice in task A).

For each task B, three counterfactual consult states:
    NONE  -- no memory          (baseline fail expected)
    RIGHT -- the task A episode (success expected)
    TWIN  -- a different task A episode (FAIL expected: wrong value)
The payoff label for a candidate = the graded success under its presence.
This is exactly the per-recall reward the intent head needs; the corpus is
small but instantiates the JOB (not values, not retrieval-correctness --
payoff).

Run:  python scripts/icm/payoff_b0.py  (~180 relay calls, ~30 min)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

from e0_v3_corpus import build, N_SESSIONS, question_symbolic  # noqa: E402
from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

_model = None


def _embedder(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


def ask_retry(policy, task_prompt, memory_block):
    prompt = (task_prompt +
              '\n Reply ONLY as JSON {"answer": "<a short phrase>"} containing the specific value.')
    if memory_block:
        prompt += "\n\nEvidence record from memory:\n" + memory_block
    body = {"model": policy.model, "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"]},
            "options": {"temperature": 0.2, "num_ctx": 8192}}
    import urllib.error
    import urllib.request

    backoff = 10.0
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
            rec = json.loads(resp["message"]["content"])
            return str(rec.get("answer", "")).strip(), (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60))
                backoff *= 1.8
                continue
            if attempt >= 2:
                return "", (time.perf_counter() - t0) * 1000
            time.sleep(2.0)
        except Exception:
            if attempt >= 2:
                return "", (time.perf_counter() - t0) * 1000
            time.sleep(2.0)
    return "", 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=str, default=os.path.join("data", "_scratch", "icm", "payoffb0"))
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    args = ap.parse_args()
    policy = DeepSeekAgentPolicy(model=args.model)
    shutil.rmtree(args.db_path, ignore_errors=True)
    rng = random.Random(43)

    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    data = build(7, N_SESSIONS)
    by_ts = {s.ts: s for s in data}
    n = min(args.tasks, N_SESSIONS)
    task_sample = rng.sample(data, n)
    try:
        # Task A: harvest episodes for the SOURCE runs only (the tasks' anchors
        # reference runs whose episodes exist; the anchor is their backup ts)
        rows_by_anchor: dict = {}
        for s in data:
            text = (f"[run {s.si:04d}] Ingested a dataset of {s.rows} rows. "
                    f"Probe latency was {s.lat}ms.")
            store.capture(session_id=f"runA-{s.si:04d}", intent_id=None,
                          text=text, episode_id=f"ep-{s.si:04d}")
            rows_by_anchor[s.ts] = (s, f"ep-{s.si:04d}")
        print(f"[taskA] {len(rows_by_anchor)} source episodes captured", flush=True)

        # Task B: verify the recorded latency for an anchored run
        results = {"none": [0, 0], "right": [0, 0], "twin": [0, 0]}
        per_task = []
        for ti, ts in enumerate([s.ts for s in task_sample]):
            target, right_ep = rows_by_anchor[ts]
            gold = str(target.lat)
            twin_ep = right_ep
            while True:
                other, oep = rows_by_anchor[rng.choice([s.ts for s in data])]
                if oep != right_ep:
                    twin_ep = oep
                    break
            twin_rec = led.get("episode", twin_ep)
            right_rec = led.get("episode", right_ep)
            task_prompt = (question := (f"Verify the recorded probe latency for the dataset ingest "
                                        f"run whose backup completed at {ts}:"))
            variants = {"right": right_rec["text"], "twin": twin_rec["text"]}
            row = {"task": ti, "anchor": ts, "gold": gold}
            for arm, block in (("none", None), ("right", variants["right"]), ("twin", variants["twin"])):
                ans, _ms = ask_retry(policy, task_prompt, block)
                ok = int(bool(gold) and gold in ans)
                results[arm][0] += ok
                results[arm][1] += 1
                row[arm] = {"answer": ans, "ok": ok}
            # the PAYOFF label per candidate = success delta under presence
            row["payoff"] = {"right": row["right"]["ok"] - row["none"]["ok"],
                             "twin": row["twin"]["ok"] - row["none"]["ok"]}
            per_task.append(row)
            time.sleep(3.0)
            if ti % 10 == 0:
                print(f"[task {ti}/{n}] none={results['none']} right={results['right']} "
                      f"twin={results['twin']}", flush=True)
    finally:
        led.close()

    out = {"tasks": n,
           "by_arm": {k: {"success": v[0], "n": v[1], "rate": round(v[0] / max(1, v[1]), 4)}
                      for k, v in results.items()},
           "per_task": per_task,
           "note": "payoff label per candidate = graded success delta under its presence -- "
                   "the per-recall reward the intent head trains on"}
    path = os.path.join("data", "_scratch", "icm", "payoff_b0_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())