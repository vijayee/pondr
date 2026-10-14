"""PAYOFF BENCHMARK v2 -- adversarial relevance (the similarity-vs-relevance inversion).

Retired so far: twins (v0 identity), env-dice payoff (v0), goal-fit (=similarity, v1).
v2 inverts: the MOST SIMILAR candidate (raw bge) is the WRONG one; the payoff
memory is LESS similar but fits the stated goal.

    Query mentions the instability symptom ("chasing a probe instability
    report ... resets") -- so the symptom episode (the run that misbehaved)
    reads as the most similar.
    The STATED GOAL, though, is to record the team's stability BASELINE:
    "the clean reference run -- state its probe latency and stability."
    The baseline episode (a clean sibling run) carries the gold values; the
    symptom episode carries wrong ones (its lats are failure-lats).

Pre-registered checks:
    P1: raw untrained bge MUST rank the distractor above the payoff episode
        (difficulty proof: similarity and relevance genuinely disagree).
    P2: null for the trained head (next step): if the head cannot beat
        P1's misranking, the "relevance beats similarity" premise dies
        having finally been tested.
Arms: NONE / DISTRACTOR (similar, wrong) / BASELINE (less similar, right).
Grading = exact gold-token containment (the baseline episode's own lat).

Run:  python scripts/icm/payoff_b2.py  (~60 relay calls, ~20 min)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

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
              '\n Reply ONLY as JSON {"answer": "<a short phrase>"} containing the specific values.')
    if memory_block:
        prompt += "\n\nEvidence record from memory:\n" + memory_block
    body = {"model": policy.model, "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"]},
            "options": {"temperature": 0.2, "num_ctx": 8192}}
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
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
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
    ap.add_argument("--db-path", type=str, default=os.path.join("data", "_scratch", "icm", "payoffb2"))
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    args = ap.parse_args()
    policy = DeepSeekAgentPolicy(model=args.model)
    shutil.rmtree(args.db_path, ignore_errors=True)
    rng = __import__("random").Random(91)

    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    fixtures = ["fixture-a", "fixture-b", "fixture-c", "fixture-d"]
    try:
        # world: per fixture, one UNSTABLE run (the symptom source) + one CLEAN
        # baseline run, value laws per rows; extra decoy runs fill the corpus
        unstable, baselines = {}, {}
        for fi, fixture in enumerate(fixtures):
            rows_u = 400 + fi * 300
            lat_u = int(8 + rows_u / 40)
            rows_b = rows_u + 120
            lat_b = int(8 + rows_b / 40)
            unstable[fixture] = {
                "text": (f"[run {fi * 2 + 1:04d}] Ingested a dataset of {rows_u} rows on {fixture}. "
                         f"The dependency probe returned a connection reset twice "
                         f"({lat_u + 30}ms, {lat_u + 55}ms); third attempt green at {lat_u}ms. "
                         f"Disk at {55 + rows_u // 80} percent."),
                "lat": lat_u}
            baselines[fixture] = {
                "text": (f"[run {fi * 2 + 2:04d}] Ingested a dataset of {rows_b} rows on {fixture}. "
                         f"The dependency probe was stable: green across 3 checks "
                         f"at {lat_b}ms, {lat_b - 2}ms and {lat_b + 1}ms. "
                         f"Disk at {55 + rows_b // 80} percent."),
                "lat": lat_b}
        si = 100
        for fixture in fixtures:
            for k in range(20):
                rows_d = 220 + ((si * 37) % 2800)
                lat_d = int(8 + rows_d / 40)
                store.capture(session_id=f"run-{si:04d}", intent_id=None,
                              text=(f"[run {si:04d}] Ingested a dataset of {rows_d} rows on {fixture}. "
                                    f"The dependency probe was stable: green at {lat_d}ms. "
                                    f"Disk at {55 + rows_d // 80} percent."),
                              episode_id=f"ep-d-{si:04d}")
                si += 1
        for fixture in fixtures:
            store.capture(session_id=f"run-u-{fixture}", intent_id=None,
                          text=unstable[fixture]["text"], episode_id=f"ep-sym-{fixture}")
            store.capture(session_id=f"run-b-{fixture}", intent_id=None,
                          text=baselines[fixture]["text"], episode_id=f"ep-base-{fixture}")
        print(f"[seed] {led.count('episode')} episodes", flush=True)

        tasks = []
        for fixture in fixtures:
            sym_ep = f"ep-sym-{fixture}"
            base_ep = f"ep-base-{fixture}"
            q = (f"We are chasing a probe instability report on {fixture} that mentioned "
                 f"connection resets. For the team's stability baseline, we need the clean "
                 f"reference run: state its probe latency and stability.")
            gold = str(baselines[fixture]["lat"])
            tasks.append({"q": q, "gold": gold, "sym": sym_ep, "base": base_ep,
                          "fixture": fixture})
        if len(tasks) < args.tasks:
            tasks = tasks * 3

        # ---- P1 difficulty proof: raw cosine must rank DISTRACTOR above BASELINE ----
        p1 = {"distractor_above": 0, "n": len(tasks)}
        for t in tasks:
            qe = np.asarray(_embedder([t["q"]]), dtype=np.float32)
            se = np.asarray(_embedder([led.get("episode", t["sym"])["text"]]), dtype=np.float32)
            be = np.asarray(_embedder([led.get("episode", t["base"])["text"]]), dtype=np.float32)
            s_s = float((qe * se).sum()); s_b = float((qe * be).sum())
            p1["distractor_above"] += int(s_s > s_b)
        p1["rate"] = round(p1["distractor_above"] / p1["n"], 3)
        print(f"[P1] raw cosine ranks the distractor FIRST on {p1['distractor_above']}/{p1['n']} tasks",
              flush=True)
        assert p1["distractor_above"] >= p1["n"] * 0.8, (
            "regime not adversarial: raw cosine must prefer the distractor")

        results = {"none": [0, 0], "distractor": [0, 0], "baseline": [0, 0]}
        per_task = []
        tasks_used = (tasks * ((args.tasks + len(tasks) - 1) // len(tasks)))[: args.tasks]
        for ti, t in enumerate(tasks_used):
            arms = {"none": None,
                    "distractor": led.get("episode", t["sym"])["text"],
                    "baseline": led.get("episode", t["base"])["text"]}
            row = {"task": ti, "fixture": t["fixture"], "gold": t["gold"]}
            for arm, block in arms.items():
                ans, _ms = ask_retry(policy, t["q"], block)
                ok = int(t["gold"] in ans.lower())
                results[arm][0] += ok
                results[arm][1] += 1
                row[arm] = {"answer": ans, "ok": ok}
            row["payoff"] = {"baseline": row["baseline"]["ok"] - row["none"]["ok"],
                             "distractor": row["distractor"]["ok"] - row["none"]["ok"]}
            per_task.append(row)
            time.sleep(3.0)
            if ti % 5 == 0:
                print(f"[task {ti}/{len(tasks_used)}]", results, flush=True)
    finally:
        led.close()

    out = {"tasks": len(per_task),
           "P1_difficulty_proof": p1,
           "by_arm": {k: {"success": v[0], "n": v[1], "rate": round(v[0] / max(1, v[1]), 4)}
                      for k, v in results.items()},
           "per_task": per_task,
           "note": "payoff vs similarity now genuinely disagree -- a head trained to follow "
                   "the GOAL must contradict untrained cosine"}
    path = os.path.join("data", "_scratch", "icm", "payoff_b2_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())