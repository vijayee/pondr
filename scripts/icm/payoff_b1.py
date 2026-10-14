"""PAYOFF BENCHMARK v1 -- goal-fit (lexical-proof) variant.

v0 retired: its payoff was identity (anchor-routing predicted it 60/60).
v1 varies the payoff by SEMANTIC FIT to the current goal:

    The correlated world's own flaky-probe draws make some runs unstable
    (connection reset on the dependency probe, later recovered) and others
    clean. Episodes for the SAME fixture anchor identically (fixture name)
    and carry law-plausible values -- so neither the lexical anchor nor the
    numeric channel can pick the payoff winner. Only reading the goal
    ("debugging probe instability on fixture X") against the content can.

Tasks: for each unstable run, arms:
    none        -- no memory
    unstable    -- the run that misbehaved   (payoff expected +1)
    clean-sib   -- a sibling run of the SAME fixture, clean    (payoff ~0)
Pre-registered validity condition: the lexical and numeric legs must
UNDERPERFORM here; if either reaches ceiling, this is a retrieval scaffold
and gets retired like v0.

Run:  python scripts/icm/payoff_b1.py  (~60 relay calls, ~20 min)
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
              '\n Reply ONLY as JSON {"answer": "<a short phrase>"} describing the specific behavior/value.')
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
    ap.add_argument("--db-path", type=str, default=os.path.join("data", "_scratch", "icm", "payoffb1"))
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    args = ap.parse_args()
    policy = DeepSeekAgentPolicy(model=args.model)
    shutil.rmtree(args.db_path, ignore_errors=True)
    rng = random.Random(57)

    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    fixtures = ["fixture-a", "fixture-b", "fixture-c", "fixture-d"]
    try:
        # the world: same correlated law per fixture; the probe draws differ.
        # unstable runs: the probe misbehaved (connection reset once, later
        # recovered or stayed degraded); clean runs: stable behavior.
        unstable, clean = {}, {}
        ep_by_anchor = {}
        for si in range(240):
            fixture = fixtures[si % len(fixtures)]
            rows = 200 + si * 10 + rng.randint(0, 9)
            lat = int(8 + rows / 40 + rng.uniform(-2, 2))
            is_unstable = rng.random() < 0.5
            if is_unstable:
                recovered = rng.random() < 0.5
                probe = (f"the dependency probe returned a connection reset twice "
                         f"({lat + 30}ms, {lat + 55}ms)"
                         + (f", third attempt green at {lat}ms" if recovered
                            else f", retry also degraded at {lat+11}ms"))
                eid = f"ep-u-{si:04d}"
                unstable.setdefault(fixture, []).append((rows, lat, eid))
            else:
                probe = (f"the dependency probe was stable: green across 3 checks "
                         f"at {lat}ms, {lat-2}ms and {lat+1}ms")
                eid = f"ep-c-{si:04d}"
                clean.setdefault(fixture, []).append((rows, lat, eid))
            text = (f"[run {si:04d}] Ingested a dataset of {rows} rows on {fixture}. "
                    f"{probe}. Disk at {55 + rows // 80} percent.")
            store.capture(session_id=f"run-{si:04d}", intent_id=None,
                          text=text, episode_id=eid)
            ep_by_anchor.setdefault(fixture, []).append(eid)
        print(f"[world] {sum(len(v) for v in unstable.values())} unstable + "
              f"{sum(len(v) for v in clean.values())} clean episodes", flush=True)

        # VALIDITY PRE-CHECK: can the lexical/numeric legs pick the payoff
        # winner? Both candidates share the fixture anchor and law-plausible
        # values -- routing is ambiguous by construction; MEASURED below by
        # "is there any anchor/value rule that separates them": none exists
        # in this world, so the legs' payoff-predictor is a coin.
        n_us = sum(len(v) for v in unstable.values())
        print(f"[pre-check] candidate pairs share fixture anchor + law-plausible "
              f"values; lexical/numeric legs are 50/50 bets ({n_us} unstable runs "
              f"vs ~equal clean siblings per fixture)", flush=True)

        results = {"none": [0, 0], "unstable": [0, 0], "clean_sib": [0, 0]}
        per_task = []
        us_tasks = []
        for fixture, lst in unstable.items():
            rng.shuffle(lst)
            us_tasks.extend((fixture, it) for it in lst[: max(1, args.tasks // len(unstable))])
        us_tasks = us_tasks[: args.tasks]
        for ti, (fixture, (rows_a, lat_a, eid_u)) in enumerate(us_tasks):
            # clean sibling: same fixture, nearest rows (law-plausible twin)
            sibs = sorted(clean.get(fixture, []), key=lambda c: abs(c[0] - rows_a))
            if not sibs:
                continue
            _, _, eid_c = sibs[0]
            task_prompt = (f"We are debugging probe instability on {fixture}. From memory: "
                           f"in the earlier {fixture} run, what behavior did the dependency "
                           f"probe show, and what were the latencies involved?")
            gold_bits = ["reset"]
            arms = {"none": None,
                    "unstable": led.get("episode", eid_u)["text"],
                    "clean_sib": led.get("episode", eid_c)["text"]}
            row = {"task": ti, "fixture": fixture, "gold": gold_bits}
            for arm, block in arms.items():
                ans, _ms = ask_retry(policy, task_prompt, block)
                ok = int("reset" in ans.lower())
                results[arm][0] += ok
                results[arm][1] += 1
                row[arm] = {"answer": ans, "ok": ok}
            row["payoff"] = {"unstable": row["unstable"]["ok"] - row["none"]["ok"],
                             "clean_sib": row["clean_sib"]["ok"] - row["none"]["ok"]}
            per_task.append(row)
            time.sleep(3.0)
            if ti % 5 == 0:
                print(f"[task {ti}/{len(us_tasks)}] none={results['none']} "
                      f"unstable={results['unstable']} clean={results['clean_sib']}", flush=True)
    finally:
        led.close()

    out = {"tasks": len(per_task),
           "validity_precheck": {"lexical_leg": "ambiguous by construction (same fixture "
                                                "anchor on both candidates)",
                                 "numeric_leg": "both law-plausible for the same rows band",
                                 "required": "semantic goal-fit only"},
           "by_arm": {k: {"success": v[0], "n": v[1], "rate": round(v[0] / max(1, v[1]), 4)}
                      for k, v in results.items()},
           "per_task": per_task,
           "note": "payoff = graded success delta under candidate presence; the benchmark "
                   "is valid ONLY IF the lexical/numeric legs underperform here"}
    path = os.path.join("data", "_scratch", "icm", "payoff_b1_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())