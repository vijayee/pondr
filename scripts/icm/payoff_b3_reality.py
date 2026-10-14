"""PAYOFF v3-REALITY -- the same candidate, two goals, REALITY as the scorer.

The only non-similarity oracle in the system: the ledger's congruence
mechanism compares prediction to a MEASURED outcome. Every prior benchmark
scored success with an n-gram or a judge -- dense-similarity instruments --
which is why they all collapsed into similarity-shaped labels. This one
scores with the FILESYSTEM: a manifest file's checksum compared to the
archive the agent recalls.

Design (two goals, ONE candidate, the reversal):
    World: dataset ingests with real files; every run leaves manifest.json
           with the data's checksum; the episode stores the archive.
    GOAL A (verify-archive): "confirm the manifest on disk matches its
           archived record" -- the candidate (the RIGHT archived run) pays
           +1: the agent compares recalled sha vs file sha -> match -> PASS.
           A swapped twin's sha yields a reported MISMATCH -> FAIL.
           None -> no archive to compare -> FAIL. Outcome = measured.
    GOAL B (fresh-reingest): "the dataset was re-ingested; write the UPDATED
           manifest by RECOMPUTING the checksum; do not trust any archived
           record" -- the SAME candidate's recalled sha is now STALE: trusting
           it writes the WRONG sha -> measured FAIL; recomputing -> PASS.
           The candidate's payoff flips: +1 under goal A, <= 0 under goal B.

Pre-registered (the labels-recovery null does not apply -- there are no
judge labels; the payoff = reality deltas; the sim-shaped pre-check runs
on the GOAL-FOLLOWING arms only for difficulty calibration).

Run:  python scripts/icm/payoff_b3_reality.py   (~72 relay calls, ~25 min)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
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
              '\n Reply ONLY as JSON {"sha": "<the checksum you wrote/trust>", '
              '"status": "<match|mismatch|recomputed|failed>", "note": "<short>"}')
    if memory_block:
        prompt += "\n\nArchived record from memory:\n" + memory_block
    body = {"model": policy.model, "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": {"type": "object", "properties": {"sha": {"type": "string"},
                                                        "status": {"type": "string"},
                                                        "note": {"type": "string"}},
                       "required": ["sha", "status", "note"]},
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
            return rec, (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
            if attempt >= 2:
                return {}, (time.perf_counter() - t0) * 1000
            time.sleep(2.0)
        except Exception:
            if attempt >= 2:
                return {}, (time.perf_counter() - t0) * 1000
            time.sleep(2.0)
    return {}, 0.0


def checksum_dir(directory: str) -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(directory)):
        with open(os.path.join(directory, fn), "rb") as f:
            h.update(f.read())
        h.update(fn.encode())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-per-goal", type=int, default=12)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--db-path", type=str, default=os.path.join("data", "_scratch", "icm", "reality"))
    args = ap.parse_args()
    policy = DeepSeekAgentPolicy(model=args.model)
    shutil.rmtree(args.db_path, ignore_errors=True)
    env_root = os.path.join(args.db_path, "env")
    import random

    rng = random.Random(71)
    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    n = args.tasks_per_goal
    world: dict[int, dict] = {}
    twins: dict[int, int] = {}
    try:
        # ---- archive sessions (task A sources): real files + real archive episodes
        for si in range(n):
            fixture = f"fixture-{chr(97 + si % 4)}-{si}"
            rows = 300 + si * 25
            directory = os.path.join(env_root, f"run-{si:04d}")
            os.makedirs(directory, exist_ok=True)
            shas = []
            for part in range(3):
                body = "\n".join(f"row-{part}-{r},value={rng.randint(0, 999)}" for r in range(rows // 3))
                with open(os.path.join(directory, f"part-{part}.csv"), "w") as f:
                    f.write(body)
                shas.append(body[:64])
            sha = checksum_dir(directory)
            text = (f"[run {si:04d}] Ingested {fixture}: {rows} rows. Manifest sha archived: {sha}. "
                    f"Probe latency {int(8 + rows / 40)}ms.")
            store.capture(session_id=f"archive-{si:04d}", intent_id=None,
                          text=text, episode_id=f"ep-arch-{si:04d}")
            world[si] = {"fixture": fixture, "rows": rows, "sha": sha,
                         "dir": directory, "ep": f"ep-arch-{si:04d}"}
        # twin mapping (another archived run, same-ish template)
        for si in range(n):
            twins[si] = (si + 1) % n
        print(f"[archive] {n} real archives on disk", flush=True)

        # ---- GOAL A: verify-archive (VERDICT task; NO LEAK: the answer is the
        # verdict, not the file sha -- the file sha in the prompt is context).
        # REALITY: the file's sha == the TRUE archive's sha (data unchanged).
        # success = the agent's verdict matches reality ("match").
        # right-recall sees the true archive -> match; twin-recall holds a
        # WRONG archive -> a trusting agent reports "mismatch" (fail); none
        # cannot know the archive -> 50/50 guess (chance baseline 0.5, NOT 0).
        outcomes_A = {"right": [0, 0], "twin": [0, 0], "none": [0, 0]}
        for si in range(n):
            w = world[si]
            file_sha = checksum_dir(w["dir"])  # reality: unchanged
            task_prompt = (f"Archived-record check for {w['fixture']} (run {si:04d}): the manifest "
                           f"on disk currently reads sha {file_sha}. Consult the archived record "
                           f"and confirm whether the archive MATCHES the on-disk manifest. "
                           f'Reply with the archived sha you trust and status "match" or "mismatch".')
            for arm, ep_id in (("right", w["ep"]), ("twin", world[twins[si]]["ep"]), ("none", None)):
                block = led.get("episode", ep_id)["text"] if ep_id else None
                rec, _ = ask_retry(policy, task_prompt, block)
                verdict = str(rec.get("status", "")).lower()
                ok = int("match" in verdict and "mismatch" not in verdict)  # reality: match
                outcomes_A[arm][0] += ok
                outcomes_A[arm][1] += 1
                time.sleep(3.0)

        # ---- GOAL B: fresh-reingest (GOAL-TRUST task; THE REVERSAL: the SAME
        # candidate whose recall passed goal A is now STALE. The new sha is
        # stated IN the prompt (no leak: the answer = trusting reality over
        # the stale memory). Success = reporting the NEW sha. Trusting the
        # stale archive = fail. The candidate's payoff flips: +1 under goal
        # A, NEGATIVE (active harm) under goal B if trust defeats the goal.
        stale_used = stale_ignored = 0
        outcomes_B = {"right": [0, 0], "twin": [0, 0], "none": [0, 0]}
        for si in range(n):
            w = world[si]
            # REALITY CHANGE: rewrite the data -> a fresh checksum
            with open(os.path.join(w["dir"], "part-0.csv"), "a") as f:
                f.write("\nrow-new-appended,value=42\n")
            fresh_sha = checksum_dir(w["dir"])
            w["fresh_sha"] = fresh_sha
            task_prompt = (f"Fresh-reingest update for {w['fixture']} (run {si:04d}): the dataset "
                           f"was just re-ingested. The NEW manifest on disk reads sha {fresh_sha}. "
                           f"Update the team archive: report the sha the archive SHOULD record "
                           f'now, and status "recomputed".')
            for arm, ep_id in (("right", w["ep"]), ("twin", world[twins[si]]["ep"]), ("none", None)):
                block = led.get("episode", ep_id)["text"] if ep_id else None
                rec, _ = ask_retry(policy, task_prompt, block)
                rsha = str(rec.get("sha", "")).strip()
                ok = int(rsha == fresh_sha)
                if arm != "none" and rsha and rsha != fresh_sha:
                    if rsha == w["sha"]:
                        stale_used += 1
                    else:
                        stale_ignored += 1
                outcomes_B[arm][0] += ok
                outcomes_B[arm][1] += 1
                time.sleep(3.0)
    finally:
        led.close()

    def rate(arms, arm):
        v = arms[arm]
        return {"success": v[0], "n": v[1], "rate": round(v[0] / max(1, v[1]), 4)}

    out = {
        "reality_oracle": "filesystem checksums (no judge, no n-gram)",
        "goal_A_verify_archive": {k: rate(outcomes_A, k) for k in outcomes_A},
        "goal_B_fresh_reingest": {"arms": {k: rate(outcomes_B, k) for k in outcomes_B},
                                  "stale_sha_used": stale_used,
                                  "stale_sha_ignored": stale_ignored},
        "reversal_claim": ("the candidate's payoff flips with the GOAL: under verify-archive the "
                           "archive episode is load-bearing; under fresh-reingest it is stale: "
                           "same candidate, different goal, reality decides both"),
    }
    path = os.path.join("data", "_scratch", "icm", "payoff_reality_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())