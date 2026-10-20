"""E0-LITE -- does episodic memory improve long-horizon task success vs context-only?

PRE-REGISTERED DESIGN (written before the run; prepared fixture, not a public
benchmark):

WORLD: four users, 20 tasks each. Each user has a verification convention
(header / excluded file / name-hash / duplicate rows). The convention is stated
ONCE in a user-profile message at the start; every task is a fresh dataset and
the convention is never restated. The verifier is mechanical and GENERIC: on a
failure it says only "verification failed" (no defect), so the profile is the
only source of the convention.

CONTEXT WINDOW: the context-only arm sees only the last 2 messages plus the
current task. At task 1 the profile is visible; from task 2 the profile has
fallen out of the window. With no convention it falls back to a naive manifest
(files=4, rows=all lines, hash_mode=content), which is wrong for every rule.
This simulates the plan's long-horizon context loss, not a real token window.

ARMS:
  MEMORY      a WaveDB EpisodicStore holds the profile and every prior task
              episode; before each task the agent recalls the user's
              convention (bge + vector search, per-user filtered) and builds
              the manifest from it.
  CONTEXT     no store; only the 2-message window + current task.
The agent is deterministic; no model calls.

METRICS: task success per arm; the memory-vs-context gap; recall latency
(mean, p95); the profile-out-of-window check for tasks 2+.

PRE-REGISTERED BAR:
  (1) POWER:     4 users x 20 tasks = 80 tasks per arm
  (2) MEMORY     success >= 0.90
  (3) CONTEXT    success <= 0.20
  (4) LIFT       memory - context >= 0.50
  (5) LATENCY    episodic recall p95 <= 250 ms per task
  (6) WINDOW     the profile is outside the context window for every task >= 2
  PASS requires all.

SCOPE: this is E0's mechanism in a prepared world: the memory store, not a
public agent benchmark. It tests whether episodic recall carries an early
user-level fact across a horizon that the context window loses.
"""

from __future__ import annotations

import io
import json
import os
import re
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from icm.episodic import EpisodicStore  # noqa: E402
from icm.layer3_env import DUPLICATE_ROWS, EXCLUDED_FILE, RULES, Layer3Env  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

OUTDIR = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "e0_lite")
RESULTS = os.path.join(OUTDIR, "results.json")
LEDGER_DB = os.path.join(OUTDIR, "ledger")

USERS = 4
TASKS = 20
WINDOW = 2
LATENCY_P95_MS = 250.0


def spec_for(rule: str, meta: dict) -> dict:
    total = meta["total_lines"]
    lc = meta["line_counts"]
    if rule == "header":
        return {"files": 4, "rows": total - 1, "hash_mode": "content"}
    if rule == "exclude":
        return {"files": 3,
                "rows": total - lc[meta["files"].index(EXCLUDED_FILE)],
                "hash_mode": "content"}
    if rule == "name_hash":
        return {"files": 4, "rows": total, "hash_mode": "content+names"}
    return {"files": 4, "rows": total - DUPLICATE_ROWS, "hash_mode": "content"}


def naive_spec(meta: dict) -> dict:
    return {"files": len(meta["files"]), "rows": meta["total_lines"],
            "hash_mode": "content"}


def parse_convention(text: str):
    m = re.search(r"convention=([a-z_]+)", text)
    return m.group(1) if m and m.group(1) in RULES else None


def embed(texts):
    return _model().encode(list(texts), batch_size=32, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)


_M = [None]


def _model():
    if _M[0] is None:
        from sentence_transformers import SentenceTransformer
        _M[0] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _M[0]


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    env = Layer3Env(os.path.join(OUTDIR, "world"), seed=23)
    ledger = ICMLedger(LEDGER_DB)
    store = EpisodicStore(ledger, embed, dim=384)
    try:
        # ---- MEMORY arm
        mem_ok, lat = [], []
        for u in range(USERS):
            rule = RULES[u % len(RULES)]
            session = f"user-{u}"
            profile = (f"USER PROFILE {session}: verification convention={rule}. "
                       f"Apply it to every dataset from this user.")
            store.capture(session, None, profile)
            for e in range(TASKS):
                sid = f"mem-u{u}-e{e:02d}"
                meta = env.materialize(sid, rule)
                t0 = time.perf_counter()
                recalled = store.recall(f"{session} verification convention", k=5)
                lat.append((time.perf_counter() - t0) * 1000.0)
                conv = None
                for rec, _sim in recalled:
                    if rec.get("session_id") != session:
                        continue
                    conv = parse_convention(rec.get("text", ""))
                    if conv:
                        break
                spec = spec_for(conv, meta) if conv else naive_spec(meta)
                check = env.check(sid, spec)
                mem_ok.append(1.0 if check["ok"] else 0.0)
                store.capture(session, None,
                              f"{session} task {e}: convention={conv or 'none'}; "
                              f"spec={json.dumps(spec)}; "
                              f"outcome={'ok' if check['ok'] else 'fail'}")
    finally:
        ledger.close()

    # ---- CONTEXT-ONLY arm (no store)
    ctx_ok, window_ok = [], []
    for u in range(USERS):
        rule = RULES[u % len(RULES)]
        session = f"user-{u}"
        profile = (f"USER PROFILE {session}: verification convention={rule}. "
                   f"Apply it to every dataset from this user.")
        history = [profile]
        for e in range(TASKS):
            sid = f"ctx-u{u}-e{e:02d}"
            meta = env.materialize(sid, rule)
            task = f"Task {e} for {session}: produce a manifest for {sid}."
            window = history[-WINDOW:] + [task]
            window_ok.append(not any("USER PROFILE" in m for m in window[1:])
                             if e >= 1 else True)
            conv = parse_convention("\n".join(window))
            spec = spec_for(conv, meta) if conv else naive_spec(meta)
            check = env.check(sid, spec)
            ctx_ok.append(1.0 if check["ok"] else 0.0)
            history = history[-WINDOW:] + [
                task, f"outcome: {'ok' if check['ok'] else 'verification failed'}"]
            history = history[-WINDOW:]

    mem_rate = round(statistics.mean(mem_ok), 3)
    ctx_rate = round(statistics.mean(ctx_ok), 3)
    lat_sorted = sorted(lat)
    p95 = round(lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))], 2)
    checks = {
        "power_80_per_arm": len(mem_ok) == USERS * TASKS and len(ctx_ok) == USERS * TASKS,
        "memory_ge_0.90": mem_rate >= 0.90,
        "context_le_0.20": ctx_rate <= 0.20,
        "lift_ge_0.50": mem_rate - ctx_rate >= 0.50,
        "latency_p95_le_250ms": p95 <= LATENCY_P95_MS,
        "profile_out_of_window": all(window_ok),
    }
    res = {
        "users": USERS, "tasks_per_user": TASKS, "window": WINDOW,
        "memory_success": mem_rate, "context_success": ctx_rate,
        "lift": round(mem_rate - ctx_rate, 3),
        "recall_latency_ms": {"mean": round(statistics.mean(lat), 2),
                              "p95": p95, "max": round(max(lat), 2)},
        "checks": checks, "pass": all(checks.values()),
        "verdict": ("PASS -- episodic recall carries an early user convention "
                    "across the context horizon" if all(checks.values()) else
                    "FAIL -- see the failing row"),
        "scope": "prepared world; context-window simulation; deterministic agent",
    }
    with io.open(RESULTS, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)
    print(f"[done] -> {RESULTS}", flush=True)
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
