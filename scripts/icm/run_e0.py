"""Full E0 -- the plan's Phase-1 evaluation, judge-free:

    M  (memory):        consult(memory, q) -> answer from recalled episodes
    C1 (compacted):     no memory (fresh context; history has been compacted away)
    C2 (context-stuff): all episode texts inlined into the prompt

The E0 claim decomposes into:
    memory task-success > compacted baseline            (C1 vs M)
    memory ~= stuffing on success but at bounded,
    deterministic context cost                          (C2 vs M)
plus the latency budget check on the MEMORY overhead (recall ms).

Scoring: exact fact-token containment ("42ms", "212", ...) -- the same
answer format is demanded of all arms; no judge model involved.
Corpus: the E0-lite work-log corpus (12 distinct facts x 5 session copies)
-- already verified to recall at rank 1 for these questions.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icm.consult import consult, consult_block  # noqa: E402
from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402
from e0_lite import QUERIES, WORK_LOGS  # noqa: E402  (sibling script, same dir)


# gold fact token per query (the distinctive number/word the answer must contain)
GOLD = {
    "deploy": "42ms",
    "deploy-1": "0.3",
    "db": "300",
    "db-1": "11 minutes",
    "bench": "88ms",
    "bench-1": "4.1k",
    "backup": "03:40",
    "backup-1": "94",
    "docker": "212",
    "docker-1": "7 minutes",
    "rotate": "24",
    "report": "three",
}
ANSWER_RULE = 'Reply ONLY as JSON {"answer": "<a short phrase>"}. The answer MUST contain the specific value (number/word) rather than a general statement.'


def ask(policy, question: str, memory_block: str | None) -> tuple[str, float]:
    prompt = question + "\n" + ANSWER_RULE
    if memory_block:
        prompt += "\n\nEvidence records:\n" + memory_block
    body = {
        "model": policy.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": {"type": "object", "properties": {"answer": {"type": "string"}},
                   "required": ["answer"]},
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request("http://localhost:11434/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
        rec = json.loads(resp["message"]["content"])
        return str(rec.get("answer", "")).strip().lower(), (time.perf_counter() - t0) * 1000
    except Exception:
        return "", (time.perf_counter() - t0) * 1000


def build_memory(store: EpisodicStore) -> None:
    episode_ids = []
    for i, (tag, text) in enumerate(WORK_LOGS):
        for v in range(5):
            episode_ids.append(store.capture(
                session_id=f"e0sess-{i}", intent_id=None,
                text=f"[session {i + 1}] " + text,
                episode_id=f"ep-{i:02d}-{v}"))
    return episode_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=str,
                    default=os.path.join("data", "_scratch", "icm", "e0"))
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    args = ap.parse_args()
    shutil.rmtree(args.db_path, ignore_errors=True)
    policy = DeepSeekAgentPolicy(model=args.model)

    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    episode_ids = build_memory(store)

    results = {"M": {"ok": 0, "lat": []}, "C1": {"ok": 0, "lat": []}, "C2": {"ok": 0, "lat": []}}
    consult_lat = []
    stuff_block = "\n".join(WORK_LOGS[i][1] for i in range(len(WORK_LOGS)))
    for tag, _text in WORK_LOGS:
        q = QUERIES[tag]
        gold = GOLD[tag].lower()
        payload = consult(store, q, k=3)
        consult_lat.append(payload["latency_ms"])
        block = consult_block(payload)
        arms = {
            "M": (q, block),
            "C1": (q, None),
            "C2": (q, "[all work-log records]\n" + stuff_block),
        }
        for arm, (question, mem_block) in arms.items():
            ans, ms = ask(policy, question, mem_block)
            results[arm]["ok"] += int(gold in ans)
            results[arm]["lat"].append(ms)
    led.close()

    n = len(WORK_LOGS)
    out = {
        "queries": n,
        "memory_vs_stuffing_vs_compacted": {
            arm: {"success": r["ok"], "rate": round(r["ok"] / n, 4),
                  "latency_ms_avg": round(sum(r["lat"]) / n, 1)}
            for arm, r in results.items()},
        "consult_latency_ms_avg": round(sum(consult_lat) / n, 2),
        "latency_budget_ms": 50,
        "memory_overhead_within_budget": bool(max(consult_lat) <= 50),
        "n_episodes": len(episode_ids),
    }
    path = os.path.join("data", "_scratch", "icm", "e0_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[done] wrote {path}")
    return 0


_model = None


def _embedder(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


if __name__ == "__main__":
    sys.exit(main())