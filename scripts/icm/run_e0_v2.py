"""E0-v2 runner -- the DECISIVE regime (content > prompt), confuser twins,
relevance-aware arms. Complements the critique's ledger:

    decisive cell = old-region (outside the stuffed window):
        M must succeed where C2 structurally cannot
    sanity cell = recent-region (inside the window): tie by construction
    relevance readout = M-arm consult top-1 correctness (the gold session's
    episode ranked FIRST) + per-error classification (recall miss vs answerer miss)

Arms: M (consult k=3) / C2 (recent-window stuffing ~18k chars) / C1 (fresh).
All relay calls retry once. Grading = exact gold-token containment.
Rows (one per question) are accumulated then scored -- no in-loop sketches.

Run:  python scripts/icm/run_e0_v2.py  (~180 relay calls, ~25 min)
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

from e0_v2_corpus import (  # noqa: E402
    N_SESSIONS,
    STUFF_RECENT,
    corpus_seed,
    episode_text,
    question_for,
    session_texts,
)
from icm.consult import consult, consult_block  # noqa: E402
from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

N_QUESTIONS = 60


def ask_retry(policy, question: str, memory_block: str | None) -> tuple[str, float]:
    prompt = (question +
              '\n\nReply ONLY as JSON {"answer": "<a short phrase>"}. '
              "The answer MUST contain the specific value rather than a general statement.")
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
    attempts = 0
    backoff = 10.0
    while attempts < 5:
        attempts += 1
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
            rec = json.loads(resp["message"]["content"])
            return str(rec.get("answer", "")).strip().lower(), (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # relay throttles bursts -- back off generously, keep trying
                wait = backoff
                backoff = min(backoff * 1.8, 90.0)
                print(f"[429] waiting {wait:.0f}s (attempt {attempts})", flush=True)
                time.sleep(wait)
                continue
            if attempts >= 3:
                print(f"[ask failed] {type(e).__name__}: {str(e)[:80]}", flush=True)
                return "", (time.perf_counter() - t0) * 1000
            time.sleep(2.0)
        except Exception as e:
            if attempts >= 3:
                print(f"[ask failed] {type(e).__name__}: {str(e)[:80]}", flush=True)
                return "", (time.perf_counter() - t0) * 1000
            time.sleep(2.0)
    return "", 0.0


_model = None


def _embedder(texts: list[str]) -> list[list[float]]:
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
                    default=os.path.join("data", "_scratch", "icm", "e0v2"))
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    args = ap.parse_args()
    policy = DeepSeekAgentPolicy(model=args.model)
    shutil.rmtree(args.db_path, ignore_errors=True)

    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    values = corpus_seed(7)
    try:
        t0 = time.time()
        nl = chr(10)
        for si in range(N_SESSIONS):
            lines = session_texts(si, values)
            store.capture(session_id=f"sess-{si:04d}", intent_id=None,
                          text=episode_text(si, values, 0) + nl +
                               nl.join(lines),
                          episode_id=f"ep-{si:04d}-0")
        print(f"[seed] {N_SESSIONS} session episodes in {time.time()-t0:.1f}s", flush=True)

        # recent-window stuffing from the corpus END (most recent = highest si)
        stuff_lines: list[str] = []
        si = N_SESSIONS - 1
        used = 0
        while si >= 0 and used < STUFF_RECENT:
            for line in session_texts(si, values):
                t = f"[session {si:04d}] " + line
                if used + len(t) > STUFF_RECENT:
                    break
                stuff_lines.append(t)
                used += len(t)
            si -= 1
        stuff_block = "\n".join(stuff_lines)
        stuff_window_lo = si + 1
        print(f"[stuff] window: sessions {stuff_window_lo}..{N_SESSIONS - 1}, "
              f"{len(stuff_lines)} lines, {used} chars", flush=True)

        rows = []
        step = N_SESSIONS // N_QUESTIONS
        samples = list(range(0, N_SESSIONS, step))[:N_QUESTIONS]
        for si in samples:
            regime = "recent" if si >= stuff_window_lo else "old"
            q, gold = question_for(si, values)
            payload = consult(store, q, k=3)
            gold_ep_prefix = f"ep-{si:04d}-"
            top1_hit = bool(payload["episodes"]) and payload["episodes"][0]["id"].startswith(gold_ep_prefix)
            in_top = any(e["id"].startswith(gold_ep_prefix) for e in payload["episodes"])
            m_ans, m_ms = ask_retry(policy, q, consult_block(payload))
            c2_ans, c2_ms = ask_retry(policy, q, "[recent work-log records]\n" + stuff_block)
            c1_ans, c1_ms = ask_retry(policy, q, None)
            rows.append({
                "session": si, "regime": regime, "gold": gold,
                "m": {"answer": m_ans, "ok": int(bool(gold) and gold in m_ans),
                      "top1": int(top1_hit), "in_top3": int(in_top), "latency_ms": round(m_ms, 1)},
                "c2": {"answer": c2_ans, "ok": int(bool(gold) and gold in c2_ans),
                       "latency_ms": round(c2_ms, 1)},
                "c1": {"answer": c1_ans, "ok": int(bool(gold) and gold in c1_ans),
                       "latency_ms": round(c1_ms, 1)},
            })
            print(f"[q {si:04d} {regime}] gold={gold} m={rows[-1]['m']['ok']} "
                  f"c2={rows[-1]['c2']['ok']} c1={rows[-1]['c1']['ok']}", flush=True)
    finally:
        led.close()

    def cell(arm: str, regime: str) -> dict:
        sel = [r for r in rows if r["regime"] == regime]
        n = len(sel)
        ok = sum(r[arm]["ok"] for r in sel)
        res = {"n": n, "success": ok, "rate": round(ok / max(1, n), 4)}
        if arm == "M":
            res["top1_relevance"] = round(sum(r["m"]["top1"] for r in sel) / max(1, n), 4)
            recalls_missed = sum(1 for r in sel if not r["m"]["ok"] and not r["m"]["in_top3"])
            res["failures"] = {"recall_miss": recalls_missed,
                               "answer_miss": max(0, n - ok - recalls_missed)}
        return res

    out = {
        "questions": len(rows),
        "stuffing_window": {"sessions": N_SESSIONS - stuff_window_lo,
                            "episodes": len(stuff_lines), "chars": used},
        "by_regime": {regime: {k: cell(k, regime) for k in ("m", "c2", "c1")} for regime in ("old", "recent")},
        "rows": rows,
    }
    path = os.path.join("data", "_scratch", "icm", "e0_v2_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())