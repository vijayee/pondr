"""E0-v2b -- anchor-aware consult on the confuser corpus.

v2 finding: dense-only recall cannot discriminate among value twins (bge sees
300 near-identical templates as duplicates; values are invisible in embedding
space). The mechanical fix -- standard hybrid practice -- is to filter by the
question's EXACT anchor token (the ts), then rank survivors by cosine.

Arms on OLD-region questions only (the decisive cell):
    M-raw  : consult k=3 (dense-only)            [already measured: ~0]
    M-anch : recall k=50 -> keep episodes containing the anchor token -> top-3
Grading = exact gold containment. Reports top-1 relevance for M-anch too.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from e0_v2_corpus import N_SESSIONS, STUFF_RECENT, corpus_seed, question_for  # noqa: E402
from icm.consult import consult, consult_block  # noqa: E402
from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402


def ask_retry(policy, question, memory_block):
    prompt = (question +
              '\n\nReply ONLY as JSON {"answer": "<a short phrase>"}. '
              "The answer MUST contain the specific value rather than a general statement.")
    if memory_block:
        prompt += "\n\nEvidence records:\n" + memory_block
    body = {"model": policy.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"]},
            "options": {"temperature": 0.2, "num_ctx": 8192}}
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
            rec = json.loads(resp["message"]["content"])
            return str(rec.get("answer", "")).strip().lower(), (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(10 * (attempt + 1), 60))
                continue
            if attempt >= 1:
                return "", 0.0
        except Exception:
            if attempt >= 1:
                return "", 0.0
    return "", 0.0


_model = None


def _embedder(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


ANCHOR_RE = re.compile(r"backup completed at (\d{2}:\d{2})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=str,
                    default=os.path.join("data", "_scratch", "icm", "e0v2"))
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--questions", type=int, default=45)
    args = ap.parse_args()
    policy = DeepSeekAgentPolicy(model=args.model)

    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    values = corpus_seed(7)
    try:
        assert led.count("episode") == N_SESSIONS, "resume assumes seeded v2 db"
        step = N_SESSIONS // args.questions
        samples = [si for si in range(0, N_SESSIONS, step) if si < N_SESSIONS][
                        :args.questions]
        samples = [si for si in samples if "ep-" + f"{si:04d}-0" is not None]
        results = []
        for si in samples:
            if si >= 225:  # recent region -- already measured in v2
                continue
            q, gold = question_for(si, values)
            m_anchor = ANCHOR_RE.search(q)
            if not m_anchor:
                print(f"[skip {si}] no anchor in question: {q[:60]}", flush=True)
                continue
            anchor = m_anchor.group(1)
            payload = consult(store, q, k=50)
            gold_ep_prefix = f"ep-{si:04d}-"
            in_top50 = any(e["id"].startswith(gold_ep_prefix) for e in payload["episodes"])
            # anchor-filtered consult payload
            kept = [e for e in payload["episodes"] if anchor in e["text"]]
            filt = {"episodes": kept[:3], "latency_ms": payload["latency_ms"],
                    "question": q}
            top1 = bool(filt["episodes"]) and filt["episodes"][0]["id"].startswith(gold_ep_prefix)
            m_ans, m_ms = ask_retry(policy, q, consult_block(filt))
            ok = int(bool(gold) and gold in m_ans)
            results.append({"si": si, "gold": gold, "ok": ok, "top1": int(top1),
                            "in_top50": int(in_top50), "kept": len(filt["episodes"]),
                            "answer": m_ans})
            print(f"[{si:04d}] anchor={anchor} kept={len(kept)} ok={ok} "
                  f"(top1={int(top1)} in50={int(in_top50)})", flush=True)
            time.sleep(3.0)
    finally:
        led.close()

    n = len(results)
    out = {
        "questions": n,
        "old_region": {"m_anchored": {"success": sum(r["ok"] for r in results),
                                      "rate": round(sum(r["ok"] for r in results) / max(1, n), 4)},
                       "top1_relevance": round(sum(r["top1"] for r in results) / max(1, n), 4)},
        "recall_layer": {"in_top50_any": round(sum(r["in_top50"] for r in results) / max(1, n), 4),
                         "note": "in_top50 = the anchor-filtering premise: the gold episode "
                                 "must exist in the wide dense recall for lexical filtering to act"},
        "results": results,
    }
    path = os.path.join("data", "_scratch", "icm", "e0_v2b_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())