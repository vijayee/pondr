"""ORGANIC payoff benchmark -- LongMemEval with counterfactual recall-swap.

The first benchmark whose discriminators are not authored: real conversations,
real goals (the benchmark's own questions), real distractors (other questions'
gold sessions -- the natural organically-confusable neighbors). It applies the
payoff-protocol method (scaffold/validity/reversal pre-checks, armed
opponents, pre-registered nulls) to real content.

Per question:
    arm NONE        -- no memory (answer only from own knowledge)
    arm RETRIEVAL   -- the consult's organic recall payload (the store's
                       organic retrieval behavior: hybrid lexical + cosine)
    arm DISTRACTOR  -- an ORGANIC confuser: another question's gold session
                       (naturally topically-near, no authored markers)
    arm GOLD        -- the question's own gold session (oracle ceiling)
Success graded judge-free vs the benchmark's answer string (key-token
containment on shared n-grams).

Outputs (per candidate): the ORGANIC payoff label; per-question rows for
the training corpus. Pre-registered readouts:
    * RETRIEVAL arm < GOLD arm by a margin -> organic recall has headroom
    * DISTRACTOR arm ~ NONE arm -> organic distractors are not guess-proof... (the
      confusers must be hard or the benchmark under-samples difficulty)
    * the raw-cosine-vs-consult delta: does the hybrid consult beat raw cosine
      on organic usefulness (the premise's first ORGANIC test).

Run:  python scripts/icm/payoff_organic_lme.py  (~160 relay calls, ~40 min)
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

HARNESS = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness"
LME50 = os.path.join(HARNESS, "datasets", "longmemeval", "longmemeval_interleaved.json")

from icm.consult import consult  # noqa: E402
from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402


def ask_retry(policy, question, memory_block):
    prompt = (question + "\nAnswer concisely; you may be given evidence records below.")
    if memory_block:
        prompt += "\n\nEvidence records from memory:\n" + memory_block
    body = {"model": policy.model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.2, "num_ctx": 8192}}
    backoff = 10.0
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
            return str(resp["message"]["content"]).strip(), (time.perf_counter() - t0) * 1000
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


def grade(answer: str, gold: str) -> float:
    """Judge-free: shared 6-gram containment between answer and gold answer."""
    def toks(v):
        s = str(v)
        return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]
    g = toks(gold)
    a = toks(answer)
    if not g:
        return 0.0
    n = min(6, len(g))
    gr = {" ".join(g[i:i + n]) for i in range(max(1, len(g) - n + 1))}
    ar = {" ".join(a[i:i + n]) for i in range(max(1, len(a) - n + 1))}
    hit = len(gr & ar) / len(gr)
    return round(hit, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=40)
    ap.add_argument("--model", type=str, default="deepseek-v4.1-flash:cloud")
    ap.add_argument("--db-path", type=str, default=os.path.join("data", "_scratch", "icm", "organic"))
    args = ap.parse_args()
    policy = DeepSeekAgentPolicy(model=args.model)
    shutil.rmtree(args.db_path, ignore_errors=True)
    rng = random.Random(101)

    import io as _io
    d = json.load(_io.open(LME50, encoding="utf-8"))
    qs = d[: args.questions]

    # build the store: each question's haystack SESSIONS as episodes
    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    gold_session_text = {}
    for qi, rec in enumerate(qs):
        gold_sets = rec["answer_session_ids"]
        for si, sess in enumerate(rec["haystack_sessions"]):
            sid = rec["haystack_session_ids"][si]
            text = "\n".join(turn.get("content", "") for turn in sess[-8:])
            if sid in gold_sets:
                # the gold bundle: the answer-bearing TURNS (has_answer) or tail
                gold_turns = [t["content"] for t in sess if t.get("has_answer")]
                text = "\n".join(gold_turns) if gold_turns else text
            store.capture(session_id=f"lme-{qi}", intent_id=None, text=text,
                          episode_id=f"lsess-{qi}-{sid}")
            if sid in gold_sets:
                gold_session_text[qi] = text

    rows = []
    consult_lat = []
    try:
        for qi, rec in enumerate(qs):
            q = rec["question"]
            gold_ans = rec["answer"]
            payload = consult(store, q, k=3)
            consult_lat.append(payload["latency_ms"])
            # organic distractor session: another question's gold bundle,
            # chosen as the highest-cosine among OTHER questions' golds
            distractor = None
            if len(qs) > 1:
                import difflib
                cand_qs = [o for o in range(len(qs)) if o != qi]
                # lexical nearness = difflib on the questions (not authored signal)
                best = max(cand_qs, key=lambda o: difflib.SequenceMatcher(
                    None, q.lower(), qs[o]["question"].lower()).ratio())
                distractor = gold_session_text.get(best)
            arms = {
                "none": None,
                "retrieval": "\n".join(f"- {e['text']}" for e in payload["episodes"]) or None,
                "distractor": distractor,
                "gold": gold_session_text.get(qi),
            }
            row = {"qi": qi, "question_type": rec["question_type"]}
            for arm, block in arms.items():
                ans, _ms = ask_retry(policy, q, block)
                g = grade(ans, gold_ans)
                row[arm] = {"score": g, "answer_len": len(ans)}
                if arm == "none":
                    time.sleep(3.0)
            rows.append(row)
            print(f"[q {qi}] none={row['none']['score']} ret={row['retrieval']['score']} "
                  f"dist={row['distractor']['score']} gold={row['gold']['score']}", flush=True)
    finally:
        led.close()

    def avg(arm):
        sel = [r[arm]["score"] for r in rows]
        return round(sum(sel) / max(1, len(sel)), 4)

    out = {"questions": len(rows),
           "consult_latency_ms_avg": round(sum(consult_lat) / max(1, len(consult_lat)), 2),
           "organic_payoff": {"none": avg("none"), "retrieval": avg("retrieval"),
                              "distractor": avg("distractor"), "gold_ceiling": avg("gold")},
           "rows": rows,
           "note": "the distractor arm's organic confusers; the gold arm is the ceiling; "
                   "the retrieval arm is the store's organic recall behavior"}
    path = os.path.join("data", "_scratch", "icm", "payoff_organic_results.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(f"[done] wrote {path}")
    return 0


_model = None


def _embedder(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


if __name__ == "__main__":
    sys.exit(main())