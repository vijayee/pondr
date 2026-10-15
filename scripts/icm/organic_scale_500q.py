"""Organic scale-up -- 50 NEW questions from the 500-record LongMemEval file.

These questions are from longmemeval_interleaved_full.json (positions 50-99),
disjoint from the 40 already tested (which came from the 50-record slice).
Each question's sessions are captured as NEW episodes in a separate
EpisodicStore (fresh db_path, no contamination from the first run's store).

Protocol (same as the 40-q organic benchmark):
    4 arms (none / retrieval / organic distractor / gold ceiling), judge
    scoring (cross-model: kimi judges deepseek), n-gram secondary.

Then the labels-recovery pre-check at new-question scale:
    Spearman(cosine, judge-payoff) across each question's candidates.
    Pre-registered: ≥0.8 = similarity-shaped at scale (confirm the null);
    <0.5 = the labels carry beyond-similarity signal (the strong thesis armed).

Run:  python scripts/icm/organic_scale_500q.py  (~250 relay calls, ~60 min)
"""

from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402

from icm.deepseek_policy import DeepSeekAgentPolicy  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402
from icm.consult import consult  # noqa: E402
from payoff_organic_lme import grade  # noqa: E402

HARNESS = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness"
FULL500 = os.path.join(HARNESS, "datasets", "longmemeval", "longmemeval_interleaved_full.json")
ANSWERER = "deepseek-v4.1-flash:cloud"
JUDGE = "kimi-k2.7-code:cloud"  # cheap kimi tier; kimi-k3 retired 2026-10-14 (token cost)
OFFSET = None  # --offset arg; disjoint ranges per chained harvest


def ask_retry(policy, question, memory_block):
    prompt = question + "\nAnswer concisely, using the evidence below if provided."
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


def judge_help(policy_j, goal_q, candidate_text) -> float:
    q = ("JUDGE: would recalling this EVIDENCE help an agent answer the GOAL? "
         'Reply ONLY JSON {"helpful": true|false, "partial": 0.0-1.0}.\n'
         f"GOAL: {goal_q}\nEVIDENCE: {candidate_text}")
    out, _ = ask_retry(policy_j, q, None)
    try:
        rec = json.loads(out)
        v = float(rec.get("partial", 0.0))
        return max(0.0, min(1.0, v)) if rec.get("helpful") else v * 0.5
    except Exception:
        return 0.0


_model = None


def _embedder(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=50)
    ap.add_argument("--offset", type=int, default=50,
                    help="start position in the 500-record file (disjoint ranges per run)")
    ap.add_argument("--db-path", type=str, default=None)
    args = ap.parse_args()
    if args.db_path is None:
        args.db_path = os.path.join("data", "_scratch", "icm", f"organic_o{args.offset}")
    policy_ans = DeepSeekAgentPolicy(model=ANSWERER)
    policy_j = DeepSeekAgentPolicy(model=JUDGE)
    shutil.rmtree(args.db_path, ignore_errors=True)

    data = json.load(io.open(FULL500, encoding="utf-8"))
    OFFSET = args.offset
    qs = data[OFFSET: OFFSET + args.questions]
    print(f"[load] {len(qs)} new questions from the 500-record file (offset {OFFSET})", flush=True)

    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)

    # ---- capture sessions as episodes ----
    qids = []
    for qi, rec in enumerate(qs):
        qid = rec["question_id"]
        qids.append(qid)
        gold_sids = set(rec["answer_session_ids"])
        for si, sess in enumerate(rec["haystack_sessions"]):
            sid = rec["haystack_session_ids"][si]
            eid = f"lsess-{OFFSET + qi}-{sid}"
            gold_turns = [t.get("content", "") for t in sess if t.get("has_answer")]
            text = "\n".join(gold_turns) if gold_turns else "\n".join(
                t.get("content", "") for t in sess[-8:])
            store.capture(session_id=f"lme-{OFFSET + qi}", intent_id=None,
                          text=text, episode_id=eid)
    texts_by_id = {i: led.get("episode", i)["text"] for i in led.ids("episode")}
    print(f"[seed] {len(led.ids('episode'))} episodes captured", flush=True)

    # ---- 4-arm protocol + judge scoring + labels-recovery ----
    import difflib

    results = {"none": [], "retrieval": [], "distractor": [], "gold": []}
    spear_rows = []
    offset = OFFSET
    ckpt_path = os.path.join(os.path.dirname(args.db_path), f"organic500_ckpt_o{OFFSET}.jsonl")
    done_qi = set()
    if os.path.exists(ckpt_path):
        with io.open(ckpt_path, encoding="utf-8") as cf:
            for line in cf:
                line = line.strip()
                if line:
                    try:
                        prev = json.loads(line)
                        done_qi.add(prev["qi"])
                        for arm in ("none", "retrieval", "distractor", "gold"):
                            if arm in prev:
                                results[arm].append(prev[arm])
                        spear_rows.append({"qi": prev["qi"],
                                           "spearman": prev.get("spearman"),
                                           "cand_scores": prev.get("cand_scores", [])})
                    except Exception:
                        continue
        print(f"[resume] checkpoint has {len(done_qi)} done questions; skipping them", flush=True)
    try:
        for qi, rec in enumerate(qs):
            if OFFSET + qi in done_qi:
                continue
            q = rec["question"]
            gold_ans = rec["answer"]
            gold_sids = set(rec["answer_session_ids"])
            gold_ep_ids = [i for i in led.ids("episode") if
                           any(sid in i for sid in gold_sids)]
            gold_blocks = [led.get("episode", i)["text"] for i in gold_ep_ids
                           if led.get("episode", i)]
            gold_block = "\n".join(gold_blocks) if gold_blocks else None

            # consult
            payload = consult(store, q, k=3)
            retrieval_block = "\n".join(f"- {e['text']}" for e in payload["episodes"])

            # organic distractor: another question's gold session (highest question-similarity)
            others = [o for o in range(len(qs)) if o != qi]
            if others:
                best = max(others, key=lambda o: difflib.SequenceMatcher(
                    None, q.lower(), qs[o]["question"].lower()).ratio())
                d_qid = OFFSET + best
                d_blocks = [led.get("episode", i)["text"] for i in led.ids("episode")
                            if i.startswith(f"lsess-{d_qid}-") and
                            any(sid in i for sid in qs[best]["answer_session_ids"])]
                distractor_block = "\n".join(d_blocks) if d_blocks else None
            else:
                distractor_block = None

            arms = {
                "none": None,
                "retrieval": retrieval_block,
                "distractor": distractor_block,
                "gold": gold_block,
            }
            row = {"qi": OFFSET + qi, "type": rec["question_type"]}
            for arm, block in arms.items():
                ans, _ms = ask_retry(policy_ans, q, block)
                g = grade(ans, gold_ans)
                results[arm].append(round(g, 3))
                row[arm] = round(g, 3)
                time.sleep(3.0)

            # judge scoring for the labels-recovery pre-check
            cand_scores = []
            cand_ids = [e["id"] for e in payload["episodes"]][:5]
            for cid in cand_ids:
                j = judge_help(policy_j, q, texts_by_id.get(cid, ""))
                cand_scores.append((cid, j))
                time.sleep(2.0)
            spear_rows.append({"qi": OFFSET + qi, "cand_scores": cand_scores})

            # cosine for the same candidates
            if cand_ids:
                c_embs = np.asarray(_embedder([texts_by_id.get(c, "") for c in cand_ids]),
                                    dtype=np.float32)
                q_emb = np.asarray(_embedder([q]), dtype=np.float32)[0]
                cos_scores = [(c_e * q_emb).sum() for c_e in c_embs]
                # spearman between judge scores and cosine scores
                from scipy.stats import spearmanr
                if len(cand_scores) >= 3:
                    js = [s for _, s in cand_scores]
                    rho = spearmanr(js, cos_scores)
                    spear_rows[-1]["spearman"] = round(float(rho.statistic), 3) if hasattr(rho, "statistic") else round(float(rho[0]), 3)
            with io.open(ckpt_path, "a", encoding="utf-8") as cf:
                cf.write(json.dumps(row | {"spearman": spear_rows[-1].get("spearman"),
                                           "cand_scores": spear_rows[-1].get("cand_scores")}) + "\n")
            print(f"[q {OFFSET + qi}] none={row['none']} ret={row['retrieval']} "
                  f"dist={row['distractor']} gold={row['gold']}", flush=True)
    finally:
        led.close()

    # ---- aggregate ----
    def avg(arm):
        vals = results[arm]
        return round(sum(vals) / max(1, len(vals)), 4)

    spearmans = [r.get("spearman") for r in spear_rows if r.get("spearman") is not None]
    spear_mean = round(sum(spearmans) / max(1, len(spearmans)), 4) if spearmans else None

    out = {
        "questions": len(qs),
        "offset_in_500": OFFSET,
        "organic_payoff": {"none": avg("none"), "retrieval": avg("retrieval"),
                           "distractor": avg("distractor"), "gold_ceiling": avg("gold")},
        "labels_recovery": {"spearman_mean": spear_mean,
                            "spearman_per_q": spear_rows,
                            "pre_registered": "spearman >= 0.8 = similarity-shaped; "
                                              "< 0.5 = armed; between = ambiguous",
                            "verdict": ("SIMILARITY-SHAPED at scale"
                                        if spear_mean and spear_mean >= 0.8 else
                                        "BEYOND SIMILARITY (the strong thesis is armed)"
                                        if spear_mean and spear_mean < 0.5 else
                                        "AMBIGUOUS") if spear_mean is not None else
                                       "NOT MEASURED (insufficient candidates)"},
    }
    path = os.path.join("data", "_scratch", "icm",
                        f"organic_500q_results_o{OFFSET}.json")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    # also the per-row rows for the training corpus
    path2 = os.path.join("data", "_scratch", "icm",
                         f"organic_500q_rows_o{OFFSET}.json")
    with io.open(path2, "w", encoding="utf-8") as f:
        f.write(json.dumps(spear_rows, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_q"}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())