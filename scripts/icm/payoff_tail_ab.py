"""Payoff A/B on the verified reversal tail -- the decisive instrument.

The 2-judge screen verified 25 reversal questions (cosine ranks the WRONG
candidate first; the judge-favored one is verified cross-model). The LOO head
then showed the trivial time-token-overlap rule ranks the judge-favored
candidate first in 20/25 -- but JUDGE labels can be time-biased themselves.

THIS script bypasses the judge entirely: each arm gets its ranked retrieval
block, the answerer answers, the answer is GRADED against the gold answer.
Mechanical, no oracle.

  Arms:  cos   -- 3 candidates in cosine order (the dense leg's rank)
         time  -- 3 candidates in time-token-Jaccard order (the trivial rule)
         gold  -- the question's gold-session episodes (ceiling)
  Pre-registered: per-question win = grade(time) > grade(cos); McNemar
  two-sided on the discordant pairs; report means per arm.

Run:  python scripts/icm/payoff_tail_ab.py   (~75 relay calls, ~20 min)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

from payoff_organic_lme import grade  # noqa: E402

CORPUS = os.path.join("data", "_scratch", "icm", "icm_head_corpus_o100.json")
FULL500 = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness" \
          r"\datasets\longmemeval\longmemeval_interleaved_full.json"
DB = os.path.join("data", "_scratch", "icm", "organic_o100", "ledger")
OUT = os.path.join("data", "_scratch", "icm", "payoff_tail_ab_results.json")

ANSWERER = "deepseek-v4.1-flash:cloud"

TIME_TOKENS = {"yesterday", "tomorrow", "today", "last", "next", "before",
               "after", "ago", "earlier", "later", "first", "second", "then",
               "week", "month", "year", "monday", "tuesday", "wednesday",
               "thursday", "friday", "saturday", "sunday", "spring", "summer",
               "fall", "autumn", "winter", "january", "february", "march",
               "april", "may", "june", "july", "august", "september",
               "october", "november", "december"}


def tokset(text: str) -> set[str]:
    return {t.strip(".,!?;:'\"()").lower() for t in text.split()}


def time_score(q: str, text: str) -> float:
    qt, ct = tokset(q), tokset(text)
    inter = (qt & ct) & TIME_TOKENS
    union = (qt | ct) & TIME_TOKENS
    return len(inter) / max(1, len(union))


def ask_retry(question, memory_block):
    prompt = question + "\nAnswer concisely, using the evidence below if provided."
    if memory_block:
        prompt += "\n\nEvidence records from memory:\n" + memory_block
    body = {"model": ANSWERER, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.2, "num_ctx": 8192}}
    backoff = 10.0
    for attempt in range(4):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
            return str(resp["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
            if attempt >= 2:
                return ""
            time.sleep(2.0)
        except Exception:
            if attempt >= 2:
                return ""
            time.sleep(2.0)
    return ""


def main() -> int:
    global ANSWERER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=ANSWERER)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N questions")
    ap.add_argument("--topk", type=int, default=3,
                    help="candidates per retrieval block; ranks only matter "
                         "through selection, so 1 is the discriminating setting")
    ap.add_argument("--skip-gold", action="store_true",
                    help="inherit gold grades from the topk=3 checkpoint instead of re-asking")
    args = ap.parse_args()
    ANSWERER = args.model

    corpus = json.load(io.open(CORPUS, encoding="utf-8"))
    recs = {100 + i: rec for i, rec in enumerate(json.load(io.open(FULL500, encoding="utf-8"))[100:300])}

    from icm.ledger import ICMLedger
    led = ICMLedger(DB)
    try:
        texts = {i: r["text"] for i in led.ids("episode") if (r := led.get("episode", i))}
    finally:
        led.close()

    # significance test: two-sided binomial on the discordant pairs (the
    # McNemar equivalent; scipy has no McNemar function)
    wins_all = wins_t + wins_c
    p = binomtest(wins_t, wins_all, 0.5).pvalue if wins_all else None

    rows = []
    ck_path = os.path.join("data", "_scratch", "icm", f"payoff_tail_ab_ckpt_k{args.topk}.jsonl")
    OUT = os.path.join("data", "_scratch", "icm", f"payoff_tail_ab_results_k{args.topk}.json")
    done = set()
    if os.path.exists(ck_path):
        for l in io.open(ck_path, encoding="utf-8"):
            if l.strip():
                r_ = json.loads(l)
                rows.append(r_); done.add(r_["qi"])
    gold_prior = {}
    if args.skip_gold and os.path.exists(os.path.join("data", "_scratch", "icm", "payoff_tail_ab_ckpt.jsonl")):
        for l in io.open(os.path.join("data", "_scratch", "icm", "payoff_tail_ab_ckpt.jsonl"), encoding="utf-8"):
            if l.strip():
                r_ = json.loads(l)
                if r_["qi"] not in done:
                    gold_prior[r_["qi"]] = r_.get("gold")
    qs = corpus if not args.limit else corpus[:args.limit]

    for q in qs:
        if q["qi"] in done:
            continue
        rec = recs[q["qi"]]
        gold_ans = rec["answer"]
        gold_ans = str(gold_ans)
        cands = q["candidates"]
        order_cos = sorted(cands, key=lambda c: -c["cos"])
        order_tim = sorted(cands, key=lambda c: -time_score(q["question"], c["text"]))
        # gold block from the dataset's answer sessions
        gold_sids = set(rec["answer_session_ids"])
        gold_texts = [t for i, t in texts.items()
                      if i.startswith(f"lsess-{q['qi']}-") and
                      any(sid in i for sid in gold_sids)]

        ans_log = {}
        row = {"qi": q["qi"], "type": q["type"], "topk": args.topk}
        for arm, order in (("cos", order_cos), ("time", order_tim)):
            block = "\n".join(f"- {c['text']}" for c in order[:args.topk])
            ans = ask_retry(q["question"], block)
            row[arm] = round(grade(ans, gold_ans), 3)
            ans_log[arm] = ans[:200]
            time.sleep(3.0)
        if gold_texts and not args.skip_gold:
            block = "\n".join(f"- {t}" for t in sorted(set(gold_texts)))
            ans = ask_retry(q["question"], block)
            row["gold"] = round(grade(ans, gold_ans), 3)
            ans_log["gold"] = ans[:200]
            time.sleep(3.0)
        elif q["qi"] in gold_prior:
            row["gold"] = gold_prior[q["qi"]]
        else:
            row["gold"] = None
        row["answers"] = ans_log

        rows.append(row)
        with io.open(ck_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[q {q['qi']}] cos={row['cos']} time={row['time']} gold={row['gold']}", flush=True)

    # ---- aggregate ----
    from scipy.stats import binomtest
    n = len(rows)
    m = {a: sum((r_.get(a) or 0) for r_ in rows) / max(1, n) for a in ("cos", "time", "gold")}
    wins_t = sum(1 for r_ in rows if r_["time"] > r_["cos"])
    wins_c = sum(1 for r_ in rows if r_["cos"] > r_["time"])
    ties = n - wins_t - wins_c
    p = None

    out = {"n": n, "model": ANSWERER,
           "means": {k: round(v, 4) for k, v in m.items()},
           "time_beats_cos": wins_t, "cos_beats_time": wins_c, "ties": ties,
           "binomial_p_two_sided": round(p, 4) if p is not None else None,
           "rows": rows}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(f"\n=== PAYOFF A/B (tail, n={n}) ===")
    print(f"  cos arm mean:  {m['cos']:.3f}")
    print(f"  time arm mean: {m['time']:.3f}")
    print(f"  gold ceiling:  {m['gold']:.3f}")
    print(f"  wins: time {wins_t} / cos {wins_c} / ties {ties}; binomial p={p}")
    print(f"  VERDICT: {'TIME RANKING WINS IN ANSWER PAYOFF' if wins_t > wins_c and (p or 1) < 0.05 else 'NOT DECISIVE'}")
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())