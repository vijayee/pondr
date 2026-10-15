"""The temporal-index leg vs the trivial time-rule -- the one honest experiment left.

THE PRE-REGISTERED DESIGN (per the round-2 critique):
  Question: how much of the 0.400 (time-rule) -> 0.680 (gold) residue is
  indexable relational structure vs needs an actual reasoner?

  Opponent: a STRUCTURED TEMPORAL INDEX (timestamp arithmetic + lexical
  anchors), NOT the trivial token-overlap rule, NOT a learned head.
  The engine:
    - accesses the FULL episode store of the question (a relational query
      should not be capped by what cosine surfaced)
    - uses ONLY generic operators detected in the question text:
        total/aggregate:  "total | overall | altogether | combined | in all"
        relative-before:  "day/night/morning before X", "prior to X", "before X"
        relative-after:   "day after X", "after X"
        earliest/first:   "first | earliest | initial | originally"
        latest/recent:    "most recent | latest | last | currently | now" (default)
      No per-question handcoding, no per-question templates.
    - anchors events lexically (content tokens after the operator), does
      date arithmetic on dataset session timestamps (haystack_dates).
    - NO learning -- the engine is fixed a priori, so no train/test split
      is needed; zero memorization risk.

  Arms (k=1 selection, graded answers, no oracle):
    cos  -- reuse prior measured grades (payoff_tail_ab k=1 run)
    time -- reuse prior measured grades
    index -- NEW: engine's relational pick, answered + graded
    gold -- reuse prior measured grades (ceiling)

  Pre-registered readout: index vs time on discordant pairs, binomial
  two-sided; the pre-registered prediction (the critique's): the query
  engine wins. gold - time residue breakdown = the deliverable.

Run:  python scripts/icm/temporal_index_leg.py [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import re
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
PRIOR_K1 = os.path.join("data", "_scratch", "icm", "payoff_tail_ab_ckpt_k1.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "temporal_index_leg_results.json")

ANSWERER = "deepseek-v4.1-flash:cloud"

STOP = {"the", "a", "an", "my", "your", "their", "his", "her", "its", "of",
        "for", "to", "in", "on", "at", "and", "or", "was", "were", "is",
        "are", "did", "do", "does", "what", "which", "when", "where", "who",
        "how", "that", "this", "with", "about", "i", "you", "we", "they"}
OP_WORDS = {"day", "night", "morning", "before", "after", "prior", "following",
            "first", "earliest", "initial", "originally", "last", "latest",
            "most", "recent", "recently", "total", "overall", "altogether",
            "combined", "sum", "currently", "now", "today", "ago", "earlier",
            "later", "week", "month", "year"}


def tok(v: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", v.lower())]


def content_toks(v: str) -> list[str]:
    return [w for w in tok(v) if w not in STOP and w not in OP_WORDS and len(w) > 2]


def detect_op(question: str) -> tuple[str, str]:
    """(operator, event-phrase-tail); priority order, first match wins.
    Ops: before | after | total | earliest | latest."""
    q = question.lower()
    m = re.search(r"\b(?:day|night|morning)\s+before\b", q)
    if m:
        return "before", q[m.end():]
    m = re.search(r"\b(?:prior to|before)\b", q)
    if m:
        return "before", q[m.end():]
    m = re.search(r"\b(?:day )?after\b", q)
    if m:
        return "after", q[m.end():]
    if re.search(r"\b(total|overall|altogether|combined|in all)\b", q):
        return "total", ""
    if re.search(r"\b(first|earliest|initial|originally)\b", q):
        return "earliest", ""
    if re.search(r"\b(most recent|latest|last time|currently|now)\b", q):
        return "latest", ""
    if re.search(r"\blast\b(?!\s+(?:year|month|week|night|time))", q):
        return "latest", ""
    return "latest", ""


def parse_date(s: str):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s.strip())
    if not m:
        return None
    return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def anchor_score(q_tokens: list[str], text: str) -> float:
    t = set(tok(text))
    if not q_tokens or not t:
        return 0.0
    return sum(1 for w in q_tokens if w in t) / len(q_tokens)


def engine_pick(question: str, eps: list[dict]) -> tuple[str | None, str, str]:
    """eps: [{sid, eid, text, date: datetime.date}] sorted by date.
    Returns (sid | None, op, detail). Generic operators only."""
    op, tail = detect_op(question)
    q_content = content_toks(question)
    if not eps:
        return None, op, "no episodes"
    if any(e["date"] is None for e in eps):
        return None, op, "missing session dates"

    if op == "total":
        # k=1 cap: the strongest content-anchored session
        best = max(eps, key=lambda e: (anchor_score(q_content, e["text"]), e["date"]))
        return best["sid"], op, f"cap: strongest anchor of {len(eps)} sessions"

    if op in ("before", "after"):
        event_toks = content_toks(tail)[:6]
        if not event_toks:
            event_toks = q_content[:6]
        scored = [(anchor_score(event_toks, e["text"]), e) for e in eps]
        ev_scored = [(s, e) for s, e in scored if s > 0]
        if not ev_scored:
            return None, op, f"no anchored event for {event_toks}"
        event = max(ev_scored, key=lambda se: (se[0], se[1]["date"]))[1]
        if op == "before":
            prev = [e for e in eps if e["date"] < event["date"]]
            if not prev:
                return None, op, "no session before the event"
            target = max(prev, key=lambda e: e["date"])
        else:
            nxt = [e for e in eps if e["date"] > event["date"]]
            if not nxt:
                return None, op, "no session after the event"
            target = min(nxt, key=lambda e: e["date"])
        return target["sid"], op, (f"event={event['sid']}(score={max(s for s, _ in ev_scored):.2f}) "
                                   f"op={op}")

    anchored = [e for e in eps if anchor_score(q_content, e["text"]) > 0]
    pool = anchored or eps
    if op == "earliest":
        target = min(pool, key=lambda e: e["date"])
        detail = f"earliest of {len(pool)}/{len(eps)} anchored"
    else:  # latest (default)
        target = max(pool, key=lambda e: e["date"])
        detail = f"latest of {len(pool)}/{len(eps)} anchored"
    return target["sid"], op, detail


def ask_retry(question: str, memory_block: str) -> str:
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


def safe(s: str) -> str:
    return str(s).encode("ascii", "replace").decode("ascii")


def main() -> int:
    global ANSWERER
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", type=str, default=ANSWERER)
    args = ap.parse_args()
    ANSWERER = args.model

    corpus = json.load(io.open(CORPUS, encoding="utf-8"))
    data = json.load(io.open(FULL500, encoding="utf-8"))
    recs = {100 + i: rec for i, rec in enumerate(data[100:300])}

    prior = {}
    for l in io.open(PRIOR_K1, encoding="utf-8"):
        if l.strip():
            r_ = json.loads(l)
            prior[r_["qi"]] = r_

    from icm.ledger import ICMLedger
    led = ICMLedger(DB)
    try:
        texts = {i: r["text"] for i in led.ids("episode") if (r := led.get("episode", i))}
    finally:
        led.close()

    # ---- per-question index: episodes with session dates, sorted by date ----
    indexes = {}
    for qi, rec in recs.items():
        eps = []
        for si, sid in enumerate(rec["haystack_session_ids"]):
            eid = f"lsess-{qi}-{sid}"
            if eid not in texts or si >= len(rec["haystack_dates"]):
                continue
            eps.append({"sid": sid, "eid": eid, "text": texts[eid],
                        "date": parse_date(rec["haystack_dates"][si])})
        eps.sort(key=lambda e: e["date"] or _dt.date(9999, 1, 1))
        indexes[qi] = eps

    qs = corpus if not args.limit else corpus[:args.limit]

    ck_path = os.path.join("data", "_scratch", "icm", "temporal_index_leg_ckpt.jsonl")
    rows = []
    done = set()
    if os.path.exists(ck_path):
        for l in io.open(ck_path, encoding="utf-8"):
            if l.strip():
                r_ = json.loads(l)
                rows.append(r_); done.add(r_["qi"])

    for q in qs:
        qi = q["qi"]
        if qi in done or qi not in prior:
            continue
        eps = indexes.get(qi, [])
        picked_sid, op, detail = engine_pick(q["question"], eps)
        if picked_sid is None:
            row = {"qi": qi, "op": op, "index": None, "cos": prior[qi].get("cos"),
                   "time": prior[qi].get("time"), "gold": prior[qi].get("gold"),
                   "detail": detail}
            rows.append(row)
            with io.open(ck_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[q {qi}] {op}: ABSTAIN ({safe(detail)})", flush=True)
            continue
        text = texts[f"lsess-{qi}-{picked_sid}"]
        ans = ask_retry(q["question"], f"- {text}")
        row = {"qi": qi, "op": op, "picked": picked_sid,
               "index": round(grade(ans, str(recs[qi]["answer"])), 3),
               "cos": prior[qi].get("cos"), "time": prior[qi].get("time"),
               "gold": prior[qi].get("gold"), "answer": ans[:200], "detail": detail}
        rows.append(row)
        with io.open(ck_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[q {qi}] {op}: index={row['index']} cos={row['cos']} "
              f"time={row['time']} gold={row['gold']} | {safe(detail)}", flush=True)
        time.sleep(3.0)

    # ---- readout ----
    live = [r_ for r_ in rows if r_.get("index") is not None]
    abstained = len(rows) - len(live)
    if not live:
        print("[readout] no live rows")
        return 1
    m = {a: sum(r_[a] for r_ in live if r_.get(a) is not None) /
         sum(1 for r_ in live if r_.get(a) is not None)
         for a in ("index", "cos", "time", "gold")}
    wi = sum(1 for r_ in live if r_["index"] > (r_.get("time") or 0))
    wt = sum(1 for r_ in live if (r_.get("time") or 0) > r_["index"])
    p = None
    if wi + wt:
        from scipy.stats import binomtest
        p = binomtest(wi, wi + wt, 0.5).pvalue
    print(f"\n=== TEMPORAL INDEX vs TIME RULE (n={len(live)}, abstained {abstained}) ===")
    print(f"  index mean: {m.get('index', 0):.3f}")
    print(f"  cos  mean:  {m.get('cos', 0):.3f}  (prior k=1)")
    print(f"  time mean:  {m.get('time', 0):.3f}  (prior k=1)")
    print(f"  gold mean:  {m.get('gold', 0):.3f}  (prior, ceiling)")
    print(f"  discordant: index {wi} / time {wt} / binomial p={p}")
    verdict = ("INDEX WINS (residue indexable)" if wi > wt and p is not None and p < 0.05
               else "INDEX LOSES (residue needs a reasoner)" if wt > wi and p is not None and p < 0.05
               else "TIE/INCONCLUSIVE")
    print(f"  VERDICT: {verdict}")
    out = {"rows": rows, "means": m, "discordant": {"index": wi, "time": wt},
           "binomial_p": p, "verdict": verdict,
           "note": "abstains carry prior arm grades only; the k=1 budget holds "
                   "(1 episode per arm); cos/time/gold reused from payoff_tail_ab k=1"}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())