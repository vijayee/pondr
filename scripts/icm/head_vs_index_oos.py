"""THE POWER-FLAGGED HEAD TEST (claim #5) -- learned head vs strongest cheap leg, OOS.

GATES (per the second agent's addendum): (i) corpus 70-80 verified reversals
with F1-gradeable golds -- NOT MET (48 verified, user override to run now);
(ii) opponent = the strongest cheap leg ON THE TARGET SEGMENT, measured --
met: o300's strongest leg is the structured index (F1 0.272, dominant on all
three buckets). The test is therefore EXPLORATORY and power-flagged; a
non-decisive result is reported as underpowered, not as a verdict.

DESIGN -- no fitting on the test segment, ever:
  Train:  o100 verified tail ONLY (25 q, 75 labeled pairs from
          icm_head_corpus_o100.json). Label = kimi judge >= 0.5 (evidence
          helpfulness). Features (all computable on any segment):
            f1 cos                     -- question-candidate cosine (bge)
            f2 time_jaccard            -- relative-time token overlap
            f3 log_len                 -- log candidate length
            f4..f8 op one-hots         -- detect_op class of the question
            f9 num_sessions            -- episodes in the question's store
          z-normalized by TRAIN stats (the o100 4/25 unscaled-head bug's fix).
  Head:   logistic regression (numpy GD, plain, seeded).
  Test:   o300 verified cells (19 graded rows across rev/near0/rejudge buckets,
          answerer deepseek-v4.1-flash, F1 + 6-gram stored).
          For each test question: rank its FULL episode store by head score
          (same full-store access the index gets), take top-1, answer + grade
          (new calls). Paired comparison vs the stored index arm (and time arm)
          on identical cells; binomial two-sided on discordants.

  Pre-registered readings:
    head decisively > index OOS  -> the strong thesis has its first organic
                                    evidence; the segmenter/lens direction opens.
    head ~= index                -> the lens earns nothing where cheap legs are
                                    measured (strong-thesis reading unchanged).
    head < index                 -> the lens actively loses; record and stop.
Run:  python scripts/icm/head_vs_index_oos.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from temporal_index_leg import detect_op  # noqa: E402

CORPUS_O100 = os.path.join("data", "_scratch", "icm", "icm_head_corpus_o100.json")
O300_CKPTS = ["data/_scratch/icm/router_oos_o300_ckpt.jsonl",
              "data/_scratch/icm/router_oos_o300_nearzero_ckpt.jsonl",
              "data/_scratch/icm/router_oos_o300_rejudge_ckpt.jsonl"]
LEDGER_O300 = r"data\_scratch\icm\organic_o300\ledger"
FULL500 = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness" \
          r"\datasets\longmemeval\longmemeval_interleaved_full.json"
OUT = os.path.join("data", "_scratch", "icm", "head_vs_index_oos_results.json")
CK = os.path.join("data", "_scratch", "icm", "head_vs_index_oos_ckpt.jsonl")

ANSWERER = "deepseek-v4.1-flash:cloud"

TIME_TOKENS = {"yesterday", "tomorrow", "today", "last", "next", "before",
               "after", "ago", "earlier", "later", "first", "second", "then",
               "week", "month", "year", "monday", "tuesday", "wednesday",
               "thursday", "friday", "saturday", "sunday", "spring", "summer",
               "fall", "autumn", "winter", "january", "february", "march",
               "april", "may", "june", "july", "august", "september",
               "october", "november", "december"}
OPS = ("before", "after", "total", "earliest", "latest")


def tokset(text: str) -> set[str]:
    return {t.strip(".,!?;:'\"()").lower() for t in text.split()}


def time_score(q: str, text: str) -> float:
    qt, ct = tokset(q), tokset(text)
    inter = (qt & ct) & TIME_TOKENS
    union = (qt | ct) & TIME_TOKENS
    return len(inter) / max(1, len(union))


def op_onehot(q: str) -> list[float]:
    op, _tail = detect_op(q)
    return [1.0 if op == o else 0.0 for o in OPS]


_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def feat_row(q: str, text: str, cos: float, n_sessions: int) -> list[float]:
    return [cos, time_score(q, text), math.log1p(len(text)), *op_onehot(q),
            float(n_sessions)]


def ask_retry(question, memory_block):
    import urllib.error
    import urllib.request
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
            return str(json.loads(urllib.request.urlopen(req, timeout=240).read())
                       ["message"]["content"]).strip()
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


def f1(a, g):
    def toks(v):
        return {w for w in re.findall(r"[a-z0-9]+", str(v).lower()) if len(w) > 2}
    G, A = toks(g), toks(a)
    if not G:
        return 0.0
    i = len(G & A)
    P = i / max(1, len(A))
    R = i / len(G)
    return round(2 * P * R / (P + R), 3) if P + R else 0.0


def grade6(a, g):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from payoff_organic_lme import grade
    return grade(a, g)


def main() -> int:
    global ANSWERER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=ANSWERER)
    args = ap.parse_args()
    ANSWERER = args.model

    # ---------------- TRAIN on o100 labels only ----------------
    corpus = json.load(io.open(CORPUS_O100, encoding="utf-8"))
    Xtr, ytr = [], []
    for q in corpus:
        texts_c = [c["text"] for c in q["candidates"]]
        embs = _emb(texts_c)
        qv = _emb([q["question"]])[0]
        n = max(len(q["candidates"]), 1)
        Xq, yq = [], []
        for c, cv in zip(q["candidates"], embs):
            cos = float(np.dot(cv, qv))
            Xq.append(feat_row(q["question"], c["text"], cos, n))
            yq.append(1.0 if c["kimi_judge"] >= 0.5 else 0.0)
        Xtr.extend(Xq); ytr.extend(yq)
    Xtr = np.asarray(Xtr, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.float32)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (Xtr - mu) / sd

    # logistic GD (seeded, plain)
    w = np.zeros(Xz.shape[1], dtype=np.float32)
    b = 0.0
    lr = 0.3
    for _ in range(4000):
        z = Xz @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        d = (p - ytr) / len(ytr)
        w -= lr * (Xz.T @ d)
        b -= lr * float(d.sum())
    acc = float(((1.0 / (1.0 + np.exp(-(Xz @ w + b))) >= 0.5) == (ytr > 0.5)).mean())
    print(f"[train] o100 pairs={len(ytr)} pos_rate={ytr.mean():.2f} "
          f"in-sample acc={acc:.3f}", flush=True)

    # ---------------- load the o300 test cells (stored arms) ----------------
    test = {}
    for pth in O300_CKPTS:
        if not os.path.exists(pth):
            continue
        for l in io.open(pth, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("index_f1") is not None and r.get("router_f1") is not None:
                    test[r["qi"]] = r  # first occurrence wins (dedup)

    data = json.load(io.open(FULL500, encoding="utf-8"))
    recs = {300 + i: rec for i, rec in enumerate(data[300:500])}
    from icm.ledger import ICMLedger
    led = ICMLedger(LEDGER_O300)
    try:
        texts = {i: rr["text"] for i in led.ids("episode") if (rr := led.get("episode", i))}
    finally:
        led.close()

    rows = []
    done = set()
    if os.path.exists(CK):
        for l in io.open(CK, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                rows.append(r); done.add(r["qi"])

    def safe(s):
        return str(s).encode("ascii", "replace").decode("ascii")

    for qi, st in sorted(test.items()):
        if qi in done:
            continue
        rec = recs[qi]
        q = rec["question"]
        gold = str(rec["answer"])
        eps = []
        for sid in rec["haystack_session_ids"]:
            eid = f"lsess-{qi}-{sid}"
            if eid in texts:
                eps.append({"sid": sid, "text": texts[eid]})
        if not eps:
            continue
        cvecs = _emb([e["text"] for e in eps])
        qv = _emb([q])[0]
        n = len(eps)
        X = np.asarray([feat_row(q, e["text"], float(np.dot(cv, qv)), n)
                        for e, cv in zip(eps, cvecs)], dtype=np.float32)
        scores = 1.0 / (1.0 + np.exp(-((X - mu) / sd) @ w + b))
        top = int(np.argmax(scores))
        picked_sid = eps[top]["sid"]
        ans = ask_retry(q, f"- {eps[top]['text']}")
        if not ans.strip():
            print(f"[q {qi}] RELAY-FAIL: empty answer; row kept with None", flush=True)
            row = {"qi": qi, "head": None, "relay_fail": True}
        else:
            row = {"qi": qi, "op": safe(detect_op(q)[0]), "picked": picked_sid,
                   "head": round(f1(ans, gold), 3), "head_6g": round(grade6(ans, gold), 3),
                   "index": st.get("index_f1"), "index_6g": st.get("index"),
                   "time": st.get("time_f1"), "gold": st.get("gold_f1"),
                   "answer": ans}
        rows.append(row)
        with io.open(CK, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[q {qi}] head={row.get('head')} index={st.get('index_f1')} "
              f"time={st.get('time_f1')} gold={st.get('gold_f1')}", flush=True)
        time.sleep(3.0)

    live = [r for r in rows if r.get("head") is not None]
    if not live:
        print("[readout] no live rows")
        return 1
    mh = sum(r["head"] for r in live) / len(live)
    mi = sum(r["index"] for r in live if r.get("index") is not None) / len(live)
    mt = sum(r["time"] for r in live if r.get("time") is not None) / len(live)
    mg = sum(r["gold"] for r in live if r.get("gold") is not None) / len(live)
    wi = sum(1 for r in live if r["head"] > (r.get("index") or 0))
    di = sum(1 for r in live if (r.get("index") or 0) > r["head"])
    wt = sum(1 for r in live if r["head"] > (r.get("time") or 0))
    dt = sum(1 for r in live if (r.get("time") or 0) > r["head"])
    pi = binomtest(wi, wi + di, 0.5).pvalue if wi + di else None
    pt = binomtest(wt, wt + dt, 0.5).pvalue if wt + dt else None
    rfails = sum(1 for r in rows if r.get("relay_fail"))
    print(f"\n=== HEAD vs STRONGEST CHEAP LEG (OOS, power-flagged; n={len(live)}, relay-fails {rfails}) ===")
    print(f"  head  F1: {mh:.3f}")
    print(f"  index F1: {mi:.3f}   (the o300 strongest cheap leg)")
    print(f"  time  F1: {mt:.3f}")
    print(f"  gold  F1: {mg:.3f}")
    print(f"  paired head vs index: {wi}/{di} p={pi if pi is None else round(pi, 4)}")
    print(f"  paired head vs time:  {wt}/{dt} p={pt if pt is None else round(pt, 4)}")
    verdict = ("head decisively wins the strongest leg OOS" if pi is not None and pi < 0.05 and wi > di
               else "head loses to the index" if pi is not None and pi < 0.05
               else "UNDERPOWERED / tie -- no verdict (the corpus gate was not met)")
    print(f"  VERDICT: {verdict}")
    out = {"train": {"pairs": len(ytr), "in_sample_acc": acc,
                     "gate_corpus": "NOT MET (48 verified; user override)"},
           "rows": rows, "n_live": len(live),
           "means": {"head": round(mh, 3), "index": round(mi, 3),
                     "time": round(mt, 3), "gold": round(mg, 3)},
           "paired": {"head_vs_index": [wi, di, None if pi is None else round(pi, 4)],
                      "head_vs_time": [wt, dt, None if pt is None else round(pt, 4)]},
           "verdict": verdict}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())