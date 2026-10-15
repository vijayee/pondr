"""OUT-OF-SAMPLE router confirmation + arm comparison on a new segment's verified tail.

The o100 tail's post-hoc dispatch (earliest -> time-rule, else -> index) was fit on
the SAME 25 questions it scored. This script applies arms to a DIFFERENT segment's
screened-only verified reversals and reports out-of-sample numbers:

  arms (k=1 selection, graded answers, no oracle):
    cos    -- consult-pool top-1 (the pool's cosine-first candidate)
    time   -- time-token-Jaccard top-1 within the pool
    index  -- temporal engine's full-store relational pick
    router -- op-dispatch: earliest -> time-pick, else -> index-pick  (fit on o100)
    gold   -- the question's gold sessions block (ceiling)

  readout: per-arm means; router vs each leg on discordant pairs + binomial
  two-sided; per-op-class split. NO fitting happens here -- the router's rule is
  frozen from the o100 result; this is the confirmation run.

Run:  python scripts/icm/router_oos_test.py --offset 300 --n 200 \
        --ckpt data/_scratch/icm/organic500_ckpt_o300.jsonl \
        --ledger data/_scratch/icm/organic_o300/ledger \
        --verif data/_scratch/icm/reversal_verification_o300.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from temporal_index_leg import engine_pick  # noqa: E402

FULL500 = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness" \
          r"\datasets\longmemeval\longmemeval_interleaved_full.json"

TIME_TOKENS = {"yesterday", "tomorrow", "today", "last", "next", "before",
               "after", "ago", "earlier", "later", "first", "second", "then",
               "week", "month", "year", "monday", "tuesday", "wednesday",
               "thursday", "friday", "saturday", "sunday", "spring", "summer",
               "fall", "autumn", "winter", "january", "february", "march",
               "april", "may", "june", "july", "august", "september",
               "october", "november", "december"}


def grade_f1(answer: str, gold: str) -> float:
    """Judge-free token F1 (soft 6-gram): catches paraphrase golds the
    6-gram containment grader floor-collapses (o300 finding: 'Twice.' vs
    "We've met up twice." graded 0.0)."""
    def toks(v):
        return {w for w in re.findall(r"[a-z0-9]+", str(v).lower()) if len(w) > 2}
    g, a = toks(gold), toks(answer)
    if not g:
        return 0.0
    inter = len(g & a)
    p = inter / max(1, len(a))
    r = inter / len(g)
    return round(2 * p * r / (p + r), 3) if p + r else 0.0


def tokset(text: str) -> set[str]:
    return {t.strip(".,!?;:'\"()").lower() for t in text.split()}


def time_score(q: str, text: str) -> float:
    qt, ct = tokset(q), tokset(text)
    inter = (qt & ct) & TIME_TOKENS
    union = (qt | ct) & TIME_TOKENS
    return len(inter) / max(1, len(union))


def parse_date(s: str):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s.strip())
    if not m:
        return None
    return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def ask_retry(question, memory_block):
    import time as _t
    import urllib.error
    import urllib.request
    prompt = question + "\nAnswer concisely, using the evidence below if provided."
    if memory_block:
        prompt += "\n\nEvidence records from memory:\n" + memory_block
    body = {"model": "deepseek-v4.1-flash:cloud",
            "messages": [{"role": "user", "content": prompt}],
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
                _t.sleep(min(backoff, 60)); backoff *= 1.8; continue
            if attempt >= 2:
                return ""
            _t.sleep(2.0)
        except Exception:
            if attempt >= 2:
                return ""
            _t.sleep(2.0)
    return ""


_model = None


def _embedder(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [np.asarray(v, dtype=np.float32) for v in vecs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--ledger", type=str, required=True)
    ap.add_argument("--verif", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_p = args.out or os.path.join("data", "_scratch", "icm",
                                     f"router_oos_o{args.offset}.json")
    ck_path = out_p.replace(".json", "_ckpt.jsonl")

    from payoff_organic_lme import grade
    from icm.ledger import ICMLedger

    ckpt_rows = {json.loads(l)["qi"]: json.loads(l)
                 for l in io.open(args.ckpt, encoding="utf-8") if l.strip()}
    verif = json.loads(io.open(args.verif, encoding="utf-8").read())
    data = json.load(io.open(FULL500, encoding="utf-8"))
    recs = {args.offset + i: rec for i, rec in enumerate(data[args.offset:args.offset + args.n])}

    led = ICMLedger(args.ledger)
    try:
        texts = {i: r["text"] for i in led.ids("episode") if (r := led.get("episode", i))}
    finally:
        led.close()

    rows = []
    done = set()
    if os.path.exists(ck_path):
        for l in io.open(ck_path, encoding="utf-8"):
            if l.strip():
                r_ = json.loads(l)
                rows.append(r_); done.add(r_["qi"])

    def safe(s):
        return str(s).encode("ascii", "replace").decode("ascii")

    targets = verif["verified_qids"]
    if args.limit:
        targets = targets[:args.limit]
    for qi in targets:
        if qi in done:
            continue
        rec = recs[qi]
        q = rec["question"]
        gold_ans = str(rec["answer"])
        cands = ckpt_rows[qi].get("cand_scores") or []
        pool = [(cid, texts.get(cid, "")) for cid, _ in cands if cid in texts]
        if not pool:
            continue
        # cos pick: the pool is consult order (cosine-first)
        cos_pick = pool[0]
        # time pick
        time_pick = max(pool, key=lambda pc: time_score(q, pc[1]))
        # index pick (full store)
        eps = []
        for si, sid in enumerate(rec["haystack_session_ids"]):
            eid = f"lsess-{qi}-{sid}"
            if eid in texts and si < len(rec["haystack_dates"]):
                eps.append({"sid": sid, "eid": eid, "text": texts[eid],
                            "date": parse_date(rec["haystack_dates"][si])})
        eps.sort(key=lambda e: e["date"] or _dt.date(9999, 1, 1))
        ipick, op, detail = engine_pick(q, eps)
        router_pick = time_pick[0] if op == "earliest" else ipick
        # gold block
        gold_sids = set(rec["answer_session_ids"])
        gold_block = "\n".join(f"- {t}" for i, t in texts.items()
                               if i.startswith(f"lsess-{qi}-") and
                               any(s in i for s in gold_sids))

        row = {"qi": qi, "op": op, "detail": safe(detail)}
        blocks = {"cos": f"- {cos_pick[1]}", "time": f"- {time_pick[1]}",
                  "router": None, "index": None, "gold": gold_block}
        picks = {"router": router_pick, "index": ipick}
        for arm in ("cos", "time", "router", "index", "gold"):
            if arm in ("router", "index"):
                if picks[arm] is None:
                    row[arm] = None
                    continue
                sid = picks[arm]
                blk = f"- {texts.get(f'lsess-{qi}-{sid}', '')}"
            else:
                blk = blocks[arm]
            ans = ask_retry(q, blk)
            if not ans.strip():
                row.setdefault("relay_fail", []).append(arm)
            row[arm] = round(grade(ans, gold_ans), 3)
            row[arm + "_f1"] = grade_f1(ans, gold_ans)
            row.setdefault("answers", {})[arm] = ans
            import time as _t2
            _t2.sleep(3.0)
        rows.append(row)
        with io.open(ck_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[q {qi}] {op}: cos={row['cos']} time={row['time']} "
              f"router={row['router']} index={row['index']} gold={row['gold']}",
              flush=True)

    live = [r_ for r_ in rows if r_.get("router") is not None]
    if not live:
        print("[readout] no live rows")
        return 1

    def mean(arm):
        v = [r_[arm] for r_ in live if r_.get(arm) is not None]
        return round(sum(v) / max(1, len(v)), 3), len(v)

    from scipy.stats import binomtest
    report = {"n_live": len(live), "means": {}, "means_f1": {}}
    print(f"\n=== OUT-OF-SAMPLE ROUTER TEST (n={len(live)}) ===")
    def meanf(arm, key):
        v = [r_[key] for r_ in live if r_.get(key) is not None]
        return round(sum(v) / max(1, len(v)), 3), len(v)
    for arm in ("cos", "time", "router", "index", "gold"):
        m_, n_ = mean(arm)
        report["means"][arm] = m_
        f1, _n2 = meanf(arm, arm + "_f1")
        report["means_f1"][arm] = f1
        print(f"  {arm:7s} 6gram: {m_:.3f} | f1: {f1:.3f} (n={n_})")
    for a, b in (("router", "time"), ("router", "index"), ("index", "time")):
        wa = sum(1 for r_ in live if r_[a] is not None and r_[b] is not None
                 and r_[a] > r_[b])
        wb = sum(1 for r_ in live if r_[a] is not None and r_[b] is not None
                 and r_[b] > r_[a])
        wfa = sum(1 for r_ in live if r_[a + "_f1"] is not None and r_[b + "_f1"] is not None and r_[a + "_f1"] > r_[b + "_f1"])
        wfb = sum(1 for r_ in live if r_[a + "_f1"] is not None and r_[b + "_f1"] is not None and r_[b + "_f1"] > r_[a + "_f1"])
        p = binomtest(wa, wa + wb, 0.5).pvalue if wa + wb else None
        pf = binomtest(wfa, wfa + wfb, 0.5).pvalue if wfa + wfb else None
        print(f"  {a} vs {b}: 6gram {wa}/{wb} p={p if p is None else round(p,4)} | f1 {wfa}/{wfb} p={pf if pf is None else round(pf,4)}")
        report[f"{a}_vs_{b}"] = {"a_6g": wa, "b_6g": wb, "p_6g": p, "a_f1": wfa, "b_f1": wfb, "p_f1": pf}
    with io.open(out_p, "w", encoding="utf-8") as f:
        f.write(json.dumps(report | {"rows": rows}, indent=2))
    print(f"[done] wrote {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())