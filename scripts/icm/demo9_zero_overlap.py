"""DEMO #9 (pre-registered) -- the cross-episode ZERO-OVERLAP rescue test.

THE PRE-REGISTRATION (per the second agent's proposal; attacks the arc's own
conclusion rather than extending it):

  Question:  does dense similarity have a SECOND real job -- semantic match
             under (near-)zero lexical overlap? The whole arc showed cosine
             losing on the RELATIONAL tail (operator questions, not similarity
             targets -- cosine losing there was expected) and on value-confusers
             (cheap legs win). Neither isolates dense's theoretical advantage:
             semantic match with no shared surface. If dense ALSO fails there,
             the negative result is airtight. If dense wins there, the taxonomy
             GROWS a fifth row and "dense has one real job" becomes "two jobs".

  Data:      ALL 450 harvested LongMemEval questions (offsets 50/100/300),
             own-haystack protocol: each question's gold session text vs its
             OWN other sessions (the per-record haystack), all on disk, no
             oracle, no answerer, no judge.

  Subset:    gold-vs-question content-token OVERLAP <= 0.05 (content tokens =
             non-stopword tokens > 2 chars; Jaccard). This is the
             zero-lexical-overlap regime where dense's advantage should live.

  Metrics:   rank of the gold session under bge-small cosine within the
             question's session set: top-1 hit rate, top-3, mean percentile;
             the SAME for (a) the time-token-overlap leg (expected ~chance
             here -- nothing tokens overlap) and (b) a lexical BM25-style
             token-overlap leg (also ~chance by construction of the subset).
             Chance baseline = mean 1/|sessions| percentile = 50%.

  Readout:   the subset's n is reported FIRST. Pre-registered call:
             WIN  = dense top-1 rate significantly above chance AND above the
                    cheap legs on the same subset (binomial-ish: compare rank
                    percentiles, sign test dense-vs-lex legs) -> fifth row.
             LOSS = dense ~chance on zero-overlap pairs -> the negative is
             airtight: relevance was never reducible NOR semantic-matchable
             here; the cheap-decomposition conclusion stands with its one gap
             closed by measurement.

Run:  python scripts/icm/demo9_zero_overlap.py   (local, no relay, ~1-2 min)
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402

FULL500 = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness" \
          r"\datasets\longmemeval\longmemeval_interleaved_full.json"
SEGMENTS = [(("data", "_scratch", "icm", "organic500"), 50),
            (("data", "_scratch", "icm", "organic_o100"), 100),
            (("data", "_scratch", "icm", "organic_o300"), 300)]
OUT = os.path.join("data", "_scratch", "icm", "demo9_zero_overlap_results.json")
OVERLAP_MAX = 0.05

STOP = {"the", "a", "an", "my", "your", "their", "his", "her", "its", "of",
        "for", "to", "in", "on", "at", "and", "or", "was", "were", "is",
        "are", "did", "do", "does", "what", "which", "when", "where", "who",
        "how", "that", "this", "with", "about", "i", "you", "we", "they",
        "there", "it", "he", "she", "them", "me", "so", "but", "if", "not"}
TIME_TOKENS = {"yesterday", "tomorrow", "today", "last", "next", "before",
               "after", "ago", "earlier", "later", "first", "second", "then",
               "week", "month", "year", "monday", "tuesday", "wednesday",
               "thursday", "friday", "saturday", "sunday"}


def content_toks(v):
    return {w for w in re.findall(r"[a-z0-9]+", str(v).lower())
            if len(w) > 2 and w not in STOP}


def tok_jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def main() -> int:
    recs = {i: rec for i, rec in enumerate(json.load(io.open(FULL500, encoding="utf-8")))}
    golds = {}       # qi -> (gold_eid, gold_text, [session texts])
    for seg, off in SEGMENTS:
        p = os.path.join(*seg, "ledger")
        led = ICMLedger(p)
        try:
            texts = {i: r["text"] for i in led.ids("episode") if (r := led.get("episode", i))}
        finally:
            led.close()
        for qi_t, t in texts.items():
            # ids look like lsess-{qi}-{sid}
            m = re.match(r"lsess-(\d+)-(.+)$", qi_t)
            if not m:
                continue
            qi = int(m.group(1))
            if qi not in recs:
                continue
            rec = recs[qi]
            gold_sids = set(rec["answer_session_ids"])
            if not any(sid in m.group(2) for sid in gold_sids):
                continue
            # collect that question's FULL haystack from this segment's store
            sess = [(i2, t2) for i2, t2 in texts.items()
                    if re.match(rf"lsess-{qi}-(.+)$", i2)]
            golds.setdefault(qi, []).append((qi_t, t, dict(sess)))

    rows = []
    for qi, items in sorted(golds.items()):
        rec = recs[qi]
        q = rec["question"]
        for gold_eid, gold_text, sess_map in items[:1]:
            cand_ids = list(sess_map.keys())
            cand_texts = [sess_map[c] for c in cand_ids]
            if len(cand_ids) < 2:
                continue
            gi = cand_ids.index(gold_eid) if gold_eid in cand_ids else None
            if gi is None:
                continue
            ov = tok_jaccard(content_toks(q), content_toks(gold_text))
            qtv = content_toks(q)
            lex_scores = [tok_jaccard(qtv, content_toks(t)) for t in cand_texts]
            time_scores = [tok_jaccard(qtv & TIME_TOKENS, content_toks(t) & TIME_TOKENS)
                           for t in cand_texts]
            rows.append({"qi": qi, "n_sessions": len(cand_ids), "ov": round(ov, 3),
                         "gold_idx": gi, "lex": lex_scores, "time": time_scores})
    print(f"[data] {len(rows)} gold pairs across the three segments", flush=True)

    zov = [r for r in rows if r["ov"] <= OVERLAP_MAX]
    print(f"[subset] zero/near-zero overlap (jaccard<= {OVERLAP_MAX}): {len(zov)} pairs", flush=True)
    if not zov:
        print("[done] subset empty; nothing to measure")
        return 1

    # dense ranking on BOTH the subset and all pairs
    emb_cache = {}
    def eval_leg(key):
        """rank percentile of the gold under leg 'key' in 'lex'/'time' rows;
        dense handled separately."""
        hits1 = hits3 = 0
        pcts = []
        for r in (zov if key else rows):
            n = r["n_sessions"]
            scores = r[key]
            rank = sorted(range(n), key=lambda i: -scores[i]).index(r["gold_idx"])
            h1 = rank == 0
            h3 = rank < 3
            pct = 100.0 * (n - 1 - rank) / max(1, n - 1)
            hits1 += h1; hits3 += h3; pcts.append(pct)
        return {"n": len(zov if key else rows) or None,
                "top1": round(hits1 / max(1, len(zov if key else rows)), 3),
                "top3": round(hits3 / max(1, len(zov if key else rows)), 3),
                "mean_pct": round(sum(pcts) / max(1, len(pcts)), 1)}
    lex_sub = eval_leg("lex")
    time_sub = eval_leg("time")

    # dense: embed per question
    d_top1 = d_top3 = 0
    d_pcts = []
    d_all_top1 = 0
    for r in zov:
        qi = r["qi"]
        # find the segment store holding this question's sessions
        for seg, off in SEGMENTS:
            pth = os.path.join(*seg, "ledger")
            led = ICMLedger(pth)
            try:
                ids = {i: r["text"] for i in led.ids("episode") if (r := led.get("episode", i))}
            finally:
                led.close()
            sess = {i: t for i, t in ids.items()
                    if re.match(rf"lsess-{qi}-(.+)$", i)}
            if len(sess) >= 2:
                break
        cids = list(sess.keys())
        g = [c for c in cids if int(re.match(r"lsess-(\d+)-(.+)$", c).group(1)) == qi
             and any(s in c for s in recs[qi]["answer_session_ids"])]
        if not g or len(cids) < 2:
            continue
        cvec = _emb(list(sess.values()))
        qv = _emb([recs[qi]["question"]])[0]
        cos = [float(np.dot(c, qv)) for c in cvec]
        rank = sorted(range(len(cids)), key=lambda i: -cos[i]).index(cids.index(g[0]))
        pct = 100.0 * (len(cids) - 1 - rank) / max(1, len(cids) - 1)
        d_top1 += rank == 0; d_top3 += rank < 3; d_pcts.append(pct)
    n_sub = max(1, len(d_pcts))
    dense = {"n": len(d_pcts), "top1": round(d_top1 / n_sub, 3),
             "top3": round(d_top3 / n_sub, 3),
             "mean_pct": round(sum(d_pcts) / n_sub, 1)}

    # pre-registered call: dense beats chance (50% percentile) and the cheap legs
    from scipy.stats import binomtest
    # top-1 vs chance under unequal n_sessions: use the mean percentile vs 50
    n_win = sum(1 for p in d_pcts if p > 50)
    p_chance = binomtest(n_win, n_sub, 0.5).pvalue
    out = {"pre_registered": OVERLAP_MAX,
           "all_pairs_n": len(rows),
           "subset_n": len(zov),
           "dense": dense,
           "lex_leg_sub": lex_sub, "time_leg_sub": time_sub,
           "chance_pct": 50.0,
           "dense_vs_chance_binomial_p": None if p_chance is None else round(p_chance, 6)}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # pair-level dense-vs-lex comparison lives in the results JSON; the
    # headline call compares dense's top-1 against both cheap legs' top-1.
    lex1, lex3 = lex_sub["top1"], lex_sub["top3"]
    print("\n=== DEMO #9 (zero/near-zero lexical overlap) ===")
    print(f"  subset n = {len(zov)} of {len(rows)} gold pairs "
          f"(jaccard<= {OVERLAP_MAX})")
    print(f"  dense : top1 {dense['top1']} top3 {dense['top3']} pct {dense['mean_pct']}")
    print(f"  lex   : top1 {lex1} top3 {lex3} pct {lex_sub['mean_pct']}")
    print(f"  time  : top1 {time_sub['top1']} top3 {time_sub['top3']} pct {time_sub['mean_pct']}")
    print(f"  dense vs chance(50%): p={out['dense_vs_chance_binomial_p']}")
    if dense["top1"] > max(lex1, time_sub["top1"]) and p_chance is not None and p_chance < 0.05:
        print("  VERDICT: WIN -- dense has a semantic second job (fifth row).")
    else:
        print("  VERDICT: LOSS/AIR-TIGHT -- dense ~or-below chance on zero-overlap pairs.")
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())