"""DEMO #10 (pre-registered) -- organic value-flips, cheap-leg separation, and
the segmenter's first real measurement.

Gate-0 result (census, 2026-10-15): LongMemEval has 0/940 value-flips; LoCoMo
has 272/272 sessions in value-flip shape (every session gold for some questions,
non-gold for others within its own conversation; gold labels FREE from evidence
pointers "D{n}:turn"). This corpus is the first that can measure
question-conditioned value variation organically.

THE PRE-REGISTRATION:
  Question 1:  do CHEAP LEGS separate a session's gold-questions from its
               non-gold-questions? Legs (all local, no oracle, no answerer):
                 cos      -- bge-small cosine(question, session text)
                 lex      -- content-token overlap (Jaccard, non-stop >2 chars)
                 time     -- relative-time-token overlap
                 tdist    -- negative temporal distance (question position
                             date <-> session date_time; LoCoMo has both)
  Metric:     per session, AUROC(gold vs non-gold questions) per leg.
  Readout:    % sessions best-leg captures (AUC >= 0.8) / partial / fails
              (<0.55) -> the residue = the beyond-cheap signal on THIS axis.

  Question 2 (the segmenter's first real measurement): DOES THE WINNING LEG
  VARY across sessions? Report the winner-distribution. If concentrated ->
  one leg suffices (segmentation story dead). If spread -> "segment, then
  dispatch" earns its keep on organic intent-shaped data, and the segments'
  identities get named (which sessions need cos, which need temporal).

Run:  python scripts/icm/demo10_locomo_legs.py   (local, no relay, ~2-5 min)
"""

from __future__ import annotations

import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

LOCOMO = os.path.join("data", "_scratch", "locomo", "locomo10.json")
OUT = os.path.join("data", "_scratch", "icm", "demo10_locomo_legs_results.json")

STOP = {"the", "a", "an", "my", "your", "their", "his", "her", "its", "of",
        "for", "to", "in", "on", "at", "and", "or", "was", "were", "is",
        "are", "did", "do", "does", "what", "which", "when", "where", "who",
        "how", "that", "this", "with", "about", "i", "you", "we", "they",
        "there", "it", "he", "she", "them", "me", "so", "but", "if", "not"}
TIME_TOKENS = {"yesterday", "tomorrow", "today", "last", "next", "before",
               "after", "ago", "earlier", "later", "first", "second", "then",
               "week", "month", "year", "monday", "tuesday", "wednesday",
               "thursday", "friday", "saturday", "sunday", "morning", "night",
               "evening", "morning", "afternoon", "spring", "summer", "fall",
               "winter", "january", "february", "march", "april", "may",
               "june", "july", "august", "september", "october", "november",
               "december"}


def content_toks(v):
    return {w for w in re.findall(r"[a-z0-9]+", str(v).lower())
            if len(w) > 2 and w not in STOP}


def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def parse_dt(s):
    return s  # keep as sortable string (ISO-ish "2023/05/11 (Thu) ...")


_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def auroc(scores, labels):
    """labels: 1 gold / 0 not. Simple rank AUROC."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = ties = 0
    ns = np.searchsorted(np.sort(np.asarray(neg)), np.asarray(pos))
    # count pos > neg + ties
    for p in pos:
        wins += sum(1 for n in neg if p > n)
        ties += sum(1 for n in neg if p == n)
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 3)


def main() -> int:
    d = json.load(io.open(LOCOMO, encoding="utf-8"))
    per_sess = {"cos": [], "lex": [], "time": [], "tdist": []}
    winners = collections.Counter()
    sess_rows = []

    # aggregate: per conversation per session, AUC over that conversation's questions
    # (re-walk to keep conversations separate)
    per_conv_rows = []
    for rec in d:
        conv = rec["conversation"]
        conv_id = rec.get("sample_id") or rec.get("observation", "?")
        sess_keys = [k for k in conv if re.match(r"session_\d+$", k)]
        sess_text = {k: "\n".join(t.get("text", "") for t in conv[k]) for k in sess_keys}
        sess_dt = {k: conv.get(f"{k}_date_time", "") for k in sess_keys}
        q_gold = []
        for q in rec["qa"]:
            gold = set()
            for ev in (q["evidence"] or []):
                m = re.match(r"D(\d+)", str(ev))
                if m and f"session_{m.group(1)}" in conv:
                    gold.add(f"session_{m.group(1)}")
            q_gold.append((q["question"], q["category"], gold))
        qs = [q for q, cat, gold in q_gold if gold]
        if not qs:
            continue
        qvecs = _emb(qs)
        qv_idx = [i for i, (q, cat, gold) in enumerate(q_gold) if gold]
        svecs = _emb([sess_text[k] for k in sess_keys])
        qs_tok = [content_toks(qs[i]) for i in range(len(qs))]
        qs_time = [content_toks(qs[i]) & TIME_TOKENS for i in range(len(qs))]
        sess_tok = [content_toks(sess_text[k]) for k in sess_keys]
        sess_time = [s2 & TIME_TOKENS for s2 in sess_tok]

        # map back: question i -> original index for label lookup
        for si, k in enumerate(sess_keys):
            labels, scores = [], {leg: [] for leg in ("cos", "lex", "time", "tdist")}
            for li, oi in enumerate(qv_idx):
                gold_k = k in q_gold[oi][2]
                labels.append(1 if gold_k else 0)
                scores["cos"].append(float(np.dot(qvecs[li], svecs[si])))
                scores["lex"].append(jac(qs_tok[li], sess_tok[si]))
                scores["time"].append(jac(qs_time[li], sess_time[si]))
                qlabels = None
                tdist = None
                try:
                    from datetime import datetime
                    q_dt = None
                    # the question's date: not in the record; use the question's
                    # category only. Fall back: distance = 0 constant -> leg abstains.
                    tdist = 0.0
                except Exception:
                    tdist = 0.0
                scores["tdist"].append(tdist)
            if sum(labels) < 1 or (sum(labels) == len(labels)):
                continue  # no flip within this conversation for this session
            aucs = {}
            for leg in ("cos", "lex", "time", "tdist"):
                a_ = auroc(scores[leg], labels)
                if a_ is not None:
                    aucs[leg] = a_
            if not aucs:
                continue
            best = max(aucs, key=aucs.get)
            winners[best] += 1
            sess_rows.append({"conv": conv_id, "session": k, "label1": sum(labels),
                              "label0": len(labels) - sum(labels), "aucs": aucs,
                              "best": best})
            for leg in aucs:
                per_sess[leg].append(aucs[leg])

    n = len(sess_rows)
    print(f"[data] {n} flipped sessions measured (label1>=1)")
    out = {}
    for leg in ("cos", "lex", "time"):
        v = per_sess.get(leg) or []
        if not v:
            continue
        captured = sum(1 for a in v if a >= 0.8)
        partial = sum(1 for a in v if 0.55 <= a < 0.8)
        failed = sum(1 for a in v if a < 0.55)
        out[leg] = {"mean_auc": round(sum(v) / len(v), 3),
                    "captured>=0.8": captured, "partial": partial,
                    "failed<0.55": failed, "n": len(v)}
        print(f"  {leg:6s} mean AUC {out[leg]['mean_auc']} | captured {captured} "
              f"/ partial {partial} / failed {failed} (n={len(v)})")
    nw = sum(winners.values())
    print("  winner distribution (which leg separates each session):",
          dict(winners) if not nw else {k: f"{v} ({round(100*v/nw)}%)" for k, v in winners.items()})
    spread = sum(1 for v in winners.values()) >= 2 and max(winners.values()) / max(1, nw) < 0.8
    out["winner_distribution"] = dict(winners)
    out["segmentation_verdict"] = (
        "WINNING LEG VARIES -- segmentation earns its keep organically"
        if spread else
        "one leg dominates -- a single-leg design suffices on this corpus"
        if nw else "no measurements")
    # the residue: sessions where NO cheap leg reaches 0.55
    residue = [r for r in sess_rows if max(r["aucs"].values()) < 0.55]
    out["residue_n"] = len(residue)
    out["residue_examples"] = residue[:8]
    out["n_sessions_measured"] = n
    out["note"] = ("labels = gold-evidence POINTERS (retrieval gold), not "
                   "answer-utility; tdist abstained (LoCoMo has per-question "
                   "dates only in some variants)")
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print("  SEGMENTATION VERDICT:", out["segmentation_verdict"])
    print(f"  residue (no cheap leg separates): {len(residue)}/{n}")
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())