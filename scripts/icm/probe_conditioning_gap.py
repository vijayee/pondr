"""ICM Probe B -- similarity-vs-usefulness gap: typology of similarity misses
and whether a cheap hand-set recency state term closes them.

UNTRACKED scratch -- never commit data outputs.

Questions (standalone):
  B1. The gold@5 misses under raw similarity -- WHY do they miss?
      abstention (no gold turn by design) / session-hit-turn-miss (near-miss) /
      near-dup shadow (a close cousin of a gold turn outranks it) /
      gold ranked low (top-10/top-20) / gold not in top-20.
  B2. Does score = cos + lam * freshness(turn, question) recover misses,
      per question_type? If a zero-training state term closes the gap,
      ternary conditioning is demonstrated cheaply; if not, the gap needs
      the learned intent lens (Phase 2).
  B3. Role prior sweep: the SAME conditioning signal (turn role) applied
      as a global constant vs switched per query by question_type. This is
      the ternary thesis in mechanical form -- only query-conditioned
      relevance is allowed to move all types at once.

NOTE: lambda/half-life sweep is on the SAME 50-q set (no train split) --
     tuning-on-test by construction. Feasibility probe only; honest eval
     needs the 500-q split.

Accounting: abstention records (no gold) count as MISSES in "gold@5_all"
(matching nothing earlier) and are excluded in "gold@5_active". Probe A's
0.80/0.88 baseline divided by non-empty-gold records only -- reconcile
against gold@5_active here.

Corpus: LongMemEval-50 interleaved, bge-small-en-v1.5 384-d. Embedding cache
at data/_scratch/icm/lme50_embeds.npz shared with probe_ternary_geometry.

Run:  python scripts/icm/probe_conditioning_gap.py
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from datetime import date as date_cls

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)

from probe_ternary_geometry import LME, cos_all, embed  # noqa: E402

CACHE = os.path.join("data", "_scratch", "icm", "lme50_embeds.npz")
HALF_LIVES = (30.0, 90.0, 180.0, 365.0)
LAMBDAS = (0.1, 0.25, 0.5, 1.0)
SHADOW_COS = 0.90


@dataclass
class Turn:
    text: str
    date: str
    freshable: bool
    role: str


def load_turns() -> tuple[list[Turn], list[dict], list[str], dict[str, set[int]]]:
    """Walk records once. Returns (turns, records, turn_qid, gold_abs).

    Turn dates inherit the session's date; turns of sessions with no date on
    record are marked not-freshable (freshness reads as the neutral 0.5).
    """
    d = json.load(io.open(LME, encoding="utf-8"))
    turns: list[Turn] = []
    turn_qid: list[str] = []
    gold_abs: dict[str, set[int]] = {}
    for rec in d:
        qid = rec["question_id"]
        dates: list[str] = rec.get("haystack_dates", [])
        for si, sess in enumerate(rec["haystack_sessions"]):
            date = dates[si].split(" ")[0] if si < len(dates) else ""
            fresh = bool(date)
            for turn in sess:
                if turn.get("has_answer"):
                    gold_abs.setdefault(qid, set()).add(len(turns))
                turns.append(Turn(f"{turn.get('role','?')}: {turn.get('content','')}".strip(), date, fresh,
                                  turn.get("role", "?")))
                turn_qid.append(qid)
    return turns, d, turn_qid, gold_abs


def ensure_embeds(texts: list[str]) -> np.ndarray:
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if z["texts"].tolist() == texts and z["X"].shape[0] == len(texts):
            return z["X"].astype(np.float32)
    X = embed(texts)
    np.savez(CACHE, texts=np.array(texts, dtype=object), X=X)
    return X


def parse_date(s: str) -> tuple:
    y, m, d = s.split(" ")[0].split("/")
    return int(y), int(m), int(d)


def days_between(a: str, b: str) -> float:
    """|question_date a - turn_date b| in days. Raises on unparseable dates."""
    qa, tb = parse_date(a), parse_date(b)
    return float(abs((date_cls(*tb) - date_cls(*qa)).days))


def freshness_vector(recs: list[dict], turn_qid: list[str], turns: list[Turn],
                     halflife: float) -> np.ndarray:
    """exp(-age/halflife) per turn, aged against its record's question_date.
    Turns with no usable date read as the neutral 0.5."""
    qdate = {r["question_id"]: r.get("question_date", "") for r in recs}
    out = np.zeros(len(turns), dtype=np.float32)
    for i, t in enumerate(turns):
        if not t.freshable or qdate.get(turn_qid[i], "") == "":
            out[i] = 0.5
            continue
        out[i] = float(np.exp(-days_between(qdate[turn_qid[i]], t.date) / halflife))
    return out


def eval_scores(sc: np.ndarray, recs: list[dict], gold_abs: dict[str, set[int]]) -> dict:
    """gold@5 over all records and over active (gold-bearing) records + per-type."""
    order = np.argsort(-sc, axis=1)
    per_type: dict[str, list[int]] = {}
    hits_all = hits_active = n_active = 0
    for qi, rec in enumerate(recs):
        gp = gold_abs.get(rec["question_id"], set())
        top5 = order[qi, :5].tolist()
        hit = 1 if set(top5) & gp else 0
        hits_all += hit
        if gp:
            hits_active += hit
            n_active += 1
        per_type.setdefault(rec["question_type"], [0, 0])
        per_type[rec["question_type"]][0] += hit
        per_type[rec["question_type"]][1] += 1
    return {
        "gold@5_all": round(hits_all / len(recs), 4),
        "gold@5_active": round(hits_active / max(1, n_active), 4),
        "per_type": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_type.items())},
    }


def main() -> None:
    t0 = time.time()
    turns, recs, turn_qid, gold_abs = load_turns()
    texts = [t.text for t in turns]
    X = ensure_embeds(texts)
    Xq = embed([r["question"] for r in recs])
    print(f"[load] {len(texts)} turns | cache {'hit' if os.path.exists(CACHE) else 'built'} "
          f"({time.time()-t0:.1f}s)", flush=True)

    S_full = cos_all(Xq, X)  # n_q x n_turns
    order0 = np.argsort(-S_full, axis=1)

    # Per-record ABSOLUTE session starts (records group contiguously in
    # turn order -- one global walk, not per-record local offsets).
    rec_bounds: list[tuple[int, list[int], list[int]]] = []  # (start, sess_starts_abs, sess_lens)
    i = 0
    for rec in recs:
        s = i
        starts, i2 = [], i
        for sess in rec["haystack_sessions"]:
            starts.append(i2)
            i2 += len(sess)
        rec_bounds.append((s, starts, [len(x) for x in rec["haystack_sessions"]]))
        i = i2
    assert i == len(turns), f"wrote {i} turns, have {len(turns)}"

    # ---- B1: typology of gold@5 misses ----
    typ = {"session_hit_turn_miss": 0, "near_dup_shadow": 0,
           "gold_in_top10": 0, "gold_in_top20": 0, "gold_not_in_top20": 0}
    miss_info: list[dict] = []
    abstention = 0
    session_hit = 0
    for qi, rec in enumerate(recs):
        qid = rec["question_id"]
        gp = gold_abs.get(qid, set())
        if not gp:
            abstention += 1
            continue
        rec_lo, sess_starts, sess_lens = rec_bounds[qi]
        rec_hi = rec_lo + sum(sess_lens)

        def sess_of(abs_i: int) -> int:
            for k in range(len(sess_starts) - 1, -1, -1):
                if abs_i >= sess_starts[k]:
                    return k
            return -1

        top5 = order0[qi, :5].tolist()

        gold_sess = {sess_of(g) for g in gp}
        top5_sess = {sess_of(t5) for t5 in top5 if rec_lo <= t5 < rec_hi}
        if gold_sess & top5_sess:
            session_hit += 1
        if set(top5) & gp:
            continue  # hit: no typology needed
        if gold_sess & top5_sess:
            typ["session_hit_turn_miss"] += 1
            miss_info.append({"qid": qid, "why": "session_hit_turn_miss"})
            continue
        # near-dup shadow: top turn close to ANY gold turn (max over golds)
        gmat = cos_all(X[np.array(sorted(gp))], X[top5])  # n_gold x 5
        if float(gmat.max()) >= SHADOW_COS:
            typ["near_dup_shadow"] += 1
            miss_info.append({"qid": qid, "why": "near_dup_shadow"})
            continue
        top10 = order0[qi, :10].tolist()
        top20 = order0[qi, :20].tolist()
        if set(top10) & gp:
            why = "gold_in_top10"
        elif set(top20) & gp:
            why = "gold_in_top20"
        else:
            why = "gold_not_in_top20"
        typ[why] += 1
        miss_info.append({"qid": qid, "why": why})

    # ---- B2: recency conditioning sweep ----
    results = {"FULL-384": eval_scores(S_full, recs, gold_abs)}
    for hl in HALF_LIVES:
        F = freshness_vector(recs, turn_qid, turns, hl)
        added = np.tile(F, (S_full.shape[0], 1))
        for lam in LAMBDAS:
            sc = S_full + lam * added
            results[f"hl={hl:g},lam={lam:g}"] = eval_scores(sc, recs, gold_abs)

    # ---- B3: role prior -- CONSTANT vs QUERY-CONDITIONAL ----
    # The 7 turn-level misses sit inside the right session (session_hit 0.94)
    # and gold turns are 87/96 user-role while the corpus is ~1:1 -- so a role
    # prior is the cheapest candidate conditioning. The question: does it have
    # to be switched per query (conditional) or can it be a global constant?
    # Conditional prior = "assistant" for single-session-assistant questions,
    # "user" otherwise (question_type read at query time; a soft oracle -- a
    # real system must INFER it, which is exactly Phase 2's intent head).
    row_of = {r["question_id"]: qi for qi, r in enumerate(recs)}
    P_c = np.zeros_like(S_full)
    for i, tqid in enumerate(turn_qid):
        P_c[row_of[tqid], i] = 1.0 if turns[i].role == "user" else 0.0
    pref = {r["question_id"]: ("assistant" if r["question_type"] == "single-session-assistant" else "user")
            for r in recs}
    P_q = np.zeros_like(S_full)
    for i, tqid in enumerate(turn_qid):
        P_q[row_of[tqid], i] = 1.0 if turns[i].role == pref[tqid] else 0.0
    role_results: dict[str, dict] = {}
    for name, P in (("constant-user", P_c), ("conditional-per-type", P_q)):
        for lam in (0.05, 0.1, 0.2, 0.4):
            role_results[f"{name},lam={lam:g}"] = eval_scores(S_full + lam * P, recs, gold_abs)
    gold_role_counts = {"user": 0, "assistant": 0}
    for gs in gold_abs.values():
        for g in gs:
            gold_role_counts[turns[g].role] += 1

    out = {
        "n_records": len(recs),
        "n_turns": len(turns),
        "gold_total": sum(len(g) for g in gold_abs.values()),
        "abstention_records": abstention,
        "misses_top5_active": len(miss_info),
        "typology_of_gold5_misses": typ,
        "session_level_hit_rate_active@5": round(session_hit / max(1, len(recs) - abstention), 4),
        "gold_turn_role_counts": gold_role_counts,
        "misses": miss_info,
        "conditions": results,
        "role_prior_conditions": role_results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = os.path.abspath(os.path.join("data", "_scratch", "icm", "probe_conditioning_gap.json"))
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[done] wrote {path}", flush=True)


if __name__ == "__main__":
    main()