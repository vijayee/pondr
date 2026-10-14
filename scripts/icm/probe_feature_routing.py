"""ICM Probe F -- does any cheap mechanical conditioning move the dominant types?

UNTRACKED scratch -- never commit data outputs.

The role prior (Probes B/D/E) is nearly exhausted: multi-session (67/251 eval)
and temporal-reasoning (67/251) barely move -- 52% of eval records. This probe
tests whether zero-training features computable at inference time move them:

  F1 session-aggregate smoothing: turns inherit cos(q, mean(turn embs of session)).
     Rationale: similarity knows WHICH session (94% session-hit at 50-q) but
     picks the wrong turn; aggregate match is a session-level prior.
  F2 lexical IDF overlap: sum(idf of shared query tokens)/sum(idf of all query
     tokens) per turn. Rationale: vector search misses needles (names/numbers);
     the repo's A2 RRF lesson, here as a conditioning feature.
  F3 type-conditioned routing: M1's inferred type picks feature+lambda per
     record. Configs chosen on the TUNE half only; eval untouched.

Protocol: stratified split (same SEED as probe D/E); sweep on tune half;
report eval gold@5 + per-type for baseline / F1 / F2 / F1+F2 / routed F3.
NOTE: all features are computed from the question + corpus only (no gold, no
benchmark labels at inference) EXCEPT the tune-half sweep protocol itself.

Run:  python scripts/icm/probe_feature_routing.py
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from collections import Counter

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)

from sklearn.linear_model import LogisticRegression

from probe_500q_split import FULL, load_full, ensure, embed, SEED  # noqa: E402

RANK = 5
LAMBDAS = (0.05, 0.1, 0.2, 0.4)
STOP = set("the a an and or of to in my me i is was were be been am are you your for on at with "
           "what when where who how why which that this it its as have has had do did does not "
           "didnt from about can could would should will there their they them we us our".split())
TOKEN_RE = re.compile(r"[a-z0-9']+")


def session_structure() -> tuple[dict[str, list[tuple[int, int]]], dict[str, int]]:
    """qid -> list of (start, end) absolute turn ranges per session; qid -> turn count."""
    d = json.load(io.open(FULL, encoding="utf-8"))
    sess_ranges: dict[str, list[tuple[int, int]]] = {}
    counts: dict[str, int] = {}
    i = 0
    for rec in d:
        qid = rec["question_id"]
        ranges = []
        for sess in rec["haystack_sessions"]:
            ranges.append((i, i + len(sess)))
            i += len(sess)
        sess_ranges[qid] = ranges
        counts[qid] = sum(e - s for s, e in ranges)
    return sess_ranges, counts


def lexical_scores(questions: list[str], texts: list[str]) -> np.ndarray:
    """(n_q, n_turns) IDF-weighted overlap in [0,1]."""
    q_tokens = [[t for t in TOKEN_RE.findall(q.lower()) if t not in STOP] for q in questions]
    df = Counter()
    for t in texts:
        df.update(set(TOKEN_RE.findall(t.lower())) - STOP)
    n = len(texts)
    idf = {w: np.log(n / max(1, df[w])) for w in df}
    turn_tokens = [set(TOKEN_RE.findall(t.lower())) - STOP for t in texts]
    out = np.zeros((len(questions), n), dtype=np.float32)
    for qi, toks in enumerate(q_tokens):
        if not toks:
            continue
        denom = sum(idf.get(w, 0.0) for w in set(toks))
        if denom <= 0:
            continue
        qset = set(toks)
        for ti, tset in enumerate(turn_tokens):
            inter = qset & tset
            if inter:
                out[qi, ti] = sum(idf.get(w, 0.0) for w in inter) / denom
    return out


def gold5(sc: np.ndarray, ev_rows: np.ndarray, qid_list: list[str],
          gold_abs: dict[str, set[int]], per_type_map: dict[str, str]) -> tuple[float, dict[str, str]]:
    order = np.argsort(-sc, axis=1)
    hits = 0
    pt: dict[str, list[int]] = {}
    for qi, r in enumerate(ev_rows):
        gp = gold_abs.get(qid_list[r], set())
        hit = 1 if set(map(int, order[qi, :RANK].tolist())) & gp else 0
        hits += hit
        qt = per_type_map[qid_list[r]]
        pt.setdefault(qt, [0, 0])
        pt[qt][0] += hit
        pt[qt][1] += 1
    return round(hits / len(ev_rows), 4), (f"{sum(v[0] for v in pt.values())}/{len(ev_rows)}", {k: f"{v[0]}/{v[1]}" for k, v in sorted(pt.items())})


def main() -> None:
    t0 = time.time()
    texts, roles, turn_qids, qtype_of, gold_abs, questions = load_full()
    X = ensure(texts)
    Xq = embed(questions)
    S = (Xq / np.maximum(1e-9, np.linalg.norm(Xq, axis=1, keepdims=True))) @ \
        (X / np.maximum(1e-9, np.linalg.norm(X, axis=1, keepdims=True))).T
    qid_list = list(dict.fromkeys(turn_qids))
    row_of = {q: i for i, q in enumerate(qid_list)}
    sess_ranges, _counts = session_structure()
    assert len(sess_ranges) == len(qid_list)
    assert sum(sum(e - s for s, e in r) for r in sess_ranges.values()) == len(texts)
    print(f"[load] {len(texts)} turns ({time.time()-t0:.1f}s)", flush=True)

    # F1: session-aggregate similarity, mapped to turns
    Sagg = np.zeros_like(S)
    for qi, q in enumerate(qid_list):
        lo_all = sess_ranges[q][0][0]
        for (s, e) in sess_ranges[q]:
            agg = X[s:e].mean(axis=0)
            agg = agg / max(1e-9, np.linalg.norm(agg))
            Sagg[qi, s:e] = float(np.dot(Xq[qi], agg))
    print(f"[F1] session-aggregate built ({time.time()-t0:.1f}s)", flush=True)

    # F2: lexical IDF overlap
    L = lexical_scores(questions, texts)
    print(f"[F2] lexical built ({time.time()-t0:.1f}s)", flush=True)

    # split (same as D/E)
    rng = np.random.default_rng(SEED)
    by_type: dict[str, list[str]] = {}
    for q in qid_list:
        by_type.setdefault(qtype_of[q], []).append(q)
    tune, ev = [], []
    for _, qs in sorted(by_type.items()):
        perm = rng.permutation(len(qs))
        half = len(qs) // 2
        tune.extend(qs[i] for i in perm[:half])
        ev.extend(qs[i] for i in perm[half:])
    tune_rows = np.array([row_of[q] for q in tune])
    ev_rows = np.array([row_of[q] for q in ev])

    def sweep(base: np.ndarray, feature: np.ndarray, rows: np.ndarray,
              lams=LAMBDAS) -> dict[str, float]:
        out = {}
        for lam in lams:
            sc = base + lam * np.broadcast_to(feature, (len(rows), S.shape[1]))
            g, _ = gold5(sc, rows, qid_list, gold_abs, qtype_of)
            out[str(lam)] = g
        return out

    # ---- TUNE half sweeps ----
    tune_S = S[tune_rows]
    tune_agg = Sagg[tune_rows]
    tune_lex = L[tune_rows]
    base_t, _ = gold5(tune_S, tune_rows, qid_list, gold_abs, qtype_of)
    sw_agg = sweep(tune_S, tune_agg, tune_rows)
    sw_lex = sweep(tune_S, tune_lex, tune_rows)
    # combo: use tune-best single lambdas for both, then joint small sweep
    best_agg = max(LAMBDAS, key=lambda l: sw_agg[str(l)])
    best_lex = max(LAMBDAS, key=lambda l: sw_lex[str(l)])
    combo = {}
    for la in (best_agg, best_agg * 2, best_agg / 2):
        for ll in (best_lex, best_lex * 2, best_lex / 2):
            sc = tune_S + la * tune_agg + ll * tune_lex
            g, _ = gold5(sc, tune_rows, qid_list, gold_abs, qtype_of)
            combo[f"agg={la:g},lex={ll:g}"] = g
    best_combo = max(combo, key=combo.get)
    print(f"[tune] base={base_t} agg={sw_agg} lex={sw_lex}", flush=True)
    print(f"[tune] combo={combo} -> {best_combo}", flush=True)

    # ---- EVAL half ----
    la, ll = (float(x) for x in best_combo.replace("agg=", "").replace("lex=", "").split(","))
    ev_base, pt_base = gold5(S[ev_rows], ev_rows, qid_list, gold_abs, qtype_of)
    ev_f1, pt_f1 = gold5(S[ev_rows] + la * Sagg[ev_rows], ev_rows, qid_list, gold_abs, qtype_of)
    ev_f2, pt_f2 = gold5(S[ev_rows] + ll * L[ev_rows], ev_rows, qid_list, gold_abs, qtype_of)
    ev_f12, pt_f12 = gold5(S[ev_rows] + la * Sagg[ev_rows] + ll * L[ev_rows], ev_rows, qid_list, gold_abs, qtype_of)

    # ---- F3 routing: inferred type -> per-type feature config from tune ----
    # train M1 as in probe E
    yq_type = np.array([qtype_of[q] for q in qid_list])
    m1 = LogisticRegression(max_iter=2000, C=4.0).fit(Xq[tune_rows], yq_type[tune_rows])
    preds_all = m1.predict(Xq)
    # for each type, pick on TUNE half: baseline vs +agg(tune-best) vs +lex(tune-best) vs both
    type_cfg = {}
    for qt in sorted(by_type):
        mask = np.array([1 if qtype_of[q] == qt else 0 for q in qid_list], dtype=bool)[tune_rows]
        rows = tune_rows[mask]
        if len(rows) < 8:
            type_cfg[qt] = ("none", 0.0, 0.0)
            continue
        cands = {}
        cands["none"] = gold5(tune_S[mask], rows, qid_list, gold_abs, qtype_of)[0]
        cands["agg"] = gold5(tune_S[mask] + la * tune_agg[mask], rows, qid_list, gold_abs, qtype_of)[0]
        cands["lex"] = gold5(tune_S[mask] + ll * tune_lex[mask], rows, qid_list, gold_abs, qtype_of)[0]
        cands["both"] = gold5(tune_S[mask] + la * tune_agg[mask] + ll * tune_lex[mask],
                              rows, qid_list, gold_abs, qtype_of)[0]
        best = max(cands, key=cands.get)
        type_cfg[qt] = (best, la, ll)
        print(f"[route] {qt}: {cands} -> {best}", flush=True)

    # apply routing on eval (predicted type, full-corpus rows)
    ev_idx_of_row = {r: i for i, r in enumerate(ev_rows)}
    sc_route = S[ev_rows].copy()
    for r in ev_rows:
        qt = preds_all[r]
        kind, la_, ll_ = type_cfg.get(qt, ("none", 0.0, 0.0))
        qi_local = ev_idx_of_row[r]
        if kind in ("agg", "both"):
            sc_route[qi_local] += la_ * Sagg[r]
        if kind in ("lex", "both"):
            sc_route[qi_local] += ll_ * L[r]
    ev_f3, pt_f3 = gold5(sc_route, ev_rows, qid_list, gold_abs, qtype_of)

    out = {
        "n_records": len(qid_list),
        "tune": {"base": base_t, "agg": sw_agg, "lex": sw_lex, "combo": combo, "best": best_combo},
        "type_cfg": {k: {"cfg": v[0], "agg": v[1], "lex": v[2]} for k, v in type_cfg.items()},
        "eval": {
            "baseline": {"g5": ev_base, "per_type": pt_base[1]},
            "f1_session_agg": {"g5": ev_f1, "lam": la, "per_type": pt_f1[1]},
            "f2_lexical": {"g5": ev_f2, "lam": ll, "per_type": pt_f2[1]},
            "f1+f2": {"g5": ev_f12, "per_type": pt_f12[1]},
            "f3_routed": {"g5": ev_f3, "per_type": pt_f3[1]},
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = os.path.join("data", "_scratch", "icm", "probe_feature_routing.json")
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[done] wrote {path}", flush=True)


if __name__ == "__main__":
    main()