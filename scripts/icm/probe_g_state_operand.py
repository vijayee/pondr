"""ICM Probe G -- state operand and conditioning placement, protocol-correct.

UNTRACKED scratch -- never commit data outputs.

Questions asked (all tuned on the tune half, evaluated on the untouched half;
non-findings will be labeled non-findings, not spun):

  G1  state operand: does folding a stand-in "current state" into the score
      beat question-only? State stand-in = aggregate embedding of the LAST
      session dated BEFORE question_date (a recency-window stand-in for the
      agent's working context). Two foldings tested:
        (a) additive score: cos(turn,q) + lam_s * cos(turn,state)
        (b) folded query:  cos(turn, normalize(q + w * state))
      Known structural caveats (stated up front, not conclusions):
        - the state aggregate includes that session's own turns (1/N
          self-inclusion) -- inherent to "state = history aggregate";
        - a "last session" state may act as little more than a recency
          prior (probe B2 already showed recency is weak);
        - LongMemEval has no agent state at all -- whatever G1 shows is
          about THIS stand-in, not about the plan's state operand.

  G2  placement: does role conditioning belong INSIDE a session-first
      two-stage rank or on top of flat retrieval?
        flat | flat+role | two-stage(2 sess x 5 trn) | two-stage+role

Run:  python scripts/icm/probe_g_state_operand.py
"""

from __future__ import annotations

import io
import json
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)

from sklearn.linear_model import LogisticRegression

from probe_500q_split import FULL, load_full, ensure, embed, SEED  # noqa: E402
from probe_feature_routing import session_structure  # noqa: E402

RANK = 5
LAM_S = (0.05, 0.1, 0.2, 0.4)
W_FOLD = (0.25, 0.5, 1.0, 2.0)
LAM_ROLE = (0.05, 0.1, 0.2)


def dmy(s: str) -> tuple | None:
    try:
        y, m, d = s.split(" ")[0].split("/")
        return int(y), int(m), int(d)
    except Exception:
        return None


def before_or_same(sess_date: str, q_date: str) -> bool:
    a, b = dmy(sess_date), dmy(q_date)
    if a is None or b is None:
        return False
    from datetime import date as D

    return D(*a) <= D(*b)


def main() -> None:
    t0 = time.time()
    texts, roles, turn_qids, qtype_of, gold_abs, questions = load_full()
    X = ensure(texts)
    Xq = embed(questions)
    Xq_norm = Xq / np.maximum(1e-9, np.linalg.norm(Xq, axis=1, keepdims=True))
    S = Xq_norm @ (X / np.maximum(1e-9, np.linalg.norm(X, axis=1, keepdims=True))).T
    qid_list = list(dict.fromkeys(turn_qids))
    row_of = {q: i for i, q in enumerate(qid_list)}
    sess_ranges, _ = session_structure()

    # session dates + question dates (own dataset walk, same order as load_full)
    d = json.load(io.open(FULL, encoding="utf-8"))
    sess_dates: dict[str, list[str | None]] = {}
    q_dates: dict[str, str] = {}
    for rec in d:
        qid = rec["question_id"]
        qd = rec.get("question_date", "")
        q_dates[qid] = qd
        dates = rec.get("haystack_dates", [])
        sess_dates[qid] = [(dates[si] if si < len(dates) else "") for si in range(len(rec["haystack_sessions"]))]

    # G1 state: last PRE-question session aggregate per record
    state_vec: dict[str, np.ndarray | None] = {}
    state_session_idx: dict[str, int | None] = {}
    excluded = 0
    for qid, ranges in sess_ranges.items():
        cands = [k for k in range(len(ranges)) if before_or_same(sess_dates[qid][k] or "", q_dates[qid])]
        if not cands:
            state_vec[qid] = None
            excluded += 1
            continue
        s0, s1 = ranges[cands[-1]]
        agg = X[s0:s1].mean(axis=0)
        n = np.linalg.norm(agg)
        state_vec[qid] = agg / max(1e-9, n)
        state_session_idx[qid] = cands[-1]
    print(f"[G1-state] built for {len(state_vec)-excluded}/{len(state_vec)} records "
          f"(excluded {excluded}; {time.time()-t0:.1f}s)", flush=True)

    # per-question turn-level session-sim vector: cos(turn, state) computed per turn
    def state_sim_row(qid: str) -> np.ndarray:
        sv = state_vec[qid]
        if sv is None:
            return None
        lo, hi = sess_ranges[qid][state_session_idx[qid]]
        # sim over ALL turns (X rows) vs state: only needed for this record's turns
        # full-corpus dot is simple and cheap at this scale:
        return (X @ sv).astype(np.float32)

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

    # role priors (turn-length): oracle and M1-inferred, ALL records
    yq_type = np.array([qtype_of[q] for q in qid_list])
    m1 = LogisticRegression(max_iter=2000, C=4.0).fit(Xq[tune_rows], yq_type[tune_rows])
    preds_all = m1.predict(Xq)
    rec_ranges: dict[str, tuple[int, int]] = {}
    lo_i = 0
    prev = None
    for ti, q in enumerate(turn_qids):
        if q != prev:
            if prev is not None:
                rec_ranges[prev] = (lo_i, ti)
            prev, lo_i = q, ti
    rec_ranges[prev] = (lo_i, len(turn_qids))
    is_u = np.array([1.0 if r == "user" else 0.0 for r in roles], dtype=np.float32)
    is_a = 1.0 - is_u
    P_oracle = np.zeros(len(roles), dtype=np.float32)
    P_infer = np.zeros(len(roles), dtype=np.float32)
    for q in qid_list:
        lo_q, hi_q = rec_ranges[q]
        wt = "assistant" if qtype_of[q] == "single-session-assistant" else "user"
        wi = "assistant" if preds_all[row_of[q]] == "single-session-assistant" else "user"
        P_oracle[lo_q:hi_q] = is_a[lo_q:hi_q] if wt == "assistant" else is_u[lo_q:hi_q]
        P_infer[lo_q:hi_q] = is_a[lo_q:hi_q] if wi == "assistant" else is_u[lo_q:hi_q]

    def own_pool_scores(qid: str, extra: np.ndarray | None, lam: float) -> list[tuple[float, int]]:
        Sc = S[row_of[qid]]
        cand = [(float(Sc[t]) + (lam * float(extra[t]) if extra is not None else 0.0), t)
                for (s, e) in sess_ranges[qid] for t in range(s, e)]
        return sorted(cand, reverse=True)

    def g5(qids: list[str], score_fn) -> float:
        hits = 0
        n = 0
        for qid in qids:
            gp = gold_abs.get(qid, set())
            if not gp:
                continue
            n += 1
            top = {t for _, t in score_fn(qid)[:RANK]}
            if top & gp:
                hits += 1
        return round(hits / max(1, n), 4)

    # ---- G1: state operand sweeps on tune half ----
    tune_has_state = [q for q in tune if state_vec[q] is not None]
    base_fn = lambda qid: own_pool_scores(qid, None, 0.0)
    g1_tune = {"flat_nostate": g5(tune_has_state, base_fn)}
    best_add, best_fold = None, None
    for lam in LAM_S:
        rows_cache: dict[str, np.ndarray] = {}
        fn = lambda qid, lam=lam: own_pool_scores(qid, rows_cache.setdefault(qid, state_sim_row(qid)), lam)
        val = g5(tune_has_state, fn)
        g1_tune[f"add lam={lam:g}"] = val
        if best_add is None or val > g1_tune[best_add]:
            best_add = f"add lam={lam:g}"
    # folded query variant: recompute S row per record (per-question re-embed)
    qid_to_row = {row_of[q]: q for q in qid_list}
    for w in W_FOLD:
        S_fold = np.zeros_like(S)
        for qi, q in enumerate(qid_list):
            sv = state_vec[q]
            if sv is None:
                S_fold[qi] = S[qi]
                continue
            qf = Xq_norm[qi] + w * sv
            qf = qf / max(1e-9, np.linalg.norm(qf))
            S_fold[qi] = qf @ X.T
        def fn2(qid, S_fold=S_fold):
            cand = sorted(((float(S_fold[row_of[qid], t]), t) for (s, e) in sess_ranges[qid] for t in range(s, e)),
                          reverse=True)
            return cand
        val = g5(tune_has_state, fn2)
        g1_tune[f"fold w={w:g}"] = val
        if best_fold is None or val > g1_tune.get(best_fold, 0):
            best_fold = f"fold w={w:g}"
    print(f"[G1 tune] {g1_tune} -> best add={best_add} fold={best_fold}", flush=True)

    # eval for the best of each family (chosen on tune) + exclusion note
    lam_best = float(best_add.split("=")[1])
    w_best = float(best_fold.split("=")[1])
    def eval_fn_add(qid):
        return own_pool_scores(qid, state_sim_row(qid), lam_best)
    S_fold_best = np.zeros_like(S)
    for qi, q in enumerate(qid_list):
        sv = state_vec[q]
        if sv is None:
            S_fold_best[qi] = S[qi]
            continue
        qf = Xq_norm[qi] + w_best * sv
        qf = qf / max(1e-9, np.linalg.norm(qf))
        S_fold_best[qi] = qf @ X.T
    def eval_fn_fold(qid):
        return sorted(((float(S_fold_best[row_of[qid], t]), t) for (s, e) in sess_ranges[qid]
                       for t in range(s, e)), reverse=True)
    eval_has = [q for q in ev if state_vec[q] is not None]
    ev_flat_ns = g5(eval_has, base_fn)
    ev_add = g5(eval_has, eval_fn_add)
    ev_fold = g5(eval_has, eval_fn_fold)

    # ---- G2: placement (tune-half lambda choice for role, then eval) ----
    def flat_role_fn(lam, P):
        return lambda qid: own_pool_scores(qid, P, lam)
    role_tune = {}
    for lam in LAM_ROLE:
        role_tune[f"oracle lam={lam:g}"] = g5(tune, flat_role_fn(lam, P_oracle))
        role_tune[f"infer lam={lam:g}"] = g5(tune, flat_role_fn(lam, P_infer))
    lam_o = max(LAM_ROLE, key=lambda l: role_tune[f"oracle lam={l:g}"])
    lam_i = max(LAM_ROLE, key=lambda l: role_tune[f"infer lam={l:g}"])

    def two_stage(qid, k_sess, k_turn, P=None, lam=0.0, base=None) -> list[tuple[float, int]]:
        """Merged candidate list of (score, turn) with REAL scores, globally sorted,
        so the caller's [:k] sees the union of the top sessions' best turns."""
        q = row_of[qid]
        sims = [(float(Sagg_loc[qid][si]), si) for si in range(len(sess_ranges[qid]))]
        sims.sort(reverse=True)
        merged = []
        for _, si in sims[:k_sess]:
            s0, s1 = sess_ranges[qid][si]
            rows = [(float((base or S[q])[t]) + lam * float(P[t] if P is not None else 0.0), t)
                    for t in range(s0, s1)]
            merged.extend(rows[:k_turn])
        merged.sort(reverse=True)
        return merged
    # session agg per record (recompute locally over own sessions only)
    Sagg_loc: dict[str, list[float]] = {}
    for qid, ranges in sess_ranges.items():
        vals = []
        for (s, e) in ranges:
            agg = X[s:e].mean(axis=0)
            agg = agg / max(1e-9, np.linalg.norm(agg))
            vals.append(float(np.dot(Xq_norm[row_of[qid]], agg)))
        Sagg_loc[qid] = vals

    ts_plain = g5(tune, lambda qid: two_stage(qid, 2, RANK))
    role_ts_tune = {}
    for lam in LAM_ROLE:
        role_ts_tune[f"ts_oracle lam={lam:g}"] = g5(tune, lambda qid, lam=lam: two_stage(qid, 2, RANK, P_oracle, lam))
        role_ts_tune[f"ts_infer lam={lam:g}"] = g5(tune, lambda qid, lam=lam: two_stage(qid, 2, RANK, P_infer, lam))
    lam_ts_o = max(LAM_ROLE, key=lambda l: role_ts_tune[f"ts_oracle lam={l:g}"])
    lam_ts_i = max(LAM_ROLE, key=lambda l: role_ts_tune[f"ts_infer lam={l:g}"])

    out = {
        "excluded_no_dated_preq_session": excluded,
        "G1": {
            "tune": g1_tune,
            "chosen": {"add": best_add, "fold": best_fold},
            "eval_on_state_available_records": {"flat_nostate": ev_flat_ns, "add": ev_add, "fold": ev_fold},
            "n_eval_records_with_state": len(eval_has),
        },
        "G2": {
            "tune": {"flat_role": role_tune, "two_stage_plain": ts_plain, "two_stage_role": role_ts_tune},
            "chosen": {"flat_oracle_lam": float(lam_o), "flat_infer_lam": float(lam_i),
                       "ts_oracle_lam": float(lam_ts_o), "ts_infer_lam": float(lam_ts_i)},
            "eval": {
                "flat_role_oracle": g5(ev, flat_role_fn(lam_o, P_oracle)),
                "flat_role_infer": g5(ev, flat_role_fn(lam_i, P_infer)),
                "flat_plain": g5(ev, base_fn),
                "two_stage_plain": g5(ev, lambda qid: two_stage(qid, 2, RANK)),
                "two_stage_role_oracle": g5(ev, lambda qid: two_stage(qid, 2, RANK, P_oracle, lam_ts_o)),
                "two_stage_role_infer": g5(ev, lambda qid: two_stage(qid, 2, RANK, P_infer, lam_ts_i)),
            },
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = os.path.join("data", "_scratch", "icm", "probe_g_state_operand.json")
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[done] wrote {path}", flush=True)


if __name__ == "__main__":
    main()