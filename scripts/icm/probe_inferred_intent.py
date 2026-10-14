"""ICM Probe E -- inferred vs oracle conditioning: the bridge to Phase 2.

UNTRACKED scratch -- never commit data outputs.

Probes B/D showed a query-conditional role prior buys +5.2pp on the untouched
500-q eval half -- but that prior read question_type from benchmark metadata
(a soft oracle). A real system must INFER the conditioning. This probe trains
the crudest possible intent heads on the tune half and measures how much of
the conditional-prior gain survives inference, and where inference damages:

  M1: 6-class linear head, question embedding -> question_type
      (reproduces the oracle signal -- upper bound of inferring THIS signal)
  M2: binary linear head, question embedding -> preferred role
      (the actual quantity the prior needs; trained on tune-half GOLD role of
       the answer-bearing turns -- a stand-in for Phase 0's harvested labels)

Retrieval on the eval half at lambda {0.05, 0.1, 0.2} (Probe D tuned 0.05 for
the hard oracle prior on the tune half; the soft prior needs its own lambda,
so both are reported -- all conditioning values below come from EVAL half
queries, and NO lambda is re-tuned here):
  baseline | oracle prior | M1-argmax prior | M2-argmax prior
           | M2-prob prior | M2-prob INVERTED control (wrong-direction check)

Readouts:
  * inference haircut: eval gain of inferred vs oracle prior
  * does inferred conditioning still self-harm where the constant prior did
    (the assistant class) or does it stay safe?
  * per-type error analysis of M2 (where would Phase 0 labels come from)

Reuse: same seed/stratified split as probe_500q_split (SEED=7, same code), so
eval half is identical and numbers cross-compare. Question embeddings are
re-embedded (cheap) rather than cached.

Run:  python scripts/icm/probe_inferred_intent.py
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
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from probe_500q_split import (  # noqa: E402
    load_full,
    ensure,
    embed,
    SEED,
)

RANK = 5


def main() -> None:
    t0 = time.time()
    texts, roles, turn_qids, qtype_of, gold_abs, questions = load_full()
    X = ensure(texts)
    Xq = embed(questions)
    print(f"[load] {X.shape[0]} turns, {Xq.shape[0]} qs ({time.time()-t0:.1f}s)", flush=True)

    S = (Xq / np.maximum(1e-9, np.linalg.norm(Xq, axis=1, keepdims=True))) @ \
        (X / np.maximum(1e-9, np.linalg.norm(X, axis=1, keepdims=True))).T

    qid_list = list(dict.fromkeys(turn_qids))
    row_of = {q: i for i, q in enumerate(qid_list)}

    # identical stratified split to probe D
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
    print(f"[split] tune={len(tune_rows)} eval={len(ev_rows)}", flush=True)

    yq_type = np.array([qtype_of[q] for q in qid_list])

    # gold-role preference per record (M2's training label, from tune half only):
    # a record prefers "assistant" iff ALL of its gold turns are assistant role.
    gold_role_set: dict[str, set[str]] = {}
    for qid in qid_list:
        gp = gold_abs.get(qid, set())
        gold_role_set[qid] = {roles[g] for g in gp}

    def pref_of(gset: set[str]) -> int:
        """1 = assistant preferred, 0 = user preferred; tie/empty -> user."""
        if gset and gset == {"assistant"}:
            return 1
        return 0

    yq_role = np.array([1 if pref_of(gold_role_set[qid_list[qi]]) else 0 for qi in range(len(qid_list))])

    Xq_train, Xq_eval = Xq[tune_rows], Xq[ev_rows]
    y_type_tr, y_type_ev = yq_type[tune_rows], yq_type[ev_rows]
    y_role_tr, y_role_ev = yq_role[tune_rows], yq_role[ev_rows]

    m1 = LogisticRegression(max_iter=2000, C=4.0)
    m1.fit(Xq_train, y_type_tr)
    p1 = m1.predict(Xq_eval)
    prob1 = m1.predict_proba(Xq_eval)
    conf1 = prob1.max(axis=1)
    a1 = accuracy_score(y_type_ev, p1)
    f1m = f1_score(y_type_ev, p1, average="macro")
    # train-half fit sanity
    a1_tr = accuracy_score(y_type_tr, m1.predict(Xq_train))
    print(f"[M1] type head: train_acc={a1_tr:.3f} eval_acc={a1:.3f} macro_f1={f1m:.3f} "
          f"conf mean={conf1.mean():.3f} p10={np.percentile(conf1,10):.3f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    m2 = LogisticRegression(max_iter=2000, C=4.0)
    m2.fit(Xq_train, y_role_tr)
    p2 = m2.predict(Xq_eval)
    prob2 = m2.predict_proba(Xq_eval)
    conf2 = prob2.max(axis=1)
    a2 = accuracy_score(y_role_ev, p2)
    a2_tr = accuracy_score(y_role_tr, m2.predict(Xq_train))
    print(f"[M2] role head: train_acc={a2_tr:.3f} eval_acc={a2:.3f} "
          f"conf mean={conf2.mean():.3f} p10={np.percentile(conf2,10):.3f} "
          f"(assistant rate ev={y_role_ev.mean():.2f})", flush=True)
    # probability diagnostics: predicted prob vs truth, per true class
    ev_true = np.array([qtype_of[q] for q in qid_list])[ev_rows]
    mean_p_true_asst = float(prob2[y_role_ev == 1, 1].mean()) if (y_role_ev == 1).any() else float("nan")
    mean_p_asst_on_user = float(prob2[y_role_ev == 0, 1].mean())
    print(f"[M2] mean p(assistant): true-assistant={mean_p_true_asst:.3f} "
          f"true-user={mean_p_asst_on_user:.3f}", flush=True)

    # M2 per-class eval detail + confusion vs question_type
    cls_report = {}
    for qt in sorted(set(qtype_of.values())):
        mask = np.array([qtype_of[q] for q in qid_list])[ev_rows] == qt
        cls_report[qt] = {"n": int(mask.sum()), "assistant_rate": round(float(y_role_ev[mask].mean()), 3)}
    pred_report = {}
    for pred_cls in (0, 1):
        mask = p2 == pred_cls
        pred_report[f"pred_{'assistant' if pred_cls else 'user'}"] = {"n": int(mask.sum()),
                                                                     "true_assistant_rate": round(float(y_role_ev[mask].mean()), 3)}

    # ---- priors and retrieval ----
    # CRITICAL: priors fill ALL records (predicted labels everywhere), and eval
    # proceeds on the untouched half. Filling eval rows only (zeros elsewhere)
    # acts as an illegitimate "own-record scope gate" and inflates everything
    # -- that was the contamination the differential test caught.
    is_user = np.array([1.0 if r == "user" else 0.0 for r in roles], dtype=np.float32)
    is_asst = 1.0 - is_user
    rec_ranges: dict[str, tuple[int, int]] = {}
    lo_i = 0
    prev = None
    for ti, q in enumerate(turn_qids):
        if q != prev:
            if prev is not None:
                rec_ranges[prev] = (lo_i, ti)
            prev, lo_i = q, ti
    rec_ranges[prev] = (lo_i, len(turn_qids))
    assert len(rec_ranges) == len(qid_list)

    preds_all1 = m1.predict(Xq)          # all 500 records (head never saw eval questions)
    probs_all2 = m2.predict_proba(Xq)
    P_oracle = np.zeros(len(roles), dtype=np.float32)  # true labels, everywhere
    P_m1 = np.zeros(len(roles), dtype=np.float32)
    P_m1c = np.zeros(len(roles), dtype=np.float32)  # confidence-gated argmax
    P_m2 = np.zeros(len(roles), dtype=np.float32)
    P_m2c = np.zeros(len(roles), dtype=np.float32)  # confidence-scaled hard prior
    P_m2p = np.zeros(len(roles), dtype=np.float32)  # soft
    P_m2_const = np.zeros(len(roles), dtype=np.float32)  # inverted control
    for qidx, q in enumerate(qid_list):
        lo_q, hi_q = rec_ranges[q]
        want_t = ("assistant" if qtype_of[q] == "single-session-assistant" else "user")
        pred1 = preds_all1[qidx]
        want1 = "assistant" if pred1 == "single-session-assistant" else "user"
        p_as = probs_all2[qidx, 1]
        want2 = "assistant" if p_as >= 0.5 else "user"
        P_oracle[lo_q:hi_q] = is_asst[lo_q:hi_q] if want_t == "assistant" else is_user[lo_q:hi_q]
        P_m1[lo_q:hi_q] = is_asst[lo_q:hi_q] if want1 == "assistant" else is_user[lo_q:hi_q]
        conf1 = float(m1.predict_proba(Xq[qidx:qidx + 1])[0].max())
        P_m1c[lo_q:hi_q] = P_m1[lo_q:hi_q] * conf1
        P_m2[lo_q:hi_q] = is_asst[lo_q:hi_q] if want2 == "assistant" else is_user[lo_q:hi_q]
        P_m2c[lo_q:hi_q] = P_m2[lo_q:hi_q] * abs(p_as - 0.5) * 2
        P_m2p[lo_q:hi_q] = p_as * is_asst[lo_q:hi_q] + (1.0 - p_as) * is_user[lo_q:hi_q]
        P_m2_const[lo_q:hi_q] = p_as * is_user[lo_q:hi_q] + (1.0 - p_as) * is_asst[lo_q:hi_q]  # inverted control

    def gold5(lam_prior: np.ndarray, lam: float) -> float:
        sc = S[ev_rows] + lam * np.broadcast_to(lam_prior, (len(ev_rows), S.shape[1]))
        order = np.argsort(-sc, axis=1)
        hits = 0
        for qi, r in enumerate(ev_rows):
            gp = gold_abs.get(qid_list[r], set())
            hits += 1 if set(map(int, order[qi, :RANK].tolist())) & gp else 0
        return round(hits / len(ev_rows), 4)

    out: dict = {
        "split": {"tune": len(tune_rows), "eval": len(ev_rows)},
        "head_accuracy": {"m1_type": {"eval": round(float(a1), 3), "train": round(float(a1_tr), 3),
                                      "macro_f1": round(float(f1m), 3)},
                          "m2_role": {"eval": round(float(a2), 3), "train": round(float(a2_tr), 3)}},
        "m2_class_report": cls_report,
        "m2_pred_report": pred_report,
        "eval": {},
    }
    # fixed lambda set {0.05, 0.1, 0.2}; no re-tuning on eval (D already chose
    # 0.05 for the hard oracle prior on the tune half).
    ev_base = gold5(np.zeros_like(P_m1), 0.0)
    out["eval"]["baseline"] = ev_base
    for name, P in (("oracle", P_oracle), ("m1_argmax", P_m1), ("m1_conf_gated", P_m1c),
                    ("m2_argmax", P_m2), ("m2_conf_gated", P_m2c),
                    ("m2_soft", P_m2p), ("m2_soft_inverted_control", P_m2_const)):
        out["eval"][name] = {"lam0.05": gold5(P, 0.05),
                             "lam0.1": gold5(P, 0.1),
                             "lam0.2": gold5(P, 0.2)}
    # per-type for the key rows at 0.05
    def per_type(P: np.ndarray, lam: float) -> dict:
        sc = S[ev_rows] + lam * np.broadcast_to(P, (len(ev_rows), S.shape[1]))
        order = np.argsort(-sc, axis=1)
        pt: dict[str, list[int]] = {}
        for qi, r in enumerate(ev_rows):
            gp = gold_abs.get(qid_list[r], set())
            hit = 1 if set(map(int, order[qi, :RANK].tolist())) & gp else 0
            qt = qtype_of[qid_list[r]]
            pt.setdefault(qt, [0, 0])
            pt[qt][0] += hit
            pt[qt][1] += 1
        return {k: f"{v[0]}/{v[1]}" for k, v in sorted(pt.items())}
    out["per_type@0.05"] = {
        "baseline": per_type(np.zeros_like(P_m1), 0.0),
        "oracle": per_type(P_oracle, 0.05),
        "m1_argmax": per_type(P_m1, 0.05),
        "m2_argmax": per_type(P_m2, 0.05),
        "m2_soft": per_type(P_m2p, 0.05),
    }
    out["elapsed_s"] = round(time.time() - t0, 1)

    path = os.path.join("data", "_scratch", "icm", "probe_inferred_intent.json")
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[done] wrote {path}", flush=True)


if __name__ == "__main__":
    main()