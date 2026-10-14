"""ICM Probe D -- scale + split check of Probe B's conditional-role-prior result.

UNTRACKED scratch -- never commit data outputs.

Probe B (50-q) showed: baseline 0.80 @5; conditional role prior (switched per
query by question_type) 0.96, but the sweep was tuning-on-test. This probe:

  1. Embeds the full 500-record LongMemEval (~10.4k turns) once (npz cache).
  2. Stratifies records by question_type into a tune half / eval half.
  3. Sweeps lambda {0, .05, .1, .2, .4} for constant-user and conditional
     priors on the TUNE half only.
  4. Reports gold@5 on the EVAL half at the lambda chosen on tune, vs baseline.
  5. Reports eval-half per-type counts for baseline vs conditional.

Readouts this answers:
  * does the conditional-prior gain survive scale and an untouched split?
  * does the constant prior's self-harming behavior replicate at scale?
  * magnitude of the gain when the conditioning prior is a real (inferred)
    feature rather than a per-record oracle -- note the prior here is still
    QUESTION_TYPE-derived (query-time metadata), i.e. soft-oracle; the
    inference problem remains Phase 2's.

Run:  python scripts/icm/probe_500q_split.py
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

HARNESS = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness"
FULL = os.path.join(HARNESS, "datasets", "longmemeval", "longmemeval_interleaved_full.json")
EMBEDDER = "BAAI/bge-small-en-v1.5"
CACHE = os.path.join("data", "_scratch", "icm", "lme500_embeds.npz")
SEED = 7
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.4)
RANK = 5


def load_full() -> tuple[list[str], list[str], list[str], dict[str, str], dict[str, set[int]], list[str]]:
    """(texts, roles, turn_qids, qtype_of, gold_abs, questions)."""
    d = json.load(io.open(FULL, encoding="utf-8"))
    texts: list[str] = []
    roles: list[str] = []
    turn_qids: list[str] = []
    gold_abs: dict[str, set[int]] = {}
    qtype_of: dict[str, str] = {}
    for rec in d:
        qid = rec["question_id"]
        qtype_of[qid] = rec["question_type"]
        gold_abs[qid] = set()
        for sess in rec["haystack_sessions"]:
            for turn in sess:
                if turn.get("has_answer"):
                    gold_abs[qid].add(len(texts))
                texts.append(f"{turn.get('role','?')}: {turn.get('content','')}".strip())
                roles.append(turn.get("role", "?"))
                turn_qids.append(qid)
    questions = [r["question"] for r in d]
    return texts, roles, turn_qids, qtype_of, gold_abs, questions


def embed(texts: list[str], batch: int = 64, show: bool = False) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(EMBEDDER)
    return np.asarray(st.encode(texts, batch_size=batch, show_progress_bar=show,
                                convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)


def ensure(texts: list[str]) -> np.ndarray:
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if z["texts"].tolist() == texts and z["X"].shape[0] == len(texts):
            return z["X"].astype(np.float32)
    X = embed(texts, show=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, texts=np.array(texts, dtype=object), X=X)
    return X


def main() -> None:
    t0 = time.time()
    texts, roles, turn_qids, qtype_of, gold_abs, questions = load_full()
    print(f"[load] {len(texts)} turns, {len(questions)} qs ({time.time()-t0:.1f}s)", flush=True)
    X = ensure(texts)
    Xq = embed(questions)
    print(f"[embed] X{X.shape} Xq{Xq.shape} ({time.time()-t0:.1f}s)", flush=True)

    S = (Xq / np.maximum(1e-9, np.linalg.norm(Xq, axis=1, keepdims=True))) @ \
        (X / np.maximum(1e-9, np.linalg.norm(X, axis=1, keepdims=True))).T

    qid_list = list(dict.fromkeys(turn_qids))
    row_of = {q: i for i, q in enumerate(qid_list)}
    assert len(qid_list) == len(questions)

    # priors per TURN: 1.0 if the turn's role matches the preferred role
    P_const = np.array([1.0 if r == "user" else 0.0 for r in roles], dtype=np.float32)
    P_cond = np.array([
        1.0 if roles[i] == ("assistant" if qtype_of[turn_qids[i]] == "single-session-assistant" else "user")
        else 0.0 for i in range(len(roles))
    ], dtype=np.float32)

    # stratified split by question type -> tune/eval halves
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

    def gold5_of(sc_sub: np.ndarray, rows: np.ndarray, per_type: bool) -> tuple[float, dict[str, str]]:
        order = np.argsort(-sc_sub, axis=1)
        hits = 0
        pt: dict[str, list[int]] = {}
        for qi, r in enumerate(rows):
            gp = gold_abs.get(qid_list[r], set())
            hit = 1 if set(map(int, order[qi, :RANK].tolist())) & gp else 0
            hits += hit
            qt = qtype_of[qid_list[r]]
            pt.setdefault(qt, [0, 0])
            pt[qt][0] += hit
            pt[qt][1] += 1
        g = hits / len(rows)
        return round(g, 4), ({k: f"{v[0]}/{v[1]}" for k, v in sorted(pt.items())} if per_type else {})

    # ---- TUNE sweep (tune half only) ----
    tune_pts = {}
    tune_scores: dict[str, dict[str, float]] = {"const": {}, "cond": {}}
    for lam in LAMBDAS:
        for name, P in (("const", P_const), ("cond", P_cond)):
            sc = S[tune_rows] + lam * np.broadcast_to(P, (len(tune_rows), S.shape[1]))
            g, _ = gold5_of(sc, tune_rows, per_type=False)
            tune_scores[name][str(lam)] = g
    best = {name: max(LAMBDAS, key=lambda l: tune_scores[name][str(l)]) for name in tune_scores}
    print(f"[tune] const={tune_scores['const']} cond={tune_scores['cond']} -> best {best}", flush=True)

    # ---- EVAL half at tuned lambdas ----
    sc_full_base = S[ev_rows]
    ev_base, pt_base = gold5_of(sc_full_base, ev_rows, per_type=True)
    out: dict = {
        "n_records": len(qid_list),
        "n_turns": len(texts),
        "split": {"tune": len(tune_rows), "eval": len(ev_rows)},
        "type_counts": {k: len(v) for k, v in sorted(by_type.items())},
        "tune_scores": tune_scores,
        "chosen_lambda": {k: float(v) for k, v in best.items()},
    }
    for name, P in (("const", P_const), ("cond", P_cond)):
        lam = float(best[name])
        sc = S[ev_rows] + lam * np.broadcast_to(P, (len(ev_rows), S.shape[1]))
        g, pt = gold5_of(sc, ev_rows, per_type=True)
        out[f"eval_{name}"] = {"lambda": lam, "gold@5": g, "per_type": pt}
    out["eval"] = {"baseline": ev_base, "baseline_per_type": pt_base}
    out["elapsed_s"] = round(time.time() - t0, 1)

    path = os.path.join("data", "_scratch", "icm", "probe_500q_split.json")
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[done] wrote {path}", flush=True)


if __name__ == "__main__":
    main()