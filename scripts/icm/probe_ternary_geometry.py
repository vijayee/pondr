"""ICM Probe A -- ternary/HDC geometry survival on real text embeddings.

UNTRACKED scratch -- never commit data outputs.

Question (standalone, not anchored to repo conclusions):
  Does a ternary {-1,0,+1} representation of dense text embeddings preserve
  enough similarity geometry to rank memories usefully?

  A1. Direct sign quantization at d=384 -- survival or collapse?
  A2. Zero-band thresholding (user's "vocabulary reduction" instinct) vs plain sign?
  A3. Bipolar random projection to higher d (2048, 10000) -- does near-orthogonality
      survive on top of SIGN quantization of a 384-d real embedding cloud?

Corpus: LongMemEval-50 interleaved sessions (~1035 turns) embedded with
bge-small-en-v1.5 (384-d). Probes = the 50 benchmark questions. Gold = turns
flagged has_answer (96 across 88 sessions). Primary readout is geometry
survival (rank agreement + similarity correlation vs full precision); gold
recall@k is reported as the retrieval-realism check. The learned projection
that would make HDC *relevant* (not just sim) is Phase-2 work, untouched here.

Run:  python scripts/icm/probe_ternary_geometry.py
"""

from __future__ import annotations

import io
import json
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HARNESS = r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness"
LME = os.path.join(HARNESS, "datasets", "longmemeval", "longmemeval_interleaved.json")
EMBEDDER = "BAAI/bge-small-en-v1.5"
SEED = 7
RANKS = (5, 10)
BANDS = (0.05, 0.1, 0.2)
PROJ_DIMS = (2048, 10000)

import torch  # noqa: E402

torch.set_num_threads(1)


def load_corpus() -> tuple[list[str], list[dict], list[set[int]]]:
    """Turn texts in record order; gold = set of global turn indices per record."""
    d = json.load(io.open(LME, encoding="utf-8"))
    texts: list[str] = []
    gold: list[set[int]] = []
    for rec in d:
        gold.append(set())
        for sess in rec["haystack_sessions"]:
            for turn in sess:
                if turn.get("has_answer"):
                    gold[-1].add(len(texts))
                texts.append(f"{turn.get('role','?')}: {turn.get('content','')}".strip())
    return texts, d, gold


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(EMBEDDER)
    return np.asarray(
        st.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )


def ternary_sign(x: np.ndarray) -> np.ndarray:  # A1
    return np.sign(x).astype(np.int8)


def ternary_band(x: np.ndarray, eps: float) -> np.ndarray:  # A2
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    out = np.zeros(x.shape, dtype=np.int8)
    out[x >= mx * eps] = 1
    out[x <= -mx * eps] = -1
    return out


_RP_W: dict[tuple[int, int], np.ndarray] = {}


def rand_project(x: np.ndarray, d_out: int) -> np.ndarray:  # A3
    d_in = x.shape[1]
    if (d_in, d_out) not in _RP_W:
        rng = np.random.default_rng(SEED)
        W = rng.standard_normal((d_in, d_out)) > 0  # +-1 matrix, seeded
        _RP_W[(d_in, d_out)] = np.where(W, np.float32(1.0), np.float32(-1.0))
    return np.sign(x @ _RP_W[(d_in, d_out)] / np.sqrt(d_in)).astype(np.int8)


def cos_all(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    an = np.linalg.norm(af, axis=1, keepdims=True)
    bn = np.linalg.norm(bf, axis=1, keepdims=True)
    an[an == 0] = 1.0
    bn[bn == 0] = 1.0
    return (af / an) @ (bf / bn).T


def _spearman_row(s_full: np.ndarray, s_test: np.ndarray, row: int) -> float:
    rf = np.argsort(np.argsort(-s_full[row]))
    rt = np.argsort(np.argsort(-s_test[row]))
    return float(np.corrcoef(rf.astype(np.float64), rt.astype(np.float64))[0, 1])


def _gold_recall(order: np.ndarray, gold_pos_sets: list[set[int]], k: int) -> float:
    hit = 0.0
    n = 0
    for i, gp in enumerate(gold_pos_sets):
        if not gp:
            continue
        n += 1
        if set(order[i, :k].tolist()) & gp:
            hit += 1.0
    return hit / max(1, n)


def main() -> None:
    t0 = time.time()
    texts, recs, gold = load_corpus()
    print(f"[load] {len(texts)} turns, gold total {sum(len(g) for g in gold)} in {time.time()-t0:.1f}s", flush=True)

    X = embed(texts)
    Xq = embed([r["question"] for r in recs])
    gold_sets = [gold[i] for i in range(len(recs))]
    print(f"[embed] X{X.shape} Xq{Xq.shape} in {time.time()-t0:.1f}s", flush=True)

    S_full = cos_all(Xq, X)  # 50 x N
    order_full = np.argsort(-S_full, axis=1)

    report: dict[str, dict] = {}

    def evaluate(name: str, Xt: np.ndarray, Qt: np.ndarray, zero_frac_x: float) -> None:
        St = cos_all(Qt, Xt)
        spear = float(np.mean([_spearman_row(S_full, St, i) for i in range(50)]))
        pear = float(np.corrcoef(S_full.ravel(), St.ravel())[0, 1])
        order_t = np.argsort(-St, axis=1)
        overlap = {str(k): round(float(np.mean([len(set(order_full[i, :k]) & set(order_t[i, :k])) / k for i in range(50)])), 4) for k in RANKS}
        report[name] = {
            "spearman": round(spear, 4),
            "pair_sim_pearson": round(pear, 4),
            "topk_overlap": overlap,
            "gold_recall@5": round(_gold_recall(order_t, gold_sets, 5), 4),
            "gold_recall@10": round(_gold_recall(order_t, gold_sets, 10), 4),
            "zero_frac": round(zero_frac_x, 4),
        }

    evaluate("T-SIGN-384", ternary_sign(X), ternary_sign(Xq), float(np.mean(ternary_sign(X) == 0)))
    for eps in BANDS:
        evaluate(f"T-BAND-{eps}", ternary_band(X, eps), ternary_band(Xq, eps), float(np.mean(ternary_band(X, eps) == 0)))
    for d_out in PROJ_DIMS:
        evaluate(f"T-RP-{d_out}", rand_project(X, d_out), rand_project(Xq, d_out), 0.0)

    out = {
        "corpus_size": len(texts),
        "gold_total": sum(len(g) for g in gold),
        "n_questions": len(recs),
        "conditions": report,
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = os.path.abspath(os.path.join("data", "_scratch", "icm", "probe_ternary_geometry.json"))
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[done] wrote {path} in {out['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()