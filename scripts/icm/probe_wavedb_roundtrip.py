"""ICM Probe C -- WaveDB 0.2.12 round-trip: does the SUBSTRATE reproduce the
numpy retrieval geometry of Probes A/B, and what do the approx index flavors
(SLSH, IVF) cost?

UNTRACKED scratch -- never commit data outputs.

For each condition: open a fresh WaveDB dir, VectorLayer.open(...), insert the
1035-turn corpus, search the 50 questions, and report
  * wavedb-vs-numpy top-k agreement for the SAME condition (substrate fidelity),
  * top-k overlap + gold recall vs the FULL-384 numpy order (condition quality),
  * per-search wall-clock (latency telemetry, E4-adjacent).

Conditions:
  FULL-384 FLAT/COSINE   float control (substrate fidelity reference)
  SIGN   FLAT/COSINE     ternary sign vectors (int -> float list) dim=384
  BAND   FLAT/COSINE     zero-band eps=0.05 dim=384
  RP10K  FLAT/COSINE     bipolar random projection dim=10000
  SIGN-SLSH / RP10K-SLSH same data through the SLSH flavor (approx)
  SIGN-IVF  / RP10K-IVF  same data through IVF (one train() call)

NOTE: SLSH params (slsh_lsh_tables/hash_bits/bucket_width) have no published
     defaults here -- we sweep a few guesses and report recall; treat SLSH rows
     as exploratory, not gates.

Run:  python scripts/icm/probe_wavedb_roundtrip.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)

from wavedb import VectorLayer, Format, IndexType, Distance, Runtime, WaveDB, WaveDBConfig

from probe_ternary_geometry import (  # noqa: E402
    load_corpus,
    cos_all,
    rand_project,
    ternary_band,
    ternary_sign,
)
from probe_conditioning_gap import ensure_embeds  # shares the embedding cache

import probe_ternary_geometry as TG  # noqa: E402  (for TG.EMBEDDER)

RANKS = (5, 10)
ROOT = os.path.abspath(os.path.join("data", "_scratch", "icm"))


def flat(v: np.ndarray) -> list[float]:
    return [float(x) for x in v]


def main() -> None:
    t0 = time.time()
    texts, recs, gold = load_corpus()
    X = ensure_embeds(texts)
    Xq = embed_queries([r["question"] for r in recs])
    gold_sets = list(gold)
    print(f"[load] {X.shape[0]} turns ({time.time()-t0:.1f}s)", flush=True)

    # numpy reference orders (same code path as Probes A/B -- no re-derivation)
    S_full_np = cos_all(Xq, X)
    order_full_np = np.argsort(-S_full_np, axis=1)

    X_sign = ternary_sign(X)
    Q_sign = ternary_sign(Xq)
    X_band = ternary_band(X, 0.05)
    Q_band = ternary_band(Xq, 0.05)
    X_rp = rand_project(X, 10000)
    Q_rp = rand_project(Xq, 10000)

    conditions = {
        # NOTE: keyword args on purpose -- the repo's positional
        # Format(FLAT, dim, COSINE) pattern lands the distance in the
        # `delimiter` field and only works because the delimiter is unused
        # and the distance enum's default (1) IS COSINE.
        "FULL-FLAT": (X, Xq, Format(IndexType.FLAT, dim=X.shape[1], distance=Distance.COSINE)),
        "SIGN-FLAT": (X_sign, Q_sign, Format(IndexType.FLAT, dim=X.shape[1], distance=Distance.COSINE)),
        "BAND-FLAT": (X_band, Q_band, Format(IndexType.FLAT, dim=X.shape[1], distance=Distance.COSINE)),
        "RP10K-FLAT": (X_rp, Q_rp, Format(IndexType.FLAT, dim=X_rp.shape[1], distance=Distance.COSINE)),
        "SIGN-SLSH-a": (X_sign, Q_sign, Format(IndexType.SLSH, dim=X.shape[1], distance=Distance.COSINE,
                                               slsh_lsh_tables=8, slsh_hash_bits=16, slsh_bucket_width=0.25)),
        "SIGN-SLSH-b": (X_sign, Q_sign, Format(IndexType.SLSH, dim=X.shape[1], distance=Distance.COSINE,
                                               slsh_lsh_tables=16, slsh_hash_bits=32, slsh_bucket_width=0.5)),
        "RP10K-SLSH": (X_rp, Q_rp, Format(IndexType.SLSH, dim=X_rp.shape[1], distance=Distance.COSINE,
                                          slsh_lsh_tables=16, slsh_hash_bits=32, slsh_bucket_width=0.5)),
        "SIGN-IVF": (X_sign, Q_sign, Format(IndexType.IVF, dim=X.shape[1], distance=Distance.COSINE, ivf_n_clusters=16)),
        "RP10K-IVF": (X_rp, Q_rp, Format(IndexType.IVF, dim=X_rp.shape[1], distance=Distance.COSINE, ivf_n_clusters=32)),
    }

    report: dict[str, dict] = {}
    for name, (Xc, Qc, fmt) in conditions.items():
        row = run_condition(name, Xc, Qc, fmt, order_full_np, gold_sets)
        report[name] = row

    out = {"n_turns": X.shape[0], "n_questions": len(recs), "conditions": report,
           "elapsed_s": round(time.time() - t0, 1)}
    path = os.path.join(ROOT, "probe_wavedb_roundtrip.json")
    io.open(path, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[done] wrote {path}", flush=True)


def embed_queries(qs: list[str]) -> np.ndarray:
    # queries share the corpus cache file only by corpus texts; embed directly
    from sentence_transformers import SentenceTransformer

    import probe_ternary_geometry as TG

    st = SentenceTransformer(TG.EMBEDDER)
    return np.asarray(
        st.encode(qs, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )


def run_condition(name: str, Xc: np.ndarray, Qc: np.ndarray, fmt: "Format",
                  order_full_np: np.ndarray, gold_sets: list[set[int]]) -> dict:
    t0 = time.time()
    db_dir = os.path.join(ROOT, "wdb", name)
    shutil.rmtree(db_dir, ignore_errors=True)
    db = WaveDB(db_dir, config=WaveDBConfig(lru_memory_mb=100))
    rt = Runtime(top_k=10, sync_only=1)
    vl = VectorLayer.open("probe", db, fmt=fmt, rt=rt)
    try:
        try:
            vl.train()  # pre-insert training (no-op for FLAT/SLSH)
        except Exception as e:
            print(f"[{name}] pre-train skipped: {e}", flush=True)
        ids = [str(i) for i in range(Xc.shape[0])]
        t_ins = time.time()
        for i, v in enumerate(Xc):
            vl.insert_sync(ids[i], flat(v))
        ins_s = time.time() - t_ins
        if fmt.index_type == IndexType.IVF:
            try:
                vl.train()  # post-insert clustering for IVF
            except Exception as e:
                print(f"[{name}] post-train failed: {e}", flush=True)

        t_ser = 0.0
        wb_orders: list[list[int]] = []
        for qi in range(Qc.shape[0]):
            tq = time.time()
            res = vl.search_sync(flat(Qc[qi]), max(RANKS))
            t_ser += time.time() - tq
            wb_orders.append([int(r.id_str) for r in res])

        wb = np.asarray([[x for x in o[: max(RANKS)]] + [-1] * (max(RANKS) - len(o)) for o in wb_orders])
        np_order = condition_numpy_order(name, Xc, Qc)
        rep = {
            "fmt": str(fmt),
            "count": vl.count(),
            "insert_s": round(ins_s, 2),
            "search_ms_avg": round(1000 * t_ser / Qc.shape[0], 3),
            # substrate fidelity: wavedb order vs numpy order of SAME condition
            "wb_vs_np_top10_overlap": _overlap(wb, np_order, 10, pad=True),
            # condition quality vs FULL numpy control
            "topk_overlap_vs_full": {str(k): _overlap(wb, order_full_np, k) for k in RANKS},
            "gold_recall@5": _gold(wb, gold_sets, 5),
            "gold_recall@10": _gold(wb, gold_sets, 10),
            "wb_vs_np_spearman10": _spearman_on_top(wb, np_order, 10),
        }
        print(f"[{name}] {rep['count']} vecs {ins_s:.1f}s ins, "
              f"{rep['search_ms_avg']}ms/search, wb-vs-np@10={rep['wb_vs_np_top10_overlap']}, "
              f"gold@5={rep['gold_recall@5']} ({time.time()-t0:.1f}s)", flush=True)
        return rep
    finally:
        vl.close()
        db.close()


def condition_numpy_order(name: str, Xc: np.ndarray, Qc: np.ndarray) -> np.ndarray:
    S = cos_all(Qc, Xc)
    return np.argsort(-S, axis=1)


def _overlap(wb: np.ndarray, np_order: np.ndarray, k: int, pad: bool = False) -> float:
    vals = []
    for i in range(wb.shape[0]):
        a = set(int(x) for x in wb[i, :k] if x >= 0)
        b = set(int(x) for x in np_order[i, :k])
        vals.append(len(a & b) / k)
    return round(float(np.mean(vals)), 4)


def _gold(wb: np.ndarray, gold_sets: list[set[int]], k: int) -> float:
    hit = n = 0
    for i, gp in enumerate(gold_sets):
        if not gp:
            continue
        n += 1
        if set(int(x) for x in wb[i, :k] if x >= 0) & gp:
            hit += 1
    return round(hit / max(1, n), 4)


def _spearman_on_top(wb: np.ndarray, np_order: np.ndarray, k: int) -> float:
    vals = []
    for i in range(wb.shape[0]):
        a = [int(x) for x in wb[i, :k] if x >= 0]
        if len(a) < 2:
            vals.append(0.0)
            continue
        rank_a = {v: j for j, v in enumerate(a)}
        b = [int(x) for x in np_order[i, :k] if int(x) in rank_a]
        if len(b) < 2:
            vals.append(0.0)
            continue
        ra = [rank_a[v] for v in b]
        rb = list(range(len(b)))
        vals.append(float(np.corrcoef(np.array(ra, dtype=float), np.array(rb, dtype=float))[0, 1]))
    return round(float(np.mean(vals)), 4)


if __name__ == "__main__":
    main()