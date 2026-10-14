"""E4 -- Phase 3 latency/energy benchmark: ternary HDC binding vs the
    Phase-2 dense baseline on WaveDB's FLAT index.

Plan §7 Phase 3 gate:
    E4: latency/energy at retrieval — the sign-flip/dot path meets the
        sub-millisecond per-candidate budget.

Benchmarks (all local, no relay):
    1. project() latency   — 384-d float -> 10k-d bipolar (per embedding)
    2. bind() latency      — elementwise multiply at 10k-d
    3. unbind() latency    — self-inverse (bipolar) roundtrip
    4. similarity()        — dot product at 10k-d
    5. WaveDB roundtrip    — insert + FLAT/COSINE search at 10k-d

The dense baseline: raw cosine on 384-d bge embeddings via FAISS-free numpy.

Run:  python scripts/icm/e4_binding_latency.py  (~30s, no relay)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from icm.binding import HDCSpace  # noqa: E402
from icm.ledger import ICMLedger  # noqa: E402
from icm.episodic import EpisodicStore  # noqa: E402

N_EPISODES = 500
N_QUERIES = 50
BATCH = 64


def bench(fn, *args, n_warmup: int = 3, n_iter: int = 100) -> dict:
    """Time fn over n_iter calls (with n_warmup), return μs/op."""
    for _ in range(n_warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    el = time.perf_counter() - t0
    return {"us_per_op": round(el * 1e6 / n_iter, 2), "n_iter": n_iter}


def bench_n(fn, *args, n_iter: int = 100) -> dict:
    """Time fn called n_iter times (counting each as one op)."""
    for _ in range(min(3, n_iter)):
        fn(*args)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    us = np.asarray(times) * 1e6
    return {"us_per_op": round(float(np.mean(us)), 2),
            "us_p95": round(float(np.percentile(us, 95)), 2),
            "n_iter": n_iter}


_model = None


def _embedder(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vecs = _model.encode(texts, batch_size=BATCH, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    return [list(map(float, v)) for v in vecs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", type=str,
                    default=os.path.join("data", "_scratch", "icm", "e4binding"))
    args = ap.parse_args()

    rng = np.random.default_rng(42)

    # ---- build a test corpus in WaveDB ----
    shutil.rmtree(args.db_path, ignore_errors=True)
    led = ICMLedger(os.path.join(args.db_path, "ledger"))
    store = EpisodicStore(led, _embedder, dim=384)
    try:
        texts = [f"Work-log entry {i}: the batch of {100 + i} records was processed "
                 f"and the checksum was {abs(hash(str(i))) % 10 ** 8:08x}."
                 for i in range(N_EPISODES)]
        embs = np.asarray(_embedder(texts), dtype=np.float32)
        eids = [store.capture(session_id=f"e4-{i:04d}", intent_id=None,
                              text=texts[i], episode_id=f"ep-{i:04d}")
                for i in range(N_EPISODES)]
        print(f"[setup] {N_EPISODES} episodes, {embs.shape[1]}-d bge", flush=True)

        # ---- HDC space ----
        hv = HDCSpace(dim=10_000)

        # ---- 1: project() latency ----
        projections = embs @ hv._Rp if hv._Rp is not None else None
        hv._ensure_Rp(embs.shape[1])

        def do_project(emb):
            return hv.project(emb)

        b_proj_batch = bench_n(do_project, embs[0], n_iter=200)
        print(f"[project] {b_proj_batch}", flush=True)

        # ---- 2: bind() latency ----
        hv_a = hv.project(embs[0]).flatten()
        hv_b = hv.project(embs[1]).flatten()
        b_bind = bench_n(hv.bind, hv_a, hv_b, n_iter=1000)
        print(f"[bind] {b_bind}", flush=True)

        # ---- 3: unbind() latency ----
        bound = hv.bind(hv_a, hv_b)
        b_unbind = bench_n(hv.unbind, bound, hv_a, n_iter=1000)
        print(f"[unbind] {b_unbind}", flush=True)

        # ---- 4: similarity() latency ----
        b_sim = bench_n(hv.similarity, hv_a, hv_b, n_iter=1000)
        print(f"[similarity] {b_sim}", flush=True)

        # ---- 5: WaveDB roundtrip at 10k-d ----
        # insert the projected vectors into a WaveDB VectorLayer at 10k-d
        from wavedb import VectorLayer, Format, IndexType, Distance, Runtime

        rt = Runtime(top_k=10, sync_only=1)
        fmt = Format(IndexType.FLAT, dim=10_000, distance=Distance.COSINE)
        vl = VectorLayer.open("e4-hdc", led._db, fmt=fmt, rt=rt)
        try:
            hvs = hv.project(embs)
            t0 = time.perf_counter()
            for i, eid in enumerate(eids):
                vl.insert_sync(eid, [float(x) for x in hvs[i]])
            ins_s = time.perf_counter() - t0
            print(f"[wavedb-insert] {N_EPISODES} vectors in {ins_s:.1f}s", flush=True)

            # search latency
            search_times = []
            for qi in range(N_QUERIES):
                qhv = [float(x) for x in hvs[qi % N_EPISODES]]
                t0 = time.perf_counter()
                vl.search_sync(qhv, 5)
                search_times.append((time.perf_counter() - t0) * 1000)
            us = np.asarray(search_times)
            print(f"[wavedb-search] avg={np.mean(us):.2f}us p95={np.percentile(us, 95):.2f}us",
                  flush=True)
            b_wavedb = {"insert_s_total": round(ins_s, 2),
                        "search_us_avg": round(float(np.mean(us)), 2),
                        "search_us_p95": round(float(np.percentile(us, 95)), 2)}
        finally:
            vl.close()

        # ---- 6: dense baseline (384-d numpy cosine over raw embeddings) ----
        def dense_cos(qi):
            return np.argsort(-(embs @ embs[qi]))[:5]

        b_dense = bench_n(lambda: dense_cos(0), n_iter=200)
        print(f"[dense-384-cosine] {b_dense}", flush=True)

        results = {
            "config": {"dim_hdc": 10_000, "dim_dense": 384,
                       "n_episodes": N_EPISODES},
            "e4_project_us": b_proj_batch,
            "e4_bind_us": b_bind,
            "e4_unbind_us": b_unbind,
            "e4_similarity_us": b_sim,
            "e4_wavedb_10k": b_wavedb,
            "dense_baseline_384_us": b_dense,
        }

        # ---- summary ----
        print("\n=== E4 SUMMARY ===")
        print(f"  project 384→10k bipolar:  {b_proj_batch['us_per_op']:.1f} μs")
        print(f"  bind (elementwise ×):     {b_bind['us_per_op']:.3f} μs")
        print(f"  unbind (self-inverse):    {b_unbind['us_per_op']:.3f} μs")
        print(f"  similarity (dot/dim):     {b_sim['us_per_op']:.3f} μs")
        print(f"  WaveDB search 10k-d:      {b_wavedb['search_us_avg']:.1f} μs avg")
        print(f"  dense 384-d cosine:       {b_dense['us_per_op']:.1f} μs")
        print(f"  E4 gate: bind+unbind+sim  "
              f"{b_bind['us_per_op'] + b_unbind['us_per_op'] + b_sim['us_per_op']:.3f} μs "
              f"(budget: 1000 μs/turn) → "
              f"{'PASS' if b_bind['us_per_op'] + b_unbind['us_per_op'] + b_sim['us_per_op'] < 1000 else 'FAIL'}")

        path = os.path.join("data", "_scratch", "icm", "e4_results.json")
        with io.open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[done] wrote {path}")
        return 0
    finally:
        led.close()


if __name__ == "__main__":
    sys.exit(main())