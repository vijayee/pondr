"""Hippo-env scale test: WaveDB VectorLayer at 384-dim, FLAT/IVF/SLSH, at scale.

Answers the original directive part 3: "see if we can get that 384 size running
at scale". Runs against the REAL WaveDB C code via the installed 0.2.1 wheel
(rebuilt with the sync_only scan fix ee97307), in the Hippo Python env, with
Hippo's actual vector config (dim=384, delimiter='/', distance=COSINE, sync_only).

For each index type x N in {10k, 50k, 100k}:
  - generate clustered 384-dim unit vectors (real neighbor structure, so recall
    is meaningful -- unlike uniform random which is concentration-of-measure
    degenerate in high dim)
  - insert N vectors one-by-one (measure insert throughput)
  - train() IVF/SLSH (measure train time)
  - run NQUERIES fresh clustered queries, measure search latency p50/p99
  - recall@10 vs exact numpy ground-truth cosine

Output: a single results table to stdout. No network, no model, no GPU.
"""
from __future__ import annotations

import os
import shutil
import statistics
import sys
import tempfile
import time

import numpy as np

from wavedb import Distance, Format, IndexType, Runtime, VectorLayer, set_quiet

set_quiet(True)

DIM = 384
NS = [10_000, 50_000, 100_000]
NQUERIES = 200
K = 10
SEED = 7

# SLSH params for 384-dim (tunable; recall is reported, not asserted).
SLSH_TABLES = 10
SLSH_HASH_BITS = 16
SLSH_BUCKET_WIDTH = 3.0


def gen_clustered(n: int, dim: int, n_clusters: int, rng) -> np.ndarray:
    """Clustered unit vectors: centroid + noise, L2-normalized (cosine space).

    Noise scale is chosen so the noise MAGNITUDE (~scale*sqrt(dim)) is a fixed
    fraction (NOISE_MAG) of the unit centroid magnitude (1.0). A bare
    scale=0.3 is degenerate in high dim: |0.3*N(0,1,384)| ~= 5.9, which drowns
    the unit centroid (centroid contributes only ~2.8% of the squared norm) ->
    the data becomes ~uniform-random on the sphere (concentration of measure),
    the true top-10 scatter across ~10/10 clusters, and ANN recall numbers are
    a pessimistic artifact, not a real index property. scale=NOISE_MAG/sqrt(dim)
    keeps |noise| ~= NOISE_MAG so the centroid signal survives -> intra-cluster
    cosine ~0.92, true top-10 concentrated in ~1 cluster, recall meaningful.
    """
    NOISE_MAG = 0.3
    cents = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    assign = rng.integers(0, n_clusters, size=n)
    noise = rng.normal(scale=NOISE_MAG / np.sqrt(dim), size=(n, dim)).astype(np.float32)
    pts = cents[assign] + noise
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts.astype(np.float32)


def ground_truth(db: np.ndarray, queries: np.ndarray, k: int) -> list[set[int]]:
    sims = queries @ db.T  # (nq, n) cosine (all unit-norm)
    # top-k per row
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    rows = np.arange(queries.shape[0])[:, None]
    order = np.argsort(-sims[rows, idx], axis=1)
    topk = idx[rows, order]
    return [set(int(j) for j in row) for row in topk]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, int((p / 100.0) * (len(xs) - 1)))
    return xs[i]


def run_one(index_type: int, n: int, tmpdir: str) -> dict:
    rng = np.random.default_rng(SEED)
    n_clusters = max(16, int(n ** 0.5))
    db = gen_clustered(n, DIM, n_clusters, rng)
    queries = gen_clustered(NQUERIES, DIM, n_clusters, rng)

    name = {IndexType.FLAT: "flat", IndexType.IVF: "ivf", IndexType.SLSH: "slsh"}[index_type]
    path = os.path.join(tmpdir, f"{name}_{n}")
    os.makedirs(path, exist_ok=True)

    if index_type == IndexType.FLAT:
        fmt = Format(index_type=IndexType.FLAT, dim=DIM, distance=Distance.COSINE)
    elif index_type == IndexType.IVF:
        fmt = Format(index_type=IndexType.IVF, dim=DIM, distance=Distance.COSINE,
                     ivf_n_clusters=n_clusters)
    else:
        fmt = Format(index_type=IndexType.SLSH, dim=DIM, distance=Distance.COSINE,
                     slsh_lsh_tables=SLSH_TABLES, slsh_hash_bits=SLSH_HASH_BITS,
                     slsh_bucket_width=SLSH_BUCKET_WIDTH)
    rt = Runtime(top_k=K, sync_only=1, ivf_nprobe=max(8, n_clusters // 4),
                 slsh_scan_radius=200)

    vl = VectorLayer.open_separate(path, "scale", fmt, rt)

    # --- insert ---
    t0 = time.perf_counter()
    for i in range(n):
        vl.insert_sync(f"v{i}", db[i].tolist())
    insert_s = time.perf_counter() - t0
    cnt = vl.count()
    assert cnt == n, f"count {cnt} != {n}"

    # --- train (IVF/SLSH) ---
    train_s = 0.0
    if index_type != IndexType.FLAT:
        t0 = time.perf_counter()
        vl.train()
        train_s = time.perf_counter() - t0

    # --- search + recall ---
    gt = ground_truth(db, queries, K)
    lats = []
    recalls = []
    for q in range(NQUERIES):
        t0 = time.perf_counter()
        res = vl.search_sync(queries[q].tolist(), K)
        lats.append(time.perf_counter() - t0)
        got = set()
        for r in res:
            # id is "v{i}"
            s = r.id
            if isinstance(s, bytes):
                s = s.decode("utf-8", "replace")
            if s.startswith("v"):
                try:
                    got.add(int(s[1:]))
                except ValueError:
                    pass
        if gt[q]:
            recalls.append(len(got & gt[q]) / len(gt[q]))

    vl.close()
    return {
        "type": name, "n": n,
        "insert_s": insert_s,
        "insert_thru": n / insert_s,
        "train_s": train_s,
        "lat_p50_ms": pct(lats, 50) * 1000,
        "lat_p99_ms": pct(lats, 99) * 1000,
        "recall@10": (statistics.mean(recalls) if recalls else float("nan")),
    }


def main() -> int:
    tmpdir = tempfile.mkdtemp(prefix="wavedb_scale384_")
    print(f"dim={DIM} cosine queries={NQUERIES} k={K} tmp={tmpdir}")
    print(f"{'type':<6} {'n':>7} {'ins_s':>8} {'ins_thru':>9} {'train_s':>8} "
          f"{'p50_ms':>8} {'p99_ms':>8} {'recall@10':>10}")
    rows = []
    for n in NS:
        for it in [IndexType.FLAT, IndexType.IVF, IndexType.SLSH]:
            try:
                r = run_one(it, n, tmpdir)
            except Exception as e:
                r = {"type": {0:"flat",1:"ivf",2:"slsh"}[it], "n": n,
                     "insert_s": float("nan"), "insert_thru": float("nan"),
                     "train_s": float("nan"), "lat_p50_ms": float("nan"),
                     "lat_p99_ms": float("nan"), "recall@10": float("nan"),
                     "_err": str(e)[:80]}
            rows.append(r)
            err = r.get("_err", "")
            print(f"{r['type']:<6} {r['n']:>7} {r['insert_s']:>8.2f} "
                  f"{r['insert_thru']:>9.0f} {r['train_s']:>8.2f} "
                  f"{r['lat_p50_ms']:>8.2f} {r['lat_p99_ms']:>8.2f} "
                  f"{r['recall@10']:>10.3f}  {('ERR='+err) if err else ''}",
                  flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())