"""Quantify the IVF/SLSH recall tuning levers at 384-dim, to back the recall
diagnosis in docs/wavedb-384-scale-results.md with data instead of hand-waving.

Builds one IVF and one SLSH index at N=50000 (clustered 384-dim cosine, same
generator as scale_384_wavedb.py), trains once, then REOPENS the on-disk index
with different Runtime params (no reinsert/retrain) and measures recall@10 +
search latency p50/p99 over 200 queries vs exact numpy ground truth.

IVF lever: ivf_nprobe (1..n_clusters; nprobe==n_clusters == exact == recall 1.0).
SLSH lever: slsh_scan_radius (Runtime; bucket_width/tables/bits are Format and
need retrain, out of scope here -- documented as a separate tuning task).
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
N = 50_000
NQUERIES = 200
K = 10
SEED = 7

SLSH_TABLES = 10
SLSH_HASH_BITS = 16
SLSH_BUCKET_WIDTH = 3.0


def gen_clustered(n, dim, n_clusters, rng):
    # noise magnitude = NOISE_MAG (centroid magnitude = 1.0); scale=0.3 is
    # degenerate in 384-dim (|0.3*N(0,1,384)|~=5.9 drowns the unit centroid ->
    # uniform-random -> recall artifact). See scale_384_wavedb.py docstring.
    NOISE_MAG = 0.3
    cents = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    assign = rng.integers(0, n_clusters, size=n)
    noise = rng.normal(scale=NOISE_MAG / np.sqrt(dim), size=(n, dim)).astype(np.float32)
    pts = cents[assign] + noise
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts.astype(np.float32)


def ground_truth(db, queries, k):
    sims = queries @ db.T
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    rows = np.arange(queries.shape[0])[:, None]
    order = np.argsort(-sims[rows, idx], axis=1)
    topk = idx[rows, order]
    return [set(int(j) for j in row) for row in topk]


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int((p / 100.0) * (len(xs) - 1)))]


def build_index(index_type, path, n_clusters):
    rng = np.random.default_rng(SEED)
    db = gen_clustered(N, DIM, n_clusters, rng)
    if index_type == IndexType.IVF:
        fmt = Format(index_type=IndexType.IVF, dim=DIM, distance=Distance.COSINE,
                     ivf_n_clusters=n_clusters)
    else:
        fmt = Format(index_type=IndexType.SLSH, dim=DIM, distance=Distance.COSINE,
                     slsh_lsh_tables=SLSH_TABLES, slsh_hash_bits=SLSH_HASH_BITS,
                     slsh_bucket_width=SLSH_BUCKET_WIDTH)
    rt = Runtime(top_k=K, sync_only=1, ivf_nprobe=max(8, n_clusters // 4),
                 slsh_scan_radius=200)
    vl = VectorLayer.open_separate(path, "scale", fmt, rt)
    t0 = time.perf_counter()
    for i in range(N):
        vl.insert_sync(f"v{i}", db[i].tolist())
    ins_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    vl.train()
    train_s = time.perf_counter() - t0
    vl.close()
    return db, ins_s, train_s


def measure(path, fmt, rt, queries, gt):
    vl = VectorLayer.open_separate(path, "scale", fmt, rt)
    lats = []
    recalls = []
    for q in range(NQUERIES):
        t0 = time.perf_counter()
        res = vl.search_sync(queries[q].tolist(), K)
        lats.append(time.perf_counter() - t0)
        got = set()
        for r in res:
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
    return (statistics.mean(recalls) if recalls else float("nan"),
            pct(lats, 50) * 1000, pct(lats, 99) * 1000)


def main():
    tmp = tempfile.mkdtemp(prefix="wavedb_sweep384_")
    n_clusters = max(16, int(N ** 0.5))
    rng = np.random.default_rng(SEED + 1)
    queries = gen_clustered(NQUERIES, DIM, n_clusters, rng)

    # ---- IVF nprobe sweep ----
    print(f"IVF build N={N} n_clusters={n_clusters}")
    db, ins_s, train_s = build_index(IndexType.IVF, os.path.join(tmp, "ivf"), n_clusters)
    gt = ground_truth(db, queries, K)
    print(f"  ins={ins_s:.1f}s train={train_s:.1f}s")
    ivf_fmt = Format(index_type=IndexType.IVF, dim=DIM, distance=Distance.COSINE,
                     ivf_n_clusters=n_clusters)
    print(f"IVF nprobe sweep (recall@10 / p50 / p99 over {NQUERIES} queries):")
    print(f"  {'nprobe':>7} {'recall@10':>10} {'p50_ms':>8} {'p99_ms':>8}")
    for npb in [8, 16, 32, 64, 128, n_clusters]:
        rt = Runtime(top_k=K, sync_only=1, ivf_nprobe=npb, slsh_scan_radius=200)
        rec, p50, p99 = measure(os.path.join(tmp, "ivf"), ivf_fmt, rt, queries, gt)
        print(f"  {npb:>7} {rec:>10.3f} {p50:>8.2f} {p99:>8.2f}", flush=True)

    # ---- SLSH scan_radius sweep ----
    print(f"\nSLSH build N={N}")
    db2, ins2, train2 = build_index(IndexType.SLSH, os.path.join(tmp, "slsh"), n_clusters)
    gt2 = ground_truth(db2, queries, K)
    print(f"  ins={ins2:.1f}s train={train2:.1f}s")
    slsh_fmt = Format(index_type=IndexType.SLSH, dim=DIM, distance=Distance.COSINE,
                      slsh_lsh_tables=SLSH_TABLES, slsh_hash_bits=SLSH_HASH_BITS,
                      slsh_bucket_width=SLSH_BUCKET_WIDTH)
    print(f"SLSH scan_radius sweep (tables={SLSH_TABLES} bits={SLSH_HASH_BITS} bw={SLSH_BUCKET_WIDTH}):")
    print(f"  {'radius':>7} {'recall@10':>10} {'p50_ms':>8} {'p99_ms':>8}")
    for rad in [50, 200, 500, 1000, 2000, 5000]:
        rt = Runtime(top_k=K, sync_only=1, ivf_nprobe=8, slsh_scan_radius=rad)
        rec, p50, p99 = measure(os.path.join(tmp, "slsh"), slsh_fmt, rt, queries, gt2)
        print(f"  {rad:>7} {rec:>10.3f} {p50:>8.2f} {p99:>8.2f}", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())