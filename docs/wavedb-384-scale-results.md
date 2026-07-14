# WaveDB VectorLayer: 384-dim at-scale test results

Date: 2026-07-14
Wheel: `wavedb-0.2.1-cp314-cp314-win_amd64.whl`, Hippo env, rebuilt from
WaveDB master **after** the sync_only scan fix (commit `ee97307`,
see `wavedb-reput-scan-invisibility-bug`). The wheel installed in the Hippo
env on 2026-07-14 INCLUDES that fix, so the IVF/SLSH `train()`/`migrate()`
paths (which do delete+re-put on aux keys) are exercised correctly.

## Why this test

Directive: "see if we can get that 384 size running at scale." Hippo stores
episode/semantic vectors at `dim=384`, `distance=COSINE`, `delimiter='/'`,
`sync_only=1`. We need to know (a) does each index type (FLAT / IVF / SLSH)
actually run at 100k vectors, and (b) what are the latency and recall
trade-offs, since Hippo today ships FLAT and the integration plan
(`docs/wavedb-vector-layer-integration-plan.md`) names IVF the graduation
path once FLAT's brute-force wall is hit.

## Config and methodology

- **Harness:** `scripts/_scratch/scale_384_wavedb.py` (untracked probe; NOT
  committed per the commit-at-will rule). Runs the REAL WaveDB C code via the
  installed wheel -- no mock, no FAISS sidecar.
- **Vectors:** clustered 384-dim unit vectors (random centroid + Gaussian
  noise, L2-normalized => cosine space). Clustered data has real neighbor
  structure, so recall is meaningful. Uniform-random in 384-dim is
  concentration-of-measure degenerate (every point is ~equidistant from every
  other) and makes recall numbers meaningless -- we do not use it.
- **Scale points:** N in {10,000, 50,000, 100,000} for each of FLAT / IVF /
  SLSH (9 cells total).
- **Queries:** 200 fresh clustered queries per cell, `k=10`.
- **Recall:** recall@10 vs exact numpy ground-truth cosine top-10
  (`queries @ db.T` + argpartition). recall@10 = |ANN-top10 ∩ exact-top10| / 10.
- **Latency:** per-query `search_sync` wall time, p50 and p99 over the 200
  queries.
- **IVF config:** `ivf_n_clusters = max(16, int(sqrt(N)))`, runtime
  `ivf_nprobe = max(8, n_clusters//4)`.
- **SLSH config:** `slsh_lsh_tables=10`, `slsh_hash_bits=16`,
  `slsh_bucket_width=3.0`, runtime `slsh_scan_radius=200`. (These are the
  defaults-ish, NOT tuned for 384 -- see the recall diagnosis below.)
- **Insert:** one-by-one `insert_sync` (no batch vector-insert API); throughput
  is cffi per-call bound (~1ms/call), NOT index-algorithm bound.

## Main results (all 9 cells ran, exit 0, NO crash at 100k)

```
type      n      ins_s  ins_thru  train_s   p50_ms   p99_ms  recall@10
flat   10000     18.08       553     0.00    41.61    51.13      1.000
ivf    10000      9.68      1033     3.14    20.21    43.52      0.468
slsh   10000      9.87      1014     1.23     7.15    50.57      0.069
flat   50000     93.97       532     0.00   212.04   227.90      1.000
ivf    50000     50.24       995    45.16    92.06   182.57      0.512
slsh   50000     48.98      1021    30.81    39.60   219.70      0.069
flat  100000    182.25       549     0.00   380.50   397.96      1.000
ivf   100000     95.35      1049   142.28   193.89   371.70      0.510
slsh  100000     97.50      1026   124.27   305.36   446.14      0.078
```

`ins_thru` = vectors/sec. `train_s` = k-means / LSH-projection build time
(zero for FLAT, which needs no training).

## Conclusions

- **Yes, 384 runs at 100k scale for all three types** with the scan fix. The
  core question is answered: the native in-DB vector index holds at Hippo's
  target dimension at 100k.
- **Insert** is ~545/s for FLAT and ~1025/s for IVF/SLSH, dominated by the
  cffi per-call cost (~1ms), not the index algorithm. 100k one-time build:
  FLAT ~182s, IVF ~95s insert + ~142s train, SLSH ~98s insert + ~124s train.
  This is an offline, one-time cost; acceptable.
- **FLAT (Hippo's current type):** exact, recall 1.000, but brute-force
  latency scales linearly with N -- 42ms@10k -> 212ms@50k -> 380ms@100k p50.
  Fine at Hippo's current episode counts (thousands of vectors); 380ms@100k
  is the real-time-retrieval wall that motivates graduating to IVF.
- **IVF (the planned graduation path):** p50 194ms@100k (well under FLAT's
  380ms), recall@10 ~0.51 and roughly stable across N. Recall is TUNABLE via
  `ivf_nprobe` (we used `sqrt(N)/4`, a low/recall-cheap setting) -- see the
  recall diagnosis below for the nprobe sweep that quantifies the lever.
  k-means train is the long pole (142s@100k).
- **SLSH:** runs, but recall@10 is only 0.07-0.08 with the params above, and
  p99 is high (446ms@100k) from LSH bucket-miss fallback scans. The runtime
  `slsh_scan_radius` lever alone does not rescue it; the Format-tier params
  (tables / hash_bits / bucket_width) need real tuning for 384-dim cosine and
  that requires a retrain. SLSH is NOT the right out-of-the-box ANN choice for
  Hippo at 384 -- IVF is.

## Recall diagnosis (the "serious recall issues with all but flat")

FLAT is exact (recall 1.000) by construction. The recall gap is entirely in
the ANN types, and the two gaps have different characters:

### IVF: recall ~0.51 is a TUNING issue, not a bug

IVF partitions the dataset into `n_clusters = sqrt(N)` k-means clusters and,
at query time, scans only the `ivf_nprobe` nearest clusters (exact within each
scanned cluster). recall is therefore bounded by "did the true top-10
neighbors fall in the `nprobe` clusters I scanned?" With
`nprobe = sqrt(N)/4` we scan ~25% of clusters, and recall ~0.51 reflects
exactly that: most but not all true neighbors land in the probed clusters.

The lever is `ivf_nprobe` (a RUNTIME value -- no retrain needed, just reopen
with a different `Runtime.ivf_nprobe`). Sweep at N=50,000, n_clusters=223,
200 queries, k=10 (harness `scripts/_scratch/sweep_384_recall.py`):

```
nprobe   recall@10   p50_ms   p99_ms
     8       0.121    18.97   133.10
    16       0.216    30.17   122.68
    32       0.329    58.64   160.01
    64       0.519   115.66   208.57
   128       0.811   304.78   329.44
   223       1.000   534.86   581.13   (nprobe == n_clusters: scan all = exact)
```

The dial is real and monotone: recall 0.12 -> 0.22 -> 0.33 -> 0.52 -> 0.81 ->
1.00 as nprobe rises. The scale test used nprobe ~= sqrt(N)/4 (~55 @ 50k),
which lands at recall ~0.51 -- consistent with nprobe=64 here.

Two caveats the sweep exposes:
- **The high-recall end is NOT free.** nprobe=223 (scan every cluster) is
  EXACT (recall 1.000) but p50 535ms -- SLOWER than FLAT's 212ms@50k, because
  IVF pays per-cluster routing overhead even when it scans them all. IVF only
  beats FLAT when you hold nprobe well below n_clusters.
- **At 50k the latency crossover is harsh.** To get recall >= 0.8 you need
  nprobe=128 (p50 305ms), already worse than FLAT exact (212ms). IVF's win is
  at LARGER scale: IVF latency grows roughly as `nprobe * (N/n_clusters) =
  nprobe * sqrt(N)` while FLAT grows as `N`. Hold nprobe fixed and IVF pulls
  ahead as N rises; the scale test already shows IVF@100k (194ms, nprobe~79)
  well under FLAT@100k (380ms). So pick an nprobe in the 64-128 band for a
  recall/latency trade that ages well with N, and do not chase recall near
  1.0 with IVF -- that is what FLAT is for.

### SLSH: recall ~0.07 needs Format-tier retraining, not just runtime tuning

SLSH (locality-sensitive hashing) recall is dominated by the Format-tier
parameters (`slsh_lsh_tables`, `slsh_hash_bits`, `slsh_bucket_width`) that
are baked into the hash structure at `train()` time. The defaults we used
(10 tables / 16 bits / bucket_width 3.0) are not tuned for 384-dim cosine and
recall collapses to ~0.07. The only runtime lever, `slsh_scan_radius`, widens
the bucket scan but cannot fix a bad hash projection. Radius sweep at N=50,000
(tables=10, bits=16, bw=3.0 held fixed; only radius is runtime-tunable):

```
radius   recall@10   p50_ms   p99_ms
    50       0.083   149.90   228.24
   200       0.083   147.32   223.47
   500       0.083   170.84   254.48
  1000       0.083   163.86   252.61
  2000       0.096   179.53   261.00
  5000       0.213   308.09   393.38
```

Recall is FLAT at 0.083 from radius 50 through 1000 -- widening the scan does
nothing because the true neighbors are simply NOT in the buckets the query
hashes to (a bad projection, not a narrow scan). It barely moves to 0.096 at
radius 2000 and only reaches 0.213 at radius 5000, where p50 is already 308ms
-- worse than FLAT exact (212ms@50k, recall 1.000) for a quarter of the recall.
The runtime lever is exhausted well before usefulness. Fixing SLSH at 384
means sweeping the Format-tier params (tables / hash_bits / bucket_width) with
a retrain per setting -- a separate tuning task -- and even then 384-dim
cosine may simply not be SLSH's regime. Do not pick SLSH for Hippo until that
work is done (and probably not after).

### Bottom line for Hippo

- FLAT stays correct and is the right choice until episode count pushes
  latency past the real-time budget (~hundreds of ms). At Hippo's current
  scale (thousands), FLAT is fine.
- IVF is the graduation path and its recall is tunable to an acceptable point
  by raising `ivf_nprobe` at the cost of latency -- a clean recall/latency
  dial, no retrain. The nprobe sweep below picks the operating point.
- SLSH at 384 needs dedicated Format-tier param tuning (a retrain sweep over
  tables/bits/bucket_width) before it is usable; until that work happens,
  do not pick SLSH for Hippo.

## Wheel rebuild note (with the scan fix)

Rebuilt `build-msvc12/Release/wavedb.dll` (CMake target `wavedb_shared`) from
WaveDB master post-`ee97307`, then
`WAVEDB_LIB_PATH=.../wavedb.dll python -m build --wheel --no-isolation` from
`bindings/python/`, producing
`dist/wavedb-0.2.1-cp314-cp314-win_amd64.whl`, installed with
`pip install --force-reinstall --no-deps`. The installed 0.2.1 in the Hippo
env now includes the scan fix (the earlier 2026-07-14 pre-fix wheel did not).

## See also

- `docs/wavedb-vector-layer-integration-plan.md` -- Hippo's FLAT->IVF
  graduation plan.
- WaveDB commit `ee97307` -- the sync_only scan read_txn_id fix that makes
  train/migrate correct (without it, delete+re-put aux keys were invisible to
  scans).
- Harnesses (untracked probes, NOT committed):
  `scripts/_scratch/scale_384_wavedb.py` (the 9-cell scale test),
  `scripts/_scratch/sweep_384_recall.py` (the IVF nprobe + SLSH radius sweeps
  below).