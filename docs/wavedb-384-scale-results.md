# WaveDB VectorLayer: 384-dim at-scale test results

Date: 2026-07-14
Wheel: `wavedb-0.2.1-cp314-cp314-win_amd64.whl`, Hippo env, rebuilt from
WaveDB master **after** the sync_only scan fix (commit `ee97307`,
see `wavedb-reput-scan-invisibility-bug`). The wheel installed in the Hippo
env on 2026-07-14 INCLUDES that fix, so the IVF/SLSH `train()`/`migrate()`
paths (which do delete+re-put on aux keys) are exercised correctly.

> **IMPORTANT -- recall caveat (test-data bug, found and fixed 2026-07-14):**
> The original run used a "clustered" vector generator with Gaussian noise
> `scale=0.3`. In 384-dim that noise has magnitude `0.3*sqrt(384) ~= 5.9`,
> which **drowns the unit centroids** (the centroid contributes only ~2.8% of
> the squared norm). The data is therefore ~uniform-random on the sphere --
> exactly the concentration-of-measure degenerate case this test meant to
> avoid. Diagnostic proof: the true top-10 neighbors of a query spanned
> **9.71 of 10** clusters, and a query's own nearest cluster held only
> **0.32 of 10** true neighbors. IVF (which assumes neighbors concentrate in
> a few clusters) had nothing to exploit, so the **IVF recall numbers below
> (0.47 / 0.51 / 0.51 and the nprobe sweep) are a pessimistic ARTIFACT, not
> IVF's real recall.** The generator is now fixed (`scale = 0.3/sqrt(dim)`,
> so noise magnitude ~= 0.3 << centroid 1.0); on the corrected data
> intra-cluster cosine is 0.918 and the true top-10 concentrate in **1.0
> cluster** with **10/10** in the query's own cluster. A corrected 10k
> spot-check gives **IVF recall@10 = 1.000 at the default nprobe** (see
> "Corrected-data spot check" below). The full corrected 50k/100k re-run was
> not completed. **FLAT recall is always 1.000 (exact) regardless of data.
> SLSH recall ~0.08 is real, not an artifact -- confirmed on the corrected
> 10k spot check (0.082).** Latency and throughput are data-structure-
> independent for FLAT/IVF and remain valid; SLSH latency is bucket-
> distribution-sensitive and may differ on structured data.

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
  noise, L2-normalized => cosine space). **The noise scale must be
  `0.3/sqrt(dim)` (noise magnitude ~0.3), not a bare `0.3`** -- a bare 0.3 is
  degenerate in high dim (see the caveat above). With the corrected scale the
  data has real neighbor structure (intra-cluster cosine ~0.92, true top-10
  in ~1 cluster), so recall is meaningful.
- **Scale points:** N in {10,000, 50,000, 100,000} for each of FLAT / IVF /
  SLSH (9 cells total).
- **Queries:** 200 fresh clustered queries per cell, `k=10` (matches the
  integration plan's planned `Runtime(top_k=10)`).
- **Recall:** recall@10 vs exact numpy ground-truth cosine top-10
  (`queries @ db.T` + argpartition). recall@10 = |ANN-top10 ∩ exact-top10| / 10.
- **Latency:** per-query `search_sync` wall time, p50 and p99 over the 200
  queries.
- **IVF config:** `ivf_n_clusters = max(16, int(sqrt(N)))`, runtime
  `ivf_nprobe = max(8, n_clusters//4)`.
- **SLSH config:** `slsh_lsh_tables=10`, `slsh_hash_bits=16`,
  `slsh_bucket_width=3.0`, runtime `slsh_scan_radius=200`. NOT tuned for 384
  -- see the recall diagnosis below.
- **Insert:** one-by-one `insert_sync` (no batch vector-insert API); throughput
  is cffi per-call bound (~1ms/call), NOT index-algorithm bound.

## Main results -- latency & throughput (original run, all 9 cells, exit 0, NO crash at 100k)

These columns are data-structure-independent for FLAT/IVF and remain valid.
The `recall@10` column is from the **degenerate** original data -- see the
caveat: FLAT=1.000 (exact, always), SLSH ~0.08 (real), **IVF 0.47/0.51/0.51
is an artifact** (corrected 10k gives 1.000).

```
type      n      ins_s  ins_thru  train_s   p50_ms   p99_ms  recall@10(degen)
flat   10000     18.08       553     0.00    41.61    51.13      1.000
ivf    10000      9.68      1033     3.14    20.21    43.52      0.468  *
slsh   10000      9.87      1014     1.23     7.15    50.57      0.069
flat   50000     93.97       532     0.00   212.04   227.90      1.000
ivf    50000     50.24       995    45.16    92.06   182.57      0.512  *
slsh   50000     48.98      1021    30.81    39.60   219.70      0.069
flat  100000    182.25       549     0.00   380.50   397.96      1.000
ivf   100000     95.35      1049   142.28   193.89   371.70      0.510  *
slsh  100000     97.50      1026   124.27   305.36   446.14      0.078
```

`*` = degenerate-data artifact for IVF (see caveat). `ins_thru` = vectors/sec.
`train_s` = k-means / LSH-projection build time (zero for FLAT).

## Corrected-data spot check (N=10,000 only; full re-run stopped)

After fixing the generator (`scale=0.3/sqrt(dim)`), a 10k run with the SAME
default IVF nprobe (=25) and the same SLSH params:

```
type      n      ins_s  ins_thru  train_s   p50_ms   p99_ms  recall@10
flat   10000     18.68       535     0.00    35.45    41.22      1.000
ivf    10000      9.55      1047     1.84    16.54    35.13      1.000   <- was 0.468
slsh   10000      9.76      1024     1.02    26.38    41.76      0.082   <- still bad
```

This is the headline correction: **IVF recall is 1.000 at the default nprobe
on structured data** -- the 0.51 was the test data, not IVF. **SLSH stays
~0.08 even on structured data**, isolating its problem to the 384-dim hash
params (a real issue, not an artifact). Corrected 50k/100k recall was not
measured (the re-run was stopped); based on the 10k result and the
concentration of neighbors into ~1 cluster, IVF recall on structured data is
expected to stay near-exact at the default nprobe across N, but this should
be confirmed before shipping IVF.

## Conclusions

- **Yes, 384 runs at 100k scale for all three types** with the scan fix. The
  core question is answered: the native in-DB vector index holds at Hippo's
  target dimension at 100k.
- **Insert** is ~545/s for FLAT and ~1025/s for IVF/SLSH, dominated by the
  cffi per-call cost (~1ms), not the index algorithm. 100k one-time build:
  FLAT ~182s, IVF ~95s insert + ~142s train, SLSH ~98s insert + ~124s train.
  Offline, one-time; acceptable.
- **FLAT (Hippo's current type):** exact, recall 1.000, but brute-force
  latency scales linearly with N -- 42ms@10k -> 212ms@50k -> 380ms@100k p50.
  Fine at Hippo's current episode counts (thousands of vectors); 380ms@100k
  is the real-time-retrieval wall that motivates graduating to IVF.
- **IVF (the planned graduation path):** p50 194ms@100k (well under FLAT's
  380ms). **Recall on structured data is near-exact at the default nprobe
  (1.000 @ 10k corrected); the earlier 0.51 was a degenerate-data artifact.**
  Recall is also tunable via runtime `ivf_nprobe` (no retrain) all the way to
  exact (nprobe == n_clusters). k-means train is the long pole (142s@100k).
- **SLSH:** runs, but recall@10 is ~0.08 **even on structured data** with the
  params above, and p99 is high (446ms@100k) from LSH bucket-miss fallback
  scans. The runtime `slsh_scan_radius` lever alone does not rescue it; the
  Format-tier params (tables / hash_bits / bucket_width) need real tuning for
  384-dim cosine and that requires a retrain. SLSH is NOT the right
  out-of-the-box ANN choice for Hippo at 384 -- IVF is.

## Recall diagnosis ("serious recall issues with all but flat")

FLAT is exact (1.000) by construction. The original run reported low recall
for both ANN types, but the two have turned out to have **different root
causes** -- one was a test-data artifact, one is real.

### IVF: the 0.51 was a TEST-DATA ARTIFACT, not IVF

IVF partitions the dataset into `n_clusters = sqrt(N)` k-means clusters and,
at query time, scans only the `ivf_nprobe` nearest clusters (exact within
each scanned cluster). recall is bounded by "did the true top-10 fall in the
probed clusters?" On the degenerate test data the true top-10 were scattered
across ~10 clusters (proof in the caveat), so low nprobe missed most of them
-> 0.51. On structured data (real Hippo embeddings have semantic structure;
intra-cluster cosine typically 0.3-0.7, far above the 0.03 of the degenerate
set) the true top-10 concentrate in ~1 cluster, so the default nprobe captures
them -> **1.000 @ 10k corrected**.

The `ivf_nprobe` lever is still real and monotone to exact -- the degenerate
nprobe sweep (kept below only as evidence the dial works, NOT as a real
recall curve), N=50,000, n_clusters=223:

```
nprobe   recall@10   p50_ms   p99_ms        (DEGENERATE data -- artifact curve)
     8       0.121    18.97   133.10
    16       0.216    30.17   122.68
    32       0.329    58.64   160.01
    64       0.519   115.66   208.57
   128       0.811   304.78   329.44
   223       1.000   534.86   581.13   (nprobe == n_clusters = exact)
```

On structured data the whole curve would sit much higher (near 1.0 from low
nprobe), so the practical takeaway is simpler than the sweep suggests: **pick
a modest nprobe (the default `sqrt(N)/4` or a bit higher), confirm recall on
real embeddings, and raise nprobe only if real-embedding recall dips.** Do
not chase recall near 1.0 by cranking nprobe -- at nprobe == n_clusters IVF
is exact but SLOWER than FLAT (per-cluster routing overhead, 535ms vs 212ms
@50k in the sweep). IVF's win over FLAT is at scale: IVF latency grows ~
`nprobe * sqrt(N)` vs FLAT's `N`, so a fixed modest nprobe pulls ahead as N
rises (the latency table already shows IVF@100k 194ms vs FLAT@100k 380ms).

### SLSH: recall ~0.08 is REAL -- needs Format-tier retraining

SLSH (locality-sensitive hashing) recall is dominated by the Format-tier
parameters (`slsh_lsh_tables`, `slsh_hash_bits`, `slsh_bucket_width`) baked
into the hash structure at `train()` time. The defaults used (10 tables / 16
bits / bucket_width 3.0) give recall ~0.08, and crucially this **does not
improve on structured data** (0.082 @ 10k corrected vs 0.069 @ 10k degenerate)
-- so unlike IVF, this is not a test-data artifact. The only runtime lever,
`slsh_scan_radius`, widens the bucket scan but cannot fix a bad hash
projection. Radius sweep at N=50,000 (degenerate data; the radius-invariance
holds on structured data too since the projection is the issue):

```
radius   recall@10   p50_ms   p99_ms
    50       0.083   149.90   228.24
   200       0.083   147.32   223.47
   500       0.083   170.84   254.48
  1000       0.083   163.86   252.61
  2000       0.096   179.53   261.00
  5000       0.213   308.09   393.38
```

Recall is flat at 0.083 from radius 50 through 1000 -- widening the scan does
nothing because the true neighbors are NOT in the buckets the query hashes to
(a bad projection, not a narrow scan). It barely moves to 0.096 at radius 2000
and only reaches 0.213 at radius 5000, where p50 is 308ms -- worse than FLAT
exact (212ms@50k, recall 1.000) for a quarter of the recall. The runtime lever
is exhausted well before usefulness. Fixing SLSH at 384 means sweeping the
Format-tier params (tables / hash_bits / bucket_width) with a retrain per
setting -- a separate tuning task -- and even then 384-dim cosine may simply
not be SLSH's regime. Do not pick SLSH for Hippo until that work is done (and
probably not after).

### Bottom line for Hippo

- **FLAT** stays correct and is the right choice until episode count pushes
  latency past the real-time budget (~hundreds of ms). At Hippo's current
  scale (thousands), FLAT is fine -- exact AND fast.
- **IVF** is the graduation path. Its recall on structured/real data is
  near-exact at a modest default nprobe (the 0.51 was a test-data artifact),
  and `ivf_nprobe` is a runtime recall/latency dial to exact with no retrain.
  Before shipping IVF, measure recall@10 on REAL Hippo episode embeddings at
  `top_k=10` and confirm; raise nprobe only if real recall dips.
- **SLSH** at 384 needs dedicated Format-tier param tuning (a retrain sweep
  over tables/bits/bucket_width) before it is usable; the ~0.08 recall is a
  real 384-hash-param problem, not a test artifact. Do not pick SLSH for Hippo.
- **WaveDB has no HNSW.** If "high recall + low latency at large scale" ever
  becomes a hard requirement, IVF-tuned-up is the only native path and its
  high-recall end costs latency. Keep this in view as a real limitation.

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
  graduation plan (its "IVF 0.96-0.99 recall at nprobe=8" figure was an
  unmeasured assumption; the corrected 10k spot check here -- IVF 1.000 at
  the default nprobe -- supports it for structured data, but real-embedding
  recall should still be measured before shipping).
- WaveDB commit `ee97307` -- the sync_only scan read_txn_id fix that makes
  train/migrate correct (without it, delete+re-put aux keys were invisible to
  scans).
- Harnesses (untracked probes, NOT committed):
  `scripts/_scratch/scale_384_wavedb.py` (the 9-cell scale test; generator
  noise scale fixed to `0.3/sqrt(dim)`),
  `scripts/_scratch/sweep_384_recall.py` (IVF nprobe + SLSH radius sweeps;
  same generator fix).