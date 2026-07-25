"""Probe: is MigrateSlshToSlsh failure a real migrate bug or bad small-scale SLSH config?

Compares SLSH search results across:
  A. fresh SLSH (insert + train, no migrate)
  B. SLSH after same-type migrate (insert + train + migrate(SLSH))
  C. SLSH after cross-type migrate (insert + train IVF + migrate(IVF->SLSH))
  D. SLSH self-query (query == a stored vector; top-1 should be itself, dist 0)

Mirrors C slsh_config: lsh_tables=2, hash_bits=8, bucket_width=1.0, L2, scan_radius=100.
Mirrors C vl_mtest_insert: 3 centers {10,0,0,0},{0,10,0,0},{0,0,10,0}, noise +-0.1, srand(7).
"""
import os
import random
import tempfile

from wavedb import Distance, Format, IndexType, Runtime, VectorLayer, set_quiet

set_quiet(True)

CENTERS = [(10.0, 0.0, 0.0, 0.0), (0.0, 10.0, 0.0, 0.0), (0.0, 0.0, 10.0, 0.0)]


def slsh_fmt(dim=4):
    return Format(index_type=IndexType.SLSH, dim=dim, distance=Distance.L2,
                  slsh_lsh_tables=2, slsh_hash_bits=8, slsh_bucket_width=1.0)


def slsh_rt():
    return Runtime(top_k=10, sync_only=1, slsh_scan_radius=100)


def insert_clustered(vl, n=60):
    random.seed(7)
    stored = {}
    for i in range(n):
        c = i % 3
        v = [CENTERS[c][d] + random.uniform(-0.1, 0.1) for d in range(4)]
        vl.insert_sync(f"v{i}", v)
        stored[f"v{i}"] = v
    return stored


def search(vl, q, k=5):
    res = vl.search_sync(list(q), k)
    return [(r.id_str, round(r.distance, 4)) for r in res]


def case(label, fn):
    d = tempfile.mkdtemp(prefix="vlprobe_")
    path = os.path.join(d, "vl")
    try:
        out = fn(path)
    finally:
        pass
    print(f"{label}: {out}")


def A_fresh(path):
    vl = VectorLayer.open_separate(path, "t", slsh_fmt(), slsh_rt())
    insert_clustered(vl)
    vl.train()
    n = vl.count()
    r1 = search(vl, CENTERS[0])           # near cluster 0 center
    r2 = search(vl, CENTERS[0], k=1)
    vl.close()
    return f"count={n} near_c0(k5)={r1} near_c0(k1)={r2}"


def B_same_type_migrate(path):
    vl = VectorLayer.open_separate(path, "t", slsh_fmt(), slsh_rt())
    insert_clustered(vl)
    vl.train()
    pre = search(vl, CENTERS[0])
    vl.migrate(slsh_fmt())                  # SLSH -> SLSH same type
    post_count = vl.count()
    post = search(vl, CENTERS[0])
    vl.close()
    return f"pre_migrate={pre} post_count={post_count} post_migrate={post}"


def C_cross_into_slsh(path):
    vl = VectorLayer.open_separate(
        path, "t",
        Format(index_type=IndexType.IVF, dim=4, distance=Distance.L2, ivf_n_clusters=3),
        Runtime(top_k=10, sync_only=1, ivf_nprobe=3, ivf_flat_until=1000))
    insert_clustered(vl)
    vl.train()
    vl.migrate(slsh_fmt())                  # IVF -> SLSH
    vl.reconfigure(slsh_rt())
    post_count = vl.count()
    post = search(vl, CENTERS[0])
    vl.close()
    return f"post_count={post_count} post_migrate_search={post}"


def D_self_query(path):
    vl = VectorLayer.open_separate(path, "t", slsh_fmt(), slsh_rt())
    stored = insert_clustered(vl)
    vl.train()
    q = stored["v10"]
    r = search(vl, q, k=1)                   # exact stored vec -> top-1 should be v10
    vl.close()
    return f"self_query_v10(k1)={r}"


if __name__ == "__main__":
    case("A fresh SLSH        ", A_fresh)
    case("B SLSH->SLSH migrate ", B_same_type_migrate)
    case("C IVF->SLSH migrate  ", C_cross_into_slsh)
    case("D SLSH self-query     ", D_self_query)