"""Probe v2: isolate whether SLSH->SLSH migrate break is a delete-then-put-same-key
(WaveDB MVCC/tombstone) issue.

Hypothesis: migrate deletes hash/<lsh_key>/<id> then rebuild re-puts the IDENTICAL
key (srand(42) => identical projections => identical lsh_key => identical key). A
fresh put of a never-deleted key (a new vector inserted AFTER migrate) should be
searchable; the re-put (deleted-then-put) keys should NOT be, if the tombstone
shadows the re-put.

Also: train() TWICE on a fresh SLSH (no delete) -- 2nd train rebuild sees old hash
entries as already_present => no-op. Search must still work (sanity).
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


def insert_clustered(vl, n=60, seed=7):
    random.seed(seed)
    for i in range(n):
        c = i % 3
        v = [CENTERS[c][d] + random.uniform(-0.1, 0.1) for d in range(4)]
        vl.insert_sync(f"v{i}", v)


def search(vl, q, k=5):
    return [(r.id_str, round(r.distance, 4)) for r in vl.search_sync(list(q), k)]


def B2_migrate_then_fresh_insert(path):
    """SLSH->SLSH migrate, then insert a FRESH vector vNEW (never deleted) near c0,
    search near c0. If vNEW found but v0..v59 not -> re-put keys invisible."""
    vl = VectorLayer.open_separate(path, "t", slsh_fmt(), slsh_rt())
    insert_clustered(vl)
    vl.train()
    before = search(vl, CENTERS[0])
    vl.migrate(slsh_fmt())
    # Fresh insert AFTER migrate -- key vec/idx/hash/<lsh_key>/vNEW never deleted.
    vl.insert_sync("vNEW", list(CENTERS[0]))
    after = search(vl, CENTERS[0], k=10)
    vl.close()
    has_new = any(i == "vNEW" for i, _ in after)
    has_old = any(i.startswith("v") and i != "vNEW" for i, _ in after)
    return f"before={before} | after_migrate+freshInsert={after} | fresh_vNEW_found={has_new} old_found={has_old}"


def B3_train_twice_no_delete(path):
    """Fresh SLSH, train, train again (no migrate/delete). 2nd train rebuild sees
    old hash as already_present => no-op. Search must still work."""
    vl = VectorLayer.open_separate(path, "t", slsh_fmt(), slsh_rt())
    insert_clustered(vl)
    vl.train()
    s1 = search(vl, CENTERS[0])
    vl.train()  # idempotent no-op rebuild
    s2 = search(vl, CENTERS[0])
    vl.close()
    return f"after_train1={s1} after_train2={s2}"


def B4_rebuild_after_manual_rehash(path):
    """Fresh SLSH, train, then call rebuild() directly (no delete, no migrate).
    Rebuild scans old hash (present) -> all already_present -> no-op. Sanity."""
    vl = VectorLayer.open_separate(path, "t", slsh_fmt(), slsh_rt())
    insert_clustered(vl)
    vl.train()
    s1 = search(vl, CENTERS[0])
    # Manually delete all hash via the migrate path is internal; instead just re-migrate.
    vl.migrate(slsh_fmt())
    # Now insert a 2nd fresh vector and train to regen -- does a fresh train after
    # a same-type migrate (which already re-put same keys) find the fresh vector?
    vl.insert_sync("vFRESH", list(CENTERS[0]))
    s2 = search(vl, CENTERS[0], k=10)
    vl.close()
    return f"after_train1={s1} after_migrate+fresh={s2}"


def case(label, fn):
    d = tempfile.mkdtemp(prefix="vlprobe2_")
    print(f"{label}: {fn(os.path.join(d, 'vl'))}")


if __name__ == "__main__":
    case("B2 migrate+fresh insert ", B2_migrate_then_fresh_insert)
    case("B3 train twice (no del) ", B3_train_twice_no_delete)
    case("B4 rebuild sanity       ", B4_rebuild_after_manual_rehash)