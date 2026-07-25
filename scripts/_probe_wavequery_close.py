"""Per-db vs process-global discriminator for the graph-query leak.

Build up RSS with N graph-query builds, then CLOSE the store + gc + malloc_trim,
and measure the drop. Decides:
  - big drop after close+trim  -> leak is per-db (freed at database_destroy;
    glibc was retaining it; a close/trim workaround bounds it). NOT a true
    process-global C leak.
  - no drop after close+trim    -> true process-global never-freed C allocation.
"""
import gc, resource, sys, ctypes
from collections import deque

sys.path.insert(0, "/root/hippo")
from src.memory.store import HippocampalStore  # noqa: E402
from src.training.oracle_labeling import (  # noqa: E402
    OracleLabelingPipeline, sample_episode_centers,
)

DB = "/root/data/db/ingest_db_dialogsum_backfilled"
libc = ctypes.CDLL("libc.so.6")
libc.malloc_trim.argtypes = [ctypes.c_size_t]
libc.malloc_trim.restype = ctypes.c_int


def rss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def bfs(pipe, c, radius=3):
    visited, q, n = set(), deque([(c, 0)]), 0
    while q:
        nid, d = q.popleft()
        if nid in visited:
            continue
        visited.add(nid); n += 1
        if d >= radius:
            continue
        for nb, _p, _dir in pipe._get_neighbors(nid):
            if nb not in visited:
                q.append((nb, d + 1))
    return n


def build(store):
    pipe = OracleLabelingPipeline(store)
    centers = sample_episode_centers(store)
    for i in range(6):
        bfs(pipe, centers[i % len(centers)])


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    store = HippocampalStore(DB)
    gc.collect()
    print(f"close-test: baseline rss={rss()}MB (N={N} builds before close)", flush=True)
    for k in range(N):
        build(store)
        gc.collect()
        print(f"  after {k+1} build-groups: rss={rss()}MB", flush=True)
    rss_before = rss()
    print(f"close-test: rss BEFORE close = {rss_before}MB", flush=True)
    store.close()
    gc.collect()
    t = libc.malloc_trim(0)
    gc.collect()
    libc.malloc_trim(0)
    rss_after = rss()
    print(f"close-test: rss AFTER close+gc+trim = {rss_after}MB (trim_rc={t})", flush=True)
    print(f"close-test: DROP = {rss_before - rss_after}MB "
          f"({'PER-DB (freed at close)' if rss_before - rss_after > 50 else 'PROCESS-GLOBAL (never freed)'})",
          flush=True)


if __name__ == "__main__":
    main()