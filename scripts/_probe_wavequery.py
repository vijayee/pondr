"""Leak discriminator: replicate extract_subgraph's WaveDB graph-query reads
BUT discard results (no nodes/edges dict, no torch). Isolates the WaveDB
graph-query / scan-iterator C path from Hippo's dict/tensor construction.

If RSS climbs ~150 MB/build here  -> leak is in WaveDB graph-query/scan path.
If RSS is flat here                -> leak is in Hippo (extract_subgraph dict
                                    or feature_for / torch tensors), not WaveDB.
"""
import gc, resource, sys, os
from collections import deque

sys.path.insert(0, "/root/hippo")
from src.memory.store import HippocampalStore  # noqa: E402
from src.training.oracle_labeling import (  # noqa: E402
    OracleLabelingPipeline, sample_episode_centers,
)


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def _bfs_discard(pipe, center_id, radius=3):
    """Same reads as extract_subgraph, but builds NO dict/edges. Returns count."""
    visited = set()
    queue = deque([(center_id, 0)])
    count = 0
    while queue:
        node_id, depth = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        count += 1
        if depth >= radius:
            continue
        for nb, _pred, _direction in pipe._get_neighbors(node_id):
            if nb not in visited:
                queue.append((nb, depth + 1))
    return count


def _libwavedb_base():
    """Print libwavedb.so load base from /proc/self/maps for ASAN addr2line."""
    try:
        for line in open("/proc/self/maps"):
            if "libwavedb" in line and "r-xp" in line:
                base = line.split("-")[0]
                print(f"LIBWAVEDB_BASE=0x{base}", flush=True)
                return int(base, 16)
    except Exception as e:
        print(f"LIBWAVEDB_BASE_ERR={e}", flush=True)
    return None


def main():
    db = "/root/data/db/ingest_db_dialogsum_backfilled"
    cap = int(os.environ.get("CAP", "20"))
    _libwavedb_base()
    store = HippocampalStore(db)
    pipe = OracleLabelingPipeline(store)
    centers = sample_episode_centers(store)
    gc.collect()
    print(f"wavequery: {len(centers)} centers radius=3 cap={cap}", flush=True)
    print(f"wavequery: baseline rss={_rss_mb()}MB", flush=True)
    for i in range(cap):
        c = centers[i % len(centers)]
        n = _bfs_discard(pipe, c)
        gc.collect()
        print(f"[build {i+1:3d}] center={c} nodes={n} rss={_rss_mb()}MB",
              flush=True)
    store.close()


if __name__ == "__main__":
    main()