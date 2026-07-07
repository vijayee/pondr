"""Confirm the walk-from-start defect: 0-result range scans at increasing
keyspace positions should get monotonically slower if `database_scan_next`
walks in-order from the leftmost leaf (O(keys-before-prefix)) rather than
seeking (which would be O(log N) ~constant regardless of position).

Each probe scans a NON-existent key just past a real prefix's region, so the
scan returns ~0 results — isolating pure "cost to reach the start position"
with no result-visit confound. Sorted keyspace order for the graph indices:
__meta < content < memory/osp < memory/pos < memory/pso < memory/spo
(osp/pos/pso/spo alphabetical; content < memory; __meta < content).

If latency grows with position, the seek defect is proven end-to-end.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import HippocampalStore  # noqa: E402


def _succ(prefix: str) -> str:
    p = prefix[:-1] if prefix.endswith("/") else prefix
    return p + "0"


def _time_zero_result_scan(db, start: str, repeats: int) -> tuple[float, int]:
    """Median ms for a [start, succ(start)) scan returning ~0 keys."""
    end = _succ(start)
    samples = []
    last_count = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        n = 0
        for _k, _v in db.create_read_stream(start=start, end=end):
            n += 1
        samples.append(time.perf_counter() - t0)
        last_count = n
    return statistics.median(samples) * 1e3, last_count


def main() -> int:
    db_path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/pod_runs/phase1b_scale/ingest_db_dialogsum"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    store = HippocampalStore(db_path)
    try:
        db = store.db
        # Non-existent late-sorting keys just past each region -> ~0 results.
        positions = [
            ("__meta/z",        "earliest"),
            ("content/ep/z",     "early (before 5002 episodes)"),
            ("memory/osp/z",     "graph osp index"),
            ("memory/pos/z",     "graph pos index"),
            ("memory/pso/z",     "graph pso index"),
            ("memory/spo/z",     "graph spo index (LAST)"),
        ]
        print(f"DB: {db_path}   repeats={repeats}\n")
        print(f"{'position':<18}{'note':<32}{'median_ms':>12}{'results':>10}")
        print("-" * 72)
        for start, note in positions:
            med, count = _time_zero_result_scan(db, start, repeats)
            print(f"{start:<18}{note:<32}{med:>12.2f}{count:>10}")
        print("\nIf median_ms grows with position (early fast, spo slow), "
              "the iterator walks from the leftmost leaf — seek defect confirmed.")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())