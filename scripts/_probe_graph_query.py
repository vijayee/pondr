"""Benchmark + bug-hunt probe for WaveDB graph queries (Phase 2a bottleneck).

Runs against the local surviving 1b DialogSum corpus via ``HippocampalStore``
so the access path matches Hippo exactly (``GraphLayer("memory", db)`` — graph
triples live under the ``memory/`` subtree, so raw keys are ``memory/spo/...``,
``memory/pso/...`` etc.).

Theories under test:

- **T3 (headline)** — one PSO range scan over ``memory/pso/follows/`` returns
  every ``(subject, object)`` edge for the ``follows`` predicate in a single
  pass. Compare its cost + result count to N per-vertex ``.out("follows")``
  queries over the same subjects. This is the Option-A speedup factor and
  confirms the batch-the-edges fix.
- **T1 (per-call overhead / cache reuse)** — warm repeated-identical query
  latency (same vertex, repeated) vs. distinct-prefix queries (each episode
  once). If repeated-identical is still ~hundreds of ms, the per-call MVCC
  snapshot + trie descent is NOT amortized across calls (points to per-call
  setup, not the scan).
- **T4 (component breakdown)** — for a fixed set of subjects, time
  query-build only, ``execute_sync`` only, ``.vertices`` marshal only,
  ``close`` only. Localizes the per-call cost.
- **T2 (range end-bound bug check)** — for a sample of subjects, compare the
  per-vertex ``.out("follows")`` count to the PSO-derived out-degree. Any
  mismatch means the SPO range scan is over- or under-shooting. Also dump
  the raw keys from one SPO range scan to eyeball whether anything outside
  ``/spo/<subject>/follows/`` leaks in.

The Hippo corpus is the test dataset (per the user's "we generate good test
data from this project"). This probe doubles as a reusable WaveDB graph
benchmark fixture.

Usage:
    python scripts/_probe_graph_query.py \\
        --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \\
        --sample 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import HippocampalStore  # noqa: E402

# The graph subtree prefix Hippo uses (store.py: self.graph = GraphLayer("memory", db)).
GRAPH_PREFIX = "memory"
DELIM = "/"


def _all_episode_ids(store: HippocampalStore) -> list[str]:
    """Episode ids that have content (same pattern as extract_backbone_sequences)."""
    ids: set[str] = set()
    start = "content/ep/"
    end = "content/ep/\x7f"
    for k, _ in store.db.create_read_stream(start=start, end=end):
        parts = k.split("/", 3)
        if len(parts) >= 3 and parts[2]:
            ids.add(parts[2])
    return sorted(ids)


def _successor_bound(prefix: str) -> str:
    """Replicate graph_ops.c ``append_successor``: strip trailing '/' then
    append '0' (0x30 > '/'=0x2F, so the bound is greater than any key sharing
    the prefix whose next byte is the delimiter or an ASCII component)."""
    p = prefix[:-1] if prefix.endswith(DELIM) else prefix
    return p + "0"


def _pso_scan_edges(store: HippocampalStore, predicate: str) -> list[tuple[str, str]]:
    """One PSO range scan over ``memory/pso/<predicate>/`` -> all (subject, object) edges.

    Keys are ``memory/pso/<predicate>/<subject>/<object>``; subject = component
    index 3 (after memory/pso/predicate), object = last component.
    """
    start = f"{GRAPH_PREFIX}/pso/{predicate}/"
    end = _successor_bound(start)
    edges: list[tuple[str, str]] = []
    for k, _ in store.db.create_read_stream(start=start, end=end):
        # k = "memory/pso/<predicate>/<subject>/<object>"
        parts = k.split(DELIM)
        # ["memory", "pso", predicate, subject, object]
        if len(parts) >= 5:
            edges.append((parts[3], parts[4]))
    return edges


def _out_scan_raw_keys(store: HippocampalStore, subject: str, predicate: str) -> list[str]:
    """Raw SPO range scan keys for ``/spo/<subject>/<predicate>/`` (debug: T2)."""
    start = f"{GRAPH_PREFIX}/spo/{subject}/{predicate}/"
    end = _successor_bound(start)
    keys: list[str] = []
    for k, _ in store.db.create_read_stream(start=start, end=end):
        keys.append(k)
    return keys


def _stats(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0}
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2]
    p99 = s[min(n - 1, int(n * 0.99))]
    return {"n": n, "min_ms": s[0] * 1e3, "median_ms": median * 1e3,
            "mean_ms": mean * 1e3, "max_ms": s[-1] * 1e3, "p99_ms": p99 * 1e3}


def _time_per_vertex_out(graph, subjects: list[str], predicate: str) -> tuple[list[float], list[int]]:
    """Per-vertex .out(predicate) via the real Hippo query path. Returns (latencies_s, counts)."""
    lat = []
    counts = []
    for s in subjects:
        t0 = time.perf_counter()
        r = graph.query().vertex(s).out(predicate).execute_sync()
        try:
            counts.append(len(r.vertices))
        finally:
            r.close()
        lat.append(time.perf_counter() - t0)
    return lat, counts


def main() -> int:
    ap = argparse.ArgumentParser(description="WaveDB graph-query benchmark + bug-hunt probe")
    ap.add_argument("--db", default="data/pod_runs/phase1b_scale/ingest_db_dialogsum")
    ap.add_argument("--predicate", default="follows",
                    help="Predicate to probe (follows = the 2a extract bottleneck)")
    ap.add_argument("--sample", type=int, default=500,
                    help="Max subjects to probe for per-vertex latency / T2 correctness")
    ap.add_argument("--repeat", type=int, default=50,
                    help="Iterations for the repeated-identical (warm cache) test")
    args = ap.parse_args()

    store = HippocampalStore(args.db)
    try:
        graph = store.graph
        all_ids = _all_episode_ids(store)
        print(f"DB: {args.db}")
        print(f"Episodes with content: {len(all_ids)}")
        if not all_ids:
            print("No episodes — aborting.")
            return 1

        # ── T3: PSO batch scan (one pass, all edges) ──
        t0 = time.perf_counter()
        edges = _pso_scan_edges(store, args.predicate)
        pso_s = time.perf_counter() - t0
        # Build adjacency from the batch (the Option-A in-memory structure).
        out_deg: dict[str, list[str]] = {}
        in_deg: dict[str, list[str]] = {}
        for s, o in edges:
            out_deg.setdefault(s, []).append(o)
            in_deg.setdefault(o, []).append(s)
        print(f"\n[T3] PSO batch scan over memory/pso/{args.predicate}/")
        print(f"     one scan: {pso_s*1e3:.1f} ms  -> {len(edges)} edges, "
              f"{len(out_deg)} distinct subjects, {len(in_deg)} distinct objects")

        # ── T3: per-vertex .out() over the SAME subjects ──
        sample_subjects = all_ids[: args.sample]
        # Warm up the layer (first query pays any one-time cost — graph_stats_compute etc.)
        _ = graph.query().vertex(sample_subjects[0]).out(args.predicate).execute_sync().close()
        lat, pv_counts = _time_per_vertex_out(graph, sample_subjects, args.predicate)
        pv_total = sum(lat)
        st = _stats(lat)
        print(f"\n[T3] Per-vertex .out({args.predicate}) over {len(sample_subjects)} subjects:")
        print(f"     total: {pv_total*1e3:.0f} ms   per-query: "
              f"median {st['median_ms']:.2f} ms  mean {st['mean_ms']:.2f} ms  "
              f"p99 {st['p99_ms']:.2f} ms  max {st['max_ms']:.2f} ms")
        if pso_s > 0:
            print(f"     *** SPEEDUP FACTOR (batch vs per-vertex): "
                  f"{pv_total / pso_s:.0f}x  ({pv_total*1e3:.0f} ms -> {pso_s*1e3:.1f} ms)")

        # ── T1: warm repeated-identical (same vertex, repeated) ──
        v = sample_subjects[0]
        rep_lat = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            r = graph.query().vertex(v).out(args.predicate).execute_sync()
            try:
                _ = r.vertices
            finally:
                r.close()
            rep_lat.append(time.perf_counter() - t0)
        rs = _stats(rep_lat)
        print(f"\n[T1] Repeated-identical query ({args.repeat}x, same vertex):")
        print(f"     median {rs['median_ms']:.3f} ms  mean {rs['mean_ms']:.3f} ms  "
              f"p99 {rs['p99_ms']:.3f} ms  max {rs['max_ms']:.3f} ms")
        print(f"     -> if this is still ~hundreds of ms, per-call snapshot/descent "
              f"is NOT amortized (per-call setup is the cost).")

        # ── T4: component breakdown over a fixed subject set ──
        fixed = sample_subjects[: min(200, len(sample_subjects))]
        # build only
        t0 = time.perf_counter()
        qs = [graph.query().vertex(s).out(args.predicate) for s in fixed]
        build_s = time.perf_counter() - t0
        # execute only
        t0 = time.perf_counter()
        results = [q.execute_sync() for q in qs]
        exec_s = time.perf_counter() - t0
        # marshal only
        t0 = time.perf_counter()
        _ = [r.vertices for r in results]
        marshal_s = time.perf_counter() - t0
        # close only
        t0 = time.perf_counter()
        for r in results:
            r.close()
        for q in qs:
            q.close()
        close_s = time.perf_counter() - t0
        n = len(fixed)
        print(f"\n[T4] Component breakdown over {n} subjects:")
        print(f"     build  : {build_s*1e3:.1f} ms  ({build_s/n*1e3:.3f} ms/query)")
        print(f"     execute: {exec_s*1e3:.1f} ms  ({exec_s/n*1e3:.3f} ms/query)  <- the scan + snapshot + optimizer")
        print(f"     marshal: {marshal_s*1e3:.1f} ms  ({marshal_s/n*1e3:.3f} ms/query)")
        print(f"     close  : {close_s*1e3:.1f} ms  ({close_s/n*1e3:.3f} ms/query)")

        # ── T2: end-bound correctness (per-vertex count == PSO degree) ──
        mismatches = 0
        checked = 0
        for s, pv_c in zip(sample_subjects, pv_counts):
            pso_c = len(out_deg.get(s, []))
            checked += 1
            if pv_c != pso_c:
                mismatches += 1
                if mismatches <= 5:
                    print(f"     MISMATCH subject={s}: per-vertex OUT={pv_c}  PSO-degree={pso_c}")
        print(f"\n[T2] End-bound correctness: {checked} subjects checked, "
              f"{mismatches} mismatches (per-vertex .out count vs PSO out-degree)")
        if mismatches == 0:
            print(f"     OK — SPO range scan returns exactly the PSO-derived degree "
                  f"(no end-bound overshoot/undershoot).")
        # Dump raw keys for one subject to eyeball range membership
        sample_subj = next((s for s in sample_subjects if len(out_deg.get(s, [])) > 0), sample_subjects[0])
        raw_keys = _out_scan_raw_keys(store, sample_subj, args.predicate)
        print(f"     Raw SPO keys for subject={sample_subj} ({len(raw_keys)}):")
        for k in raw_keys[:5]:
            in_range = k.startswith(f"{GRAPH_PREFIX}/spo/{sample_subj}/{args.predicate}/")
            print(f"       {'OK ' if in_range else 'LEAK'} {k}")
        if raw_keys:
            all_in = all(k.startswith(f"{GRAPH_PREFIX}/spo/{sample_subj}/{args.predicate}/") for k in raw_keys)
            print(f"     all {len(raw_keys)} keys in range: {all_in}")

        # ── Summary ──
        print(f"\n=== SUMMARY ===")
        print(f"PSO batch: {pso_s*1e3:.1f} ms for {len(edges)} edges")
        print(f"Per-vertex: {pv_total*1e3:.0f} ms for {len(sample_subjects)} queries "
              f"({st['median_ms']:.2f} ms median)")
        if pso_s > 0:
            print(f"Option-A speedup: {pv_total / pso_s:.0f}x")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())