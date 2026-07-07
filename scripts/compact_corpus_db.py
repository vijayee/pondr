"""Compact a bloated WaveDB corpus DB by reloading every key into a fresh DB.

Reads an existing (write-amplified) WaveDB database and writes every key/value
into a NEW database created with the current WaveDB (sub-block packing +
64-bit offsets + cross-hbtrie dirty-propagation flush fix). The logical
content is identical; only the on-disk layout is compacted.

Why SORT before inserting:
  The source DB was built by episode-order ingestion (non-monotonic key
  order), which triggers WaveDB's insertion-order-dependent split bug (the
  right sibling of a leaf split is orphaned — scan-visible via sibling
  pointers but not flush-collected, and the scan also emits duplicate keys).
  Sorting the keys lexicographically before `batch_sync` makes insertion
  monotonic, which avoids the split orphan entirely, and duplicate stream
  entries collapse automatically (same-key puts overwrite). The destination
  therefore contains exactly the source's UNIQUE (key, value) set, which is
  the faithful reload of "all the old database's values."

Verification (must all pass before the source is touched):
  - dst unique (key, value) set == source unique (key, value) set
    (no missing, no extra) — compared via create_read_stream on both sides
  - no NUL bytes in any scanned dst key (the Hippo scan gate)
  - dst reopens cleanly (close + reopen + scan count stable)
  - dst size reported alongside source size

Usage:
    python scripts/compact_corpus_db.py \\
        --src data/pod_runs/phase1b_scale/ingest_db_dialogsum \\
        --dst data/pod_runs/phase1b_scale/ingest_db_dialogsum_compact \\
        --swap

--swap (after successful verification) replaces the source directory with the
compact one (src renamed to *_bloated_backup, dst renamed to src). Without
--swap it leaves both in place for manual inspection.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import wavedb


def _stream_items(db) -> list[tuple[str, bytes]]:
    return [(k, v) for k, v in db.create_read_stream(start="", end="\xff")]


def reload(src: Path, dst: Path, batch_size: int) -> tuple[int, int, int]:
    """Copy every (key, value) from src DB into a fresh dst DB, SORTED by key.

    Returns (n_stream_entries, n_unique_keys, n_value_bytes).
    """
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    src_db = wavedb.WaveDB(str(src))
    items = _stream_items(src_db)
    src_db.close()
    n_stream = len(items)

    # Sort by key lexicographically. Duplicate stream entries (a symptom of
    # the source's split corruption) collapse on insert (same key overwrites);
    # the destination ends up with the source's UNIQUE (key, value) set.
    items.sort(key=lambda kv: kv[0])
    n_unique = len({k for k, _ in items})

    dst_db = wavedb.WaveDB(str(dst))
    batch: list[dict] = []
    n = 0
    n_bytes = 0
    bi = 0
    for key, value in items:
        v = value if value is not None else b""
        batch.append({"type": "put", "key": key, "value": v})
        n_bytes += len(v)
        if len(batch) >= batch_size:
            try:
                dst_db.batch_sync(batch)
            except Exception as e:
                print(f"  batch_sync FAILED at batch #{bi} (n={n}, batch_size={len(batch)}): {e}", flush=True)
                dst_db.close()
                raise
            n += len(batch)
            batch.clear()
            bi += 1
            if n % (batch_size * 50) == 0:
                print(f"  reloaded {n} stream entries...", flush=True)
    if batch:
        try:
            dst_db.batch_sync(batch)
        except Exception as e:
            print(f"  batch_sync FAILED at final batch (n={n}, batch_size={len(batch)}): {e}", flush=True)
            dst_db.close()
            raise
        n += len(batch)
    dst_db.close()
    return n_stream, n_unique, n_bytes


def verify(src: Path, dst: Path) -> tuple[bool, dict]:
    """Verify dst holds exactly the source's unique (key, value) set.

    Compares via create_read_stream on both sides (NOT get_sync — get_sync is
    a separate broken path on DBs built by unsorted ingestion). Returns
    (ok, stats).
    """
    print("Verifying...")
    src_db = wavedb.WaveDB(str(src))
    dst_db = wavedb.WaveDB(str(dst))

    src_items = _stream_items(src_db)
    dst_items = _stream_items(dst_db)
    src_db.close()
    dst_db.close()

    src_kv = set((k, v) for k, v in src_items)
    dst_kv = set((k, v) for k, v in dst_items)
    missing = src_kv - dst_kv
    extra = dst_kv - src_kv

    stats = {
        "src_stream": len(src_items),
        "dst_stream": len(dst_items),
        "src_unique_kv": len(src_kv),
        "dst_unique_kv": len(dst_kv),
        "missing": len(missing),
        "extra": len(extra),
        "dups_in_src_stream": len(src_items) - len(src_kv),
    }
    print(f"  source stream entries:  {stats['src_stream']}")
    print(f"  source unique (k,v):    {stats['src_unique_kv']}")
    print(f"  source stream dups:     {stats['dups_in_src_stream']}")
    print(f"  compact stream entries: {stats['dst_stream']}")
    print(f"  compact unique (k,v):   {stats['dst_unique_kv']}")
    print(f"  (k,v) missing in compact: {stats['missing']}")
    print(f"  (k,v) extra in compact:   {stats['extra']}")

    ok = (stats["missing"] == 0 and stats["extra"] == 0)
    if not ok:
        for k, v in sorted(missing)[:10]:
            print(f"    MISSING: {k!r} -> {v!r}")
        for k, v in sorted(extra)[:10]:
            print(f"    EXTRA:   {k!r} -> {v!r}")

    # NUL-free scan gate (Hippo convention) on the compact DB.
    nul = [k for k, _ in dst_items if "\x00" in k]
    stats["nul_keys"] = len(nul)
    print(f"  NUL-free scan gate: {'PASS' if not nul else 'FAIL'} ({len(nul)} NUL keys)")
    ok = ok and not nul
    return ok, stats


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def swap(src: Path, dst: Path) -> None:
    """Replace src with dst; keep the old src as *_bloated_backup."""
    backup = src.with_name(src.name + "_bloated_backup")
    if backup.exists():
        shutil.rmtree(backup)
    os.rename(src, backup)
    os.rename(dst, src)
    print(f"  source moved to backup: {backup}")
    print(f"  compact DB now at:      {src}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="source (bloated) WaveDB dir")
    ap.add_argument("--dst", required=True, type=Path, help="destination (compact) WaveDB dir")
    ap.add_argument("--batch-size", type=int, default=512, help="keys per batch_sync (default 512; larger values can exceed the C batch buffer with large value payloads)")
    ap.add_argument("--swap", action="store_true", help="after verify, replace src with dst (keep backup)")
    args = ap.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    if not (src / "data.wdbp").exists():
        print(f"ERROR: no data.wdbp in src {src}", file=sys.stderr)
        return 2

    src_size_before = _dir_size(src)
    print(f"Source: {src}")
    print(f"  source size: {src_size_before:,} bytes ({src_size_before/1024/1024:.1f} MB)")

    print("Reloading keys (SORTED) into compact DB...")
    n_stream, n_unique, n_bytes = reload(src, dst, args.batch_size)
    print(f"  read {n_stream} stream entries; {n_unique} unique keys; {n_bytes:,} value bytes written.")

    ok, stats = verify(src, dst)
    if not ok:
        print("VERIFICATION FAILED; not swapping. Compact DB left at dst for inspection.", file=sys.stderr)
        return 1

    dst_size = _dir_size(dst)
    ratio = src_size_before / dst_size if dst_size else float("inf")
    print(f"  compact size: {dst_size:,} bytes ({dst_size/1024/1024:.1f} MB)")
    print(f"  shrink ratio: {ratio:.1f}x  (was {src_size_before/1024/1024:.1f} MB -> {dst_size/1024/1024:.1f} MB)")

    if args.swap:
        print("Swapping compact DB into source location...")
        swap(src, dst)
    else:
        print(f"  --swap not set; compact DB left at {dst} for inspection.")

    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())