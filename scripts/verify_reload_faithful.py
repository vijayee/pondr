"""Verify the sorted reload is byte-faithful: source (k,v) set == dst (k,v) set."""
import sys
from pathlib import Path
from collections import Counter
import wavedb

SRC = Path("data/pod_runs/phase1b_scale/ingest_db_dialogsum")
DST = Path("data/pod_runs/phase1b_scale/ingest_db_dialogsum_compact_sorted_diag")

src_db = wavedb.WaveDB(str(SRC))
src_items = [(k, v) for k, v in src_db.create_read_stream(start="", end="\xff")]
src_db.close()

dst_db = wavedb.WaveDB(str(DST))
dst_items = [(k, v) for k, v in dst_db.create_read_stream(start="", end="\xff")]
dst_db.close()

print(f"src stream entries: {len(src_items)}", flush=True)
print(f"dst stream entries: {len(dst_items)}", flush=True)

# duplicate keys in source?
src_key_counts = Counter(k for k, _ in src_items)
dups = {k: c for k, c in src_key_counts.items() if c > 1}
print(f"src duplicate keys: {len(dups)} (total dup excess={sum(c-1 for c in dups.values())})", flush=True)
for k, c in list(dups.items())[:12]:
    print(f"  dup x{c}: {k!r}", flush=True)

src_kv = Counter((k, v) for k, v in src_items)
dst_kv = Counter((k, v) for k, v in dst_items)

# unique (k,v) comparison
src_unique = set(src_kv)
dst_unique = set(dst_kv)
print(f"src unique (k,v): {len(src_unique)}", flush=True)
print(f"dst unique (k,v): {len(dst_unique)}", flush=True)
print(f"(k,v) missing in dst: {len(src_unique - dst_unique)}", flush=True)
print(f"(k,v) extra in dst:   {len(dst_unique - src_unique)}", flush=True)

# NUL-free scan gate
nul = [k for k, _ in dst_items if "\x00" in k]
print(f"dst keys with NUL: {len(nul)}", flush=True)

# size
sz = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
src_sz = sum(f.stat().st_size for f in SRC.rglob("*") if f.is_file())
print(f"dst size: {sz/1024/1024:.1f} MB  src size: {src_sz/1024/1024:.1f} MB", flush=True)
print("DONE", flush=True)