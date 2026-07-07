"""Test UNSORTED reload with the 0.1.14 fixes (Bug A split-orphan + Bug B cross-hbtrie).
If both fixes hold, an unsorted reload is byte-faithful (no sort workaround needed)."""
import shutil
from pathlib import Path
import wavedb

SRC = Path("data/pod_runs/phase1b_scale/ingest_db_dialogsum")
DST = Path("data/pod_runs/phase1b_scale/ingest_db_dialogsum_compact_UNSORTED_test")

if DST.exists():
    shutil.rmtree(DST)

src_db = wavedb.WaveDB(str(SRC))
items = [(k, v) for k, v in src_db.create_read_stream(start="", end="\xff")]
src_db.close()
src_kv = set(items)
print(f"src stream entries: {len(items)}  unique (k,v): {len(src_kv)}", flush=True)

# NO sorting — insert in stream order (the original failing path)
dst_db = wavedb.WaveDB(str(DST))
batch = []
n = 0
for k, v in items:
    batch.append({"type": "put", "key": k, "value": v if v is not None else b""})
    if len(batch) >= 512:
        dst_db.batch_sync(batch)
        n += len(batch); batch.clear()
        if n % 25600 == 0:
            print(f"  wrote {n}...", flush=True)
if batch:
    dst_db.batch_sync(batch); n += len(batch)
dst_db.close()
print(f"wrote {n} (unsorted)", flush=True)

dst_db2 = wavedb.WaveDB(str(DST))
dst_items = [(k, v) for k, v in dst_db2.create_read_stream(start="", end="\xff")]
dst_db2.close()
dst_kv = set(dst_items)
missing = src_kv - dst_kv
extra = dst_kv - src_kv
nul = [k for k, _ in dst_items if "\x00" in k]
sz = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
print(f"dst stream: {len(dst_items)}  unique (k,v): {len(dst_kv)}", flush=True)
print(f"missing: {len(missing)}  extra: {len(extra)}  NUL: {len(nul)}", flush=True)
print(f"size: {sz/1024/1024:.1f} MB", flush=True)
print("BYTE-FAITHFUL" if (len(missing) == 0 and len(extra) == 0 and not nul) else "LOSS REMAINS", flush=True)
print("DONE", flush=True)