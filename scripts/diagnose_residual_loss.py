"""Find the specific keys lost during a sorted full reload with the fixed DLL."""
import sys
from pathlib import Path
import wavedb

SRC = Path("data/pod_runs/phase1b_scale/ingest_db_dialogsum")
DST = Path("data/pod_runs/phase1b_scale/ingest_db_dialogsum_compact_sorted_diag")

src_db = wavedb.WaveDB(str(SRC))
items = [(k, v) for k, v in src_db.create_read_stream(start="", end="\xff")]
src_db.close()
print(f"src keys: {len(items)}", flush=True)

# sort by key; write real values (matches the 8-loss condition)
items.sort(key=lambda kv: kv[0])
print(f"items sorted: {len(items)}", flush=True)

import shutil
if DST.exists():
    shutil.rmtree(DST)
dst_db = wavedb.WaveDB(str(DST))
ops = []
for k, v in items:
    ops.append({"type": "put", "key": k, "value": v if v is not None else b""})
    if len(ops) >= 1024:
        dst_db.batch_sync(ops)
        ops = []
if ops:
    dst_db.batch_sync(ops)
dst_db.close()
print("wrote + closed dst", flush=True)

# reopen and scan
dst_db2 = wavedb.WaveDB(str(DST))
dst_keys = set(k for k, _ in dst_db2.create_read_stream(start="", end="\xff"))
dst_db2.close()
print(f"dst scan after reopen: {len(dst_keys)}", flush=True)

src_keyset = set(k for k, _ in items)
missing = src_keyset - dst_keys
extra = dst_keys - src_keyset
print(f"missing: {len(missing)}", flush=True)
print(f"extra: {len(extra)}", flush=True)
print("\n--- MISSING KEYS (first 30) ---", flush=True)
for k in sorted(missing)[:30]:
    print(repr(k), flush=True)
# group missing by index prefix
from collections import Counter
pref = Counter()
for k in missing:
    parts = k.split("/")
    if len(parts) >= 3:
        pref[parts[1] + "/" + parts[2]] += 1
    else:
        pref["/".join(parts[:2])] += 1
print("\n--- missing by index ---", flush=True)
for p, c in pref.most_common():
    print(f"  {p}: {c}", flush=True)
print("DONE", flush=True)