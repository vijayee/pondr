"""Decisive reclamation test.

If page-file stale-region reclamation is wired up, repeatedly OVERWRITING the
same keys (same paths, new values) keeps the file ~flat — old CoW copies get
reused. If reclamation is dead code (page_file_get_reusable_blocks has no
callers), each overwrite CoW-appends fresh bytes and the file grows ~linearly
with the number of overwrite passes. Scratch — not committed.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import wavedb

base = Path(tempfile.mkdtemp(prefix="hippo_reclaim_"))
db = wavedb.WaveDB(str(base))
N = 500
keys = [f"memory/spo/ep_{i:06d}/has_entity/entity_0" for i in range(N)]

db.put_sync(keys[0], b"seed")
for i, k in enumerate(keys):
    db.put_sync(k, f"v{i}".encode())
db.close()
sz = (base / "data.wdbp").stat().st_size
print(f"after initial write of {N} keys: {sz/1024/1024:.2f} MB")

for rep in range(1, 6):
    db = wavedb.WaveDB(str(base))
    for i, k in enumerate(keys):
        db.put_sync(k, f"overwrite-{rep}-{i}".encode())
    db.close()
    sz = (base / "data.wdbp").stat().st_size
    print(f"after overwrite pass {rep}: {sz/1024/1024:.2f} MB")

sz_final = (base / "data.wdbp").stat().st_size
print(f"\nfinal {sz_final/1024/1024:.2f} MB for {N} live keys "
      f"=> {(sz_final / N / 1024):.1f} KB/key after 5 overwrites")
print("flat => reclamation works; ~linear growth => reclamation is dead code")
shutil.rmtree(base)