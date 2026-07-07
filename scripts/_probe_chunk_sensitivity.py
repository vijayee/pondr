"""Chunk-size / delimiter / node-size sensitivity probe.

Isolates whether the ~24.5 KB/key cost for divergent keys is:
  (A) too many chunk levels (chunk_size=4 → ~10 levels for a 40-byte key,
      each its own bnode), so a LARGER chunk_size collapses levels and drops
      cost; or
  (B) each bnode consumes a full 4KB block regardless of size, so chunk_size
      changes nothing and only btree_node_size / block packing matters.

Matrix (500 divergent-prefix keys each, one batch_sync + close):
  - chunk=4  delim='/'  node=4096   (baseline = current Hippo config)
  - chunk=16 delim='/'  node=4096
  - chunk=64 delim='/'  node=4096
  - chunk=4  delim=''   node=4096    (no delimiter → whole key chunked as one path)
  - chunk=64 delim=''   node=4096
  - chunk=4  delim='/'  node=65536  (16x bigger B+tree nodes)
  - chunk=4  delim='/'  node=4096, FLAT keys (no path structure) — control

Scratch — not committed.
"""

import shutil
import tempfile
from pathlib import Path

import wavedb
from wavedb import WaveDB, WaveDBConfig

N = 500
KEYS_DIV = [f"memory/spo/ep_{i:06d}/has_entity/entity_0000" for i in range(N)]
KEYS_FLAT = [f"k{i:06d}" for i in range(N)]
VALS = [f"value-{i}".encode() for i in range(N)]


def measure(label: str, keys: list[str], chunk_size: int, delimiter: str,
            btree_node_size: int) -> float:
    base = Path(tempfile.mkdtemp(prefix="hippo_cs_"))
    cfg = WaveDBConfig(chunk_size=chunk_size, btree_node_size=btree_node_size)
    db = WaveDB(str(base), delimiter=delimiter, config=cfg)
    ops = [{"type": "put", "key": k, "value": v} for k, v in zip(keys, VALS)]
    db.batch_sync(ops)
    db.close()
    sz = (base / "data.wdbp").stat().st_size
    per_key = sz / N
    shutil.rmtree(base)
    print(f"{label:<34}  {sz/1024/1024:>8.2f} MB  {per_key/1024:>6.1f} KB/key  "
          f"~{per_key/4096:.1f} pages/key")
    return sz


print("Divergent-prefix keys (memory/spo/ep_XXX/has_entity/entity_0000):")
measure("chunk=4 delim='/' node=4096 [BASE]", KEYS_DIV, 4, "/", 4096)
measure("chunk=16 delim='/' node=4096", KEYS_DIV, 16, "/", 4096)
measure("chunk=64 delim='/' node=4096", KEYS_DIV, 64, "/", 4096)
measure("chunk=4 delim='' node=4096", KEYS_DIV, 4, "", 4096)
measure("chunk=64 delim='' node=4096", KEYS_DIV, 64, "", 4096)
measure("chunk=4 delim='/' node=65536", KEYS_DIV, 4, "/", 65536)
print()
print("Flat keys (control, no path structure):")
measure("chunk=4 delim='/' node=4096 FLAT", KEYS_FLAT, 4, "/", 4096)
measure("chunk=64 delim='' node=4096 FLAT", KEYS_FLAT, 64, "", 4096)
print()
print("Reading: if chunk_size matters → (A) chunk-levels; if only node_size matters → (B) block-per-bnode.")