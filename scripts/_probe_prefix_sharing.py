"""Isolate whether WaveDB's per-key cost is driven by trie-node (page) creation
along the key path — i.e. shared-prefix keys should be cheap, divergent-prefix
keys expensive.

If confirmed, the amplification is structural to the HBTrie (sparse pages per
key path), not MVCC. Scratch — not committed.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import wavedb


def measure(label: str, keys: list[str]) -> None:
    base = Path(tempfile.mkdtemp(prefix="hippo_pfx_"))
    db = wavedb.WaveDB(str(base))
    for k in keys:
        db.put_sync(k, b"v")
    db.close()
    sz = (base / "data.wdbp").stat().st_size
    per_key = sz / len(keys)
    print(f"{label:>22}  N={len(keys):>5}  data.wdbp={sz/1024/1024:>7.2f} MB  "
          f"per_key={per_key/1024:>6.1f} KB  (~{per_key/4096:.1f} pages)")
    shutil.rmtree(base)


N = 2000

# High shared prefix: all keys under one long common prefix, diverging only at
# the leaf (mimics many triples for ONE episode).
shared = [f"memory/spo/ep_000001/has_entity/entity_{i:04d}" for i in range(N)]
measure("shared-prefix (1 eps)", shared)

# Divergent prefix: each key under a unique episode id (mimics one triple per
# episode across many episodes — minimal prefix sharing).
divergent = [f"memory/spo/ep_{i:06d}/has_entity/entity_0000" for i in range(N)]
measure("divergent-prefix (N eps)", divergent)

# Flat single-segment keys (no path structure) — baseline for the trie.
flat = [f"k{i:06d}" for i in range(N)]
measure("flat keys", flat)