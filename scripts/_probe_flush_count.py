"""Decisive flush-count test for the WaveDB write-amplification symptom.

The pod's DialogSum run wrote 4995 episodes through `process_corpus.py`, which
uses ONE store for the whole run (open at start, close at end). Each
`encode_episode` calls `HippocampalStore.encode_episode` -> a handful of
`store.db.put_sync` calls (content keys + graph triples via `batch_sync`).

Two competing theories for the 4.69 GB bloat:

  (A) Flush-count-driven: every `put_sync` (or `batch_sync`) CoW-appends new
      4KB pages and never reclaims, so N puts -> ~N pages of stale garbage
      regardless of key structure. Then a single-transaction bulk reload
      (all keys in ONE `batch_sync`, one close) would NOT bloat — and the
      Hippo-side workaround "compact via reload" is viable for the immediate
      corpora.

  (B) Structural per-key: divergent-prefix keys each cost ~6 pages of sparse
      bnode, and CoW + dead reclamation only compounds it. Then even a single
      `batch_sync` bloats, and we need the WaveDB reclamation/vacuum fix.

This probe runs both and compares:

  - DIVERGENT_500_PUTS: 500 `put_sync` calls (one per key) + one close.
      Reuses the divergent-prefix key shape from _probe_prefix_sharing.
  - DIVERGENT_500_BATCH: the same 500 keys in ONE `batch_sync` + one close.

If BATCH is dramatically smaller than PUTS, theory (A) holds and the
workaround is viable. If BATCH ~= PUTS (both ~hundreds of MB), theory (B)
holds and the fix must be in WaveDB. Scratch — not committed.
"""

import shutil
import tempfile
from pathlib import Path

import wavedb

N = 500
KEYS = [f"memory/spo/ep_{i:06d}/has_entity/entity_0000" for i in range(N)]
VALS = [f"value-{i}".encode() for i in range(N)]


def divergent_puts() -> float:
    base = Path(tempfile.mkdtemp(prefix="hippo_fc_puts_"))
    db = wavedb.WaveDB(str(base))
    for k, v in zip(KEYS, VALS):
        db.put_sync(k, v)
    db.close()
    sz = (base / "data.wdbp").stat().st_size
    shutil.rmtree(base)
    return sz


def divergent_batch() -> float:
    base = Path(tempfile.mkdtemp(prefix="hippo_fc_batch_"))
    db = wavedb.WaveDB(str(base))
    ops = [{"type": "put", "key": k, "value": v} for k, v in zip(KEYS, VALS)]
    db.batch_sync(ops)
    db.close()
    sz = (base / "data.wdbp").stat().st_size
    shutil.rmtree(base)
    return sz


def main() -> None:
    puts = divergent_puts()
    batch = divergent_batch()
    print(f"DIVERGENT_500_PUTS   : {puts/1024/1024:>8.2f} MB "
          f"({puts/N/1024:.1f} KB/key, {puts/4096:.0f} pages)")
    print(f"DIVERGENT_500_BATCH  : {batch/1024/1024:>8.2f} MB "
          f"({batch/N/1024:.1f} KB/key, {batch/4096:.0f} pages)")
    ratio = puts / batch if batch else float("inf")
    print(f"\nputs/batch ratio = {ratio:.2f}x")
    if ratio > 5:
        print("=> flush-count-driven (A): single-transaction bulk reload "
              "compacts; Hippo workaround viable for the immediate corpora.")
    else:
        print("=> structural per-key (B): bloat is in the per-key sparse "
              "bnode path, not flush count; fix must be in WaveDB "
              "(reclamation + vacuum).")
    # Compare against the flat baseline (no path structure) to isolate
    # structural cost from the CoW/reclamation cost.
    base = Path(tempfile.mkdtemp(prefix="hippo_fc_flat_"))
    db = wavedb.WaveDB(str(base))
    flat_keys = [f"k{i:06d}" for i in range(N)]
    ops = [{"type": "put", "key": k, "value": v} for k, v in zip(flat_keys, VALS)]
    db.batch_sync(ops)
    db.close()
    flat = (base / "data.wdbp").stat().st_size
    shutil.rmtree(base)
    print(f"FLAT_500_BATCH       : {flat/1024/1024:>8.2f} MB "
          f"({flat/N/1024:.1f} KB/key, {flat/4096:.0f} pages)")
    print(f"\npath-cost ratio (divergent_batch / flat_batch) = "
          f"{batch/flat if flat else float('inf'):.2f}x — the per-key "
          f"page-bloat multiplier from divergent trie paths alone.")


if __name__ == "__main__":
    main()