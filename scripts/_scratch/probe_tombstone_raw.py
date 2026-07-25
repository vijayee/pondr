"""Minimal raw KV probe: does delete-then-put of an IDENTICAL key work in WaveDB core?

put K=v1 -> get K (v1) -> delete K -> get K (gone) -> put K=v2 -> get K (v2? or gone?).
If the last get returns gone, it's a core tombstone/MVCC bug (re-put shadowed by
its own tombstone). Tests via the plain WaveDB KV API (no vector layer).
"""
import os
import tempfile
from wavedb import WaveDB

d = tempfile.mkdtemp(prefix="tomb_")
db = WaveDB(os.path.join(d, "db"))

K = "test/key1"
db.put_sync(K, "v1")
g1 = db.get_sync(K)
db.del_sync(K)
g2 = db.get_sync(K)
db.put_sync(K, "v2")
g3 = db.get_sync(K)

# Also test via batch (migrate/rebuild use batch_sync, not put_sync)
K2 = "test/key2"
db.batch_sync([{"type": "put", "key": K2, "value": "v1"}])
gb1 = db.get_sync(K2)
db.batch_sync([{"type": "del", "key": K2}])
gb2 = db.get_sync(K2)
db.batch_sync([{"type": "put", "key": K2, "value": "v2"}])
gb3 = db.get_sync(K2)

# And: delete + re-put in the SAME batch (different txn) -- rebuild uses separate batches.
K3 = "test/key3"
db.batch_sync([{"type": "put", "key": K3, "value": "v1"}])
db.batch_sync([{"type": "del", "key": K3}, {"type": "put", "key": K3, "value": "v2"}])  # del+put same txn
gc3 = db.get_sync(K3)

db.close()
print(f"put_sync path:   put={g1} del={g2} re-put={g3!r}  (expect v2; bug if None)")
print(f"batch_sync path: put={gb1} del={gb2} re-put={gb3!r}  (expect v2; bug if None)")
print(f"same-batch del+put: {gc3!r}  (expect v2)")