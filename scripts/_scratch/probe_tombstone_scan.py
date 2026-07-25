"""Probe v3: does a RANGE SCAN see a delete-then-put-same-key value?

Point get works (probe_tombstone_raw). Vector search uses a prefix range scan,
not a point get. If the scan path mishandles tombstones for re-put keys, the
SLSH same-type migrate search returns [] while a point get of the same key
would succeed. This pinpoints the bug to the scan/iterator path.
"""
import os
import tempfile
from wavedb import WaveDB

d = tempfile.mkdtemp(prefix="tomb_scan_")
db = WaveDB(os.path.join(d, "db"))

# 3 keys under a common prefix. Middle one gets deleted then re-put (same key).
db.put_sync("pfx/a", "A1")
db.put_sync("pfx/b", "B1")
db.put_sync("pfx/c", "C1")
db.del_sync("pfx/b")          # tombstone
db.put_sync("pfx/b", "B2")    # re-put same key

def scan(start, end):
    return list(db.create_read_stream(start=start, end=end))

allkeys = scan("pfx/", "pfx0")   # '0' = 0x30, just past '/'=0x2f
print("scan all pfx/:", [(k, v) for k, v in allkeys])
print("  -> 'pfx/b' present in scan?", any(k == "pfx/b" for k, _ in allkeys))
print("  -> point get pfx/b:", db.get_sync("pfx/b"))

# Also test the delete-then-put via batch (migrate uses batch_sync for aux deletes)
db.put_sync("pfx2/x", "X1")
db.batch_sync([{"type": "del", "key": "pfx2/x"}])
db.batch_sync([{"type": "put", "key": "pfx2/x", "value": "X2"}])
allkeys2 = scan("pfx2/", "pfx20")
print("scan pfx2/ (batch del+re-put):", [(k, v) for k, v in allkeys2])
print("  -> 'pfx2/x' present in scan?", any(k == "pfx2/x" for k, _ in allkeys2))
print("  -> point get pfx2/x:", db.get_sync("pfx2/x"))

db.close()