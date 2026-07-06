"""Pinpoint the exact step where the graph reopen segfaults."""
import os, sys, tempfile, time
from wavedb import GraphLayer, WaveDB, WaveDBConfig

def fresh_db():
    d = tempfile.mkdtemp()
    return os.path.join(d, "db"), WaveDBConfig(wal_sync_mode="debounced")

N = 50
p, cfg = fresh_db()
print(f"[1] create db", flush=True)
db = WaveDB(p, config=cfg)
print(f"[2] create graph", flush=True)
g = GraphLayer("memory", db)
ops = []
for i in range(N):
    ops += g.expand_triple(f"C{i:05d}", "subClassOf", "Parent")
print(f"[3] expand_triple done: {len(ops)} ops", flush=True)
db.batch_sync(ops)
print(f"[4] batch_sync done", flush=True)
g.close()
print(f"[5] g.close done", flush=True)
db.close()
print(f"[6] db.close done", flush=True)

print(f"[7] reopen db2", flush=True)
db2 = WaveDB(p, config=WaveDBConfig(wal_sync_mode="debounced"))
print(f"[8] reopen done", flush=True)
g2 = GraphLayer("memory", db2)
print(f"[9] graph2 done", flush=True)
res = g2.query().vertex("Parent").in_("subClassOf").execute_sync()
print(f"[10] query done", flush=True)
got = res.count  # property, not a method
print(f"[11] count={got}", flush=True)
verts = res.vertices
print(f"[12] vertices len={len(verts)} first={verts[0] if verts else None}", flush=True)
res.close()
g2.close()
print(f"[13] g2.close done", flush=True)
db2.close()
print(f"[14] db2.close done — ALL OK", flush=True)