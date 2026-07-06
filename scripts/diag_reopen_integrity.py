"""Verify WaveDB reopen DATA INTEGRITY after the O_BINARY WAL fix.

Writes N records, closes, reopens, and checks the data is fully queryable
(not just that reopen doesn't hang). Tests both graph triples and plain
content puts, at small (50) and large (1500) sizes.
"""
import os
import sys
import tempfile
import time

from wavedb import GraphLayer, WaveDB, WaveDBConfig


def fresh_db():
    d = tempfile.mkdtemp()
    return os.path.join(d, "db"), WaveDBConfig(wal_sync_mode="debounced")


def test_graph(n):
    p, cfg = fresh_db()
    db = WaveDB(p, config=cfg)
    g = GraphLayer("memory", db)
    ops = []
    for i in range(n):
        ops += g.expand_triple(f"C{i:05d}", "subClassOf", "Parent")
    db.batch_sync(ops)
    g.close()
    db.close()

    # reopen
    t = time.time()
    db2 = WaveDB(p, config=WaveDBConfig(wal_sync_mode="debounced"))
    g2 = GraphLayer("memory", db2)
    # Query: all vertices that subClassOf Parent -> should be N
    res = g2.query().vertex("Parent").in_("subClassOf").execute_sync()
    got = res.count()
    g2.close()
    db2.close()
    dt = time.time() - t
    ok = (got == n)
    print(f"[graph n={n}] reopen {dt:.2f}s  count={got}  expect={n}  {'OK' if ok else 'FAIL DATA LOSS'}", flush=True)
    return ok


def test_plain(n):
    p, cfg = fresh_db()
    db = WaveDB(p, config=cfg)
    ops = [{"type": "put", "key": f"content/k{i:05d}", "value": str(i)} for i in range(n)]
    db.batch_sync(ops)
    db.close()

    t = time.time()
    db2 = WaveDB(p, config=WaveDBConfig(wal_sync_mode="debounced"))
    got = 0
    for i in range(n):
        v = db2.get_sync(f"content/k{i:05d}")
        if v is not None:
            got += 1
    db2.close()
    dt = time.time() - t
    ok = (got == n)
    print(f"[plain n={n}] reopen {dt:.2f}s  count={got}  expect={n}  {'OK' if ok else 'FAIL DATA LOSS'}", flush=True)
    return ok


results = []
for n in (50, 200, 1500):
    results.append(("graph", n, test_graph(n)))
for n in (50, 200, 1500):
    results.append(("plain", n, test_plain(n)))

print("\n=== SUMMARY ===", flush=True)
fails = [(k, n) for (k, n, ok) in results if not ok]
if fails:
    print(f"FAILURES: {fails}", flush=True)
    sys.exit(1)
else:
    print("ALL PASS — no data loss on reopen", flush=True)