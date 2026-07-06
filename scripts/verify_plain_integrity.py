"""Verify the WAL data-loss fix on the CLEAN build, with correct bytes
comparison and a determinism loop (the O_BINARY bug was non-deterministic)."""
import os, sys, tempfile
from wavedb import WaveDB, WaveDBConfig

def fresh_db():
    d = tempfile.mkdtemp()
    return os.path.join(d, "db"), WaveDBConfig(wal_sync_mode="debounced")

ok_all = True
for trial in range(3):
    for n in (50, 500, 1500, 3000):
        p, cfg = fresh_db()
        db = WaveDB(p, config=cfg)
        ops = [{"type": "put", "key": f"content/k{i:05d}", "value": f"v{i}"} for i in range(n)]
        db.batch_sync(ops)
        db.close()

        db2 = WaveDB(p, config=WaveDBConfig(wal_sync_mode="debounced"))
        got = sum(1 for i in range(n) if db2.get_sync(f"content/k{i:05d}") is not None)
        val_ok = db2.get_sync("content/k00000") == b"v0" if n > 0 else True
        db2.close()
        ok = got == n and val_ok
        ok_all = ok_all and ok
        if not ok:
            print(f"[trial {trial} n={n}] got={got} expect={n} val_ok={val_ok} FAIL", flush=True)

print("ALL OK (3 trials x 4 sizes)" if ok_all else "FAILURES", flush=True)
sys.exit(0 if ok_all else 1)