"""Diagnose the WaveDB reopen hang: is it per-batch op count, graph keys, or total keys?

Hypothesis from the user: WaveDB has a batch-size limit on what it can record in
one WAL record at once. If a single batch_sync submits more ops than the WAL can
hold in one record, the WAL may be left in a state that hangs recovery (reopen),
even though the write itself returned.

We compare four write shapes at the same total key count (n=1500 graph triples =
~4500 root-namespace ops after expand_triple, vs 1500 plain content puts), then
reopen each and time it. Output is flushed per line so a hang on one variant
doesn't hide the others' results.
"""
import os
import sys
import tempfile
import time

from wavedb import GraphLayer, WaveDB, WaveDBConfig


def fresh_db():
    d = tempfile.mkdtemp()
    return os.path.join(d, "db"), WaveDBConfig(wal_sync_mode="debounced")


def reopen_ok(p, label):
    t = time.time()
    try:
        db2 = WaveDB(p, config=WaveDBConfig(wal_sync_mode="debounced"))
        dt = time.time() - t
        db2.close()
        print(f"  [{label}] reopen OK in {dt:.2f}s", flush=True)
        return True
    except Exception as e:
        print(f"  [{label}] reopen RAISED {type(e).__name__}: {e}", flush=True)
        return False


def variant_plain_content_one_batch(n):
    p, cfg = fresh_db()
    db = WaveDB(p, config=cfg)
    ops = [{"type": "put", "key": f"content/k{i:05d}", "value": str(i)} for i in range(n)]
    db.batch_sync(ops)
    db.close()
    return p


def variant_graph_one_big_batch(n):
    p, cfg = fresh_db()
    db = WaveDB(p, config=cfg)
    g = GraphLayer("memory", db)
    ops = []
    for i in range(n):
        ops += g.expand_triple(f"C{i:05d}", "subClassOf", "Parent")
    print(f"  graph_one_big_batch: {n} triples -> {len(ops)} root ops", flush=True)
    db.batch_sync(ops)
    g.close()
    db.close()
    return p


def variant_graph_many_small_batches(n):
    p, cfg = fresh_db()
    db = WaveDB(p, config=cfg)
    g = GraphLayer("memory", db)
    for i in range(n):
        g.insert_sync(f"C{i:05d}", "subClassOf", "Parent")
    g.close()
    db.close()
    return p


def variant_graph_chunked_batches(n, chunk):
    p, cfg = fresh_db()
    db = WaveDB(p, config=cfg)
    g = GraphLayer("memory", db)
    buf = []
    for i in range(n):
        buf += g.expand_triple(f"C{i:05d}", "subClassOf", "Parent")
        if len(buf) >= chunk:
            db.batch_sync(buf)
            buf = []
    if buf:
        db.batch_sync(buf)
    g.close()
    db.close()
    return p


N = 1500
variants = [
    ("plain_content_one_batch", lambda: variant_plain_content_one_batch(N)),
    ("graph_one_big_batch", lambda: variant_graph_one_big_batch(N)),
    ("graph_many_small_batches", lambda: variant_graph_many_small_batches(N)),
    ("graph_chunked_50", lambda: variant_graph_chunked_batches(N, 50)),
]

for label, make in variants:
    print(f"== {label} (n={N}) ==", flush=True)
    t = time.time()
    p = make()
    print(f"  write done in {time.time()-t:.2f}s", flush=True)
    reopen_ok(p, label)