"""Scan-compatibility probe for the chunk_size / delimiter fix.

The 60x write-amplification fix is `chunk_size=64 + delimiter=0x01` (a
non-appearing delimiter collapses the HBTrie to a flat B+tree on the full
key). But Hippo retrieval depends on THREE prefix-scan patterns:

  1. POS prefix scan: `memory/pos/has_entity/{E}/` → all episodes with entity E.
  2. SPO prefix scan: `memory/spo/{eid}/` → all predicates/objects of one ep.
  3. Content prefix scan: `content/ep/{eid}/` → all fields of one ep.

This probe loads a small realistic corpus under each config and verifies
every scan returns the EXPECTED keys (count + membership). If chunk=64
delim=0x01 breaks any scan, the config fix is unsafe and we fall back to
chunk=16 delim='/' (3x, no scan risk). Scratch — not committed.
"""

import shutil
import tempfile
from pathlib import Path

import wavedb
from wavedb import WaveDB, WaveDBConfig


def build_corpus(db):
    # 3 episodes, 2 entities, 2 topics, 1 tone — enough to exercise every scan.
    eps = ["ep_000001", "ep_000002", "ep_000003"]
    # SPO + POS triples (skip OSP — drop candidate).
    for eid in eps:
        db.put_sync(f"content/ep/{eid}/summary", b"sum")
        db.put_sync(f"content/ep/{eid}/text", b"text")
        db.put_sync(f"content/ep/{eid}/ts", b"ts")
    # ep1 has entity_A + entity_B; ep2 has entity_A; ep3 has entity_B
    triples = [
        ("ep_000001", "has_entity", "E:entity_A"),
        ("ep_000001", "has_entity", "E:entity_B"),
        ("ep_000002", "has_entity", "E:entity_A"),
        ("ep_000003", "has_entity", "E:entity_B"),
        ("ep_000001", "has_topic", "T:topic_X"),
        ("ep_000002", "has_topic", "T:topic_X"),
        ("ep_000003", "has_topic", "T:topic_Y"),
        ("ep_000001", "has_tone", "A:frustrated"),
    ]
    for eid, pred, obj in triples:
        db.put_sync(f"memory/spo/{eid}/{pred}/{obj}", b"1")
        db.put_sync(f"memory/pos/{pred}/{obj}/{eid}", b"1")


def scan(db, start):
    """Prefix scan via create_read_stream(start, end=start+\x7f)."""
    end = start + "\x7f"
    return [k for k, _ in db.create_read_stream(start=start, end=end)]


def run(label, chunk, delim, node=4096):
    base = Path(tempfile.mkdtemp(prefix="hippo_scan_"))
    cfg = WaveDBConfig(chunk_size=chunk, btree_node_size=node)
    db = WaveDB(str(base), delimiter=delim, config=cfg)
    build_corpus(db)
    db.close()
    # Reopen read-only-ish for scans.
    db = WaveDB(str(base), delimiter=delim, config=cfg)

    results = {}
    # POS: episodes with entity_A = ep1, ep2
    k = scan(db, "memory/pos/has_entity/E:entity_A/")
    results["POS has_entity E:entity_A"] = set(k)
    # POS: episodes with entity_B = ep1, ep3
    k = scan(db, "memory/pos/has_entity/E:entity_B/")
    results["POS has_entity E:entity_B"] = set(k)
    # POS: episodes with topic_X = ep1, ep2
    k = scan(db, "memory/pos/has_topic/T:topic_X/")
    results["POS has_topic T:topic_X"] = set(k)
    # SPO: all predicates of ep_000001 = has_entity×2, has_topic, has_tone (4)
    k = scan(db, "memory/spo/ep_000001/")
    results["SPO ep_000001 preds"] = set(k)
    # Content: all fields of ep_000001 = summary, text, ts (3)
    k = scan(db, "content/ep/ep_000001/")
    results["content ep_000001 fields"] = set(k)
    db.close()
    sz = (base / "data.wdbp").stat().st_size
    shutil.rmtree(base)

    # Expected
    exp = {
        "POS has_entity E:entity_A": {
            "memory/pos/has_entity/E:entity_A/ep_000001",
            "memory/pos/has_entity/E:entity_A/ep_000002",
        },
        "POS has_entity E:entity_B": {
            "memory/pos/has_entity/E:entity_B/ep_000001",
            "memory/pos/has_entity/E:entity_B/ep_000003",
        },
        "POS has_topic T:topic_X": {
            "memory/pos/has_topic/T:topic_X/ep_000001",
            "memory/pos/has_topic/T:topic_X/ep_000002",
        },
        "SPO ep_000001 preds": {
            "memory/spo/ep_000001/has_entity/E:entity_A",
            "memory/spo/ep_000001/has_entity/E:entity_B",
            "memory/spo/ep_000001/has_topic/T:topic_X",
            "memory/spo/ep_000001/has_tone/A:frustrated",
        },
        "content ep_000001 fields": {
            "content/ep/ep_000001/summary",
            "content/ep/ep_000001/text",
            "content/ep/ep_000001/ts",
        },
    }
    ok = all(results[name] == exp[name] for name in exp)
    print(f"\n=== {label} (data.wdbp={sz/1024:.1f} KB) ===")
    for name in exp:
        mark = "OK" if results[name] == exp[name] else "FAIL"
        if results[name] != exp[name]:
            print(f"  [{mark}] {name}")
            print(f"         expected {len(exp[name])}: {sorted(exp[name])}")
            print(f"         got      {len(results[name])}: {sorted(results[name])}")
        else:
            print(f"  [{mark}] {name} ({len(results[name])} keys)")
    print("ALL SCANS PASS" if ok else "SOME SCANS FAILED")
    return ok


ok1 = run("chunk=4 delim='/' [BASE]", 4, "/")
ok2 = run("chunk=16 delim='/'", 16, "/")
ok3 = run("chunk=64 delim='/'", 64, "/")
ok4 = run("chunk=64 delim=0x01 [60x FIX]", 64, chr(1))
ok5 = run("chunk=32 delim=0x01", 32, chr(1))
print("\n=== SUMMARY ===")
print(f"chunk=4  delim='/'  : {'PASS' if ok1 else 'FAIL'}")
print(f"chunk=16 delim='/'  : {'PASS' if ok2 else 'FAIL'}")
print(f"chunk=64 delim='/'  : {'PASS' if ok3 else 'FAIL'}")
print(f"chunk=64 delim=0x01 : {'PASS' if ok4 else 'FAIL'}  <- 60x fix")
print(f"chunk=32 delim=0x01 : {'PASS' if ok5 else 'FAIL'}")