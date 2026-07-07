"""Local reproduction probe for the WaveDB write-amplification symptom.

The pod's DialogSum scale run wrote 4995 episodes into a WaveDB DB whose
``data.wdbp`` page file ballooned to 4.69 GB (~960 KB/episode, ~1000x over the
~1 KB logical content + a handful of graph triples). Zero WAL files — the
bloat is the page file itself.

This probe reproduces the write pattern at small scale, fully offline, with
stubbed extraction (no GLiNER/Bonsai): it opens a fresh store, runs one session
per "conversation", encodes N episodes with realistic per-episode graph
triples (~2.5 entities, ~0.6 topics, ~1 tone, ~0.4 decisions, ~3 relations,
chained ``follows``, user/session scope + ``at_time``), closes, and measures:

  - ``data.wdbp`` size
  - total key count + breakdown by prefix (content/ep/, memory/spo/, memory/pos/)
  - per-episode KB
  - bytes per graph key

Running N = 10, 50, 200 reveals whether growth is **linear** (per-insert page
overhead — sparse pages) or **super-linear** (MVCC version accumulation across
transactions). Scratch — not committed.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.episode import Episode
from src.memory.store import HippocampalStore


def make_episode(i: int, user: str, session: str, follows: str | None) -> Episode:
    rng = random.Random(i)
    ents = [f"entity_{rng.randint(0, 40)}", f"entity_{rng.randint(0, 40)}"][: rng.randint(1, 3)]
    topics = [f"topic_{rng.randint(0, 60)}"] if rng.random() < 0.6 else []
    tones = [rng.choice(["neutral", "curious", "excited", "frustrated"])]
    decisions = [f"decision_{rng.randint(0, 30)}"] if rng.random() < 0.4 else []
    rels = []
    for _ in range(rng.randint(2, 4)):
        rels.append({
            "subject": rng.choice(["User", "Assistant", ents[0] if ents else "User"]),
            "predicate": rng.choice(["explains", "decides", "questions", "suggests"]),
            "object": f"concept_{rng.randint(0, 50)}",
        })
    return Episode(
        id=f"ep_{i:06d}",
        timestamp=f"2026-07-06T{i:04d}:00:00Z",
        summary=f"summary for episode {i} " * 3,
        full_text=f"User: message number {i} " * 8 + f"\nAssistant: response number {i} " * 8,
        entities=ents,
        topics=topics,
        tones=tones,
        decisions=decisions,
        relations=rels,
        follows=follows,
        user_id=user,
        session_id=session,
    )


def measure(db_dir: Path) -> dict:
    wdbp = db_dir / "data.wdbp"
    size = wdbp.stat().st_size if wdbp.exists() else 0
    return {"data_wdbp_bytes": size, "data_wdbp_mb": size / 1024 / 1024}


def count_keys(db_dir: Path, n_episodes: int) -> dict:
    """Reopen the DB read-only-ish and scan prefixes. WaveDB has no read-only
    flag in this binding, so we just reopen (the DB is closed) and scan."""
    import wavedb
    db = wavedb.WaveDB(str(db_dir))
    prefixes = {
        "content/ep/": b"content/ep/",
        "memory/spo/": b"memory/spo/",
        "memory/pos/": b"memory/pos/",
        "memory/osp/": b"memory/osp/",
    }
    counts = {}
    total = 0
    for name, pfx in prefixes.items():
        c = 0
        start = pfx.decode()
        end = start + "\x7f"
        for k, _ in db.create_read_stream(start=start, end=end):
            c += 1
        counts[name] = c
        total += c
    db.close()
    return {"key_counts": counts, "total_keys": total}


def run(n: int, base: Path) -> dict:
    db_dir = base / f"db_n{n}"
    if db_dir.exists():
        shutil.rmtree(db_dir)
    store = HippocampalStore(str(db_dir))
    user = "U:probe"
    # One session per "conversation" of n episodes, mirroring process_corpus.
    session = store.next_session_id()
    store.open_session(user, session, "2026-07-06T00:00:00Z")
    follows = None
    for i in range(n):
        ep = make_episode(i, user, session, follows)
        store.encode_episode(ep)
        follows = ep.id
    store.close_session(session, "2026-07-06T23:59:59Z")
    store.close()
    m = measure(db_dir)
    kc = count_keys(db_dir, n)
    per_ep_kb = m["data_wdbp_bytes"] / n / 1024
    total_keys = kc["total_keys"]
    bytes_per_key = m["data_wdbp_bytes"] / total_keys if total_keys else 0
    result = {
        "n_episodes": n,
        "data_wdbp_mb": round(m["data_wdbp_mb"], 2),
        "key_counts": kc["key_counts"],
        "total_keys": total_keys,
        "kb_per_episode": round(per_ep_kb, 1),
        "bytes_per_key": round(bytes_per_key, 1),
    }
    print(f"N={n:>4}  data.wdbp={result['data_wdbp_mb']:>8.2f} MB  "
          f"keys={total_keys:>6}  kb/ep={per_ep_kb:>6.1f}  bytes/key={bytes_per_key:>6.1f}  "
          f"spo={kc['key_counts'].get('memory/spo/',0)} pos={kc['key_counts'].get('memory/pos/',0)} "
          f"content={kc['key_counts'].get('content/ep/',0)}")
    return result


def main() -> None:
    base = Path(tempfile.mkdtemp(prefix="hippo_amp_"))
    print(f"probe base: {base}")
    print("linear growth => per-insert page overhead (sparse pages); "
          "super-linear => MVCC version accumulation\n")
    results = []
    for n in (10, 50, 200, 500):
        try:
            results.append(run(n, base))
        except Exception as e:
            print(f"N={n} FAILED: {e!r}")
    # Growth check
    if len(results) >= 2:
        print("\ngrowth ratio (kb/ep should be ~flat if linear):")
        for r in results:
            print(f"  N={r['n_episodes']:>4}  kb/ep={r['kb_per_episode']}")
    # Don't clean up so we can inspect, but print the path.
    print(f"\nleft at {base} for inspection")


if __name__ == "__main__":
    main()