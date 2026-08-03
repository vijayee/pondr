"""Backfill the in-WaveDB BM25 inverted index over existing episode full_text.

A2 one-time step for a corpus ingested before ``--hybrid-retrieval`` was on.
Reads each active episode's ``content/ep/{eid}/text`` and splices
``bm25_index_ops`` into a per-episode ``batch_sync`` (mirrors the encode path,
which writes the index in the same atomic batch as the content). Idempotent:
``bm25_index_ops`` returns ``[]`` when ``content/idx/docterms/{eid}`` already
exists, so a rerun reports ``Indexed 0`` and corrupts nothing -- safe to run
repeatedly, safe to run with the flag already on (encodes after this script
index themselves).

Indexes ACTIVE episodes only (``default_episode_ids`` excludes
deprecated/superseded), matching the live path: the index never holds dead
episodes, so the BM25 list is the source of truth for "active" at query time
(see ``BM25Search.search``).

    python scripts/build_bm25_index.py --db /workspace/volumes/hippo/memory_db

No SQLite/FTS5 -- the index lives under ``content/idx/`` inside WaveDB via
HBTrie range scan. No new training/GPU/LLM-call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config as _master_config  # noqa: E402
from src.memory.bm25_index import b2s, bm25_index_ops  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill the in-WaveDB BM25 inverted index over episode full_text.",
    )
    parser.add_argument("--db", required=True, help="WaveDB store path (ingested corpus).")
    args = parser.parse_args()

    # The index hook in ``_content_ops`` reads this global at call time. The
    # backfill calls ``bm25_index_ops`` directly (not ``encode_episode``), so
    # the flag is not strictly required here -- but setting it makes the
    # intent explicit and keeps a concurrent encode (if any) consistent.
    _master_config.hybrid_retrieval = True

    store = HippocampalStore(args.db)
    try:
        eids = store.default_episode_ids()
        n = 0
        for eid in eids:
            text = b2s(store.db.get_sync(f"content/ep/{eid}/text"))
            if not text:
                continue  # no full_text -> nothing to index (tokenize("") == [])
            ops = bm25_index_ops(store.db, eid, text)
            if ops:
                store.db.batch_sync(ops)
                n += 1
        print(f"Indexed {n} episodes ({len(eids)} active).")
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())