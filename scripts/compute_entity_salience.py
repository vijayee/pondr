"""Compute entity salience from existing encoded episodes (Phase 1c, step 1).

Scans the POS index of ``has_entity`` ONCE
(``memory/pos/has_entity/{E:entity}/{eid}``) to count, per entity, how many
distinct episodes mention it, then persists salience via
``HippocampalStore.write_entity_salience_batch`` (one sorted ``batch_sync`` so
the salience keys are ``get_sync``-safe — see ``docs/Phase 1c.md`` §0.3).

This is the batch path — it does NOT call a per-encode increment. Run it after
ingesting a corpus (or after topping one up) to (re)compute salience for every
entity in the graph. Cheap: one prefix scan over the ``has_entity`` POS subtree,
O(total has_entity triples), no per-episode hydration, no LLM.

Usage:
    python scripts/compute_entity_salience.py --db ./data/memory_db
    python scripts/compute_entity_salience.py --db /workspace/volumes/hippo/memory_db
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import HippocampalStore  # noqa: E402


def _strip_prefix(vid: str, prefix: str) -> str:
    return vid[len(prefix):] if vid.startswith(prefix) else vid


def _iter_entity_episode_pairs(store: HippocampalStore):
    """Yield ``(entity, episode_id)`` for every ``has_entity`` triple.

    POS key = ``memory/pos/has_entity/{E:entity}/{eid}``. One scan over the
    whole ``has_entity`` POS subtree — O(total has_entity triples), no
    per-episode work. Yields tuples (NOT bare keys) because
    ``create_read_stream`` yields ``(key, value)``.
    """
    start = "memory/pos/has_entity/"
    end = "memory/pos/has_entity/\x7f"
    for k, _ in store.db.create_read_stream(start=start, end=end):
        # k = memory/pos/has_entity/{E:entity}/{eid}
        parts = k.split("/", 4)
        if len(parts) < 5:
            continue
        entity_node, eid = parts[3], parts[4]
        yield _strip_prefix(entity_node, "E:"), eid


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute entity salience from the has_entity POS index.")
    ap.add_argument("--db", default="./data/memory_db", help="WaveDB store directory")
    ap.add_argument("--top", type=int, default=20, help="print this many top entities")
    args = ap.parse_args()

    store = HippocampalStore(args.db)

    counts: Counter[str] = Counter()
    last_ep: dict[str, str] = {}
    last_ep_ts: dict[str, str] = {}  # Phase 3b: max-mention timestamp per entity
    n_triples = 0
    nul_keys = 0
    for entity, eid in _iter_entity_episode_pairs(store):
        counts[entity] += 1
        last_ep[entity] = eid  # last-seen wins; scan order is trie order
        # Phase 3b step 10: track the LATEST mention timestamp per entity so the
        # retrieval hot path can compute recency without a per-query get_episode.
        # Load by unit type: the has_entity POS scan now includes documents
        # (docs emit ``(doc, has_entity, E:x)``); a ``doc_`` id has no episode
        # content, so use the document's ``ingested_at`` instead of get_episode.
        if eid.startswith("doc_"):
            doc = store.get_document(eid, load_bodies=False)
            ts = doc.ingested_at if doc is not None else None
        else:
            ep = store.get_episode(eid)
            ts = ep.timestamp if ep is not None else None
        if ts:
            prev = last_ep_ts.get(entity)
            if prev is None or ts > prev:
                last_ep_ts[entity] = ts
        n_triples += 1
        if "\x00" in eid or "\x00" in entity:
            nul_keys += 1

    store.write_entity_salience_batch(dict(counts), last_ep, last_ep_ts)

    print(f"Scanned {n_triples} has_entity triples; salience for {len(counts)} entities.")
    print(f"NUL-bearing keys skipped in scan: {nul_keys}")
    print(f"Top {args.top} entities by episode-mention count:")
    for entity, count in counts.most_common(args.top):
        print(f"  {entity}: {count} episodes")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())