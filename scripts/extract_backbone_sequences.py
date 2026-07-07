"""Extract temporal state-transition pairs for JGS backbone pre-training (Phase 2a).

Walks each conversation's ``follows`` turn-chain in the surviving WaveDB corpus
and emits ``(state_t, state_{t+1})`` embedding pairs (forward + reverse) for
JEPA pre-training. **No Oracle, no OpenAI.** See ``docs/Phase 2a.md`` §0.2.

Corrected vs a draft that used the broken WaveDB API (``.execute()`` /
``r.subject`` / ``r.id`` / wrong embedding key / 1536-dim):

- Episode ids come from a ``content/ep/`` scan (only episodes that actually have
  content), via the same pattern as ``VectorSearch._all_episode_ids``.
- ``follows`` is stored as ``(ep_N, follows, ep_{N-1})`` — the later episode
  points at its predecessor. So a chain **start** is an episode with no
  ``.out("follows")`` (no predecessor = first turn of a conversation), and the
  forward walk uses ``.in_("follows")`` (successors). The real graph API is
  ``.execute_sync()`` -> ``.vertices`` (+ ``result.close()`` in finally).
- Embeddings are 384-dim (``BAAI/bge-small-en-v1.5``), read from
  ``store.get_episode(eid).summary_embedding``. They are NOT persisted in the
  surviving DB by default, so ``--embed-source on-demand`` backfills them (via
  ``VectorSearch``'s sentence-transformer embedder) and persists with
  ``store.set_summary_embedding``. For a no-model-download dev smoke, use
  ``--embed-source stub`` (deterministic hash embeddings — shape-correct only).

Usage (dev smoke, real DB, no model download):
    python scripts/extract_backbone_sequences.py \
        --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \
        --output data/training/backbone/sequences.jsonl \
        --embed-source stub --limit 50

Usage (pod, real backfill + full extraction):
    python scripts/extract_backbone_sequences.py \
        --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \
        --output data/training/backbone/sequences.jsonl \
        --embed-source on-demand
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import HippocampalStore  # noqa: E402
from src.retrieval.vector_search import VectorSearch  # noqa: E402

EMBED_DIM = 384


def _all_episode_ids(store) -> list[str]:
    """Episode ids that have content, via a content/ep scan (see VectorSearch)."""
    ids: set[str] = set()
    for k, _ in store.db.create_read_stream(start="content/ep/", end="content/ep/\x7f"):
        parts = k.split("/", 3)
        if len(parts) >= 3 and parts[2]:
            ids.add(parts[2])
    return sorted(ids)


def _stub_embedding(summary: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic 384-dim 'embedding' from the summary hash.

    Shape-correct and reproducible but carries no semantics — for dev smoke
    tests of the extraction/walking logic only. NOT for training.
    """
    h = hashlib.sha256(summary.encode("utf-8")).digest()
    out = []
    i = 0
    while len(out) < dim:
        out.append(((h[i % len(h)] << 8) | h[(i + 1) % len(h)]) / 65535.0 * 2 - 1)
        i += 2
    return out


def _get_embedding(store, eid: str, vs: VectorSearch | None,
                   embed_source: str, cache: dict[str, list[float]]) -> list[float] | None:
    if eid in cache:
        return cache[eid]
    ep = store.get_episode(eid)
    if ep is None or not ep.summary:
        return None
    if ep.summary_embedding is not None and len(ep.summary_embedding) > 0:
        cache[eid] = ep.summary_embedding
        return ep.summary_embedding
    if embed_source == "stub":
        vec = _stub_embedding(ep.summary)
    elif embed_source == "on-demand":
        if vs is None:
            raise RuntimeError("on-demand embed requires a VectorSearch embedder")
        vec = vs._embed([ep.summary])[0]  # persists below
        store.set_summary_embedding(eid, vec)
    else:  # "persisted" — no backfill; skip episodes without an embedding
        return None
    cache[eid] = vec
    return vec


def _successors(store, eid: str) -> list[str]:
    """Episodes that follow ``eid`` in time (``.in_("follows")`` -> successors).

    The triple is ``(ep_N, follows, ep_{N-1})``, so ``in_("follows")`` from
    ``ep_{N-1}`` returns ``ep_N`` (the next turn).
    """
    result = store.graph.query().vertex(eid).in_("follows").execute_sync()
    try:
        succ = [v for v in result.vertices]
    finally:
        result.close()
    return succ


def _is_chain_start(store, eid: str) -> bool:
    """True if ``eid`` has no predecessor (no ``.out("follows")``)."""
    result = store.graph.query().vertex(eid).out("follows").execute_sync()
    try:
        return len(result.vertices) == 0
    finally:
        result.close()


def _walk_chain(store, start_id: str, max_len: int = 64) -> list[str]:
    """Walk a follows chain forward from a start, guarding cycles."""
    chain = [start_id]
    current = start_id
    seen = {start_id}
    while len(chain) < max_len:
        succ = _successors(store, current)
        nxt = next((s for s in succ if s not in seen), None)
        if nxt is None:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    return chain


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract backbone training sequences from a WaveDB corpus")
    parser.add_argument("--db", default="data/pod_runs/phase1b_scale/ingest_db_dialogsum",
                        help="WaveDB store path (surviving encoded corpus)")
    parser.add_argument("--output", default="data/training/backbone/sequences.jsonl")
    parser.add_argument("--min-chain-length", type=int, default=2,
                        help="Minimum chain length to emit (>=2 for at least one transition)")
    parser.add_argument("--embed-source", choices=["persisted", "on-demand", "stub"],
                        default="on-demand",
                        help="persisted: only use stored embeddings; on-demand: backfill via "
                             "sentence-transformers (bge-small, 384-dim); stub: deterministic "
                             "hash embeddings for dev smoke (no model download)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the number of chain starts used (0 = all). For dev smoke.")
    parser.add_argument("--scan-limit", type=int, default=0,
                        help="Cap the number of episodes scanned for start-detection "
                             "(0 = all). Per-vertex graph queries are ~50-100ms, so the full "
                             "5,002-episode scan takes minutes; set e.g. 300 for a fast dev smoke.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = HippocampalStore(args.db)
    vs = VectorSearch(store) if args.embed_source == "on-demand" else None
    try:
        all_ids = _all_episode_ids(store)
        print(f"Episodes with content: {len(all_ids)}")

        scan_ids = all_ids[: args.scan_limit] if args.scan_limit else all_ids
        starts = [eid for eid in scan_ids if _is_chain_start(store, eid)]
        if args.limit:
            starts = starts[: args.limit]
        print(f"Chain starts (no predecessor): {len(starts)}")

        cache: dict[str, list[float]] = {}
        pairs: list[dict] = []
        chains_used = 0
        chains_dropped_no_emb = 0

        for start_id in starts:
            chain = _walk_chain(store, start_id)
            if len(chain) < args.min_chain_length:
                continue
            embs: list[tuple[str, list[float]]] = []
            for eid in chain:
                vec = _get_embedding(store, eid, vs, args.embed_source, cache)
                if vec is None:
                    # The first episode missing an embedding drops the whole
                    # chain (subsequent turns are unreachable) — count chains,
                    # not episodes.
                    chains_dropped_no_emb += 1
                    break
                embs.append((eid, vec))
            if len(embs) < args.min_chain_length:
                continue
            chains_used += 1
            for i in range(len(embs) - 1):
                eid_t, vec_t = embs[i]
                eid_t1, vec_t1 = embs[i + 1]
                pairs.append({"type": "forward", "state_t": vec_t, "state_t_plus_1": vec_t1,
                              "episode_t": eid_t, "episode_t_plus_1": eid_t1,
                              "chain_id": start_id, "position": i})
                pairs.append({"type": "reverse", "state_t": vec_t1, "state_t_plus_1": vec_t,
                              "episode_t": eid_t1, "episode_t_plus_1": eid_t,
                              "chain_id": start_id, "position": i})

        with open(output_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        fwd = sum(1 for p in pairs if p["type"] == "forward")
        rev = len(pairs) - fwd
        print(f"Chains used: {chains_used}  (dropped {chains_dropped_no_emb} chains w/o embedding)")
        print(f"Pairs: {len(pairs)}  (forward {fwd}, reverse {rev})")
        print(f"Embedding dim: {len(pairs[0]['state_t']) if pairs else 'n/a'}")
        print(f"Output: {output_path}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())