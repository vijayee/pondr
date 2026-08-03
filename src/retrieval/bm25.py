"""BM25 search + Reciprocal Rank Fusion (Tencent-survey A2, query side).

``BM25Search`` reads the inverted index ``src/memory/bm25_index.py`` writes
(postings + df + doclen + stats, all under ``content/idx/``) at QUERY time via
HBTrie range scans -- no in-memory rebuild, no SQLite. ``rrf_fuse`` fuses the
graph, vector, and BM25 ranked id-lists into one ranking (parameter-free,
``k=60``). The retriever's hybrid path (``retriever._retrieve_hybrid``) calls
both.

This module imports ``tokenize`` + the key-layout helpers from
``..memory.bm25_index`` (retrieval -> memory; the retriever already imports
``HippocampalStore`` from memory, so this is the established direction -- no
cycle). It holds a ``db`` ref (the WaveDB handle) and reads committed index
state; it writes nothing.

The index is kept consistent with episode lifecycle by the store's
``set_episode_state`` / supersede unindex (UNCONDITIONAL on deprecate -- cleans
orphan postings even across a flag flip, no-op when the episode was never
indexed), so the index is the source of truth for "active episodes":
``BM25Search`` does NOT re-filter by ``default_episode_ids`` at query time.
The only query-time filter is ``allowed_episode_ids`` (user-scope).
"""

from __future__ import annotations

import math
from typing import Optional

from ..memory.bm25_index import (
    DOCLEN,
    STATS,
    TOK,
    TOKDF,
    b2s,
    read_int,
    read_stats,
    safe_term,
    tokenize,
)

# Reciprocal Rank Fusion constant (Tencent ``RRF_K=60`` in search-utils.ts:18).
# Parameter-free: ``score = sum_i 1/(k + rank_i + 1)``. 60 is the standard
# literature default; tunable later (A2 ships it fixed).
_RRF_K = 60

# Standard BM25 saturation params.
_K1 = 1.5
_B = 0.75

# Cap the per-term posting range scan so a term appearing in very many docs
# (high df -> low idf anyway -> contributes little) cannot blow up query
# latency. The scan streams from HBTrie and breaks at the cap, so it does NOT
# materialize the full posting list.
_MAX_POSTINGS_PER_TERM = 1024


class BM25Search:
    """Read-only BM25 over the in-DB inverted index.

    Holds a WaveDB ``db`` ref; ``search`` streams each query term's posting
    range, scores docs with BM25 (idf from ``tokdf`` + doclen from ``doclen`` +
    avgdl from ``stats``), and returns the top-k ``(eid, score)`` pairs. The
    store's encode-time hook writes the index; this class never writes.
    """

    def __init__(self, db) -> None:
        self.db = db

    def search(
        self,
        terms: list[str],
        k: int,
        allowed_episode_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Score docs by BM25 over the query terms; return top-k ``(eid, score)``.

        ``allowed_episode_ids`` (user-scope): when not None, only eids in the
        set are scored (the per-posting check is cheaper than scoring-then-
        filtering). When None, the index is the source of truth (it already
        excludes deprecated/superseded episodes via the store's unconditional
        unindex), so no further active-set filter is needed.

        Empty terms / empty corpus (``N == 0``) -> ``[]``. Never raises on a
        corrupt/absent key (``read_int`` / ``read_stats`` default to 0).
        """
        if not terms or k <= 0:
            return []
        stats = read_stats(self.db)
        n_docs = stats["N"]
        if n_docs <= 0:
            return []
        total_len = stats["total_len"]
        avgdl = (total_len / n_docs) if n_docs else 0.0

        scores: dict[str, float] = {}
        doclen_cache: dict[str, int] = {}
        for term in set(terms):  # unique query terms; order irrelevant for a sum
            st = safe_term(term)
            df = read_int(self.db, f"{TOKDF}/{st}")
            if df <= 0:
                continue  # term not in the corpus -> no postings
            # idf (the BM25+ form, always non-negative for df <= N).
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            if idf <= 0.0:
                continue  # df > N (shouldn't happen) or a degenerate term
            start = f"{TOK}/{st}/"
            end = f"{TOK}/{st}/\x7f"
            count = 0
            for key_bytes, val_bytes in self.db.create_read_stream(
                start=start, end=end
            ):
                if count >= _MAX_POSTINGS_PER_TERM:
                    break  # common-term scan cap (latency bound)
                count += 1
                # key = "content/idx/tok/{safe_term}/{eid}" -> last component.
                eid = b2s(key_bytes).rsplit("/", 1)[-1]
                if not eid:
                    continue
                if allowed_episode_ids is not None and eid not in allowed_episode_ids:
                    continue
                tf = int(b2s(val_bytes) or "0")
                if tf <= 0:
                    continue
                dl = doclen_cache.get(eid)
                if dl is None:
                    dl = read_int(self.db, f"{DOCLEN}/{eid}")
                    doclen_cache[eid] = dl
                norm = (dl / avgdl) if avgdl > 0 else 0.0
                denom = tf + _K1 * (1.0 - _B + _B * norm)
                if denom <= 0.0:
                    continue
                scores[eid] = scores.get(eid, 0.0) + idf * tf * (_K1 + 1.0) / denom
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]


def rrf_fuse(
    ranked_id_lists: list[list[str]],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion -- parameter-free merge of ranked id-lists.

    ``score[eid] = sum over each list where eid appears: 1 / (k + rank + 1)``
    (``rank`` is 0-based position in that list; absence from a list contributes
    nothing). Returns ``[(eid, score)]`` sorted by score desc. Empty lists /
    all-empty -> ``[]``. The graph, vector, and BM25 lists fuse with NO weight
    tuning and NO score-scale normalization (RRF uses only rank positions, so
    the graph's heuristic ~10 scale and BM25's ~5 scale never collide).
    """
    if not ranked_id_lists:
        return []
    scores: dict[str, float] = {}
    for ranked in ranked_id_lists:
        for rank, eid in enumerate(ranked):
            if not eid:
                continue
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
    if not scores:
        return []
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)