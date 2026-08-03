"""BM25 inverted index hosted INSIDE WaveDB (Tencent-survey A2).

A lexical retrieval path Pondr lacked: an episode whose ``full_text`` contains
the query words but whose entities/topics the planner didn't surface (and whose
summary embedding doesn't cosine-rank) was invisible to graph+vector retrieval.
A2 adds a BM25 inverted index over episode ``full_text``, fused with the graph
and vector ranked lists via Reciprocal Rank Fusion (see ``src/retrieval/bm25.py``).

HARD CONSTRAINT: NO SQLite/FTS5 -- HippocampalStore is WaveDB end-to-end. The
inverted index exploits HBTrie's native lexicographic RANGE SCAN
(``db.create_read_stream(start, end=".../\\x7f")``) to live entirely inside
WaveDB, no in-memory rebuild:

    content/idx/tok/{safe_term}/{eid}   -> str(tf)     (posting)
    content/idx/tokdf/{safe_term}        -> str(df)     (document frequency)
    content/idx/doclen/{eid}            -> str(doc_len) (doc length in tokens)
    content/idx/docterms/{eid}          -> json(terms)  (unindex key)
    content/idx/stats                   -> json(N,total_len)  (avgdl source)

The index ops are spliced into the SAME atomic ``batch_sync`` as
``HippocampalStore.encode_episode`` (live, persisted, atomic -- can't drift from
the content it indexes), and deleted on forget/supersede (mirrors
``_unindex_embedding``). Terms are safe-encoded via the ``safe_edge_component``
logic (store.py:85-100) so a ``/`` or NUL in a term never splits the key into
the wrong namespace.

This is a LEAF module: it imports nothing from ``store`` (avoids a
store<->bm25_index cycle). It inlines the 3-line ``safe_term`` helper (with a
pointer to ``store.safe_edge_component``) and the 4-line ``b2s`` decode helper
rather than importing them. Episodes-only for A2; the key layout is generic
(eid can be ``ep_*``/``{doc}_sec_NNN``/``scene_*`` later) so documents/sections
extend WITHOUT a schema change. Flag-gated (``config.hybrid_retrieval``, read
at call time by the store hook -- the master-config convention), default OFF,
byte-identical when off (no ops are appended).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

# Key layout -- one scannable ``content/idx/`` prefix. PUBLIC so the query
# side (``src/retrieval/bm25.py``) reads the SAME layout the index side writes
# -- re-defining it there would risk drift. ``_DOCTERMS`` stays private (only
# the unindex path reads it).
TOK = "content/idx/tok"
TOKDF = "content/idx/tokdf"
DOCLEN = "content/idx/doclen"
_DOCTERMS = "content/idx/docterms"
STATS = "content/idx/stats"

# Small English stopword set. ASCII-only (the codebase is ASCII per
# commit-at-will). Stripped from BOTH index tokens and query tokens so a
# stopword never contributes a posting / a BM25 score. Kept tiny on purpose --
# over-aggressive stemming/stopwording is a de-wonk smell, not a feature.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "is",
    "it", "for", "with", "as", "by", "this", "that", "was", "were", "be", "are",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def b2s(v) -> str:
    """Decode a WaveDB value (bytes) to str; '' for missing/None.

    Local copy of ``store._b2s`` -- this module imports nothing from ``store``
    (avoids a cycle). Kept byte-identical to the store helper. PUBLIC so the
    query side (``src/retrieval/bm25.py``) decodes posting/doclen values the
    same way the index side wrote them.
    """
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)


def safe_term(term: str) -> str:
    """A key-path component for a posting/df key, hashing any term with ``/``
    or NUL.

    Local copy of ``store.safe_edge_component`` logic (store.py:85-100) -- this
    module imports nothing from ``store`` (avoids a cycle). A term like
    ``foo/bar`` -> ``h_<sha256[:16]>`` so a literal slash never splits the key
    into ``content/idx/tok/foo/bar`` (which would scan as a sub-prefix). The eid
    component (``ep_NNNNNN``) is slash-free so it is NOT hashed. PUBLIC so the
    query side builds the same scan key the index side wrote.
    """
    if "/" not in term and "\x00" not in term:
        return term
    return "h_" + hashlib.sha256(term.encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> list[str]:
    """``[a-z0-9]+`` lowercase word tokens minus the stopword set.

    Empty/None text -> []. ASCII-only (the codebase is ASCII). Used by BOTH the
    index path (``bm25_index_ops``) and the query path (``BM25Search``) so the
    index and the query share one tokenization -- a query token that was
    stopword-stripped at index time is also stripped at query time.
    """
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def read_stats(db) -> dict:
    """Read ``content/idx/stats`` -> ``{"N": int, "total_len": int}``; zeros if
    absent/corrupt (a corrupt stats is treated as empty, never raises). PUBLIC
    -- the query side reads stats for avgdl via this same helper."""
    raw = b2s(db.get_sync(STATS))
    if not raw:
        return {"N": 0, "total_len": 0}
    try:
        s = json.loads(raw)
    except (ValueError, TypeError):
        return {"N": 0, "total_len": 0}
    if not isinstance(s, dict):
        return {"N": 0, "total_len": 0}
    return {"N": int(s.get("N", 0) or 0), "total_len": int(s.get("total_len", 0) or 0)}


def read_int(db, key: str) -> int:
    """Read a str-encoded int key; 0 if absent/corrupt (never raises). PUBLIC --
    the query side reads df/doclen via this same helper."""
    raw = b2s(db.get_sync(key))
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def bm25_index_ops(db, eid: str, text: str) -> list[dict]:
    """Build the put ops to index ``eid`` for ``text``.

    Idempotent: reads ``content/idx/docterms/{eid}``; if present (already
    indexed) -> returns ``[]``. Episodes are append-only (a new ``eid`` per
    turn), so a duplicate ``_content_ops`` call is a SKIP, not a double-count --
    this guard makes a re-encode safe and makes the backfill script rerunnable.

    Else: reads current ``tokdf/{term}`` + ``stats`` (get_sync), increments df
    per unique term, ``N += 1``, ``total_len += doc_len``, and returns puts for
    postings + doclen + docterms + updated tokdf + updated stats. The caller
    (``_content_ops``) splices these into the SAME ``batch_sync`` as the
    content puts, so the index is atomic with the content (can't drift). Reads
    happen at op-build time, so they see committed state (encodes are serial on
    the main thread; the async-distill worker writes edges only, not content).
    """
    if b2s(db.get_sync(f"{_DOCTERMS}/{eid}")):
        return []  # already indexed (append-only guard)
    if not text:
        return []  # empty doc -> no postings, no docterms, no stats bump
    terms = tokenize(text)
    doc_len = len(terms)
    tf = Counter(terms)  # term -> count (unique keys)
    ops: list[dict] = []
    stats = read_stats(db)
    n_new = stats["N"] + 1
    total_len_new = stats["total_len"] + doc_len
    unique_terms = sorted(tf)  # deterministic op order
    for term in unique_terms:
        st = safe_term(term)
        ops.append({"type": "put", "key": f"{TOK}/{st}/{eid}",
                    "value": str(tf[term])})
        df = read_int(db, f"{TOKDF}/{st}")
        ops.append({"type": "put", "key": f"{TOKDF}/{st}",
                    "value": str(df + 1)})
    ops.append({"type": "put", "key": f"{DOCLEN}/{eid}", "value": str(doc_len)})
    ops.append({"type": "put", "key": f"{_DOCTERMS}/{eid}",
                "value": json.dumps(unique_terms)})
    ops.append({"type": "put", "key": STATS,
                "value": json.dumps({"N": n_new, "total_len": total_len_new})})
    return ops


def bm25_unindex_ops(db, eid: str) -> list[dict]:
    """Build del + decrement ops to remove ``eid`` from the index.

    Reads ``docterms/{eid}``; if absent -> ``[]`` (no-op -- never indexed, or
    already unindexed, or the flag was off at encode time). Else: for each term,
    ``del`` the posting + decrement ``tokdf`` (DEL the key when df->0, never
    write ``"0"`` -- a zero-df key would skew the corpus df and waste a scan);
    ``del doclen`` + ``del docterms``; decrement ``stats`` (``N -= 1``,
    ``total_len -= old doclen``, both floored at 0). The caller splices these
    into the state ``batch_sync`` (forget/supersede), mirroring
    ``_unindex_embedding``.
    """
    raw = b2s(db.get_sync(f"{_DOCTERMS}/{eid}"))
    if not raw:
        return []
    try:
        terms = json.loads(raw)
    except (ValueError, TypeError):
        terms = []
    if not isinstance(terms, list):
        terms = []
    ops: list[dict] = []
    stats = read_stats(db)
    old_doclen = read_int(db, f"{DOCLEN}/{eid}")
    for term in terms:
        st = safe_term(term)
        ops.append({"type": "del", "key": f"{TOK}/{st}/{eid}"})
        df = read_int(db, f"{TOKDF}/{st}")
        if df <= 1:
            ops.append({"type": "del", "key": f"{TOKDF}/{st}"})
        else:
            ops.append({"type": "put", "key": f"{TOKDF}/{st}",
                        "value": str(df - 1)})
    ops.append({"type": "del", "key": f"{DOCLEN}/{eid}"})
    ops.append({"type": "del", "key": f"{_DOCTERMS}/{eid}"})
    n_new = max(0, stats["N"] - 1)
    total_len_new = max(0, stats["total_len"] - old_doclen)
    ops.append({"type": "put", "key": STATS,
                "value": json.dumps({"N": n_new, "total_len": total_len_new})})
    return ops