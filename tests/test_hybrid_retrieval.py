"""A2: RRF hybrid BM25 + vector + graph retrieval (Tencent-survey item 3).

Two new modules: ``src/memory/bm25_index.py`` (ingest side -- a BM25 inverted
index hosted INSIDE WaveDB via HBTrie range scan, NO SQLite/FTS5, written in
the same atomic ``batch_sync`` as ``encode_episode``) and ``src/retrieval/bm25.py``
(query side -- ``BM25Search`` + ``rrf_fuse``). The retriever's
``_retrieve_hybrid`` fuses three ranked id-lists (graph, vector, BM25) via
Reciprocal Rank Fusion (``k=60``, parameter-free). Flag-gated
(``config.hybrid_retrieval``), default OFF, byte-identical when off.

These tests pin: the index writes + BM25Search finds lexical matches; BM25
ranks a lexical match first; RRF fusion unit behavior; the A2 value test (a
lexical-miss episode invisible to graph+vector surfaces via BM25+RRF); the
byte-identical-OFF guarantee; unindex-on-forget keeps the index clean;
user-scope filtering; ``safe_term`` handling of slash query terms; and stats
consistency across index/unindex.

Offline: installed ``wavedb`` (CPU). No GLiNER/Bonsai/torch.
"""

from __future__ import annotations

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)",
                allow_module_level=True)

from src.config import config
from src.memory.bm25_index import (
    bm25_index_ops,
    bm25_unindex_ops,
    read_stats,
    safe_term,
    tokenize,
)
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.bm25 import BM25Search, rrf_fuse
from src.retrieval.retriever import HippocampalRetriever


# ── helpers ───────────────────────────────────────────────────────────────────

class _StubPlanner:
    """Returns a fixed plan, ignoring the prompt (deterministic tests)."""

    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history: list | None = None) -> dict:
        return self._plan


def _ep(eid, entities=None, topics=None, summary=None, full_text=None,
        ts="2026-07-03T10:00:00"):
    return Episode(
        id=eid, timestamp=ts, summary=summary or f"summary {eid}",
        full_text=full_text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [],
    )


@pytest.fixture
def hybrid_on():
    """Set the master-config flag ON for tests that exercise the store's
    encode-time index hook (``store.encode_episode`` -> ``_content_ops``).
    Restored after so the global never leaks into sibling tests."""
    prev = config.hybrid_retrieval
    config.hybrid_retrieval = True
    try:
        yield
    finally:
        config.hybrid_retrieval = prev


# ── BM25 index + search ───────────────────────────────────────────────────────

def test_bm25_index_write_and_search(tmp_path, hybrid_on):
    """encode -> _content_ops indexes full_text; BM25Search finds it by lexicon."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", summary="db notes",
                             full_text="postgres WAL tuning and replication"))
    bm = BM25Search(store.db)
    hits = bm.search(tokenize("postgres wal"), k=10)
    ids = [eid for eid, _ in hits]
    assert "ep_001" in ids
    assert hits[0][1] > 0.0  # BM25 score is positive for a match
    store.close()


def test_bm25_ranks_lexical_match_first(tmp_path, hybrid_on):
    """Two episodes (postgres / redis); a postgres query ranks the postgres eid first."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_redis", summary="redis notes",
                             full_text="redis cache pipelining and persistence"))
    store.encode_episode(_ep("ep_pg", summary="pg notes",
                             full_text="postgres replication slots and WAL"))
    bm = BM25Search(store.db)
    hits = bm.search(tokenize("postgres replication"), k=10)
    ids = [eid for eid, _ in hits]
    assert ids and ids[0] == "ep_pg"
    store.close()


def test_bm25_empty_corpus_returns_empty(tmp_path):
    """No index -> N=0 -> search returns [] (never raises)."""
    store = HippocampalStore(str(tmp_path / "db"))
    bm = BM25Search(store.db)
    assert bm.search(tokenize("anything here"), k=10) == []
    assert bm.search([], k=10) == []  # empty terms
    store.close()


# ── RRF fusion ───────────────────────────────────────────────────────────────

def test_rrf_fuse_unit():
    """An eid at rank 0 in all three lists outscores one at rank 0 in one list;
    an eid absent from one list still scores from the other two."""
    fused = rrf_fuse([["a", "b", "c"], ["a", "x", "y"], ["a", "b", "z"]])
    scores = dict(fused)
    # 'a' is rank 0 in all three -> 3 * 1/61
    # 'b' is rank 1 in list 0 + rank 1 in list 2 -> 2 * 1/62
    assert scores["a"] > scores["b"]
    # 'x' appears in only one list (rank 1) -> 1/62
    assert "x" in scores
    assert scores["x"] == pytest.approx(1.0 / (60 + 1 + 1))


def test_rrf_fuse_missing_list():
    """An eid only in the graph list scores 1/(60+0+1) and appears in output."""
    fused = rrf_fuse([["only_graph"], ["a", "b"], ["c", "d"]])
    scores = dict(fused)
    assert "only_graph" in scores
    assert scores["only_graph"] == pytest.approx(1.0 / (60 + 0 + 1))
    # Empty lists / all-empty -> [].
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], [], []]) == []


# ── Retriever hybrid path ─────────────────────────────────────────────────────

def test_hybrid_retrieve_surfaces_lexical_miss(tmp_path, hybrid_on):
    """THE A2 value test. An episode whose full_text contains the query words
    but has NO entities/topics the plan matches (graph miss) is invisible to
    the graph path; with the flag ON, BM25+RRF surfaces it. With the flag OFF
    (graph returns >=3 -> no vector fallback), it stays invisible."""
    store = HippocampalStore(str(tmp_path / "db"))
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    # Three Alice episodes -> graph returns >=3 (so the off path's vector
    # fallback does NOT fire). Their full_text has no postgres words.
    for i in range(3):
        store.encode_episode(_ep(
            f"ep_a{i}", entities=["Alice"], summary=f"alice chat {i}",
            full_text=f"User: chatting about alice topic number {i}"))
    # The lexical-miss episode: no entities (graph-invisible) but full_text
    # carries the distinctive query words.
    store.encode_episode(_ep(
        "ep_target", entities=[], summary="target",
        full_text="postgres WAL tuning and replication setup"))

    # OFF: graph path only (3 Alice hits, no fallback) -> target absent.
    retr_off = HippocampalRetriever(store, planner=_StubPlanner(plan))
    off_ids = {r["episode_id"] for r in retr_off.retrieve("postgres replication")}
    assert "ep_target" not in off_ids

    # ON: BM25 finds the target by lexical match; RRF fuses it in.
    retr_on = HippocampalRetriever(store, planner=_StubPlanner(plan))
    retr_on.hybrid_retrieval = True
    retr_on.bm25 = BM25Search(store.db)
    on_results = retr_on.retrieve("postgres replication")
    on_ids = {r["episode_id"] for r in on_results}
    assert "ep_target" in on_ids
    # Provenance: the hybrid path stamps strategy="hybrid" on every result.
    target = next(r for r in on_results if r["episode_id"] == "ep_target")
    assert target["strategy"] == "hybrid"
    store.close()


def test_hybrid_off_byte_identical(tmp_path):
    """Flag OFF: a retriever with hybrid_retrieval=False/bm25=None produces the
    SAME retrieve() output as a baseline retriever built without touching the
    hybrid attrs. Pins that the early-return branch leaves the off path
    untouched (byte-identical-OFF)."""
    store = HippocampalStore(str(tmp_path / "db"))
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    store.encode_episode(_ep("ep_1", entities=["Alice"], summary="first",
                             full_text="User: alice one"))
    store.encode_episode(_ep("ep_2", entities=["Alice"], summary="second",
                             full_text="User: alice two"))

    baseline = HippocampalRetriever(store, planner=_StubPlanner(plan))
    explicit_off = HippocampalRetriever(store, planner=_StubPlanner(plan))
    explicit_off.hybrid_retrieval = False
    explicit_off.bm25 = None

    base = baseline.retrieve("alice")
    off = explicit_off.retrieve("alice")
    assert [r["episode_id"] for r in base] == [r["episode_id"] for r in off]
    assert [r.get("score") for r in base] == [r.get("score") for r in off]
    assert [r.get("kind") for r in base] == [r.get("kind") for r in off]
    # No strategy key leaks on the off path.
    assert all("strategy" not in r for r in off)
    store.close()


# ── Index lifecycle / scope ───────────────────────────────────────────────────

def test_bm25_unindex_on_forget(tmp_path, hybrid_on):
    """encode -> set_episode_state('deprecated') -> _unindex_embedding removes
    the BM25 postings; the eid no longer surfaces in BM25Search (the index is
    kept consistent with episode lifecycle, mirroring the vector unindex)."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", summary="db notes",
                             full_text="postgres WAL tuning and replication"))
    bm = BM25Search(store.db)
    assert "ep_001" in [e for e, _ in bm.search(tokenize("postgres"), k=10)]
    store.set_episode_state("ep_001", "deprecated")
    assert "ep_001" not in [e for e, _ in bm.search(tokenize("postgres"), k=10)]
    store.close()


def test_bm25_user_scope(tmp_path, hybrid_on):
    """alice/bob episodes; allowed_episode_ids={alice_ep} returns only alice's."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_alice", summary="alice",
                             full_text="postgres WAL tuning by alice"))
    store.encode_episode(_ep("ep_bob", summary="bob",
                             full_text="postgres WAL tuning by bob"))
    bm = BM25Search(store.db)
    hits = bm.search(tokenize("postgres wal"), k=10,
                     allowed_episode_ids={"ep_alice"})
    ids = [e for e, _ in hits]
    assert ids == ["ep_alice"]
    # Unscoped -> both.
    assert set(e for e, _ in bm.search(tokenize("postgres wal"), k=10)) == {"ep_alice", "ep_bob"}
    store.close()


def test_safe_term_handles_slash_in_query(tmp_path, hybrid_on):
    """A query term containing '/' is safe-encoded (hashed) so it does not
    split the range-scan key into a wrong prefix. ``tokenize``'s ``[a-z0-9]+``
    regex never EMITS slash-terms (it splits on '/'), so the slash-term has no
    posting -> returns nothing for it alone, but a co-occurring clean term still
    finds the episode and the search does not raise. ``safe_term`` unit below
    pins the hash."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_1", summary="s", full_text="foo bar baz qux"))
    bm = BM25Search(store.db)
    # "foo/bar" -> hashed -> no posting (index tokenized to foo/bar/baz/qux,
    # none of which is the literal "foo/bar"). "baz" matches -> ep_1 found.
    hits = bm.search(["foo/bar", "baz"], k=10)
    assert "ep_1" in [e for e, _ in hits]
    # Pure slash-term -> no crash, no hits.
    assert bm.search(["foo/bar"], k=10) == []
    store.close()


def test_safe_term_unit():
    """safe_term leaves a clean term unchanged; hashes a '/' or NUL term to
    'h_<sha256[:16]>' so it can never introduce a '/' into the key path."""
    assert safe_term("postgres") == "postgres"
    assert safe_term("foo/bar").startswith("h_")
    assert "/" not in safe_term("foo/bar")
    assert safe_term("a\x00b").startswith("h_")


def test_stats_consistent_after_index_unindex(tmp_path):
    """Direct index/unindex ops keep content/idx/stats consistent (N +
    total_len), so avgdl never drifts across a forget."""
    store = HippocampalStore(str(tmp_path / "db"))
    db = store.db
    db.batch_sync(bm25_index_ops(db, "ep_1", "alpha beta gamma delta"))
    db.batch_sync(bm25_index_ops(db, "ep_2", "epsilon zeta"))
    s = read_stats(db)
    assert s["N"] == 2
    assert s["total_len"] == len(tokenize("alpha beta gamma delta")) + len(tokenize("epsilon zeta"))
    # Idempotent re-index is a no-op (docterms guard).
    db.batch_sync(bm25_index_ops(db, "ep_1", "alpha beta gamma delta"))
    assert read_stats(db)["N"] == 2
    # Unindex ep_1 -> N=1, total_len reflects only ep_2.
    db.batch_sync(bm25_unindex_ops(db, "ep_1"))
    s = read_stats(db)
    assert s["N"] == 1
    assert s["total_len"] == len(tokenize("epsilon zeta"))
    # Unindex again (already gone) -> no-op, stats unchanged.
    db.batch_sync(bm25_unindex_ops(db, "ep_1"))
    assert read_stats(db)["N"] == 1
    store.close()