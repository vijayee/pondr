"""Offline tests for VectorSearch + the retriever's semantic fallback.

No faiss / numpy / sentence-transformers required: these run against the
pure-Python cosine backend with a deterministic bag-of-words stub embedder
(word overlap → cosine similarity). The pod run exercises the faiss +
sentence-transformers path; this validates the indexing/search/persist logic
and the retriever wiring.
"""

from __future__ import annotations

import hashlib

import pytest

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever
from src.retrieval.vector_search import VectorSearch, _l2_normalize


class _BowEmbedder:
    """Deterministic bag-of-words embedder: word overlap → cosine similarity.

    Each word hashes to a dim index and increments that slot, so texts sharing
    words have higher cosine. ``encode`` returns plain python lists (no numpy).
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for w in t.lower().split():
                # strip punctuation so "wal" and "wal," match
                w = "".join(c for c in w if c.isalnum())
                if not w:
                    continue
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            out.append(vec)
        return out


def _ep(eid, summary, entities=None, topics=None, tones=None):
    return Episode(
        id=eid, timestamp="2026-07-03T10:00:00", summary=summary,
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
    )


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


# ── VectorSearch ──


def test_build_and_search_ranks_word_overlap(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "WAL config pain and corruption"))
    store.encode_episode(_ep("ep_002", "encryption key rotation schedule"))
    store.encode_episode(_ep("003", "alice decided on the database schema"))
    vs = VectorSearch(store, embedder=_BowEmbedder())

    n = vs.build_index()
    assert n == 3

    hits = vs.search("wal config corruption", k=3)
    assert hits[0][0] == "ep_001"  # shares wal/config/corruption-ish
    store.close()


def test_build_index_skips_empty_summaries(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "real summary here"))
    store.encode_episode(_ep("ep_002", ""))  # empty summary → skipped
    vs = VectorSearch(store, embedder=_BowEmbedder())
    assert vs.build_index() == 1
    assert vs._ids == ["ep_001"]
    store.close()


def test_empty_store_builds_zero_index(tmp_path):
    store = _store(tmp_path)
    vs = VectorSearch(store, embedder=_BowEmbedder())
    assert vs.build_index() == 0
    assert vs.search("anything", k=5) == []
    store.close()


def test_search_returns_at_most_k(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.encode_episode(_ep(f"ep_{i:03d}", f"shared word number {i}"))
    vs = VectorSearch(store, embedder=_BowEmbedder())
    vs.build_index()
    hits = vs.search("shared word", k=2)
    assert len(hits) == 2
    store.close()


def test_save_load_round_trip_preserves_ranking(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "wal config pain"))
    store.encode_episode(_ep("ep_002", "encryption key rotation"))
    db = str(tmp_path / "db")
    vs = VectorSearch(store, embedder=_BowEmbedder())
    vs.build_index()
    vs.save(db)
    before = vs.search("wal config", k=2)
    store.close()

    # Reload into a fresh store + VectorSearch. Index vectors are persisted
    # (pure-Python fallback file); the embedder is still needed to encode the
    # query at search time.
    store2 = _store(tmp_path)
    vs2 = VectorSearch(store2, embedder=_BowEmbedder())
    n = vs2.load(db)
    assert n == 2
    after = vs2.search("wal config", k=2)
    assert [eid for eid, _ in after] == [eid for eid, _ in before]
    assert after[0][0] == "ep_001"
    store2.close()


def test_build_index_persists_embeddings_and_caches(tmp_path):
    """Second build_index reads cached embeddings — embedder not called again."""
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "wal config pain"))
    emb = _BowEmbedder()
    vs = VectorSearch(store, embedder=emb)
    vs.build_index()
    first_calls = emb.calls
    # Embedding now persisted under content/ep/ep_001/embedding.
    raw = store.db.get_sync("content/ep/ep_001/embedding")
    assert raw, "embedding was not persisted"

    # Second build reads from the store — no new embedder calls.
    vs.build_index()
    assert emb.calls == first_calls
    store.close()


def test_l2_normalize_basic():
    v = _l2_normalize([3.0, 4.0])
    assert abs(sum(x * x for x in v) - 1.0) < 1e-9
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]  # zero vector unchanged


class _NumpyLikeVec:
    """Mimics a numpy array: has .tolist() and is iterable. sentence-transformers
    returns numpy float32 arrays whose scalars json.dumps can't serialize;
    VectorSearch._embed must convert via .tolist()+float()."""

    def __init__(self, vals):
        self._v = vals

    def tolist(self):
        return self._v

    def __iter__(self):
        return iter(self._v)


class _NumpyLikeEmbedder:
    def __init__(self, dim=8):
        self.dim = dim

    def encode(self, texts):
        return [_NumpyLikeVec([float(len(t) % (i + 1)) for i in range(self.dim)])
                for t in texts]


def test_build_index_handles_numpy_like_vectors(tmp_path):
    """Embedder returns numpy-like vectors (with .tolist()); embeddings persist
    as JSON-serializable Python floats (regression for float32 serialization)."""
    import json
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "summary one"))
    store.encode_episode(_ep("ep_002", "summary two"))
    vs = VectorSearch(store, embedder=_NumpyLikeEmbedder())
    n = vs.build_index()
    assert n == 2
    # Persisted embeddings must be JSON (Python floats, not numpy scalars).
    raw = store.db.get_sync("content/ep/ep_001/embedding")
    parsed = json.loads(raw)  # would raise if numpy scalars slipped through
    assert all(isinstance(x, float) for x in parsed)
    # Search works end-to-end with the numpy-like embedder.
    hits = vs.search("summary one", k=2)
    assert hits[0][0] == "ep_001"
    store.close()


# ── retriever semantic fallback wiring ──


class _StubPlanner:
    """Returns a plan that matches nothing graph-side, forcing semantic fallback."""

    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str) -> dict:
        return self._plan


def test_retriever_semantic_fallback_fills_subthree_results(tmp_path):
    """Graph traversal returns <3 → semantic fallback adds vector hits."""
    store = _store(tmp_path)
    # Episodes that the graph axis won't match but the embeddings will.
    store.encode_episode(_ep("ep_001", "wal config pain",
                             tones=["curious"], entities=["WAL"]))
    store.encode_episode(_ep("ep_002", "wal corruption on reopen",
                             tones=["curious"], entities=["WAL"]))
    store.encode_episode(_ep("ep_003", "encryption key rotation",
                             tones=["curious"], entities=["RSA"]))
    vs = VectorSearch(store, embedder=_BowEmbedder())
    vs.build_index()

    # Plan asks for a tone the episodes don't have → graph returns 0 → fallback.
    retr = HippocampalRetriever(store, planner=_StubPlanner(
        {"tones": ["frustrated"], "entity_mode": "union"}))
    retr.vector_search = vs
    results = retr.retrieve("wal config corruption", use_semantic=True)

    ids = {r["episode_id"] for r in results}
    assert "ep_001" in ids  # semantic hit (shares wal/config)
    # Semantic hits carry the 0.5 discount.
    for r in results:
        if r["episode_id"] in {"ep_001", "ep_002"}:
            assert r["score"] <= 0.5 + 1e-9
    store.close()


def test_retriever_semantic_disabled_when_no_index(tmp_path):
    """No vector_search → fallback is a no-op (graph-only retrieval)."""
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "wal config pain", tones=["frustrated"]))
    retr = HippocampalRetriever(store, planner=_StubPlanner(
        {"tones": ["frustrated"], "entity_mode": "union"}))
    assert retr.vector_search is None
    results = retr.retrieve("wal config pain", use_semantic=True)
    assert [r["episode_id"] for r in results] == ["ep_001"]
    store.close()


def test_retriever_auto_loads_persisted_index(tmp_path):
    """auto_load_index=True attaches a persisted VectorSearch."""
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "wal config pain"))
    db = str(tmp_path / "db")
    vs = VectorSearch(store, embedder=_BowEmbedder())
    vs.build_index()
    vs.save(db)
    store.close()

    store2 = _store(tmp_path)
    retr = HippocampalRetriever(store2, auto_load_index=True)
    assert retr.vector_search is not None
    # On the pod the auto-loaded VectorSearch lazy-loads sentence-transformers
    # for the query embedding; here we inject the stub embedder it would use.
    retr.vector_search.embedder = _BowEmbedder()
    hits = retr.vector_search.search("wal config", k=1)
    assert hits[0][0] == "ep_001"
    store2.close()


def test_auto_load_index_missing_is_noop(tmp_path):
    """No persisted index → auto_load leaves vector_search None (no error)."""
    store = _store(tmp_path)
    retr = HippocampalRetriever(store, auto_load_index=True)
    assert retr.vector_search is None
    store.close()