"""Tests for the in-DB WaveDB VectorLayer integration (wavedb>=0.2.2).

Covers the store-owned layer lifecycle, the single ``_index_embedding``
chokepoint (live insert on encode + backfill on set_summary_embedding), the
``_unindex_embedding`` delete hook on forget AND supersede (the two-
chopepoint de-wonk: ``supersede_episode`` writes ``state="superseded"``
directly, bypassing ``set_episode_state``), the ``WavedbVectorStore`` adapter
(search + encode for the gate-embedder reuse), best-effort failure isolation,
lifecycle, and the ``vector_index_enabled=False`` gating fallback.

Requires the native VectorLayer (wavedb>=0.2.0); skipped cleanly otherwise.
Uses a 384-dim deterministic stub embedder (matching the layer's 384 dim) so
the vectors actually enter the in-DB COSINE index -- no faiss / numpy /
sentence-transformers needed.
"""

from __future__ import annotations

import hashlib

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)", allow_module_level=True)

from src.gnn.semantic_memory import SemanticMemoryWriter
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever
from src.retrieval.wavedb_vector_store import WavedbVectorStore


class _Bow384:
    """Deterministic 384-dim bag-of-words embedder (matches the layer dim)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for w in t.lower().split():
                w = "".join(c for c in w if c.isalnum())
                if not w:
                    continue
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            out.append(vec)
        return out


def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": True, "embedding_dim": 384}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


def _ep(eid, summary, embedding=None):
    return Episode(
        id=eid,
        timestamp="2026-07-14T10:00:00",
        summary=summary,
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=[],
        topics=[],
        tones=[],
        summary_embedding=embedding,
    )


# ── 1. lifecycle / ownership ──


def test_store_opens_vector_layer(tmp_path):
    store = _store(tmp_path)
    assert store.vector_layer is not None
    assert store.vector_layer.count() == 0
    store.close()


def test_close_then_reopen_preserves_vectors(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    vec = embed.encode(["alice database schema"])[0]
    store.encode_episode(_ep("ep_001", "alice database schema", embedding=vec))
    assert store.vector_layer.count() == 1
    store.close()
    # Reopen on the same path: vectors + __format persist; the layer reopens.
    store2 = _store(tmp_path)
    assert store2.vector_layer is not None
    assert store2.vector_layer.count() == 1
    hits = store2.vector_layer.search_sync(vec, 1)
    assert hits and hits[0].id_str == "ep_001"
    store2.close()


# ── 2. chokepoint: backfill path (set_summary_embedding) ──


def test_set_summary_embedding_indexes_and_searches(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "alice database schema"))
    vec = embed.encode(["alice database schema"])[0]
    store.set_summary_embedding("ep_001", vec)
    assert store.vector_layer.count() == 1
    hits = store.vector_layer.search_sync(vec, 1)
    assert hits and hits[0].id_str == "ep_001"
    store.close()


# ── 3. chokepoint: live path (encode_episode with embedding) ──


def test_encode_episode_with_embedding_is_immediately_searchable(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    vec = embed.encode(["wal config corruption"])[0]
    store.encode_episode(_ep("ep_001", "wal config corruption", embedding=vec))
    # No build_index call -- the live insert is the win.
    assert store.vector_layer.count() == 1
    hits = store.vector_layer.search_sync(vec, 1)
    assert hits and hits[0].id_str == "ep_001"
    store.close()


def test_encode_episode_without_embedding_does_not_index(tmp_path):
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "no embedding here"))  # no summary_embedding
    assert store.vector_layer.count() == 0
    store.close()


# ── 4 + 5. WavedbVectorStore adapter (search + encode) ──


def test_adapter_search_returns_similarity_best_first(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "wal config corruption",
                             embedding=embed.encode(["wal config corruption"])[0]))
    store.encode_episode(_ep("ep_002", "encryption key rotation",
                             embedding=embed.encode(["encryption key rotation"])[0]))
    adapter = WavedbVectorStore(store, embedder=embed)
    hits = adapter.search("wal config corruption", k=2)
    assert hits[0][0] == "ep_001"
    # Identical query/stored vector -> cosine distance 0 -> similarity 1.0.
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    # Adapter exposes .encode (Embedder protocol for the gate-embedder reuse).
    enc = adapter.encode(["wal config corruption"])
    assert len(enc) == 1 and len(enc[0]) == 384
    store.close()


def test_adapter_search_returns_empty_when_layer_disabled(tmp_path):
    store = _store(tmp_path, vector_index_enabled=False)
    assert store.vector_layer is None
    adapter = WavedbVectorStore(store, embedder=_Bow384())
    assert adapter.search("anything", k=5) == []
    store.close()


# ── 6. delete-on-forget (set_episode_state deprecated) ──


def test_forget_removes_episode_from_vector_index(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    vec = embed.encode(["alice database schema"])[0]
    store.encode_episode(_ep("ep_001", "alice database schema", embedding=vec))
    assert store.vector_layer.count() == 1
    store.set_episode_state("ep_001", "deprecated")
    assert store.vector_layer.count() == 0
    assert store.vector_layer.search_sync(vec, 1) == []
    # Content stays (soft-forget): retrievable via include_inactive.
    assert store.get_episode("ep_001") is not None
    store.close()


# ── 7. delete-on-supersede (the two-chokepoint de-wonk) ──


def test_supersede_episode_removes_old_keeps_new(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    vec_old = embed.encode(["alice database schema"])[0]
    vec_new = embed.encode(["alice database schema v2"])[0]
    store.encode_episode(_ep("ep_001", "alice database schema", embedding=vec_old))
    store.encode_episode(_ep("ep_002", "alice database schema v2", embedding=vec_new))
    assert store.vector_layer.count() == 2
    # supersede_episode writes state="superseded" DIRECTLY (not via
    # set_episode_state); hooking only set_episode_state would leave ep_001 in
    # the vector index. This test catches that regression.
    SemanticMemoryWriter(store).supersede_episode("ep_002", "ep_001")
    assert store.vector_layer.count() == 1
    ids = {r.id_str for r in store.vector_layer.search_sync(vec_old, 5)}
    assert "ep_001" not in ids
    assert "ep_002" in {r.id_str for r in store.vector_layer.search_sync(vec_new, 5)}
    store.close()


# ── 8. best-effort failure isolation ──


def test_index_embedding_failure_is_logged_not_raised(tmp_path, capsys):
    embed = _Bow384()
    store = _store(tmp_path)
    vec = embed.encode(["alice database schema"])[0]

    def boom(*a, **k):
        raise RuntimeError("simulated vector-layer failure")

    store.vector_layer.insert_sync = boom
    # Must not raise; the episode still encodes.
    store.encode_episode(_ep("ep_001", "alice database schema", embedding=vec))
    captured = capsys.readouterr()
    assert "[vector-index-fail]" in captured.err
    assert store.get_episode("ep_001") is not None
    store.close()


# ── 10. gating: vector_index_enabled=False falls back to FAISS VectorSearch ──


def test_disabled_vector_layer_falls_back_to_vectorsearch(tmp_path):
    store = _store(tmp_path, vector_index_enabled=False)
    assert store.vector_layer is None
    # No FAISS ids file on disk -> _try_load_vector_index leaves vector_search
    # None (graph-only), i.e. it does NOT attach the WavedbVectorStore adapter.
    retr = HippocampalRetriever(store, auto_load_index=True)
    assert retr.vector_search is None
    store.close()


def test_retriever_auto_load_prefers_wavedb_adapter(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    store.encode_episode(_ep("ep_001", "wal config corruption",
                             embedding=embed.encode(["wal config corruption"])[0]))
    retr = HippocampalRetriever(store, auto_load_index=True)
    assert isinstance(retr.vector_search, WavedbVectorStore)
    store.close()