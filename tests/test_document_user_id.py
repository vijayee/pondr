"""Tests for ``Document.user_id`` -- the write-side of the retrieval boundary.

A document with ``user_id`` set gets ownership graph edges
(``(U:{user}, owns_document, doc)`` + the reverse ``owned_by``) and a
``content/doc/{id}/user`` content key; ``get_document`` reads it back; and
``delete_document`` retracts the edges symmetrically (no orphans). A document
with ``user_id=None`` (pre-user-scope / unscoped) is byte-identical to today --
no edges, no content key.

Offline: installed ``wavedb`` (CPU) + the keyword extractor / bag-of-words
embedder stubs (reused from ``test_doc_retrieval``). No GLiNER/Bonsai.
"""

from __future__ import annotations

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)", allow_module_level=True)

from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.store import HippocampalStore, _b2s

from tests.test_doc_retrieval import _Bow384, _KWExtractor, _MD


def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": True, "embedding_dim": 384}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


def _ingest(store, tmp_path, *, user_id, source="doc.md"):
    src = tmp_path / source
    src.write_text(_MD, encoding="utf-8")
    chunker = HierarchicalChunker(max_section_tokens=200, min_section_tokens=1)
    pipe = UnifiedIngestionPipeline(store, chunker=chunker)
    doc_id, _ = pipe.ingest(
        str(src), extractor=_KWExtractor(), relation_extractor=None,
        embedder=_Bow384(), user_id=user_id,
    )
    return doc_id


def _has_triple(store, s, p, o):
    """True if the ``(s, p, o)`` SPO key exists in the graph index."""
    key = f"memory/spo/{s}/{p}/{o}"
    return store.db.get_sync(key) is not None


def test_encode_with_user_id_writes_ownership_edges_and_content_key(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, user_id="alice")
    assert _has_triple(store, "U:alice", "owns_document", doc_id)
    assert _has_triple(store, doc_id, "owned_by", "U:alice")
    assert _b2s(store.db.get_sync(f"content/doc/{doc_id}/user")) == "alice"
    store.close()


def test_get_document_reads_back_user_id(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, user_id="alice")
    doc = store.get_document(doc_id, load_bodies=False)
    assert doc is not None
    assert doc.user_id == "alice"
    store.close()


def test_encode_without_user_id_writes_no_ownership_edges(tmp_path):
    """A user_id=None doc is byte-identical to pre-user-scope: no edges, no key."""
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, user_id=None)
    assert not _has_triple(store, "U:alice", "owns_document", doc_id)
    assert not _has_triple(store, doc_id, "owned_by", "U:alice")
    assert store.db.get_sync(f"content/doc/{doc_id}/user") is None
    doc = store.get_document(doc_id, load_bodies=False)
    assert doc is not None
    assert doc.user_id is None
    store.close()


def test_delete_document_retracts_ownership_edges(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, user_id="alice")
    assert _has_triple(store, "U:alice", "owns_document", doc_id)
    assert store.delete_document(doc_id)
    # Both ownership edges + the content key are gone (no orphans).
    assert not _has_triple(store, "U:alice", "owns_document", doc_id)
    assert not _has_triple(store, doc_id, "owned_by", "U:alice")
    assert store.db.get_sync(f"content/doc/{doc_id}/user") is None
    assert store.get_document(doc_id, load_bodies=False) is None
    store.close()


def test_reingest_update_preserves_user_id(tmp_path):
    """Re-ingesting the same source (UPDATE path) keeps the doc's owner."""
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, user_id="alice", source="doc.md")
    # Re-ingest the same source (created=False -> update in place).
    doc_id2 = _ingest(store, tmp_path, user_id="alice", source="doc.md")
    assert doc_id == doc_id2
    assert _has_triple(store, "U:alice", "owns_document", doc_id)
    assert _b2s(store.db.get_sync(f"content/doc/{doc_id}/user")) == "alice"
    store.close()