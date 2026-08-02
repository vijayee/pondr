"""Tests for the user-scope store primitives (the read side of the boundary).

``episode_ids_for_user`` / ``document_ids_for_user`` / ``claim_unscoped_documents``
are the only user-scoped reads the retriever consults; everything else
(``default_*_ids``) is across-all-users. These verify they return the RIGHT id
sets (a user's own episodes/docs, not another user's), are empty for an
unknown user, and that the backfill is idempotent + never clobbers an existing
owner.

Offline: installed ``wavedb`` (CPU), a keyword extractor stub + a 384-dim
bag-of-words embedder (reused from ``test_doc_retrieval``). No GLiNER/Bonsai.
"""

from __future__ import annotations

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)", allow_module_level=True)

from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.episode import Episode
from src.memory.store import HippocampalStore

# Shared offline fixtures (read-only test infrastructure).
from tests.test_doc_retrieval import _Bow384, _KWExtractor, _MD


def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": True, "embedding_dim": 384}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


def _ingest_doc(store, tmp_path, *, user_id, source="doc.md"):
    """Write the fixture markdown to disk + run the real pipeline with a user."""
    src = tmp_path / source
    src.write_text(_MD, encoding="utf-8")
    chunker = HierarchicalChunker(max_section_tokens=200, min_section_tokens=1)
    pipe = UnifiedIngestionPipeline(store, chunker=chunker)
    doc_id, _ = pipe.ingest(
        str(src), extractor=_KWExtractor(), relation_extractor=None,
        embedder=_Bow384(), user_id=user_id,
    )
    return doc_id


def _encode_episode(store, eid, *, user_id, session_id, entity="Alice"):
    store.encode_episode(Episode(
        id=eid, timestamp="2026-07-03T10:00:00",
        summary=f"{eid} summary", full_text=f"User: hi {eid}",
        entities=[entity], topics=["database_design"], tones=["curious"],
        user_id=user_id, session_id=session_id,
    ))


# ── episode_ids_for_user ──

def test_episode_ids_for_user_returns_only_that_users_episodes(tmp_path):
    store = _store(tmp_path)
    _encode_episode(store, "ep_001", user_id="alice", session_id="S:a1", entity="Alice")
    _encode_episode(store, "ep_002", user_id="alice", session_id="S:a1", entity="Alice")
    _encode_episode(store, "ep_003", user_id="bob", session_id="S:b1", entity="Bob")
    assert store.episode_ids_for_user("alice") == {"ep_001", "ep_002"}
    assert store.episode_ids_for_user("bob") == {"ep_003"}
    store.close()


def test_episode_ids_for_user_empty_for_unknown_user(tmp_path):
    store = _store(tmp_path)
    _encode_episode(store, "ep_001", user_id="alice", session_id="S:a1")
    assert store.episode_ids_for_user("nobody") == set()
    store.close()


def test_episode_ids_for_user_excludes_unscoped_episodes(tmp_path):
    """An episode encoded with no user_id is in NO user's set (strict scope)."""
    store = _store(tmp_path)
    # Unscoped episode (pre-provenance / no user_id).
    store.encode_episode(Episode(
        id="ep_old", timestamp="2026-01-01T00:00:00", summary="old",
        full_text="old", entities=["Alice"], topics=["database_design"],
    ))
    _encode_episode(store, "ep_001", user_id="alice", session_id="S:a1")
    assert store.episode_ids_for_user("alice") == {"ep_001"}
    assert "ep_old" not in store.episode_ids_for_user("alice")
    store.close()


# ── document_ids_for_user ──

def test_document_ids_for_user_returns_only_that_users_docs(tmp_path):
    store = _store(tmp_path)
    a_doc = _ingest_doc(store, tmp_path, user_id="alice", source="a.md")
    b_doc = _ingest_doc(store, tmp_path, user_id="bob", source="b.md")
    assert store.document_ids_for_user("alice") == {a_doc}
    assert store.document_ids_for_user("bob") == {b_doc}
    store.close()


def test_document_ids_for_user_empty_for_unknown_user(tmp_path):
    store = _store(tmp_path)
    _ingest_doc(store, tmp_path, user_id="alice", source="a.md")
    assert store.document_ids_for_user("nobody") == set()
    store.close()


def test_document_ids_for_user_excludes_unscoped_docs(tmp_path):
    """A doc ingested with user_id=None is in NO user's set until claimed."""
    store = _store(tmp_path)
    unscoped = _ingest_doc(store, tmp_path, user_id=None, source="u.md")
    assert store.document_ids_for_user("alice") == set()
    assert unscoped not in store.document_ids_for_user("alice")
    store.close()


# ── claim_unscoped_documents ──

def test_claim_unscoped_documents_stamps_ownerless_docs(tmp_path):
    store = _store(tmp_path)
    u1 = _ingest_doc(store, tmp_path, user_id=None, source="u1.md")
    u2 = _ingest_doc(store, tmp_path, user_id=None, source="u2.md")
    assert store.document_ids_for_user("alice") == set()
    claimed = store.claim_unscoped_documents("alice")
    assert claimed == 2
    assert store.document_ids_for_user("alice") == {u1, u2}
    # The content key is stamped too (the cheap per-doc owner read).
    from src.memory.store import _b2s
    assert _b2s(store.db.get_sync(f"content/doc/{u1}/user")) == "alice"
    store.close()


def test_claim_unscoped_documents_is_idempotent(tmp_path):
    store = _store(tmp_path)
    _ingest_doc(store, tmp_path, user_id=None, source="u.md")
    assert store.claim_unscoped_documents("alice") == 1
    # Second call: every doc is now owned -> 0 claimed, no clobber.
    assert store.claim_unscoped_documents("alice") == 0
    assert store.claim_unscoped_documents("bob") == 0  # bob does NOT steal it
    assert len(store.document_ids_for_user("alice")) == 1
    assert store.document_ids_for_user("bob") == set()
    store.close()


def test_claim_unscoped_documents_skips_already_owned(tmp_path):
    """A doc already owned by another user is NOT stolen by the claim."""
    store = _store(tmp_path)
    a_doc = _ingest_doc(store, tmp_path, user_id="alice", source="a.md")
    _ingest_doc(store, tmp_path, user_id=None, source="u.md")
    claimed = store.claim_unscoped_documents("bob")
    assert claimed == 1  # only the unscoped one
    assert store.document_ids_for_user("alice") == {a_doc}
    store.close()


def test_claim_unscoped_documents_no_docs(tmp_path):
    store = _store(tmp_path)
    assert store.claim_unscoped_documents("alice") == 0
    store.close()