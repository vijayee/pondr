"""Tests for ``DocumentRetriever`` -- the Phase 1c aggregation layer.

Phase 1c Refinement 1 closes the RAG-replacement pillar's aggregation gap: a
query matching MULTIPLE sections of the same document must surface ONE
document result with the matched sections highlighted + counted, not a wall
of separate section chunks. ``DocumentRetriever.aggregate_results`` groups
``kind="section"`` (semantic-fallback) and ``kind="document"`` (graph-path)
results by their parent document and builds one document result per parent;
conversation episodes / semantic memories pass through unchanged.

Reuses the fixtures from ``test_doc_retrieval`` (the shared doc-ingestion
helpers): ``_Bow384`` (384-dim bag-of-words embedder), ``_KWExtractor`` (keyword
entity/topic stub), ``_store`` (wavedb + vector layer), ``_MD`` (a 2-section
markdown doc: Alice/Storage + Bob/Networking), and ``_ingest`` (runs the real
``UnifiedIngestionPipeline``).

Covers:

* Multi-section aggregation: a semantic-fallback query hitting BOTH sections
  -> ONE document result with ``matched_sections == 2``, ``total_sections``
  correct, ``kind="document"``, both headings in the summary.
* Single-section aggregation: one section hit -> one document result with
  ``matched_sections == 1``.
* Graph-doc + section merge: a graph-path document result and a semantic-
  fallback section result for the SAME doc merge into one document result
  (no duplicate doc entries).
* Conversation pass-through: a plain episode result (no ``kind``) passes
  through unchanged alongside the aggregated document result.
* ``store_has_documents`` guard: True for a store with ingested docs, False
  for an empty / conversation-only store.
* Retriever hook: ``HippocampalRetriever.retrieve`` aggregates when
  ``document_retriever`` is attached and is byte-identical when it is
  ``None`` (the conversation-only path).
"""

from __future__ import annotations

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)", allow_module_level=True)

from src.retrieval.document_retriever import DocumentRetriever, store_has_documents
from src.retrieval.retriever import HippocampalRetriever
from src.retrieval.wavedb_vector_store import WavedbVectorStore

# Shared doc-ingestion fixtures (kept in test_doc_retrieval to avoid
# duplication; imported here as read-only test infrastructure).
from tests.test_doc_retrieval import _Bow384, _MD, _ingest, _store  # noqa: F401


def _retriever_with_semantic(store, embed, planner_entities):
    """Build a retriever whose graph path misses (so the semantic fallback
    fires) and attach the DocumentRetriever + a real vector layer store."""
    class _StubPlanner:
        def plan(self, prompt, history=None):
            # An entity the graph has no edge for -> traversal returns [] ->
            # the retriever's semantic fallback fires (<3 results).
            return {"entities": planner_entities, "entity_mode": "union"}

    retr = HippocampalRetriever(store, planner=_StubPlanner())
    retr.vector_search = WavedbVectorStore(store, embedder=embed)
    retr.document_retriever = DocumentRetriever(store)
    return retr


# ── aggregation ──

def test_multi_section_query_aggregates_into_one_document(tmp_path):
    """A semantic-fallback query hitting BOTH sections -> ONE document result."""
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)
    doc = store.get_document(doc_id, load_bodies=False)
    total = len(doc.sections)
    assert total >= 2

    retr = _retriever_with_semantic(store, embed, planner_entities=["zzzznomatch"])
    # A query whose bag-of-words overlaps BOTH sections ("alice storage" +
    # "bob networking") -> the sections surface via the fallback. (With a tiny
    # corpus the top-k fallback returns all sections, so ``matched_sections``
    # equals ``total_sections`` -- the assertion is "multiple sections aggregate
    # into ONE document result", not "exactly two".)
    results = retr.retrieve("Alice Bob storage networking", use_semantic=True)

    doc_results = [r for r in results if r.get("kind") == "document"]
    assert len(doc_results) == 1, "matched sections should aggregate into one doc"
    r = doc_results[0]
    assert r["episode_id"] == doc_id
    assert r["matched_sections"] >= 2, "at least two sections counted as matched"
    assert r["matched_sections"] == r["total_sections"] == total
    assert "Alice" in r["summary"] and "Bob" in r["summary"], (
        "both matched-section headings should be in the summary"
    )
    store.close()


def test_single_section_aggregates_into_one_document(tmp_path):
    """A single section result fed to the aggregator -> ONE document result
    with ``matched_sections == 1`` (the controlled single-section case; the
    live semantic fallback returns top-k, so a corpus-wide single-section hit
    is exercised via a direct ``aggregate_results`` call)."""
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)
    doc = store.get_document(doc_id, load_bodies=False)
    sec = doc.sections[0]
    section_result = {
        "episode_id": sec.id, "kind": "section", "summary": doc.title,
        "text": store.get_section_body(doc_id, sec.id) or "", "timestamp": doc.ingested_at,
        "entities": list(sec.entities), "topics": list(sec.topics),
        "tones": [], "decisions": [], "session_id": None, "user_id": None,
        "follows": None, "score": 0.8, "source_path": doc.source_path,
        "section_heading": sec.heading, "doc_id": doc_id,
    }
    agg = DocumentRetriever(store)
    out = agg.aggregate_results([section_result])
    doc_results = [r for r in out if r.get("kind") == "document"]
    assert len(doc_results) == 1
    assert doc_results[0]["matched_sections"] == 1
    assert doc_results[0]["episode_id"] == doc_id
    store.close()


def test_graph_doc_and_section_merge_into_one(tmp_path):
    """A graph-path doc hit + a semantic section hit for the SAME doc merge."""
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)
    # Synthesize a graph-path document result (kind="document", one matched
    # section body in text) + a section result for the same doc, and feed both
    # directly to the aggregator.
    doc = store.get_document(doc_id, load_bodies=False)
    sec_id = doc.sections[0].id
    sec_body = store.get_section_body(doc_id, sec_id) or ""
    graph_doc_result = {
        "episode_id": doc_id, "kind": "document", "summary": doc.title,
        "text": sec_body, "timestamp": doc.ingested_at,
        "entities": list(doc.entities), "topics": list(doc.topics),
        "tones": [], "decisions": [], "session_id": None, "user_id": None,
        "follows": None, "score": 0.9, "source_path": doc.source_path,
        "matched_section": doc.sections[0].heading,
    }
    section_result = {
        "episode_id": sec_id, "kind": "section", "summary": doc.title,
        "text": sec_body, "timestamp": doc.ingested_at,
        "entities": [], "topics": [], "tones": [], "decisions": [],
        "session_id": None, "user_id": None, "follows": None, "score": 0.5,
        "source_path": doc.source_path, "section_heading": doc.sections[0].heading,
        "doc_id": doc_id,
    }
    agg = DocumentRetriever(store)
    out = agg.aggregate_results([graph_doc_result, section_result])
    doc_results = [r for r in out if r.get("kind") == "document"]
    assert len(doc_results) == 1, "graph doc + section must merge (no duplicate)"
    assert doc_results[0]["episode_id"] == doc_id
    # Both contributions counted as matched sections.
    assert doc_results[0]["matched_sections"] == 2
    store.close()


def test_conversation_episode_passes_through(tmp_path):
    """A plain episode result (no kind) passes through unchanged."""
    embed = _Bow384()
    store = _store(tmp_path)
    _ingest(store, tmp_path, extractor=None, embedder=embed)
    ep = {
        "episode_id": "ep_0001", "summary": "a chat", "text": "hello",
        "timestamp": "2026-01-01", "entities": ["Alice"], "topics": ["Storage"],
        "tones": ["neutral"], "decisions": [], "session_id": "sess_1",
        "user_id": "u_1", "follows": None, "score": 0.7,
    }
    agg = DocumentRetriever(store)
    out = agg.aggregate_results([ep])
    # The episode passes through unchanged (no kind -> not a document section).
    assert ep in out
    assert all(r.get("kind") != "document" for r in out)
    store.close()


# ── store_has_documents guard ──

def test_store_has_documents_true_for_doc_store(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    assert store_has_documents(store) is False  # before ingest: no doc edges
    _ingest(store, tmp_path, extractor=None, embedder=embed)
    assert store_has_documents(store) is True  # after ingest: has_section edges
    store.close()


def test_store_has_documents_false_for_empty_store(tmp_path):
    store = _store(tmp_path)
    assert store_has_documents(store) is False
    store.close()


# ── retriever hook (None = byte-identical pre-1c path) ──

def test_retriever_without_document_retriever_is_passthrough(tmp_path):
    """``document_retriever is None`` -> aggregation is a no-op (results returned
    as sections, unchanged)."""
    embed = _Bow384()
    store = _store(tmp_path)
    _ingest(store, tmp_path, extractor=None, embedder=embed)

    class _StubPlanner:
        def plan(self, prompt, history=None):
            return {"entities": ["zzzznomatch"], "entity_mode": "union"}

    retr = HippocampalRetriever(store, planner=_StubPlanner())
    retr.vector_search = WavedbVectorStore(store, embedder=embed)
    # document_retriever left None (the conversation-only / pre-1c path).
    assert retr.document_retriever is None
    results = retr.retrieve("Alice storage", use_semantic=True)
    # No aggregation -> section results pass through as kind="section".
    sec_results = [r for r in results if r.get("kind") == "section"]
    assert sec_results, "section results pass through when no aggregator"
    assert not any(r.get("kind") == "document" for r in results), (
        "no document aggregation should occur without the aggregator"
    )
    store.close()


# ── build_ponder integration: the store_has_documents guard drives attachment ──

def test_build_ponder_attaches_document_retriever_when_docs_present(tmp_path):
    """``build_ponder`` attaches ``DocumentRetriever`` iff the store has docs.

    With an ingested document (``has_section`` edges present), the built
    orchestrator's retriever carries a live ``DocumentRetriever``. This exercises
    the full ``store_has_documents`` -> attach path end-to-end (the runtime
    entrypoint, not a manually-attached aggregator).
    """
    from src.runtime import DEFAULT_BACKBONE_PATH, DEFAULT_GATE_PATH, build_ponder
    from src.memory.store import HippocampalStore

    embed = _Bow384()
    db_path = str(tmp_path / "memory_db")
    # Ingest a doc into the SAME store path build_ponder will reopen, so the
    # ``has_section`` edges are present when the guard probe runs.
    store = HippocampalStore(db_path, config={"vector_index_enabled": True,
                                              "embedding_dim": 384})
    _ingest(store, tmp_path, extractor=None, embedder=embed)
    store.close()

    orch = build_ponder(
        db_path,
        backbone_path=DEFAULT_BACKBONE_PATH,
        gate_path=DEFAULT_GATE_PATH,
        embedder_source="stub",
        device="cpu",
        live_encode=False,
    )
    try:
        assert orch.retriever.document_retriever is not None, (
            "build_ponder must attach a DocumentRetriever when the store has docs"
        )
    finally:
        orch.store.close()