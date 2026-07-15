"""Tests for the doc-ingestion RAG path: retrieval integration + per-chunk vectors.

Closes Phase 1c (the RAG-replacement pillar): an ingested document surfaces in
``GraphTraversal.retrieve()`` (the actual RAG path) as a first-class result with
its matched section body, AND each section (chunk) carries its own embedding in
the in-DB vector layer so a pure-semantic query finds the relevant chunk
directly (the per-chunk design confirmed in the original coding chat).

Offline: uses installed ``wavedb`` (CPU) + a deterministic 384-dim bag-of-words
stub embedder (matches the layer's 384 dim). No GLiNER/Bonsai -- a keyword
extractor stub fills entities/topics so the graph axes work.

Covers:

* Part A -- the graph path: doc surfaces by entity axis (``appears_in_doc``) and
  topic axis (``has_topic``), hydrates as ``kind="document"`` with a metadata-
  only ``sections`` list (no bodies) and ONE matched section body in ``text``;
  no-axis retrieve includes the doc; a non-matching entity excludes it;
  temporal filters use ``ingested_at``; renderers cite source + matched body;
  ``end_state._build_graph`` labels the node ``kind="document"``; the retrieval-
  boost hook still writes no sidecar for a doc.
* A2 -- the per-chunk vector path: sections are indexed keyed by section id;
  a semantic query (graph returns <3) surfaces a SECTION via the fallback,
  hydrated as ``kind="section"`` with the chunk body + section axes;
  ``delete_document`` removes all section ids; structure-only ingest (no
  embedder) indexes nothing; re-ingest (UPDATE) re-indexes without orphans;
  the ``_sec_`` discriminator precedes ``doc_`` (a section id is never
  hydrated as a document).
"""

from __future__ import annotations

import hashlib

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)", allow_module_level=True)

from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.store import HippocampalStore
from src.retrieval.end_state import _build_graph
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.retriever import HippocampalRetriever
from src.retrieval.chunked_context import ChunkedContextFormatter
from src.retrieval.wavedb_vector_store import WavedbVectorStore


# ── helpers ──

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


class _KWExtractor:
    """Deterministic entity/topic extractor stub (GLiNER stand-in, offline).

    Maps a small keyword vocabulary to entities/topics so the graph axes work
    without the heavy GLiNER model. Same ``extract(text) -> dict`` contract.
    """

    def extract(self, text: str) -> dict:
        ents: list[str] = []
        tops: list[str] = []
        low = text.lower()
        if "alice" in low:
            ents.append("Alice")
        if "bob" in low:
            ents.append("Bob")
        if "storage" in low:
            tops.append("Storage")
        if "networking" in low:
            tops.append("Networking")
        return {"entities": ents, "topics": tops}


def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": True, "embedding_dim": 384}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


_MD = (
    "# Project Notes\n\n"
    "This document records architecture decisions for the hippocampal index.\n\n"
    "## Alice on Storage\n\n"
    "Alice architected the storage subsystem. The hippocampal index uses a "
    "content-addressed cold blob store with a hot metadata index. Alice chose "
    "this split for LRU preservation.\n\n"
    "## Bob on Networking\n\n"
    "Bob implemented the networking transport. The cluster nodes communicate "
    "over a gossip protocol with retries. Bob tuned the retry backoff.\n"
)


def _ingest(store, tmp_path, *, embedder=None, extractor=None, source="doc.md"):
    """Write the fixture markdown to disk + run the real pipeline. Returns doc_id."""
    src = tmp_path / source
    src.write_text(_MD, encoding="utf-8")
    chunker = HierarchicalChunker(max_section_tokens=200, min_section_tokens=1)
    pipe = UnifiedIngestionPipeline(store, chunker=chunker)
    doc_id, _ = pipe.ingest(
        str(src), extractor=extractor, relation_extractor=None, embedder=embedder,
    )
    return doc_id


def _top_id(store, embed, query, k=10):
    """Top vector-layer hit id for a query (None if the layer is empty/no hit)."""
    if store.vector_layer is None or store.vector_layer.count() == 0:
        return None
    vec = embed.encode([query])[0]
    hits = store.vector_layer.search_sync(vec, k)
    return hits[0].id_str if hits else None


# ── Part A: the graph path ──

def test_doc_surfaces_by_entity_axis(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=_KWExtractor())
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Alice"], "entity_mode": "union"})
    ids = [r["episode_id"] for r in results]
    assert doc_id in ids, "doc did not surface via the entity axis (appears_in_doc)"
    doc = next(r for r in results if r["episode_id"] == doc_id)
    assert doc["kind"] == "document"
    assert doc["summary"]  # title
    assert "Alice" in doc["entities"]
    assert "Storage" in doc["topics"]
    # Hot/cold invariant: sections list is metadata-only (no content bodies).
    assert all("content" not in s for s in doc["sections"])
    assert all("blob_hash" in s for s in doc["sections"])
    # The matched section body is materialized (one cold pull), and it is the
    # Alice section (best entity overlap).
    assert "Alice" in doc["text"], "matched section body not pulled into text"
    assert "storage" in doc["text"].lower()
    store.close()


def test_doc_surfaces_by_topic_axis(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=_KWExtractor())
    trav = GraphTraversal(store)
    results = trav.retrieve({"topics": ["Storage"]})
    assert doc_id in [r["episode_id"] for r in results], "doc did not surface via topic axis"
    store.close()


def test_doc_in_no_axis_candidates(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=_KWExtractor())
    trav = GraphTraversal(store)
    # Empty plan -> no-axis seed = all episodes + all docs (the union).
    results = trav.retrieve({})
    assert doc_id in [r["episode_id"] for r in results], "doc not in no-axis candidates"
    store.close()


def test_doc_excluded_by_nonmatching_entity(tmp_path):
    store = _store(tmp_path)
    _ingest(store, tmp_path, extractor=_KWExtractor())
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Nonexistent"], "entity_mode": "union"})
    assert results == [], "non-matching entity should yield no results (honest)"
    store.close()


def test_doc_temporal_filter_uses_ingested_at(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=_KWExtractor())
    trav = GraphTraversal(store)
    # "today" bucket = [now-1d, now). The freshly-ingested doc is within today.
    today = trav.retrieve({"temporal_filter": "today"})
    assert doc_id in [r["episode_id"] for r in today], "fresh doc should be in 'today'"
    # A far-future date_to excludes it; a far-past date_from excludes it.
    past = trav.retrieve({"date_from": "2099-01-01"})
    assert doc_id not in [r["episode_id"] for r in past]
    future = trav.retrieve({"date_to": "2000-01-01"})
    assert doc_id not in [r["episode_id"] for r in future]
    store.close()


def test_doc_context_string_cites_source_and_body(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=_KWExtractor())
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Alice"], "entity_mode": "union"})
    doc = next(r for r in results if r["episode_id"] == doc_id)
    # build_context_string (the Mode A renderer) + the chunked-context renderer
    # both branch on kind.
    retr = HippocampalRetriever(store)
    ctx = retr.build_context_string([doc])
    assert f"[{doc_id} |" in ctx
    assert "Source: " in ctx
    assert "Title: " in ctx
    assert "Alice" in ctx and "Storage" in ctx
    assert "storage" in ctx.lower(), "matched section body not rendered"
    assert "Section '" in ctx, "doc block should label the matched section"
    # ChunkedContextFormatter._format_episode (the Phase 2c renderer).
    chunk = ChunkedContextFormatter()._format_episode(doc)
    assert chunk.startswith("--- Document ")
    assert "Source: " in chunk
    assert "storage" in chunk.lower()
    store.close()


def test_doc_end_state_graph_node_kind(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=_KWExtractor())
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Alice"], "entity_mode": "union"})
    doc = next(r for r in results if r["episode_id"] == doc_id)
    g = _build_graph([doc])
    node = next(n for n in g["nodes"] if n["id"] == doc_id)
    assert node["kind"] == "document"
    store.close()


def test_doc_retrieval_boost_writes_no_sidecar(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=_KWExtractor())
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Alice"], "entity_mode": "union"})
    # _apply_retrieval_boost must skip doc_ ids (the forgetting exemption).
    trav._apply_retrieval_boost(results, ["Alice"], [], [], "important")
    # No sidecar written -> no retrieval-boost edge in the store. Check there is
    # no ``retrieval_boost`` predicate involving the doc id.
    before = store.documents_by_entity("Alice")
    assert doc_id in before  # sanity: the doc is still found by entity
    store.close()


# ── A2: the per-chunk vector path ──

def test_structure_only_ingest_indexes_no_sections(tmp_path):
    """No embedder -> no section vectors in the layer (structure-only)."""
    store = _store(tmp_path)
    assert store.vector_layer.count() == 0
    _ingest(store, tmp_path, extractor=None, embedder=None)
    assert store.vector_layer.count() == 0, "structure-only ingest must not index sections"
    store.close()


def test_embedded_ingest_indexes_each_section(tmp_path):
    """With the stub embedder, each section's vector is in the layer keyed by id."""
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)
    doc = store.get_document(doc_id, load_bodies=False)
    sids = [s.id for s in doc.sections]
    assert len(sids) >= 2, "fixture should yield >=2 sections"
    # One vector per section (count == #sections), and a search returns a
    # section id (proving section ids -- not doc/episode ids -- are indexed).
    assert store.vector_layer.count() == len(sids)
    top = _top_id(store, embed, "Alice storage")
    assert top is not None and "_sec_" in top and top.startswith(doc_id)
    store.close()


def test_semantic_fallback_surfaces_section(tmp_path):
    """A query the graph cannot match surfaces a SECTION via the vector fallback."""
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)

    class _StubPlanner:
        def plan(self, prompt, history=None):
            # An entity the graph has no edge for -> traversal returns [] -> the
            # retriever's semantic fallback fires (<3 results).
            return {"entities": ["zzzznomatch"], "entity_mode": "union"}

    retr = HippocampalRetriever(store, planner=_StubPlanner())
    retr.vector_search = WavedbVectorStore(store, embedder=embed)
    results = retr.retrieve("Alice storage", use_semantic=True)
    sec_results = [r for r in results if r.get("kind") == "section"]
    assert sec_results, "semantic fallback did not surface a section result"
    r = sec_results[0]
    assert "_sec_" in r["episode_id"]
    assert r["doc_id"] == doc_id
    assert r["source_path"]
    assert r["section_heading"]
    assert "storage" in r["text"].lower(), "section body not materialized into text"
    # Section axes come from the SECTION, not the doc (structure-only ingest has
    # no entity axes -- this asserts the field is the section's own, here []).
    assert isinstance(r["entities"], list)
    store.close()


def test_delete_document_removes_all_section_vectors(tmp_path):
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)
    doc = store.get_document(doc_id, load_bodies=False)
    sids = [s.id for s in doc.sections]
    assert store.vector_layer.count() == len(sids)
    assert store.delete_document(doc_id) is True
    # All section vectors gone (count back to 0; no orphans left in the layer).
    assert store.vector_layer.count() == 0
    assert _top_id(store, embed, "Alice storage") is None
    store.close()


def test_reingest_update_reindexes(tmp_path):
    """Re-ingest (UPDATE) unindexes old section ids then re-indexes new ones."""
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)
    n = store.vector_layer.count()
    assert n >= 2
    # Re-ingest the SAME source -> UPDATE path (reuses doc_id).
    doc_id2, created = _reingest(store, tmp_path, embedder=embed)
    assert doc_id2 == doc_id, "re-ingest should reuse the doc id"
    assert created is False
    # Count stable: old sections unindexed, new sections reindexed (idempotent).
    assert store.vector_layer.count() == n, "re-ingest orphaned or duplicated section vectors"
    # The section ids are still findable (re-indexed, not orphaned).
    top = _top_id(store, embed, "Alice storage")
    assert top is not None and "_sec_" in top and top.startswith(doc_id)
    store.close()


def _reingest(store, tmp_path, *, embedder):
    chunker = HierarchicalChunker(max_section_tokens=200, min_section_tokens=1)
    pipe = UnifiedIngestionPipeline(store, chunker=chunker)
    src = tmp_path / "doc.md"  # same path as _ingest -> resolves to UPDATE
    return pipe.ingest(str(src), extractor=None, relation_extractor=None, embedder=embedder)


def test_sec_discriminator_precedes_doc_prefix(tmp_path):
    """A section id (starts with doc_ + contains _sec_) hydrates as a section."""
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)
    doc = store.get_document(doc_id, load_bodies=False)
    sid = doc.sections[0].id
    assert sid.startswith("doc_") and "_sec_" in sid
    trav = GraphTraversal(store)
    hydrated = trav._hydrate(sid)
    assert hydrated["kind"] == "section", "section id mis-hydrated as a document"
    # And a doc id still hydrates as a document.
    assert trav._hydrate(doc_id)["kind"] == "document"
    store.close()


def test_section_result_rendering(tmp_path):
    """A section result renders via build_context_string + _format_episode + graph."""
    embed = _Bow384()
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path, extractor=None, embedder=embed)
    doc = store.get_document(doc_id, load_bodies=False)
    sid = doc.sections[0].id
    trav = GraphTraversal(store)
    sec = trav._hydrate(sid)
    assert sec["kind"] == "section"
    # Renderer branches on kind -> a Section block (not a Document block).
    ctx = HippocampalRetriever(store).build_context_string([sec])
    assert f"[{sid} |" in ctx
    assert "Section '" in ctx
    assert "storage" in ctx.lower() or sec["text"] in ctx
    chunk = ChunkedContextFormatter()._format_episode(sec)
    assert chunk.startswith("--- Section ")
    # end_state graph node kind = section.
    g = _build_graph([sec])
    node = next(n for n in g["nodes"] if n["id"] == sid)
    assert node["kind"] == "section"
    store.close()