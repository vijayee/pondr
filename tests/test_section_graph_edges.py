"""Tests for the per-section graph edges: entity->section reverse + per-section
has_topic (Phase 2c+ follow-on to the doc-ingestion RAG pillar).

Before this: ``_document_graph_ops`` emitted ``has_topic`` at DOC level only and
``has_entity`` per-section with no reverse, so the graph entity axis found docs,
not the relevant section chunk. Now each section emits ``(sec, has_topic, T:t)``
AND the reverse ``(E:x, appears_in_section, sec)`` so:

* the entity axis (``_get_episodes_by_entity`` third union) lands on a SECTION
  (hydrated ``kind="section"`` -- the chunk), not just the whole doc;
* the topic axis picks up per-section topics with NO code change.

``appears_in_section`` is a hash-tail predicate (GNN-invisible; kept OUT of
KNOWN_PREDICATES/_NODE_PREDICATES). Delete is symmetric (no orphan).

Offline: installed wavedb (CPU) + a keyword-extractor stub. No GLiNER/Bonsai.
"""

from __future__ import annotations

import pytest

wavedb = pytest.importorskip("wavedb")

from src.gnn.graph_loader import KNOWN_PREDICATES
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.store import HippocampalStore
from src.retrieval.graph_traversal import GraphTraversal


class _KWExtractor:
    """Deterministic entity/topic extractor stub (GLiNER stand-in, offline)."""

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


_MD = (
    "# Project Notes\n\n"
    "## Alice on Storage\n\n"
    "Alice architected the storage subsystem. Storage uses a content-addressed "
    "cold blob store. Alice chose this split.\n\n"
    "## Bob on Networking\n\n"
    "Bob implemented the networking transport. Networking nodes gossip.\n"
)


def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": False}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


def _ingest(store, tmp_path, *, source="doc.md"):
    src = tmp_path / source
    src.write_text(_MD, encoding="utf-8")
    chunker = HierarchicalChunker(max_section_tokens=200, min_section_tokens=1)
    pipe = UnifiedIngestionPipeline(store, chunker=chunker)
    doc_id, _ = pipe.ingest(str(src), extractor=_KWExtractor(), relation_extractor=None)
    return doc_id


def _section_ids(results):
    return [r["episode_id"] for r in results if r.get("kind") == "section"]


def _section_by_text(results, needle):
    """Find the section result whose text contains ``needle`` (case-insensitive)."""
    nl = needle.lower()
    for r in results:
        if r.get("kind") == "section" and nl in r.get("text", "").lower():
            return r
    return None


# ── entity -> section reverse edge ──

def test_entity_axis_lands_on_section(tmp_path):
    store = _store(tmp_path)
    _ingest(store, tmp_path)
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Alice"], "entity_mode": "union"})
    secs = _section_ids(results)
    assert secs, f"entity axis found no section chunk: {secs}"
    # The Alice/Storage section surfaces as a kind="section" chunk (not the doc).
    sec = _section_by_text(results, "storage subsystem")
    assert sec is not None, "the Alice/Storage section did not surface on the entity axis"
    assert sec["kind"] == "section"
    assert "Alice" in sec["entities"]
    assert "storage" in sec["text"].lower()
    store.close()


def test_entity_axis_excludes_nonmatching_section(tmp_path):
    store = _store(tmp_path)
    _ingest(store, tmp_path)
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Alice"], "entity_mode": "union"})
    # The Bob/Networking section must NOT surface on the Alice entity axis.
    assert _section_by_text(results, "networking transport") is None, (
        "non-matching section leaked via the entity axis"
    )
    store.close()


# ── per-section has_topic ──

def test_topic_axis_lands_on_section(tmp_path):
    store = _store(tmp_path)
    _ingest(store, tmp_path)
    trav = GraphTraversal(store)
    results = trav.retrieve({"topics": ["Storage"]})
    # The Storage section is on the topic axis (per-section has_topic works).
    assert _section_by_text(results, "storage subsystem") is not None, (
        "per-section has_topic did not put the section on the topic axis"
    )
    # The Networking section is NOT on the Storage topic axis.
    assert _section_by_text(results, "networking transport") is None
    store.close()


# ── delete symmetry (no orphan reverse edges) ──

def test_delete_removes_section_reverse_edges(tmp_path):
    store = _store(tmp_path)
    doc_id = _ingest(store, tmp_path)
    trav = GraphTraversal(store)
    assert _section_ids(trav.retrieve({"entities": ["Alice"], "entity_mode": "union"}))
    assert _section_ids(trav.retrieve({"topics": ["Storage"]}))
    # Delete the doc -> _document_graph_ops(delete=True) removes the per-section
    # has_topic + appears_in_section reverse edges symmetrically.
    assert store.delete_document(doc_id)
    after = trav.retrieve({"entities": ["Alice"], "entity_mode": "union"})
    assert after == [], "section reverse edges survived delete (orphan)"
    after_t = trav.retrieve({"topics": ["Storage"]})
    assert after_t == [], "per-section has_topic survived delete (orphan)"
    store.close()


# ── hash-tail predicate: GNN-invisible ──

def test_appears_in_section_is_gnn_invisible():
    # The new predicate must NOT enter the trained-GNN predicate vocab (keeps the
    # checkpoint loadable + invisible until retrain -- the A3 template).
    assert "appears_in_section" not in KNOWN_PREDICATES
    assert "has_section" not in KNOWN_PREDICATES
    assert "appears_in_doc" not in KNOWN_PREDICATES