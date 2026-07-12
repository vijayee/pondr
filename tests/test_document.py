"""Unit tests for the document/record ingestion store layer (task #17).

Offline: uses the installed ``wavedb`` package (CPU), no GLiNER/Bonsai. Mirrors
``test_store.py`` (tmp-store fixture). Gates the load-bearing design:

* hot/cold split -- section bodies live in the cold blob store, NOT the hot
  memory store (the LRU-pollution gate);
* content-addressed dedup -- two docs sharing an identical section share one
  blob;
* shared-blob delete safety -- deleting one doc does not physically delete a
  blob another doc references (refcount decrement, GC for orphans);
* upsert by source_path -- re-ingesting a source updates in place (one doc id,
  no duplicate), hash-diffing its sections;
* findability (documents_by_entity / by_topic) + real delete.
"""

from src.memory.document import Document, DocumentSection
from src.memory.store import HippocampalStore


def _doc(
    doc_id: str = "doc_000001",
    source_path: str = "readme.md",
    sections: list[dict] | None = None,
    entities: list[str] | None = None,
    topics: list[str] | None = None,
    citations: list[str] | None = None,
    relations: list[dict] | None = None,
    authors: list[str] | None = None,
    language: str | None = None,
) -> Document:
    """Build a Document with explicit sections (bypasses the chunker)."""
    secs = []
    for i, s in enumerate(sections or [{"heading": "Intro", "content": "Body."}]):
        secs.append(DocumentSection(
            id=f"{doc_id}_sec_{i:03d}",
            heading=s["heading"],
            level=s.get("level", 1),
            content=s["content"],
            parent_section=s.get("parent_section"),
            entities=s.get("entities", []),
            topics=s.get("topics", []),
        ))
    return Document(
        id=doc_id,
        source_type="markdown",
        source_path=source_path,
        title="Test Doc",
        ingested_at="2026-07-11T00:00:00",
        sections=secs,
        entities=entities or [],
        topics=topics or [],
        citations=citations or [],
        relations=relations or [],
        authors=authors or [],
        language=language,
    )


def _bodies_in_hot_store(store: HippocampalStore, needles: list[str]) -> bool:
    """True if any section body leaked into the hot (memory) store."""
    for k, v in store.db.create_read_stream(start="content/doc/", end="content/doc/\x7f"):
        val = v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)
        for n in needles:
            if n in val:
                return True
    return False


def test_encode_document_writes_metadata_and_graph_triples(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    doc = _doc(entities=["Alice"], topics=["Storage"],
               relations=[{"subject": "doc_000001", "predicate": "cites", "object": "rfc1"}],
               citations=["rfc1"],
               sections=[{"heading": "Intro", "content": "Intro body about WaveDB.",
                         "entities": ["WaveDB"], "topics": ["Storage"]},
                         {"heading": "Next", "content": "Next body.", "level": 2,
                          "parent_section": "doc_000001_sec_000"}])
    store.encode_document(doc)

    # Graph triples present (scan the doc node's out-edges).
    has_entity_keys = list(store.db.create_read_stream(
        start="memory/spo/doc_000001/has_entity/", end="memory/spo/doc_000001/has_entity/\x7f"))
    assert has_entity_keys, "doc has_entity edge missing"
    has_section_keys = list(store.db.create_read_stream(
        start="memory/spo/doc_000001/has_section/", end="memory/spo/doc_000001/has_section/\x7f"))
    assert len(has_section_keys) == 2, "has_section edges missing"
    appears_keys = list(store.db.create_read_stream(
        start="memory/spo/E:Alice/appears_in_doc/", end="memory/spo/E:Alice/appears_in_doc/\x7f"))
    assert appears_keys, "appears_in_doc back-pointer missing"
    inst_keys = list(store.db.create_read_stream(
        start="memory/spo/doc_000001/instanceOf/", end="memory/spo/doc_000001/instanceOf/\x7f"))
    assert inst_keys, "instanceOf Document missing"
    cites_keys = list(store.db.create_read_stream(
        start="memory/spo/doc_000001/cites/", end="memory/spo/doc_000001/cites/\x7f"))
    assert cites_keys, "cites edge missing"
    child_keys = list(store.db.create_read_stream(
        start="memory/spo/doc_000001_sec_001/child_of/", end="memory/spo/doc_000001_sec_001/child_of/\x7f"))
    assert child_keys, "child_of edge missing"
    store.close()


def test_encode_document_keeps_hot_lru_clean(tmp_path):
    """Section bodies live in the cold blob store, NOT the hot memory store."""
    store = HippocampalStore(str(tmp_path / "mem"))
    body_a = "Intro body about WaveDB storage layer. " * 20
    body_b = "Next section body about HBTrie. " * 20
    doc = _doc(sections=[{"heading": "Intro", "content": body_a},
                         {"heading": "Next", "content": body_b}])
    store.encode_document(doc)

    # The bodies must NOT be in the memory store.
    assert not _bodies_in_hot_store(store, [body_a, body_b]), "body leaked into hot LRU"
    # But the blob_hash refs + headings ARE in the memory store.
    got = store.get_document(doc.id)
    assert got is not None
    assert got.sections[0].content == body_a, "cold body not roundtripped"
    assert got.sections[1].content == body_b
    store.close()


def test_content_addressed_dedup(tmp_path):
    """Two docs sharing an identical section write ONE blob (dedup hit)."""
    store = HippocampalStore(str(tmp_path / "mem"))
    shared = "Identical shared section body for dedup. " * 20
    d1 = _doc("doc_000001", "a.md",
              sections=[{"heading": "Shared", "content": shared}])
    d2 = _doc("doc_000002", "b.md",
              sections=[{"heading": "Shared", "content": shared}])
    store.encode_document(d1)
    store.encode_document(d2)

    h1 = d1.sections[0].blob_hash
    h2 = d2.sections[0].blob_hash
    assert h1 == h2, "identical bodies hashed to different keys"
    bs = store._blob_store()
    assert bs.refcount(h1) == 2, f"shared blob refcount should be 2, got {bs.refcount(h1)}"
    # Both docs roundtrip the body from the one shared blob.
    assert store.get_document("doc_000001").sections[0].content == shared
    assert store.get_document("doc_000002").sections[0].content == shared
    store.close()


def test_delete_document_decrements_blob_refcount_not_blob(tmp_path):
    """Deleting one of two docs sharing a blob leaves the blob in place."""
    store = HippocampalStore(str(tmp_path / "mem"))
    shared = "Identical shared section body for dedup. " * 20
    store.encode_document(_doc("doc_000001", "a.md",
                               sections=[{"heading": "Shared", "content": shared}]))
    store.encode_document(_doc("doc_000002", "b.md",
                               sections=[{"heading": "Shared", "content": shared}]))
    h = store.get_document("doc_000001").sections[0].blob_hash
    bs = store._blob_store()

    assert store.delete_document("doc_000002") is True
    assert bs.refcount(h) == 1, "refcount should drop to 1 (surviving doc)"
    assert bs.get_blob(h) is not None, "blob physically deleted -- surviving doc corrupted"
    # The surviving doc still roundtrips.
    assert store.get_document("doc_000001").sections[0].content == shared
    store.close()


def test_reingest_same_source_updates_in_place(tmp_path):
    """Re-ingesting a source reuses ONE doc_id (no duplicate), hash-diffs sections."""
    store = HippocampalStore(str(tmp_path / "mem"))
    body_unchanged = "Unchanged section body. " * 20
    body_removed = "Section to be removed on re-ingest. " * 20
    body_added = "New section added on re-ingest. " * 20

    # First ingest.
    doc = _doc("doc_000001", "src.md", sections=[
        {"heading": "Keep", "content": body_unchanged},
        {"heading": "Drop", "content": body_removed},
    ])
    store.encode_document(doc)
    assert store.default_document_ids() == ["doc_000001"]

    h_keep = doc.sections[0].blob_hash
    h_drop = doc.sections[1].blob_hash
    bs = store._blob_store()
    assert bs.refcount(h_keep) == 1
    assert bs.refcount(h_drop) == 1

    # Re-ingest: reuse doc_id (upsert), keep one section, drop one, add one.
    doc2 = _doc("doc_000001", "src.md", sections=[
        {"heading": "Keep", "content": body_unchanged},   # unchanged -> reuse blob
        {"heading": "Added", "content": body_added},      # new
    ])
    store.encode_document(doc2, update=True)

    # Still exactly one doc.
    assert store.default_document_ids() == ["doc_000001"], "upsert created a duplicate"
    got = store.get_document("doc_000001")
    assert len(got.sections) == 2
    headings = [s.heading for s in got.sections]
    assert headings == ["Keep", "Added"], f"sections after upsert wrong: {headings}"

    # Unchanged section's blob reused (refcount unchanged at 1).
    assert doc2.sections[0].blob_hash == h_keep, "unchanged section got a new hash"
    assert bs.refcount(h_keep) == 1, "unchanged blob refcount changed"
    # Removed section's blob refcount decremented to 0 (orphaned, not deleted).
    assert bs.refcount(h_drop) == 0, "removed blob refcount not decremented"
    assert bs.get_blob(h_drop) is not None, "orphan blob physically deleted before gc"
    # New section got a new blob.
    assert doc2.sections[1].blob_hash != h_keep
    assert bs.refcount(doc2.sections[1].blob_hash) == 1
    store.close()


def test_get_document_roundtrips(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    doc = _doc(entities=["Alice", "Bob"], topics=["Storage", "Perf"],
               authors=["Victor"], language="en",
               sections=[{"heading": "A", "content": "AAA " * 20},
                         {"heading": "B", "content": "BBB " * 20, "level": 2}])
    store.encode_document(doc)
    got = store.get_document("doc_000001")
    assert got is not None
    assert got.title == "Test Doc"
    assert got.source_path == "readme.md"
    assert got.authors == ["Victor"]
    assert got.language == "en"
    assert got.entities == ["Alice", "Bob"]
    assert got.topics == ["Storage", "Perf"]
    assert len(got.sections) == 2
    assert got.sections[0].content.startswith("AAA")
    assert got.sections[1].level == 2
    store.close()


def test_get_document_missing_returns_none(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    assert store.get_document("doc_nope") is None
    store.close()


def test_default_document_ids_scans_content_doc(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    store.encode_document(_doc("doc_000001", "a.md"))
    store.encode_document(_doc("doc_000002", "b.md"))
    assert store.default_document_ids() == ["doc_000001", "doc_000002"]
    store.close()


def test_delete_document_removes_edges_and_content(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    doc = _doc(entities=["Alice"], sections=[{"heading": "A", "content": "Body A " * 20}])
    store.encode_document(doc)
    assert store.default_document_ids() == ["doc_000001"]

    assert store.delete_document("doc_000001") is True
    assert store.default_document_ids() == []
    assert store.get_document("doc_000001") is None
    assert store.document_id_by_source("readme.md") is None
    # Graph edges gone.
    keys = list(store.db.create_read_stream(
        start="memory/spo/doc_000001/", end="memory/spo/doc_000001/\x7f"))
    assert not keys, "graph edges not removed on delete"
    # Idempotent delete.
    assert store.delete_document("doc_000001") is False
    store.close()


def test_documents_by_entity_and_topic(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    store.encode_document(_doc("doc_000001", "a.md", entities=["Alice"],
                               topics=["Storage"]))
    store.encode_document(_doc("doc_000002", "b.md", entities=["Bob"],
                               topics=["Perf"]))
    assert store.documents_by_entity("Alice") == ["doc_000001"]
    assert store.documents_by_entity("Bob") == ["doc_000002"]
    assert store.documents_by_topic("Storage") == ["doc_000001"]
    # Episode entities don't pollute doc findability.
    from src.memory.episode import Episode
    store.encode_episode(Episode(id="ep_000001", timestamp="2026-07-11",
                                  summary="s", full_text="t", entities=["Alice"]))
    assert store.documents_by_entity("Alice") == ["doc_000001"], "episode polluted doc query"
    store.close()


def test_document_predicates_in_hash_tail_not_known():
    """Document predicates stay out of KNOWN_PREDICATES + _NODE_PREDICATES."""
    from src.gnn.graph_loader import KNOWN_PREDICATES
    from src.training.oracle_labeling import _NODE_PREDICATES
    new = {"has_section", "child_of", "cites", "appears_in_doc"}
    assert not (new & set(KNOWN_PREDICATES)), "doc predicate leaked into KNOWN_PREDICATES"
    assert not (new & set(_NODE_PREDICATES)), "doc predicate leaked into _NODE_PREDICATES"
    # And the ontology declares them + the DocumentSection class.
    from src.memory.ontology import SEED_ONTOLOGY
    for p in new:
        assert p in SEED_ONTOLOGY["properties"], f"ontology missing predicate {p}"
    assert "DocumentSection" in SEED_ONTOLOGY["classes"]