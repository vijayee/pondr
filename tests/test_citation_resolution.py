"""Phase 3c: citation resolution + email provenance + cited_from (D5).

Offline (no GLiNER/Bonsai/server): constructs Documents + Episodes directly and
encodes them, then asserts the Phase 3c graph edges. Three citation pieces:

1. **doc->doc ``cites`` resolution**: ``Document.citations`` literals are
   resolved to Document node ids via ``find_document_by_title_or_url`` (title /
   URL match); unresolved literals are kept verbatim. ``resolved_citations`` is
   persisted so an update/delete emits symmetric deletes.
2. **email provenance**: ``in_reply_to`` / ``references`` edges from the
   parser's ``metadata`` maps (Message-ID -> section id).
3. **cited_from**: a best-effort ``(eid, cited_from, doc_id)`` edge when an
   episode's text references a known doc title; no match -> no edge.

Gated off -> byte-identical (zero citation/provenance edges).
"""

from __future__ import annotations

from src.config import config as master_config
from src.memory.document import Document, DocumentSection
from src.memory.episode import Episode
from src.memory.store import HippocampalStore


# ── helpers ───────────────────────────────────────────────────────────────

def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _doc(doc_id, title, source_path="x.md", citations=None, sections=None,
         metadata=None, state_assertions=None):
    secs = []
    for i, s in enumerate(sections or [{"heading": "Intro", "content": "Body."}]):
        secs.append(DocumentSection(
            id=f"{doc_id}_sec_{i:03d}", heading=s["heading"],
            level=s.get("level", 1), content=s["content"],
            parent_section=s.get("parent_section"),
        ))
    return Document(
        id=doc_id, source_type="markdown", source_path=source_path,
        title=title, ingested_at="2026-07-15T00:00:00",
        sections=secs, citations=citations or [],
        metadata=metadata or {}, state_assertions=state_assertions or [],
    )


def _cites_targets(store, doc_id):
    """The object ids of the doc's ``cites`` out-edges."""
    r = store.graph.query().vertex(doc_id).out("cites").execute_sync()
    try:
        return sorted(r.vertices)
    finally:
        r.close()


def _edges(store, subj, pred):
    r = store.graph.query().vertex(subj).out(pred).execute_sync()
    try:
        return sorted(r.vertices)
    finally:
        r.close()


# ── find_document_by_title_or_url ──

def test_find_by_title_exact(tmp_path):
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Policy A"))
    assert store.find_document_by_title_or_url("Policy A") == "doc_000001"
    store.close()


def test_find_by_title_substring(tmp_path):
    """``"Policy A"`` resolves a doc titled ``"Policy A: Remote Work"``."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Policy A: Remote Work"))
    assert store.find_document_by_title_or_url("Policy A") == "doc_000001"
    store.close()


def test_find_by_url_in_source_path(tmp_path):
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Spec",
                                source_path="https://repo/spec.md"))
    assert store.find_document_by_title_or_url("https://repo/spec.md") == \
        "doc_000001"
    store.close()


def test_find_returns_none_when_unresolved(tmp_path):
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Policy A"))
    assert store.find_document_by_title_or_url("Nonexistent Doc") is None
    store.close()


# ── doc->doc cites resolution ──

def test_cites_resolved_to_document_node(tmp_path):
    """A citation literal matching another doc's title -> ``(doc, cites,
    target_doc_id)`` edge to the Document node (not the literal)."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Policy A: Remote Work"))
    store.encode_document(_doc("doc_000002", "Meeting Notes",
                                citations=["Policy A: Remote Work"]))
    assert _cites_targets(store, "doc_000002") == ["doc_000001"]
    store.close()


def test_cites_unresolved_keeps_literal(tmp_path):
    """A citation literal that matches no doc -> the literal itself is the
    cite object (byte-identical to pre-Phase-3c)."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Meeting Notes",
                                citations=["some-unresolved-literal"]))
    assert _cites_targets(store, "doc_000001") == ["some-unresolved-literal"]
    store.close()


def test_resolved_citations_persisted(tmp_path):
    """``resolved_citations`` is read back so an update/delete emits symmetric
    deletes for the resolved edges (resolution is store-state-dependent)."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Policy A"))
    store.encode_document(_doc("doc_000002", "Notes",
                                citations=["Policy A"]))
    got = store.get_document("doc_000002")
    assert got.resolved_citations == ["doc_000001"]
    store.close()


def test_citation_resolution_disabled_keeps_literal(tmp_path):
    """``--no-citation-resolution`` -> literals kept as-is (byte-identical)."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Policy A"))
    saved = master_config.citation_resolution_enabled
    master_config.citation_resolution_enabled = False
    try:
        store.encode_document(_doc("doc_000002", "Notes",
                                    citations=["Policy A"]))
        # No resolution -> the literal is the cite object.
        assert _cites_targets(store, "doc_000002") == ["Policy A"]
    finally:
        master_config.citation_resolution_enabled = saved
    store.close()


# ── email in_reply_to / references provenance ──

def test_email_in_reply_to_edges(tmp_path):
    """A 2-message email thread (root + reply) -> ``in_reply_to`` edge from
    the reply's section to the root's section."""
    store = _store(tmp_path)
    # The parser emits one section per message in emission order; section ids
    # are ``{doc_id}_sec_{i:03d}``. Two messages: sec_000 (root), sec_001 (reply).
    # The parser stores message_ids + in_reply_to keys/values ANGLE-STRIPPED
    # (``_strip_angle``); the ``references`` map value is the RAW header string
    # (brackets intact), which the store strips at lookup time.
    meta = {
        "message_ids": ["root@example.com", "reply@example.com"],
        "in_reply_to": {"reply@example.com": "root@example.com"},
        "references": {"reply@example.com": "<root@example.com>"},
    }
    doc = _doc("doc_000001", "Thread", source_path="t.mbox",
               sections=[{"heading": "root", "content": "root body"},
                         {"heading": "reply", "content": "reply body"}],
               metadata=meta)
    store.encode_document(doc)

    irt = _edges(store, "doc_000001_sec_001", "in_reply_to")
    assert irt == ["doc_000001_sec_000"], f"got {irt}"
    refs = _edges(store, "doc_000001_sec_001", "references")
    assert refs == ["doc_000001_sec_000"], f"got {refs}"
    store.close()


def test_email_citation_off_writes_no_in_reply_to(tmp_path):
    """``--no-citation-resolution`` -> zero ``in_reply_to`` edges (the de-wonk
    cold-start gate)."""
    store = _store(tmp_path)
    meta = {
        "message_ids": ["root@example.com", "reply@example.com"],
        "in_reply_to": {"reply@example.com": "root@example.com"},
        "references": {},
    }
    saved = master_config.citation_resolution_enabled
    master_config.citation_resolution_enabled = False
    try:
        store.encode_document(_doc("doc_000001", "Thread",
                                    source_path="t.mbox",
                                    sections=[{"heading": "root", "content": "r"},
                                              {"heading": "reply", "content": "p"}],
                                    metadata=meta))
        assert _edges(store, "doc_000001_sec_001", "in_reply_to") == []
    finally:
        master_config.citation_resolution_enabled = saved
    store.close()


def test_email_no_metadata_no_edges(tmp_path):
    """A doc with no email metadata -> no in_reply_to/references edges."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Plain",
                                sections=[{"heading": "a", "content": "b"}]))
    assert _edges(store, "doc_000001_sec_000", "in_reply_to") == []
    store.close()


# ── cited_from (episode -> doc) ──

def test_cited_from_on_title_match(tmp_path):
    """An episode whose text references a known doc title -> ``(eid,
    cited_from, doc_id)``."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Remote Work Policy"))
    store.encode_episode(Episode(
        id="ep_001", timestamp="2026-07-15T10:00:00Z",
        summary="policy mention", full_text="Per the Remote Work Policy, ...",
        entities=[], topics=[],
    ))
    assert _edges(store, "ep_001", "cited_from") == ["doc_000001"]
    store.close()


def test_cited_from_none_on_no_match(tmp_path):
    """An episode that references no known doc title -> no ``cited_from``."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Remote Work Policy"))
    store.encode_episode(Episode(
        id="ep_001", timestamp="2026-07-15T10:00:00Z",
        summary="x", full_text="A totally unrelated message about cooking.",
        entities=[], topics=[],
    ))
    assert _edges(store, "ep_001", "cited_from") == []
    store.close()


def test_cited_from_disabled_no_edge(tmp_path):
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Remote Work Policy"))
    saved = master_config.citation_resolution_enabled
    master_config.citation_resolution_enabled = False
    try:
        store.encode_episode(Episode(
            id="ep_001", timestamp="2026-07-15T10:00:00Z",
            summary="x", full_text="Per the Remote Work Policy, ...",
            entities=[], topics=[],
        ))
        assert _edges(store, "ep_001", "cited_from") == []
    finally:
        master_config.citation_resolution_enabled = saved
    store.close()


# ── assertion edges from a document (D1 doc path) ──

def test_document_state_assertions_write_entity_edges(tmp_path):
    """A doc with ``state_assertions`` -> ``(E:entity, state, value)`` edges
    with sidecar ``asserted_by`` = the asserting section (the D1 doc path)."""
    store = _store(tmp_path)
    doc = _doc("doc_000001", "Config",
               sections=[{"heading": "Settings", "content": "Status: open"}],
               state_assertions=[{"entity": "status", "value": "open",
                                  "section": "doc_000001_sec_000"}])
    store.encode_document(doc)
    # The entity state edge exists.
    assert _edges(store, "E:status", "state") == ["open"]
    # Sidecar provenance: asserted_by = the section.
    meta = store.get_edge_meta("E:status", "state", "open")
    assert meta.get("asserted_by") == "doc_000001_sec_000"
    store.close()


def test_assertion_extraction_disabled_no_entity_edges(tmp_path):
    """``--no-assertions`` -> zero entity ``state`` edges from a doc (byte-
    identical to pre-Phase-3c)."""
    store = _store(tmp_path)
    saved = master_config.assertion_extraction_enabled
    master_config.assertion_extraction_enabled = False
    try:
        store.encode_document(_doc("doc_000001", "Config",
            sections=[{"heading": "Settings", "content": "Status: open"}],
            state_assertions=[{"entity": "status", "value": "open",
                               "section": "doc_000001_sec_000"}]))
        assert _edges(store, "E:status", "state") == []
    finally:
        master_config.assertion_extraction_enabled = saved
    store.close()


# ── cold-start: no citations, no metadata -> byte-identical ──

def test_plain_doc_has_no_phase4_edges(tmp_path):
    """A plain doc (no citations, no email metadata, no state assertions) ->
    zero cites/in_reply_to/state edges (the cold-start no-op)."""
    store = _store(tmp_path)
    store.encode_document(_doc("doc_000001", "Plain",
                                sections=[{"heading": "a", "content": "b"}]))
    assert _cites_targets(store, "doc_000001") == []
    assert _edges(store, "doc_000001_sec_000", "in_reply_to") == []
    store.close()