"""Document data model -- a peer top-level unit alongside ``Episode``.

Where ``Episode`` is the atomic unit of *conversational* memory (one
user+assistant turn), ``Document`` is the atomic unit of *ingested*
memory -- a projection of an external source (a markdown file, a PDF, a
web page, a transcript) into the hippocampal index. The two are peers, not
subclasses: a Document is NOT a kind of Episode (it has no forgetting
fields, no user/session scope, no chat-turn structure); cross-reference
edges in the graph link them when they overlap.

Load-bearing design (see the ingestion plan, ``mellow-jumping-token.md``):

* **Documents are not forgotten.** They carry no ``state`` / ``validity`` /
  ``salience`` / ``decay`` fields and never enter the 3b decay/archive sweep.
  Removal is an explicit ``delete_document`` (real removal, not
  decay-to-archive). Ownership of a graph node by the document subsystem is
  signaled by the ``doc_`` id prefix -- the same convention ``ep_`` uses for
  episodes -- so the forgetting sweeps skip doc-owned edges by prefix.
* **Hot/cold split.** A ``DocumentSection`` (the leaf chunk, the retrieval
  unit) holds a ``blob_hash`` reference, NOT its body. The bulky section body
  lives in a separate content-addressed cold store keyed by that hash; the
  memory (hot) store keeps only the small render-time metadata + graph
  pointers. This keeps large chunk bodies from flushing the memory store's
  100MB LRU. ``content`` on a ``DocumentSection`` is therefore an in-memory
  convenience populated only by ``store.get_document`` (which pulls each body
  from the cold store); it is empty on a freshly-constructed section before
  the store resolves its blob.
* **Structure-based chunking.** Sections are derived from the source's own
  structure (markdown headings, later PDF/DOCX/web native structure), not
  arbitrary token windows. Each section carries an ``embedding`` (the
  semantic-retrieval unit) and a ``blob_hash`` (the content-addressed cold-
  store key, dedup + incremental-re-ingest enabler).

``DocumentSection`` and ``Document`` are dataclasses, peers to ``Episode``
(see ``episode.py``). They intentionally share no superclass: the forgetting
exemption is expressed by *absence* of the forgetting fields, not by a flag.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DocumentSection:
    """One leaf chunk of a document -- the retrieval unit.

    A section is a node in the document's hierarchy. Its ``level`` records the
    structural depth (markdown ``#`` -> 1, ``##`` -> 2, ...; structure-less
    inputs flatten to 1) and ``parent_section`` records the parent section id
    (``None`` at the root), so the graph can carry ``(sec, child_of, parent)``
    edges that preserve the tree the parser saw. The body (``content``) is the
    in-memory text pulled from the cold store by ``get_document``; on a section
    constructed for encoding it is present (the encoder hashes it into a blob
    and stores only the ``blob_hash``), on a section loaded by ``get_document``
    it is rehydrated from the cold store.

    ``embedding`` is the per-section dense vector for semantic search (the
    chunk IS the semantic-retrieval unit); ``None`` until the embedding
    backfill populates it. ``blob_hash`` is the content-addressed cold-store
    key (``sha256[:16]`` of the body); the store fills it at encode time.
    """

    id: str
    heading: str
    level: int
    content: str = ""
    parent_section: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    blob_hash: Optional[str] = None


@dataclass
class Document:
    """A projection of an external source into the hippocampal index.

    The source of truth stays external (git / filesystem / Drive); the
    Document is Pondr's *derived* semantic projection -- a materialized view
    (coding-chat IDX 8). Identity is keyed by ``source_path`` (the
    ``doc_by_source`` index), so re-ingesting the same source UPDATES the
    document in place (reuses its id, hash-diffs its sections) rather than
    creating a duplicate -- explicit re-ingest, no file watcher.

    Fields mirror the design chat (IDX 88) with two deliberate omissions: no
    forgetting fields (documents are exempt) and section bodies are not
    stored on the document (they live in the cold store behind
    ``DocumentSection.blob_hash``). ``entities`` / ``topics`` are the
    doc-level union of their sections' extractions (the graph carries both
    per-section ``has_entity`` and doc-level ``has_entity`` pointers).
    ``relations`` are Bonsai-extracted triples over the doc; ``citations``
    are the doc-level citation targets (``(doc, cites, target)`` edges).
    """

    id: str
    source_type: str
    source_path: str
    title: str
    ingested_at: str
    sections: list[DocumentSection] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    created_at: Optional[str] = None
    language: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    @classmethod
    def from_parse(
        cls,
        doc_id: str,
        parsed,
        extracted: dict,
        relations: list[dict],
        ingested_at: Optional[str] = None,
    ) -> "Document":
        """Build a ``Document`` from a parser + chunker + extraction pass.

        Mirrors ``Episode.from_extraction``: the caller runs the parser and
        chunker (``parsed`` is the chunked ``ParsedDocument`` -- duck-typed to
        avoid a ``memory`` -> ``ingestion`` import cycle), the per-section +
        doc-level GLiNER extractions (``extracted``), and the Bonsai relation
        pass (``relations``); this assembles the unit model.

        ``parsed.sections`` are the chunker's normalized leaf sections, each
        with ``heading`` / ``level`` / ``content`` / ``parent_index`` (the
        index of the parent section in the SAME list, or ``None`` at the root
        -- the chunker emits an index, not an id, because it does not know
        ``doc_id``). ``extracted`` carries ``sections`` -- a list aligned 1:1
        with ``parsed.sections``, each ``{"entities": [...], "topics": [...]}``
        -- plus optional doc-level ``entities`` / ``topics`` / ``citations``
        (when absent the doc-level sets are the union of the section sets,
        which is the encoding-time default). Section ids are compound
        (``{doc_id}_sec_{i:03d}``), so there is one id space across the whole
        document and no separate section counter; ``parent_index`` is mapped
        to the parent's compound id here.

        ``blob_hash`` is left ``None`` here: the store fills it at encode time
        (hashing the body into the cold store). ``embedding`` is copied from
        ``raw.embedding`` when the pipeline pre-embedded the section (the
        per-chunk vector-index path); otherwise it is ``None`` and a later
        backfill fills it.
        """
        if ingested_at is None:
            ingested_at = datetime.now().isoformat()

        sec_extractions = extracted.get("sections") or []
        sections: list[DocumentSection] = []
        doc_entities: set[str] = set()
        doc_topics: set[str] = set()
        for i, raw in enumerate(getattr(parsed, "sections", [])):
            sid = f"{doc_id}_sec_{i:03d}"
            pidx = getattr(raw, "parent_index", None)
            parent = f"{doc_id}_sec_{pidx:03d}" if pidx is not None else None
            sec_ext = sec_extractions[i] if i < len(sec_extractions) else {}
            sec_ents = sec_ext.get("entities", [])
            sec_tops = sec_ext.get("topics", [])
            doc_entities.update(sec_ents)
            doc_topics.update(sec_tops)
            sections.append(DocumentSection(
                id=sid,
                heading=getattr(raw, "heading", "") or "",
                level=int(getattr(raw, "level", 1) or 1),
                content=getattr(raw, "content", "") or "",
                parent_section=parent,
                entities=list(sec_ents),
                topics=list(sec_tops),
                embedding=getattr(raw, "embedding", None),
            ))

        # Doc-level entities/topics default to the section union; an explicit
        # set in ``extracted`` (e.g. a doc-level GLiNER pass) overrides.
        entities = extracted.get("entities")
        entities = list(doc_entities) if entities is None else list(entities)
        topics = extracted.get("topics")
        topics = list(doc_topics) if topics is None else list(topics)
        citations = list(extracted.get("citations", []))

        return cls(
            id=doc_id,
            source_type=getattr(parsed, "source_type", "text"),
            source_path=getattr(parsed, "source_path", ""),
            title=getattr(parsed, "title", "") or "",
            ingested_at=ingested_at,
            sections=sections,
            authors=list(getattr(parsed, "authors", []) or []),
            created_at=getattr(parsed, "created_at", None),
            language=getattr(parsed, "language", None),
            metadata=dict(getattr(parsed, "metadata", {}) or {}),
            entities=entities,
            topics=topics,
            relations=relations,
            citations=citations,
        )