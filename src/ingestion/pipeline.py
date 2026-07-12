"""UnifiedIngestionPipeline -- parser -> chunker -> extraction -> store.

Composes the three ingestion stages into one ``ingest(source_path)`` call:

1. ``detect_type`` -> parser -> ``ParsedDocument`` (native structure).
2. ``HierarchicalChunker`` -> leaf-sized, hierarchy-wired sections.
3. Per-section GLiNER extract (entities/topics) + doc-level Bonsai relations ->
   ``Document.from_parse`` -> ``store.encode_document`` (upsert by source_path).

Extraction is INJECTABLE and OPTIONAL, not constructed here. The heavy GPU
deps (GLiNER, Bonsai) are the CALLER's responsibility: the CLI constructs them
when available and passes them in; when absent (CPU dev box, offline tests) the
pipeline runs structure-only (empty entities/topics/relations) and still
produces a valid, retrievable Document. This keeps the pipeline pure +
offline-testable while the CLI owns the lazy-heavy-deps policy. A document
ingested structure-only can be re-ingested later with extractors to fill in
its entities/topics in place (the upsert hash-diff reuses unchanged blobs).

Upsert (the user's directive): identity is resolved by ``source_path`` BEFORE
encoding -- a source already ingested is an in-place UPDATE (reuses its id,
hash-diffs its sections), never a duplicate. Returns ``(doc_id, created)`` so
the CLI can print ``created doc_NNNNNN`` vs ``updated doc_NNNNNN``.
"""

from typing import Optional

from ..memory.document import Document
from .chunker import HierarchicalChunker
from .parsers import detect_type, get_parser


# Cap on the text handed to Bonsai for doc-level relation extraction. A large
# doc would otherwise blow the prompt; relations over the first ~8k chars
# capture the document's main assertions (Bonsai's response is also max_tokens
# bounded). A Phase-2 refinement would run Bonsai per-section.
_BONSAI_TEXT_CAP = 8000


class UnifiedIngestionPipeline:
    """End-to-end ingestion: source -> Document projection in the store."""

    def __init__(self, store, *, chunker: Optional[HierarchicalChunker] = None):
        self.store = store
        self.chunker = chunker or HierarchicalChunker()

    def ingest(
        self,
        source_path: str,
        *,
        source_type: str = "auto",
        extractor=None,
        relation_extractor=None,
    ) -> tuple[str, bool]:
        """Ingest (or re-ingest) a source. Returns ``(doc_id, created)``.

        ``source_type="auto"`` infers from the path extension (``text``
        fallback); an explicit type selects that parser. ``extractor`` (a
        ``GLiNERExtractor``) and ``relation_extractor`` (a
        ``BonsaiRelationExtractor``) are optional -- when ``None`` the
        pipeline runs structure-only (no entities/topics/relations). Re-ingest
        resolves identity by ``source_path`` and updates in place.
        """
        if source_type == "auto":
            source_type = detect_type(source_path)
        parser = get_parser(source_type)
        parsed = parser.parse(source_path)
        parsed = self.chunker.chunk(parsed)

        # Per-section extraction (entities/topics) -- the retrieval axes that
        # make sections + the doc findable by entity/topic.
        sec_extractions: list[dict] = []
        if extractor is not None:
            for sec in parsed.sections:
                ext = extractor.extract(sec.content)
                sec_extractions.append({
                    "entities": list(ext.get("entities", [])),
                    "topics": list(ext.get("topics", [])),
                })

        # Doc-level relations (Bonsai) over a capped concatenation of the doc.
        relations: list[dict] = []
        if relation_extractor is not None:
            relations = list(relation_extractor.extract(self._doc_text(parsed)))

        extracted = {"sections": sec_extractions}

        # Upsert by source_path: reuse the existing doc_id on re-ingest.
        existing = self.store.document_id_by_source(source_path)
        if existing is None:
            doc_id = self.store.next_document_id()
            created = True
        else:
            doc_id = existing
            created = False

        doc = Document.from_parse(doc_id, parsed, extracted, relations)
        self.store.encode_document(doc, update=not created)
        return doc_id, created

    @staticmethod
    def _doc_text(parsed) -> str:
        """Concatenate section headings + bodies for doc-level extraction.

        Capped at ``_BONSAI_TEXT_CAP`` chars so a large doc does not blow the
        relation-extraction prompt; the first sections carry the document's
        main assertions (and Bonsai's response is max_tokens bounded anyway).
        """
        out: list[str] = []
        for sec in parsed.sections:
            chunk = (sec.heading + "\n" + sec.content) if sec.heading else sec.content
            out.append(chunk.strip())
        text = "\n\n".join(out)
        return text[:_BONSAI_TEXT_CAP]