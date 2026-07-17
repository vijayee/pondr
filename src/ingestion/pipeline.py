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

from ..encoding.assertion_extractor import extract_state_assertions
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
        embedder=None,
        doc_kind_tagger=None,
    ) -> tuple[str, bool]:
        """Ingest (or re-ingest) a source. Returns ``(doc_id, created)``.

        ``source_type="auto"`` infers from the path extension (``text``
        fallback); an explicit type selects that parser. ``extractor`` (a
        ``GLiNERExtractor``) and ``relation_extractor`` (a
        ``BonsaiRelationExtractor``) are optional -- when ``None`` the
        pipeline runs structure-only (no entities/topics/relations). Re-ingest
        resolves identity by ``source_path`` and updates in place.

        ``embedder`` (any object with ``encode(texts: list[str]) ->
        list[list[float]]`` -- the same protocol ``VectorSearch`` /
        ``WavedbVectorStore`` satisfy) is optional -- when set, each section's
        text (``heading + "\\n" + content``, the chunk content per the chat
        design) is embedded in ONE batched ``encode`` call and assigned to
        ``sec.embedding`` before ``Document.from_parse``, so the per-chunk
        vector rides through ``encode_document`` into BOTH the hot
        ``sec/embedding`` key AND the in-DB vector layer (the per-chunk doc-RAG
        semantic path). When ``None`` (structure-only ingest, no embedder), no
        embeddings are written -- sections stay findable via the graph
        entity/topic axes but not via the semantic fallback (mirrors episodes'
        ``set_summary_embedding`` backfill model).

        ``doc_kind_tagger`` (Phase 3c Sec 7.11; any object with
        ``classify_doc_kind(text) -> Optional[str]`` -- a ``BonsaiDecider``
        satisfies this) is optional -- when set, the doc's semantic KIND is
        tagged at ingest (one zero-shot HTTP call over ``_doc_text``) and
        written to ``doc.doc_kind``. When ``None`` (structure-only / cold-
        start) OR the call fails / returns an out-of-vocab label, ``doc_kind``
        stays the ``"other"`` default -> byte-identical to pre-7.11 (NO
        fabricated label). The tag lets the complementary-temporal guard fire
        on a semantic signal (both sources ``point_in_time_snapshot``) instead
        of a filename month-prefix, which is inert on real enterprise docs.
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

        # Per-section embeddings (the per-chunk vector-index path): ONE batched
        # encode over every section's chunk text. Assigned to the RawSection so
        # ``Document.from_parse`` carries it through to ``encode_document``,
        # which persists the hot key AND indexes the vector layer.
        if embedder is not None and parsed.sections:
            sec_texts = [
                (s.heading + "\n" + s.content) if s.heading else s.content
                for s in parsed.sections
            ]
            vecs = embedder.encode(sec_texts)
            for s, vec in zip(parsed.sections, vecs):
                s.embedding = list(vec)

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

        # Phase 3c (D1): per-section state assertions -- the deterministic
        # normalizer over each section's content (catches explicit ``key:
        # value`` / ``key is value`` / change-verb patterns -- Jira/Linear/
        # Confluence status fields, config snippets, spec tables), plus a
        # doc-level pass that also lifts Bonsai ``has_state`` relations. Each
        # assertion is tagged with its asserting section id (``asserted_by``);
        # the store writes ``(E:entity, state, value)`` edges with that
        # provenance. Empty for structure-only ingests (no extractor) and
        # docs with no explicit state claims -- the cold-start no-op (D6).
        # Section ids mirror ``Document.from_parse``'s ``{doc_id}_sec_{i:03d}``.
        state_assertions: list[dict] = []
        for i, sec in enumerate(parsed.sections):
            sid = f"{doc_id}_sec_{i:03d}"
            for a in extract_state_assertions(sec.content, None, None):
                state_assertions.append({"entity": a["entity"],
                                         "value": a["value"], "section": sid})
        for a in extract_state_assertions(self._doc_text(parsed), None, relations):
            # Doc-level assertions default ``asserted_by`` to the doc id
            # (set by the store when ``section`` is absent).
            state_assertions.append({"entity": a["entity"], "value": a["value"]})
        doc.state_assertions = state_assertions

        # Phase 3c Sec 7.11: semantic doc-kind tag (zero-shot Bonsai at ingest).
        # Injected (a BonsaiDecider or any duck-typed classify_doc_kind); when
        # None (structure-only / cold-start) doc_kind stays the "other" default
        # -> byte-identical to pre-7.11. Best-effort: a failed call (down
        # server, parse error) or an out-of-vocab label (classify_doc_kind
        # returns None) also leaves "other" -- NO fabricated label. The tag
        # lets the complementary-temporal guard fire on a semantic signal
        # (both sources point_in_time_snapshot) instead of a filename month-
        # prefix, which is inert on real enterprise docs (the bench finding).
        if doc_kind_tagger is not None:
            try:
                kind = doc_kind_tagger.classify_doc_kind(self._doc_text(parsed))
            except Exception:  # noqa: BLE001 -- best-effort; cold-start safe
                kind = None
            if kind is not None:
                doc.doc_kind = kind

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