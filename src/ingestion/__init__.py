"""Document/record ingestion pipeline (task #17, the RAG-replacement pillar).

Format-pluggable ingestion that turns an external source (a markdown file, a
PDF, a web page, a transcript) into a ``Document`` projection in the
hippocampal index. The source of truth stays external (git / filesystem /
Drive); this is Pondr's *derived* semantic projection -- a materialized view.

The pipeline is explicit/conscious ONLY (no file watcher, no subconscious
sync layer -- the user's directive): a source is ingested by an explicit
``ingest_document`` call, and re-ingesting an already-ingested source UPDATES
it in place (reuses its id, hash-diffs its sections) rather than creating a
duplicate.

Three stages, composable:

1. ``parsers`` -- ``DocumentParser.parse(path) -> ParsedDocument`` (native
   structure: markdown headings -> sections). First slice: ``MarkdownParser``
   + ``PlainTextParser`` (zero-dep, offline-testable). Later phases add
   PDF/DOCX/web/email/code.
2. ``chunker`` -- ``HierarchicalChunker.chunk(parsed) -> ParsedDocument`` --
   structure-based, not arbitrary token windows. Leaf sizing
   (``max_section_tokens`` / ``min_section_tokens``) sub-splits oversized
   sections on paragraph boundaries and merges tiny ones into their parent;
   structure-less inputs get a degenerate paragraph split. The embedding-
   based semantic-boundary splitter is a Phase-2 item (needs the embedder).
3. ``pipeline`` -- ``UnifiedIngestionPipeline`` composes parser -> chunker ->
   per-section GLiNER extract + doc-level Bonsai relations -> ``Document`` ->
   ``store.encode_document``. GLiNER/Bonsai constructed lazily (heavy, GPU).
"""

from .chunker import HierarchicalChunker
from .parsers import (
    MarkdownParser,
    PlainTextParser,
    ParsedDocument,
    RawSection,
    detect_type,
)
from .pipeline import UnifiedIngestionPipeline
from .pdf_parser import PDFParser
from .code_parser import CodeParser
from .docx_parser import DocxParser
from .web_parser import WebParser

__all__ = [
    "HierarchicalChunker",
    "MarkdownParser",
    "PlainTextParser",
    "ParsedDocument",
    "RawSection",
    "detect_type",
    "UnifiedIngestionPipeline",
    "PDFParser",
    "CodeParser",
    "DocxParser",
    "WebParser",
]