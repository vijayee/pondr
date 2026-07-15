"""Format-pluggable document parsers -- source -> native structure.

A parser turns an external source into a ``ParsedDocument``: a title +
metadata + a flat list of ``RawSection`` leaves, each carrying the
structural level the source itself provides (markdown ``#`` -> 1, ``##`` -> 2,
...). The chunker (``chunker.py``) then normalizes that list (leaf sizing +
parent wiring). Parsers are additive: each format gets its own parser
registering in ``detect_type``. First slice: ``MarkdownParser`` (headings via
stdlib regex) + ``PlainTextParser`` (degenerate paragraph split). Later
phases add PDF (pymupdf TOC + font-size), web (trafilatura h1-h6), DOCX
(python-docx heading styles), email (thread structure), code (tree-sitter
AST) -- all behind the same ``parse -> ParsedDocument`` interface.

Zero external deps in this slice (stdlib only) so the parsers are testable
offline. ``RawSection.parent_index`` is left ``None`` here; the chunker wires
the hierarchy (a parser only emits the flat, ordered, level-tagged list the
chunker builds the tree from).
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class RawSection:
    """One structural unit a parser emits, before chunker normalization.

    ``level`` is the structural depth the source provides (markdown heading
    level; structure-less inputs flatten to 1). ``parent_index`` is the
    index of the parent section in the SAME list (``None`` at the root); the
    parser leaves it ``None`` and the chunker wires it once it has built the
    tree.
    """

    heading: str
    level: int
    content: str
    parent_index: Optional[int] = None
    # Optional per-chunk dense vector, filled by the pipeline's embedder AFTER
    # chunking (so the chunker -- which builds new RawSection objects -- cannot
    # drop it). ``Document.from_parse`` copies it to ``DocumentSection.embedding``.
    embedding: Optional[list[float]] = None


@dataclass
class ParsedDocument:
    """The output of a parser: metadata + an ordered list of raw sections."""

    source_type: str
    source_path: str
    sections: list[RawSection] = field(default_factory=list)
    title: str = ""
    authors: list[str] = field(default_factory=list)
    created_at: Optional[str] = None
    language: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class DocumentParser(Protocol):
    """``parse(source_path) -> ParsedDocument`` -- the parser contract."""

    def parse(self, source_path: str) -> ParsedDocument: ...


# Extension -> source_type. ``detect_type`` picks a parser by extension; an
# explicit ``--type`` from the CLI overrides. Add rows here as parsers land.
_TYPE_BY_EXT = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "web", ".htm": "web",
    # Source code -- one row per supported extension (CodeParser infers the
    # language from the extension).
    ".py": "code", ".js": "code", ".mjs": "code", ".ts": "code",
    ".c": "code", ".h": "code", ".cpp": "code", ".cc": "code", ".cxx": "code",
    ".hpp": "code", ".go": "code", ".rs": "code", ".java": "code",
}


def detect_type(source_path: str) -> str:
    """Infer a source_type from the path extension (``"text"`` fallback)."""
    ext = os.path.splitext(source_path)[1].lower()
    return _TYPE_BY_EXT.get(ext, "text")


# A markdown ATX heading: 1-6 ``#`` + a space + the heading text. The trailing
# ``#``-run (closing) is optional. Compiled once.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+\s*)?$")


@dataclass
class MarkdownParser:
    """Markdown -> sections by ATX heading (``#`` level -> section level).

    The first heading at level 1 is treated as the document title if no title
    was set; a leading block of text before any heading becomes a level-1
    section (``""`` heading) so it is not lost. Front-matter / fenced code
    blocks are NOT special-cased in the first slice (their ``#`` inside a
    fence would be misread as a heading) -- a Phase-2 refinement strips
    fences; the first-slice tests use plain markdown.
    """

    source_type: str = "markdown"

    def parse(self, source_path: str) -> ParsedDocument:
        with open(source_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        return self.parse_text(text, source_path)

    def parse_text(self, text: str, source_path: str = "") -> ParsedDocument:
        title = ""
        sections: list[RawSection] = []
        current: Optional[RawSection] = None
        # Buffer leading text before the first heading as a root section.
        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                if current is not None:
                    current.content = current.content.strip()
                    sections.append(current)
                level = len(m.group(1))
                heading = m.group(2).strip()
                if level == 1 and not title:
                    title = heading
                current = RawSection(heading=heading, level=level, content="")
            else:
                if current is None:
                    current = RawSection(heading="", level=1, content="")
                current.content += line + "\n"
        if current is not None:
            current.content = current.content.strip()
            if current.content or current.heading:
                sections.append(current)
        return ParsedDocument(
            source_type=self.source_type,
            source_path=source_path,
            sections=sections,
            title=title,
        )


@dataclass
class PlainTextParser:
    """Plain text -> a degenerate paragraph split (structure-less fallback).

    Structure-based chunking has no boundaries to split on for a heading-less
    text; this parser splits on blank lines (paragraphs), each paragraph a
    level-1 section with an empty heading. This is the offline-testable
    fallback; the embedding-based semantic-boundary splitter is a Phase-2
    item (needs the embedder) and would replace this for ambiguous inputs.
    """

    source_type: str = "text"

    def parse(self, source_path: str) -> ParsedDocument:
        with open(source_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        return self.parse_text(text, source_path)

    def parse_text(self, text: str, source_path: str = "") -> ParsedDocument:
        sections: list[RawSection] = []
        for para in re.split(r"\n\s*\n", text.strip()):
            body = para.strip()
            if body:
                sections.append(RawSection(heading="", level=1, content=body))
        # First line of the first paragraph as a title heuristic.
        title = ""
        if sections:
            first_line = sections[0].content.splitlines()[0] if sections[0].content else ""
            title = first_line[:80]
        return ParsedDocument(
            source_type=self.source_type,
            source_path=source_path,
            sections=sections,
            title=title,
        )


_PARSERS = {
    "markdown": MarkdownParser,
    "text": PlainTextParser,
    # The four format parsers live in their own modules (each lazy-imports its
    # heavy dep inside ``parse``). They are imported lazily here to keep this
    # module importable without their deps AND to avoid a circular import
    # (those modules ``from .parsers import ParsedDocument, RawSection``).
    "pdf": "src.ingestion.pdf_parser.PDFParser",
    "code": "src.ingestion.code_parser.CodeParser",
    "docx": "src.ingestion.docx_parser.DocxParser",
    "web": "src.ingestion.web_parser.WebParser",
}


def get_parser(source_type: str) -> DocumentParser:
    """Instantiate the parser for a source_type (``"auto"`` -> text fallback)."""
    cls = _PARSERS.get(source_type, PlainTextParser)
    if isinstance(cls, str):
        # Lazy import of a format parser module path.
        import importlib
        mod_name, _, attr = cls.rpartition(".")
        cls = getattr(importlib.import_module(mod_name), attr)
    return cls()