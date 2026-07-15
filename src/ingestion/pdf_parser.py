"""PDF parser -- pypdf -> ParsedDocument.

Structure-preserving: when the PDF carries a document outline (the TOC /
bookmarks), each outline entry becomes a ``RawSection`` at its nesting depth,
with the page text under that heading as ``content``. TOC-less PDFs fall back to
ONE section per page (heading ``Page {n}``, level 1) -- coarse but honest, since
a page is the only structural boundary ``extract_text`` can see. ``pypdf`` is
the chosen dep (already installed in this env; lighter/more stable than
pymupdf for the text-only first slice -- font-size-based heading detection is a
later refinement).

Lazy import (GLiNER pattern): ``pypdf`` is imported inside ``parse`` so the
ingestion package stays importable without it, and a missing dep raises a
clear ``RuntimeError`` with the install hint rather than a bare ``ImportError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .parsers import ParsedDocument, RawSection


class PDFParser:
    """``parse(source_path) -> ParsedDocument`` via pypdf."""

    source_type: str = "pdf"

    def parse(self, source_path: str) -> ParsedDocument:
        try:
            import pypdf  # lazy: keeps the package importable without the dep
        except ImportError as exc:
            raise RuntimeError(
                "PDFParser needs pypdf: pip install -e .[ingestion]"
            ) from exc

        reader = pypdf.PdfReader(source_path)
        page_texts: list[str] = []
        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception:
                # extract_text can raise on malformed pages; an empty page is
                # honest -- the section just has no body.
                page_texts.append("")

        title = ""
        created_at: Optional[str] = None
        meta = reader.metadata
        if meta is not None:
            try:
                raw_title = meta.get("/Title", "") if hasattr(meta, "get") else getattr(meta, "title", "")
                if raw_title:
                    title = str(raw_title).strip()
            except Exception:
                pass
            try:
                cd = meta.creation_date if hasattr(meta, "creation_date") else None
                if cd is not None:
                    created_at = cd.isoformat()
            except Exception:
                pass

        sections: list[RawSection] = []
        outline = getattr(reader, "outline", None)

        if outline:
            # Walk the outline recursively. A list entry is a Destination (a
            # heading); a nested list is its children (deeper headings). The
            # page a Destination points at bounds the text for that section.
            try:
                sections = self._sections_from_outline(outline, page_texts, reader)
            except Exception:
                # A malformed outline should not lose the whole doc; fall back
                # to the per-page path.
                sections = []
        if not sections:
            # Per-page fallback for TOC-less PDFs.
            for i, text in enumerate(page_texts, start=1):
                body = text.strip()
                if body:
                    sections.append(
                        RawSection(heading=f"Page {i}", level=1, content=body)
                    )

        if not title:
            # Filename stem as the last-resort title.
            import os
            title = os.path.splitext(os.path.basename(source_path))[0]

        return ParsedDocument(
            source_type=self.source_type,
            source_path=source_path,
            sections=sections,
            title=title,
            created_at=created_at,
        )

    def _sections_from_outline(self, outline, page_texts: list[str], reader=None) -> list[RawSection]:
        """Walk a pypdf outline tree -> one RawSection per heading.

        Each section's ``content`` is the page text from its target page up to
        the next sibling/parent heading's page. Nesting depth -> ``level`` (the
        top outline list is level 1). Destinations that fail to resolve to a
        page index get an empty body (best-effort; the heading still records a
        section boundary). ``reader`` is the open ``PdfReader`` so a
        ``Destination.page`` (a ``PageObject`` / indirect ref) can be resolved to
        a 0-based page index via ``reader.get_page_index``.
        """
        out: list[RawSection] = []

        def page_index(dest) -> Optional[int]:
            try:
                # pypdf outline items expose their target page via the ``page``
                # property (a Destination) or as a raw ``/Page`` dict entry. The
                # value is a PageObject (an indirect ref), NOT an int and with no
                # ``page_number`` attr -- so resolve it to a 0-based index via
                # the reader. ``get_page_index`` is the pypdf API for that.
                if hasattr(dest, "page"):
                    pg = dest.page
                elif hasattr(dest, "get"):
                    pg = dest.get("/Page")
                else:
                    pg = None
                if pg is None:
                    return None
                if isinstance(pg, int):
                    return pg
                if reader is not None:
                    get_idx = getattr(reader, "get_page_index", None)
                    if get_idx is not None:
                        return get_idx(pg)
                return None
            except Exception:
                return None

        def title_of(dest) -> str:
            try:
                t = dest.get("/Title") if hasattr(dest, "get") else getattr(dest, "title", "")
                return str(t).strip() if t else ""
            except Exception:
                return ""

        # Flatten the nested tree into an ordered list of (level, page, title)
        # by recursing. ``outline`` is a list whose items are either
        # Destination objects or nested lists (the children of the preceding
        # Destination).
        flat: list[tuple[int, Optional[int], str]] = []

        def walk(items, depth: int) -> None:
            for item in items:
                if isinstance(item, list):
                    walk(item, depth + 1)
                else:
                    flat.append((depth, page_index(item), title_of(item)))

        walk(outline, 1)
        flat = [f for f in flat if f[2]]  # drop anonymous headings

        for idx, (level, pg, heading) in enumerate(flat):
            if pg is None:
                content = ""
            else:
                # Content spans from this heading's page to the page before
                # the next heading (any depth) that starts on a later page.
                end_page = len(page_texts)
                for nxt_level, nxt_pg, _ in flat[idx + 1:]:
                    if nxt_pg is not None and nxt_pg > pg:
                        end_page = nxt_pg
                        break
                content = "\n".join(page_texts[pg:end_page]).strip()
            if content or heading:
                out.append(RawSection(heading=heading, level=level, content=content))
        return out