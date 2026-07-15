"""Web / HTML parser -- beautifulsoup4 + lxml -> ParsedDocument.

Structure-preserving: each ``h1``-``h6`` opens a new ``RawSection`` at that
heading level, with the text up to the next heading (of any level) as its
``content``. A leading block before any heading becomes a level-1 section.
``title`` = the first ``<title>`` (else the first ``<h1>``, else the filename
stem). bs4 is chosen over trafilatura (which flattens to article text) because
the chunker is HIERARCHICAL -- keeping the heading structure lets the chunker
wire parent/child relationships. The ``lxml`` parser backend is specified
explicitly for stable, lenient parsing of real-world HTML.

Lazy import (GLiNER pattern): ``bs4`` + ``lxml`` are imported inside ``parse``
so the ingestion package stays importable without them, and a missing dep
raises a clear ``RuntimeError`` with the install hint. ``parse_text(html,
source_path)`` is the no-temp-file mirror tests use.
"""

from __future__ import annotations

import os
from typing import Optional

from .parsers import ParsedDocument, RawSection


class WebParser:
    """``parse(source_path) -> ParsedDocument`` via beautifulsoup4 + lxml."""

    source_type: str = "web"

    def parse(self, source_path: str) -> ParsedDocument:
        with open(source_path, "r", encoding="utf-8", errors="replace") as fh:
            html = fh.read()
        return self.parse_text(html, source_path)

    def parse_text(self, html: str, source_path: str = "") -> ParsedDocument:
        try:
            from bs4 import BeautifulSoup, NavigableString, Tag  # lazy import
        except ImportError as exc:
            raise RuntimeError(
                "WebParser needs beautifulsoup4 + lxml: pip install -e .[ingestion]"
            ) from exc

        soup = BeautifulSoup(html, "lxml")

        title = ""
        if soup.title and soup.title.get_text(strip=True):
            title = soup.title.get_text(strip=True)

        headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        sections: list[RawSection] = []

        if not headings:
            # No headings: the whole body text is one root section.
            body = soup.get_text(separator="\n", strip=True)
            if body:
                sections.append(RawSection(heading="", level=1, content=body))
        else:
            # Single document-order walk: a heading Tag opens a new section at
            # its level; every other visible text node appends to the current
            # section (creating a root section for any leading block before the
            # first heading -- the previous sibling-walk dropped that). Heading
            # text is captured by the heading branch (set as the section's
            # ``heading`` + prepended to the body so each chunk is self-
            # describing), so strings INSIDE a heading are skipped here to avoid
            # duplication. ``<script>``/``<style>``/``<head>``/``<title>`` text is
            # skipped (non-content).
            current: Optional[RawSection] = None

            def _inside_heading(node) -> bool:
                # Any heading ancestor -> the string is heading text (handled by
                # the heading branch, e.g. a <span> inside an <h2>).
                p = node.parent
                while p is not None:
                    if isinstance(p, Tag) and p.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                        return True
                    p = p.parent
                return False

            def _flush() -> None:
                nonlocal current
                if current is None:
                    return
                body = current.content.strip()
                heading = current.heading
                if body or heading:
                    # Self-describing: the heading text leads the body so a
                    # chunk reads as a complete unit even shown alone.
                    content = (heading + "\n" + body).strip() if (heading and body) else (body or heading)
                    sections.append(RawSection(heading=heading, level=current.level, content=content))
                current = None

            for node in soup.descendants:
                if isinstance(node, Tag) and node.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    _flush()
                    level = int(node.name[1])  # 'h2' -> 2
                    heading_text = node.get_text(strip=True)
                    if not title and level == 1:
                        title = heading_text
                    current = RawSection(heading=heading_text, level=level, content="")
                elif isinstance(node, NavigableString):
                    if not node.strip():
                        continue
                    pname = node.parent.name if node.parent else ""
                    if pname in ("script", "style", "title", "head"):
                        continue
                    if _inside_heading(node):
                        continue  # heading text captured by the heading branch
                    if current is None:
                        # Leading block before the first heading -> a root section.
                        current = RawSection(heading="", level=1, content="")
                    current.content += str(node).strip() + "\n"
            _flush()

        if not title:
            title = os.path.splitext(os.path.basename(source_path))[0]

        return ParsedDocument(
            source_type=self.source_type,
            source_path=source_path,
            sections=sections,
            title=title,
        )