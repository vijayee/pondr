"""Tests for the PDF parser (pypdf).

Skipped cleanly when pypdf is absent (``importorskip``). Builds a real 2-page
PDF in memory with text (via pypdf's writer + a text-bearing page built from a
minimal content stream), exercises the per-page fallback path (TOC-less PDFs ->
one section per page), and the filename-stem title fallback.
"""

from __future__ import annotations

import os

import pytest

pypdf = pytest.importorskip("pypdf")

from src.ingestion.pdf_parser import PDFParser


def _write_text_pdf(path: str, pages: list[str]) -> None:
    """Write a minimal PDF whose pages carry the given text.

    Builds each page from a hand-rolled content stream (the standard ``BT ...
    Tj`` text object) so ``extract_text`` returns the page's text. Avoids a
    reportlab dependency by attaching the content stream as a raw ``StreamObject``.
    """
    from pypdf import PdfWriter
    from pypdf.generic import (
        DictionaryObject, NameObject, StreamObject,
    )

    writer = PdfWriter()
    for body in pages:
        escaped = body.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        cs = StreamObject()
        cs.set_data(stream.encode("latin-1"))
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Contents")] = writer._add_object(cs)
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font)
        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_ref
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = writer._add_object(resources)
    with open(path, "wb") as fh:
        writer.write(fh)


def test_pdf_per_page_fallback_sections(tmp_path):
    fix = tmp_path / "doc.pdf"
    _write_text_pdf(str(fix), ["Alice storage subsystem", "Bob networking transport"])
    parsed = PDFParser().parse(str(fix))
    assert parsed.source_type == "pdf"
    # No outline -> per-page fallback: one section per page (>=2).
    assert len(parsed.sections) >= 2
    # Page text is extractable and bucketed under a section.
    joined = "\n".join(s.content for s in parsed.sections)
    assert "Alice" in joined or "storage" in joined.lower()
    # Title falls back to the filename stem (no /Title metadata set).
    assert parsed.title == "doc"
    # Headings are the per-page labels.
    assert any("Page" in s.heading for s in parsed.sections)


def test_pdf_missing_dep_clear_error(tmp_path, monkeypatch):
    """A missing pypdf raises a clear RuntimeError with the install hint."""
    fix = tmp_path / "doc.pdf"
    _write_text_pdf(str(fix), ["hello"])
    import builtins
    real_import = builtins.__import__

    def _no_pypdf(name, *a, **k):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_pypdf)
    with pytest.raises(RuntimeError, match="pip install -e .\\[ingestion\\]"):
        PDFParser().parse(str(fix))