"""Tests for the web/HTML parser (beautifulsoup4 + lxml).

The structure tests are skipped cleanly when bs4/lxml are absent
(``importorskip`` INSIDE each test, not at module level -- a module-level
``importorskip`` would skip the whole file, including the missing-dep error
test, which is meant to run WITHOUT bs4). The missing-dep test needs no deps:
``WebParser.parse_text`` lazy-imports bs4 and raises the ``RuntimeError`` at
that import, before parsing.
"""

from __future__ import annotations

import pytest

from src.ingestion.web_parser import WebParser


_HTML = """<html><head><title>Project Notes</title></head><body>
<p>Intro about the hippocampal index.</p>
<h1>Alice on Storage</h1>
<p>Alice architected the storage subsystem.</p>
<h2>Cold blob store</h2>
<p>The cold store is content-addressed.</p>
<h1>Bob on Networking</h1>
<p>Bob implemented the networking transport.</p>
</body></html>"""


def test_web_headings_to_sections():
    pytest.importorskip("bs4", reason="beautifulsoup4 not installed")
    pytest.importorskip("lxml", reason="lxml not installed")
    parsed = WebParser().parse_text(_HTML, "doc.html")
    assert parsed.source_type == "web"
    headings = [s.heading for s in parsed.sections]
    assert "Alice on Storage" in headings
    assert "Bob on Networking" in headings
    # Levels track the heading number (h1 -> 1, h2 -> 2).
    alice = next(s for s in parsed.sections if s.heading == "Alice on Storage")
    assert alice.level == 1
    cold = next((s for s in parsed.sections if s.heading == "Cold blob store"), None)
    assert cold is not None and cold.level == 2
    # Body text bucketed under the right heading.
    assert "storage" in alice.content.lower()
    bob = next(s for s in parsed.sections if s.heading == "Bob on Networking")
    assert "networking" in bob.content.lower()
    # Title from <title>.
    assert parsed.title == "Project Notes"


def test_web_missing_dep_clear_error(monkeypatch):
    # Runs WITHOUT bs4 installed (the box where the clear error matters): the
    # parser raises at the lazy ``from bs4 import ...`` before it parses.
    import builtins
    real_import = builtins.__import__

    def _no_bs4(name, *a, **k):
        if name == "bs4":
            raise ImportError("no bs4")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_bs4)
    with pytest.raises(RuntimeError, match="pip install -e .\\[ingestion\\]"):
        WebParser().parse_text(_HTML, "doc.html")


def test_web_no_headings_one_section():
    pytest.importorskip("bs4", reason="beautifulsoup4 not installed")
    pytest.importorskip("lxml", reason="lxml not installed")
    parsed = WebParser().parse_text(
        "<html><body><p>Just a block of text.</p></body></html>", "x.html")
    assert len(parsed.sections) == 1
    assert "block of text" in parsed.sections[0].content.lower()