"""Tests for the code parser (tree-sitter primary, stdlib ast Python fallback).

The ast fallback (Python-only, zero deps) is always exercised so the test
runs without tree-sitter installed. The tree-sitter path is checked too when
``tree_sitter_languages`` is importable (``importorskip``).
"""

from __future__ import annotations

import pytest

from src.ingestion.code_parser import CodeParser

_PY_SRC = '''"""Module docstring."""

import os
from pathlib import Path


def greet(name):
    """Say hi."""
    return f"hi {name}"


class Foo:
    """A class."""

    def method(self, x):
        return x + 1
'''


def test_code_ast_fallback_python():
    """The stdlib ast backend parses Python defs + the module root (no deps)."""
    parsed = CodeParser().parse_text(_PY_SRC, "m.py", "python")
    assert parsed.source_type == "code"
    assert parsed.language == "python"
    assert parsed.title == "m"
    headings = {s.heading for s in parsed.sections}
    # The module root holds the docstring + imports (outside the top-level defs).
    assert any(h == "<module>" for h in headings)
    assert "def greet(name)" in headings
    assert "class Foo" in headings
    # The nested method is a separate section.
    assert "def method(self, x)" in headings
    # The greet section's content is its full source span.
    greet = next(s for s in parsed.sections if s.heading == "def greet(name)")
    assert "return f\"hi {name}\"" in greet.content
    # The module-root content holds the docstring + imports, NOT the def bodies.
    root = next(s for s in parsed.sections if s.heading == "<module>")
    assert "import os" in root.content
    assert "def greet" not in root.content


def test_code_ast_module_root_excludes_top_defs():
    """Top-level def bodies are excluded from the module-root section."""
    parsed = CodeParser().parse_text(_PY_SRC, "m.py", "python")
    root = next(s for s in parsed.sections if s.heading == "<module>")
    foo = next(s for s in parsed.sections if s.heading == "class Foo")
    # The class body (and its method) is in the Foo section, not the root.
    assert "def method" not in root.content
    assert "def method" in foo.content


def test_code_unknown_extension_single_section():
    """An unknown extension -> one module section (source preserved)."""
    parsed = CodeParser().parse_text("just some text\n", "x.unknown", None)
    assert len(parsed.sections) == 1
    assert "just some text" in parsed.sections[0].content


# Source with a multi-byte (non-ASCII) char before a def on the same line and
# inside the module root: ast col_offset / tree-sitter byte offsets are UTF-8
# BYTE offsets, so spans must be computed against the encoded bytes (a str
# slice with byte offsets mis-slices non-ASCII source and drops the tail).
# chr(0x00e9) builds a real e-acute (2 UTF-8 bytes) at runtime; chr(10) a real
# newline -- so THIS test file stays pure ASCII (no non-ASCII literal, no
# backslash-escape pitfall in the heredoc that wrote it).
_E = chr(0x00e9)
_N = chr(10)
_NONASCII_SRC = (
    "# heading: " + _E + "clair" + _N + _N
    + "import os" + _N + _N + _N
    + "def make(" + _E + "):" + _N
    + "    return " + _E + " + 1" + _N
)


def test_code_ast_nonascii_byte_offsets():
    """Non-ASCII source is sliced byte-accurately (no truncation past the char)."""
    parsed = CodeParser().parse_text(_NONASCII_SRC, "m.py", "python")
    make = next(s for s in parsed.sections if s.heading.startswith("def make"))
    # The full def body survives (the return line is past the multi-byte char).
    assert ("return " + _E + " + 1") in make.content
    root = next(s for s in parsed.sections if s.heading == "<module>")
    assert (_E + "clair") in root.content


# tree-sitter path (multi-language) -- only when the dep is installed. The
# ``importorskip`` is INSIDE each tree-sitter test (NOT at module level): a
# module-level ``importorskip`` skips the WHOLE file, which would silently
# disable the zero-dep ast-fallback tests above (their whole point is to run
# WITHOUT tree-sitter).
def test_code_tree_sitter_python():
    """When tree-sitter is available, it is preferred over the ast fallback."""
    pytest.importorskip("tree_sitter_languages", reason="tree-sitter not installed")
    parsed = CodeParser().parse_text(_PY_SRC, "m.py", "python")
    headings = {s.heading for s in parsed.sections}
    assert "def greet(name)" in headings or "def greet" in " ".join(headings)
    assert "class Foo" in headings or any("Foo" in h for h in headings)


def test_code_tree_sitter_missing_dep_raises_for_non_python():
    """A non-Python file with no tree-sitter raises a clear RuntimeError."""
    import builtins
    real_import = builtins.__import__

    def _no_ts(name, *a, **k):
        if name == "tree_sitter_languages":
            raise ImportError("no ts")
        return real_import(name, *a, **k)

    builtins.__import__ = _no_ts
    try:
        with pytest.raises(RuntimeError, match="pip install -e .\\[ingestion\\]"):
            CodeParser().parse_text("fn main() {}", "x.go", "go")
    finally:
        builtins.__import__ = real_import
