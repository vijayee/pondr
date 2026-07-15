"""Source-code parser -- tree-sitter (primary) / stdlib ast (Python fallback).

A code file is structured by its definitions: a ``function_definition`` /
``class_definition`` / ``method_definition`` node is one ``RawSection`` whose
``content`` is the full source span of the node (``src[start:end]``) and whose
``heading`` is the name + signature. A **module-root section at level 1** holds
the module docstring + imports + top-level statements so they are not lost when
every top-level def becomes its own level-2 section. Nested defs nest by depth.

Two backends:

* **tree-sitter** (``tree_sitter_languages.get_parser``) -- multi-language: py,
  js/mjs, ts, c/h, cpp, go, rs, java. Lazy import; on a box without it the
  Python-only ast fallback still delivers the repo-dogfood win for ``.py``.
* **stdlib ``ast``** -- Python-only, zero deps. Used when tree-sitter is absent
  AND the file is Python. Non-Python files with no tree-sitter raise the clear
  ``RuntimeError`` (honest: we cannot parse them without the dep).

``parse_text(text, source_path, language)`` is the no-temp-file mirror tests
use. ``parse(source_path)`` reads the file and infers the language by
extension.
"""

from __future__ import annotations

import os
from typing import Optional

from .parsers import ParsedDocument, RawSection


# Extension -> language id. Used by both the tree-sitter path (to pick the
# parser) and the ast fallback (to decide whether ast is applicable).
_LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript",
    ".ts": "typescript",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

# Definition node types to treat as one section per backend. tree-sitter node
# types vary slightly across grammars; the union covers the supported langs.
_TS_DEF_TYPES = {
    "function_definition", "class_definition", "method_definition",
    "decorated_definition",  # a def/class with decorators wraps the real node
}

# Ast node types for the Python fallback.
_AST_DEF_TYPES = {"FunctionDef", "AsyncFunctionDef", "ClassDef"}


def _lang_for(source_path: str, language: Optional[str]) -> Optional[str]:
    if language:
        return language
    ext = os.path.splitext(source_path)[1].lower()
    return _LANG_BY_EXT.get(ext)


class CodeParser:
    """``parse(source_path) -> ParsedDocument`` over source-code structure."""

    source_type: str = "code"

    def parse(self, source_path: str) -> ParsedDocument:
        with open(source_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        language = _lang_for(source_path, None)
        return self.parse_text(text, source_path, language)

    def parse_text(
        self, text: str, source_path: str = "", language: Optional[str] = None
    ) -> ParsedDocument:
        language = _lang_for(source_path, language)
        if language is None:
            # Unknown extension: parse as a single module section (honest: no
            # structure extracted, but the source is preserved as one chunk).
            sections = [RawSection(heading="", level=1, content=text)] if text.strip() else []
        elif language == "python":
            # Prefer tree-sitter; fall back to stdlib ast if it is absent.
            sections = self._try_tree_sitter(text, language) or self._ast_sections(text)
        else:
            sections = self._try_tree_sitter(text, language)
            if not sections:
                raise RuntimeError(
                    f"CodeParser needs tree_sitter_languages for {language!r}: "
                    f"pip install -e .[ingestion]"
                )
        title = os.path.splitext(os.path.basename(source_path))[0]
        return ParsedDocument(
            source_type=self.source_type,
            source_path=source_path,
            sections=sections,
            title=title,
            language=language,
        )

    # ── tree-sitter backend ──

    def _try_tree_sitter(self, text: str, language: str) -> list[RawSection]:
        try:
            from tree_sitter_languages import get_parser  # lazy
        except ImportError:
            return []
        try:
            parser = get_parser(language)
        except Exception:
            # Grammar not bundled for this language -> fall back to caller.
            return []
        tree = parser.parse(bytes(text, "utf-8"))
        return self._ts_walk(tree.root_node, text)

    def _ts_walk(self, root, text: str) -> list[RawSection]:
        """Emit a module-root section + one section per def, nested by depth.

        The module root (level 1) holds everything OUTSIDE top-level defs
        (docstring, imports, top-level statements). Each top-level def is level
        2; nested defs deepen by one per enclosing def. ``decorated_definition``
        wraps the real node, so unwrap it to get the name/signature.

        tree-sitter ``start_byte``/``end_byte`` are UTF-8 BYTE offsets, so the
        source is encoded once and all spans slice the BYTES then decode (slicing
        a ``str`` with byte offsets is wrong for non-ASCII source).
        """
        src_b = text.encode("utf-8")

        def seg(a: int, b: int) -> str:
            return src_b[a:b].decode("utf-8", "replace")

        sections: list[RawSection] = []

        def unwrap(node):
            # decorated_definition -> the inner function/class node
            if node.type == "decorated_definition" and node.child_count >= 2:
                inner = node.child_by_field_name("definition")
                return inner or node.children[-1]
            return node

        def name_of(node) -> str:
            n = node.child_by_field_name("name")
            return seg(n.start_byte, n.end_byte) if n is not None else "<anon>"

        def params_of(node) -> str:
            params = node.child_by_field_name("parameters")
            if params is None:
                return ""
            return seg(params.start_byte, params.end_byte)

        def heading_for(node) -> str:
            if node.type == "class_definition":
                return f"class {name_of(node)}"
            if node.type in ("function_definition", "method_definition"):
                return f"def {name_of(node)}{params_of(node)}"
            return name_of(node)

        # Collect top-level def node spans so the module root can exclude them.
        top_def_spans: list[tuple[int, int]] = []
        for child in root.children:
            real = unwrap(child)
            if real.type in _TS_DEF_TYPES:
                top_def_spans.append((real.start_byte, real.end_byte))

        # Build the module-root content from the gaps between top-level defs.
        root_parts: list[str] = []
        cursor = 0
        for start, end in sorted(top_def_spans):
            if start > cursor:
                root_parts.append(seg(cursor, start))
            cursor = max(cursor, end)
        if cursor < len(src_b):
            root_parts.append(seg(cursor, len(src_b)))
        root_body = "".join(root_parts).strip()
        if root_body:
            sections.append(RawSection(heading="<module>", level=1, content=root_body))

        # Recurse defs, nesting by depth (top-level = level 2).
        def walk(node, depth: int) -> None:
            for child in node.children:
                real = unwrap(child)
                if real.type in _TS_DEF_TYPES:
                    heading = heading_for(real)
                    body = seg(real.start_byte, real.end_byte)
                    sections.append(
                        RawSection(heading=heading, level=depth + 1, content=body)
                    )
                    walk(real, depth + 1)
                else:
                    walk(child, depth)

        walk(root, 1)
        return sections

    # ── stdlib ast backend (Python-only) ──

    def _ast_sections(self, text: str) -> list[RawSection]:
        import ast

        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Unparseable -> one module section (honest: structure not extractable).
            return [RawSection(heading="", level=1, content=text)] if text.strip() else []

        # Python ast ``col_offset`` / ``end_col_offset`` are UTF-8 BYTE offsets
        # into the source, so spans must be computed against the ENCODED bytes
        # (not the ``str`` line lengths -- that mis-slices non-ASCII source).
        # Precompute the byte offset of the start of each 1-based line.
        src_b = text.encode("utf-8")
        line_start_b = [0]
        for ln in text.splitlines(keepends=True):
            line_start_b.append(line_start_b[-1] + len(ln.encode("utf-8")))

        def span(node) -> tuple[int, int]:
            start = line_start_b[node.lineno - 1] + node.col_offset
            # end_lineno/end_col_offset may be absent on old Pythons; fallback.
            end_ln = getattr(node, "end_lineno", node.lineno) or node.lineno
            end_col = getattr(node, "end_col_offset", 0) or 0
            end = line_start_b[end_ln - 1] + end_col
            return start, min(end, len(src_b))

        def seg(start: int, end: int) -> str:
            return src_b[start:end].decode("utf-8", "replace")

        sections: list[RawSection] = []

        top_def_spans: list[tuple[int, int]] = []
        top_defs: list[ast.AST] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top_def_spans.append(span(node))
                top_defs.append(node)

        # Module root: source outside the top-level defs.
        cursor = 0
        root_parts: list[str] = []
        for start, end in sorted(top_def_spans):
            if start > cursor:
                root_parts.append(seg(cursor, start))
            cursor = max(cursor, end)
        if cursor < len(src_b):
            root_parts.append(seg(cursor, len(src_b)))
        root_body = "".join(root_parts).strip()
        if root_body:
            sections.append(RawSection(heading="<module>", level=1, content=root_body))

        def heading_for(node) -> str:
            if isinstance(node, ast.ClassDef):
                return f"class {node.name}"
            # FunctionDef / AsyncFunctionDef: reconstruct the signature.
            kind = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            args = ", ".join(a.arg for a in node.args.args)
            return f"{kind}{node.name}({args})"

        def walk(node, depth: int) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    s, e = span(child)
                    sections.append(
                        RawSection(
                            heading=heading_for(child),
                            level=depth + 1,
                            content=seg(s, e),
                        )
                    )
                    walk(child, depth + 1)

        for top in top_defs:
            s, e = span(top)
            sections.append(
                RawSection(heading=heading_for(top), level=2, content=seg(s, e))
            )
            walk(top, 2)
        return sections