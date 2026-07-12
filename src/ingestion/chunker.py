"""HierarchicalChunker -- structure-based chunking, not arbitrary token windows.

Takes a ``ParsedDocument`` (a flat, ordered, level-tagged list of raw sections
from a parser) and returns a normalized ``ParsedDocument`` whose sections are
retrieval-sized leaves with the hierarchy wired. The chat's design (IDX 88,
"the document's own structure provides natural boundaries") is
structure-based; we close two gaps it glosses over:

* **Leaf sizing.** A 50-page chapter under one ``#`` would be one giant chunk
  -- one embedding (embedders cap ~512 tokens) and one retrieval pull that
  blows the context window (the failure IDX 136 describes for episodes).
  ``max_section_tokens`` (default 512, the embedder cap) sub-splits an
  oversized section on paragraph boundaries; ``min_section_tokens`` (default
  64) merges a too-small section into its parent. Each chunk ends up
  one-embedding-pass + retrieval-sized. Token estimate reuses the codebase's
  ``len(text)//4`` (``ssm_chunker.py``, ``chunked_context.py``).
* **Structure-less fallback.** A heading-less input has no structure to split
  on. The plain-text parser already paragraph-splits it; the chunker applies
  leaf sizing to those paragraphs. The embedding-based semantic-boundary
  splitter (cosine-similarity drops between passages) is a Phase-2 item --
  it needs the lazy-init embedder (the same deferred pattern as GLiNER) and
  would replace paragraph-splitting for genuinely ambiguous inputs.

Parent wiring: the chunker computes each section's parent from the level
stack (a section's parent is the most recent section at a lower level), then
FLATTENS to a depth-first list with stable indices so ``parent_index`` can
point at the parent's position in that list. ``Document.from_parse`` maps
``parent_index`` to the compound section id (``{doc_id}_sec_{i:03d}``).

Sub-split sections: an oversized section sub-split on paragraphs becomes N
sibling leaves at the SAME level, all sharing the original parent. The
heading gets a ``(N)`` suffix on the splits so each is distinguishable.
"""

from dataclasses import dataclass
from typing import Optional

from .parsers import ParsedDocument, RawSection


def _estimate_tokens(text: str) -> int:
    """Rough token count (chars // 4) -- the codebase's standing estimate.

    Same heuristic as ``ssm_chunker.SSMChunker._estimate_tokens`` and
    ``chunked_context._estimate_tokens``: a coarse char->token approximation
    good enough for sizing decisions (the embedder's hard cap is what
    ``max_section_tokens`` defaults to).
    """
    return max(1, len(text) // 4)


def _split_paragraphs(content: str) -> list[str]:
    """Split a section body on blank lines, dropping empties."""
    import re
    parts = re.split(r"\n\s*\n", content.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class HierarchicalChunker:
    """Structure-based leaf-sizing chunker.

    ``max_section_tokens`` / ``min_section_tokens`` bound leaf size. A
    ``None`` ``max_section_tokens`` disables the upper bound (sections are
    never sub-split). ``semantic_split_threshold`` is reserved for the Phase-2
    embedding-based splitter and is a no-op in this slice.
    """

    max_section_tokens: Optional[int] = 512
    min_section_tokens: int = 64
    semantic_split_threshold: Optional[float] = None

    def chunk(self, parsed: ParsedDocument) -> ParsedDocument:
        """Normalize ``parsed`` into leaf-sized, hierarchy-wired sections.

        Returns a NEW ``ParsedDocument`` (the input is not mutated) whose
        ``sections`` are the flattened, parent-wired leaves. Doc metadata
        (title/authors/...) is preserved.
        """
        # 1. Sub-split oversized sections on paragraphs (same level, same
        #    parent stack -- wiring recomputed after, so we only sub-split the
        #    body here).
        expanded: list[RawSection] = []
        for sec in parsed.sections:
            if (self.max_section_tokens is not None
                    and _estimate_tokens(sec.content) > self.max_section_tokens):
                expanded.extend(self._subsplit(sec))
            else:
                expanded.append(sec)

        # 2. Wire parents from the level stack + flatten with stable indices.
        wired = self._wire_parents(expanded)

        # 3. Merge too-small leaves into their parent (min_section_tokens).
        merged = self._merge_tiny(wired) if self.min_section_tokens > 0 else wired

        # Re-index parent_index after the merge (indices shifted).
        merged = self._reindex(merged)

        return ParsedDocument(
            source_type=parsed.source_type,
            source_path=parsed.source_path,
            sections=merged,
            title=parsed.title,
            authors=list(parsed.authors),
            created_at=parsed.created_at,
            language=parsed.language,
            metadata=dict(parsed.metadata),
        )

    def _subsplit(self, sec: RawSection) -> list[RawSection]:
        """Split an oversized section on paragraphs into <=max leaves.

        Greedily packs paragraphs into leaves up to ``max_section_tokens``;
        a single paragraph larger than the cap becomes its own leaf (we do not
        split mid-paragraph in this slice -- a Phase-2 refinement). Splits
        share the original level; the heading gets a ``(N)`` suffix.
        """
        paras = _split_paragraphs(sec.content)
        if not paras:
            return [sec]
        out: list[RawSection] = []
        buf = ""
        count = 0
        cap = self.max_section_tokens
        for p in paras:
            p_tokens = _estimate_tokens(p)
            if buf and _estimate_tokens(buf) + p_tokens > cap:
                count += 1
                out.append(RawSection(
                    heading=f"{sec.heading} ({count})" if sec.heading else f"({count})",
                    level=sec.level, content=buf.strip()))
                buf = ""
            buf = (buf + "\n\n" + p) if buf else p
        if buf.strip():
            count += 1
            out.append(RawSection(
                heading=f"{sec.heading} ({count})" if sec.heading else f"({count})",
                level=sec.level, content=buf.strip()))
        return out or [sec]

    def _wire_parents(self, sections: list[RawSection]) -> list[RawSection]:
        """Set ``parent_index`` from a level stack, in a stable flat list.

        The parent of a section at level L is the most recent section at a
        level < L (the nearest ancestor); sections at level 1 have no parent.
        Sibling sub-splits (same level, same heading prefix) share the parent
        of the original section. Returns the same list with ``parent_index``
        filled in.
        """
        out: list[RawSection] = []
        # Stack of (level, index) for open ancestors.
        stack: list[tuple[int, int]] = []
        for sec in sections:
            while stack and stack[-1][0] >= sec.level:
                stack.pop()
            parent_index = stack[-1][1] if stack else None
            out.append(RawSection(
                heading=sec.heading, level=sec.level, content=sec.content,
                parent_index=parent_index))
            stack.append((sec.level, len(out) - 1))
        return out

    def _merge_tiny(self, sections: list[RawSection]) -> list[RawSection]:
        """Merge leaves below ``min_section_tokens`` into their parent's body.

        A tiny section's body is appended to its parent's body (the parent's
        heading/level stand); the tiny section is dropped. Root-level tiny
        sections (no parent) are kept as-is (nothing to merge into). Operates
        on parent indices that will be re-indexed next, so it merges by
        PARENT INDEX, not by final id.
        """
        if not sections:
            return []
        drop: set[int] = set()
        # Append children into parents, processing deepest first so a parent
        # that is itself tiny may absorb a child before being absorbed.
        bodies = [s.content for s in sections]
        levels = [s.level for s in sections]
        parents = [s.parent_index for s in sections]
        order = sorted(range(len(sections)),
                       key=lambda i: -levels[i])  # deepest first
        for i in order:
            if i in drop:
                continue
            if _estimate_tokens(bodies[i]) >= self.min_section_tokens:
                continue
            p = parents[i]
            if p is None or p in drop:
                continue  # nothing to merge into -> keep
            # Merge this tiny section into its parent.
            sep = "\n\n" if bodies[p] else ""
            bodies[p] = bodies[p] + sep + bodies[i]
            drop.add(i)
        return [RawSection(heading=sections[i].heading, level=sections[i].level,
                           content=bodies[i], parent_index=parents[i])
                for i in range(len(sections)) if i not in drop]

    def _reindex(self, sections: list[RawSection]) -> list[RawSection]:
        """Remap ``parent_index`` after a list compaction (merge/drop).

        Builds old->new index map for the survivors and rewrites each
        section's ``parent_index``; an orphaned parent (dropped) is re-pointed
        to its grandparent if available, else None.
        """
        if not sections:
            return []
        # The input's parent_index refers to positions in the PRE--compaction
        # list; ``_merge_tiny`` already used the pre-compaction indices. We
        # rebuild a clean parent stack from levels instead -- the level stack
        # is robust to compaction (parents are nearest lower-level survivor).
        return self._wire_parents(sections)