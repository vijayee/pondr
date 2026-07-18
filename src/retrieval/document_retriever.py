"""Document-aware retrieval that aggregates section hits into document results.

Phase 1c Refinement 1 (the RAG-replacement pillar's aggregation layer). When a
query matches MULTIPLE sections from the same document -- the realistic case
once a 100-page PDF is ingested as ~200 chunked sections -- the raw result
list contains one entry per matched section (``kind="section"``), plus possibly
a doc-level graph hit (``kind="document"``) for the same doc. Returning those
verbatim sends the generator a wall of undifferentiated chunks with no signal
that sections 3, 7, and 12 are all from the same source.

``DocumentRetriever.aggregate_results`` groups section/document results by
their parent document and returns ONE document result per parent (with the
matched sections highlighted and counted), leaving non-document (conversation
episode, semantic-memory) results untouched. The final list is re-sorted by
score so documents and episodes interleave by relevance.

Adaptation note (vs. ``docs/Phase 1c.md`` sec 3.4 reference): the reference
detected document sections by scanning the graph for an outgoing ``child_of``
edge and counted sections via a ``has_section`` POS scan. The shipped pipeline
is richer: hydrated section results already carry ``kind="section"`` and a
``doc_id`` field (``graph_traversal._hydrate_section``), and doc results carry
``kind="document"``. ``HippocampalStore.get_document(doc_id, load_bodies=False)``
returns the title / timestamp / source_path / full section list (metadata-only,
no cold pull). This implementation uses those existing fields/APIs directly --
it groups by ``doc_id``/``episode_id`` and reads doc metadata via
``get_document`` -- rather than re-scanning the graph. The aggregation
semantics (best score, entity/topic union, matched-section summary, document
context) match the reference.

Construction is guarded (see ``runtime.build_ponder``): the retriever only
attaches a ``DocumentRetriever`` when the store actually has document nodes
(a cheap ``has_section`` POS probe returns >0), so conversation-only corpora
skip aggregation entirely -- zero overhead, no behavior change.
"""

from __future__ import annotations

from typing import Any, Optional

from ..memory.store import HippocampalStore


def _b2s(v: Any) -> str:
    """Decode a WaveDB value to ``str`` (bytes-safe); ``None`` -> empty."""
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)


class DocumentRetriever:
    """Wraps the graph traversal engine with document-aware result aggregation.

    ``aggregate_results`` is a pure post-processing pass over the ranked result
    list; it does not re-rank by retrieval score (the caller already scored the
    inputs) -- it only re-sorts the merged document+episode list by score at
    the end. Document metadata (title/timestamp/source_path/total_sections)
    comes from ``store.get_document(doc_id, load_bodies=False)`` (no cold pull);
    the matched sections' bodies come from the section results' already-
    materialized ``text`` (the semantic-fallback / hydrate path pulled them), so
    aggregation itself does ZERO cold pulls.
    """

    def __init__(self, store: HippocampalStore) -> None:
        self.store = store

    def aggregate_results(self, raw_results: list[dict]) -> list[dict]:
        """Aggregate raw section/document results into per-document results.

        - ``kind="section"`` results group under their ``doc_id`` (falling back
          to the graph ``child_of`` edge when ``doc_id`` is absent, for
          forward-compat with results hydrated by older code paths).
        - ``kind="document"`` results group under their own ``episode_id``;
          their already-materialized matched-section body seeds the document's
          ``text`` and their ``matched_section`` heading is counted as a match.
        - Everything else (conversation episodes, semantic memories) passes
          through unchanged.

        Returns a new list (inputs are not mutated), re-sorted by score.
        """
        groups: dict[str, dict] = {}
        order: list[str] = []           # doc_ids in first-appearance order
        regular: list[dict] = []

        for r in raw_results:
            kind = r.get("kind")
            doc_id: Optional[str] = None
            if kind == "section":
                doc_id = r.get("doc_id") or self._parent_via_child_of(r["episode_id"])
            elif kind == "document":
                doc_id = r["episode_id"]

            if doc_id is None:
                regular.append(r)
                continue

            group = groups.get(doc_id)
            if group is None:
                group = self._new_group(doc_id, r)
                groups[doc_id] = group
                order.append(doc_id)
            else:
                self._merge_into(group, r, doc_id)

        doc_results = [self._build_result(doc_groups=groups[d], doc_id=d) for d in order]
        all_results = doc_results + regular
        all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return all_results

    # ── group construction / merge ──

    def _new_group(self, doc_id: str, r: dict) -> dict:
        """Start a group for ``doc_id`` from the first section/document result."""
        title, ts, source_path, total = self._doc_meta(doc_id)
        sections: list[dict] = []
        primary_text = ""
        if r.get("kind") == "document":
            # A graph-path doc hit: its already-materialized matched section is
            # the first matched section, and its body seeds the document text.
            primary_text = r.get("text", "")
            heading = r.get("matched_section", "")
            sections.append({
                "heading": heading, "text": primary_text,
                "score": r.get("score", 0.0),
                "entities": list(r.get("entities", [])),
                "topics": list(r.get("topics", [])),
            })
        else:  # section
            sections.append(self._section_summary(r))
        return {
            "document_id": doc_id,
            "title": title,
            "timestamp": ts,
            "source_path": source_path or r.get("source_path", ""),
            "total_sections": total,
            "sections": sections,
            "best_score": r.get("score", 0.0),
            "entities": set(r.get("entities", [])),
            "topics": set(r.get("topics", [])),
            "primary_text": primary_text,
        }

    def _merge_into(self, group: dict, r: dict, doc_id: str) -> None:
        """Fold a later section/document result for the same doc into the group."""
        if r.get("kind") == "document":
            # Prefer an existing materialized body; only seed if empty.
            if not group["primary_text"] and r.get("text"):
                group["primary_text"] = r["text"]
                group["sections"].insert(0, {
                    "heading": r.get("matched_section", ""), "text": r["text"],
                    "score": r.get("score", 0.0),
                    "entities": list(r.get("entities", [])),
                    "topics": list(r.get("topics", [])),
                })
        else:  # section
            group["sections"].append(self._section_summary(r))
        group["best_score"] = max(group["best_score"], r.get("score", 0.0))
        group["entities"].update(r.get("entities", []))
        group["topics"].update(r.get("topics", []))

    def _section_summary(self, r: dict) -> dict:
        """Project a section result down to the fields aggregation needs."""
        return {
            "heading": r.get("section_heading") or r.get("matched_section", "")
                       or "Untitled",
            "text": r.get("text", ""),
            "score": r.get("score", 0.0),
            "entities": list(r.get("entities", [])),
            "topics": list(r.get("topics", [])),
        }

    def _build_result(self, doc_groups: dict, doc_id: str) -> dict:
        """Build the final document result dict from an accumulated group."""
        g = doc_groups
        g["sections"].sort(key=lambda s: s["score"], reverse=True)
        matched = len(g["sections"])
        total = g["total_sections"]
        section_lines = [
            f"Section '{s['heading']}': {s['text']}" for s in g["sections"][:5]
        ]
        text = g["primary_text"] or self._build_document_context(g, total)
        return {
            "episode_id": doc_id,
            "kind": "document",
            "type": "document",
            "score": g["best_score"],
            "summary": (
                f"Document: {g['title']}\n"
                f"Relevant sections ({matched} matched):\n"
                + "\n".join(f"  - {s}" for s in section_lines)
            ),
            "text": text,
            "timestamp": g["timestamp"],
            "entities": sorted(g["entities"]),
            "topics": sorted(g["topics"]),
            "tones": [],
            "decisions": [],
            "session_id": None,
            "user_id": None,
            "follows": None,
            "source_path": g["source_path"],
            "matched_sections": matched,
            "total_sections": total,
            "sections": g["sections"],
        }

    def _build_document_context(self, group: dict, total_sections: int) -> str:
        """Fallback context text when no graph-path doc body seeded the group."""
        parts = [
            f"Document: {group['title']}",
            f"Relevant sections: {len(group['sections'])} of {total_sections}",
            "",
        ]
        for s in group["sections"][:5]:
            parts.append(f"--- {s['heading']} ---")
            parts.append(s["text"][:500])
            parts.append("")
        return "\n".join(parts)

    # ── store helpers ──

    def _doc_meta(self, doc_id: str) -> tuple[str, str, str, int]:
        """Return (title, timestamp, source_path, total_section_count).

        Uses ``get_document(load_bodies=False)`` -- metadata-only, no cold pull.
        Falls back to (doc_id, "", "", 0) when the doc no longer resolves (a
        deleted-mid-query race), so aggregation degrades rather than crashes.
        """
        doc = self.store.get_document(doc_id, load_bodies=False)
        if doc is None:
            return doc_id, "", "", 0
        return (
            doc.title or doc_id,
            getattr(doc, "ingested_at", "") or "",
            getattr(doc, "source_path", "") or "",
            len(doc.sections),
        )

    def _parent_via_child_of(self, section_id: str) -> Optional[str]:
        """Fallback parent-doc lookup for a section result lacking ``doc_id``.

        A document section has an outgoing ``child_of`` edge to its parent doc.
        Used only when a section result did not carry ``doc_id`` (forward-compat
        with results hydrated by code paths that predate the ``doc_id`` field);
        the shipped ``_hydrate_section`` always sets ``doc_id``, so this is
        defensive.
        """
        try:
            query = self.store.graph.query().vertex(section_id).out("child_of")
            result = query.execute_sync()
            try:
                vids = list(result.vertices)
            finally:
                result.close()
            return vids[0] if vids else None
        except Exception:
            return None


def store_has_documents(store: HippocampalStore) -> bool:
    """Cheap probe: does this store contain any document section edges?

    Returns True when a ``has_section`` POS scan yields at least one key --
    i.e. the corpus has at least one ingested document with sections. Used by
    ``runtime.build_ponder`` to decide whether to attach a ``DocumentRetriever``:
    conversation-only corpora (DialogSum/SAMSum, no ``has_section`` edges) skip
    aggregation entirely, so retrieval is byte-identical to the pre-1c path.
    """
    start = "memory/pos/has_section/"
    end = "memory/pos/has_section/\x7f"
    try:
        for _ in store.db.create_read_stream(start=start, end=end):
            return True
    except Exception:
        return False
    return False


__all__ = ["DocumentRetriever", "store_has_documents"]