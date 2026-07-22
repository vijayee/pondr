"""ChunkedContextFormatter: render a ChunkedContext into the LLM context string.

Phase 2c. The generation model receives:

- **Primary**: the full text of the most-relevant episodes (the detail).
- **Compressed summary**: the union of topics from the compressed (gist)
  episodes — NOT the raw SSM state vector (Bonsai consumes text, not state).
- **Working-memory state**: the active domains / recent focus from the WM
  metadata (a textual preamble, not a tensor).
- **EXPAND instructions**: how to request the full text of a compressed episode.

The hard cap at ``max_context_tokens`` (len(text)//4 estimate) matches
``HippocampalRetriever.build_context_string``; episodes beyond the cap are
dropped, not truncated, so a half-episode never enters context.

This module imports torch only for the type hint (the formatter consumes the
ChunkedContext which holds a WorkingMemoryState). The actual formatting is
text-only — no model call.
"""

from __future__ import annotations

from typing import Optional

from ..subconscious.ssm_chunker import ChunkedContext
from ..subconscious.working_memory import WorkingMemoryState


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


class ChunkedContextFormatter:
    """Format a ``ChunkedContext`` into a text string for the generation model.

    Only the ``bonsai`` consumer is live; the ``consumer`` parameter is kept for
    forward-compatibility (a different consumer would get a different layout;
    for now all consumers get the structured format below).
    """

    def format_for_llm(
        self,
        chunked: ChunkedContext,
        consumer: str = "bonsai",
        working_memory: Optional[WorkingMemoryState] = None,
        max_tokens: int = 4000,
    ) -> str:
        """Produce the context string. Text-only (no LLM call here).

        Sections: [RETRIEVED CONTEXT — PRIMARY] (full text + metadata),
        [COMPRESSED CONTEXT — SUMMARY] (topic union from secondary episodes),
        [WORKING MEMORY STATE] (active domains / recent focus from WM metadata),
        and EXPAND instructions.
        """
        parts: list[str] = [
            "You have access to relevant past conversations.",
            "Primary episodes are shown in full; secondary episodes are compressed",
            "(only their topics are listed). Use EXPAND(episode_id) to retrieve the",
            "full text of any compressed episode if you need detail.",
            "",
        ]
        token_count = len("\n".join(parts)) // 4

        # ── PRIMARY (full text) ──
        primary_lines: list[str] = ["[RETRIEVED CONTEXT — PRIMARY]"]
        for ep in chunked.primary_chunks:
            chunk = self._format_episode(ep)
            chunk_tokens = len(chunk) // 4
            if token_count + chunk_tokens > max_tokens:
                break  # drop, don't truncate
            primary_lines.append(chunk)
            token_count += chunk_tokens
        parts.append("\n".join(primary_lines))

        # ── COMPRESSED (topic union from secondary episodes — NOT the state vector) ──
        if chunked.has_compressed:
            topics = sorted({
                t for ep in chunked.secondary_episodes
                for t in ep.get("topics", []) if t
            })
            comp_lines = [
                "[COMPRESSED CONTEXT — SUMMARY]",
                "The following topics are available in compressed form. If you need",
                "specific details, use EXPAND(episode_id) to retrieve full text.",
                f"Compressed topics: {', '.join(topics) if topics else '(none extracted)'}",
                f"Expandable episode ids: {', '.join(sorted(chunked.expandable_ids))}",
            ]
            parts.append("\n".join(comp_lines))

        # ── WORKING MEMORY STATE (text preamble from WM metadata) ──
        if working_memory is not None and working_memory.metadata:
            meta = working_memory.metadata
            focus = meta.get("last_query_type", "(none)")
            domains = meta.get("active_domains", [])
            wm_lines = [
                "[WORKING MEMORY STATE]",
                f"Current conversation focus: {focus}",
                f"Active domains: {', '.join(domains) if domains else '(none)'}",
            ]
            parts.append("\n".join(wm_lines))

        return "\n\n".join(parts)

    def _format_episode(self, ep: dict) -> str:
        eid = ep.get("episode_id", "")
        ts = ep.get("timestamp", "")
        entities = ep.get("entities", [])
        topics = ep.get("topics", [])
        tones = ep.get("tones", [])
        summary = ep.get("summary", "")
        text = ep.get("text", "")
        kind = ep.get("kind")
        if kind == "section":
            # Section (per-chunk) result: the chunk body is in ``text``
            # (materialized at hydrate); the renderer needs no store/cold pull.
            lines = [f"--- Section {eid} ({ts}) ---"]
            src = ep.get("source_path", "")
            if src:
                lines.append(f"Source: {src}")
            if summary:
                lines.append(f"Title: {summary}")
            # STRM 1f-6: surface the LLM prose description as a one-line handle
            # so the LLM gets BOTH a meaning-level description AND the full code
            # body below (serves "reasoning over recalled code"). Additive: only
            # when ``embed_text`` is non-empty (code docs ingested with a
            # summarizer); absent -> byte-identical to pre-1f-6.
            embed_text = ep.get("embed_text", "")
            if embed_text:
                lines.append(f"Description: {embed_text}")
            if entities:
                lines.append(f"Entities: {', '.join(entities)}")
            if topics:
                lines.append(f"Topics: {', '.join(topics)}")
            heading = ep.get("section_heading", "")
            if text:
                if heading:
                    lines.append(f"Section '{heading}': {text}")
                else:
                    lines.append(f"Section: {text}")
            return "\n".join(lines)
        if kind == "document":
            # Document result (graph-path hit): the matched section body is in
            # ``text`` (already materialized at hydrate), so the renderer needs
            # no store/cold pull.
            lines = [f"--- Document {eid} ({ts}) ---"]
            src = ep.get("source_path", "")
            if src:
                lines.append(f"Source: {src}")
            if summary:
                lines.append(f"Title: {summary}")
            # STRM 1f-6: surface the LLM prose description as a one-line handle
            # (see the section branch above for rationale). Additive: only when
            # ``embed_text`` is non-empty; absent -> byte-identical to pre-1f-6.
            embed_text = ep.get("embed_text", "")
            if embed_text:
                lines.append(f"Description: {embed_text}")
            if entities:
                lines.append(f"Entities: {', '.join(entities)}")
            if topics:
                lines.append(f"Topics: {', '.join(topics)}")
            matched = ep.get("matched_section", "")
            if text:
                if matched:
                    lines.append(f"Section '{matched}': {text}")
                else:
                    lines.append(f"Section: {text}")
            return "\n".join(lines)
        lines = [f"--- Episode {eid} ({ts}) ---"]
        if entities:
            lines.append(f"Entities: {', '.join(entities)}")
        if topics:
            lines.append(f"Topics: {', '.join(topics)}")
        if tones:
            lines.append(f"Tone: {', '.join(tones)}")
        if summary:
            lines.append(f"Summary: {summary}")
        if text:
            lines.append(f"Full text: {text}")
        return "\n".join(lines)