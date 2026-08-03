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

B3 (``config.reclaim_enabled``, default OFF): when ON, the PRIMARY band
replaces that break-on-overflow with a demotion cascade -- lowest-SCORE
episodes demote FULL -> SUMMARY -> DROP, then emergency body-truncation,
re-measuring after each step until the budget fits (mirrors A4's
``format_fade_block``). Under budget ON -> all items render at FULL unchanged
(byte-identical to OFF). Only over-budget ON changes behavior. See
``src/retrieval/reclaim.py``.

This module imports torch only for the type hint (the formatter consumes the
ChunkedContext which holds a WorkingMemoryState). The actual formatting is
text-only — no model call.
"""

from __future__ import annotations

from typing import Optional

from ..config import config
from ..subconscious.ssm_chunker import ChunkedContext
from ..subconscious.working_memory import WorkingMemoryState
from .reclaim import ReclaimItem, reclaim_to_budget


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


# B3 emergency body-truncation marker (ASCII; appended after the sliced body so
# a truncated chunk is still recognizable -- the LLM sees it kept its id +
# metadata, only the body is shortened).
_TRUNCATED_MARKER = " [... truncated ...]"


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
        if config.reclaim_enabled:
            # B3: replace break-on-overflow (drop the lowest-SCORE tail
            # entirely) with a demotion cascade on the lowest-SCORE primary
            # items: FULL -> SUMMARY (episodes only) -> DROP, then emergency
            # body-truncation, re-measuring after each step until the budget
            # fits (mirrors A4's ``format_fade_block`` cascade). The cascade
            # operates on the PRIMARY band only; its budget is ``max_tokens``
            # MINUS the prelude tokens already counted (apples-to-apples with
            # the off loop, which starts ``token_count`` at the prelude) so
            # under-budget ON returns all ``full_text`` unchanged -> byte-
            # identical to off, and over-budget ON is the only behavioral
            # delta. ``build_context_string`` (the search_memory renderer) is
            # deferred (it renders ``Summary:`` by default so mild is a no-op
            # there except under B2 ``verbatim``; ``reclaim_to_budget`` is
            # reusable for that follow-on).
            primary_budget = max_tokens - token_count
            if primary_budget > 0 and chunked.primary_chunks:
                items = [self._reclaim_item(ep) for ep in chunked.primary_chunks]
                kept = reclaim_to_budget(
                    items,
                    primary_budget,
                    headroom=config.reclaim_headroom,
                    min_keep=config.reclaim_min_keep,
                    truncate_chars=config.reclaim_truncate_chars,
                )
                primary_lines.extend(kept)
                token_count += sum(len(c) // 4 for c in kept)
        else:
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

    def _format_episode(
        self,
        ep: dict,
        *,
        summary_only: bool = False,
        truncate_chars: Optional[int] = None,
    ) -> str:
        """Render one PRIMARY-band item to text.

        ``summary_only`` (B3 mild rung, episodes only) omits the ``Full text:``
        line, keeping ``Summary:`` + the entities/topics/tones header -- a
        content-preserving demotion (the reverse of B2 ``verbatim``). Ignored
        for section/document/scene (no separate summary form). ``truncate_chars``
        (B3 emergency rung) slices the body to ``chars`` + a marker, reusing
        the SAME header/metadata so a truncated chunk is still recognizable.
        Both default off -> byte-identical to the pre-B3 render.
        """
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
                body = self._truncate_body(text, truncate_chars)
                if heading:
                    lines.append(f"Section '{heading}': {body}")
                else:
                    lines.append(f"Section: {body}")
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
                body = self._truncate_body(text, truncate_chars)
                if matched:
                    lines.append(f"Section '{matched}': {body}")
                else:
                    lines.append(f"Section: {body}")
            return "\n".join(lines)
        if kind == "scene":
            # Scene block (B1): the LLM-authored topic-level macro-memory. The
            # Markdown body (``text``) is the system's synthesized understanding
            # of one topic for a user; ``summary`` is the topic (the scene's
            # handle), ``heat`` is the scene-level forgetting signal. Scenes
            # ride the SAME ChunkedContext as episodes/docs/sections -- one
            # retrieval pipeline, NOT a separate [SCENE MEMORY] macro lane (the
            # fade-inject path owns macro lanes; scene blocks are retrieved).
            lines = [f"--- Scene {eid} (topic: {summary}, heat: "
                     f"{ep.get('heat', 0.0):.2f}) ---"]
            if topics:
                lines.append(f"Topics: {', '.join(topics)}")
            if text:
                lines.append(self._truncate_body(text, truncate_chars))
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
        if text and not summary_only:
            body = self._truncate_body(text, truncate_chars)
            lines.append(f"Full text: {body}")
        return "\n".join(lines)

    @staticmethod
    def _truncate_body(text: str, truncate_chars: Optional[int]) -> str:
        """B3 emergency body-truncation: slice ``text`` to ``truncate_chars`` +
        a marker. ``None`` or a body already shorter than the cap -> unchanged
        (byte-identical). The marker is appended only when slicing occurs so
        non-truncated renders are untouched.
        """
        if truncate_chars is None or len(text) <= truncate_chars:
            return text
        return text[:truncate_chars] + _TRUNCATED_MARKER

    def _render_truncated(self, ep: dict, chars: int) -> tuple[str, int]:
        """B3 emergency-truncation callback for a ``ReclaimItem``: re-render the
        item with its body sliced to ``chars`` (same header/metadata), return
        ``(text, tokens)``. Used by ``reclaim_to_budget`` only when the cascade
        reaches the emergency rung.
        """
        s = self._format_episode(ep, truncate_chars=chars)
        return s, len(s) // 4

    def _reclaim_item(self, ep: dict) -> ReclaimItem:
        """Build the ``ReclaimItem`` view of one primary chunk for the cascade.

        Episodes (no ``kind`` / kind not section/document/scene) carry a
        SUMMARY rung (``summary_only=True``); sections/docs/scenes do not
        (``summary_text=None`` -> mild skips them). The mild rung is offered
        only when it actually saves tokens (the episode has a ``text`` body to
        drop -> ``summary_tokens < full_tokens``); a textless episode has no
        shorter form and goes straight to aggressive DROP. ``score`` drives
        demote/drop order (lowest first); unscored items default to 0.0
        (demoted first -- the conservative "least confident" choice).
        """
        full = self._format_episode(ep)
        full_tokens = len(full) // 4
        kind = ep.get("kind")
        if kind in ("section", "document", "scene"):
            summary_text: Optional[str] = None
            summary_tokens: Optional[int] = None
        else:
            summary = self._format_episode(ep, summary_only=True)
            summary_tokens = len(summary) // 4
            if summary_tokens < full_tokens:
                summary_text = summary
            else:
                summary_text = None
                summary_tokens = None
        return ReclaimItem(
            score=float(ep.get("score", 0.0)),
            full_text=full,
            full_tokens=full_tokens,
            summary_text=summary_text,
            summary_tokens=summary_tokens,
            truncate=lambda chars, ep=ep: self._render_truncated(ep, chars),
        )