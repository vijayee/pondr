"""Prompt compression for query planning (Phase 2c, Task 5).

The problem (docs/Phase 2c.md §7.1): the chat said "compress the prompt through
the SSM before Bonsai sees it" and "pass the compressed state + key entities to
Bonsai." But Bonsai (the query planner) consumes **text**, not a state vector —
the 2a backbone has no text-decoder head (it's a JEPA predictor in embedding
space). So compression here = produce a **text** prompt ≤ ``bonsai_max_input``
(2000 chars) that preserves the planning-relevant signal (entities + recent
focus), not a state vector.

The compressed prompt is:

  [WM preamble: active domains + recent topics from working_memory.metadata]
  [key spans extracted from the prompt: GLiNER if available, else a cheap
   titlecase/regex span extractor — the same one the end-state ``extract`` path
   uses conceptually]
  [a tail-truncated copy of the original prompt, hard-capped at
   bonsai_max_input chars]

The hard cap (2000 chars) prevents Bonsai context overflow. The planner is then
called with this compressed text. The SSM state is NOT fed to Bonsai; it
influences the *preamble* text, which is what Bonsai can read.

GLiNER is gated behind a flag (``use_gliner``); the cheap span extractor is the
default so the latency target holds and the module imports without gliner
installed.
"""

from __future__ import annotations

import re
from typing import Optional

from ..subconscious.working_memory import WorkingMemoryState


# Cheap span extraction (no model). Matches TitleCase sequences, ALL_CAPS
# acronyms, and quoted strings — a reasonable entity proxy without GLiNER.
_TITLECASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")
_QUOTED_RE = re.compile(r'"([^"]{1,60})"')


def _cheap_spans(text: str, max_spans: int = 12) -> list[str]:
    """Extract candidate entity spans with regex/titlecase (no model).

    Returns up to ``max_spans`` de-duplicated spans. This is a deliberate
    fallback for when GLiNER is unavailable — it catches proper nouns,
    acronyms, and quoted names, which is most of what planning needs.
    """
    spans: list[str] = []
    seen: set[str] = set()
    for rx in (_TITLECASE_RE, _ACRONYM_RE, _QUOTED_RE):
        for m in rx.finditer(text):
            s = m.group(1).strip()
            if s and s.lower() not in seen and len(s) <= 60:
                seen.add(s.lower())
                spans.append(s)
                if len(spans) >= max_spans:
                    return spans
    return spans


def _gliner_spans(text: str, max_spans: int = 12) -> list[str]:
    """Extract entity spans via GLiNER if available; else fall back to cheap.

    Lazy-imported so this module imports without gliner. GLiNER labels: the
    planning-relevant entity types (person, org, tech, event). On any import
    or runtime error, fall back to the cheap extractor (never raise — the
    planner must always get a prompt).
    """
    try:
        from src.encoding.gliner_extractor import get_gliner_extractor  # type: ignore
    except Exception:
        return _cheap_spans(text, max_spans)
    try:
        extractor = get_gliner_extractor()
        labels = ["person", "organization", "technology", "software", "event",
                  "topic", "concept"]
        hits = extractor.extract([text], labels=labels)  # type: ignore[attr-defined]
        spans: list[str] = []
        seen: set[str] = set()
        for h in (hits[0] if hits else []):
            s = (h.get("text") or "").strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                spans.append(s)
                if len(spans) >= max_spans:
                    return spans
        if not spans:
            return _cheap_spans(text, max_spans)
        return spans
    except Exception:
        return _cheap_spans(text, max_spans)


def _wm_preamble(working_memory: Optional[WorkingMemoryState]) -> str:
    """Text preamble from WM metadata — the only WM influence on the planner."""
    if working_memory is None or not working_memory.metadata:
        return ""
    meta = working_memory.metadata
    domains = meta.get("active_domains", [])
    focus = meta.get("last_query_type", "")
    lines: list[str] = []
    if domains:
        lines.append(f"Active domains: {', '.join(domains)}")
    if focus:
        lines.append(f"Recent focus: {focus}")
    return "\n".join(lines) + "\n\n" if lines else ""


def compress_prompt_for_planning(
    prompt: str,
    working_memory: Optional[WorkingMemoryState] = None,
    embedder=None,
    config=None,
    use_gliner: bool = False,
) -> str:
    """Compress ``prompt`` to a text string ≤ ``bonsai_max_input`` chars.

    Short prompts (≤ ``short_prompt_threshold``, 500 chars) pass through
    byte-identical. Longer prompts get a WM preamble + extracted key spans + a
    tail-truncated copy, hard-capped at ``bonsai_max_input`` (2000 chars).

    Returns text (never a state vector). The planner (Bonsai) consumes this.

    Args:
        prompt: the raw user query.
        working_memory: optional WM snapshot (its metadata → the preamble).
        embedder: unused by the text path (kept for API symmetry / future state
            path); may be None.
        config: a ``Phase2cConfig`` (or anything with ``.prompt_compression``).
            Defaults to the module defaults.
        use_gliner: if True, use GLiNER for span extraction (slower; gated so
            the default cheap path holds the latency target).
    """
    if config is not None:
        pc = config.prompt_compression
        short_threshold = pc.short_prompt_threshold
        max_input = pc.bonsai_max_input
    else:
        short_threshold = 500
        max_input = 2000

    if len(prompt) <= short_threshold:
        return prompt

    # 1. WM preamble (text from metadata — NOT the state tensor).
    preamble = _wm_preamble(working_memory)

    # 2. Key spans extracted from the prompt.
    spans = _gliner_spans(prompt) if use_gliner else _cheap_spans(prompt)
    spans_block = ""
    if spans:
        spans_block = "Key entities: " + ", ".join(spans) + "\n\n"

    # 3. Tail-truncated copy of the original prompt.
    # Reserve room for the preamble + spans; truncate the raw prompt to fit.
    overhead = len(preamble) + len(spans_block)
    raw_budget = max(0, max_input - overhead)
    # Keep the head of the prompt (entities usually appear early) + a tail
    # marker so the planner knows it was truncated.
    if len(prompt) <= raw_budget:
        truncated = prompt
    else:
        head = prompt[: max(0, raw_budget - 20)]
        truncated = head + "\n[...truncated]"

    compressed = preamble + spans_block + truncated

    # Hard cap — never hand Bonsai more than max_input chars.
    if len(compressed) > max_input:
        compressed = compressed[:max_input]
    return compressed