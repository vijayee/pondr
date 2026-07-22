"""Bonsai-based query planning: natural language → structured graph query.

The planner turns a free-text question into the ``query_plan`` dict that
``GraphTraversal.retrieve`` consumes (see ``graph_traversal.py`` for the plan
shape). It talks to the **local Bonsai llama-server** at ``config.bonsai_endpoint``
via its OpenAI-compatible ``/chat/completions`` API — NOT to OpenAI. The
endpoint is configurable and the connection is opened lazily, so the class is
constructible offline and the module imports without the server present.

A deterministic ``plan_rule_based`` fallback handles the common question shapes
(tone, entity, entity+topic, temporal chain, cross-entity intersection) without
any server. ``plan()`` uses the server when reachable and falls back to the
rule-based planner on any failure, so retrieval degrades gracefully offline —
and the offline test suite can exercise planning without Bonsai.
"""

from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import datetime

import requests

from ..config import config

# Prompt body from docs/Phase 1b.md §6. The {prompt} slot is filled per query.
BONSAI_QUERY_PROMPT = """Convert this question into a structured memory query.
Return ONLY valid JSON, no other text.

RECENT CONVERSATION (use it to resolve pronouns and implicit references):
{conversation_context}

CURRENT QUESTION: {prompt}

The memory graph stores episodes with these attributes:
- entities: [Person, Project, Technology, Concept]
- topics: [database_design, configuration, graph_database, performance,
           decision_making, ai_architecture, api_design, security]
- tones: [frustrated, excited, curious, neutral]
- decisions: specific choices made (e.g., "use_hbtrie", "add_optimizer")
- temporal: episodes linked by "follows" edges

Query parameters:
- entities: list of entities to search for
- topics: list of topics to filter by
- tones: list of emotional tones to filter by
- entity_mode: "intersection" (episodes containing ALL entities) or
               "union" (episodes containing ANY entity)
- temporal_after: if the question asks "what happened after X", the
                  keyword to find the anchor episode, or null
- temporal_before: if the question asks "what led up to X", the keyword,
                   or null
- temporal_filter: "today", "this_week", "last_week", "this_month", or null
- date_from: ISO date for the start of an ABSOLUTE range (e.g., "2025-06-01"), or null
- date_to: ISO date for the end of an ABSOLUTE range (e.g., "2025-06-30"), or null
- limit: max episodes to return (default 5)

ABSOLUTE vs RELATIVE time:
- "What happened in June 2025?" -> date_from="2025-06-01", date_to="2025-06-30"
- "What did we discuss last week?" -> temporal_filter="last_week"
- "What happened between March and May?" -> date_from="2025-03-01", date_to="2025-05-31"
- Do NOT set both date_from/date_to and temporal_filter in the same query.

IMPORTANT RULES:
- "What was I frustrated about?" → tones=["frustrated"], entity_mode="union"
- "What did Alice and I decide?" → entities=["Alice"], entity_mode="union"
  (NOT intersection — "Alice and I" means episodes involving Alice)
- "What did Alice say about databases?" → entities=["Alice"],
  topics=["database_design"], entity_mode="union"
- "What happened after we implemented morphisms?" → temporal_after="morphism"
- "Why did we choose X over Y?" → topics=["decision_making"],
  entities=["X", "Y"], entity_mode="union"
- If the question is about a specific person's opinion, entity_mode is
  "union" (episodes involving that person)
- If the question is about when two specific things were discussed
  TOGETHER, entity_mode is "intersection"

PRONOUN / IMPLICIT-REFERENCE RESOLUTION (use the RECENT CONVERSATION):
- "he" / "she" → the person mentioned in recent context (as an entity).
- "it" / "that" → the topic/entity most recently discussed.
- "we discussed" / "we decided" → the people in the conversation as entities.
- If the current question has no extractable entity/topic but recent context
  makes the referent clear, pull entities/topics from the recent context.

Return ONLY valid JSON:
{{"entities": [], "topics": [], "tones": [], "entity_mode": "union",
  "temporal_after": null, "temporal_before": null,
  "temporal_filter": null, "date_from": null, "date_to": null,
  "limit": 5}}"""


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _default_plan() -> dict:
    """The canonical empty plan (matches the prompt's return shape)."""
    return {
        "entities": [],
        "topics": [],
        "tones": [],
        "entity_mode": "union",
        "temporal_after": None,
        "temporal_before": None,
        "temporal_filter": None,
        "date_from": None,
        "date_to": None,
        "limit": config.default_retrieval_limit,
    }


# ── rule-based planner (no server) ──

# Capitalized tokens that are NOT entities (question words, pronouns, sentence
# starters). Matched case-insensitively against Capitalized tokens.
_ENTITY_STOPLIST = {
    "what", "who", "where", "when", "why", "how", "which",
    "did", "was", "were", "is", "are", "do", "does", "can", "could",
    "the", "a", "an", "i", "we", "they", "he", "she", "it",
    "let", "now", "about", "after", "before", "and", "or", "with",
    "on", "in", "to", "of", "for", "did", "happened", "say", "said",
}

# Words after "after"/"before" that are verbs/pronouns, not the anchor noun.
_TEMPORAL_SKIP = {
    "we", "i", "they", "he", "she", "it",
    "implemented", "did", "started", "finished", "built", "added",
    "deployed", "fixed", "wrote", "the", "a", "an", "had", "have",
}

# Joint predicates that, with 2+ entities, mean intersection (discussed TOGETHER).
_JOINT_PREDICATES = {
    "disagree", "disagreed", "disagreement", "agree", "agreed",
    "discuss", "discussed", "discussion", "debate", "debated",
    "both", "together", "versus", "vs", "compare", "compared", "comparison",
}

# Tone keyword → canonical tone label.
_TONE_MAP = {
    "frustrated": "frustrated", "frustration": "frustrated", "frustrating": "frustrated",
    "angry": "frustrated", "annoyed": "frustrated",
    "excited": "excited", "exciting": "excited", "enthusiastic": "excited",
    "curious": "curious", "wondered": "curious", "wondering": "curious",
    "neutral": "neutral",
}

# Topic keyword → canonical topic label (subset of the planner prompt's list).
_TOPIC_MAP = {
    "database": "database_design", "db": "database_design", "postgres": "database_design",
    "hbtrie": "database_design", "trie": "database_design",
    "decide": "decision_making", "decision": "decision_making", "choose": "decision_making",
    "chose": "decision_making", "choosing": "decision_making",
    "performance": "performance", "slow": "performance", "fast": "performance",
    "latency": "performance", "optimize": "performance", "optimization": "performance",
    "security": "security", "encrypt": "security", "encryption": "security",
    "aes": "security", "key": "security",
    "api": "api_design", "endpoint": "api_design", "rest": "api_design",
    "graph": "graph_database", "traversal": "graph_database", "gremlin": "graph_database",
    "config": "configuration", "configuration": "configuration", "wal": "configuration",
    "ai": "ai_architecture", "model": "ai_architecture", "neural": "ai_architecture",
    "transformer": "ai_architecture", "llm": "ai_architecture",
}


def _extract_entities(prompt: str) -> list[str]:
    """Capitalized tokens that aren't question words / pronouns → entities."""
    out: list[str] = []
    for tok in re.findall(r"\b[A-Z][a-z]+\b", prompt):
        if tok.lower() not in _ENTITY_STOPLIST and tok not in out:
            out.append(tok)
    return out


def _extract_temporal_anchor(prompt: str, anchor_word: str) -> str | None:
    """Extract the content noun after ``anchor_word`` (``after``/``before``)."""
    m = re.search(rf"\b{anchor_word}\s+(.+)", prompt.lower())
    if not m:
        return None
    words = re.findall(r"[a-z_]+", m.group(1))
    content = [w for w in words if w not in _TEMPORAL_SKIP]
    return content[0] if content else None


# ── absolute date-range extraction (Phase 1c) ──

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_PAT = r"(" + "|".join(_MONTHS) + r")"


def _month_range(month: int, year: int) -> tuple[str, str]:
    """First and last day of ``month``/``year`` as ISO date strings."""
    last = monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"


def _extract_date_range(prompt: str) -> tuple[str | None, str | None]:
    """Parse an explicit absolute date range from the prompt.

    Recognizes ``"in <Month> <Year>"`` (single-month range) and
    ``"between <Month> [<Year>] and <Month> [<Year>]"`` (multi-month range).
    A missing year defaults to the current year. Returns ``(date_from, date_to)``
    as ISO date strings; either may be ``None`` when no explicit range is found.
    Used by the rule-based planner; the Bonsai server path parses dates itself.
    """
    lower = prompt.lower()
    # "in June 2025" / "in June"
    m = re.search(rf"\bin {_MONTH_PAT}\b(?:\s+(\d{{4}}))?", lower)
    if m:
        month = _MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else datetime.now().year
        return _month_range(month, year)
    # "between March and May" / "between March 2025 and May 2025"
    m = re.search(
        rf"\bbetween {_MONTH_PAT}(?:\s+(\d{{4}}))?\s+and\s+{_MONTH_PAT}(?:\s+(\d{{4}}))?",
        lower,
    )
    if m:
        m1, y1, m2, y2 = m.group(1), m.group(2), m.group(3), m.group(4)
        now_year = datetime.now().year
        y1 = int(y1) if y1 else (int(y2) if y2 else now_year)
        y2 = int(y2) if y2 else (int(y1) if y1 else now_year)
        start, _ = _month_range(_MONTHS[m1], y1)
        _, end = _month_range(_MONTHS[m2], y2)
        return start, end
    return None, None


# Pronouns / implicit references that trigger context-based resolution in the
# rule-based planner. "I" is excluded (first person); "we" is included because
# "we discussed/decided" should pull the conversation's people as entities.
_PRONOUN_RE = re.compile(r"\b(he|she|it|that|we|they)\b")


def _resolve_from_context(
    conversation_history: list[dict] | None,
) -> tuple[list[str], list[str]]:
    """Extract ``(entities, topics)`` from recent conversation turns.

    Used by the rule-based planner to resolve pronouns / implicit references
    when the current prompt has no extractable entity/topic of its own. Reuses
    ``_extract_entities`` (Capitalized non-stoplist tokens) and the ``_TOPIC_MAP``
    keyword scan over the last ~6 messages.
    """
    if not conversation_history:
        return [], []
    text = " ".join(
        m.get("content", "")
        for m in conversation_history[-6:]
        if isinstance(m, dict)
    )
    entities = _extract_entities(text)
    lower = text.lower()
    topics: list[str] = []
    for word, topic in _TOPIC_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", lower) and topic not in topics:
            topics.append(topic)
    return entities, topics


class BonsaiQueryPlanner:
    """Converts natural-language questions into structured query parameters.

    Uses the local Bonsai server when reachable; falls back to a deterministic
    rule-based planner otherwise (and via ``plan_rule_based`` directly for tests).
    """

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        temperature: float | None = None,
        timeout: float = 30.0,
        force_rule_based: bool = False,
    ) -> None:
        self.model = model or config.bonsai_model
        self.endpoint = (endpoint or config.bonsai_endpoint).rstrip("/")
        self.temperature = temperature if temperature is not None else config.bonsai_temperature
        self.timeout = timeout
        # When True, ``plan`` skips the Bonsai server entirely and goes straight
        # to ``plan_rule_based``. Use this for offline trace generation and
        # acceptance probes that must be deterministic and must NOT depend on an
        # external LLM server being up (a flapping server both makes traces
        # non-reproducible AND can hang ``requests.post`` past its timeout when a
        # large conversation_history payload meets a half-alive endpoint that
        # accepts the connection but never drains it). Default False preserves
        # the server-first-then-fallback behavior every production caller relies
        # on. NOTE: ``endpoint=None`` does NOT disable the server -- it resolves
        # to ``config.bonsai_endpoint``; pass ``force_rule_based=True`` for a
        # truly offline planner.
        self.force_rule_based = force_rule_based

    # ── public API ──

    def plan(self, prompt: str, conversation_history: list[dict] | None = None) -> dict:
        """Plan a query, preferring the Bonsai server and falling back to rules.

        ``conversation_history`` (last few turns) is threaded to both paths so
        pronouns / implicit references in ``prompt`` can be resolved against
        recent context. Optional and backward-compatible (``None`` = plan from
        the prompt alone, the Phase 1b behavior).

        Any server-side failure (connection, non-200, parse) is swallowed and
        the rule-based plan is returned, so retrieval still works offline. Use
        ``plan_via_server`` to surface server errors verbatim (for live tests).
        With ``force_rule_based=True`` the server is never contacted.
        """
        if self.force_rule_based:
            return self.plan_rule_based(prompt, conversation_history)
        try:
            return self.plan_via_server(prompt, conversation_history)
        except RuntimeError:
            # plan_via_server wraps every server-side failure (connection,
            # non-200, parse) in RuntimeError; fall back to the rule-based
            # planner so retrieval still works offline. Unexpected code errors
            # are NOT RuntimeError and propagate rather than being masked.
            return self.plan_rule_based(prompt, conversation_history)

    def plan_via_server(
        self,
        prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """Plan via the Bonsai server; raise on any failure (verbatim errors)."""
        url = f"{self.endpoint}/chat/completions"
        content = BONSAI_QUERY_PROMPT.format(
            prompt=prompt,
            conversation_context=self._format_context(conversation_history),
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Bonsai request to {url} failed: {e}") from e
        if resp.status_code != 200:
            raise RuntimeError(
                f"Bonsai endpoint {url} returned HTTP {resp.status_code}: {resp.text}"
            )
        try:
            outer = resp.json()
            content = outer["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Bonsai response missing choices[0].message.content: {outer}") from e
        return self._parse_plan(content)

    def plan_rule_based(
        self,
        prompt: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """Deterministic, server-free planner for common question shapes.

        When ``conversation_history`` is supplied and ``prompt`` contains a
        pronoun / implicit reference (he/she/it/that/we/they), entities and
        topics are pulled from the recent context to resolve the referent — the
        rule-based analog of what Bonsai does in the server path via the prompt.
        """
        plan = _default_plan()
        lower = prompt.lower()

        # Temporal chain anchors take precedence — they re-anchor the candidate
        # set to a follows-chain, so axis filters are secondary.
        if "after" in lower:
            anchor = _extract_temporal_anchor(prompt, "after")
            if anchor:
                plan["temporal_after"] = anchor
        if "before" in lower or "led up to" in lower or "led to" in lower:
            anchor = (
                _extract_temporal_anchor(prompt, "before")
                or _extract_temporal_anchor(prompt, "to")
            )
            if anchor:
                plan["temporal_before"] = anchor

        # Tones.
        for word, tone in _TONE_MAP.items():
            if re.search(rf"\b{re.escape(word)}\b", lower) and tone not in plan["tones"]:
                plan["tones"].append(tone)

        # Entities.
        entities = _extract_entities(prompt)
        plan["entities"] = entities

        # entity_mode: 2+ entities + a joint predicate → intersection; else union.
        if len(entities) >= 2 and any(p in lower for p in _JOINT_PREDICATES):
            plan["entity_mode"] = "intersection"
        else:
            plan["entity_mode"] = "union"

        # Topics.
        for word, topic in _TOPIC_MAP.items():
            if re.search(rf"\b{re.escape(word)}\b", lower) and topic not in plan["topics"]:
                plan["topics"].append(topic)

        # Pronoun / implicit-reference resolution from conversation context
        # (Phase 1c). Only when the prompt actually contains a pronoun AND
        # context is supplied — otherwise leave the prompt-derived plan alone.
        if conversation_history and _PRONOUN_RE.search(lower):
            ctx_entities, ctx_topics = _resolve_from_context(conversation_history)
            if not entities and ctx_entities:
                # Prompt had no extractable entity; fill from context and
                # re-evaluate entity_mode with the resolved entities.
                entities = ctx_entities
                plan["entities"] = entities
                if len(entities) >= 2 and any(p in lower for p in _JOINT_PREDICATES):
                    plan["entity_mode"] = "intersection"
                else:
                    plan["entity_mode"] = "union"
            for t in ctx_topics:
                if t not in plan["topics"]:
                    plan["topics"].append(t)

        # Absolute date range (Phase 1c) — takes precedence over the relative
        # bucket filter; the two are mutually exclusive (see BONSAI_QUERY_PROMPT).
        date_from, date_to = _extract_date_range(prompt)
        if date_from or date_to:
            plan["date_from"] = date_from
            plan["date_to"] = date_to
        else:
            # Relative temporal bucket filter.
            if "today" in lower:
                plan["temporal_filter"] = "today"
            elif "last week" in lower:
                plan["temporal_filter"] = "last_week"
            elif "this week" in lower:
                plan["temporal_filter"] = "this_week"
            elif "this month" in lower:
                plan["temporal_filter"] = "this_month"

        return plan

    # ── helpers ──

    @staticmethod
    def _format_context(conversation_history: list[dict] | None) -> str:
        """Format recent turns for the BONSAI_QUERY_PROMPT context slot.

        Last ~6 messages (≈3 exchanges), ``"role: content"`` per line. Returns
        ``"(no prior context)"`` when no history is supplied so the prompt slot
        is always filled.
        """
        if not conversation_history:
            return "(no prior context)"
        recent = conversation_history[-6:]
        # Defensive .get — a malformed history message shouldn't crash the
        # server path (and force a rule-based fallback); skip empty turns.
        lines = [
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent
            if isinstance(m, dict) and (m.get("content") or m.get("role"))
        ]
        return "\n".join(lines) if lines else "(no prior context)"

    @staticmethod
    def _parse_plan(content: str) -> dict:
        """Parse the model's JSON content into a normalized plan dict.

        Strips accidental ``` fences and falls back to the outermost ``{...}``
        span. Coerces missing fields to the canonical defaults so a partial model
        response still yields a usable plan.
        """
        body = content.strip()
        fence = _FENCE_RE.match(body)
        if fence:
            body = fence.group(1).strip()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            start, end = body.find("{"), body.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(body[start : end + 1])
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Bonsai returned unparseable JSON: {content!r}") from e
            else:
                raise RuntimeError(f"Bonsai returned unparseable JSON: {content!r}") from None

        plan = _default_plan()
        if isinstance(data, dict):
            for k in plan:
                if k in data and data[k] is not None:
                    plan[k] = data[k]
        return plan