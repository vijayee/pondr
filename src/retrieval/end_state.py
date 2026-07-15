"""End-state dispatch: direct / format / synthesize / extract (Phase 2c, axis b).

What to DO with the retrieved results (chat [144]; docs/Ponder Engine Chat
Facts.md §3.2). Not every retrieval ends in an LLM call:

- ``direct``    — the results ARE the answer. No LLM. Return episodes/summaries.
- ``format``    — results are context for another consumer. No LLM. Return a
                  formatted context string/structure for the named consumer.
- ``extract``   — results → structured data. No synthesis LLM. Use a cheap
                  local span/field extractor (the episode dicts already carry
                  ``entities``/``topics``/``tones``/``decisions`` from the graph
                  traversal — pull those, no model).
- ``synthesize`` — LLM reasons across episodes. Calls the generation model
                  (Bonsai). The only end state that invokes an LLM.

The dispatch is a pure function of the ``EndStatePlan`` + the retrieved context.
The ``synthesize`` path is delegated to a caller-supplied callable so this
module stays free of the Bonsai/requests dependency (the orchestrator passes
``mode_a._complete`` or a stub).
"""

from __future__ import annotations

from typing import Callable, Optional

from ..subconscious.presentation_gate import (
    END_DIRECT, END_EXTRACT, END_FORMAT, END_SYNTHESIZE, EndStatePlan,
)
from ..subconscious.ssm_chunker import ChunkedContext
from ..subconscious.working_memory import WorkingMemoryState
from .chunked_context import ChunkedContextFormatter


# A synthesize callable: (context_string, conversation_history) -> response str.
SynthesizeFn = Callable[[str, Optional[list[dict]]], str]


def dispatch_end_state(
    plan: EndStatePlan,
    chunked: ChunkedContext,
    formatter: ChunkedContextFormatter,
    episodes: list[dict],
    query: str,
    working_memory: Optional[WorkingMemoryState] = None,
    consumer: str = "bonsai",
    synthesize: Optional[SynthesizeFn] = None,
    conversation_history: Optional[list[dict]] = None,
    max_context_tokens: int = 4000,
) -> dict:
    """Dispatch on ``plan.end_state`` and return the result dict.

    For ``direct``/``format``/``extract``: returns WITHOUT an LLM call (the
    "database you can talk to" behavior). For ``synthesize``: calls the
    ``synthesize`` callable (the generation model); raises if it is None.
    """
    if plan.end_state == END_DIRECT:
        return _dispatch_direct(plan, episodes, query)
    if plan.end_state == END_FORMAT:
        return _dispatch_format(plan, chunked, formatter, working_memory, consumer,
                                max_context_tokens, query)
    if plan.end_state == END_EXTRACT:
        return _dispatch_extract(plan, episodes, query)
    if plan.end_state == END_SYNTHESIZE:
        return _dispatch_synthesize(plan, chunked, formatter, working_memory, consumer,
                                    synthesize, conversation_history, max_context_tokens,
                                    query)
    raise ValueError(f"unknown end_state: {plan.end_state!r}")


def _dispatch_direct(plan: EndStatePlan, episodes: list[dict], query: str) -> dict:
    """Results are the answer — return episodes/summaries as-is. No LLM."""
    return {
        "type": END_DIRECT,
        "query": query,
        "episodes": [
            {
                "episode_id": e.get("episode_id"),
                "timestamp": e.get("timestamp", ""),
                "summary": e.get("summary", ""),
                "text": e.get("text", ""),
                "entities": e.get("entities", []),
                "topics": e.get("topics", []),
                "tones": e.get("tones", []),
                "score": e.get("score", 0.0),
            }
            for e in episodes
        ],
        "end_state_plan": plan,
        "supported": True,
    }


def _dispatch_format(
    plan: EndStatePlan,
    chunked: ChunkedContext,
    formatter: ChunkedContextFormatter,
    working_memory: Optional[WorkingMemoryState],
    consumer: str,
    max_context_tokens: int,
    query: str,
) -> dict:
    """Results are context for another consumer — return formatted context. No LLM."""
    context = formatter.format_for_llm(
        chunked, consumer=consumer, working_memory=working_memory,
        max_tokens=max_context_tokens,
    )
    return {
        "type": END_FORMAT,
        "query": query,
        "context": context,
        "format_spec": plan.format_spec,
        "consumer": plan.format_spec.get("consumer", consumer) if plan.format_spec else consumer,
        "end_state_plan": plan,
        "supported": True,
    }


def _dispatch_extract(plan: EndStatePlan, episodes: list[dict], query: str) -> dict:
    """Results → structured data via a cheap local extractor. No synthesis LLM.

    The episode dicts already carry structured fields from the graph traversal
    (``entities``/``topics``/``tones``/``decisions``); the extractor pulls those
    per the schema, no model call. ``schema`` shape:
    ``{"type": "list"|"graph"|"table", "item_type": "decision"|"entity"|"topic"|...}``.
    """
    schema = plan.extract_schema or {}
    etype = (schema.get("type") or "list").lower()
    item_type = (schema.get("item_type") or "entity").lower()

    if item_type == "decision":
        items = [d for e in episodes for d in e.get("decisions", []) if d]
    elif item_type == "entity":
        seen: set[str] = set()
        items = []
        for e in episodes:
            for ent in e.get("entities", []):
                if ent and ent not in seen:
                    seen.add(ent)
                    items.append(ent)
    elif item_type == "topic":
        seen_t: set[str] = set()
        items = []
        for e in episodes:
            for t in e.get("topics", []):
                if t and t not in seen_t:
                    seen_t.add(t)
                    items.append(t)
    else:
        # Fallback: pull the named field from each episode (best-effort).
        items = []
        for e in episodes:
            val = e.get(item_type, [])
            if isinstance(val, list):
                items.extend(val)
            elif val:
                items.append(val)

    if etype == "graph":
        data = _build_graph(episodes)
    elif etype == "table":
        data = [
            {
                "episode_id": e.get("episode_id"),
                "timestamp": e.get("timestamp", ""),
                item_type: [x for x in (e.get(item_type, []) if isinstance(e.get(item_type), list)
                                       else [e.get(item_type)] if e.get(item_type) else [])],
            }
            for e in episodes
        ]
    else:  # "list"
        data = items

    return {
        "type": END_EXTRACT,
        "query": query,
        "schema": schema,
        "data": data,
        "end_state_plan": plan,
        "supported": True,
    }


def _build_graph(episodes: list[dict]) -> dict:
    """Build a simple entity→episode adjacency from the episode dicts."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_entities: set[str] = set()
    for e in episodes:
        eid = e.get("episode_id")
        if eid:
            # ``kind`` comes from the hydrated result dict (``"section"`` for a
            # per-chunk vector hit, ``"document"`` for a graph-path doc hit);
            # episode dicts carry no ``kind`` key, so they fall through to
            # ``"episode"``. ``rel`` stays ``appears_in`` for UI symmetry across
            # the three result kinds. (Section ids start with ``doc_``, so the
            # ``kind`` field -- not the prefix -- is the discriminator.)
            kind = e.get("kind") or "episode"
            nodes.append({"id": eid, "kind": kind, "timestamp": e.get("timestamp", "")})
        for ent in e.get("entities", []):
            if ent and ent not in seen_entities:
                seen_entities.add(ent)
                nodes.append({"id": ent, "kind": "entity"})
            if ent and eid:
                edges.append({"src": ent, "dst": eid, "rel": "appears_in"})
    return {"nodes": nodes, "edges": edges}


def _dispatch_synthesize(
    plan: EndStatePlan,
    chunked: ChunkedContext,
    formatter: ChunkedContextFormatter,
    working_memory: Optional[WorkingMemoryState],
    consumer: str,
    synthesize: Optional[SynthesizeFn],
    conversation_history: Optional[list[dict]],
    max_context_tokens: int,
    query: str,
) -> dict:
    """LLM reasons across episodes. The ONLY end state that calls the LLM."""
    if synthesize is None:
        raise RuntimeError(
            "synthesize end_state requires a synthesize callable "
            "(the orchestrator passes mode_a._complete or a stub)"
        )
    context = formatter.format_for_llm(
        chunked, consumer=consumer, working_memory=working_memory,
        max_tokens=max_context_tokens,
    )
    response = synthesize(context, conversation_history)
    return {
        "type": END_SYNTHESIZE,
        "query": query,
        "response": response,
        "context": context,
        "model_size": plan.model_size,
        "end_state_plan": plan,
        "supported": True,
    }