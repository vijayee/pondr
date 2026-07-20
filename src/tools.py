"""Ponder Engine tool surface -- the subconscious capabilities the conscious
consumer LLM invokes during generation.

The Ponder Engine is an *artificial subconscious*: it auto-retrieves memory on
the shared prompt (the existing ``HippocampalRetriever`` + ``build_context_string``
path) AND exposes a small set of OpenAI-style tools the conscious consumer LLM
calls to (a) report which retrieved units were useful -- the feedback signal
that drives per-unit salience -- and (b) manipulate how memory is retrieved
mid-generation (expand a compressed chunk, re-search with a refined query).
Two consumers share this ONE interface:

* The **external LLM** (the canonical conscious model) -- the host runs the
  OpenAI tool-calling protocol with ``TOOL_SCHEMAS`` and dispatches each tool
  call through ``dispatch_tool`` (or the optional ``run_tool_loop`` helper).
* **Ponder's own Bonsai self-chat** (``PonderOrchestrator.query``) -- emits
  ``record_feedback`` as a Bonsai tool call during synthesis, with a structured
  fallback when Bonsai tool-calling is unsupported (see ``orchestrator.py``).

``dispatch_tool`` is the single seam a consumer host calls: it routes by name
to the method that owns the capability (``record_feedback`` on the store -- it
owns the boost store; ``expand_unit`` / ``search_memory`` on the orchestrator
-- it owns the retriever + expand handler). It is best-effort and NEVER raises:
an unknown tool or malformed args returns a short error string so a tool
failure can't break the consumer's agent loop.

The schemas use only ASCII descriptions (cp1252-safe) and JSON-Schema arg types
matching the OpenAI tool-definition shape.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

# ── OpenAI-style tool definitions ──
# Each entry is ``{"type": "function", "function": {"name", "description",
# "parameters": <JSON schema>}}`` -- the shape ``/chat/completions`` ``tools``
# takes. ASCII-only descriptions (cp1252-safe).

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "record_feedback",
            "description": (
                "Report how useful each retrieved memory unit was for answering "
                "the user's question, on a 1-5 scale (1=useless, 3=neutral, "
                "5=essential). Each rating adjusts that unit's retrieval salience "
                "for future queries (useful units resurface, useless ones fade). "
                "Call this ONCE after forming your answer with the units you "
                "actually used. unit_id is the [id | ...] header from the context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "judgments": {
                        "type": "array",
                        "description": "One {unit_id, rating} per context unit you judged.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "unit_id": {"type": "string"},
                                "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                                "slot_index": {
                                    "type": "integer",
                                    "description": (
                                        "Optional: the 0-based index of the WM ring slot "
                                        "this unit occupied when it was surfaced (for STRM "
                                        "relevance-head training). Omit if unknown."
                                    ),
                                },
                            },
                            "required": ["unit_id", "rating"],
                        },
                    },
                },
                "required": ["judgments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": (
                "Pull the FULL text of a retrieved memory unit that was shown "
                "compressed (a gist). Use when a cited context unit looks "
                "relevant but its snippet is too short to answer from. Returns "
                "the unit's complete text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string", "description": "The unit's id (the [id | ...] header)."},
                },
                "required": ["unit_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Re-search memory mid-generation with a refined query and/or "
                "explicit entities/topics. Use when the initial context was "
                "insufficient and you can phrase a better retrieval. Returns "
                "fresh ranked context units."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "entities": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional entity axes to bias the retrieval.",
                    },
                    "topics": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional topic axes to bias the retrieval.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _err(msg: str) -> str:
    """A short tool-result error string (never raises)."""
    return json.dumps({"error": msg})


def dispatch_tool(orchestrator, name: str, args: Any) -> str:
    """Route a tool call to the owning method; return a result string.

    ``orchestrator`` is a ``PonderOrchestrator`` (carries ``store`` +
    ``retriever`` + the ``expand_unit`` / ``search_memory`` methods). ``args`` is
    the tool's ``arguments`` -- already-parsed JSON (a dict) OR a JSON string
    (parsed best-effort). NEVER raises: unknown tool, bad args, or an underlying
    failure all return a short error string so the consumer's loop can't break.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except (ValueError, TypeError):
            return _err(f"malformed arguments for {name}")
    if not isinstance(args, dict):
        return _err(f"arguments for {name} must be an object")

    try:
        if name == "record_feedback":
            judgments = args.get("judgments")
            if not isinstance(judgments, list):
                return _err("record_feedback requires a 'judgments' array")
            store = getattr(orchestrator, "store", None)
            if store is None:
                return _err("no store configured; feedback not persisted")
            # Thread the orchestrator's current query into the STRM 2a raw-rating
            # tap (set at the top of query(); None outside a query / in tests).
            query = getattr(orchestrator, "_current_query", None)
            n = store.record_feedback(judgments, query=query)
            return json.dumps({"ok": True, "applied": n})

        if name == "expand":
            unit_id = args.get("unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                return _err("expand requires a 'unit_id' string")
            text = orchestrator.expand_unit(unit_id)
            return text if isinstance(text, str) else _err(f"unit {unit_id} not found")

        if name == "search_memory":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return _err("search_memory requires a 'query' string")
            context = orchestrator.search_memory(
                query,
                entities=args.get("entities"),
                topics=args.get("topics"),
            )
            return context if isinstance(context, str) else _err("search_memory returned nothing")

        return _err(f"unknown tool: {name}")
    except Exception as e:  # noqa: BLE001 - never break the consumer's loop
        return _err(f"{name} failed: {e}")


def run_tool_loop(
    llm_call: Callable[[list[dict], Optional[list[dict]]], tuple[str, Optional[list[dict]]]],
    prompt: str,
    messages: list[dict],
    dispatch: Callable[[str, Any], str],
    max_iters: int = 4,
    tools: Optional[list[dict]] = None,
) -> dict:
    """Optional convenience agent loop for a host (and Ponder self-chat).

    ``llm_call(messages, tools) -> (content, tool_calls)`` is the consumer's
    model call. Each emitted ``tool_calls`` entry is dispatched via ``dispatch``
    and fed back as a ``tool``-role message; the loop repeats until the model
    emits no tool_calls or ``max_iters`` is hit. Returns ``{"content",
    "tool_messages", "iterations", "collected", "loop_turns", "exhausted"}``.

    ``tools`` defaults to the full ``TOOL_SCHEMAS`` (the external-consumer
    surface); self-chat passes a gated subset (``TOOL_SCHEMAS`` when feedback
    is on, ``LOOP_TOOLS`` when it is off) so ``record_feedback``'s boost
    side-effect stays behind the ``feedback_salience_enabled`` gate -- see
    ``orchestrator._synthesize``. The host may run its OWN loop and call
    ``dispatch_tool`` directly instead -- this is a convenience, not required.

    ``iterations`` is the count of tool calls DISPATCHED; ``loop_turns`` is the
    count of model calls made; ``exhausted`` is True iff the final turn still
    emitted tool_calls (the loop hit ``max_iters`` mid-conversation, not a
    clean stop) so a caller can log/observe truncation.
    """
    if tools is None:
        tools = TOOL_SCHEMAS
    tool_messages: list[dict] = list(messages)
    collected: list[dict] = []
    content = ""
    turns = 0
    exhausted = False
    for _ in range(max_iters):
        turns += 1
        text, tool_calls = llm_call(tool_messages, tools)
        if text:
            content = text
        if not tool_calls:
            exhausted = False
            break
        exhausted = True  # this turn still wanted tools; cleared on a clean stop
        # Echo the assistant's tool_calls back into the transcript so the next
        # call sees the request, then append each dispatched result.
        tool_messages.append({"role": "assistant", "content": text or "",
                              "tool_calls": tool_calls})
        for call in tool_calls:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            cname = fn.get("name", "")
            cargs = fn.get("arguments", {})
            cid = call.get("id") if isinstance(call, dict) else None
            result = dispatch(cname, cargs)
            collected.append({"name": cname, "result": result})
            tool_messages.append({"role": "tool", "tool_call_id": cid or "",
                                  "content": result})
    return {"content": content, "tool_messages": tool_messages,
            "iterations": len(collected), "collected": collected,
            "loop_turns": turns, "exhausted": exhausted}


# A compact subset offered to the NON-loop Bonsai self-chat synthesis path (the
# full set is for the external consumer). The full self-chat TOOL LOOP
# (``orchestrator._synthesize`` with ``self_chat_tool_loop_enabled=True``) uses
# ``TOOL_SCHEMAS`` when feedback is on, or ``LOOP_TOOLS`` when it is off -- that
# is the path through which ``search_memory`` enters self-chat mid-generation.
SELF_CHAT_TOOLS: list[dict] = [TOOL_SCHEMAS[0], TOOL_SCHEMAS[1]]

# Retrieval-only tool set for the self-chat loop when feedback is DISABLED:
# ``expand`` + ``search_memory`` (``record_feedback`` is excluded so the boost
# side-effect stays behind the ``feedback_salience_enabled`` gate even inside
# the loop -- ``dispatch_tool`` does not re-check that gate itself).
LOOP_TOOLS: list[dict] = [TOOL_SCHEMAS[1], TOOL_SCHEMAS[2]]


def feedback_instruction(units: list[dict]) -> str:
    """The system note telling the model it MAY call record_feedback.

    ``units`` is the retrieved-episodes list (each has ``episode_id`` + ``kind``).
    Capped at 12 so the instruction stays bounded for the 4K Bonsai context.
    ASCII-only.
    """
    cap = 12
    lines = [
        "After answering, you MAY call the record_feedback tool to rate how "
        "useful each cited context unit was (1=useless, 3=neutral, 5=essential). "
        "Be critical; 3 means it was present but didn't help. Rate only units you "
        "actually used. Cited unit ids (use these as unit_id):",
    ]
    for u in units[:cap]:
        lines.append(f"- {u.get('episode_id', '?')}")
    if len(units) > cap:
        lines.append(f"- (... {len(units) - cap} more, omitted)")
    return "\n".join(lines)