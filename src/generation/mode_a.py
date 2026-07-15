"""Mode A: context-window adapter for any LLM.

Mode A is the simplest integration shape from ``docs/Phase 1b.md``: retrieve
relevant episodes, build a structured context string, and hand it to an LLM
alongside the user's prompt. The LLM never knows about the memory system — it
just receives curated context.

The generator talks to the **local Bonsai llama-server** at
``config.bonsai_endpoint`` via its OpenAI-compatible ``/chat/completions`` API —
NOT to OpenAI. ``config.generation_model`` defaults to the Bonsai GGUF so
generation costs nothing; override ``GENERATION_MODEL`` for a different backend.
The LLM call is isolated in ``_complete`` so tests can stub it without a server.
"""

from __future__ import annotations

from typing import Optional

import requests

from ..config import config
from ..retrieval.retriever import HippocampalRetriever

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to past conversations. "
    "Use the provided context to answer the user's question accurately. "
    "If the context doesn't contain the answer, say so."
)


def _normalize_tool_calls(tool_calls: list) -> list[dict]:
    """Coerce a server's ``tool_calls`` list to plain dicts for ``dispatch_tool``.

    Bonsai/llama-server returns the OpenAI shape (each call has ``id`` +
    ``function: {name, arguments}`` with ``arguments`` as a JSON STRING); this
    keeps that shape and just ensures plain dict types (no pydantic-ish
    objects some servers return). ``arguments`` is left as the raw string --
    ``dispatch_tool`` parses it best-effort.
    """
    out: list[dict] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        out.append({
            "id": call.get("id"),
            "type": call.get("type", "function"),
            "function": {
                "name": fn.get("name", "") if isinstance(fn, dict) else "",
                "arguments": fn.get("arguments", {}) if isinstance(fn, dict) else {},
            },
        })
    return out


class ModeAGenerator:
    """Context-window adapter: retrieve → build context → LLM completion."""

    def __init__(
        self,
        retriever: HippocampalRetriever,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
    ) -> None:
        self.retriever = retriever
        self.model = model or config.generation_model
        self.endpoint = (endpoint or config.bonsai_endpoint).rstrip("/")
        self.temperature = temperature if temperature is not None else config.generation_temperature
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        max_context_tokens: Optional[int] = None,
    ) -> dict:
        """Generate a response using retrieved context.

        Returns:
            ``{"response", "retrieved_episodes", "context_used", "model"}``.
        """
        # Phase 1c: pass conversation_history to the retriever so the planner
        # can resolve pronouns / implicit references against recent context
        # (not just to the LLM, which the lines below already do).
        episodes = self.retriever.retrieve(prompt, conversation_history=conversation_history)
        context = self.retriever.build_context_string(episodes, max_context_tokens)

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if conversation_history:
            # Last 10 turns — keeps the prompt bounded for the 4K Bonsai context.
            messages.extend(conversation_history[-10:])
        messages.append({
            "role": "user",
            "content": f"Context from past conversations:\n{context}\n\nUser: {prompt}",
        })

        response, _ = self._complete(messages)
        return {
            "response": response,
            "retrieved_episodes": episodes,
            "context_used": context,
            "model": self.model,
        }

    def generate_with_routing(
        self,
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        max_context_tokens: Optional[int] = None,
    ) -> dict:
        """Generate using the subconscious Retrieval Gate (Phase 2b, opt-in).

        Calls ``retriever.retrieve_with_routing`` and acts on the pathway:

        - ``graph_retrieve`` / ``conscious_deliberation`` (supported): build the
          context from the retrieved episodes and complete via the local Bonsai
          endpoint (reuses ``_complete``). The gate's predicted ``model_size`` is
          recorded in the result for the future model-size ladder — generation
          itself still uses the single configured Bonsai model (the ladder is a
          later phase; we don't fake a multi-model selection).
        - ``ssm_direct`` / ``process_exec`` / ``tool_plan`` (unsupported): the
          routed pathway has no executor wired yet, so ``response`` is ``None``
          and ``supported=False`` — surfaced honestly, not faked.

        Returns ``{"response", "route", "retrieved_episodes", "model_used",
        "context_used", "supported"}``.
        """
        retrieval_result = self.retriever.retrieve_with_routing(
            prompt, conversation_history=conversation_history
        )
        route = retrieval_result["route"]

        if not retrieval_result["supported"]:
            return {
                "response": None,
                "route": route,
                "retrieved_episodes": [],
                "model_used": None,
                "context_used": None,
                "supported": False,
            }

        episodes = retrieval_result.get("results", [])
        context = retrieval_result.get("context") or self.retriever.build_context_string(
            episodes, max_context_tokens
        )
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history[-10:])
        messages.append({
            "role": "user",
            "content": f"Context from past conversations:\n{context}\n\nUser: {prompt}",
        })
        response, _ = self._complete(messages)
        return {
            "response": response,
            "route": route,
            "retrieved_episodes": episodes,
            "model_used": self.model,
            "context_used": context,
            "supported": True,
        }

    def generate_with_working_memory(
        self,
        prompt: str,
        conversation_history: Optional[list[dict]] = None,
        max_context_tokens: Optional[int] = None,
        session=None,
        end_state: Optional[str] = None,
        format_spec: Optional[dict] = None,
        extract_schema: Optional[dict] = None,
        model_size: Optional[str] = None,
    ) -> dict:
        """Phase 2c: full Working-Memory + chunking + presentation pipeline.

        Delegates to the ``PonderOrchestrator.query()`` contract: it routes,
        retrieves, updates working memory, plans presentation (both axes), and
        dispatches on the end state. For ``ssm_direct``/``process_exec``/
        ``tool_plan`` pathways (unsupported) returns ``supported=False`` —
        honest, not faked (mirrors ``generate_with_routing``).

        ``session`` is a ``PonderOrchestrator``. When ``session`` is None this
        falls back to the plain ``generate`` path (no WM/chunking) so the method
        is callable without a configured orchestrator.

        Returns the orchestrator's result dict (see
        ``PonderOrchestrator.query``). For ``direct``/``format``/``extract`` the
        dict has no ``response`` (no LLM call); for ``synthesize`` it has
        ``response`` (the Bonsai completion).
        """
        if session is None:
            # No orchestrator configured — degrade to the plain Phase 1b path
            # (no working memory / chunking / end-state dispatch).
            return self.generate(prompt, conversation_history=conversation_history,
                                 max_context_tokens=max_context_tokens)
        return session.query(
            prompt,
            consumer="bonsai",
            conversation_history=conversation_history,
            end_state=end_state,
            format_spec=format_spec,
            extract_schema=extract_schema,
            model_size=model_size,
        )

    def _complete(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> tuple[str, Optional[list[dict]]]:
        """LLM completion over the local Bonsai endpoint.

        Returns ``(content, tool_calls)`` -- ``content`` is the message text
        (``""`` when the model emitted only tool calls), ``tool_calls`` is the
        parsed ``choices[0].message.tool_calls`` list or ``None`` when absent
        (no tools passed, or the model chose not to call any). ``tools`` /
        ``tool_choice`` (OpenAI tool-calling protocol) are forwarded to the
        Bonsai payload when given; both default ``None`` so existing callers
        that pass no tools are unaffected (they unpack ``(content, None)``).

        Raises ``RuntimeError`` with the verbatim server response on any
        failure (connection, non-200, parse) so the caller sees the raw output.
        Tests override this method to stub the server out. ``tool_calls``
        (when present) are normalized to plain dicts (``{"id", "function":
        {"name", "arguments"}}``) so ``dispatch_tool`` can consume them.
        """
        url = f"{self.endpoint}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
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
            msg = outer["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Bonsai response missing choices[0].message: {outer}") from e
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tool_calls = _normalize_tool_calls(tool_calls)
        return content, tool_calls