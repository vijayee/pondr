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
        episodes = self.retriever.retrieve(prompt)
        context = self.retriever.build_context_string(episodes, max_context_tokens)

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if conversation_history:
            # Last 10 turns — keeps the prompt bounded for the 4K Bonsai context.
            messages.extend(conversation_history[-10:])
        messages.append({
            "role": "user",
            "content": f"Context from past conversations:\n{context}\n\nUser: {prompt}",
        })

        response = self._complete(messages)
        return {
            "response": response,
            "retrieved_episodes": episodes,
            "context_used": context,
            "model": self.model,
        }

    def _complete(self, messages: list[dict]) -> str:
        """LLM completion over the local Bonsai endpoint; returns the content.

        Raises ``RuntimeError`` with the verbatim server response on any failure
        (connection, non-200, parse) so the caller sees the raw output. Tests
        override this method to stub the server out.
        """
        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
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
            return outer["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Bonsai response missing choices[0].message.content: {outer}") from e