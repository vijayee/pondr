"""Orchestrates the full retrieval pipeline: plan → traverse → load → context.

``HippocampalRetriever`` ties together the query planner (NL → structured plan)
and the graph traversal (plan → ranked episodes), and builds the structured
context string that Mode A generation consumes. Both LLM-facing pieces (planner,
and later the generator) use the local Bonsai llama-server at
``config.bonsai_endpoint`` — no OpenAI spend.

Semantic fallback (vector search over summary embeddings) is deferred to Phase F
(FAISS + local sentence-transformers embeddings). Until then ``use_semantic`` is
a no-op: ``_semantic_fallback`` returns an empty list, so retrieval is graph-
traversal-only. The hook is wired so Phase F only needs to fill in
``_semantic_fallback`` and ``_embed``.
"""

from __future__ import annotations

from typing import Optional

from ..config import config
from ..memory.store import HippocampalStore
from .graph_traversal import GraphTraversal
from .query_planner import BonsaiQueryPlanner


class HippocampalRetriever:
    """Full retrieval pipeline: plan → traverse → (semantic fallback) → context.

    Phase 1b context strategy: fixed top-N episodes, full text, hard cutoff at
    the token limit. Phase 2.5 adds SSM chunking and JEPA presentation gating.
    """

    def __init__(
        self,
        store: HippocampalStore,
        planner: Optional[BonsaiQueryPlanner] = None,
    ) -> None:
        self.store = store
        self.planner = planner or BonsaiQueryPlanner()
        self.traversal = GraphTraversal(store)
        # Phase F: VectorSearch(store). None until FAISS + embeddings land.
        self.vector_search = None

    def retrieve(self, prompt: str, use_semantic: bool = True) -> list[dict]:
        """Retrieve relevant episodes for a natural-language prompt.

        Args:
            prompt: The user's question.
            use_semantic: Fall back to semantic search if graph traversal
                returns fewer than 3 results. No-op until Phase F (no vector
                index yet) — kept wired so Phase F only fills in the hook.

        Returns:
            Ranked list of episode dicts (see ``GraphTraversal.retrieve`` for
            the shape), highest score first.
        """
        query_plan = self.planner.plan(prompt)
        results = self.traversal.retrieve(query_plan)

        if use_semantic and len(results) < 3:
            semantic_results = self._semantic_fallback(prompt, query_plan)
            existing_ids = {r["episode_id"] for r in results}
            for sr in semantic_results:
                if sr["episode_id"] not in existing_ids:
                    results.append(sr)
                    existing_ids.add(sr["episode_id"])
            results.sort(key=lambda r: r["score"], reverse=True)

        return results

    def retrieve_with_plan(self, query_plan: dict) -> list[dict]:
        """Traverse directly with a caller-supplied plan (skips the planner).

        Lets tests exercise the traverse→load path deterministically without
        the planner (or its server fallback) in the loop.
        """
        return self.traversal.retrieve(query_plan)

    def _semantic_fallback(self, prompt: str, query_plan: dict) -> list[dict]:
        """Semantic fallback over summary embeddings.

        Phase F: embed ``prompt`` with the local sentence-transformers model,
        run ``self.vector_search.search``, and hydrate hits into episode dicts
        with a discounted score (×0.5) so graph-traversal matches rank higher.
        Returns ``[]`` until the vector index exists.
        """
        if self.vector_search is None:
            return []
        # Phase F fills this in once VectorSearch is implemented.
        return []

    def build_context_string(self, episodes: list[dict], max_tokens: Optional[int] = None) -> str:
        """Build a structured context string for Mode A generation.

        Each episode is formatted as ``[id | date]`` + entities/topics/tones +
        summary — structured so the generator doesn't have to infer that Alice
        is a person or that the tone was frustrated. Hard cutoff at
        ``max_tokens`` (chars//4 estimate); episodes beyond the cutoff are
        dropped, not truncated, so a half-episode never enters context.
        """
        if max_tokens is None:
            max_tokens = config.max_context_tokens

        parts = [
            "You have access to relevant past conversations.",
            "Each is formatted as [Episode ID | Date]: Summary with metadata.",
            "Use this context to answer the user's question. If the context",
            "doesn't contain the answer, say so rather than guessing.",
            "",
        ]
        token_count = len("\n".join(parts)) // 4

        for ep in episodes:
            chunk = (
                f"[{ep.get('episode_id', '')} | {ep.get('timestamp', '')}]\n"
                f"Entities: {', '.join(ep.get('entities', []))}\n"
                f"Topics: {', '.join(ep.get('topics', []))}\n"
                f"Tone: {', '.join(ep.get('tones', []))}\n"
                f"Summary: {ep.get('summary', '')}\n"
                "\n"
            )
            chunk_tokens = len(chunk) // 4
            if token_count + chunk_tokens > max_tokens:
                break
            parts.append(chunk)
            token_count += chunk_tokens

        return "\n".join(parts)