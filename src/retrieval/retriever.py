"""Orchestrates the full retrieval pipeline: plan → traverse → load → context.

``HippocampalRetriever`` ties together the query planner (NL → structured plan)
and the graph traversal (plan → ranked episodes), and builds the structured
context string that Mode A generation consumes. Both LLM-facing pieces (planner,
and later the generator) use the local Bonsai llama-server at
``config.bonsai_endpoint`` — no OpenAI spend.

Semantic fallback (Phase F): when graph traversal returns fewer than 3
results, the retriever falls back to ``VectorSearch`` over summary embeddings
(local sentence-transformers, FAISS on the pod / pure-Python cosine offline).
Hits are hydrated into episode dicts with a 0.5 score discount so graph-
traversal matches rank higher. The index is auto-loaded from
``{db}/vector_index_ids.json`` when ``auto_load_index=True``; otherwise
``vector_search`` stays None and the fallback is a no-op (the graph-only path).

Phase 2b adds the **Retrieval Gate** (``RetrievalGate``, ``src/subconscious``):
an optional subconscious router consulted *before* retrieval via
``retrieve_with_routing``. The gate predicts domain(s)/pathway/model-size/
deliberation; the retriever then acts on the pathway. The existing
``retrieve()`` is unchanged (still returns ``list[dict]``) so ``ModeAGenerator``
keeps working — the routing path is opt-in. The gate + embedder are injected as
already-constructed objects so this module does NOT import torch at import time
(the retrieval package stays usable without it).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from ..config import config
from ..memory.store import HippocampalStore
from .graph_traversal import GraphTraversal
from .query_planner import BonsaiQueryPlanner

if TYPE_CHECKING:  # torch/subconscious only needed for type hints, not at runtime
    from ..subconscious.retrieval_gate import RetrievalGate
    from ..subconscious.routing import RoutingDecision, RoutingOutcome


class HippocampalRetriever:
    """Full retrieval pipeline: plan → traverse → (semantic fallback) → context.

    Phase 1b context strategy: fixed top-N episodes, full text, hard cutoff at
    the token limit. Phase 2.5 adds SSM chunking and JEPA presentation gating.
    Phase 2b adds subconscious routing via ``retrieve_with_routing`` (opt-in;
    pass a ``retrieval_gate`` + ``embedder``).
    """

    def __init__(
        self,
        store: HippocampalStore,
        planner: Optional[BonsaiQueryPlanner] = None,
        auto_load_index: bool = False,
        retrieval_gate: "Optional[RetrievalGate]" = None,
        embedder: Optional[object] = None,
    ) -> None:
        self.store = store
        self.planner = planner or BonsaiQueryPlanner()
        self.traversal = GraphTraversal(store)
        # Phase F: VectorSearch over summary embeddings. Auto-loaded from
        # {db}/vector_index_ids.json when auto_load_index is set (the live pod
        # pipeline); tests pass a stub VectorSearch or leave it None.
        self.vector_search = None
        if auto_load_index:
            self._try_load_vector_index()

        # Phase 2b: subconscious routing (opt-in). The gate embeds the prompt via
        # the injected embedder (the real bge-small VectorSearch embedder, or a
        # stub in tests). If no embedder is passed but a VectorSearch index was
        # auto-loaded, reuse its embedder for routing.
        self.gate = retrieval_gate
        self._route_embedder = embedder
        if self.gate is not None and self._route_embedder is None and self.vector_search is not None:
            self._route_embedder = self.vector_search
        self._outcome_trainer = None  # lazily built on first record_outcome

    def _try_load_vector_index(self) -> None:
        """Attach + load a persisted VectorSearch index if one exists."""
        from pathlib import Path
        from .vector_search import VectorSearch
        ids_path = Path(self.store.db_path) / VectorSearch.IDS_NAME
        if not ids_path.exists():
            return
        vs = VectorSearch(self.store)
        try:
            vs.load(self.store.db_path)
        except (OSError, json.JSONDecodeError, RuntimeError):
            # Corrupt index file, unreadable ids JSON, or faiss-saved index
            # without faiss installed — degrade to graph-only retrieval.
            return
        self.vector_search = vs

    def retrieve(
        self,
        prompt: str,
        conversation_history: list[dict] | None = None,
        use_semantic: bool = True,
    ) -> list[dict]:
        """Retrieve relevant episodes for a natural-language prompt.

        Args:
            prompt: The user's question.
            conversation_history: Recent turns for pronoun / implicit-reference
                resolution by the planner (Phase 1c). Optional, backward
                compatible (``None`` = plan from the prompt alone).
            use_semantic: Fall back to semantic search if graph traversal
                returns fewer than 3 results. No-op until Phase F (no vector
                index yet) — kept wired so Phase F only fills in the hook.

        Returns:
            Ranked list of episode dicts (see ``GraphTraversal.retrieve`` for
            the shape), highest score first.
        """
        query_plan = self.planner.plan(prompt, conversation_history)
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

    # ── Phase 2b: subconscious routing ──

    def retrieve_with_routing(
        self,
        prompt: str,
        conversation_history: list[dict] | None = None,
        use_semantic: bool = True,
    ) -> dict:
        """Retrieve with the subconscious Retrieval Gate consulted first.

        Returns ``{"type", "route", "results", "context", "supported"}``:

        - ``graph_retrieve`` / ``conscious_deliberation`` → runs the existing
          ``retrieve`` pipeline (plan → traverse → semantic fallback) and
          builds the context string. ``conscious_deliberation`` additionally
          flags the result for System 2 (the generator decides what to do with
          it). The gate's predicted ``domains`` are recorded in ``route`` but
          do NOT filter traversal — the Phase 1b graph traversal is
          domain-agnostic (it scores on entities/topics/tones), so filtering
          by domain here would be theater. Domain-aware traversal is a later
          phase; the route carries the domains for that future hook.
        - ``ssm_direct`` → answer from Working Memory. No Working-Memory/SSM
          state holder is wired into the pipeline yet (Phase 2.5), so this is
          ``supported=False`` with empty results — the caller (e.g.
          ``ModeAGenerator.generate_with_routing``) surfaces it honestly rather
          than faking a response.
        - ``process_exec`` / ``tool_plan`` → no stored-process or tool-planning
          infrastructure exists yet; ``supported=False``, empty results.

        Raises ``RuntimeError`` if no gate was configured (this method is only
        meaningful with a ``RetrievalGate``).
        """
        if self.gate is None:
            raise RuntimeError(
                "retrieve_with_routing requires a retrieval_gate at construction"
            )
        if self._route_embedder is None:
            raise RuntimeError(
                "retrieve_with_routing requires an embedder (pass embedder= or "
                "auto_load_index=True with a persisted vector index)"
            )

        route = self.gate.route_text(prompt, self._route_embedder)

        if route.pathway in ("graph_retrieve", "conscious_deliberation"):
            results = self.retrieve(prompt, conversation_history=conversation_history,
                                    use_semantic=use_semantic)
            context = self.build_context_string(results) if results else None
            return {
                "type": route.pathway,
                "route": route,
                "results": results,
                "context": context,
                "supported": True,
            }

        # ssm_direct / process_exec / tool_plan: routed but not yet executable
        # end-to-end. Return the route + an honest unsupported flag + empty
        # results; never fake a response.
        return {
            "type": route.pathway,
            "route": route,
            "results": [],
            "context": None,
            "supported": False,
        }

    def record_outcome(
        self,
        prompt: str,
        route: "RoutingDecision",
        outcome: "RoutingOutcome",
    ) -> None:
        """Record a routing outcome for the outcome-based trainer (Phase 2b).

        No-op unless a gate was configured. The (embedding, context, decision,
        outcome) tuple is pushed to the gate's ``OutcomeBasedTrainer`` replay
        buffer; ``train_from_outcomes`` is the caller's responsibility (the live
        pipeline calls it on a schedule). The prompt is re-embedded here so the
        replay entry is self-contained.
        """
        if self.gate is None or self._route_embedder is None:
            return
        # Lazy-import the trainer (torch/subconscious) so this module stays
        # importable without torch when no gate is configured.
        from ..subconscious.training.routing_training import OutcomeBasedTrainer
        if self._outcome_trainer is None:
            self._outcome_trainer = OutcomeBasedTrainer(self.gate)
        import torch  # local: only needed when actually recording
        vec = self._route_embedder.encode([prompt])[0]
        emb = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        emb = emb.to(next(self.gate.parameters()).device)
        self._outcome_trainer.record_outcome(emb, context=None,
                                             decision=route, outcome=outcome)

    def _semantic_fallback(self, prompt: str, query_plan: dict) -> list[dict]:
        """Semantic fallback over summary embeddings.

        Embed ``prompt`` with the local sentence-transformers model, run
        ``self.vector_search.search``, and hydrate hits into episode dicts with
        a discounted score (×0.5) so graph-traversal matches rank higher.
        Returns ``[]`` when no vector index is configured.
        """
        if self.vector_search is None:
            return []
        hits = self.vector_search.search(prompt, k=config.default_retrieval_limit)
        out: list[dict] = []
        for eid, sim in hits:
            ep = self.traversal._hydrate(eid)
            ep["score"] = sim * 0.5  # discount so graph matches rank higher
            out.append(ep)
        return out

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