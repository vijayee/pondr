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
    from .document_retriever import DocumentRetriever


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

        # Phase 1c: document-aware aggregation (Refinement 1). Set externally by
        # ``runtime.build_ponder`` (only when the store has document section
        # edges -- ``store_has_documents`` probe). ``None`` = conversation-only
        # corpus -> aggregation is a no-op and retrieval is byte-identical to
        # the pre-1c path. When set, ``retrieve`` post-processes its results
        # through ``DocumentRetriever.aggregate_results`` so multi-section hits
        # surface as one document result.
        self.document_retriever: Optional["DocumentRetriever"] = None

    def _try_load_vector_index(self) -> None:
        """Attach a vector backend for the semantic fallback.

        Prefers the in-DB WaveDB VectorLayer (``store.vector_layer``) via the
        ``WavedbVectorStore`` adapter when the store opened one -- the index is
        maintained live by the store (insert on encode, delete on forget), so
        there is nothing to load. Falls back to the persisted FAISS
        ``VectorSearch`` sidecar (``{db}/vector_index_ids.json``) when the
        layer is absent/disabled (old wavedb or ``vector_index_enabled=False``).
        """
        if getattr(self.store, "vector_layer", None) is not None:
            from .wavedb_vector_store import WavedbVectorStore
            self.vector_search = WavedbVectorStore(self.store)
            return
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
        signal: str = "routine",
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
            signal: The caller's affective/task signal (Phase 3b forgetting —
                ``important``/``routine``/``correction``/...), threaded to the
                traversal's retrieval-boost hook so matched edges strengthen
                with use. Defaults to ``"routine"`` (no-op for cold-start).

        Returns:
            Ranked list of episode dicts (see ``GraphTraversal.retrieve`` for
            the shape), highest score first.
        """
        query_plan = self.planner.plan(prompt, conversation_history)
        results = self.traversal.retrieve(query_plan, signal=signal)

        if use_semantic and len(results) < 3:
            semantic_results = self._semantic_fallback(prompt, query_plan)
            existing_ids = {r["episode_id"] for r in results}
            for sr in semantic_results:
                if sr["episode_id"] not in existing_ids:
                    results.append(sr)
                    existing_ids.add(sr["episode_id"])
        # Kind-aware diversity rerank (Phase 2c+): sort by score (the per-unit
        # feedback boost is already applied in both score sites), then cap the
        # run of one kind so a wall of sections (or episodes) can't drown the
        # other kind in the top-K. Gated on ``kind_diversity_cap > 0``; when 0
        # this is a pure score sort (the pre-2c+ behavior). Independent of
        # feedback_salience_enabled. Replaces the old bare ``results.sort``.
        results = self._kind_aware_rerank(results)

        # Phase 1c: aggregate multi-section hits into one document result when a
        # ``DocumentRetriever`` is attached (set by ``runtime.build_ponder`` for
        # corpora that have document section edges). No-op when ``None``
        # (conversation-only corpus). ``retrieve_with_routing`` calls this
        # method, so the routed graph path is covered transitively.
        if self.document_retriever is not None:
            results = self.document_retriever.aggregate_results(results)

        return results

    def retrieve_with_plan(self, query_plan: dict, signal: str = "routine") -> list[dict]:
        """Traverse directly with a caller-supplied plan (skips the planner).

        Lets tests exercise the traverse→load path deterministically without
        the planner (or its server fallback) in the loop.
        """
        return self.traversal.retrieve(query_plan, signal=signal)

    def retrieve_by_embedding(
        self,
        query_emb,
        signal: str = "routine",
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Vector search with a PRE-COMPUTED query embedding (no text re-embed).

        STRM Phase 4 Step 5: the salience trigger fires state-conditioned
        retrieval -- the query is the salient anchor's 384-d doc vector (the
        episode the WM state flagged as being-forgotten), NOT the prompt text.
        Reuses the same vector index the ``use_semantic`` fallback uses, hydrates
        hits into episode dicts in the same shape as ``retrieve`` (with the same
        0.5 score discount so prompt-driven graph matches rank higher), and
        applies the per-unit feedback boost. ``[]`` when no vector index is
        configured (no-op -- byte-identical to a no-salience turn).

        Args:
            query_emb: ``[384]`` / ``[1,384]`` tensor or list[float] -- the
                state-conditioned query (a bge-space 384-d vector).
            signal: the caller's affective/task signal, threaded to the
                retrieval-boost hook (same as ``retrieve``).
            limit: max hits (defaults to ``config.default_retrieval_limit``).
        """
        if self.vector_search is None:
            return []
        # Tensor -> flat list[float] (the C/Python search backends take a 1-D
        # list). Accept [384] or [1,384] (the anchor doc_emb is [1,384]) by
        # flattening to 1-D first.
        if hasattr(query_emb, "detach"):
            import torch  # local: only needed for the tensor->list conversion
            v = query_emb.detach().cpu().to(torch.float32).reshape(-1)
            vec = [float(x) for x in v.tolist()]
        else:
            try:
                vec = [float(x) for x in query_emb]
            except (TypeError, ValueError):
                return []
        if not vec:
            return []
        k = limit if limit is not None else config.default_retrieval_limit
        hits = self.vector_search.search_by_vector(vec, k=k)
        out: list[dict] = []
        for eid, sim in hits:
            ep = self.traversal._hydrate(eid)
            ep["score"] = sim * 0.5  # discount so graph matches rank higher
            out.append(ep)
        # Same boost path as the semantic fallback so no scored result bypasses
        # the per-unit feedback boost.
        self.traversal._apply_unit_boost(out)
        return out

    # ── Phase 2b: subconscious routing ──

    def retrieve_with_routing(
        self,
        prompt: str,
        conversation_history: list[dict] | None = None,
        use_semantic: bool = True,
        signal: str = "routine",
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
                                    use_semantic=use_semantic, signal=signal)
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
        # Phase 2c+: semantic-fallback hits are boost-aware too (one shared
        # helper with the graph score site so no scored result bypasses the
        # per-unit feedback boost).
        self.traversal._apply_unit_boost(out)
        return out

    def _kind_aware_rerank(self, results: list[dict]) -> list[dict]:
        """Sort by score, then cap the run of any one ``kind`` in the top-K.

        Greedy walk over the score-sorted list: allow at most
        ``config.kind_diversity_cap`` CONSECUTIVE results of the same kind
        (``section``/``document``/``episode`` -- ``episode`` when ``kind`` is
        absent, the episode-dict default) before the next slot is taken from a
        DIFFERENT kind if one remains. This prevents a wall of section chunks
        (or a wall of episodes) drowning the other kind in the top-K. Score
        order is preserved WITHIN each kind. ``kind_diversity_cap=0`` disables
        the cap -> pure score sort (the pre-2c+ behavior). Independent of
        ``feedback_salience_enabled``.
        """
        cap = config.kind_diversity_cap
        results = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
        if cap <= 0 or not results:
            return results

        def _kind(r: dict) -> str:
            return r.get("kind") or "episode"

        remaining = list(results)
        out: list[dict] = []
        while remaining:
            # Count how many of the current leading kind are already at the tail.
            run = 0
            if out:
                last_kind = _kind(out[-1])
                for r in reversed(out):
                    if _kind(r) == last_kind:
                        run += 1
                    else:
                        break
            if run >= cap:
                # The tail is saturated with one kind -- pick the next result of
                # a DIFFERENT kind if any remains (keep score order: the first
                # non-matching remaining item is the highest-scoring other kind).
                pick = None
                for r in remaining:
                    if _kind(r) != _kind(out[-1]):
                        pick = r
                        break
                if pick is None:
                    # No other kind left -- append the rest in score order.
                    out.extend(remaining)
                    remaining = []
                    break
                out.append(pick)
                remaining.remove(pick)
            else:
                out.append(remaining.pop(0))
        return out

    def build_with_chunking(
        self,
        query: str,
        episodes: list[dict],
        presentation_plan,
        working_memory=None,
        ssm_chunker=None,
        consumer: str = "bonsai",
    ) -> tuple[str, "ChunkedContext"]:
        """Phase 2c: chunk episodes per ``presentation_plan`` and build the context.

        1. ``SSMChunker.chunk(episodes, plan)`` → ``ChunkedContext`` (primary full
           text + compressed gist + secondary episode dicts for EXPAND).
        2. ``ChunkedContextFormatter.format_for_llm(chunked, consumer, working_memory)``
           → the context string for the generation model.

        ``retrieve()`` / ``retrieve_with_routing()`` are unchanged (back-compat).
        The chunker is injected (the orchestrator owns it) so this module does
        NOT import the torch/subconscious chunker at module load.
        """
        if ssm_chunker is None:
            raise RuntimeError("build_with_chunking requires an ssm_chunker")
        from .chunked_context import ChunkedContextFormatter
        chunked = ssm_chunker.chunk(episodes, presentation_plan)
        formatter = ChunkedContextFormatter()
        context = formatter.format_for_llm(chunked, consumer=consumer,
                                           working_memory=working_memory)
        return context, chunked

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
            eid = ep.get("episode_id", "")
            kind = ep.get("kind")
            if kind == "section":
                # Section (per-chunk) result: the matched chunk body is in
                # ``text`` (materialized at hydrate), so no store/cold pull here.
                body = ep.get("text", "")
                heading = ep.get("section_heading", "")
                chunk = (
                    f"[{eid} | {ep.get('timestamp', '')}]\n"
                    f"Source: {ep.get('source_path', '')}\n"
                    f"Title: {ep.get('summary', '')}\n"
                    f"Entities: {', '.join(ep.get('entities', []))}\n"
                    f"Topics: {', '.join(ep.get('topics', []))}\n"
                    + (f"Section '{heading}': {body}\n" if heading else
                       (f"Section: {body}\n" if body else ""))
                    + "\n"
                )
            elif kind == "document":
                # Document result (graph-path hit): cite source + title + the
                # matched section body (in ``text`` at hydrate, no cold pull).
                matched = ep.get("matched_section", "")
                body = ep.get("text", "")
                chunk = (
                    f"[{eid} | {ep.get('timestamp', '')}]\n"
                    f"Source: {ep.get('source_path', '')}\n"
                    f"Title: {ep.get('summary', '')}\n"
                    f"Entities: {', '.join(ep.get('entities', []))}\n"
                    f"Topics: {', '.join(ep.get('topics', []))}\n"
                    + (f"Section '{matched}': {body}\n" if matched else
                       (f"Section: {body}\n" if body else ""))
                    + "\n"
                )
            else:
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