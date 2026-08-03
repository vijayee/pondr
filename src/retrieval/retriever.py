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
    from .bm25 import BM25Search
    from .document_retriever import DocumentRetriever


# User-scope vector over-fetch factor. When user-scope is ON, the semantic /
# embedding paths over-fetch by this multiple of ``k`` THEN filter the hits by
# shape (ep_*/doc_*/section) against the user's owned id sets, taking the top
# ``k`` after filtering. The over-fetch compensates for the global flat vector
# index returning other users' content (which gets filtered out) so a user's
# own k results still surface. 3x is an honest, documented knob, not a silent
# cap: if a user owns a small fraction of the corpus, the post-filter can still
# come up short (raise this); if the corpus is single-user it's pure overhead
# (leave user-scope OFF -- the default). See ``_filter_vector_hits_by_scope``.
_USER_SCOPE_FETCH_MULT = 3


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
        user_id: Optional[str] = None,
    ) -> None:
        self.store = store
        self.planner = planner or BonsaiQueryPlanner()
        self.traversal = GraphTraversal(store)
        # User-scope (retrieval user boundary): when set, every retrieve path
        # intersects its candidate set with this user's owned episode + document
        # id sets (strict scope -- no cross-user, no unscoped, no memories).
        # ``None`` = unscoped (the pre-user-scope global path, byte-identical).
        # The retriever is constructed ONCE in ``runtime.build_ponder`` and
        # injected into the orchestrator, so holding ``user_id`` here covers all
        # orchestrator call sites without per-call threading. Computed per call
        # (not cached at init) so episodes/docs ingested during the session are
        # visible to later retrieves.
        self.user_id = user_id
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

        # A2 RRF hybrid retrieval (Tencent-survey item 3): when
        # ``hybrid_retrieval`` is set AND a ``BM25Search`` is attached (both by
        # ``runtime.build_ponder`` when ``--hybrid-retrieval`` is on),
        # :meth:`retrieve` early-returns through :meth:`_retrieve_hybrid`, which
        # fuses THREE ranked id-lists (graph, vector, BM25) via Reciprocal Rank
        # Fusion. The existing graph+vector-fallback body is byte-identical when
        # the flag is off (the early-return branch is not entered). ``bm25`` is
        # lazy-built in ``_retrieve_hybrid`` if a caller sets ``hybrid_retrieval``
        # without attaching one (e.g. a test that toggles the global directly).
        self.hybrid_retrieval = False
        self.bm25: "Optional[BM25Search]" = None

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

    # ── User-scope (retrieval user boundary) ──

    def _user_scope_sets(
        self,
    ) -> tuple[Optional[set[str]], Optional[set[str]], Optional[set[str]]]:
        """The query user's owned episode + document + scene id sets (strict scope).

        Returns ``(allowed_ep, allowed_doc, allowed_scene)``. When
        :attr:`user_id` is ``None`` (user-scope OFF) returns ``(None, None, None)``
        so every retrieve path skips filtering -> byte-identical to the pre-user-
        scope path. When set, all three sets are recomputed per call from the
        store's SPO indices -- a user who ingests a doc mid-session sees it on
        the next retrieve (no init-time cache). The scans are cheap (SPO prefix
        range reads). Empty for an unknown user (no sessions / no owned docs /
        no scenes) -> strict scope returns nothing, the honest result.
        """
        if self.user_id is None:
            return None, None, None
        allowed_ep = self.store.episode_ids_for_user(self.user_id)
        allowed_doc = self.store.document_ids_for_user(self.user_id)
        allowed_scene = self.store.scene_ids_for_user(self.user_id)
        return allowed_ep, allowed_doc, allowed_scene

    @staticmethod
    def _filter_vector_hits_by_scope(
        hits: list[tuple[str, float]],
        allowed_episode_ids: Optional[set[str]],
        allowed_document_ids: Optional[set[str]],
        allowed_scene_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Filter vector-search hits by shape against the user's owned ids.

        The vector index is ONE flat global layer (``episodes``) holding
        ``ep_*``, ``{doc_id}_sec_NNN``, ``M:*``, and (B1) ``scene_*`` together --
        no user partition, so the boundary is an id-set intersection, not a
        key-prefix. Each hit is kept iff its kind is owned by the query user:

        * ``ep_*`` -> ``allowed_episode_ids`` (``None`` = pass).
        * ``{doc_id}_sec_NNN`` -> parent ``doc_id`` in ``allowed_document_ids``
          (``None`` = pass). ``_sec_`` check precedes the ``doc_`` check (section
          ids start with ``doc_`` -- mirrors ``_hydrate``).
        * ``doc_*`` (no ``_sec_``) -> ``allowed_document_ids`` (``None`` = pass).
        * ``scene_*`` -> ``allowed_scene_ids`` (``None`` = pass). Scenes ARE in
          the vector index (D1 -- bodies are embedded + inserted), so a semantic
          search can return a ``scene_*`` id; this branch enforces scope on it.
        * ``M:*`` -> dropped under strict scope (any set not ``None``); kept when
          user-scope is fully off.

        When all three allowed sets are ``None`` returns ``hits`` unchanged
        (byte-identical). The caller over-fetches (``k * _USER_SCOPE_FETCH_MULT``)
        before calling this, then takes the top ``k`` of the filtered list. Once
        we reach the loop the early ``all None`` return means strict scope is on,
        so ``M:`` is always dropped here.
        """
        if (allowed_episode_ids is None and allowed_document_ids is None
                and allowed_scene_ids is None):
            return hits
        out: list[tuple[str, float]] = []
        for eid, sim in hits:
            if eid.startswith("M:"):
                continue  # memories dropped under strict scope (unwired, no data)
            if "_sec_" in eid:
                if allowed_document_ids is None:
                    out.append((eid, sim))
                else:
                    doc_id = eid.rsplit("_sec_", 1)[0]
                    if doc_id in allowed_document_ids:
                        out.append((eid, sim))
                continue
            if eid.startswith("doc_"):
                if allowed_document_ids is None or eid in allowed_document_ids:
                    out.append((eid, sim))
                continue
            if eid.startswith("scene_"):
                if allowed_scene_ids is None or eid in allowed_scene_ids:
                    out.append((eid, sim))
                continue
            # episode (ep_*)
            if allowed_episode_ids is None or eid in allowed_episode_ids:
                out.append((eid, sim))
        return out

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
        allowed_ep, allowed_doc, allowed_scene = self._user_scope_sets()
        # A2: when hybrid retrieval is on AND a BM25Search is attached (both set
        # by ``runtime.build_ponder`` under ``--hybrid-retrieval``), take the
        # RRF fusion path -- graph + vector + BM25 ranked lists fused by
        # Reciprocal Rank Fusion. The existing graph+vector-fallback body below
        # is byte-identical when the flag is off (this branch is not entered:
        # ``hybrid_retrieval`` defaults False, ``bm25`` defaults None).
        if self.hybrid_retrieval and self.bm25 is not None:
            return self._retrieve_hybrid(
                prompt, query_plan, signal,
                allowed_ep, allowed_doc, allowed_scene,
            )
        results = self.traversal.retrieve(
            query_plan, signal=signal,
            allowed_episode_ids=allowed_ep, allowed_document_ids=allowed_doc,
            allowed_scene_ids=allowed_scene,
        )

        if use_semantic and len(results) < 3:
            semantic_results = self._semantic_fallback(
                prompt, query_plan, allowed_ep, allowed_doc, allowed_scene)
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

    def _retrieve_hybrid(
        self,
        prompt: str,
        query_plan: dict,
        signal: str,
        allowed_ep: "Optional[set[str]]",
        allowed_doc: "Optional[set[str]]",
        allowed_scene: "Optional[set[str]]",
    ) -> list[dict]:
        """RRF hybrid retrieval -- fuse graph + vector + BM25 ranked lists.

        Tencent-survey A2: the graph-walk ranking, the vector-similarity
        ranking, and the BM25 lexical ranking are each turned into an ordered
        id-list, then fused by Reciprocal Rank Fusion (parameter-free
        ``score = sum 1/(k + rank + 1)``, ``k=60``). RRF uses only RANK
        positions, so the three lists' wildly different score scales (graph
        heuristic ~10, vector cosine ~0.5, BM25 ~5) never collide -- no weight
        tuning, no normalization. This one-ups Tencent (which fuses only
        BM25+vector) by folding the graph-walk in as a third RRF list with zero
        formula change.

        The lexical list is the NEW component: an episode whose ``full_text``
        contains the query words but whose entities/topics the planner didn't
        surface (graph miss) and whose summary embedding doesn't cosine-rank
        (vector miss) was invisible to the pre-A2 path -- BM25 finds it.

        User-scope: the vector + BM25 lists over-fetch (``k *
        _USER_SCOPE_FETCH_MULT``) when scoped, then filter by the user's owned
        id sets; the graph list is already scoped by ``traversal.retrieve``.
        ``allowed_ep`` (episode ids) is the BM25 filter; the vector filter reuses
        ``_filter_vector_hits_by_scope`` (ep/doc/scene shapes).

        RRF score replaces the per-list score on every hydrated result; the
        per-unit feedback boost is NOT re-applied here (it is incompatible with
        RRF's rank semantics -- a 0.25..4.0 score multiplier would swamp a
        ~0.02 RRF score). The boost IS honored: it shaped the graph list's
        internal sort (``traversal.retrieve`` applies it), which shaped the
        graph ranks, which shaped the RRF contribution. ``strategy="hybrid"`` is
        stamped on every result as additive provenance (the context builder
        ignores unknown dict keys, so it is NOT LLM-facing -- matches Tencent's
        ``strategy`` field, B2-steal).
        """
        # Lazy imports so the OFF path (the common case) never imports the
        # hybrid modules, and a test that toggles the global without a
        # build_ponder-attached BM25Search still works.
        from .bm25 import BM25Search, rrf_fuse
        from ..memory.bm25_index import tokenize

        k = config.default_retrieval_limit
        scoped = (allowed_ep is not None or allowed_doc is not None
                  or allowed_scene is not None)
        fetch_k = k * _USER_SCOPE_FETCH_MULT if scoped else k

        # 1. Graph list (ranked hydrated dicts; already scoped + boost-scored).
        graph_results = self.traversal.retrieve(
            query_plan, signal=signal,
            allowed_episode_ids=allowed_ep, allowed_document_ids=allowed_doc,
            allowed_scene_ids=allowed_scene,
        )
        graph_rank = [r["episode_id"] for r in graph_results]
        graph_by_id = {r["episode_id"]: r for r in graph_results}

        # 2. Vector list (cosine over summary embeddings). Over-fetch + scope-
        # filter when scoped, take top-k. Empty when no vector index configured.
        vector_rank: list[str] = []
        if self.vector_search is not None:
            hits = self.vector_search.search(prompt, k=fetch_k)
            if scoped:
                hits = self._filter_vector_hits_by_scope(
                    hits, allowed_ep, allowed_doc, allowed_scene)
                hits = hits[:k]
            vector_rank = [eid for eid, _ in hits]

        # 3. BM25 list (lexical over full_text). Over-fetch via fetch_k; the
        # ``allowed_ep`` set is the user-scope filter (episodes-only for A2).
        bm25_search = self.bm25
        if bm25_search is None:
            bm25_search = BM25Search(self.store.db)
            self.bm25 = bm25_search  # cache so later calls reuse it
        bm25_hits = bm25_search.search(
            tokenize(prompt), k=fetch_k, allowed_episode_ids=allowed_ep)
        bm25_rank = [eid for eid, _ in bm25_hits]

        # 4. Fuse the three ranked id-lists (RRF, rank-only -- no score-scale
        # normalization needed). Empty lists contribute nothing.
        fused = rrf_fuse([graph_rank, vector_rank, bm25_rank])
        fused = fused[:k]

        # 5. Hydrate: reuse the graph dict when the eid was a graph hit (its
        # text/salience/etc are already loaded), else read content/ep/{eid}/.
        # RRF score replaces the per-list score; ``strategy`` is additive.
        results: list[dict] = []
        for eid, rrf_score in fused:
            d = graph_by_id.get(eid)
            if d is None:
                d = self.traversal._hydrate(eid)
            d["score"] = rrf_score
            d["strategy"] = "hybrid"
            results.append(d)

        # 6. Same tail as the off path: kind-aware diversity rerank (sorts by
        # the RRF score, which preserves fused order), then optional document
        # aggregation. ``_apply_unit_boost`` is intentionally NOT called (see
        # the docstring: RRF rank semantics vs score-multiplier boost).
        results = self._kind_aware_rerank(results)
        if self.document_retriever is not None:
            results = self.document_retriever.aggregate_results(results)
        return results

    def retrieve_with_plan(self, query_plan: dict, signal: str = "routine") -> list[dict]:
        """Traverse directly with a caller-supplied plan (skips the planner).

        Lets tests exercise the traverse→load path deterministically without
        the planner (or its server fallback) in the loop. Threads the user-
        scope id sets (computed from :attr:`user_id`) so a scoped retriever
        filters here too; ``user_id=None`` -> ``None`` sets -> byte-identical.
        """
        allowed_ep, allowed_doc, allowed_scene = self._user_scope_sets()
        return self.traversal.retrieve(
            query_plan, signal=signal,
            allowed_episode_ids=allowed_ep, allowed_document_ids=allowed_doc,
            allowed_scene_ids=allowed_scene,
        )

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
        allowed_ep, allowed_doc, allowed_scene = self._user_scope_sets()
        # User-scope: over-fetch from the GLOBAL flat vector index, then filter
        # by shape against the user's owned ids, then take the top ``k``. When
        # user-scope is OFF (all allowed sets ``None``) ``fetch_k == k`` and the
        # filter is a no-op -> byte-identical.
        scoped = (allowed_ep is not None or allowed_doc is not None
                  or allowed_scene is not None)
        fetch_k = k * _USER_SCOPE_FETCH_MULT if scoped else k
        hits = self.vector_search.search_by_vector(vec, k=fetch_k)
        if scoped:
            hits = self._filter_vector_hits_by_scope(
                hits, allowed_ep, allowed_doc, allowed_scene)
            hits = hits[:k]
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

    def _semantic_fallback(
        self,
        prompt: str,
        query_plan: dict,
        allowed_episode_ids: Optional[set[str]] = None,
        allowed_document_ids: Optional[set[str]] = None,
        allowed_scene_ids: Optional[set[str]] = None,
    ) -> list[dict]:
        """Semantic fallback over summary embeddings.

        Embed ``prompt`` with the local sentence-transformers model, run
        ``self.vector_search.search``, and hydrate hits into episode dicts with
        a discounted score (×0.5) so graph-traversal matches rank higher.
        Returns ``[]`` when no vector index is configured.

        User-scope: when the allowed sets are not ``None``, over-fetch (``k *
        _USER_SCOPE_FETCH_MULT``) then filter hits by shape against the user's
        owned ids, taking the top ``k`` after filtering. When all ``None``
        (user-scope OFF) ``fetch_k == k`` and the filter is a no-op ->
        byte-identical. The sets are threaded from :meth:`retrieve` (computed
        once per call) so the graph + semantic paths share one scope read.
        """
        if self.vector_search is None:
            return []
        k = config.default_retrieval_limit
        scoped = (allowed_episode_ids is not None
                  or allowed_document_ids is not None
                  or allowed_scene_ids is not None)
        fetch_k = k * _USER_SCOPE_FETCH_MULT if scoped else k
        hits = self.vector_search.search(prompt, k=fetch_k)
        if scoped:
            hits = self._filter_vector_hits_by_scope(
                hits, allowed_episode_ids, allowed_document_ids,
                allowed_scene_ids)
            hits = hits[:k]
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
            elif kind == "scene":
                # Scene block (B1): the LLM-authored topic-level macro-memory.
                # ``summary`` is the scene's topic (its handle), ``text`` is the
                # Markdown body, ``heat`` is the scene-level forgetting signal.
                # Rendered as a peer of episodes/docs/sections in the SAME context
                # string -- scenes ride one retrieval pipeline, NOT a separate
                # [SCENE MEMORY] macro lane. The topic + heat header orients the
                # LLM; the body is the synthesized understanding.
                body = ep.get("text", "")
                chunk = (
                    f"[{eid} | {ep.get('timestamp', '')}]\n"
                    f"Scene (topic: {ep.get('summary', '')}, "
                    f"heat: {ep.get('heat', 0.0):.2f})\n"
                    f"Topics: {', '.join(ep.get('topics', []))}\n"
                    + (f"{body}\n" if body else "")
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