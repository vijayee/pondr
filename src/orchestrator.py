"""PonderOrchestrator: compose Working Memory + chunking + presentation + retrieval (Phase 2c).

The orchestrator owns the cross-query ``WorkingMemory`` instance (the retriever
has no constructor slot for it — docs/Phase 2c.md §0) and wires the 2c pipeline:

  1. embed prompt → ``working_memory.update`` (state evolves; persists across queries)
  2. route (Retrieval Gate) — or skip if no gate
  3. compress the prompt for planning (Task 5) → ``planner.plan`` (Bonsai)
  4. retrieve (graph traversal; or ``retrieve_with_routing`` if a gate is set)
  5. inject each retrieved episode into WM as a step (gist)
  6. Presentation Gate axis (a): ``plan`` chunking strategy
  6b. Presentation Gate axis (b): ``plan_end_state`` — heuristic default or caller
      override (→ ``record_override`` to the ReplayBuffer)
  7. ``SSMChunker.chunk`` → ChunkedContext
  8. ``ChunkedContextFormatter.format_for_llm``
  9. ``dispatch_end_state`` → ``direct``/``format``/``extract`` return WITHOUT an
     LLM call; only ``synthesize`` calls the generation model.

For ``ssm_direct``/``process_exec``/``tool_plan`` pathways (unsupported — no
process/tool/System-2 infra): return ``supported=False`` (honest, mirroring 2b).
``graph_retrieve``/``conscious_deliberation`` run the full pipeline.

Session save/load reuses the shipped ``state_serializer`` + ``HippocampalStore``
(per-user cross-session; the save TRIGGER policy is still open —
docs/Phase 2c.md §15, memory hippo-phase-2c-status). File-first so tests need no
WaveDB; WaveDB-backed persistence is optional (pass a ``store``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .config import Phase2cConfig
from .generation.mode_a import ModeAGenerator
from .retrieval.chunked_context import ChunkedContextFormatter
from .retrieval.end_state import dispatch_end_state
from .retrieval.expand_handler import ExpandHandler
from .retrieval.prompt_compress import compress_prompt_for_planning
from .retrieval.retriever import HippocampalRetriever
from .subconscious.presentation_gate import (
    PresentationGate, PresentationOutcome,
)
from .subconscious.ssm_chunker import SSMChunker
from .subconscious.state_serializer import (
    deserialize, serialize, snapshot_from_instance,
)
from .subconscious.working_memory import WorkingMemory, WorkingMemoryState


class PonderOrchestrator:
    """Compose the Phase 2c pipeline. Owns the cross-query Working Memory.

    The backbone, embedder, retriever, and mode_a are injected (already
    constructed) so this module imports torch only transitively through the
    subconscious package, and the retrieval/generation packages stay usable
    without a backbone configured (tests construct an orchestrator with a real
    backbone + stub embedder).
    """

    def __init__(
        self,
        store,
        retriever: HippocampalRetriever,
        backbone,
        embedder,
        mode_a: ModeAGenerator,
        config: Phase2cConfig,
        user_id: Optional[str] = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.mode_a = mode_a
        self.config = config
        self.user_id = user_id

        # The cross-query Working Memory (persistent state). embedder injected so
        # WM can embed episodes/queries on demand.
        self.working_memory = WorkingMemory(
            backbone, embedder=embedder, decay_alpha=config.working_memory.decay_alpha
        )
        self.ssm_chunker = SSMChunker(backbone, embedder, config)
        self.presentation_gate = PresentationGate(config, embedder)
        # Wire the chunker's primary-chunk cap into the gate so the gate's
        # primary_chunk_count never exceeds what the chunker will keep.
        self.presentation_gate.set_chunker_cfg(config.ssm_chunker)
        self.expand_handler = ExpandHandler(
            self.ssm_chunker, self.working_memory, store=store
        )
        self.formatter = ChunkedContextFormatter()
        self.embedder = embedder

        self.sessions_dir = Path(config.session.state_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Lazy: try to restore a saved session for this user (if a store is set).
        if store is not None and user_id is not None:
            self.load_session(user_id)

    # ── main entry ──

    def query(
        self,
        user_prompt: str,
        consumer: str = "bonsai",
        conversation_history: Optional[list[dict]] = None,
        end_state: Optional[str] = None,
        format_spec: Optional[dict] = None,
        extract_schema: Optional[dict] = None,
        model_size: Optional[str] = None,
    ) -> dict:
        """Run the full 2c pipeline and return the result dict.

        End-state dispatch: ``direct``/``format``/``extract`` return WITHOUT an
        LLM call; only ``synthesize`` (or the default when no end_state is
        specified) calls Bonsai. A caller override of the gate's end-state
        default is recorded to the override ReplayBuffer.
        """
        # 1. embed prompt; update WM (state persists across queries).
        prompt_emb = self.working_memory.embed([user_prompt])[0]
        self.working_memory.update(prompt_emb)
        self.working_memory.set_metadata("last_query_type", self._classify_query(user_prompt))
        wm_snapshot = self.working_memory.snapshot()

        # 2. compress the prompt for planning (text ≤ bonsai_max_input). Done
        #    BEFORE routing/retrieval so Bonsai (the planner) never sees >2000
        #    chars in either the gate or no-gate path (docs/Phase 2c.md §7).
        plan_prompt = compress_prompt_for_planning(
            user_prompt, working_memory=wm_snapshot, embedder=self.embedder,
            config=self.config,
        )

        # 3. route + retrieve in ONE call when a gate is wired (avoid double
        #    gate invocation); else plain retrieve. The retriever's own ``gate``
        #    is the source of truth for whether routing is available.
        route = None
        pathway = "graph_retrieve"
        gate = getattr(self.retriever, "gate", None) if self.retriever is not None else None
        if gate is not None:
            routing_result = self.retriever.retrieve_with_routing(
                plan_prompt, conversation_history=conversation_history
            )
            route = routing_result["route"]
            pathway = route.pathway
            if not routing_result["supported"]:
                # ssm_direct / process_exec / tool_plan — honest unsupported.
                return {
                    "response": None, "route": route, "retrieved_episodes": [],
                    "context_used": None, "chunked": None,
                    "working_memory_state": self.working_memory.snapshot(),
                    "presentation_plan": None, "end_state_plan": None,
                    "supported": False,
                }
            episodes = routing_result.get("results", [])
        else:
            episodes = self.retriever.retrieve(
                plan_prompt, conversation_history=conversation_history
            )

        # 4. inject each retrieved episode into WM as a gist step.
        if episodes and self.embedder is not None:
            summaries = [e.get("summary", "") or e.get("text", "") for e in episodes]
            embs = self.working_memory.embed(summaries)
            for emb in embs:
                self.working_memory.inject(emb)
            self.working_memory.set_metadata(
                "active_domains", sorted({d for e in episodes for d in e.get("topics", [])})[:5]
            )

        # 5. Presentation Gate axis (a): chunking strategy.
        presentation_plan = self.presentation_gate.plan(
            user_prompt, episodes, working_memory=wm_snapshot,
            retrieval_gate_pathway=pathway,
        )

        # 6b. Presentation Gate axis (b): end state (heuristic default or override).
        end_state_plan = self.presentation_gate.plan_end_state(
            user_prompt, episodes, working_memory=wm_snapshot,
            caller_end_state=end_state, format_spec=format_spec,
            extract_schema=extract_schema, model_size=model_size,
        )

        # 7. chunk → ChunkedContext.
        chunked = self.ssm_chunker.chunk(episodes, presentation_plan)

        # 8/9. format + dispatch on end state.
        # Reset the expand handler's per-query counter for the outcome signal.
        self.expand_handler.expand_count = 0

        # The synthesize callable: build messages and call mode_a._complete.
        def _synthesize(context: str, history: Optional[list[dict]]) -> str:
            messages: list[dict] = [{"role": "system", "content":
                "You are a helpful assistant with access to past conversations."}]
            if history:
                messages.extend(history[-10:])
            messages.append({
                "role": "user",
                "content": f"Context from past conversations:\n{context}\n\nUser: {user_prompt}",
            })
            return self.mode_a._complete(messages)

        wm_state_final = self.working_memory.snapshot()
        result = dispatch_end_state(
            end_state_plan, chunked, self.formatter, episodes, user_prompt,
            working_memory=wm_state_final, consumer=consumer,
            synthesize=_synthesize, conversation_history=conversation_history,
            max_context_tokens=4000,
        )

        # Augment with the orchestration bookkeeping the doc's §8.1 contract lists.
        result["route"] = route
        result["retrieved_episodes"] = episodes
        result["chunked"] = chunked
        result["working_memory_state"] = wm_state_final
        result["presentation_plan"] = presentation_plan
        result["end_state_plan"] = end_state_plan
        result["supported"] = result.get("supported", True)
        return result

    def _classify_query(self, prompt: str) -> str:
        """Cheap query-type tag for the WM metadata (the WM preamble)."""
        low = (prompt or "").lower()
        if any(w in low for w in ("why", "how did", "compare")):
            return "reasoning"
        if any(w in low for w in ("list", "json", "graph", "table")):
            return "extraction"
        if any(w in low for w in ("summarize", "overview", "everything")):
            return "summarization"
        return "factual"

    # ── session persistence (reuses the shipped state serializer) ──

    def save_session(self, session_id: Optional[str] = None) -> Path:
        """Persist the current WM state to disk (and optionally the store).

        ``session_id`` defaults to ``user_id``. File-first so tests need no
        WaveDB. The save TRIGGER policy is intentionally NOT wired here — the
        caller decides when to save (memory hippo-phase-2c-status).
        """
        sid = session_id or self.user_id
        if sid is None:
            raise ValueError("save_session requires a session_id or a user_id")
        snap = snapshot_from_instance(
            self.working_memory,
            input_count=self.working_memory.input_count,
            timestamp=time.time(),
            metadata=self.working_memory._metadata,
        )
        blob = serialize(snap)
        path = self.sessions_dir / f"{sid}.json"
        path.write_text(blob, encoding="utf-8")
        # Optional WaveDB-backed persistence (per-user cross-session).
        if self.store is not None:
            self.store.save_jgs_state(sid, blob, scope="working_memory")
        return path

    def load_session(self, session_id: Optional[str] = None) -> bool:
        """Restore WM state from disk (or the store). Returns False if none saved."""
        sid = session_id or self.user_id
        if sid is None:
            return False
        # Store first (the per-user cross-session source of truth); fall back to disk.
        blob = None
        if self.store is not None:
            blob = self.store.load_jgs_state(sid, scope="working_memory")
        if not blob:
            path = self.sessions_dir / f"{sid}.json"
            if path.exists():
                blob = path.read_text(encoding="utf-8")
        if not blob:
            return False
        snap = deserialize(blob)
        self.working_memory.reset()  # ensure state is initialized, then overwrite
        self.working_memory.restore(snap)
        return True

    # ── EXPAND (delegated to the handler) ──

    def expand(self, episode_id: str, chunked) -> tuple[str, WorkingMemoryState]:
        """EXPAND a compressed episode: load full text + inject into WM."""
        return self.expand_handler.handle_expand(episode_id, chunked)

    # ── outcome recording ──

    def record_outcome(
        self,
        presentation_plan,
        expand_count: int = 0,
        unused_primary_count: int = 0,
        user_satisfaction: float = 0.0,
    ) -> None:
        """Record a presentation outcome to the gate's buffer (seeds a future gate)."""
        self.presentation_gate.record_outcome(
            presentation_plan,
            PresentationOutcome(
                expand_count=expand_count,
                unused_primary_count=unused_primary_count,
                user_satisfaction=user_satisfaction,
            ),
        )