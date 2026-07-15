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
(per-user cross-session). The runtime gap is closed (2026-07-14): ``query``
now persists each (prompt, response) exchange as a new episode via an injected
``HippocampalEncoder`` (always-encode by default; ``auto_persist=False`` opts
out; ``end_conversation`` closes the conversation session). Pure DI -- the
caller that wants live-encode constructs and injects the encoder; no encoder
injected (tests, WM-only) -> no-op. File-first so tests need no WaveDB;
WaveDB-backed persistence is optional (pass a ``store``).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .config import Phase2cConfig, config as _runtime_config
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
from .tools import (
    SELF_CHAT_TOOLS, dispatch_tool, feedback_instruction,
)

if TYPE_CHECKING:
    from .encoding.encoder import HippocampalEncoder


# Signal -> persistence profile (2026-07-14). The ``signal`` arg modulates HOW
# strongly a live-encoded episode persists, not WHETHER (always-encode is the
# default; ``auto_persist=False`` opts out). ``utility_decay_rate`` is the lever
# the forgetting dream pass fades (``utility_score *= (1 - decay_rate)**days``);
# ``salience`` feeds the heuristic scorer + entity-salience compose. Unknown
# signals fall back to the ``routine`` defaults (the Episode field defaults).
_SIGNAL_PROFILES = {
    "important":   {"salience": 0.8, "decay_rate": 0.005},   # persists longest
    "routine":     {"salience": 0.5, "decay_rate": 0.01},    # Episode defaults
    "satisfied":   {"salience": 0.7, "decay_rate": 0.008},
    "correction":  {"salience": 0.6, "decay_rate": 0.008},
    "frustration": {"salience": 0.3, "decay_rate": 0.03},    # fades fastest
}


def _parse_json_array(text: str) -> list[dict]:
    """Best-effort extraction of a JSON array of objects from a model reply.

    The fallback rating call asks for a bare JSON array, but a small model may
    wrap it in prose or fences. This finds the first ``[`` ... ``]`` span and
    parses it, then keeps only dicts with a ``unit_id``. Returns ``[]`` on any
    failure (the caller treats empty as no-op, not an error).
    """
    if not text:
        return []
    s = text.strip()
    # Strip a code fence if present.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Find the first balanced [...] span (the array the model was asked for).
    start = s.find("[")
    if start == -1:
        return []
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(s[start:i + 1])
                except (ValueError, TypeError):
                    return []
                if not isinstance(parsed, list):
                    return []
                return [o for o in parsed
                        if isinstance(o, dict) and o.get("unit_id")]
    return []


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
        encoder: Optional[HippocampalEncoder] = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.mode_a = mode_a
        self.config = config
        self.user_id = user_id
        # Live-encode (2026-07-14): persist each exchange as an episode. The
        # encoder is injected (DI pattern, like retriever/mode_a/embedder) -- a
        # caller that wants live-encode constructs a ``HippocampalEncoder`` and
        # passes it here; ``query(auto_persist=True)`` then encodes every
        # exchange. ``None`` (tests, WM-only) -> no-op. Pure DI (no lazy heavy
        # construction) so ``query()`` never loads GLiNER unless a real encoder
        # was explicitly wired in.
        self._encoder = encoder

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
            # Phase 3a Task 7: also restore the durable presentation-outcome
            # buffers (EXPAND-frequency signal) so they survive restarts.
            self.load_outcomes(user_id)

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
        signal: str = "routine",
        auto_persist: bool = True,
    ) -> dict:
        """Run the full 2c pipeline and return the result dict.

        End-state dispatch: ``direct``/``format``/``extract`` return WITHOUT an
        LLM call; only ``synthesize`` (or the default when no end_state is
        specified) calls Bonsai. A caller override of the gate's end-state
        default is recorded to the override ReplayBuffer.

        ``signal`` (Phase 3b) is the caller's affective/task signal
        (``important``/``routine``/``correction``/...) threaded through to the
        retrieval-boost hook so query-matched edges strengthen with use, AND
        modulating how strongly the live-encoded episode persists (salience +
        decay rate; see ``_SIGNAL_PROFILES``). Defaults to ``"routine"`` (a
        no-op until something is actually retrieved).

        ``auto_persist`` (default True) encodes the (prompt, response) exchange
        as a new episode after the response is built (closes the runtime gap --
        the system learns from use). Set False to opt out. Best-effort: a
        persistence failure is logged and never loses the response. The encoded
        episode id (when persisted) is returned as ``result["persisted_episode_id"]``.
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
                plan_prompt, conversation_history=conversation_history, signal=signal,
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
                plan_prompt, conversation_history=conversation_history, signal=signal,
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
        # Phase 2c+: pass the record_feedback tool so the model can rate the
        # context units it used; after the call, dispatch any record_feedback
        # tool calls. If the model emits none (Bonsai tool-calling may be
        # unsupported on a Q2_0 8B), fall back to one small structured rating
        # call. Best-effort: a feedback failure never loses the response.
        feedback_enabled = _runtime_config.feedback_salience_enabled
        feedback_state = {"count": 0}

        def _synthesize(context: str, history: Optional[list[dict]]) -> str:
            messages: list[dict] = [{"role": "system", "content":
                "You are a helpful assistant with access to past conversations."}]
            if history:
                messages.extend(history[-10:])
            user_content = f"Context from past conversations:\n{context}\n\nUser: {user_prompt}"
            if feedback_enabled:
                user_content += "\n\n" + feedback_instruction(episodes)
            messages.append({"role": "user", "content": user_content})
            tools = SELF_CHAT_TOOLS if feedback_enabled else None
            try:
                content, tool_calls = self.mode_a._complete(messages, tools=tools)
            except Exception as e:  # noqa: BLE001 - generation failure -> empty answer
                print(f"[synthesize-fail] {e}", file=sys.stderr)
                return ""
            content = content or ""
            if feedback_enabled:
                feedback_state["count"] = self._dispatch_feedback(
                    tool_calls, episodes, content, feedback_state["count"]
                )
            return content

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

        # Phase 3a Task 7: auto-record the presentation outcome with the
        # MEASURED expand_count (the durable salience signal from 2c §15).
        # ``unused_primary_count`` and ``user_satisfaction`` are NOT directly
        # measured here (we don't observe which primary chunks the model attended
        # to, nor collect a satisfaction rating) — they stay 0 (caller-supplied
        # via ``record_outcome`` if available). ``expand_count`` is the real
        # durable signal. Recording happens after every query so the buffer is
        # populated without a caller remembering to call ``record_outcome``.
        measured_expand = int(getattr(self.expand_handler, "expand_count", 0))
        self.presentation_gate.record_outcome(
            presentation_plan,
            PresentationOutcome(
                expand_count=measured_expand,
                unused_primary_count=0,   # not measured (see above)
                user_satisfaction=0.0,    # not measured (caller-supplied)
            ),
        )
        result["measured_expand_count"] = measured_expand
        # Phase 2c+: how many record_feedback judgments were applied this turn
        # (0 when feedback is disabled, the model emitted none, or the fallback
        # also yielded nothing). Observability only -- never blocks the response.
        result["feedback_collected"] = feedback_state["count"]

        # 2026-07-14: close the runtime gap -- persist the (prompt, response)
        # exchange as a new episode so the system learns from use. Always-encode
        # by default; ``auto_persist=False`` opts out. Best-effort: a persistence
        # failure never loses the response the user already has.
        if auto_persist:
            self._persist_exchange(user_prompt, result, signal)
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

    # ── live-encode: persist each exchange as an episode (2026-07-14) ──

    def _get_encoder(self):
        """Return the injected HippocampalEncoder, or ``None``.

        ``None`` when no encoder was injected (tests, WM-only orchestrator) --
        ``_persist_exchange`` then no-ops. Pure DI: the caller that wants
        live-encode constructs and injects the encoder (mirrors retriever/
        mode_a/embedder). No lazy construction, so ``query()`` never loads
        GLiNER unless a real encoder was explicitly wired in.
        """
        if self.store is None:
            return None
        return self._encoder

    def _persist_exchange(self, user_prompt: str, result: dict, signal: str) -> None:
        """Encode the (prompt, response) exchange as a new episode.

        Best-effort: any failure is logged to stderr and swallowed -- a
        persistence failure must never lose the response the user already has.
        Skips when there is no encoder, or when the result carries no
        non-empty string response (the ``direct``/``format``/``extract`` end
        states that produce no string, and the ``supported=False`` early
        return with ``response: None``).
        """
        try:
            encoder = self._get_encoder()
            if encoder is None:
                return
            response = result.get("response")
            if not isinstance(response, str) or not response.strip():
                return
            if encoder.session_id is None:
                encoder.start_session()  # one conversation session per instance
            prof = _SIGNAL_PROFILES.get(signal, _SIGNAL_PROFILES["routine"])
            # Role-tagged segments (OpenAI vocabulary). Today: user + assistant.
            # system (boilerplate prompt) and tool/tool_call are reserved --
            # appended here when those pathways are wired, not as flat strings.
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
            # ``working_memory.embed`` returns [1,384] TENSORS (not JSON-serializable
            # for the store's summary_embedding); pass the raw embedder's ``.encode``
            # instead, which yields one 1-D float vector per text. The encoder
            # coerces it to ``list[float]`` for JSON persistence.
            raw_encode = self.embedder.encode if self.embedder is not None else None
            episode = encoder.encode_messages(
                messages,
                origin="live",
                salience=prof["salience"],
                utility_decay_rate=prof["decay_rate"],
                embedder=raw_encode,
                degrade_on_extract_fail=True,
            )
            result["persisted_episode_id"] = episode.id
        except Exception as e:  # noqa: BLE001 - never lose the response
            print(f"[persist-fail] {e}", file=sys.stderr)

    # ── Phase 2c+: feedback salience + consumer tool surface ──

    def _dispatch_feedback(
        self,
        tool_calls: Optional[list[dict]],
        episodes: list[dict],
        content: str,
        already: int,
    ) -> int:
        """Dispatch any ``record_feedback`` tool calls; fall back if none.

        Self-chat feedback path: if the synthesis returned ``record_feedback``
        tool calls, dispatch each via ``dispatch_tool`` (-> ``store.record_feedback``).
        If the model emitted NONE (Bonsai tool-calling may be unsupported on a
        Q2_0 8B), make ONE small structured rating call asking only for a JSON
        array of {unit_id, rating}, parse it best-effort, and apply it. Best-
        effort: any failure is logged and swallowed -- a feedback failure never
        loses the response. Returns the cumulative count applied this turn.
        """
        if not _runtime_config.feedback_salience_enabled or self.store is None:
            return already
        count = already
        if tool_calls:
            for call in tool_calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                if fn.get("name") == "record_feedback":
                    result = dispatch_tool(self, "record_feedback", fn.get("arguments", {}))
                    try:
                        parsed = json.loads(result) if isinstance(result, str) else {}
                        count += int(parsed.get("applied", 0))
                    except (ValueError, TypeError):
                        pass
            if count > already:
                return count  # tool path worked -- skip the fallback
        # Fallback: no usable record_feedback tool call -> one structured call.
        if not episodes:
            return count
        try:
            count += self._feedback_fallback_call(episodes, content)
        except Exception as e:  # noqa: BLE001 - fallback is best-effort
            print(f"[feedback-fallback-fail] {e}", file=sys.stderr)
        return count

    def _feedback_fallback_call(self, episodes: list[dict], content: str) -> int:
        """One structured rating call when Bonsai emits no tool call.

        Asks the model ONLY for a JSON array of ``{"unit_id","rating"}`` over the
        cited units (capped), parses best-effort, and applies it via
        ``store.record_feedback``. Returns the count applied. The model's prior
        ``content`` (the answer) is included so the rating is grounded in what it
        actually said. No tools passed (the fallback exists precisely because
        tool-calling may be unsupported).
        """
        cap = 12
        units = [
            {"unit_id": e.get("episode_id", ""), "kind": e.get("kind", "episode")}
            for e in episodes[:cap] if e.get("episode_id")
        ]
        if not units:
            return 0
        lines = [
            "Rate how useful each cited memory unit was for the answer you just "
            "gave, on a 1-5 scale (1=useless, 3=neutral, 5=essential). Be critical. "
            "Reply with ONLY a JSON array, no prose, of objects like "
            '{"unit_id":"<id>","rating":5}. Units:',
        ]
        for u in units:
            lines.append(f'- {u["unit_id"]}')
        prompt = "\n".join(lines)
        messages = [
            {"role": "system", "content": "You rate memory units for usefulness."},
            {"role": "user", "content": f"Your answer was:\n{content[:1500]}\n\n{prompt}"},
        ]
        text, _ = self.mode_a._complete(messages)
        if not text:
            return 0
        judgments = _parse_json_array(text)
        if not judgments:
            return 0
        return self.store.record_feedback(judgments)

    def expand_unit(self, unit_id: str) -> Optional[str]:
        """Consumer tool: return the FULL text of a retrieved unit.

        Resolves the unit by its id shape: an episode (``ep_*``) -> the episode
        text; a section (``{doc_id}_sec_NNN``) -> the section body (cold pull);
        a document (``doc_*``) -> the doc with all section bodies loaded. The
        external LLM calls this via ``dispatch_tool("expand", ...)`` to pull a
        compressed gist's full text. Returns ``None`` for a missing unit
        (``dispatch_tool`` turns that into an error string).
        """
        if not unit_id or self.store is None:
            return None
        try:
            if "_sec_" in unit_id:
                # A section id: ``{doc_id}_sec_{i:03d}``. Split on the FIRST
                # ``_sec_`` so a doc_id containing ``_sec_`` (unlikely) still
                # resolves -- the section id is the full compound string.
                head, _, _rest = unit_id.partition("_sec_")
                doc_id = head
                return self.store.get_section_body(doc_id, unit_id)
            if unit_id.startswith("doc_"):
                doc = self.store.get_document(unit_id, load_bodies=True)
                if doc is None:
                    return None
                parts = [f"Title: {doc.title}", f"Source: {doc.source_path}"]
                for sec in doc.sections:
                    head = sec.heading or "(section)"
                    parts.append(f"\n## {head}\n{sec.content}")
                return "\n".join(parts)
            # Episode: return summary + full text.
            ep = self.store.get_episode(unit_id)
            if ep is None:
                return None
            parts = []
            if ep.summary:
                parts.append(f"Summary: {ep.summary}")
            if ep.full_text:
                parts.append(ep.full_text)
            return "\n".join(parts)
        except Exception as e:  # noqa: BLE001 - expand is best-effort
            print(f"[expand-fail] {e}", file=sys.stderr)
            return None

    def search_memory(
        self,
        query: str,
        entities: Optional[list[str]] = None,
        topics: Optional[list[str]] = None,
    ) -> str:
        """Consumer tool: re-retrieve mid-generation with a refined query/axes.

        Runs the retriever with a literal query plan (the entities/topics axes
        the consumer named) and builds the context string. The external LLM
        calls this via ``dispatch_tool("search_memory", ...)`` when the initial
        context was insufficient. Returns the formatted context (empty string
        when nothing is found).
        """
        if self.retriever is None or not query:
            return ""
        try:
            plan = {
                "entities": entities or [],
                "topics": topics or [],
                "tones": [],
                "entity_mode": "union",
                "limit": _runtime_config.default_retrieval_limit,
            }
            results = self.retriever.retrieve_with_plan(plan)
            if not results:
                return ""
            return self.retriever.build_context_string(results)
        except Exception as e:  # noqa: BLE001 - search is best-effort
            print(f"[search_memory-fail] {e}", file=sys.stderr)
            return ""

    def end_conversation(self) -> None:
        """Close the live-encode conversation session.

        Caller-invoked at conversation boundaries (mirrors the open save-trigger
        policy -- the caller decides when a conversation ends). An unclosed
        session is graceful, not broken: episodes still carry ``at_time``; only
        ``ended_at`` is absent. No-op when no encoder or no open session.
        """
        encoder = self._get_encoder()
        if encoder is None or encoder.session_id is None:
            return
        encoder.end_session()

    # ── session persistence (reuses the shipped state serializer) ──

    def save_session(self, session_id: Optional[str] = None) -> Path:
        """Persist the current WM state to disk (and optionally the store).

        ``session_id`` defaults to ``user_id``. File-first so tests need no
        WaveDB. This persists the WM SSM state (the caller decides when); it is
        distinct from the per-exchange episode persistence, which ``query``
        does automatically (``auto_persist``).
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

    # ── Phase 3b: active-forget + reconsolidation API ──

    def forget(self, episode_id: str, validity_end: "Optional[str]" = None) -> None:
        """Active-forget an episode: deprecate, never delete.

        Sets ``content/ep/{eid}/state = "deprecated"`` (+ ``validity_end`` if
        given) via ``store.set_episode_state``. The episode stops appearing in
        default queries (the ``default_episode_ids`` state/validity filter) and
        in axis queries (``is_episode_active``); its content + graph triples
        are untouched, so it stays retrievable via ``include_inactive=True`` and
        reversible (a subsequent ``set_episode_state(..., "current")`` revives
        it). No store configured is a no-op (WM-only orchestrator).
        """
        if self.store is None:
            return
        self.store.set_episode_state(episode_id, "deprecated", validity_end=validity_end)

    def reconsolidate(
        self,
        old_episode_id: str,
        new_episode_id: str,
        validity_end: "Optional[str]" = None,
    ) -> None:
        """Record that ``new_episode_id`` supersedes ``old_episode_id``.

        Writes the MVCC supersession chain atomically: the ``supersedes`` (new
        -> old) + ``superseded_by`` (old -> new) graph edges and the old
        episode's ``state="superseded"`` + ``validity_end``. The old episode
        drops out of default/axis queries; the new one (encoded by the caller)
        stays ``current``. Contradiction-resolution and active reconsolidation
        both land here. No store configured is a no-op. See
        ``SemanticMemoryWriter.supersede_episode``.
        """
        if self.store is None:
            return
        from .gnn.semantic_memory import SemanticMemoryWriter
        SemanticMemoryWriter(self.store).supersede_episode(
            new_episode_id, old_episode_id, when=validity_end,
        )

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

    # ── presentation-outcome persistence (Phase 3a Task 7) ──

    def save_outcomes(self, user_id: Optional[str] = None) -> Optional[str]:
        """Persist the gate's outcome/override buffers to the store (durable signal).

        Returns the blob, or ``None`` if no store or user is configured. The
        save TRIGGER policy mirrors ``save_session`` — the caller decides when
        (e.g. at session end / periodically); ``query()`` auto-records into the
        in-memory buffer, and this method flushes it to disk.
        """
        sid = user_id or self.user_id
        if sid is None or self.store is None:
            return None
        import json
        blob = json.dumps(self.presentation_gate.serialize_buffers(), ensure_ascii=False)
        self.store.save_presentation_outcomes(sid, blob)
        return blob

    def load_outcomes(self, user_id: Optional[str] = None) -> bool:
        """Restore the gate's outcome/override buffers from the store. False if none."""
        import json
        sid = user_id or self.user_id
        if sid is None or self.store is None:
            return False
        blob = self.store.load_presentation_outcomes(sid)
        if not blob:
            return False
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            return False
        self.presentation_gate.load_buffers(data)
        return True