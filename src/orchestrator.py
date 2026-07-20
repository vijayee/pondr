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

import torch
from torch import Tensor

from .config import Phase2cConfig, config as _runtime_config
from .generation.mode_a import ModeAGenerator
from .retrieval.chunked_context import ChunkedContextFormatter
from .retrieval.end_state import dispatch_end_state
from .retrieval.expand_handler import ExpandHandler
from .retrieval.prompt_compress import compress_prompt_for_planning
from .retrieval.retriever import HippocampalRetriever
from .subconscious.presentation_gate import (
    CHUNKED, DIRECT, PresentationGate, PresentationOutcome, PresentationPlan,
)
from .subconscious.salience import format_salience_gap
from .subconscious.ssm_chunker import SSMChunker
from .subconscious.state_serializer import (
    deserialize, serialize, snapshot_from_instance,
)
from .subconscious.recoverability_head import pool_state_tensors
from .subconscious.relevance_score import (
    score_ring_slots, score_ring_slots_with_doc_embs,
)
from .subconscious.working_memory import WorkingMemory, WorkingMemoryState
from .tools import (
    LOOP_TOOLS, SELF_CHAT_TOOLS, TOOL_SCHEMAS, dispatch_tool,
    feedback_instruction, run_tool_loop,
)

if TYPE_CHECKING:
    from .encoding.encoder import HippocampalEncoder

from .encoding.distill_worker import DistillWorker


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


def _sum_record_feedback_applied(collected: Optional[list[dict]]) -> int:
    """Sum the ``applied`` counts from a tool loop's ``record_feedback`` calls.

    ``run_tool_loop`` returns ``collected`` as ``[{"name", "result"}, ...]`` where
    ``result`` is the ``dispatch_tool`` return string. ``record_feedback``'s
    result is JSON ``{"ok": True, "applied": N}``; this parses each and sums
    ``N``. Mirrors the per-call parse in ``_dispatch_feedback``. Non-
    ``record_feedback`` entries (expand / search_memory) and any parse failure
    contribute 0 -- a retrieval-only loop yields 0, which triggers the
    structured fallback (mirroring the one-shot path's "skip fallback if a tool
    worked" early-return).
    """
    total = 0
    if not collected:
        return total
    for entry in collected:
        if not isinstance(entry, dict) or entry.get("name") != "record_feedback":
            continue
        result = entry.get("result")
        try:
            parsed = json.loads(result) if isinstance(result, str) else {}
            total += int(parsed.get("applied", 0))
        except (ValueError, TypeError):
            pass
    return total


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
        relevance_head=None,
        graduation_proxy=None,
        graduation_head=None,
        recoverability_head=None,
        latent_dynamics_head=None,
        ring_capacity: Optional[int] = None,
        context_builder=None,
        strm_salience: bool = False,
        salience_thresholds=None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.mode_a = mode_a
        self.config = config
        self.user_id = user_id
        # STRM Phase 2a relevance head (optional, DI like the encoder). When
        # wired it scores each WM ring slot's relevance to the current query
        # (``r_i in [0,1]``); Phase 3's context-builder consumes ``r_i`` as the
        # slot-selection bias. ``None`` (default, flag off / no checkpoint) ->
        # no relevance scoring at serve (byte-identical to pre-2a).
        self.relevance_head = relevance_head
        # STRM Phase 2d v1 graduation proxy (optional, DI like the encoder).
        # When wired it scores each WM ring slot's graduation (the
        # parameter-free ``integral(r_i dt)`` heuristic the v2 head must beat);
        # Phase 4's LTM-promotion path consumes the decision. ``None`` (default,
        # flag off) -> no graduation scoring at serve (byte-identical to pre-2d).
        self.graduation_proxy = graduation_proxy
        # STRM Phase 2d v2 graduation head (optional, DI like the proxy). When
        # wired it scores each WM ring slot's ``later_needed`` probability (the
        # learned classifier the v1 proxy is the baseline for). Phase 4's LTM-
        # promotion path consumes the decision; this round only attaches it
        # (completes the full serve-wiring of all STRM read-out heads).
        # ``None`` (default, flag off / no checkpoint) -> no v2 graduation
        # scoring at serve (byte-identical to pre-Phase-4).
        self.graduation_head = graduation_head
        # STRM Phase 2b recoverability head (optional, DI like the relevance
        # head). When wired it scores how forgotten a past anchor is from the
        # live WM pooled state; Phase 4's salience trigger consumes the
        # ``recoverability < theta`` term (low = likely forgotten = salient).
        # ``None`` (default, flag off / no checkpoint) -> no recoverability
        # scoring at serve (byte-identical to pre-Phase-4).
        self.recoverability_head = recoverability_head
        # STRM Phase 2c latent-dynamics head (optional, DI like the relevance
        # head). When wired it predicts the next WM state + emits a per-turn
        # surprise signal; Phase 4's salience trigger consumes the
        # ``surprise < surprise_cap`` term (high surprise -> suppress).
        # ``None`` (default, flag off / no checkpoint) -> no latent-dynamics
        # scoring at serve (byte-identical to pre-Phase-4).
        self.latent_dynamics_head = latent_dynamics_head
        # STRM Phase 3 context-builder (optional, DI like the relevance head).
        # When wired it attends over the WM ring with the 2a ``r_i`` as an
        # additive bias and selects top-m primary context instead of the
        # heuristic PresentationGate (see ``_plan_with_context_builder``).
        # Requires the ring ON + a relevance head; any exception / empty ring /
        # no matching slots falls back to the heuristic so the turn never
        # crashes. ``None`` (default, flag off) -> heuristic PresentationGate
        # (byte-identical to pre-3).
        self.context_builder = context_builder
        # STRM Phase 4 salience trigger (Step 4). When ``strm_salience`` is ON
        # AND all three read-out heads (2a relevance, 2b recoverability, 2c
        # latent-dynamics) are wired AND the ring is ON AND thresholds are
        # loaded, the pre-retrieval hook (``_run_salience_hook``) scores every
        # ring slot for salience and stashes the anchors here for Step 5
        # (state-conditioned retrieval + pin-tagged re-inject) and Step 6
        # (freshness watermark + stale-uncertain signal). ``strm_salience=False``
        # (the default) -> the hook never runs -> ``_salience_anchors`` stays
        # None -> byte-identical to pre-Step-4. Best-effort: any failure in the
        # hook is swallowed (anchors stay None, the turn proceeds unchanged).
        self.strm_salience = bool(strm_salience)
        self.salience_thresholds = salience_thresholds
        self._salience_anchors = None
        # Step 5: episodes the salience trigger proactively recalled from LTM
        # (state-conditioned, pin-tagged re-inject). Reset to None each turn;
        # merged into the prompt-driven ``episodes`` (salience first, dedup by
        # episode_id) and injected with ``pin=True`` so W_A retains them. None
        # when the trigger is off / disarmed / failed -> no merge -> byte-
        # identical to pre-Step-5.
        self._salience_fired_episodes: Optional[list] = None
        # Step 6: freshness watermark. ``_turn_count`` increments per armed
        # query; ``_source_entry_turn`` records the turn each source_id first
        # appeared in the ring at salience-scoring time (age = turn_count -
        # entry_turn). A young anchor (age < strm_salience_freshness_lag) whose
        # retrieval returned nothing emits a ``stale_uncertain`` signal instead
        # of being silently suppressed (the episode may be known but not yet
        # fully ingested by Thread 2's async-distill worker). Both inert when
        # the trigger is off / disarmed -> byte-identical.
        self._salience_turn_count = 0
        self._source_entry_turn: dict = {}
        # Step 6: the per-turn salience signals (recall | stale_uncertain) +
        # the consumer-facing gap text. None when off / disarmed / failed ->
        # absent from the result dict -> byte-identical.
        self._salience_signals: Optional[list] = None
        # Live-encode (2026-07-14): persist each exchange as an episode. The
        # encoder is injected (DI pattern, like retriever/mode_a/embedder) -- a
        # caller that wants live-encode constructs a ``HippocampalEncoder`` and
        # passes it here; ``query(auto_persist=True)`` then encodes every
        # exchange. ``None`` (tests, WM-only) -> no-op. Pure DI (no lazy heavy
        # construction) so ``query()`` never loads GLiNER unless a real encoder
        # was explicitly wired in.
        self._encoder = encoder
        # Async episode distillation (Phase 3c): when ``async_distill_enabled``
        # is on AND an encoder is wired, a single-worker background FIFO fills
        # each turn's graph edges after the response returns (the 22 s
        # extraction runs off-thread; the stub content + vector index is written
        # synchronously so the turn is retrievable immediately). ``None`` (the
        # default, flag off, or no encoder) -> the synchronous
        # ``_persist_exchange`` path, byte-identical to pre-async. See
        # async-distill-stub.md + src/encoding/distill_worker.py.
        self._distill_worker: Optional[DistillWorker] = None
        if encoder is not None and getattr(_runtime_config, "async_distill_enabled", False):
            self._distill_worker = DistillWorker(encoder, store)
        # Self-chat tool-loop transcript surfaced onto the query result (D6).
        # Declared here so the attribute always exists; ``query`` resets it to
        # None before the synthesize call and sets it to the loop dict only when
        # the loop path ran (the non-loop path leaves it None).
        self._last_loop = None
        # STRM 2a raw-rating tap: the current user query, set at the top of
        # query() and cleared on every return path so the record_feedback tool
        # path can thread it into feedback.jsonl. None outside a query.
        self._current_query: Optional[str] = None
        # STRM 2d replay-logger turn counter: a per-orchestrator monotonic id so
        # the v2 graduation label generator can order turns within a session
        # (the WM ring's FIFO eviction makes "compressed out then re-recalled"
        # a turn-gap question). Incremented once per query when
        # ``strm_graduation_logging`` is on; untouched (stays 0) when off, so
        # the flag-off path is byte-identical (the logger never runs).
        self._graduation_turn_counter: int = 0

        # The cross-query Working Memory (persistent state). embedder injected so
        # WM can embed episodes/queries on demand. ``ring_capacity`` overrides
        # the instance config's ring_capacity (default None -> config, which is 0
        # = ring OFF, byte-identical to Phase 2c). The STRM 2a/2d serve-time
        # read-out heads (relevance scoring, graduation replay logging) need the
        # ring ON; a serve flag threads a K>0 here so the ring populates. K=0
        # (the default) keeps the ring off and the shipped path byte-identical.
        self.working_memory = WorkingMemory(
            backbone, embedder=embedder, decay_alpha=config.working_memory.decay_alpha,
            ring_capacity=ring_capacity,
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

    # ── STRM Phase 4 Step 4: salience trigger ──

    def _salience_armed(self) -> bool:
        """True iff the salience trigger has everything it needs to run: the
        ``--strm-salience`` flag is on, all three read-out heads (2a relevance,
        2b recoverability, 2c latent-dynamics) are wired, the thresholds sidecar
        is loaded, and the ring is ON (salience reads ring slots). A missing
        piece disarms the trigger -- the salience AND needs all three scores, so
        a missing head means no anchor can be salient. Flag-off (the default)
        disarms here so the query() seam skips the state capture + hook entirely
        (byte-identical to pre-Step-4)."""
        return (
            self.strm_salience
            and self.relevance_head is not None
            and self.recoverability_head is not None
            and self.latent_dynamics_head is not None
            and self.salience_thresholds is not None
            and self.working_memory.ring_capacity > 0
        )

    def _run_salience_hook(self, prompt_emb, prev_state_tensors, signal: str) -> None:
        """Score the WM ring for salience + fire state-conditioned retrieval.

        Step 4: compute ``SalienceAnchor`` per ring slot and stash on
        ``self._salience_anchors``.
        Step 5: for each SALIENT anchor (budget-capped), fire
        ``retrieve_by_embedding`` with the anchor's 384-d doc vector as the
        state-conditioned query (the episode the WM state flagged as being-
        forgotten), dedup by ``episode_id``, and stash the merged list on
        ``self._salience_fired_episodes`` -- the caller merges it into the
        prompt-driven ``episodes`` (salience first) and injects with
        ``pin=True`` so ``W_A`` retains the proactive recall.
        Step 6: per salient anchor, emit a typed consumer signal on
        ``self._salience_signals`` -- ``recall`` (retrieval returned hits) or
        ``stale_uncertain`` (young anchor, age < ``strm_salience_freshness_lag``,
        retrieval returned nothing -- the episode may still be ingesting by
        Thread 2's async-distill worker; do not lie by omission). An OLD anchor
        that got nothing back is silently dropped. The caller surfaces
        ``salience_signals`` + ``format_salience_gap(...)`` in the result dict.

        Flag-off (when the hook never runs) is byte-identical: no anchors, no
        fired episodes, no signals, no merge. Best-effort: any failure is
        swallowed (``_salience_anchors`` AND ``_salience_fired_episodes`` AND
        ``_salience_signals`` stay ``None``, the turn proceeds unchanged) -- a
        proactive-recall heuristic must never crash the turn. Caller (``query``)
        only invokes this when ``_salience_armed``.
        """
        try:
            ring_slots = self.working_memory.ring_buffer()
            if not ring_slots:
                self._salience_anchors = None
                self._salience_fired_episodes = None
                self._salience_signals = None
                return
            state_tensors = self.working_memory.state_tensors()
            from src.subconscious.salience import (
                compute_salience,
                salient_anchors,
                SALIENCE_RETRIEVAL_BUDGET,
            )
            self._salience_anchors = compute_salience(
                ring_slots=ring_slots,
                state_tensors=state_tensors,
                prev_state_tensors=prev_state_tensors,
                working_memory=self.working_memory,
                relevance_head=self.relevance_head,
                recoverability_head=self.recoverability_head,
                latent_dynamics_head=self.latent_dynamics_head,
                embedder=self.embedder,
                query_emb=prompt_emb,
                thresholds=self.salience_thresholds,
            )
            # Step 5: fire state-conditioned retrieval per salient anchor.
            # Step 6: track per-anchor retrieval outcome + age to emit the
            # ``recall`` (got hits) / ``stale_uncertain`` (young anchor, no hits)
            # consumer signals. A young anchor (age < freshness lag) that got
            # nothing back is NOT silently suppressed -- the episode may be known
            # but not yet fully ingested by Thread 2's async-distill worker, so
            # surface a stated gap (proposal sec 5: don't lie by omission). An
            # OLD anchor that got nothing back is silently dropped (it had its
            # chance). A retrieval exception is treated as no hits.
            lag = int(getattr(_runtime_config, "strm_salience_freshness_lag", 3))
            fired: list[dict] = []
            seen_ids: set = set()
            signals: list[dict] = []
            for anchor in salient_anchors(self._salience_anchors)[:SALIENCE_RETRIEVAL_BUDGET]:
                sid = anchor.source_id
                # Freshness watermark: record the turn this source_id first
                # appeared at salience-scoring time. age = turns since.
                if sid is not None and sid not in self._source_entry_turn:
                    self._source_entry_turn[sid] = self._salience_turn_count
                age = (self._salience_turn_count
                       - self._source_entry_turn.get(sid, self._salience_turn_count))
                hits: list = []
                if anchor.doc_emb is not None and self.retriever is not None:
                    try:
                        hits = self.retriever.retrieve_by_embedding(
                            anchor.doc_emb, signal=signal,
                        )
                    except Exception:  # noqa: BLE001 - per-anchor best-effort
                        hits = []
                    for ep in hits:
                        eid = ep.get("episode_id")
                        if eid is None or eid in seen_ids:
                            continue
                        seen_ids.add(eid)
                        fired.append(ep)
                got_hits = bool(hits)
                # Emit the per-anchor consumer signal.
                if got_hits:
                    kind = "recall"
                elif age < lag:
                    # Young + failed -> stale-uncertain (the episode may still be
                    # ingesting). Not suppressed.
                    kind = "stale_uncertain"
                else:
                    # Old + failed -> silently dropped (had its chance).
                    continue
                signals.append({
                    "anchor_source_id": sid,
                    "kind": kind,
                    "text": anchor.text,
                    "r_i": anchor.r_i,
                    "rec_i": anchor.rec_i,
                    "age": age,
                })
            self._salience_fired_episodes = fired
            self._salience_signals = signals
            # Prune the entry-turn watermark to source_ids still in the ring so
            # it does not grow unbounded across a long session.
            live = {s.source_id for s in ring_slots if s.source_id is not None}
            self._source_entry_turn = {
                k: v for k, v in self._source_entry_turn.items() if k in live
            }
        except Exception:
            # A proactive-recall heuristic must never crash the turn. Swallow,
            # leave no anchors + no fired episodes + no signals, and the rest of
            # query() proceeds unchanged.
            self._salience_anchors = None
            self._salience_fired_episodes = None
            self._salience_signals = None

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
        # Foreground-priority yielding (Phase 3c async-distill): mark the
        # foreground busy for the duration of the response build so the
        # background distill worker's GPU steps (GLiNER + 10-pass Bonsai) block
        # and run only in the gaps between turns. Cleared at return. No-op when
        # there is no worker (the synchronous default).
        if self._distill_worker is not None:
            self._distill_worker.foreground_busy.set()
        # STRM 2a raw-rating tap: remember the current query so the
        # record_feedback tool path (tools.dispatch_tool -> store.record_feedback)
        # can thread it into feedback.jsonl. Cleared on every return path (the
        # early-return at the route gate below + the happy-path tail) so it never
        # leaks into the next query -- if a new return path is added, clear there.
        self._current_query = user_prompt
        # STRM Phase 4 Step 4/5: salience trigger. If armed, capture the pre-step
        # WM state (the 2c surprise term needs surprise(z_t, z_{t+1}) -> both
        # states) BEFORE the query step mutates it. Flag-off (the default) skips
        # the capture + the hook entirely -> byte-identical to pre-Step-4. Reset
        # the per-turn stashes so a skipped/failed turn never leaks the previous
        # turn's anchors / fired episodes.
        salience_armed = self._salience_armed()
        if not salience_armed:
            self._salience_anchors = None
            self._salience_fired_episodes = None
            self._salience_signals = None
        else:
            # Step 6: advance the freshness-watermark turn counter for this
            # armed query (the hook computes anchor age = turn_count - entry_turn).
            self._salience_turn_count += 1
        prev_state_tensors = None
        if salience_armed and self.working_memory.state is not None:
            prev_state_tensors = [t.clone() for t in self.working_memory.state_tensors()]
        # 1. embed prompt; update WM (state persists across queries).
        prompt_emb = self.working_memory.embed([user_prompt])[0]
        self.working_memory.update(prompt_emb)
        self.working_memory.set_metadata("last_query_type", self._classify_query(user_prompt))
        wm_snapshot = self.working_memory.snapshot()
        # STRM Phase 4 Step 4/5: score the ring for salience (state-conditioned,
        # pre-retrieval) AND fire state-conditioned retrieval per salient anchor
        # (budget-capped, dedup by episode_id). Stashes anchors (Step 6) + fired
        # episodes (merged into the prompt-driven set below, pin-tagged re-
        # inject). Best-effort: any failure leaves both stashes None (no-op) ->
        # flag-off byte-identical.
        if salience_armed:
            self._run_salience_hook(prompt_emb, prev_state_tensors, signal)

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
                # Release the foreground gate on this early-return path too, or
                # the distill worker would stay paused until the next query
                # (foreground_busy.set() above is not re-cleared by the happy-
                # path tail below, which this return skips).
                if self._distill_worker is not None:
                    self._distill_worker.foreground_busy.clear()
                self._current_query = None
                return {
                    "response": None, "route": route, "retrieved_episodes": [],
                    "context_used": None, "chunked": None,
                    "working_memory_state": self.working_memory.snapshot(),
                    "presentation_plan": None, "end_state_plan": None,
                    "supported": False,
                    # Armed-only (see the happy-path augmentation): the
                    # salience hook ran before this route gate, so report its
                    # retrieval count + signals even on the unsupported-route
                    # early return for a consistent budget/signal contract.
                    # Absent when off.
                    **({"salience_retrieval_count": len(self._salience_fired_episodes or [])}
                       if salience_armed else {}),
                    **({"salience_signals": self._salience_signals or [],
                        "salience_gap_text": format_salience_gap(self._salience_signals or [])}
                       if salience_armed else {}),
                }
            episodes = routing_result.get("results", [])
        else:
            episodes = self.retriever.retrieve(
                plan_prompt, conversation_history=conversation_history, signal=signal,
            )

        # STRM Phase 4 Step 5: merge salience-fired episodes (state-conditioned
        # proactive recall) into the prompt-driven set. Salience first, dedup by
        # episode_id -- a salience-fired episode already in the prompt-driven set
        # is kept in its salience position (and pin-tagged on inject); the
        # prompt-driven duplicate is dropped so the same episode is not injected
        # twice. Flag-off / disarmed / failed -> ``_salience_fired_episodes`` is
        # None -> ``salience_fired_ids`` stays empty -> no merge -> byte-
        # identical to pre-Step-5.
        salience_fired_ids: set = set()
        if self._salience_fired_episodes:
            fired = [ep for ep in self._salience_fired_episodes
                     if ep.get("episode_id") is not None]
            salience_fired_ids = {ep["episode_id"] for ep in fired}
            episodes = fired + [ep for ep in episodes
                                if ep.get("episode_id") not in salience_fired_ids]

        # 4. inject each retrieved episode into WM as a gist step.
        if episodes and self.embedder is not None:
            summaries = [e.get("summary", "") or e.get("text", "") for e in episodes]
            embs = self.working_memory.embed(summaries)
            # Thread provenance (episode_id + summary) into each inject so the
            # WM ring slots carry ``source_id``/``text`` when the ring is ON --
            # the STRM 2a relevance head scores per slot and the 2d replay logger
            # + label generator match on ``source_id`` (a slot is "later needed"
            # if its source_id re-appears after a ring gap). When the ring is OFF
            # (the default) provenance is ignored, so this is byte-identical to
            # the pre-2d path.
            # STRM Phase 4 Step 5: salience-fired episodes (state-conditioned
            # proactive recall) inject with ``pin=True`` so W_A retains them
            # over the next K steps; prompt-driven episodes inject with
            # ``pin=False`` (unchanged). ``salience_fired_ids`` is empty when the
            # trigger is off / disarmed / failed -> every inject pin=False ->
            # byte-identical to pre-Step-5. Pin is itself gated on ring_capacity
            # > 0 (Step 3), which holds whenever salience is armed.
            for emb, ep in zip(embs, episodes):
                self.working_memory.inject(
                    emb, source_id=ep.get("episode_id"),
                    text=ep.get("summary", "") or ep.get("text", ""),
                    pin=(ep.get("episode_id") in salience_fired_ids),
                )
            self.working_memory.set_metadata(
                "active_domains", sorted({d for e in episodes for d in e.get("topics", [])})[:5]
            )

        # STRM 2d replay logger (Step 5): when ``strm_graduation_logging`` is
        # on, snapshot the WM ring slots for THIS turn to replay.jsonl so the
        # v2 graduation labels can accumulate (one record per ring slot per
        # turn; later_needed is filled later by the label generator). The ring
        # is now fully populated (the query step + the recalled-episode
        # injects). Best-effort: a logger failure never breaks the query.
        if getattr(_runtime_config, "strm_graduation_logging", False):
            try:
                self._write_graduation_replay(prompt_emb, signal)
            except Exception as e:  # noqa: BLE001 - logging is best-effort
                print(f"[graduation-replay-fail] {e}", file=sys.stderr)

        # 5. Presentation Gate axis (a): chunking strategy.
        # STRM Phase 3: when the context-builder is wired AND the ring is on AND
        # a 2a relevance head is loaded, attend over the WM ring with r_i as a
        # bias and select top-m primary context instead of the heuristic
        # PresentationGate. The builder reorders ``episodes`` (selected first) +
        # emits a PresentationPlan with ``primary_chunk_count = m``; the chunker
        # then takes the first m (the selected ones) as primary. Any exception,
        # empty ring, or no matching slots falls back to the heuristic so the
        # turn never crashes. The ``else`` branch is the pre-Phase-3 code
        # verbatim -> byte-identical when the builder flag is off.
        if (self.context_builder is not None
                and self.working_memory.ring_capacity > 0
                and self.relevance_head is not None):
            try:
                presentation_plan, ordered_episodes = self._plan_with_context_builder(
                    user_prompt, episodes, prompt_emb)
            except Exception as e:  # noqa: BLE001 - builder is best-effort
                print(f"[context-builder-fail] {e}", file=sys.stderr)
                presentation_plan = self.presentation_gate.plan(
                    user_prompt, episodes, working_memory=wm_snapshot,
                    retrieval_gate_pathway=pathway,
                )
                ordered_episodes = episodes
        else:
            presentation_plan = self.presentation_gate.plan(
                user_prompt, episodes, working_memory=wm_snapshot,
                retrieval_gate_pathway=pathway,
            )
            ordered_episodes = episodes

        # 6b. Presentation Gate axis (b): end state (heuristic default or override).
        end_state_plan = self.presentation_gate.plan_end_state(
            user_prompt, episodes, working_memory=wm_snapshot,
            caller_end_state=end_state, format_spec=format_spec,
            extract_schema=extract_schema, model_size=model_size,
        )

        # 7. chunk → ChunkedContext.
        chunked = self.ssm_chunker.chunk(ordered_episodes, presentation_plan)

        # 8/9. format + dispatch on end state.
        # Reset the expand handler's per-query counter for the outcome signal.
        self.expand_handler.expand_count = 0

        # The synthesize callable: build messages and call mode_a._complete.
        # Phase 2c+: the self-chat TOOL LOOP (self_chat_tool_loop_enabled, the
        # default) lets Bonsai call expand / search_memory mid-generation to
        # ground its answer beyond the pre-retrieved context, plus
        # record_feedback for salience (gated by feedback_salience_enabled). A
        # live probe confirmed the 8B Bonsai emits native, parseable
        # tool_calls (finish_reason "tool_calls"), so the loop is the primary
        # path; the structured-JSON fallback stays as a safety net for when the
        # model emits no record_feedback (loop on OR off). When the loop is OFF
        # the body is byte-identical to the one-shot path (the A/B regression
        # guard). Best-effort: a feedback or loop failure never loses the
        # response.
        feedback_enabled = _runtime_config.feedback_salience_enabled
        loop_enabled = _runtime_config.self_chat_tool_loop_enabled
        feedback_state = {"count": 0}
        # Loop transcript surfaced onto the result dict by query() (D6). Reset
        # per query; set to the loop dict only when the loop path ran.
        self._last_loop = None

        def _synthesize(context: str, history: Optional[list[dict]]) -> str:
            sys_content = "You are a helpful assistant with access to past conversations."
            if loop_enabled:
                # Bounds redundant tool calls from the 8B: only call a tool
                # when the provided context is genuinely insufficient. Loop-
                # path-only so the one-shot path stays byte-identical.
                sys_content += (" Only call a tool when the provided context is"
                                " genuinely insufficient; if you can answer"
                                " from it, do so without calling tools.")
            messages: list[dict] = [{"role": "system", "content": sys_content}]
            if history:
                messages.extend(history[-10:])
            user_content = f"Context from past conversations:\n{context}\n\nUser: {user_prompt}"
            if feedback_enabled:
                user_content += "\n\n" + feedback_instruction(episodes)
            messages.append({"role": "user", "content": user_content})

            if loop_enabled:
                # Loop path: run_tool_loop drives the multi-turn tool
                # conversation (call -> dispatch -> append tool result ->
                # repeat). The tool SET is the gate for the record_feedback
                # boost side-effect inside the loop: TOOL_SCHEMAS (all 3)
                # when feedback is on, LOOP_TOOLS (expand + search_memory)
                # when off -- dispatch_tool does not re-check
                # feedback_salience_enabled, so the set must.
                loop_tools = TOOL_SCHEMAS if feedback_enabled else LOOP_TOOLS
                dispatch_fn = lambda name, args: dispatch_tool(self, name, args)
                try:
                    loop = run_tool_loop(
                        self.mode_a._complete, "", messages, dispatch_fn,
                        max_iters=_runtime_config.self_chat_tool_loop_max_iters,
                        tools=loop_tools,
                    )
                except Exception as e:  # noqa: BLE001 - loop failure -> empty answer
                    print(f"[synthesize-loop-fail] {e}", file=sys.stderr)
                    return ""
                if loop.get("exhausted"):
                    print("[synthesize-loop-exhausted] hit max_iters mid-conversation",
                          file=sys.stderr)
                content = loop.get("content") or ""
                if feedback_enabled and self.store is not None:
                    # The store-is-not-None guard is required: the fallback
                    # below (_feedback_fallback_call) dereferences self.store.
                    fb_sum = _sum_record_feedback_applied(loop.get("collected"))
                    if fb_sum == 0:
                        try:
                            fb_sum = self._feedback_fallback_call(episodes, content)
                        except Exception as e:  # noqa: BLE001 - fallback is best-effort
                            print(f"[feedback-fallback-fail] {e}", file=sys.stderr)
                    feedback_state["count"] += fb_sum
                self._last_loop = loop  # surfaced on result by query() (D6)
                return content

            # One-shot path (loop disabled) -- byte-identical to the pre-loop
            # body: one _complete + _dispatch_feedback.
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
        # STRM Phase 4 Step 5: surface the per-turn salience-fired retrieval
        # count so the deferred Step 7 eval can measure the proactive-recall
        # budget against fixed-interval RAG at equal budget WITHOUT re-
        # instrumenting. Armed-only -> the key is ABSENT when the flag is off
        # (byte-identical result dict to pre-Step-5).
        if salience_armed:
            result["salience_retrieval_count"] = len(self._salience_fired_episodes or [])
        # STRM Phase 4 Step 6: surface the per-anchor salience signals
        # (recall | stale_uncertain) + the consumer-facing gap text. A young
        # anchor whose retrieval returned nothing emits ``stale_uncertain`` ("I
        # may know this but have not finished ingesting it") instead of being
        # silently suppressed (proposal sec 5: don't lie by omission). Armed-
        # only -> both keys ABSENT when the flag is off (byte-identical).
        if salience_armed:
            result["salience_signals"] = self._salience_signals or []
            result["salience_gap_text"] = format_salience_gap(self._salience_signals or [])

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

        # Phase 2c+: when the self-chat tool loop ran, surface its transcript
        # for live-dogfood observability (the synthesize end-state only; the
        # non-loop path leaves self._last_loop None and adds nothing -- so the
        # result keys stay byte-identical to the one-shot path when the loop is
        # off). loop_exhausted is True iff the loop hit max_iters mid-
        # conversation (a truncated tool trajectory, not a clean stop).
        if self._last_loop is not None:
            result["loop_tool_messages"] = self._last_loop.get("tool_messages")
            result["loop_collected"] = self._last_loop.get("collected")
            result["loop_exhausted"] = self._last_loop.get("exhausted", False)
            self._last_loop = None

        # 2026-07-14: close the runtime gap -- persist the (prompt, response)
        # exchange as a new episode so the system learns from use. Always-encode
        # by default; ``auto_persist=False`` opts out. Best-effort: a persistence
        # failure never loses the response the user already has.
        if auto_persist:
            self._persist_exchange(user_prompt, result, signal)
        # The foreground response is fully built + persisted; release the
        # background distill worker so it can fill this turn's (and any queued
        # turn's) graph edges in the now-idle GPU gap. No-op without a worker.
        if self._distill_worker is not None:
            self._distill_worker.foreground_busy.clear()
        self._current_query = None
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
            if self._distill_worker is not None:
                # Async-distill path: pre-allocate the id on the main thread
                # (the persisted counter is never touched by the worker),
                # build + write the stub (content + vector index -- the one
                # synchronous cost the design keeps), set the follows chain,
                # then hand the stub to the worker for the 22 s fill. The
                # response has already returned by the time the fill runs.
                episode_id = self.store.next_episode_id()
                episode = encoder.encode_messages_stub(
                    messages,
                    episode_id,
                    origin="live",
                    salience=prof["salience"],
                    utility_decay_rate=prof["decay_rate"],
                    embedder=raw_encode,
                )
                self.store.encode_episode_content(episode_id, episode)
                encoder.last_episode_id = episode_id
                result["persisted_episode_id"] = episode_id
                self._distill_worker.enqueue(episode, episode_id)
            else:
                # Synchronous path (the default, flag off): extract + build +
                # store in one fused call on the main thread, byte-identical to
                # pre-async. ``encode_messages`` sets last_episode_id itself.
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

    # ── STRM 2d: replay logger (v2 graduation training substrate) ──

    _REPLAY_PATH = Path("data/training/strm_graduation/replay.jsonl")

    def _write_graduation_replay(self, prompt_emb: Tensor, signal: str) -> None:
        """Append one replay.jsonl record per WM ring slot for THIS turn.

        Gated by ``strm_graduation_logging`` (the caller checks the flag; this
        method does the I/O). Each record captures the inputs the v2
        graduation head + its label generator need:

          * ``state_t_pooled`` (1536) -- the 0a-validated pooled WM state
            (shared with RecoverabilityHead), the v2 head's first feature.
          * ``slot_y_t`` (256) -- the slot's recurrent readout, the v2 head's
            second feature.
          * ``r_i`` (float or null) -- the 2a relevance head's per-slot score
            for THIS turn (re-embedded from the slot's text against the query;
            null when no relevance head is loaded or the slot has no text).
            The v1 ``integral(r_i dt)`` proxy is scored later from a slot's
            ``r_i`` stream, so the v2-beat-v1 gate has both on the same slots.
          * ``llm_signal`` -- the turn's affective signal (the v2 head's third
            feature; the ``forgetting.LLM_SIGNAL_MODIFIERS`` vocabulary).
          * ``source_id`` / ``text`` -- provenance the label generator matches
            on (a slot is ``later_needed`` if its ``source_id`` re-appears in a
            later turn AFTER a ring gap -- "compressed out then re-recalled").
          * ``turn_id`` / ``session_id`` / ``slot_index`` -- ordering keys.
          * ``later_needed`` -- null now; the label generator fills it.

        The append is best-effort (the caller wraps it in a try) and writes
        one JSONL line per slot. Tensors are moved to CPU + ``.tolist()`` for
        JSON. The query step itself is in the ring (its ``source_id``/``text``
        are None for the raw prompt) -- kept, so the log mirrors the WM content
        exactly; the label generator ignores None-``source_id`` slots.
        """
        slots = self.working_memory.ring_buffer()
        if not slots:
            return
        self._graduation_turn_counter += 1
        turn_id = self._graduation_turn_counter
        encoder = self._get_encoder()
        session_id = (
            encoder.session_id
            if encoder is not None and getattr(encoder, "session_id", None)
            else self.user_id
        ) or "default"

        # Pool the live WM state once for this turn (the v2 head's first
        # feature). state_tensors() is the live, on-device per-layer state;
        # pool_state_tensors means over d_state per layer -> [1, 1536].
        state_pooled = pool_state_tensors(self.working_memory.state_tensors())
        state_list = state_pooled.squeeze(0).to(torch.float32).tolist()

        # r_i: only when a 2a relevance head is loaded. Re-embed each slot's
        # text (bge-small, 384-d -- the SAME vector the 2a generator built its
        # doc vectors from) and score against the query embedding. Slots with
        # no text (e.g. the raw query step, None-provenance recalls) get null.
        # The loop is factored into ``relevance_score.score_ring_slots`` (shared
        # with the Phase 3 context-builder path); this call is byte-identical to
        # the pre-Phase-3 inline loop -- same embed, same device moves, same
        # ``predict`` -> ``float(r[j].item())`` assignment into ``r_is``.
        _slots, r_is = score_ring_slots(
            self.working_memory, self.relevance_head, self.embedder,
            prompt_emb, slots=slots,
        )

        # Append one JSONL line per slot (oldest-first, the ring's order).
        self._REPLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self._REPLAY_PATH, "a", encoding="utf-8") as f:
            for slot_index, slot in enumerate(slots):
                y_list = slot.y.to(torch.float32).squeeze(0).tolist()
                rec = {
                    "turn_id": turn_id,
                    "session_id": session_id,
                    "slot_index": slot_index,
                    "source_id": slot.source_id,
                    "text": slot.text,
                    "slot_y_t": y_list,
                    "state_t_pooled": state_list,
                    "r_i": r_is[slot_index],
                    "llm_signal": signal,
                    "later_needed": None,
                    # Phase 4 Step 3: whether this slot was re-injected with the
                    # pin tag (a salience-fired recall). Non-breaking extra key --
                    # ``generate_graduation_labels.py`` shallow-copies and is
                    # key-agnostic; ``graduation_training.py`` ignores extra keys.
                    # Lets a future retention surrogate ask whether pinned slots
                    # stay relevant (high r_i) over K steps. Always False until
                    # Step 5 wires the salience re-inject.
                    "pinned": bool(slot.pinned),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── STRM Phase 3: learned context-builder (learned PresentationGate) ──

    def _plan_with_context_builder(
        self,
        user_prompt: str,
        episodes: list[dict],
        prompt_emb: Tensor,
    ) -> tuple[PresentationPlan, list[dict]]:
        """Attend over the WM ring with the 2a ``r_i`` as a bias and select
        top-m primary context. Returns ``(PresentationPlan, ordered_episodes)``
        where ``ordered_episodes`` is ``episodes`` reordered with the builder-
        selected ones first and the plan's ``primary_chunk_count = m``.

        Caller guards: this is only invoked when ``self.context_builder`` is
        wired, the ring is ON, and a 2a relevance head is loaded. Any failure
        here (empty ring, no matching slots, builder exception) propagates to
        the ``try/except`` at the call site, which falls back to the heuristic
        PresentationGate so the turn never crashes.

        The builder attends over WM ring slots that (a) carry text AND (b) map
        to a retrieved episode via ``source_id``. Episodes with no ring slot
        stay compressed (no ``y_t`` to score); ring slots with no matching
        retrieved episode are excluded (not surfaced -- avoids a context leak).
        """
        if not episodes:
            # nothing to plan -- let the caller's heuristic fallback handle it
            raise RuntimeError("no episodes to plan with")

        slots = self.working_memory.ring_buffer()
        if not slots:
            raise RuntimeError("WM ring empty (ring off / not yet populated)")

        # Only ring slots that map to a retrieved episode AND carry text can be
        # scored + surfaced. Episodes are dicts with ``episode_id``.
        ep_ids = {ep.get("episode_id") for ep in episodes if ep.get("episode_id")}
        matching = [s for s in slots
                    if s.source_id in ep_ids and s.text and str(s.text).strip()]
        if not matching:
            raise RuntimeError("no ring slots map to a retrieved episode with text")

        # r_i for the matching slots (frozen 2a head) + the re-embedded doc
        # vectors the builder's W_doc path fuses. ``score_ring_slots_with_doc_embs``
        # is the same r_i loop as the graduation logger's ``score_ring_slots``;
        # the doc_embs are slot-aligned (None where unscored, but ``matching``
        # has no None-text slots so all are scored).
        m_slots, r_is, doc_embs = score_ring_slots_with_doc_embs(
            self.working_memory, self.relevance_head, self.embedder,
            prompt_emb, slots=matching,
        )

        # Stack the per-slot tensors the builder consumes. ``s.y`` is [1,256];
        # squeeze to [256]. ``r`` defaults to 0.5 where r_i is None (defensive --
        # matching slots all have text so r_i should be non-None, but a None
        # head/embedder path returns None and we degrade rather than crash).
        slots_y = torch.stack(
            [s.y.to(torch.float32).squeeze(0).reshape(-1) for s in m_slots]
        )                                                            # [K, 256]
        slots_doc_emb = torch.stack(
            [e.to(torch.float32).squeeze(0).reshape(-1) for e in doc_embs
             if e is not None]
        )                                                            # [K, 384]
        r = torch.tensor(
            [ri if ri is not None else 0.5 for ri in r_is],
            dtype=torch.float32,
        )                                                            # [K]

        # Builder selects top-m slot indices (descending score). ``m`` is the
        # builder's serve-time fixed top_m (from the checkpoint); clamped to K
        # + to len(episodes) inside predict (topk clamps to K).
        top_m_idx, _ = self.context_builder.predict(
            slots_y, slots_doc_emb, prompt_emb, r,
        )
        if not top_m_idx:
            raise RuntimeError("context-builder returned no selection")

        # Map selected slot indices -> source_ids -> retrieved episodes, reorder
        # (selected first), preserving first-seen order within each group.
        selected_ids = [m_slots[i].source_id for i in top_m_idx]
        ep_by_id = {ep.get("episode_id"): ep
                    for ep in episodes if ep.get("episode_id")}
        selected_eps = [ep_by_id[sid] for sid in selected_ids if sid in ep_by_id]
        # de-dup by episode_id (a source_id could in principle map to one ep)
        seen: set = set()
        selected_eps = [e for e in selected_eps
                        if not (e.get("episode_id") in seen
                                or seen.add(e.get("episode_id")))]
        selected_id_set = {e.get("episode_id") for e in selected_eps}
        rest = [ep for ep in episodes if ep.get("episode_id") not in selected_id_set]
        ordered = selected_eps + rest

        m = min(len(selected_eps), len(episodes))
        strategy = DIRECT if m >= len(ordered) else CHUNKED
        return PresentationPlan(
            strategy=strategy,
            primary_chunk_count=m,
            primary_chunk_size=0,
            compressed_chunk_count=max(0, len(ordered) - m),
            expand_threshold=self.presentation_gate.cfg.expand_threshold,
            rationale=f"context-builder: {m} selected of {len(matching)} ring slots",
        ), ordered

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
        return self.store.record_feedback(judgments, query=self._current_query)

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

    def drain(self, timeout: float = 5.0) -> bool:
        """Teardown: stop the background distill worker, finish in-flight +
        queued fills, join the worker thread. No-op when async-distill is off.

        This PERMANENTLY stops the worker -- call at process exit / orchestrator
        disposal (``serve_ponder.py`` calls it on shutdown), NOT per
        conversation (the worker must stay alive across conversations). Returns
        True if the worker joined within ``timeout``. Best-effort: a hard exit
        may lose in-flight encodes -- the stub keeps the turn vector-retrievable.
        """
        if self._distill_worker is None:
            return True
        return self._distill_worker.drain(timeout=timeout)

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