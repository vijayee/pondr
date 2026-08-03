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
    LOOP_TOOLS, REMEMBER_SCHEMA, SEARCH_MEMORY_DRILLDOWN_SCHEMA,
    SELF_CHAT_TOOLS, TOOL_SCHEMAS, dispatch_tool,
    feedback_instruction, run_tool_loop,
)

if TYPE_CHECKING:
    from .encoding.encoder import HippocampalEncoder
    from .subconscious.consolidation_worker import ConsolidationWorker
    from .subconscious.fade import FadeMemory
    from .subconscious.scene_worker import SceneAuthoringWorker

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

# ── Tier-2 recall menu (the on-demand ``remember`` tool) ──
# The candidate set is SYSTEM-PROPOSED (the LLM names nothing) and LLM-FILTERED
# (the LLM reads the menu tool-result and uses the relevant items). Two sources:
#   * R4 -- fade-forgotten anchors from THIS session (``cos < cos_gist``, the
#     signal ``format_fade_block`` skips); a fresh strict enumerator (not
#     ``fading_anchors`` -- that one keeps the ``+epsilon`` R3 band + a
#     ``max_depth`` cap, neither of which belong in a recall candidate set).
#   * WaveDB-tail -- vector hits beyond the tier-1 top-k cutoff
#     (``config.default_retrieval_limit``); over-fetch ``tier1_k + _REMEMBER_TAIL_N``
#     via the SAME ``vector_search.search(prompt)`` the tier-1 semantic fallback
#     uses (NOT ``search_by_vector`` -- that would re-embed with a different
#     embedder and not reproduce tier-1's ranking), then take ``hits[tier1_k:]``.
# No cross-tier cosine dedup (the search return carries no episode vector; dedup
# would need re-embedding every hit -- expensive + fiddly, and the menu is LLM-
# filtered anyway). Per-source caps bound the redundancy instead.
_BLURB_CHARS = 240            # max chars of blurb/summary shown per menu item
_REMEMBER_TAIL_N = 8          # extra WaveDB hits beyond the tier-1 cutoff to fetch
_REMEMBER_R4_CAP = 4          # max R4 items in the menu
_REMEMBER_TAIL_CAP = 4       # max WaveDB-tail items in the menu
_REMEMBER_TOTAL_CAP = 8       # max items total (R4 + tail)
_REMEMBER_MENU_MAX_TOKENS = 512  # soft cap on the rendered menu (~4 chars/token)
# Over-fetch multiplier when user-scope is on (mirrors retriever's
# ``_USER_SCOPE_FETCH_MULT``): the vector index is one flat global layer, so the
# tail hits are over-fetched then filtered by the query user's owned ids -- the
# tail would be starved by the filter otherwise (and, without the filter, cross-
# user content would leak into the menu under ``--retrieval-user-scope``).
_REMEMBER_SCOPE_FETCH_MULT = 3


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


# Validated-compaction review resolution command (the serve REPL): ``keep <i>`` /
# ``accept <i>`` / ``edit <i>: <new gist>`` (1-indexed against the worker's
# ``pending_reviews``). A line that does not match falls through to a normal
# query (reviews stay pending). Only honored when the consolidation worker is
# present AND validate is on (gated default OFF).
_REVIEW_RE = re.compile(
    r"^\s*(keep|accept|edit)\s+(\d+)\s*(?::\s*(.+?))?\s*$", re.IGNORECASE
)


def _format_remember_menu(items: list[dict]) -> str:
    """Render the tier-2 recall menu as a labeled text block for the LLM.

    Two sections -- ``[FADE R4 -- fading from this session]`` (the R4 fade-
    forgotten anchors, most-faded-first) and ``[LONG AGO -- recalled from
    WaveDB tail]`` (vector hits beyond the tier-1 cutoff, score-desc). Each
    item is one short labeled line. A ``len // 4``-char soft cap (≈4 chars/token)
    is applied DROP-not-truncate against ``_REMEMBER_MENU_MAX_TOKENS``: once the
    running char budget is spent, the remaining items are dropped (the menu is a
    maybe-list, not a must-include -- truncating a line mid-fact would be worse
    than omitting it). Returns ``""`` when ``items`` is empty (the dispatch
    branch turns that into an honest "nothing to recall" error string).
    """
    if not items:
        return ""
    r4 = [i for i in items if i.get("source") == "fade_r4"]
    tail = [i for i in items if i.get("source") == "wavedb_tail"]
    lines: list[str] = [
        "[REMEMBER MENU -- system-flagged maybes; use the relevant ones in your answer]",
    ]
    if r4:
        lines.append("[FADE R4 -- fading from this session]")
        for i in r4:
            cos = i.get("cos")
            tag = f" (cos {cos:.2f})" if isinstance(cos, (int, float)) else ""
            lines.append(f"- [fading{tag}] {i.get('blurb', '')}")
    if tail:
        lines.append("[LONG AGO -- recalled from WaveDB tail]")
        for i in tail:
            lines.append(f"- [long ago] {i.get('summary', '')}")
    text = "\n".join(lines)
    cap_chars = _REMEMBER_MENU_MAX_TOKENS * 4
    if len(text) <= cap_chars:
        return text
    # Drop-not-truncate: keep whole lines until the budget is spent. The header
    # always survives (it is short + orients the LLM); subsequent lines are added
    # only while they fit. The last line kept may straddle the budget by one
    # line -- acceptable (a maybe-list, not a verbatim quote).
    kept = [lines[0]]
    used = len(lines[0])
    for line in lines[1:]:
        if used + 1 + len(line) > cap_chars:
            break
        kept.append(line)
        used += 1 + len(line)
    return "\n".join(kept)


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
        identity_instance: bool = False,
        capture_pre_state: bool = False,
        fade_memory: "Optional[FadeMemory]" = None,
        fade_memory_top_k: int = 5,
        fade_inject: bool = False,
        consolidation_worker: "Optional[ConsolidationWorker]" = None,
        tier2_recall_menu: bool = False,
        scene_blocks: bool = False,
        scene_worker: "Optional[SceneAuthoringWorker]" = None,
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
        # Phase 1a: prompt-turn counter for the ``strm_ring_text``
        # conversation-slot provenance (``source_id=f"{session_id}#msg{turn}"``).
        # Incremented once per ``query`` when ``strm_ring_text`` is ON; untouched
        # (stays 0) when off, so the flag-off path is byte-identical (no
        # provenance is passed to ``update``). Monotonic per orchestrator
        # lifetime (NOT reset on ``load_session`` — matches
        # ``_graduation_turn_counter``): the ``session_id`` prefix in the
        # source_id already separates sessions, so a carried counter still
        # gives correct per-session ids (``sessionA#msg5`` != ``sessionB#msg5``).
        # The gap metric needs a message's source_id to REPEAT across the later
        # turns whose rings it appears in — the prefix+turn id does that.
        self._strm_ring_text_turn_counter: int = 0
        # Fade memory (Phase A, optional DI like the encoder). When wired it
        # ingests each (user, assistant) exchange -- building the SSM-A fade
        # state over a real session -- and on each query runs ``recall``,
        # surfacing the routed recalls as ``result["fade_recalls"]``. Phase A
        # is observability + ingest only: the recalls are NOT fed into the
        # retrieval/presentation/LLM-context flow yet, so the user-facing
        # response is byte-identical whether the flag is on or off. ``None``
        # (the default, flag off) -> no fade wiring -> byte-identical to
        # pre-fade. Sits ALONGSIDE WorkingMemory (not replacing it). Best-
        # effort: any failure in recall/ingest is swallowed (the turn proceeds
        # unchanged), mirroring ``_run_salience_hook``.
        self._fade = fade_memory
        self._fade_top_k = fade_memory_top_k
        # Phase B: when True, the fade recalls are formatted into a ``[FADE MEMORY]``
        # block and prepended to the LLM user message on synthesize turns (the only
        # end state that calls the LLM). ``False`` (default) -> the recalls stay
        # observability-only (``result["fade_recalls"]``), byte-identical to Phase A.
        self._fade_inject = fade_inject
        self._fade_regime_names: dict[int, str] = {}
        self._format_fade_block = None
        # A3 (corrected): scene-block -> system-prompt-suffix formatter. ``None``
        # (default, flag off) -> no system-prompt append -> byte-identical to
        # pre-A3. Set below when ``self._scene_blocks`` is on (lazy import, like
        # the fade-block seam).
        self._format_scene_block = None
        # Gist-on-forgetting consolidation worker (Phase C, optional DI like the
        # distill worker). When wired, ``tick()`` is called at the query tail
        # (after the fade ingest) to enqueue fading anchors for background
        # gist-ing; the worker thread consolidates them BETWEEN turns (it shares
        # this orchestrator's foreground-busy gate via its own event, set/cleared
        # alongside the distill worker's below). ``None`` (default, flag off) ->
        # no consolidation -> byte-identical to Phase B. Best-effort: tick + the
        # worker's per-job loop swallow all failures (a Bonsai outage leaves
        # anchors at R4, retried next sweep -- no fabricated gist).
        self._consolidation_worker = consolidation_worker
        # Scene blocks (B1): LLM-authored topic-level macro-memory stored IN
        # WaveDB (``content/scene/{id}``, NOT files). The worker authors/revises
        # one scene per ingest BATCH in the background (between turns, sharing
        # this orchestrator's foreground-busy gate via its own event -- assigned
        # below alongside the consolidation + distill workers). ``None`` (default,
        # flag off) -> no scene worker -> no scene writes -> ``default_scene_ids``
        # is empty -> byte-identical to pre-B1. Best-effort: tick + the worker's
        # per-job loop swallow all failures (a Bonsai outage leaves the macro
        # layer untouched, retried next batch -- no fabricated scene).
        self._scene_blocks = bool(scene_blocks)
        self._scene_worker = scene_worker
        if fade_memory is not None:
            from src.subconscious.fade import (  # local, light import
                REGIME_NAME, format_fade_block,
            )
            self._fade_regime_names = dict(REGIME_NAME)
            self._format_fade_block = format_fade_block
        # A3 (corrected): scene blocks are session-stable macro memory -> render
        # in the system-prompt suffix (cache-friendly), NOT the user message.
        # Lazy import mirrors the fade-block seam above; the ``= None`` default
        # is set beside ``self._format_fade_block`` above.
        if self._scene_blocks:
            from src.subconscious.scene_format import (  # local, light import
                format_scene_block,
            )
            self._format_scene_block = format_scene_block

        # Tier-2 recall menu (the on-demand ``remember`` tool). When True, the
        # loop-path synthesize appends ``REMEMBER_SCHEMA`` to the tool set the
        # consumer sees (a NEW list -- the module-level lists are never mutated,
        # so the flag-off path hands the consumer the exact prior tool objects,
        # byte-identical). The LLM calls ``remember`` mid-generation when it
        # senses a gap; ``remember_menu`` builds a SYSTEM-PROPOSED candidate set
        # (R4 fade-forgotten anchors from this session ∪ WaveDB-tail hits beyond
        # the tier-1 top-k cutoff) and the LLM FILTERS it by using the relevant
        # items in its answer -- one round-trip. ``False`` (default) -> the
        # schema is absent, ``remember_menu`` short-circuits to "" -> byte-
        # identical to pre-tier-2. Loop-path-only: the one-shot path (loop off)
        # never gets the schema and cannot dispatch ``remember``.
        self._tier2_recall_menu = bool(tier2_recall_menu)
        # The structured menu from the last ``remember`` call, surfaced on the
        # result dict for observation (``result["remember_menu"]``). Reset per
        # query; set only when ``remember_menu`` actually ran. None when the
        # flag is off / the LLM never called ``remember`` -> key ABSENT ->
        # byte-identical.
        self._last_remember_menu: Optional[list] = None

        # The cross-query Working Memory (persistent state). embedder injected so
        # WM can embed episodes/queries on demand. ``ring_capacity`` overrides
        # the instance config's ring_capacity (default None -> config, which is 0
        # = ring OFF, byte-identical to Phase 2c). The STRM 2a/2d serve-time
        # read-out heads (relevance scoring, graduation replay logging) need the
        # ring ON; a serve flag threads a K>0 here so the ring populates. K=0
        # (the default) keeps the ring off and the shipped path byte-identical.
        self.working_memory = WorkingMemory(
            backbone, embedder=embedder, decay_alpha=config.working_memory.decay_alpha,
            ring_capacity=ring_capacity, identity_instance=identity_instance,
            capture_pre_state=capture_pre_state,
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
        # The consolidation worker mutates ssm_a/blurbs (which Seam-2 recall
        # reads), so it MUST yield to the foreground too -- gate it on the same
        # set/clear lifecycle so consolidation runs only between turns.
        if self._consolidation_worker is not None:
            self._consolidation_worker.foreground_busy.set()
        # The scene authoring worker mutates the store (scene writes) between
        # turns; gate it on the same set/clear lifecycle so authoring runs only
        # in the gaps between turns (D3 -- mirrors the consolidation worker).
        if self._scene_worker is not None:
            self._scene_worker.foreground_busy.set()
        # Validated compaction: a review-resolution command (keep/accept/edit
        # <i>) is consumed as a meta-command THIS turn -- it resolves a deferred
        # consolidation review instead of running a normal query (so the line
        # "keep 1" is not sent to Bonsai as a prompt). The worker is blocked for
        # the whole query (foreground_busy set above), so ``resolve``'s
        # ``consolidate`` call is race-free. A non-matching line falls through to
        # the normal query; pending reviews then surface in the result. Only when
        # the worker is present AND validate is on (gated default OFF).
        if (self._consolidation_worker is not None
                and self._consolidation_worker.validate):
            ack = self._try_resolve_review(user_prompt)
            if ack is not None:
                if self._distill_worker is not None:
                    self._distill_worker.foreground_busy.clear()
                if self._consolidation_worker is not None:
                    self._consolidation_worker.foreground_busy.clear()
                if self._scene_worker is not None:
                    self._scene_worker.foreground_busy.clear()
                self._current_query = None
                return {
                    "response": ack,
                    "pending_consolidation_reviews":
                        self._consolidation_worker.pending_reviews(),
                    "retrieved_episodes": [], "context_used": None,
                    "chunked": None,
                    "working_memory_state": self.working_memory.snapshot(),
                    "presentation_plan": None, "end_state_plan": None,
                    "supported": True,
                }
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
        # Phase 1a (strm_ring_text): when ON, thread the prompt's text +
        # ``source_id=f"{session_id}#msg{turn}"`` into ``update`` so the
        # conversation slot carries provenance + ``slot_type=0`` and survives the
        # live scorer's ``text is not None`` filter -> the live gate scores the
        # FULL ring (conversation + retrieved docs), not retrieved-docs only
        # (task #46/#47 H2 content shift). Flag-OFF: ``update(prompt_emb)`` with
        # no provenance -> byte-identical to pre-Phase-1a (conversation slots
        # dropped by the scorer; shipped path unchanged).
        if getattr(_runtime_config, "strm_ring_text", False):
            self._strm_ring_text_turn_counter += 1
            encoder = self._get_encoder()
            session_id = (
                encoder.session_id
                if encoder is not None and getattr(encoder, "session_id", None)
                else self.user_id
            ) or "default"
            self.working_memory.update(
                prompt_emb, text=user_prompt,
                source_id=f"{session_id}#msg{self._strm_ring_text_turn_counter}",
            )
        else:
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
                if self._consolidation_worker is not None:
                    self._consolidation_worker.foreground_busy.clear()
                if self._scene_worker is not None:
                    self._scene_worker.foreground_busy.clear()
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

        # Fade memory recall (Phase A observability only -- NOT fed into the
        # retrieval/presentation/LLM-context flow). Runs on every query when a
        # FadeMemory is wired, surfaces the routed recalls as
        # ``result["fade_recalls"]``. Best-effort: swallowed on any failure so
        # the turn never breaks. Empty when ``self._fade is None`` (flag off).
        fade_recalls: list = []
        if self._fade is not None:
            try:
                names = self._fade_regime_names
                for r in self._fade.recall(user_prompt, top_k=self._fade_top_k):
                    aid = r.anchor_id
                    fade_recalls.append({
                        "anchor_id": aid,
                        "regime": r.regime,
                        "regime_name": names.get(r.regime, "?"),
                        "cos": r.cos,
                        # A4: prompt relevance (cos(bge(prompt), bge(anchor))),
                        # surfaced from BlurbStore.retrieve in FadeMemory.recall.
                        # The within-regime drop key for the fade-block budget
                        # cascade (the free router ``cos`` is recoverability).
                        # Observability + the drop-key source -- NOT LLM-facing
                        # (only the rendered block string reaches the LLM).
                        "cos_q": r.cos_q,
                        "content": r.content,
                        "blurb": r.blurb,
                        # Phase C observability: how many times this anchor has
                        # been consolidated (gist-of-gist depth; 0 = verbatim,
                        # never gisted) and whether its blurb is a real gist
                        # (prior_gist is set the moment ``consolidate`` runs).
                        # Both 0/False when the consolidation flag is off.
                        "consolidation_count": self._fade.consolidation_count(aid),
                        "consolidated": self._fade.prior_gist(aid) is not None,
                    })
            except Exception as e:  # noqa: BLE001 - observability only
                print(f"[fade-recall-fail] {e}", file=sys.stderr)

        # 4. inject each retrieved episode into WM as a gist step.
        if episodes and self.embedder is not None:
            # STRM 1f-6: prefer the prose ``embed_text`` handle (the LLM-written
            # code-section summary) so a retrieved CODE doc is injected into the
            # ring by MEANING, not the synthetic ``summary`` string (which
            # inlines the full code and embeds poorly against prose queries -- the
            # 1f-5 code-doc mis-rank, median -0.801). Absent (text docs, conv
            # slots, summarizer down) -> falls back to ``summary``/``text`` =
            # byte-identical to pre-1f-6. NOTE: line ~679 (the slot ``text``)
            # stays ``summary or text`` -- that is the recalled content the
            # formatter + salience scorer read; touching it would swap full code
            # for truncated code (the binding code-recall concern).
            summaries = [e.get("embed_text", "") or e.get("summary", "")
                         or e.get("text", "") for e in episodes]
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

        # A3 (corrected): scene blocks are session-stable macro memory -> render
        # in the system-prompt suffix (cache-friendly), NOT the user-message
        # context. Pull them out of the episode set here, AFTER the salience merge
        # + WM-ring injects above and BEFORE the context builder / presentation
        # gate / chunker, so they never enter ChunkedContext -> never in the
        # user-message ``context`` string. ``scene_results`` is closed over by
        # ``_synthesize`` below for system-prompt injection. Gated on
        # ``self._scene_blocks``: flag off -> no filter -> scenes (if any) stay in
        # the user context via ``_format_episode``'s ``kind == "scene"`` branch
        # -> byte-identical to pre-A3. Empty when no scenes retrieved -> no
        # system append -> byte-identical to flag-off.
        scene_results: list = []
        if self._scene_blocks:
            scene_results = [ep for ep in episodes if ep.get("kind") == "scene"]
            episodes = [ep for ep in episodes if ep.get("kind") != "scene"]

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
        # Tier-2 recall menu: reset the per-query structured-menu stash so a
        # skipped / no-``remember``-call turn never leaks the previous turn's
        # menu (the key is ABSENT on the result when this stays None).
        self._last_remember_menu = None

        def _synthesize(context: str, history: Optional[list[dict]]) -> str:
            sys_content = "You are a helpful assistant with access to past conversations."
            if loop_enabled:
                # Tools are OPTIONAL -- phrased as guidance, not an order. The
                # model may answer directly from the context when it suffices,
                # and reach for a tool only when the context is genuinely
                # insufficient (this bounds redundant tool calls from the 8B
                # without forbidding the model from ever calling one). Loop-
                # path-only so the one-shot path stays byte-identical.
                sys_content += (" Tools are available if you need them, but"
                                " optional -- when the provided context already"
                                " answers the question, just answer directly;"
                                " reach for a tool only when the context is"
                                " genuinely insufficient.")
                # B2: drill-down self-awareness (loop-path-only, MAY-phrased --
                # never imperative per [[llm-tool-use-prompts-optional]]). When
                # ``--drill-down`` is on, the search_memory tool accepts a
                # ``verbatim`` option (the user's literal words vs a summary)
                # and context units may show a ``[strategy:...]`` tag for how
                # they were found. OFF -> no note -> byte-identical.
                if _runtime_config.drill_down_enabled:
                    sys_content += (" The search_memory tool accepts a `verbatim`"
                                    " option when you need the user's literal"
                                    " words rather than a summary; context units"
                                    " may show a `[strategy:...]` tag indicating"
                                    " how they were found.")
            # A3 (corrected): inject the session-stable scene blocks into the
            # system-prompt SUFFIX (NOT the user message). ``scene_results`` is
            # the per-query retrieved scene subset pulled out of ``episodes``
            # before chunking (closed over here). Mirrors the fade-inject
            # best-effort swallow below. Empty / ``None`` / flag off -> no
            # append -> byte-identical to pre-A3.
            if scene_results and self._format_scene_block is not None:
                try:
                    scene_block = self._format_scene_block(scene_results)
                except Exception as e:  # noqa: BLE001 - never break the turn
                    print(f"[scene-inject-fail] {e}", file=sys.stderr)
                    scene_block = ""
                if scene_block:
                    sys_content = f"{sys_content}\n\n{scene_block}"
            messages: list[dict] = [{"role": "system", "content": sys_content}]
            if history:
                messages.extend(history[-10:])
            user_content = f"Context from past conversations:\n{context}\n\nUser: {user_prompt}"
            if feedback_enabled:
                user_content += "\n\n" + feedback_instruction(episodes)
            if self._fade is not None and self._fade_inject:
                # Phase B: prepend the fade-memory block (the fading working memory
                # of THIS conversation) ahead of the retrieved cross-session context.
                # R4 (forgotten) is a signal, not content -- ``format_fade_block``
                # skips it; the block is omitted entirely when no R1/R3 recalls exist
                # (so an all-R4 turn stays byte-identical to flag-off). Best-effort:
                # a render failure leaves ``user_content`` unchanged (the turn
                # proceeds), mirroring the recall seam's swallow.
                try:
                    # A4: pass the fade-block token budget so ``format_fade_block``
                    # can drop recalls regime-cascade (R3 before R1; lowest cos_q
                    # within a regime) when the block overflows. Default config
                    # 1024 is inert at current top_k=5 x blurb_chars=600 (~750
                    # tokens) -> no drop -> byte-identical to pre-A4.
                    block = self._format_fade_block(
                        fade_recalls,
                        max_tokens=self._fade.cfg.fade_block_max_tokens)
                except Exception as e:  # noqa: BLE001 - never break the turn
                    print(f"[fade-inject-fail] {e}", file=sys.stderr)
                    block = ""
                if block:
                    user_content = f"{block}\n\n{user_content}"
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
                # Tier-2 recall menu: append the ``remember`` schema to a NEW
                # list (never mutate the module-level lists -- that would leak
                # the schema into the flag-off path and break byte-identity).
                # Loop-path-only: the one-shot path below never sees it, so
                # ``remember`` cannot be dispatched when the loop is off.
                if self._tier2_recall_menu:
                    loop_tools = [*loop_tools, REMEMBER_SCHEMA]
                # B2: when ``--drill-down`` is on, swap the ``search_memory``
                # entry for the variant WITH the ``verbatim`` param (the
                # conversation-vs-memory split). Build a NEW list -- never
                # mutate the module-level ``TOOL_SCHEMAS``/``LOOP_TOOLS`` (same
                # discipline as the ``remember`` append above) so the flag-off
                # path hands the model the exact prior tool set. Loop-path-only
                # (the one-shot path's ``SELF_CHAT_TOOLS`` has no
                # ``search_memory``) -> byte-identical when the loop is off.
                if _runtime_config.drill_down_enabled:
                    loop_tools = [
                        (SEARCH_MEMORY_DRILLDOWN_SCHEMA
                         if t.get("function", {}).get("name") == "search_memory"
                         else t)
                        for t in loop_tools
                    ]
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
        # Fade memory (Phase A): surface the routed recalls for observation. Key
        # ABSENT when ``self._fade is None`` (flag off) -> byte-identical to
        # pre-fade. NOT fed into the LLM context this phase.
        if self._fade is not None:
            result["fade_recalls"] = fade_recalls
        # Validated compaction: surface deferred consolidation reviews (gists the
        # fidelity judge flagged as corrupt, or held unvalidated). Key ABSENT
        # when there is no validating worker or no pending reviews -> byte-
        # identical to the non-validating path. NOT a resolution turn (those
        # returned above); this just attaches the question for the user to answer
        # on the NEXT line.
        if (self._consolidation_worker is not None
                and self._consolidation_worker.validate):
            pending = self._consolidation_worker.pending_reviews()
            if pending:
                result["pending_consolidation_reviews"] = pending

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
        # Tier-2 recall menu: surface the structured menu the LLM's ``remember``
        # call produced (R4 fade-forgotten ∪ WaveDB-tail items) for observation.
        # Key ABSENT when the flag is off, the loop is off, or the LLM never
        # called ``remember`` this turn (``_last_remember_menu`` stays None) ->
        # byte-identical to pre-tier-2.
        if self._last_remember_menu is not None:
            result["remember_menu"] = self._last_remember_menu
            self._last_remember_menu = None

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
        if self._consolidation_worker is not None:
            self._consolidation_worker.foreground_busy.clear()
        if self._scene_worker is not None:
            # B1 macro-forgetting: decay every scene's heat + evict below-floor,
            # in the foreground-clear window (the worker is STILL blocked on
            # ``foreground_busy`` here, so this store mutation can't race the
            # worker's authoring mutation -- the clear below releases it after).
            # Best-effort, swallowed: a decay failure never breaks the turn.
            # Runs once per turn -- a scene touched this turn (retrieval boost or
            # UPDATE) stays warm; an untouched scene decays toward eviction.
            try:
                self._scene_worker.decay_tick()
            except Exception as de:  # noqa: BLE001 - best-effort macro-forgetting
                print(f"[scene-decay-fail] {de}", file=sys.stderr)
            self._scene_worker.foreground_busy.clear()
        self._current_query = None
        return result

    def _try_resolve_review(self, user_prompt: str) -> Optional[str]:
        """Parse + apply a deferred-review resolution command, or fall through.

        Returns an ack message when ``user_prompt`` matches ``keep <i>`` /
        ``accept <i>`` / ``edit <i>: <text>`` (applied, or failed-stale), so the
        caller short-circuits the normal query and surfaces the ack + remaining
        reviews. Returns ``None`` when the line is not a resolution command ->
        the caller runs a normal query (reviews stay pending). Only meaningful
        when the consolidation worker is present and validating; the caller
        gates on that before calling. ``resolve`` runs in the foreground (worker
        blocked on the gate), so its ``consolidate`` call is race-free.
        """
        if self._consolidation_worker is None:
            return None
        m = _REVIEW_RE.match(user_prompt or "")
        if not m:
            return None
        action = m.group(1).lower()
        idx_str = m.group(2)
        text = m.group(3)
        idx = int(idx_str) - 1
        if action == "edit":
            if not text or not text.strip():
                return f"[review #{idx_str}: edit needs new gist text]"
            ok = self._consolidation_worker.resolve(idx, "edit", text.strip())
        else:
            ok = self._consolidation_worker.resolve(idx, action)
        if ok:
            return f"[resolved review #{idx_str}: {action}]"
        return f"[review #{idx_str} not found / invalid]"

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
            response = result.get("response")
            has_response = isinstance(response, str) and response.strip()
            # Fade memory ingest (Phase A): one chunk per exchange, joined user+
            # assistant. Runs BEFORE the encoder check so the fade advances even
            # when no HippocampalEncoder is wired (the fade memory is independent
            # of the episode store). Best-effort, swallowed on any failure so
            # the turn never breaks. ``None`` (flag off) -> no-op.
            if self._fade is not None and has_response:
                try:
                    self._fade.ingest(f"User: {user_prompt}\nAssistant: {response}")
                except Exception as fe:  # noqa: BLE001 - never break persist
                    print(f"[fade-ingest-fail] {fe}", file=sys.stderr)
                # Phase C: enqueue fading anchors for background gist-ing. Read-
                # only on the memory (a ``fading_anchors`` sweep + queue puts), so
                # safe to run here even though the foreground gate is still held
                # (the worker thread blocks on the gate before any mutation).
                # Best-effort, swallowed -- a tick failure never breaks the turn.
                if self._consolidation_worker is not None:
                    try:
                        self._consolidation_worker.tick()
                    except Exception as ce:  # noqa: BLE001 - never break persist
                        print(f"[consolidation-tick-fail] {ce}", file=sys.stderr)
            encoder = self._get_encoder()
            if encoder is None:
                return
            if not has_response:
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
            # B1 scene blocks: enqueue ONE authoring job for the just-persisted
            # episode. ``topic_hint`` is the union of the episode's topics (the
            # macro axis the scene lives on); an empty topic list -> the worker's
            # ``tick`` is a no-op (no topic -> no scene). Read-only on the store
            # (it builds the job + queues it), so safe here with the foreground
            # gate still held (the worker thread blocks before any mutation).
            # Best-effort, swallowed -- a tick failure never breaks the turn.
            # NB: on the async-distill path the stub episode's topics may not be
            # filled yet (the 22s extract fills them) -> this tick no-ops and the
            # scene is authored from a later topic-bearing episode instead.
            if self._scene_worker is not None and episode is not None:
                try:
                    hint_topics = list(getattr(episode, "topics", []) or [])
                    hint = hint_topics[0] if hint_topics else ""
                    if hint:
                        self._scene_worker.tick(
                            self.user_id, [episode.id], topic_hint=hint)
                except Exception as se:  # noqa: BLE001 - never break persist
                    print(f"[scene-tick-fail] {se}", file=sys.stderr)
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
            if unit_id.startswith("scene_"):
                # A scene block (B1): return the topic + heat + cited episodes
                # + the Markdown body. Drill-down for the expand tool -- the
                # existing ``expand`` routes ``scene_*`` ids here via
                # ``dispatch_tool`` -> ``expand_memory`` -> ``expand_unit`` (no
                # separate ``expand_scene`` tool). Full scene-scene drill-down
                # (following ``cites`` to the underlying episodes) is a B2
                # follow-on; this wires expandability of the macro layer.
                sc = self.store.get_scene(unit_id)
                if sc is None:
                    return None
                parts = [
                    f"Scene (topic: {sc.get('topic') or ''})",
                    f"Heat: {float(sc.get('heat') or 0.0):.2f}",
                    f"Updated: {sc.get('updated_ts') or ''}",
                ]
                src = sc.get("source_eps") or []
                if src:
                    if _runtime_config.drill_down_enabled:
                        # B2 drill-down: follow ``cites`` ONE HOP -- each cited
                        # episode's one-line summary, so the LLM picks which to
                        # ``expand(ep_id)`` for verbatim (the scene -> cited-ep
                        # summary -> verbatim chain = "every symbol a path back
                        # to ground truth"). Capped at 12 (a scene citing many
                        # eps shows the first dozen); best-effort per cited ep --
                        # a missing/unreadable one is skipped, no crash. Falls
                        # back to the bare id list when NONE hydrate (keeps the
                        # expand contract: the cited ids are always visible).
                        cited = []
                        for ep_id in src[:12]:
                            try:
                                epc = self.store.get_episode(ep_id)
                                if epc is not None:
                                    cited.append(
                                        f"  - {ep_id}: "
                                        f"{epc.summary or '(no summary)'}")
                            except Exception:  # noqa: BLE001 - best-effort per ep
                                pass
                        if cited:
                            parts.append("Cites:")
                            parts.extend(cited)
                        else:
                            parts.append("Cites: " + ", ".join(src))
                    else:
                        parts.append("Cites: " + ", ".join(src))
                body = sc.get("body") or ""
                if body:
                    parts.append(body)
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
        verbatim: bool = False,
    ) -> str:
        """Consumer tool: re-retrieve mid-generation with a refined query/axes.

        Runs the retriever with a literal query plan (the entities/topics axes
        the consumer named) and builds the context string. The external LLM
        calls this via ``dispatch_tool("search_memory", ...)`` when the initial
        context was insufficient. Returns the formatted context (empty string
        when nothing is found).

        ``verbatim`` (B2 conversation-vs-memory split, default False): when True,
        the context string renders each episode's FULL TEXT (the user's literal
        words) alongside the summary -- the "what did the user literally say"
        intent vs the default "what do I remember" (summary) intent. The param
        is added to the LLM-visible schema ONLY when ``--drill-down`` is on
        (gated); the handler always accepts it, defaulting to summary. Flag off
        + ``verbatim=False`` -> byte-identical to pre-B2.
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
            return self.retriever.build_context_string(results, verbatim=verbatim)
        except Exception as e:  # noqa: BLE001 - search is best-effort
            print(f"[search_memory-fail] {e}", file=sys.stderr)
            return ""

    def remember_menu(self) -> str:
        """Tier-2 recall menu -- the on-demand ``remember`` tool's handler.

        Builds a SYSTEM-PROPOSED candidate set of "maybes" -- things NOT in
        context that might be worth recalling -- and returns it as a labeled
        text block the LLM FILTERS (it reads the tool-result and uses the
        relevant items in its answer, one round-trip). Two sources:

        * **R4** -- fade-forgotten anchors from THIS session: anchors whose
          ``cos(state_t, bge(anchor))`` has crossed below ``cos_gist`` (the
          signal ``format_fade_block`` skips on the inject path). A fresh
          strict enumerator (``cos < cos_gist``, not in the recency ring) --
          NOT ``FadeMemory.fading_anchors`` (that keeps the ``+epsilon`` R3
          band + a ``max_depth`` cap; a gisted-then-faded anchor with a real
          gist blurb is good recall content, so neither filter belongs here).
        * **WaveDB-tail** -- vector-index hits beyond the tier-1 top-k cutoff
          (``config.default_retrieval_limit``). Over-fetch
          ``tier1_k + _REMEMBER_TAIL_N`` via the SAME
          ``vector_search.search(prompt)`` the tier-1 semantic fallback uses
          (NOT ``search_by_vector`` -- the WM embedder is not guaranteed to be
          the same object as the vector-search embedder, so re-embedding would
          not reproduce tier-1's ranking), then take ``hits[tier1_k:]``. The
          flat global vector index is then filtered by the query user's owned
          ids (``_user_scope_sets`` + ``_filter_vector_hits_by_scope`` -- the
          same boundary the tier-1 semantic fallback applies) so cross-user
          content cannot leak into the menu under ``--retrieval-user-scope``;
          over-fetched by ``_REMEMBER_SCOPE_FETCH_MULT`` when scoped so the tail
          survives the filter. ``user_id is None`` -> no filter, no over-fetch
          -> byte-identical to the pre-scope path.

        Lazy -- computed only when the LLM calls ``remember``, zero cost on
        turns that do not. Best-effort at EVERY stage: an R4 failure and a tail
        failure are each logged and the partial menu returned (never raises;
        ``dispatch_tool``'s outer ``except`` is the final net). Returns ``""``
        when the flag is off or no maybes exist (the dispatch branch turns that
        into an honest "nothing to recall" error string). The structured items
        are stashed on ``self._last_remember_menu`` and surfaced on the result
        dict by ``query`` for observation.
        """
        if not self._tier2_recall_menu:
            return ""
        items: list[dict] = []
        # R4 source -- fade-forgotten anchors (this session).
        if self._fade is not None:
            try:
                cos_gist = self._fade.cfg.cos_gist
                for aid in self._fade.blurbs._ids:
                    if aid in self._fade.ring:           # recency verbatim window
                        continue
                    blurb = self._fade.blurbs.text(aid)
                    if blurb is None:
                        continue
                    cos = self._fade._recoverability(aid)
                    if cos is None or cos >= cos_gist:    # strict R4 only
                        continue
                    items.append({
                        "source": "fade_r4", "anchor_id": aid,
                        "cos": cos, "blurb": blurb[:_BLURB_CHARS],
                    })
            except Exception as e:  # noqa: BLE001 - R4 is best-effort
                print(f"[remember-r4-fail] {e}", file=sys.stderr)
        # WaveDB-tail source -- vector hits beyond the tier-1 top-k cutoff.
        q = getattr(self, "_current_query", "") or ""
        if (q and self.retriever is not None
                and self.retriever.vector_search is not None):
            try:
                tier1_k = _runtime_config.default_retrieval_limit
                # User-scope: the vector index is one flat GLOBAL layer, so the
                # tail hits must be filtered by the query user's owned ids --
                # the SAME ``_user_scope_sets`` / ``_filter_vector_hits_by_scope``
                # the tier-1 semantic fallback uses -- else cross-user content
                # leaks into the menu under ``--retrieval-user-scope`` (the
                # boundary shipped in 339cdb9). When ``user_id is None`` (scope
                # off) both helpers are no-ops -> byte-identical to the off path.
                # Over-fetch when scoped (``_REMEMBER_SCOPE_FETCH_MULT`` mirrors
                # retriever's ``_USER_SCOPE_FETCH_MULT``) so the tail survives
                # the filter instead of being starved by it.
                allowed_ep, allowed_doc, allowed_scene = self.retriever._user_scope_sets()
                scoped = (allowed_ep is not None or allowed_doc is not None
                          or allowed_scene is not None)
                fetch_k = tier1_k + _REMEMBER_TAIL_N
                if scoped:
                    fetch_k *= _REMEMBER_SCOPE_FETCH_MULT
                hits = self.retriever.vector_search.search(q, k=fetch_k)
                if scoped:
                    hits = self.retriever._filter_vector_hits_by_scope(
                        hits, allowed_ep, allowed_doc, allowed_scene)
                for eid, sim in hits[tier1_k:]:          # the tail beyond cutoff
                    ep = self.retriever.traversal._hydrate(eid)
                    items.append({
                        "source": "wavedb_tail", "episode_id": eid,
                        "score": float(sim),
                        "summary": (ep.get("summary") or "")[:_BLURB_CHARS],
                    })
            except Exception as e:  # noqa: BLE001 - tail is best-effort
                print(f"[remember-tail-fail] {e}", file=sys.stderr)
        # Sort: R4 most-faded-first (lowest cos), tail by score desc; cap total.
        r4 = sorted([i for i in items if i["source"] == "fade_r4"],
                    key=lambda i: i["cos"])
        tail = sorted([i for i in items if i["source"] == "wavedb_tail"],
                      key=lambda i: -i["score"])
        items = (r4[:_REMEMBER_R4_CAP] + tail[:_REMEMBER_TAIL_CAP])[:_REMEMBER_TOTAL_CAP]
        text = _format_remember_menu(items)
        self._last_remember_menu = items  # structured observability (surfaces on result)
        return text

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
        """Teardown: stop the background workers, finish in-flight + queued
        jobs, join their threads. No-op when both async-distill and fade-
        consolidation are off.

        This PERMANENTLY stops the workers -- call at process exit / orchestrator
        disposal (``serve_ponder.py`` calls it on shutdown), NOT per
        conversation (the workers must stay alive across conversations). Returns
        True if every wired worker joined within ``timeout``. Best-effort: a
        hard exit may lose in-flight encodes / consolidations -- the stub keeps
        the turn vector-retrievable and a lost consolidation leaves the anchor
        at R4 (retried next sweep, no data loss).
        """
        ok = True
        if self._distill_worker is not None:
            ok = self._distill_worker.drain(timeout=timeout) and ok
        if self._consolidation_worker is not None:
            ok = self._consolidation_worker.drain(timeout=timeout) and ok
        if self._scene_worker is not None:
            ok = self._scene_worker.drain(timeout=timeout) and ok
        return ok

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