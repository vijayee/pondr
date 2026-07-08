"""Presentation Gate: decide *how to present* retrieved context (Phase 2c).

Two orthogonal axes (docs/Phase 2c.md §5, docs/Ponder Engine Chat Facts.md §3):

- **Axis (a) — chunking strategy**: ``direct`` (no chunking, all full text),
  ``chunked`` (top-N primary full text + rest compressed into the SSM state),
  ``summary_only`` (all compressed). Decided from episode count + query
  specificity.
- **Axis (b) — end state**: ``direct`` / ``format`` / ``synthesize`` / ``extract``
  — what to DO with the retrieved results. Not every retrieval ends in an LLM
  call (chat [143]/[144]).

**Why heuristic, not a learned JGS instance (docs/Phase 2c.md §5.1).** There is
no supervised training data for either axis (no Oracle pairs), and the outcome
signals (EXPAND frequency, unused-primary count, user satisfaction, caller
overrides) are not wired live yet. A trained gate's heads would be dead params
(a de-wonk "weird/dead" flag). So the heuristic IS the real
strategy-selection logic, and ``record_outcome`` / ``record_override`` feed
``ReplayBuffer``s that seed a *future* learned gate (deferred — mirrors Phase
2b's "routed but not executed" honesty). The buffer is the only training
signal available; it is collected now even though learning is not.

The end-state axis uses an **explicit API with a heuristic default** (chat
[145]/[146]): the caller may specify ``end_state``; when they don't, the
heuristic picks a default; a caller override is recorded as the training
signal. "The API is the interface. JEPA is the optimization." The end-state
feedback signal is weak (the difference between ``direct`` and ``synthesize``
is latency/cost/format, not correctness), so inferring an unobservable
preference implicitly is fragile — the explicit override is the honest path.

This module is stdlib-only (no torch import) so it can run in any context;
the heuristic is a pure function of query text + episode count.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .routing import Embedder  # noqa: F401  (re-exported for constructor symmetry)


# ── Axis (a): chunking strategy ──

# Strategy vocabulary.
DIRECT = "direct"           # all episodes as full text, no compression
CHUNKED = "chunked"         # top-N primary full text + rest compressed
SUMMARY_ONLY = "summary_only"  # all compressed; only gist + topic summary
STRATEGIES = (DIRECT, CHUNKED, SUMMARY_ONLY)

# Query-specificity heuristics. Summarization verbs flip toward summary_only;
# factual lookups (show/who/when/where/list-what) lean direct. Cheap keyword
# matching — NOT NLP. The learned gate (deferred) replaces this.
_SUMMARIZATION_KEYWORDS = (
    "summarize", "summary", "overview", "overall", "everything about",
    "all about", "recap", "tl;dr", "the gist of", "high-level",
)
_FACTUAL_KEYWORDS = (
    "show me", "what did", "who said", "when did", "where did",
    "what is", "what was", "list the", "which ", "how much", "how many",
)


@dataclass
class PresentationPlan:
    """Axis (a) output: how to chunk/compress the retrieved episodes."""
    strategy: str            # one of STRATEGIES
    primary_chunk_count: int
    primary_chunk_size: int  # max tokens per primary chunk (len(text)//4 estimate)
    compressed_chunk_count: int
    expand_threshold: float  # confidence threshold for auto-EXPAND (Phase 4a reads this)
    rationale: str           # human-readable, for debugging


@dataclass
class PresentationOutcome:
    """Outcome signals that train a future learned presentation gate.

    Wired live in a later phase (the buffer is the seed). ``expand_count``
    (something important was compressed → should have been primary) and
    ``unused_primary_count`` (primary too large) are the chat's training signal
    (chat [130]; docs/Ponder Engine Chat Facts.md §3.1).
    """
    expand_count: int = 0
    unused_primary_count: int = 0
    user_satisfaction: float = 0.0


# ── Axis (b): end state ──

# End-state vocabulary (chat [144]).
END_DIRECT = "direct"          # results ARE the answer; no LLM
END_FORMAT = "format"          # results are context for another consumer
END_SYNTHESIZE = "synthesize"  # LLM reasons across episodes
END_EXTRACT = "extract"        # results → structured data
END_STATES = (END_DIRECT, END_FORMAT, END_SYNTHESIZE, END_EXTRACT)

# End-state heuristic keyword sets.
_EXTRACT_KEYWORDS = (
    "list all", "list of", "as json", "to json", "as a graph", "dependency graph",
    "as a table", "extract the", "structured", "schema",
)
_SYNTHESIS_KEYWORDS = (
    "why", "how did", "compare", "tradeoff", "trade-off", "explain",
    "analyze", "reason about", "what should", "recommend",
)
_FACTUAL_LOOKUP_KEYWORDS = (
    "show me", "what did", "who said", "when did", "where did",
    "what is", "what was", "how much", "how many", "which one",
)


@dataclass
class EndStatePlan:
    """Axis (b) output: what to do with the retrieved results."""
    end_state: str
    format_spec: Optional[dict] = None    # for "format": {"consumer":..., "purpose":..., "max_tokens":...}
    extract_schema: Optional[dict] = None  # for "extract": {"type":"list"|"graph"|"table", "item_type":...}
    model_size: Optional[str] = None       # for "synthesize": "bonsai" (only live model now)
    jepa_default: bool = True              # True = gate picked it; False = caller overrode
    rationale: str = ""


# ── Replay buffers (seeds for the deferred learned gate) ──

@dataclass
class ReplayBuffer:
    """A bounded ring buffer of (plan, outcome) or (override) records.

    No-op learning — the buffer is the training-data seed for a future learned
    gate/router. ``capacity`` bounds memory; oldest records are evicted. The
    records are plain dicts (JSON-safe) so a future trainer can persist them.
    """
    capacity: int = 1000
    records: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        # deque(maxlen=...) auto-evicts; keep the existing records if reloaded.
        if not self.records:
            self.records = deque(maxlen=self.capacity)
        else:
            self.records = deque(self.records, maxlen=self.capacity)

    def push(self, record: dict) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def to_list(self) -> list[dict]:
        """JSON-safe snapshot of the records (for persistence)."""
        return list(self.records)

    @staticmethod
    def from_list(records: list, capacity: int = 1000) -> "ReplayBuffer":
        """Reconstruct a buffer from a persisted record list (capacity re-bounded)."""
        buf = ReplayBuffer(capacity=capacity)
        buf.records = deque(records or [], maxlen=capacity)
        return buf


class PresentationGate:
    """Heuristic presentation planner + outcome/override buffers.

    Pure-logic (no model) — strategy and end state are functions of query text
    and episode count. The ``ReplayBuffer``s collect outcome/override signals
    for the deferred learned gate; no learning happens here.
    """

    def __init__(self, config, embedder: Optional[Embedder] = None) -> None:
        # ``config`` is a Phase2cConfig (or anything with .presentation_gate +
        # .replay_capacity). embedder is accepted for symmetry/future use; the
        # heuristic does not need it.
        self.cfg = config.presentation_gate
        self.embedder = embedder
        cap = getattr(config, "replay_capacity", 1000)
        self.outcome_buffer = ReplayBuffer(capacity=cap)   # axis (a) outcomes
        self.override_buffer = ReplayBuffer(capacity=cap)  # axis (b) overrides

    # ── Axis (a): chunking strategy ──

    def plan(
        self,
        query: str,
        retrieved_episodes: list[dict],
        working_memory=None,            # WorkingMemoryState, unused by the heuristic
        retrieval_gate_pathway: Optional[str] = None,
    ) -> PresentationPlan:
        """Pick a chunking strategy. Deterministic.

        - ``direct`` if episodes ≤ ``direct_max_episodes`` AND the query is
          specific (not a summarization verb).
        - ``summary_only`` if episodes ≥ ``summary_only_min_episodes`` OR the
          query is a summarization.
        - else ``chunked``.

        ``primary_chunk_count`` = ``min(episodes, max_primary_chunks)`` for
        chunked; = ``episodes`` for direct; = 0 for summary_only.
        """
        n = len(retrieved_episodes)
        is_summary = _has_any(query, _SUMMARIZATION_KEYWORDS)
        is_specific = _has_any(query, _FACTUAL_KEYWORDS)

        if n == 0:
            return PresentationPlan(
                strategy=DIRECT, primary_chunk_count=0,
                primary_chunk_size=0, compressed_chunk_count=0,
                expand_threshold=self.cfg.expand_threshold,
                rationale="no episodes retrieved → direct (nothing to compress)",
            )

        if is_summary or n >= self.cfg.summary_only_min_episodes:
            return PresentationPlan(
                strategy=SUMMARY_ONLY, primary_chunk_count=0,
                primary_chunk_size=0, compressed_chunk_count=n,
                expand_threshold=self.cfg.expand_threshold,
                rationale=f"summary_only: {'summarization query' if is_summary else f'{n}≥{self.cfg.summary_only_min_episodes} episodes'}",
            )

        if n <= self.cfg.direct_max_episodes and is_specific:
            return PresentationPlan(
                strategy=DIRECT, primary_chunk_count=n,
                primary_chunk_size=0, compressed_chunk_count=0,
                expand_threshold=self.cfg.expand_threshold,
                rationale=f"direct: {n}≤{self.cfg.direct_max_episodes} episodes + specific query",
            )

        # chunked: top-N primary, rest compressed.
        primary = min(n, self._max_primary_chunks())
        return PresentationPlan(
            strategy=CHUNKED, primary_chunk_count=primary,
            primary_chunk_size=0, compressed_chunk_count=n - primary,
            expand_threshold=self.cfg.expand_threshold,
            rationale=f"chunked: {n} episodes → {primary} primary + {n - primary} compressed",
        )

    def _max_primary_chunks(self) -> int:
        # The chunker's max_primary_chunks cap; the gate reads it from the same
        # config object the orchestrator passes to the chunker.
        chunker_cfg = getattr(self, "_chunker_cfg", None)
        if chunker_cfg is not None:
            return chunker_cfg.max_primary_chunks
        return 5  # default; the orchestrator wires the real cap via set_chunker_cfg

    def set_chunker_cfg(self, chunker_cfg) -> None:
        """Let the orchestrator tell the gate the chunker's primary-chunk cap."""
        self._chunker_cfg = chunker_cfg

    def record_outcome(self, plan: PresentationPlan, outcome: PresentationOutcome) -> None:
        """Store a (plan, outcome) record in the outcome buffer.

        No-op learning — the buffer seeds a future learned gate. Called by the
        orchestrator after a query completes with EXPAND/unused-primary counts.
        """
        self.outcome_buffer.push({
            "strategy": plan.strategy,
            "primary_chunk_count": plan.primary_chunk_count,
            "compressed_chunk_count": plan.compressed_chunk_count,
            "expand_count": outcome.expand_count,
            "unused_primary_count": outcome.unused_primary_count,
            "user_satisfaction": outcome.user_satisfaction,
        })

    # ── Axis (b): end state ──

    def plan_end_state(
        self,
        query: str,
        retrieved_episodes: list[dict],
        working_memory=None,
        caller_end_state: Optional[str] = None,
        format_spec: Optional[dict] = None,
        extract_schema: Optional[dict] = None,
        model_size: Optional[str] = None,
    ) -> EndStatePlan:
        """Pick an end state. Heuristic default when the caller doesn't specify;
        else honor the caller (and record the override as a training signal).

        Heuristic:
        - ``extract`` for list/graph/json/table verbs.
        - ``synthesize`` for why/how/compare reasoning, OR >direct_max episodes.
        - ``direct`` for factual lookups with ≤direct_max episodes.
        - ``format`` only when a non-bonsai consumer is named (caller-side).

        Deterministic. The caller's ``format_spec``/``extract_schema``/``model_size``
        are passed through unchanged when the caller specifies the end state.
        """
        if caller_end_state is not None:
            if caller_end_state not in END_STATES:
                raise ValueError(
                    f"unknown end_state {caller_end_state!r}; want one of {END_STATES}"
                )
            jepa_default = self._heuristic_end_state(query, retrieved_episodes)
            if caller_end_state != jepa_default:
                self.record_override(query, retrieved_episodes, jepa_default, caller_end_state)
            return EndStatePlan(
                end_state=caller_end_state,
                format_spec=format_spec, extract_schema=extract_schema,
                model_size=model_size, jepa_default=False,
                rationale=f"caller-specified end_state={caller_end_state!r} "
                          f"(heuristic default was {jepa_default!r})",
            )

        chosen = self._heuristic_end_state(query, retrieved_episodes)
        # The only live generation model is the local Bonsai; "synthesize" uses
        # it (default "bonsai" when the caller didn't specify). The other end
        # states make no LLM call, so model_size is None.
        if chosen == END_SYNTHESIZE:
            chosen_model = model_size or "bonsai"
        else:
            chosen_model = None
        return EndStatePlan(
            end_state=chosen,
            format_spec=format_spec if chosen == END_FORMAT else None,
            extract_schema=extract_schema if chosen == END_EXTRACT else None,
            model_size=chosen_model,
            jepa_default=True,
            rationale=self._end_state_rationale(chosen, query, retrieved_episodes),
        )

    def _heuristic_end_state(self, query: str, episodes: list[dict]) -> str:
        n = len(episodes)
        if _has_any(query, _EXTRACT_KEYWORDS):
            return END_EXTRACT
        if _has_any(query, _SYNTHESIS_KEYWORDS) or n > self.cfg.direct_max_episodes:
            return END_SYNTHESIZE
        if _has_any(query, _FACTUAL_LOOKUP_KEYWORDS) and n <= self.cfg.direct_max_episodes:
            return END_DIRECT
        # Default: synthesize (reason across whatever was retrieved).
        return END_SYNTHESIZE

    def _end_state_rationale(self, chosen: str, query: str, episodes: list[dict]) -> str:
        n = len(episodes)
        if chosen == END_EXTRACT:
            return "extract: extract/structured-data verb in query"
        if chosen == END_SYNTHESIZE:
            if _has_any(query, _SYNTHESIS_KEYWORDS):
                return "synthesize: reasoning verb (why/how/compare) in query"
            return f"synthesize: {n}>{self.cfg.direct_max_episodes} episodes → reason across them"
        if chosen == END_DIRECT:
            return f"direct: factual lookup + {n}≤{self.cfg.direct_max_episodes} episodes"
        return "format"

    def record_override(
        self,
        query: str,
        episodes: list[dict],
        jepa_predicted: str,
        caller_chose: str,
    ) -> None:
        """Caller overrode the gate's end-state default → push a training record.

        The override is the only signal for the deferred learned end-state
        router (the end-state feedback signal is weak, chat [146]). No-op
        learning; the buffer is the seed.
        """
        self.override_buffer.push({
            "query": query,
            "episode_count": len(episodes),
            "episode_ids": [e.get("episode_id") for e in episodes if e.get("episode_id")],
            "jepa_predicted": jepa_predicted,
            "caller_chose": caller_chose,
        })

    # ── buffer persistence (Phase 3a Task 7: durable EXPAND-frequency signal) ──

    def serialize_buffers(self) -> dict:
        """JSON-safe snapshot of both ReplayBuffers for cross-session persistence.

        The records are already JSON-safe dicts (``record_outcome`` /
        ``record_override`` build them from primitives), so this is a direct
        ``list(deque)``. Phase 3a Task 7 persists this so the EXPAND-frequency
        salience signal survives restarts (the 2c §15 blocker).
        """
        return {
            "capacity": self.outcome_buffer.capacity,
            "outcome": self.outcome_buffer.to_list(),
            "override": self.override_buffer.to_list(),
        }

    def load_buffers(self, data: dict) -> None:
        """Restore both buffers from a ``serialize_buffers`` snapshot (in place)."""
        cap = int(data.get("capacity", self.outcome_buffer.capacity))
        self.outcome_buffer = ReplayBuffer.from_list(data.get("outcome", []), capacity=cap)
        self.override_buffer = ReplayBuffer.from_list(data.get("override", []), capacity=cap)


# ── helpers ──

def _has_any(text: str, keywords) -> bool:
    """Case-insensitive substring match against any keyword."""
    if not text:
        return False
    low = text.lower()
    return any(kw in low for kw in keywords)