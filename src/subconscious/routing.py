"""Routing vocab, decision/outcome dataclasses for the Retrieval Gate (Phase 2b).

The Retrieval Gate is the first trained JGS instance (``retrieval_gate.py``). It
routes a query *before* retrieval into a structured decision: which domain(s)
to query, which pathway to use, which meta-skills are required, what model size
is needed, and whether conscious deliberation is required.

The vocabularies here are **the labels the Phase 1d Oracle actually emits**
(see ``src/training/prompts.py:jepa_routing_prompt`` and
``scripts/generate_jepa_training_data.py``), not the larger speculative sets in
the original ``docs/Phase 2b.md`` draft. Aligning the head sizes to the real
labels means every Oracle record maps to a non-zero training target — a 13-skill
head would waste capacity on 5 skills the Oracle never labels.

This module is dependency-light (stdlib only — ``reward`` takes no tensors and
the module imports none) so the subconscious package stays importable on a fresh
dev box with only torch installed. The text embedder is an *injected* ``Embedder``
Protocol (structurally compatible with ``src/retrieval/vector_search.py:Embedder``);
it is NOT imported here, so the package has no hard dep on ``sentence_transformers``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from .gate import GateContext, GateDecision

if TYPE_CHECKING:
    import torch


# ── Vocabularies (match the Oracle's real labels — see module docstring) ──

AVAILABLE_DOMAINS: list[str] = [
    "database",        # WaveDB, Postgres, HBTrie, SQL, configuration, performance
    "coding",          # Python, Rust, Dart, tree-sitter, AST parsing, code review
    "robotics",        # actuators, sensors, inverse kinematics, control policies
    "economics",       # Spark Ledger, monetary theory, QE, zk-SNARKs
    "ai_architecture",  # neural networks, cognitive systems, memory models
    "personal",        # user preferences, relationships, emotional patterns
]

PATHWAYS: list[str] = [
    "ssm_direct",              # Answer from Working Memory. No retrieval.
    "graph_retrieve",          # Query the memory graph.
    "process_exec",            # Execute a stored process.
    "tool_plan",               # Plan a multi-step tool strategy.
    "conscious_deliberation",  # Engage System 2 for complex reasoning.
]

META_SKILLS: list[str] = [
    "factual_recall",       # Retrieve and restate known information
    "basic_synthesis",      # Combine multiple pieces of information
    "pattern_recognition",  # Identify patterns across episodes
    "decomposition",        # Break complex task into sub-tasks
    "process_selection",    # Choose the right stored process
    "creative_synthesis",   # Generate novel ideas or designs
    "security_analysis",    # Identify security implications
    "tradeoff_analysis",    # Evaluate competing constraints
]

MODEL_SIZES: list[str] = ["1B", "3B", "8B", "70B", "175B"]


@runtime_checkable
class Embedder(Protocol):
    """Maps text to fixed-dim float vectors.

    Structurally compatible with ``src/retrieval/vector_search.py:Embedder``
    (``encode(list[str]) -> list[list[float]]``). The Retrieval Gate takes this
    as a constructor/``route_text`` argument so the subconscious package does
    NOT import ``sentence_transformers`` — the caller injects the real embedder
    (bge-small-en-v1.5, 384-dim) or a deterministic stub for offline tests.
    """

    def encode(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class RoutingDecision:
    """Output of the Retrieval Gate for one query.

    ``domains``/`meta_skills`` are multi-label lists (possibly empty). ``pathway``
    and ``model_size`` are single-label strings drawn from their vocabularies.
    ``needs_deliberation`` is a discrete bool. ``confidence`` comes from the
    underlying ``GateDecision`` (excite−inhibit spread). ``gate_decision`` keeps
    the raw gate output for the outcome-based trainer; it is ``None`` only when a
    caller builds a ``RoutingDecision`` by hand (e.g. the outcome trainer's
    recorded "wrong" decisions — see the tests).

    This holds **discrete, detached** choices — no logits. The outcome trainer
    re-runs ``RetrievalGate.forward`` on the stored embedding+context to recover
    fresh logits for the policy-gradient step, so the decision carries no tensor
    state that could leak across steps.
    """

    domains: list[str]
    pathway: str
    meta_skills: list[str]
    model_size: str
    needs_deliberation: bool
    confidence: float
    gate_decision: Optional[GateDecision] = None


@dataclass
class RoutingOutcome:
    """Observed outcome of a routing decision, for REINFORCE personalization.

    Each field is a bool signal. ``reward`` reduces them to a scalar with the
    weighting from ``docs/Phase 2b.md`` §3.3:

    - user accepted the response: **+1.0**
    - user corrected / rejected:   **−1.0**
    - had to delegate unexpectedly (route was too low): **−0.3**
    - used a larger model than needed (overkill):       **−0.1**
    - response was fast (efficiency bonus):             **+0.1**
    - had to EXPAND mid-response (route was incomplete): **−0.3**

    These are the Phase 2b *signal definitions*; wiring real user-feedback /
    efficiency / delegation detectors into the live pipeline is a later phase
    (the integration records outcomes via ``HippocampalRetriever.record_outcome``
    when the signals become available; until then the trainer is exercised in
    tests with synthetic outcomes).
    """

    user_accepted: bool = False
    user_corrected: bool = False
    had_to_delegate: bool = False
    model_was_overkill: bool = False
    response_fast: bool = False
    had_to_expand: bool = False

    def reward(self) -> float:
        """Scalar reward from the bool signals (doc §3.3 weighting)."""
        r = 0.0
        r += 1.0 if self.user_accepted else 0.0
        r -= 1.0 if self.user_corrected else 0.0
        r -= 0.3 if self.had_to_delegate else 0.0
        r -= 0.1 if self.model_was_overkill else 0.0
        r += 0.1 if self.response_fast else 0.0
        r -= 0.3 if self.had_to_expand else 0.0
        return r


@dataclass
class RoutingReplayEntry:
    """One (state, decision, outcome) tuple for the outcome-based trainer.

    Stores the prompt embedding and the gate context that produced ``decision``,
    so the trainer can re-run ``RetrievalGate.forward`` to recover fresh logits
    for the policy-gradient step (the decision itself only carries discrete
    choices). Mirrors the generic ``ReplayEntry`` shape but with routing-specific
    fields; uses the shared ``ReplayBuffer`` for storage.

    ``filled`` is set ``True`` by ``record_outcome`` (outcomes are recorded
    complete, not filled in later); ``train_from_outcomes`` checks it defensively.
    """

    embedding: "torch.Tensor"  # [batch, input_dim] (batch=1 for a recorded decision)
    context: Optional[GateContext]   # GateContext at decision time, or None
    decision: RoutingDecision
    outcome: Optional[RoutingOutcome] = None
    filled: bool = field(default=False)