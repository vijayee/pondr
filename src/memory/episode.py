"""Episode data model — the atomic unit of episodic memory."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Episode:
    """One complete conversational exchange (user message + assistant response).

    The atomic unit of encoding: the smallest unit that contains everything
    needed for retrieval — who, what, how felt, what decided, what next.

    Content (``summary``, ``full_text``, ``timestamp``) is the "neocortical"
    payload stored in the WaveDB HBTrie; structure (entities/topics/tones/
    decisions/relations/follows) is the "hippocampal index" stored as triples in
    the WaveDB Graph layer. See ``store.py``.
    """

    id: str
    timestamp: str
    summary: str
    full_text: str
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    tones: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    follows: Optional[str] = None
    # User/session scope for global chat history. None = unscoped (backward-
    # compatible with the single-session Phase 1a model). When set, the store
    # writes (U:user, has_session, S:session), (S:session, has_episode, ep),
    # (ep, in_session, S:session), (ep, at_time, <ts>), and the session's
    # started_at/ended_at + follows_session chain. See ontology.py.
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    salience: float = 0.5
    state: str = "current"
    validity_start: Optional[str] = None
    validity_end: Optional[str] = None

    # ── Downstream-system fields (populated by Phase 1a, read by Phase 2-4) ──
    # Present from the start so the store schema never has to migrate later.
    # Phase 1a writes safe defaults; later phases update them on retrieval /
    # consolidation. Phase 1a persists retrieval_count / ltp_phase /
    # utility_decay_rate; the rest are reserved for Phase 3+ to populate.
    retrieval_count: int = 0
    ltp_phase: str = "early"  # "early" | "late" — long-term potentiation stage
    consolidation_window_start: Optional[str] = None
    utility_decay_rate: float = 0.01
    retrieval_timestamps: list[str] = field(default_factory=list)
    saturation_flags: int = 0

    def __post_init__(self) -> None:
        # A fact's validity begins when the episode was encoded. Reconsolidation
        # sets validity_end when a newer episode supersedes this one — the old
        # version is preserved (MVCC), just not returned by default queries.
        if self.validity_start is None:
            self.validity_start = self.timestamp

    @classmethod
    def from_extraction(
        cls,
        episode_id: str,
        user_message: str,
        assistant_response: str,
        extracted: dict,
        relations: list[dict],
        follows: Optional[str] = None,
        timestamp: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "Episode":
        """Create an Episode from extraction results.

        Args:
            episode_id: Stable id (e.g. ``ep_0001``).
            user_message: The user side of the turn.
            assistant_response: The assistant side of the turn.
            extracted: Output of ``GLiNERExtractor.extract`` — keys
                ``entities`` / ``topics`` / ``tones`` / ``decisions`` / ``discovered``.
            relations: Output of ``BonsaiRelationExtractor.extract`` — list of
                ``{"subject", "predicate", "object"}`` triples.
            follows: Id of the previous episode in the conversation chain.
            timestamp: ISO timestamp; defaults to now (override for tests).
        """
        full_text = f"User: {user_message}\nAssistant: {assistant_response}"
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Simple summary: first 200 chars of the assistant response.
        summary = assistant_response[:200]
        if len(assistant_response) > 200:
            summary += "..."

        return cls(
            id=episode_id,
            timestamp=timestamp,
            summary=summary,
            full_text=full_text,
            entities=extracted.get("entities", []),
            topics=extracted.get("topics", []),
            tones=extracted.get("tones", []),
            decisions=extracted.get("decisions", []),
            relations=relations,
            follows=follows,
            user_id=user_id,
            session_id=session_id,
        )