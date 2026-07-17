"""Orchestrates the full encoding pipeline.

Composes the two extractors (GLiNER for entities/topics/tones/decisions +
open discovery, Bonsai for relations) with the ``HippocampalStore``. Each
conversation turn becomes one ``Episode`` written atomically to WaveDB.

The encoder is **session-scoped**: it is constructed for a user, and each
conversation is one session (``start_session`` … ``encode_turn`` … ``end_session``).
Episodes chain via ``follows`` *within* a session; sessions chain via
``follows_session`` *across* a user's chats. Globally-unique episode ids come
from a persisted counter on the store, and every episode carries an ``at_time``
edge so cross-session temporal queries scan by timestamp. See
``ontology.py`` and ``store.py`` for the User/Session/Episode hierarchy.

Model defaults are not repeated here — they live on the extractors, which
pull from ``config``. The constructor only takes optional overrides for
experiments, and threads them through.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

from ..memory.episode import Episode
from ..memory.store import HippocampalStore
from .assertion_extractor import extract_state_assertions
from .bonsai_relations import BonsaiRelationExtractor
from .gliner_extractor import GLiNERExtractor


class HippocampalEncoder:
    """Orchestrates extraction and storage of conversation episodes."""

    # Foreground-priority yielding hook (Phase 3c async-distill). Class-level
    # default so instances built via ``object.__new__`` (test fixtures that skip
    # ``__init__``'s GLiNER load) still have the attribute -- ``_extract`` reads
    # it unconditionally. The background distill worker sets an instance
    # attribute (shadowing this) to a blocking callable, and clears it back to
    # ``None`` after the fill. ``None`` = no yielding, byte-identical to the
    # synchronous path.
    pause_gate: Optional[object] = None

    def __init__(
        self,
        store: HippocampalStore,
        user_id: str,
        gliner2_model: Optional[str] = None,
        gliner_decoder_model: Optional[str] = None,
        bonsai_model: Optional[str] = None,
        bonsai_endpoint: Optional[str] = None,
        gliner_device: str = "cpu",
        gliner_timing: bool = False,
    ):
        self.store = store
        self.user_id = user_id
        # Model selection defaults come from config via the extractors; pass
        # overrides through only when provided. gliner_device/gliner_timing
        # thread through to GLiNERExtractor so the live-encode path can run
        # GLiNER on CUDA (the ~20s/conv CPU bottleneck) with an OOM-safe CPU
        # fallback, and optionally log per-stage extraction timing.
        self.gliner = GLiNERExtractor(
            gliner2_model=gliner2_model,
            gliner_decoder_model=gliner_decoder_model,
            device=gliner_device,
            timing=gliner_timing,
        )
        self.bonsai = BonsaiRelationExtractor(
            model=bonsai_model,
            endpoint=bonsai_endpoint,
        )
        self.session_id: Optional[str] = None
        # Intra-session follows chain; reset at each start_session so unrelated
        # conversations don't link into one global chain.
        self.last_episode_id: Optional[str] = None

    def start_session(self, started_at: Optional[str] = None) -> str:
        """Open a new chat session under this encoder's user.

        Returns the new session id (``S:NNNN``). Resets the intra-session
        ``follows`` chain so this conversation's first episode doesn't follow
        the previous conversation's last episode.
        """
        if started_at is None:
            started_at = datetime.now().isoformat()
        self.session_id = self.store.next_session_id()
        self.store.open_session(self.user_id, self.session_id, started_at)
        self.last_episode_id = None
        return self.session_id

    def end_session(self, ended_at: Optional[str] = None) -> None:
        """Close the current session (record ended_at). No-op if none open."""
        if self.session_id is None:
            return
        if ended_at is None:
            ended_at = datetime.now().isoformat()
        self.store.close_session(self.session_id, ended_at)
        self.session_id = None
        self.last_episode_id = None

    def _extract(self, full_text: str, *, degrade_on_extract_fail: bool) -> dict:
        """GLiNER extraction with optional graceful degradation.

        ``degrade_on_extract_fail=False`` (default, corpus path) re-raises so
        ``scripts/process_corpus.py``'s per-conversation isolation handles the
        failure exactly as before. ``degrade_on_extract_fail=True`` (live path)
        extends the existing Bonsai-degrades-to-empty philosophy to GLiNER: a
        transient hiccup yields empty extraction rather than dropping the turn,
        so a live exchange still persists (retrievable via no-axis + the
        backfilled embedding).
        """
        try:
            # Yield to the foreground before the GLiNER GPU call (mirrors the
            # per-Bonsai-call gate in extract_isolated). None (sync path) no-op.
            if self.pause_gate is not None:
                self.pause_gate()
            return self.gliner.extract(full_text)
        except Exception as e:  # noqa: BLE001 - intentional broad guard
            if not degrade_on_extract_fail:
                raise
            print(f"[gliner-fail] degrade_on_extract_fail: {e}", file=sys.stderr)
            return {
                "entities": [], "entity_classes": {}, "topics": [],
                "tones": [], "decisions": [], "discovered": [],
            }

    def _extract_relations(self, full_text: str, episode_id: str) -> list[dict]:
        """Bonsai relation extraction; degrades to ``[]`` on any failure.

        Relations are supplementary -- an episode with no relations is still
        fully usable for entity/topic/tone/decision + semantic retrieval -- so
        a Bonsai failure (unparseable JSON from over-extraction truncation,
        transient server error) degrades to empty relations rather than failing
        the turn. This keeps one hiccup from dropping a whole conversation's
        episodes.
        """
        try:
            return self.bonsai.extract(full_text)
        except Exception as e:  # noqa: BLE001 - intentional broad guard
            print(f"[bonsai-fail] {episode_id}: {e}", file=sys.stderr)
            return []

    def _build_state_assertions(
        self, full_text: str, decisions: list[str], relations: list[dict]
    ) -> list[dict]:
        """Build the episode's ``state_assertions`` (Phase 3c, D1).

        The deterministic normalizer (``extract_state_assertions``) scans the
        full text + decision spans for explicit ``entity -> value`` field
        patterns, AND lifts any Bonsai ``has_state``/``state`` relations, in
        one deduped union (Bonsai wins on overlap, deterministic fills when
        Bonsai returns none). The store gates the WRITE of ``(E:entity, state,
        value)`` edges on ``config.assertion_extraction_enabled``; this method
        only populates the episode field, so the extraction is always free +
        inert (no patterns -> empty list -> no edges downstream).
        """
        try:
            return extract_state_assertions(full_text, decisions, relations)
        except Exception as e:  # noqa: BLE001 - never let a regex hiccup drop a turn
            print(f"[assertion-fail] {e}", file=sys.stderr)
            return []

    def _apply_overrides(
        self, episode: Episode, *, salience, utility_decay_rate,
        summary_embedding, embedder,
    ) -> None:
        """Apply caller-supplied overrides to an in-memory episode (no store).

        ``salience`` / ``utility_decay_rate`` set the episode's persistence
        levers (the forgetting dream pass fades ``utility_score *= (1 -
        decay_rate)**days``). The embedding: an explicit ``summary_embedding``
        wins; else if an ``embedder`` callable is given, backfill it from the
        episode summary so the live episode is semantically retrievable without
        a separate FAISS rebuild (the graph path surfaces it immediately). The
        embedder returns one vector per text (lists, numpy arrays, or 1-D
        tensors all coerce to ``list[float]`` for JSON persistence).
        """
        if salience is not None:
            episode.salience = salience
        if utility_decay_rate is not None:
            episode.utility_decay_rate = utility_decay_rate
        if summary_embedding is not None:
            episode.summary_embedding = summary_embedding
        elif embedder is not None:
            vec = embedder([episode.summary])[0]
            episode.summary_embedding = [float(x) for x in vec]

    def _apply_overrides_and_store(
        self, episode: Episode, *, salience, utility_decay_rate,
        summary_embedding, embedder,
    ) -> None:
        """Apply caller-supplied overrides, then store atomically.

        Thin wrapper over ``_apply_overrides`` + ``store.encode_episode``; the
        synchronous corpus/live path. The async-distill path calls
        ``_apply_overrides`` alone (the stub), then stores content and edges
        separately via ``store.encode_episode_content`` / ``encode_episode_edges``.
        """
        self._apply_overrides(
            episode, salience=salience, utility_decay_rate=utility_decay_rate,
            summary_embedding=summary_embedding, embedder=embedder,
        )
        self.store.encode_episode(episode)

    def encode_turn(
        self,
        user_message: str,
        assistant_response: str,
        *,
        salience: Optional[float] = None,
        utility_decay_rate: Optional[float] = None,
        summary_embedding: Optional[list[float]] = None,
        embedder=None,
        degrade_on_extract_fail: bool = False,
        origin: str = "corpus",
    ) -> Episode:
        """Encode a single conversation turn.

        Requires an open session (call ``start_session`` first, or use
        ``encode_conversation`` which manages the session lifecycle).

        Keyword-only overrides (all optional; defaults preserve the corpus
        behavior): ``salience`` / ``utility_decay_rate`` set the persistence
        levers, ``summary_embedding`` / ``embedder`` backfill the semantic
        embedding, ``degrade_on_extract_fail`` makes a GLiNER hiccup degrade to
        empty extraction instead of failing the turn, ``origin`` tags the
        episode source (``"corpus"`` default / ``"live"``).
        """
        if self.session_id is None:
            raise RuntimeError("encode_turn requires an open session; call start_session() first.")

        episode_id = self.store.next_episode_id()
        full_text = f"User: {user_message}\nAssistant: {assistant_response}"

        extracted = self._extract(full_text, degrade_on_extract_fail=degrade_on_extract_fail)
        relations = self._extract_relations(full_text, episode_id)

        episode = Episode.from_extraction(
            episode_id=episode_id,
            user_message=user_message,
            assistant_response=assistant_response,
            extracted=extracted,
            relations=relations,
            follows=self.last_episode_id,
            user_id=self.user_id,
            session_id=self.session_id,
            origin=origin,
        )
        episode.state_assertions = self._build_state_assertions(
            full_text, episode.decisions, episode.relations
        )

        self._apply_overrides_and_store(
            episode, salience=salience, utility_decay_rate=utility_decay_rate,
            summary_embedding=summary_embedding, embedder=embedder,
        )
        self.last_episode_id = episode_id
        return episode

    def encode_messages(
        self,
        messages: list[dict],
        *,
        origin: str = "corpus",
        salience: Optional[float] = None,
        utility_decay_rate: Optional[float] = None,
        summary_embedding: Optional[list[float]] = None,
        embedder=None,
        degrade_on_extract_fail: bool = False,
    ) -> Episode:
        """Encode a turn from already-role-tagged segments (the live path).

        ``messages`` is the OpenAI Chat Completions shape (``{role, content,
        ...}``) -- the same shape the orchestrator receives as
        ``conversation_history``. Extraction runs over the joined ``full_text``
        (so GLiNER/Bonsai see the same text the corpus path would), the episode
        is built via ``Episode.from_messages`` (which derives ``full_text`` +
        ``summary`` from the segments), then the same override/store path as
        ``encode_turn``. One encode pipeline, two entry points.

        Requires an open session (call ``start_session`` first).
        """
        if self.session_id is None:
            raise RuntimeError("encode_messages requires an open session; call start_session() first.")

        episode_id = self.store.next_episode_id()
        full_text = Episode._join_messages(messages)

        extracted = self._extract(full_text, degrade_on_extract_fail=degrade_on_extract_fail)
        relations = self._extract_relations(full_text, episode_id)

        episode = Episode.from_messages(
            episode_id=episode_id,
            messages=messages,
            extracted=extracted,
            relations=relations,
            follows=self.last_episode_id,
            user_id=self.user_id,
            session_id=self.session_id,
            origin=origin,
        )
        episode.state_assertions = self._build_state_assertions(
            full_text, episode.decisions, episode.relations
        )

        self._apply_overrides_and_store(
            episode, salience=salience, utility_decay_rate=utility_decay_rate,
            summary_embedding=summary_embedding, embedder=embedder,
        )
        self.last_episode_id = episode_id
        return episode

    # ── Async-distill entry points (stub-then-fill + pre-allocated id) ──
    #
    # The synchronous ``encode_messages`` fuses build + extract + store into one
    # call on the main thread. The async path splits it so the 22 s extraction
    # runs on a background worker while the response has already returned:
    #
    #   main thread:  id = store.next_episode_id()
    #                 episode = encode_messages_stub(messages, id, ...)  # no extract, no store
    #                 store.encode_episode_content(id, episode)         # stub: content + vector
    #                 self.last_episode_id = id                          # follows chain
    #                 enqueue(worker, id, episode)  ->  return response
    #   worker:       episode = encode_messages_fill(episode, id)       # GLiNER + 10-pass Bonsai
    #                 store.encode_episode_edges(id, episode)           # fill: graph edges
    #
    # Contract (see plan async-distill-stub.md): the stub runs on the main
    # thread and may READ ``last_episode_id`` / ``session_id`` / ``user_id``
    # (for the ``follows`` + scope fields); the fill runs on the worker and
    # touches NONE of that state -- it only reads ``episode.full_text`` and the
    # handed-in id. Neither calls ``next_episode_id`` (the caller pre-allocated
    # the id so the persisted counter is main-thread-only). Neither touches
    # ``last_episode_id`` (the orchestrator sets it on the main thread).

    def encode_messages_stub(
        self,
        messages: list[dict],
        episode_id: str,
        *,
        origin: str = "corpus",
        salience: Optional[float] = None,
        utility_decay_rate: Optional[float] = None,
        summary_embedding: Optional[list[float]] = None,
        embedder=None,
    ) -> Episode:
        """Build the stub episode: content fields + summary embedding, NO
        extraction, NO store.

        Derives ``full_text`` + ``summary`` from the role-tagged segments via
        ``Episode.from_messages`` with empty extraction, applies the
        salience/decay/embedding overrides (the embedding is the one synchronous
        cost the async path keeps -- one embedder call, ms), and returns the
        episode in memory. The caller (orchestrator) stores it via
        ``store.encode_episode_content`` and sets ``last_episode_id``.

        Does NOT call ``next_episode_id`` (``episode_id`` is pre-allocated by
        the caller) and does NOT mutate ``last_episode_id`` (the caller sets it
        after the stub write). Reads ``last_episode_id`` / ``session_id`` /
        ``user_id`` for the ``follows`` + scope fields -- main-thread-only state.

        Requires an open session (call ``start_session`` first).
        """
        if self.session_id is None:
            raise RuntimeError(
                "encode_messages_stub requires an open session; call start_session() first."
            )

        episode = Episode.from_messages(
            episode_id=episode_id,
            messages=messages,
            extracted={},
            relations=[],
            follows=self.last_episode_id,
            user_id=self.user_id,
            session_id=self.session_id,
            origin=origin,
        )
        self._apply_overrides(
            episode, salience=salience, utility_decay_rate=utility_decay_rate,
            summary_embedding=summary_embedding, embedder=embedder,
        )
        return episode

    def encode_messages_fill(
        self,
        episode: Episode,
        episode_id: str,
        *,
        degrade_on_extract_fail: bool = True,
    ) -> Episode:
        """Run the heavy extraction on a stub episode and populate its graph
        fields in memory (NO store).

        GLiNER (entities/topics/tones/decisions) + Bonsai relations + the
        deterministic state-assertion normalizer run over ``episode.full_text``
        (set by ``encode_messages_stub``); the results populate
        ``entities``/``entity_classes``/``topics``/``tones``/``decisions``/
        ``relations``/``state_assertions``. The caller (worker) then stores the
        edges via ``store.encode_episode_edges(episode_id, episode)``.

        Worker-safe: reads ONLY ``episode.full_text`` and ``episode_id``. Does
        NOT call ``next_episode_id``, does NOT touch ``last_episode_id`` /
        ``session_id`` / ``WorkingMemory``, does NOT store. Defaults
        ``degrade_on_extract_fail=True`` (the live async path: a transient
        GLiNER hiccup yields empty extraction, the stub keeps the turn
        vector-retrievable).
        """
        full_text = episode.full_text
        extracted = self._extract(full_text, degrade_on_extract_fail=degrade_on_extract_fail)
        relations = self._extract_relations(full_text, episode_id)

        episode.entities = extracted.get("entities", [])
        episode.entity_classes = extracted.get("entity_classes", {})
        episode.topics = extracted.get("topics", [])
        episode.tones = extracted.get("tones", [])
        episode.decisions = extracted.get("decisions", [])
        episode.relations = relations
        episode.state_assertions = self._build_state_assertions(
            full_text, episode.decisions, relations
        )
        return episode

    def encode_conversation(self, turns: list[tuple[str, str]]) -> list[Episode]:
        """Encode a full conversation as one session.

        Opens a session, encodes each turn (chained via ``follows`` within the
        session), then closes the session. Each conversation is its own session
        under the user — unrelated conversations are NOT linked into one global
        episode chain; cross-session order is via ``follows_session`` +
        ``at_time``.
        """
        self.start_session()
        episodes: list[Episode] = []
        try:
            for user_msg, assistant_msg in turns:
                ep = self.encode_turn(user_msg, assistant_msg)
                episodes.append(ep)
        finally:
            self.end_session()
        return episodes