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

from datetime import datetime
from typing import Optional

from ..memory.episode import Episode
from ..memory.store import HippocampalStore
from .bonsai_relations import BonsaiRelationExtractor
from .gliner_extractor import GLiNERExtractor


class HippocampalEncoder:
    """Orchestrates extraction and storage of conversation episodes."""

    def __init__(
        self,
        store: HippocampalStore,
        user_id: str,
        gliner2_model: Optional[str] = None,
        gliner_decoder_model: Optional[str] = None,
        bonsai_model: Optional[str] = None,
        bonsai_endpoint: Optional[str] = None,
    ):
        self.store = store
        self.user_id = user_id
        # Model selection defaults come from config via the extractors; pass
        # overrides through only when provided.
        self.gliner = GLiNERExtractor(
            gliner2_model=gliner2_model,
            gliner_decoder_model=gliner_decoder_model,
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

    def encode_turn(self, user_message: str, assistant_response: str) -> Episode:
        """Encode a single conversation turn.

        Requires an open session (call ``start_session`` first, or use
        ``encode_conversation`` which manages the session lifecycle).
        """
        if self.session_id is None:
            raise RuntimeError("encode_turn requires an open session; call start_session() first.")

        episode_id = self.store.next_episode_id()
        full_text = f"User: {user_message}\nAssistant: {assistant_response}"

        # 1. Extract entities, topics, tones, decisions (+ open discovery).
        extracted = self.gliner.extract(full_text)

        # 2. Extract relations.
        relations = self.bonsai.extract(full_text)

        # 3. Create episode, scoped to the current user/session.
        episode = Episode.from_extraction(
            episode_id=episode_id,
            user_message=user_message,
            assistant_response=assistant_response,
            extracted=extracted,
            relations=relations,
            follows=self.last_episode_id,
            user_id=self.user_id,
            session_id=self.session_id,
        )

        # 4. Store atomically (content + graph index in one batch_sync).
        self.store.encode_episode(episode)
        self.last_episode_id = episode_id

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