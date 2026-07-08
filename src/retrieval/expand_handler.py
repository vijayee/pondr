"""ExpandHandler: chunking-level EXPAND (load full text of a compressed episode).

Phase 2c. The generation model (or any consumer) may request the full text of
an episode that was compressed into the SSM gist. This handler:

1. Loads the full text via ``SSMChunker.expand`` (in-memory secondary set first,
   then the WaveDB store).
2. Injects the loaded episode into working memory as an embedding-step (the
   retrieved content refreshes the WM state — "retrieval is a signal that the
   memory matters", chat [092]; docs/Ponder Engine Chat Facts.md §7).
3. Returns ``(full_text, updated_snapshot)`` so the caller can inject the full
   text into the generation context and resume.

This is the **chunking-level** EXPAND loader only. The *trigger* — when to
auto-EXPAND mid-generation on low decoder confidence — is Phase 4a
(docs/Phase 2c.md §0, EXPAND is double-specified; this phase does the loader).

Counts EXPAND invocations for the Presentation Gate outcome signal
(``expand_count`` — something important was compressed → should have been
primary; chat [130]).
"""

from __future__ import annotations

from typing import Optional

from ..subconscious.ssm_chunker import (
    ChunkedContext,
    EpisodeNotFound,
    EpisodeNotExpandable,
    SSMChunker,
)
from ..subconscious.working_memory import WorkingMemory, WorkingMemoryState


class ExpandHandler:
    """Chunking-level EXPAND: load full text + inject into working memory."""

    def __init__(self, chunker: SSMChunker, working_memory: WorkingMemory, store=None) -> None:
        self._chunker = chunker
        self._wm = working_memory
        self._store = store
        self.expand_count = 0  # for the PresentationOutcome signal

    def handle_expand(
        self,
        episode_id: str,
        chunked: ChunkedContext,
        embedder=None,
        store=None,
    ) -> tuple[str, WorkingMemoryState]:
        """Load full text of a compressed episode and inject it into WM.

        Args:
            episode_id: the id to expand (must be a compressed/gist episode).
            chunked: the ChunkedContext from the current query's ``chunk()``.
            embedder: optional embedder to step the loaded episode into WM
                (defaults to the WM's own injected embedder).
            store: optional store fallback (defaults to the handler's store).

        Returns:
            ``(full_text, updated_snapshot)``. The snapshot reflects the WM
            state after absorbing the expanded episode.

        Raises:
            ``EpisodeNotExpandable`` if the id is a primary chunk (already full
            text), ``EpisodeNotFound`` if unknown.
        """
        store = store if store is not None else self._store
        ep = self._chunker.expand(episode_id, chunked, store=store)
        full_text = ep.get("text", "") or ep.get("summary", "")

        # Inject the expanded episode's summary as a WM step (retrieval refresh).
        if self._wm is not None:
            emb = self._embed_episode(ep, embedder)
            if emb is not None:
                self._wm.inject(emb)

        self.expand_count += 1
        snapshot = self._wm.snapshot() if self._wm is not None else None
        return full_text, snapshot

    def _embed_episode(self, ep: dict, embedder):
        """Embed the episode summary for the WM injection step."""
        text = ep.get("summary", "") or ep.get("text", "")
        if not text:
            return None
        embdr = embedder if embedder is not None else getattr(self._wm, "_embedder", None)
        if embdr is None:
            return None
        return self._wm.embed([text])[0] if hasattr(self._wm, "embed") else None

    @property
    def outcome_expand_count(self) -> int:
        """EXPAND count for the PresentationOutcome (reset per query by the orchestrator)."""
        return self.expand_count