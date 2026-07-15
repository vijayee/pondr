"""SSM Chunker: compress less-relevant retrieved episodes into the SSM state.

Phase 2c. The generation model has a finite context window. Rather than
truncate retrieved episodes (silently dropping some the model never knows
existed), the chunker splits the ranked episode list into:

- **primary chunks**: the most-relevant episodes, kept as full text (the detail).
- **compressed state**: the remaining episodes, stepped into a SSM as
  summary embeddings (the gist — recoverable on demand via EXPAND).

The generation model receives the primary full text PLUS a working-memory
state encoding the gist of everything else. "You remember the gist of
everything you've read. You remember the exact words of almost nothing. When
you need the exact words, you go back to the source. The SSM is the gist.
EXPAND is going back to the source." (chat [128]; docs/Ponder Engine Chat
Facts.md §2).

The compressor is a *separate* ``WorkingMemory`` instance so it does not
pollute the user's persistent working memory — each chunk() call compresses
into a fresh, ephemeral state. (The user's WM is updated by the orchestrator;
this chunker only builds the per-query compressed context.)

Episode dicts are the shape ``GraphTraversal._hydrate`` produces:
``episode_id``, ``text`` (full_text), ``summary``, ``timestamp``, ``entities``,
``topics``, ``tones``, ``decisions``, ``score``. Episodes are assumed already
sorted by retrieval relevance (highest score first) — the chunker does not
re-rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch import Tensor

from .configs import INSTANCE_CONFIGS, InstanceConfig
from .working_memory import WorkingMemory, WorkingMemoryState


class EpisodeNotExpandable(Exception):
    """Raised when EXPAND is asked for an episode that is already primary.

    A primary-chunk episode is already full text — there is nothing to expand.
    The caller should not call ``expand`` on a primary id; this signals a logic
    error in the caller (e.g. the EXPAND handler mis-routing).
    """


class EpisodeNotFound(KeyError):
    """Raised when EXPAND is asked for an episode id the chunker never saw."""


@dataclass
class ChunkedContext:
    """The result of chunking a ranked episode list for presentation.

    ``primary_chunks`` carry full text; ``compressed_state`` carries the gist of
    the rest as an SSM recurrent state; ``secondary_episodes`` retains the
    compressed episode dicts (their topics feed the formatter's compressed
    summary, and EXPAND can resolve them in-memory before hitting the store).
    ``chunk_map`` and ``expandable_ids`` support EXPAND.
    """
    primary_chunks: list[dict]
    compressed_state: Optional[WorkingMemoryState]
    chunk_map: dict[str, int]            # episode_id → primary index, or -1 (compressed)
    expandable_ids: set[str]             # the compressed episode ids (EXPAND targets)
    total_episodes: int
    primary_token_count: int             # len(text)//4 estimate, summed over primary
    compressed_episode_count: int
    secondary_episodes: list[dict] = field(default_factory=list)  # the compressed dicts

    @property
    def has_compressed(self) -> bool:
        return self.compressed_episode_count > 0


def _estimate_tokens(text: str) -> int:
    """The codebase's len(text)//4 token estimate (no tokenizer dep)."""
    return len(text) // 4


class SSMChunker:
    """Splits ranked episodes into primary full-text + compressed SSM gist.

    Owns an ephemeral compressor ``WorkingMemory`` (separate from the user's
    persistent WM) so compressing episodes into gist does not mutate the user's
    awareness state.
    """

    def __init__(
        self,
        backbone,
        embedder,
        config,
        instance_config: Optional[InstanceConfig] = None,
    ) -> None:
        self.backbone = backbone
        self.embedder = embedder
        self._cfg = config  # Phase2cConfig (or anything with .ssm_chunker)
        chunk_cfg = config.ssm_chunker
        self.max_primary_tokens = chunk_cfg.max_primary_tokens
        self.max_primary_chunks = chunk_cfg.max_primary_chunks
        cfg = instance_config or INSTANCE_CONFIGS["working_memory"]
        # Ephemeral compressor: fresh state per chunk() call (see compress_episodes).
        self._compressor = WorkingMemory(
            backbone, config=cfg, embedder=embedder, decay_alpha=1.0
        )

    def chunk(
        self,
        episodes: list[dict],
        presentation_plan,
    ) -> ChunkedContext:
        """Split ``episodes`` (ranked, highest score first) into primary + compressed.

        ``presentation_plan`` is the ``PresentationPlan`` from the Presentation
        Gate (axis a); its ``primary_chunk_count`` caps how many primary chunks
        we keep (further bounded by ``max_primary_chunks`` and the token budget).
        Episodes that do not fit the primary budget are compressed into the SSM
        state. ``expandable_ids`` is exactly the compressed set.
        """
        primary_cap = min(
            getattr(presentation_plan, "primary_chunk_count", self.max_primary_chunks),
            self.max_primary_chunks,
        )
        primary_chunks: list[dict] = []
        chunk_map: dict[str, int] = {}
        token_count = 0
        secondary: list[dict] = []

        for ep in episodes:
            eid = ep.get("episode_id")
            if eid is None:
                continue
            text = ep.get("text", "") or ep.get("summary", "")
            tok = _estimate_tokens(text)
            if (
                len(primary_chunks) < primary_cap
                and token_count + tok <= self.max_primary_tokens
            ):
                primary_chunks.append(ep)
                chunk_map[eid] = len(primary_chunks) - 1
                token_count += tok
            else:
                secondary.append(ep)
                chunk_map[eid] = -1

        compressed_state = self.compress_episodes(secondary) if secondary else None
        return ChunkedContext(
            primary_chunks=primary_chunks,
            compressed_state=compressed_state,
            chunk_map=chunk_map,
            expandable_ids={ep["episode_id"] for ep in secondary if ep.get("episode_id")},
            total_episodes=len(episodes),
            primary_token_count=token_count,
            compressed_episode_count=len(secondary),
            secondary_episodes=list(secondary),
        )

    def compress_episodes(self, episodes: list[dict]) -> WorkingMemoryState:
        """Embed each episode summary and step the compressor SSM sequentially.

        Returns the final recurrent state (gist of all the episodes). The
        compressor is reset before each call so the gist is scoped to this
        chunk() — never aliased to a previous query's compression.
        """
        if not episodes:
            raise ValueError("compress_episodes called with no episodes")
        self._compressor.reset()
        summaries = [ep.get("summary", "") or ep.get("text", "") for ep in episodes]
        embs = self._compressor.embed(summaries) if self.embedder is not None else []
        for emb in embs:
            self._compressor.inject(emb)
        return self._compressor.snapshot(
            metadata={"compressed_episode_ids": [ep.get("episode_id") for ep in episodes]}
        )

    def expand(
        self,
        episode_id: str,
        chunked: ChunkedContext,
        store=None,
    ) -> dict:
        """EXPAND: load the full text of a compressed episode on demand.

        This is the **chunking-level** EXPAND (the compressed→full-text loader).
        The *trigger* logic (when to auto-EXPAND mid-generation, on low decoder
        confidence) is Phase 4a — see docs/Phase 2c.md §0 (EXPAND is
        double-specified; this phase implements the loader only).

        ``chunked`` carries the ``chunk_map`` / ``expandable_ids`` from the
        ``chunk()`` call, so this method can distinguish the three cases: a
        primary id (already full text — ``EpisodeNotExpandable``), a compressed
        id (load from the store), or an unknown id (``EpisodeNotFound``).
        Primary ids are resolved from the in-memory primary_chunks (already
        full text, no I/O); compressed ids are loaded from ``store``.
        """
        if episode_id not in chunked.chunk_map:
            raise EpisodeNotFound(episode_id)
        idx = chunked.chunk_map[episode_id]
        if idx >= 0:
            # Primary chunk — already full text. EXPAND is meaningless here.
            raise EpisodeNotExpandable(
                f"episode {episode_id!r} is a primary chunk (already full text); "
                f"EXPAND is only for compressed (gist) episodes"
            )
        # Compressed: resolve from the in-memory secondary episodes first
        # (the full text is retained in the ChunkedContext), then the store.
        for ep in chunked.secondary_episodes:
            if ep.get("episode_id") == episode_id:
                return ep
        if store is None:
            raise RuntimeError(
                "SSMChunker.expand: episode not found in the in-memory secondary "
                "set and no store supplied to load it from"
            )
        # Document / section result (the unified doc+episode RAG path): expand
        # pulls the body on demand (``expand`` is NOT the hot retrieve path, so a
        # cold pull is fine here). Build the same dict shape as an episode so
        # downstream chunking/formatting (which read ``.get``) is unaffected.
        # Section ids (``{doc_id}_sec_{i:03d}``) start with ``doc_`` AND contain
        # ``_sec_``, so the ``_sec_`` check MUST precede the ``doc_`` check (else
        # a section id would hit ``get_document(section_id)`` -> None ->
        # EpisodeNotFound). For a doc id, the matched body is the first
        # non-empty section (on-demand EXPAND has no query axes, so there is no
        # "matched" section to pick).
        if "_sec_" in episode_id:
            doc_id = episode_id.rsplit("_sec_", 1)[0]
            doc = store.get_document(doc_id, load_bodies=True)
            if doc is None:
                raise EpisodeNotFound(episode_id)
            sec = next((s for s in doc.sections if s.id == episode_id), None)
            if sec is None:
                raise EpisodeNotFound(episode_id)
            return {
                "episode_id": episode_id,
                "summary": doc.title,
                "text": sec.content or "",
                "timestamp": doc.ingested_at,
                "entities": list(getattr(sec, "entities", []) or []),
                "topics": list(getattr(sec, "topics", []) or []),
                "tones": [],
                "decisions": [],
                "score": 0.0,
                "kind": "section",
                "source_path": doc.source_path,
                "section_heading": sec.heading,
                "doc_id": doc_id,
            }
        if episode_id.startswith("doc_"):
            doc = store.get_document(episode_id, load_bodies=True)
            if doc is None:
                raise EpisodeNotFound(episode_id)
            text = ""
            for sec in doc.sections:
                if sec.content:
                    text = sec.content
                    break
            return {
                "episode_id": episode_id,
                "summary": doc.title,
                "text": text,
                "timestamp": doc.ingested_at,
                "entities": list(getattr(doc, "entities", []) or []),
                "topics": list(getattr(doc, "topics", []) or []),
                "tones": [],
                "decisions": [],
                "score": 0.0,
                "kind": "document",
                "source_path": doc.source_path,
            }
        ep = store.get_episode(episode_id)
        if ep is None:
            raise EpisodeNotFound(episode_id)
        return {
            "episode_id": episode_id,
            "summary": ep.summary,
            "text": ep.full_text,
            "timestamp": ep.timestamp,
            "entities": list(getattr(ep, "entities", []) or []),
            "topics": list(getattr(ep, "topics", []) or []),
            "tones": list(getattr(ep, "tones", []) or []),
            "decisions": list(getattr(ep, "decisions", []) or []),
            "score": 0.0,
        }