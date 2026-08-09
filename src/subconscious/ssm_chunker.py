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


# Named gist-cue presets for the ``mamba3`` decode backend. Each is a completion
# cue handed to ``voice.ephemeral_gist``; selected via the ``gist_cue_preset``
# ctor arg. A non-None preset OVERRIDES ``gist_cue``; ``None`` (the default) ->
# ``gist_cue`` is used as-is (byte-identical to the pre-preset path). Inert for
# the ``topics`` backend and when query-conditioning is on (those paths do not
# read ``_gist_cue`` -- query-conditioning builds ``Q: {query}\nA:`` directly).
#
# - ``summary``: the original bare completion cue (the byte-identical default).
# - ``instruct-labeled``: extract every concrete fact as ``subject: value``
#   lines. The toy 4-needle eval scored 4/4 (vs 0/4 for ``Summary:``); at scale
#   (12-item LongMemEval forced-secondary stress) 58% (7/12) -- 5/5 single-
#   discrete-fact hits, multi-item/reasoning still hard
#   ([[pondr-mamba3-gist-eval-result]]). Generic examples (no needle values) so
#   the gist is real extraction, not cue-echo.
# - ``instruct-additive``: ``instruct-labeled`` PLUS explicit instructions to
#   capture counts/quantities, time/date/sequence markers, and user preferences
#   -- the three things the failure diagnosis showed the labeled cue dropped.
#   ADDITIVE, not suppressive: a suppressive "do not list the assistant's
#   advice" cue dropped load-bearing assistant-sourced facts (item 3's needle
#   lived in the assistant's shift sheet) -> strictly worse. At scale 67%
#   (8/12): fixes the preference (item 4) and temporal (item 12) failures
#   without breaking the single-fact hits. Thin (12 items; full set 50/500
#   pending); two structural fixes + one sampling loss vs the labeled cue.
GIST_CUE_PRESETS: dict[str, str] = {
    "summary": "Summary:",
    "instruct-labeled": (
        "List every concrete fact from the context WITH its subject, as "
        "'subject: value' on its own line (e.g. 'colleague name: John Smith', "
        "'meeting date: March 15', 'locker code: AB-123'). Cover names, numbers, "
        "dates, codes, IDs. Be exhaustive and literal. Use ONLY the provided context."
    ),
    "instruct-additive": (
        "List every concrete fact from the context WITH its subject, as "
        "'subject: value' on its own line. Cover names, numbers, dates, codes, IDs, "
        "preferences, activities, and roles. Be exhaustive and literal. Make sure to "
        "ALSO capture, when present: (1) any counts or quantities the user states "
        "(e.g. 'restaurants tried: five', 'projects led: two'); (2) any time, date, "
        "or sequence markers (e.g. 'webinar attended: two months ago', 'workshop "
        "attended: last Saturday', 'first event: the workshop'); and (3) the user's "
        "preferences and self-described activities (e.g. 'preferred software: Vim', "
        "'current role: team lead'). Examples: 'colleague name: John Smith', 'meeting "
        "date: March 15', 'locker code: AB-123', 'preferred editor: Vim', 'meetings "
        "attended: three', 'course started: last June'. Use ONLY the provided context."
    ),
}


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
    the rest as an SSM recurrent state (the bge-into-backbone path); ``secondary_episodes``
    retains the compressed episode dicts (their topics feed the formatter's compressed
    summary, and EXPAND can resolve them in-memory before hitting the store).
    ``chunk_map`` and ``expandable_ids`` support EXPAND.

    ``compressed_gist`` (the Mamba3 path, default None): when the gist backend is
    ``"mamba3"`` a textual summary is DECODED from the secondary episodes (the
    deferred Phase 2c "decode a summary from the SSM state" path -- realized via
    a Mamba3 LM prefill+decode), and the formatter emits THIS text instead of the
    topic union. ``None`` (the default ``"topics"`` backend) -> the formatter
    emits the topic union from ``secondary_episodes`` (byte-identical to pre-
    mamba3). An EMPTY string decode (``""``) also falls back to the topic union
    (the formatter treats a falsy gist as "no gist" -- topics > an empty
    ``Summary:`` line). Only one of ``compressed_state`` / a non-empty
    ``compressed_gist`` is produced per chunk() (the backend selects the path);
    the other is None / empty.
    """
    primary_chunks: list[dict]
    compressed_state: Optional[WorkingMemoryState]
    chunk_map: dict[str, int]            # episode_id → primary index, or -1 (compressed)
    expandable_ids: set[str]             # the compressed episode ids (EXPAND targets)
    total_episodes: int
    primary_token_count: int             # len(text)//4 estimate, summed over primary
    compressed_episode_count: int
    secondary_episodes: list[dict] = field(default_factory=list)  # the compressed dicts
    compressed_gist: Optional[str] = None  # mamba3-decoded textual gist (None = topic union)

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
        *,
        voice=None,
        gist_backend: str = "topics",
        gist_cue: str = "Summary:",
        gist_cue_preset: Optional[str] = None,
    ) -> None:
        self.backbone = backbone
        self.embedder = embedder
        self._cfg = config  # Phase2cConfig (or anything with .ssm_chunker)
        chunk_cfg = config.ssm_chunker
        self.max_primary_tokens = chunk_cfg.max_primary_tokens
        self.max_primary_chunks = chunk_cfg.max_primary_chunks
        cfg = instance_config or INSTANCE_CONFIGS["working_memory"]
        # Ephemeral compressor: fresh state per chunk() call (see compress_episodes).
        # This is the bge-into-backbone path (the "topics" backend, default).
        self._compressor = WorkingMemory(
            backbone, config=cfg, embedder=embedder, decay_alpha=1.0
        )
        # Mamba3 gist backend (the deferred Phase 2c path, default OFF). When
        # ``gist_backend == "mamba3"`` and ``voice`` is a Mamba3 LM, the
        # secondary episodes are decoded into a TEXTUAL gist (see
        # ``compress_gist_mamba3``) instead of being stepped into the (falsified)
        # backbone SSM whose state was computed-then-discarded (the ablation
        # [[pondr-backbone-ablation-result]] falsified the backbone's identity
        # objective; the formatter never read ``compressed_state`` anyway).
        # ``voice`` is a ``Mamba3Voice`` (or any object exposing
        # ``ephemeral_gist(texts, cue, max_new_tokens)``); ``None`` (default) ->
        # the topics backend runs (byte-identical). ``gist_cue`` is the
        # completion cue handed to the decoder (a base LM, not instruction-tuned
        # -- a completion-style cue elicits the summary; tunable for sweeps).
        # ``gist_cue_preset`` (optional): a named key into ``GIST_CUE_PRESETS``
        # (``"summary"`` / ``"instruct-labeled"`` / ``"instruct-additive"``).
        # When set it OVERRIDES ``gist_cue`` (an explicit preset wins); ``None``
        # (default) -> ``gist_cue`` is used as-is (byte-identical to pre-preset).
        self.voice = voice
        self.gist_backend = gist_backend
        if gist_cue_preset is not None:
            if gist_cue_preset not in GIST_CUE_PRESETS:
                raise ValueError(
                    f"unknown gist_cue_preset {gist_cue_preset!r}; "
                    f"choose from {sorted(GIST_CUE_PRESETS)}"
                )
            self._gist_cue = GIST_CUE_PRESETS[gist_cue_preset]
        else:
            self._gist_cue = gist_cue

    def chunk(
        self,
        episodes: list[dict],
        presentation_plan,
        *,
        query: Optional[str] = None,
    ) -> ChunkedContext:
        """Split ``episodes`` (ranked, highest score first) into primary + compressed.

        ``presentation_plan`` is the ``PresentationPlan`` from the Presentation
        Gate (axis a); its ``primary_chunk_count`` caps how many primary chunks
        we keep (further bounded by ``max_primary_chunks`` and the token budget).
        Episodes that do not fit the primary budget are compressed into the SSM
        state. ``expandable_ids`` is exactly the compressed set.

        ``query`` (optional, keyword-only): when the gist backend is ``mamba3`` and
        a ``query`` is supplied, the gist is decoded QUERY-CONDITIONED -- the cue
        becomes ``Q: {query}\\nA:`` (the carry path's proven completion shape,
        [[pondr-mamba3-carry-wired]]) instead of the fixed ``gist_cue``. The base
        443M is a recall machine, not a summarizer ([[pondr-mamba3-gist-eval-result]]):
        a global ``Summary:`` cue degenerates (0/8 needles), while a targeted Q-A
        cue that names the asked-for facts recalls them (3/8). Threading the user's
        question into the decode moves the secondary-episodes gist from the
        degenerate "summarize everything" path to the honest "recall what was
        asked" path. ``None`` (default) -> the fixed ``gist_cue`` (byte-identical to
        the pre-query-conditioned mamba3 path); the ``topics`` backend ignores it.
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

        # Compress the secondary set. Two backends, selected by ``gist_backend``:
        #   "topics" (default): step bge summary embeddings into the ephemeral
        #     backbone SSM -> ``compressed_state`` (the gist as a recurrent state
        #     vector). The formatter never reads this state (it emits the topic
        #     union from ``secondary_episodes``), but computing it keeps the
        #     ChunkedContext byte-identical to the pre-mamba3 path.
        #   "mamba3": DECODE a textual gist from the secondary episodes via the
        #     Mamba3 voice -> ``compressed_gist``. The formatter emits THIS text
        #     instead of the topic union, and the backbone compressor is SKIPPED
        #     (no dead-weight SSM step on the falsified backbone). Falls back to
        #     no compression (no state, no gist) when the voice is absent -- the
        #     formatter then emits the topic union from ``secondary_episodes``.
        if secondary:
            if self.gist_backend == "mamba3" and self.voice is not None:
                compressed_state = None
                compressed_gist = self.compress_gist_mamba3(secondary, query=query)
            elif self.gist_backend == "mamba3":
                # mamba3 requested but no voice loaded -> skip the backbone
                # compressor (the user opted out of the topics path); the
                # formatter falls back to the topic union from secondary_episodes.
                compressed_state = None
                compressed_gist = None
            else:
                compressed_state = self.compress_episodes(secondary)
                compressed_gist = None
        else:
            compressed_state = None
            compressed_gist = None
        return ChunkedContext(
            primary_chunks=primary_chunks,
            compressed_state=compressed_state,
            chunk_map=chunk_map,
            expandable_ids={ep["episode_id"] for ep in secondary if ep.get("episode_id")},
            total_episodes=len(episodes),
            primary_token_count=token_count,
            compressed_episode_count=len(secondary),
            secondary_episodes=list(secondary),
            compressed_gist=compressed_gist,
        )

    def compress_episodes(self, episodes: list[dict]) -> WorkingMemoryState:
        """Embed each episode summary and step the compressor SSM sequentially.

        Returns the final recurrent state (gist of all the episodes). The
        compressor is reset before each call so the gist is scoped to this
        chunk() — never aliased to a previous query's compression.

        (The "topics" backend. Note: the formatter never reads this state -- it
        emits the topic union from ``secondary_episodes``; the state is
        computed for the byte-identical pre-mamba3 path. The mamba3 backend
        skips this entirely.)
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

    def compress_gist_mamba3(self, episodes: list[dict], *, query: Optional[str] = None) -> str:
        """Decode a textual gist of ``episodes`` via the Mamba3 voice (ephemeral).

        The deferred Phase 2c path (docs/Phase 2c.md lines 552-559): the original
        chat sketch "compresses the prompt by chunking it, stepping each chunk
        through the SSM, and decoding a summary from the SSM state." That decode
        step was not implementable on the Phase 2a backbone (a JEPA predictor
        with no text-decoder head); a Mamba3 LM IS the decoder. Here the
        secondary episode FULL TEXTS (text, falling back to summary) are
        concatenated and prefilled into a FRESH Mamba3 recurrent state, then a
        completion cue elicits a greedy-decoded summary continuation -- the gist
        Bonsai consumes (text, not a state vector).

        The full text (not the summary) is ingested because the whole point of
        the decode path is to surface DETAILS the topic-union labels cannot --
        the summary is already a gist, and summarizing a summary loses the
        facts the decoder is meant to retain. (The bge-into-backbone path uses
        summaries; the decode path needs the content.)

        ``query`` (optional): when supplied, the cue is QUERY-CONDITIONED --
        ``Q: {query}\\nA:`` (the carry path's proven completion shape,
        [[pondr-mamba3-carry-wired]]) -- instead of the fixed ``_gist_cue``. The
        base 443M is a recall machine, not a summarizer ([[pondr-mamba3-gist-eval-
        result]]): a global ``Summary:`` cue degenerates (0/8 needles), while a
        targeted Q-A cue that names the asked-for facts recalls them (3/8). So a
        query-conditioned decode is the honest "recall what was asked" path vs the
        degenerate "summarize everything" path. ``None`` -> ``_gist_cue``
        (byte-identical to the pre-query-conditioned mamba3 path).

        ``voice.ephemeral_gist`` allocates its own ``InferenceParams`` and
        discards it -- the voice's carried state (if any) is never touched, so
        this is safe against a shared fade voice. Returns "" on a cold/empty
        decode (the formatter then emits the topic union as a fallback).
        """
        if self.voice is None:
            return ""
        texts = [ep.get("text", "") or ep.get("summary", "") for ep in episodes]
        cue = f"Q: {query}\nA:" if query is not None else self._gist_cue
        try:
            return self.voice.ephemeral_gist(texts, cue)
        except Exception:
            # Best-effort: a decode failure must not break the query. The
            # formatter falls back to the topic union from secondary_episodes.
            return ""

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