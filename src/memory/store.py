"""WaveDB wrapper for the hippocampal memory system.

Graph layer (``memory`` subtree) = hippocampal index (sparse pointers:
who/what/when/tone/decisions/relations). HBTrie (``content/`` subtree) =
neocortical store (content: summary, full text, timestamp).

Each episode is written as ONE atomic ``WaveDB.batch_sync`` that merges content
puts with graph-index ops from ``GraphLayer.expand_triple`` — content and index
share a single transaction / WAL record, so encoding is all-or-nothing. The
GraphLayer opens a subtree named "memory" inside the parent database, so the
graph triples and the HBTrie content share one database with namespace
isolation. The WaveDB binding's ``close()`` refuses to destroy the database
while child handles are open, so ``HippocampalStore.close()`` closes the graph
layer first.
"""

import json
from typing import Optional

from wavedb import GraphLayer, WaveDB, WaveDBConfig

from .episode import Episode
from .ontology import SEED_ONTOLOGY


def _b2s(v) -> str:
    """Decode a WaveDB value (bytes) to str; '' for missing/None."""
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)


class HippocampalStore:
    """Wraps WaveDB for hippocampal memory operations."""

    def __init__(self, db_path: str, config: dict | None = None):
        cfg = WaveDBConfig(
            lru_memory_mb=config.get("lru_memory_mb", 100) if config else 100,
            wal_sync_mode=config.get("wal_sync_mode", "debounced") if config else "debounced",
        )
        self.db = WaveDB(db_path, config=cfg)
        self.db_path = db_path  # exposed so retrieval can load a persisted vector index
        # "memory" is the subtree prefix the graph layer lives under.
        self.graph = GraphLayer("memory", self.db)
        self._seed_ontology()

    # ---- ontology ----

    def _seed_ontology(self) -> None:
        """Store the seed taxonomy as subClassOf triples in the graph layer.

        Run ONCE per database, gated by a marker key (``content/system/
        ontology_seeded``). Reopening a store must NOT re-insert the ~1448
        seed triples: ``GraphLayer.insert_sync`` on an already-seeded graph is
        slow, and at scale on a reopened database it can hang (the graph's
        HBTrie index has crossed the btree-split threshold). The marker makes
        reopen a pure read of one key. Not part of the per-episode atomic
        batch (ontology seeding has no content to stay consistent with).
        """
        marker = "content/system/ontology_seeded"
        if _b2s(self.db.get_sync(marker)):
            return
        ops: list[dict] = []
        for parent, info in SEED_ONTOLOGY["classes"].items():
            for child in info.get("subclasses", []):
                ops += self.graph.expand_triple(child, "subClassOf", parent)
        ops.append({"type": "put", "key": marker, "value": "1"})
        self.db.batch_sync(ops)

    # ---- encode ----

    def encode_episode(self, episode: Episode) -> None:
        """Store episode content + graph index in ONE atomic batch.

        Content (HBTrie, ``content/ep/{id}/...``) and graph index (``memory``
        subtree) are written through a single ``WaveDB.batch_sync`` so an
        episode is either fully stored or not at all — no content without its
        index entries, no index entries without their content. Graph triples
        are expanded into root-namespace ops via ``GraphLayer.expand_triple``
        (shipped in WaveDB 0.1.4) and spliced into the same batch; the batch is
        only submitted once every op has been built, so a mid-build exception
        leaves the store untouched.
        """
        ops: list[dict] = []
        eid = episode.id

        # ── HBTrie: content (neocortical store), root namespace under content/ ──
        ops += [
            {"type": "put", "key": f"content/ep/{eid}/summary", "value": episode.summary},
            {"type": "put", "key": f"content/ep/{eid}/text", "value": episode.full_text},
            {"type": "put", "key": f"content/ep/{eid}/ts", "value": episode.timestamp},
            {"type": "put", "key": f"content/ep/{eid}/salience", "value": str(episode.salience)},
            {"type": "put", "key": f"content/ep/{eid}/state", "value": episode.state},
            # Downstream-system fields (defaults at encode time; Phase 2-4
            # update them on retrieval / consolidation). Persisted now so the
            # store schema is stable from the start.
            {"type": "put", "key": f"content/ep/{eid}/retrieval_count", "value": str(episode.retrieval_count)},
            {"type": "put", "key": f"content/ep/{eid}/ltp_phase", "value": episode.ltp_phase},
            {"type": "put", "key": f"content/ep/{eid}/decay_rate", "value": str(episode.utility_decay_rate)},
            # Phase 1b: persistence for the remaining downstream fields so the
            # store schema is fully stable. saturation_flags / retrieval_timestamps
            # are always written (defaults are meaningful); consolidation_window_start
            # and summary_embedding are optional and only written when set, like
            # validity_start above.
            {"type": "put", "key": f"content/ep/{eid}/saturation_flags", "value": str(episode.saturation_flags)},
            {"type": "put", "key": f"content/ep/{eid}/retrieval_timestamps", "value": json.dumps(episode.retrieval_timestamps)},
        ]
        if episode.consolidation_window_start:
            ops.append({"type": "put", "key": f"content/ep/{eid}/consolidation_window_start", "value": episode.consolidation_window_start})
        if episode.summary_embedding:
            ops.append({"type": "put", "key": f"content/ep/{eid}/embedding", "value": json.dumps(episode.summary_embedding)})

        # ── Graph: sparse pointers (hippocampal index) via expand_triple ──
        # expand_triple returns root-namespace ops (the "memory/" subtree prefix
        # is already prepended by the C helper) — splice into the SAME batch_sync
        # so content and graph indices share one atomic transaction / WAL record.
        for entity in episode.entities:
            ops += self.graph.expand_triple(eid, "has_entity", f"E:{entity}")
            ops += self.graph.expand_triple(f"E:{entity}", "in_episode", eid)
        for topic in episode.topics:
            ops += self.graph.expand_triple(eid, "has_topic", f"T:{topic}")
        for tone in episode.tones:
            ops += self.graph.expand_triple(eid, "has_tone", f"A:{tone}")
        for decision in episode.decisions:
            ops += self.graph.expand_triple(eid, "has_decision", f"D:{decision}")
        for rel in episode.relations:
            ops += self.graph.expand_triple(rel["subject"], rel["predicate"], rel["object"])
        if episode.follows:
            ops += self.graph.expand_triple(eid, "follows", episode.follows)

        # State / validity tracking.
        ops += self.graph.expand_triple(eid, "state", episode.state)
        if episode.validity_start:
            ops += self.graph.expand_triple(eid, "validity_start", episode.validity_start)

        # ── User / Session scope (global chat history) ──
        # When the episode is scoped (encoder opened a session under a user),
        # link it into the User → Session → Episode hierarchy and stamp its
        # at_time so cross-session temporal queries can scan by timestamp.
        # has_session is idempotent (duplicate triples aren't stored), so
        # writing it per-episode keeps the graph self-consistent even if
        # open_session wasn't called; open_session adds started_at +
        # follows_session, close_session adds ended_at.
        if episode.user_id and episode.session_id:
            user = f"U:{episode.user_id}"
            sess = episode.session_id
            ops += self.graph.expand_triple(user, "has_session", sess)
            ops += self.graph.expand_triple(sess, "has_episode", eid)
            ops += self.graph.expand_triple(eid, "in_session", sess)
            ops += self.graph.expand_triple(eid, "at_time", episode.timestamp)

        self.db.batch_sync(ops)

    # ---- retrieve ----

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Load an episode's content from HBTrie.

        Returns the content fields (summary/text/ts/salience/state plus the
        three persisted downstream fields retrieval_count/ltp_phase/
        utility_decay_rate); the graph-side structure (entities/topics/
        relations) is queried separately via the Graph layer. Phase 1a's tests
        only assert content retrieval.
        """
        summary = _b2s(self.db.get_sync(f"content/ep/{episode_id}/summary"))
        if not summary:
            return None

        text = _b2s(self.db.get_sync(f"content/ep/{episode_id}/text"))
        ts = _b2s(self.db.get_sync(f"content/ep/{episode_id}/ts"))
        salience_str = _b2s(self.db.get_sync(f"content/ep/{episode_id}/salience")) or "0.5"
        state = _b2s(self.db.get_sync(f"content/ep/{episode_id}/state")) or "current"
        retrieval_count_str = _b2s(self.db.get_sync(f"content/ep/{episode_id}/retrieval_count")) or "0"
        ltp_phase = _b2s(self.db.get_sync(f"content/ep/{episode_id}/ltp_phase")) or "early"
        decay_rate_str = _b2s(self.db.get_sync(f"content/ep/{episode_id}/decay_rate")) or "0.01"
        saturation_flags_str = _b2s(self.db.get_sync(f"content/ep/{episode_id}/saturation_flags")) or "0"
        retrieval_timestamps_raw = _b2s(self.db.get_sync(f"content/ep/{episode_id}/retrieval_timestamps"))
        consolidation_window_start = _b2s(self.db.get_sync(f"content/ep/{episode_id}/consolidation_window_start")) or None
        embedding_raw = _b2s(self.db.get_sync(f"content/ep/{episode_id}/embedding"))

        try:
            salience = float(salience_str)
        except ValueError:
            salience = 0.5
        try:
            retrieval_count = int(retrieval_count_str)
        except ValueError:
            retrieval_count = 0
        try:
            utility_decay_rate = float(decay_rate_str)
        except ValueError:
            utility_decay_rate = 0.01
        try:
            saturation_flags = int(saturation_flags_str)
        except ValueError:
            saturation_flags = 0
        try:
            retrieval_timestamps = json.loads(retrieval_timestamps_raw) if retrieval_timestamps_raw else []
        except (ValueError, TypeError):
            retrieval_timestamps = []
        try:
            summary_embedding = json.loads(embedding_raw) if embedding_raw else None
        except (ValueError, TypeError):
            summary_embedding = None

        return Episode(
            id=episode_id,
            timestamp=ts,
            summary=summary,
            full_text=text,
            salience=salience,
            state=state,
            retrieval_count=retrieval_count,
            ltp_phase=ltp_phase,
            utility_decay_rate=utility_decay_rate,
            saturation_flags=saturation_flags,
            retrieval_timestamps=retrieval_timestamps,
            consolidation_window_start=consolidation_window_start,
            summary_embedding=summary_embedding,
        )

    def set_summary_embedding(self, episode_id: str, embedding: list[float]) -> None:
        """Backfill an episode's summary embedding (Phase F vector index build).

        Writes ``content/ep/{episode_id}/embedding`` so ``get_episode`` and
        ``VectorSearch`` can read it back without re-running the encoder.
        """
        import json
        self.db.put_sync(
            f"content/ep/{episode_id}/embedding",
            json.dumps(embedding),
        )

    # ---- user / session / global ids ----

    def _counter_next(self, key: str) -> int:
        """Read-modify-write a persisted counter in HBTrie, returning the new value.

        Counters live under ``content/system/`` (e.g. ``episode_counter``,
        ``session_counter``) so episode and session ids are globally unique and
        monotonically increasing across all users/sessions and across restarts.

        NOTE: this RMW is not atomic across concurrently-writing encoders — fine
        for the single-threaded Phase 1a corpus run. Phase 2+ needs a CAS or
        native counter primitive if multiple encoders write concurrently.
        """
        cur_raw = _b2s(self.db.get_sync(f"content/system/{key}"))
        cur = int(cur_raw) if cur_raw else 0
        nxt = cur + 1
        self.db.batch_sync([{"type": "put", "key": f"content/system/{key}", "value": str(nxt)}])
        return nxt

    def next_episode_id(self) -> str:
        """Globally-unique, monotonically-increasing episode id (``ep_NNNNNN``)."""
        return f"ep_{self._counter_next('episode_counter'):06d}"

    def next_session_id(self) -> str:
        """Globally-unique session id (``S:NNNN``)."""
        return f"S:{self._counter_next('session_counter'):04d}"

    def open_session(self, user_id: str, session_id: str, started_at: str) -> None:
        """Open a chat session under a user.

        Writes ``(U:user, has_session, S:session)`` and ``(S, started_at, ts)``,
        and chains ``(S, follows_session, S_prev)`` to the user's previous latest
        session (tracked by a per-user pointer in HBTrie) so the user's chats
        form a cross-session temporal chain. All ops share one ``batch_sync``.
        """
        user = f"U:{user_id}"
        ops: list[dict] = []
        ops += self.graph.expand_triple(user, "has_session", session_id)
        ops += self.graph.expand_triple(session_id, "started_at", started_at)
        prev = _b2s(self.db.get_sync(f"content/system/last_session/{user_id}"))
        if prev:
            ops += self.graph.expand_triple(session_id, "follows_session", prev)
        ops.append({"type": "put", "key": f"content/system/last_session/{user_id}", "value": session_id})
        self.db.batch_sync(ops)

    def close_session(self, session_id: str, ended_at: str) -> None:
        """Record a session's end timestamp. Idempotent."""
        self.db.batch_sync(self.graph.expand_triple(session_id, "ended_at", ended_at))

    def list_sessions(self, user_id: str) -> list[str]:
        """All session ids for a user (scan the has_session SPO index)."""
        user = f"U:{user_id}"
        start = f"memory/spo/{user}/has_session/"
        end = f"memory/spo/{user}/has_session/\x7f"
        keys = [k for k, _ in self.db.create_read_stream(start=start, end=end)]
        return [k.rsplit("/", 1)[-1] for k in keys]

    def list_session_episodes(self, session_id: str) -> list[str]:
        """All episode ids in a session (scan the has_episode SPO index)."""
        start = f"memory/spo/{session_id}/has_episode/"
        end = f"memory/spo/{session_id}/has_episode/\x7f"
        keys = [k for k, _ in self.db.create_read_stream(start=start, end=end)]
        return [k.rsplit("/", 1)[-1] for k in keys]

    # ---- lifecycle ----

    def close(self) -> None:
        # Close the graph layer first: the binding's WaveDB.close() refuses to
        # destroy the database while the graph layer's subtree reference is open.
        try:
            self.graph.close()
        finally:
            self.db.close()