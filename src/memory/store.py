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

    # ---- entity salience (Phase 1c) ----

    def get_entity_salience(self, entity: str) -> float:
        """Salience score in ``[0, 1]``. Phase 1c: mention_count only (log-scaled).

        Recency and structural factors are Phase 3 (GNN salience). Returns ``0.0``
        for an unknown entity. This is a ``get_sync`` point lookup on a single
        known key — safe on the sorted-built compact corpora; the salience keys
        are written in a sorted batch by ``write_entity_salience_batch`` (see
        ``docs/Phase 1c.md`` §0.3 for the get_sync caveat).
        """
        count_str = _b2s(self.db.get_sync(f"content/entity/{entity}/mention_count"))
        if not count_str:
            return 0.0
        try:
            count = int(count_str)
        except ValueError:
            return 0.0
        # Log-scaled: 0 mentions -> 0; 1 -> ~0.1; 100 -> ~0.4; capped at 1.0.
        # Gentle curve so a dominant entity doesn't fully drown out rarer ones.
        return min(1.0, 0.1 + 0.3 * (count ** 0.5) / 10.0)

    def write_entity_salience_batch(
        self,
        counts: dict[str, int],
        last_ep: dict[str, str],
    ) -> None:
        """Persist salience for all entities in ONE sorted ``batch_sync``.

        Writes ``content/entity/{entity}/mention_count`` and
        ``content/entity/{entity}/last_mentioned``. Keys are SORTED before
        submission so subsequent ``get_sync`` reads on them are reliable (sorted
        insertion avoids the insertion-order-dependent split path — see
        ``docs/Phase 1c.md`` §0.3). Called by ``scripts/compute_entity_salience.py``;
        do NOT call per-encode (unsorted incremental writes are not get_sync-safe).

        Entities containing ``/`` or NUL are skipped: the encoder assumes
        ``/``-free entity strings for its own ``memory/spo/{eid}/has_entity/E:{entity}``
        keys, and a ``/`` would split the salience key into the wrong namespace.
        """
        ops: list[dict] = []
        for entity in sorted(counts):
            if "/" in entity or "\x00" in entity:
                continue
            ops.append({
                "type": "put",
                "key": f"content/entity/{entity}/mention_count",
                "value": str(counts[entity]),
            })
            ops.append({
                "type": "put",
                "key": f"content/entity/{entity}/last_mentioned",
                "value": last_ep.get(entity, ""),
            })
        if ops:
            self.db.batch_sync(ops)

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

    def next_memory_id(self) -> str:
        """Globally-unique semantic-memory id (``M:NNNN``).

        Phase 3a: semantic memories (DiffPool cluster abstractions) are ``M:``
        nodes that ``abstracts`` their source episodes. Counter lives under
        ``content/system/memory_counter`` like the episode/session counters.
        """
        return f"M:{self._counter_next('memory_counter'):04d}"

    # ---- abstraction / default-query filtering (Phase 3a) ----

    def is_abstracted(self, episode_id: str) -> bool:
        """True if ``episode_id`` has been abstracted into a semantic memory.

        Abstracted episodes stay retrievable (content untouched) but are
        excluded from default queries (spec §371). Single ``get_sync`` on a
        known key.
        """
        return _b2s(self.db.get_sync(f"content/ep/{episode_id}/abstracted")) == "1"

    def default_episode_ids(self, include_abstracted: bool = False) -> list[str]:
        """All episode ids, optionally excluding abstracted ones.

        Default queries retrieve the non-abstracted set (spec §371). The
        retrieval layer's enumeration delegates here so abstracted episodes are
        excluded uniformly. ``include_abstracted=True`` opts in (e.g. an
        explicit "show me everything including abstractions" path).
        """
        ids: set[str] = set()
        for k, _ in self.db.create_read_stream(start="content/ep/", end="content/ep/\x7f"):
            parts = k.split("/", 3)
            if len(parts) >= 3 and parts[2]:
                ids.add(parts[2])
        if include_abstracted:
            return sorted(ids)
        return sorted(eid for eid in ids if not self.is_abstracted(eid))

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

    # ---- JGS recurrent-state persistence (Phase 2c plumbing) ----

    def save_jgs_state(self, user_id: str, blob: str, scope: str = "working_memory") -> None:
        """Persist a serialized JGS recurrent-state snapshot for a user.

        Phase 2c plumbing: the *mechanism* only. The ``when`` (save-trigger
        policy) is decided by the orchestrator/session layer, not here. Working
        Memory is per-user and cross-session — a user has working memory, and
        the sessions live inside it — so the state is keyed by user (not by
        session), mirroring the cross-session ``content/system/last_session``
        pointer. ``scope`` lets multiple distinct JGS instances coexist under
        one user (e.g. ``"working_memory"`` now, a future gate id later) — the
        default is the only instance that persists in 2c.

        ``blob`` is the NUL-free ASCII text produced by
        ``src.subconscious.state_serializer.serialize``. It is written with a
        single ``put_sync`` (a known key, so the sorted-insertion caveat that
        governs the salience/entity keys does not apply). The blob must not
        contain ``/``-split namespaces or NUL — ``state_serializer`` guarantees
        this; we guard the user-supplied ``user_id``/``scope`` the same way
        ``write_entity_salience_batch`` guards entity strings.
        """
        if "/" in user_id or "\x00" in user_id:
            raise ValueError(f"invalid user_id for JGS state key: {user_id!r}")
        if "/" in scope or "\x00" in scope or not scope:
            raise ValueError(f"invalid scope for JGS state key: {scope!r}")
        if not blob:
            raise ValueError("JGS state blob is empty — refusing to store (load would read as None)")
        if "\x00" in blob:
            raise ValueError("JGS state blob contains a NUL byte — refusing to store")
        self.db.put_sync(f"content/system/user/{user_id}/{scope}/state", blob)

    def load_jgs_state(self, user_id: str, scope: str = "working_memory") -> Optional[str]:
        """Return the serialized JGS state blob for ``user_id``/``scope``, or ``None``.

        Inverse of :meth:`save_jgs_state`. ``None`` means no state has been saved
        yet (fresh user) — the caller should ``reset_state`` a fresh instance
        rather than restore. Single-key ``get_sync`` is safe on the compact
        corpora (see the get_sync caveat in ``docs/Phase 1c.md`` §0.3).
        """
        if "/" in user_id or "\x00" in user_id:
            raise ValueError(f"invalid user_id for JGS state key: {user_id!r}")
        if "/" in scope or "\x00" in scope or not scope:
            raise ValueError(f"invalid scope for JGS state key: {scope!r}")
        raw = self.db.get_sync(f"content/system/user/{user_id}/{scope}/state")
        blob = _b2s(raw)
        return blob if blob else None

    # ---- presentation outcome persistence (Phase 3a Task 7) ----

    def save_presentation_outcomes(self, user_id: str, blob: str) -> None:
        """Persist the serialized PresentationGate outcome/override buffers.

        Phase 3a Task 7: the EXPAND-frequency salience signal (2c §15) was
        in-memory only — ``presentation_gate.outcome_buffer``/``override_buffer``
        were ``deque``s that died on restart and ``record_outcome`` was never
        auto-invoked. This + :meth:`load_presentation_outcomes` make the signal
        durable and cross-session, mirroring :meth:`save_jgs_state`. The blob is
        the JSON from ``PresentationGate.serialize_buffers``.
        """
        if "/" in user_id or "\x00" in user_id:
            raise ValueError(f"invalid user_id for outcome key: {user_id!r}")
        if not blob or "\x00" in blob:
            raise ValueError("presentation-outcome blob is empty or contains NUL")
        self.db.put_sync(f"content/system/user/{user_id}/presentation_outcomes/state", blob)

    def load_presentation_outcomes(self, user_id: str) -> Optional[str]:
        """Return the serialized outcome-buffer blob for ``user_id``, or ``None``."""
        if "/" in user_id or "\x00" in user_id:
            raise ValueError(f"invalid user_id for outcome key: {user_id!r}")
        raw = self.db.get_sync(f"content/system/user/{user_id}/presentation_outcomes/state")
        blob = _b2s(raw)
        return blob if blob else None

    # ---- lifecycle ----

    def close(self) -> None:
        # Close the graph layer first: the binding's WaveDB.close() refuses to
        # destroy the database while the graph layer's subtree reference is open.
        try:
            self.graph.close()
        finally:
            self.db.close()