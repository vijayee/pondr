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
import math
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from wavedb import GraphLayer, WaveDB, WaveDBConfig

from .blob_store import BlobStore, blob_hash
from .document import Document, DocumentSection
from .episode import Episode
from .ontology import SEED_ONTOLOGY


def _b2s(v) -> str:
    """Decode a WaveDB value (bytes) to str; '' for missing/None."""
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)


# Phase 3b step 10: entity-salience recency half-life (days). A mention at the
# half-life is worth half a same-instant mention; 4x half-life -> 1/16. The
# mention factor (Phase 1c) and structural factor (Phase 3a head) are both
# already in [0,1], so recency multiplies in as [0,1]. Constant (not a Config
# knob) to match the Phase 1c mention formula, which is also hardcoded here.
_SALIENCE_RECENCY_HALF_LIFE_DAYS = 30.0


def safe_edge_component(part: str) -> str:
    """A graph-key path component, hashing any part containing ``/`` or NUL.

    ``E:some/thing`` -> ``h_<sha256[:16]>`` so a literal slash in a node id
    never splits the key into the wrong namespace. Shared by the
    ``archive/edge/...`` key builder (``gnn.semantic_memory``) and the
    ``content/edge/...`` per-edge sidecar key builder (``memory.edge_meta``),
    so live-graph keys (``memory/spo/{s}/{p}/E:some/thing``, literal slash) and
    sidecar/archive keys (hashed) never collide. Lives here in the low
    ``store`` layer so both dependents reach it without a ``memory``->``gnn``
    import.
    """
    import hashlib
    return part if "/" not in part and "\x00" not in part else (
        "h_" + hashlib.sha256(part.encode("utf-8")).hexdigest()[:16]
    )


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
        # Cold, content-addressed blob store for document section bodies. LAZY:
        # opened on first document op so episode-only workloads (consolidation,
        # salience compute, retrieval) never create a second WaveDB instance.
        # Sibling of the memory db by default (data/memory_db -> data/document_db)
        # so a single data dir holds both; overridable via the config dict.
        self._blob: Optional[BlobStore] = None
        self._blob_db_path = (
            (config.get("document_db_path") if config else None)
            or os.path.join(os.path.dirname(db_path.rstrip("/").rstrip("\\")), "document_db")
        )
        self._blob_lru_mb = (config.get("document_store_lru_mb") if config else None) or 16
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
            # Provenance (2026-07-14): episode source + role-tagged segments.
            # origin is always written (default "corpus"); messages is optional
            # (None on pre-provenance episodes) so old episodes read back
            # role-unaware and consumers fall back to full_text.
            {"type": "put", "key": f"content/ep/{eid}/origin", "value": episode.origin},
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
        if episode.messages:
            ops.append({"type": "put", "key": f"content/ep/{eid}/messages", "value": json.dumps(episode.messages, ensure_ascii=False)})

        # ── Graph: sparse pointers (hippocampal index) via expand_triple ──
        # expand_triple returns root-namespace ops (the "memory/" subtree prefix
        # is already prepended by the C helper) — splice into the SAME batch_sync
        # so content and graph indices share one atomic transaction / WAL record.
        for entity in episode.entities:
            ops += self.graph.expand_triple(eid, "has_entity", f"E:{entity}")
            ops += self.graph.expand_triple(f"E:{entity}", "in_episode", eid)
            # A3: entity -> class typing edge (E:Alice instanceOf Person) when
            # the extractor assigned a seed class. Bridges the entity partition
            # and the class DAG so ontology-decay reassignment can find a
            # deprecated class's entities at the graph layer. Skipped for
            # open-discovery entities (untyped) and pre-A3 episodes.
            cls = episode.entity_classes.get(entity)
            if cls:
                ops += self.graph.expand_triple(f"E:{entity}", "instanceOf", cls)
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
        origin = _b2s(self.db.get_sync(f"content/ep/{episode_id}/origin")) or "corpus"
        messages_raw = _b2s(self.db.get_sync(f"content/ep/{episode_id}/messages"))

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
        try:
            messages = json.loads(messages_raw) if messages_raw else None
        except (ValueError, TypeError):
            messages = None

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
            origin=origin,
            messages=messages,
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

    # ---- entity salience (Phase 1c + Phase 3b composition) ----

    def get_entity_salience(self, entity: str) -> float:
        """Salience score in ``[0, 1]``.

        Phase 1c base: ``mention_count`` only (log-scaled). Phase 3b step 10
        COMPOSES three factors once the consolidation dream pass has persisted a
        structural salience for the entity:

            salience = mention_factor * structural_factor * recency_factor

        - ``mention_factor``  -- the Phase 1c log-scaled mention count, [0,1].
        - ``structural_factor`` -- the trained SalienceHead output (sigmoid'd
          + clipped to [0,1]), read from ``content/entity/{e}/structural_salience``.
        - ``recency_factor``  -- ``0.5 ** (age_days / half_life)`` from
          ``content/entity/{e}/last_mentioned_ts``; neutral ``1.0`` when no
          timestamp is present (cold start).

        **Cold-start fallback:** when ``structural_salience`` is absent (no
        consolidation pass has run, or this entity was never scored), the result
        is the Phase 1c mention-only value -- byte-identical to pre-3b behavior.
        This keeps retrieval and the GNN cold-start prior (``features.py``)
        unchanged on a fresh corpus; 3b composition only takes effect after the
        first dream pass persists structural salience. The recency factor is a
        neutral multiplier (1.0) when its timestamp is absent, so a corpus that
        has structural but not ``last_mentioned_ts`` still composes mention x
        structural (recency deferred, not blocking).

        Returns ``0.0`` for an unknown entity (no mention_count). Two
        ``get_sync`` point lookups in the composed path (mention + structural),
        three when recency is present -- all on single known keys.
        """
        count_str = _b2s(self.db.get_sync(f"content/entity/{entity}/mention_count"))
        if not count_str:
            return 0.0
        try:
            count = int(count_str)
        except ValueError:
            return 0.0
        # Log-scaled mention factor: 0 mentions -> 0; 1 -> ~0.1; 100 -> ~0.4; capped 1.0.
        # Gentle curve so a dominant entity doesn't fully drown out rarer ones.
        mention = min(1.0, 0.1 + 0.3 * (count ** 0.5) / 10.0)

        # Phase 3b: structural salience from the trained GNN head (sigmoid'd [0,1]).
        struct_str = _b2s(self.db.get_sync(f"content/entity/{entity}/structural_salience"))
        structural = None
        if struct_str:
            try:
                structural = float(struct_str)
            except ValueError:
                structural = None
        # Cold-start: no structural salience -> mention-only (backward compatible;
        # a corpus that never ran the consolidation dream pass behaves as Phase 1c).
        if structural is None:
            return mention
        structural = min(1.0, max(0.0, structural))

        # Recency factor (neutral 1.0 when no last_mentioned_ts).
        recency = self._recency_factor(entity)

        composed = mention * structural * recency
        return min(1.0, max(0.0, composed))

    def _recency_factor(self, entity: str) -> float:
        """Recency multiplier in [0,1]; ``1.0`` when no timestamp is persisted.

        ``0.5 ** (age_days / _SALIENCE_RECENCY_HALF_LIFE_DAYS)``: a mention at the
        half-life is worth half a fresh one. Uses the real wall clock (this is
        the retrieval hot path, not the dream pass); unparseable or absent
        timestamps decay to neutral (no modulation), never to zero.

        Parses ISO-8601 flexibly (``datetime.fromisoformat``): accepts both
        ``2026-07-01T00:00:00`` and ``2026-07-01T00:00:00Z`` -- episode timestamps
        are stored verbatim and may lack the ``Z`` suffix; a naive timestamp is
        read as UTC.
        """
        ts_str = _b2s(self.db.get_sync(f"content/entity/{entity}/last_mentioned_ts"))
        if not ts_str:
            return 1.0
        try:
            last = datetime.fromisoformat(ts_str)
        except ValueError:
            return 1.0
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_days = max(0.0, (now - last).total_seconds() / 86400.0)
        if age_days <= 0.0:
            return 1.0
        return float(0.5 ** (age_days / _SALIENCE_RECENCY_HALF_LIFE_DAYS))

    def write_entity_salience_batch(
        self,
        counts: dict[str, int],
        last_ep: dict[str, str],
        last_ep_ts: Optional[dict[str, str]] = None,
    ) -> None:
        """Persist salience for all entities in ONE sorted ``batch_sync``.

        Writes ``content/entity/{entity}/mention_count`` and
        ``content/entity/{entity}/last_mentioned``. When ``last_ep_ts`` is given
        (Phase 3b step 10), also writes ``content/entity/{entity}/last_mentioned_ts``
        -- the ISO timestamp of the latest-mentioning episode, so the retrieval
        hot path (``get_entity_salience``) can compute recency without an extra
        ``get_episode`` lookup per entity per query. ``last_ep`` stays the episode
        id (kept for back-compat / inspection); ``last_ep_ts`` is the recency key.

        Keys are SORTED before submission so subsequent ``get_sync`` reads on them
        are reliable (sorted insertion avoids the insertion-order-dependent split
        path — see ``docs/Phase 1c.md`` §0.3). Called by
        ``scripts/compute_entity_salience.py``; do NOT call per-encode (unsorted
        incremental writes are not get_sync-safe).

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
            if last_ep_ts is not None:
                ts = last_ep_ts.get(entity, "")
                if ts:
                    ops.append({
                        "type": "put",
                        "key": f"content/entity/{entity}/last_mentioned_ts",
                        "value": ts,
                    })
        if ops:
            self.db.batch_sync(ops)

    def persist_node_salience(self, node_id: str, score: float) -> None:
        """Persist a node's STRUCTURAL salience (Phase 3b, the trained GNN head).

        ``score`` is the sigmoid'd+clipped SalienceHead output in [0, 1]. Written
        for ENTITY nodes (``E:`` prefix) to ``content/entity/{bare}/structural_salience``
        so ``get_entity_salience`` (step 10) can compose mention_count x recency x
        structural. Non-entity nodes (topics/tones) are not persisted: their
        structural salience is used in-memory at dream-pass composition only (the
        retrieval hot path does not look up topic/tone structural salience).
        ``/``-bearing or NUL-bearing entity strings are skipped (same constraint as
        ``write_entity_salience_batch``).
        """
        if not node_id.startswith("E:"):
            return
        entity = node_id[2:]
        if "/" in entity or "\x00" in entity:
            return
        self.db.put_sync(
            f"content/entity/{entity}/structural_salience",
            repr(float(score)),
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

    def default_episode_ids(
        self,
        include_abstracted: bool = False,
        include_inactive: bool = False,
    ) -> list[str]:
        """All episode ids, applying the default-query filters.

        Default queries (spec §371) retrieve only the *active* set. Three
        flags can take an episode OUT of that set, all collected in ONE scan
        over ``content/ep/`` (no per-episode ``get_sync`` -- important since
        retrieval enumerates here per query):

        - **abstracted** (Phase 3a) -- ``content/ep/{eid}/abstracted == "1"``.
          The episode was rolled into a semantic memory; content is untouched.
        - **state != "current"** (Phase 3b) -- ``content/ep/{eid}/state`` set to
          ``deprecated`` (active-forget) or ``superseded`` (reconsolidation).
          The episode is NOT deleted; deprecate, don't delete.
        - **validity_end set** (Phase 3b) -- ``content/ep/{eid}/validity_end``
          records when the episode stopped being authoritative.

        ``include_abstracted`` and ``include_inactive`` independently opt back
        in (e.g. an explicit historical "show me everything" query). The two
        axes are independent: an abstracted-but-current episode is excluded on
        the abstracted axis only; a deprecated-but-unabstracted episode on the
        state axis only.

        The Phase 3b state/validity_end axis is gated on the master
        ``forgetting_enabled`` flag so ``--no-forget`` makes a corpus with prior
        deprecations behave as if forgetting were never deployed (everything
        current). The 3a ``abstracted`` axis is NOT gated -- it is independent of
        the forgetting system.
        """
        from ..config import config as _master_config
        forget_on = _master_config.forgetting_enabled
        ids: set[str] = set()
        abstracted: set[str] = set()
        state: dict[str, str] = {}
        validity_end: dict[str, str] = {}
        for k, v in self.db.create_read_stream(start="content/ep/", end="content/ep/\x7f"):
            parts = k.split("/", 3)
            if len(parts) < 3 or not parts[2]:
                continue
            eid = parts[2]
            ids.add(eid)
            field = parts[3] if len(parts) >= 4 else ""
            if field == "abstracted" and _b2s(v) == "1":
                abstracted.add(eid)
            elif field == "state":
                state[eid] = _b2s(v)
            elif field == "validity_end":
                validity_end[eid] = _b2s(v)
        if include_abstracted and include_inactive:
            return sorted(ids)
        out = []
        for eid in sorted(ids):
            if not include_abstracted and eid in abstracted:
                continue
            # Phase 3b state/validity_end axis -- gated on forgetting_enabled.
            if forget_on and not include_inactive and (
                state.get(eid, "current") != "current" or validity_end.get(eid)
            ):
                continue
            out.append(eid)
        return out

    # ---- episode-level forgetting state (Phase 3b) ----

    def set_episode_state(
        self, episode_id: str, state: str, validity_end: "str | None" = None
    ) -> None:
        """Set an episode's lifecycle ``state`` (+ optional ``validity_end``).

        Phase 3b write path for active-forget (``state="deprecated"``) and
        reconsolidation-supersession (``state="superseded"``). Writes the
        ``content/ep/{eid}/state`` content key (and ``validity_end`` if given),
        which ``default_episode_ids`` reads to exclude the episode from default
        queries. The episode is NOT deleted -- its content stays retrievable via
        ``include_inactive=True`` and its graph triples are untouched.

        Note: the graph triple ``(eid, "state", old_state)`` written at encode
        is NOT updated here; retrieval ``_hydrate`` does not read it, and the
        content key is the filter's source of truth. Updating the graph triple
        would need an expand_triple(delete old)+(put new) and gains nothing
        the filter uses.
        """
        ops: list[dict] = [
            {"type": "put", "key": f"content/ep/{episode_id}/state", "value": state}
        ]
        if validity_end is not None:
            ops.append({
                "type": "put",
                "key": f"content/ep/{episode_id}/validity_end",
                "value": validity_end,
            })
        self.db.batch_sync(ops)

    def episode_state(self, episode_id: str) -> str:
        """The episode's lifecycle state (``"current"`` if never set)."""
        return _b2s(self.db.get_sync(f"content/ep/{episode_id}/state")) or "current"

    def episode_validity_end(self, episode_id: str) -> "str | None":
        """The episode's ``validity_end`` timestamp, or ``None`` if unset."""
        raw = _b2s(self.db.get_sync(f"content/ep/{episode_id}/validity_end"))
        return raw or None

    def is_episode_active(self, episode_id: str) -> bool:
        """True if the episode is in the default-query active set.

        Mirrors ``default_episode_ids`` 's state/validity_end exclusion: an
        episode is active iff ``state == "current"`` and ``validity_end`` is
        unset. Used by the retrieval axis-query path (``_find_candidates``) so a
        deprecated/superseded episode is excluded from axis queries too -- not
        just from the no-axis enumeration path. Two point ``get_sync`` calls.
        """
        if self.episode_state(episode_id) != "current":
            return False
        return self.episode_validity_end(episode_id) is None

    # ---- edge-level forgetting sidecar (Phase 3b) ----
    # Thin wrappers over src/memory/edge_meta.py. Deferred import avoids a
    # module-level store<->edge_meta cycle (edge_meta imports store for
    # safe_edge_component + _b2s). Callers may also use edge_meta directly.

    def get_edge_meta(self, subject: str, predicate: str, object: str) -> dict:
        """Read an edge's forgetting sidecar (default_meta() if none yet)."""
        from .edge_meta import get_edge_meta as _get
        return _get(self, subject, predicate, object)

    def update_edge_meta_batch(
        self, updates: list[tuple[str, str, str, dict]]
    ) -> None:
        """Write many edge sidecars in one atomic ``batch_sync``."""
        from .edge_meta import batch_update_edge_meta as _batch
        _batch(self, updates)

    def set_edge_state(
        self,
        subject: str,
        predicate: str,
        object: str,
        state: str,
        validity_end: Optional[str] = None,
    ) -> dict:
        """Set an edge's ``state`` (+ optional ``validity_end``) via RMW."""
        from .edge_meta import set_edge_state as _set
        return _set(self, subject, predicate, object, state, validity_end)

    def is_edge_current(self, subject: str, predicate: str, object: str) -> bool:
        """True if an edge's sidecar state is ``current`` (or no sidecar yet)."""
        from .edge_meta import is_edge_current as _cur
        return _cur(self, subject, predicate, object)

    # ---- ontology-class decay (Phase 3b step 9) ----
    # Per-class metadata lives under ``content/class/{safe(class)}/...``. The
    # seed ontology writes ONLY ``subClassOf`` graph triples (no content/class
    # entries), so seed classes have no last_seen/discovered/state and are
    # NEVER decay-eligible -- decay targets DISCOVERED classes (runtime-
    # invented labels promoted via Bonsai, a deferred path). The mechanism is
    # shipped now so promotion lands into a decay-ready namespace; today the
    # decay pass is a no-op (no discovered classes exist).

    def _class_key(self, class_name: str, field: str) -> str:
        return f"content/class/{safe_edge_component(class_name)}/{field}"

    def persist_class_last_seen(self, class_name: str, ts: str) -> None:
        """Stamp ``content/class/{c}/last_seen = ts`` (the class-use signal)."""
        self.db.put_sync(self._class_key(class_name, "last_seen"), ts)

    def class_last_seen(self, class_name: str) -> "Optional[str]":
        raw = _b2s(self.db.get_sync(self._class_key(class_name, "last_seen")))
        return raw or None

    def mark_class_discovered(self, class_name: str) -> None:
        """Mark a class as runtime-discovered (decay-eligible, unlike seed)."""
        self.db.put_sync(self._class_key(class_name, "discovered"), "1")

    def is_class_discovered(self, class_name: str) -> bool:
        return _b2s(self.db.get_sync(self._class_key(class_name, "discovered"))) == "1"

    def set_class_state(self, class_name: str, state: str) -> None:
        """Set ``content/class/{c}/state`` (e.g. ``"deprecated"`` -- decay)."""
        self.db.put_sync(self._class_key(class_name, "state"), state)

    def class_state(self, class_name: str) -> str:
        return _b2s(self.db.get_sync(self._class_key(class_name, "state"))) or "current"

    def scan_classes(self) -> "list[str]":
        """All class names with a ``content/class/`` entry (discovered or touched).

        Seed classes have no ``content/class/`` entry (the seed writes only
        ``subClassOf`` graph triples), so they don't appear here -- which is
        exactly the gate that keeps decay off the seed ontology.
        """
        names: set[str] = set()
        for k, _ in self.db.create_read_stream(
            start="content/class/", end="content/class/\x7f"
        ):
            parts = k.split("/")
            # content/class/{name}/field -> parts[2] is the (hashed) name.
            if len(parts) >= 3 and parts[2]:
                names.add(parts[2])
        return sorted(names)

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

    # ---- documents (ingestion; task #17) ----
    #
    # Documents are a peer top-level unit to episodes, with a hot/cold split:
    # small metadata + graph pointers live HERE in the memory (hot) store;
    # section bodies live in the content-addressed cold store (BlobStore).
    # Identity is keyed by ``source_path`` (the ``doc_by_source`` index) so
    # re-ingesting a source UPDATES in place rather than duplicating -- explicit
    # ingest, no file watcher (user directive). Documents are EXEMPT from the
    # 3b forgetting system: no state/validity/salience fields, and the ``doc_``
    # id prefix is the ownership signal the decay sweeps skip by prefix.
    #
    # Content-key layout (hot store):
    #   content/doc/{id}/{field}            doc-level metadata
    #   content/doc/{id}/sec/{sid}/{field}  per-section metadata + blob_hash ref
    #   content/doc/{id}/section_ids        JSON list (enumerates sections for
    #                                      update/delete edge cleanup)
    #   content/doc_by_source/{safe(path)}  source_path -> doc_id (upsert index)
    # Bodies + embeddings live in the cold store (blob/body/{hash},
    # blob/refs/{hash}); only the blob_hash ref is stored here.

    _DOC_FIELDS = (
        "title", "source_type", "source_path", "authors", "ingested_at",
        "created_at", "language", "metadata", "entities", "topics",
        "citations", "relations", "section_ids",
    )
    _SEC_FIELDS = (
        "heading", "level", "parent", "blob_hash", "entities", "topics",
        "embedding",
    )

    def _blob_store(self) -> BlobStore:
        """Lazily open + memoize the cold blob store.

        Episode-only workloads (consolidation, salience compute, retrieval)
        never touch a document and so never open the second WaveDB instance --
        the blob store is created on first document op, not at store init.
        """
        if self._blob is None:
            self._blob = BlobStore(self._blob_db_path, lru_memory_mb=self._blob_lru_mb)
        return self._blob

    def next_document_id(self) -> str:
        """Globally-unique, monotonically-increasing document id (``doc_NNNNNN``).

        Counter lives under ``content/system/document_counter`` beside the
        episode/session/memory counters (same RMW caveat as ``next_episode_id``).
        """
        return f"doc_{self._counter_next('document_counter'):06d}"

    def document_id_by_source(self, source_path: str) -> Optional[str]:
        """Resolve upsert identity: the doc id for ``source_path``, or ``None``.

        The ``doc_by_source`` index is keyed by ``safe_edge_component(path)``
        (a source path contains ``/`` and would otherwise split the key into
        the wrong namespace; the hash folds it to one component). Re-ingesting
        the same source string resolves to the same key -> the same doc id ->
        an in-place UPDATE rather than a duplicate.
        """
        raw = _b2s(self.db.get_sync(
            f"content/doc_by_source/{safe_edge_component(source_path)}"))
        return raw or None

    def _document_graph_ops(self, doc: Document, delete: bool) -> list[dict]:
        """Build expand_triple ops for a doc's graph pointers (put or delete).

        Triples (per the ingestion plan): ``(doc, instanceOf, Document)``,
        ``(doc, has_entity, E:x)`` + ``(E:x, appears_in_doc, doc)``,
        ``(doc, has_topic, T:t)``, ``(doc, has_section, sid)``,
        ``(sid, child_of, parent)``, ``(sid, has_entity, E:x)``,
        ``(doc, cites, target)``, plus each Bonsai ``relations`` triple. The
        four document-specific predicates (``has_section``/``child_of``/
        ``appears_in_doc``/``cites``) are snake_case hash-tail predicates,
        checkpoint-safe + GNN-invisible until retrain (the A3 ``instanceOf``
        template). Reversing a put uses the SAME ``(s, p, o)`` with
        ``delete=True`` so the SPO/POS/OSP index entries all come out.
        """
        ops: list[dict] = []

        def emit(s: str, p: str, o: str) -> None:
            ops.extend(self.graph.expand_triple(s, p, o, delete=delete))

        emit(doc.id, "instanceOf", "Document")
        for e in doc.entities:
            emit(doc.id, "has_entity", f"E:{e}")
            emit(f"E:{e}", "appears_in_doc", doc.id)
        for t in doc.topics:
            emit(doc.id, "has_topic", f"T:{t}")
        for sec in doc.sections:
            emit(doc.id, "has_section", sec.id)
            if sec.parent_section:
                emit(sec.id, "child_of", sec.parent_section)
            for e in sec.entities:
                emit(sec.id, "has_entity", f"E:{e}")
        for target in doc.citations:
            emit(doc.id, "cites", target)
        for rel in doc.relations:
            emit(rel["subject"], rel["predicate"], rel["object"])
        return ops

    def _document_metadata_ops(self, doc: Document) -> list[dict]:
        """Build the hot-store content puts for a doc's metadata + sections."""
        ops: list[dict] = []
        d = f"content/doc/{doc.id}"

        def put(key: str, value) -> None:
            ops.append({"type": "put", "key": key, "value": value})

        put(f"{d}/title", doc.title)
        put(f"{d}/source_type", doc.source_type)
        put(f"{d}/source_path", doc.source_path)
        put(f"{d}/authors", json.dumps(doc.authors))
        put(f"{d}/ingested_at", doc.ingested_at)
        if doc.created_at:
            put(f"{d}/created_at", doc.created_at)
        if doc.language:
            put(f"{d}/language", doc.language)
        put(f"{d}/metadata", json.dumps(doc.metadata))
        put(f"{d}/entities", json.dumps(doc.entities))
        put(f"{d}/topics", json.dumps(doc.topics))
        put(f"{d}/citations", json.dumps(doc.citations))
        put(f"{d}/relations", json.dumps(doc.relations))
        put(f"{d}/section_ids", json.dumps([s.id for s in doc.sections]))
        for sec in doc.sections:
            s = f"{d}/sec/{sec.id}"
            put(f"{s}/heading", sec.heading)
            put(f"{s}/level", str(sec.level))
            put(f"{s}/parent", sec.parent_section or "")
            put(f"{s}/blob_hash", sec.blob_hash or "")
            put(f"{s}/entities", json.dumps(sec.entities))
            put(f"{s}/topics", json.dumps(sec.topics))
            if sec.embedding is not None:
                put(f"{s}/embedding", json.dumps(sec.embedding))
        # Upsert index (overwritten in place on update; same key for the same
        # source_path, so the put is idempotent across re-ingests).
        put(f"content/doc_by_source/{safe_edge_component(doc.source_path)}", doc.id)
        return ops

    def _document_delete_content_ops(
        self, doc_id: str, section_ids: list[str]
    ) -> list[dict]:
        """Build del ops for a doc's known hot-store content keys.

        Deletes every doc-level field + every per-section field for the given
        section ids. Used by update (delete-then-rewrite, so removed sections'
        keys don't linger) and by delete (full removal). The ``doc_by_source``
        index lives under a different key and is handled by the caller.
        """
        ops: list[dict] = [{"type": "del", "key": f"content/doc/{doc_id}/{f}"}
                           for f in self._DOC_FIELDS]
        for sid in section_ids:
            for f in self._SEC_FIELDS:
                ops.append({"type": "del", "key": f"content/doc/{doc_id}/sec/{sid}/{f}"})
        return ops

    def encode_document(self, doc: Document, *, update: bool = False) -> None:
        """Store a document's metadata + graph pointers + section bodies.

        Hot/cold split (the user's LRU concern): section BODIES go to the cold
        content-addressed BlobStore (``put_blob``, idempotent = dedup); the hot
        memory store keeps only small metadata + graph pointers + a
        ``blob_hash`` ref per section. The memory-store writes (metadata +
        graph) are ONE atomic ``batch_sync``, mirroring ``encode_episode``;
        the blob writes precede it (a blob must exist before its hash is
        referenced). Cross-store atomicity is NOT possible (two WaveDB
        instances); on a memory-batch failure the blobs are orphaned and GC'd
        later by ``BlobStore.gc_zero_ref`` -- acceptable for the single-
        threaded first slice, documented in the plan.

        Upsert (``update=True``): the doc.id is the REUSED id (the pipeline
        resolved it via ``document_id_by_source``). The old sections' blob
        hashes are loaded (without their bodies) to compute a refcount DELTA --
        unchanged sections keep their ref (and, in Phase 2, their embedding),
        removed sections drop theirs, new/changed sections get one. The old
        hot-store keys + graph edges are deleted, then the new set written,
        all in the one memory batch. ``doc_by_source`` is overwritten in place.
        """
        bs = self._blob_store()
        # Load the OLD doc once (update path): reused for BOTH the refcount
        # delta (step 2) and the delete-then-rewrite ops (step 3), so a re-
        # ingest reads the prior metadata a single time, not twice.
        old_doc: Optional[Document] = None
        if update:
            old_doc = self.get_document(doc.id, load_bodies=False)
        # 1. Hash each new section's body + write its blob (idempotent = dedup
        #    across documents AND across re-ingests of the same doc).
        for sec in doc.sections:
            sec.blob_hash = blob_hash(sec.content)
            bs.put_blob(sec.blob_hash, sec.content)
        # 2. Refcount delta vs the old section hashes (CREATE: old is empty).
        old_hashes: list[str] = (
            [s.blob_hash for s in old_doc.sections if s.blob_hash]
            if old_doc is not None else []
        )
        new_counts = Counter(s.blob_hash for s in doc.sections if s.blob_hash)
        old_counts = Counter(old_hashes)
        # NOTE: ``Counter`` ``+``/``-`` silently drop non-positive counts, so a
        # removed section's hash (delta < 0) would be dropped and ``decr_ref``
        # never called -- a refcount leak that orphans shared blobs. Compute
        # the delta with a plain dict so negative entries survive.
        delta = {h: new_counts[h] - old_counts[h]
                 for h in set(new_counts) | set(old_counts)}
        for h, n in delta.items():
            if n > 0:
                for _ in range(n):
                    bs.incr_ref(h)
            elif n < 0:
                for _ in range(-n):
                    bs.decr_ref(h)
        # 3. ONE atomic memory batch: delete old content + graph edges, write
        #    new content + graph edges + the upsert index.
        ops: list[dict] = []
        if old_doc is not None:
            ops += self._document_delete_content_ops(
                doc.id, [s.id for s in old_doc.sections])
            ops += self._document_graph_ops(old_doc, delete=True)
        ops += self._document_metadata_ops(doc)
        ops += self._document_graph_ops(doc, delete=False)
        self.db.batch_sync(ops)

    def get_document(
        self, doc_id: str, load_bodies: bool = True
    ) -> Optional[Document]:
        """Load a document's metadata + sections from the hot (+ cold) store.

        Mirrors ``get_episode``. Doc-level + per-section metadata + graph
        pointers' refs (``blob_hash``) come from the hot store; section BODIES
        are pulled from the cold store keyed by ``blob_hash`` -- but only when
        ``load_bodies`` is set (a cold pull, kept out of the hot LRU by the
        split). ``load_bodies=False`` is the update/delete path: it recovers
        the section structure + blob hashes without the cold pull. Graph
        structure (entities/topics/relations/citations) is read back from the
        hot-store metadata fields (written at encode), not re-queried from the
        graph layer. Returns ``None`` for a missing doc (no title key).
        """
        title = _b2s(self.db.get_sync(f"content/doc/{doc_id}/title"))
        if not title:
            return None
        source_type = _b2s(self.db.get_sync(f"content/doc/{doc_id}/source_type")) or "text"
        source_path = _b2s(self.db.get_sync(f"content/doc/{doc_id}/source_path"))
        authors_raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/authors"))
        ingested_at = _b2s(self.db.get_sync(f"content/doc/{doc_id}/ingested_at"))
        created_at = _b2s(self.db.get_sync(f"content/doc/{doc_id}/created_at")) or None
        language = _b2s(self.db.get_sync(f"content/doc/{doc_id}/language")) or None
        metadata_raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/metadata"))
        entities_raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/entities"))
        topics_raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/topics"))
        citations_raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/citations"))
        relations_raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/relations"))
        section_ids_raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/section_ids"))

        def _loads(raw, default):
            try:
                return json.loads(raw) if raw else default
            except (ValueError, TypeError):
                return default

        bs = self._blob_store() if load_bodies else None
        sections: list[DocumentSection] = []
        for sid in _loads(section_ids_raw, []):
            s = f"content/doc/{doc_id}/sec/{sid}"
            blob_hash = _b2s(self.db.get_sync(f"{s}/blob_hash")) or None
            content = ""
            if load_bodies and blob_hash:
                body = bs.get_blob(blob_hash)
                content = body if body is not None else ""
            embedding = None
            emb_raw = _b2s(self.db.get_sync(f"{s}/embedding"))
            if emb_raw:
                embedding = _loads(emb_raw, None)
            sections.append(DocumentSection(
                id=sid,
                heading=_b2s(self.db.get_sync(f"{s}/heading")),
                level=int(_b2s(self.db.get_sync(f"{s}/level")) or "1"),
                content=content,
                parent_section=_b2s(self.db.get_sync(f"{s}/parent")) or None,
                entities=_loads(_b2s(self.db.get_sync(f"{s}/entities")), []),
                topics=_loads(_b2s(self.db.get_sync(f"{s}/topics")), []),
                embedding=embedding,
                blob_hash=blob_hash,
            ))

        return Document(
            id=doc_id,
            source_type=source_type,
            source_path=source_path,
            title=title,
            ingested_at=ingested_at,
            sections=sections,
            authors=_loads(authors_raw, []),
            created_at=created_at,
            language=language,
            metadata=_loads(metadata_raw, {}),
            entities=_loads(entities_raw, []),
            topics=_loads(topics_raw, []),
            relations=_loads(relations_raw, []),
            citations=_loads(citations_raw, []),
        )

    def default_document_ids(self) -> list[str]:
        """All document ids (scan ``content/doc/``).

        Documents are NOT forgotten, so there is no state/validity filter --
        a deleted document's metadata is gone, so it simply doesn't appear.
        Parallel to ``default_episode_ids`` but simpler. The scan range
        ``content/doc/`` .. ``content/doc/\\x7f`` excludes
        ``content/doc_by_source`` by trie ordering (``_`` 0x5F > ``/`` 0x2F,
        so the by-source index sorts beyond the range's end key).
        """
        ids: set[str] = set()
        for k, _ in self.db.create_read_stream(
            start="content/doc/", end="content/doc/\x7f"
        ):
            parts = k.split("/", 3)
            if len(parts) < 3 or not parts[2]:
                continue
            ids.add(parts[2])
        return sorted(ids)

    def documents_by_entity(self, entity: str) -> list[str]:
        """Document ids that link a given entity (``(doc, has_entity, E:x)``).

        Scans the POS index ``memory/pos/has_entity/E:{entity}/`` and keeps
        ``doc_`` subjects. Store-level findability for Phase 1; the full
        ``GraphTraversal.retrieve()`` integration is Phase 2.
        """
        return self._documents_by_pos("has_entity", f"E:{entity}")

    def documents_by_topic(self, topic: str) -> list[str]:
        """Document ids that carry a given topic (``(doc, has_topic, T:t)``)."""
        return self._documents_by_pos("has_topic", f"T:{topic}")

    def _documents_by_pos(self, predicate: str, obj: str) -> list[str]:
        start = f"memory/pos/{predicate}/{obj}/"
        end = f"memory/pos/{predicate}/{obj}/\x7f"
        out: list[str] = []
        for k, _ in self.db.create_read_stream(start=start, end=end):
            # k = memory/pos/{predicate}/{obj}/{subject}
            subj = k.rsplit("/", 1)[-1]
            if subj.startswith("doc_"):
                out.append(subj)
        return sorted(set(out))

    def delete_document(self, doc_id: str) -> bool:
        """Real removal of a document + its sections (no archive record).

        The user's "deleted, not archived" directive: the doc's hot metadata,
        per-section metadata, graph edges, and ``doc_by_source`` index entry
        are all deleted in ONE atomic ``batch_sync``, and each section's blob
        refcount is DECREMENTED (shared-blob safety: a blob may back more than
        one document, so the body is NOT physically deleted here -- zero-ref
        orphans are left for ``BlobStore.gc_zero_ref`` to sweep). Returns
        ``False`` if the doc was already absent (idempotent), ``True`` if
        removed. A lightweight audit tombstone is intentionally NOT written
        (honest delete, per the plan's default).
        """
        doc = self.get_document(doc_id, load_bodies=False)
        if doc is None:
            return False
        ops: list[dict] = []
        ops += self._document_delete_content_ops(
            doc_id, [s.id for s in doc.sections])
        ops += self._document_graph_ops(doc, delete=True)
        ops.append({"type": "del",
                    "key": f"content/doc_by_source/{safe_edge_component(doc.source_path)}"})
        self.db.batch_sync(ops)
        # Decrement each section's blob refcount AFTER the memory batch (the
        # doc is no longer findable); zero-ref blobs are orphaned, not deleted
        # (shared-blob safety). ``--gc-blobs`` sweeps them later.
        bs = self._blob_store()
        for sec in doc.sections:
            if sec.blob_hash:
                bs.decr_ref(sec.blob_hash)
        return True

    def gc_blobs(self) -> int:
        """Sweep zero-refcount orphan blobs from the cold store. Returns count."""
        return self._blob_store().gc_zero_ref()

    # ---- lifecycle ----

    def close(self) -> None:
        # Close children before the db: the binding's WaveDB.close() refuses
        # to destroy the database while child handles are open. The graph
        # layer is always open; the blob store is opened lazily and may be
        # None for episode-only workloads.
        try:
            self.graph.close()
        finally:
            if self._blob is not None:
                try:
                    self._blob.close()
                finally:
                    self.db.close()
            else:
                self.db.close()