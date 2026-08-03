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
from .bm25_index import bm25_index_ops, bm25_unindex_ops
from .document import Document, DocumentSection
from .episode import Episode
from .ontology import SEED_ONTOLOGY


def _has_vector_layer() -> bool:
    """True if the installed wavedb exposes the native VectorLayer (>=0.2.0).

    Gated so Hippo keeps working on old WaveDB (<0.2.0) and in the offline test
    suite (no vector build needed): when False, ``HippocampalStore`` leaves
    ``self.vector_layer = None`` and the retriever falls back to the FAISS
    ``VectorSearch`` path.
    """
    import wavedb
    return hasattr(wavedb, "VectorLayer")


def _b2s(v) -> str:
    """Decode a WaveDB value (bytes) to str; '' for missing/None."""
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)


def _norm_title(s: "Optional[str]") -> str:
    """Normalize a document title for citation matching (Phase 3c, D5).

    Lowercase, collapse whitespace, strip wrapping punctuation. "Policy A:
    Remote Work" -> "policy a remote work". Used by
    ``find_document_by_title_or_url`` so a citation literal "Policy A" matches
    a doc titled "Policy A: Remote Work" (substring containment).
    """
    if not s:
        return ""
    import re as _re
    return _re.sub(r"\s+", " ", s.strip().strip(" .,:;!?\"'()[]").lower()).strip()


# Phase 3b step 10: entity-salience recency half-life (days). A mention at the
# half-life is worth half a same-instant mention; 4x half-life -> 1/16. The
# mention factor (Phase 1c) and structural factor (Phase 3a head) are both
# already in [0,1], so recency multiplies in as [0,1]. Constant (not a Config
# knob) to match the Phase 1c mention formula, which is also hardcoded here.
_SALIENCE_RECENCY_HALF_LIFE_DAYS = 30.0

# Phase 2c+: per-unit feedback boost clamp bounds. The boost is a salience
# WEIGHT (default 1.0 = neutral) that multiplies a unit's retrieval score. It
# is clamped at WRITE (``write_unit_boost_batch``) so runaway feedback can't
# push a unit to 100x or 0x -- the read path is a raw multiply (no clamp there,
# so a clamped value is honored exactly). 0.25 .. 4.0 means a judged unit can
# at most halve (rating 1 repeatedly) or quadruple (rating 5 repeatedly) its
# raw score vs an unjudged one.
_BOOST_MIN = 0.25
_BOOST_MAX = 4.0


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
        # Native in-DB vector index (WaveDB VectorLayer, wavedb>=0.2.2). Opened
        # on the SAME database_t as self.db (VectorLayer.open shares it and
        # registers with db._open_subtrees), so it must close before db.close()
        # (see close()). FLAT/COSINE is exact + training-free; IVF graduation
        # is a later config flip + one train() call, not wired here. Gated on
        # vector_index_enabled + the installed wavedb exposing VectorLayer; when
        # absent, self.vector_layer stays None and the retriever falls back to
        # the FAISS VectorSearch path.
        self.vector_layer = None
        self._vector_dim = None
        if (config.get("vector_index_enabled", True) if config else True) and _has_vector_layer():
            from wavedb import VectorLayer, Format, IndexType, Distance, Runtime
            dim = (config.get("embedding_dim", 384) if config else 384)
            self.vector_layer = VectorLayer.open(
                "episodes",
                self.db,
                fmt=Format(IndexType.FLAT, dim, Distance.COSINE),
                rt=Runtime(top_k=10, sync_only=1),
            )
            self._vector_dim = dim
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
        eid = episode.id
        ops = self._content_ops(eid, episode) + self._edge_ops(eid, episode)
        self.db.batch_sync(ops)
        # Live upsert into the in-DB vector index. The VectorLayer insert is
        # its own atomic op on the same database_t -- it cannot ride in the
        # content batch_sync above -- so it happens after, best-effort. This
        # closes the FAISS-sidecar "not updated live" caveat: a newly encoded
        # episode is immediately searchable via search_sync, no offline rebuild.
        if episode.summary_embedding:
            self._index_embedding(eid, episode.summary_embedding, episode.summary or "")

    def _content_ops(self, eid: str, episode) -> list[dict]:
        """Content (neocortical) puts under ``content/ep/{eid}/...`` -- the
        stub half of an episode (no graph edges).

        ``encode_episode`` merges these with ``_edge_ops`` into ONE atomic
        ``batch_sync`` (the byte-identical synchronous path). The async-distill
        path calls them separately: ``encode_episode_content`` writes the stub
        (content + vector index) so a turn is immediately retrievable by
        embedding while its extraction runs in the background; the worker later
        calls ``encode_episode_edges`` to fill the graph index. Ordering is
        always content-before-edges, never edges without content.
        """
        ops: list[dict] = [
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
        # A2 BM25 inverted index (Tencent-survey item 3): lexical postings over
        # ``full_text`` hosted INSIDE WaveDB via HBTrie range scan (NO SQLite/FTS5).
        # Gated on the master-config flag (read at call time, the codebase
        # convention -- same pattern as ``assertion_extraction_enabled`` /
        # ``forgetting_enabled``). The index ops ride the SAME atomic
        # ``batch_sync`` as the content puts, so the index can't drift from the
        # content it indexes. Episodes-only for A2 (scenes use
        # ``_scene_content_ops``; the ``content/idx/`` key layout is generic so
        # docs/sections extend later without a schema change). ``bm25_index_ops``
        # is idempotent (``docterms``-exists guard) -> a re-encode is a skip, not a
        # double-count. Flag off -> appends nothing -> byte-identical.
        from ..config import config as _master_config
        if _master_config.hybrid_retrieval and episode.full_text:
            ops += bm25_index_ops(self.db, eid, episode.full_text)
        return ops

    def _edge_ops(self, eid: str, episode) -> list[dict]:
        """Graph index ops (hippocampal sparse pointers) -- the fill half.

        ``encode_episode`` merges these with ``_content_ops`` into one atomic
        ``batch_sync``. The async-distill worker calls this via
        ``encode_episode_edges`` AFTER the stub content is stored, populating
        ``entities``/``relations``/``state_assertions``/etc. from extraction.
        expand_triple returns root-namespace ops (the "memory/" subtree prefix
        is already prepended by the C helper).
        """
        ops: list[dict] = []
        # ── Graph: sparse pointers (hippocampal index) via expand_triple ──
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

        # ── Phase 3c: entity-state assertion edges (D1) ──
        # The production writer of ``(E:entity, state, value)`` edges -- the A2
        # anomaly-resolver input. Each assertion carries provenance
        # (``asserted_by=eid``, ``asserted_at=timestamp``) on the edge sidecar
        # via ``_assertion_edge_ops`` (RMW-merged, idempotent, never revives a
        # tombstone). Gated on ``assertion_extraction_enabled``; an empty
        # ``state_assertions`` list (no explicit state claims in the turn) is
        # a no-op, so a plain conversation is byte-identical to pre-Phase-3c.
        from ..config import config as _master_config
        if episode.state_assertions:
            if _master_config.assertion_extraction_enabled:
                for a in episode.state_assertions:
                    ent = a.get("entity")
                    val = a.get("value")
                    if not ent or val is None:
                        continue
                    ops += self._assertion_edge_ops(
                        ent, val, eid, episode.timestamp
                    )

        # ── Phase 3c: best-effort cited_from provenance (D5) ──
        # When the episode text references a known document's title or
        # source_path, link ``(eid, cited_from, doc_id)`` -- the graph-
        # visible complement to the assertion sidecar's ``asserted_by``
        # (which IS the primary supported_by link). Gated on
        # ``citation_resolution_enabled``; no edge when no match. O(N_docs)
        # reads + a ``default_document_ids`` scan per encode -- best-effort
        # for the first slice; a future title-index can replace the scan
        # (matters only for corpora with many docs AND many episodes).
        if _master_config.citation_resolution_enabled:
            text_lower = episode.full_text.lower()
            for _doc_id in self.default_document_ids():
                _title = _b2s(self.db.get_sync(f"content/doc/{_doc_id}/title"))
                if len(_title) >= 4 and _title.lower() in text_lower:
                    ops += self.graph.expand_triple(eid, "cited_from", _doc_id)
                    continue
                _sp = _b2s(self.db.get_sync(f"content/doc/{_doc_id}/source_path"))
                if len(_sp) >= 4 and _sp in episode.full_text:
                    ops += self.graph.expand_triple(eid, "cited_from", _doc_id)

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
        return ops

    def encode_episode_content(self, eid: str, episode) -> None:
        """Stub write: content + vector index, NO graph edges.

        The async-distill path: the main thread pre-allocates ``eid`` and calls
        this so the turn is immediately retrievable by embedding (``search_sync``)
        while its 22 s extraction runs on the background worker. The episode is
        graph-invisible until ``encode_episode_edges`` fills the edges. Atomic:
        one ``batch_sync`` of content puts + the best-effort vector-index upsert.
        """
        self.db.batch_sync(self._content_ops(eid, episode))
        if episode.summary_embedding:
            self._index_embedding(eid, episode.summary_embedding, episode.summary or "")

    def encode_episode_edges(self, eid: str, episode) -> None:
        """Fill write: graph edges only, for content already stored via
        ``encode_episode_content`` (or ``encode_episode``).

        The async-distill worker calls this after extraction populates the
        episode's ``entities``/``topics``/``tones``/``decisions``/``relations``/
        ``state_assertions``. One atomic ``batch_sync`` of edge ops; content is
        untouched (the stub already wrote it).
        """
        self.db.batch_sync(self._edge_ops(eid, episode))

    # ---- scene blocks (B1: LLM-authored topic-level macro-memory) ----
    #
    # A scene block is the system's synthesized understanding of one TOPIC for a
    # user -- an LLM-authored Markdown summary stored IN WaveDB (not a file on
    # disk) under ``content/scene/{scene_id}/...``. It is the deployable, no-
    # training symbolic macro layer tier 1 lacked AND the substrate the trained-
    # but-unwired GNN scene-ontology head will later write onto. Authored on
    # ingest by ``SceneAuthoringWorker`` with 4 actions (CREATE/UPDATE/MERGE/
    # skip); heat decays per tick and cold scenes are evicted = macro-forgetting
    # (the scene-level analog of fade R4). Retrieved via the SAME graph+vector
    # pipeline as episodes/docs -- ``has_topic`` surfaces scenes on the topics
    # axis, and the body embedding is indexed so scenes match semantically too.
    # Flag-gated (``--scene-blocks``) default OFF, byte-identical when off: no
    # scene is ever written with the flag off, so ``default_scene_ids()`` is
    # empty and no ``scene_`` id ever enters the graph or vector index.

    def _scene_content_ops(self, scene_id, body, topic, heat, updated_ts,
                           user_id, source_eps, body_embedding=None) -> list[dict]:
        """Content puts under ``content/scene/{scene_id}/...`` -- the stub half
        of a scene block (no graph edges). Mirrors ``_content_ops``: ``encode_scene``
        merges these with ``_scene_edge_ops`` into ONE atomic ``batch_sync``. The
        Markdown body is the VALUE of the ``body`` field, never a file on disk.
        ``heat`` is a stringified float (the scene-level forgetting signal);
        ``source_eps`` is a json list of cited episode ids (provenance for
        drill-down via the ``cites`` edge); ``embedding`` is the body vector
        (parity with ``content/ep/{id}/embedding``)."""
        ops: list[dict] = [
            {"type": "put", "key": f"content/scene/{scene_id}/body", "value": body or ""},
            {"type": "put", "key": f"content/scene/{scene_id}/topic", "value": topic or ""},
            {"type": "put", "key": f"content/scene/{scene_id}/heat", "value": str(float(heat))},
            {"type": "put", "key": f"content/scene/{scene_id}/updated_ts", "value": updated_ts or ""},
            {"type": "put", "key": f"content/scene/{scene_id}/user_id", "value": user_id or ""},
            {"type": "put", "key": f"content/scene/{scene_id}/source_eps",
             "value": json.dumps(list(source_eps or []))},
        ]
        if body_embedding:
            ops.append({"type": "put",
                        "key": f"content/scene/{scene_id}/embedding",
                        "value": json.dumps(body_embedding)})
        return ops

    def _scene_edge_ops(self, scene_id, topic, user_id, source_eps,
                        delete: bool = False) -> list[dict]:
        """Graph index ops for a scene block (put or delete). Mirrors
        ``_document_graph_ops``'s ``emit`` closure + symmetric reversal: a delete
        emits the SAME ``(s, p, o)`` with ``delete=True`` so SPO/POS/OSP all come
        out -- no orphan edges. Four edge types:

        * ``(scene, instanceOf, SceneBlock)`` -- typing.
        * ``(scene, has_topic, T:{topic})`` -- surfaces scenes on the topics axis
          (``graph.query().vertex(f"T:{topic}").in_("has_topic")``), the SAME axis
          episodes/docs use; the object is raw ``T:{topic}`` (NOT safe-hashed) so
          a scene and an episode sharing a topic land on the SAME ``T:`` node.
        * ``(U:{user}, owns_scene, scene)`` forward + ``(scene, owned_by, U:{user})``
          reverse -- mirrors doc ownership; the scope scan uses the forward SPO
          edge ``memory/spo/{user}/owns_scene/`` (see ``scene_ids_for_user``).
        * ``(scene, cites, ep_id)`` per source episode -- provenance for drill-down.
        """
        ops: list[dict] = []

        def emit(s: str, p: str, o: str) -> None:
            ops.extend(self.graph.expand_triple(s, p, o, delete=delete))

        emit(scene_id, "instanceOf", "SceneBlock")
        if topic:
            emit(scene_id, "has_topic", f"T:{topic}")
        if user_id:
            user_node = f"U:{user_id}"
            emit(user_node, "owns_scene", scene_id)
            emit(scene_id, "owned_by", user_node)
        for ep_id in (source_eps or []):
            if ep_id:
                emit(scene_id, "cites", ep_id)
        return ops

    def encode_scene(self, scene_id, body, topic, heat, updated_ts, user_id,
                     source_eps, body_embedding=None) -> None:
        """Store a scene block's content + graph index in ONE atomic batch, then
        best-effort upsert the body embedding into the in-DB vector index.

        Mirrors ``encode_episode`` (``store.py:173``): content + edges in one
        ``batch_sync``; the VectorLayer insert is its own atomic op on the same
        database_t (can't ride in the content batch), so it happens after, best-
        effort (a vector-index hiccup must never fail an encode). Idempotent: re-
        encoding the same ``scene_id`` overwrites in place (``batch_sync`` puts
        replace); with ``topic`` held stable the ``has_topic``/``owned_by`` edges
        are unchanged and only ``cites`` churns."""
        ops = (self._scene_content_ops(scene_id, body, topic, heat, updated_ts,
                                       user_id, source_eps, body_embedding)
               + self._scene_edge_ops(scene_id, topic, user_id, source_eps, delete=False))
        self.db.batch_sync(ops)
        if body_embedding:
            self._index_embedding(scene_id, body_embedding, body or "")

    def get_scene(self, scene_id: str) -> "Optional[dict]":
        """Load a scene block's content fields. Returns ``None`` for an absent
        scene (detected via the always-written ``heat`` key -- ``get_sync``
        returns ``None`` for a missing key, distinct from ``_b2s``'s ``""`` for
        an empty one). Edges (``has_topic``/``owned_by``/``cites``) are NOT read
        here -- the hydrate scan reads them, mirroring ``get_episode``."""
        heat_raw = self.db.get_sync(f"content/scene/{scene_id}/heat")
        if heat_raw is None:
            return None
        body = _b2s(self.db.get_sync(f"content/scene/{scene_id}/body"))
        topic = _b2s(self.db.get_sync(f"content/scene/{scene_id}/topic"))
        updated_ts = _b2s(self.db.get_sync(f"content/scene/{scene_id}/updated_ts"))
        user_id = _b2s(self.db.get_sync(f"content/scene/{scene_id}/user_id"))
        source_eps_raw = _b2s(self.db.get_sync(f"content/scene/{scene_id}/source_eps"))
        try:
            heat = float(_b2s(heat_raw))
        except ValueError:
            heat = 0.0
        try:
            source_eps = json.loads(source_eps_raw) if source_eps_raw else []
        except (ValueError, TypeError):
            source_eps = []
        return {
            "scene_id": scene_id,
            "body": body,
            "topic": topic,
            "heat": heat,
            "updated_ts": updated_ts,
            "user_id": user_id or None,
            "source_eps": source_eps,
        }

    def delete_scene(self, scene_id: str) -> None:
        """Evict a scene block: delete ALL its content keys, its graph edges
        (symmetric reversal -- no orphan SPO/POS/OSP entries), and its vector
        index entry, in ONE atomic ``batch_sync`` + a best-effort unindex.

        The eviction chokepoint, called by ``SceneAuthoringWorker`` when
        ``heat < floor`` (macro-forgetting) and on MERGE (the merged body
        supersedes the source scene). Mirrors the symmetric delete guarantee at
        ``_document_graph_ops`` (``store.py:1572``): re-derive the scene's
        topic/user/source_eps from content, build ``_scene_edge_ops(delete=True)``,
        plus a content-delete op per ``content/scene/{scene_id}/*`` key (enumerate
        via a range scan so a future-added field is cleaned up too)."""
        sc = self.get_scene(scene_id)
        ops: list[dict] = []
        # Content deletes: enumerate every field key under the scene's prefix.
        start = f"content/scene/{scene_id}/"
        end = f"content/scene/{scene_id}/\x7f"
        for k, _ in self.db.create_read_stream(start=start, end=end):
            ops.append({"type": "del", "key": k})
        # Edge deletes (symmetric reversal) -- only emit if the scene existed
        # (a delete on an absent scene is a no-op content-wise; edges already gone).
        if sc is not None:
            ops += self._scene_edge_ops(
                scene_id, sc.get("topic") or "", sc.get("user_id"),
                sc.get("source_eps") or [], delete=True)
        if ops:
            self.db.batch_sync(ops)
        # Best-effort vector-index removal (delete_sync on an absent id is a no-op).
        self._unindex_embedding(scene_id)

    def touch_scene(self, scene_id: str, delta: float = 0.2) -> None:
        """Bump a scene's heat by ``delta`` (clamped to [0, 1]) in a single-key
        RMW. The cheap path the retrieval-boost hook calls -- no full re-encode.
        A missing scene is a no-op (the retrieval boost can fire on a scene id
        that was evicted between hydrate and boost). Never raises."""
        try:
            cur_raw = self.db.get_sync(f"content/scene/{scene_id}/heat")
            if cur_raw is None:
                return
            try:
                cur = float(_b2s(cur_raw))
            except ValueError:
                return
            new_heat = max(0.0, min(1.0, cur + delta))
            self.db.batch_sync([{"type": "put",
                                 "key": f"content/scene/{scene_id}/heat",
                                 "value": str(new_heat)}])
        except Exception:  # noqa: BLE001 - touch is best-effort, never breaks retrieval
            pass

    def scene_ids_for_user(self, user_id: str) -> set:
        """All scene ids owned by a user (scan the ``owns_scene`` SPO index).

        EXACT mirror of ``document_ids_for_user`` (``store.py:1312``): scenes
        have no session axis -- ownership is user-level (a scene is a peer top-
        level unit like a doc). The forward edge ``(U:{user}, owns_scene, scene)``
        is written by ``_scene_edge_ops``; this scans the SPO index by
        subject+predicate. Empty for an unknown user. Scenes with ``user_id=None``
        are NOT in any user's set -> excluded under strict scope."""
        user = f"U:{user_id}"
        start = f"memory/spo/{user}/owns_scene/"
        end = f"memory/spo/{user}/owns_scene/\x7f"
        out: set = set()
        for k, _ in self.db.create_read_stream(start=start, end=end):
            scene_id = k.rsplit("/", 1)[-1]
            if scene_id:
                out.add(scene_id)
        return out

    def default_scene_ids(self) -> list:
        """All scene ids (scan ``content/scene/``).

        EXACT mirror of ``default_memory_ids`` (``store.py:922``) /
        ``default_document_ids`` (``store.py:1979``). Scenes are forgotten by
        HEAT (the scene-level forgetting signal), not by the episode state/
        validity axis, so there is no state filter here -- a deleted scene's
        content is gone, so it simply doesn't appear. Sorted for stable
        enumeration. Empty when no scenes exist (cold start / flag off) ->
        byte-identical to pre-B1 retrieval."""
        ids: set = set()
        for k, _ in self.db.create_read_stream(
                start="content/scene/", end="content/scene/\x7f"):
            parts = k.split("/", 3)
            if len(parts) < 3 or not parts[2]:
                continue
            ids.add(parts[2])
        return sorted(ids)

    # ---- task canvas (B4: structural short-term task-state memory) ----
    #
    # A task canvas is a per-task Mermaid flowchart the LLM authors + reads each
    # turn -- a STRUCTURAL task-state axis (phases: done|doing|paused|blocked)
    # the fade cannot provide (fade is prose-only; R4 never fires on code). The
    # offload-stack L2 analog of Tencent's Mermaid canvas. Stored IN WaveDB
    # under ``content/canvas/{canvas_id}/...`` (mirrors B1 scenes; NOT a file on
    # disk). The canvas is STRUCTURAL -> NO vector index (no ``_index_embedding``
    # call); it is the ONE active canvas per user, read by id after the L1.5
    # lifecycle gate sets it, NOT retrieved via the graph+vector pipeline (so it
    # never enters ``episodes``/``ChunkedContext`` -- different from scenes, which
    # ride the topics axis). One active canvas per user (``active`` = "1"/"0");
    # a reclaim pass in the persist tail keeps a per-user historical floor.
    # Flag-gated (``--task-canvas``) default OFF, byte-identical when off: no
    # canvas is ever written with the flag off, so ``canvas_ids_for_user()`` is
    # empty and no ``canvas_`` id ever enters the graph.

    def _canvas_content_ops(self, canvas_id, mermaid, task_label, progress,
                            created_ts, updated_ts, user_id, active,
                            node_mapping=None) -> list[dict]:
        """Content puts under ``content/canvas/{canvas_id}/...`` -- the stub half
        of a canvas (no graph edges). Mirrors ``_scene_content_ops``: ``encode_canvas``
        merges these with ``_canvas_edge_ops`` into ONE atomic ``batch_sync``.
        ``mermaid`` is the Mermaid string (the VALUE, never a file on disk);
        ``progress`` is a stringified int 0-100 (parsed from the ``%%{...}%%``
        header on write); ``active`` is "1"/"0" (one active canvas per user);
        ``node_mapping`` is a json dict of ``{node_id: source_id}`` provenance
        (stored verbatim in v1 -- episode resolution is a v2 follow-on)."""
        ops: list[dict] = [
            {"type": "put", "key": f"content/canvas/{canvas_id}/mermaid",
             "value": mermaid or ""},
            {"type": "put", "key": f"content/canvas/{canvas_id}/task_label",
             "value": task_label or ""},
            {"type": "put", "key": f"content/canvas/{canvas_id}/progress",
             "value": str(int(progress or 0))},
            {"type": "put", "key": f"content/canvas/{canvas_id}/created_ts",
             "value": created_ts or ""},
            {"type": "put", "key": f"content/canvas/{canvas_id}/updated_ts",
             "value": updated_ts or ""},
            {"type": "put", "key": f"content/canvas/{canvas_id}/user_id",
             "value": user_id or ""},
            {"type": "put", "key": f"content/canvas/{canvas_id}/active",
             "value": str(active or "0")},
            {"type": "put", "key": f"content/canvas/{canvas_id}/node_mapping",
             "value": json.dumps(node_mapping or {})},
        ]
        return ops

    def _canvas_edge_ops(self, canvas_id, task_label, user_id,
                         delete: bool = False) -> list[dict]:
        """Graph index ops for a canvas (put or delete). Mirrors
        ``_scene_edge_ops``'s ``emit`` closure + symmetric reversal. Three edge
        types (NO ``cites`` -- v1 has no episode links):

        * ``(canvas, instanceOf, CanvasBlock)`` -- typing.
        * ``(canvas, has_topic, T:{task_label})`` -- optional topic axis
          (grouping; the canvas is read by id, not retrieved, so this is a
          future-facing index, not a retrieval path).
        * ``(U:{user}, owns_canvas, canvas)`` forward + ``(canvas, owned_by,
          U:{user})`` reverse -- the scope scan uses the forward SPO edge
          ``memory/spo/{user}/owns_canvas/`` (see ``canvas_ids_for_user``).
        """
        ops: list[dict] = []

        def emit(s: str, p: str, o: str) -> None:
            ops.extend(self.graph.expand_triple(s, p, o, delete=delete))

        emit(canvas_id, "instanceOf", "CanvasBlock")
        if task_label:
            emit(canvas_id, "has_topic", f"T:{task_label}")
        if user_id:
            user_node = f"U:{user_id}"
            emit(user_node, "owns_canvas", canvas_id)
            emit(canvas_id, "owned_by", user_node)
        return ops

    def encode_canvas(self, canvas_id, mermaid, task_label, progress,
                      created_ts, updated_ts, user_id, active,
                      node_mapping=None) -> None:
        """Store a canvas's content + graph index in ONE atomic batch. Mirrors
        ``encode_scene`` MINUS the vector index (the canvas is structural -> no
        ``_index_embedding`` call). Idempotent: re-encoding the same
        ``canvas_id`` overwrites in place (``batch_sync`` puts replace); with
        ``task_label`` + ``user_id`` held stable the edges are unchanged."""
        ops = (self._canvas_content_ops(canvas_id, mermaid, task_label, progress,
                                        created_ts, updated_ts, user_id, active,
                                        node_mapping)
               + self._canvas_edge_ops(canvas_id, task_label, user_id, delete=False))
        self.db.batch_sync(ops)

    def get_canvas(self, canvas_id: str) -> "Optional[dict]":
        """Load a canvas's content fields. Returns ``None`` for an absent
        canvas (detected via the always-written ``active`` key -- ``get_sync``
        returns ``None`` for a missing key, distinct from ``_b2s``'s ``""`` for
        an empty one). Mirrors ``get_scene``."""
        active_raw = self.db.get_sync(f"content/canvas/{canvas_id}/active")
        if active_raw is None:
            return None
        mermaid = _b2s(self.db.get_sync(f"content/canvas/{canvas_id}/mermaid"))
        task_label = _b2s(self.db.get_sync(f"content/canvas/{canvas_id}/task_label"))
        progress_raw = _b2s(self.db.get_sync(f"content/canvas/{canvas_id}/progress"))
        created_ts = _b2s(self.db.get_sync(f"content/canvas/{canvas_id}/created_ts"))
        updated_ts = _b2s(self.db.get_sync(f"content/canvas/{canvas_id}/updated_ts"))
        user_id = _b2s(self.db.get_sync(f"content/canvas/{canvas_id}/user_id"))
        node_mapping_raw = _b2s(
            self.db.get_sync(f"content/canvas/{canvas_id}/node_mapping"))
        try:
            progress = int(progress_raw) if progress_raw else 0
        except ValueError:
            progress = 0
        try:
            node_mapping = json.loads(node_mapping_raw) if node_mapping_raw else {}
        except (ValueError, TypeError):
            node_mapping = {}
        return {
            "canvas_id": canvas_id,
            "mermaid": mermaid,
            "task_label": task_label,
            "progress": progress,
            "created_ts": created_ts,
            "updated_ts": updated_ts,
            "user_id": user_id or None,
            "active": _b2s(active_raw) == "1",
            "node_mapping": node_mapping,
        }

    def touch_canvas(self, canvas_id: str, mermaid=None, progress=None,
                     node_mapping=None) -> None:
        """Re-write a canvas's mutable fields (``mermaid``/``progress``/
        ``node_mapping``) + bump ``updated_ts`` (naive-local, no Z -- matches
        episode timestamps so a mixed-result sort never TypeErrors). The cheap
        path ``update_canvas`` calls -- no full re-encode (edges + ``user_id``/
        ``task_label``/``active``/``created_ts`` unchanged). A missing canvas is
        a no-op. Never raises (mirrors ``touch_scene``'s best-effort contract)."""
        try:
            if self.db.get_sync(f"content/canvas/{canvas_id}/active") is None:
                return
            ops: list[dict] = [{
                "type": "put",
                "key": f"content/canvas/{canvas_id}/updated_ts",
                "value": datetime.now().isoformat(),
            }]
            if mermaid is not None:
                ops.append({"type": "put",
                            "key": f"content/canvas/{canvas_id}/mermaid",
                            "value": mermaid})
            if progress is not None:
                ops.append({"type": "put",
                            "key": f"content/canvas/{canvas_id}/progress",
                            "value": str(int(progress))})
            if node_mapping is not None:
                ops.append({"type": "put",
                            "key": f"content/canvas/{canvas_id}/node_mapping",
                            "value": json.dumps(node_mapping)})
            self.db.batch_sync(ops)
        except Exception:  # noqa: BLE001 - touch is best-effort, never breaks a turn
            pass

    def delete_canvas(self, canvas_id: str) -> None:
        """Evict a canvas: delete ALL its content keys + its graph edges
        (symmetric reversal -- no orphan SPO/POS/OSP entries) in ONE atomic
        ``batch_sync``. Mirrors ``delete_scene``. Content-op type is ``"del"``
        (NOT ``"delete"`` -- the de-wonk fix; ``"delete"`` would be silently
        wrong). NO ``_unindex_embedding`` -- the canvas is never vector-indexed.
        The range scan enumerates every field key so a future-added field is
        cleaned up too."""
        cv = self.get_canvas(canvas_id)
        ops: list[dict] = []
        start = f"content/canvas/{canvas_id}/"
        end = f"content/canvas/{canvas_id}/\x7f"
        for k, _ in self.db.create_read_stream(start=start, end=end):
            ops.append({"type": "del", "key": k})
        if cv is not None:
            ops += self._canvas_edge_ops(
                canvas_id, cv.get("task_label") or "", cv.get("user_id"),
                delete=True)
        if ops:
            self.db.batch_sync(ops)

    def canvas_ids_for_user(self, user_id: str) -> set:
        """All canvas ids owned by a user (scan the ``owns_canvas`` SPO index).
        EXACT mirror of ``scene_ids_for_user`` with the canvas predicate. Empty
        for an unknown user; canvases with ``user_id=None`` are NOT in any
        user's set (excluded under strict scope)."""
        user = f"U:{user_id}"
        start = f"memory/spo/{user}/owns_canvas/"
        end = f"memory/spo/{user}/owns_canvas/\x7f"
        out: set = set()
        for k, _ in self.db.create_read_stream(start=start, end=end):
            canvas_id = k.rsplit("/", 1)[-1]
            if canvas_id:
                out.add(canvas_id)
        return out

    def set_active_canvas(self, user_id: str, canvas_id: str) -> None:
        """Mark ``canvas_id`` as the ONE active canvas for ``user_id``: flip the
        prior active canvas's ``active`` field to "0" (if any), set the new one's
        to "1". Enforces one-active-per-user. A ``canvas_id`` of ``None``/empty
        just clears the prior (the ``clear`` lifecycle case). Best-effort never
        raises -- a corrupt prior-active is skipped, not fatal."""
        try:
            # Flip the prior active canvas (if any) to historical.
            prior = self.get_active_canvas(user_id)
            if prior is not None and prior["canvas_id"] != canvas_id:
                self.db.batch_sync([{
                    "type": "put",
                    "key": f"content/canvas/{prior['canvas_id']}/active",
                    "value": "0",
                }])
            if canvas_id:
                self.db.batch_sync([{
                    "type": "put",
                    "key": f"content/canvas/{canvas_id}/active",
                    "value": "1",
                }])
        except Exception:  # noqa: BLE001 - active flip is best-effort
            pass

    def get_active_canvas(self, user_id: str) -> "Optional[dict]":
        """The ONE active canvas for ``user_id`` (``active == "1"``), or
        ``None``. Scans ``canvas_ids_for_user`` + reads each ``active`` flag;
        returns the first active. (At most one by the ``set_active_canvas``
        invariant; a pre-existing double-active from a race resolves to the
        lowest id deterministically.)"""
        for cid in sorted(self.canvas_ids_for_user(user_id)):
            if _b2s(self.db.get_sync(f"content/canvas/{cid}/active")) == "1":
                return self.get_canvas(cid)
        return None

    def reclaim_canvases(self, user_id: str, *, min_keep: int = 15) -> int:
        """Per-user historical-canvas reclaim: NEVER delete the active canvas;
        keep at least ``min_keep`` historical canvases; beyond the floor, delete
        oldest-by-``updated_ts``. Returns the number deleted. Mirrors the
        scene heat-evict macro-forgetting but on the ``updated_ts`` axis (a
        canvas has no heat signal -- recency is the forgetting signal). Called
        by the orchestrator's persist-tail reclaim step (gated on
        ``--task-canvas``). Best-effort never raises."""
        try:
            ids = self.canvas_ids_for_user(user_id)
            if not ids:
                return 0
            canvases = [c for c in (self.get_canvas(cid) for cid in ids) if c]
            active = [c for c in canvases if c["active"]]
            historical = [c for c in canvases if not c["active"]]
            if len(historical) <= min_keep:
                return 0
            # Oldest-by-updated_ts beyond the floor. Empty/missing ts sorts
            # oldest (treated as "" -> sorts first) -- the conservative choice.
            historical.sort(key=lambda c: c.get("updated_ts") or "")
            to_drop = historical[:len(historical) - min_keep]
            for c in to_drop:
                # Defensive: never delete an active canvas even if the active
                # flag flipped between the partition above and here.
                if c["active"]:
                    continue
                self.delete_canvas(c["canvas_id"])
            return len(to_drop)
        except Exception:  # noqa: BLE001 - reclaim is best-effort
            return 0

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

    def _index_embedding(self, eid: str, vec, summary: str = "") -> None:
        """Best-effort upsert of an embedding into the in-DB vector index.

        The single chokepoint for embedding writes: called from
        ``encode_episode`` (the live path), ``set_summary_embedding`` (the
        backfill path used by ``build_vector_index.py``), and
        ``encode_document`` (per-section document-chunk embeddings). Id-agnostic:
        the layer now holds ``ep_*`` episode ids, ``{doc_id}_sec_{i:03d}``
        section ids (per-chunk document vectors -- the doc-RAG semantic path),
        and (hypothetically) ``doc_*`` ids -- though documents are NOT
        vector-indexed (no doc-level embedding is ever written); only their
        sections are. Best-effort + logged + never raises -- a vector-index
        hiccup must never fail an encode (mirrors ``_persist_exchange``'s
        philosophy). An episode/section with a missing vector entry is still
        retrievable via the graph path.
        """
        import sys
        if self.vector_layer is None:
            return
        # Dim guard: the layer was opened at self._vector_dim (384 for bge-
        # small). insert_sync sends the vector to C without an explicit length
        # (the C layer reads fmt.dim floats), so a short vector (e.g. a stub
        # embedder's 64-dim vector in a test) would over-read or mismatch.
        # Skip silently on mismatch -- best-effort, and keeps stub-embedder
        # tests noise-free. The real embedder always matches.
        if self._vector_dim is not None and len(vec) != self._vector_dim:
            return
        try:
            self.vector_layer.insert_sync(eid, vec, metadata=(summary or "").encode("utf-8"))
        except Exception as e:  # noqa: BLE001 - vector index is best-effort
            print(f"[vector-index-fail] {eid}: {e}", file=sys.stderr)

    def set_summary_embedding(self, episode_id: str, embedding: list[float]) -> None:
        """Backfill an episode's summary embedding (Phase F vector index build).

        Writes ``content/ep/{episode_id}/embedding`` so ``get_episode`` and
        ``VectorSearch`` can read it back without re-running the encoder. Also
        upserts into the in-DB VectorLayer (when enabled), so the offline bulk
        build populates BOTH the FAISS sidecar and the in-DB index in one pass.
        """
        import json
        self.db.put_sync(
            f"content/ep/{episode_id}/embedding",
            json.dumps(embedding),
        )
        self._index_embedding(
            episode_id, embedding, self.get_episode(episode_id).summary or ""
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

    # ---- per-unit feedback boost (Phase 2c+) ----

    def get_unit_boost(self, unit_id: str) -> float:
        """Per-unit feedback salience weight; ``1.0`` on a fresh/unknown unit.

        A "unit" is any retrieval result id verbatim -- an episode (``ep_*``),
        a document (``doc_*``), or a section (``{doc_id}_sec_NNN``). The boost is
        a MULTIPLIER on the unit's raw retrieval score (the chat's ``boost``
        field, default 1 = neutral), written by ``record_feedback`` (a 1-5
        model-usefulness rating) and clamped to ``[_BOOST_MIN, _BOOST_MAX]`` at
        write time.

        Cold-start: ``1.0`` on a missing key -- NO cold-start penalty (unlike
        ``get_entity_salience``, which returns 0.0 for an unknown entity). So
        a fresh corpus scores byte-identically to pre-2c+ retrieval until a unit
        is actually judged. Ids with ``/`` or NUL are skipped (defensive, mirror
        ``write_entity_salience_batch``) and read as 1.0. One ``get_sync`` point
        lookup.
        """
        if not unit_id or "/" in unit_id or "\x00" in unit_id:
            return 1.0
        raw = _b2s(self.db.get_sync(f"content/unit_boost/{unit_id}"))
        if not raw:
            return 1.0
        try:
            return float(raw)
        except ValueError:
            return 1.0

    def write_unit_boost_batch(self, boosts: dict[str, float]) -> None:
        """Persist a batch of per-unit boosts in ONE sorted ``batch_sync``.

        Keys are SORTED before submission (the load-bearing ``get_sync``-safe
        constraint -- mirror ``write_entity_salience_batch``: sorted insertion
        avoids the insertion-order-dependent split path). Each value is clamped
        to ``[_BOOST_MIN, _BOOST_MAX]`` at WRITE (the clamp is the policy, not
        the read path). Ids with ``/`` or NUL are skipped. Empty input is a
        no-op. Called by ``record_feedback`` (the single boost-mutation site).
        """
        ops: list[dict] = []
        for unit_id in sorted(boosts):
            if not unit_id or "/" in unit_id or "\x00" in unit_id:
                continue
            val = min(_BOOST_MAX, max(_BOOST_MIN, float(boosts[unit_id])))
            ops.append({
                "type": "put",
                "key": f"content/unit_boost/{unit_id}",
                "value": repr(val),
            })
        if ops:
            self.db.batch_sync(ops)

    def persist_unit_boost(self, unit_id: str, boost: float) -> None:
        """Single-point convenience wrapper over ``write_unit_boost_batch``."""
        self.write_unit_boost_batch({unit_id: boost})

    def record_feedback(self, judgments: list[dict],
                        query: Optional[str] = None) -> int:
        """Apply per-unit 1-5 usefulness ratings -> boost mutations.

        Each judgment is ``{"unit_id": str, "rating": int 1..5}``. The rating
        maps to a multiplicative nudge: ``useful = (rating-1)/4`` (1 -> 0, 3 ->
        0.5, 5 -> 1), then ``new = clamp(old * (0.5 + useful), _BOOST_MIN,
        _BOOST_MAX)`` -- rating 1 halves the boost, 3 is neutral, 5 grows it
        1.5x. Repeated judgments compound (over many turns a useful unit rises,
        a useless one falls), bounded by the clamp. ONE sorted
        ``write_unit_boost_batch`` (idempotent re-rating of the same unit in one
        call applies them in sorted-id order).

        STRM Phase 2a raw-rating tap: when ``strm_relevance_logging`` is ON, the
        RAW per-unit rating is appended to
        ``data/training/strm_relevance/feedback.jsonl`` BEFORE the boost
        reduction -- today only the reduced multiplier is persisted and the raw
        rating / query / slot_index (the 2a relevance head's training signal)
        are lost. The tap record is ``{unit_id, rating, query, slot_index,
        timestamp}`` (``slot_index`` from the judgment if the consumer knew
        which ring slot it was rating, else ``null``; ``query`` threaded from
        the orchestrator's current query). Best-effort, NEVER raises: a tap
        failure is logged and swallowed (a logging failure must never break the
        boost reduction or the consumer's loop).

        Best-effort, NEVER raises: a bad rating / unknown unit / store error is
        logged to stderr and swallowed (mirror ``_persist_exchange`` -- a
        feedback failure must never break the consumer's loop or lose the
        response). Returns the count of judgments actually applied (0 on a
        whole-method failure).
        """
        if not judgments:
            return 0
        # STRM 2a raw-rating tap (BEFORE the reduction; best-effort, never
        # raises -- a tap failure must not break the boost write).
        try:
            from ..config import config as _master_config
            if getattr(_master_config, "strm_relevance_logging", False):
                self._append_relevance_tap(judgments, query)
        except Exception as e:  # noqa: BLE001 - tap is best-effort
            import sys
            print(f"[record_feedback-tap-fail] {e}", file=sys.stderr)
        try:
            updates: dict[str, float] = {}
            for j in judgments:
                try:
                    unit_id = j["unit_id"]
                    rating = int(j["rating"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not unit_id or "/" in unit_id or "\x00" in unit_id:
                    continue
                if rating < 1 or rating > 5:
                    continue
                useful = (rating - 1) / 4.0
                old = self.get_unit_boost(unit_id)
                new = old * (0.5 + useful)
                updates[unit_id] = new
            if updates:
                self.write_unit_boost_batch(updates)
            return len(updates)
        except Exception as e:  # noqa: BLE001 - never break the consumer's loop
            import sys
            print(f"[record_feedback-fail] {e}", file=sys.stderr)
            return 0

    def _append_relevance_tap(self, judgments: list[dict],
                              query: Optional[str]) -> None:
        """Append the raw per-unit rating to the STRM 2a feedback JSONL.

        ``data/training/strm_relevance/feedback.jsonl`` (gitignored,
        regenerable). One line per judgment: ``{unit_id, rating, query,
        slot_index, timestamp}``. ``slot_index`` is taken from the judgment if
        the consumer supplied it (the external LLM / tool layer that knows
        which ring slot it is rating), else ``null``. ``query`` is the
        orchestrator's current query (threaded through ``record_feedback``),
        else ``null``. The tap is a side-effect-only append -- it never
        affects the boost reduction and never raises (the caller wraps it).
        """
        tap_path = os.path.join("data", "training", "strm_relevance",
                                "feedback.jsonl")
        os.makedirs(os.path.dirname(tap_path), exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with open(tap_path, "a", encoding="utf-8") as f:
            for j in judgments:
                try:
                    unit_id = j.get("unit_id")
                    rating = j.get("rating")
                    slot_index = j.get("slot_index")  # may be absent -> None
                except (AttributeError, TypeError):
                    continue
                if not isinstance(unit_id, str) or not unit_id:
                    continue
                if not isinstance(rating, int):
                    continue
                f.write(json.dumps({
                    "unit_id": unit_id,
                    "rating": rating,
                    "query": query if isinstance(query, str) else None,
                    "slot_index": slot_index if isinstance(slot_index, int) else None,
                    "timestamp": ts,
                }, ensure_ascii=False) + "\n")

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

    def default_memory_ids(self) -> "list[str]":
        """All semantic-memory ids (``M:NNNN``), scanning ``content/mem/``.

        Parallel to ``default_document_ids`` but for semantic memories (the
        DiffPool cluster abstractions the consolidator writes). M-nodes are
        NOT forgotten (no state/validity filter), so every written memory is a
        retrieval candidate -- the abstract SURROGATE for its source episodes
        (spec 371). The scan range ``content/mem/`` .. ``content/mem/\\x7f``
        enumerates every ``content/mem/{mid}/...`` key prefix; parts[2] is the
        ``M:NNNN`` id. Sorted so retrieval candidate enumeration is stable.
        """
        ids: set[str] = set()
        for k, _ in self.db.create_read_stream(
            start="content/mem/", end="content/mem/\x7f"
        ):
            parts = k.split("/", 3)
            if len(parts) < 3 or not parts[2]:
                continue
            ids.add(parts[2])
        return sorted(ids)

    # ---- episode-level forgetting state (Phase 3b) ----

    def _unindex_embedding(self, eid: str) -> None:
        """Best-effort removal of an embedding from the in-DB vector index.

        The single delete chokepoint, called from ``set_episode_state`` when an
        episode is deprecated/superseded, and from ``encode_document`` /
        ``delete_document`` for per-section document-chunk embeddings. Id-
        agnostic: ``delete_sync`` on an absent id is a no-op, so this is safe to
        call on any id (``ep_*``, ``{doc_id}_sec_{i:03d}``, or a stale ``doc_*``).
        Best-effort + logged + never raises -- a vector-index hiccup must never
        fail a state change or a document encode/delete. NOTE:
        ``SemanticMemoryWriter.supersede_episode`` writes
        ``state="superseded"`` directly (NOT via this method), so it calls this
        helper itself; hooking only ``set_episode_state`` would miss
        reconsolidation/anomaly-driven supersession.

        A2 BM25 unindex rides this SAME chokepoint (not a separate mirror at
        each call site) so no deprecate/supersede path can orphan the lexical
        index. UNCONDITIONAL on the flag: ``bm25_unindex_ops`` is self-gating
        (reads ``content/idx/docterms/{eid}``; absent -> ``[]``), so it is a
        no-op for never-indexed ids (flag was off at encode, or a scene/section
        id) AND it cleans orphan postings from a prior flag-ON period even when
        the flag is now off -- the index stays consistent with episode lifecycle
        regardless of flag flips. For non-episode ids (scenes/docs/sections)
        ``docterms`` is absent -> no-op. Best-effort + never raises, mirroring
        the vector path.
        """
        import sys
        if self.vector_layer is not None:
            try:
                self.vector_layer.delete_sync(eid)
            except Exception as e:  # noqa: BLE001 - vector index is best-effort
                print(f"[vector-index-fail] delete {eid}: {e}", file=sys.stderr)
        # A2 BM25 unindex (same chokepoint as the vector delete; see above).
        try:
            unindex_ops = bm25_unindex_ops(self.db, eid)
            if unindex_ops:
                self.db.batch_sync(unindex_ops)
        except Exception as e:  # noqa: BLE001 - lexical index is best-effort
            print(f"[bm25-index-fail] unindex {eid}: {e}", file=sys.stderr)

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

        When the in-DB vector layer is enabled, a deprecated/superseded episode
        is also removed from the vector index immediately (the FAISS sidecar
        could not delete; this closes that caveat), best-effort via
        ``_unindex_embedding``. A later revival to ``"current"`` re-inserts on
        its next encode (rare; not auto-reinserted here -- avoids a re-embed
        cost on a rare path).

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
        if state in ("deprecated", "superseded"):
            self._unindex_embedding(episode_id)

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

    # ---- Phase 3c: entity-state assertions + citation + tombstone ----

    def _assertion_edge_ops(
        self, entity: str, value, asserted_by: str, asserted_at: str
    ) -> list[dict]:
        """Build the atomic ops for one assertion edge + its provenance sidecar.

        ``(E:entity, state, value)`` graph triple (idempotent via
        ``expand_triple``) + an edge-sidecar put recording ``asserted_by`` /
        ``asserted_at`` (the episode/document/section id that asserted the
        value + when). The sidecar is RMW-merged over the existing sidecar
        (``get_edge_meta`` returns ``default_meta`` for a fresh edge, or the
        live sidecar for a re-asserted value), so re-asserting the SAME value
        preserves forgetting state (utility/decay/access_count) and NEVER
        revives a tombstoned value (a ``state="superseded"`` edge stays
        superseded -- the adjudicator owns tombstones). The provenance is
        updated to the latest asserting unit (latest-asserting wins, the A2
        resolver's signal).

        Returns ops ready to splice into a caller's ``batch_sync`` (the
        graph ops + one sidecar put). The RMW read happens before the batch
        is submitted (single-threaded -> consistent).
        """
        from .edge_meta import edge_meta_put_op, get_edge_meta
        subj = f"E:{entity}"
        obj = str(value)
        ops = self.graph.expand_triple(subj, "state", obj)
        meta = get_edge_meta(self, subj, "state", obj)
        meta["asserted_by"] = asserted_by
        meta["asserted_at"] = asserted_at
        ops.append(edge_meta_put_op(subj, "state", obj, meta))
        return ops

    def find_document_by_title_or_url(self, literal: "Optional[str]") -> "Optional[str]":
        """Resolve a citation literal to a Document node id, best-effort.

        Scans every document's ``title`` + ``source_path`` (via
        ``default_document_ids``). A non-URL literal matches on normalized
        title equality, then title substring containment either way (so
        "Policy A" resolves a doc titled "Policy A: Remote Work"). A URL
        literal (``http(s)://``) matches when it is a substring of a doc's
        ``source_path`` or ``title``. Returns the first matching doc id, or
        ``None``. O(N_docs) point reads -- cheap for a small corpus, best-
        effort for large (the citation path is not hot). Used by
        ``encode_document`` to resolve ``Document.citations`` (D5).
        """
        if not literal:
            return None
        lit = literal.strip()
        if not lit:
            return None
        lit_lower = lit.lower()
        is_url = lit_lower.startswith(("http://", "https://"))
        lit_norm = _norm_title(lit)
        for doc_id in self.default_document_ids():
            title = _b2s(self.db.get_sync(f"content/doc/{doc_id}/title"))
            source_path = _b2s(self.db.get_sync(f"content/doc/{doc_id}/source_path"))
            if is_url:
                if source_path and lit in source_path:
                    return doc_id
                if title and lit in title:
                    return doc_id
            else:
                tnorm = _norm_title(title)
                if tnorm and lit_norm:
                    if tnorm == lit_norm:
                        return doc_id
                    if lit_norm in tnorm or tnorm in lit_norm:
                        return doc_id
                if source_path and lit and lit in source_path:
                    return doc_id
        return None

    def supersede_assertion(
        self, entity: str, old_value, new_unit: str, when: str
    ) -> None:
        """Fact-level tombstone (D2): mark ``(E:entity, state, old_value)`` superseded.

        The chat's fact-level MVCC: the OLD assertion edge's sidecar gets
        ``state="superseded"`` + ``superseded_by=new_unit`` (the asserting
        unit that contradicted it) + ``superseded_at=when`` + ``validity_end=
        when``. The edge itself is NOT deleted (MVCC -- it stays retrievable
        via include_inactive); ``is_edge_current`` then returns False for it,
        so once the anomaly subgraph load threads edge-currentness (D2),
        the tombstoned old value drops out of the live-values set and the
        ``_detect_contradictory_state`` detector goes quiet -- the
        contradiction is *resolved*, not just recorded. RMW-merged over the
        existing sidecar (preserves utility/decay/access_count + the
        original ``asserted_by``/``asserted_at`` provenance).
        """
        from .edge_meta import get_edge_meta, update_edge_meta
        # Accept either a raw entity name (``team``) or a pre-prefixed id
        # (``E:team`` -- what ``anom["node"]`` carries from the detector);
        # the assertion edges are stored under ``E:{name}``.
        name = entity[2:] if entity.startswith("E:") else entity
        subj = f"E:{name}"
        obj = str(old_value)
        meta = get_edge_meta(self, subj, "state", obj)
        meta["state"] = "superseded"
        meta["superseded_by"] = new_unit
        meta["superseded_at"] = when
        meta["validity_end"] = when
        update_edge_meta(self, subj, "state", obj, meta)

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

    def create_class(self, name: str, parent: "Optional[str]", when: str) -> None:
        """Create a runtime-discovered class in ONE atomic ``batch_sync``.

        Writes ``(name, subClassOf, parent)`` (when ``parent`` is given) + the
        discovered marker + last_seen stamp, so the new class is immediately
        decay-eligible and reassignment-ready (the mechanism A3 / Bonsai-gated
        promotion lands into). Composite helper (not a primitive): the three
        writes share one transaction/WAL record with the graph triple, mirroring
        ``encode_episode``'s content+graph batch.

        ``name`` / ``parent`` must be slash- and NUL-free: ``scan_classes``
        reads ``content/class/{name}/field`` and returns ``parts[2]`` verbatim,
        so a literal slash would split the key and the class would surface under
        its hash, not its name (silent + confusing). ``safe_edge_component``
        would hash it for the graph key, but the content/class sidecar would
        still split -- so reject up front rather than half-hash.
        """
        for part, label in ((name, "name"), (parent, "parent")):
            if part is None:
                continue
            if "/" in part or "\x00" in part or not part:
                raise ValueError(
                    f"create_class: {label} must be non-empty and free of "
                    f"'/' and NUL (got {part!r})"
                )
        ops: list[dict] = []
        if parent:
            ops += self.graph.expand_triple(name, "subClassOf", parent)
        ops.append({"type": "put", "key": self._class_key(name, "discovered"), "value": "1"})
        ops.append({"type": "put", "key": self._class_key(name, "last_seen"), "value": when})
        if ops:
            self.db.batch_sync(ops)

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

    # ---- Retrieval user-scope primitives (the read-side of the user boundary) ----
    # The encoder writes ``(U:{user}, has_session, S)`` + ``(S, has_episode, eid)``
    # at encode (``_edge_ops``); the document write path adds
    # ``(U:{user}, owns_document, doc_id)`` (``_document_graph_ops``). These
    # methods enumerate the user's owned ids so the retriever can intersect its
    # candidate set with them. They are the ONLY user-scoped reads the retriever
    # consults; everything else (``default_*_ids``) is across-all-users.

    def episode_ids_for_user(self, user_id: str) -> set[str]:
        """All episode ids owned by a user (the union of their sessions).

        Reuses :meth:`list_sessions` + :meth:`list_session_episodes` -- no new
        key scans, just the two-level walk over the existing ``has_session`` /
        ``has_episode`` SPO index. Empty for an unknown user (no sessions). The
        retriever intersects its candidate set with this; episodes NOT in this
        set are excluded under strict user-scope. Episodes ingested with
        ``user_id=None`` (pre-provenance / unscoped) are NOT in any user's set,
        so they are excluded too (the deliberate strict-ness; see
        ``claim_unscoped_documents`` for the docs-side backfill).
        """
        out: set[str] = set()
        for sess in self.list_sessions(user_id):
            out.update(self.list_session_episodes(sess))
        return out

    def document_ids_for_user(self, user_id: str) -> set[str]:
        """All document ids owned by a user (scan the ``owns_document`` SPO index).

        Documents have no session axis -- ownership is user-level only (a doc
        is a peer top-level unit, not a chat turn). The forward edge
        ``(U:{user}, owns_document, doc_id)`` is written by
        ``_document_graph_ops`` when ``doc.user_id`` is set; this scans the SPO
        index by subject+predicate (the same pattern as :meth:`list_sessions`).
        Empty for an unknown user. Docs with ``user_id=None`` (pre-user-scope /
        unscoped) are NOT in any user's set -> excluded under strict scope until
        :meth:`claim_unscoped_documents` stamps them.

        (The POS index orders object-before-subject, so it cannot be prefix-
        scanned by user -- the SPO index is the right one here.)
        """
        user = f"U:{user_id}"
        start = f"memory/spo/{user}/owns_document/"
        end = f"memory/spo/{user}/owns_document/\x7f"
        out: set[str] = set()
        for k, _ in self.db.create_read_stream(start=start, end=end):
            # k = memory/spo/{user}/owns_document/{doc_id}
            doc_id = k.rsplit("/", 1)[-1]
            if doc_id:
                out.add(doc_id)
        return out

    def claim_unscoped_documents(self, user_id: str) -> int:
        """One-time backfill: stamp ``user_id`` on every unscoped document.

        Strict user-scope EXCLUDES unscoped content, so existing docs (encoded
        before the ``user_id`` field / with no owner) would vanish the moment
        ``--retrieval-user-scope`` turns on. This walks ``default_document_ids``,
        skips docs that already have an owner (``content/doc/{id}/user`` present
        or an ``owned_by`` edge -- the cheap content-key check first, the graph
        check as a backstop), and stamps ``user_id`` on the rest (content key +
        the two ownership graph edges). Idempotent: a second call is a no-op
        (every doc is now owned). Never clobbers an existing owner. Returns the
        number of docs claimed. Run via ``serve_ponder --claim-docs`` or once
        per store before enabling user-scope.
        """
        user_node = f"U:{user_id}"
        claimed = 0
        for doc_id in self.default_document_ids():
            # Cheap already-owned check: the content key (one get_sync).
            if _b2s(self.db.get_sync(f"content/doc/{doc_id}/user")):
                continue
            # Backstop: an owner edge without the content key (shouldn't happen
            # in normal encode, but a partial write could leave one). Skip if any
            # user already owns this doc. The ``owned_by`` edge is
            # ``(doc_id, owned_by, U:{user})`` -> SPO scan by subject+predicate.
            owner_start = f"memory/spo/{doc_id}/owned_by/"
            owner_end = f"memory/spo/{doc_id}/owned_by/\x7f"
            if any(True for _ in self.db.create_read_stream(
                    start=owner_start, end=owner_end)):
                continue
            ops: list[dict] = [
                {"type": "put", "key": f"content/doc/{doc_id}/user",
                 "value": user_id},
            ]
            ops += self.graph.expand_triple(user_node, "owns_document", doc_id)
            ops += self.graph.expand_triple(doc_id, "owned_by", user_node)
            self.db.batch_sync(ops)
            claimed += 1
        return claimed

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
        # Phase 3c (D1/D5): persisted so update/delete are symmetric.
        "resolved_citations", "state_assertions",
        # Phase 3c Sec 7.11: semantic doc-kind tag (zero-shot at ingest).
        "doc_kind",
        # User scope (retrieval boundary): conditional at encode, so a delete
        # no-ops on a pre-user-scope doc (WaveDB ``del_sync`` ignores absent
        # keys -- mirrors the conditional ``embedding``/``summary`` deletes).
        "user",
    )
    _SEC_FIELDS = (
        "heading", "level", "parent", "blob_hash", "entities", "topics",
        "embedding", "summary",
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

    def next_scene_id(self) -> str:
        """Globally-unique, monotonically-increasing scene block id
        (``scene_NNNNNN``).

        Counter lives under ``content/system/scene_counter`` beside the
        episode/session/memory/document counters -- a NEW counter key, no
        collision with any existing one (same RMW caveat as
        ``next_episode_id``). Mirrors ``next_document_id`` (``store.py:1714``).
        """
        return f"scene_{self._counter_next('scene_counter'):06d}"

    def next_canvas_id(self) -> str:
        """Globally-unique, monotonically-increasing task-canvas id
        (``canvas_NNNNNN``).

        Counter lives under ``content/system/canvas_counter`` -- a NEW counter
        key, no collision (same non-atomic-RMW caveat as ``next_scene_id``).
        Mirrors ``next_scene_id`` (``store.py:1754``)."""
        return f"canvas_{self._counter_next('canvas_counter'):06d}"

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

    def document_source_path(self, doc_id: str) -> Optional[str]:
        """Light reverse lookup: the ``source_path`` for ``doc_id``, or ``None``.

        One hot-store ``get_sync`` (no section / cold-body pull, unlike
        ``get_document``). Used by the Phase 3c contradiction decider context
        (``_gather_entity_context``) to resolve an assertion's ``asserted_by``
        doc/section id back to its source filename so the complementary-temporal
        guard can recognize month-named point-in-time records
        (``docs/jan-status.md``). ``None`` for a missing doc or an
        episode/``None`` provenance id (caller falls through to the LLM).
        """
        if not isinstance(doc_id, str) or not doc_id:
            return None
        raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/source_path"))
        return raw or None

    def document_kind(self, doc_id: str) -> Optional[str]:
        """Light reverse lookup: the ``doc_kind`` tag for ``doc_id``, or ``None``.

        Byte-identical shape to ``document_source_path``: one hot-store
        ``get_sync`` on ``content/doc/{doc_id}/doc_kind`` (no section / cold-
        body pull). Used by the Phase 3c Sec 7.11 contradiction decider context
        (``_gather_entity_context``) so the complementary-temporal guard can
        fire on the semantic ``point_in_time_snapshot`` tag instead of a
        filename month-prefix -- production-sound on real enterprise docs that
        carry no month in their names. ``None`` for a missing doc or an
        episode/``None`` provenance id (caller falls through to the month-prefix
        fallback / the LLM). A doc encoded before this field has no key ->
        ``None`` (back-compatible, same as a never-tagged doc).
        """
        if not isinstance(doc_id, str) or not doc_id:
            return None
        raw = _b2s(self.db.get_sync(f"content/doc/{doc_id}/doc_kind"))
        return raw or None

    def _document_graph_ops(self, doc: Document, delete: bool) -> list[dict]:
        """Build expand_triple ops for a doc's graph pointers (put or delete).

        Triples (per the ingestion plan): ``(doc, instanceOf, Document)``,
        ``(doc, has_entity, E:x)`` + ``(E:x, appears_in_doc, doc)``,
        ``(doc, has_topic, T:t)``, ``(doc, has_section, sid)``,
        ``(sid, child_of, parent)``, ``(sid, has_entity, E:x)`` +
        ``(E:x, appears_in_section, sid)`` (the entity->SECTION reverse, so the
        graph entity axis lands on the relevant chunk, not just the doc),
        ``(sid, has_topic, T:t)`` (per-section topic axis), ``(doc, cites,
        target)``, plus each Bonsai ``relations`` triple. The document-specific
        predicates (``has_section``/``child_of``/``appears_in_doc``/``cites``/
        ``appears_in_section``) are snake_case hash-tail predicates,
        checkpoint-safe + GNN-invisible until retrain (the A3 ``instanceOf``
        template). ``has_topic``/``has_entity`` ARE known -> doc/section nodes
        enter GNN BFS on retrain (distribution shift, NOT a checkpoint break).
        Reversing a put uses the SAME ``(s, p, o)`` with ``delete=True`` so the
        SPO/POS/OSP index entries all come out -- the per-section ``has_topic``
        + ``appears_in_section`` reverse edges are deleted symmetrically, no
        orphan.
        """
        ops: list[dict] = []

        def emit(s: str, p: str, o: str) -> None:
            ops.extend(self.graph.expand_triple(s, p, o, delete=delete))

        emit(doc.id, "instanceOf", "Document")
        # User scope (retrieval boundary): when the doc carries a ``user_id``,
        # own it in the graph so ``document_ids_for_user`` can scope retrieval.
        # Both the forward edge (user -> doc, the scoping scan) and the reverse
        # (doc -> user, a cheap per-doc owner lookup) are emitted; ``emit`` uses
        # the SAME ``(s, p, o)`` with ``delete=True`` on the delete path, so a
        # doc update/delete retracts both symmetrically (no orphans). Gated on
        # ``doc.user_id`` -> byte-identical for pre-user-scope docs.
        if doc.user_id:
            user_node = f"U:{doc.user_id}"
            emit(user_node, "owns_document", doc.id)
            emit(doc.id, "owned_by", user_node)
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
                emit(f"E:{e}", "appears_in_section", sec.id)  # entity->section reverse
            for t in sec.topics:
                emit(sec.id, "has_topic", f"T:{t}")           # per-section topic axis
        # Phase 3c (D5): cite targets are the STORE-resolved set -- a doc_id
        # when ``find_document_by_title_or_url`` matched the literal, else the
        # literal itself. ``resolved_citations`` is PERSISTED so the delete
        # path emits symmetric deletes for exactly the edges that were written
        # (resolution is store-state-dependent + not reproducible at delete
        # time); fall back to the literal ``citations`` for docs encoded before
        # this field / when citation resolution is disabled (byte-identical).
        cite_targets = doc.resolved_citations or doc.citations
        for target in cite_targets:
            emit(doc.id, "cites", target)
        for rel in doc.relations:
            emit(rel["subject"], rel["predicate"], rel["object"])

        # Phase 3c (D1): entity-state assertion edges. These are ENTITY-owned
        # (subject = E:entity, NOT doc-owned), so they are PUT-ONLY -- a doc
        # update/delete does NOT retract a fact the doc asserted (it may be
        # asserted elsewhere, and a doc updating X->Y is exactly the
        # contradiction the resolver should catch, not silently delete). The
        # sidecar is RMW-merged (``_assertion_edge_ops``) so re-assertion
        # preserves forgetting state and never revives a tombstoned value.
        if not delete:
            from ..config import config as _master_config
            if _master_config.assertion_extraction_enabled and doc.state_assertions:
                for a in doc.state_assertions:
                    ent = a.get("entity")
                    val = a.get("value")
                    if not ent or val is None:
                        continue
                    asserted_by = a.get("section") or doc.id
                    ops += self._assertion_edge_ops(
                        ent, val, asserted_by, doc.ingested_at
                    )

        # Phase 3c (D5): email thread provenance edges. The email parser
        # collects per-message Message-IDs + in_reply_to/references maps in
        # ``doc.metadata``; map each Message-ID to its section id (one section
        # per message, in emission order) and emit (section, in_reply_to,
        # parent_section) + (section, references, ref_section) for in-set refs.
        # PUT is gated on ``citation_resolution_enabled`` (the de-wonk cold-
        # start: citation off -> zero in_reply_to edges); DELETE always emits
        # so a flag flip after encoding still cleans up (delete_sync no-ops
        # an absent edge). Section ids are doc-specific -> safe to delete
        # symmetrically (no shared-edge concern, unlike assertions).
        meta = doc.metadata or {}
        message_ids = meta.get("message_ids") or []
        if message_ids:
            from ..config import config as _master_config
            cit_on = _master_config.citation_resolution_enabled
            if delete or cit_on:
                mid_to_sid = {
                    mid: f"{doc.id}_sec_{i:03d}"
                    for i, mid in enumerate(message_ids)
                }
                in_reply_to = meta.get("in_reply_to") or {}
                for mid, parent in in_reply_to.items():
                    sid = mid_to_sid.get(mid)
                    psid = mid_to_sid.get(parent)
                    if sid and psid and sid != psid:
                        emit(sid, "in_reply_to", psid)
                references = meta.get("references") or {}
                for mid, refs_raw in references.items():
                    sid = mid_to_sid.get(mid)
                    if not sid:
                        continue
                    for tok in str(refs_raw).split():
                        ref_mid = tok.strip().strip("<>")
                        rsid = mid_to_sid.get(ref_mid)
                        if rsid and rsid != sid:
                            emit(sid, "references", rsid)
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
        # Phase 3c Sec 7.11: semantic doc-kind tag (zero-shot at ingest).
        put(f"{d}/doc_kind", doc.doc_kind)
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
        # Phase 3c (D1/D5): the resolved cite targets + state assertions
        # actually written to the graph, persisted so update/delete emit
        # symmetric ops (resolution is store-state-dependent, not reproducible
        # at delete time). Empty when citation resolution / assertion
        # extraction is disabled -> the graph-ops builder falls back to the
        # literal ``citations`` (byte-identical to pre-Phase-3c).
        put(f"{d}/resolved_citations", json.dumps(doc.resolved_citations))
        put(f"{d}/state_assertions", json.dumps(doc.state_assertions))
        # User scope (retrieval boundary): written ONLY when set (conditional,
        # like ``created_at``/``embedding``) so a pre-user-scope / unscoped doc
        # is byte-identical. The graph carries the scoping edge; this content
        # key is the cheap per-doc owner read (avoids a graph scan in
        # ``claim_unscoped_documents``'s already-owned check).
        if doc.user_id:
            put(f"{d}/user", doc.user_id)
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
            # STRM 1f-6: prose code-section summary (embedding handle). Written
            # ONLY when present (conditional, like ``embedding``) so a store
            # built without a summarizer is byte-identical to pre-1f-6. The
            # recalled content (``{s}/blob_hash`` -> cold body) is untouched.
            if sec.summary:
                put(f"{s}/summary", sec.summary)
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
        # Phase 3c (D5): resolve citation literals to Document nodes ONCE here
        # so the put + the persisted ``resolved_citations`` (read back on
        # update/delete for symmetric deletes) agree. When citation
        # resolution is disabled, ``resolved_citations`` = the literals ->
        # byte-identical to pre-Phase-3c. ``doc.state_assertions`` is populated
        # by the ingestion pipeline (deterministic normalizer + Bonsai
        # has_state); the graph-ops builder writes the assertion edges.
        from ..config import config as _master_config
        if _master_config.citation_resolution_enabled and doc.citations:
            doc.resolved_citations = [
                self.find_document_by_title_or_url(c) or c
                for c in doc.citations
            ]
        else:
            doc.resolved_citations = list(doc.citations)
        ops: list[dict] = []
        if old_doc is not None:
            ops += self._document_delete_content_ops(
                doc.id, [s.id for s in old_doc.sections])
            ops += self._document_graph_ops(old_doc, delete=True)
            # Unindex the OLD section vectors (per-chunk vector indexing): the
            # old sections are being removed from the hot store + graph, so
            # their vector-layer entries must go too. Best-effort + idempotent
            # (``delete_sync`` no-ops on an absent id, e.g. a structure-only old
            # section that was never indexed). Unchanged sections are re-indexed
            # below with the same vector (idempotent overwrite) -- wasteful but
            # correct for the first slice; a per-section embedding delta is
            # deferred (mirrors the blob-refcount delta already done for bodies).
            for s in old_doc.sections:
                self._unindex_embedding(s.id)
        ops += self._document_metadata_ops(doc)
        ops += self._document_graph_ops(doc, delete=False)
        self.db.batch_sync(ops)
        # 4. Index each NEW section that has an embedding -- one vector per
        #    chunk (the per-chunk doc-RAG semantic path). Sections with no
        #    embedding (structure-only ingest, no embedder) are simply not
        #    indexed; they stay findable via the graph entity/topic axes but not
        #    via the semantic fallback. Best-effort + never raises.
        for s in doc.sections:
            if s.embedding is not None:
                self._index_embedding(s.id, s.embedding, s.heading)

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
        graph layer. Returns ``None`` for a missing doc (no ``source_type`` key
        -- the existence sentinel; ``title`` can legitimately be empty for a
        heading-less source, so it is NOT the sentinel).
        """
        # Existence sentinel: source_type is ALWAYS written at encode (defaults
        # to "text") and never empty for a real doc. title can be empty (a
        # markdown doc with only ``##`` headings, or a plain-text file), so
        # gating on title would make such docs unreadable -- and break the
        # encode_document UPDATE path (old_doc would load as None -> old section
        # vectors orphaned). source_type is the safe marker.
        source_type = _b2s(self.db.get_sync(f"content/doc/{doc_id}/source_type"))
        if not source_type:
            return None
        title = _b2s(self.db.get_sync(f"content/doc/{doc_id}/title"))
        source_path = _b2s(self.db.get_sync(f"content/doc/{doc_id}/source_path"))
        # Phase 3c Sec 7.11: semantic doc-kind tag ("" for a pre-7.11 doc ->
        # coerced to the "other" default by the ctor below, back-compatible).
        doc_kind = _b2s(self.db.get_sync(f"content/doc/{doc_id}/doc_kind"))
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
        resolved_citations_raw = _b2s(self.db.get_sync(
            f"content/doc/{doc_id}/resolved_citations"))
        state_assertions_raw = _b2s(self.db.get_sync(
            f"content/doc/{doc_id}/state_assertions"))
        # User scope (retrieval boundary): absent key -> ``None`` (byte-identical
        # for a pre-user-scope / unscoped doc).
        user_id = _b2s(self.db.get_sync(f"content/doc/{doc_id}/user")) or None

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
            # STRM 1f-6: prose summary (embedding handle). Absent key -> None
            # (byte-identical to a pre-1f-6 / no-summarizer store).
            summary = _b2s(self.db.get_sync(f"{s}/summary")) or None
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
                summary=summary,
            ))

        return Document(
            id=doc_id,
            source_type=source_type,
            source_path=source_path,
            title=title,
            ingested_at=ingested_at,
            doc_kind=doc_kind or "other",
            sections=sections,
            authors=_loads(authors_raw, []),
            created_at=created_at,
            language=language,
            metadata=_loads(metadata_raw, {}),
            entities=_loads(entities_raw, []),
            topics=_loads(topics_raw, []),
            relations=_loads(relations_raw, []),
            citations=_loads(citations_raw, []),
            resolved_citations=_loads(resolved_citations_raw, []),
            state_assertions=_loads(state_assertions_raw, []),
            user_id=user_id,
        )

    def get_section_body(self, doc_id: str, section_id: str) -> Optional[str]:
        """Materialize ONE section's body from the cold store (single-section pull).

        Reads the hot ``content/doc/{doc_id}/sec/{section_id}/blob_hash`` ref
        (no cold pull) and, when present, pulls that one blob from the cold
        BlobStore. This is the retrieval-path cold read: ``_hydrate_document``
        calls it at most once per doc result to fill ``text`` with the matched
        section's body. Cold-read only -- it never writes the hot store, so a
        retrieved body does NOT enter the memory LRU as a permanent resident
        (the blob store has its own LRU; this just reads through it). Returns
        ``None`` when the section or its blob is absent (the caller falls back
        to an empty ``text``).
        """
        blob_hash = _b2s(self.db.get_sync(
            f"content/doc/{doc_id}/sec/{section_id}/blob_hash"))
        if not blob_hash:
            return None
        return self._blob_store().get_blob(blob_hash)

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
        # Unindex each section's per-chunk vector (best-effort; ``delete_sync``
        # no-ops on an absent id, e.g. a structure-only section never indexed).
        # Done AFTER the memory batch so the doc is already unfindable if the
        # vector delete races a concurrent search.
        for sec in doc.sections:
            self._unindex_embedding(sec.id)
        return True

    def gc_blobs(self) -> int:
        """Sweep zero-refcount orphan blobs from the cold store. Returns count."""
        return self._blob_store().gc_zero_ref()

    # ---- lifecycle ----

    def close(self) -> None:
        # Close children before the db: the binding's WaveDB.close() refuses
        # to destroy the database while child handles are open. The graph
        # layer is always open; the blob store is opened lazily and may be
        # None for episode-only workloads; the vector layer (VectorLayer.open
        # shares self.db's database_t) is opened in __init__ when enabled.
        try:
            if self.vector_layer is not None:
                self.vector_layer.close()
            self.graph.close()
        finally:
            if self._blob is not None:
                try:
                    self._blob.close()
                finally:
                    self.db.close()
            else:
                self.db.close()