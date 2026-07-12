"""Phase 1b graph traversal — pattern completion over the hippocampal index.

Backed by ``store.graph`` (a WaveDB ``GraphLayer`` over the ``memory/`` subtree)
and ``store.db`` (the parent ``WaveDB``). Adapts the ``docs/Phase 1b.md`` design
to the REAL WaveDB Python graph API (the doc's assumed API is wrong in two
places — noted inline):

- ``GraphQuery.execute_sync()`` returns a ``GraphResult`` whose ``.vertices`` is a
  flat ``list[str]`` of vertex ids. The doc's ``r.subject`` / ``r.id`` row model
  does not exist.
- ``.has(predicate, value)`` requires a concrete value, so "all episodes with
  predicate P regardless of value" is NOT expressible via the builder. We use
  ``db.create_read_stream`` over the content namespace instead (see
  ``_get_all_episode_ids``).

Predicates are snake_case (``has_entity``, ``has_topic``, ``has_tone``,
``has_decision``, ``in_episode``, ``in_session``, ``follows``, ``at_time``) —
the ontology registry's camelCase names never appear in the SPO index. Node id
conventions: ``E:{entity}``, ``T:{topic}``, ``A:{tone}``, ``D:{decision}``,
``ep_000001``, ``S:0001``, ``U:{user}``.

Candidate-set union/intersection across the entity/topic/tone axes is computed
Python-side — WaveDB's set ops exist in C but are not bound in Python, and this
slice does not need them.

Scoring (entity×10, topic×5, tone×3, recency×0.1) follows ``docs/Phase 1b.md``
and is a placeholder ranker; the Phase 3 GNN consolidator replaces it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..config import config as _config
from ..memory.store import HippocampalStore

# Phase 3b retrieval-time boost (lazy import inside the hook to keep the
# module-import graph acyclic and avoid loading forgetting machinery on the
# cold path where forgetting_enabled is False).
_LOG = logging.getLogger(__name__)

# Axis weights from docs/Phase 1b.md §query-scoring. Recency is applied to the
# candidate's ordinal rank within the result set (newest = highest), a stand-in
# for the encoded timestamp until Phase 1c's temporal index.
_W_ENTITY = 10.0
_W_TOPIC = 5.0
_W_TONE = 3.0
_W_RECENCY = 0.1

_MAX_FOLLOWS_HOPS = 5

# Temporal bucket windows, anchored to "now". ``last_week`` is the half-open
# interval [now-14d, now-7d) — i.e. the week before this week. (The doc's
# last_week logic was inverted and could never match; this is the fix.)
_TEMPORAL_BUCKETS_DAYS: dict[str, tuple[float, Optional[float]]] = {
    "today": (1.0, None),
    "this_week": (7.0, None),
    "last_week": (14.0, 7.0),
    "this_month": (30.0, None),
}


def _strip_prefix(vid: str, prefix: str) -> str:
    """Strip a node-id prefix (``E:``/``T:``/``A:``/``D:``/``U:``); return vid unchanged if absent."""
    return vid[len(prefix):] if vid.startswith(prefix) else vid


class GraphTraversal:
    """Pattern-completion retrieval over the hippocampal graph index."""

    def __init__(self, store: HippocampalStore) -> None:
        self.store = store
        self.graph = store.graph
        self.db = store.db

    # ── primitive graph queries (one axis each) ──

    def _exec_vertices(self, query) -> list[str]:
        """Run a one-shot ``GraphQuery`` and collect its vertices.

        ``execute_sync`` consumes the query handle (it is NULL afterwards). The
        returned ``GraphResult`` wraps a heap-allocated C ``graph_result_t``, so
        we close it explicitly in a ``finally`` rather than relying on ``__del__``
        GC across the many queries a single ``retrieve`` call can issue.
        """
        result = query.execute_sync()
        try:
            return list(result.vertices)
        finally:
            result.close()

    def _filter_current_edges(
        self, episode_ids: list[str], predicate: str, obj: str
    ) -> list[str]:
        """Edge-level forgetting filter (Phase 3b).

        Drop episodes whose ``(ep, predicate, obj)`` association has been
        deprecated/superseded/archived in its per-edge sidecar. This is the
        EDGE granularity (vs the episode-level filter in
        ``store.default_episode_ids``): a single stale ``(ep, has_entity,
        E:Alice)`` edge excludes the episode from an Alice query but NOT from a
        Bob query -- the episode stays live for its other associations. The
        episode's graph triple is NOT deleted (deprecate, don't delete); only
        the sidecar state marks it stale.

        Gated on ``config.forgetting_enabled`` (default True). A missing sidecar
        (the common cold-start case -- no edge has been touched by forgetting)
        means ``current``, so the filter is a no-op until something is actually
        deprecated. One ``get_sync`` point lookup per candidate edge.
        """
        if not _config.forgetting_enabled:
            return episode_ids
        return [
            eid for eid in episode_ids
            if self.store.is_edge_current(eid, predicate, obj)
        ]

    def _get_episodes_by_entity(self, entity: str) -> list[str]:
        """Episodes that mention ``entity`` — ``E:{entity} --out:in_episode--> ep``.

        The store writes the reverse edge ``(E:{entity}, in_episode, eid)`` for
        entities (topic/tone/decision only write the forward ``(eid, has_*,
        leaf)`` edge), so the lookup follows ``in_episode`` OUT of ``E:{entity}``,
        not into it. The Phase 3b edge-level filter then drops episodes whose
        ``(ep, has_entity, E:{entity})`` association is deprecated/superseded.
        """
        q = self.graph.query().vertex(f"E:{entity}").out("in_episode")
        eps = self._exec_vertices(q)
        return self._filter_current_edges(eps, "has_entity", f"E:{entity}")

    def _get_episodes_by_topic(self, topic: str) -> list[str]:
        q = self.graph.query().vertex(f"T:{topic}").in_("has_topic")
        eps = self._exec_vertices(q)
        return self._filter_current_edges(eps, "has_topic", f"T:{topic}")

    def _get_episodes_by_tone(self, tone: str) -> list[str]:
        q = self.graph.query().vertex(f"A:{tone}").in_("has_tone")
        eps = self._exec_vertices(q)
        return self._filter_current_edges(eps, "has_tone", f"A:{tone}")

    def _get_all_episode_ids(self) -> list[str]:
        """All episode ids — scan the content namespace (complete).

        The doc's POS-scan over ``has_entity`` only yields entity-bearing
        episodes, which would drop entity-less episodes from the candidate seed
        for a tone/topic-only query. The content scan is complete: every episode
        has a ``content/ep/{id}/summary`` key, so the unique ``content/ep/{id}/``
        prefixes enumerate all episodes regardless of their graph triples.

        Delegates to ``store.default_episode_ids`` so abstracted episodes
        (Phase 3a semantic memories) are excluded from the default retrieval
        candidate set (spec §371).
        """
        return self.store.default_episode_ids(include_abstracted=False)

    # ── candidate-set construction (Python-side set ops) ──

    def _find_candidates(
        self,
        entities: list[str],
        topics: list[str],
        tones: list[str],
        entity_mode: str = "union",
    ) -> set[str]:
        """Build the candidate episode set from the query axes.

        Starts from the first specified axis's match set, then intersects each
        further non-empty axis set (unioned across that axis's values). If an
        axis is specified but matches no episodes, the query is over-constrained
        and an empty set is returned. With no axis specified, every episode is a
        candidate (filtered later by temporal constraints / scoring).
        """
        candidates: Optional[set[str]] = None  # None until the first axis is applied

        if entities:
            if entity_mode == "intersection":
                ent_sets = [set(self._get_episodes_by_entity(e)) for e in entities]
                entity_eps: set[str] = set.intersection(*ent_sets) if ent_sets else set()
            else:  # "union" (default)
                entity_eps = set()
                for e in entities:
                    entity_eps |= set(self._get_episodes_by_entity(e))
            if not entity_eps:
                return set()
            candidates = entity_eps.copy()

        if topics:
            topic_eps: set[str] = set()
            for t in topics:
                topic_eps |= set(self._get_episodes_by_topic(t))
            if not topic_eps:
                return set()
            candidates = topic_eps if candidates is None else (candidates & topic_eps)

        if tones:
            tone_eps: set[str] = set()
            for a in tones:
                tone_eps |= set(self._get_episodes_by_tone(a))
            if not tone_eps:
                return set()
            candidates = tone_eps if candidates is None else (candidates & tone_eps)

        if candidates is None:
            # No axis specified — start from all episodes.
            return set(self._get_all_episode_ids())
        # Phase 3b: the no-axis path goes through ``_get_all_episode_ids`` ->
        # ``default_episode_ids`` which already excludes deprecated/superseded
        # episodes. The axis path builds candidates from graph lookups, which
        # bypass that filter -- so drop episode-level-deprecated episodes here.
        # (Edge-level deprecation is already handled per-axis in
        # ``_get_episodes_by_*``; this is the episode-level pass.)
        return self._filter_active_episodes(candidates)

    def _filter_active_episodes(self, episode_ids: set[str]) -> set[str]:
        """Episode-level forgetting filter on an axis-derived candidate set.

        Drops episodes whose lifecycle state is ``deprecated``/``superseded`` or
        whose ``validity_end`` is set. Gated on ``config.forgetting_enabled``;
        a no-op until something is actually deprecated.
        """
        if not _config.forgetting_enabled:
            return episode_ids
        return {eid for eid in episode_ids if self.store.is_episode_active(eid)}

    # ── follows chain (temporal order) ──

    def _follows_neighbors(self, episode_id: str, direction: str) -> list[str]:
        """One hop along the ``follows`` chain.

        ``follows`` points from an episode to the one BEFORE it (``ep2 follows
        ep1`` means ep2 is the later one). So:
        - forward in time (episodes AFTER ``episode_id``) = ``in_("follows")``
          (episodes whose ``follows`` points AT this one).
        - backward in time (the episode this one follows) = ``out("follows")``.
        """
        q = self.graph.query().vertex(episode_id)
        if direction == "forward":
            q = q.in_("follows")
        else:
            q = q.out("follows")
        return self._exec_vertices(q)

    def _follow_chain(self, episode_id: str, direction: str, max_hops: int = _MAX_FOLLOWS_HOPS) -> list[str]:
        """BFS along ``follows`` up to ``max_hops``.

        Returns the chain INCLUDING ``episode_id``. Each hop can fan out (an
        episode may be followed by several later episodes), so this is a BFS, not
        a single-path walk.
        """
        chain: list[str] = [episode_id]
        frontier: list[str] = [episode_id]
        visited: set[str] = {episode_id}
        for _ in range(max_hops):
            next_frontier: list[str] = []
            for ep in frontier:
                for nb in self._follows_neighbors(ep, direction):
                    if nb and nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
                        chain.append(nb)
            if not next_frontier:
                break
            frontier = next_frontier
        return chain

    def _find_anchor(self, candidates: list[str], keyword: str) -> Optional[str]:
        """First candidate whose summary contains ``keyword`` (case-insensitive)."""
        kw = keyword.lower()
        for eid in candidates:
            ep = self.store.get_episode(eid)
            if ep and kw in (ep.summary or "").lower():
                return eid
        return None

    # ── temporal filtering ──

    def _filter_temporal(self, candidates: set[str], bucket: str) -> set[str]:
        """Keep candidates whose timestamp falls in ``bucket``.

        Buckets are anchored to ``datetime.now()``. ``last_week`` is the half-open
        interval [now-14d, now-7d). Unknown buckets apply no filter.
        """
        if bucket not in _TEMPORAL_BUCKETS_DAYS:
            return candidates
        back_days, end_days = _TEMPORAL_BUCKETS_DAYS[bucket]
        now = datetime.now()
        start = now - timedelta(days=back_days)
        end = now - timedelta(days=end_days) if end_days is not None else None

        out: set[str] = set()
        for eid in candidates:
            ep = self.store.get_episode(eid)
            if not ep or not ep.timestamp:
                continue
            try:
                ts = datetime.fromisoformat(ep.timestamp)
            except ValueError:
                continue
            if ts >= start and (end is None or ts < end):
                out.add(eid)
        return out

    def _filter_date_range(
        self,
        candidates: set[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> set[str]:
        """Keep candidates whose timestamp falls in ``[date_from, date_to]``.

        ``date_from`` / ``date_to`` are ISO date or datetime strings; either may
        be ``None`` (one-sided range). A bare date (``"2025-06-01"``) parses at
        midnight. Inclusive on both ends. O(candidates) with one ``get_episode``
        per candidate — same cost shape as ``_filter_temporal``. This is the
        Phase 1c long-range path: O(candidates) instead of walking a ``follows``
        chain capped at ``_MAX_FOLLOWS_HOPS``.
        """
        lo = self._parse_dt(date_from) if date_from else None
        hi = self._parse_dt(date_to) if date_to else None
        out: set[str] = set()
        for eid in candidates:
            ep = self.store.get_episode(eid)
            if not ep or not ep.timestamp:
                continue
            ts = self._parse_dt(ep.timestamp)
            if ts is None:
                continue
            if lo is not None and ts < lo:
                continue
            if hi is not None and ts > hi:
                continue
            out.add(eid)
        return out

    @staticmethod
    def _parse_dt(s: str) -> Optional[datetime]:
        """Parse an ISO date/datetime string; return ``None`` on failure."""
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    # ── hydration (scope rehydration deferred from Phase 1a) ──

    def _user_for_session(self, session_id: str) -> Optional[str]:
        """Resolve ``U:{user}`` from ``S:{session}`` via the POS index of has_session.

        ``(U:user, has_session, S:sess)`` is stored as POS
        ``memory/pos/has_session/{sess}/{user}``, so scanning that prefix yields
        the user (the subject) as the last component.
        """
        start = f"memory/pos/has_session/{session_id}/"
        end = f"memory/pos/has_session/{session_id}/\x7f"
        for k, _ in self.db.create_read_stream(start=start, end=end):
            return _strip_prefix(k.rsplit("/", 1)[-1], "U:")
        return None

    def hydrate_episode(self, eid: str) -> dict:
        """Public alias for ``_hydrate`` — load content + graph-side fields.

        Exposed for Phase 1d training-data generators (and other read-only
        consumers) that need a hydrated episode dict without running a full
        ``retrieve`` query plan. Returns the same shape as ``_hydrate``.
        """
        return self._hydrate(eid)

    def episodes_by_entity(self, entity: str) -> list[str]:
        """Public alias for ``_get_episodes_by_entity`` (read-only consumers)."""
        return self._get_episodes_by_entity(entity)

    def episodes_by_topic(self, topic: str) -> list[str]:
        """Public alias for ``_get_episodes_by_topic`` (read-only consumers)."""
        return self._get_episodes_by_topic(topic)

    def _hydrate(self, eid: str) -> dict:
        """Load content + graph-side fields for one episode into a result dict.

        Graph fields come from a SINGLE scan over ``memory/spo/{eid}/`` (all
        outgoing edges), bucketed by predicate — one scan per episode instead of
        one per axis. The user is one extra POS scan (reverse of ``has_session``)
        only when the episode is session-scoped.
        """
        ep = self.store.get_episode(eid)
        result: dict = {
            "episode_id": eid,
            "summary": ep.summary if ep else "",
            "text": ep.full_text if ep else "",
            "timestamp": ep.timestamp if ep else "",
            "entities": [],
            "topics": [],
            "tones": [],
            "decisions": [],
            "session_id": None,
            "user_id": None,
            "follows": None,
            "score": 0.0,
        }
        if not ep:
            return result

        start = f"memory/spo/{eid}/"
        end = f"memory/spo/{eid}/\x7f"
        for k, _ in self.db.create_read_stream(start=start, end=end):
            # k = memory/spo/{eid}/{predicate}/{object}
            parts = k.split("/", 4)
            if len(parts) < 5:
                continue
            predicate, obj = parts[3], parts[4]
            if predicate == "has_entity":
                result["entities"].append(_strip_prefix(obj, "E:"))
            elif predicate == "has_topic":
                result["topics"].append(_strip_prefix(obj, "T:"))
            elif predicate == "has_tone":
                result["tones"].append(_strip_prefix(obj, "A:"))
            elif predicate == "has_decision":
                result["decisions"].append(_strip_prefix(obj, "D:"))
            elif predicate == "in_session":
                result["session_id"] = obj
            elif predicate == "follows":
                result["follows"] = obj
            # state / at_time / validity_start are not retrieval-facing.

        if result["session_id"]:
            result["user_id"] = self._user_for_session(result["session_id"])
        return result

    # ── scoring ──

    def _score_candidates(
        self,
        hydrated: list[dict],
        entities: list[str],
        topics: list[str],
        tones: list[str],
    ) -> list[dict]:
        """Heuristic ranker: axis-match counts × weights + recency ordinal.

        Entity matches are weighted by per-entity salience
        (``store.get_entity_salience``): each match contributes
        ``_W_ENTITY * (0.5 + 0.5 * salience)`` ∈ [_W_ENTITY/2, _W_ENTITY], so a
        high-salience entity match is worth up to 2× a low-salience one. Phase 3
        GNN salience replaces this.

        Recency is the candidate's rank by timestamp within the result set
        (newest = highest rank), so the recency term is a small tiebreaker
        bounded by ``_W_RECENCY * len(candidates)`` — never enough to override an
        axis match.
        """
        ent_set = {e.lower() for e in entities}
        top_set = {t.lower() for t in topics}
        ton_set = {a.lower() for a in tones}

        times: list[tuple[str, datetime]] = []
        for r in hydrated:
            try:
                times.append((r["episode_id"], datetime.fromisoformat(r["timestamp"])))
            except (ValueError, TypeError):
                times.append((r["episode_id"], datetime.min))
        times.sort(key=lambda x: x[1])
        recency_rank = {eid: i for i, (eid, _) in enumerate(times)}

        for r in hydrated:
            score = 0.0
            # Entity matches — salience-weighted (Phase 1c). A high-salience
            # entity match is worth up to 2× a low-salience one; an unknown
            # entity (salience 0.0) still scores _W_ENTITY/2 so a query entity
            # not present in the salience index isn't zeroed out. Phase 3 GNN
            # salience replaces this heuristic. Dedup by lowercased key so a
            # duplicate in r["entities"] (a corrupt stream entry) can't
            # double-count; salience is looked up with the original-case entity
            # because the salience key is case-sensitive.
            matched_entities = {e.lower() for e in r["entities"]} & ent_set
            seen: set[str] = set()
            for e in r["entities"]:
                key = e.lower()
                if key in matched_entities and key not in seen:
                    seen.add(key)
                    salience = self.store.get_entity_salience(e)
                    score += _W_ENTITY * (0.5 + 0.5 * salience)
            score += _W_TOPIC * len({t.lower() for t in r["topics"]} & top_set)
            score += _W_TONE * len({a.lower() for a in r["tones"]} & ton_set)
            score += _W_RECENCY * recency_rank[r["episode_id"]]
            r["score"] = score
        return hydrated

    # ── public entry point ──

    def retrieve(self, query_plan: dict, limit: Optional[int] = None, signal: Optional[str] = None) -> list[dict]:
        """Run a query plan against the graph and return ranked episode dicts.

        The plan is the output of the query planner (or a literal dict in tests):
        ``entities`` / ``topics`` / ``tones`` (lists), ``entity_mode``
        (``"union"`` | ``"intersection"``), ``temporal_filter`` (a bucket name),
        ``temporal_after`` / ``temporal_before`` (keywords anchoring a ``follows``
        chain walk), and ``limit``.

        Each result dict carries content (summary/text/timestamp) plus the
        graph-side fields hydrated from the index (entities/topics/tones/
        decisions/session_id/user_id/follows) and a ``score``.

        ``signal`` (Phase 3b) is the LLM-mediated importance signal threaded from
        the orchestrator (``important``/``routine``/``satisfied``/``frustration``
        /``correction``) used by the retrieval-time boost hook. Defaults to the
        plan's ``signal`` field, then ``"routine"``. The boost is NON-BLOCKING:
        a sidecar write failure is logged and never breaks retrieval.
        """
        entities = query_plan.get("entities") or []
        topics = query_plan.get("topics") or []
        tones = query_plan.get("tones") or []
        entity_mode = query_plan.get("entity_mode", "union")
        if signal is None:
            signal = query_plan.get("signal") or "routine"
        temporal_filter = query_plan.get("temporal_filter")
        temporal_after = query_plan.get("temporal_after")
        temporal_before = query_plan.get("temporal_before")
        date_from = query_plan.get("date_from")
        date_to = query_plan.get("date_to")
        if limit is None:
            limit = query_plan.get("limit") or _config.default_retrieval_limit

        candidates = self._find_candidates(entities, topics, tones, entity_mode)
        if not candidates:
            return []

        # Temporal chain mode: re-anchor the candidate set to a follows-chain
        # rooted at the keyword-matching episode. If no anchor matches, fall
        # through with the axis-derived candidates.
        if temporal_after:
            anchor = self._find_anchor(list(candidates), temporal_after)
            if anchor:
                candidates = set(self._follow_chain(anchor, "forward"))
        elif temporal_before:
            anchor = self._find_anchor(list(candidates), temporal_before)
            if anchor:
                candidates = set(self._follow_chain(anchor, "backward"))

        # Absolute date-range filter (Phase 1c). Mutually exclusive with the
        # relative temporal_filter bucket in the planner; the code tolerates both
        # (range applied first, then bucket) but the planner should not emit both.
        if date_from or date_to:
            candidates = self._filter_date_range(candidates, date_from, date_to)
            if not candidates:
                return []

        if temporal_filter:
            candidates = self._filter_temporal(candidates, temporal_filter)
            if not candidates:
                return []

        hydrated = [self._hydrate(eid) for eid in candidates]
        scored = self._score_candidates(hydrated, entities, topics, tones)
        scored.sort(key=lambda r: r["score"], reverse=True)
        results = scored[:limit]

        # Phase 3b: strengthen the edges this retrieval actually matched. The
        # data is cleanest here -- results are scored, limited, and hydrated with
        # the matched entities/topics/tones. Fire AFTER results are in hand so a
        # sidecar write can never delay or break the answer. Non-blocking.
        self._apply_retrieval_boost(results, entities, topics, tones, signal)
        return results

    def _apply_retrieval_boost(
        self,
        results: list[dict],
        query_entities: list[str],
        query_topics: list[str],
        query_tones: list[str],
        signal: str,
    ) -> None:
        """Retrieval-time forgetting boost (Phase 3b, hot path).

        For each returned result, find the ``(ep, predicate, obj)`` edges that
        matched this query (entity/topic/tone axes), and apply
        ``forgetting.apply_retrieval_boost`` to each edge's sidecar -- the
        ``on_retrieve`` strengthening that makes memories persist with use. All
        touched edges land in ONE ``batch_sync`` (one RMW read pass, one write
        batch). Non-blocking: any failure is logged and swallowed so retrieval
        never breaks on a sidecar write.

        Gated on ``config.forgetting_enabled``. ``access_count++`` is a
        read-modify-write on the sidecar; under concurrent queries increments
        can be lost (the same single-user assumption 2c makes -- documented in
        ``docs/Phase 3b.md``).
        """
        if not _config.forgetting_enabled:
            return
        if not results:
            return
        # Lazy import keeps the forgetting machinery off the cold path and the
        # module-import graph acyclic.
        from ..memory.edge_meta import batch_update_edge_meta, get_edge_meta
        from ..memory.forgetting import apply_retrieval_boost

        ent_set = {e.lower() for e in query_entities}
        top_set = {t.lower() for t in query_topics}
        ton_set = {a.lower() for a in query_tones}
        if not (ent_set or top_set or ton_set):
            return  # no-axis query: nothing matched an edge to boost

        now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        updates: list[tuple[str, str, str, dict]] = []
        for r in results:
            eid = r["episode_id"]
            # Documents are exempt from the forgetting system: skip doc-owned
            # results so no retrieval-boost sidecar is ever written for a
            # document edge. In Phase 1 ``retrieve()`` returns episodes only
            # (docs are not yet wired through retrieval), so this is defensive
            # for the Phase-2 integration that will mix docs into the results.
            if eid.startswith("doc_"):
                continue
            # Hydration strips the E:/T:/A: prefix (graph_traversal._hydrate);
            # re-prepend it to recover the graph-triple object the sidecar keys on.
            for e in r["entities"]:
                if e.lower() in ent_set:
                    obj = f"E:{e}"
                    meta = get_edge_meta(self.store, eid, "has_entity", obj)
                    updates.append((
                        eid, "has_entity", obj,
                        apply_retrieval_boost(meta, llm_signal=signal, now_ts=now_ts),
                    ))
            for t in r["topics"]:
                if t.lower() in top_set:
                    obj = f"T:{t}"
                    meta = get_edge_meta(self.store, eid, "has_topic", obj)
                    updates.append((
                        eid, "has_topic", obj,
                        apply_retrieval_boost(meta, llm_signal=signal, now_ts=now_ts),
                    ))
            for a in r["tones"]:
                if a.lower() in ton_set:
                    obj = f"A:{a}"
                    meta = get_edge_meta(self.store, eid, "has_tone", obj)
                    updates.append((
                        eid, "has_tone", obj,
                        apply_retrieval_boost(meta, llm_signal=signal, now_ts=now_ts),
                    ))
        if not updates:
            return
        try:
            batch_update_edge_meta(self.store, updates)
        except Exception as exc:  # non-blocking: never break retrieval
            _LOG.warning("retrieval boost write failed (%d edges): %s", len(updates), exc)