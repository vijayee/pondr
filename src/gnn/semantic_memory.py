"""Semantic-memory storage for the Phase 3a consolidation loop (Task 5).

The consolidation loop's DiffPool head clusters related episodes and abstracts
them into a **semantic memory** — a single node that ``abstracts`` its source
episodes and carries their gist. This module writes/reads those memories and
the supporting bookkeeping:

- **``abstracts`` edges** — ``(M:NNNN, abstracts, ep_000001)`` per source episode.
  ``M:`` is a new node-kind prefix (semantic Memory), consistent with the graph's
  id-prefix-typing convention (``E:``/``T:``/``A:``/``D:``/``S:``/``U:``/``ep_``).
- **``supersedes`` edges** — ``(M:new, supersedes, M:old)`` when a fresh
  abstraction replaces a stale one (the predicate was declared in
  ``ontology.py`` but never written until now).
- **abstracted flag** — ``content/ep/{eid}/abstracted = 1`` on each source
  episode. Abstracted episodes stay retrievable (their content is untouched) but
  are EXCLUDED from default queries (spec §371) — the store's
  ``default_episode_ids(include_abstracted=False)`` filters them.
- **``consolidation_window_start``** — set on each source episode (the field
  existed at ``Episode.consolidation_window_start`` but was never written).
- **archive subtree** — pruned low-salience edges are COPIED to ``archive/edge/...``
  with a reason + timestamp, then removed from the live graph. Archive is
  recoverable and never deleted (spec §371).

All writes go through ONE ``store.db.batch_sync`` so an abstraction is atomic:
the M node, its abstracts edges, and the source episodes' abstracted flags
either all land or none do.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..memory.store import HippocampalStore

# ``_b2s`` decodes WaveDB bytes→str ('' for missing). It's a module-level helper
# in ``store.py`` (not a method), so import it rather than reaching through the
# store instance. ``safe_edge_component`` hashes ``/``-bearing key components
# (shared with ``memory.edge_meta``'s sidecar key builder).
from ..memory.store import _b2s, safe_edge_component


def _utc_now() -> str:
    """ISO-8601 UTC timestamp. (Module-level helper so tests can monkeypatch.)"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SemanticMemoryWriter:
    """Writes semantic memories + supporting edges/flags to the store.

    Stateless beyond the store handle. The consolidation loop (Task 6) calls
    ``create_abstract`` for each DiffPool cluster it accepts; ``archive_edge``
    for each low-salience edge it prunes.
    """

    def __init__(self, store: "HippocampalStore") -> None:
        self.store = store

    # ── semantic memories ──

    def create_abstract(
        self,
        source_episode_ids: list[str],
        summary: str,
        *,
        text: str = "",
        embedding: Optional[list[float]] = None,
        supersedes: Optional[str] = None,
        when: Optional[str] = None,
    ) -> str:
        """Create one semantic memory abstracting ``source_episode_ids``.

        Writes atomically: a new ``M:NNNN`` node (content + abstracts edges),
        marks each source episode abstracted, sets its
        ``consolidation_window_start``, and optionally writes a ``supersedes``
        edge to a prior abstract. Returns the new memory id (``M:NNNN``).
        """
        if not source_episode_ids:
            raise ValueError("create_abstract requires at least one source episode")
        if not summary:
            raise ValueError("create_abstract requires a non-empty summary")
        ts = when or _utc_now()
        mid = self.store.next_memory_id()

        ops: list[dict] = []
        # M-node content (mirrors episode content layout under content/mem/).
        ops.append({"type": "put", "key": f"content/mem/{mid}/summary", "value": summary})
        ops.append({"type": "put", "key": f"content/mem/{mid}/text", "value": text})
        ops.append({"type": "put", "key": f"content/mem/{mid}/ts", "value": ts})
        ops.append({"type": "put", "key": f"content/mem/{mid}/abstracted_from",
                     "value": json.dumps(source_episode_ids)})
        if embedding:
            ops.append({"type": "put", "key": f"content/mem/{mid}/embedding",
                         "value": json.dumps(embedding)})

        # abstracts edges + source-episode bookkeeping.
        for eid in source_episode_ids:
            ops += self.store.graph.expand_triple(mid, "abstracts", eid)
            ops.append({"type": "put", "key": f"content/ep/{eid}/abstracted", "value": "1"})
            ops.append({"type": "put",
                        "key": f"content/ep/{eid}/consolidation_window_start", "value": ts})

        if supersedes:
            ops += self.store.graph.expand_triple(mid, "supersedes", supersedes)

        self.store.db.batch_sync(ops)
        return mid

    def supersede(self, new_memory_id: str, old_memory_id: str) -> None:
        """Record that ``new_memory_id`` supersedes ``old_memory_id`` (M->M)."""
        ops = self.store.graph.expand_triple(new_memory_id, "supersedes", old_memory_id)
        self.store.db.batch_sync(ops)

    def supersede_episode(
        self,
        new_episode_id: str,
        old_episode_id: str,
        *,
        when: Optional[str] = None,
    ) -> None:
        """Record that episode ``new`` supersedes episode ``old`` (E->E, Phase 3b).

        Writes the MVCC supersession chain atomically in ONE ``batch_sync``:

        * ``(new, supersedes, old)`` graph edge (forward chain link).
        * ``(old, superseded_by, new)`` graph edge (back-pointer for queries
          that want "what replaced this?"). ``superseded_by`` is added to
          ``CONVERSATIONAL_PROPERTIES`` in the same change.
        * ``content/ep/{old}/state = "superseded"`` +
          ``content/ep/{old}/validity_end = <ts>`` (the old episode stops
          appearing in default queries; it is NOT deleted).

        The new episode's own state/validity is left untouched (it stays
        ``current``). This only records the relationship; it does not create
        the new episode (the caller encodes that separately).
        """
        ts = when or _utc_now()
        ops: list[dict] = []
        ops += self.store.graph.expand_triple(new_episode_id, "supersedes", old_episode_id)
        ops += self.store.graph.expand_triple(old_episode_id, "superseded_by", new_episode_id)
        ops.append({"type": "put", "key": f"content/ep/{old_episode_id}/state",
                     "value": "superseded"})
        ops.append({"type": "put", "key": f"content/ep/{old_episode_id}/validity_end",
                     "value": ts})
        self.store.db.batch_sync(ops)

    def get_abstract(self, memory_id: str) -> Optional[dict]:
        """Read a semantic memory back. ``None`` if it doesn't exist."""
        summary = _b2s(self.store.db.get_sync(f"content/mem/{memory_id}/summary"))
        if not summary:
            return None
        text = _b2s(self.store.db.get_sync(f"content/mem/{memory_id}/text")) or ""
        ts = _b2s(self.store.db.get_sync(f"content/mem/{memory_id}/ts")) or ""
        sources_raw = _b2s(self.store.db.get_sync(f"content/mem/{memory_id}/abstracted_from"))
        try:
            sources = json.loads(sources_raw) if sources_raw else []
        except (ValueError, TypeError):
            sources = []
        return {"id": memory_id, "summary": summary, "text": text, "ts": ts, "sources": sources}

    def abstracted_episodes(self, memory_id: str) -> list[str]:
        """The source episodes a semantic memory abstracts (via the graph)."""
        out: list[str] = []
        q = self.store.graph.query().vertex(memory_id).out("abstracts")
        result = q.execute_sync()
        try:
            out = list(result.vertices)
        finally:
            result.close()
        return out

    # ── edge archive (prune, never delete) ──

    def archive_edge(
        self,
        subject: str,
        predicate: str,
        object: str,
        *,
        reason: str = "",
        when: Optional[str] = None,
        remove_from_graph: bool = True,
    ) -> str:
        """Copy a triple to the ``archive/`` subtree, then remove it from the live graph.

        Archive is recoverable and never deleted (spec §371). The archived
        record is a JSON value at ``archive/edge/{s}/{p}/{o}`` (falling back to
        a hashed key if any component contains ``/``). When
        ``remove_from_graph`` is True (default), the live triple is deleted in
        the SAME atomic batch — so the edge is either archived-and-removed or
        untouched.
        """
        ts = when or _utc_now()
        record = json.dumps({
            "subject": subject, "predicate": predicate, "object": object,
            "reason": reason, "archived_at": ts,
        }, ensure_ascii=False)
        archive_key = self._archive_key(subject, predicate, object)
        ops: list[dict] = [
            {"type": "put", "key": archive_key, "value": record},
        ]
        if remove_from_graph:
            ops += self.store.graph.expand_triple(subject, predicate, object, delete=True)
        self.store.db.batch_sync(ops)
        return archive_key

    def read_archived_edge(self, archive_key: str) -> Optional[dict]:
        """Read an archived edge record by its ``archive/edge/...`` key."""
        raw = _b2s(self.store.db.get_sync(archive_key))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _archive_key(subject: str, predicate: str, object: str) -> str:
        """``archive/edge/{s}/{p}/{o}``, hashing any ``/``-bearing component."""
        return (
            f"archive/edge/{safe_edge_component(subject)}/"
            f"{safe_edge_component(predicate)}/{safe_edge_component(object)}"
        )


__all__ = ["SemanticMemoryWriter"]