"""Per-edge forgetting metadata sidecar (Phase 3b).

Edges are otherwise stateless ``(s, p, o)`` triples in the graph layer. This
module adds a lazy-created per-edge metadata namespace
``content/edge/{s}/{p}/{o}`` (one JSON blob per edge) carrying the forgetting
fields -- the same shape as the ``meta`` dict operated on by
``src/memory/forgetting.py``::

    utility_score, utility_decay_rate, base_decay_rate, state, access_count,
    reconsolidation_count, ltp_phase, consolidation_window_start,
    retrieval_timestamps, saturation_flags, validity_end

Lazy-create: an edge with no sidecar is treated as ``forgetting.default_meta()``
on read; a sidecar is only written when a retrieval boost or dream-state decay
touches the edge. This bounds write amplification -- only edges that are
actually used/decayed get a sidecar.

Key hashing reuses ``store.safe_edge_component`` (hashes any component with
``/`` or NUL), so ``content/edge/...`` never collides with the live graph key
``memory/spo/{s}/{p}/{o}`` (literal slash) or the archive key
``archive/edge/{s}/{p}/{o}``.

RMW caveat: ``set_edge_state`` and the retrieval-time boost do a
read-modify-write on the blob. Like ``store._counter_next``, this is NOT atomic
across concurrent writers -- a concurrent boost can lose an increment.
Acceptable under the single-user assumption the rest of the system makes
(documented in ``docs/Phase 3b.md``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .forgetting import default_meta
from .store import _b2s, safe_edge_component

if TYPE_CHECKING:
    from .store import HippocampalStore

__all__ = [
    "edge_meta_key",
    "get_edge_meta",
    "edge_meta_put_op",
    "update_edge_meta",
    "batch_update_edge_meta",
    "set_edge_state",
    "is_edge_current",
]


def edge_meta_key(subject: str, predicate: str, object: str) -> str:
    """``content/edge/{s}/{p}/{o}``, hashing any ``/``-bearing component."""
    return (
        f"content/edge/{safe_edge_component(subject)}/"
        f"{safe_edge_component(predicate)}/{safe_edge_component(object)}"
    )


def get_edge_meta(
    store: "HippocampalStore", subject: str, predicate: str, object: str
) -> dict:
    """Read an edge's sidecar meta dict; ``default_meta()`` if none exists yet.

    Never raises on a missing sidecar (lazy-create contract). Returns a fresh
    dict each call (safe to mutate and write back). Old sidecars missing newer
    fields are merged over ``default_meta()`` so schema additions degrade
    gracefully.
    """
    raw = _b2s(store.db.get_sync(edge_meta_key(subject, predicate, object)))
    if not raw:
        return default_meta()
    try:
        meta = json.loads(raw)
    except (ValueError, TypeError):
        return default_meta()
    merged = default_meta()
    merged.update(meta)
    return merged


def edge_meta_put_op(
    subject: str, predicate: str, object: str, meta: dict
) -> dict:
    """A ``batch_sync`` put-op for the meta blob (no store access; for batching).

    Callers composing a larger atomic batch (e.g. the consolidation ``_apply``
    that also writes abstracts edges) include this op directly rather than
    issuing a separate ``batch_sync``.
    """
    return {
        "type": "put",
        "key": edge_meta_key(subject, predicate, object),
        "value": json.dumps(meta, ensure_ascii=False),
    }


def update_edge_meta(
    store: "HippocampalStore", subject: str, predicate: str, object: str, meta: dict
) -> None:
    """Write one edge's sidecar (single ``batch_sync``)."""
    store.db.batch_sync([edge_meta_put_op(subject, predicate, object, meta)])


def batch_update_edge_meta(
    store: "HippocampalStore",
    updates: list[tuple[str, str, str, dict]],
) -> None:
    """Write many edge sidecars in ONE atomic ``batch_sync``.

    ``updates`` is a list of ``(subject, predicate, object, meta)`` tuples. The
    retrieval-time boost and the consolidation dream pass use this so all edges
    touched by one retrieval / one center land atomically.
    """
    if not updates:
        return
    ops = [edge_meta_put_op(s, p, o, meta) for (s, p, o, meta) in updates]
    store.db.batch_sync(ops)


def set_edge_state(
    store: "HippocampalStore",
    subject: str,
    predicate: str,
    object: str,
    state: str,
    validity_end: "str | None" = None,
) -> dict:
    """Set an edge's ``state`` (+ optional ``validity_end``) via read-modify-write.

    Used by active-forget (``state='deprecated'``), soft-archive
    (``state='archived'``), and reconsolidation (``state='superseded'`` on the
    old edge). Returns the written meta dict. RMW caveat: not atomic across
    concurrent writers (see module docstring).
    """
    meta = get_edge_meta(store, subject, predicate, object)
    meta["state"] = state
    if validity_end is not None:
        meta["validity_end"] = validity_end
    update_edge_meta(store, subject, predicate, object, meta)
    return meta


def is_edge_current(
    store: "HippocampalStore", subject: str, predicate: str, object: str
) -> bool:
    """True if an edge's sidecar ``state == 'current'`` (or no sidecar yet).

    The edge-level default-query filter (``graph_traversal._get_episodes_by_*``)
    calls this to skip deprecated/superseded/archived associations. A missing
    sidecar means the edge has never been touched by forgetting, so it is
    current. One ``get_sync`` point lookup per candidate edge.
    """
    return get_edge_meta(store, subject, predicate, object)["state"] == "current"