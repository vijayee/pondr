"""The R4 -> long-term-memory pull hook (the fact-sink hookup).

When the fade memory's consolidation loop gists a fading anchor
(``fade_consolidation``), the gister's extracted facts are staged as a sidecar
and then DROPPED (``fact_sink`` was ``None`` in v1). ``GraphFactSink`` is the
consumer: it writes the facts to the WaveDB hippocampal graph so they survive
the fade -- the anchor's blurb may go away (Regime 4) but the structured facts
stay queryable in the long-term-memory graph layer.

Two write shapes, mirroring the encoder's ``HippocampalStore._edge_ops``
pattern exactly:

- **Relation triples** go through ``graph.expand_triple`` as-is (the same
  call ``_edge_ops`` makes for ``episode.relations``) -- subject /
  predicate / object in the extractor's own vocabulary. ``expand_triple``
  is idempotent, so the multi-pass consolidation loop re-extracting the
  same SOURCE-blurb facts each pass re-asserts the same triples (a no-op
  on the second write), not duplicates.
- **State assertions** go through ``HippocampalStore._assertion_edge_ops``
  (the Phase 3c provenance writer): ``(E:entity, state, value)`` +
  an edge sidecar ``asserted_by`` / ``asserted_at``. The asserted_by
  marker is ``fade-anchor:{anchor_id}`` -- the anchor is an int-keyed fade
  chunk with NO episode id (the fade ingests BEFORE an episode is
  pre-allocated), so the consolidation pass itself is the honest asserting
  unit and the provenance records that (latest-asserting-wins still
  updates the sidecar on re-assertion).

One atomic ``batch_sync`` per ``write`` -- a fact-sink failure is caught by
the worker's best-effort call site (logged, swallowed), and the
consolidation itself has already succeeded by then. The assertion sidecar
is an RMW-merge (``get_edge_meta`` then the batch), whose race window against
a same-edge foreground encode is the pre-existing Phase 3c pattern (the
worker writes only in the foreground-clear windows between turns).
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ..memory.store import HippocampalStore

# The provenance marker for assertion edges written by a fade consolidation
# pass. The anchor is an int-keyed fade chunk (no episode id exists to cite),
# so the marker names the anchor instead of pretending to be one.
_ASSERTED_BY_TEMPLATE = "fade-anchor:{anchor_id}"


def _utc_now() -> str:
    """ISO-8601 UTC timestamp (the semantic-memory helper's shape)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


class GraphFactSink:
    """Write a consolidated anchor's facts to the WaveDB graph.

    Implements the ``FactSink`` protocol structurally (``write(anchor_id,
    facts, state_assertions)``); ``ConsolidationWorker`` calls it best-effort
    after a successful consolidate. Constructed by ``build_ponder`` only when
    the fact-sink flag is on (byte-identical-OFF gate: flag off -> no sink ->
    the worker's ``fact_sink`` stays ``None`` -> no graph writes).
    """

    def __init__(self, store: "HippocampalStore") -> None:
        self.store = store

    def write(self, anchor_id: int, facts: list[dict],
              state_assertions: list[dict]) -> None:
        """Write ``facts`` relation triples + ``state_assertions`` to the graph.

        Malformed entries (missing subject/predicate/object or entity/value)
        are SKIPPED -- mirrors ``_edge_ops``'s guards for the same lists; one
        bad entry never blocks the rest of the batch.
        """
        ops: list[dict] = []
        for rel in facts:
            if not isinstance(rel, dict):
                continue
            subj = rel.get("subject")
            pred = rel.get("predicate")
            obj = rel.get("object")
            if not subj or not pred or not obj:
                continue
            ops += self.store.graph.expand_triple(subj, pred, obj)
        asserted_by = _ASSERTED_BY_TEMPLATE.format(anchor_id=anchor_id)
        asserted_at = _utc_now()
        for a in state_assertions:
            if not isinstance(a, dict):
                continue
            ent = a.get("entity")
            val = a.get("value")
            if not ent or val is None:
                continue  # the _edge_ops guard: entity + value both needed
            ops += self.store._assertion_edge_ops(
                ent, val, asserted_by, asserted_at)
        if ops:
            self.store.db.batch_sync(ops)