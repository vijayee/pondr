"""Tests for ``src/subconscious/graph_fact_sink.py`` -- the R4 ->
long-term-memory pull hook (the fact-sink hookup).

CPU, self-contained against a REAL WaveDB store (tmp db): the sink's job is a
graph WRITE, so the graph itself is the real code under test. The worker's
best-effort call-site plumbing is already covered by ``test_consolidation.py``
(_RecordingSink); here the real sink's write shapes are verified:

- relation triples land in the graph (queryable via ``graph.query()``),
- state assertions land with ``asserted_by``/``asserted_at`` provenance
  (``_assertion_edge_ops``'s Phase 3c sidecar),
- re-writes are idempotent (``expand_triple`` does not duplicate),
- malformed entries are skipped without crashing or writing the rest.
"""

from __future__ import annotations

from src.memory.store import HippocampalStore
from src.subconscious.graph_fact_sink import (  # noqa: F401 - protocol shape
    GraphFactSink,
)


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _out(store, subj, pred):
    return store.graph.query().vertex(subj).out(pred).execute_sync()


def test_writes_relation_triples_and_assertion_provenance(tmp_path) -> None:
    store = _store(tmp_path)
    sink = GraphFactSink(store)
    sink.write(3, facts=[
        {"subject": "E:Alice", "predicate": "works_at", "object": "Co"},
        {"subject": "E:Bob", "predicate": "likes", "object": "tea"},
    ], state_assertions=[
        {"entity": "Alice", "value": "remote"},
    ])
    assert set(_out(store, "E:Alice", "works_at").vertices) == {"Co"}
    assert set(_out(store, "E:Bob", "likes").vertices) == {"tea"}
    # The assertion edge + its Phase 3c provenance sidecar.
    assert set(_out(store, "E:Alice", "state").vertices) == {"remote"}
    meta = store.get_edge_meta("E:Alice", "state", "remote")
    assert meta.get("asserted_by") == "fade-anchor:3"
    assert meta.get("asserted_at")  # the consolidation timestamp is stamped
    store.close()


def test_rewrites_are_idempotent(tmp_path) -> None:
    # The consolidation loop re-extracts the SAME source-blurb facts on every
    # pass; expand_triple must not duplicate and the sidecar must be RMW-merged
    # (never a tombstone revival / doubled edge).
    store = _store(tmp_path)
    sink = GraphFactSink(store)
    for _ in range(2):
        sink.write(1, facts=[
            {"subject": "E:Alice", "predicate": "works_at", "object": "Co"},
        ], state_assertions=[
            {"entity": "Alice", "value": "remote"},
        ])
    assert _out(store, "E:Alice", "works_at").count == 1
    assert _out(store, "E:Alice", "state").count == 1
    store.close()


def test_malformed_entries_are_skipped(tmp_path) -> None:
    # One bad entry must not block the rest of the batch (the _edge_ops guard
    # mirrored): non-dict payloads, missing keys, and falsy values are skipped.
    store = _store(tmp_path)
    sink = GraphFactSink(store)
    sink.write(7, facts=[
        "not-a-dict",
        {"subject": "E:X", "predicate": "pred"},          # no object
        {"predicate": "pred", "object": "o"},             # no subject
        {"subject": "E:Ok", "predicate": "knows", "object": "E:V"},
    ], state_assertions=[
        "not-a-dict",
        {"entity": "", "value": "v"},                     # empty entity
        {"entity": "Y"},                                  # no value
        {"entity": "Fine", "value": "on"},
    ])
    # Only the well-formed entries landed.
    assert set(_out(store, "E:Ok", "knows").vertices) == {"E:V"}
    assert set(_out(store, "E:Fine", "state").vertices) == {"on"}
    store.close()