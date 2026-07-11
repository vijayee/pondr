"""Tests for the Phase 3b episode-level forgetting filter (store.py).

The episode-level granularity: active-forget and reconsolidation-supersession
deprecate the *entire episode* via ``set_episode_state``. ``default_episode_ids``
then excludes it (deprecate, don't delete). The episode is NOT removed -- its
content + graph triples stay, and ``include_inactive=True`` returns it for
historical queries. The state axis is INDEPENDENT of the 3a abstracted axis.
"""

from __future__ import annotations

from src.memory.episode import Episode
from src.memory.store import HippocampalStore, _b2s
from src.gnn.semantic_memory import SemanticMemoryWriter


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid, entities=None):
    store.encode_episode(Episode(
        id=eid, timestamp="t", summary=f"s {eid}", full_text=f"f {eid}",
        entities=entities or [],
    ))


# ── defaults / getters ──
def test_episode_state_defaults_to_current(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    assert store.episode_state("ep_000001") == "current"
    assert store.episode_validity_end("ep_000001") is None
    store.close()


def test_set_episode_state_deprecates_excludes_from_default(tmp_path):
    store = _store(tmp_path)
    for i in range(1, 4):
        _encode(store, f"ep_00000{i}")
    assert store.default_episode_ids() == ["ep_000001", "ep_000002", "ep_000003"]

    store.set_episode_state("ep_000002", "deprecated")
    assert store.episode_state("ep_000002") == "deprecated"
    # default query excludes the deprecated episode
    assert store.default_episode_ids() == ["ep_000001", "ep_000003"]
    # historical opt-in returns it
    assert store.default_episode_ids(include_inactive=True) == [
        "ep_000001", "ep_000002", "ep_000003",
    ]
    store.close()


def test_set_episode_state_with_validity_end_excludes(tmp_path):
    store = _store(tmp_path)
    for i in range(1, 4):
        _encode(store, f"ep_00000{i}")
    store.set_episode_state("ep_000001", "deprecated", validity_end="2026-09-01")
    assert store.episode_validity_end("ep_000001") == "2026-09-01"
    assert store.default_episode_ids() == ["ep_000002", "ep_000003"]
    assert store.default_episode_ids(include_inactive=True) == [
        "ep_000001", "ep_000002", "ep_000003",
    ]
    store.close()


def test_set_episode_state_superseded_excludes(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    store.set_episode_state("ep_000001", "superseded")
    assert store.default_episode_ids() == []
    assert store.default_episode_ids(include_inactive=True) == ["ep_000001"]
    store.close()


# ── "deprecate, don't delete": content + graph triples survive ──
def test_set_episode_state_does_not_delete_content(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001", entities=["Alice"])
    store.set_episode_state("ep_000001", "deprecated")
    # content keys still present
    assert _b2s(store.db.get_sync("content/ep/ep_000001/summary")) == "s ep_000001"
    # the graph triple (ep, has_entity, E:Alice) still resolves via the reverse
    # in_episode edge (the convention retrieval uses, graph_traversal.py:96)
    q = store.graph.query().vertex("E:Alice").out("in_episode")
    result = q.execute_sync()
    try:
        eps = list(result.vertices)
    finally:
        result.close()
    assert "ep_000001" in eps
    store.close()


def test_set_episode_state_is_reversible(tmp_path):
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    store.set_episode_state("ep_000001", "deprecated")
    assert store.default_episode_ids() == []
    # un-forget: restore to current
    store.set_episode_state("ep_000001", "current")
    assert store.episode_state("ep_000001") == "current"
    assert store.default_episode_ids() == ["ep_000001"]
    store.close()


# ── axis independence (R3 load-bearing): abstracted vs inactive are separate ──
def test_state_axis_independent_of_abstracted_axis(tmp_path):
    store = _store(tmp_path)
    for i in range(1, 4):
        _encode(store, f"ep_00000{i}")
    # abstract ep_000001 (3a); deprecate ep_000002 (3b); leave ep_000003 active
    w = SemanticMemoryWriter(store)
    w.create_abstract(["ep_000001"], "Alice gist")
    store.set_episode_state("ep_000002", "deprecated")

    # both axes on (default): only the untouched episode survives
    assert store.default_episode_ids() == ["ep_000003"]

    # abstracted axis off (inactive still on): abstracted returns, deprecated still out
    assert store.default_episode_ids(include_abstracted=True) == [
        "ep_000001", "ep_000003",
    ]

    # inactive axis off (abstracted still on): deprecated returns, abstracted still out
    assert store.default_episode_ids(include_inactive=True) == [
        "ep_000002", "ep_000003",
    ]

    # both axes off (everything): historical full set
    assert store.default_episode_ids(
        include_abstracted=True, include_inactive=True
    ) == ["ep_000001", "ep_000002", "ep_000003"]
    store.close()


# ── regression: the scan-rewrite still excludes abstracted (3a behavior) ──
def test_default_episode_ids_still_excludes_abstracted(tmp_path):
    store = _store(tmp_path)
    for i in range(1, 4):
        _encode(store, f"ep_00000{i}", entities=["Alice"])
    w = SemanticMemoryWriter(store)
    w.create_abstract(["ep_000001", "ep_000002"], "Alice interactions")
    # unchanged 3a behavior (no 3b state set on any episode)
    assert store.default_episode_ids() == ["ep_000003"]
    assert store.default_episode_ids(include_abstracted=True) == [
        "ep_000001", "ep_000002", "ep_000003",
    ]
    store.close()


# ── master gate: forgetting_enabled=False restores deprecated episodes ──
def test_forgetting_disabled_restores_deprecated_to_default(tmp_path):
    """The Phase 3b state/validity_end axis is gated on forgetting_enabled.

    With the master flag off, a corpus that has prior deprecations behaves as if
    forgetting were never deployed: deprecated/superseded episodes reappear in
    default queries. The 3a abstracted axis is NOT gated (independent of 3b)."""
    from src.config import config as _config

    store = _store(tmp_path)
    for i in range(1, 4):
        _encode(store, f"ep_00000{i}")
    store.set_episode_state("ep_000002", "deprecated")
    # forgetting on (default): deprecated excluded.
    assert _config.forgetting_enabled is True
    assert store.default_episode_ids() == ["ep_000001", "ep_000003"]

    saved = _config.forgetting_enabled
    _config.forgetting_enabled = False
    try:
        # forgetting off: the state axis is skipped -> deprecated reappears.
        assert store.default_episode_ids() == ["ep_000001", "ep_000002", "ep_000003"]
    finally:
        _config.forgetting_enabled = saved
    # restored: deprecated excluded again.
    assert store.default_episode_ids() == ["ep_000001", "ep_000003"]
    store.close()


def test_forgetting_disabled_does_not_ungate_abstracted_axis(tmp_path):
    """``forgetting_enabled=False`` gates ONLY the 3b state axis. The 3a
    abstracted axis keeps excluding regardless (it is not a forgetting concept)."""
    from src.config import config as _config

    store = _store(tmp_path)
    for i in range(1, 4):
        _encode(store, f"ep_00000{i}")
    w = SemanticMemoryWriter(store)
    w.create_abstract(["ep_000001"], "gist")
    saved = _config.forgetting_enabled
    _config.forgetting_enabled = False
    try:
        # abstracted still excluded even with forgetting off.
        assert store.default_episode_ids() == ["ep_000002", "ep_000003"]
        # the historical opt-in still returns it.
        assert store.default_episode_ids(include_abstracted=True) == [
            "ep_000001", "ep_000002", "ep_000003",
        ]
    finally:
        _config.forgetting_enabled = saved
    store.close()