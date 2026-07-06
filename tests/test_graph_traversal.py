"""Offline tests for Phase 1b graph traversal (pattern completion).

No GLiNER / Bonsai — episodes are constructed directly and encoded into a
tmp_path WaveDB store, then a ``GraphTraversal`` is driven with literal
``query_plan`` dicts (no query planner needed). Covers the union/intersection
candidate logic, topic/tone axes, the ``follows`` chain walk, scope
rehydration (entities/topics/tones/decisions/session/user from the graph), and
the NUL-free scan regression gate (the WaveDB HBTrie scan-corruption fix is the
gate — a corrupt key in any scan the traversal issues means the bug is back).
"""

from datetime import datetime, timedelta

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.graph_traversal import GraphTraversal


def _ep(
    eid: str,
    summary: str = "",
    entities: list[str] | None = None,
    topics: list[str] | None = None,
    tones: list[str] | None = None,
    decisions: list[str] | None = None,
    follows: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    ts: str | None = None,
) -> Episode:
    return Episode(
        id=eid,
        timestamp=ts or "2026-07-03T10:00:00",
        summary=summary or f"summary {eid}",
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [],
        topics=topics or [],
        tones=tones or [],
        decisions=decisions or [],
        follows=follows,
        user_id=user_id,
        session_id=session_id,
    )


def _traversal(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    return store, GraphTraversal(store)


# ── axis queries ──


def test_entity_union_returns_both(tmp_path):
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice"]))
    store.encode_episode(_ep("ep_002", entities=["Bob"]))
    store.encode_episode(_ep("ep_003", entities=["Carol"]))

    results = trav.retrieve({"entities": ["Alice", "Bob"], "entity_mode": "union"})
    ids = {r["episode_id"] for r in results}
    assert ids == {"ep_001", "ep_002"}, ids
    store.close()


def test_entity_intersection_returns_only_shared(tmp_path):
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice", "Bob"]))
    store.encode_episode(_ep("ep_002", entities=["Alice"]))

    results = trav.retrieve({"entities": ["Alice", "Bob"], "entity_mode": "intersection"})
    ids = {r["episode_id"] for r in results}
    assert ids == {"ep_001"}, ids  # only ep_001 has both
    store.close()


def test_topic_axis(tmp_path):
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", topics=["databases"]))
    store.encode_episode(_ep("ep_002", topics=["networking"]))
    store.encode_episode(_ep("ep_003", topics=["databases"]))

    results = trav.retrieve({"topics": ["databases"]})
    assert {r["episode_id"] for r in results} == {"ep_001", "ep_003"}
    store.close()


def test_tone_axis(tmp_path):
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", tones=["frustrated"]))
    store.encode_episode(_ep("ep_002", tones=["curious"]))

    results = trav.retrieve({"tones": ["frustrated"]})
    assert {r["episode_id"] for r in results} == {"ep_001"}
    store.close()


def test_no_match_returns_empty(tmp_path):
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice"]))

    assert trav.retrieve({"entities": ["Zelda"]}) == []
    store.close()


def test_no_axis_returns_all(tmp_path):
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice"]))
    store.encode_episode(_ep("ep_002", entities=[]))  # entity-less episode

    results = trav.retrieve({})
    # No axis → all episodes, including the entity-less one (content scan is complete).
    assert {r["episode_id"] for r in results} == {"ep_001", "ep_002"}
    store.close()


# ── follows chain ──


def test_follows_chain_forward(tmp_path):
    """temporal_after walks forward (later episodes) from the keyword anchor.

    Chain ORDER is asserted on ``_follow_chain`` directly — ``retrieve`` re-sorts
    by score, which would obscure BFS order.
    """
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", summary="we implemented morphisms"))
    store.encode_episode(_ep("ep_002", summary="then profiling", follows="ep_001"))
    store.encode_episode(_ep("ep_003", summary="then cleanup", follows="ep_002"))

    assert trav._follow_chain("ep_001", "forward") == ["ep_001", "ep_002", "ep_003"]

    # Via retrieve, the same episodes are returned (as a set — order is by score).
    results = trav.retrieve({"temporal_after": "morphisms"})
    assert {r["episode_id"] for r in results} == {"ep_001", "ep_002", "ep_003"}
    store.close()


def test_follows_chain_backward(tmp_path):
    """temporal_before walks backward (earlier episodes) from the anchor."""
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", summary="first design"))
    store.encode_episode(_ep("ep_002", summary="then implementation", follows="ep_001"))
    store.encode_episode(_ep("ep_003", summary="final cleanup", follows="ep_002"))

    assert trav._follow_chain("ep_003", "backward") == ["ep_003", "ep_002", "ep_001"]

    results = trav.retrieve({"temporal_before": "cleanup"})
    assert {r["episode_id"] for r in results} == {"ep_001", "ep_002", "ep_003"}
    store.close()


def test_temporal_chain_no_anchor_falls_back(tmp_path):
    """If the keyword matches nothing, the axis candidates are returned unchanged."""
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice"]))
    store.encode_episode(_ep("ep_002", entities=["Alice"], follows="ep_001"))

    results = trav.retrieve({"entities": ["Alice"], "temporal_after": "nonexistent"})
    assert {r["episode_id"] for r in results} == {"ep_001", "ep_002"}
    store.close()


# ── temporal bucket filter ──


def test_temporal_filter_this_week(tmp_path):
    store, trav = _traversal(tmp_path)
    now = datetime.now()
    recent = (now - timedelta(days=2)).isoformat()
    old = (now - timedelta(days=40)).isoformat()
    store.encode_episode(_ep("ep_001", ts=recent))
    store.encode_episode(_ep("ep_002", ts=old))

    results = trav.retrieve({"temporal_filter": "this_month"})
    assert {r["episode_id"] for r in results} == {"ep_001"}
    store.close()


# ── hydration + scoring ──


def test_hydrate_populates_graph_fields(tmp_path):
    """Scope rehydration: entities/topics/tones/decisions/session/user from the graph."""
    store, trav = _traversal(tmp_path)
    store.encode_episode(
        _ep(
            "ep_001",
            summary="decided on hbtrie",
            entities=["Alice"],
            topics=["databases"],
            tones=["curious"],
            decisions=["use_hbtrie"],
            user_id="victor",
            session_id="S:0001",
        )
    )

    results = trav.retrieve({"entities": ["Alice"]})
    assert len(results) == 1
    r = results[0]
    assert r["episode_id"] == "ep_001"
    assert r["entities"] == ["Alice"]
    assert r["topics"] == ["databases"]
    assert r["tones"] == ["curious"]
    assert r["decisions"] == ["use_hbtrie"]
    assert r["session_id"] == "S:0001"
    assert r["user_id"] == "victor"
    store.close()


def test_scoring_ranks_higher_match_first(tmp_path):
    """An episode matching more query-axis values outscores one matching fewer.

    Under intersection candidate semantics every candidate matches all specified
    axes, so score differentiation comes from a multi-value UNION axis: an
    episode that matches more of the query's values for that axis scores higher.
    """
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice"]))
    store.encode_episode(_ep("ep_002", entities=["Alice", "Bob"]))

    results = trav.retrieve({"entities": ["Alice", "Bob"], "entity_mode": "union"})
    assert len(results) == 2
    assert results[0]["episode_id"] == "ep_002"  # matches both query entities
    assert results[0]["score"] > results[1]["score"]
    store.close()


def test_limit_truncates(tmp_path):
    store, trav = _traversal(tmp_path)
    for i in range(6):
        store.encode_episode(_ep(f"ep_{i:03d}", entities=["Alice"]))

    results = trav.retrieve({"entities": ["Alice"], "limit": 2})
    assert len(results) == 2
    store.close()


# ── regression gate ──


def test_reopen_traversal_works(tmp_path):
    """Graph traversal works after closing and reopening the store.

    Reopen regression gate for the WaveDB 0.1.10 fix (graph-query segfault on
    reopened tries — scan iterator missed lazy-load of child/child_bnode and
    skipped empty-value leaves on a reopened trie). Encodes 40 episodes (past
    the >38 btree-split threshold), closes, reopens, then traverses + scans.
    """
    path = str(tmp_path / "db")
    store = HippocampalStore(path)
    for i in range(40):
        store.encode_episode(_ep(f"ep_{i:03d}", entities=[f"E{i}"], topics=[f"T{i % 5}"]))
    store.close()

    store2 = HippocampalStore(path)
    trav = GraphTraversal(store2)
    results = trav.retrieve({"entities": ["E39"]})
    assert results and results[0]["episode_id"] == "ep_039"
    keys = [k for k, _ in store2.db.create_read_stream(start="memory/", end=None)]
    assert not any("\x00" in k for k in keys), "corrupt graph keys after reopen"
    store2.close()


def test_traversal_scans_are_nul_free(tmp_path):
    """Every scan the traversal issues must be NUL-free (HBTrie scan-corruption gate).

    Encoding enough episodes to cross the >38-entry btree-split threshold, then
    running a full retrieve (which scans memory/spo, memory/pos, and content/ep),
    must not surface any NUL-padded mis-split keys.
    """
    store, trav = _traversal(tmp_path)
    for i in range(40):
        store.encode_episode(
            _ep(
                f"ep_{i:03d}",
                entities=[f"E{i}"],
                topics=[f"T{i % 5}"],
                tones=["curious"],
                user_id="victor",
                session_id=f"S:{i:04d}",
            )
        )

    # Force every scan path: axis query, all-episodes content scan, hydration
    # SPO scan, and the has_session POS scan for user resolution.
    results = trav.retrieve({"entities": [f"E{39}"]})
    assert results
    assert results[0]["user_id"] == "victor"

    keys = [k for k, _ in store.db.create_read_stream(start="memory/", end=None)]
    assert keys, "no graph keys stored"
    corrupt = [k for k in keys if "\x00" in k]
    assert not corrupt, f"corrupt graph keys (scan bug regressed): {corrupt[:3]}"


# ── entity salience (Phase 1c) ──


def test_salience_weighted_scoring(tmp_path):
    """High-salience entity matches outscore low-salience ones (Phase 1c).

    Two episodes each matching one query entity: Alice (salience ~0.31 from 50
    mentions) vs Bob (salience ~0.13 from 1 mention). Alice's entity term
    (~6.56) beats Bob's (~5.65) by more than the recency tiebreaker (Bob is one
    day newer → +0.1), so Alice ranks higher despite being older.
    """
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice"], ts="2026-07-01T00:00:00"))
    store.encode_episode(_ep("ep_002", entities=["Bob"], ts="2026-07-02T00:00:00"))

    # Persist salience via the sorted-batch store method so the keys are
    # get_sync-safe (NOT raw put_sync).
    store.write_entity_salience_batch(
        counts={"Alice": 50, "Bob": 1},
        last_ep={"Alice": "ep_001", "Bob": "ep_002"},
    )

    results = trav.retrieve(
        {"entities": ["Alice", "Bob"], "entity_mode": "union", "limit": 5}
    )
    by_id = {r["episode_id"]: r for r in results}
    assert "ep_001" in by_id and "ep_002" in by_id
    assert by_id["ep_001"]["score"] > by_id["ep_002"]["score"]
    store.close()


def test_salience_unknown_entity_still_scores(tmp_path):
    """An entity with no salience entry still contributes _W_ENTITY/2 (not 0)."""
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice"], ts="2026-07-01T00:00:00"))
    # No write_entity_salience_batch call → get_entity_salience("Alice") == 0.0.
    results = trav.retrieve({"entities": ["Alice"], "limit": 5})
    assert results and results[0]["episode_id"] == "ep_001"
    # Entity match with salience 0.0 = _W_ENTITY * 0.5 = 5.0; recency rank 0 → 0.
    assert results[0]["score"] == 5.0
    store.close()


def test_compute_entity_salience_script(tmp_path):
    """The batch script counts has_entity triples correctly and persists them.

    Mirrors what scripts/compute_entity_salience.py does, against a tmp store,
    so the scan/parse/batch-persist path is exercised without a real corpus.
    """
    from collections import Counter

    from scripts.compute_entity_salience import _iter_entity_episode_pairs

    store, _ = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice", "Bob"]))
    store.encode_episode(_ep("ep_002", entities=["Alice"]))
    store.encode_episode(_ep("ep_003", entities=["Alice"]))

    counts: Counter[str] = Counter()
    last_ep: dict[str, str] = {}
    for entity, eid in _iter_entity_episode_pairs(store):
        counts[entity] += 1
        last_ep[entity] = eid
    store.write_entity_salience_batch(dict(counts), last_ep)

    assert counts["Alice"] == 3
    assert counts["Bob"] == 1
    # get_sync reads the sorted-written keys back.
    assert store.get_entity_salience("Alice") > store.get_entity_salience("Bob")
    store.close()


# ── temporal date-range (Phase 1c) ──


def test_date_range_query(tmp_path):
    """Absolute date_from/date_to filters episodes by timestamp."""
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", ts="2025-06-15T10:00:00"))
    store.encode_episode(_ep("ep_002", ts="2025-07-15T10:00:00"))
    store.encode_episode(_ep("ep_003", ts="2025-08-15T10:00:00"))

    results = trav.retrieve({"date_from": "2025-06-01", "date_to": "2025-07-31", "limit": 5})
    ids = {r["episode_id"] for r in results}
    assert ids == {"ep_001", "ep_002"}, ids
    store.close()


def test_date_range_and_entity_combined(tmp_path):
    """Absolute date range combines with an entity filter."""
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", entities=["Alice"], ts="2025-06-15T10:00:00"))
    store.encode_episode(_ep("ep_002", entities=["Bob"], ts="2025-06-15T10:00:00"))
    store.encode_episode(_ep("ep_003", entities=["Alice"], ts="2025-08-15T10:00:00"))

    results = trav.retrieve(
        {
            "entities": ["Alice"],
            "entity_mode": "union",
            "date_from": "2025-06-01",
            "date_to": "2025-07-31",
            "limit": 5,
        }
    )
    ids = {r["episode_id"] for r in results}
    assert ids == {"ep_001"}, ids  # Bob excluded (entity); ep_003 excluded (August)
    store.close()


def test_date_range_one_sided(tmp_path):
    """date_from alone (open-ended upper bound) keeps everything at/after it."""
    store, trav = _traversal(tmp_path)
    store.encode_episode(_ep("ep_001", ts="2025-05-15T10:00:00"))
    store.encode_episode(_ep("ep_002", ts="2025-07-15T10:00:00"))

    results = trav.retrieve({"date_from": "2025-06-01", "limit": 5})
    ids = {r["episode_id"] for r in results}
    assert ids == {"ep_002"}, ids
    store.close()