"""Phase 3b step 10: entity-salience composition tests.

``get_entity_salience`` composes three factors once the consolidation dream pass
has persisted a structural salience for the entity:

    salience = mention_factor * structural_factor * recency_factor

Cold-start fallback: when ``structural_salience`` is absent, the result is the
Phase 1c mention-only value (byte-identical to pre-3b), so a fresh corpus and
the GNN cold-start prior (``features.py``) are unchanged. Recency is a neutral
multiplier (1.0) when ``last_mentioned_ts`` is absent.

These tests persist salience keys directly (sorted ``batch_sync`` via the store
method, NOT raw ``put_sync``) so the ``get_sync`` reads are reliable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.memory.store import HippocampalStore


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _persist(store, entity, *, count, structural=None, last_ts=None):
    """Persist salience keys for one entity via the sorted-batch store method."""
    store.write_entity_salience_batch(
        counts={entity: count},
        last_ep={entity: "ep_001"},
        last_ep_ts={entity: last_ts} if last_ts else None,
    )
    if structural is not None:
        store.persist_node_salience(f"E:{entity}", structural)


def _mention(count: int) -> float:
    """The Phase 1c mention-only formula (mirrors store.get_entity_salience)."""
    return min(1.0, 0.1 + 0.3 * (count ** 0.5) / 10.0)


# ── cold-start fallback ──

def test_cold_start_no_structural_is_mention_only(tmp_path):
    """No structural_salience -> mention-only, byte-identical to Phase 1c."""
    store = _store(tmp_path)
    _persist(store, "Alice", count=50)  # no structural
    assert store.get_entity_salience("Alice") == _mention(50)
    store.close()


def test_unknown_entity_is_zero(tmp_path):
    store = _store(tmp_path)
    assert store.get_entity_salience("Nobody") == 0.0
    store.close()


def test_cold_start_with_recency_ts_but_no_structural_is_mention_only(tmp_path):
    """Recency alone (without structural) does NOT modulate -- structural is the
    composition gate; recency is a multiplier applied only when structural is
    present. So a corpus that has last_mentioned_ts but never ran consolidation
    still behaves mention-only."""
    store = _store(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=365)
    _persist(store, "Alice", count=50, last_ts=_iso(old))
    assert store.get_entity_salience("Alice") == _mention(50)
    store.close()


# ── composition ──

def test_composed_is_mention_times_structural_when_no_recency(tmp_path):
    """structural present, no last_mentioned_ts -> recency neutral (1.0)."""
    store = _store(tmp_path)
    _persist(store, "Alice", count=50, structural=0.8)  # no last_ts
    assert store.get_entity_salience("Alice") == _mention(50) * 0.8
    store.close()


def test_composed_in_unit_interval(tmp_path):
    """R4 gate: composed salience is always in [0,1]."""
    store = _store(tmp_path)
    for s in (0.0, 0.25, 0.5, 0.9, 1.0):
        _persist(store, f"E{s}", count=1000, structural=s)
        val = store.get_entity_salience(f"E{s}")
        assert 0.0 <= val <= 1.0, (s, val)
    store.close()


def test_structural_zero_zeroes_composed(tmp_path):
    """structural=0 -> composed=0 (a structurally-irrelevant entity drops out)."""
    store = _store(tmp_path)
    _persist(store, "Alice", count=50, structural=0.0)
    assert store.get_entity_salience("Alice") == 0.0
    store.close()


def test_structural_one_equals_mention_when_recency_neutral(tmp_path):
    store = _store(tmp_path)
    _persist(store, "Alice", count=50, structural=1.0)  # no last_ts
    assert store.get_entity_salience("Alice") == _mention(50)
    store.close()


# ── recency ──

def test_recency_decays_old_mention(tmp_path):
    """An old mention scores lower than a fresh one (same count, same structural)."""
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)
    fresh = now - timedelta(days=1)
    old = now - timedelta(days=120)  # 4 half-lives (30d) -> 1/16
    _persist(store, "Fresh", count=50, structural=0.5, last_ts=_iso(fresh))
    _persist(store, "Old", count=50, structural=0.5, last_ts=_iso(old))
    f = store.get_entity_salience("Fresh")
    o = store.get_entity_salience("Old")
    assert f > o, (f, o)
    # 1 day vs 120 days: fresh ~ mention*0.5*~0.977; old ~ mention*0.5*0.0625.
    assert o < f / 10.0, (o, f)
    store.close()


def test_recency_neutral_when_ts_absent(tmp_path):
    """No last_mentioned_ts -> recency 1.0 (no modulation)."""
    store = _store(tmp_path)
    _persist(store, "Alice", count=50, structural=0.6)  # no last_ts
    # recency neutral -> composed == mention * structural exactly.
    assert abs(store.get_entity_salience("Alice") - _mention(50) * 0.6) < 1e-12
    store.close()


def test_recency_unparseable_ts_is_neutral(tmp_path):
    """A garbage timestamp decays to neutral (1.0), never to zero."""
    store = _store(tmp_path)
    # Plant a structural + a garbage last_mentioned_ts directly.
    store.write_entity_salience_batch(
        counts={"Alice": 50}, last_ep={"Alice": "ep_001"},
        last_ep_ts={"Alice": "not-a-timestamp"},
    )
    store.persist_node_salience("E:Alice", 0.6)
    assert abs(store.get_entity_salience("Alice") - _mention(50) * 0.6) < 1e-12
    store.close()


def test_recency_parses_timestamp_without_z_suffix(tmp_path):
    """Episode timestamps are stored verbatim and may lack the ``Z`` suffix
    (e.g. ``2026-07-01T00:00:00``). The recency parser must still decay them --
    a strict ``%Y-%m-%dT%H:%M:%SZ`` parse would silently fail and return neutral
    for every real episode timestamp. Regression gate for the parser fix."""
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)
    old_no_z = (now - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%S")  # no Z
    fresh_no_z = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    _persist(store, "Fresh", count=50, structural=0.5, last_ts=fresh_no_z)
    _persist(store, "Old", count=50, structural=0.5, last_ts=old_no_z)
    f = store.get_entity_salience("Fresh")
    o = store.get_entity_salience("Old")
    assert f > o, (f, o)  # recency parsed -> old decays below fresh
    assert o < f / 10.0, (o, f)
    store.close()


def test_recency_future_ts_clamped_to_one(tmp_path):
    """A future timestamp (age < 0) -> recency 1.0 (clamped, not > 1)."""
    store = _store(tmp_path)
    future = datetime.now(timezone.utc) + timedelta(days=10)
    _persist(store, "Alice", count=50, structural=0.6, last_ts=_iso(future))
    assert abs(store.get_entity_salience("Alice") - _mention(50) * 0.6) < 1e-12
    store.close()


# ── write path ──

def test_write_batch_persists_last_mentioned_ts(tmp_path):
    store = _store(tmp_path)
    ts = "2026-06-01T00:00:00Z"
    store.write_entity_salience_batch(
        counts={"Alice": 50}, last_ep={"Alice": "ep_001"},
        last_ep_ts={"Alice": ts},
    )
    from src.memory.store import _b2s
    raw = _b2s(store.db.get_sync("content/entity/Alice/last_mentioned_ts"))
    assert raw == ts
    # mention_count + last_mentioned still written (back-compat).
    assert _b2s(store.db.get_sync("content/entity/Alice/mention_count")) == "50"
    assert _b2s(store.db.get_sync("content/entity/Alice/last_mentioned")) == "ep_001"
    store.close()


def test_write_batch_without_last_ep_ts_writes_no_ts_key(tmp_path):
    """The 2-arg form (no last_ep_ts) writes no last_mentioned_ts (back-compat)."""
    store = _store(tmp_path)
    store.write_entity_salience_batch(
        counts={"Alice": 50}, last_ep={"Alice": "ep_001"},
    )
    from src.memory.store import _b2s
    assert _b2s(store.db.get_sync("content/entity/Alice/last_mentioned_ts")) == ""
    store.close()