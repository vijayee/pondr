"""Unit tests for HippocampalStore (WaveDB content + graph index).

These run offline — they use the installed ``wavedb`` package (CPU), no GLiNER
or Bonsai. They verify the atomic encode path that ``GraphLayer.expand_triple``
+ a single ``batch_sync`` provides, and that the graph index scan is clean
(the WaveDB HBTrie scan-corruption fix from 2026-07-04 is the regression gate
here — a corrupt key in the scan means the bug is back).
"""

from src.memory.episode import Episode
from src.memory.store import HippocampalStore


def _make_episode(eid: str = "ep_001") -> Episode:
    return Episode(
        id=eid,
        timestamp="2026-07-03T10:00:00",
        summary="Test episode",
        full_text="User: Hi\nAssistant: Hello",
        entities=["Alice"],
        topics=["database_design"],
        tones=["curious"],
        decisions=["use_hbtrie"],
        relations=[{"subject": "Alice", "predicate": "explains", "object": "HBTrie"}],
    )


def test_encode_and_retrieve_episode(tmp_path):
    """Episode content can be stored and retrieved from WaveDB."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    ep = _make_episode()
    store.encode_episode(ep)
    loaded = store.get_episode("ep_001")

    assert loaded is not None
    assert loaded.summary == "Test episode"
    assert loaded.state == "current"
    store.close()


def test_get_episode_missing_returns_none(tmp_path):
    """Retrieving a non-existent episode returns None, not raising."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    assert store.get_episode("does_not_exist") is None
    store.close()


def test_graph_triples_stored(tmp_path):
    """Encoding stores triples in the graph layer and the index scan is clean.

    The regression gate: after encoding, scanning the ``memory/`` subtree must
    return keys with no NUL padding (the HBTrie scan-corruption bug produced
    NUL-padded mis-split keys once the btree split at >38 entries). The ontology
    seed alone writes 1448 graph keys, well past the split threshold.
    """
    store = HippocampalStore(str(tmp_path / "test_db"))
    store.encode_episode(_make_episode())

    graph_keys = [k for k, _ in store.db.create_read_stream(start="memory/", end=None)]
    assert graph_keys, "no graph triples stored"
    corrupt = [k for k in graph_keys if "\x00" in k]
    assert not corrupt, f"corrupt graph keys (scan bug regressed): {corrupt[:3]}"

    # The entity index entry E:Alice in_episode ep_001 must be present and clean.
    alice_keys = [
        k for k, _ in store.db.create_read_stream(
            start="memory/spo/E:Alice/in_episode/",
            end="memory/spo/E:Alice/in_episode/\x7f",
        )
    ]
    assert "memory/spo/E:Alice/in_episode/ep_001" in alice_keys, alice_keys
    assert not any("\x00" in k for k in alice_keys)
    store.close()


def test_follows_chain_stored(tmp_path):
    """The `follows` relation is written as a graph triple."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    ep1 = Episode(
        id="ep_001", timestamp="2026-07-03T10:00:00", summary="s1",
        full_text="User: a\nAssistant: b", entities=[], topics=[], tones=[],
    )
    ep2 = Episode(
        id="ep_002", timestamp="2026-07-03T10:00:01", summary="s2",
        full_text="User: c\nAssistant: d", entities=[], topics=[], tones=[],
        follows="ep_001",
    )
    store.encode_episode(ep1)
    store.encode_episode(ep2)

    follows_keys = [
        k for k, _ in store.db.create_read_stream(
            start="memory/spo/ep_002/follows/",
            end="memory/spo/ep_002/follows/\x7f",
        )
    ]
    assert any("ep_001" in k for k in follows_keys), follows_keys
    store.close()


def test_ontology_seeded_once(tmp_path):
    """The seed taxonomy is written as subClassOf triples at init."""
    store = HippocampalStore(str(tmp_path / "test_db"))
    sub_keys = [
        k for k, _ in store.db.create_read_stream(
            start="memory/spo/", end="memory/spo/\x7f",
        )
    ]
    subclass_keys = [k for k in sub_keys if "subClassOf" in k]
    assert subclass_keys, "no subClassOf triples from ontology seed"
    assert not any("\x00" in k for k in subclass_keys)
    store.close()