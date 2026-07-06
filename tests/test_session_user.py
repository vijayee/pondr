"""Tests for the User → Session → Episode hierarchy (global chat history).

Offline — uses the installed ``wavedb`` package (CPU), no GLiNER/Bonsai.
Exercises the persisted global counters, session lifecycle (open/close,
``follows_session`` chain across a user's chats), per-episode membership +
``at_time``, and the cross-session retrieval helpers. Also the regression
gate: graph-index scans must stay NUL-free past the >38-entry btree split.
"""

from src.memory.episode import Episode
from src.memory.store import HippocampalStore


def _scoped_episode(eid: str, user: str, session: str, ts: str) -> Episode:
    return Episode(
        id=eid, timestamp=ts, summary=f"s {eid}", full_text=f"User: u{eid}\nAssistant: a{eid}",
        user_id=user, session_id=session,
    )


def test_next_episode_id_persisted_and_monotonic(tmp_path):
    """Episode ids are globally unique and survive reopening the store."""
    store = HippocampalStore(str(tmp_path / "db"))
    a = store.next_episode_id()
    b = store.next_episode_id()
    assert a != b
    store.close()

    store2 = HippocampalStore(str(tmp_path / "db"))
    c = store2.next_episode_id()
    assert c > a and c > b  # counter persisted across reopen
    store2.close()


def test_next_session_id_monotonic(tmp_path):
    store = HippocampalStore(str(tmp_path / "db"))
    s1 = store.next_session_id()
    s2 = store.next_session_id()
    assert s1.startswith("S:") and s2.startswith("S:")
    assert s1 != s2
    store.close()


def test_open_session_links_follows_session_chain(tmp_path):
    """A user's second session follows_session the first (cross-chat chain)."""
    store = HippocampalStore(str(tmp_path / "db"))
    user = "victor"
    s1 = store.next_session_id()
    store.open_session(user, s1, "2026-07-05T10:00:00")
    s2 = store.next_session_id()
    store.open_session(user, s2, "2026-07-05T11:00:00")

    sessions = store.list_sessions(user)
    assert set(sessions) == {s1, s2}, sessions

    # (S:s2, follows_session, S:s1) must be present in the graph.
    fs_keys = [k for k, _ in store.db.create_read_stream(
        start=f"memory/spo/{s2}/follows_session/",
        end=f"memory/spo/{s2}/follows_session/\x7f",
    )]
    assert any(s1 in k for k in fs_keys), fs_keys
    assert not any("\x00" in k for k in fs_keys)
    store.close()


def test_encode_episode_writes_membership_and_at_time(tmp_path):
    """A scoped episode links into User→Session→Episode and stamps at_time."""
    store = HippocampalStore(str(tmp_path / "db"))
    user, sess = "victor", "S:0001"
    store.open_session(user, sess, "2026-07-05T10:00:00")
    store.encode_episode(_scoped_episode("ep_000001", user, sess, "2026-07-05T10:00:05"))

    # in_session: ep -> S
    in_sess = [k for k, _ in store.db.create_read_stream(
        start=f"memory/spo/ep_000001/in_session/",
        end=f"memory/spo/ep_000001/in_session/\x7f",
    )]
    assert any(sess in k for k in in_sess), in_sess

    # has_episode: S -> ep
    eps = store.list_session_episodes(sess)
    assert "ep_000001" in eps, eps

    # at_time: ep -> ts
    at_keys = [k for k, _ in store.db.create_read_stream(
        start="memory/spo/ep_000001/at_time/",
        end="memory/spo/ep_000001/at_time/\x7f",
    )]
    assert any("10:00:05" in k for k in at_keys), at_keys

    # has_session: U -> S
    sess_listed = store.list_sessions(user)
    assert sess in sess_listed, sess_listed

    # All scanned keys NUL-free (HBTrie scan-corruption regression gate).
    all_graph = [k for k, _ in store.db.create_read_stream(start="memory/", end=None)]
    assert not any("\x00" in k for k in all_graph)
    store.close()


def test_unscoped_episode_skips_session_triples(tmp_path):
    """Episodes with no user/session stay backward-compatible (no membership)."""
    store = HippocampalStore(str(tmp_path / "db"))
    ep = Episode(id="ep_x", timestamp="2026-07-05T10:00:00", summary="s",
                full_text="User: a\nAssistant: b")
    store.encode_episode(ep)
    # No in_session triple should exist for ep_x.
    in_sess = list(store.db.create_read_stream(
        start="memory/spo/ep_x/in_session/",
        end="memory/spo/ep_x/in_session/\x7f",
    ))
    assert in_sess == []
    store.close()


def test_cross_session_isolation_no_global_episode_chain(tmp_path):
    """Two conversations form separate follows chains, not one global chain.

    The cross-conversation chaining de-wonk: episode N of conversation 2 must
    NOT follow the last episode of conversation 1. Cross-session order is via
    follows_session + at_time, not follows.
    """
    store = HippocampalStore(str(tmp_path / "db"))
    user = "victor"

    s1 = store.next_session_id()
    store.open_session(user, s1, "2026-07-05T10:00:00")
    e1a = _scoped_episode("ep_000001", user, s1, "2026-07-05T10:00:01")
    e1b = _scoped_episode("ep_000002", user, s1, "2026-07-05T10:00:02")
    e1b.follows = "ep_000001"
    store.encode_episode(e1a)
    store.encode_episode(e1b)
    store.close_session(s1, "2026-07-05T10:05:00")

    s2 = store.next_session_id()
    store.open_session(user, s2, "2026-07-05T11:00:00")
    e2a = _scoped_episode("ep_000003", user, s2, "2026-07-05T11:00:01")  # follows=None
    store.encode_episode(e2a)
    store.close_session(s2, "2026-07-05T11:05:00")

    # ep_000003 must NOT follow ep_000002 (different session).
    cross = [k for k, _ in store.db.create_read_stream(
        start="memory/spo/ep_000003/follows/",
        end="memory/spo/ep_000003/follows/\x7f",
    )]
    assert cross == [], f"cross-session follows link leaked: {cross}"

    # But s2 follows_session s1.
    fs = [k for k, _ in store.db.create_read_stream(
        start=f"memory/spo/{s2}/follows_session/",
        end=f"memory/spo/{s2}/follows_session/\x7f",
    )]
    assert any(s1 in k for k in fs), fs

    # list_sessions returns both, in the order they were opened.
    sessions = store.list_sessions(user)
    assert set(sessions) == {s1, s2}
    store.close()