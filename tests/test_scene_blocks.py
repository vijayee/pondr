"""Tests for scene blocks (B1) -- the LLM-authored topic-level macro-memory layer.

Covers three layers:

1. Store layer (``src/memory/store.py``): ``encode_scene``/``get_scene`` round-
   trip, ``scene_ids_for_user`` two-user exclusion, ``default_scene_ids`` empty-
   when-none + sorted, ``delete_scene`` symmetric-reversal (no content/spo/
   vector residue), ``touch_scene`` RMW + clamp.
2. Retrieval layer: ``_hydrate_scene`` 12-key base + ``kind="scene"`` + topics,
   ``_filter_user_scope`` ``scene_`` branch, ``_filter_vector_hits_by_scope``
   ``scene_`` branch, ``build_context_string`` ``kind=="scene"`` branch.
3. Authoring worker (``src/subconscious/scene_worker.py``): the four-action gate
   with a stubbed decider (CREATE/UPDATE/MERGE/skip/None), the ``maxScenes`` cap
   + coldest-eviction, and heat decay + below-floor eviction (macro-forgetting).

Plus THE byte-identical-OFF gate: with the flag off (no ``SceneAuthoringWorker``),
no scene writes occur, ``default_scene_ids()`` is empty, no ``scene_*`` id enters
the graph or vector index, and ``retrieve()`` is byte-identical to pre-B1.

Offline throughout: the worker uses a ``_StubDecider`` (no Bonsai HTTP). A live
LLM gate is sketched but SKIPPED when the local Bonsai server is down (mirrors
the tier-2 live-gate pattern -- the offline suite pins the contract; the live
gate is a smoke test, not a correctness gate).
"""

from __future__ import annotations

import hashlib
import json
import time

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)",
                allow_module_level=True)

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.scene_worker import SceneAuthoringWorker


# ── stubs ──

class _StubEmbedder:
    """Deterministic 384-dim embedder (SHA256 stretch -> normalized). Mirrors
    ``tests/test_fade_serve_integration._StubEmbedder`` so scenes embed the same
    way the WM bge-small embedder would (384-d, cosine)."""
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            buf = bytearray()
            h = hashlib.sha256(t.encode("utf-8")).digest()
            counter = 0
            while len(buf) < self.dim:
                buf += hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
                counter += 1
            vec = [(b / 127.5 - 1.0) for b in buf[: self.dim]]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


class _StubDecider:
    """A stub ``author_scene`` decider for the four-action gate. ``queue`` is a
    list of verdict dicts consumed FIFO; each call pops one. ``calls`` records
    every invocation (the offered-input distinction -- asserts the worker OFFERED
    the merge candidate even when the model picks another action). ``None`` in the
    queue models cold-start / Bonsai-down (defer, no write)."""
    def __init__(self, queue: list | None = None) -> None:
        self._queue = list(queue or [])
        self.calls: list[dict] = []

    def author_scene(self, topic, existing_body, candidate_summaries, user_id,
                     heat_budget, merge_candidate=None):
        self.calls.append({
            "topic": topic, "existing_body": existing_body,
            "candidate_summaries": list(candidate_summaries),
            "user_id": user_id, "heat_budget": heat_budget,
            "merge_candidate": (dict(merge_candidate) if merge_candidate else None),
        })
        if not self._queue:
            return None
        return self._queue.pop(0)


class _TopicPlanner:
    """Stub planner: a topics-axis plan (the shared ``storage`` topic -- the axis
    scenes share with episodes). A topics-only plan (no entity axis) so a
    non-existent entity never collapses the candidate set (``_find_candidates``
    returns empty when a specified entity fails to match -- a pre-existing
    quirk unrelated to scene blocks)."""
    def plan(self, prompt, history=None):
        return {"topics": ["storage"], "entity_mode": "union", "limit": 20}


# ── fixtures ──

def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": True, "embedding_dim": 384}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


_EMBED = _StubEmbedder()


def _embed(text: str) -> list[float]:
    return _EMBED.encode([text])[0]


def _vsearch(store, query: str, k: int = 10) -> list[tuple[str, float]]:
    """Vector search via the in-DB layer (mirrors ``wavedb_vector_store.search``:
    embed the query, ``search_sync``, convert distance -> similarity). Returns
    ``[]`` when the vector layer is absent/disabled."""
    vl = getattr(store, "vector_layer", None)
    if vl is None:
        return []
    results = vl.search_sync(_embed(query), k)
    return [(r.id_str, 1.0 - float(r.distance)) for r in results]


def _encode_scene(store, sid, *, body, topic, heat, user_id, source_eps,
                  updated_ts="2026-08-01T10:00:00"):
    store.encode_scene(sid, body=body, topic=topic, heat=heat,
                       updated_ts=updated_ts, user_id=user_id,
                       source_eps=source_eps, body_embedding=_embed(body))


def _worker(store, decider, **kw):
    w = SceneAuthoringWorker(store, decider, _StubEmbedder(), **kw)
    # No foreground contention in tests -- leave the default (unset) event.
    return w


def _drain(w):
    w.drain(timeout=5.0)


def _wait_queue():
    # The worker is a daemon thread; give it a beat to process enqueued jobs.
    time.sleep(0.1)


# ──────────────────────────────────────────────────────────────────────────
# 1. Store layer
# ──────────────────────────────────────────────────────────────────────────

def test_encode_get_scene_roundtrip(tmp_path):
    store = _store(tmp_path)
    sid = store.next_scene_id()
    assert sid == "scene_000001"
    _encode_scene(store, sid, body="# storage notes", topic="storage",
                  heat=0.8, user_id="alice", source_eps=["ep_001", "ep_002"])
    sc = store.get_scene(sid)
    assert sc is not None
    assert sc["scene_id"] == sid
    assert sc["body"] == "# storage notes"
    assert sc["topic"] == "storage"
    assert abs(sc["heat"] - 0.8) < 1e-6
    assert sc["user_id"] == "alice"
    assert sc["source_eps"] == ["ep_001", "ep_002"]
    # The four edge types are present in the graph index.
    assert store.db.get_sync(f"memory/spo/{sid}/has_topic/T:storage") is not None
    assert store.db.get_sync(f"memory/spo/U:alice/owns_scene/{sid}") is not None
    assert store.db.get_sync(f"memory/spo/{sid}/cites/ep_001") is not None
    assert store.db.get_sync(f"memory/spo/{sid}/instanceOf/SceneBlock") is not None
    store.close()


def test_next_scene_id_increments(tmp_path):
    store = _store(tmp_path)
    assert store.next_scene_id() == "scene_000001"
    assert store.next_scene_id() == "scene_000002"
    assert store.next_scene_id() == "scene_000003"
    store.close()


def test_scene_ids_for_user_two_user_exclusion(tmp_path):
    store = _store(tmp_path)
    a1 = store.next_scene_id(); a2 = store.next_scene_id(); b1 = store.next_scene_id()
    _encode_scene(store, a1, body="a1", topic="t1", heat=0.5, user_id="alice",
                  source_eps=[])
    _encode_scene(store, a2, body="a2", topic="t2", heat=0.5, user_id="alice",
                  source_eps=[])
    _encode_scene(store, b1, body="b1", topic="t3", heat=0.5, user_id="bob",
                  source_eps=[])
    assert store.scene_ids_for_user("alice") == {a1, a2}
    assert store.scene_ids_for_user("bob") == {b1}
    assert store.scene_ids_for_user("nobody") == set()
    store.close()


def test_default_scene_ids_empty_and_sorted(tmp_path):
    store = _store(tmp_path)
    assert store.default_scene_ids() == []  # cold start
    a = store.next_scene_id(); b = store.next_scene_id(); c = store.next_scene_id()
    assert (a, b, c) == ("scene_000001", "scene_000002", "scene_000003")
    # Encode out of order to confirm the scan SORTS (b, c, a insertion).
    _encode_scene(store, b, body="b", topic="t", heat=0.5, user_id="u",
                  source_eps=[])
    _encode_scene(store, c, body="c", topic="t", heat=0.5, user_id="u",
                  source_eps=[])
    _encode_scene(store, a, body="a", topic="t", heat=0.5, user_id="u",
                  source_eps=[])
    assert store.default_scene_ids() == [a, b, c]  # sorted by id
    store.close()


def test_delete_scene_leaves_no_residue(tmp_path):
    """The eviction chokepoint: symmetric-reversal + content + vector unindex."""
    store = _store(tmp_path)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="body to embed", topic="storage", heat=0.9,
                  user_id="alice", source_eps=["ep_001"])
    assert sid in store.default_scene_ids()
    # Vector entry present.
    hits = _vsearch(store, "body to embed", k=10)
    assert any(h[0] == sid for h in hits)
    store.delete_scene(sid)
    # No content keys.
    assert store.get_scene(sid) is None
    assert store.default_scene_ids() == []
    # No SPO/POS/OSP edge residue for ANY of the four edge types.
    for k, _ in store.db.create_read_stream(start=f"memory/spo/{sid}/",
                                            end=f"memory/spo/{sid}/\x7f"):
        pytest.fail(f"orphan SPO edge after delete: {k}")
    # The forward owns_scene edge is gone too.
    assert store.db.get_sync(f"memory/spo/U:alice/owns_scene/{sid}") is None
    assert store.db.get_sync(f"memory/spo/{sid}/cites/ep_001") is None
    # No vector residue.
    hits = _vsearch(store, "body to embed", k=10)
    assert not any(h[0] == sid for h in hits)
    store.close()


def test_touch_scene_rmw_and_clamp(tmp_path):
    store = _store(tmp_path)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="b", topic="t", heat=0.5, user_id="u",
                  source_eps=[])
    store.touch_scene(sid, delta=0.2)
    assert abs(store.get_scene(sid)["heat"] - 0.7) < 1e-6
    # Clamp at 1.0.
    store.touch_scene(sid, delta=10.0)
    assert store.get_scene(sid)["heat"] == 1.0
    # Clamp at 0.0 + missing-scene no-op.
    store.touch_scene(sid, delta=-10.0)
    assert store.get_scene(sid)["heat"] == 0.0
    store.touch_scene("scene_999999", delta=0.5)  # missing -> no-op, no raise
    store.close()


# ──────────────────────────────────────────────────────────────────────────
# 2. Retrieval layer
# ──────────────────────────────────────────────────────────────────────────

def test_hydrate_scene_12_keys_and_topics(tmp_path):
    store = _store(tmp_path)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="# storage macro", topic="storage",
                  heat=0.7, user_id="alice", source_eps=["ep_001"])
    trav = GraphTraversal(store)
    r = trav._hydrate_scene(sid)
    base_keys = {"episode_id", "summary", "text", "timestamp", "entities",
                 "topics", "tones", "decisions", "session_id", "user_id",
                 "follows", "score"}
    assert base_keys <= set(r.keys())
    assert r["kind"] == "scene"
    assert r["episode_id"] == sid
    assert r["summary"] == "storage"        # topic is the handle
    assert r["text"] == "# storage macro"    # body is the content
    assert r["user_id"] == "alice"
    assert r["heat"] == 0.7
    assert r["source_eps"] == ["ep_001"]
    assert "storage" in r["topics"]          # from the has_topic edge
    assert r["entities"] == [] and r["tones"] == [] and r["decisions"] == []
    store.close()


def test_hydrate_scene_missing_fallback(tmp_path):
    store = _store(tmp_path)
    trav = GraphTraversal(store)
    r = trav._hydrate_scene("scene_999999")
    assert r["kind"] == "scene"
    assert r["text"] == "" and r["topics"] == [] and r["heat"] == 0.0
    store.close()


def test_filter_user_scope_scene_branch(tmp_path):
    """``_filter_user_scope`` keeps owned scenes, drops another user's, passes
    all when ``allowed_scene_ids is None`` (scope off)."""
    a = {"scene_000001"}; b = {"scene_000002"}
    cand = a | b | {"ep_001"}
    # alice scoped -> only alice's scene + the episode (ep allowed None).
    out = GraphTraversal._filter_user_scope(
        cand, allowed_episode_ids=None, allowed_document_ids=None,
        allowed_scene_ids=a)
    assert "scene_000001" in out and "scene_000002" not in out
    assert "ep_001" in out  # episodes pass (ep allowed None)
    # scene scope off (None) -> all scenes pass.
    out = GraphTraversal._filter_user_scope(
        cand, allowed_episode_ids=None, allowed_document_ids=None,
        allowed_scene_ids=None)
    assert out == cand
    # strict scope on (all sets set) -> M: dropped.
    out = GraphTraversal._filter_user_scope(
        {"scene_000001", "M:mem1"}, allowed_episode_ids=set(),
        allowed_document_ids=set(), allowed_scene_ids=a)
    assert "M:mem1" not in out and "scene_000001" in out


def test_filter_vector_hits_by_scope_scene_branch(tmp_path):
    """The vector path enforces scene scope (D1 -- scenes are in the index)."""
    hits = [("scene_000001", 0.9), ("scene_000002", 0.8), ("ep_001", 0.7)]
    # alice scoped -> only alice's scene + the episode.
    out = HippocampalRetriever._filter_vector_hits_by_scope(
        hits, allowed_episode_ids={"ep_001"}, allowed_document_ids=set(),
        allowed_scene_ids={"scene_000001"})
    assert [h[0] for h in out] == ["scene_000001", "ep_001"]
    # scope off (all None) -> byte-identical.
    out = HippocampalRetriever._filter_vector_hits_by_scope(
        hits, allowed_episode_ids=None, allowed_document_ids=None,
        allowed_scene_ids=None)
    assert out == hits


def test_build_context_string_scene_branch(tmp_path):
    store = _store(tmp_path)
    retr = HippocampalRetriever(store, planner=_TopicPlanner(), user_id=None)
    scene = {
        "episode_id": "scene_000001", "kind": "scene",
        "summary": "storage", "text": "# storage macro\nlots of detail",
        "timestamp": "2026-08-01T10:00:00Z", "topics": ["storage"],
        "heat": 0.7, "entities": [], "tones": [], "decisions": [],
    }
    ctx = retr.build_context_string([scene], max_tokens=1000)
    assert "scene_000001" in ctx
    assert "Scene (topic: storage, heat: 0.70)" in ctx
    assert "# storage macro" in ctx
    store.close()


def test_scenes_surface_on_topics_axis(tmp_path):
    """A scene shares the ``T:{topic}`` axis with episodes, so a topics-axis query
    surfaces it via the SAME retrieve pipeline (not a separate lane)."""
    store = _store(tmp_path)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="# storage macro", topic="storage",
                  heat=0.9, user_id="alice", source_eps=["ep_001"])
    trav = GraphTraversal(store)
    results = trav.retrieve({"topics": ["storage"], "entity_mode": "union",
                            "limit": 20})
    ids = {r["episode_id"] for r in results}
    assert sid in ids
    assert any(r["kind"] == "scene" for r in results if r["episode_id"] == sid)
    store.close()


def test_retriever_scene_scope_two_users(tmp_path):
    """alice's scene is a candidate for alice's query, NOT bob's; the vector path
    (D1) + graph path both enforce it."""
    store = _store(tmp_path)
    a = store.next_scene_id(); b = store.next_scene_id()
    _encode_scene(store, a, body="alice on storage", topic="storage",
                  heat=0.9, user_id="alice", source_eps=[])
    _encode_scene(store, b, body="bob on storage", topic="storage",
                  heat=0.9, user_id="bob", source_eps=[])
    # Graph path (topics axis).
    retr_a = HippocampalRetriever(store, planner=_TopicPlanner(), user_id="alice")
    ids = {r["episode_id"] for r in retr_a.retrieve("storage", use_semantic=False)}
    assert a in ids and b not in ids
    # Vector path: alice's vector search returns only alice's scene.
    hits = _vsearch(store, "storage", k=10)
    scene_hits = [h[0] for h in hits if h[0].startswith("scene_")]
    assert a in scene_hits and b in scene_hits  # index is global
    filtered = HippocampalRetriever._filter_vector_hits_by_scope(
        hits, allowed_episode_ids=set(), allowed_document_ids=set(),
        allowed_scene_ids=store.scene_ids_for_user("alice"))
    fids = {h[0] for h in filtered}
    assert a in fids and b not in fids
    store.close()


# ──────────────────────────────────────────────────────────────────────────
# 3. Byte-identical-OFF (THE gate)
# ──────────────────────────────────────────────────────────────────────────

def test_byte_identical_off_no_scenes(tmp_path):
    """Flag off -> no worker -> no scene writes -> ``default_scene_ids()`` empty
    -> no ``scene_*`` id in graph or vector index -> retrieve() byte-identical to
    pre-B1. This is the contract the ``build_ponder`` gate upholds."""
    store = _store(tmp_path)
    # Populate an episode (the pre-B1 corpus).
    store.encode_episode(Episode(
        id="ep_001", timestamp="2026-08-01T10:00:00",
        summary="We chose Postgres", full_text="User: storage?\nAssistant: Postgres",
        entities=["Postgres"], topics=["storage"], tones=[], decisions=[],
        user_id="alice", session_id="S:a1",
    ))
    assert store.default_scene_ids() == []  # no scenes
    # No scene_ id in the graph SPO index.
    for k, _ in store.db.create_read_stream(start="memory/spo/scene_",
                                            end="memory/spo/scene_\x7f"):
        pytest.fail(f"scene_ edge present with flag off: {k}")
    # Vector search returns zero scene_* ids.
    hits = _vsearch(store, "storage", k=20)
    assert not any(h[0].startswith("scene_") for h in hits)
    # retrieve() returns only the episode.
    retr = HippocampalRetriever(store, planner=_TopicPlanner(), user_id="alice")
    results = retr.retrieve("storage", use_semantic=False)
    ids = {r["episode_id"] for r in results}
    assert "ep_001" in ids
    assert not any(i.startswith("scene_") for i in ids)
    store.close()


# ──────────────────────────────────────────────────────────────────────────
# 4. Authoring worker -- the four-action gate
# ──────────────────────────────────────────────────────────────────────────

def _batch_eps(store, n=2, user="alice", topic="storage"):
    """Encode n episodes for a user (the ingest batch the worker authors from)."""
    for i in range(n):
        store.encode_episode(Episode(
            id=f"ep_{i+1:03d}", timestamp="2026-08-01T10:00:00",
            summary=f"notes on {topic} part {i}", full_text=f"text {i}",
            entities=[], topics=[topic], tones=[], decisions=[],
            user_id=user, session_id="S:a1",
        ))
    return [f"ep_{i+1:03d}" for i in range(n)]


def test_worker_create_writes_scene(tmp_path):
    store = _store(tmp_path)
    eps = _batch_eps(store)
    decider = _StubDecider([{
        "action": "CREATE", "topic": "storage",
        "body": "# storage scene", "merge_with": None, "reason": "fresh",
    }])
    w = _worker(store, decider)
    assert w.tick("alice", eps, "storage") == 1
    _wait_queue(); _drain(w)
    assert len(decider.calls) == 1
    assert decider.calls[0]["topic"] == "storage"
    assert decider.calls[0]["heat_budget"] == 24
    sids = store.default_scene_ids()
    assert len(sids) == 1
    sc = store.get_scene(sids[0])
    assert sc["body"] == "# storage scene"
    assert sc["topic"] == "storage"
    assert sc["user_id"] == "alice"
    assert sc["source_eps"] == eps
    assert sc["heat"] == 1.0
    store.close()


def test_worker_update_bumps_heat_and_unions_source(tmp_path):
    store = _store(tmp_path)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="old body", topic="storage", heat=0.4,
                  user_id="alice", source_eps=["ep_001"])
    eps = ["ep_001", "ep_002"]  # ep_001 already cited, ep_002 new
    _batch_eps(store)
    decider = _StubDecider([{
        "action": "UPDATE", "topic": "storage", "body": "new body",
        "merge_with": None, "reason": "revise",
    }])
    w = _worker(store, decider)
    w.tick("alice", eps, "storage")
    _wait_queue(); _drain(w)
    sc = store.get_scene(sid)
    assert sc["body"] == "new body"            # replaced
    assert sc["topic"] == "storage"            # stable (existing topic enforced)
    assert set(sc["source_eps"]) == {"ep_001", "ep_002"}  # unioned
    assert abs(sc["heat"] - 0.6) < 1e-6         # 0.4 + 0.2 bump, < 1.0
    assert len(store.default_scene_ids()) == 1  # no new scene
    store.close()


def test_worker_merge_deletes_source(tmp_path):
    """D5 + D7: the worker offers ONE merge candidate; MERGE folds onto the
    target and DELETES the source scene. A verdict merging into an unoffered
    target is deferred (no write)."""
    store = _store(tmp_path)
    # Source scene (the topic's existing scene).
    src = store.next_scene_id()
    _encode_scene(store, src, body="storage notes", topic="storage cluster",
                  heat=0.5, user_id="alice", source_eps=["ep_001"])
    # Target scene (an overlapping topic the worker will offer as merge candidate).
    tgt = store.next_scene_id()
    _encode_scene(store, tgt, body="cluster architecture", topic="storage cluster design",
                  heat=0.6, user_id="alice", source_eps=["ep_002"])
    _batch_eps(store)
    eps = ["ep_001", "ep_002", "ep_003"]
    # The worker pre-filters the merge candidate (lexical overlap "storage cluster").
    decider = _StubDecider([{
        "action": "MERGE", "topic": "storage cluster", "body": "merged body",
        "merge_with": tgt, "reason": "fold",
    }])
    w = _worker(store, decider)
    w.tick("alice", eps, "storage cluster")
    _wait_queue(); _drain(w)
    # The merge candidate offered was the OTHER scene (not the existing one).
    assert decider.calls[0]["merge_candidate"] is not None
    assert decider.calls[0]["merge_candidate"]["scene_id"] == tgt
    # Source deleted, target rewritten.
    assert store.get_scene(src) is None
    t = store.get_scene(tgt)
    assert t["body"] == "merged body"
    assert set(t["source_eps"]) == {"ep_001", "ep_002", "ep_003"}
    assert t["heat"] == 0.8  # 0.6 + 0.2
    assert store.default_scene_ids() == [tgt]
    store.close()


def test_worker_merge_unoffered_target_defers(tmp_path):
    """A verdict that merges into an unoffered scene defers (never writes)."""
    store = _store(tmp_path)
    src = store.next_scene_id()
    _encode_scene(store, src, body="storage notes", topic="storage cluster",
                  heat=0.5, user_id="alice", source_eps=["ep_001"])
    tgt = store.next_scene_id()
    _encode_scene(store, tgt, body="cluster architecture", topic="storage cluster design",
                  heat=0.6, user_id="alice", source_eps=["ep_002"])
    bogus = store.next_scene_id()
    _encode_scene(store, bogus, body="unrelated", topic="unrelated topic",
                  heat=0.9, user_id="alice", source_eps=[])
    _batch_eps(store)
    decider = _StubDecider([{
        "action": "MERGE", "topic": "storage cluster", "body": "merged body",
        "merge_with": bogus, "reason": "wrong target",  # NOT the offered candidate
    }])
    w = _worker(store, decider)
    w.tick("alice", ["ep_001", "ep_002", "ep_003"], "storage cluster")
    _wait_queue(); _drain(w)
    # No mutation -- source + target + bogus all unchanged.
    assert store.get_scene(src)["body"] == "storage notes"
    assert store.get_scene(tgt)["body"] == "cluster architecture"
    assert store.get_scene(bogus)["body"] == "unrelated"
    store.close()


def test_worker_merge_no_candidate_offered_defers(tmp_path):
    """D7: if the worker offered NO merge candidate (no overlapping scene), a
    MERGE verdict (with any ``merge_with``) MUST defer -- the model can't invent
    a merge target. Pins the ``merge_candidate is None`` guard in ``_merge``."""
    store = _store(tmp_path)
    src = store.next_scene_id()
    _encode_scene(store, src, body="storage notes", topic="storage",
                  heat=0.5, user_id="alice", source_eps=["ep_001"])
    # A scene the model might try to invent as a target (no lexical overlap with
    # "storage" -> NOT offered as a merge candidate by _pick_merge_candidate).
    other = store.next_scene_id()
    _encode_scene(store, other, body="unrelated topic body", topic="zzz unrelated",
                  heat=0.9, user_id="alice", source_eps=["ep_002"])
    _batch_eps(store, n=2, user="alice", topic="storage")
    decider = _StubDecider([{
        "action": "MERGE", "topic": "storage", "body": "merged body",
        "merge_with": other, "reason": "invented target",  # no candidate offered
    }])
    w = _worker(store, decider)
    w.tick("alice", ["ep_001", "ep_002"], "storage")
    _wait_queue(); _drain(w)
    # The offered merge_candidate was None (no overlap) -> defer, no mutation.
    assert decider.calls[0]["merge_candidate"] is None
    assert store.get_scene(src)["body"] == "storage notes"
    assert store.get_scene(other)["body"] == "unrelated topic body"
    store.close()


def test_worker_skip_defers(tmp_path):
    store = _store(tmp_path)
    eps = _batch_eps(store)
    decider = _StubDecider([{"action": "skip", "body": "", "topic": "storage"}])
    w = _worker(store, decider)
    w.tick("alice", eps, "storage")
    _wait_queue(); _drain(w)
    assert store.default_scene_ids() == []  # no write
    store.close()


def test_worker_none_defers(tmp_path):
    """Cold-start / Bonsai-down: ``None`` verdict -> defer, no fabricated scene."""
    store = _store(tmp_path)
    eps = _batch_eps(store)
    decider = _StubDecider([None])
    w = _worker(store, decider)
    w.tick("alice", eps, "storage")
    _wait_queue(); _drain(w)
    assert store.default_scene_ids() == []
    assert len(decider.calls) == 1  # the worker DID call (offered the inputs)
    store.close()


def test_worker_tick_noop_conditions(tmp_path):
    """``tick`` is a no-op with no user, no batch, or no topic hint (dedup too)."""
    store = _store(tmp_path)
    decider = _StubDecider([])
    w = _worker(store, decider)
    assert w.tick(None, ["ep_001"], "storage") == 0
    assert w.tick("alice", [], "storage") == 0
    assert w.tick("alice", ["ep_001"], None) == 0
    assert w.tick("alice", ["ep_001"], "") == 0
    assert w.tick("alice", ["ep_001"], "storage") == 1
    assert w.tick("alice", ["ep_001"], "storage") == 0  # dedup
    _drain(w)
    store.close()


# ──────────────────────────────────────────────────────────────────────────
# 5. Heat / eviction (macro-forgetting)
# ──────────────────────────────────────────────────────────────────────────

def test_decay_tick_multiplies_heat(tmp_path):
    store = _store(tmp_path)
    s = store.next_scene_id()
    _encode_scene(store, s, body="b", topic="t", heat=1.0, user_id="alice",
                  source_eps=[])
    w = _worker(store, _StubDecider([]), heat_decay=0.5)
    w.decay_tick()
    assert abs(store.get_scene(s)["heat"] - 0.5) < 1e-6
    w.decay_tick()
    assert abs(store.get_scene(s)["heat"] - 0.25) < 1e-6
    _drain(w)
    store.close()


def test_decay_evicts_below_floor(tmp_path):
    store = _store(tmp_path)
    s = store.next_scene_id()
    _encode_scene(store, s, body="b", topic="t", heat=0.1, user_id="alice",
                  source_eps=[])
    w = _worker(store, _StubDecider([]), heat_decay=0.5, heat_floor=0.05)
    w.decay_tick()  # 0.1 * 0.5 = 0.05 ... < floor 0.05? 0.05 < 0.05 is False
    # 0.05 is NOT below floor 0.05 (strict <), so survives this tick.
    assert store.get_scene(s) is not None
    w.decay_tick()  # 0.05 * 0.5 = 0.025 < 0.05 -> evict
    assert store.get_scene(s) is None
    assert s not in store.default_scene_ids()
    # No orphan owns_scene edge.
    assert store.db.get_sync(f"memory/spo/U:alice/owns_scene/{s}") is None
    _drain(w)
    store.close()


def test_create_at_cap_evicts_coldest(tmp_path):
    store = _store(tmp_path)
    # Fill to cap=2 with two scenes, one cold (0.2) one warm (0.8).
    cold = store.next_scene_id(); warm = store.next_scene_id()
    _encode_scene(store, cold, body="cold", topic="cold topic", heat=0.2,
                  user_id="alice", source_eps=[])
    _encode_scene(store, warm, body="warm", topic="warm topic", heat=0.8,
                  user_id="alice", source_eps=[])
    _batch_eps(store)
    decider = _StubDecider([{
        "action": "CREATE", "topic": "new topic", "body": "new body",
        "merge_with": None, "reason": "fresh",
    }])
    w = _worker(store, decider, max_scenes_per_user=2)
    w.tick("alice", ["ep_001", "ep_002"], "new topic")
    _wait_queue(); _drain(w)
    ids = set(store.default_scene_ids())
    assert cold not in ids          # coldest evicted
    assert warm in ids              # warm kept
    assert len(ids) == 2            # cap held
    # New scene landed.
    assert any(store.get_scene(i)["body"] == "new body" for i in ids)
    # No orphan edge for the evicted scene.
    assert store.db.get_sync(f"memory/spo/U:alice/owns_scene/{cold}") is None
    store.close()


def test_decay_tick_eviction_bound(tmp_path):
    """``max_evict_per_tick`` bounds evictions so a large scene store can't stall
    the gap between turns."""
    store = _store(tmp_path)
    sids = []
    for i in range(4):
        s = store.next_scene_id()
        sids.append(s)
        _encode_scene(store, s, body=f"b{i}", topic=f"t{i}", heat=0.1,
                      user_id="alice", source_eps=[])
    w = _worker(store, _StubDecider([]), heat_decay=0.5, heat_floor=0.05,
                max_evict_per_tick=2)
    evicted = w.decay_tick()  # 0.1*0.5=0.05 not < 0.05 -> none this tick
    assert evicted == 0
    evicted = w.decay_tick()  # 0.05*0.5=0.025 < 0.05 -> 4 candidates, bound 2
    assert evicted == 2
    assert len(store.default_scene_ids()) == 2  # 2 remain for next tick
    _drain(w)
    store.close()


# ──────────────────────────────────────────────────────────────────────────
# 6. Live LLM gate (SKIPPED when Bonsai down)
# ──────────────────────────────────────────────────────────────────────────

def test_live_bonsai_author_and_retrieve(tmp_path):
    """Live smoke gate: ingest a multi-episode batch, the real Bonsai LLM authors
    a scene, and it is retrievable via BOTH the topics axis AND the vector path
    (D1), with two-user scope holding end-to-end. SKIPPED when the local Bonsai
    server is down (mirrors the tier-2 live-gate pattern -- the offline suite pins
    the contract; this is a smoke test)."""
    from src.gnn.bonsai_decider import BonsaiDecider

    decider = BonsaiDecider()
    if not decider.health_check():
        pytest.skip("local Bonsai server not running (start scripts/start_llama_server.ps1)")

    store = _store(tmp_path)
    eps = _batch_eps(store, n=3, user="alice", topic="storage architecture")
    w = SceneAuthoringWorker(store, decider, _StubEmbedder())
    w.tick("alice", eps, "storage architecture")
    # Allow the live LLM call to complete (longer than the offline beat).
    for _ in range(40):
        time.sleep(0.25)
        if store.default_scene_ids():
            break
    _drain(w)
    sids = store.default_scene_ids()
    if not sids:
        pytest.skip("Bonsai did not author a scene (8B may have picked skip) -- "
                     "offered-tools assertion is the real gate, covered offline")
    sid = sids[0]
    sc = store.get_scene(sid)
    assert sc["user_id"] == "alice"
    # Topics-axis retrieval surfaces the scene.
    trav = GraphTraversal(store)
    results = trav.retrieve({"topics": [sc["topic"]], "entity_mode": "union",
                             "limit": 20})
    assert sid in {r["episode_id"] for r in results}
    # Two-user scope holds: bob's retrieval excludes alice's scene.
    retr_b = HippocampalRetriever(store, planner=_TopicPlanner(), user_id="bob")
    bob_ids = {r["episode_id"] for r in retr_b.retrieve("storage", use_semantic=False)}
    assert sid not in bob_ids
    store.close()