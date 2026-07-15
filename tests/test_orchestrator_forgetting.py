"""Phase 3b active-forget + reconsolidation API tests (orchestrator level).

Covers the user-triggered forgetting surface added in step 7:

* ``orchestrator.forget(eid)`` — episode-level deprecate (``state="deprecated"``);
  the episode drops out of default retrieval, its content is NOT deleted, and
  the deprecation is reversible.
* ``orchestrator.reconsolidate(old, new)`` — the MVCC supersession chain:
  ``(new, supersedes, old)`` + ``(old, superseded_by, new)`` graph edges + old
  ``state="superseded"`` + ``validity_end``; default queries exclude the old
  episode.
* ``signal`` threading — ``orchestrator.query(..., signal="important")`` reaches
  the retrieval-boost hook so a query-matched edge gains a sidecar.

Mirrors ``tests/test_orchestrator.py`` (tmp_path store, stub planner/embedder/
mode_a, ReferenceSSM backbone). No GLiNER, no Bonsai, no GPU.
"""

from __future__ import annotations

import hashlib

from src.config import Phase2cConfig
from src.memory.episode import Episode
from src.memory.store import HippocampalStore, _b2s
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig


# ── stubs (mirror test_orchestrator.py) ──

class _StubEmbedder:
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


class _StubPlanner:
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return self._plan


class _StubModeA:
    def __init__(self, reply: str = "SYNTH RESPONSE") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def _complete(self, messages: list[dict], tools=None, tool_choice=None) -> tuple:
        self.calls.append(messages)
        return self.reply, None


def _ep(eid, entities=None, topics=None, summary=None, text=None,
        ts="2026-07-03T10:00:00") -> Episode:
    return Episode(
        id=eid, timestamp=ts,
        summary=summary or f"summary {eid}",
        full_text=text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [],
    )


def _orchestrator(
    tmp_path, plan: dict, episodes: list[Episode],
    user_id: str | None = "victor",
) -> tuple[PonderOrchestrator, HippocampalStore]:
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in episodes:
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                      embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=_StubModeA(), config=cfg,
        user_id=user_id,
    )
    return orch, store


def _ids(episodes: list[dict]) -> set[str]:
    return {e["episode_id"] for e in episodes}


# ── active-forget ──

def test_forget_excludes_episode_from_default_retrieval(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [
        _ep("ep_001", entities=["Alice"], summary="Alice likes Postgres"),
        _ep("ep_002", entities=["Alice"], summary="Alice likes MySQL"),
    ]
    orch, store = _orchestrator(tmp_path, plan, eps)

    # baseline: both Alice episodes retrievable.
    assert _ids(orch.retriever.retrieve("Alice")) == {"ep_001", "ep_002"}

    orch.forget("ep_001")
    # deprecated -> excluded from default retrieval (state filter).
    assert _ids(orch.retriever.retrieve("Alice")) == {"ep_002"}
    store.close()


def test_forget_does_not_delete_content(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice likes Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps)

    orch.forget("ep_001")
    # content key still present (not deleted).
    assert _b2s(store.db.get_sync("content/ep/ep_001/summary")) == "Alice likes Postgres"
    # state was written.
    assert store.episode_state("ep_001") == "deprecated"
    store.close()


def test_forget_is_reversible(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice likes Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps)

    orch.forget("ep_001")
    assert _ids(orch.retriever.retrieve("Alice")) == set()
    # revive via the store's write path (the orchestrator exposes forget only;
    # revival is a direct state set, mirroring how the API is symmetric).
    store.set_episode_state("ep_001", "current", validity_end=None)
    # validity_end was never set by forget (no validity_end arg), so the
    # episode is active again once state flips back to current.
    assert _ids(orch.retriever.retrieve("Alice")) == {"ep_001"}
    store.close()


def test_forget_no_store_is_noop(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"])]
    # Construct an orchestrator with store=None (WM-only) by bypassing the
    # fixture: build a store-less orchestrator directly.
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in eps:
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                      embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=None, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=_StubModeA(), config=cfg,
        user_id=None,
    )
    # forget with no store must not raise.
    orch.forget("ep_001")
    # state unchanged.
    assert store.episode_state("ep_001") == "current"
    store.close()


# ── reconsolidation ──

def test_reconsolidate_writes_supersedes_chain_and_supersedes_old(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [
        _ep("ep_old", entities=["Alice"], summary="Alice likes Postgres"),
        _ep("ep_new", entities=["Alice"], summary="Alice now likes MySQL"),
    ]
    orch, store = _orchestrator(tmp_path, plan, eps)

    orch.reconsolidate("ep_old", "ep_new")
    # forward chain link in the graph: ep_new -supersedes-> ep_old
    q = store.graph.query().vertex("ep_new").out("supersedes")
    result = q.execute_sync()
    try:
        superseded = list(result.vertices)
    finally:
        result.close()
    assert superseded == ["ep_old"]
    # back-pointer: ep_old -superseded_by-> ep_new
    q = store.graph.query().vertex("ep_old").out("superseded_by")
    result = q.execute_sync()
    try:
        back = list(result.vertices)
    finally:
        result.close()
    assert back == ["ep_new"]
    # old episode state superseded + validity_end set.
    assert store.episode_state("ep_old") == "superseded"
    assert store.episode_validity_end("ep_old") is not None
    # new episode untouched (still current).
    assert store.episode_state("ep_new") == "current"
    store.close()


def test_reconsolidate_excludes_old_from_default_queries(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [
        _ep("ep_old", entities=["Alice"], summary="Alice likes Postgres"),
        _ep("ep_new", entities=["Alice"], summary="Alice now likes MySQL"),
    ]
    orch, store = _orchestrator(tmp_path, plan, eps)

    assert _ids(orch.retriever.retrieve("Alice")) == {"ep_old", "ep_new"}
    orch.reconsolidate("ep_old", "ep_new")
    # old drops out; new stays.
    assert _ids(orch.retriever.retrieve("Alice")) == {"ep_new"}
    store.close()


def test_reconsolidate_no_store_is_noop(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_old", entities=["Alice"]))
    store.encode_episode(_ep("ep_new", entities=["Alice"]))
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                      embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=None, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=_StubModeA(), config=cfg,
        user_id=None,
    )
    orch.reconsolidate("ep_old", "ep_new")  # must not raise
    assert store.episode_state("ep_old") == "current"
    store.close()


# ── signal threading through query -> retrieval boost ──

def test_query_signal_threads_to_retrieval_boost(tmp_path):
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Alice"], summary="Alice likes Postgres")]
    orch, store = _orchestrator(tmp_path, plan, eps)

    orch.query("What did Alice say?", signal="important")
    # the matched Alice edge got a retrieval-boost sidecar.
    meta = store.get_edge_meta("ep_001", "has_entity", "E:Alice")
    assert meta["access_count"] == 1
    # important => boosted down (more persistent than baseline 0.01).
    assert meta["utility_decay_rate"] < 0.01
    store.close()


def test_query_default_signal_is_routine_and_boosts_less_than_important(tmp_path):
    # routine (modifier 1.0) still boosts; important (1.5) boosts more. The
    # default ``signal="routine"`` must thread through to the hook, and an
    # explicit ``important`` must persist more (lower decay).
    plan = {"entities": ["Alice"], "entity_mode": "union"}

    orch_r, store_r = _orchestrator(tmp_path / "r", plan,
                                    [_ep("ep_001", entities=["Alice"])])
    orch_r.query("What did Alice say?")  # default signal -> routine
    routine = store_r.get_edge_meta("ep_001", "has_entity", "E:Alice")["utility_decay_rate"]
    store_r.close()

    orch_i, store_i = _orchestrator(tmp_path / "i", plan,
                                    [_ep("ep_001", entities=["Alice"])])
    orch_i.query("What did Alice say?", signal="important")
    important = store_i.get_edge_meta("ep_001", "has_entity", "E:Alice")["utility_decay_rate"]
    store_i.close()

    # routine threaded + boosted (below baseline)...
    assert routine < 0.01
    # ...and important persists more than routine.
    assert important < routine