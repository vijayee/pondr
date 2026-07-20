"""Live-encode tests: the orchestrator persists each exchange as an episode.

Closes the runtime gap (2026-07-14): ``query()`` encodes the (prompt, response)
exchange as a new ``Episode`` via an injected ``HippocampalEncoder`` so the
system learns from use. ``signal`` modulates HOW strongly the new episode
persists (salience + decay rate), not WHETHER; ``auto_persist=False`` opts out.

The encoder is real (``HippocampalEncoder``) but its heavy extractors are
stubbed: ``object.__new__`` bypasses ``__init__``'s GLiNER model load, then we
set ``store`` / ``user_id`` / ``session_id`` / ``last_episode_id`` / ``gliner``
/ ``bonsai`` by hand. This exercises the real ``encode_messages`` /
``start_session`` / ``end_session`` path with no GLiNER, no Bonsai, no GPU.
Mirrors the stubs in ``tests/test_orchestrator.py`` (stub embedder / planner /
mode_a, tmp_path WaveDB store, ReferenceSSM backbone).
"""

from __future__ import annotations

import hashlib
import time

import pytest
import torch

from src.config import Phase2cConfig, config
from src.encoding.encoder import HippocampalEncoder
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig


# ── stubs (mirror tests/test_orchestrator.py) ──

class _StubEmbedder:
    """Deterministic 384-dim embedder (SHA256 stretch -> normalized)."""
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
    """Returns a fixed plan keyed on entities, ignoring the prompt text."""
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return self._plan


class _StubModeA:
    """Stub ModeAGenerator -- records _complete calls, returns canned text."""
    def __init__(self, reply: str = "SYNTH RESPONSE") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def _complete(self, messages: list[dict], tools=None, tool_choice=None) -> tuple:
        self.calls.append(messages)
        return self.reply, None


class _StubGliner:
    """Stub GLiNERExtractor.extract -- fixed extraction (no model load)."""
    def __init__(self, extracted: dict | None = None) -> None:
        self._extracted = extracted or {
            "entities": ["Postgres"], "entity_classes": {"Postgres": "Technology"},
            "topics": ["storage"], "tones": ["neutral"], "decisions": [],
            "discovered": [],
        }

    def extract(self, text: str) -> dict:
        return {**self._extracted}


class _StubBonsai:
    """Stub BonsaiRelationExtractor.extract -- fixed relations (no server)."""
    def __init__(self, relations: list[dict] | None = None) -> None:
        self._relations = relations or []

    def extract(self, text: str) -> list[dict]:
        return [dict(r) for r in self._relations]


# ── fixtures ──

def _ep(eid, entities=None, topics=None, tones=None, decisions=None,
        summary=None, text=None, ts="2026-07-03T10:00:00") -> Episode:
    return Episode(
        id=eid, timestamp=ts,
        summary=summary or f"summary {eid}",
        full_text=text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=tones or [],
        decisions=decisions or [],
    )


def _make_encoder(store: HippocampalStore, user_id: str = "victor",
                  gliner: _StubGliner | None = None,
                  bonsai: _StubBonsai | None = None) -> HippocampalEncoder:
    """Build a real HippocampalEncoder with stubbed extractors.

    ``object.__new__`` skips ``__init__`` (which eagerly loads GLiNER models);
    we set the attributes ``encode_messages`` / ``start_session`` /
    ``end_session`` actually use. Exercises the real encode path with no GPU.
    """
    enc = object.__new__(HippocampalEncoder)
    enc.store = store
    enc.user_id = user_id
    enc.session_id = None
    enc.last_episode_id = None
    enc.gliner = gliner if gliner is not None else _StubGliner()
    enc.bonsai = bonsai if bonsai is not None else _StubBonsai()
    return enc


def _orch_with_encoder(
    tmp_path, plan: dict, episodes: list[Episode], *,
    reply: str = "SYNTH RESPONSE", user_id: str = "victor",
    gliner: _StubGliner | None = None, bonsai: _StubBonsai | None = None,
) -> tuple[PonderOrchestrator, HippocampalStore, HippocampalEncoder]:
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in episodes:
        store.encode_episode(ep)
    encoder = _make_encoder(store, user_id=user_id, gliner=gliner, bonsai=bonsai)
    embedder = _StubEmbedder()
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                     embedder=embedder)
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=embedder, mode_a=_StubModeA(reply=reply), config=cfg,
        user_id=user_id, encoder=encoder,
    )
    return orch, store, encoder


def _has_graph_edge(store: HippocampalStore, s: str, p: str, o: str) -> bool:
    """True if the graph carries the (s, p, o) edge (used for the follows chain)."""
    r = store.graph.query().vertex(s).out(p).execute_sync()
    try:
        return o in list(r.vertices)
    finally:
        r.close()


# ── 1. two queries persist two chained episodes ──

def test_two_queries_persist_two_chained_episodes(tmp_path, monkeypatch):
    """Two synthesize queries -> two live episodes; ep2 follows ep1 in-session.

    Sync-path invariant: the ``follows`` edge is written by the FOREGROUND
    ``encode_messages`` call, so it is present immediately after ``query()``
    returns. ``async_distill_enabled`` now defaults ON (the stub-then-fill path
    defers graph edges to the background worker, filled by ``drain()`` -- see
    ``test_async_distill_stub_then_fill_end_to_end``); this test selects the
    SYNC path explicitly so the immediate-edge contract it asserts holds.
    """
    monkeypatch.setattr(config, "async_distill_enabled", False)
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, _ = _orch_with_encoder(tmp_path, plan, eps)
    r1 = orch.query("Why did we choose Postgres?")
    r2 = orch.query("Why is Postgres better than MySQL?")
    ep1 = r1["persisted_episode_id"]
    ep2 = r2["persisted_episode_id"]
    assert ep1 and ep2 and ep1 != ep2
    ids = store.default_episode_ids()
    assert ep1 in ids and ep2 in ids
    # The encoder's intra-session follows chain links ep2 -> ep1.
    assert _has_graph_edge(store, ep2, "follows", ep1)
    store.close()


# ── 2. the response is returned unchanged (encode is post-response) ──

def test_response_returned_unchanged(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, _ = _orch_with_encoder(tmp_path, plan, eps, reply="MY REPLY")
    res = orch.query("Why did we choose Postgres?")
    assert res["response"] == "MY REPLY"
    assert "persisted_episode_id" in res
    store.close()


# ── 3. signal modulates salience + decay (how strongly, not whether) ──

def test_signal_profiles_modulate_salience_and_decay(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, _ = _orch_with_encoder(tmp_path, plan, eps)
    r_imp = orch.query("Why did we choose Postgres?", signal="important")
    r_fru = orch.query("Why is Postgres fast?", signal="frustration")
    r_rou = orch.query("Why is Postgres reliable?", signal="routine")
    ep_imp = store.get_episode(r_imp["persisted_episode_id"])
    ep_fru = store.get_episode(r_fru["persisted_episode_id"])
    ep_rou = store.get_episode(r_rou["persisted_episode_id"])
    assert ep_imp.salience == pytest.approx(0.8)
    assert ep_imp.utility_decay_rate == pytest.approx(0.005)
    assert ep_fru.salience == pytest.approx(0.3)
    assert ep_fru.utility_decay_rate == pytest.approx(0.03)
    assert ep_rou.salience == pytest.approx(0.5)
    assert ep_rou.utility_decay_rate == pytest.approx(0.01)
    store.close()


# ── 4. auto_persist=False opts out ──

def test_auto_persist_false_skips_encoding(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, _ = _orch_with_encoder(tmp_path, plan, eps)
    before = set(store.default_episode_ids())
    res = orch.query("Why did we choose Postgres?", auto_persist=False)
    after = set(store.default_episode_ids())
    assert "persisted_episode_id" not in res
    assert before == after  # no new episode encoded
    store.close()


# ── 5. no encoder injected -> no-op (the existing-suite regression contract) ──

def test_no_encoder_injected_is_noop(tmp_path):
    """Pure DI: no encoder injected -> _get_encoder returns None -> no persistence.

    This is the contract that keeps the existing 600-test suite green: tests
    that don't inject an encoder silently skip live-encode rather than crashing.
    """
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in eps:
        store.encode_episode(ep)
    embedder = _StubEmbedder()
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                     embedder=embedder)
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=embedder, mode_a=_StubModeA(), config=cfg, user_id="victor",
    )  # no encoder= -> self._encoder is None
    before = set(store.default_episode_ids())
    res = orch.query("Why did we choose Postgres?")  # auto_persist default True
    after = set(store.default_episode_ids())
    assert "persisted_episode_id" not in res
    assert before == after
    store.close()


# ── 6. a forced encode failure is logged; the response is not lost ──

def test_encode_failure_logged_response_still_returned(tmp_path, capsys, monkeypatch):
    """Sync-path invariant: a foreground ``encode_messages`` failure logs and
    prevents persistence (no ``persisted_episode_id``). Under async distill
    (now the default) the foreground calls ``encode_messages_stub`` -- which
    succeeds and persists a stub -- while ``encode_messages`` runs on the
    background worker, so this contract only holds on the sync path. Select it
    explicitly; the async failure surface is covered by
    ``test_async_distill_stub_then_fill_end_to_end``.
    """
    monkeypatch.setattr(config, "async_distill_enabled", False)
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, encoder = _orch_with_encoder(tmp_path, plan, eps, reply="MY REPLY")

    def _boom(*a, **k):
        raise RuntimeError("forced encode failure")
    encoder.encode_messages = _boom  # simulate a persistence-layer failure

    res = orch.query("Why did we choose Postgres?")
    assert res["response"] == "MY REPLY"          # response not lost
    assert "persisted_episode_id" not in res       # nothing persisted
    assert "persist-fail" in capsys.readouterr().err
    store.close()


# ── 7. end_conversation closes the live-encode session ──

def test_end_conversation_closes_session(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, encoder = _orch_with_encoder(tmp_path, plan, eps)
    orch.query("Why did we choose Postgres?")
    assert encoder.session_id is not None   # opened lazily on first query
    orch.end_conversation()
    assert encoder.session_id is None       # closed
    # A second call with no open session is a graceful no-op (no crash).
    orch.end_conversation()
    store.close()


# ── 8. summary_embedding is backfilled from the embedder ──

def test_summary_embedding_backfilled(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, _ = _orch_with_encoder(tmp_path, plan, eps)
    res = orch.query("Why did we choose Postgres?")
    ep = store.get_episode(res["persisted_episode_id"])
    assert ep.summary_embedding is not None
    assert len(ep.summary_embedding) == 384  # stub embedder dim
    assert all(isinstance(x, float) for x in ep.summary_embedding)
    store.close()


# ── 9. origin round-trips: live -> "live", corpus -> "corpus" ──

def test_origin_round_trip(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, _ = _orch_with_encoder(tmp_path, plan, eps)
    res = orch.query("Why did we choose Postgres?")
    # Live-encoded episode (orchestrator path) is tagged "live".
    assert store.get_episode(res["persisted_episode_id"]).origin == "live"
    # A pre-provenance seed episode (no origin key) reads back as "corpus".
    assert store.get_episode("ep_001").origin == "corpus"
    # The corpus encode path (encode_turn default) also tags "corpus".
    enc2 = _make_encoder(store, user_id="victor")
    eps2 = enc2.encode_conversation([("User asked about Postgres", "We chose it")])
    assert store.get_episode(eps2[0].id).origin == "corpus"
    store.close()


# ── 10. messages round-trip, including a tool-role segment ──

def test_messages_round_trip_including_tool_role(tmp_path):
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch, store, _ = _orch_with_encoder(tmp_path, plan, eps)
    res = orch.query("Why did we choose Postgres?")
    live_ep = store.get_episode(res["persisted_episode_id"])
    assert live_ep.messages is not None
    assert [m["role"] for m in live_ep.messages] == ["user", "assistant"]
    assert live_ep.messages[0]["content"] == "Why did we choose Postgres?"
    assert live_ep.messages[1]["content"] == "SYNTH RESPONSE"
    # A pre-provenance seed episode reads back messages=None (role-unaware;
    # consumers fall back to full_text).
    assert store.get_episode("ep_001").messages is None
    # A from_messages episode with a tool-role segment round-trips the role
    # + tool_call_id (the OpenAI tool-result shape). The assistant segment
    # carries non-empty content so the derived summary is non-empty (get_episode
    # returns None for an empty summary).
    enc2 = _make_encoder(store, user_id="victor")
    enc2.start_session()
    tool_messages = [
        {"role": "user", "content": "Run the migration"},
        {"role": "assistant", "content": "Calling the migration tool", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "migrate", "arguments": "{}"}}]},
        {"role": "tool", "content": "migration ok", "tool_call_id": "call_1"},
    ]
    ep = enc2.encode_messages(tool_messages, origin="live")
    enc2.end_session()
    rt = store.get_episode(ep.id)
    assert [m["role"] for m in rt.messages] == ["user", "assistant", "tool"]
    assert rt.messages[2]["role"] == "tool"
    assert rt.messages[2]["tool_call_id"] == "call_1"
    assert rt.messages[2]["content"] == "migration ok"
    # The assistant tool_calls array survives the JSON round-trip too.
    assert rt.messages[1]["tool_calls"][0]["id"] == "call_1"
    store.close()


# ── 11. from_messages stays readable when no assistant segment has content ──

def test_from_messages_summary_fallback_when_no_assistant_content(tmp_path):
    """No assistant content -> summary falls back to the last non-empty segment.

    Without the fallback the summary would be empty, and ``get_episode`` returns
    ``None`` for an empty summary -- making the episode write-only. A system +
    user message set (no assistant content) must still round-trip.
    """
    store = HippocampalStore(str(tmp_path / "db"))
    enc = _make_encoder(store, user_id="victor")
    enc.start_session()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Just a user question with no answer yet"},
    ]
    ep = enc.encode_messages(msgs, origin="live")
    enc.end_session()
    rt = store.get_episode(ep.id)
    assert rt is not None                       # readable (not write-only)
    assert rt.summary                          # non-empty (fell back to user content)
    assert "Just a user question" in rt.summary
    assert [m["role"] for m in rt.messages] == ["system", "user"]
    store.close()


# ── 12. async-distill: stub written synchronously, edges filled by the worker ──

def test_async_distill_stub_then_fill_end_to_end(tmp_path, monkeypatch):
    """With ``async_distill_enabled`` on, ``query()`` writes the stub (content +
    embedding) synchronously and returns immediately; the graph edges are filled
    by the background worker. The stub is content-retrievable right away; the
    entity edge appears only after the worker's fill (drain)."""
    monkeypatch.setattr(config, "async_distill_enabled", True)
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]

    class _SlowGliner(_StubGliner):
        """Sleeps so the worker's fill is still in flight right after query()
        returns -- makes the 'no edges yet' assertion deterministic."""
        def extract(self, text):
            time.sleep(0.3)
            return super().extract(text)

    orch, store, _ = _orch_with_encoder(
        tmp_path, plan, eps, reply="ASYNC REPLY", gliner=_SlowGliner(),
    )
    try:
        res = orch.query("Why did we choose Postgres?")
        eid = res["persisted_episode_id"]
        # Stub written synchronously: the episode is content-retrievable
        # immediately (origin + embedding land on the main thread).
        ep = store.get_episode(eid)
        assert ep is not None
        assert ep.origin == "live"
        assert ep.summary_embedding is not None
        # Edges NOT yet filled -- the slow GLiNER is still in flight on the
        # worker thread, so the graph is thin right after the response returns.
        assert not _has_graph_edge(store, eid, "has_entity", "E:Postgres"), (
            "entity edge appeared synchronously -- the fill ran on the main thread"
        )
        # Drain: the worker finishes the fill + writes the graph edges.
        joined = orch.drain(timeout=10.0)
        assert joined, "distill worker did not join within timeout"
        assert _has_graph_edge(store, eid, "has_entity", "E:Postgres"), (
            "entity edge missing after drain -- the fill did not write edges"
        )
    finally:
        orch.drain(timeout=10.0)
        store.close()