"""B2: drill-down chain + conversation-vs-memory tool split + ``strategy`` field.

Three components under ONE master flag ``config.drill_down_enabled`` (default
OFF, byte-identical when off):

(a) Scene drill-down: ``expand_unit``'s scene branch follows ``cites`` ONE HOP
    -- each cited episode's one-line summary, so the LLM picks which to
    ``expand(ep_id)`` for verbatim (scene -> cited-ep summary -> verbatim).
    Rides ``--scene-blocks`` (inert without it).
(b) Conversation-vs-memory split: ONE ``search_memory`` tool with an optional
    ``verbatim`` param (default false=summary, true=full text). The param is
    added to the LLM-visible schema ONLY when the flag is on (gated); the
    handler always accepts it.
(c) ``strategy`` field: stamp ``graph``/``vector`` on the graph + semantic
    paths (``hybrid`` stays UNCONDITIONAL) and surface ``[strategy:...]`` in
    ``build_context_string`` so the LLM sees HOW each result was found.

These tests pin: scene drill-down ON/OFF, verbatim rendering ON/OFF,
``search_memory`` forwards verbatim, schema gating (the variant has ``verbatim``,
the base does NOT), ``dispatch_tool`` forwards verbatim, strategy stamps
graph/vector/hybrid (hybrid unconditional), strategy surface ON/OFF, and the
loop-path tool-use prompt note. Offline: tmp_path WaveDB store, stub
planner/embedder/mode_a. No GLiNER, no Bonsai server, no GPU.
"""

from __future__ import annotations

import hashlib
import json

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)",
                allow_module_level=True)

import torch  # noqa: E402 -- JGSBackbone needs torch; gated by wavedb skip above

from src.config import Phase2cConfig, config as _config
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.tools import (
    LOOP_TOOLS, SEARCH_MEMORY_DRILLDOWN_SCHEMA, TOOL_SCHEMAS, dispatch_tool,
)


# ── stubs (mirror tests/test_feedback_salience.py) ──────────────────────────

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
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return self._plan


class _ScriptedModeA:
    """Stub ModeAGenerator -- pops a queued ``(content, tool_calls)`` per call.
    Records the tool set AND the first message's system content per call so the
    schema-swap + tool-use-prompt tests can assert on both."""
    def __init__(self, responses: list[tuple]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def _complete(self, messages: list[dict], tools=None, tool_choice=None) -> tuple:
        sys_content = messages[0]["content"] if messages else ""
        self.calls.append({"tools": tools, "sys_content": sys_content})
        if self.responses:
            return self.responses.pop(0)
        return ("", None)


def _ep(eid, entities=None, topics=None, summary=None, text=None,
        ts="2026-07-03T10:00:00") -> Episode:
    return Episode(
        id=eid, timestamp=ts,
        summary=summary or f"summary {eid}",
        full_text=text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or [], tones=[], decisions=[],
    )


def _tool_call(name, args, cid="call_1"):
    return [{"id": cid, "type": "function",
             "function": {"name": name, "arguments": args}}]


def _orch(tmp_path, plan, episodes, mode_a, *, cfg=None, user_id="victor"):
    store = HippocampalStore(str(tmp_path / "db"))
    for ep in episodes:
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                     embedder=_StubEmbedder())
    c = cfg or Phase2cConfig()
    c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=mode_a, config=c, user_id=user_id,
    )
    return orch, store


def _embed(text: str) -> list[float]:
    return _StubEmbedder().encode([text])[0]


def _encode_scene(store, sid, *, body, topic, heat, user_id, source_eps,
                  updated_ts="2026-08-01T10:00:00"):
    store.encode_scene(sid, body=body, topic=topic, heat=heat,
                      updated_ts=updated_ts, user_id=user_id,
                      source_eps=source_eps, body_embedding=_embed(body))


@pytest.fixture
def drill_on():
    """Set the master-config flag ON for the test; restored after so the global
    never leaks into sibling tests (mirrors ``hybrid_on`` in test_hybrid)."""
    prev = _config.drill_down_enabled
    _config.drill_down_enabled = True
    try:
        yield
    finally:
        _config.drill_down_enabled = prev


# ── (a) scene drill-down in expand_unit ──────────────────────────────────────

def test_scene_drill_down_on_lists_cited_summaries(tmp_path, drill_on):
    """Flag ON: expand_unit(scene) follows ``cites`` ONE HOP -- each cited
    episode's one-line summary, so the LLM picks which to expand for verbatim."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [
        _ep("ep_001", entities=["Postgres"], summary="We chose Postgres"),
        _ep("ep_002", entities=["Postgres"], summary="Backup strategy decided"),
    ]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="# storage notes", topic="storage", heat=0.8,
                  user_id="victor", source_eps=["ep_001", "ep_002"])
    out = orch.expand_unit(sid)
    assert out is not None
    # One-hop form: a ``Cites:`` header + a ``- ep_id: summary`` line per ep.
    assert "Cites:" in out
    assert "- ep_001: We chose Postgres" in out
    assert "- ep_002: Backup strategy decided" in out
    # NOT the bare comma-joined id list (the pre-B2 / flag-off form).
    assert "Cites: ep_001, ep_002" not in out
    store.close()


def test_scene_drill_down_off_is_bare_id_list(tmp_path):
    """Flag OFF: expand_unit(scene) returns the pre-B2 ``Cites: ep_1, ep_2``
    line (byte-identical to pre-B2)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="# storage notes", topic="storage", heat=0.8,
                  user_id="victor", source_eps=["ep_001"])
    out = orch.expand_unit(sid)
    assert out is not None
    assert "Cites: ep_001" in out
    # No one-hop summary line.
    assert "- ep_001:" not in out
    store.close()


def test_scene_drill_down_missing_cited_ep_skipped(tmp_path, drill_on):
    """A cited episode that does not hydrate (missing / unreadable) is skipped,
    no crash. The remaining cited eps still list; if NONE hydrate, fall back to
    the bare id list (the expand contract: cited ids are always visible)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="real episode")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    sid = store.next_scene_id()
    # ep_001 hydrates; ep_missing does not (never encoded).
    _encode_scene(store, sid, body="# notes", topic="t", heat=0.5,
                  user_id="victor", source_eps=["ep_001", "ep_missing"])
    out = orch.expand_unit(sid)
    assert out is not None
    assert "- ep_001: real episode" in out
    # The missing cited ep is skipped (no line, no crash).
    assert "ep_missing" not in out
    store.close()


def test_scene_drill_down_all_missing_falls_back_to_id_list(tmp_path, drill_on):
    """When NONE of the cited eps hydrate, fall back to the bare id list so the
    cited ids are never lost (the expand contract)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = []
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    sid = store.next_scene_id()
    _encode_scene(store, sid, body="# notes", topic="t", heat=0.5,
                  user_id="victor", source_eps=["ep_gone_1", "ep_gone_2"])
    out = orch.expand_unit(sid)
    assert out is not None
    assert "Cites: ep_gone_1, ep_gone_2" in out
    store.close()


# ── (b) verbatim rendering in build_context_string ───────────────────────────

def test_build_context_string_verbatim_on_appends_full_text(tmp_path, drill_on):
    """verbatim=True: the EPISODE chunk appends ``Full text: {text}`` after the
    summary (the "what did the user literally say" intent)."""
    store = HippocampalStore(str(tmp_path / "db"))
    retr = HippocampalRetriever(store, planner=_StubPlanner({}), user_id=None)
    ep = {
        "episode_id": "ep_001", "timestamp": "2026-08-01T10:00:00Z",
        "entities": ["Postgres"], "topics": ["storage"], "tones": [],
        "summary": "We chose Postgres", "text": "User: why\nAssistant: because",
    }
    ctx = retr.build_context_string([ep], max_tokens=1000, verbatim=True)
    assert "Summary: We chose Postgres" in ctx
    assert "Full text: User: why" in ctx
    store.close()


def test_build_context_string_verbatim_off_no_full_text(tmp_path):
    """verbatim=False (default): NO ``Full text`` line -> byte-identical to
    pre-B2."""
    store = HippocampalStore(str(tmp_path / "db"))
    retr = HippocampalRetriever(store, planner=_StubPlanner({}), user_id=None)
    ep = {
        "episode_id": "ep_001", "timestamp": "2026-08-01T10:00:00Z",
        "entities": ["Postgres"], "topics": ["storage"], "tones": [],
        "summary": "We chose Postgres", "text": "User: why\nAssistant: because",
    }
    ctx = retr.build_context_string([ep], max_tokens=1000)
    assert "Summary: We chose Postgres" in ctx
    assert "Full text:" not in ctx
    store.close()


def test_build_context_string_scene_unaffected_by_verbatim(tmp_path, drill_on):
    """Sections/documents/scenes already render their body as ``text`` (no
    separate gist form); verbatim does NOT change them (no double-render)."""
    store = HippocampalStore(str(tmp_path / "db"))
    retr = HippocampalRetriever(store, planner=_StubPlanner({}), user_id=None)
    scene = {
        "episode_id": "scene_000001", "kind": "scene", "summary": "storage",
        "text": "# storage macro\nlots of detail", "timestamp": "2026-08-01",
        "topics": ["storage"], "heat": 0.7, "entities": [], "tones": [],
        "decisions": [],
    }
    on = retr.build_context_string([scene], max_tokens=1000, verbatim=True)
    off = retr.build_context_string([scene], max_tokens=1000, verbatim=False)
    # Scene branch is identical either way (body already rendered).
    assert on == off
    assert "# storage macro" in on
    assert "Full text:" not in on
    store.close()


# ── (b) search_memory forwards verbatim ─────────────────────────────────────

def test_search_memory_forwards_verbatim(tmp_path, drill_on):
    """orchestrator.search_memory(verbatim=True) forwards the flag to
    build_context_string (the handler always accepts it; the schema gating is
    LLM-visible only)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres",
               text="User: literal words here")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    captured: list[dict] = []
    orig = orch.retriever.build_context_string

    def _capture(results, max_tokens=None, verbatim=False):
        captured.append({"verbatim": verbatim, "n": len(results)})
        return orig(results, max_tokens=max_tokens, verbatim=verbatim)

    orch.retriever.build_context_string = _capture
    out = orch.search_memory("Postgres", entities=["Postgres"], verbatim=True)
    assert captured and captured[0]["verbatim"] is True
    assert "Full text:" in out  # verbatim reached the renderer
    store.close()


def test_dispatch_tool_forwards_verbatim(tmp_path, drill_on):
    """dispatch_tool("search_memory", {"verbatim": True}) forwards verbatim to
    orchestrator.search_memory (the handler honors it whenever passed)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres",
               text="User: literal words here")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    captured: list[dict] = []
    orig = orch.retriever.build_context_string

    def _capture(results, max_tokens=None, verbatim=False):
        captured.append({"verbatim": verbatim})
        return orig(results, max_tokens=max_tokens, verbatim=verbatim)

    orch.retriever.build_context_string = _capture
    out = dispatch_tool(orch, "search_memory",
                        {"query": "Postgres", "entities": ["Postgres"],
                         "verbatim": True})
    assert captured and captured[0]["verbatim"] is True
    assert "Full text:" in out
    store.close()


def test_dispatch_tool_search_memory_verbatim_defaults_false(tmp_path, drill_on):
    """dispatch_tool without ``verbatim`` defaults to False (summary intent)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres",
               text="User: literal words here")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    out = dispatch_tool(orch, "search_memory",
                        {"query": "Postgres", "entities": ["Postgres"]})
    assert "Summary: We chose Postgres" in out
    assert "Full text:" not in out
    store.close()


# ── (b) schema gating ─────────────────────────────────────────────────────────

def test_search_memory_drilldown_schema_has_verbatim():
    """The drilldown variant has the ``verbatim`` prop; the base schema does NOT
    (byte-identical-off: the LLM never sees the param when the flag is off)."""
    base = next(t for t in TOOL_SCHEMAS
                if t["function"]["name"] == "search_memory")
    base_props = base["function"]["parameters"]["properties"]
    assert "verbatim" not in base_props
    dd_props = SEARCH_MEMORY_DRILLDOWN_SCHEMA["function"]["parameters"]["properties"]
    assert "verbatim" in dd_props
    assert dd_props["verbatim"]["type"] == "boolean"
    assert dd_props["verbatim"]["default"] is False
    # Same name (it's a swap-in, not a new tool).
    assert (SEARCH_MEMORY_DRILLDOWN_SCHEMA["function"]["name"]
            == "search_memory")


def test_synthesize_loop_swaps_schema_when_drill_down_on(tmp_path, drill_on):
    """With the flag ON + the loop on, the loop-path tool set swaps the
    search_memory entry for the variant WITH ``verbatim``. Feedback disabled so
    loop_tools = LOOP_TOOLS (expand + search_memory); the swap replaces only the
    search_memory entry (expand is untouched)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("FINAL ANSWER", None)])
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved_fb
    tools = mode_a.calls[0]["tools"]
    names = [t["function"]["name"] for t in tools]
    assert names == ["expand", "search_memory"]
    sm = next(t for t in tools if t["function"]["name"] == "search_memory")
    assert "verbatim" in sm["function"]["parameters"]["properties"]
    store.close()


def test_synthesize_loop_base_schema_when_drill_down_off(tmp_path):
    """Flag OFF: the loop-path tool set is the exact LOOP_TOOLS (search_memory
    entry has NO ``verbatim``) -> byte-identical to pre-B2."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("FINAL ANSWER", None)])
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved_fb
    tools = mode_a.calls[0]["tools"]
    # The flag-off path hands the model the exact LOOP_TOOLS object (identity).
    assert tools is LOOP_TOOLS
    sm = next(t for t in tools if t["function"]["name"] == "search_memory")
    assert "verbatim" not in sm["function"]["parameters"]["properties"]
    store.close()


def test_synthesize_loop_prompt_note_when_drill_down_on(tmp_path, drill_on):
    """Flag ON + loop ON: the system prompt includes the MAY-phrased
    ``verbatim``/strategy note. Loop-path-only, never imperative."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("FINAL ANSWER", None)])
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved_fb
    sys_content = mode_a.calls[0]["sys_content"]
    assert "verbatim" in sys_content
    assert "[strategy:...]" in sys_content
    store.close()


def test_synthesize_loop_prompt_note_absent_when_drill_down_off(tmp_path):
    """Flag OFF (or loop off): the note is ABSENT -> byte-identical system
    prompt to pre-B2."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    mode_a = _ScriptedModeA([("FINAL ANSWER", None)])
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, plan, eps, mode_a)
        orch.query("Why did we choose Postgres?")
    finally:
        _config.feedback_salience_enabled = saved_fb
    sys_content = mode_a.calls[0]["sys_content"]
    assert "verbatim" not in sys_content
    assert "[strategy:...]" not in sys_content
    store.close()


# ── (c) strategy stamps ──────────────────────────────────────────────────────

def test_strategy_stamp_graph_on(tmp_path, drill_on):
    """Flag ON: graph-only ``traversal.retrieve`` stamps ``strategy="graph"`` on
    every result."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", entities=["Postgres"], summary="pg"))
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Postgres"], "entity_mode": "union"})
    assert results
    assert all(r.get("strategy") == "graph" for r in results)
    store.close()


def test_strategy_stamp_graph_off_no_key(tmp_path):
    """Flag OFF: NO ``strategy`` key on graph results -> byte-identical dict to
    pre-B2 (the renderer's ``strat`` is None)."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", entities=["Postgres"], summary="pg"))
    trav = GraphTraversal(store)
    results = trav.retrieve({"entities": ["Postgres"], "entity_mode": "union"})
    assert results
    assert all("strategy" not in r for r in results)
    store.close()


def test_strategy_stamp_vector_on(tmp_path, drill_on):
    """Flag ON: the semantic-fallback path stamps ``strategy="vector"``."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", entities=["Postgres"], summary="pg"))
    retr = HippocampalRetriever(store, planner=_StubPlanner({}),
                                embedder=_StubEmbedder())

    class _FakeVS:
        def search(self, query, k=5):
            return [("ep_001", 0.9)]

    retr.vector_search = _FakeVS()
    out = retr._semantic_fallback("pg", {"entities": ["Postgres"]}, None, None,
                                  None)
    assert out
    assert all(r.get("strategy") == "vector" for r in out)
    store.close()


def test_strategy_stamp_vector_off_no_key(tmp_path):
    """Flag OFF: the semantic-fallback path does NOT stamp ``strategy``."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001", entities=["Postgres"], summary="pg"))
    retr = HippocampalRetriever(store, planner=_StubPlanner({}),
                                embedder=_StubEmbedder())

    class _FakeVS:
        def search(self, query, k=5):
            return [("ep_001", 0.9)]

    retr.vector_search = _FakeVS()
    out = retr._semantic_fallback("pg", {"entities": ["Postgres"]}, None, None,
                                  None)
    assert out
    assert all("strategy" not in r for r in out)
    store.close()


def test_strategy_stamp_hybrid_unconditional(tmp_path, drill_on):
    """The hybrid stamp (``_retrieve_hybrid``) stays UNCONDITIONAL: results
    carry ``strategy="hybrid"`` regardless of ``--drill-down`` (the existing
    test_hybrid_retrieval assertion holds). With drill_down ON the renderer
    would surface ``[strategy:hybrid]``; the stamp itself is not gated."""
    from src.retrieval.bm25 import BM25Search
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    store = HippocampalStore(str(tmp_path / "db"))
    for i in range(3):
        store.encode_episode(_ep(
            f"ep_a{i}", entities=["Postgres"], summary=f"pg {i}",
            text=f"User: postgres chat {i}"))
    retr = HippocampalRetriever(store, planner=_StubPlanner(plan))
    retr.hybrid_retrieval = True
    retr.bm25 = BM25Search(store.db)
    results = retr.retrieve("postgres")
    assert results
    assert all(r.get("strategy") == "hybrid" for r in results)
    store.close()


def test_strategy_stamp_hybrid_unconditional_off(tmp_path):
    """Flag OFF: hybrid results STILL carry ``strategy="hybrid"`` (unconditional
    stamp) but the renderer does NOT surface it -> byte-identical LLM output."""
    from src.retrieval.bm25 import BM25Search
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    store = HippocampalStore(str(tmp_path / "db"))
    for i in range(3):
        store.encode_episode(_ep(
            f"ep_a{i}", entities=["Postgres"], summary=f"pg {i}",
            text=f"User: postgres chat {i}"))
    retr = HippocampalRetriever(store, planner=_StubPlanner(plan))
    retr.hybrid_retrieval = True
    retr.bm25 = BM25Search(store.db)
    results = retr.retrieve("postgres")
    assert results
    assert all(r.get("strategy") == "hybrid" for r in results)
    store.close()


# ── (c) strategy surface in build_context_string ─────────────────────────────

def test_strategy_surface_on_prefixes_chunk(tmp_path, drill_on):
    """Flag ON: a result carrying ``strategy="graph"`` is prefixed
    ``[strategy:graph]`` in the rendered context (LLM-facing provenance)."""
    store = HippocampalStore(str(tmp_path / "db"))
    retr = HippocampalRetriever(store, planner=_StubPlanner({}), user_id=None)
    ep = {
        "episode_id": "ep_001", "timestamp": "2026-08-01T10:00:00Z",
        "entities": ["Postgres"], "topics": ["storage"], "tones": [],
        "summary": "We chose Postgres", "strategy": "graph",
    }
    ctx = retr.build_context_string([ep], max_tokens=1000)
    assert "[strategy:graph]" in ctx
    # The prefix sits ABOVE the chunk header.
    assert ctx.index("[strategy:graph]") < ctx.index("[ep_001 |")
    store.close()


def test_strategy_surface_off_no_prefix(tmp_path):
    """Flag OFF: NO ``[strategy:...]`` prefix even when the dict carries
    ``strategy="hybrid"`` -> byte-identical LLM output."""
    store = HippocampalStore(str(tmp_path / "db"))
    retr = HippocampalRetriever(store, planner=_StubPlanner({}), user_id=None)
    ep = {
        "episode_id": "ep_001", "timestamp": "2026-08-01T10:00:00Z",
        "entities": ["Postgres"], "topics": ["storage"], "tones": [],
        "summary": "We chose Postgres", "strategy": "hybrid",
    }
    ctx = retr.build_context_string([ep], max_tokens=1000)
    assert "[strategy:" not in ctx
    store.close()


# ── byte-identical-OFF end-to-end (search_memory tool response) ───────────────

def test_search_memory_byte_identical_off(tmp_path):
    """Flag OFF: ``search_memory`` (verbatim default False) renders the SAME
    context as pre-B2 -- no ``Full text`` line, no ``[strategy:]`` prefix, even
    when a result happens to carry ``strategy="hybrid"``."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres",
               text="User: literal")]
    mode_a = _ScriptedModeA([("ans", None)])
    orch, store = _orch(tmp_path, plan, eps, mode_a)
    out = dispatch_tool(orch, "search_memory",
                        {"query": "Postgres", "entities": ["Postgres"]})
    assert "Summary: We chose Postgres" in out
    assert "Full text:" not in out
    assert "[strategy:" not in out
    store.close()