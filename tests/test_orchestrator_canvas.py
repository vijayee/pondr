"""B4: orchestrator wiring of the Mermaid task canvas -- L1.5 lifecycle gate,
user-message injection, the ``update_canvas`` Bonsai-loop tool, schema gating,
per-query reset, surfacing, and persist-tail reclaim. Offline throughout: a
stub canvas decider (no Bonsai HTTP), stub embedder/planner/mode_a, tmp WaveDB
store. No GLiNER, no GPU.

Master flag is ``config.task_canvas_enabled`` (default OFF, byte-identical when
off). The orchestrator ctor also takes ``task_canvas`` (sets the lazy formatter
+ the gate's decider) -- both are flipped on together (mirrors serve_ponder).
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
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.tools import LOOP_TOOLS, UPDATE_CANVAS_SCHEMA, dispatch_tool


# ── stubs (mirror tests/test_drill_down.py) ─────────────────────────────────

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
    def plan(self, prompt: str, conversation_history=None) -> dict:
        return {"entities": ["Postgres"], "entity_mode": "union"}


class _ScriptedModeA:
    """Pops a queued ``(content, tool_calls)`` per call; records the tool set, the
    system content, AND the last user-message content (the canvas-injection
    assertion reads the user message)."""
    def __init__(self, responses: list[tuple]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def _complete(self, messages: list[dict], tools=None, tool_choice=None) -> tuple:
        sys_content = messages[0]["content"] if messages else ""
        user_content = next((m["content"] for m in reversed(messages)
                             if m.get("role") == "user"), "")
        self.calls.append({"tools": tools, "sys_content": sys_content,
                           "user_content": user_content})
        if self.responses:
            return self.responses.pop(0)
        return ("", None)


class _StubCanvasDecider:
    """Stub ``judge_task_lifecycle``: pops a queued verdict per call. ``None``
    models cold-start / Bonsai-down (gate defers). Records every call."""
    def __init__(self, queue: list | None = None) -> None:
        self._queue = list(queue or [])
        self.calls: list[dict] = []

    def judge_task_lifecycle(self, user_id, recent_messages, active_canvas,
                             historical_canvases):
        self.calls.append({"user_id": user_id,
                           "active": (active_canvas or {}).get("canvas_id"),
                           "n_hist": len(historical_canvases)})
        if not self._queue:
            return None
        return self._queue.pop(0)


def _ep(eid):
    return Episode(
        id=eid, timestamp="2026-08-01T10:00:00", summary=f"summary {eid}",
        full_text=f"User: u{eid}\nAssistant: a{eid}", entities=["Postgres"],
        topics=["storage"], tones=[], decisions=[],
    )


def _tool_call(name, args, cid="call_1"):
    return [{"id": cid, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]


def _orch(tmp_path, mode_a, *, decider=None, task_canvas=True, user_id="victor"):
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                     embedder=_StubEmbedder())
    c = Phase2cConfig()
    c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=mode_a, config=c, user_id=user_id,
        task_canvas=task_canvas, canvas_decider=decider,
    )
    return orch, store


@pytest.fixture
def canvas_on():
    """Flip the master-config flag ON for the test; restored after."""
    prev = _config.task_canvas_enabled
    _config.task_canvas_enabled = True
    try:
        yield
    finally:
        _config.task_canvas_enabled = prev


def _encode(store, cid, *, label="task", progress=0, active="1", user="victor",
            ts="2026-08-01T10:00:00", mermaid=None):
    store.encode_canvas(
        cid, mermaid=mermaid or f"%%{{progress: {progress}}}%%\nflowchart TD",
        task_label=label, progress=progress, created_ts=ts, updated_ts=ts,
        user_id=user, active=active, node_mapping={})


# ── (1) L1.5 gate: the 5-case lifecycle ──────────────────────────────────────

def test_gate_create_opens_canvas(tmp_path, canvas_on):
    """create: isLongTask + no active -> a fresh canvas is created + set active."""
    decider = _StubCanvasDecider([{"taskCompleted": False, "isLongTask": True,
                                   "isContinuation": False,
                                   "continuationCanvasId": "",
                                   "newTaskLabel": "build feature"}])
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]), decider=decider)
    orch._run_canvas_gate("let us build a feature", [])
    assert orch._active_canvas_id is not None
    cv = store.get_canvas(orch._active_canvas_id)
    assert cv is not None
    assert cv["task_label"] == "build-feature"
    assert cv["active"] is True
    assert "flowchart TD" in cv["mermaid"]
    store.close()


def test_gate_resume_switches_active(tmp_path, canvas_on):
    """switch/resume: isContinuation + continuationCanvasId -> set that canvas
    active (the prior active flips to historical)."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    a = store.next_canvas_id(); b = store.next_canvas_id()
    _encode(store, a, label="old-task", active="1")
    _encode(store, b, label="yesterday-task", active="0")
    decider = _StubCanvasDecider([{"taskCompleted": False, "isLongTask": False,
                                   "isContinuation": True,
                                   "continuationCanvasId": b,
                                   "newTaskLabel": ""}])
    retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                    embedder=_StubEmbedder())
    c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=_ScriptedModeA([("ans", None)]),
        config=c, user_id="victor", task_canvas=True, canvas_decider=decider)
    orch._run_canvas_gate("continue yesterday's task", [])
    assert orch._active_canvas_id == b
    assert store.get_canvas(b)["active"] is True
    assert store.get_canvas(a)["active"] is False
    store.close()


def test_gate_resume_wins_over_create_when_both_set(tmp_path, canvas_on):
    """Precedence: a named resume (isContinuation + continuationCanvasId)
    must win over a coincident isLongTask when there is no active canvas. The
    prompt documents ``isContinuation > isLongTask``; the orchestrator checks
    resume BEFORE create so a stale/ambiguous dual signal resumes the named
    task rather than opening a fresh canvas."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    a = store.next_canvas_id(); b = store.next_canvas_id()
    _encode(store, a, label="old-task", active="0")
    _encode(store, b, label="yesterday-task", active="0")  # none active
    decider = _StubCanvasDecider([{"taskCompleted": False, "isLongTask": True,
                                   "isContinuation": True,
                                   "continuationCanvasId": b,
                                   "newTaskLabel": "brand-new-task"}])
    retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                    embedder=_StubEmbedder())
    c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=_ScriptedModeA([("ans", None)]),
        config=c, user_id="victor", task_canvas=True, canvas_decider=decider)
    orch._run_canvas_gate("continue yesterday's task", [])
    assert orch._active_canvas_id == b          # resumed, NOT a new canvas
    assert store.get_canvas(b)["active"] is True
    # No new canvas was created (only canvas_000001 + canvas_000002 exist).
    assert store.next_canvas_id() == "canvas_000003"
    store.close()


def test_gate_resume_with_bogus_id_falls_through_to_create(tmp_path, canvas_on):
    """A resume with an unknown continuationCanvasId falls through to create
    (isLongTask, no active) -- the resume branch never crashes on a bad id."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    a = store.next_canvas_id()
    _encode(store, a, label="old-task", active="0")  # none active
    decider = _StubCanvasDecider([{"taskCompleted": False, "isLongTask": True,
                                   "isContinuation": True,
                                   "continuationCanvasId": "canvas_999999",
                                   "newTaskLabel": "brand-new-task"}])
    retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                    embedder=_StubEmbedder())
    c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=_ScriptedModeA([("ans", None)]),
        config=c, user_id="victor", task_canvas=True, canvas_decider=decider)
    orch._run_canvas_gate("start a brand new task", [])
    # Bogus resume fell through; create opened a fresh canvas.
    assert orch._active_canvas_id == "canvas_000002"
    assert store.get_canvas("canvas_000002")["active"] is True
    store.close()


def test_gate_clear_flips_active_to_historical(tmp_path, canvas_on):
    """clear: taskCompleted -> flip active to historical; no active this turn."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    a = store.next_canvas_id()
    _encode(store, a, label="done-task", active="1")
    decider = _StubCanvasDecider([{"taskCompleted": True, "isLongTask": False,
                                   "isContinuation": False,
                                   "continuationCanvasId": "",
                                   "newTaskLabel": ""}])
    retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                    embedder=_StubEmbedder())
    c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=_ScriptedModeA([("ans", None)]),
        config=c, user_id="victor", task_canvas=True, canvas_decider=decider)
    orch._run_canvas_gate("we are done, ship it", [])
    assert orch._active_canvas_id is None
    assert store.get_canvas(a)["active"] is False
    store.close()


def test_gate_keep_leaves_active_unchanged(tmp_path, canvas_on):
    """keep: all-false -> the active canvas is unchanged."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    a = store.next_canvas_id()
    _encode(store, a, label="ongoing", active="1")
    decider = _StubCanvasDecider([{"taskCompleted": False, "isLongTask": False,
                                   "isContinuation": False,
                                   "continuationCanvasId": "",
                                   "newTaskLabel": ""}])
    retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                    embedder=_StubEmbedder())
    c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=_ScriptedModeA([("ans", None)]),
        config=c, user_id="victor", task_canvas=True, canvas_decider=decider)
    orch._run_canvas_gate("still working on it", [])
    assert orch._active_canvas_id == a
    assert store.get_canvas(a)["active"] is True
    store.close()


def test_gate_create_if_missing_when_no_active(tmp_path, canvas_on):
    """create-if-missing: isLongTask with no active opens a fresh canvas."""
    decider = _StubCanvasDecider([{"taskCompleted": False, "isLongTask": True,
                                   "isContinuation": False,
                                   "continuationCanvasId": "",
                                   "newTaskLabel": "new thing"}])
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]), decider=decider)
    orch._run_canvas_gate("start a new long task", [])
    assert orch._active_canvas_id is not None
    assert store.get_active_canvas("victor") is not None
    store.close()


# ── (2) gate failure is best-effort ──────────────────────────────────────────

def test_gate_failure_falls_back_to_store_active(tmp_path, canvas_on):
    """Gate returns None (cold-start / Bonsai-down) -> keep the store's current
    active (or None); never break the turn."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    a = store.next_canvas_id()
    _encode(store, a, label="prior", active="1")
    decider = _StubCanvasDecider([None])  # Bonsai-down
    retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                    embedder=_StubEmbedder())
    c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=_ScriptedModeA([("ans", None)]),
        config=c, user_id="victor", task_canvas=True, canvas_decider=decider)
    orch._run_canvas_gate("anything", [])
    # Falls back to the store's current active.
    assert orch._active_canvas_id == a
    store.close()


def test_gate_failure_no_active_leaves_none(tmp_path, canvas_on):
    """Gate None + no prior active -> _active_canvas_id stays None -> no injection
    -> byte-identical to flag-off for that turn."""
    decider = _StubCanvasDecider([None])
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]), decider=decider)
    orch._run_canvas_gate("cold start", [])
    assert orch._active_canvas_id is None
    store.close()


# ── (3) injection (user-message prepend, per-turn dynamic) ───────────────────

def test_injection_prepends_canvas_block_when_on(tmp_path, canvas_on):
    """Flag ON + active canvas -> the user message starts with ``[TASK CANVAS]``.
    Order: canvas -> context -> User (no fade block here)."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    a = store.next_canvas_id()
    _encode(store, a, label="build-api", active="1",
            mermaid="%%{progress: 20}%%\nflowchart TD\n    N1[\"doing\"]")
    decider = _StubCanvasDecider([{"taskCompleted": False, "isLongTask": False,
                                   "isContinuation": False,
                                   "continuationCanvasId": "",
                                   "newTaskLabel": ""}])  # keep
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                        embedder=_StubEmbedder())
        c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
        orch = PonderOrchestrator(
            store=store, retriever=retriever,
            backbone=JGSBackbone(BackboneConfig()), embedder=_StubEmbedder(),
            mode_a=_ScriptedModeA([("FINAL ANSWER", None)]), config=c,
            user_id="victor", task_canvas=True, canvas_decider=decider)
        orch.query("what should I do next?")
    finally:
        _config.feedback_salience_enabled = saved_fb
    user_content = orch.mode_a.calls[0]["user_content"]
    assert user_content.startswith("[TASK CANVAS")
    assert "flowchart TD" in user_content
    assert "Context from past conversations" in user_content
    store.close()


def test_injection_absent_when_flag_off(tmp_path):
    """Flag OFF -> no canvas block in the user message -> byte-identical to pre-B4."""
    store = HippocampalStore(str(tmp_path / "db"))
    store.encode_episode(_ep("ep_001"))
    a = store.next_canvas_id()
    _encode(store, a, label="build-api", active="1")
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                        embedder=_StubEmbedder())
        c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
        # task_canvas=False (ctor flag off) -> no formatter, no gate.
        orch = PonderOrchestrator(
            store=store, retriever=retriever,
            backbone=JGSBackbone(BackboneConfig()), embedder=_StubEmbedder(),
            mode_a=_ScriptedModeA([("FINAL ANSWER", None)]), config=c,
            user_id="victor", task_canvas=False, canvas_decider=None)
        orch.query("what should I do next?")
    finally:
        _config.feedback_salience_enabled = saved_fb
    user_content = orch.mode_a.calls[0]["user_content"]
    assert "[TASK CANVAS" not in user_content
    store.close()


# ── (4) update_canvas tool (Bonsai-loop) ─────────────────────────────────────

def test_update_canvas_write_stores_mermaid(tmp_path, canvas_on):
    """write: stores the Mermaid via touch_canvas + re-parses progress."""
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]),
                        decider=_StubCanvasDecider([{"taskCompleted": False,
                                                     "isLongTask": True,
                                                     "isContinuation": False,
                                                     "continuationCanvasId": "",
                                                     "newTaskLabel": "build"}]))
    orch._run_canvas_gate("start building", [])
    cid = orch._active_canvas_id
    out = json.loads(orch.update_canvas(
        mmd_content="%%{progress: 50}%%\nflowchart TD\n    N1[\"doing\"]",
        file_action="write"))
    assert out["ok"] is True
    assert out["canvas_id"] == cid
    assert out["progress"] == 50
    assert store.get_canvas(cid)["progress"] == 50
    store.close()


def test_update_canvas_replace_splices_blocks(tmp_path, canvas_on):
    """replace: splice replace_blocks into the stored Mermaid by node id."""
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]),
                        decider=_StubCanvasDecider([{"taskCompleted": False,
                                                     "isLongTask": True,
                                                     "isContinuation": False,
                                                     "continuationCanvasId": "",
                                                     "newTaskLabel": "build"}]))
    orch._run_canvas_gate("start building", [])
    cid = orch._active_canvas_id
    # Seed the canvas with two nodes via a write.
    orch.update_canvas(
        mmd_content="%%{progress: 0}%%\nflowchart TD\n    N1[\"old\"]\n    N2[\"keep\"]",
        file_action="write")
    out = json.loads(orch.update_canvas(
        file_action="replace",
        replace_blocks=[{"node_id": "N1", "new_block": "    N1[\"new\"]"}]))
    assert out["ok"] is True
    cv = store.get_canvas(cid)
    assert "N1[\"new\"]" in cv["mermaid"]
    assert "N2[\"keep\"]" in cv["mermaid"]
    store.close()


def test_update_canvas_no_active_returns_error(tmp_path, canvas_on):
    """No active canvas -> a short error JSON string (never raises)."""
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]),
                        decider=_StubCanvasDecider([None]))
    orch._run_canvas_gate("nothing active", [])
    assert orch._active_canvas_id is None
    out = json.loads(orch.update_canvas(mmd_content="%%{progress: 0}%%\nflowchart TD",
                                       file_action="write"))
    assert "error" in out
    store.close()


def test_dispatch_tool_update_canvas(tmp_path, canvas_on):
    """dispatch_tool("update_canvas", ...) forwards to orchestrator.update_canvas."""
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]),
                        decider=_StubCanvasDecider([{"taskCompleted": False,
                                                     "isLongTask": True,
                                                     "isContinuation": False,
                                                     "continuationCanvasId": "",
                                                     "newTaskLabel": "build"}]))
    orch._run_canvas_gate("start building", [])
    out = dispatch_tool(orch, "update_canvas",
                        {"mmd_content": "%%{progress: 30}%%\nflowchart TD",
                         "file_action": "write"})
    assert json.loads(out)["ok"] is True
    store.close()


def test_dispatch_tool_update_canvas_bad_file_action(tmp_path, canvas_on):
    """A bad file_action -> a short error string (the dispatch guard)."""
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]),
                        decider=_StubCanvasDecider([None]))
    out = dispatch_tool(orch, "update_canvas",
                        {"mmd_content": "x", "file_action": "bogus"})
    assert "file_action" in out
    store.close()


# ── (5) schema gating (loop-path-only, new-list discipline) ──────────────────

def test_update_canvas_schema_has_expected_params():
    props = UPDATE_CANVAS_SCHEMA["function"]["parameters"]["properties"]
    assert "mmd_content" in props
    assert "file_action" in props
    assert props["file_action"]["default"] == "write"
    assert "replace_blocks" in props
    assert "node_mapping" in props
    assert UPDATE_CANVAS_SCHEMA["function"]["name"] == "update_canvas"


def test_loop_tools_append_update_canvas_when_on(tmp_path, canvas_on):
    """Flag ON + loop ON: loop_tools includes ``update_canvas`` (new list)."""
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, _ScriptedModeA([("FINAL ANSWER", None)]),
                            decider=_StubCanvasDecider([{"taskCompleted": False,
                                                         "isLongTask": False,
                                                         "isContinuation": False,
                                                         "continuationCanvasId": "",
                                                         "newTaskLabel": ""}]))
        orch.query("what next?")
    finally:
        _config.feedback_salience_enabled = saved_fb
    tools = orch.mode_a.calls[0]["tools"]
    names = [t["function"]["name"] for t in tools]
    assert "update_canvas" in names


def test_loop_tools_no_update_canvas_when_off(tmp_path):
    """Flag OFF -> loop_tools is the exact LOOP_TOOLS (no update_canvas) ->
    byte-identical to pre-B4."""
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        store = HippocampalStore(str(tmp_path / "db"))
        store.encode_episode(_ep("ep_001"))
        retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                        embedder=_StubEmbedder())
        c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
        orch = PonderOrchestrator(
            store=store, retriever=retriever,
            backbone=JGSBackbone(BackboneConfig()), embedder=_StubEmbedder(),
            mode_a=_ScriptedModeA([("FINAL ANSWER", None)]), config=c,
            user_id="victor", task_canvas=False, canvas_decider=None)
        orch.query("what next?")
    finally:
        _config.feedback_salience_enabled = saved_fb
    tools = orch.mode_a.calls[0]["tools"]
    assert tools is LOOP_TOOLS
    assert "update_canvas" not in [t["function"]["name"] for t in tools]
    store.close()


# ── (6) per-query reset + surfacing ──────────────────────────────────────────

def test_per_query_reset_clears_active_canvas(tmp_path, canvas_on):
    """_active_canvas_id is reset to None at the top of each query; the gate
    re-sets it. After a keep turn (no active created) it stays None."""
    orch, store = _orch(tmp_path, _ScriptedModeA([("ans", None)]),
                        decider=_StubCanvasDecider([{"taskCompleted": False,
                                                     "isLongTask": False,
                                                     "isContinuation": False,
                                                     "continuationCanvasId": "",
                                                     "newTaskLabel": ""}]))
    orch._active_canvas_id = "canvas_bogus"
    orch._run_canvas_gate("fresh turn", [])
    # keep + no active -> None (the bogus leaked value is gone).
    assert orch._active_canvas_id is None
    store.close()


def test_surfacing_task_canvas_present_when_set(tmp_path, canvas_on):
    """result["task_canvas"] is present when the gate set _last_canvas (create),
    absent when None -> byte-identical to flag-off."""
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, _ScriptedModeA([("FINAL ANSWER", None)]),
                            decider=_StubCanvasDecider([{"taskCompleted": False,
                                                         "isLongTask": True,
                                                         "isContinuation": False,
                                                         "continuationCanvasId": "",
                                                         "newTaskLabel": "build"}]))
        result = orch.query("start a long task", auto_persist=False)
    finally:
        _config.feedback_salience_enabled = saved_fb
    assert "task_canvas" in result
    assert result["task_canvas"]["action"] == "create"
    store.close()


def test_surfacing_task_canvas_absent_when_none(tmp_path, canvas_on):
    """Gate returns None + no prior active -> _last_canvas stays None -> key
    absent (byte-identical to flag-off for that turn)."""
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, _ScriptedModeA([("FINAL ANSWER", None)]),
                            decider=_StubCanvasDecider([None]))
        result = orch.query("cold start", auto_persist=False)
    finally:
        _config.feedback_salience_enabled = saved_fb
    assert "task_canvas" not in result
    store.close()


# ── (7) reclaim in persist tail ──────────────────────────────────────────────

def test_reclaim_wired_in_persist_tail(tmp_path, canvas_on):
    """Flag ON + auto_persist -> reclaim_canvases is called in the persist tail."""
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        orch, store = _orch(tmp_path, _ScriptedModeA([("FINAL ANSWER", None)]),
                            decider=_StubCanvasDecider([{"taskCompleted": False,
                                                         "isLongTask": False,
                                                         "isContinuation": False,
                                                         "continuationCanvasId": "",
                                                         "newTaskLabel": ""}]))
        called = {"n": 0}
        orig = store.reclaim_canvases

        def _spy(user_id, *, min_keep=15):
            called["n"] += 1
            called["min_keep"] = min_keep
            return orig(user_id, min_keep=min_keep)

        store.reclaim_canvases = _spy
        orch.query("anything", auto_persist=True)
    finally:
        _config.feedback_salience_enabled = saved_fb
    assert called["n"] == 1
    assert called["min_keep"] == _config.canvas_min_keep
    store.close()


def test_reclaim_not_called_when_flag_off(tmp_path):
    """Flag OFF -> reclaim_canvases is NOT called in the persist tail."""
    saved_fb = _config.feedback_salience_enabled
    _config.feedback_salience_enabled = False
    try:
        store = HippocampalStore(str(tmp_path / "db"))
        store.encode_episode(_ep("ep_001"))
        retriever = HippocampalRetriever(store, planner=_StubPlanner(),
                                        embedder=_StubEmbedder())
        c = Phase2cConfig(); c.session.state_dir = str(tmp_path / "sessions")
        orch = PonderOrchestrator(
            store=store, retriever=retriever,
            backbone=JGSBackbone(BackboneConfig()), embedder=_StubEmbedder(),
            mode_a=_ScriptedModeA([("FINAL ANSWER", None)]), config=c,
            user_id="victor", task_canvas=False, canvas_decider=None)
        called = {"n": 0}
        orig = store.reclaim_canvases
        store.reclaim_canvases = lambda *a, **k: (called.__setitem__("n", called["n"] + 1), orig(*a, **k))[1]
        orch.query("anything", auto_persist=True)
    finally:
        _config.feedback_salience_enabled = saved_fb
    assert called["n"] == 0
    store.close()