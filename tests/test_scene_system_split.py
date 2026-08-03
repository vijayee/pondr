"""A3 (corrected): scene blocks -> system-prompt suffix (cache-stable split).

The Tencent-survey A3 ("stable-vs-dynamic prompt-cache split") was originally
phrased as "move ``[FADE MEMORY]`` to the system prompt" -- backwards, since the
fade block is per-turn dynamic and the system prompt is already the cache-stable
prefix. The corrected A3 renders the SESSION-STABLE scene blocks (B1,
``--scene-blocks``) into the system-prompt suffix instead of the user-message
context blob; the fade block stays in the user message. These tests pin:

- the split is active only when ``--scene-blocks`` is on (off -> scenes stay in
  the user context via ``_format_episode``'s ``kind == "scene"`` branch ->
  byte-identical to pre-A3);
- ``--scene-blocks`` on + no scenes retrieved -> messages byte-identical to off
  (the messages-level gate the suite previously lacked);
- on + a scene retrieved -> scene body in the SYSTEM message, NOT the user
  message;
- ``format_scene_block`` returns ``""`` on empty (no empty header);
- inject failure is swallowed (the turn proceeds, system prompt unchanged).

Stubs mirror ``tests/test_fade_serve_integration.py`` (the orchestrator harness
that already wires fade) + ``tests/test_scene_blocks.py`` (the scene store
helpers). No bge, no Bonsai, no token-LM.
"""

from __future__ import annotations

import hashlib

from src.config import Phase2cConfig
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.scene_format import format_scene_block


# ── stubs (mirror tests/test_fade_serve_integration.py + test_scene_blocks.py) ──

class _StubEmbedder:
    """Deterministic 384-dim embedder (SHA256 stretch -> normalized). Mirrors the
    scene_blocks + fade test embedders so scenes embed the same way the WM
    bge-small embedder would (384-d, cosine)."""
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
    """Returns a fixed plan dict (topics-axis so scenes surface on ``has_topic``)."""
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return self._plan


class _StubModeA:
    def __init__(self, reply: str = "SYNTH RESPONSE") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def _complete(self, messages, tools=None, tool_choice=None) -> tuple:
        self.calls.append(messages)
        return self.reply, None


_EMBED = _StubEmbedder()

# Topics-axis plan (the shared ``storage`` topic -- the axis scenes share with
# episodes). Topics-only (no entity axis) so a non-existent entity never
# collapses the candidate set (``_find_candidates`` returns empty when a
# specified entity fails to match -- a pre-existing quirk unrelated to scenes).
_TOPICS_PLAN = {"topics": ["storage"], "entity_mode": "union", "limit": 20}


# ── fixtures ──

def _ep(eid, entities=None, summary=None, text=None, topics=None,
        ts="2026-08-01T10:00:00") -> Episode:
    return Episode(
        id=eid, timestamp=ts,
        summary=summary or f"summary {eid}",
        full_text=text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=topics or ["storage"],
        tones=[], decisions=[],
    )


def _embed(text: str) -> list[float]:
    return _EMBED.encode([text])[0]


def _encode_scene(store, sid, *, body, topic, heat, user_id, source_eps,
                  updated_ts="2026-08-01T10:00:00"):
    store.encode_scene(sid, body=body, topic=topic, heat=heat,
                       updated_ts=updated_ts, user_id=user_id,
                       source_eps=source_eps, body_embedding=_embed(body))


def _orchestrator(
    tmp_path,
    episodes: list[Episode] | None = None,
    reply: str = "SYNTH RESPONSE",
    db_subdir: str = "db",
    scene_blocks: bool = False,
) -> PonderOrchestrator:
    store = HippocampalStore(str(tmp_path / db_subdir))
    for ep in (episodes or []):
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(_TOPICS_PLAN),
                                     embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / db_subdir / "sessions")
    mode_a = _StubModeA(reply=reply)
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=mode_a, config=cfg,
        user_id="victor",
        scene_blocks=scene_blocks,
    )
    return orch


def _system_msg_contents(calls):
    """Yield the ``content`` of every ``system`` role message across the LLM call
    list. Sibling of ``_user_msg_contents`` (which only scans user role) -- the
    scene block lives in the SYSTEM prompt under A3, so the user-role scanner
    would not see it."""
    for msgs in calls:
        for m in msgs:
            if m.get("role") == "system":
                yield m["content"]


def _user_msg_contents(calls):
    """Yield the ``content`` of every ``user`` role message across the LLM call
    list (mirrors ``tests/test_fade_serve_integration._user_msg_contents``)."""
    for msgs in calls:
        for m in msgs:
            if m.get("role") == "user":
                yield m["content"]


# ──────────────────────────────────────────────────────────────────────────
# format_scene_block unit tests
# ──────────────────────────────────────────────────────────────────────────

def test_format_scene_block_empty():
    """No scenes -> ``""`` (no empty header -> the orchestrator's no-append path
    stays byte-identical to flag-off). Mirrors ``format_fade_block``."""
    assert format_scene_block([]) == ""
    assert format_scene_block(None) == ""  # tolerant: best-effort by construction


def test_format_scene_block_renders():
    """A scene renders as a ``[SCENE MEMORY]`` header + a per-scene head
    (id/topic/heat) + Topics line + the Markdown body. Mirrors the
    ``_format_episode`` ``kind == "scene"`` shape, but as a system-prompt block."""
    scenes = [
        {"episode_id": "scene_000001", "kind": "scene", "summary": "storage",
         "text": "# storage macro\nlots of detail", "topics": ["storage"],
         "heat": 0.7, "entities": [], "tones": [], "decisions": []},
        {"episode_id": "scene_000002", "kind": "scene", "summary": "redis",
         "text": "# redis cache notes", "topics": ["redis", "cache"],
         "heat": 0.42, "entities": [], "tones": [], "decisions": []},
    ]
    block = format_scene_block(scenes)
    assert "[SCENE MEMORY" in block
    assert "--- Scene scene_000001 (topic: storage, heat: 0.70) ---" in block
    assert "Topics: storage" in block
    assert "# storage macro" in block
    assert "--- Scene scene_000002 (topic: redis, heat: 0.42) ---" in block
    assert "Topics: redis, cache" in block
    assert "# redis cache notes" in block
    # A scene with no body still renders its head (no body line, no raise).
    head_only = format_scene_block([
        {"episode_id": "scene_000003", "kind": "scene", "summary": "x",
         "text": "", "topics": [], "heat": 0.1}])
    assert "--- Scene scene_000003" in head_only
    assert "Topics" not in head_only  # no topics -> no Topics line


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator seam: scene -> system-prompt suffix
# ──────────────────────────────────────────────────────────────────────────

def test_off_no_scenes_messages_byte_identical(tmp_path):
    """``scene_blocks=True`` + an EMPTY scene store -> ``mode_a.calls`` deep-equal
    to a ``scene_blocks=False`` run. The messages-level byte-identical gate the
    suite previously lacked (the fade tests pin it only indirectly, with
    scene_blocks unset). No scenes retrieved -> filter removes nothing ->
    ``scene_results`` empty -> no system-prompt append."""
    eps = [_ep("ep_001", summary="We chose Postgres for storage",
               text="User: storage?\nAssistant: Postgres WAL")]
    on = _orchestrator(tmp_path, eps, reply="LLM SAID THIS",
                       db_subdir="on", scene_blocks=True)
    off = _orchestrator(tmp_path, eps, reply="LLM SAID THIS",
                        db_subdir="off", scene_blocks=False)
    on.query("What have we figured out about storage for the write-ahead log?")
    off.query("What have we figured out about storage for the write-ahead log?")
    assert on.mode_a.calls and off.mode_a.calls  # both reached the LLM
    assert on.mode_a.calls == off.mode_a.calls  # byte-identical (no scenes)
    on.store.close()
    off.store.close()


def test_on_scene_in_system_not_user(tmp_path):
    """``scene_blocks=True`` + a scene in the store (topic matches the query) ->
    the scene body is in the SYSTEM message (``[SCENE MEMORY]`` suffix), NOT the
    user message. This is the A3 split: session-stable macro memory in the
    cacheable system prompt, per-query evidence in the user message."""
    eps = [_ep("ep_001", summary="We chose Postgres for storage",
               text="User: storage?\nAssistant: Postgres WAL")]
    orch = _orchestrator(tmp_path, eps, reply="LLM SAID THIS",
                         db_subdir="on", scene_blocks=True)
    # Encode a scene into the same store AFTER construction (the retriever reads
    # it live). user_id matches the orchestrator; the retriever has user_id=None
    # -> scope is None -> the scene is visible (the split is the focus here, not
    # scope; two-user scope is pinned in test_scene_blocks.py).
    sid = orch.store.next_scene_id()
    _encode_scene(orch.store, sid, body="# storage macro understanding",
                  topic="storage", heat=0.9, user_id="victor",
                  source_eps=["ep_001"])
    orch.query("What have we figured out about storage for the write-ahead log?")
    assert orch.mode_a.calls  # reached the LLM (synthesize end state)
    sys_msgs = list(_system_msg_contents(orch.mode_a.calls))
    user_msgs = list(_user_msg_contents(orch.mode_a.calls))
    # The scene is in the SYSTEM prompt...
    assert any("[SCENE MEMORY" in c for c in sys_msgs)
    assert any("storage macro understanding" in c for c in sys_msgs)
    assert any("topic: storage" in c for c in sys_msgs)
    # ...and NOT in the user message (the split removed it from the context blob).
    assert not any("storage macro understanding" in c for c in user_msgs)
    assert not any("[SCENE MEMORY" in c for c in user_msgs)
    orch.store.close()


def test_off_with_existing_scenes_renders_in_user_context(tmp_path):
    """``scene_blocks=False`` + a scene in the store -> the scene still renders in
    the USER message (pre-A3 behavior via ``_format_episode``'s
    ``kind == "scene"`` branch); the system prompt is unchanged. Pins the gate:
    the split is active ONLY when ``--scene-blocks`` is on."""
    eps = [_ep("ep_001", summary="We chose Postgres for storage",
               text="User: storage?\nAssistant: Postgres WAL")]
    orch = _orchestrator(tmp_path, eps, reply="LLM SAID THIS",
                         db_subdir="off", scene_blocks=False)
    sid = orch.store.next_scene_id()
    _encode_scene(orch.store, sid, body="# storage macro understanding",
                  topic="storage", heat=0.9, user_id="victor",
                  source_eps=["ep_001"])
    orch.query("What have we figured out about storage for the write-ahead log?")
    assert orch.mode_a.calls  # reached the LLM
    sys_msgs = list(_system_msg_contents(orch.mode_a.calls))
    user_msgs = list(_user_msg_contents(orch.mode_a.calls))
    # Flag off -> no split -> scene in the USER context, system prompt unchanged.
    assert any("storage macro understanding" in c for c in user_msgs)
    assert not any("[SCENE MEMORY" in c for c in sys_msgs)
    assert not any("storage macro understanding" in c for c in sys_msgs)
    orch.store.close()


def test_scene_inject_failure_swallowed(tmp_path):
    """A ``format_scene_block`` failure is swallowed (the turn proceeds, system
    prompt unchanged) -- mirroring the fade-inject best-effort swallow. Injects a
    raising formatter via the orchestrator's DI slot."""
    eps = [_ep("ep_001", summary="We chose Postgres for storage",
               text="User: storage?\nAssistant: Postgres WAL")]
    orch = _orchestrator(tmp_path, eps, reply="LLM SAID THIS",
                         db_subdir="on", scene_blocks=True)
    sid = orch.store.next_scene_id()
    _encode_scene(orch.store, sid, body="# storage macro understanding",
                  topic="storage", heat=0.9, user_id="victor",
                  source_eps=["ep_001"])

    def _raising(scenes):
        raise RuntimeError("boom")

    orch._format_scene_block = _raising  # inject a failing formatter
    res = orch.query("What have we figured out about storage for the write-ahead log?")
    # The turn proceeded (the swallow did not break it).
    assert orch.mode_a.calls
    assert res["response"] == "LLM SAID THIS"
    # The system prompt is unchanged (the failing block was not appended).
    sys_msgs = list(_system_msg_contents(orch.mode_a.calls))
    assert not any("[SCENE MEMORY" in c for c in sys_msgs)
    orch.store.close()