"""Phase-A serve integration tests for the FadeMemory wiring.

Exercises the orchestrator seams added in Phase A (``fade_memory`` DI kwarg +
recall/ingest/result surfacing) on a tmp_path WaveDB store with the same stubs
``test_orchestrator.py`` uses (stub embedder/planner/mode_a, ReferenceSSM
backbone). No bge, no Ollama, no token-LM -- the FadeMemory uses a synthetic
embedder + ``voice=None`` (the built-in passthrough).

Scope (Phase A): the fade recalls are surfaced as ``result["fade_recalls"]``
for observation and the fade ingests each exchange, but the recalls are NOT
fed into the LLM context -- so the user-facing response is byte-identical to
flag-off. These tests pin that contract.
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
from src.subconscious.fade import FadeConfig, FadeMemory


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


# ── fixtures ──

def _ep(eid, entities=None, summary=None, text=None,
        ts="2026-07-03T10:00:00") -> Episode:
    return Episode(
        id=eid, timestamp=ts,
        summary=summary or f"summary {eid}",
        full_text=text or f"User: u{eid}\nAssistant: a{eid}",
        entities=entities or [], topics=[], tones=[], decisions=[],
    )


def _orchestrator(
    tmp_path,
    plan: dict | None = None,
    episodes: list[Episode] | None = None,
    reply: str = "SYNTH RESPONSE",
    fade_memory=None,
    db_subdir: str = "db",
    fade_inject: bool = False,
) -> PonderOrchestrator:
    store = HippocampalStore(str(tmp_path / db_subdir))
    for ep in (episodes or []):
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan or {}),
                                     embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / db_subdir / "sessions")
    mode_a = _StubModeA(reply=reply)
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=mode_a, config=cfg,
        user_id="victor",
        fade_memory=fade_memory,
        fade_inject=fade_inject,
    )
    return orch


def _fade() -> FadeMemory:
    """A FadeMemory with a synthetic embedder + ``voice=None`` (passthrough).

    ``decay=0.5`` + ``cos_gist=0.01`` so a same-doc anchor transitions to gist
    within a couple of ingests on the synthetic embedder (whose cross-doc floor
    is ~0.01, unlike real bge's ~0.37 -- see docs/fade-cross-domain-eval-result.md).
    """
    return FadeMemory(
        FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.01, ring_capacity=8),
        _StubEmbedder(), None,
    )


# ── tests ──

def _user_msg_contents(calls):
    """Yield the ``content`` of every ``user`` role message across the LLM call
    list (``mode_a.calls`` is a list of message-lists). Used to scan for the
    ``[FADE MEMORY]`` block without assuming which call was the synthesize one
    (the orchestrator may make several LLM calls per query -- tool loop etc.)."""
    for msgs in calls:
        for m in msgs:
            if m.get("role") == "user":
                yield m["content"]


def test_flag_off_no_fade_recalls(tmp_path):
    """``fade_memory=None`` -> ``fade_recalls`` ABSENT (byte-identical to pre-fade)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS")
    res = orch.query("Why did we choose Postgres?")
    assert "fade_recalls" not in res
    assert res["response"] == "LLM SAID THIS"
    orch.store.close()


def test_flag_on_surfaces_fade_recalls(tmp_path):
    """``fade_memory`` wired -> ``fade_recalls`` is a list; the first exchange
    is ingested into the fade blurb store."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    fade = _fade()
    orch = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                         fade_memory=fade)
    # First query: the fade has no prior anchors yet -> empty recalls, but the
    # exchange IS ingested (after the response is built). "Why..." forces the
    # synthesize end-state so a response is produced (the ingest only runs on
    # a non-empty response).
    res1 = orch.query("Why did we choose Postgres for the write-ahead log?")
    assert "fade_recalls" in res1
    assert isinstance(res1["fade_recalls"], list)
    assert len(fade.blurbs) == 1                  # the first exchange ingested
    # Second query: the fade now has one anchor to route.
    res2 = orch.query("Why did we choose Postgres?")
    assert isinstance(res2["fade_recalls"], list)
    assert len(fade.blurbs) == 2                  # second exchange ingested
    orch.store.close()


def test_flag_on_response_byte_identical_to_off(tmp_path):
    """The fade never touches the LLM context in Phase A -> the messages
    passed to the LLM (``mode_a.calls``) are identical whether the flag is on
    or off. Comparing the stub reply alone would be trivially true (the stub
    ignores its messages); comparing the actual messages is the real proof
    that no fade recalls leaked into the context. Uses separate db subdirs so
    the two runs have identical, non-leaking store state."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch_off = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                            db_subdir="off")
    res_off = orch_off.query("Why did we choose Postgres?")
    orch_off.store.close()

    eps2 = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch_on = _orchestrator(tmp_path, plan, eps2, reply="LLM SAID THIS",
                            fade_memory=_fade(), db_subdir="on")
    res_on = orch_on.query("Why did we choose Postgres?")
    assert res_on["response"] == res_off["response"]
    # The real byte-identical proof: the LLM saw the SAME messages with the
    # fade on as off (no fade recalls appended to the context).
    assert orch_on.mode_a.calls == orch_off.mode_a.calls
    assert "fade_recalls" in res_on
    assert "fade_recalls" not in res_off
    orch_on.store.close()


def test_fade_ingest_failure_does_not_break_query(tmp_path):
    """A FadeMemory whose ``ingest`` raises -> the query still returns a
    response (best-effort swallow, mirroring _run_salience_hook)."""

    class _BrokenIngest:
        # The orchestrator only calls .ingest() / .recall() and fetches
        # REGIME_NAME from the module (not the instance), so no other attrs.
        def ingest(self, chunk_text) -> int:
            raise RuntimeError("ingest boom")

        def recall(self, query_text, top_k=5):
            return []

    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                        fade_memory=_BrokenIngest())
    res = orch.query("Why did we choose Postgres?")  # must not raise
    assert res["response"] == "LLM SAID THIS"
    orch.store.close()


def test_fade_recall_failure_does_not_break_query(tmp_path):
    """A FadeMemory whose ``recall`` raises -> the query still returns a
    response and ``fade_recalls`` is present-and-empty (the recall seam
    swallows the exception, leaving the list ``[]``; the result-augment seam
    still adds the key because ``self._fade is not None``)."""

    class _BrokenRecall:
        def ingest(self, chunk_text) -> int:
            return 0

        def recall(self, query_text, top_k=5):
            raise RuntimeError("recall boom")

    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                        fade_memory=_BrokenRecall())
    res = orch.query("Why did we choose Postgres?")  # must not raise
    assert res["response"] == "LLM SAID THIS"
    # Recall swallowed -> the list stays empty; the key is still present
    # (the result-augment seam fires whenever self._fade is not None).
    assert res.get("fade_recalls") == []
    orch.store.close()


# ── Phase B: --fade-inject feeds recalls into the LLM context ──

def test_inject_off_messages_byte_identical_to_off(tmp_path):
    """``fade_inject=False`` (the default) -> the fade recalls are NOT fed into
    the LLM context, so ``mode_a.calls`` (the LLM messages) are byte-identical
    to flag-off. This is the Phase A contract, preserved when --fade-inject is
    absent. (Compare to ``test_inject_on_messages_contain_fade_block``.)"""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch_off = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                             db_subdir="off")
    orch_off.query("Why did we choose Postgres?")
    off_calls = [c for c in orch_off.mode_a.calls]
    orch_off.store.close()

    eps2 = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch_on = _orchestrator(tmp_path, plan, eps2, reply="LLM SAID THIS",
                            fade_memory=_fade(), db_subdir="on",
                            fade_inject=False)
    orch_on.query("Why did we choose Postgres?")
    # The LLM saw the SAME messages with the fade on (inject OFF) as off -- no
    # fade block leaked into the context.
    assert orch_on.mode_a.calls == off_calls
    orch_on.store.close()


def test_inject_on_messages_contain_fade_block(tmp_path):
    """``fade_inject=True`` -> the fade recalls are formatted into a
    ``[FADE MEMORY]`` block and prepended to the LLM user message on synthesize
    turns, so ``mode_a.calls`` CONTAIN the block (the LLM saw the fade) and are
    NOT identical to flag-off. Two queries: the first ingests an anchor (no
    prior anchors -> empty recalls -> no block); the second recalls that anchor
    (still in-ring -> R1 verbatim) -> the block is injected."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                        fade_memory=_fade(), db_subdir="on",
                        fade_inject=True)
    # Query 1: no prior anchors -> empty recalls -> no block yet.
    orch.query("Why did we choose Postgres for the write-ahead log?")
    calls_after_q1 = len(orch.mode_a.calls)
    # Query 2: anchor 0 is now in-ring -> R1 verbatim -> the block is injected.
    res = orch.query("Why did we choose Postgres?")
    assert len(orch.mode_a.calls) > calls_after_q1  # query 2 reached the LLM
    # The recalls were non-empty (R1) and content-bearing.
    assert isinstance(res.get("fade_recalls"), list)
    assert any(r["regime"] == 1 for r in res["fade_recalls"])
    # The LLM user message CONTAINS the [FADE MEMORY] block (find the synthesize
    # call -- the one whose user message holds the fade block; the orchestrator
    # may make several LLM calls per query, so scan rather than assume a count).
    q2_user_msgs = list(_user_msg_contents(orch.mode_a.calls[calls_after_q1:]))
    assert any("[FADE MEMORY" in c for c in q2_user_msgs)
    assert any("[verbatim, recent]" in c for c in q2_user_msgs)
    # Query 1's calls did NOT contain the block (no prior anchors -> empty).
    q1_user_msgs = list(_user_msg_contents(orch.mode_a.calls[:calls_after_q1]))
    assert not any("[FADE MEMORY" in c for c in q1_user_msgs)
    # And query 2 is NOT byte-identical to flag-off (the block changed the msg).
    orch_off = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                             db_subdir="off2")
    orch_off.query("Why did we choose Postgres for the write-ahead log?")
    calls_off_after_q1 = len(orch_off.mode_a.calls)
    orch_off.query("Why did we choose Postgres?")
    # The off-run's query-2 user messages have NO fade block (the on-run did --
    # proven above -- so the two runs are not byte-identical).
    off_q2 = list(_user_msg_contents(orch_off.mode_a.calls[calls_off_after_q1:]))
    assert all("[FADE MEMORY" not in c for c in off_q2)
    orch.store.close()
    orch_off.store.close()


def test_inject_r4_only_no_block(tmp_path):
    """When every recall is R4 (forgotten), ``format_fade_block`` returns ``""``
    (R4 is a signal, not LLM content) -> the block is omitted entirely and the
    LLM messages are byte-identical to flag-off. R4 stays in ``fade_recalls``
    as the structured signal (the future long-term-memory pull consumes it).
    Uses a stub FadeMemory whose ``recall`` returns only R4 recalls so the test
    does not need many cross-domain ingests to drive a real anchor to R4."""
    from src.subconscious.fade import REGIME_FORGOTTEN, Recall

    class _R4OnlyFade:
        def ingest(self, chunk_text) -> int:
            return 0

        def recall(self, query_text, top_k=5):
            return [Recall(0, REGIME_FORGOTTEN, 0.05,
                           content="[forgotten]", blurb=None)]

    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                        fade_memory=_R4OnlyFade(), db_subdir="on",
                        fade_inject=True)
    res = orch.query("Why did we choose Postgres?")
    # R4 is present as the structured signal in fade_recalls...
    assert res.get("fade_recalls")
    assert all(r["regime"] == REGIME_FORGOTTEN for r in res["fade_recalls"])
    # ...but NOT injected into the LLM messages (R4 is a signal, not content):
    # no call's user message contains the block.
    assert not any("[FADE MEMORY" in c
                   for c in _user_msg_contents(orch.mode_a.calls))
    # Byte-identical to flag-off (no block when all recalls are R4): every LLM
    # call -- not just the last -- matches the off-run's calls.
    orch_off = _orchestrator(tmp_path, plan, eps, reply="LLM SAID THIS",
                             db_subdir="off")
    orch_off.query("Why did we choose Postgres?")
    assert orch.mode_a.calls == orch_off.mode_a.calls
    orch.store.close()
    orch_off.store.close()