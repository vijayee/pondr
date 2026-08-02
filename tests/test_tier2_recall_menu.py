"""Tier-2 recall menu -- the on-demand ``remember`` tool.

The tiered context-assembly model has tier 0 (fade, ``--fade-inject``) and
tier 1 (WaveDB top-k "related from long ago") both auto-stuffed (zero round-
trips). Tier 2 is the "maybes" lane: an on-demand ``remember`` tool the LLM
calls mid-generation when it senses a gap. The menu is SYSTEM-PROPOSED (the
LLM names nothing -- the system builds the candidate set) and LLM-FILTERED
(the LLM reads the tool-result and uses the relevant items). Sources: R4
fade-forgotten anchors from this session (``cos < cos_gist`` -- the signal
``format_fade_block`` skips) ∪ WaveDB-tail hits beyond the tier-1 top-k cutoff
(``config.default_retrieval_limit``). One round-trip. Flag-gated default OFF,
byte-identical when off; loop-path-only (the one-shot path never gets the
schema).

Reuses the fade-serve stubs (``_StubEmbedder`` / ``_StubPlanner`` / ``_ep`` /
``_fade``). Adds a mode_a stub that records ``(messages, tools)`` per call and
can emit a ``remember`` tool_call so the loop iterates. Offline: no Bonsai,
no GLiNER, no token-LM (the fade stub / real ``_fade()`` provide the R4 source;
a stub ``vector_search`` provides the WaveDB-tail source).
"""

from __future__ import annotations

import pytest

from src.config import Phase2cConfig, config as _gcfg
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator, _REMEMBER_TOTAL_CAP
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.tools import (
    LOOP_TOOLS, REMEMBER_SCHEMA, SELF_CHAT_TOOLS, TOOL_SCHEMAS,
)

# Reuse the fade-serve stubs (deterministic 384-d embedder, stub planner, the
# real FadeMemory helper, the episode builder).
from tests.test_fade_serve_integration import (
    _StubEmbedder, _StubPlanner, _ep, _fade,
)


# ── stubs ──

class _ModeARecorder:
    """mode_a stub that records ``(messages, tools)`` per call.

    The existing ``_StubModeA`` records only ``messages`` and drops ``tools``,
    which makes a tool-schema byte-identical proof vacuous. This recorder keeps
    both so the load-bearing assertion -- the flag-off path hands the consumer
    the EXACT prior tool objects (no ``REMEMBER_SCHEMA``) -- can be made.

    When ``emit_remember`` is set, the FIRST call returns a ``remember``
    tool_call (so ``run_tool_loop`` dispatches it and iterates); every later
    call returns ``(reply, None)`` so the loop terminates with ``content=reply``.
    """
    def __init__(self, reply: str = "SYNTH RESPONSE", emit_remember: bool = False) -> None:
        self.reply = reply
        self.emit_remember = emit_remember
        self.calls: list[tuple[list[dict], object]] = []
        self._fired = False

    def _complete(self, messages, tools=None, tool_choice=None) -> tuple:
        self.calls.append((messages, tools))
        if self.emit_remember and not self._fired:
            self._fired = True
            tool_calls = [{
                "id": "call_remember_1", "type": "function",
                "function": {"name": "remember", "arguments": "{}"},
            }]
            return self.reply, tool_calls
        return self.reply, None


class _R4Fade:
    """FadeMemory stub: two anchors both below ``cos_gist`` (strict R4),
    neither in the recency ring.

    Exposes EXACTLY the surface ``PonderOrchestrator.remember_menu`` reads
    (``blurbs._ids`` / ``blurbs.text`` / ``ring`` / ``cfg.cos_gist`` /
    ``_recoverability``) and nothing more. Used both for the direct unit test
    (``remember_menu`` called directly) and the dispatched test (``query`` --
    the recall/ingest seams swallow the missing ``recall``/``ingest`` attrs).
    """
    class cfg:
        cos_gist = 0.40

    class blurbs:
        _ids = [0, 1]

        @staticmethod
        def text(aid):
            return f"blurb {aid}"

    ring: set = set()  # empty -> neither anchor is in the verbatim window
    _cos = {0: 0.05, 1: 0.10}

    def _recoverability(self, aid):
        return self._cos[aid]


class _StubVectorSearch:
    """Vector-search stub: ``search(query, k)`` returns a fixed hit list,
    truncated to ``k`` (so the tier1_k + tail_n over-fetch then ``hits[tier1_k:]``
    slice in ``remember_menu`` is exercised). ``search_by_vector`` is unused by
    ``remember_menu`` (it deliberately uses ``search`` to reproduce tier-1's
    ranking)."""

    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self._hits = hits

    def search(self, query: str, k: int = 5):
        return self._hits[:k]


def _orch2(tmp_path, *, plan=None, episodes=None, reply="SYNTH RESPONSE",
           fade_memory=None, db_subdir="db", fade_inject=False,
           tier2_recall_menu=False, vector_search=None, mode_a=None,
           retriever_user_id=None):
    """Build an orchestrator for tier-2 tests.

    Mirrors ``test_fade_serve_integration._orchestrator`` but threads the
    ``tier2_recall_menu`` flag, lets a stub ``vector_search`` be injected onto
    the retriever, defaults ``mode_a`` to the ``(messages, tools)`` recorder,
    and optionally sets the retriever's ``user_id`` (the retrieval user-scope
    boundary -- ``None`` = scope off, byte-identical; the orchestrator's own
    ``user_id`` stays "victor" and does not affect retrieval scope).
    """
    store = HippocampalStore(str(tmp_path / db_subdir))
    for ep in (episodes or []):
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan or {}),
                                     embedder=_StubEmbedder(),
                                     user_id=retriever_user_id)
    if vector_search is not None:
        retriever.vector_search = vector_search
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / db_subdir / "sessions")
    if mode_a is None:
        mode_a = _ModeARecorder(reply=reply)
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=mode_a, config=cfg, user_id="victor",
        fade_memory=fade_memory, fade_inject=fade_inject,
        tier2_recall_menu=tier2_recall_menu,
    )
    return orch


def _tail_eps():
    """Eight episodes ``ep_t0..ep_t7`` whose ids the stub vector-search returns;
    tier-1 cutoff is 5, so ``ep_t5..ep_t7`` are the WaveDB-tail maybes."""
    return [_ep(f"ep_t{i}", entities=["Postgres"], summary=f"tail summary {i}")
            for i in range(8)]


# ── 1. unit: remember_menu() builds the menu directly ──

def test_remember_menu_unit(tmp_path):
    """Direct ``remember_menu()`` call: R4 items are strict (cos < cos_gist),
    tail items are the hits beyond the tier-1 cutoff, the head (``hits[:tier1_k]``)
    is excluded, the total is capped, the structured stash is set, and the
    rendered string is non-empty + within the soft token cap."""
    hits = [(f"ep_t{i}", 0.9 - 0.05 * i) for i in range(8)]  # 8 hits, score desc
    orch = _orch2(tmp_path, episodes=_tail_eps(), db_subdir="unit",
                 tier2_recall_menu=True,
                 vector_search=_StubVectorSearch(hits),
                 fade_memory=_R4Fade())
    orch._current_query = "test query"           # the tail source reads this
    text = orch.remember_menu()
    assert text                                  # non-empty -> dispatch returns it
    assert "[FADE R4" in text
    assert "[LONG AGO" in text
    items = orch._last_remember_menu
    assert isinstance(items, list) and items
    assert len(items) <= _REMEMBER_TOTAL_CAP
    r4 = [i for i in items if i["source"] == "fade_r4"]
    tail = [i for i in items if i["source"] == "wavedb_tail"]
    assert len(r4) == 2                          # both stub anchors are R4
    assert all(i["cos"] < 0.40 for i in r4)     # strict R4 (cos < cos_gist)
    # R4 most-faded-first: lowest cos (0.05) before higher (0.10).
    assert r4[0]["cos"] < r4[1]["cos"]
    assert len(tail) == 3                        # 8 hits - tier1_k(5) = 3
    tail_ids = {i["episode_id"] for i in tail}
    assert tail_ids == {"ep_t5", "ep_t6", "ep_t7"}   # the head is excluded
    assert not (tail_ids & {"ep_t0", "ep_t1", "ep_t2", "ep_t3", "ep_t4"})
    # Soft token cap (~4 chars/token): drop-not-truncate keeps it bounded.
    assert len(text) <= 512 * 4 + 64
    orch.store.close()


# ── 2. flag off -> empty + no stash ──

def test_remember_menu_off_returns_empty(tmp_path):
    """``tier2_recall_menu=False`` -> ``remember_menu`` short-circuits to "" and
    leaves ``_last_remember_menu`` None (the flag-off path)."""
    orch = _orch2(tmp_path, episodes=_tail_eps(), db_subdir="off",
                 tier2_recall_menu=False,
                 vector_search=_StubVectorSearch([("ep_t0", 0.9)]),
                 fade_memory=_R4Fade())
    orch._current_query = "test query"
    assert orch.remember_menu() == ""
    assert orch._last_remember_menu is None
    orch.store.close()


# ── 3. byte-identical off (the load-bearing assertion) ──

def test_byte_identical_off(tmp_path):
    """Flag OFF -> the consumer (``mode_a._complete``) sees the EXACT prior tool
    objects (no ``REMEMBER_SCHEMA``) and identical messages. Flag ON (LLM stub
    never calls ``remember``) -> ``REMEMBER_SCHEMA`` IS appended to the loop-path
    tool set (a NEW list; the module-level lists are not mutated) and the
    messages are otherwise unchanged. The tools-list comparison is the real
    byte-identical proof (comparing the stub reply alone would be vacuous)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]

    orch_off = _orch2(tmp_path, plan=plan, episodes=eps, reply="LLM SAID THIS",
                     db_subdir="off", tier2_recall_menu=False,
                     mode_a=_ModeARecorder(reply="LLM SAID THIS"))
    orch_off.query("Why did we choose Postgres?")
    off_calls = orch_off.mode_a.calls
    orch_off.store.close()

    eps2 = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch_on = _orch2(tmp_path, plan=plan, episodes=eps2, reply="LLM SAID THIS",
                    db_subdir="on", tier2_recall_menu=True,
                    mode_a=_ModeARecorder(reply="LLM SAID THIS"))
    orch_on.query("Why did we choose Postgres?")
    on_calls = orch_on.mode_a.calls
    orch_on.store.close()

    # The loop path is the first _complete call (synthesize). feedback is ON by
    # default -> the base set is TOOL_SCHEMAS (not LOOP_TOOLS). Compute the
    # expected base so the test is robust to the feedback default.
    feedback_on = _gcfg.feedback_salience_enabled
    expected_base = TOOL_SCHEMAS if feedback_on else LOOP_TOOLS

    # OFF: no REMEMBER_SCHEMA, the exact prior list object (not a copy).
    off_tools = [c[1] for c in off_calls]
    assert off_tools[0] is expected_base
    assert REMEMBER_SCHEMA not in off_tools[0]

    # ON: REMEMBER_SCHEMA appended as a NEW list; the base entries are unchanged.
    on_tools = [c[1] for c in on_calls]
    assert on_tools[0] == [*expected_base, REMEMBER_SCHEMA]
    assert on_tools[0] is not expected_base         # a new list, not a mutation
    assert on_tools[0][:len(expected_base)] == expected_base
    assert REMEMBER_SCHEMA in on_tools[0]

    # Messages identical off vs on (the schema is appended, nothing else moves;
    # the LLM never called remember, so no tool-result message differs).
    assert [c[0] for c in off_calls] == [c[0] for c in on_calls]


# ── 4. the LLM calls remember -> menu dispatched ──

def test_remember_tool_call_dispatched(tmp_path):
    """Loop enabled + the stub emits a ``remember`` tool_call: the menu string
    appears as a ``tool``-role message in the loop transcript, ``mode_a`` got >=2
    calls (the follow-up sees the menu tool-result), ``result["remember_menu"]``
    is the structured list, and the response is the stub reply."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch = _orch2(tmp_path, plan=plan, episodes=eps, reply="LLM SAID THIS",
                 db_subdir="on", tier2_recall_menu=True,
                 fade_memory=_R4Fade(),
                 mode_a=_ModeARecorder(reply="LLM SAID THIS", emit_remember=True))
    res = orch.query("Why did we choose Postgres?")
    assert res["response"] == "LLM SAID THIS"
    # The loop iterated: first call emitted remember, second call saw the result.
    assert len(orch.mode_a.calls) >= 2
    loop_msgs = res.get("loop_tool_messages") or []
    tool_results = [m.get("content", "") for m in loop_msgs if m.get("role") == "tool"]
    assert any("[REMEMBER MENU" in c or "[FADE R4" in c for c in tool_results)
    # The structured menu is surfaced for observation.
    assert isinstance(res.get("remember_menu"), list) and res["remember_menu"]
    assert all(i["source"] == "fade_r4" for i in res["remember_menu"])  # no tail (no vector_search)
    orch.store.close()


# ── 5. remember_menu failure is swallowed ──

def test_remember_menu_failure_swallowed(tmp_path):
    """If ``remember_menu`` raises, ``dispatch_tool``'s outer ``except`` wraps it
    into an ``_err`` JSON tool-result and the loop continues -> the query still
    returns the stub reply (a tool failure can't break the consumer's loop)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch = _orch2(tmp_path, plan=plan, episodes=eps, reply="LLM SAID THIS",
                 db_subdir="on", tier2_recall_menu=True, fade_memory=_R4Fade(),
                 mode_a=_ModeARecorder(reply="LLM SAID THIS", emit_remember=True))

    def _boom():
        raise RuntimeError("remember boom")
    orch.remember_menu = _boom                    # instance override -> dispatch sees the raise

    res = orch.query("Why did we choose Postgres?")
    assert res["response"] == "LLM SAID THIS"
    loop_msgs = res.get("loop_tool_messages") or []
    tool_results = [m.get("content", "") for m in loop_msgs if m.get("role") == "tool"]
    assert any(c.startswith('{"error"') for c in tool_results)   # the _err string
    orch.store.close()


# ── 6. loop disabled -> remember is never dispatched (loop-path-only) ──

def test_remember_loop_disabled_no_call(tmp_path, monkeypatch):
    """Tier-2 ON but the tool loop OFF -> one-shot path: ``_complete`` is called
    once with ``SELF_CHAT_TOOLS``/``None`` (no ``REMEMBER_SCHEMA``), ``remember``
    is never dispatched (even if the stub emits the call -- the one-shot path's
    ``_dispatch_feedback`` ignores non-record_feedback calls), and
    ``result["remember_menu"]`` is ABSENT. Pins the loop-path-only contract."""
    monkeypatch.setattr(_gcfg, "self_chat_tool_loop_enabled", False)
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_ep("ep_001", entities=["Postgres"], summary="We chose Postgres")]
    orch = _orch2(tmp_path, plan=plan, episodes=eps, reply="LLM SAID THIS",
                 db_subdir="on", tier2_recall_menu=True, fade_memory=_R4Fade(),
                 mode_a=_ModeARecorder(reply="LLM SAID THIS", emit_remember=True))

    called: list = []
    orch.remember_menu = lambda: called.append(1) or ""   # spy: must NOT be called

    res = orch.query("Why did we choose Postgres?")
    assert res["response"] == "LLM SAID THIS"
    assert called == []                          # the one-shot path never dispatched remember
    assert "remember_menu" not in res             # ABSENT (loop off -> _last_remember_menu stays None)
    # The one-shot path's tool set has no REMEMBER_SCHEMA.
    one_shot_tools = orch.mode_a.calls[0][1]
    assert one_shot_tools is None or REMEMBER_SCHEMA not in one_shot_tools
    if _gcfg.feedback_salience_enabled:
        assert one_shot_tools is SELF_CHAT_TOOLS   # feedback on -> SELF_CHAT_TOOLS (no remember)
    orch.store.close()


# ── 7. user-scope: the tail filters cross-user hits (the 339cdb9 boundary) ──

def test_remember_menu_tail_user_scoped(tmp_path):
    """Under retriever user-scope (``retriever.user_id`` set), the WaveDB-tail
    source filters cross-user hits via ``_user_scope_sets`` +
    ``_filter_vector_hits_by_scope`` -- only the query user's OWNED episodes
    survive into the tail. This is the read-side boundary shipped in 339cdb9,
    applied to the tier-2 tail which (pre-fix) called ``vector_search.search``
    directly and would have leaked bob's content into alice's ``remember`` menu.

    Alice owns 8 episodes, bob owns 4; the stub vector-search returns all 12
    interleaved (so bob's are genuinely in the over-fetched candidate set).
    With the filter, bob's are dropped; ``hits[tier1_k:]`` is alice-only.
    """
    def _owned(eid, user, sess, summary):
        return Episode(id=eid, timestamp="2026-07-03T10:00:00",
                       summary=summary, full_text=summary,
                       entities=["Alice"], topics=[], tones=[],
                       user_id=user, session_id=sess)

    alice_eps = [_owned(f"ep_a{i}", "alice", "S:a1", f"alice {i}") for i in range(8)]
    bob_eps = [_owned(f"ep_b{i}", "bob", "S:b1", f"bob {i}") for i in range(4)]
    # Interleaved hits, descending score; bob's are interspersed so they sit in
    # the over-fetched set (the filter must drop them, not the score cutoff).
    hits = []
    for i in range(8):
        hits.append((f"ep_a{i}", 0.95 - 0.01 * i))
        if i < 4:
            hits.append((f"ep_b{i}", 0.945 - 0.01 * i))

    orch = _orch2(tmp_path, episodes=alice_eps + bob_eps, db_subdir="scope",
                 tier2_recall_menu=True, retriever_user_id="alice",
                 vector_search=_StubVectorSearch(hits))
    orch._current_query = "tell me about Alice"

    # Sanity: the retriever's scope set is alice-only (bob is in the store but
    # not alice's) -- so this is a real exclusion, not an empty corpus.
    allowed_ep, _, _ = orch.retriever._user_scope_sets()
    assert "ep_a0" in allowed_ep and "ep_b0" not in allowed_ep

    orch.remember_menu()
    tail = [i for i in orch._last_remember_menu if i["source"] == "wavedb_tail"]
    tail_ids = {i["episode_id"] for i in tail}
    assert tail_ids, "tail should have alice's episodes beyond the tier-1 cutoff"
    assert all(i.startswith("ep_a") for i in tail_ids)        # alice only
    assert not any(i.startswith("ep_b") for i in tail_ids)    # bob excluded
    orch.store.close()


def test_remember_menu_tail_scope_off_is_byte_identical(tmp_path):
    """``retriever_user_id=None`` -> ``_user_scope_sets`` returns ``(None, None, None)``
    -> no filter, no over-fetch multiplier -> the tail is byte-identical to the
    pre-scope path (bob's hits survive into the tail). Pins the byte-identical-
    off contract for the tail source specifically."""
    def _owned(eid, user, sess, summary):
        return Episode(id=eid, timestamp="2026-07-03T10:00:00",
                       summary=summary, full_text=summary,
                       entities=["Alice"], topics=[], tones=[],
                       user_id=user, session_id=sess)

    alice_eps = [_owned(f"ep_a{i}", "alice", "S:a1", f"alice {i}") for i in range(6)]
    bob_eps = [_owned(f"ep_b{i}", "bob", "S:b1", f"bob {i}") for i in range(4)]
    hits = [(f"ep_a{i}", 0.95 - 0.01 * i) for i in range(6)] + \
           [(f"ep_b{i}", 0.90 - 0.01 * i) for i in range(4)]
    orch = _orch2(tmp_path, episodes=alice_eps + bob_eps, db_subdir="noscope",
                 tier2_recall_menu=True, retriever_user_id=None,
                 vector_search=_StubVectorSearch(hits))
    orch._current_query = "tell me about Alice"
    orch.remember_menu()
    tail = [i for i in orch._last_remember_menu if i["source"] == "wavedb_tail"]
    tail_ids = {i["episode_id"] for i in tail}
    # tier1_k(5) of 10 hits -> 5 tail hits, BOTH users present (no filter).
    bob_tail = {i for i in tail_ids if i.startswith("ep_b")}
    assert bob_tail, "scope OFF -> bob's hits are NOT filtered out of the tail"
    orch.store.close()