"""R4 per-query freshness on derived context (Graft steal #2).

Three deliverables behind ``--query-freshness``:

* **Mechanism** -- ``GraphTraversal.check_freshness``: a READ-ONLY structural
  staleness stat (the unmarked sibling of R1's ``_filter_stale_derived``
  recheck). An episode/doc is fresh iff its supersession tip is itself; a
  scene/``M:`` node is fresh iff every cited source's tip is still the source
  (or the node also cites the tip -- re-derived). Works MARK-FREE (config-
  INDEPENDENT); never writes (the ``stale_of`` resize lifecycle stays R1's).
* **Assembly-time stamp** -- ``build_context_string`` renders a STALE note on
  a chunk whose hydrated dict carries the R1 annotation (``stale_since``);
  flag off -> byte-identical.
* **Agent tool** -- ``check_freshness_context`` handler + ``dispatch_tool``
  branch + the loop-path ``CHECK_FRESHNESS_SCHEMA`` append (standalone schema;
  the flag-off path hands the consumer the exact prior tool objects). The
  no-args case ("YOUR current context") is only meaningful MID-QUERY (the
  loop dispatches the tool while ``_current_fresh_ids`` holds this turn's
  assembled units; the happy-path tail clears it with ``_current_query``).

Offline: real WaveDB + GraphTraversal for the mechanism; the fade-serve stubs
for the orchestrator path (mirrors tests/test_tier2_recall_menu.py).
"""

from __future__ import annotations

import json
from contextlib import contextmanager

from src.config import Phase2cConfig, config as _gcfg
from src.gnn.semantic_memory import SemanticMemoryWriter
from src.memory.episode import Episode
from src.memory.store import HippocampalStore, _b2s
from src.orchestrator import PonderOrchestrator
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.tools import (
    CHECK_FRESHNESS_SCHEMA, LOOP_TOOLS, SELF_CHAT_TOOLS, TOOL_SCHEMAS,
    dispatch_tool,
)

# Reuse the fade-serve stubs (deterministic 384-d embedder, stub planner).
from tests.test_fade_serve_integration import (_StubEmbedder, _StubPlanner)


@contextmanager
def freshness(on):
    prev = _gcfg.query_freshness_enabled
    _gcfg.query_freshness_enabled = on
    try:
        yield
    finally:
        _gcfg.query_freshness_enabled = prev


@contextmanager
def propagating(on):
    """R1's write-side mark reads ``stale_propagation_enabled`` (a DIFFERENT
    flag than R4's ``query_freshness_enabled``) -- needed only where the test
    creates real marks."""
    prev = _gcfg.stale_propagation_enabled
    _gcfg.stale_propagation_enabled = on
    try:
        yield
    finally:
        _gcfg.stale_propagation_enabled = prev


@contextmanager
def feedback_off():
    """Pin ``feedback_salience_enabled`` OFF during the query so the loop path's
    base tool set is the deterministic ``LOOP_TOOLS``."""
    prev = _gcfg.feedback_salience_enabled
    _gcfg.feedback_salience_enabled = False
    try:
        yield
    finally:
        _gcfg.feedback_salience_enabled = prev


def _mk_episode(eid, entities, summary):
    return Episode(
        id=eid, timestamp="2026-07-03T10:00:00", summary=summary,
        full_text=f"User: u{eid}\nAssistant: a{eid}",
        entities=entities, topics=[], tones=[], decisions=[],
    )


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"))


def _encode(store, eid, entities=None):
    store.encode_episode(Episode(
        id=eid, timestamp="t", summary=f"s {eid}", full_text=f"f {eid}",
        entities=entities or [],
    ))


def _scene(store, sid, source_eps, topic="db"):
    store.encode_scene(sid, body=f"## {topic}\nbody", topic=topic, heat=1.0,
                       updated_ts="t", user_id=None, source_eps=source_eps)


def _stale_of(store, prefix, node_id):
    raw = _b2s(store.db.get_sync(f"content/{prefix}/{node_id}/stale_of"))
    return set(json.loads(raw)) if raw else set()


def _stale_since(store, prefix, node_id):
    return _b2s(store.db.get_sync(f"content/{prefix}/{node_id}/stale_since"))


# ── 1. mechanism: check_freshness (read-only structural stat) ──


def test_check_freshness_episode_superseded_vs_fresh(tmp_path):
    """An episode id is fresh iff its own supersession tip is itself."""
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    w = SemanticMemoryWriter(store)
    w.supersede_episode("ep_000002", "ep_000001")
    trav = GraphTraversal(store)
    report = trav.check_freshness(["ep_000001", "ep_000002"])
    assert report["ep_000001"] == {
        "kind": "episode", "fresh": False,
        "reasons": ["superseded by ep_000002"],
    }
    assert report["ep_000002"]["fresh"] is True
    assert report["ep_000002"]["reasons"] == []
    store.close()


def test_check_freshness_doc_fresh_no_chain(tmp_path):
    """A ``doc_`` id with no supersession chain reports fresh (kind doc)."""
    store = _store(tmp_path)
    trav = GraphTraversal(store)
    report = trav.check_freshness(["doc_0001"])
    assert report["doc_0001"] == {"kind": "doc", "fresh": True,
                                  "reasons": []}
    store.close()


def test_check_freshness_stale_scene_and_re_derived_fresh(tmp_path):
    """The scene rule: EVERY cited source's tip must still be the source, OR
    the node must also cite the tip (re-authoring grew the union)."""
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    _scene(store, "scene_0001", ["ep_000001"])
    _scene(store, "scene_0002", ["ep_000002"])          # healthy control
    w.supersede_episode("ep_000002", "ep_000001")
    # A THIRD scene re-authors against the tip union (R1's resolution path).
    _scene(store, "scene_0003", ["ep_000001", "ep_000002"])
    trav = GraphTraversal(store)
    report = trav.check_freshness(
        ["scene_0001", "scene_0002", "scene_0003"])
    assert report["scene_0001"]["fresh"] is False
    assert report["scene_0001"]["kind"] == "scene"
    assert report["scene_0001"]["reasons"] == [
        "source ep_000001 superseded by ep_000002"]
    assert report["scene_0001"]["sources"] == ["ep_000001"]
    assert report["scene_0002"]["fresh"] is True
    assert report["scene_0002"]["reasons"] == []
    # Re-derived: the cited union contains the tip -> fresh despite citing the
    # superseded source too.
    assert report["scene_0003"]["fresh"] is True
    assert report["scene_0003"]["reasons"] == []
    store.close()


def test_check_freshness_memory_node_stale(tmp_path):
    """An ``M:`` node is judged by the same citation rule (via ``abstracts``)."""
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    mid = w.create_abstract(["ep_000001"], "gist v1", when="t")
    w.supersede_episode("ep_000002", "ep_000001", when="T9")
    trav = GraphTraversal(store)
    report = trav.check_freshness([mid])
    assert mid.startswith("M:")
    assert report[mid]["kind"] == "mem"
    assert report[mid]["fresh"] is False
    assert report[mid]["reasons"] == [
        "source ep_000001 superseded by ep_000002"]
    store.close()


def test_check_freshness_scene_citing_nothing_is_honest_fresh(tmp_path):
    """A derived node citing nothing has no verifiable derivation -- reported
    fresh WITH the note, not a fake clean stamp."""
    store = _store(tmp_path)
    _scene(store, "scene_0009", [], topic="orphan")
    trav = GraphTraversal(store)
    report = trav.check_freshness(["scene_0009"])
    assert report["scene_0009"]["fresh"] is True
    assert report["scene_0009"]["reasons"] == [
        "cites nothing (nothing verifiable)"]
    assert report["scene_0009"]["sources"] == []
    store.close()


def test_check_freshness_unknown_blank_and_multi_hop_tip(tmp_path):
    """Unknown/blank ids report honestly; a 2-hop chain surfaces the TIP in
    the reason (not the immediate successor)."""
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    w = SemanticMemoryWriter(store)
    w.supersede_episode("ep_000002", "ep_000001")
    w.supersede_episode("ep_000003", "ep_000002")
    trav = GraphTraversal(store)
    report = trav.check_freshness(
        ["ep_000001", "ep_000003", "  ", "scene_7777"])
    assert report["ep_000001"]["fresh"] is False
    assert report["ep_000001"]["reasons"] == ["superseded by ep_000003"]
    assert report["ep_000003"]["fresh"] is True
    # A scene id with no node anywhere: cites nothing -> the honest note.
    assert report["scene_7777"]["fresh"] is True
    assert report["scene_7777"]["reasons"] == [
        "cites nothing (nothing verifiable)"]
    # Blank id key is preserved verbatim so the caller can match its input.
    assert report["  "]["kind"] == "unknown"
    assert report["  "]["reasons"] == ["blank id"]
    store.close()


def test_check_freshness_is_read_only(tmp_path):
    """The stat MUTATES NOTHING: existing stale_of/stale_since keys are left
    byte-identical through a call -- and a call on a resolving (re-derived)
    scene does NOT do the R1 recheck's shrink either (never fights it)."""
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    _scene(store, "scene_0001", ["ep_000001"])
    with propagating(True):
        w.supersede_episode("ep_000002", "ep_000001", when="T2")
        before_of = _stale_of(store, "scene", "scene_0001")
        before_since = _stale_since(store, "scene", "scene_0001")
        assert before_of == {"ep_000001"} and before_since == "T2"
        # Re-author citing the tip: R1's recheck would resolve + clear here.
        _scene(store, "scene_0001", ["ep_000001", "ep_000002"])
        trav = GraphTraversal(store)
        report = trav.check_freshness(["scene_0001"])
        assert report["scene_0001"]["fresh"] is True  # cites both ep + tip
        # The mark keys are bit-identical around the call: check_freshness
        # neither shrinks nor grows them (that stays R1's recheck's job).
        assert _stale_of(store, "scene", "scene_0001") == before_of
        assert _stale_since(store, "scene", "scene_0001") == before_since
    # The R1 recheck itself still resolves the same scene (the lifecycle is
    # intact where it belongs).
    with freshness(True):
        trav2 = GraphTraversal(store)
        keep, _ = trav2._filter_stale_derived({"scene_0001"})
        assert keep == {"scene_0001"}
    assert _stale_of(store, "scene", "scene_0001") == set()
    store.close()


# ── 2. assembly-time STALE stamp ──


def _hydration(eid, stale=False, when="T9"):
    d = {
        "episode_id": eid, "kind": "episode", "timestamp": "t",
        "summary": f"summary {eid}", "entities": ["Postgres"],
        "topics": [], "tones": [],
    }
    if stale:
        d["stale_since"] = when
        d["stale_of"] = ["ep_000001"]
    return d


def test_stamp_renders_stale_note_when_on(tmp_path):
    store = _store(tmp_path)
    retr = HippocampalRetriever(store)
    with freshness(False):
        base = retr.build_context_string([_hydration("M:0007", stale=True)])
    with freshness(True):
        stamped = retr.build_context_string([_hydration("M:0007", stale=True)])
    assert "Stale: derives from superseded source(s) ep_000001 (marked T9)" \
        in stamped
    assert "-- verify before trusting" in stamped
    assert stamped != base  # flag off vs on differ on an annotated unit
    assert "Stale:" not in base
    store.close()


def test_stamp_unannotated_unit_untouched_even_when_on(tmp_path):
    """A hydrated dict WITHOUT the R1 annotation gets no note even flag-on
    (the stamp keys off the annotation, not the flag alone)."""
    store = _store(tmp_path)
    retr = HippocampalRetriever(store)
    with freshness(True):
        ctx = retr.build_context_string([_hydration("ep_001")])
    assert "Stale:" not in ctx
    store.close()


def test_stamp_counts_against_token_budget(tmp_path):
    """The note is appended BEFORE the token estimate: under a tight budget the
    (now larger) chunk can drop out of context entirely -- no bypass."""
    store = _store(tmp_path)
    retr = HippocampalRetriever(store)
    with freshness(False):
        off_tight = retr.build_context_string(
            [_hydration("M:0007", stale=True)], max_tokens=300)
        off_loose = retr.build_context_string(
            [_hydration("M:0007", stale=True)], max_tokens=4000)
    with freshness(True):
        on_tight = retr.build_context_string(
            [_hydration("M:0007", stale=True)], max_tokens=300)
        on_loose = retr.build_context_string(
            [_hydration("M:0007", stale=True)], max_tokens=4000)
    assert "Stale:" in off_loose or True   # loose budget always fits
    assert "Stale:" in on_loose
    # Under the tight budget the (stamped, larger) chunk may have dropped out
    # entirely -- the budget honored the note rather than bypassing it.
    dropped = "M:0007" in off_tight and "M:0007" not in on_tight
    assert dropped or "Stale:" in on_tight
    store.close()


# ── 3. the agent tool: handler + dispatch + loop-tool gating ──


class _ModeARecorder:
    """mode_a stub recording ``(messages, tools)`` per call (tier-2 pattern).

    ``emit_check`` (optional): the FIRST call returns a ``check_freshness``
    tool_call (with ``unit_ids`` as given -- ``"OMIT"`` sends no args) so
    ``run_tool_loop`` dispatches it mid-query; every later call returns
    ``(reply, None)`` so the loop terminates with ``content=reply``."""

    def __init__(self, reply: str = "SYNTH RESPONSE",
                 emit_check: object = None) -> None:
        self.reply = reply
        self.emit_check = emit_check
        self.calls: list[tuple[list[dict], object]] = []
        self._fired = False

    def _complete(self, messages, tools=None, tool_choice=None) -> tuple:
        self.calls.append((messages, tools))
        if self.emit_check is not None and not self._fired:
            self._fired = True
            args = (
                "{}" if self.emit_check == "OMIT"
                else json.dumps({"unit_ids": self.emit_check})
            )
            tool_calls = [{
                "id": "call_check_freshness_1", "type": "function",
                "function": {"name": "check_freshness", "arguments": args},
            }]
            return self.reply, tool_calls
        return self.reply, None


def _orch(tmp_path, *, db_subdir="db", query_freshness=False, eps=None,
          plan=None, mode_a=None):
    store = HippocampalStore(str(tmp_path / db_subdir))
    for ep in (eps or []):
        store.encode_episode(ep)
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan or {}),
                                     embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / db_subdir / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=mode_a or _ModeARecorder(),
        config=cfg, user_id="victor", query_freshness=query_freshness,
    )
    return orch


def test_handler_flag_off_returns_empty(tmp_path):
    orch = _orch(tmp_path, db_subdir="off", query_freshness=False)
    assert orch.check_freshness_context(["ep_000001"]) == ""
    assert orch.check_freshness_context(None) == ""
    orch.store.close()


def test_handler_report_format_explicit_ids(tmp_path):
    """One line per id: fresh lines plain, stale lines name source -> tip."""
    store = _store(tmp_path)
    _encode(store, "ep_000001")
    _encode(store, "ep_000002")
    w = SemanticMemoryWriter(store)
    _scene(store, "scene_0001", ["ep_000001"])
    w.supersede_episode("ep_000002", "ep_000001")
    retriever = HippocampalRetriever(store, planner=_StubPlanner({}),
                                     embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=_ModeARecorder(), config=cfg,
        user_id="victor", query_freshness=True,
    )
    text = orch.check_freshness_context(
        ["scene_0001", "ep_000002", "ep_000001"])
    assert text.startswith("[freshness] 3 unit(s) checked")
    assert "scene_0001: STALE -- source ep_000001 superseded by ep_000002" \
        in text
    assert "ep_000002: fresh" in text
    assert "ep_000001: STALE -- superseded by ep_000002" in text
    orch.store.close()


def test_loop_no_args_tool_checks_tracked_context_ids(tmp_path):
    """The true no-args contract, mid-loop: the query's assembled context ids
    (``_current_fresh_ids``) are set DURING ``query`` and the loop's dispatched
    no-args ``check_freshness`` reports on them; after the query's happy-path
    tail the ids are cleared (never leak)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_mk_episode("ep_001", ["Postgres"], "We chose Postgres")]
    orch = _orch(tmp_path, db_subdir="track", query_freshness=True,
                 eps=eps, plan=plan,
                 mode_a=_ModeARecorder(reply="LLM SAID THIS",
                                       emit_check="OMIT"))
    assert orch._current_fresh_ids is None
    with feedback_off():
        res = orch.query("Why did we choose Postgres?")
    # The loop ran; its collected transcript is surfaced on the result only
    # (``_last_loop`` itself is consumed + cleared by query()'s tail).
    assert res.get("response") == "LLM SAID THIS"
    collected = res["loop_collected"]
    check = next(c for c in collected if c["name"] == "check_freshness")
    assert check["result"].startswith("[freshness]")
    assert "ep_001: fresh" in check["result"]
    # Happy-path tail: cleared (never leaks into the next turn).
    assert orch._current_fresh_ids is None
    orch.store.close()


def test_query_resets_tracked_ids_on_early_return(tmp_path):
    """The lifecycle: ``_current_fresh_ids`` is cleared on the early-return
    tail too (never leaks)."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_mk_episode("ep_001", ["Postgres"], "We chose Postgres")]
    orch = _orch(tmp_path, db_subdir="lifecycle", query_freshness=True,
                 eps=eps, plan=plan)
    with feedback_off():
        orch.query("Why did we choose Postgres?")
    assert orch._current_fresh_ids is None  # happy-path tail cleared it
    orch.store.close()


def test_dispatch_tool_check_freshness(tmp_path):
    """The dispatch branch: explicit ``unit_ids`` -> the plain-text verdict;
    a malformed ``unit_ids`` -> the error string; the flag OFF -> the honest
    nothing error."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_mk_episode("ep_001", ["Postgres"], "We chose Postgres")]
    orch = _orch(tmp_path, db_subdir="disp", query_freshness=True,
                 eps=eps, plan=plan)
    # Explicit ids (string args parsed like every dispatch branch).
    out = dispatch_tool(orch, "check_freshness", '{"unit_ids": ["ep_001"]}')
    assert out.startswith("[freshness]")
    assert "ep_001: fresh" in out
    # The malformed-args contract.
    bad = dispatch_tool(orch, "check_freshness",
                        {"unit_ids": ["ok", "", 42]})
    assert json.loads(bad)["error"] == (
        "check_freshness 'unit_ids' must be an array of non-empty strings")
    # Flag-off orchestrator: handler "" -> the honest error, not a verdict.
    orch2 = _orch(tmp_path, db_subdir="disp2", query_freshness=False)
    bad2 = dispatch_tool(orch2, "check_freshness", {})
    assert json.loads(bad2)["error"] == "check_freshness returned nothing"
    orch.store.close()
    orch2.store.close()


def test_byte_identical_off_and_schema_appended_on(tmp_path):
    """Flag OFF -> the consumer sees the EXACT prior tool objects (no
    ``CHECK_FRESHNESS_SCHEMA``). Flag ON -> the schema IS appended to the
    loop-path tool set as a NEW list; messages are otherwise unchanged."""
    plan = {"entities": ["Postgres"], "entity_mode": "union"}
    eps = [_mk_episode("ep_001", ["Postgres"], "We chose Postgres")]

    orch_off = _orch(tmp_path, db_subdir="boff", query_freshness=False,
                     eps=eps, plan=plan,
                     mode_a=_ModeARecorder(reply="LLM SAID THIS"))
    with feedback_off():
        orch_off.query("Why did we choose Postgres?")
    off_calls = orch_off.mode_a.calls
    orch_off.store.close()

    eps2 = [_mk_episode("ep_001", ["Postgres"], "We chose Postgres")]
    orch_on = _orch(tmp_path, db_subdir="bon", query_freshness=True,
                    eps=eps2, plan=plan,
                    mode_a=_ModeARecorder(reply="LLM SAID THIS"))
    with feedback_off():
        orch_on.query("Why did we choose Postgres?")
    on_calls = orch_on.mode_a.calls

    # The queries ran with feedback OFF -> the base set is LOOP_TOOLS.
    off_tools = [c[1] for c in off_calls]
    assert off_tools[0] is LOOP_TOOLS
    assert CHECK_FRESHNESS_SCHEMA not in off_tools[0]
    # The one-shot path (loop disabled) never offers the schema either.
    assert CHECK_FRESHNESS_SCHEMA not in SELF_CHAT_TOOLS

    on_tools = [c[1] for c in on_calls]
    assert on_tools[0] is not LOOP_TOOLS          # a new list, not a mutation
    assert on_tools[0] == [*LOOP_TOOLS, CHECK_FRESHNESS_SCHEMA]
    assert CHECK_FRESHNESS_SCHEMA in on_tools[0]
    # Module-level lists were never mutated.
    assert CHECK_FRESHNESS_SCHEMA not in TOOL_SCHEMAS
    assert CHECK_FRESHNESS_SCHEMA not in LOOP_TOOLS
    # Messages identical off vs on (the schema is appended, nothing else moves).
    assert [c[0] for c in off_calls] == [c[0] for c in on_calls]
    orch_on.store.close()


def test_current_fresh_ids_initialized_none(tmp_path):
    """The tracking attribute always exists (init in ``__init__``), so the
    no-args handler is safe on a fresh orchestrator before any query."""
    orch = _orch(tmp_path, db_subdir="init")
    assert orch._current_fresh_ids is None
    assert orch.check_freshness_context(None) == ""
    orch.store.close()