"""Live gate for the Tier-2 recall menu against the local Bonsai server.

The offline suite (``test_tier2_recall_menu.py``) proves the wiring with stub
mode_a + a stub vector_search. This is the end-to-end check against the REAL
8B Ternary-Bonsai at ``config.bonsai_endpoint`` (``localhost:8080/v1``) + the
REAL in-DB WaveDB vector index: (1) the ``remember`` tool is genuinely OFFERED
to the live LLM in the loop-path tool set (the byte-identical-off contract's
on-path counterpart), the loop runs, and any tool the model calls is known;
(2) the load-bearing user-scope gate holds end-to-end -- a real vector search
over a two-user corpus returns BOTH users' episodes as candidates, but
``remember_menu``'s tail filter excludes the cross-user (bob's) hits so only
the query user's (alice's) content reaches the menu.

Skipped automatically when Bonsai is unreachable (``GET /v1/models`` probe --
mirrors ``test_self_chat_live.py``) or when ``wavedb.VectorLayer`` is absent
(mirrors ``test_retrieval_user_scope.py``). Run with the local Bonsai server
up (see memory ``bonsai-server-local-startup``): ``$env:BONSAI_MODEL="8B";
& ../Bonsai-demo/scripts/start_llama_server.ps1``.

The model is a small Q2 8B and is nudged to answer directly when the context
suffices, so a ``remember`` call is NOT guaranteed -- the loop test asserts the
PATH executed + ``REMEMBER_SCHEMA`` was offered + any tool called is known, and
IF the model did call ``remember`` the surfaced menu is alice-only (the scope
gate, live). The scope test (2) is LLM-independent and is the strong assertion.
"""

from __future__ import annotations

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)", allow_module_level=True)

import requests

from src.config import Phase2cConfig, config
from src.generation.mode_a import ModeAGenerator
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.tools import REMEMBER_SCHEMA


# ── stubs (deterministic; mirror test_self_chat_live / test_fade_serve) ──

class _StubEmbedder:
    """Deterministic 384-d embedder (SHA256 stretch -> normalized). The SAME
    embedder is used for the episode ``summary_embedding`` writes AND the
    ``WavedbVectorStore.search`` query re-embed, so query and stored vectors
    live in one consistent space (the stub is not semantic, but the scope gate
    only needs the candidates to surface, not semantically-correct ranking)."""
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts):
        import hashlib
        out = []
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
    """Fixed plan over the shared ``Storage`` entity so the graph path retrieves
    alice's episodes (the retriever's user-scope intersects the candidate set
    to alice's only)."""
    def __init__(self, plan: dict) -> None:
        self._plan = plan

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return self._plan


class _RecordingModeA(ModeAGenerator):
    """``ModeAGenerator`` that records the ``tools`` list passed to each
    ``_complete`` call, then delegates to the real Bonsai HTTP call. Lets the
    live test assert ``REMEMBER_SCHEMA`` was actually offered to the LLM in the
    loop-path tool set (the on-path counterpart of the byte-identical-off
    contract)."""
    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.tool_sets: list = []   # list of tools-lists (one per _complete call)

    def _complete(self, messages, tools=None, tool_choice=None):
        self.tool_sets.append(tools)
        return super()._complete(messages, tools, tool_choice)


# ── corpus ──

def _owned_ep(eid, user, sess, summary, emb):
    return Episode(
        id=eid, timestamp="2026-07-03T10:00:00",
        summary=summary, full_text=f"User: {summary}\nAssistant: {summary}",
        entities=["Storage"], topics=["storage"], tones=[], decisions=[],
        user_id=user, session_id=sess, summary_embedding=emb,
    )


def _two_user_corpus(store):
    """Alice owns 8 storage episodes, bob owns 4 -- ALL with the SAME summary
    text so they embed to near-identical vectors and interleave in the vector
    ranking (bob's are genuine candidates, not ranked out). The scope filter
    must drop bob's; without it bob would leak into alice's ``remember`` menu."""
    emb = _StubEmbedder()
    txt = "Alice and Bob architected the storage subsystem together"
    vec = emb.encode([txt])[0]
    alice = [_owned_ep(f"ep_a{i}", "alice", "S:a1", txt, vec) for i in range(8)]
    bob = [_owned_ep(f"ep_b{i}", "bob", "S:b1", txt, vec) for i in range(4)]
    for ep in alice + bob:
        store.encode_episode(ep)
    return alice, bob


@pytest.fixture(scope="module")
def orch_live(tmp_path_factory):
    url = config.bonsai_endpoint.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable: {e}")

    tmp_path = tmp_path_factory.mktemp("tier2live")
    store = HippocampalStore(str(tmp_path / "db"),
                            config={"vector_index_enabled": True, "embedding_dim": 384})
    alice, bob = _two_user_corpus(store)
    plan = {"entities": ["Storage"], "entity_mode": "union", "limit": 20}
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                     embedder=_StubEmbedder(),
                                     user_id="alice", auto_load_index=True)
    # Force the stub embedder onto the real in-DB vector backend so
    # ``search(query)`` re-embeds the query with the SAME embedder that produced
    # the episode vectors (else it would fall back to a real sentence-transformers
    # model -- heavy + a different vector space than the indexed stub vectors).
    if retriever.vector_search is not None:
        retriever.vector_search.embedder = _StubEmbedder()

    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    mode_a = _RecordingModeA(retriever)
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=JGSBackbone(BackboneConfig()),
        embedder=_StubEmbedder(), mode_a=mode_a, config=cfg, user_id="alice",
        tier2_recall_menu=True,
    )
    yield orch, set(e.id for e in alice), set(e.id for e in bob)
    store.close()


# ── 1. the load-bearing live user-scope gate (LLM-independent) ──

def test_tier2_live_remember_menu_tail_is_scoped(orch_live):
    """Real in-DB vector index + two-user corpus: the raw over-fetched search
    returns BOTH alice's and bob's episode ids (bob IS a candidate), but
    ``remember_menu``'s tail filter excludes bob's so the menu is alice-only.
    This is the end-to-end (real vector layer, real scope filter) counterpart
    of the offline ``test_remember_menu_tail_user_scoped`` (which used a stub
    vector_search)."""
    orch, alice_ids, bob_ids = orch_live
    orch._current_query = "tell me about the storage subsystem"
    # Sanity: bob's episodes are genuine vector candidates (in the over-fetch).
    from src.orchestrator import _REMEMBER_TAIL_N
    from src.config import config as _cfg
    tier1_k = _cfg.default_retrieval_limit
    raw = orch.retriever.vector_search.search(
        orch._current_query, k=(tier1_k + _REMEMBER_TAIL_N) * 3)
    raw_ids = {eid for eid, _ in raw}
    assert bob_ids <= raw_ids, "bob should be a vector candidate (proves the filter, not ranking, excludes him)"
    # The menu: only alice's tail survives the scope filter.
    text = orch.remember_menu()
    items = orch._last_remember_menu or []
    tail = [i for i in items if i["source"] == "wavedb_tail"]
    assert text and tail, "tail should surface alice's episodes beyond the tier-1 cutoff"
    tail_ids = {i["episode_id"] for i in tail}
    assert tail_ids <= alice_ids, f"cross-user leak: {tail_ids - alice_ids}"
    assert not (tail_ids & bob_ids), f"bob leaked into alice's menu: {tail_ids & bob_ids}"


# ── 2. the live LLM loop: remember is offered + scope holds if called ──

def test_tier2_live_loop_offers_remember_and_stays_scoped(orch_live):
    """Real Bonsai loop with ``tier2_recall_menu=True``: the loop-path tool set
    handed to the live LLM includes ``REMEMBER_SCHEMA`` (the on-path counterpart
    of the byte-identical-off contract), the loop runs, any tool the model calls
    is a known retrieval/feedback/remember tool, and IF the model called
    ``remember`` the surfaced ``result["remember_menu"]`` is alice-only (the
    scope gate, live, end-to-end through the real dispatch). A ``remember`` call
    is NOT guaranteed from an 8B Q2 model -- the offered-tools assertion is the
    load-bearing one; the scoped-menu assertion is conditional on the call."""
    orch, alice_ids, bob_ids = orch_live
    res = orch.query("What did we decide about the storage subsystem?",
                     auto_persist=False)
    assert res["end_state_plan"].end_state == "synthesize"
    # The loop path ran.
    assert "loop_tool_messages" in res
    assert isinstance(res.get("response"), str) and res["response"].strip()
    # REMEMBER_SCHEMA was offered to the live LLM in (at least) one loop call.
    offered = [t for t in orch.mode_a.tool_sets if t]
    assert offered, "the loop path should have passed tools to _complete"
    assert any(REMEMBER_SCHEMA in t for t in offered), \
        "REMEMBER_SCHEMA was not offered to the live LLM (tier2 on -> loop-path append)"
    # Any tool the loop dispatched is a known tool (never an unknown/leak).
    names = [c.get("name") for c in res["loop_collected"]]
    assert all(n in ("expand", "search_memory", "record_feedback", "remember") for n in names)
    # IF the model called remember, the surfaced menu is alice-only.
    if "remember" in names:
        menu = res.get("remember_menu") or []
        assert menu, "remember was called but no menu was surfaced"
        tail_ids = {i["episode_id"] for i in menu if i["source"] == "wavedb_tail"}
        assert tail_ids <= alice_ids, f"cross-user leak in live menu: {tail_ids - alice_ids}"
        assert not (tail_ids & bob_ids), f"bob leaked into live menu: {tail_ids & bob_ids}"
    # The response never leaks raw tool-call JSON.
    assert "tool_calls" not in (res["response"] or "")