"""STRM Phase 4 Step 5: state-conditioned retrieval + pin-tagged re-injection.

When the salience trigger (Step 4) finds a salient anchor, the orchestrator
fires ``retrieve_by_embedding`` with the anchor's 384-d doc vector as the
state-conditioned query (the episode the WM state flagged as being-forgotten),
dedups by ``episode_id``, merges the fired episodes into the prompt-driven set
(salience first), and re-injects them with ``pin=True`` so ``W_A`` retains the
proactive recall over the next K steps. Flag-off / disarmed / failed -> no
merge, every inject ``pin=False``, no ``salience_retrieval_count`` key ->
byte-identical to pre-Step-5.

Two layers tested:

1. ``HippocampalRetriever.retrieve_by_embedding`` -- vector search with a
   pre-computed query embedding (no text re-embed). ``[]`` when no vector index
   is configured (the stub-harness path: no ``auto_load_index``). The
   ``search_by_vector`` backend methods skip the embed step.
2. The orchestrator merge + pin + per-turn count, with ``retrieve_by_embedding``
   monkeypatched so the test controls the fired episodes deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Phase2cConfig
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.latent_dynamics_head import load_latent_dynamics_head
from src.subconscious.recoverability_head import load_recoverability_head
from src.subconscious.relevance_head import load_relevance_head
from src.subconscious.salience import SALIENCE_RETRIEVAL_BUDGET, SalienceThresholds

from tests.test_orchestrator import _StubEmbedder, _StubModeA, _StubPlanner, _ep

_REC_CKPT = Path("data/training/strm_recoverability/best.pt")
_LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")
_REL_CKPT = Path("data/training/strm_relevance/best.pt")
_HEADS_PRESENT = _REC_CKPT.exists() and _LD_CKPT.exists() and _REL_CKPT.exists()


def _thresh(theta=1e18, phi=-1e18, surprise_cap=1e18):
    """Permissive thresholds so every SCORED anchor is salient (the AND passes
    for any non-None scores). Lets the tests exercise the retrieval/merge/pin
    wiring without controlling the head outputs."""
    return SalienceThresholds(
        theta=theta, phi=phi, surprise_cap=surprise_cap,
        theta_percentile=30.0, phi_percentile=70.0, surprise_cap_percentile=80.0,
        basis="test", n_recoverability=0, n_relevance=0, n_latent_dynamics=0,
    )


def _build(tmp_path, *, strm_salience=False, salience_thresholds=None,
           recoverability_head=None, latent_dynamics_head=None,
           relevance_head=None, ring_capacity=16, reply="SYNTH RESPONSE"):
    """Build an orchestrator on the stub harness with the salience trigger wired."""
    store = HippocampalStore(str(tmp_path / "db"))
    ep = _ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")
    store.encode_episode(ep)
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                     embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=_StubModeA(reply=reply),
        config=cfg, user_id="victor", ring_capacity=ring_capacity,
        recoverability_head=recoverability_head, latent_dynamics_head=latent_dynamics_head,
        relevance_head=relevance_head,
        strm_salience=strm_salience, salience_thresholds=salience_thresholds,
    )
    return orch, store


def _preinject(orch, slots_text):
    """Pre-inject (source_id, text) episodes into the WM ring so the salience
    hook (which fires BEFORE the current turn's retrieve+inject) has text slots
    to score. Simulates prior turns' recalled episodes sitting in the ring.
    Returns the list of source_ids injected."""
    sids = []
    for sid, txt in slots_text:
        emb = orch.working_memory.embed([txt])[0]
        orch.working_memory.inject(emb, source_id=sid, text=txt)
        sids.append(sid)
    return sids


# ── retrieve_by_embedding + search_by_vector (the retriever layer) ──

def test_retrieve_by_embedding_no_index_is_empty(tmp_path):
    """No vector index configured (the stub harness: no auto_load_index) ->
    retrieve_by_embedding is a no-op ([]). This is the byte-identical fallback
    when no vector layer is loaded."""
    store = HippocampalStore(str(tmp_path / "db"))
    retriever = HippocampalRetriever(store, planner=_StubPlanner({}),
                                     embedder=_StubEmbedder())
    try:
        assert retriever.vector_search is None
        out = retriever.retrieve_by_embedding([0.1] * 384)
        assert out == []
    finally:
        store.close()


def test_retrieve_by_embedding_accepts_tensor_and_list(tmp_path, monkeypatch):
    """retrieve_by_embedding accepts a tensor OR a list[float] query and
    forwards the flat 384-d list to search_by_vector."""
    store = HippocampalStore(str(tmp_path / "db"))
    retriever = HippocampalRetriever(store, planner=_StubPlanner({}),
                                     embedder=_StubEmbedder())
    # Attach a fake vector_search whose search_by_vector records the query.
    class _FakeVS:
        def search_by_vector(self, vec, k=5):
            self.seen = list(vec)
            self.k = k
            return [("ep_x", 0.9)]
        def search(self, query, k=5):  # pragma: no cover - not used here
            return []
    fake = _FakeVS()
    retriever.vector_search = fake
    try:
        import torch
        # tensor query [1,384]
        out = retriever.retrieve_by_embedding(torch.zeros(1, 384))
        assert [e["episode_id"] for e in out] == ["ep_x"]
        assert len(fake.seen) == 384 and all(x == 0.0 for x in fake.seen)
        # list query [384]
        out2 = retriever.retrieve_by_embedding([0.5] * 384, limit=3)
        assert fake.k == 3
        assert out2[0]["score"] == pytest.approx(0.45)  # 0.9 * 0.5 discount
    finally:
        store.close()


# ── orchestrator merge + pin + count ──

@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_salience_flag_off_no_count_key_and_byte_identical(tmp_path):
    """Flag off -> salience_retrieval_count key is ABSENT (byte-identical result
    dict to pre-Step-5) and no retrieval is fired. The merge is a no-op
    (_salience_fired_episodes is None)."""
    orch, store = _build(tmp_path, strm_salience=False,
                        salience_thresholds=_thresh(), ring_capacity=16)
    orch_plain, store_plain = _build(tmp_path, strm_salience=False, ring_capacity=16)
    fired_calls = []
    orch.retriever.retrieve_by_embedding = lambda *a, **k: fired_calls.append(1) or []
    try:
        res = orch.query("What did Alice say?")
        assert "salience_retrieval_count" not in res
        assert fired_calls == []  # the hook never ran -> no retrieval fired
        res_plain = orch_plain.query("What did Alice say?")
        # deterministic subset byte-identical (excludes response/WM-state)
        for k in ("presentation_plan", "chunked", "supported"):
            assert res[k] == res_plain[k]
        assert res["retrieved_episodes"] == res_plain["retrieved_episodes"]
    finally:
        store.close()
        store_plain.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_salience_fires_retrieval_merges_and_pins(tmp_path, monkeypatch):
    """Armed + a salient anchor (pre-injected text slot) -> the hook fires
    retrieve_by_embedding, the fired episode is merged FIRST into
    retrieved_episodes, re-injected with pin=True (its ring slot carries
    pinned=True), and salience_retrieval_count is surfaced."""
    rec = load_recoverability_head(str(_REC_CKPT), device="cpu")
    ld = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    rel = load_relevance_head(str(_REL_CKPT), device="cpu")
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    # Pre-inject a text slot the hook will score as salient (permissive
    # thresholds -> any scored anchor salient).
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])

    fired_ep = {
        "episode_id": "ep_recall_1", "summary": "Alice chose Postgres",
        "entities": ["Alice"], "topics": ["Postgres"], "tones": [],
        "timestamp": "2026-07-01T10:00:00", "score": 0.42,
    }
    call_count = [0]

    def _fake_rbe(query_emb, signal="routine", limit=None):
        call_count[0] += 1
        return [dict(fired_ep)]
    orch.retriever.retrieve_by_embedding = _fake_rbe
    try:
        res = orch.query("What did Alice say?")
        assert call_count[0] == 1                 # one salient anchor -> one retrieval
        assert res["salience_retrieval_count"] == 1
        ids = [e["episode_id"] for e in res["retrieved_episodes"]]
        # the fired episode is merged in (salience first)
        assert "ep_recall_1" in ids
        assert ids[0] == "ep_recall_1"
        # its WM ring slot is pinned (pin-tagged re-inject)
        pinned = [s for s in orch.working_memory.ring_buffer()
                  if s.source_id == "ep_recall_1"]
        assert pinned and pinned[0].pinned is True
        # a prompt-driven slot is NOT pinned
        plain = [s for s in orch.working_memory.ring_buffer()
                 if s.source_id == "ep_001"]
        if plain:
            assert plain[0].pinned is False
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_salience_budget_cap(tmp_path, monkeypatch):
    """More salient anchors than SALIENCE_RETRIEVAL_BUDGET -> at most BUDGET
    retrieve_by_embedding calls (the proactive-recall budget cap)."""
    rec = load_recoverability_head(str(_REC_CKPT), device="cpu")
    ld = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    rel = load_relevance_head(str(_REL_CKPT), device="cpu")
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=32)
    # pre-inject MORE than BUDGET text slots -> all salient under permissive thresholds
    slots = [(f"ep_old_{i}", f"forgotten fact number {i}") for i in range(SALIENCE_RETRIEVAL_BUDGET + 3)]
    _preinject(orch, slots)
    call_count = [0]

    def _fake_rbe(query_emb, signal="routine", limit=None):
        call_count[0] += 1
        return [{"episode_id": f"recall_{call_count[0]}", "summary": "s",
                 "entities": [], "topics": [], "tones": [],
                 "timestamp": "2026-07-01T10:00:00", "score": 0.3}]
    orch.retriever.retrieve_by_embedding = _fake_rbe
    try:
        res = orch.query("What did Alice say?")
        assert call_count[0] == SALIENCE_RETRIEVAL_BUDGET
        assert res["salience_retrieval_count"] == SALIENCE_RETRIEVAL_BUDGET
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_salience_dedup_by_episode_id(tmp_path, monkeypatch):
    """Salience-fired episodes dedup by episode_id (no duplicate injects), and a
    fired episode that also appears in the prompt-driven set is kept in its
    salience position (the prompt-driven duplicate is dropped)."""
    rec = load_recoverability_head(str(_REC_CKPT), device="cpu")
    ld = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    rel = load_relevance_head(str(_REL_CKPT), device="cpu")
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])
    # fired retrieval returns ep_001 (which the prompt-driven retrieve ALSO
    # returns, since the stub plan matches Alice) + a unique one.
    def _fake_rbe(query_emb, signal="routine", limit=None):
        return [
            {"episode_id": "ep_001", "summary": "Alice said use Postgres",
             "entities": ["Alice"], "topics": [], "tones": [],
             "timestamp": "2026-07-03T10:00:00", "score": 0.5},
            {"episode_id": "ep_recall_unique", "summary": "unique recall",
             "entities": [], "topics": [], "tones": [],
             "timestamp": "2026-07-01T10:00:00", "score": 0.4},
        ]
    orch.retriever.retrieve_by_embedding = _fake_rbe
    try:
        res = orch.query("What did Alice say?")
        ids = [e["episode_id"] for e in res["retrieved_episodes"]]
        # no duplicates
        assert len(ids) == len(set(ids))
        # ep_001 appears once (deduped), salience-fired eps first
        assert ids.count("ep_001") == 1
        assert ids[0] in ("ep_001", "ep_recall_unique")
        assert "ep_recall_unique" in ids
    finally:
        store.close()


@pytest.mark.skipif(not _HEADS_PRESENT, reason="STRM head checkpoints not trained")
def test_salience_retrieval_failure_is_noop(tmp_path, monkeypatch):
    """retrieve_by_embedding raises -> the per-anchor try/except swallows it:
    no merge, count=0, byte-identical retrieved_episodes to flag-off. A
    proactive-recall heuristic must never crash the turn."""
    rec = load_recoverability_head(str(_REC_CKPT), device="cpu")
    ld = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    rel = load_relevance_head(str(_REL_CKPT), device="cpu")
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=_thresh(),
                        recoverability_head=rec, latent_dynamics_head=ld,
                        relevance_head=rel, ring_capacity=16)
    _preinject(orch, [("ep_old", "Alice chose Postgres for the audit log")])

    def _boom(query_emb, signal="routine", limit=None):
        raise RuntimeError("simulated vector search failure")
    orch.retriever.retrieve_by_embedding = _boom
    orch_plain, store_plain = _build(tmp_path, ring_capacity=16)
    try:
        res = orch.query("What did Alice say?")
        assert res["salience_retrieval_count"] == 0
        # no merge -> byte-identical retrieved_episodes to a plain flag-off orchestrator
        res_plain = orch_plain.query("What did Alice say?")
        assert res["retrieved_episodes"] == res_plain["retrieved_episodes"]
    finally:
        store.close()
        store_plain.close()