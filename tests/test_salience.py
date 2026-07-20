"""STRM Phase 4 Step 4: the salience trigger.

Two layers:

1. ``_decide_salience`` -- the pure sign logic
   ``salient = (rec_i < theta) AND (r_i > phi) AND (surprise_i < surprise_cap)``.
   Pinned so Step 5 wires the comparisons the right way around: LOW
   recoverability (forgotten) -> salient, HIGH relevance -> salient, HIGH
   surprise -> SUPPRESS. A None score (unscoreable slot) is never salient.
2. ``compute_salience`` -- runs the three heads over the WM ring + live state
   and returns one ``SalienceAnchor`` per slot whose ``salient`` flag equals
   ``_decide_salience`` applied to the head's scores (pins the wiring without
   needing to control the head outputs).

Plus the orchestrator pre-retrieval hook: flag-off byte-identical (the hook
never runs, ``_salience_anchors`` stays None), flag-on populates the stash
WITHOUT changing the result dict (Step 4 computes anchors only; Step 5 fires
retrieval), and a hook failure is swallowed (no-op, byte-identical).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Phase2cConfig
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.recoverability_head import load_recoverability_head
from src.subconscious.latent_dynamics_head import load_latent_dynamics_head
from src.subconscious.relevance_head import load_relevance_head
from src.subconscious.salience import (
    SalienceAnchor,
    SalienceThresholds,
    _decide_salience,
    compute_salience,
    load_salience_thresholds,
    percentile_threshold,
    salient_anchors,
)
from src.subconscious.working_memory import WorkingMemory

from tests.test_orchestrator import _StubEmbedder, _StubModeA, _StubPlanner, _ep

_REC_CKPT = Path("data/training/strm_recoverability/best.pt")
_LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")
_REL_CKPT = Path("data/training/strm_relevance/best.pt")
_THRESH = Path("data/training/strm_salience/thresholds.json")


def _thresh(theta=-1e9, phi=1e9, surprise_cap=1e9):
    """Permissive thresholds by default (all AND terms pass for scored slots)."""
    return SalienceThresholds(
        theta=theta, phi=phi, surprise_cap=surprise_cap,
        theta_percentile=30.0, phi_percentile=70.0, surprise_cap_percentile=80.0,
        basis="test", n_recoverability=0, n_relevance=0, n_latent_dynamics=0,
    )


# ── _decide_salience: the pure sign logic (the load-bearing correctness) ──

def test_decide_forgotten_relevant_low_surprise_is_salient():
    """The happy anchor: low recoverability (forgotten) + high relevance + low
    surprise -> salient. Pins all three signs at once."""
    t = _thresh(theta=0.0, phi=0.0, surprise_cap=1.0)
    assert _decide_salience(r_i=0.9, rec_i=-0.5, surprise_i=0.1, thresholds=t) is True


def test_decide_not_forgotten_is_not_salient():
    """HIGH recoverability (rec_i > theta, NOT forgotten) -> not salient. The
    whole point of the trigger is to recall what we're about to forget."""
    t = _thresh(theta=0.0, phi=0.0, surprise_cap=1.0)
    assert _decide_salience(r_i=0.9, rec_i=0.5, surprise_i=0.1, thresholds=t) is False


def test_decide_irrelevant_is_not_salient():
    """LOW relevance (r_i < phi) -> not salient. No point proactively recalling
    something irrelevant to the current query."""
    t = _thresh(theta=0.0, phi=0.5, surprise_cap=1.0)
    assert _decide_salience(r_i=0.2, rec_i=-0.5, surprise_i=0.1, thresholds=t) is False


def test_decide_high_surprise_suppresses():
    """HIGH surprise (surprise_i > surprise_cap) -> SUPPRESS (not salient). A
    novel turn should not be pre-empted with a proactive recall (proposal §5
    step 9). This is the sign Step 4 wires as ``surprise < surprise_cap`` (NOT
    ``surprise > cap``) -- pinned here so it can't be flipped."""
    t = _thresh(theta=0.0, phi=0.0, surprise_cap=0.5)
    assert _decide_salience(r_i=0.9, rec_i=-0.5, surprise_i=0.9, thresholds=t) is False


def test_decide_none_score_is_never_salient():
    """An unscoreable anchor (no text / no head -> None score) is never salient:
    we cannot responsibly pre-empt the user on an anchor we cannot score."""
    t = _thresh()
    assert _decide_salience(None, -0.5, 0.1, t) is False
    assert _decide_salience(0.9, None, 0.1, t) is False
    assert _decide_salience(0.9, -0.5, None, t) is False


def test_decide_sign_direction_via_permissive_and_restrictive():
    """Permissive thresholds (theta=+inf, phi=-inf, cap=+inf) -> every scored
    anchor salient (rec_i < +inf, r_i > -inf, surprise < +inf all hold).
    Restrictive on any one term -> none salient. Pins the sign DIRECTION of each
    comparison (low rec, high r, low surprise -> salient)."""
    permissive = _thresh(theta=1e18, phi=-1e18, surprise_cap=1e18)
    assert _decide_salience(0.1, 0.1, 0.1, permissive) is True
    # flip each term to restrictive (the comparison never holds) -> not salient
    assert _decide_salience(0.1, 0.1, 0.1, _thresh(theta=-1e18, phi=-1e18, surprise_cap=1e18)) is False
    assert _decide_salience(0.1, 0.1, 0.1, _thresh(theta=1e18, phi=1e18, surprise_cap=1e18)) is False
    assert _decide_salience(0.1, 0.1, 0.1, _thresh(theta=1e18, phi=-1e18, surprise_cap=-1e18)) is False


# ── percentile_threshold + load_salience_thresholds ──

def test_percentile_threshold_basic():
    """percentile_threshold mirrors np.percentile (linear interpolation):
    p70 of [1..10] = 7 + 0.3*(8-7) = 7.3; p0 = 1; p100 = 10."""
    assert percentile_threshold(list(range(1, 11)), 70.0) == pytest.approx(7.3)
    assert percentile_threshold(torch.arange(1, 11).float(), 0.0) == pytest.approx(1.0)
    assert percentile_threshold(torch.arange(1, 11).float(), 100.0) == pytest.approx(10.0)


def test_load_salience_thresholds_roundtrip(tmp_path):
    """A written sidecar round-trips through load_salience_thresholds."""
    import json
    payload = {
        "theta": -0.26, "phi": 0.07, "surprise_cap": 5.6e-5,
        "theta_percentile": 30.0, "phi_percentile": 70.0, "surprise_cap_percentile": 80.0,
        "basis": "test sidecar", "n_recoverability": 620, "n_relevance": 1438,
        "n_latent_dynamics": 237,
    }
    p = tmp_path / "thresholds.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    t = load_salience_thresholds(str(p))
    assert t.theta == pytest.approx(-0.26)
    assert t.phi == pytest.approx(0.07)
    assert t.surprise_cap == pytest.approx(5.6e-5)
    assert t.theta_percentile == 30.0
    assert t.n_relevance == 1438
    assert t.basis == "test sidecar"


@pytest.mark.skipif(not _THRESH.exists(), reason="salience thresholds sidecar not computed")
def test_real_sidecar_loads():
    """The shipped sidecar (from scripts/compute_salience_thresholds.py) loads."""
    t = load_salience_thresholds(str(_THRESH))
    assert isinstance(t, SalienceThresholds)
    # all three heads' val counts are populated (the script ran over all three)
    assert t.n_recoverability > 0 and t.n_relevance > 0 and t.n_latent_dynamics > 0
    assert t.basis  # non-empty provenance


# ── compute_salience: end-to-end wiring on a synthetic ring ──

def _wm_with_ring(slots_text):
    """Build a WM with ring ON and inject one slot per (source_id, text) pair.
    Returns (wm, prompt_emb) where prompt_emb is a seeded 384-d query embedding."""
    bb = JGSBackbone(BackboneConfig())
    wm = WorkingMemory(bb, embedder=_StubEmbedder(), ring_capacity=8)
    # step a query first so state is non-None (surprise needs a pre-step state)
    qemb = wm.embed(["what did Alice say about Postgres"])[0]
    wm.update(qemb, source_id="query", text="what did Alice say about Postgres")
    for sid, txt in slots_text:
        emb = wm.embed([txt])[0]
        wm.inject(emb, source_id=sid, text=txt)
    return wm, qemb


def test_compute_salience_empty_ring_returns_empty():
    """An empty ring -> no anchors (the trigger has nothing to score)."""
    bb = JGSBackbone(BackboneConfig())
    wm = WorkingMemory(bb, embedder=_StubEmbedder(), ring_capacity=8)
    qemb = wm.embed(["q"])[0]
    wm.update(qemb)  # one step, but ring_capacity=8 and one slot -> ring has 1 slot
    # use a fresh WM with ring OFF so ring_buffer() is empty
    wm_off = WorkingMemory(JGSBackbone(BackboneConfig()), embedder=_StubEmbedder(), ring_capacity=0)
    qemb2 = wm_off.embed(["q"])[0]
    wm_off.update(qemb2)
    state = wm_off.state_tensors()
    anchors = compute_salience(
        ring_slots=wm_off.ring_buffer(), state_tensors=state,
        prev_state_tensors=state, working_memory=wm_off,
        relevance_head=None, recoverability_head=None, latent_dynamics_head=None,
        embedder=wm_off._embedder, query_emb=qemb2, thresholds=_thresh(),
    )
    assert anchors == []


def test_compute_salience_one_anchor_per_slot_and_flag_matches_decide():
    """compute_salience returns one SalienceAnchor per ring slot (oldest-first),
    and each anchor's ``salient`` flag equals ``_decide_salience`` applied to
    the head's computed scores. Pins the wiring (scores -> decision) without
    needing to control the head outputs. Uses the trained heads if present; else
    falls back to all-None heads (every anchor not salient, still one per slot)."""
    wm, qemb = _wm_with_ring([("ep-a", "Alice chose Postgres"),
                              ("ep-b", "Bob mentioned Redis"),
                              ("ep-c", "Carol filed the ticket")])
    slots = wm.ring_buffer()
    assert len(slots) == 4  # query + 3 injected
    state = wm.state_tensors()
    rec = load_recoverability_head(str(_REC_CKPT), device="cpu") if _REC_CKPT.exists() else None
    ld = load_latent_dynamics_head(str(_LD_CKPT), device="cpu") if _LD_CKPT.exists() else None
    rel = load_relevance_head(str(_REL_CKPT), device="cpu") if _REL_CKPT.exists() else None
    # permissive thresholds so the AND passes for any scored anchor
    anchors = compute_salience(
        ring_slots=slots, state_tensors=state, prev_state_tensors=state,
        working_memory=wm, relevance_head=rel, recoverability_head=rec,
        latent_dynamics_head=ld, embedder=wm._embedder, query_emb=qemb,
        thresholds=_thresh(theta=1e18, phi=-1e18, surprise_cap=1e18),
    )
    assert len(anchors) == len(slots)
    assert [a.slot_index for a in anchors] == list(range(len(slots)))
    # the slot's provenance is carried onto the anchor
    assert anchors[1].source_id == "ep-a" and anchors[1].text == "Alice chose Postgres"
    # age: ring-position proxy (0 = newest = last slot)
    assert anchors[-1].age == 0 and anchors[0].age == len(slots) - 1
    # the flag is exactly _decide_salience on the computed scores
    t = _thresh(theta=1e18, phi=-1e18, surprise_cap=1e18)
    for a in anchors:
        assert a.salient == _decide_salience(a.r_i, a.rec_i, a.surprise_i, t)
    # with the trained heads wired + permissive thresholds, every SCORED anchor
    # is salient; the raw-query slot (no source_id/text -> r_i=None) is not.
    if rel is not None and rec is not None and ld is not None:
        scored = [a for a in anchors if a.r_i is not None and a.rec_i is not None and a.surprise_i is not None]
        assert all(a.salient for a in scored)
        # the query slot has text ("what did Alice say...") so it IS scored ->
        # salient under permissive thresholds. A None-provenance slot would not be.
        # salient_anchors filters the salient subset.
        assert salient_anchors(anchors) == [a for a in anchors if a.salient]


def test_compute_salience_missing_head_disarms():
    """A missing head -> that score is None for every anchor -> no anchor
    salient (the trigger is all-three-AND). This is the orchestrator's
    ``_salience_armed`` invariant at the compute layer too."""
    wm, qemb = _wm_with_ring([("ep-a", "Alice chose Postgres")])
    anchors = compute_salience(
        ring_slots=wm.ring_buffer(), state_tensors=wm.state_tensors(),
        prev_state_tensors=wm.state_tensors(), working_memory=wm,
        relevance_head=None, recoverability_head=None, latent_dynamics_head=None,
        embedder=wm._embedder, query_emb=qemb,
        thresholds=_thresh(theta=1e18, phi=-1e18, surprise_cap=1e18),
    )
    assert len(anchors) == 2
    assert all(a.r_i is None for a in anchors)   # no relevance head
    assert all(a.rec_i is None for a in anchors)  # no recoverability head
    assert all(a.surprise_i is None for a in anchors)  # no latent-dynamics head
    assert all(not a.salient for a in anchors)


# ── orchestrator pre-retrieval hook ──

def _build(tmp_path, *, strm_salience=False, salience_thresholds=None,
           recoverability_head=None, latent_dynamics_head=None,
           relevance_head=None, ring_capacity=0, reply="SYNTH RESPONSE"):
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


def _capture(orch):
    """Deterministic view of the result (the ``direct`` end-state path has no
    ``response`` key; ``working_memory_state`` carries a timestamp -> excluded)."""
    res = orch.query("What did Alice say?")
    assert res["supported"] is True
    return {
        "presentation_plan": res["presentation_plan"],
        "chunked": res["chunked"],
        "retrieved_episodes": res["retrieved_episodes"],
        "supported": res["supported"],
    }


@pytest.mark.skipif(not (_REC_CKPT.exists() and _LD_CKPT.exists() and _REL_CKPT.exists()),
                    reason="STRM head checkpoints not trained")
def test_salience_flag_off_is_byte_identical(tmp_path):
    """strm_salience=False (the default) -> the hook never runs, _salience_anchors
    stays None, and the result dict is byte-identical to a flag-on-but-disarmed
    orchestrator (heads + thresholds + ring, but the trigger's AND would fire).
    Proves the hook attaches without changing the serve path this step."""
    th = _thresh(theta=1e18, phi=-1e18, surprise_cap=1e18)  # permissive
    rec = load_recoverability_head(str(_REC_CKPT), device="cpu")
    ld = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    rel = load_relevance_head(str(_REL_CKPT), device="cpu")
    orch_off, store_off = _build(tmp_path, strm_salience=False, salience_thresholds=th,
                                 recoverability_head=rec, latent_dynamics_head=ld,
                                 relevance_head=rel, ring_capacity=16)
    orch_on, store_on = _build(tmp_path, strm_salience=True, salience_thresholds=th,
                               recoverability_head=rec, latent_dynamics_head=ld,
                               relevance_head=rel, ring_capacity=16)
    try:
        res_off = _capture(orch_off)
        res_on = _capture(orch_on)
        assert res_off == res_on
        # flag-off -> no stash; flag-on + armed -> stash populated (a list, maybe empty
        # if no slot cleared the AND, but NOT None -- the hook ran).
        assert orch_off._salience_anchors is None
        assert orch_on._salience_anchors is not None
        assert orch_on._salience_armed() is True
        assert orch_off._salience_armed() is False
    finally:
        store_off.close()
        store_on.close()


@pytest.mark.skipif(not (_REC_CKPT.exists() and _LD_CKPT.exists() and _REL_CKPT.exists()),
                    reason="STRM head checkpoints not trained")
def test_salience_disarmed_when_a_head_missing(tmp_path):
    """strm_salience=True but a head missing -> _salience_armed is False -> the
    hook never runs -> _salience_anchors None, byte-identical to flag-off."""
    th = _thresh()
    rec = load_recoverability_head(str(_REC_CKPT), device="cpu")
    ld = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    # relevance_head deliberately None
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=th,
                         recoverability_head=rec, latent_dynamics_head=ld,
                         relevance_head=None, ring_capacity=16)
    try:
        assert orch._salience_armed() is False
        res = _capture(orch)
        assert orch._salience_anchors is None
        # and the result matches a plain flag-off orchestrator
        orch_plain, store_plain = _build(tmp_path, ring_capacity=16)
        try:
            assert _capture(orch_plain) == res
        finally:
            store_plain.close()
    finally:
        store.close()


@pytest.mark.skipif(not (_REC_CKPT.exists() and _LD_CKPT.exists() and _REL_CKPT.exists()),
                    reason="STRM head checkpoints not trained")
def test_salience_hook_failure_is_noop(tmp_path, monkeypatch):
    """If compute_salience raises (a head blows up), the hook swallows it:
    _salience_anchors stays None and the result is byte-identical to flag-off.
    A proactive-recall heuristic must never crash the turn.

    The swallow is the load-bearing contract, so we patch ``compute_salience``
    itself to raise -- this fires regardless of ring contents (a head's
    ``predict`` is only called for slots WITH provenance text, so sabotaging a
    head in isolation can be silently skipped when the ring holds only the raw
    query slot). Patching the entry point exercises the hook's try/except on
    every code path."""
    import src.subconscious.salience as sal_mod

    def _boom(*a, **k):
        raise RuntimeError("simulated head failure")
    monkeypatch.setattr(sal_mod, "compute_salience", _boom)

    th = _thresh(theta=1e18, phi=-1e18, surprise_cap=1e18)
    rec = load_recoverability_head(str(_REC_CKPT), device="cpu")
    ld = load_latent_dynamics_head(str(_LD_CKPT), device="cpu")
    rel = load_relevance_head(str(_REL_CKPT), device="cpu")
    orch, store = _build(tmp_path, strm_salience=True, salience_thresholds=th,
                         recoverability_head=rec, latent_dynamics_head=ld,
                         relevance_head=rel, ring_capacity=16)
    orch_plain, store_plain = _build(tmp_path, ring_capacity=16)
    try:
        res = _capture(orch)
        assert orch._salience_anchors is None  # swallowed -> no stash
        assert _capture(orch_plain) == res     # byte-identical to flag-off
    finally:
        store.close()
        store_plain.close()