"""STRM Phase 2d graduation-head tests (v1 proxy + v2 head + trainer + labeler).

CPU-only, no backbone, no embedder, no WaveDB, no replay data. The v1 proxy is
parameter-free (its math is exact, not learned), so the tests assert the
``integral(r_i dt)`` computation + threshold directly. The v2 head is a learned
classifier whose TRAINING RUN is deferred (Step 5 ships the trainer + replay
labels); here we validate the module's predict shape/range, the
``llm_signal`` one-hot encoding (reusing the ``forgetting.LLM_SIGNAL_MODIFIERS``
vocabulary), the broadcast/bad-dim guards, the checkpoint round-trip, the AUC
gate + v1-score helpers, the replay-label generator (re-appearance after a
ring gap), and that the v2 trainer clears the synthetic gate (v2 beats the v1
r_i proxy when slot_y carries the signal and r_i does not) -- the same surface
the 2b/2c/2a head tests cover for their heads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.graduation_head import (
    DEFAULT_GRADUATION_THRESHOLD,
    LLM_SIGNAL_DIM,
    LLM_SIGNAL_VOCAB,
    SLOT_DIM,
    STATE_DIM_POOLED,
    GraduationHeadV2,
    GraduationProxyV1,
    encode_llm_signal,
    load_graduation_head,
)
from src.subconscious.recoverability_head import STATE_DIM_POOLED as _REC_STATE_DIM
from src.subconscious.training.graduation_training import (
    GraduationTrainingConfig,
    _auc,
    fit_graduation,
    load_replay_labeled,
    v1_scores_per_record,
)
from src.memory.forgetting import LLM_SIGNAL_MODIFIERS
from scripts.generate_graduation_labels import label_later_needed, load_replay
from src.config import Phase2cConfig, config as _config

# Orchestrator harness stubs (shared with tests/test_orchestrator.py -- the
# replay-logger integration test mirrors that suite's offline harness).
from tests.test_orchestrator import _StubEmbedder, _StubPlanner, _StubModeA, _ep


# sanity: the v2 head reuses the recoverability head's pooled-state dim
assert STATE_DIM_POOLED == _REC_STATE_DIM == 4 * 384


# ── v1 proxy: integral(r_i dt) math ──

def test_graduation_proxy_v1_integral_math():
    proxy = GraduationProxyV1(threshold=1.0, dt=1.0)
    # sum(r_i * dt) -- order-independent
    assert proxy.integrate_relevance([0.5, 0.5, 0.5, 0.5]) == 2.0
    assert proxy.integrate_relevance([0.5, 0.5, 0.5, 0.5][::-1]) == 2.0
    assert proxy.integrate_relevance([1.0, 0.0, 0.0]) == 1.0
    # empty stream -> 0.0 (a slot never scored is not graduated)
    assert proxy.integrate_relevance([]) == 0.0
    # graduation_score is an alias for integrate_relevance
    assert proxy.graduation_score([0.25, 0.25]) == 0.5


def test_graduation_proxy_v1_dt_scales_the_integral():
    # dt scales the integral uniformly (the time-step the r_i stream is sampled
    # at). dt=2 doubles the score for the same r stream.
    p1 = GraduationProxyV1(dt=1.0)
    p2 = GraduationProxyV1(dt=2.0)
    stream = [0.4, 0.6, 0.5]
    assert p1.integrate_relevance(stream) == pytest.approx(1.5)
    assert p2.integrate_relevance(stream) == pytest.approx(3.0)


def test_graduation_proxy_v1_accepts_tensor_stream():
    proxy = GraduationProxyV1()
    r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    # a tensor stream is flattened + converted (same result as the list)
    assert proxy.integrate_relevance(r) == pytest.approx(2.0)
    # 2-D tensor is flattened (order-independent integral)
    assert proxy.integrate_relevance(r.reshape(2, 2)) == pytest.approx(2.0)


def test_graduation_proxy_v1_threshold_behavior():
    # graduate(score) = score >= threshold. A consistently-relevant slot clears
    # the default threshold; a never-relevant slot does not.
    proxy = GraduationProxyV1(threshold=1.0)
    assert proxy.graduate([0.5, 0.5, 0.5, 0.5]) is True      # 2.0 >= 1.0
    assert proxy.graduate([0.1, 0.1, 0.1]) is False          # 0.3 < 1.0
    # boundary: exactly the threshold graduates (>=)
    assert proxy.graduate([1.0]) is True
    # a higher threshold tightens promotion
    strict = GraduationProxyV1(threshold=3.0)
    assert strict.graduate([0.5, 0.5, 0.5, 0.5]) is False    # 2.0 < 3.0
    assert strict.graduate([1.0, 1.0, 1.0]) is True         # 3.0 >= 3.0


def test_graduation_proxy_v1_default_threshold_is_named_constant():
    # the default threshold is the named, uncalibrated constant (one obvious
    # calibration lever; Phase 4 / a later sweep sets it against the v2 head)
    proxy = GraduationProxyV1()
    assert proxy.threshold == DEFAULT_GRADUATION_THRESHOLD
    assert proxy.dt == 1.0


def test_graduation_proxy_v1_rejects_nonpositive_dt():
    with pytest.raises(ValueError, match="dt must be > 0"):
        GraduationProxyV1(dt=0.0)
    with pytest.raises(ValueError, match="dt must be > 0"):
        GraduationProxyV1(dt=-1.0)


def test_graduation_proxy_v1_forward_returns_score_not_bool():
    # forward returns the SCORE (the v2-beat metric), not the boolean -- callers
    # compare to threshold for the promote/don't decision.
    proxy = GraduationProxyV1(threshold=1.0)
    assert proxy.forward([0.5, 0.5]) == 1.0
    assert isinstance(proxy.forward([0.5, 0.5]), float)


def test_graduation_proxy_v1_has_no_parameters():
    # the v1 proxy is parameter-free -- it is a heuristic baseline, not a
    # learned head. No trainable params, no checkpoint.
    proxy = GraduationProxyV1()
    assert sum(p.numel() for p in proxy.parameters()) == 0


# ── llm_signal one-hot encoding (reuses the forgetting vocabulary) ──

def test_llm_signal_vocab_matches_forgetting_modifiers():
    # the v2 head reuses the forgetting.LLM_SIGNAL_MODIFIERS vocabulary -- it
    # does NOT invent a new signal taxonomy.
    assert LLM_SIGNAL_VOCAB == tuple(LLM_SIGNAL_MODIFIERS.keys())
    assert LLM_SIGNAL_DIM == len(LLM_SIGNAL_VOCAB) == 5


def test_encode_llm_signal_onehot_each_vocab_key():
    # each vocab key -> a distinct one-hot of width LLM_SIGNAL_DIM
    seen = set()
    for sig in LLM_SIGNAL_VOCAB:
        v = encode_llm_signal(sig)
        assert v.shape == (LLM_SIGNAL_DIM,)
        assert int(v.sum().item()) == 1                  # exactly one hot
        idx = int(v.argmax().item())
        seen.add(idx)
    assert seen == set(range(LLM_SIGNAL_DIM))             # all positions used


def test_encode_llm_signal_none_and_unknown_are_zeros():
    # None / unknown -> all-zeros (a missing signal, not a silent mis-wire: the
    # v2 head learns to treat absence as its own evidence).
    assert torch.all(encode_llm_signal(None) == 0.0)
    assert torch.all(encode_llm_signal("not_a_real_signal") == 0.0)
    assert torch.all(encode_llm_signal(123) == 0.0)       # non-str -> zeros
    assert encode_llm_signal(None).shape == (LLM_SIGNAL_DIM,)


# ── v2 head: predict shape/range + broadcast + bad dims ──

def test_graduation_head_v2_predict_shape_and_range():
    head = GraduationHeadV2()
    state = torch.randn(7, STATE_DIM_POOLED)
    slot = torch.randn(7, SLOT_DIM)
    sig = torch.randn(7, LLM_SIGNAL_DIM)
    p = head.predict(state, slot, sig)                       # [7, 1]
    assert p.shape == (7, 1)
    # sigmoid -> [0, 1]
    assert bool((p >= 0.0).all()) and bool((p <= 1.0).all())
    # 1-D inputs broadcast -> [1, 1]
    p1 = head.predict(torch.randn(STATE_DIM_POOLED), torch.randn(SLOT_DIM),
                      torch.randn(LLM_SIGNAL_DIM))
    assert p1.shape == (1, 1)


def test_graduation_head_v2_predict_rejects_bad_dims():
    head = GraduationHeadV2()
    state = torch.randn(3, STATE_DIM_POOLED)
    slot = torch.randn(3, SLOT_DIM)
    sig = torch.randn(3, LLM_SIGNAL_DIM)
    # state dim mismatch
    with pytest.raises(ValueError, match="state_pooled dim"):
        head.predict(torch.randn(3, 64), slot, sig)
    # slot dim mismatch
    with pytest.raises(ValueError, match="slot_y dim"):
        head.predict(state, torch.randn(3, 100), sig)
    # llm_signal dim mismatch
    with pytest.raises(ValueError, match="llm_signal_onehot dim"):
        head.predict(state, slot, torch.randn(3, 2))
    # batch mismatch (state batch != slot batch, neither is 1)
    with pytest.raises(ValueError, match="incompatible"):
        head.predict(torch.randn(3, STATE_DIM_POOLED), torch.randn(5, SLOT_DIM),
                     sig)


def test_graduation_head_v2_uses_encoded_llm_signal():
    # wiring encode_llm_signal into predict: a one-hot signal changes the score
    # vs the all-zeros (missing) signal, all else equal.
    head = GraduationHeadV2()
    state = torch.randn(STATE_DIM_POOLED)
    slot = torch.randn(SLOT_DIM)
    p_none = head.predict(state, slot, encode_llm_signal(None)).item()
    p_imp = head.predict(state, slot, encode_llm_signal("important")).item()
    assert p_none != p_imp


# ── v2 head: checkpoint round-trip + dim-mismatch ──

def test_graduation_head_v2_round_trips_through_loader(tmp_path):
    head = GraduationHeadV2()
    sd = head.state_dict()
    # save in the loader's checkpoint shape
    ckpt = {"head": sd, "state_dim_pooled": STATE_DIM_POOLED, "slot_dim": SLOT_DIM,
            "llm_signal_dim": LLM_SIGNAL_DIM, "hidden_dim": 128, "go": False}
    p = tmp_path / "grad.pt"
    torch.save(ckpt, p)
    loaded = load_graduation_head(str(p), device="cpu")
    # the loaded head's predict matches the original on the same inputs
    state = torch.randn(5, STATE_DIM_POOLED)
    slot = torch.randn(5, SLOT_DIM)
    sig = torch.randn(5, LLM_SIGNAL_DIM)
    with torch.no_grad():
        a = head.predict(state, slot, sig)
        b = loaded.predict(state, slot, sig)
    assert torch.allclose(a, b, atol=1e-6)


def test_graduation_head_v2_dim_mismatch_raises(tmp_path):
    head = GraduationHeadV2()
    ckpt = {"head": head.state_dict(), "state_dim_pooled": STATE_DIM_POOLED,
            "slot_dim": SLOT_DIM, "llm_signal_dim": LLM_SIGNAL_DIM,
            "hidden_dim": 128, "go": False}
    p = tmp_path / "grad.pt"
    torch.save(ckpt, p)
    # lie about slot_dim -> the rebuilt MLP's first Linear expects a different
    # input width -> torch raises a size-mismatch on load_state_dict.
    ckpt2 = dict(ckpt)
    ckpt2["slot_dim"] = 64
    p2 = tmp_path / "mismatch.pt"
    torch.save(ckpt2, p2)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_graduation_head(str(p2), device="cpu")


# ── AUC helper (rank-based, tie-aware) ──

def test_auc_known_value_perfect_separation():
    # two positives score above two negatives -> AUC 1.0
    assert _auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert _auc([0.1, 0.2, 0.9, 0.95], [0, 0, 1, 1]) == pytest.approx(1.0)
    # inverted -> AUC 0.0
    assert _auc([0.9, 0.95, 0.1, 0.2], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_auc_all_ties_is_chance():
    # all scores tied -> average ranks -> AUC 0.5 (chance), no crash
    assert _auc([0.5, 0.5, 0.5], [0, 1, 1]) == pytest.approx(0.5)
    assert _auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)


def test_auc_single_class_is_chance():
    # one class empty -> 0.5 (the gate then fails on the chance floor, not a
    # crash). Both directions.
    assert _auc([0.1, 0.2, 0.3], [0, 0, 0]) == 0.5
    assert _auc([0.1, 0.2, 0.3], [1, 1, 1]) == 0.5


# ── v1 scores: cumulative sum(r_i dt) per (session, source_id) by turn ──

def _rec(turn_id, session_id, source_id, r_i, label=None,
         slot0=0.0, signal="routine"):
    """Build a minimal replay record (the orchestrator's replay.jsonl shape).

    ``later_needed`` defaults to ``None`` -- the replay logger writes null and
    the label generator fills it; tests pass an explicit bool only when they
    need a labeled record (the v2 trainer / the loader drop test).
    """
    return {
        "turn_id": turn_id, "session_id": session_id, "source_id": source_id,
        "slot_index": 0, "text": f"text-{source_id}" if source_id else None,
        "slot_y_t": [slot0] + [0.0] * (SLOT_DIM - 1),
        "state_t_pooled": [0.0] * STATE_DIM_POOLED,
        "r_i": r_i, "llm_signal": signal, "later_needed": label,
    }


def test_v1_scores_cumulative_per_source_by_turn():
    recs = [
        _rec(1, "s1", "x", 0.5),     # x@1: cum 0.5
        _rec(3, "s1", "x", 0.3),     # x@3: cum 0.8
        _rec(2, "s1", "y", 0.9),     # y@2: cum 0.9
        _rec(1, "s2", "z", 1.0),     # s2/z@1: cum 1.0 (different session)
    ]
    scores = v1_scores_per_record(recs, dt=1.0)
    assert scores == pytest.approx([0.5, 0.8, 0.9, 1.0])
    # dt scales the integral uniformly
    scores2 = v1_scores_per_record(recs, dt=2.0)
    assert scores2 == pytest.approx([1.0, 1.6, 1.8, 2.0])


def test_v1_scores_null_r_i_contributes_zero_but_carries_cumulative():
    # a null r_i at a turn adds 0.0 but does not reset the cumulative sum.
    recs = [
        _rec(1, "s1", "x", 0.4),     # cum 0.4
        _rec(2, "s1", "x", None),    # cum 0.4 (null -> +0.0)
        _rec(3, "s1", "x", 0.6),     # cum 1.0
    ]
    assert v1_scores_per_record(recs, dt=1.0) == pytest.approx([0.4, 0.4, 1.0])


def test_v1_scores_none_source_id_records_stay_zero():
    # the raw query step (source_id=None) cannot be grouped -> v1 score 0.0.
    recs = [
        _rec(1, "s1", None, 0.9),
        _rec(2, "s1", "x", 0.5),
    ]
    assert v1_scores_per_record(recs, dt=1.0) == pytest.approx([0.0, 0.5])


# ── label generator: re-appearance after a ring gap ──

def test_label_later_needed_re_appearance_after_gap():
    # session s1, present turns 1..5.
    # A: appears at 1,2 then absent at 3, re-appears at 4 -> A@1,A@2 later_needed
    # B: appears at 1,2,3 consecutively (no gap) -> all False
    # C: appears at 1 and 5, absent at 2,3,4 -> C@1 later_needed
    recs = []
    for t in [1, 2, 4]:
        recs.append(_rec(t, "s1", "A", 0.5, label=True, slot0=0.0))
    for t in [1, 2, 3]:
        recs.append(_rec(t, "s1", "B", 0.5, label=True, slot0=0.0))
    for t in [1, 5]:
        recs.append(_rec(t, "s1", "C", 0.5, label=True, slot0=0.0))
    # A None-source_id slot (the raw query step) -- must stay null (unmatchable).
    recs.append(_rec(1, "s1", None, 0.5, label=True, slot0=0.0))

    labeled = label_later_needed(recs)
    # build a (turn, source_id) -> later_needed map (None source -> null)
    by_key = {(r["turn_id"], r["source_id"]): r["later_needed"] for r in labeled}
    # A@1 and A@2: re-appears at 4 with turn 3 absent -> True
    assert by_key[(1, "A")] is True
    assert by_key[(2, "A")] is True
    # A@4: no later appearance -> False
    assert by_key[(4, "A")] is False
    # B@1,B@2,B@3: consecutive presence, no gap -> False
    assert by_key[(1, "B")] is False
    assert by_key[(2, "B")] is False
    assert by_key[(3, "B")] is False
    # C@1: re-appears at 5 with 2,3,4 absent -> True. C@5: no later -> False.
    assert by_key[(1, "C")] is True
    assert by_key[(5, "C")] is False
    # None-source slot -> later_needed null (trainer drops it)
    assert by_key[(1, None)] is None


def test_label_later_needed_does_not_mutate_input():
    # the replay log records carry ``later_needed: null`` (the logger's shape);
    # the labeler returns a NEW list of shallow copies with the field filled,
    # leaving the originals' null untouched.
    recs = [_rec(1, "s1", "x", 0.5), _rec(3, "s1", "x", 0.5)]
    assert recs[0]["later_needed"] is None
    labeled = label_later_needed(recs)
    # originals untouched (a NEW shallow copy per record)
    assert recs[0]["later_needed"] is None
    assert recs[1]["later_needed"] is None
    # x appears at turns 1 and 3 only (no turn 2 in the session) -> no gap ->
    # both False (consecutive-ish presence is not a re-recall after eviction).
    assert labeled[0]["later_needed"] is False
    assert labeled[1]["later_needed"] is False


def test_load_replay_drops_malformed_lines(tmp_path):
    p = tmp_path / "replay.jsonl"
    p.write_text(
        json.dumps({"turn_id": 1, "session_id": "s1", "source_id": "x"}) + "\n"
        + "not json\n"
        + json.dumps({"turn_id": 2}) + "\n"            # missing session_id
        + json.dumps({"turn_id": 3, "session_id": "s2", "source_id": "y"}) + "\n",
        encoding="utf-8",
    )
    recs = load_replay(str(p))
    assert len(recs) == 2
    assert recs[0]["turn_id"] == 1
    assert recs[1]["session_id"] == "s2"


# ── v2 trainer clears the synthetic gate (v2 beats v1, chance floor) ──

def test_fit_graduation_v2_beats_v1_on_synthetic_gate(tmp_path):
    # Two sessions, 10 records each (5 positive / 5 negative). The label is
    # perfectly separated by slot_y_t[0] (+3.0 positive, -3.0 negative) -> the
    # v2 MLP learns it -> v2_auc ~ 1.0. r_i is a CONSTANT 0.5 with one record
    # per (session, source_id), so each v1 score = 0.5 (a single r_i in the
    # cumulative) -> all v1 scores tied -> v1_auc = 0.5 (chance). state + signal
    # are all-zeros (uninformative). The gate: v2 beats v1 AND clears the
    # chance floor -> go=True, best_v2_auc > best_v1_auc.
    records: list[dict] = []
    for sess in ("s1", "s2"):
        for i in range(5):
            records.append(_rec(
                turn_id=i * 2, session_id=sess, source_id=f"{sess}-neg-{i}",
                r_i=0.5, label=False, slot0=-3.0))
            records.append(_rec(
                turn_id=i * 2 + 1, session_id=sess, source_id=f"{sess}-pos-{i}",
                r_i=0.5, label=True, slot0=+3.0))
    cfg = GraduationTrainingConfig(
        epochs=30, learning_rate=3e-3, val_fraction=0.5, min_val_n=8,
        gate_auc_min=0.5, checkpoint_dir=str(tmp_path / "ckpt"),
    )
    result = fit_graduation(records, cfg)
    assert result["go"] is True
    assert result["best_v2_auc"] > result["best_v1_auc"]
    assert result["best_v2_auc"] >= 0.95          # clean separation
    assert result["best_v1_auc"] == pytest.approx(0.5)   # constant r_i -> chance
    # the checkpoint was written + round-trips through the loader.
    best = tmp_path / "ckpt" / "best.pt"
    assert best.exists()
    loaded = load_graduation_head(str(best), device="cpu")
    # the loaded head separates the two classes on a fresh batch
    pos = torch.tensor([[+3.0] + [0.0] * (SLOT_DIM - 1)], dtype=torch.float32)
    neg = torch.tensor([[-3.0] + [0.0] * (SLOT_DIM - 1)], dtype=torch.float32)
    st = torch.zeros(1, STATE_DIM_POOLED)
    sig = torch.zeros(1, LLM_SIGNAL_DIM)
    with torch.no_grad():
        p_pos = loaded.predict(st, pos, sig).item()
        p_neg = loaded.predict(st, neg, sig).item()
    assert p_pos > p_neg


def test_fit_graduation_empty_records_raises():
    with pytest.raises(RuntimeError, match="no records"):
        fit_graduation([])


def test_load_replay_labeled_drops_null_and_missing(tmp_path):
    p = tmp_path / "labeled.jsonl"
    p.write_text(
        json.dumps(_rec(1, "s1", "x", 0.5, label=True)) + "\n"          # keep
        + json.dumps(_rec(2, "s1", "y", 0.5, label=False)) + "\n"      # keep
        + json.dumps(_rec(3, "s1", None, 0.5, label=None)) + "\n"      # null label -> drop
        + json.dumps({"turn_id": 4, "session_id": "s1", "source_id": "z"}) + "\n"  # missing fields -> drop
        + "garbage\n",
        encoding="utf-8",
    )
    recs = load_replay_labeled(str(p))
    # only the two records with a bool label AND all feature fields survive.
    assert len(recs) == 2
    assert all(isinstance(r["later_needed"], bool) for r in recs)


# ── replay-logger integration: orchestrator writes replay.jsonl (ring ON) ──

def test_orchestrator_writes_graduation_replay_when_logging_on(tmp_path):
    """With the ring ON + strm_graduation_logging on, query() appends one
    replay.jsonl record per WM ring slot per turn. The records carry the v2
    head's three feature fields (state_t_pooled, slot_y_t, llm_signal) + the
    label-generator match keys (turn_id, session_id, source_id, text) +
    later_needed: null (filled later by the label generator)."""
    from src.orchestrator import PonderOrchestrator
    from src.memory.store import HippocampalStore
    from src.retrieval.retriever import HippocampalRetriever
    from src.subconscious.backbone import JGSBackbone
    from src.subconscious.configs import BackboneConfig

    # Build a tiny orchestrator with the ring ON (ring_capacity=8) on a tmp
    # store with one retrievable episode. Mirror tests/test_orchestrator.py's
    # stub harness (no Bonsai, no GLiNER).
    store = HippocampalStore(str(tmp_path / "db"))
    ep = _ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")
    store.encode_episode(ep)
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                    embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    replay_path = tmp_path / "replay.jsonl"
    # Redirect the class-level replay path so the test never touches the real
    # data dir.
    saved_path = PonderOrchestrator._REPLAY_PATH
    PonderOrchestrator._REPLAY_PATH = replay_path
    saved_flag = _config.strm_graduation_logging
    _config.strm_graduation_logging = True
    try:
        orch = PonderOrchestrator(
            store=store, retriever=retriever, backbone=backbone,
            embedder=_StubEmbedder(), mode_a=_StubModeA(reply="R"),
            config=cfg, user_id="victor", ring_capacity=8,
        )
        res = orch.query("What did Alice say?")
        assert res["supported"] is True
    finally:
        PonderOrchestrator._REPLAY_PATH = saved_path
        _config.strm_graduation_logging = saved_flag
    store.close()

    assert replay_path.exists(), "replay logger wrote nothing (ring off?)"
    import json as _json
    lines = [l for l in replay_path.read_text(encoding="utf-8").splitlines() if l]
    assert lines, "replay.jsonl is empty"
    recs = [_json.loads(l) for l in lines]
    # one record per ring slot for this turn
    assert len(recs) >= 1
    for r in recs:
        # the three v2-head feature fields are present + right shape
        assert len(r["slot_y_t"]) == SLOT_DIM
        assert len(r["state_t_pooled"]) == STATE_DIM_POOLED
        assert "r_i" in r                       # null (no relevance head loaded)
        assert r["r_i"] is None
        assert "llm_signal" in r
        # label-generator match keys
        assert r["turn_id"] == 1
        assert r["session_id"] == "victor"      # no encoder -> user_id
        assert r["later_needed"] is None        # filled later by the labeler
    # the recalled episode's source_id is threaded into a ring slot
    source_ids = {r["source_id"] for r in recs}
    assert "ep_001" in source_ids
    # the raw query step carries a None source_id (the labeler drops it)
    assert None in source_ids


def test_orchestrator_no_replay_when_logging_off(tmp_path):
    """strm_graduation_logging off (the default) -> no replay.jsonl written,
    byte-identical to the pre-2d path. Guards the flag-off invariant."""
    from src.orchestrator import PonderOrchestrator
    from src.memory.store import HippocampalStore
    from src.retrieval.retriever import HippocampalRetriever
    from src.subconscious.backbone import JGSBackbone
    from src.subconscious.configs import BackboneConfig

    store = HippocampalStore(str(tmp_path / "db"))
    ep = _ep("ep_001", entities=["Alice"], summary="Alice said use Postgres")
    store.encode_episode(ep)
    plan = {"entities": ["Alice"], "entity_mode": "union"}
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan),
                                    embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / "sessions")
    replay_path = tmp_path / "replay.jsonl"
    saved_path = PonderOrchestrator._REPLAY_PATH
    PonderOrchestrator._REPLAY_PATH = replay_path
    saved_flag = _config.strm_graduation_logging
    _config.strm_graduation_logging = False
    try:
        orch = PonderOrchestrator(
            store=store, retriever=retriever, backbone=backbone,
            embedder=_StubEmbedder(), mode_a=_StubModeA(reply="R"),
            config=cfg, user_id="victor", ring_capacity=8,
        )
        orch.query("What did Alice say?")
    finally:
        PonderOrchestrator._REPLAY_PATH = saved_path
        _config.strm_graduation_logging = saved_flag
    store.close()
    assert not replay_path.exists()