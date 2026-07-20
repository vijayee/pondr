"""STRM Phase 2d graduation-head tests (v1 proxy + v2 head shell).

CPU-only, no backbone, no embedder, no WaveDB, no replay data. The v1 proxy is
parameter-free (its math is exact, not learned), so the tests assert the
``integral(r_i dt)`` computation + threshold directly. The v2 head is a learned
classifier whose TRAINING RUN is deferred (Step 5 ships the trainer + replay
labels); here we validate the module's predict shape/range, the
``llm_signal`` one-hot encoding (reusing the ``forgetting.LLM_SIGNAL_MODIFIERS``
vocabulary), the broadcast/bad-dim guards, and the checkpoint round-trip --
the same surface the 2b/2c/2a head tests cover for their heads.
"""

from __future__ import annotations

import pytest
import torch

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
from src.memory.forgetting import LLM_SIGNAL_MODIFIERS


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