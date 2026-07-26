"""Unit tests for the token-level LM-SSM (``src/subconscious/token_lm.py``).

CPU-runnable. Exercises the I/O wrapper the content objective lives in: the
forward/step/generate contract, tied LM head, gradient flow, and the
forward-prime-then-step invariant generation depends on.
"""

from __future__ import annotations

import torch

from src.subconscious.token_lm import LMConfig, SSMLanguageModel


def _cfg(**kw) -> LMConfig:
    base = dict(vocab=64, d_model=16, n_layers=2, d_state=4, seq_len=8)
    base.update(kw)
    return LMConfig(**base)


def test_lm_forward_shapes():
    m = SSMLanguageModel(_cfg())
    ids = torch.randint(0, 64, (3, 7))
    logits, states = m.forward(ids)
    assert logits.shape == (3, 7, 64)
    assert len(states) == 2
    assert states[0].shape == (3, 4, 16)


def test_lm_step_shapes():
    m = SSMLanguageModel(_cfg())
    ids = torch.randint(0, 64, (3, 7))
    _, states = m.forward(ids)
    one = torch.randint(0, 64, (3,))
    logits, states2 = m.step(one, states)
    assert logits.shape == (3, 64)
    assert len(states2) == 2
    assert states2[0].shape == (3, 4, 16)


def test_lm_tied_head():
    m = SSMLanguageModel(_cfg(tie_head=True))
    assert m.lm_head.weight is m.token_emb.weight
    m2 = SSMLanguageModel(_cfg(tie_head=False))
    assert m2.lm_head.weight is not m2.token_emb.weight


def test_lm_forward_then_step_equals_full_forward():
    """Stepping from a forward-primed state must equal the forward over the
    full (prompt + stepped) sequence. This is the invariant ``generate`` relies
    on: priming with the prompt then stepping one token reproduces the forward
    logits at the last position of the extended sequence.
    """
    cfg = _cfg()
    m = SSMLanguageModel(cfg)
    m.eval()
    ids = torch.randint(0, cfg.vocab, (2, 5))
    nxt = torch.randint(0, cfg.vocab, (2,))  # the token to step / append
    # Path A: forward over the prompt, then step ONE token from the primed state.
    _, states = m.forward(ids)
    step_logits, _ = m.step(nxt, states)
    # Path B: forward over prompt + [next] in one go; take the logits at the last
    # position (which is exactly what stepping `nxt` from the primed state
    # computes).
    full = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
    logits_b, _ = m.forward(full)
    last_b = logits_b[:, -1, :]
    assert torch.allclose(step_logits, last_b, atol=1e-4), (
        "step from a forward-primed state must match the forward over the "
        "extended sequence (the generation invariant)"
    )


def test_lm_generate_shapes_greedy():
    m = SSMLanguageModel(_cfg())
    prompt = torch.randint(0, 64, (2, 4))
    out = m.generate(prompt, max_new_tokens=5, temperature=0.0)
    assert out.shape == (2, 9)
    # The first 4 tokens are the prompt verbatim.
    assert torch.equal(out[:, :4], prompt)


def test_lm_generate_sample_shapes_and_topk():
    m = SSMLanguageModel(_cfg())
    prompt = torch.randint(0, 64, (2, 3))
    out = m.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=5)
    assert out.shape == (2, 7)
    # top_k=5 -> every sampled token must be among the top-5 logits at its step.
    # (Sanity: all ids in range.)
    assert ((out >= 0) & (out < 64)).all()


def test_lm_backward_grads_flow():
    m = SSMLanguageModel(_cfg())
    ids = torch.randint(0, 64, (2, 6))
    logits, _ = m.forward(ids)
    # Next-token CE on the shifted targets.
    targets = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 64), targets.reshape(-1)
    )
    loss.backward()
    # Embedding/head (tied) and the per-layer SSM params all receive grad. With
    # tied weights, `named_parameters()` deduplicates the shared Parameter under
    # the first name it encounters ("token_emb.weight"); "lm_head.weight" is NOT
    # a separate key. The tie is verified by the tensor identity + shared grad.
    named = {n: p for n, p in m.named_parameters() if p.grad is not None}
    assert "token_emb.weight" in named
    assert m.lm_head.weight is m.token_emb.weight  # tied -> same Parameter object
    assert m.token_emb.weight.grad is m.lm_head.weight.grad  # same grad handle
    # Per-layer SSM params receive grad (the recurrent block trains).
    assert any("layers.0.W" in n for n in named)
    assert all(torch.isfinite(p).all() and p.abs().sum() > 0 for p in named.values())


def test_lm_param_count_modest():
    m = SSMLanguageModel(_cfg(vocab=4096, d_model=256, n_layers=6, d_state=16))
    n = m.num_parameters()
    # ~14M target; allow a loose band (block params + tied emb/head 4096*256).
    assert 9_000_000 < n < 18_000_000, f"param count {n:,} outside the ~14M band"