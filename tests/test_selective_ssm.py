"""Unit tests for the owned ``SelectiveSSM`` backend (``src/subconscious/ssm.py``).

CPU-runnable. Exercises the ``SSMBackend`` protocol contract and the
forward==step equivalence the live-serve / generation path depends on.
"""

from __future__ import annotations

import torch

from src.subconscious import make_ssm
from src.subconscious.configs import BackboneConfig


def _cfg(**kw) -> BackboneConfig:
    base = dict(d_model=32, d_state=8, ssm_backend="selective")
    base.update(kw)
    return BackboneConfig(**base)


def test_factory_returns_selective():
    m = make_ssm("selective", _cfg())
    assert type(m).__name__ == "SelectiveSSM"
    assert m.d_model == 32 and m.d_state == 8


def test_init_state_shape():
    m = make_ssm("selective", _cfg())
    s = m.init_state(2, torch.device("cpu"), torch.float32)
    assert s.shape == (2, 8, 32)
    assert torch.all(s == 0)


def test_forward_shapes():
    m = make_ssm("selective", _cfg())
    x = torch.randn(2, 5, 32)
    y, st = m.forward(x)
    assert y.shape == (2, 5, 32)
    assert st.shape == (2, 8, 32)


def test_step_shapes():
    m = make_ssm("selective", _cfg())
    x = torch.randn(2, 32)
    s = m.init_state(2, torch.device("cpu"), torch.float32)
    y, s1 = m.step(x, s)
    assert y.shape == (2, 32)
    assert s1.shape == (2, 8, 32)


def test_forward_equals_step_loop():
    """The sequence ``forward`` must equal stepping one token at a time.

    This is the invariant the generation / live-serve path relies on: ``step`` is
    not a stub, it is the same recurrence as ``forward``.
    """
    m = make_ssm("selective", _cfg())
    m.eval()
    x = torch.randn(2, 6, 32)
    y_seq, _ = m.forward(x)
    s = m.init_state(2, torch.device("cpu"), torch.float32)
    ys = []
    for t in range(6):
        yt, s = m.step(x[:, t, :], s)
        ys.append(yt)
    y_loop = torch.stack(ys, dim=1)
    assert torch.allclose(y_seq, y_loop, atol=1e-5)


def test_backward_grads_flow():
    m = make_ssm("selective", _cfg())
    x = torch.randn(2, 5, 32)
    y, _ = m.forward(x)
    y.sum().backward()
    grads = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
    # Every parameter should receive a gradient (no detached/broken path).
    assert len(grads) == 8
    assert all(torch.isfinite(p).all() and p.abs().sum() > 0 for p in grads.values())


def test_state_evolution_is_selective():
    """Δ modulates BOTH decay and write: a large-Δ input replaces state, a
    near-zero-Δ input keeps it. This is the selectivity the content objective
    needs (write-and-keep), and the property ReferenceSSM's coupled gate lacks.
    """
    m = make_ssm("selective", _cfg())
    m.eval()
    s0 = torch.randn(1, 8, 32)
    # A large-norm input -> strong Δ -> state moves substantially.
    x_big = torch.randn(1, 32) * 5.0
    _, s_big = m.step(x_big, s0.clone())
    # A zero input -> softplus(W_Δ(0)) is small but nonzero; state barely moves.
    x_zero = torch.zeros(1, 32)
    _, s_zero = m.step(x_zero, s0.clone())
    big_move = (s_big - s0).abs().mean().item()
    zero_move = (s_zero - s0).abs().mean().item()
    assert big_move > zero_move