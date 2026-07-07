"""Pluggable SSM backend.

The JGS backbone is not coupled to a specific SSM implementation. It talks to
any ``SSMBackend`` through a small protocol: given an input token (or sequence)
and a recurrent state, produce an output and a new state.

Three backends share this protocol:

- ``ReferenceSSM`` — a minimal **selective** state-space model implemented in
  pure PyTorch. Real and trainable, but NOT Mamba3-faithful. Zero dependencies
  beyond torch, so it runs on CPU for the dev loop and unit tests. This is the
  default.
- ``Mamba3PyTorchBackend`` — wraps the community pure-PyTorch Mamba3 reference
  (``rishikksh20/mamba3-pytorch``), lazily imported. A faithful-but-slow CPU
  path for users who want Mamba3 dynamics without a CUDA pod.
- ``Mamba3CUDABackend`` — wraps the official ``mamba_ssm.Mamba3`` (Triton /
  TileLang / CuTe DSL kernels), lazily imported. CUDA/Hopper, pod-only.

The recurrent ``step()`` path is exposed for Phase 2b+ live instance inference;
Phase 2a pre-training uses ``forward()`` over whole sequences (the parallel /
prefill path), which is what runs on Ampere/Ada without Hopper.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from .configs import BackboneConfig


@runtime_checkable
class SSMBackend(Protocol):
    """Minimal recurrent SSM interface used by the JGS backbone."""

    @property
    def d_model(self) -> int: ...

    @property
    def d_state(self) -> int: ...

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Fresh recurrent state ``[batch, d_state, d_model]``."""

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Process a sequence ``[batch, seq, d_model]``.

        Returns ``(output [batch, seq, d_model], final_state [batch, d_state, d_model])``.
        """

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """One recurrent step. ``x`` is ``[batch, d_model]``.

        Returns ``(output [batch, d_model], new_state)``. Used by live instance
        inference (Phase 2b+); not on the 2a pre-training path.
        """


class ReferenceSSM(nn.Module):
    """Minimal selective state-space model (pure PyTorch, CPU-runnable).

    A discrete-time selective SSM: at each step the transition matrix is
    input-dependent (``A_t = diag(sigma(W_A x_t))``), so the model can gate
    information in/out of the recurrent state — the core selective-scan idea,
    if not Mamba3's exact discretization. State shape ``[batch, d_state, d_model]``
    matches the doc's JGS state layout so the backbone/gate code is unchanged
    when a real Mamba3 backend is swapped in.

    This is a *stand-in* for dev/tests, not a fidelity target. The faithful path
    is ``Mamba3PyTorchBackend`` / ``Mamba3CUDABackend``.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self._d_model = config.d_model
        self._d_state = config.d_state

        # Input-dependent selective gates and projections. d_state independent
        # state channels, each a vector of size d_model.
        self.W_A = nn.Linear(config.d_model, config.d_state)   # retention gate
        self.W_B = nn.Linear(config.d_model, config.d_state * config.d_model)
        self.W_C = nn.Linear(config.d_state * config.d_model, config.d_model)
        self.D = nn.Parameter(torch.ones(config.d_model))      # skip connection

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def d_state(self) -> int:
        return self._d_state

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch, self._d_state, self._d_model, device=device, dtype=dtype)

    def _step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        # x: [batch, d_model], state: [batch, d_state, d_model]
        gate = torch.sigmoid(self.W_A(x))                      # [batch, d_state]
        b = self.W_B(x).view(x.shape[0], self._d_state, self._d_model)
        # Selective retention: per-channel decay + write.
        g = gate.unsqueeze(-1)                                 # [batch, d_state, 1]
        new_state = g * b + (1.0 - g) * state
        flat = new_state.reshape(x.shape[0], -1)                # [batch, d_state*d_model]
        y = self.W_C(flat) + self.D * x
        return y, new_state

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        # x: [batch, seq, d_model]
        batch, seq, _ = x.shape
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)
        outputs = []
        for t in range(seq):
            y, state = self._step(x[:, t, :], state)
            outputs.append(y)
        out = torch.stack(outputs, dim=1)                       # [batch, seq, d_model]
        return out, state

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        return self._step(x, state)


def make_ssm(backend: str, config: BackboneConfig) -> SSMBackend:
    """Factory for an SSM backend by name.

    ``backend``: ``"reference"`` | ``"mamba3-pytorch"`` | ``"mamba3-cuda"``.
    The Mamba3 variants are lazily imported so this module has no hard
    dependency on ``mamba_ssm`` or the community reference (neither builds on
    this Windows dev box).
    """
    if backend == "reference":
        return ReferenceSSM(config)
    if backend == "mamba3-pytorch":
        return Mamba3PyTorchBackend(config)
    if backend == "mamba3-cuda":
        return Mamba3CUDABackend(config)
    raise ValueError(f"unknown SSM backend: {backend!r}")


class Mamba3PyTorchBackend(nn.Module):
    """Wraps the community pure-PyTorch Mamba3 reference (faithful, slow).

    Lazily imports ``mamba3`` from ``rishikksh20/mamba3-pytorch``. Run on CPU/GPU
    when you want real Mamba3 dynamics without a CUDA pod. The reference's
    recurrent scan is O(L) serial — fine for dev, too slow for production
    pre-training.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        try:
            from mamba3 import Mamba3 as _Mamba3  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised only when installed
            raise ImportError(
                "mamba3-pytorch backend requires the community reference: "
                "pip install git+https://github.com/rishikksh20/mamba3-pytorch"
            ) from e
        self._d_model = config.d_model
        self._d_state = config.d_state
        self.model = _Mamba3(d_model=config.d_model, d_state=config.d_state)
        # Stack layers ourselves (Mamba3 has no n_layers kwarg — see §0.1).

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def d_state(self) -> int:
        return self._d_state

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch, self._d_state, self._d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        # The community reference is a sequence model; recurrent state is
        # handled internally. We expose a zero state placeholder for the
        # protocol and rely on the sequence output for pre-training.
        out = self.model(x)
        batch = x.shape[0]
        final = self.init_state(batch, x.device, x.dtype)
        return out, final

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:  # pragma: no cover
        raise NotImplementedError(
            "mamba3-pytorch reference does not expose a per-step API; use forward()"
        )


class Mamba3CUDABackend(nn.Module):
    """Wraps the official ``mamba_ssm.Mamba3`` (CUDA/Hopper kernels, pod-only).

    Lazily imports ``mamba_ssm``. Build from source on the pod:
    ``MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation
    git+https://github.com/state-spaces/mamba.git``. Note the official
    ``step()`` decode path is only tested on H100 (see §0.1); 2a uses
    ``forward()`` (Triton prefill) which runs on Ampere/Ada.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        try:
            from mamba_ssm import Mamba3 as _Mamba3  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised only on the pod
            raise ImportError(
                "mamba3-cuda backend requires the official package: see §0.1"
            ) from e
        self._d_model = config.d_model
        self._d_state = config.d_state
        self.model = _Mamba3(
            d_model=config.d_model,
            d_state=config.d_state,
            headdim=64,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def d_state(self) -> int:
        return self._d_state

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch, self._d_state, self._d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        # The official Mamba3 manages recurrent state internally for the
        # parallel/prefill path; we expose a zero-state placeholder to satisfy
        # the SSMBackend protocol (unused on the 2a pre-training path, which
        # only consumes the sequence output). Per-step state is 2b+ (see step).
        out = self.model(x)
        batch = x.shape[0]
        final = self.init_state(batch, x.device, x.dtype)
        return out, final

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:  # pragma: no cover
        # Official step() uses inference_params and is H100-tested. Wire up in
        # Phase 2b when live instance inference is needed.
        raise NotImplementedError("per-step decode wired in Phase 2b")