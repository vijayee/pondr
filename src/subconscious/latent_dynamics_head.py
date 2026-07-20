"""LatentDynamicsHead: a linear next-state predictor over the WM recurrent state.

STRM Phase 2c -- the cheapest of the four read-out heads. It predicts
``z_{t+1}`` from ``z_t`` and emits a per-step *surprise* signal
``||A z_t + b - z_{t+1}||^2``. The Phase 0b probe de-risked this: a linear
predictor fits the dynamics (R^2=0.297 over the constant-mean baseline) AND
its L2-residual surprise-AUC (0.7625) beats a JEPA cosine predictor (0.565).
Linear cannot collapse (frozen backbone, no representation drift), so the
EMA/stop-grad/negatives anti-collapse machinery JEPA needs is NOT required
here -- which is why the head is a single ``nn.Linear`` rather than a JEPA
predictor. (JEPA is reserved for a future generative rollout, k>1
imagination; see docs/STRM-architecture-proposal.md §6.2.)

This is a plain ``nn.Module``, NOT a ``JGSInstance`` subclass. The
``JGSInstance`` pattern (doc_kind_head, retrieval_gate) is for heads that
STEP the SSM over an input sequence; this head only READS the already-
produced recurrent state and predicts the next. Forcing it into
``JGSInstance`` would drag in unused input_proj / output_proj / LoRA / gate
machinery -- the "works but for the wrong reason" smell -- so it is a lean
module holding one ``nn.Linear(384, 384)``. It is not registered in
``INSTANCE_CONFIGS`` (that registry is for ``JGSInstance`` configs); its
shape is fixed by the 0b-validated "last layer, mean over d_state" projection.

Training is a closed-form ridge fit (``training/latent_dynamics_training.py``)
baked into the ``nn.Linear``'s weight/bias -- no epoch loop, no optimizer.
The surprise signal is consumed downstream by Phase 4 (salience trigger /
state-conditioned retrieval), which is also where the head is wired into the
serve path. The loader ``load_latent_dynamics_head`` stands up the trained
module from its checkpoint for that wiring.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

# The state representation the 0b probe validated for this head: last SSM
# layer, mean over the d_state channel -> [384]. (The 1536-dim "pooled" rep
# was underdetermined at N=957 < D=1537 and gave a false NO-GO.)
STATE_DIM = 384


class LatentDynamicsHead(nn.Module):
    """Linear ``z_{t+1} = A z_t + b`` over the WM recurrent state's last layer.

    The single ``nn.Linear(STATE_DIM, STATE_DIM)`` holds A (weight) and b
    (bias). The closed-form ridge fit bakes the feature standardization into
    these params, so ``predict`` operates on the RAW projected state -- no
    runtime standardization needed. ``surprise`` is the L2 prediction residual
    the Phase 4 salience trigger reads as "this transition was unexpected".
    """

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.linear = nn.Linear(self.state_dim, self.state_dim, bias=True)

    # ── state projection ──

    def project(self, state_tensors: list[Tensor]) -> Tensor:
        """Map the live per-layer WM state -> the z vector this head predicts on.

        ``state_tensors`` is ``WorkingMemory.state_tensors()``: a list of 4
        per-layer tensors of shape ``[1, d_state=16, d_model=384]`` (or
        ``[16, 384]`` when the batch dim is squeezed). We take the LAST layer
        and mean over the d_state axis -> ``[state_dim]`` (or ``[1, state_dim]``
        if a leading batch dim is present). This is the 0b-validated "last"
        representation; the head's Linear was fit on it.

        Returns a 2-D ``[1, state_dim]`` tensor for caller convenience (the
        Linear's first dim is its in_features, so a ``[state_dim]`` 1-D input
        would also work, but a batched shape is the common downstream case).
        """
        if not state_tensors:
            raise ValueError("LatentDynamicsHead.project called with no state tensors")
        last = state_tensors[-1].to(torch.float32)   # [1, 16, 384] or [16, 384]
        if last.dim() == 3:
            # [batch, d_state, d_model] -> [batch, d_model] (mean over d_state)
            z = last.mean(dim=1)
        elif last.dim() == 2:
            # [d_state, d_model] -> [d_model]
            z = last.mean(dim=0).unsqueeze(0)
        else:
            raise ValueError(
                f"LatentDynamicsHead.project: last-layer state has unsupported "
                f"ndim={last.dim()} (expected 2 [d_state,d_model] or 3 "
                f"[batch,d_state,d_model])"
            )
        return z

    # ── prediction + surprise ──

    def predict(self, z: Tensor) -> Tensor:
        """``z_{t+1}_hat = A z_t + b`` -- the one-step next-state prediction."""
        return self.linear(z.to(torch.float32))

    def surprise(self, z_t: Tensor, z_tp1: Tensor) -> Tensor:
        """Surprise = mean-squared prediction residual over the state dims.

        ``surprise = ((A z_t + b) - z_{t+1})^2 .mean(-1)``. Higher = the actual
        next state was less predictable from the current state = more
        surprising. The L2 residual (NOT cosine) is the metric the 0b probe
        found discriminative (surprise-AUC 0.7625); a cosine-trained JEPA
        predictor underperformed here (0.565) because it is scale-invariant and
        calibrates magnitude poorly. Returns a per-row scalar (``[]`` or
        ``[batch]``) matching the leading dim of ``z_t``.
        """
        pred = self.predict(z_t)
        return ((pred - z_tp1.to(pred.dtype).to(pred.device)) ** 2).mean(dim=-1)

    # ── load ──

    @classmethod
    def from_state_dict(cls, sd: dict, state_dim: int = STATE_DIM) -> "LatentDynamicsHead":
        """Build a head and load a raw ``nn.Linear`` state_dict into it.

        The checkpoint stores the Linear's ``state_dict`` directly (under
        ``"linear"``), so this just constructs the module and ``load_state_dict``
        s it. Strict by default: a shape mismatch (e.g. a different state_dim)
        is a hard error, not a silent mis-wire.
        """
        head = cls(state_dim=state_dim)
        missing, unexpected = head.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"latent-dynamics head state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        head.eval()
        return head


def load_latent_dynamics_head(
    path: str,
    device: str = "auto",
    map_location: str = "cpu",
) -> LatentDynamicsHead:
    """Load a trained LatentDynamicsHead checkpoint -> ready-to-serve module.

    The checkpoint is ``{"linear": state_dict, "state_dim": int, "r2": float,
    "surprise_auc": float}`` (see ``latent_dynamics_training.fit_latent_dynamics``).
    Unlike the ``JGSInstance`` loaders this takes NO backbone -- the head reads
    WM state at serve, it does not own a backbone. ``state_dim`` is read before
    construction so a checkpoint fit on a different rep (e.g. a future 1536-dim
    variant) loads into a matching Linear. Moves to the resolved device, eval.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dim = int(ckpt.get("state_dim", STATE_DIM)) if isinstance(ckpt, dict) else STATE_DIM
    sd = ckpt["linear"] if isinstance(ckpt, dict) and "linear" in ckpt else ckpt
    head = LatentDynamicsHead.from_state_dict(sd, state_dim=state_dim)
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return head.to(dev).eval()