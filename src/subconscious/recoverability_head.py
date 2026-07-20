"""RecoverabilityHead: predict how forgotten a past anchor is from the WM state.

STRM Phase 2b -- the recoverability read-out. Given the current recurrent
state ``state_t`` and a past anchor ``u_i`` (lag ``k = t - i``), it emits a
scalar ``e_hat(i,t)`` estimating the decoder's reconstruction error of ``u_i``
from ``state_t`` -- i.e. how much the SSM has forgotten ``u_i`` by step ``t``.
Phase 4 consumes this as the recoverability signal that drives state-
conditioned retrieval (which past anchor to surface / refresh).

The Phase 0a probe de-risked this: a ridge regressor ``P([state_t, u_i]) ->
e_hat`` scored AUC 0.810 against a per-split-median "forgotten" label, beating
the free monotonic-forgetting-in-k baseline (0.732). Both the label-generating
decoder ``D`` and the probe ``P`` were ridge in the probe, so -- by parity
with the 2c latent-dynamics head (linear because the 0b probe proved linear
works) -- this head is the probe's closed-form ridge baked into a single
``nn.Linear(1920, 1)`` (1920 = 1536-dim pooled state + 384-dim anchor). A
nonlinear MLP upgrade is deferred (same status as 2c's deferred MLP upgrade);
it would only be warranted if the linear head under-performed, which it did
not (0.810 GO).

This is a plain ``nn.Module``, NOT a ``JGSInstance`` subclass -- it READS the
already-produced recurrent state and the anchor, it does not step the SSM. It
is not registered in ``INSTANCE_CONFIGS`` (that registry is for ``JGSInstance``
configs). The state projection is the 0a-validated "pooled" rep (per-layer
mean over d_state, 4 layers concatenated -> [1536]); the anchor is the raw
384-dim input embedding. Training is closed-form
(``training/recoverability_training.py``); the decoder ``D`` is fit as a
ridge train-side only (to generate the e(i,t) labels) and is NOT saved -- the
head at serve is just ``P``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# The state representation the 0a probe validated for this head: per-layer
# mean over the d_state channel, 4 layers concatenated -> [1536].
STATE_DIM_POOLED = 4 * 384
# The anchor u_i is the raw 384-dim input embedding.
ANCHOR_DIM = 384
# P's input is [pooled state ; anchor u_i].
INPUT_DIM = STATE_DIM_POOLED + ANCHOR_DIM


class RecoverabilityHead(nn.Module):
    """Ridge ``e_hat(i,t) = P([state_t ; u_i])`` over the pooled WM state.

    A single ``nn.Linear(INPUT_DIM, 1)`` holds P's weights. The closed-form
    ridge fit bakes feature standardization into these params, so ``predict``
    operates on RAW [state ; u_i] -- no runtime standardization. The scalar
    output is the predicted reconstruction error of ``u_i`` from ``state_t``;
    higher = more forgotten. The decoder ``D`` that produced the training
    labels is train-side only and lives in the trainer, not this module.
    """

    def __init__(
        self,
        state_dim_pooled: int = STATE_DIM_POOLED,
        anchor_dim: int = ANCHOR_DIM,
    ) -> None:
        super().__init__()
        self.state_dim_pooled = int(state_dim_pooled)
        self.anchor_dim = int(anchor_dim)
        input_dim = self.state_dim_pooled + self.anchor_dim
        self.linear = nn.Linear(input_dim, 1, bias=True)

    # ── state projection ──

    def project_state(self, state_tensors: list[Tensor]) -> Tensor:
        """Map the live per-layer WM state -> the pooled vector P reads.

        ``state_tensors`` is ``WorkingMemory.state_tensors()``: a list of 4
        per-layer tensors of shape ``[1, d_state=16, d_model=384]`` (or
        ``[16, 384]`` when the batch dim is squeezed). We mean over the
        d_state axis per layer and concatenate the 4 layers -> ``[1, 1536]``.
        This is the 0a-validated "pooled" representation; P was fit on it.

        Returns a 2-D ``[1, state_dim_pooled]`` tensor.
        """
        if not state_tensors:
            raise ValueError(
                "RecoverabilityHead.project_state called with no state tensors"
            )
        if len(state_tensors) != 4:
            raise ValueError(
                f"RecoverabilityHead.project_state: expected 4 per-layer state "
                f"tensors, got {len(state_tensors)}"
            )
        # Mean over d_state per layer -> [1, 384] each, then cat -> [1, 1536].
        per_layer = []
        for st in state_tensors:
            s = st.to(torch.float32)            # [1, 16, 384] or [16, 384]
            if s.dim() == 3:
                per_layer.append(s.mean(dim=1))        # [1, 384]
            elif s.dim() == 2:
                per_layer.append(s.mean(dim=0).unsqueeze(0))  # [1, 384]
            else:
                raise ValueError(
                    f"RecoverabilityHead.project_state: per-layer state has "
                    f"unsupported ndim={s.dim()} (expected 2 [d_state,d_model] "
                    f"or 3 [batch,d_state,d_model])"
                )
        return torch.cat(per_layer, dim=1)     # [1, state_dim_pooled]

    # ── prediction ──

    def predict(self, state_pooled: Tensor, anchor: Tensor) -> Tensor:
        """``e_hat(i,t) = P([state_t ; u_i])`` -> per-row scalar forgetting score.

        ``state_pooled`` is ``[batch, 1536]`` (from ``project_state``) and
        ``anchor`` is ``[batch, 384]`` (or both 1-D, broadcastable). They are
        concatenated along the feature dim and pushed through the Linear.
        Returns ``[batch, 1]`` (or ``[1, 1]`` for 1-D inputs).
        """
        s = state_pooled.to(torch.float32)
        u = anchor.to(torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        x = torch.cat([s, u], dim=1)
        return self.linear(x)

    forward = predict

    # ── load ──

    @classmethod
    def from_state_dict(
        cls,
        sd: dict,
        state_dim_pooled: int = STATE_DIM_POOLED,
        anchor_dim: int = ANCHOR_DIM,
    ) -> "RecoverabilityHead":
        """Build a head and load a raw ``nn.Linear`` state_dict into it.

        The checkpoint stores the Linear's ``state_dict`` directly (under
        ``"linear"``). Strict (a shape mismatch from a different
        state_dim_pooled/anchor_dim is a hard error, not a silent mis-wire).
        """
        head = cls(state_dim_pooled=state_dim_pooled, anchor_dim=anchor_dim)
        missing, unexpected = head.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"recoverability head state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        head.eval()
        return head


def load_recoverability_head(
    path: str,
    device: str = "auto",
    map_location: str = "cpu",
) -> RecoverabilityHead:
    """Load a trained RecoverabilityHead checkpoint -> ready-to-serve module.

    The checkpoint is ``{"linear": state_dict, "state_dim_pooled": int,
    "anchor_dim": int, "ridge_auc": float, "k_auc": float, "go": bool, ...}``
    (see ``recoverability_training.fit_recoverability``). Like the
    latent-dynamics loader this takes NO backbone -- the head reads WM state
    + an anchor at serve. ``state_dim_pooled``/``anchor_dim`` are read before
    construction so a checkpoint fit on a different rep loads into a matching
    Linear. Moves to the resolved device, eval.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        state_dim_pooled = int(ckpt.get("state_dim_pooled", STATE_DIM_POOLED))
        anchor_dim = int(ckpt.get("anchor_dim", ANCHOR_DIM))
        sd = ckpt["linear"] if "linear" in ckpt else ckpt
    else:
        state_dim_pooled, anchor_dim = STATE_DIM_POOLED, ANCHOR_DIM
        sd = ckpt
    head = RecoverabilityHead.from_state_dict(
        sd, state_dim_pooled=state_dim_pooled, anchor_dim=anchor_dim
    )
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return head.to(dev).eval()