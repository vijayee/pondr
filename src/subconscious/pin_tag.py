"""STRM Phase 4 Step 3: the pin tag — a token-type embedding added to ``u_{t+1}``.

When the salience trigger (Step 4) re-injects a recalled LTM episode, it calls
``WorkingMemory.inject(emb, ..., pin=True)``. A pinned injection ADDS this
module's single 384-d parameter to the input embedding BEFORE the SSM step, so
``W_A`` (the SSM's ``Linear(384 -> 16)`` input projection) sees a tagged input
and retains the pinned episode over the next K steps. ``d_model`` stays 384 —
the pin is an embedding ADDED to ``u_{t+1}``, NOT an extra input feature (a
concat would change ``u``'s width and break ``W_A``). See the plan's de-wonk
note #2 and the architecture proposal §4.5.

v1 ships **default-initialized** (deterministic non-zero). It is a faithful
non-stub: the pin vector is real, additive, and flows through the SSM, so
``pin=True`` measurably changes the step output vs ``pin=False`` (pinned by
``test_pin_adds_embedding_to_input``). Whether LEARNING this vector (a
retention surrogate: does a pinned slot's ``r_i`` stay high over K steps?)
beats the default init is a question for the deferred Step 7 eval; if the eval
NO-GOs at the retention axis, fit it then. See the plan's "Out of scope" note.

The ``u_t`` naming collision: ``u_t`` already denotes the SSM *input vector*
(``strm_traces.py``, ``test_strm_heads.py``). The pin is the input-side
embedding; the per-slot bookkeeping flag is ``RingSlot.pinned: bool``
(``working_memory.py``), never ``u_t``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

# Matches the SSM input width (d_model) and the bge-small embedding dim. The pin
# is added to u_{t+1}, so it MUST share the input width -- a different dim would
# either fail to broadcast or (if concatenated) break W_A's Linear(384 -> 16).
D_MODEL = 384

# Deterministic init bounds. Small (token-type-embedding scale, ~BERT word_emb
# std) and non-zero so pin=True is not a silent no-op. A fixed generator makes
# the default-init reproducible across processes (no Date.now/random here).
_INIT_SEED = 1337
_INIT_LO, INIT_HI = -0.02, 0.02


def _default_init(d_model: int = D_MODEL) -> Tensor:
    """Deterministic non-zero pin vector (fixed generator -> reproducible)."""
    g = torch.Generator().manual_seed(_INIT_SEED)
    pin = torch.empty(d_model, dtype=torch.float32)
    pin.uniform_(_INIT_LO, INIT_HI, generator=g)
    return pin


class PinTag(nn.Module):
    """One ``nn.Parameter([d_model])`` token-type embedding, added to ``u_{t+1}``.

    ``forward(input_embedding)`` returns ``input_embedding + pin`` (broadcasts
    over the batch dim). The pin is added BEFORE the SSM step in
    ``WorkingMemory.step`` (caller's responsibility); this module only owns the
    vector and the add. ``pin=False`` (the default everywhere except the salience
    re-inject) never calls this module, so the off-path is byte-identical.
    """

    def __init__(self, d_model: int = D_MODEL, init: Optional[Tensor] = None) -> None:
        super().__init__()
        if init is not None:
            if init.shape != (d_model,):
                raise ValueError(
                    f"PinTag init shape {tuple(init.shape)} != ({d_model},). "
                    "The pin must share the SSM input width (see D_MODEL)."
                )
            pin = init.detach().to(torch.float32).clone()
        else:
            pin = _default_init(d_model)
        self.pin = nn.Parameter(pin)

    def forward(self, input_embedding: Tensor) -> Tensor:
        """Return ``input_embedding + pin`` (broadcasts ``pin`` over the batch)."""
        return input_embedding + self.pin

    @classmethod
    def from_state_dict(cls, sd: dict, d_model: int = D_MODEL) -> "PinTag":
        """Build a PinTag and load a raw ``pin`` parameter into it (strict)."""
        tag = cls(d_model=d_model)
        missing, unexpected = tag.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"pin tag state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        tag.eval()
        return tag


def load_pin_tag(path: str, device: str = "auto", map_location: str = "cpu") -> PinTag:
    """Load a trained PinTag checkpoint -> ready-to-serve module.

    The checkpoint is ``{"pin": state_dict, "d_model": int, ...}`` (a future
    retention surrogate would write this; v1 has no trained checkpoint, so this
    loader is wired now and exercised by the round-trip test once a surrogate
    runs). Takes NO backbone — the pin is a free parameter. Moves to the
    resolved device, eval.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "pin" in ckpt:
        d_model = int(ckpt.get("d_model", D_MODEL))
        sd = ckpt["pin"]
    else:
        d_model = D_MODEL
        sd = ckpt
    tag = PinTag.from_state_dict(sd, d_model=d_model)
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return tag.to(dev).eval()