"""Serialize/deserialize a JGS instance's recurrent state to a portable blob.

Phase 2c plumbing. Working Memory (and any other ``JGSInstance`` whose state
should survive across queries/sessions) keeps its continuous state in
``JGSInstance.state``: a list of per-layer tensors ``[batch, d_state, d_model]``
(detached after each step — no BPTT). For the configured backbone that is four
``[1, 16, 384]`` float32 tensors (~24,576 floats, ~96 KB).

This module is the **mechanism** only. It does NOT decide when to save, what to
key by, or how to tie saves to session boundaries — that policy lives in the
orchestrator/session layer and is intentionally left out here so the serializer
stays policy-agnostic and unit-testable in isolation.

Format: a single JSON object (NUL-free ASCII text, safe as a WaveDB text value
— mirrors the ``content/ep/{id}/embedding`` JSON pattern in ``store.py`` and
avoids the historical raw-bytes NUL/empty-value pitfalls):

    {
      "v": 1,
      "dtype": "float32",
      "shapes": [[1,16,384], ...],          # one per layer
      "tensors": "<base64 of concatenated raw float32 bytes>",
      "input_count": int,
      "timestamp": float,
      "metadata": { ... }                     # arbitrary JSON-safe dict
    }

Round-trip is element-exact: tensors are stored as raw little-endian float32
bytes (base64 for text safety) and rebuilt with ``torch.from_numpy``. No
precision loss, no randomness, no dependence on the SSM backend.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

# Only float32 is stored. The Phase 2a backbone trains in float32; the bf16
# autocast path (pretrain.py:206) uses the modern torch.amp.autocast API and is
# fixed, but the serialized recurrent state is still widened to float32 for
# stable cross-dtype round-trips. A non-floating-point state (int/bool) raises
# — it's a semantic type change, not a precision loss. Other floating-point
# dtypes (float16/bfloat16) are losslessly widened to float32.
_SUPPORTED_DTYPE = torch.float32
_FORMAT_VERSION = 1


@dataclass
class JGSSnapshot:
    """A serializable snapshot of a JGS instance's recurrent state + bookkeeping.

    ``state_tensors`` are detached clones (caller-independent of the live
    instance). ``input_count`` / ``timestamp`` / ``metadata`` are opaque to the
    serializer — the caller defines their meaning (e.g. number of absorbed
    queries, wall-clock time of the last update, active domains).
    """

    state_tensors: list[Tensor]
    input_count: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def snapshot_from_instance(
    instance: Any,
    input_count: int = 0,
    timestamp: float = 0.0,
    metadata: Optional[dict[str, Any]] = None,
) -> JGSSnapshot:
    """Capture a detached snapshot of ``instance.state``.

    The instance must have been stepped at least once (or ``reset_state`` called)
    so ``instance.state`` is not ``None``. The returned tensors are clones —
    mutating them or the live instance afterward does not affect the other.
    """
    state = getattr(instance, "state", None)
    if state is None:
        raise ValueError(
            "instance.state is None — call reset_state() or step() before "
            "snapshotting so the per-layer shapes/device/dtype are known."
        )
    return JGSSnapshot(
        state_tensors=[t.detach().cpu().contiguous().clone() for t in state],
        input_count=input_count,
        timestamp=timestamp,
        metadata=dict(metadata) if metadata else {},
    )


def restore_to_instance(
    instance: Any,
    snapshot: JGSSnapshot,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """Load ``snapshot`` into ``instance.state``.

    Restored tensors are moved to ``device``/``dtype`` (default: the instance's
    current parameter device/dtype, or CPU/float32 if the instance has no
    parameters yet). **Caveat:** ``JGSInstance._ensure_state`` re-initializes the
    state to zeros if ``self.state[0].device != input_embedding.device`` on the
    next step. To keep the restored state live, pass the same ``device`` the
    instance will step on, or move the restored tensors there before stepping.
    For the offline plumbing/tests this is moot (everything is CPU).
    """
    if not snapshot.state_tensors:
        raise ValueError("cannot restore an empty snapshot (no state tensors)")

    if device is None:
        try:
            device = next(instance.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")
    if dtype is None:
        try:
            dtype = next(instance.parameters()).dtype
        except (StopIteration, AttributeError):
            dtype = _SUPPORTED_DTYPE

    instance.state = [
        t.to(device=device, dtype=dtype).contiguous()
        for t in snapshot.state_tensors
    ]


def serialize(snapshot: JGSSnapshot) -> str:
    """Serialize a ``JGSSnapshot`` to a NUL-free ASCII text blob.

    Returns a JSON string (no embedded NUL bytes — safe for WaveDB's text value
    path). Round-trips element-exactly through :func:`deserialize`.
    """
    tensors = snapshot.state_tensors
    if not tensors:
        raise ValueError("cannot serialize a snapshot with no state tensors")

    shapes: list[list[int]] = []
    parts: list[bytes] = []
    for t in tensors:
        tt = t.detach().cpu().contiguous()
        if not tt.is_floating_point():
            # An int/bool/uint state is a semantic type change, not a precision
            # loss — reject so a wrong-dtype instance is never silently coerced.
            raise TypeError(f"JGS state tensor must be floating-point, got {tt.dtype}")
        # Widening cast to float32 (lossless for float16/bfloat16; float32 is a
        # no-op). The blob always stores float32 so the round-trip dtype is fixed.
        tt = tt.to(_SUPPORTED_DTYPE)
        shapes.append(list(tt.shape))
        parts.append(np.ascontiguousarray(tt.numpy(), dtype=np.float32).tobytes())

    payload = b"".join(parts)
    blob = json.dumps(
        {
            "v": _FORMAT_VERSION,
            "dtype": "float32",
            "shapes": shapes,
            "tensors": base64.b64encode(payload).decode("ascii"),
            "input_count": int(snapshot.input_count),
            "timestamp": float(snapshot.timestamp),
            "metadata": snapshot.metadata,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    if "\x00" in blob:
        # Defensive — JSON + base64 never produces NUL; if this ever fires the
        # WaveDB value would be silently truncated. Fail loudly instead.
        raise ValueError("serialized blob contains a NUL byte — format invariant violated")
    return blob


def deserialize(blob: str) -> JGSSnapshot:
    """Inverse of :func:`serialize`. Element-exact round-trip."""
    if not blob:
        raise ValueError("cannot deserialize an empty blob")
    if "\x00" in blob:
        raise ValueError("blob contains a NUL byte — corrupted or wrong format")

    obj = json.loads(blob)
    version = obj.get("v")
    if version != _FORMAT_VERSION:
        raise ValueError(f"unsupported JGS state blob version: {version!r} (want {_FORMAT_VERSION})")
    if obj.get("dtype") != "float32":
        raise ValueError(f"unsupported dtype in blob: {obj.get('dtype')!r}")

    shapes = obj["shapes"]
    raw = base64.b64decode(obj["tensors"])
    # np.frombuffer returns a read-only view of the bytes; copy so the tensors
    # are writable and own their storage (torch.from_numpy needs a writable
    # ndarray, and we don't want to alias the base64-decoded buffer).
    flat = np.frombuffer(raw, dtype=np.float32).copy()

    expected = sum(int(np.prod(s)) for s in shapes)
    if flat.size != expected:
        raise ValueError(
            f"state tensor element count mismatch: blob has {flat.size} "
            f"floats but shapes imply {expected}"
        )

    tensors: list[Tensor] = []
    offset = 0
    for s in shapes:
        n = int(np.prod(s))
        chunk = flat[offset:offset + n].reshape(s)
        tensors.append(torch.from_numpy(chunk.copy()).contiguous())
        offset += n

    return JGSSnapshot(
        state_tensors=tensors,
        input_count=int(obj.get("input_count", 0)),
        timestamp=float(obj.get("timestamp", 0.0)),
        metadata=dict(obj.get("metadata", {}) or {}),
    )