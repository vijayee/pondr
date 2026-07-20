"""Shared r_i scoring for the WM ring (STRM Phase 2a relevance head read-out).

Factored out of ``orchestrator._write_graduation_replay`` (Phase 2d) so the
Phase 3 context-builder and the 2d graduation logger share ONE r_i code path.
The graduation logger must stay byte-identical to its pre-Phase-3 behavior, so
this helper mirrors the original inline loop verbatim (re-embed each slot's
text via ``working_memory.embed``, move every operand to the relevance head's
device, ``relevance_head.predict`` -> ``r``); the only change is WHERE the
code lives.

Two forms:

* ``score_ring_slots`` -- returns ``(slots, r_is)``. Used by the 2d graduation
  logger (it writes one JSONL record per slot and ignores the doc embeddings).
  Byte-identical to the pre-refactor inline code.
* ``score_ring_slots_with_doc_embs`` -- returns ``(slots, r_is, doc_embs)``.
  Used by the Phase 3 context-builder path, which needs the re-embedded
  ``slot_doc_emb`` (384-d) the relevance head consumed, so its ``W_doc`` path
  is active. The graduation logger does NOT call this form.

Slots with no text (e.g. the raw query step, None-provenance recalls), or when
no relevance head / no embedder is wired, get ``r_is[i] = None`` -- matching
the original behavior exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .working_memory import WorkingMemory


def _score(
    working_memory: "WorkingMemory",
    relevance_head,
    embedder,
    prompt_emb: Tensor,
    slots: list,
) -> tuple[list[Optional[float]], list]:
    """The shared r_i loop. Returns ``(r_is, doc_embs)`` where BOTH are
    length-``len(slots)`` lists aligned to slot position: ``r_is[i]`` is slot
    ``i``'s relevance (or ``None`` if it wasn't scored), and ``doc_embs[i]`` is
    slot ``i``'s re-embedded doc vector (or ``None``). Mirrors the
    orchestrator's pre-refactor lines 706-734 byte-for-byte for the r_i values
    themselves; the doc-embedding bookkeeping is added for the builder path
    (the graduation logger ignores ``doc_embs``).
    """
    r_is: list[Optional[float]] = [None] * len(slots)
    doc_embs: list = [None] * len(slots)
    if relevance_head is None or embedder is None:
        return r_is, doc_embs
    idx_text = [(i, s.text) for i, s in enumerate(slots)
                if s.text is not None and str(s.text).strip()]
    if not idx_text:
        return r_is, doc_embs
    # The relevance head may live on a different device than the slot readouts
    # (CUDA backbone) vs the bge embedder (CPU) vs the query embedding (CPU).
    # Move every operand to the HEAD's device before the fused predict -- a
    # mixed-device addmm crashes the whole logger (and thus the turn). ``.to``
    # on an already-right-device tensor is a no-op, so this is free when they
    # already agree.
    head_dev = next(relevance_head.parameters()).device
    doc_emb_tensors = working_memory.embed([t for _, t in idx_text])  # [K',384] each
    ys = torch.cat(
        [slots[i].y.to(torch.float32).squeeze(0).reshape(1, -1)
         for i, _ in idx_text], dim=0).to(head_dev)   # [K', 256]
    ds = torch.cat(
        [e.to(torch.float32).squeeze(0).reshape(1, -1)
         for e in doc_emb_tensors], dim=0).to(head_dev)  # [K', 384]
    q = prompt_emb.to(torch.float32).squeeze(0).reshape(1, -1).to(head_dev)  # [1, 384]
    with torch.no_grad():
        r = relevance_head.predict(ys, ds, q)    # [K', 1]
    for j, (i, _) in enumerate(idx_text):
        r_is[i] = float(r[j].item())
        doc_embs[i] = doc_emb_tensors[j]
    return r_is, doc_embs


def score_ring_slots(
    working_memory: "WorkingMemory",
    relevance_head,
    embedder,
    prompt_emb: Tensor,
    slots: Optional[list] = None,
) -> tuple[list, list[Optional[float]]]:
    """Compute ``(slots, r_is)`` for the current WM ring.

    Mirrors ``orchestrator._write_graduation_replay``'s r_i loop
    byte-for-byte. Pass ``slots=working_memory.ring_buffer()`` explicitly to
    snapshot once and reuse; if ``None``, calls ``ring_buffer()`` inside (the
    graduation logger's existing behavior).
    """
    if slots is None:
        slots = working_memory.ring_buffer()
    r_is, _doc_embs = _score(working_memory, relevance_head, embedder,
                             prompt_emb, slots)
    return slots, r_is


def score_ring_slots_with_doc_embs(
    working_memory: "WorkingMemory",
    relevance_head,
    embedder,
    prompt_emb: Tensor,
    slots: Optional[list] = None,
) -> tuple[list, list[Optional[float]], list]:
    """Compute ``(slots, r_is, doc_embs)`` for the current WM ring.

    Same r_i loop as ``score_ring_slots`` (byte-identical r_is), but also
    returns the re-embedded doc vectors (384-d) the Phase 3 context-builder
    fuses via ``W_doc``. ``doc_embs`` is slot-aligned: length ``len(slots)``
    with ``None`` at the same positions ``r_is`` has ``None`` (slots with no
    text / no head / no embedder). The caller stacks the non-None entries in
    slot order -- NOT in ``idx_text`` order -- so the builder's ``W_doc`` path
    scores the SAME vectors the frozen relevance head consumed.
    """
    if slots is None:
        slots = working_memory.ring_buffer()
    r_is, doc_embs = _score(working_memory, relevance_head, embedder,
                            prompt_emb, slots)
    return slots, r_is, doc_embs