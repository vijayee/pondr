"""Phase 1c tests: the mixed-ring trace generator's alignment-critical helpers.

The generator (``scripts/generate_onyx_mixed_ring_traces.py``) emits one
training record per turn whose ring = conversation slots (slot_type=0) +
retrieved doc-episode slots (slot_type=1). Two helpers are alignment-
critical -- a misalignment would silently corrupt the slot-type embedding's
training labels (de-wonk hazard), so they get their own tests:

1. ``_infer_slot_type`` -- the ``RingSlot.slot_type`` field is the source of
   truth; the source_id prefix (``#msg`` -> 0, ``__ep`` -> 1) is the
   fallback; an UNKNOWN prefix MUST raise (never silently default to 0).
2. ``_build_mixed_record`` -- the ``kept`` filter (drop slots with no
   ``h``/no text) and the ``slot_types`` output MUST align with the
   returned ``source_ids`` + ``slots_*`` (same ``kept`` order); the record
   is byte-identical to ``generate_lmsys_serve_traces._build_record`` for
   the shared fields, PLUS ``slot_types`` [K'] long.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_onyx_mixed_ring_traces import (  # noqa: E402
    _build_mixed_record,
    _infer_slot_type,
)
from src.subconscious.latent_dynamics_head import LatentDynamicsHead  # noqa: E402
from src.subconscious.working_memory import RingSlot  # noqa: E402


def _slot(text: str, source_id: str, slot_type: int | None, y_seed: int = 0,
          h=True) -> RingSlot:
    """Build a minimal RingSlot with a 4-layer [1,16,384] fp16 state."""
    g = torch.Generator().manual_seed(y_seed)
    y = torch.randn(1, 256, generator=g)
    h_list = None
    if h:
        h_list = [torch.randn(1, 16, 384, generator=g).to(torch.float16)
                  for _ in range(4)]
    return RingSlot(y, source_id, text, pinned=False, h=h_list,
                    slot_type=slot_type)


# ── _infer_slot_type ──

def test_infer_prefers_recorded_slot_type():
    assert _infer_slot_type("s__ep0001", 1) == 1
    assert _infer_slot_type("s#msg3", 0) == 0


def test_infer_falls_back_to_source_id_prefix():
    # Field None -> prefix rule.
    assert _infer_slot_type("abc#msg4", None) == 0
    assert _infer_slot_type("abc__ep0012", None) == 1


def test_infer_unknown_prefix_raises():
    # An unknown source_id prefix MUST raise -- never silently default to 0
    # (a mis-tagged slot would corrupt the slot-type embedding's labels).
    with pytest.raises(ValueError):
        _infer_slot_type("unknown-prefix-no-marker", None)


def test_infer_empty_source_id_defaults_conversation():
    # No source_id + no field -> 0 (the conservative conversation default).
    assert _infer_slot_type(None, None) == 0
    assert _infer_slot_type("", None) == 0


# ── _build_mixed_record ──

def _build_inputs(n_conv=2, n_ret=2, with_no_text=False, with_no_h=False):
    """Build a parallel (ring, doc_embs, source_ids, slot_types) test bundle."""
    ring: list[RingSlot] = []
    doc_embs: list[torch.Tensor] = []
    source_ids: list[str] = []
    slot_types: list[int | None] = []
    idx = 0
    for _ in range(n_conv):
        sid = f"sess#msg{idx}"
        st = 0
        ring.append(_slot(f"conv text {idx}", sid, st, y_seed=idx,
                          h=False if with_no_h else True))
        doc_embs.append(torch.randn(384))
        source_ids.append(sid)
        slot_types.append(st)
        idx += 1
    for _ in range(n_ret):
        sid = f"sess__ep{idx:04d}"
        st = 1
        ring.append(_slot(f"retrieved episode {idx}", sid, st, y_seed=idx,
                          h=False if with_no_h else True))
        doc_embs.append(torch.randn(384))
        source_ids.append(sid)
        slot_types.append(st)
        idx += 1
    if with_no_text:
        # Append a slot with empty text (should be dropped by the kept filter).
        ring.append(_slot("", "sess#msg99", 0, y_seed=99))
        doc_embs.append(torch.randn(384))
        source_ids.append("sess#msg99")
        slot_types.append(0)
    return ring, doc_embs, source_ids, slot_types


def test_build_returns_none_for_small_ring():
    ring, embs, sids, sts = _build_inputs(n_conv=1, n_ret=1)  # K=2 < 3
    q = torch.randn(384)
    rec = _build_mixed_record(ring, q, LatentDynamicsHead(), embs, sids, sts,
                              False, "q")
    assert rec is None


def test_build_emits_slot_types_aligned_with_source_ids():
    ring, embs, sids, sts = _build_inputs(n_conv=2, n_ret=3)  # K=5
    q = torch.randn(384)
    rec = _build_mixed_record(ring, q, LatentDynamicsHead(), embs, sids, sts,
                              False, "q")
    assert rec is not None
    # slot_types is a long tensor of the SAME length as the kept slots (5).
    assert rec["slot_types"].dtype == torch.long
    assert rec["slot_types"].shape == (5,)
    # source_ids + slot_types align: conv (0) first, then retrieved (1).
    assert list(rec["slot_types"].tolist()) == [0, 0, 1, 1, 1]
    assert rec["source_ids"] == sids
    # Shared fields present + byte-identical format to _build_record.
    assert rec["slots_h_raw"].shape == (5, 4, 16, 384)
    assert rec["slots_h_raw"].dtype == torch.float16
    assert rec["slots_z"].shape == (5, 384)
    assert rec["slots_y"].shape == (5, 256)
    assert rec["slots_doc_emb"].shape == (5, 384)
    assert rec["cos"].shape == (5,)
    assert rec["labels"].shape == (5,)
    # Gold = top-1-cos (exactly one positive label).
    assert int(rec["labels"].sum().item()) == 1
    assert "question" not in rec  # emit_question=False


def test_build_drops_no_text_and_no_h_slots_and_keeps_alignment():
    # A slot with empty text + (separately) would be dropped; here test the
    # no-text drop keeps slot_types aligned with the surviving slots.
    ring, embs, sids, sts = _build_inputs(n_conv=2, n_ret=3, with_no_text=True)
    q = torch.randn(384)
    rec = _build_mixed_record(ring, q, LatentDynamicsHead(), embs, sids, sts,
                              False, "q")
    assert rec is not None
    # The no-text slot (sess#msg99) was dropped -> 5 surviving slots, not 6.
    assert rec["slot_types"].shape == (5,)
    assert "sess#msg99" not in rec["source_ids"]
    # Alignment: each surviving source_id's slot_type matches its prefix.
    for sid, st in zip(rec["source_ids"], rec["slot_types"].tolist()):
        assert (st == 1) == ("__ep" in sid)
        assert (st == 0) == ("#msg" in sid)


def test_build_emit_question_carries_text():
    ring, embs, sids, sts = _build_inputs(n_conv=2, n_ret=2)
    q = torch.randn(384)
    rec = _build_mixed_record(ring, q, LatentDynamicsHead(), embs, sids, sts,
                              True, "the user question")
    assert rec is not None
    assert rec["question"] == "the user question"


def test_build_gold_aligns_with_max_cos():
    # Make slot 2 (the first retrieved) the obvious top-1-cos by giving it an
    # embedding nearly identical to the query.
    ring, embs, sids, sts = _build_inputs(n_conv=2, n_ret=2)
    q = torch.randn(384)
    qn = q / (q.norm() + 1e-9)
    embs[2] = qn * 5.0  # slot 2 dominates cosine
    rec = _build_mixed_record(ring, q, LatentDynamicsHead(), embs, sids, sts,
                              False, "q")
    assert rec is not None
    # The argmax label is at the surviving index of slot 2 (kept index 2).
    assert int(rec["labels"].argmax().item()) == 2