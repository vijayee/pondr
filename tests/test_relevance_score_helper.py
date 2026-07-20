"""STRM Phase 3 Step 1: ``relevance_score`` helper byte-identical regression.

The r_i scoring loop was factored out of ``orchestrator._write_graduation_replay``
into ``src.subconscious.relevance_score.score_ring_slots`` so the Phase 3
context-builder can share the SAME r_i code path as the 2d graduation logger.
The graduation logger's ``replay.jsonl`` must not change a single byte, so this
test runs a verbatim copy of the PRE-refactor inline loop and the new helper on
the same ``(slots, prompt_emb, working_memory, relevance_head, embedder)`` and
asserts the resulting ``r_is`` are equal to 7 decimals.

It also covers the second form ``score_ring_slots_with_doc_embs`` (the builder
path): the ``r_is`` must match the first form exactly, and the returned
``doc_embs`` must be the slot-aligned re-embedded doc vectors the helper used
internally (so the builder's ``W_doc`` path scores the SAME vectors the frozen
relevance head saw).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.relevance_score import (
    score_ring_slots,
    score_ring_slots_with_doc_embs,
)
from src.subconscious.relevance_head import RelevanceHead, SLOT_DIM, DOC_DIM, QUERY_DIM
from src.subconscious.working_memory import RingSlot


# ── harness ──────────────────────────────────────────────────────────────────

class _MockWM:
    """Minimal stand-in for WorkingMemory: only ``embed`` is exercised by the
    helper. Returns one ``[1, 384]`` tensor per text. Real ``WorkingMemory.embed``
    is a PURE function of the text (the bge embedder is stateless), so the mock
    is too: the same text yields the same vector on every call. A stateful RNG
    would break the byte-identical comparison, since the inline copy and the
    helper each call ``embed`` once and must see the SAME vectors. The helper's
    device moves (``.to(head_dev)``) are no-ops on CPU, so this is sufficient."""

    def __init__(self, dim: int = DOC_DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[Tensor]:
        out: list[Tensor] = []
        for t in texts:
            # deterministic-per-text: SHA256 stretch -> normalize, mirroring
            # tests.test_orchestrator._StubEmbedder.encode exactly (so the mock
            # honors the same "embed is pure" contract the real embedder does).
            import hashlib
            buf = bytearray()
            h = hashlib.sha256(t.encode("utf-8")).digest()
            counter = 0
            while len(buf) < self.dim:
                buf += hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
                counter += 1
            vec = [(b / 127.5 - 1.0) for b in buf[: self.dim]]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append(
                torch.tensor([v / norm for v in vec], dtype=torch.float32).unsqueeze(0)
            )   # [1, 384]
        return out


def _slot(y: Tensor, source_id: Optional[str], text: Optional[str]) -> RingSlot:
    return RingSlot(y=y.clone(), source_id=source_id, text=text)


def _build_slots() -> list[RingSlot]:
    """A small ring mirroring real provenance: one raw query slot (None
    source_id / no text), one empty-text slot, and three scored slots with
    text. The helper must skip the no-text slots (r_is = None) and score the
    rest."""
    torch.manual_seed(123)
    return [
        _slot(torch.randn(1, SLOT_DIM), None, None),                 # raw query step
        _slot(torch.randn(1, SLOT_DIM), "ep_001", "   "),             # blank text
        _slot(torch.randn(1, SLOT_DIM), "ep_002", "Alice said use Postgres"),
        _slot(torch.randn(1, SLOT_DIM), "ep_003", "The deploy is Tuesday"),
        _slot(torch.randn(1, SLOT_DIM), "ep_004", "Bob likes Python"),
    ]


def _inline_pre_refactor(wm, relevance_head, embedder, prompt_emb, slots):
    """A VERBATIM copy of the pre-Phase-3 inline r_i loop that lived in
    ``orchestrator._write_graduation_replay`` (lines ~706-734). Frozen here so
    any drift in ``relevance_score._score`` is caught by direct comparison."""
    r_is: list[Optional[float]] = [None] * len(slots)
    if relevance_head is None or embedder is None:
        return r_is
    idx_text = [(i, s.text) for i, s in enumerate(slots)
                if s.text is not None and str(s.text).strip()]
    if not idx_text:
        return r_is
    head_dev = next(relevance_head.parameters()).device
    doc_emb_tensors = wm.embed([t for _, t in idx_text])   # [K',384] each
    ys = torch.cat(
        [slots[i].y.to(torch.float32).squeeze(0).reshape(1, -1)
         for i, _ in idx_text], dim=0).to(head_dev)        # [K', 256]
    ds = torch.cat(
        [e.to(torch.float32).squeeze(0).reshape(1, -1)
         for e in doc_emb_tensors], dim=0).to(head_dev)    # [K', 384]
    q = prompt_emb.to(torch.float32).squeeze(0).reshape(1, -1).to(head_dev)  # [1, 384]
    with torch.no_grad():
        r = relevance_head.predict(ys, ds, q)              # [K', 1]
    for j, (i, _) in enumerate(idx_text):
        r_is[i] = float(r[j].item())
    return r_is


def _make_head() -> RelevanceHead:
    torch.manual_seed(7)
    return RelevanceHead()   # default dims 256/384/384/128


# ── tests ────────────────────────────────────────────────────────────────────

def test_byte_identical_to_inline():
    """The helper's ``r_is`` equal the pre-refactor inline loop's ``r_is`` to
    7 decimals on the same inputs."""
    wm = _MockWM()
    head = _make_head()
    prompt_emb = torch.randn(1, QUERY_DIM, dtype=torch.float32)
    slots = _build_slots()

    golden = _inline_pre_refactor(wm, head, wm, prompt_emb, slots)
    _, got = score_ring_slots(wm, head, wm, prompt_emb, slots=slots)

    assert len(got) == len(golden) == len(slots)
    # the no-text slots (0 and 1) are unscored in both
    assert golden[0] is None and got[0] is None
    assert golden[1] is None and got[1] is None
    # the scored slots match to 7 decimals
    for i in (2, 3, 4):
        assert golden[i] is not None
        assert got[i] is not None
        assert abs(golden[i] - got[i]) < 1e-7, f"slot {i} drifted: {golden[i]} vs {got[i]}"
        # r_i is a sigmoid -> in [0, 1]
        assert 0.0 <= got[i] <= 1.0


def test_byte_identical_when_slots_snapshot_passed():
    """Passing ``slots=`` explicitly (the graduation logger's behavior) is the
    same as letting the helper snapshot inside -- but the helper must NOT
    re-fetch from a (mock) ring_buffer when slots are given. Verify the explicit
    path matches the inline copy too (the orchestrator call passes slots)."""
    wm = _MockWM()
    head = _make_head()
    prompt_emb = torch.randn(1, QUERY_DIM, dtype=torch.float32)
    slots = _build_slots()

    golden = _inline_pre_refactor(wm, head, wm, prompt_emb, slots)
    _, got = score_ring_slots(wm, head, wm, prompt_emb, slots=slots)
    for g, v in zip(golden, got):
        if g is None:
            assert v is None
        else:
            assert abs(g - v) < 1e-7


def test_no_head_returns_all_none():
    """No relevance head -> all r_is are None (the graduation logger writes
    r_i: null). Byte-identical to the inline copy."""
    wm = _MockWM()
    prompt_emb = torch.randn(1, QUERY_DIM, dtype=torch.float32)
    slots = _build_slots()
    golden = _inline_pre_refactor(wm, None, wm, prompt_emb, slots)
    _, got = score_ring_slots(wm, None, wm, prompt_emb, slots=slots)
    assert all(v is None for v in got)
    assert golden == got


def test_no_embedder_returns_all_none():
    wm = _MockWM()
    head = _make_head()
    prompt_emb = torch.randn(1, QUERY_DIM, dtype=torch.float32)
    slots = _build_slots()
    golden = _inline_pre_refactor(wm, head, None, prompt_emb, slots)
    _, got = score_ring_slots(wm, head, None, prompt_emb, slots=slots)
    assert all(v is None for v in got)
    assert golden == got


def test_with_doc_embs_matches_first_form_and_aligns():
    """The builder path: ``score_ring_slots_with_doc_embs`` returns r_is that
    match ``score_ring_slots`` byte-for-byte, plus slot-aligned doc embeddings
    that are the SAME vectors the frozen relevance head consumed (so the
    builder's W_doc path scores identical inputs)."""
    wm = _MockWM()
    head = _make_head()
    prompt_emb = torch.randn(1, QUERY_DIM, dtype=torch.float32)
    slots = _build_slots()

    _, r_ref = score_ring_slots(wm, head, wm, prompt_emb, slots=slots)
    _, r_full, doc_embs = score_ring_slots_with_doc_embs(
        wm, head, wm, prompt_emb, slots=slots)

    # r_is identical to the first form
    assert len(r_full) == len(r_ref)
    for a, b in zip(r_ref, r_full):
        if a is None:
            assert b is None
        else:
            assert abs(a - b) < 1e-7

    # doc_embs slot-aligned: length == len(slots); None where unscored;
    # [1, DOC_DIM] tensor where scored.
    assert len(doc_embs) == len(slots)
    assert doc_embs[0] is None        # no text
    assert doc_embs[1] is None        # blank text
    for i in (2, 3, 4):
        assert doc_embs[i] is not None
        assert doc_embs[i].shape == (1, DOC_DIM)

    # the doc_embs the helper returned are the SAME vectors it fed the head: re-
    # embed the scored texts through the SAME wm and compare (the helper must
    # not re-embed a second time with different RNG state). Because _MockWM is
    # seeded fresh per construction, rebuild a wm, re-embed in idx_text order,
    # and compare to the scored doc_embs in slot order.
    wm2 = _MockWM()                   # same seed -> same sequence
    scored_idx = [i for i, s in enumerate(slots)
                  if s.text is not None and str(s.text).strip()]
    fresh = wm2.embed([slots[i].text for i in scored_idx])
    for j, i in enumerate(scored_idx):
        assert torch.allclose(doc_embs[i], fresh[j], atol=1e-7), \
            f"doc_emb slot {i} drifted from the vector fed the head"


def test_doc_embs_not_in_idx_order_when_slots_skipped():
    """Regression guard for the slot-alignment bug: doc_embs must be indexed by
    SLOT position (with None holes for skipped slots), NOT by idx_text position.
    A naive return of ``doc_emb_tensors`` in idx_text order would misalign with
    ``r_is`` for the builder. Build a ring where slot 0 is scored and slot 4 is
    blank -- the doc_emb list must have None at position 4, not at 0."""
    torch.manual_seed(99)
    slots = [
        _slot(torch.randn(1, SLOT_DIM), "ep_a", "first scored"),    # scored
        _slot(torch.randn(1, SLOT_DIM), "ep_b", "second scored"),    # scored
        _slot(torch.randn(1, SLOT_DIM), "ep_c", "   "),             # blank
        _slot(torch.randn(1, SLOT_DIM), "ep_d", "third scored"),    # scored
        _slot(torch.randn(1, SLOT_DIM), None, None),                # no text
    ]
    wm = _MockWM()
    head = _make_head()
    prompt_emb = torch.randn(1, QUERY_DIM, dtype=torch.float32)
    _, r_is, doc_embs = score_ring_slots_with_doc_embs(
        wm, head, wm, prompt_emb, slots=slots)
    # alignment: r_is and doc_embs holes coincide
    for i, (r, d) in enumerate(zip(r_is, doc_embs)):
        assert (r is None) == (d is None), f"slot {i}: r_is/doc_emb misaligned"
    # the scored slots are 0, 1, 3 (NOT 0,1,2) -- a idx_text-ordered return would
    # put a tensor at position 2 and None at position 4, which this catches.
    assert doc_embs[0] is not None and doc_embs[1] is not None
    assert doc_embs[2] is None
    assert doc_embs[3] is not None
    assert doc_embs[4] is None