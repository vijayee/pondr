"""Focused unit tests for ``ExpandHandler`` (Phase 2c, chunking-level EXPAND).

The orchestrator-level EXPAND scenarios live in ``tests/test_orchestrator.py``
(``test_expand_loads_full_text_and_injects_into_wm`` /
``test_expand_on_primary_raises_not_expandable``); this file exercises the
handler directly so the ``SSMChunker.expand`` + working-memory injection +
``expand_count`` (Presentation Gate outcome signal) contract is pinned without
the orchestrator wiring.

CPU, ReferenceSSM, deterministic hash stub embedder (no sentence_transformers,
no WaveDB). Mirrors the stubs in ``tests/test_ssm_chunker.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest
import torch

from src.config import Phase2cConfig
from src.retrieval.expand_handler import ExpandHandler
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.ssm_chunker import (
    EpisodeNotExpandable,
    SSMChunker,
)
from src.subconscious.working_memory import WorkingMemory


class _StubEmbedder:
    """Deterministic 384-dim hash embedder (shape-only, not semantic)."""
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            buf = bytearray()
            counter = 0
            h = hashlib.sha256(t.encode("utf-8")).digest()
            while len(buf) < self.dim:
                buf += hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
                counter += 1
            vec = [(b / 127.5 - 1.0) for b in buf[: self.dim]]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


@dataclass
class _FakeEpisode:
    summary: str
    full_text: str
    timestamp: str = "2026-01-01"


class _FakeStore:
    def __init__(self, eps: dict[str, _FakeEpisode]) -> None:
        self._eps = eps

    def get_episode(self, eid: str):
        return self._eps.get(eid)


def _plan(primary_chunk_count: int = 5):
    class _P:
        pass
    p = _P()
    p.primary_chunk_count = primary_chunk_count
    return p


def _ep(eid: str, text: str = "x" * 400, summary: str = "summary " + "y" * 40,
        score: float = 1.0) -> dict:
    return {
        "episode_id": eid, "text": text, "summary": summary,
        "timestamp": "2026-01-01",
        "entities": [], "topics": [], "tones": [], "decisions": [],
        "score": score,
    }


def _chunker(max_primary_chunks: int = 3) -> SSMChunker:
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.ssm_chunker.max_primary_chunks = max_primary_chunks
    return SSMChunker(bb, _StubEmbedder(), cfg)


def _working_memory() -> WorkingMemory:
    return WorkingMemory(JGSBackbone(BackboneConfig()), embedder=_StubEmbedder())


# ── handle_expand ──

def test_handle_expand_loads_full_text_and_injects_into_wm():
    """EXPAND a compressed episode -> full text returned, WM state moves,
    expand_count incremented (the Presentation Gate outcome signal)."""
    chunker = _chunker(max_primary_chunks=3)
    eps = [_ep(f"e{i}", text="FULL TEXT " * 200, summary=f"summary e{i}")
           for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    expandable = sorted(ctx.expandable_ids)
    assert expandable, "expected at least one compressed episode"
    target = expandable[0]

    wm = _working_memory()
    # Step the WM once with a dummy query so it has a non-None baseline state
    # (there is no orchestrator query here to do that). handle_expand's inject
    # then moves the state further.
    wm.update(torch.zeros(1, 384))
    handler = ExpandHandler(chunker, wm)
    state_before = [t.clone() for t in wm.state]

    full_text, snap = handler.handle_expand(target, ctx)

    assert "FULL TEXT" in full_text
    # WM absorbed the expanded episode's summary as an injection step.
    assert not all(torch.equal(a, b)
                   for a, b in zip(state_before, wm.state))
    assert snap is not None  # snapshot returned
    assert handler.expand_count == 1
    assert handler.outcome_expand_count == 1


def test_handle_expand_on_primary_raises_not_expandable():
    """EXPAND on a primary chunk (already full text) raises, expand_count
    unchanged (no successful expansion -> no outcome signal)."""
    chunker = _chunker(max_primary_chunks=3)
    eps = [_ep(f"e{i}", text="FULL TEXT " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    primary_ids = [ep["episode_id"] for ep in ctx.primary_chunks]
    assert primary_ids, "expected at least one primary chunk"

    wm = _working_memory()
    handler = ExpandHandler(chunker, wm)

    with pytest.raises(EpisodeNotExpandable):
        handler.handle_expand(primary_ids[0], ctx)
    assert handler.expand_count == 0


def test_handle_expand_falls_back_to_store_when_secondary_set_dropped():
    """When the in-memory secondary set is unavailable, EXPAND loads from the
    store fallback passed at construction (handler._store)."""
    chunker = _chunker(max_primary_chunks=3)
    eps = [_ep(f"e{i}", text="FULL TEXT " * 200, summary=f"summary e{i}")
           for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    ctx.secondary_episodes = []  # simulate the secondary set being dropped
    target = sorted(ctx.expandable_ids)[0]
    store = _FakeStore({target: _FakeEpisode(summary="sum " + target,
                                              full_text="STORE FULL TEXT " + target)})

    wm = _working_memory()
    handler = ExpandHandler(chunker, wm, store=store)
    full_text, _ = handler.handle_expand(target, ctx)
    assert "STORE FULL TEXT" in full_text
    assert handler.expand_count == 1


def test_handle_expand_without_wm_returns_full_text_and_none_snapshot():
    """A handler constructed with ``working_memory=None`` (WM injection
    skipped) still returns the full text; the snapshot is None and
    expand_count still increments (the outcome signal is independent of WM)."""
    chunker = _chunker(max_primary_chunks=3)
    eps = [_ep(f"e{i}", text="FULL TEXT " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    target = sorted(ctx.expandable_ids)[0]

    handler = ExpandHandler(chunker, None)  # no WM
    full_text, snap = handler.handle_expand(target, ctx)
    assert "FULL TEXT" in full_text
    assert snap is None
    assert handler.expand_count == 1