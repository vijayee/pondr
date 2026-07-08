"""Offline tests for the Phase 2c SSM Chunker.

All CPU, ReferenceSSM, with a deterministic hash stub embedder (no
sentence_transformers, no WaveDB). A tiny fake store provides full text for
EXPAND. Verifies the primary/compressed split, the token budget, the gist
state shape, and the three EXPAND cases.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from src.config import Phase2cConfig
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.ssm_chunker import (
    ChunkedContext,
    EpisodeNotExpandable,
    EpisodeNotFound,
    SSMChunker,
)
from src.subconscious.working_memory import WorkingMemoryState


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
            vec = [(b / 127.5 - 1.0) for b in buf[:self.dim]]
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
    """Minimal PresentationPlan stand-in — only primary_chunk_count is read."""
    class _P:
        pass
    p = _P()
    p.primary_chunk_count = primary_chunk_count
    return p


def _ep(eid: str, text: str = "x" * 400, summary: str = "summary " + "y" * 40, score: float = 1.0) -> dict:
    return {
        "episode_id": eid,
        "text": text,
        "summary": summary,
        "timestamp": "2026-01-01",
        "entities": [], "topics": [], "tones": [], "decisions": [],
        "score": score,
    }


def _chunker(max_primary_tokens: int = 4096, max_primary_chunks: int = 5) -> SSMChunker:
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.ssm_chunker.max_primary_tokens = max_primary_tokens
    cfg.ssm_chunker.max_primary_chunks = max_primary_chunks
    return SSMChunker(bb, _StubEmbedder(), cfg)


# ── primary/compressed split ──

def test_direct_small_set_all_primary():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="abcd" * 10) for i in range(2)]  # ~40 tokens each
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=5))
    assert len(ctx.primary_chunks) == 2
    assert ctx.compressed_episode_count == 0
    assert ctx.compressed_state is None
    assert ctx.expandable_ids == set()
    assert ctx.total_episodes == 2


def test_chunked_some_primary_some_compressed():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(12)]  # ~100 tokens each
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=5))
    assert len(ctx.primary_chunks) == 5
    assert ctx.compressed_episode_count == 7
    assert ctx.compressed_state is not None
    assert ctx.expandable_ids == {f"e{i}" for i in range(5, 12)}
    assert ctx.total_episodes == 12


def test_token_budget_caps_primary_below_chunk_count():
    # max_primary_tokens=300, each ep ~100 tokens → at most 3 primary even though
    # primary_chunk_count=5.
    chunker = _chunker(max_primary_tokens=300, max_primary_chunks=5)
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=5))
    assert len(ctx.primary_chunks) <= 5
    assert ctx.primary_token_count <= 300
    assert ctx.compressed_episode_count == 8 - len(ctx.primary_chunks)


def test_chunk_cap_binds_when_smaller_than_plan():
    chunker = _chunker(max_primary_chunks=3, max_primary_tokens=100000)
    eps = [_ep(f"e{i}", text="x") for i in range(10)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=5))
    assert len(ctx.primary_chunks) == 3
    assert ctx.compressed_episode_count == 7


# ── compressed state shape ──

def test_compressed_state_shape_is_4x_1_16_384():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="x" * 400) for i in range(7)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=2))
    assert ctx.compressed_state is not None
    assert isinstance(ctx.compressed_state, WorkingMemoryState)
    assert len(ctx.compressed_state.state_tensors) == 4
    for t in ctx.compressed_state.state_tensors:
        assert t.shape == (1, 16, 384)


def test_compressed_state_metadata_records_ids():
    chunker = _chunker()
    eps = [_ep(f"e{i}") for i in range(7)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=2))
    assert ctx.compressed_state.metadata["compressed_episode_ids"] == [
        "e2", "e3", "e4", "e5", "e6"
    ]


def test_chunk_map_distinguishes_primary_and_compressed():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    for i in range(3):
        assert ctx.chunk_map[f"e{i}"] == i       # primary → index
    for i in range(3, 8):
        assert ctx.chunk_map[f"e{i}"] == -1      # compressed


def test_empty_episodes():
    chunker = _chunker()
    ctx = chunker.chunk([], _plan(primary_chunk_count=5))
    assert ctx.primary_chunks == []
    assert ctx.compressed_state is None
    assert ctx.total_episodes == 0


# ── EXPAND ──

def test_expand_compressed_loads_full_text():
    """EXPAND resolves a compressed episode from the in-memory secondary set
    first (the full text is retained in the ChunkedContext)."""
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    loaded = chunker.expand("e5", ctx, store=None)  # no store needed — in-memory
    assert loaded["episode_id"] == "e5"
    assert "word" in loaded["text"]


def test_expand_compressed_falls_back_to_store_when_not_in_memory():
    """When the secondary set has been dropped, EXPAND loads from the store."""
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    ctx.secondary_episodes = []  # simulate the secondary set being unavailable
    store = _FakeStore({"e5": _FakeEpisode(summary="sum e5", full_text="FULL TEXT e5")})
    loaded = chunker.expand("e5", ctx, store=store)
    assert loaded["episode_id"] == "e5"
    assert loaded["text"] == "FULL TEXT e5"


def test_expand_primary_raises_not_expandable():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    with pytest.raises(EpisodeNotExpandable):
        chunker.expand("e0", ctx, store=_FakeStore({}))


def test_expand_unknown_raises_not_found():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    with pytest.raises(EpisodeNotFound):
        chunker.expand("nope", ctx, store=_FakeStore({}))


def test_expand_compressed_without_store_raises():
    """When the secondary set is empty AND no store is given, EXPAND raises."""
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    ctx.secondary_episodes = []  # not in memory → needs the store
    with pytest.raises(RuntimeError, match="store"):
        chunker.expand("e5", ctx, store=None)


# ── compressor isolation ──

def test_compress_is_fresh_per_call():
    """Two chunk() calls must not alias compressed state (ephemeral compressor)."""
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx1 = chunker.chunk(eps, _plan(primary_chunk_count=3))
    ctx2 = chunker.chunk(eps, _plan(primary_chunk_count=3))
    # Same inputs → same gist (compressor is reset each call, deterministic).
    for a, b in zip(ctx1.compressed_state.state_tensors, ctx2.compressed_state.state_tensors):
        import torch
        assert torch.equal(a, b)


# ── latency ──

def test_chunk_50_episodes_under_300ms():
    import time as _t
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 50) for i in range(50)]
    start = _t.perf_counter()
    chunker.chunk(eps, _plan(primary_chunk_count=5))
    elapsed_ms = (_t.perf_counter() - start) * 1000
    # CPU torch per-step is slower than the doc's imagined numpy <50ms; the
    # corrected realistic bound is <300ms for 50 episodes (docs/Phase 2c.md §9.2).
    # ~45 compress steps + 5 primary overhead.
    assert elapsed_ms < 300.0, f"chunk 50 eps took {elapsed_ms:.1f}ms"