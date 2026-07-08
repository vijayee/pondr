"""Offline tests for the Phase 2c ChunkedContextFormatter + ExpandHandler.

CPU, ReferenceSSM, stub embedder. Verifies the three context sections (primary
full text / compressed topic summary / working-memory preamble), the token
cap, EXPAND full-text loading + WM injection, and the expand-count outcome
signal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest
import torch

from src.config import Phase2cConfig
from src.retrieval.chunked_context import ChunkedContextFormatter
from src.retrieval.expand_handler import ExpandHandler
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.presentation_gate import (
    CHUNKED, DIRECT, PresentationGate, PresentationPlan, SUMMARY_ONLY,
)
from src.subconscious.ssm_chunker import (
    ChunkedContext, EpisodeNotExpandable, EpisodeNotFound, SSMChunker,
)
from src.subconscious.working_memory import WorkingMemory


class _StubEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            buf = bytearray()
            h = hashlib.sha256(t.encode("utf-8")).digest()
            counter = 0
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


def _ep(eid, text="primary text " * 20, summary="sum", topics=None, entities=None,
        tones=None, score=1.0) -> dict:
    return {
        "episode_id": eid, "text": text, "summary": summary,
        "timestamp": "2026-01-01", "entities": entities or [],
        "topics": topics or [], "tones": tones or [],
        "decisions": [], "score": score,
    }


def _chunker(max_primary_tokens=4096, max_primary_chunks=5):
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.ssm_chunker.max_primary_tokens = max_primary_tokens
    cfg.ssm_chunker.max_primary_chunks = max_primary_chunks
    return SSMChunker(bb, _StubEmbedder(), cfg)


def _plan(strategy=CHUNKED, primary_chunk_count=5):
    return PresentationPlan(
        strategy=strategy, primary_chunk_count=primary_chunk_count,
        primary_chunk_size=0, compressed_chunk_count=0, expand_threshold=0.5,
        rationale="test",
    )


# ── formatter ──

def test_format_produces_primary_section():
    chunker = _chunker()
    eps = [_ep("e0", topics=["db", "perf"]), _ep("e1", topics=["db"])]
    ctx = chunker.chunk(eps, _plan(strategy=DIRECT, primary_chunk_count=2))
    out = ChunkedContextFormatter().format_for_llm(ctx)
    assert "[RETRIEVED CONTEXT — PRIMARY]" in out
    assert "e0" in out
    assert "primary text" in out


def test_format_produces_compressed_section_with_topics():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200, topics=[f"topic_{i}", "shared"])
           for i in range(8)]
    ctx = chunker.chunk(eps, _plan(strategy=CHUNKED, primary_chunk_count=2))
    out = ChunkedContextFormatter().format_for_llm(ctx)
    assert "[COMPRESSED CONTEXT — SUMMARY]" in out
    assert "shared" in out  # topic union from secondary episodes
    assert "EXPAND(episode_id)" in out
    # secondary ids e2..e7 are expandable
    for i in range(2, 8):
        assert f"e{i}" in out


def test_format_omits_compressed_section_when_none():
    chunker = _chunker()
    eps = [_ep("e0"), _ep("e1")]
    ctx = chunker.chunk(eps, _plan(strategy=DIRECT, primary_chunk_count=2))
    out = ChunkedContextFormatter().format_for_llm(ctx)
    assert "[COMPRESSED CONTEXT — SUMMARY]" not in out


def test_format_includes_working_memory_preamble():
    chunker = _chunker()
    eps = [_ep("e0", text="word " * 200) for _ in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=2))
    # Build a fake WM snapshot with metadata.
    from src.subconscious.state_serializer import JGSSnapshot
    import torch
    wm = JGSSnapshot(
        state_tensors=[torch.zeros(1, 16, 384) for _ in range(4)],
        input_count=3, timestamp=0.0,
        metadata={"last_query_type": "factual", "active_domains": ["database", "coding"]},
    )
    out = ChunkedContextFormatter().format_for_llm(ctx, working_memory=wm)
    assert "[WORKING MEMORY STATE]" in out
    assert "factual" in out
    assert "database" in out


def test_format_compressed_section_uses_topics_not_state_vector():
    """The compressed section is TEXT (topic union), never the raw SSM tensor."""
    chunker = _chunker()
    eps = [_ep("e0", text="word " * 200, topics=["alpha", "beta"])]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=0))  # all compressed
    out = ChunkedContextFormatter().format_for_llm(ctx)
    # The tensor bytes never appear; only topic names.
    assert "alpha" in out and "beta" in out
    assert "tensor" not in out.lower()


def test_format_respects_token_cap():
    chunker = _chunker(max_primary_tokens=100000, max_primary_chunks=20)
    eps = [_ep(f"e{i}", text="word " * 100) for i in range(20)]  # ~125 tokens each
    ctx = chunker.chunk(eps, _plan(strategy=DIRECT, primary_chunk_count=20))
    out = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=300)
    # Hard cap at 300 tokens → only ~2 primary episodes fit.
    assert "e0" in out  # at least the first fits
    assert "e19" not in out  # the last is dropped (not truncated)


# ── ExpandHandler ──

def _wm(embedder=_StubEmbedder()) -> WorkingMemory:
    bb = JGSBackbone(BackboneConfig())
    return WorkingMemory(bb, embedder=embedder)


def test_expand_loads_full_text_and_injects_into_wm():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    wm = _wm()
    wm.reset()
    state_before = [t.clone() for t in wm.state]
    handler = ExpandHandler(chunker, wm)
    full_text, snap = handler.handle_expand("e5", ctx)
    assert "word" in full_text
    # WM state moved (the expanded episode was injected as a step).
    assert not any(torch.equal(a, b) for a, b in zip(state_before, wm.state))
    assert handler.expand_count == 1


def test_expand_on_primary_raises_not_expandable():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    handler = ExpandHandler(chunker, _wm())
    with pytest.raises(EpisodeNotExpandable):
        handler.handle_expand("e0", ctx)


def test_expand_unknown_raises_not_found():
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    handler = ExpandHandler(chunker, _wm())
    with pytest.raises(EpisodeNotFound):
        handler.handle_expand("nope", ctx)


def test_expand_resolves_secondary_from_store_when_not_in_memory():
    """If the secondary set was dropped (e.g. reloaded context), fall back to store."""
    chunker = _chunker()
    eps = [_ep(f"e{i}", text="word " * 200) for i in range(8)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=3))
    # Simulate the secondary set being unavailable by emptying it.
    ctx.secondary_episodes = []
    store = _FakeStore({"e5": _FakeEpisode(summary="sum e5", full_text="STORE TEXT e5")})
    handler = ExpandHandler(chunker, _wm(), store=store)
    full_text, _ = handler.handle_expand("e5", ctx)
    assert full_text == "STORE TEXT e5"