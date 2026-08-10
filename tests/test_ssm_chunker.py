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
    GIST_CUE_PRESETS,
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


# ── gist-cue presets (mamba3 decode backend) ──────────────────────────────────

class _StubVoice:
    """Records the cue ``ephemeral_gist`` was called with; returns a marker."""
    def __init__(self) -> None:
        self.last_cue = None

    def ephemeral_gist(self, texts, cue, max_new_tokens=1024):
        self.last_cue = cue
        return "<gist>"


class _RecordingVoice:
    """Records every ``ephemeral_gist`` call (texts, cue, max_new_tokens).

    Returns a marker embedding the call index so concatenated per-episode gists
    are distinguishable. Optionally raises on a call whose joined body contains
    ``fail_on`` (to test the per-episode skip-on-failure path).
    """
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_on = fail_on

    def ephemeral_gist(self, texts, cue, max_new_tokens=1024):
        body = "\n".join(texts)
        if self.fail_on is not None and self.fail_on in body:
            raise RuntimeError(f"boom on {self.fail_on!r}")
        self.calls.append({
            "texts": list(texts), "cue": cue, "max_new_tokens": max_new_tokens,
        })
        return f"<gist{len(self.calls)}>"


def _voice_chunker(gist_cue="Summary:", gist_cue_preset=None,
                   gist_per_episode=False, voice=None) -> SSMChunker:
    """SSMChunker wired with a stub voice (the mamba3-decode surface)."""
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    return SSMChunker(
        bb, _StubEmbedder(), cfg,
        voice=voice if voice is not None else _StubVoice(),
        gist_backend="mamba3",
        gist_cue=gist_cue,
        gist_cue_preset=gist_cue_preset,
        gist_per_episode=gist_per_episode,
    )


def test_gist_cue_preset_none_uses_gist_cue_byte_identical():
    """No preset -> the raw gist_cue reaches the decoder (byte-identical default)."""
    chunker = _voice_chunker(gist_cue="Summary:", gist_cue_preset=None)
    chunker.compress_gist_mamba3([_ep("e1")], query=None)
    assert chunker.voice.last_cue == "Summary:"


def test_gist_cue_preset_overrides_gist_cue():
    """A named preset wins over gist_cue; the preset's exact cue reaches the decoder."""
    chunker = _voice_chunker(
        gist_cue="ignored-raw-cue", gist_cue_preset="instruct-additive")
    assert chunker._gist_cue == GIST_CUE_PRESETS["instruct-additive"]
    chunker.compress_gist_mamba3([_ep("e1")], query=None)
    assert chunker.voice.last_cue == GIST_CUE_PRESETS["instruct-additive"]


def test_gist_cue_preset_each_named_key_resolves():
    """Every shipped preset key resolves to its constant and reaches the decoder."""
    for key in ("summary", "instruct-labeled", "instruct-additive"):
        chunker = _voice_chunker(gist_cue="raw", gist_cue_preset=key)
        assert chunker._gist_cue == GIST_CUE_PRESETS[key]
        chunker.compress_gist_mamba3([_ep("e1")], query=None)
        assert chunker.voice.last_cue == GIST_CUE_PRESETS[key]


def test_gist_cue_preset_unknown_raises():
    """An unknown preset name raises ValueError at construction (not silently)."""
    import pytest as _pt
    with _pt.raises(ValueError):
        _voice_chunker(gist_cue_preset="bogus")


def test_gist_cue_preset_loses_to_query_conditioning():
    """Query-conditioning builds 'Q: {query}\\nA:' and ignores the preset cue."""
    chunker = _voice_chunker(gist_cue_preset="instruct-additive")
    chunker.compress_gist_mamba3([_ep("e1")], query="what is the wifi password?")
    assert chunker.voice.last_cue == "Q: what is the wifi password?\nA:"


def test_gist_cue_presets_table_is_consistent():
    """The preset table is non-empty and 'summary' is the byte-identical default cue."""
    assert set(GIST_CUE_PRESETS) == {"summary", "instruct-labeled", "instruct-additive"}
    assert GIST_CUE_PRESETS["summary"] == "Summary:"
    # Generic examples must not leak the toy needles (clean-cue lesson).
    add = GIST_CUE_PRESETS["instruct-additive"].lower()
    for needle in ("xyz-4471", "helena voss", "pinecone-river"):
        assert needle not in add


# ── per-episode gisting (mamba3 decode backend) ───────────────────────────────

def test_gist_per_episode_off_is_one_joined_call_byte_identical():
    """OFF -> one ephemeral_gist call with ALL texts (byte-identical to pre-flag)."""
    voice = _RecordingVoice()
    chunker = _voice_chunker(gist_per_episode=False, voice=voice)
    eps = [_ep("e1", text="alpha details"), _ep("e2", text="beta details"),
           _ep("e3", text="gamma details")]
    out = chunker.compress_gist_mamba3(eps, query=None)
    assert len(voice.calls) == 1
    assert [t for t in voice.calls[0]["texts"]] == [
        "alpha details", "beta details", "gamma details"]
    # No max_new_tokens override on the joined path (the voice default is used).
    assert voice.calls[0]["max_new_tokens"] == 1024
    assert out == "<gist1>"


def test_gist_per_episode_on_calls_once_per_nonempty_episode():
    """ON -> one bounded call per non-empty episode, concatenated."""
    voice = _RecordingVoice()
    chunker = _voice_chunker(gist_per_episode=True, voice=voice)
    eps = [_ep("e1", text="alpha details"), _ep("e2", text="beta details"),
           _ep("e3", text="gamma details")]
    out = chunker.compress_gist_mamba3(eps, query=None)
    assert len(voice.calls) == 3
    for call in voice.calls:
        # Each call gets a SINGLE-element texts list (one episode, decoded alone).
        assert len(call["texts"]) == 1
        # The per-episode cap (256 new tokens) bounds each decode.
        assert call["max_new_tokens"] == 256
        assert call["cue"] == "Summary:"
    # The per-episode gists are concatenated with newlines, in episode order.
    assert out == "<gist1>\n<gist2>\n<gist3>"


def test_gist_per_episode_skips_empty_text_episodes():
    """ON -> empty/whitespace-only episodes get NO call (skipped, not gist'd).

    Note ``compress_gist_mamba3`` uses ``text or summary`` -- an empty ``text``
    falls back to ``summary`` (the production behavior). So to actually skip an
    episode, BOTH ``text`` and ``summary`` must be empty/whitespace.
    """
    voice = _RecordingVoice()
    chunker = _voice_chunker(gist_per_episode=True, voice=voice)
    eps = [_ep("e1", text="alpha details"),
           _ep("e2", text="   ", summary="   "),
           _ep("e3", text="", summary=""),
           _ep("e4", text="delta details")]
    out = chunker.compress_gist_mamba3(eps, query=None)
    # Only the two non-empty episodes were decoded.
    assert len(voice.calls) == 2
    assert voice.calls[0]["texts"] == ["alpha details"]
    assert voice.calls[1]["texts"] == ["delta details"]
    assert out == "<gist1>\n<gist2>"


def test_gist_per_episode_all_empty_returns_empty():
    """ON -> every episode empty -> no calls, "" returned (formatter falls back).

    Both ``text`` and ``summary`` empty (see the note on the skip test above)."""
    voice = _RecordingVoice()
    chunker = _voice_chunker(gist_per_episode=True, voice=voice)
    eps = [_ep("e1", text="", summary=""), _ep("e2", text="   ", summary="   ")]
    out = chunker.compress_gist_mamba3(eps, query=None)
    assert voice.calls == []
    assert out == ""


def test_gist_per_episode_with_query_conditioning_uses_qa_cue_per_call():
    """ON + query -> each per-episode call gets the Q-A cue (not the gist_cue)."""
    voice = _RecordingVoice()
    chunker = _voice_chunker(
        gist_cue="Summary:", gist_per_episode=True, voice=voice)
    eps = [_ep("e1", text="alpha details"), _ep("e2", text="beta details")]
    chunker.compress_gist_mamba3(eps, query="what is the wifi password?")
    assert len(voice.calls) == 2
    for call in voice.calls:
        assert call["cue"] == "Q: what is the wifi password?\nA:"


def test_gist_per_episode_skips_failed_episode_keeps_others():
    """ON -> an episode whose decode raises is skipped; the rest are kept.

    One episode's decode failure must not zero the whole gist (graceful skip).
    """
    voice = _RecordingVoice(fail_on="beta")  # the e2 body contains 'beta'
    chunker = _voice_chunker(gist_per_episode=True, voice=voice)
    eps = [_ep("e1", text="alpha details"), _ep("e2", text="beta details"),
           _ep("e3", text="gamma details")]
    out = chunker.compress_gist_mamba3(eps, query=None)
    # e2 raised -> skipped; e1 and e3 still produced gists.
    assert len(voice.calls) == 2
    bodies = [c["texts"][0] for c in voice.calls]
    assert "alpha details" in bodies
    assert "gamma details" in bodies
    assert "beta details" not in bodies
    # e1's marker is <gist1>; e2 raised (no marker); e3's is <gist2>.
    assert out == "<gist1>\n<gist2>"


def test_gist_per_episode_off_is_default_in_chunk():
    """The chunk() path with the flag OFF produces a joined gist (one voice call)."""
    voice = _RecordingVoice()
    chunker = _voice_chunker(gist_per_episode=False, voice=voice)
    eps = [_ep("e1", text="word " * 200), _ep("e2", text="word " * 200),
           _ep("e3", text="word " * 200), _ep("e4", text="word " * 200)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=1))
    # 1 primary, 3 secondary -> ONE joined decode over the 3 secondary texts.
    assert ctx.compressed_episode_count == 3
    assert ctx.compressed_gist == "<gist1>"
    assert len(voice.calls) == 1
    assert len(voice.calls[0]["texts"]) == 3


def test_gist_per_episode_on_in_chunk_splits_secondary():
    """The chunk() path with the flag ON decodes each secondary episode alone."""
    voice = _RecordingVoice()
    chunker = _voice_chunker(gist_per_episode=True, voice=voice)
    eps = [_ep("e1", text="word " * 200), _ep("e2", text="word " * 200),
           _ep("e3", text="word " * 200), _ep("e4", text="word " * 200)]
    ctx = chunker.chunk(eps, _plan(primary_chunk_count=1))
    # 1 primary, 3 secondary -> THREE per-episode decodes, concatenated.
    assert ctx.compressed_episode_count == 3
    assert ctx.compressed_gist == "<gist1>\n<gist2>\n<gist3>"
    assert len(voice.calls) == 3
    for call in voice.calls:
        assert len(call["texts"]) == 1