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

from src.config import Phase2cConfig, config
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


class _StubGistVoice:
    """A stub Mamba3Voice for the gist-backend tests (no torch, no HF).

    Records the episode texts + cue it received and returns a canned gist, so
    the tests can assert the chunker threaded the secondary episodes through
    ``ephemeral_gist`` and the formatter emitted the decoded text.
    """

    def __init__(self, gist: str = "DECODED GIST") -> None:
        self.gist = gist
        self.calls: list[tuple[list[str], str]] = []

    def ephemeral_gist(self, texts, cue, max_new_tokens=128):
        self.calls.append((list(texts), cue))
        return self.gist


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


# ── Mamba3 gist backend (the deferred Phase 2c "decode a summary" path) ──


def test_topics_backend_default_is_byte_identical_compressed_gist_none():
    """The default 'topics' backend produces NO decoded gist (compressed_gist is
    None) and the formatter emits the topic union -- byte-identical to the
    pre-mamba3 path (the bge-into-backbone SSM compressor still runs but its
    state is never read)."""
    eps = [_ep(f"e{i}", text="word " * 200, topics=[f"topic_{i}", "shared"],
               summary=f"sum {i}") for i in range(6)]
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    chunker = SSMChunker(bb, _StubEmbedder(), cfg)  # default gist_backend="topics"
    ctx = chunker.chunk(eps, _plan(strategy=CHUNKED, primary_chunk_count=2))
    assert ctx.compressed_gist is None              # no decoded gist
    assert ctx.compressed_state is not None         # the bge-into-backbone path ran
    out = ChunkedContextFormatter().format_for_llm(ctx)
    assert "[COMPRESSED CONTEXT — SUMMARY]" in out
    assert "Compressed topics:" in out              # topic union, not a decoded gist
    assert "Summary: " not in out.split("[COMPRESSED")[1]  # no decoded gist line


def test_mamba3_backend_decodes_gist_and_skips_backbone_compressor():
    """The 'mamba3' backend threads the secondary episodes through the voice's
    ephemeral_gist (which returns a canned gist), skips the dead-weight backbone
    compressor (compressed_state is None), and the formatter emits the decoded
    gist text instead of the topic union."""
    voice = _StubGistVoice(gist="The user discussed a rental car and a vet visit.")
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    chunker = SSMChunker(bb, _StubEmbedder(), cfg, voice=voice,
                         gist_backend="mamba3", gist_cue="Summary:")
    eps = [_ep(f"e{i}", text=f"episode {i} body " * 30,
               topics=[f"topic_{i}", "shared"], summary=f"summary {i}")
           for i in range(6)]
    ctx = chunker.chunk(eps, _plan(strategy=CHUNKED, primary_chunk_count=2))
    # The voice received the SECONDARY episodes' texts (e2..e5, 4 of them).
    assert len(voice.calls) == 1
    texts, cue = voice.calls[0]
    assert cue == "Summary:"
    assert len(texts) == 4
    # The decode path ingests the FULL TEXT (not the summary) -- the decoder
    # needs the content to summarize; summarizing a summary loses the facts.
    assert texts[0].startswith("episode 2 body")  # text preferred over summary
    # The backbone compressor was skipped (no dead-weight SSM step).
    assert ctx.compressed_state is None
    # The decoded gist was captured on the ChunkedContext.
    assert ctx.compressed_gist == "The user discussed a rental car and a vet visit."


def test_mamba3_backend_formatter_emits_gist_not_topics():
    """The formatter emits the decoded gist text (a 'Summary:' line) and keeps
    the EXPAND ids line; the topic-union line is NOT emitted under mamba3."""
    voice = _StubGistVoice(gist="Rental car plate ABC-9921; vet Dr. Voss at 3pm.")
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    chunker = SSMChunker(bb, _StubEmbedder(), cfg, voice=voice,
                         gist_backend="mamba3")
    eps = [_ep(f"e{i}", text=f"body {i} " * 30, topics=[f"t{i}"],
               summary=f"sum {i}") for i in range(5)]
    ctx = chunker.chunk(eps, _plan(strategy=CHUNKED, primary_chunk_count=2))
    out = ChunkedContextFormatter().format_for_llm(ctx)
    assert "[COMPRESSED CONTEXT — SUMMARY]" in out
    assert "Summary: Rental car plate ABC-9921; vet Dr. Voss at 3pm." in out
    assert "EXPAND(episode_id)" in out        # EXPAND ids line kept
    for i in range(2, 5):
        assert f"e{i}" in out                 # expandable ids still listed
    # The topic-union line is NOT emitted under mamba3.
    assert "Compressed topics:" not in out


def test_mamba3_backend_without_voice_falls_back_to_topics_formatter():
    """mamba3 requested but no voice loaded -> no compression (no SSM state, no
    gist); the formatter falls back to the topic union from secondary_episodes.
    The backbone compressor is NOT run (the user opted out of the topics path)."""
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    chunker = SSMChunker(bb, _StubEmbedder(), cfg, voice=None,
                         gist_backend="mamba3")
    eps = [_ep(f"e{i}", text=f"body {i} " * 30, topics=[f"t{i}", "shared"],
               summary=f"sum {i}") for i in range(4)]
    ctx = chunker.chunk(eps, _plan(strategy=CHUNKED, primary_chunk_count=2))
    assert ctx.compressed_state is None       # backbone compressor skipped
    assert ctx.compressed_gist is None        # no voice -> no decoded gist
    out = ChunkedContextFormatter().format_for_llm(ctx)
    # Formatter falls back to the topic union from the secondary episodes.
    assert "Compressed topics:" in out
    assert "shared" in out


def test_mamba3_backend_empty_decode_falls_back_to_topics_formatter():
    """A cold/empty decode (voice returns '') -> the formatter treats a falsy
    gist as 'no gist' and falls back to the topic union from secondary_episodes
    (topics > an empty 'Summary:' line for the LLM). compressed_gist is still
    recorded as '' on the ChunkedContext (the decode happened; it was empty)."""
    voice = _StubGistVoice(gist="")
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    chunker = SSMChunker(bb, _StubEmbedder(), cfg, voice=voice,
                         gist_backend="mamba3")
    eps = [_ep(f"e{i}", text=f"body {i} " * 30, topics=[f"t{i}", "shared"],
               summary=f"sum {i}") for i in range(4)]
    ctx = chunker.chunk(eps, _plan(strategy=CHUNKED, primary_chunk_count=2))
    assert ctx.compressed_gist == ""          # empty decode, recorded
    out = ChunkedContextFormatter().format_for_llm(ctx)
    # Formatter falls back to the topic union (truthy check on the gist).
    assert "Compressed topics:" in out
    assert "shared" in out
    assert "Summary: " not in out.split("[COMPRESSED")[1]


def test_mamba3_backend_no_secondary_episodes():
    """No secondary episodes -> no gist decode, no SSM state; the formatter
    omits the compressed section entirely (byte-identical to topics backend
    with no secondary)."""
    voice = _StubGistVoice()
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    chunker = SSMChunker(bb, _StubEmbedder(), cfg, voice=voice,
                         gist_backend="mamba3")
    eps = [_ep("e0"), _ep("e1")]
    ctx = chunker.chunk(eps, _plan(strategy=DIRECT, primary_chunk_count=2))
    assert ctx.compressed_gist is None
    assert ctx.compressed_state is None
    assert voice.calls == []                   # no decode attempted
    out = ChunkedContextFormatter().format_for_llm(ctx)
    assert "[COMPRESSED CONTEXT — SUMMARY]" not in out


def test_mamba3_backend_query_conditioned_cue_is_q_a():
    """``chunk(..., query=q)`` threads the user's question into the gist decode:
    the cue becomes ``Q: {q}\\nA:`` (the carry path's proven completion shape),
    NOT the fixed ``gist_cue``. The 443M is a recall machine, not a summarizer --
    a targeted Q-A cue recalls the asked-for facts where a global ``Summary:``
    cue degenerates (see ``pondr-mamba3-gist-eval-result``)."""
    voice = _StubGistVoice(gist="The license plate is XYZ-4471.")
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    chunker = SSMChunker(bb, _StubEmbedder(), cfg, voice=voice,
                         gist_backend="mamba3", gist_cue="Summary:")
    eps = [_ep(f"e{i}", text=f"episode {i} body " * 30,
               topics=[f"topic_{i}"], summary=f"summary {i}")
           for i in range(4)]
    ctx = chunker.chunk(eps, _plan(strategy=CHUNKED, primary_chunk_count=2),
                       query="What is the license plate of my rental car?")
    assert len(voice.calls) == 1
    _texts, cue = voice.calls[0]
    assert cue == "Q: What is the license plate of my rental car?\nA:"
    assert ctx.compressed_gist == "The license plate is XYZ-4471."


def test_mamba3_backend_no_query_is_byte_identical_fixed_cue():
    """``chunk(...)`` with NO ``query`` (the default at every call site until a
    flag opts in) uses the fixed ``gist_cue`` -- byte-identical to the pre-query-
    conditioned mamba3 path. The query-conditioning branch only fires when a
    query is explicitly passed."""
    voice = _StubGistVoice(gist="gist")
    bb = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    chunker = SSMChunker(bb, _StubEmbedder(), cfg, voice=voice,
                         gist_backend="mamba3", gist_cue="Summary:")
    eps = [_ep(f"e{i}", text=f"episode {i} body " * 30,
               topics=[f"topic_{i}"], summary=f"summary {i}")
           for i in range(4)]
    chunker.chunk(eps, _plan(strategy=CHUNKED, primary_chunk_count=2))  # no query
    assert voice.calls[0][1] == "Summary:"     # the fixed cue, byte-identical


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


# ── B3: active token-reclamation cascade (config.reclaim_enabled) ──


def test_reclaim_off_break_on_overflow_unchanged(monkeypatch):
    """Flag OFF (default) -> the exact break-on-overflow path (regression guard)."""
    monkeypatch.setattr(config, "reclaim_enabled", False)
    chunker = _chunker(max_primary_tokens=100000, max_primary_chunks=20)
    eps = [_ep(f"e{i}", text="word " * 100) for i in range(20)]
    ctx = chunker.chunk(eps, _plan(strategy=DIRECT, primary_chunk_count=20))
    out = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=300)
    assert "e0" in out
    assert "e19" not in out  # dropped by break-on-overflow, not truncated


def test_reclaim_on_keeps_tail_in_summary_where_off_drops(monkeypatch):
    """ON over budget: the lowest-SCORE tail demotes FULL -> SUMMARY instead of
    being dropped entirely, so MORE episodes stay in context (degraded)."""
    eps = [_ep(f"e{i}", text="word " * 100, summary=f"sum {i}") for i in range(5)]
    chunker = _chunker(max_primary_tokens=100000, max_primary_chunks=20)
    ctx = chunker.chunk(eps, _plan(strategy=DIRECT, primary_chunk_count=5))

    monkeypatch.setattr(config, "reclaim_enabled", False)
    out_off = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=300)
    monkeypatch.setattr(config, "reclaim_enabled", True)
    out_on = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=300)

    # OFF break-on-overflow drops the tail; ON keeps it in summary form.
    assert "e4" not in out_off
    assert "e4" in out_on
    assert "Summary: sum 4" in out_on
    # Every kept episode renders a `Summary:` line, so its count == # kept.
    # ON keeps more episodes (degraded) than OFF's break-on-overflow.
    assert out_on.count("Summary:") > out_off.count("Summary:")


def test_reclaim_on_under_budget_byte_identical(monkeypatch):
    """ON under budget -> no demotion -> byte-identical to OFF (both all FULL)."""
    eps = [_ep(f"e{i}", text="word " * 20, summary=f"sum {i}") for i in range(3)]
    chunker = _chunker(max_primary_tokens=100000, max_primary_chunks=20)
    ctx = chunker.chunk(eps, _plan(strategy=DIRECT, primary_chunk_count=3))

    monkeypatch.setattr(config, "reclaim_enabled", False)
    out_off = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=10_000)
    monkeypatch.setattr(config, "reclaim_enabled", True)
    out_on = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=10_000)
    assert out_on == out_off


def test_reclaim_on_emergency_truncates_oversized_section(monkeypatch):
    """A single oversized SECTION (no summary form -> mild skips; min_keep
    blocks aggressive) -> emergency truncates its body. OFF drops it."""
    sec = {
        "episode_id": "sec0", "kind": "section", "text": "body " * 2000,
        "timestamp": "2026-01-01", "topics": [], "entities": [],
        "summary": "sec title", "score": 1.0,
    }
    chunker = _chunker(max_primary_tokens=100000, max_primary_chunks=20)
    ctx = chunker.chunk([sec], _plan(strategy=DIRECT, primary_chunk_count=1))

    monkeypatch.setattr(config, "reclaim_enabled", False)
    out_off = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=300)
    monkeypatch.setattr(config, "reclaim_enabled", True)
    out_on = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=300)

    assert "sec0" not in out_off  # OFF break-on-overflow drops the oversized body
    assert "sec0" in out_on  # ON keeps it (truncated, not dropped)
    assert "[... truncated ...]" in out_on  # emergency body-truncation marker


def test_reclaim_mild_demotes_episode_not_section(monkeypatch):
    """mild demotes EPISODES (have a summary form) but NOT sections (no summary
    form) -- the section stays FULL (its only reclamation rung is DROP/truncate)."""
    ep = _ep("e0", text="word " * 200, summary="sum 0", score=0.9)  # ~250 tok
    sec = {
        "episode_id": "sec0", "kind": "section", "text": "body " * 20,
        "timestamp": "2026-01-01", "topics": [], "entities": [],
        "summary": "sec title", "score": 0.5,  # ~32 tok
    }
    chunker = _chunker(max_primary_tokens=100000, max_primary_chunks=20)
    ctx = chunker.chunk([ep, sec], _plan(strategy=DIRECT, primary_chunk_count=2))

    monkeypatch.setattr(config, "reclaim_enabled", True)
    out = ChunkedContextFormatter().format_for_llm(ctx, max_tokens=300)

    # The episode demoted: its summary is present, its full text is gone.
    assert "Summary: sum 0" in out
    assert "Full text: word" not in out
    # The section unaffected by mild: its body still present in full.
    assert "Section:" in out
    assert "body " in out


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