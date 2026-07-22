"""Tests for STRM 1f-6: LLM prose summaries as the code-section embedding handle.

The 1f-5 doc-kind split showed bilinear (Head A) clears the live z_logit gate on
TEXT docs (median 3.969, 2/3) but MIS-RANKS CODE docs (median -0.801, 0/3) -- raw
code embeds poorly against prose queries. 1f-6 gives each code section a 1-2
sentence LLM prose description used as the *embedding handle* (the stored
``{s}/embedding`` + the STRM inject-time ``y_t``), while the recalled code body
stays as ``content``/``text`` (the binding user concern: "recalling code and
reasoning over it will be critical to an llm writing code").

These tests pin the three invariants of the design:

* **Never-raise + cold-start safe** (``CodeSectionSummarizer``): a down server,
  non-JSON, missing field, or batch failure -> per-section ``None`` (or all-None
  on a hard failure) so the pipeline falls back to embedding the raw source,
  byte-identical to a not-wired ingest. The ``<module>`` root section is skipped.
* **Embed-handle vs recall-content decoupled** (``pipeline`` + ``store``): a stub
  summarizer populates ``sec.summary``; the embedder encodes the PROSE
  (``heading + summary``), NOT the raw source; ``sec_texts`` (the doc-kind tagger
  input) keeps the RAW content (so code docs are not mis-classified as text); the
  summary persists to ``{s}/summary`` and round-trips through ``get_document``.
  With ``summarizer=None`` the embedder encodes raw content and no ``{s}/summary``
  key is written -> byte-identical to pre-1f-6.
* **Code-recall preservation** (the binding concern): ``format_for_llm`` on a
  retrieved code doc WITH summaries present still renders the FULL code body in
  the ``Title:``/``Section:`` lines -- ``embed_text`` did NOT leak into
  ``summary``/``text``. The ``Description:`` line is additive (present only with
  ``embed_text``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)", allow_module_level=True)

from src.ingestion.code_summarizer import CodeSectionSummarizer
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.code_parser import CodeParser
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.store import HippocampalStore
from src.retrieval.chunked_context import ChunkedContextFormatter


# ── OracleResult stub (mirrors src.training.oracle_labeling.OracleResult) ──

@dataclass
class _FakeResult:
    response: dict
    error: Optional[str] = None
    cached: bool = False


def _make_summarizer(monkeypatch, batch_return):
    """Build a CodeSectionSummarizer with its OracleClient.generate_batch stubbed.

    ``batch_return`` is either a list of ``_FakeResult`` (one per prompt the
    summarizer decides to send) or an Exception instance to raise on call.
    """
    s = CodeSectionSummarizer(model="stub:cloud")

    def _fake_generate_batch(prompts, **kwargs):
        if isinstance(batch_return, BaseException):
            raise batch_return
        # The summarizer skips <module> + empty sections, so the number of
        # prompts it sends may be < the number of items. Return one result per
        # prompt actually sent, preserving order.
        assert len(batch_return) == len(prompts), (
            f"stub expected {len(batch_return)} prompts, got {len(prompts)}")
        return list(batch_return)

    monkeypatch.setattr(s._client, "generate_batch", _fake_generate_batch)
    return s


# ── 1. CodeSectionSummarizer: never-raise + JSON parse + <module> skip ──

def test_summarizer_parses_json_envelope(monkeypatch):
    """A {"summary": "..."} envelope -> the prose string, stripped."""
    items = [("def greet", "def greet(name): return name"),
             ("def bar", "def bar(): pass")]
    s = _make_summarizer(monkeypatch, [
        _FakeResult(response={"summary": "  Greets a name.  "}),
        _FakeResult(response={"summary": "Bar does nothing."}),
    ])
    out = s.summarize_sections(items)
    assert out == ["Greets a name.", "Bar does nothing."]


def test_summarizer_skips_module_root(monkeypatch):
    """The <module> root section is skipped -> None (no prompt sent for it)."""
    items = [("<module>", "import os\nimport sys"),
             ("def greet", "def greet(name): return name")]
    # Only ONE prompt is sent (for def greet); the stub gets exactly one result.
    s = _make_summarizer(monkeypatch, [_FakeResult(response={"summary": "Greets."})])
    out = s.summarize_sections(items)
    assert out[0] is None          # <module> skipped
    assert out[1] == "Greets."


def test_summarizer_skips_empty_content(monkeypatch):
    """Empty/whitespace content -> None (no prompt sent)."""
    items = [("def empty", "   "), ("def real", "def real(): return 1")]
    s = _make_summarizer(monkeypatch, [_FakeResult(response={"summary": "Real."})])
    out = s.summarize_sections(items)
    assert out[0] is None
    assert out[1] == "Real."


def test_summarizer_never_raises_on_batch_exception(monkeypatch):
    """A hard batch failure (server down) -> all-None, NO exception raised."""
    items = [("def greet", "def greet(name): return name"),
             ("def bar", "def bar(): pass")]
    s = _make_summarizer(monkeypatch, ConnectionError("server down"))
    out = s.summarize_sections(items)   # must not raise
    assert out == [None, None]


def test_summarizer_per_section_failure_routes_to_none(monkeypatch):
    """A per-section error / non-dict response / missing key -> None for that
    section only; good sections still return their prose."""
    items = [("def a", "def a(): pass"), ("def b", "def b(): pass"),
             ("def c", "def c(): pass"), ("def d", "def d(): pass")]
    s = _make_summarizer(monkeypatch, [
        _FakeResult(response={"summary": "A does it."}),          # ok
        _FakeResult(response={}, error="timeout"),                # error -> None
        _FakeResult(response={"not_summary": "x"}),               # missing key -> None
        _FakeResult(response={"summary": ""}),                    # empty -> None
    ])
    out = s.summarize_sections(items)
    assert out[0] == "A does it."
    assert out[1] is None
    assert out[2] is None
    assert out[3] is None


def test_summarizer_empty_input_returns_empty(monkeypatch):
    s = _make_summarizer(monkeypatch, [])
    assert s.summarize_sections([]) == []


# ── 2. Pipeline + store: embed-handle decoupled + round-trip ──

_PY_SRC = '''"""Module docstring for the encoder."""

import os


def encode_document(doc):
    """Encode a document into a dense vector."""
    return [0.1, 0.2, 0.3]


class Encoder:
    """A stateful encoder."""

    def fit(self, data):
        return self
'''


class _RecordingBow384:
    """384-dim bag-of-words embedder that RECORDS every text it encodes, so the
    test can assert the embedder saw prose (not raw source) when a summary is
    present, and raw source when it is not."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.recorded: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        out: list[list[float]] = []
        for t in texts:
            self.recorded.append(t)
            vec = [0.0] * self.dim
            for w in t.lower().split():
                w = "".join(c for c in w if c.isalnum())
                if not w:
                    continue
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            out.append(vec)
        return out


class _StubSummarizer:
    """Deterministic stand-in for CodeSectionSummarizer (no network).

    Returns a fixed prose string per non-``<module>`` section so the test is
    offline + reproducible. Mirrors the real contract
    ``summarize_sections(items) -> list[Optional[str]]``.
    """

    def summarize_sections(self, items):
        out: list[Optional[str]] = []
        for heading, content in items:
            if heading == "<module>":
                out.append(None)
                continue
            if not content or not content.strip():
                out.append(None)
                continue
            out.append(f"Prose handle for {heading}.")
        return out


def _store(tmp_path):
    return HippocampalStore(str(tmp_path / "db"),
                            config={"vector_index_enabled": True, "embedding_dim": 384})


def _ingest_code(store, tmp_path, *, embedder, summarizer, source="encoder.py"):
    """Write the fixture .py to disk + run the real pipeline. Returns doc_id."""
    src = tmp_path / source
    src.write_text(_PY_SRC, encoding="utf-8")
    chunker = HierarchicalChunker(max_section_tokens=200, min_section_tokens=1)
    pipe = UnifiedIngestionPipeline(store, chunker=chunker)
    doc_id, _ = pipe.ingest(
        str(src), source_type="auto",
        embedder=embedder, summarizer=summarizer,
    )
    return doc_id


def test_pipeline_embeds_prose_not_raw_when_summarizer_present(tmp_path):
    """With a summarizer, the embedder encodes ``heading + summary`` (prose), NOT
    the raw ``heading + content``. The prose handle reaches the stored embedding."""
    store = _store(tmp_path)
    try:
        emb = _RecordingBow384()
        doc_id = _ingest_code(store, tmp_path, embedder=emb,
                              summarizer=_StubSummarizer())
        assert emb.recorded, "embedder was not called"
        # Sections WITH a summary encode prose (``heading + summary``); the
        # ``<module>`` root (no summary, skipped) falls back to raw content --
        # byte-identical to pre-1f-6 (the designed fallback). Match each encoded
        # text to its section by heading prefix.
        doc = store.get_document(doc_id, load_bodies=False)
        sec_by_heading = {s.heading: s for s in doc.sections}
        assert len(emb.recorded) == len(doc.sections)
        for text, sec in zip(emb.recorded, doc.sections):
            if sec.summary:
                # Prose-derived: the prose marker is present, the raw BODY is not.
                # (The heading prefix is legitimately present -- it is part of
                # ``heading + summary``; the invariant is the raw code body is
                # gone, replaced by prose.)
                assert "Prose handle for" in text, f"raw leaked for {sec.heading!r}: {text!r}"
                assert "return [0.1, 0.2, 0.3]" not in text, (
                    f"raw body leaked into embed for {sec.heading!r}: {text!r}")
            else:
                # No summary (the <module> root) -> raw content fallback.
                assert sec.heading == "<module>"
                assert "import os" in text or "Module docstring" in text
        # The summary persisted + round-trips through get_document.
        summaries = [s.summary for s in doc.sections]
        assert any(s for s in summaries), f"no summaries persisted: {summaries}"
        # The <module> root section stayed None (skipped).
        root = sec_by_heading["<module>"]
        assert root.summary is None
    finally:
        store.close()


def test_pipeline_embeds_raw_when_summarizer_none(tmp_path):
    """With no summarizer, the embedder encodes raw ``heading + content`` and no
    ``{s}/summary`` key is written -> byte-identical to pre-1f-6."""
    store = _store(tmp_path)
    try:
        emb = _RecordingBow384()
        doc_id = _ingest_code(store, tmp_path, embedder=emb, summarizer=None)
        assert emb.recorded, "embedder was not called"
        # Raw source must appear in at least one encoded text (the def body).
        joined = "\n".join(emb.recorded)
        assert "def encode_document" in joined, f"raw source missing: {emb.recorded}"
        # No section carries a summary.
        doc = store.get_document(doc_id, load_bodies=False)
        assert all(s.summary is None for s in doc.sections)
        # And no {s}/summary key was written to the store.
        for s in doc.sections:
            assert store.db.get_sync(f"content/doc/{doc_id}/{s.id}/summary") is None
    finally:
        store.close()


def test_summary_round_trips_through_store(tmp_path):
    """A summary written on ingest is read back by get_document (load_bodies
    path included, since _hydrate_section uses get_document)."""
    store = _store(tmp_path)
    try:
        doc_id = _ingest_code(store, tmp_path, embedder=_RecordingBow384(),
                              summarizer=_StubSummarizer())
        doc_meta = store.get_document(doc_id, load_bodies=False)
        non_root = [s for s in doc_meta.sections if s.heading != "<module>"]
        assert non_root, "expected non-root sections"
        assert all(s.summary and s.summary.startswith("Prose handle for")
                   for s in non_root)
    finally:
        store.close()


# ── 3. Code-recall preservation (the binding user concern) ──

def test_format_for_llm_preserves_full_code_body_with_summaries(tmp_path):
    """The binding concern: with summaries present, the LLM context STILL
    contains the FULL code body -- ``embed_text`` did NOT leak into
    ``summary``/``text``. A ``Description:`` line is added (additive)."""
    fmt = ChunkedContextFormatter()
    store = _store(tmp_path)
    try:
        doc_id = _ingest_code(store, tmp_path, embedder=_RecordingBow384(),
                              summarizer=_StubSummarizer())
        doc = store.get_document(doc_id, load_bodies=True)
        # Build a hydrated section episode the way graph_traversal would, with
        # the prose summary surfaced as embed_text (mirrors _hydrate_section).
        sec = next(s for s in doc.sections if s.heading.startswith("def encode_document"))
        body = store.get_section_body(doc_id, sec.id)
        episode = {
            "episode_id": sec.id,
            "kind": "section",
            "summary": doc.title,            # doc title (UNCHANGED)
            "text": body or sec.content,     # the FULL recalled code body
            "timestamp": doc.ingested_at,
            "source_path": doc.source_path,
            "section_heading": sec.heading,
            "embed_text": sec.summary or "",  # prose handle (additive)
            "entities": [], "topics": [],
        }
        rendered = fmt._format_episode(episode)
        # The full code body must still be present (recall preservation).
        assert "def encode_document" in rendered, "full code body lost from context"
        assert "return [0.1, 0.2, 0.3]" in rendered, "code body truncated"
        # The prose description is surfaced additively (the stub wrote prose for
        # the full-signature heading).
        assert f"Description: Prose handle for {sec.heading}." in rendered
        # The embed_text did NOT replace the recalled body in the Section line.
        assert f"Section '{sec.heading}': def encode_document" in rendered
    finally:
        store.close()


def test_format_for_llm_byte_identical_when_no_embed_text(tmp_path):
    """A section episode with no embed_text renders WITHOUT a Description line
    -> byte-identical to pre-1f-6."""
    store = _store(tmp_path)
    try:
        doc_id = _ingest_code(store, tmp_path, embedder=_RecordingBow384(),
                              summarizer=None)
        doc = store.get_document(doc_id, load_bodies=True)
        sec = next(s for s in doc.sections if s.heading.startswith("def encode_document"))
        body = store.get_section_body(doc_id, sec.id)
        episode_no_et = {
            "episode_id": sec.id, "kind": "section", "summary": doc.title,
            "text": body or sec.content, "timestamp": doc.ingested_at,
            "source_path": doc.source_path, "section_heading": sec.heading,
            "entities": [], "topics": [],
        }
        episode_with_et_empty = {**episode_no_et, "embed_text": ""}
        fmt = ChunkedContextFormatter()
        a = fmt._format_episode(episode_no_et)
        b = fmt._format_episode(episode_with_et_empty)
        assert a == b, "empty embed_text changed the render"
        assert "Description:" not in a
        assert "def encode_document" in a   # code still recalled
    finally:
        store.close()