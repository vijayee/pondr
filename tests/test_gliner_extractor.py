"""Unit tests for GLiNERExtractor.

These require the ``gliner`` and ``gliner2`` packages plus the downloaded
HuggingFace models — they run on the RunPod GPU pod, not locally. The whole
module is skipped when the packages are not importable, so the offline test
suite stays green.

Live inference assertions (entity/topic/tone/decision recall) follow the
plan's §8.1 expectations but are written leniently: GLiNER is a probabilistic
model, so we assert the *category* of result rather than an exact string match
where the plan's labels depend on the model's free-form output.
"""

import pytest

# Skips the whole module if gliner/gliner2 aren't installed (offline / non-GPU).
pytest.importorskip("gliner")
pytest.importorskip("gliner2")

from src.encoding.gliner_extractor import GLiNERExtractor  # noqa: E402


def _extractor():
    return GLiNERExtractor()


def test_extract_entities():
    """GLiNER extracts entities from conversation text."""
    extractor = _extractor()
    text = "Alice suggested using HBTrie for the WaveDB storage layer."
    result = extractor.extract(text)

    assert "entities" in result
    # At least one of the expected entities should surface; which one depends
    # on the model's threshold on this exact sentence.
    assert any(e in result["entities"] for e in ("Alice", "WaveDB", "HBTrie")), result["entities"]


def test_extract_tones():
    """GLiNER extracts emotional tones (multi-label classification)."""
    extractor = _extractor()
    text = "I'm so frustrated with this configuration. It's incredibly confusing."
    result = extractor.extract(text)
    # Probabilistic multi-label output: assert a valid tone surfaced, not a
    # specific label (the model may weight "frustrated" vs "curious" either way).
    assert isinstance(result["tones"], list)
    assert len(result["tones"]) > 0, result["tones"]
    known_tones = {"frustrated", "excited", "curious", "neutral"}
    assert all(t in known_tones for t in result["tones"]), result["tones"]


def test_extract_topics():
    """GLiNER extracts topics as free-form spans (varies with the corpus)."""
    extractor = _extractor()
    text = "The WAL sync modes need better documentation. DEBOUNCED vs ASYNC is unclear."
    result = extractor.extract(text)
    # Topics are spans now (not a fixed label set), so assert non-empty
    # multi-word-ish spans rather than membership in a closed taxonomy.
    assert isinstance(result["topics"], list)
    assert len(result["topics"]) > 0, result["topics"]
    # Regression guard for the char-split bug: a topic must be a whole span,
    # not a single character leaked from an iterated string.
    assert all(isinstance(t, str) and len(t) > 1 for t in result["topics"]), result["topics"]


def test_extract_decisions():
    """GLiNER extracts decisions as content spans (not a yes/no label)."""
    extractor = _extractor()
    text = "I've decided to go with DEBOUNCED for the WAL sync mode."
    result = extractor.extract(text)
    assert len(result["decisions"]) > 0, result["decisions"]
    # Regression guard for the char-split bug: decisions must be whole spans,
    # never single chars (the old multi_label:False classification returned the
    # bare string "decision" which iterated to ['d','e','c','i','s','i','o','n']).
    assert all(isinstance(d, str) and len(d) > 1 for d in result["decisions"]), result["decisions"]


def test_open_discovery():
    """GLiNER-Decoder discovers entity types not in the schema."""
    extractor = _extractor()
    text = "The Kubernetes deployment uses Helm charts for the staging environment."
    result = extractor.extract(text)
    # Discovered items are free-form {text, label}; we only assert non-empty.
    assert isinstance(result["discovered"], list)
    assert len(result["discovered"]) > 0, result["discovered"]


def test_extract_returns_all_keys():
    """extract() returns the full documented shape."""
    extractor = _extractor()
    result = extractor.extract("Alice uses WaveDB.")
    for key in ("entities", "topics", "tones", "decisions", "discovered"):
        assert key in result, f"missing key {key}"


def test_promotion_buffer_accumulates():
    """Repeated discoveries of the same label accumulate in the buffer."""
    extractor = _extractor()
    extractor._buffer_discoveries([{"text": "a", "label": "concept"}])
    extractor._buffer_discoveries([{"text": "b", "label": "concept"}])
    assert extractor.discovery_buffer["concept"] == ["a", "b"]