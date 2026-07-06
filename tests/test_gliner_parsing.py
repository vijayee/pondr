"""Offline tests for GLiNERExtractor's parsing logic.

These do NOT need ``gliner`` / ``gliner2`` — they validate ``_extract_stable``'s
wiring (decisions/topics come from entity spans; tones from a classification
coerced via ``_as_list``) by injecting a stub GLiNER2, so they run in the
offline suite. The live model-quality tests live in ``test_gliner_extractor.py``
(gated on the packages being installed).

Regression context: the first DialogSum scale run produced
``"decisions": ["d","e","c","i","s","i","o","n"]`` because the old
``multi_label: False`` "decisions" classification returned the bare string
``"decision"`` and ``_extract_stable`` iterated it char-by-char. These tests
guard that path can't recur.
"""

from __future__ import annotations

from collections import defaultdict

from src.encoding.gliner_extractor import GLiNERExtractor, _as_list


class _StubExtractor:
    """Stand-in for GLiNER2: returns a canned schema result."""

    def __init__(self, result: dict) -> None:
        self._result = result

    def extract(self, text, schema, threshold):
        return self._result


def _offline_extractor(result: dict) -> GLiNERExtractor:
    """Build a GLiNERExtractor without the heavy __init__ (no model download)."""
    ext = GLiNERExtractor.__new__(GLiNERExtractor)
    ext.threshold = 0.3
    ext.extractor = _StubExtractor(result)
    ext.discoverer = _StubExtractor({"entities": []})  # not used by _extract_stable
    ext.discovery_buffer = defaultdict(list)  # so extract() is safe to call too
    return ext


def test_as_list_coerces_string():
    """A bare string (multi_label:False output) becomes a 1-element list, not chars."""
    assert _as_list("decision") == ["decision"]
    assert _as_list(["a", "b"]) == ["a", "b"]
    assert _as_list(None) == []
    assert _as_list([]) == []


def test_extract_stable_pulls_decisions_and_topics_from_entity_spans():
    """decisions + topics come from the `entities` dict (spans), not classifications."""
    ext = _offline_extractor({
        "entities": {
            "person": ["Alice"],
            "project": ["WaveDB"],
            "technology": ["HBTrie"],
            "decision": ["go with DEBOUNCED"],
            "topic": ["WAL config", "sync modes"],
        },
        "tones": ["frustrated", "curious"],
    })
    result = ext._extract_stable("ignored by stub")
    assert set(result["entities"]) == {"Alice", "WaveDB", "HBTrie"}
    assert result["decisions"] == ["go with DEBOUNCED"]
    assert result["topics"] == ["WAL config", "sync modes"]
    assert result["tones"] == ["frustrated", "curious"]


def test_extract_stable_char_split_regression_guard():
    """A bare-string classification must NOT char-split (the bug that produced
    ['d','e','c','i','s','i','o','n'] on DialogSum)."""
    ext = _offline_extractor({
        "entities": {},
        # Simulate GLiNER2 returning a bare string for a classification task.
        "tones": "frustrated",
    })
    result = ext._extract_stable("ignored by stub")
    assert result["tones"] == ["frustrated"]  # whole label, not chars


def test_extract_stable_drops_empty_and_none_sentinel():
    """Empty spans / a 'none' tone sentinel are filtered out."""
    ext = _offline_extractor({
        "entities": {"decision": ["", "real decision"], "topic": ["", "real topic"]},
        "tones": ["none", "neutral", ""],
    })
    result = ext._extract_stable("ignored by stub")
    assert result["decisions"] == ["real decision"]
    assert result["topics"] == ["real topic"]
    assert result["tones"] == ["neutral"]


def test_extract_stable_handles_missing_categories():
    """A schema result missing categories yields empty lists, not KeyError."""
    ext = _offline_extractor({})
    result = ext._extract_stable("ignored by stub")
    assert result["entities"] == []
    assert result["decisions"] == []
    assert result["topics"] == []
    assert result["tones"] == []