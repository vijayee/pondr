"""Extraction-quality measurement against the hand-labeled sample conversations.

This is a quality MEASUREMENT, not a pass/fail gate: it prints per-conversation
entity/topic/tone recall so model quality can be tracked. Requires GLiNER
(RunPod); skipped when ``gliner``/``gliner2`` aren't importable.

Run on the pod:
    pytest tests/test_extraction_quality.py -s
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("gliner")
pytest.importorskip("gliner2")

from src.encoding.gliner_extractor import GLiNERExtractor  # noqa: E402


def _recall(expected: set, extracted: set) -> float:
    if not expected:
        return 1.0
    return len(expected & extracted) / len(expected)


def _topic_recall(expected_labels: list[str], extracted_spans: list[str]) -> float:
    """Recall for span-based topics against labeled intent.

    Topics are free-form spans now (not a fixed label set), so exact set
    intersection with the labeled ``expected_topics`` is meaningless — a
    conversation about "database_design" yields a span like "HBTrie storage
    layer", not the literal label. Instead, tokenize each expected label on
    ``_`` and count it as recalled when any extracted span contains one of its
    keywords as a case-insensitive substring ("database_design" → "database"
    hits span "database storage layer"). This keeps the measurement meaningful
    for span output while still flagging a collapsed extraction (no spans →
    0 recall).
    """
    if not expected_labels:
        return 1.0
    spans_low = [s.lower() for s in extracted_spans if isinstance(s, str)]
    hits = 0
    for label in expected_labels:
        keywords = [k for k in label.lower().split("_") if k]
        if not keywords:
            continue
        if any(kw in span for kw in keywords for span in spans_low):
            hits += 1
    return hits / len(expected_labels)


def test_extraction_matches_expected(capsys):
    """Print entity/topic/tone recall per conversation. Asserts only that the
    run completes and produces sane recall floors, so CI doesn't flake on
    probabilistic model output while still catching a broken extraction path."""
    extractor = GLiNERExtractor()

    path = Path(__file__).resolve().parent.parent / "data" / "sample_conversations.jsonl"
    with open(path, encoding="utf-8") as f:
        convs = [json.loads(line) for line in f]

    entity_recalls, topic_recalls, tone_recalls = [], [], []
    for conv in convs:
        full_text = " ".join(f"User: {u} Assistant: {a}" for u, a in conv["turns"])
        result = extractor.extract(full_text)

        er = _recall(set(conv.get("expected_entities", [])), set(result["entities"]))
        tr = _topic_recall(conv.get("expected_topics", []), result["topics"])
        nr = _recall(set(conv.get("expected_tones", [])), set(result["tones"]))
        entity_recalls.append(er)
        topic_recalls.append(tr)
        tone_recalls.append(nr)
        print(f"{conv['id']}: entity_recall={er:.2f}, topic_recall={tr:.2f}, tone_recall={nr:.2f}")

    mean_entity = sum(entity_recalls) / len(entity_recalls)
    mean_topic = sum(topic_recalls) / len(topic_recalls)
    mean_tone = sum(tone_recalls) / len(tone_recalls)
    print(f"\nMEAN: entity={mean_entity:.2f}, topic={mean_topic:.2f}, tone={mean_tone:.2f}")

    # Floors: a broken extraction path (wrong schema wiring, empty results)
    # collapses toward 0; a working one stays well above. Tuned loose so
    # model variance doesn't flake CI.
    assert mean_entity > 0.2, f"entity recall collapsed: {mean_entity:.2f}"
    assert mean_topic > 0.2, f"topic recall collapsed: {mean_topic:.2f}"
    assert mean_tone > 0.2, f"tone recall collapsed: {mean_tone:.2f}"