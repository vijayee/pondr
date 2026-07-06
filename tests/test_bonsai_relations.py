"""Tests for BonsaiRelationExtractor.

Two layers:

1. **Offline parser tests** (``_parse_relations``) — run everywhere; they verify
   the JSON-recovery logic that handles fenced output, trailing prose, bare
   lists, and malformed responses. No llama-server needed.
2. **Live extraction tests** — require the Bonsai ``llama-server`` HTTP endpoint
   (RunPod). Skipped automatically when the endpoint is unreachable, so the
   offline suite stays green.

The prompt text is asserted verbatim against ``BONSAI_RELATION_PROMPT`` to guard
the relation-type list against drift.
"""

import pytest
import requests

from src.config import config
from src.encoding.bonsai_relations import BONSAI_RELATION_PROMPT, BonsaiRelationExtractor


# ── Offline parser tests ──────────────────────────────────────────────────

def test_prompt_lists_relation_types():
    """The prompt enumerates the documented relation types verbatim."""
    for rt in ("explains", "decides", "expresses", "questions", "suggests",
               "concerns", "involves", "contradicts", "follows_up_on"):
        assert rt in BONSAI_RELATION_PROMPT, f"relation type {rt!r} missing from prompt"


def test_parse_clean_envelope():
    r = BonsaiRelationExtractor._parse_relations(
        '{"relations": [{"subject": "a", "predicate": "b", "object": "c"}]}'
    )
    assert r == [{"subject": "a", "predicate": "b", "object": "c"}]


def test_parse_fenced_json():
    r = BonsaiRelationExtractor._parse_relations(
        "```json\n{\"relations\": [{\"subject\": \"a\", \"predicate\": \"b\", \"object\": \"c\"}]}\n```"
    )
    assert r == [{"subject": "a", "predicate": "b", "object": "c"}]


def test_parse_trailing_prose():
    r = BonsaiRelationExtractor._parse_relations('{"relations": []}\n\nSorry, here is the answer.')
    assert r == []


def test_parse_bare_list_envelope():
    """The model may return a bare list instead of the {relations: [...]} envelope."""
    r = BonsaiRelationExtractor._parse_relations(
        '[{"subject": "a", "predicate": "b", "object": "c"}]'
    )
    assert r == [{"subject": "a", "predicate": "b", "object": "c"}]


def test_parse_filters_incomplete_relations():
    """Relations missing subject/predicate/object are dropped, not passed through."""
    r = BonsaiRelationExtractor._parse_relations(
        '{"relations": [{"subject": "a", "predicate": "b", "object": "c"}, {"subject": "x"}]}'
    )
    assert r == [{"subject": "a", "predicate": "b", "object": "c"}]


def test_parse_malformed_raises():
    with pytest.raises(RuntimeError):
        BonsaiRelationExtractor._parse_relations("not json at all")


def test_construct_is_offline():
    """Constructing the extractor opens no connection."""
    ext = BonsaiRelationExtractor()
    assert ext.endpoint == config.bonsai_endpoint.rstrip("/")


# ── Live extraction tests (skipped when the endpoint is down) ─────────────

@pytest.fixture(scope="session")
def bonsai_live():
    """Yield a live extractor, skipping the whole live suite if the endpoint
    is unreachable. Probes ``/v1/models`` (a cheap GET that lists loaded models
    without invoking inference)."""
    url = config.bonsai_endpoint.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable: {e}")
    return BonsaiRelationExtractor()


def test_extract_explains_relation(bonsai_live):
    """Bonsai extracts an 'explains' relation."""
    text = "User: What is DEBOUNCED? Assistant: DEBOUNCED batches fsync calls every 250ms."
    relations = bonsai_live.extract(text)
    explains_rels = [r for r in relations if r["predicate"] == "explains"]
    assert explains_rels, relations


def test_extract_decides_relation(bonsai_live):
    """Bonsai extracts a 'decides' relation."""
    text = "User: I'll go with DEBOUNCED then. Assistant: Good choice."
    relations = bonsai_live.extract(text)
    decides_rels = [r for r in relations if r["predicate"] == "decides"]
    assert decides_rels, relations


def test_extract_returns_well_formed_triples(bonsai_live):
    """Every returned relation has subject/predicate/object string fields."""
    text = "User: Alice suggested HBTrie for WaveDB. Assistant: Good idea."
    relations = bonsai_live.extract(text)
    for r in relations:
        assert {"subject", "predicate", "object"} <= r.keys()
        for v in (r["subject"], r["predicate"], r["object"]):
            assert isinstance(v, str)