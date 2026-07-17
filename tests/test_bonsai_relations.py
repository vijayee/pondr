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


def test_parse_truncated_recovers_complete_relations():
    """A JSON stream truncated mid-object (max_tokens cut) still yields the
    complete relations before the cut, instead of raising."""
    truncated = (
        '{"relations": ['
        '{"subject": "Alice", "predicate": "decides", "object": "use HBTrie"}, '
        '{"subject": "Bob", "predicate": "suggests", "object": "WAL config"}, '
        '{"subject": "Alice", "predicate": "explains", "object": "the B+tr'
    )
    r = BonsaiRelationExtractor._parse_relations(truncated)
    assert r == [
        {"subject": "Alice", "predicate": "decides", "object": "use HBTrie"},
        {"subject": "Bob", "predicate": "suggests", "object": "WAL config"},
    ]


def test_parse_caps_to_max_relations():
    """Over-extraction is capped to _MAX_RELATIONS salient relations."""
    rel = '{"subject": "a", "predicate": "explains", "object": "b"}'
    body = '{"relations": [' + ', '.join([rel] * 20) + ']}'
    r = BonsaiRelationExtractor._parse_relations(body)
    from src.encoding.bonsai_relations import _MAX_RELATIONS
    assert len(r) == _MAX_RELATIONS


def test_prompt_caps_relation_count():
    """The prompt tells the model to emit at most a small number of relations."""
    assert "AT MOST 6" in BONSAI_RELATION_PROMPT or "at most 6" in BONSAI_RELATION_PROMPT


def test_construct_is_offline():
    """Constructing the extractor opens no connection."""
    ext = BonsaiRelationExtractor()
    assert ext.endpoint == config.bonsai_endpoint.rstrip("/")


# ── Isolated 10-pass extractor (offline, _post monkeypatched) ──────────────

def _stub_post_factory(calls: list, payload_per_call=None):
    """Return a _post stub that records each call and returns a fixed relation
    JSON. ``payload_per_call`` lets a test vary the returned body by call index
    (e.g. to make one pass raise)."""
    def _post(self, prompt, text, *, max_tokens=768):
        calls.append(prompt)
        idx = len(calls) - 1
        if payload_per_call is not None:
            return payload_per_call(idx)
        # Default: a relation whose predicate is a paraphrase -- isolation must
        # force-normalize it to the class name (the prompt's exact string).
        return '{"relations": [{"subject": "x", "predicate": "is_now", "object": "y"}]}'
    return _post


def test_extract_single_is_default_dispatch(monkeypatch):
    """With bonsai_isolation_extraction=False (the default), extract() makes ONE
    HTTP call (the V1 merged prompt), not 10."""
    monkeypatch.setattr(config, "bonsai_isolation_extraction", False)
    ext = BonsaiRelationExtractor()
    calls = []
    monkeypatch.setattr(BonsaiRelationExtractor, "_post", _stub_post_factory(calls))
    rels = ext.extract("User: q\nAssistant: a")
    assert len(calls) == 1, f"V1 path must be one call, got {len(calls)}"
    assert BONSAI_RELATION_PROMPT.splitlines()[0] in calls[0]
    # The paraphrased predicate is NOT force-normalized on the V1 path.
    assert rels == [{"subject": "x", "predicate": "is_now", "object": "y"}]


def test_isolated_runs_one_pass_per_class_and_normalizes_predicate(monkeypatch):
    """Isolated mode runs one pass per class (10 calls) and force-normalizes
    each relation's predicate to the exact class name -- the ternary 8B
    paraphrases predicates, but the merged graph must carry canonical names."""
    from src.encoding.bonsai_relations import ISOLATION_CLASSES
    monkeypatch.setattr(config, "bonsai_isolation_extraction", True)
    ext = BonsaiRelationExtractor()
    calls = []
    monkeypatch.setattr(BonsaiRelationExtractor, "_post", _stub_post_factory(calls))
    rels = ext.extract("User: q\nAssistant: a")

    assert len(calls) == len(ISOLATION_CLASSES) == 10
    assert {r["predicate"] for r in rels} == {c[0] for c in ISOLATION_CLASSES}
    # Every relation carries a canonical predicate (no paraphrased "is_now").
    assert all(r["predicate"] != "is_now" for r in rels)


def test_isolated_degrades_failed_pass_to_empty(monkeypatch):
    """A single class's HTTP/parse failure degrades to empty for that class --
    the other 9 classes still land. One hiccup does not drop the extraction."""
    monkeypatch.setattr(config, "bonsai_isolation_extraction", True)
    ext = BonsaiRelationExtractor()
    calls = []

    def per_call(idx):
        if idx == 3:  # the 4th class (questions) raises mid-stream
            raise RuntimeError("simulated HTTP 500")
        return '{"relations": [{"subject": "x", "predicate": "p", "object": "y"}]}'

    monkeypatch.setattr(BonsaiRelationExtractor, "_post", _stub_post_factory(calls, per_call))
    rels = ext.extract("User: q\nAssistant: a")

    assert len(calls) == 10            # all 10 passes attempted
    assert len(rels) == 9              # the failed class contributed nothing
    from src.encoding.bonsai_relations import ISOLATION_CLASSES
    failed_pred = ISOLATION_CLASSES[3][0]
    assert all(r["predicate"] != failed_pred for r in rels)


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