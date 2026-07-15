"""Phase 4 (D8): EnterpriseRAG-Bench 'Conflicting Info' eval -- deterministic only.

Vendored offline fixture subset (``tests/fixtures/enterpriserag/pairs.json``):
near-duplicate document pairs where a NEWER doc supersedes facts from an
OLDER one (the bench's Conflicting-Info ground truth) + each pair's
``expected_doc_id`` (citation ground truth). No network, no model, no server.

Two deterministic-path signals are asserted:

1. **Contradiction recall** on the ``catchable`` pairs: encoding both docs
   writes two ``(E:entity, state, V)`` edges for the same normalized entity;
   the deterministic detector sees >=2 distinct live values -> the
   contradiction is flagged. The threshold is the deterministic-normalizer
   CEILING: a paraphrased-only conflict (pair 5, ``catchable=false``) the
   normalizer cannot see is HONESTLY counted as a miss -- that is Bonsai's
   job (out of scope for this deterministic-only slice, exercised via
   FakeDecider in ``test_contradiction.py`` + the live dogfood step).
2. **Citation resolve-rate**: ``find_document_by_title_or_url`` resolves the
   newer doc's title to its ``expected_doc_id``.

We do NOT run the bench's full LLM-judge harness (correctness x completeness,
three-judge consensus); we take its labels, not its scorer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.encoding.assertion_extractor import extract_state_assertions
from src.memory.document import Document, DocumentSection
from src.memory.store import HippocampalStore


FIXTURE = Path(__file__).parent / "fixtures" / "enterpriserag" / "pairs.json"

# The deterministic-normalizer ceiling: every CATCHABLE pair must be flagged.
# Pair 5 (paraphrased-only) is an honest miss and excluded from the recall
# numerator + denominator (it is asserted separately as a documented miss).
CATCHABLE_RECALL_THRESHOLD = 0.75  # 3/4 -> green; the fixtures are built for 4/4
CITATION_RESOLVE_THRESHOLD = 0.80  # 4/5 -> green; the fixtures are built for 5/5


@pytest.fixture(scope="module")
def pairs():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["pairs"]


def _encode_pair(tmp_path, pair):
    """Encode old + new doc into a FRESH store; return the store.

    A fresh store per pair keeps the hardcoded ``expected_doc_id`` (``doc_000002``
    = the second doc encoded) valid for every pair."""
    store = HippocampalStore(str(tmp_path / pair["id"] / "db"))
    for i, key in enumerate(("old_doc", "new_doc"), start=1):
        d = pair[key]
        assertions = [
            {"entity": a["entity"], "value": a["value"],
             "section": f"doc_{i:06d}_sec_000"}
            for a in extract_state_assertions(d["body"])
        ]
        doc = Document(
            id=f"doc_{i:06d}", source_type="markdown",
            source_path=d["source_path"], title=d["title"],
            ingested_at="2026-07-15T00:00:00",
            sections=[DocumentSection(id=f"doc_{i:06d}_sec_000",
                                       heading=d["title"], level=1,
                                       content=d["body"])],
            state_assertions=assertions,
        )
        store.encode_document(doc)
    return store


def _entity_state_values(store, entity_name):
    """The distinct CURRENT state values on ``(E:entity_name, state, ?)``."""
    subj = f"E:{entity_name}"
    r = store.graph.query().vertex(subj).out("state").execute_sync()
    try:
        vals = []
        for v in r.vertices:
            if store.is_edge_current(subj, "state", v) and v not in vals:
                vals.append(v)
    finally:
        r.close()
    return vals


# ── contradiction recall (deterministic ceiling) ──

def test_contradiction_recall_on_catchable_pairs(tmp_path, pairs):
    """Every CATCHABLE pair produces >=2 distinct live state values for the
    conflicting entity -> the deterministic detector flags the contradiction.

    Recall is measured over the catchable pairs only; pair 5
    (``catchable=false``, paraphrased-only) is an honest deterministic miss,
    asserted separately. The threshold (0.75) is the deterministic-normalizer
    ceiling -- a paraphrased conflict the normalizer cannot see is Bonsai's
    job, not a failure of this slice.
    """
    catchable = [p for p in pairs if p["catchable"]]
    assert catchable, "fixture must contain at least one catchable pair"
    hits = 0
    for pair in catchable:
        store = _encode_pair(tmp_path, pair)
        try:
            vals = _entity_state_values(store, pair["conflicting_entity"])
            flagged = len(vals) >= 2
            if flagged:
                hits += 1
            else:
                pytest.fail(
                    f"{pair['id']}: expected >=2 live state values for "
                    f"'{pair['conflicting_entity']}', got {vals}")
        finally:
            store.close()
    recall = hits / len(catchable)
    assert recall >= CATCHABLE_RECALL_THRESHOLD, (
        f"contradiction recall {recall:.2f} < {CATCHABLE_RECALL_THRESHOLD} "
        f"({hits}/{len(catchable)} catchable pairs flagged)")


def test_paraphrased_pair_is_honest_miss(tmp_path, pairs):
    """Pair 5 (paraphrased-only, no field shape) is an HONEST deterministic
    miss: the normalizer extracts no colliding assertions, so no
    contradiction is flagged. This documents the deterministic ceiling -- the
    Bonsai ``has_state`` path (out of scope here) is what would catch it."""
    missed = [p for p in pairs if not p["catchable"]]
    assert missed, "fixture should include at least one honest-miss pair"
    for pair in missed:
        store = _encode_pair(tmp_path, pair)
        try:
            vals = _entity_state_values(store, pair["conflicting_entity"])
            # Either no assertions, or a single value -- NOT a >=2 collision.
            assert len(vals) < 2, (
                f"{pair['id']} was expected to be a deterministic MISS but "
                f"the normalizer flagged {vals} -- update the fixture or the "
                f"normalizer")
        finally:
            store.close()


# ── citation resolve-rate ──

def test_citation_resolve_rate(tmp_path, pairs):
    """``find_document_by_title_or_url`` resolves the newer doc's title to its
    ``expected_doc_id`` (citation ground truth)."""
    hits = 0
    for pair in pairs:
        store = _encode_pair(tmp_path, pair)
        try:
            resolved = store.find_document_by_title_or_url(
                pair["new_doc"]["title"])
            if resolved == pair["expected_doc_id"]:
                hits += 1
            else:
                pytest.fail(
                    f"{pair['id']}: expected {pair['expected_doc_id']}, "
                    f"got {resolved}")
        finally:
            store.close()
    rate = hits / len(pairs)
    assert rate >= CITATION_RESOLVE_THRESHOLD, (
        f"citation resolve-rate {rate:.2f} < {CITATION_RESOLVE_THRESHOLD} "
        f"({hits}/{len(pairs)} expected_doc_ids resolved)")


# ── fixture integrity ──

def test_fixture_has_expected_shape(pairs):
    """The vendored subset is well-formed (the test does not silently skip)."""
    assert len(pairs) >= 5, "expected at least 5 pairs in the fixture subset"
    for p in pairs:
        assert {"id", "catchable", "old_doc", "new_doc", "expected_doc_id",
                "conflicting_entity"} <= p.keys(), p
        assert {"title", "source_path", "body"} <= p["old_doc"].keys()
        assert {"title", "source_path", "body"} <= p["new_doc"].keys()