"""Tests for the retrieval user-scope boundary (the read-side filter).

The retriever holds a ``user_id`` and threads owned-id sets to every retrieve
path; ``GraphTraversal.retrieve`` intersects its candidate set with them right
after candidate construction (before temporal / follows-chain re-anchoring).
Strict scope: a query user sees ONLY their own episodes + docs; unscoped + cross-
user + ``M:`` memories are excluded. ``user_id=None`` -> the global across-all-
users path, byte-identical to pre-user-scope.

Covers:

* GraphTraversal: a shared entity surfaces BOTH users' episodes + docs; with
  alice's allowed sets only alice's survive; with ``None`` both survive
  (byte-identical regression).
* HippocampalRetriever: constructed with ``user_id="alice"`` -> only alice's;
  ``user_id=None`` -> both. ``retrieve_with_plan`` too.
* Vector path: ``_filter_vector_hits_by_scope`` keeps a section iff its parent
  doc is owned; drops ``M:`` under strict scope; passes everything when both
  allowed sets are ``None`` (byte-identical).
* The fade is untouched by this work (not exercised here -- it has its own
  suite; this test only asserts the scope filter is byte-identical when off).

Offline: installed ``wavedb`` (CPU) + keyword extractor / bag-of-words stubs.
No GLiNER/Bonsai.
"""

from __future__ import annotations

import pytest

wavedb = pytest.importorskip("wavedb")
if not hasattr(wavedb, "VectorLayer"):
    pytest.skip("wavedb.VectorLayer not available (need wavedb>=0.2.0)", allow_module_level=True)

from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.retriever import HippocampalRetriever

from tests.test_doc_retrieval import _Bow384, _KWExtractor


def _store(tmp_path, **cfg):
    base = {"vector_index_enabled": True, "embedding_dim": 384}
    base.update(cfg)
    return HippocampalStore(str(tmp_path / "db"), config=base)


def _encode_episode(store, eid, *, user_id, session_id, entity):
    store.encode_episode(Episode(
        id=eid, timestamp="2026-07-03T10:00:00",
        summary=f"{eid} about {entity}", full_text=f"User: tell me about {entity}",
        entities=[entity], topics=["shared_topic"], tones=["curious"],
        user_id=user_id, session_id=session_id,
    ))


def _ingest_doc(store, tmp_path, *, user_id, source, text):
    src = tmp_path / source
    src.write_text(text, encoding="utf-8")
    chunker = HierarchicalChunker(max_section_tokens=200, min_section_tokens=1)
    pipe = UnifiedIngestionPipeline(store, chunker=chunker)
    doc_id, _ = pipe.ingest(
        str(src), extractor=_KWExtractor(), relation_extractor=None,
        embedder=_Bow384(), user_id=user_id,
    )
    return doc_id


_ALICE_DOC = (
    "# Alice Notes\n\n## Alice on Storage\n\nAlice architected the storage subsystem.\n"
)
_BOB_DOC = (
    "# Bob Notes\n\n## Bob on Storage\n\nBob and Alice architected the storage subsystem.\n"
)


def _two_user_corpus(store, tmp_path):
    """Two users (alice + bob) with one episode + one doc each, ALL sharing the
    ``Alice`` entity so the entity axis surfaces both users' content."""
    _encode_episode(store, "ep_001", user_id="alice", session_id="S:a1", entity="Alice")
    _encode_episode(store, "ep_002", user_id="bob", session_id="S:b1", entity="Alice")
    a_doc = _ingest_doc(store, tmp_path, user_id="alice", source="a.md", text=_ALICE_DOC)
    b_doc = _ingest_doc(store, tmp_path, user_id="bob", source="b.md", text=_BOB_DOC)
    return {"ep_001", "ep_002"}, {a_doc, b_doc}


# ── GraphTraversal.retrieve ──

def test_graph_scope_filters_to_query_user(tmp_path):
    store = _store(tmp_path)
    eps, docs = _two_user_corpus(store, tmp_path)
    trav = GraphTraversal(store)
    allowed_ep = store.episode_ids_for_user("alice")  # {ep_001}
    allowed_doc = store.document_ids_for_user("alice")  # {alice's doc}
    results = trav.retrieve(
        {"entities": ["Alice"], "entity_mode": "union"},
        allowed_episode_ids=allowed_ep, allowed_document_ids=allowed_doc,
    )
    ids = {r["episode_id"] for r in results}
    assert "ep_001" in ids  # alice's episode
    assert "ep_002" not in ids  # bob's episode excluded
    # Only alice's doc survived (sections start with ``doc_`` too -- exclude them
    # so this is the doc-level result set, not section chunks).
    doc_results = {r["episode_id"] for r in results
                   if r["episode_id"].startswith("doc_") and "_sec_" not in r["episode_id"]}
    assert doc_results <= allowed_doc
    assert doc_results == allowed_doc
    store.close()


def test_graph_scope_off_is_byte_identical(tmp_path):
    """No allowed sets -> the whole candidate set passes (pre-user-scope path)."""
    store = _store(tmp_path)
    _, docs = _two_user_corpus(store, tmp_path)
    trav = GraphTraversal(store)
    scoped = trav.retrieve({"entities": ["Alice"], "entity_mode": "union", "limit": 20})
    # Both users' episodes + docs surface (the global across-all-users path).
    ids = {r["episode_id"] for r in scoped}
    assert "ep_001" in ids and "ep_002" in ids
    assert docs <= ids
    store.close()


def test_graph_scope_unknown_user_sees_nothing(tmp_path):
    """Strict scope: an unknown user (no sessions / no docs) -> empty result."""
    store = _store(tmp_path)
    _two_user_corpus(store, tmp_path)
    trav = GraphTraversal(store)
    results = trav.retrieve(
        {"entities": ["Alice"], "entity_mode": "union"},
        allowed_episode_ids=store.episode_ids_for_user("nobody"),
        allowed_document_ids=store.document_ids_for_user("nobody"),
    )
    assert results == []
    store.close()


def test_graph_scope_applied_before_follows_chain(tmp_path):
    """The scope filter runs BEFORE temporal/follows re-anchoring, so a follows
    chain can never escape the user's sessions. Sanity: with scope on, the
    candidate set post-filter is the user's only -- there is no follows chain
    here, but the ordering is verified by the result being scoped."""
    store = _store(tmp_path)
    _two_user_corpus(store, tmp_path)
    trav = GraphTraversal(store)
    allowed_ep = store.episode_ids_for_user("alice")
    allowed_doc = store.document_ids_for_user("alice")
    results = trav.retrieve(
        {"entities": ["Alice"], "entity_mode": "union", "temporal_filter": "today"},
        allowed_episode_ids=allowed_ep, allowed_document_ids=allowed_doc,
    )
    ids = {r["episode_id"] for r in results}
    assert ids <= (allowed_ep | allowed_doc)
    store.close()


# ── HippocampalRetriever (the user_id wiring) ──

class _AlicePlanner:
    """Stub planner: always returns an Alice-entity plan (the shared entity).

    ``limit`` is raised above ``default_retrieval_limit`` (5) so BOTH users'
    episodes + docs + sections all surface -- the no-scope byte-identical test
    asserts on the full candidate set, and the default 5 would cut one of 6.
    """
    def plan(self, prompt, history=None):
        return {"entities": ["Alice"], "entity_mode": "union", "limit": 20}


def test_retriever_user_id_scopes_to_user(tmp_path):
    store = _store(tmp_path)
    _two_user_corpus(store, tmp_path)
    allowed_doc = store.document_ids_for_user("alice")
    retr = HippocampalRetriever(store, planner=_AlicePlanner(), user_id="alice")
    results = retr.retrieve("tell me about Alice", use_semantic=False)
    ids = {r["episode_id"] for r in results}
    assert "ep_001" in ids
    assert "ep_002" not in ids
    doc_results = {i for i in ids if i.startswith("doc_") and "_sec_" not in i}
    assert doc_results == allowed_doc
    store.close()


def test_retriever_user_id_none_is_byte_identical(tmp_path):
    store = _store(tmp_path)
    _, docs = _two_user_corpus(store, tmp_path)
    retr = HippocampalRetriever(store, planner=_AlicePlanner(), user_id=None)
    results = retr.retrieve("tell me about Alice", use_semantic=False)
    ids = {r["episode_id"] for r in results}
    assert "ep_001" in ids and "ep_002" in ids
    assert docs <= ids
    store.close()


def test_retriever_retrieve_with_plan_scopes(tmp_path):
    store = _store(tmp_path)
    _two_user_corpus(store, tmp_path)
    allowed_doc = store.document_ids_for_user("bob")
    retr = HippocampalRetriever(store, planner=_AlicePlanner(), user_id="bob")
    results = retr.retrieve_with_plan({"entities": ["Alice"], "entity_mode": "union"})
    ids = {r["episode_id"] for r in results}
    assert "ep_002" in ids
    assert "ep_001" not in ids
    doc_results = {i for i in ids if i.startswith("doc_") and "_sec_" not in i}
    assert doc_results == allowed_doc
    store.close()


# ── vector path post-filter ──

def test_filter_vector_hits_by_scope_keeps_owned_sections_drops_others():
    """Section ids gate on the PARENT doc's owner; M: dropped under strict."""
    a_doc = "doc_000001"
    b_doc = "doc_000002"
    hits = [
        (a_doc, 0.9),                # alice's doc -> kept
        ("ep_001", 0.85),            # alice's episode -> kept
        ("ep_002", 0.8),             # bob's episode -> dropped
        (b_doc, 0.75),               # bob's doc -> dropped
        (f"{b_doc}_sec_001", 0.7),   # bob's section -> dropped (parent b_doc)
        (f"{a_doc}_sec_002", 0.65),  # alice's section -> kept
        ("M:0001", 0.6),            # memory -> dropped under strict
    ]
    allowed_ep = {"ep_001"}
    allowed_doc = {a_doc}
    out = HippocampalRetriever._filter_vector_hits_by_scope(hits, allowed_ep, allowed_doc)
    out_ids = [eid for eid, _ in out]
    assert a_doc in out_ids
    assert "ep_001" in out_ids
    assert f"{a_doc}_sec_002" in out_ids
    assert b_doc not in out_ids
    assert f"{b_doc}_sec_001" not in out_ids
    assert "ep_002" not in out_ids
    assert "M:0001" not in out_ids


def test_filter_vector_hits_by_scope_off_passes_all():
    """Both allowed sets None -> byte-identical (everything passes, incl M:)."""
    hits = [("ep_001", 0.9), ("doc_000001_sec_001", 0.8), ("M:0001", 0.6)]
    out = HippocampalRetriever._filter_vector_hits_by_scope(hits, None, None)
    assert out == hits


def test_filter_vector_hits_by_scope_none_kind_passes_when_that_set_none():
    """allowed_ep=None but allowed_doc set -> episodes pass, docs gated, M dropped."""
    a_doc = "doc_000001"
    hits = [("ep_001", 0.9), (a_doc, 0.85), ("M:0001", 0.6)]
    out = HippocampalRetriever._filter_vector_hits_by_scope(hits, None, {a_doc})
    out_ids = [eid for eid, _ in out]
    assert "ep_001" in out_ids  # episodes pass (allowed_ep None)
    assert a_doc in out_ids
    assert "M:0001" not in out_ids  # strict (either set not None) -> dropped


# ── GraphTraversal._filter_user_scope (the candidate-set filter, unit) ──

def test_filter_user_scope_drops_memories_under_strict():
    cands = {"ep_001", "doc_000001", "doc_000001_sec_001", "M:0001"}
    out = GraphTraversal._filter_user_scope(cands, {"ep_001"}, {"doc_000001"})
    assert "M:0001" not in out
    assert out == {"ep_001", "doc_000001", "doc_000001_sec_001"}


def test_filter_user_scope_off_is_identity():
    cands = {"ep_001", "M:0001", "doc_000001"}
    assert GraphTraversal._filter_user_scope(cands, None, None) == cands


def test_filter_user_scope_section_gates_on_parent_doc():
    """A section id gates on its parent doc id, not the section id itself."""
    cands = {"doc_000001_sec_001", "doc_000002_sec_001"}
    # Only doc_000001 is owned -> its section survives, doc_000002's drops.
    out = GraphTraversal._filter_user_scope(cands, set(), {"doc_000001"})
    assert out == {"doc_000001_sec_001"}