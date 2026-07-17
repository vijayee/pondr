"""Regression guards for the ``encode_episode`` content/edge split (async-distill).

Phase 3c async distillation factors ``encode_episode`` into two op-builders --
``_content_ops`` (the ``content/ep/{eid}/...`` puts) and ``_edge_ops`` (the
graph triples + state_assertion + cited_from + scope edges) -- so a turn can be
stored as a stub (content + vector index, no edges) and filled later by a
background worker. These tests pin the two load-bearing properties of that
split, offline (no GLiNER / Bonsai):

  1. ``encode_episode`` is still one atomic batch of content + edges -- i.e. the
     split is byte-identical to the pre-split path. Guarded by comparing the
     full key set of a single ``encode_episode`` against stub-then-fill (test 2);
     they must be equal, so the merged path and the split path land the same
     keys.
  2. ``encode_episode_content`` then ``encode_episode_edges`` produces the SAME
     final content + graph keys as a single ``encode_episode`` -- the split is
     lossless.
  3. A stub-only episode is content-retrievable + vector-retrievable but
     graph-invisible (no entity/session edges) -- the dummy state the user sees
     while extraction is in flight.
  4. ``encode_episode_edges`` fills the graph edges for an already-stored stub
     -- the episode becomes graph-reachable.
"""

from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.retrieval.vector_search import VectorSearch


def _ep(eid: str = "ep_001") -> Episode:
    return Episode(
        id=eid,
        timestamp="2026-07-16T10:00:00",
        summary="Alice chose Postgres for the database",
        full_text="User: what db?\nAssistant: Alice chose Postgres",
        entities=["Alice", "Postgres"],
        topics=["database_design"],
        tones=["decisive"],
        decisions=["use_postgres"],
        relations=[{"subject": "Alice", "predicate": "decides", "object": "use_postgres"}],
        # A small stub vector so the embedding key + VectorSearch path are
        # exercised. _content_ops writes the content/ep/{eid}/embedding key
        # whenever this is set; VectorSearch.build_index reads it back.
        summary_embedding=[1.0, 0.0, 0.0, 0.0],
    )


class _StubEmbedder:
    """4-dim stub embedder matching the episode's summary_embedding dim so
    VectorSearch.build_index + search work end-to-end offline."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for w in t.lower().split():
                w = "".join(c for c in w if c.isalnum())
                if w:
                    vec[hash(w) % self.dim] += 1.0
            out.append(vec)
        return out


def _scan(store: HippocampalStore, prefix: str) -> list[str]:
    """Bounded prefix scan (``end = prefix + \\x7f`` bounds the subtree; the
    ``\\x7f`` sentinel sorts after any real key char, matching the convention in
    test_store.py). ``end=None`` would run to the end of the whole DB, pulling
    in adjacent subtrees."""
    end = prefix + "\x7f"
    return [k for k, _ in store.db.create_read_stream(start=prefix, end=end)]


def _all_keys(store: HippocampalStore) -> set[str]:
    """Every content + graph key in the store -- the byte-identical comparison
    surface. The ontology seed is identical across fresh stores, so a key-set
    diff isolates the episode's writes."""
    return set(_scan(store, "content/")) | set(_scan(store, "memory/"))


def _entity_edge_keys(store: HippocampalStore, eid: str) -> list[str]:
    """Graph keys that link the episode to its entities (has_entity / in_episode
    / instanceOf). Empty for a stub-only episode; populated after the fill."""
    out = []
    for k in _scan(store, "memory/spo/"):
        # has_entity: (eid, has_entity, E:...); in_episode: (E:..., in_episode, eid)
        if f"/{eid}" in k or f"{eid}/" in k:
            if "has_entity" in k or "in_episode" in k or "instanceOf" in k:
                out.append(k)
    return out


def test_split_is_lossless_stub_then_fill_equals_single_encode(tmp_path):
    """stub-then-fill lands the SAME content + graph keys as one encode_episode.

    This is the byte-identical / no-regression guard for the split: the merged
    path (encode_episode = _content_ops + _edge_ops in one batch_sync) and the
    split path (encode_episode_content + encode_episode_edges) must produce
    identical stores.
    """
    ep = _ep("ep_042")

    store_merged = HippocampalStore(str(tmp_path / "merged"))
    store_merged.encode_episode(ep)

    store_split = HippocampalStore(str(tmp_path / "split"))
    store_split.encode_episode_content(ep.id, ep)
    store_split.encode_episode_edges(ep.id, ep)

    assert _all_keys(store_merged) == _all_keys(store_split), (
        "split path diverged from merged path -- the refactor is not lossless"
    )
    store_merged.close()
    store_split.close()


def test_stub_only_is_content_retrievable_and_graph_invisible(tmp_path):
    """A stub episode is retrievable by content + embedding but has NO graph
    edges -- the dummy state while extraction is in flight."""
    store = HippocampalStore(str(tmp_path / "db"))
    ep = _ep("ep_077")
    store.encode_episode_content(ep.id, ep)

    # Content is present: get_episode reads it back.
    loaded = store.get_episode("ep_077")
    assert loaded is not None
    assert loaded.summary == ep.summary
    assert loaded.full_text == ep.full_text

    # Vector-retrievable: the embedding key persisted and VectorSearch indexes it.
    assert store.db.get_sync("content/ep/ep_077/embedding") is not None
    vs = VectorSearch(store, embedder=_StubEmbedder())
    n = vs.build_index()
    assert n >= 1
    hits = [h[0] for h in vs.search("database", k=5)]
    assert "ep_077" in hits, hits

    # Graph-invisible: no entity/session edges for this episode yet.
    assert _entity_edge_keys(store, "ep_077") == [], "stub wrote graph edges"
    store.close()


def test_fill_adds_graph_edges_to_stub(tmp_path):
    """encode_episode_edges populates the graph edges for an already-stored
    stub -- the episode becomes graph-reachable."""
    store = HippocampalStore(str(tmp_path / "db"))
    ep = _ep("ep_077")
    store.encode_episode_content(ep.id, ep)
    assert _entity_edge_keys(store, "ep_077") == []

    store.encode_episode_edges(ep.id, ep)

    edges = _entity_edge_keys(store, "ep_077")
    assert edges, "fill did not write entity edges"
    # The E:Alice in_episode ep_077 pointer must be present.
    assert any("E:Alice" in k and "in_episode" in k and "ep_077" in k for k in edges), edges
    store.close()


def test_fill_does_not_touch_content(tmp_path):
    """The fill writes edges only; content keys are unchanged by
    encode_episode_edges (the stub already wrote them). Guards against an
    accidental double-write of content in the edge op-builder."""
    store = HippocampalStore(str(tmp_path / "db"))
    ep = _ep("ep_077")
    store.encode_episode_content(ep.id, ep)
    content_before = _scan(store, "content/")

    store.encode_episode_edges(ep.id, ep)
    content_after = _scan(store, "content/")

    assert sorted(content_before) == sorted(content_after), (
        "fill mutated content keys -- _edge_ops must be edge-only"
    )
    store.close()