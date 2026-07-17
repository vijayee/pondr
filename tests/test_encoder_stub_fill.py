"""Offline contract tests for the encoder stub/fill split (async-distill).

The async-distill path splits ``encode_messages`` into ``encode_messages_stub``
(main thread: content + embedding, no extraction, no store) and
``encode_messages_fill`` (worker: GLiNER + Bonsai + state-assertions, no store,
no encoder-state mutation). These pin the three load-bearing contracts of that
split, WITHOUT the GLiNER/Bonsai deps:

  - the stub does not extract and does not store; it builds the episode in
    memory with the pre-allocated id and the ``follows`` chain.
  - the fill never calls ``next_episode_id`` (the counter is main-thread-only)
    and never mutates ``last_episode_id`` / ``session_id`` (worker-safe).
  - stub + fill populate the episode identically to the fused ``encode_messages``
    path (lossless at the encoder level) when the extractors return the same
    outputs -- the store-level lossless guard is test_store_stub_fill.py.

The encoder is built with ``__new__`` + manual attributes so GLiNER/Bonsai are
never constructed; ``_extract`` / ``_extract_relations`` are monkeypatched to
fixed offline values. ``_build_state_assertions`` is a deterministic regex
normalizer (no model), so it runs for real.
"""

import pytest

from src.encoding.encoder import HippocampalEncoder
from src.memory.store import HippocampalStore


def _bare_encoder(store: HippocampalStore, user_id: str = "u1") -> HippocampalEncoder:
    """An encoder with only the attributes stub/fill touch -- no GLiNER/Bonsai
    constructed. ``_extract`` / ``_extract_relations`` are patched per-test."""
    enc = HippocampalEncoder.__new__(HippocampalEncoder)
    enc.store = store
    enc.user_id = user_id
    enc.session_id = None
    enc.last_episode_id = None
    return enc


def _stub_embedder():
    def _emb(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    return _emb


def _mock_extract(enc, extracted=None):
    enc._extract = lambda full_text, *, degrade_on_extract_fail: extracted or {
        "entities": ["Alice", "Postgres"],
        "entity_classes": {"Alice": "Person"},
        "topics": ["database_design"],
        "tones": ["decisive"],
        "decisions": ["use_postgres"],
        "discovered": [],
    }


def _mock_relations(enc, rels=None):
    enc._extract_relations = lambda full_text, episode_id: rels or [
        {"subject": "Alice", "predicate": "decides", "object": "use_postgres"},
    ]


def test_stub_does_not_extract_or_store(tmp_path):
    """The stub builds the episode in memory with empty extraction and does NOT
    store -- nothing is retrievable until the caller writes the stub."""
    store = HippocampalStore(str(tmp_path / "db"))
    enc = _bare_encoder(store)
    enc.start_session()

    # If the stub tried to extract, this would raise (no gliner attr). It must
    # not touch _extract / _extract_relations at all.
    enc._extract = lambda *a, **k: pytest.fail("stub must not extract")
    enc._extract_relations = lambda *a, **k: pytest.fail("stub must not extract relations")

    msgs = [{"role": "user", "content": "what db?"},
            {"role": "assistant", "content": "Alice chose Postgres"}]
    ep = enc.encode_messages_stub(msgs, "ep_777", embedder=_stub_embedder())

    assert ep.id == "ep_777"
    assert ep.entities == [] and ep.topics == [] and ep.relations == []
    assert ep.state_assertions == []
    assert ep.summary_embedding == [1.0, 0.0, 0.0, 0.0]
    assert ep.user_id == "u1" and ep.session_id == enc.session_id
    # Nothing stored yet.
    assert store.get_episode("ep_777") is None
    store.close()


def test_follows_chain_correct_without_worker(tmp_path):
    """Turn N+1's ``follows == id_N`` is correct even when turn N's worker fill
    has NOT run -- the chain is set from ``last_episode_id`` on the main thread
    at stub time, independent of extraction completion."""
    store = HippocampalStore(str(tmp_path / "db"))
    enc = _bare_encoder(store)
    enc.start_session()
    enc._extract = lambda *a, **k: {}
    enc._extract_relations = lambda *a, **k: []

    msgs1 = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    msgs2 = [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]

    # Turn N: pre-allocate id, stub, write stub content, set follows chain --
    # but do NOT run the worker fill.
    id_n = store.next_episode_id()
    ep_n = enc.encode_messages_stub(msgs1, id_n, embedder=_stub_embedder())
    store.encode_episode_content(id_n, ep_n)
    enc.last_episode_id = id_n

    # Turn N+1: pre-allocate id, stub. Its `follows` must be id_n.
    id_n1 = store.next_episode_id()
    ep_n1 = enc.encode_messages_stub(msgs2, id_n1, embedder=_stub_embedder())

    assert ep_n.follows is None            # first turn in the session
    assert ep_n1.follows == id_n           # chain correct without the worker
    assert id_n1 != id_n
    store.close()


def test_fill_uses_passed_id_and_never_touches_counter(tmp_path):
    """The worker fill uses the handed-in id and never calls
    ``next_episode_id`` (the persisted counter is main-thread-only)."""
    store = HippocampalStore(str(tmp_path / "db"))
    enc = _bare_encoder(store)
    enc.start_session()
    _mock_extract(enc)
    _mock_relations(enc)

    # Sabotage the counter: if the fill calls it, this raises.
    store.next_episode_id = lambda: pytest.fail("fill must not call next_episode_id")

    msgs = [{"role": "user", "content": "what db?"},
            {"role": "assistant", "content": "Alice chose Postgres"}]
    ep = enc.encode_messages_stub(msgs, "ep_999", embedder=_stub_embedder())
    filled = enc.encode_messages_fill(ep, "ep_999")

    assert filled.id == "ep_999"
    assert filled.entities == ["Alice", "Postgres"]
    assert filled.entity_classes == {"Alice": "Person"}
    assert filled.topics == ["database_design"]
    assert filled.relations == [{"subject": "Alice", "predicate": "decides", "object": "use_postgres"}]
    # Nothing stored by the fill (the worker stores edges separately).
    assert store.get_episode("ep_999") is None
    store.close()


def test_fill_never_mutates_last_episode_id(tmp_path):
    """The worker fill is worker-safe: it does not mutate ``last_episode_id``
    (the orchestrator sets it on the main thread; the worker only reads the
    handed-in id and ``episode.full_text``)."""
    store = HippocampalStore(str(tmp_path / "db"))
    enc = _bare_encoder(store)
    enc.start_session()
    _mock_extract(enc)
    _mock_relations(enc)
    enc.last_episode_id = "ep_PREV"

    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    ep = enc.encode_messages_stub(msgs, "ep_424", embedder=_stub_embedder())
    enc.encode_messages_fill(ep, "ep_424")

    assert enc.last_episode_id == "ep_PREV", "fill mutated last_episode_id"
    store.close()


def test_stub_then_fill_matches_fused_encode_messages(tmp_path):
    """stub + fill populate the episode identically to the fused synchronous
    ``encode_messages`` when the extractors return the same outputs -- the
    encoder-level lossless guard (the store-level guard is test_store_stub_fill)."""
    store = HippocampalStore(str(tmp_path / "db"))
    enc = _bare_encoder(store)
    enc.start_session()
    _mock_extract(enc)
    _mock_relations(enc)

    msgs = [{"role": "user", "content": "what db?"},
            {"role": "assistant", "content": "Alice chose Postgres"}]

    # Fused path (the synchronous encode_messages): pre-allocate id, set
    # last_episode_id as the fused path does, store via encode_episode.
    id_a = store.next_episode_id()
    enc.last_episode_id = None
    ep_fused = enc.encode_messages(msgs, embedder=_stub_embedder())

    # Split path: stub + fill, store via content + edges.
    enc.last_episode_id = None
    id_b = store.next_episode_id()
    ep_stub = enc.encode_messages_stub(msgs, id_b, embedder=_stub_embedder())
    enc.encode_messages_fill(ep_stub, id_b)

    # The extraction-derived fields must match (ids differ by construction).
    for field in ("entities", "entity_classes", "topics", "tones", "decisions"):
        assert getattr(ep_stub, field) == getattr(ep_fused, field), field
    assert ep_stub.relations == ep_fused.relations
    assert ep_stub.state_assertions == ep_fused.state_assertions
    store.close()