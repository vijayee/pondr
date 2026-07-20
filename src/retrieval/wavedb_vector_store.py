"""WaveDB VectorLayer adapter -- the in-DB vector index as a retriever backend.

Drop-in replacement for ``VectorSearch`` (the FAISS sidecar) when
``HippocampalStore.vector_layer`` is open. The store maintains the index live
(insert on ``encode_episode`` / ``set_summary_embedding``, delete on
forget/supersede), so this adapter does NO building -- it only embeds the query
and reads ``vector_layer.search_sync``.

Contract parity with ``VectorSearch`` (so the retriever's ``self.vector_search``
slot is unchanged):

* ``search(query: str, k: int = 5) -> list[tuple[str, float]]`` -- text in,
  ``[(episode_id, score)]`` out, **highest similarity first**. The WaveDB
  COSINE metric reports a *distance* = ``1 - cosine_similarity`` (lower =
  closer, best-first from the C layer); ``VectorSearch`` returns a
  *similarity* (higher = closer) and the retriever sorts
  ``reverse=True``. So the adapter converts ``score = 1.0 - distance`` to
  keep the retriever's ranking direction correct.
* ``encode(texts)`` -- satisfies the ``Embedder`` protocol, so the retriever's
  gate-embedder reuse (``retriever.py`` treats ``self.vector_search`` as an
  ``Embedder`` when no route embedder is passed) keeps working.

The embedder defaults to the same lazy-loaded local sentence-transformers
model ``VectorSearch`` uses (``config.embedding_model``); tests inject a stub
and set ``adapter.embedder`` directly (the existing pattern).
"""

from __future__ import annotations

from typing import Optional

from .vector_search import Embedder, _sentence_transformers_embedder


class WavedbVectorStore:
    """Read-only adapter over ``store.vector_layer`` (the in-DB vector index).

    ``store`` is a ``HippocampalStore`` with ``vector_layer`` open. ``embedder``
    defaults to the local sentence-transformers model (lazy-loaded); pass a
    stub for tests. The index itself is maintained by the store (live insert /
    delete), so there is no ``build_index`` work here -- ``build_index`` is a
    no-op returning the current count for API parity.
    """

    def __init__(self, store, embedder: Optional[Embedder] = None) -> None:
        self.store = store
        self.embedder = embedder

    # ── embedder (mirrors VectorSearch so tests can set .embedder directly) ──

    def _get_embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = _sentence_transformers_embedder()
        return self.embedder

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Satisfy the ``Embedder`` protocol (gate-embedder reuse in the retriever)."""
        return self._get_embedder().encode(texts)

    def _embed_query(self, query: str) -> list[float]:
        vec = self._get_embedder().encode([query])[0]
        # sentence-transformers / faiss return numpy float32 arrays; convert to
        # Python floats (the C layer takes a float[] either way, but keep parity
        # with VectorSearch._embed). Pure-Python stubs already return floats.
        if hasattr(vec, "tolist"):
            return [float(x) for x in vec.tolist()]
        return [float(x) for x in vec]

    # ── search ──

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Nearest episodes to ``query`` as ``[(episode_id, similarity)]``, best first.

        Returns ``[]`` when the in-DB vector layer is absent/disabled (the
        retriever then degrades to graph-only, same as a missing FAISS index).
        """
        vl = getattr(self.store, "vector_layer", None)
        if vl is None:
            return []
        vec = self._embed_query(query)
        results = vl.search_sync(vec, k)
        # WaveDB COSINE distance = 1 - cosine_similarity (lower = closer, best
        # first from the C layer). Convert to similarity (higher = closer) to
        # match VectorSearch's contract + the retriever's reverse=True sort.
        return [(r.id_str, 1.0 - float(r.distance)) for r in results]

    def search_by_vector(self, vec: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Nearest episodes to a PRE-COMPUTED ``vec`` (no re-embed), best first.

        STRM Phase 4 Step 5: the salience trigger fires retrieval with a
        state-conditioned query (the anchor's 384-d doc vector), not a text
        prompt, so the embed step is skipped. Same distance->similarity
        conversion + empty-when-no-layer contract as ``search``.
        """
        vl = getattr(self.store, "vector_layer", None)
        if vl is None:
            return []
        vec = [float(x) for x in vec]
        results = vl.search_sync(vec, k)
        return [(r.id_str, 1.0 - float(r.distance)) for r in results]

    # ── API-parity no-ops (the index is maintained live by the store) ──

    def build_index(self) -> int:
        """No-op: the in-DB index is maintained live by the store chokepoint.

        Returns the current vector count for API parity with ``VectorSearch``
        (which returns the number of indexed episodes). The offline bulk
        backfill of pre-existing episodes goes through ``set_summary_embedding``
        in the store, which dual-writes into this layer for free.
        """
        vl = getattr(self.store, "vector_layer", None)
        return vl.count() if vl is not None else 0