"""FAISS-backed vector search over episode summary embeddings.

Phase 1b Phase F: a semantic fallback for the retriever. ``VectorSearch``
embeds episode summaries with a **local sentence-transformers model**
(``config.embedding_model = BAAI/bge-small-en-v1.5``, 384-dim) — NOT OpenAI —
and builds a L2-normalized inner-product (cosine) index for nearest-neighbor
lookup. Embeddings are persisted by the store under ``content/ep/{eid}/embedding``
(Phase A); ``build_index`` reads them back, falling back to encoding on the fly
if an episode has no persisted embedding yet.

Backends:
- **faiss** (``IndexFlatIP``) when available — used on the pod for the full
  corpus.
- **pure-Python cosine** fallback (no faiss / no numpy) — keeps offline tests
  dependency-free and handles small corpora.

The embedder is any object with ``encode(texts: list[str]) -> list[list[float]]``.
Tests pass a deterministic stub; the pod uses ``sentence_transformers
.SentenceTransformer(config.embedding_model)``.
"""

from __future__ import annotations

import json
import math
from typing import Optional, Protocol

from ..config import config


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


def _sentence_transformers_embedder() -> Embedder:
    """Lazy-load the local sentence-transformers model (pod only)."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer(config.embedding_model)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorSearch:
    """Cosine nearest-neighbor over episode summary embeddings.

    ``store`` is a ``HippocampalStore``. ``embedder`` defaults to the local
    sentence-transformers model (lazy-loaded); pass a stub for tests. ``dim``
    is inferred from the first embedded vector if omitted.
    """

    INDEX_NAME = "vector_index.faiss"
    IDS_NAME = "vector_index_ids.json"

    def __init__(self, store, embedder: Optional[Embedder] = None, dim: Optional[int] = None) -> None:
        self.store = store
        self.embedder = embedder
        self.dim = dim
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []  # normalized
        self._faiss_index = None  # set by _rebuild_faiss when faiss is available

    # ── embedder ──

    def _get_embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = _sentence_transformers_embedder()
        return self.embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._get_embedder().encode(texts)
        out: list[list[float]] = []
        for v in vecs:
            # sentence-transformers (and faiss) return numpy float32 arrays;
            # convert to Python floats so json.dumps in set_summary_embedding
            # works. Pure-Python stub embedders already return Python floats.
            if hasattr(v, "tolist"):
                out.append([float(x) for x in v.tolist()])
            else:
                out.append([float(x) for x in v])
        return out

    # ── episode enumeration ──

    def _all_episode_ids(self) -> list[str]:
        """All episode ids via a content/ep/ scan (sorted, deduped).

        Delegates to ``store.default_episode_ids`` so abstracted episodes
        (Phase 3a semantic memories) are excluded from the default retrieval
        candidate set (spec §371). Pass ``include_abstracted=True`` on the
        store helper to opt in.
        """
        return self.store.default_episode_ids(include_abstracted=False)

    def _summary_for(self, eid: str) -> str:
        ep = self.store.get_episode(eid)
        return (ep.summary or "") if ep else ""

    # ── index build / load / save ──

    def build_index(self) -> int:
        """Embed every episode summary (persisted or on-the-fly) + build the index.

        Reads persisted embeddings from ``content/ep/{eid}/embedding`` first;
        any episode without one is embedded on the fly AND persisted (so the
        build is idempotent across re-runs). Episodes with empty summaries are
        skipped (no signal to embed). Returns the number of episodes indexed.
        """
        import json
        ids = self._all_episode_ids()
        kept = [(eid, self._summary_for(eid)) for eid in ids]
        kept = [(eid, s) for eid, s in kept if s]
        if not kept:
            self._ids, self._vectors = [], []
            self._faiss_index = None
            return 0

        cached: list[list[float]] = []
        to_embed: list[tuple[int, str]] = []  # (position, text)
        for pos, (eid, summary) in enumerate(kept):
            raw = self.store.db.get_sync(f"content/ep/{eid}/embedding")
            if raw:
                try:
                    cached.append(json.loads(raw))
                    continue
                except (json.JSONDecodeError, TypeError):
                    pass  # corrupt → re-embed below
            cached.append([])  # placeholder, filled after batch embed
            to_embed.append((pos, summary))

        if to_embed:
            new_vecs = self._embed([s for _, s in to_embed])
            for (pos, _), vec in zip(to_embed, new_vecs):
                cached[pos] = vec
                # Persist so the next build_index reuses it.
                self.store.set_summary_embedding(kept[pos][0], vec)

        if self.dim is None:
            self.dim = len(cached[0]) if cached else 0
        self._vectors = [_l2_normalize(v) for v in cached]
        self._ids = [eid for eid, _ in kept]
        self._rebuild_faiss()
        return len(self._ids)

    def _rebuild_faiss(self) -> None:
        """If faiss is available, mirror self._vectors into an IndexFlatIP."""
        self._faiss_index = None
        if not self._vectors:
            return
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return
        dim = len(self._vectors[0])
        index = faiss.IndexFlatIP(dim)
        arr = np.array(self._vectors, dtype="float32")
        index.add(arr)
        self._faiss_index = index

    def save(self, db_path: str) -> None:
        """Persist the index + id list under the db directory.

        With faiss, writes a real .faiss index. Without faiss, writes the
        vectors as JSON so the pure-Python backend can reload them.
        """
        from pathlib import Path
        base = Path(db_path)
        base.mkdir(parents=True, exist_ok=True)
        with open(base / self.IDS_NAME, "w", encoding="utf-8") as f:
            json.dump(self._ids, f)
        if self._faiss_index is not None:
            import faiss  # type: ignore
            faiss.write_index(self._faiss_index, str(base / self.INDEX_NAME))
        else:
            # Pure-Python fallback: stash vectors in the ids file's sibling.
            with open(base / "vector_index_vectors.json", "w", encoding="utf-8") as f:
                json.dump(self._vectors, f)

    def load(self, db_path: str) -> int:
        """Load a persisted index. Returns the number of indexed episodes."""
        from pathlib import Path
        base = Path(db_path)
        ids_path = base / self.IDS_NAME
        if not ids_path.exists():
            self._ids, self._vectors = [], []
            self._faiss_index = None
            return 0
        self._ids = json.loads(ids_path.read_text(encoding="utf-8"))
        faiss_path = base / self.INDEX_NAME
        if faiss_path.exists():
            try:
                import faiss  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "Vector index was built with faiss but faiss is not "
                    "installed; install faiss to load it."
                ) from e
            self._faiss_index = faiss.read_index(str(faiss_path))
            # Search uses the faiss index directly; _vectors stays empty. We
            # only need dim for any later re-embed of missing episodes.
            if self.dim is None:
                self.dim = self._faiss_index.d
            self._vectors = []
        else:
            vec_path = base / "vector_index_vectors.json"
            self._vectors = json.loads(vec_path.read_text(encoding="utf-8")) if vec_path.exists() else []
            if self.dim is None and self._vectors:
                self.dim = len(self._vectors[0])
        return len(self._ids)

    # ── search ──

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to k ``(episode_id, score)`` pairs, highest score first."""
        if not self._ids:
            return []
        qvec = _l2_normalize(self._embed([query])[0])
        if self._faiss_index is not None:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
            arr = np.array([qvec], dtype="float32")
            scores, idxs = self._faiss_index.search(arr, min(k, len(self._ids)))
            out = []
            for sc, ix in zip(scores[0].tolist(), idxs[0].tolist()):
                if ix < 0:
                    continue
                out.append((self._ids[ix], float(sc)))
            return out
        # Pure-Python fallback.
        scored = [(self._ids[i], _cosine(qvec, self._vectors[i])) for i in range(len(self._ids))]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]