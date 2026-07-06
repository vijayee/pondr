"""Build + persist the FAISS vector index over episode summary embeddings.

Phase F on-pod step. ``VectorSearch.build_index`` reads persisted embeddings
from ``content/ep/{eid}/embedding`` first, and embeds (with the local
sentence-transformers model, ``config.embedding_model = BAAI/bge-small-en-v1.5``)
+ persists any episode that doesn't yet have one. It then builds a L2-
normalized cosine index (``faiss.IndexFlatIP`` on the pod, or the pure-Python
fallback offline) and persists it under the db directory.

Run on the pod (GPU embeddings) after ``process_corpus.py`` has ingested a
corpus:

    python scripts/build_vector_index.py --db /workspace/volumes/hippo/memory_db

Offline (no sentence-transformers / no faiss) the script works against a small
store if ``WAVEDB_LIB_PATH`` etc. are set and a stub embedder is injected — the
test suite exercises ``VectorSearch`` directly with a stub instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import HippocampalStore  # noqa: E402
from src.retrieval.vector_search import VectorSearch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build + persist the episode vector index.")
    parser.add_argument("--db", required=True, help="WaveDB store path (ingested corpus).")
    args = parser.parse_args()

    store = HippocampalStore(args.db)
    try:
        vs = VectorSearch(store)
        n = vs.build_index()
        print(f"Indexed {n} episodes.")
        vs.save(args.db)
        print(f"Saved index → {Path(args.db) / VectorSearch.IDS_NAME}")
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())