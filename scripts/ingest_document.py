"""CLI: ingest (or delete) a document/record into the hippocampal memory.

The RAG-replacement ingestion entrypoint (task #17). Explicit only -- no file
watcher (user directive): a source is ingested by an explicit call, and re-
ingesting an already-ingested source UPDATES it in place (reuses its doc id,
hash-diffs its sections) rather than creating a duplicate. A document is
stored as a hot/cold split: small metadata + graph pointers in the memory
store; section bodies in the content-addressed cold blob store.

Actions (mutually exclusive):

* ``--source PATH``      ingest (or re-ingest) a source; prints
                         ``created doc_NNNNNN`` or ``updated doc_NNNNNN``.
* ``--delete DOC_ID``    explicit delete (real removal, no archive record);
                         shared blobs are refcount-decremented, not deleted.
* ``--gc-blobs``         sweep zero-refcount orphan blobs from the cold store.

Heavy extraction deps (GLiNER for entities/topics, Bonsai for relations) are
constructed LAZILY and only when available; if they are not installed (a CPU
dev box), the ingest runs structure-only (no entities/topics/relations) and
still produces a valid, retrievable Document -- re-ingest later on a box with
the models to fill them in via the upsert. Mirrors ``run_consolidation.py``
(argparse, sys.path insert, ASCII-only help).
"""

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a script (mirrors run_consolidation.py:
# insert the REPO ROOT so ``src`` resolves as a package, not ``scripts/src``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _build_store(db_path: str, doc_db: str):
    from src.memory.store import HippocampalStore

    cfg = {}
    if doc_db:
        cfg["document_db_path"] = doc_db
    return HippocampalStore(db_path, config=cfg or None)


def _maybe_extractors(extract: bool):
    """Construct GLiNER + Bonsai if available; return (gliner, bonsai).

    When ``extract`` is False or the heavy deps are missing, returns
    ``(None, None)`` so the pipeline runs structure-only. A missing dep prints
    a warning (so the operator knows extraction was skipped) but does NOT
    abort the ingest -- structure-only ingestion is still useful and re-
    ingest later fills the entities/topics in place.
    """
    if not extract:
        return None, None
    gliner = bonsai = None
    try:
        from src.encoding.gliner_extractor import GLiNERExtractor
        gliner = GLiNERExtractor()
    except Exception as exc:  # ImportError on a CPU box, or a model error
        print(f"warning: GLiNER unavailable, skipping entity/topic extraction: {exc}")
    try:
        from src.encoding.bonsai_relations import BonsaiRelationExtractor
        bonsai = BonsaiRelationExtractor()
    except Exception as exc:
        print(f"warning: Bonsai unavailable, skipping relation extraction: {exc}")
    return gliner, bonsai


def _maybe_embedder(store, extract: bool):
    """Construct the per-chunk section embedder if available; return it or None.

    Reuses the same vector backend episodes use (the in-DB WaveDB VectorLayer
    via ``WavedbVectorStore`` when the store opened one, else the FAISS
    ``VectorSearch`` sidecar) -- both expose ``.encode`` (lazy-loading the same
    bge-small model). The embedder produces one vector per section (the per-
    chunk doc-RAG semantic path). When ``extract`` is False (``--no-extract``) or
    the model/deps are missing, returns ``None`` so the pipeline runs structure-
    only (sections are findable via the graph axes, not the semantic fallback).
    """
    if not extract:
        return None
    try:
        if getattr(store, "vector_layer", None) is not None:
            from src.retrieval.wavedb_vector_store import WavedbVectorStore
            return WavedbVectorStore(store)
        from src.retrieval.vector_search import VectorSearch
        return VectorSearch(store)
    except Exception as exc:  # ImportError / model-load error on a CPU box
        print(f"warning: section embedder unavailable, skipping per-chunk vector indexing: {exc}")
        return None


def _ingest(args) -> int:
    store = _build_store(args.db, args.doc_db)
    try:
        from src.config import config
        from src.ingestion.chunker import HierarchicalChunker
        from src.ingestion.pipeline import UnifiedIngestionPipeline

        ic = config.ingestion
        chunker = HierarchicalChunker(
            max_section_tokens=ic.max_section_tokens,
            min_section_tokens=ic.min_section_tokens,
            semantic_split_threshold=ic.semantic_split_threshold,
        )
        gliner, bonsai = _maybe_extractors(not args.no_extract)
        embedder = _maybe_embedder(store, not args.no_extract)
        pipe = UnifiedIngestionPipeline(store, chunker=chunker)
        doc_id, created = pipe.ingest(
            args.source,
            source_type=args.type,
            extractor=gliner,
            relation_extractor=bonsai,
            embedder=embedder,
        )
        print(f"{'created' if created else 'updated'} {doc_id}")
        return 0
    finally:
        store.close()


def _delete(args) -> int:
    store = _build_store(args.db, args.doc_db)
    try:
        if store.delete_document(args.delete):
            print(f"deleted {args.delete}")
            return 0
        print(f"absent {args.delete}")
        return 1
    finally:
        store.close()


def _gc(args) -> int:
    store = _build_store(args.db, args.doc_db)
    try:
        removed = store.gc_blobs()
        print(f"gc removed {removed} orphan blob(s)")
        return 0
    finally:
        store.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ingest (or delete) a document/record into hippocampal memory.",
    )
    ap.add_argument("--db", default="./data/memory_db",
                    help="memory (hot) WaveDB store directory")
    ap.add_argument("--doc-db", default=None,
                    help="document (cold) blob store directory (default: sibling of --db)")
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--source", help="ingest (or re-ingest) this source path")
    sub.add_argument("--delete", metavar="DOC_ID", help="explicitly delete a document id")
    sub.add_argument("--gc-blobs", action="store_true",
                     help="sweep zero-refcount orphan blobs from the cold store")
    ap.add_argument("--type", default="auto",
                    choices=["auto", "markdown", "text", "pdf", "code", "docx", "web"],
                    help="source type: auto|markdown|text|pdf|code|docx|web (default: auto by extension)")
    ap.add_argument("--no-extract", action="store_true",
                    help="skip GLiNER/Bonsai extraction (structure-only ingest)")
    args = ap.parse_args()

    if args.gc_blobs:
        return _gc(args)
    if args.delete:
        return _delete(args)
    if args.source:
        return _ingest(args)
    ap.error("no action specified")  # unreachable: group is required
    return 2


if __name__ == "__main__":
    raise SystemExit(main())