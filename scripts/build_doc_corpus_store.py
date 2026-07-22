"""Phase 1f-1: build a PERSISTED document corpus store for the doc-ring retrain.

The conv-pair mixed ring (Phase 1c) is ill-posed for the cross-slot Transformer
because its two slot types are content-overlapping (a prior message ``u_j``
appears both as a type-0 conversation slot and as a type-1 retrieved episode
``u_j + a_j``) -- single-argmax gold is contradictory and the transformer
overfits the tie-breaking noise (see [[pondr-strm-phase1d-self-match-rootcause]],
DeepSeek's section-2 diagnosis). Phase 1f replaces that content-overlapping
retrieved pool with REAL documents (this repo's .md/.py/.txt) so the two slot
types are GENUINELY distinct (chat messages vs external docs) -> single-argmax
gold becomes well-posed -> the transformer's type-conditioned advantage should
re-emerge.

This script ingests a glob of repo files into a PERSISTED WaveDB store at
``data/training/strm_relevance/doc_corpus_store/`` via the EXISTING production
pipeline ``UnifiedIngestionPipeline.ingest`` -> ``store.encode_document`` (NOT
``encode_episode``). Docs live under ``content/doc/{doc_id}/...`` and surface as
type-1 ring slots via ``HippocampalRetriever.retrieve`` /
``GraphTraversal._hydrate_document`` / ``_hydrate_section``, aggregated to one
slot per doc by ``DocumentRetriever`` (attached by ``build_ponder`` when
``store_has_documents(store)``). The store persists; both the 1f generator
(``generate_onyx_doc_ring_traces.py``) and the acceptance probe
(``probe_strm_selectivity_real.py --doc-store``) open it READ-ONLY. This mirrors
production (the personal doc corpus persists across conversations).

Required vs optional extractors (per the approved plan):
- GLiNER ``extractor`` + section ``embedder`` are REQUIRED -- entity/topic edges
  + section embeddings are what make docs retrievable. If either is unavailable
  the run still ingests structure-only but the final coverage assert
  (``n_docs_with_edges > 0``) fails loudly so a silently-empty / unfindable
  corpus is impossible to miss (de-wonk).
- Bonsai ``relation_extractor`` (needs local Ollama ``:11434``, ~22.8s/doc for
  the 10x isolation pass) is OPTIONAL and OFF by default -- relations are
  supplementary; docs stay findable via GLiNER entity/topic + semantic. Pass
  ``--relations`` to enable.
- ``doc_kind_tagger`` is OPTIONAL and skipped (``doc_kind`` stays the cold-start
  ``"other"`` default; the tag is for the complementary-temporal guard, not
  retrieval) -- mirrors ``ingest_document`` cold-start.

Standalone (never touches ``DEFAULT_BACKBONE_PATH`` / ``build_ponder`` /
``serve_ponder``). Re-ingesting an existing store upserts by ``source_path``
(reuses the doc id, hash-diffs sections) -- idempotent, never duplicates.
"""
import argparse
import glob
import sys
from pathlib import Path

# Make ``src`` importable when run as a script (mirrors ingest_document.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows cp1252 console cannot encode some emoji GLiNER prints during model
# load (e.g. the brain char) -> UnicodeEncodeError -> GLiNERExtractor()
# construction raises -> the extractor is silently dropped -> docs ingest with
# no entity/topic edges (unfindable by graph traversal). Reconfigure stdio to
# UTF-8 so the extractor loads. No-op on POSIX (already UTF-8).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _build_store(db_path: str):
    from src.memory.store import HippocampalStore
    return HippocampalStore(db_path)


def _maybe_gliner():
    """Construct the GLiNER entity/topic extractor if available, else None.

    GLiNER is REQUIRED for docs to be findable by the graph entity/topic axes;
    a None return prints a loud warning (the run will still ingest structure-
    only but the final ``n_docs_with_edges`` assert will fail -- by design, so a
    broken extractor is visible, not silent).
    """
    try:
        from src.encoding.gliner_extractor import GLiNERExtractor
        return GLiNERExtractor()
    except Exception as exc:  # ImportError on a CPU box, or a model error
        print(f"WARNING: GLiNER unavailable -- docs will ingest structure-only "
              f"(entity/topic edges missing -> unfindable by graph traversal): "
              f"{exc}", file=sys.stderr)
        return None


def _maybe_bonsai():
    """Construct the Bonsai relation extractor if available, else None.

    OPTIONAL (relations are supplementary). Construction is best-effort; the
    pipeline's own try/except guards the per-doc ``extract`` call, so a server
    that is up at construction but drops mid-run degrades to ``[]`` per doc
    rather than aborting.
    """
    try:
        from src.encoding.bonsai_relations import BonsaiRelationExtractor
        return BonsaiRelationExtractor()
    except Exception as exc:
        print(f"warning: Bonsai unavailable, skipping relation extraction: {exc}",
              file=sys.stderr)
        return None


def _maybe_embedder(store):
    """Construct the per-chunk section embedder if available, else None.

    Reuses the same vector backend episodes use (the in-DB WaveDB VectorLayer
    via ``WavedbVectorStore`` when the store opened one, else the FAISS
    ``VectorSearch`` sidecar) -- both expose ``.encode``. REQUIRED for the
    semantic-fallback retrieval path; None -> loud warning.
    """
    try:
        if getattr(store, "vector_layer", None) is not None:
            from src.retrieval.wavedb_vector_store import WavedbVectorStore
            return WavedbVectorStore(store)
        from src.retrieval.vector_search import VectorSearch
        return VectorSearch(store)
    except Exception as exc:
        print(f"WARNING: section embedder unavailable -- per-chunk vector "
              f"indexing skipped (semantic retrieval path missing): {exc}",
              file=sys.stderr)
        return None


def _expand_docs(docs_globs: list[str], max_docs: int) -> list[str]:
    """Expand comma-separated glob patterns to a deduped, sorted file list.

    Each ``--docs-glob`` value may itself be comma-separated. Recursive globs
    (``**``) require ``recursive=True``. Dedupes (a file may match two patterns)
    and caps at ``max_docs`` when > 0. Returns absolute, sorted paths.
    """
    seen: set[str] = set()
    out: list[str] = []
    for pattern in docs_globs:
        for sub in pattern.split(","):
            sub = sub.strip()
            if not sub:
                continue
            for path in glob.glob(sub, recursive=True):
                p = Path(path).resolve()
                if not p.is_file():
                    continue
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
    out.sort()
    if max_docs and len(out) > max_docs:
        out = out[:max_docs]
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 1f-1: build a persisted doc corpus store for the "
                    "doc-ring retrain (task #55).")
    p.add_argument("--docs-glob", nargs="+",
                   default=["docs/**/*.md", "README.md", "src/**/*.py", "CLAUDE.md"],
                   help="one or more glob patterns (each may be comma-separated) "
                        "of repo files to ingest. Default ingests repo .md + .py.")
    p.add_argument("--max-docs", type=int, default=0,
                   help="cap the number of docs ingested (0 = all; dev knob -- "
                        "the 1f-0 smoke uses ~20).")
    p.add_argument("--store",
                   default="data/training/strm_relevance/doc_corpus_store",
                   help="persisted WaveDB store directory (the 1f generator + "
                        "the acceptance probe open this read-only).")
    p.add_argument("--device", default="auto",
                   help="device for GLiNER (auto|cpu|cuda; default auto).")
    p.add_argument("--relations", action="store_true",
                   help="enable Bonsai doc-level relation extraction (needs local "
                        "Ollama :11434; ~22.8s/doc). OFF by default -- relations "
                        "are supplementary; docs stay findable via GLiNER + "
                        "semantic.")
    p.add_argument("--no-extract", action="store_true",
                   help="skip GLiNER + embedder too (structure-only ingest). NOT "
                        "recommended -- the final n_docs_with_edges assert will "
                        "fail. Provided for parity with ingest_document.py.")
    p.add_argument("--no-gliner", action="store_true",
                   help="skip GLiNER entity/topic extraction but KEEP the section "
                        "embedder. Workaround for the WaveDB memory_pool double-"
                        "free ([[wavedb-memory-pool-doublefree]]): GLiNER entity/"
                        "topic edges grow the encode_document batch large enough "
                        "to trigger ``batch_sync failed`` on entity-dense docs; "
                        "with GLiNER off the batch stays small (section + has_section "
                        "+ doc metadata only) and succeeds, while the embedder "
                        "keeps docs findable via the semantic fallback. The final "
                        "assert uses ``store_has_documents`` (the has_section edge) "
                        "so it still passes with GLiNER off.")
    p.add_argument("--no-state-assertions", action="store_true",
                   help="skip the Phase 3c deterministic state-assertion normalizer "
                        "(key:value / change-verb edges). Second memory_pool lever: "
                        "assertion-dense docs (a 60-section spec with 200+ state "
                        "claims) grow the encode_document batch large enough to "
                        "trigger ``batch_sync failed`` EVEN with --no-gliner; "
                        "skipping the assertions shrinks the batch to section + "
                        "has_section + doc-metadata volume and lets the doc ingest. "
                        "Assertions are SUPPLEMENTARY for retrieval (docs stay "
                        "findable via GLiNER + the semantic embedder), so this is a "
                        "safe corpus-build workaround -- production ingest keeps "
                        "them on. Combine with --no-gliner for the smallest batch.")
    args = p.parse_args()

    store_path = Path(args.store)
    store_path.mkdir(parents=True, exist_ok=True)
    print(f"doc corpus store: {store_path}", flush=True)

    files = _expand_docs(args.docs_glob, args.max_docs)
    print(f"  {len(files)} file(s) to ingest"
          + (f" (capped at --max-docs {args.max_docs})" if args.max_docs else ""),
          flush=True)
    if not files:
        print("ERROR: no files matched --docs-glob", file=sys.stderr)
        return 1

    store = _build_store(str(store_path))
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

        if args.no_extract:
            gliner = None
            embedder = None
            print("  --no-extract: structure-only ingest (no GLiNER, no embeddings)",
                  flush=True)
        elif args.no_gliner:
            gliner = None
            embedder = _maybe_embedder(store)
            print("  --no-gliner: GLiNER OFF (memory_pool workaround), embedder ON "
                  "(semantic findability)", flush=True)
        else:
            gliner = _maybe_gliner()
            embedder = _maybe_embedder(store)
        bonsai = _maybe_bonsai() if args.relations else None
        if args.relations and bonsai is None:
            print("  --relations set but Bonsai unavailable; proceeding without "
                  "relation extraction", file=sys.stderr)
        # doc_kind_tagger intentionally None (cold-start "other"; the tag is for
        # the complementary-temporal guard, not retrieval). Mirrors ingest_document
        # cold-start. de-wonk: do NOT fabricate a doc_kind.
        doc_kind_tagger = None

        pipe = UnifiedIngestionPipeline(store, chunker=chunker)

        n_ok = 0
        n_skip = 0
        n_sections_total = 0
        doc_ids: list[str] = []
        for i, fpath in enumerate(files, 1):
            try:
                doc_id, created = pipe.ingest(
                    fpath,
                    source_type="auto",
                    extractor=gliner,
                    relation_extractor=bonsai,
                    embedder=embedder,
                    doc_kind_tagger=doc_kind_tagger,
                    state_assertions=not args.no_state_assertions,
                )
            except Exception as exc:  # noqa: BLE001 - one bad doc must not kill the build
                print(f"  [{i}/{len(files)}] SKIP {fpath}: {exc}", file=sys.stderr)
                n_skip += 1
                continue
            n_ok += 1
            doc_ids.append(doc_id)
            # Section count via metadata-only (no cold pull) for the coverage sum.
            try:
                doc = store.get_document(doc_id, load_bodies=False)
                n_secs = len(doc.sections) if doc is not None else 0
            except Exception:  # noqa: BLE001 - coverage is best-effort
                n_secs = 0
            n_sections_total += n_secs
            print(f"  [{i}/{len(files)}] {'created' if created else 'updated'} "
                  f"{doc_id}  ({n_secs} sections)  <- {fpath}", flush=True)

        # Coverage summary + the findability assert. ``store_has_documents`` is
        # the SAME probe ``build_ponder`` uses to decide whether to attach a
        # DocumentRetriever -- if it is False, the 1f generator's retriever will
        # NOT aggregate and the acceptance probe's ring will have NO doc slots ->
        # the whole phase is invalid. Fail loudly here rather than downstream.
        from src.retrieval.document_retriever import store_has_documents
        has_docs = store_has_documents(store)
        n_docs_with_edges = n_ok if has_docs else 0

        print()
        print("=" * 64)
        print(f"docs ingested:        {n_ok}")
        print(f"docs skipped:         {n_skip}")
        print(f"sections total:       {n_sections_total}")
        print(f"store_has_documents:  {has_docs}")
        print(f"docs with edges:      {n_docs_with_edges}")
        print(f"GLiNER:               {'on' if gliner is not None else 'off'}")
        print(f"embedder:             {'on' if embedder is not None else 'off (no semantic path)'}")
        print(f"Bonsai relations:     {'on' if bonsai is not None else 'off'}")
        print("=" * 64)

        summary = {
            "store": str(store_path),
            "n_files_matched": len(files),
            "n_docs_ingested": n_ok,
            "n_docs_skipped": n_skip,
            "n_sections_total": n_sections_total,
            "store_has_documents": has_docs,
            "n_docs_with_edges": n_docs_with_edges,
            "gliner": gliner is not None,
            "embedder": embedder is not None,
            "bonsai_relations": bonsai is not None,
            "state_assertions": not args.no_state_assertions,
            "doc_ids": doc_ids,
        }
        summary_path = store_path / "build_summary.json"
        summary_path.write_text(
            __import__("json").dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote coverage summary -> {summary_path}", flush=True)

        if n_ok == 0:
            print("ERROR: no docs ingested successfully", file=sys.stderr)
            return 1
        if n_docs_with_edges == 0:
            print("ERROR: n_docs_with_edges == 0 -- docs are NOT findable by "
                  "graph traversal (GLiNER missing? embedder missing? empty "
                  "files?). The 1f generator would surface zero doc slots. "
                  "Fix the extractor/embedder and re-run (idempotent upsert).",
                  file=sys.stderr)
            return 1
        print("OK: doc corpus store is findable.", flush=True)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())