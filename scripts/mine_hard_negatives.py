"""Mine HARD negatives for the STRM 2a relevance head (the GPU job).

WHY. ``scripts/generate_relevance_data.py`` samples negatives UNIFORM-RANDOM
from the 511K-doc ERAG pool (``rng.choice(non_gold, size=k_neg, replace=False)``
at ``build_records``). Random negatives are trivially unrelated to the query
(cosine ~= 0), so the 2a head never learns "close-but-wrong -> low" -> at serve
it scores ~0.9+ on EVERYTHING (Probe 3 selectivity NO-GO: probe-minus-filler
``r_i`` gap 0.006-0.187, gate needs >= 0.2). The fix is a pure data change: swap
random negatives for the non-gold docs CLOSEST to the query. This script
produces the ``{query_id: [hard_neg_doc_ids]}`` map; the generator consumes it.

WHAT. One-shot miner:
  1. Bulk-embed all 511K docs (raw ``title + "\\n" + content``, single string per
     doc -- the MINING embed, an approximate cosine proxy; the FAITHFUL
     parse+chunk+mean-pool ``embed_doc`` path is reserved for the small selected
     set in the generator). Bypasses the ``_STEmbedder`` wrapper to call
     ``SentenceTransformer.encode(batch_size=256, show_progress_bar=True,
     convert_to_numpy=True)`` directly (keeps float32 arrays, skips the slow
     ``list(map(float, row))`` conversion).
  2. Embed all 480 questions the same way.
  3. ``faiss.IndexFlatIP(384)`` over L2-normalized doc vectors (inner product =
     cosine). Per query: search the top ``k_neg + len(gold) + margin`` docs,
     drop the query's gold, take the next ``k_neg`` closest as hard negatives.
  4. Write ``{query_id: [doc_id, ...]}`` JSON (tiny -- SCP back from the pod).

RUN ON THE POD. 511K bge-small encodes in ~5 min on a 3090 (parse/read
overhead dominates -> ~15-40 min wall-clock). Well under $1 at $0.22/hr
(`AMPERE_24`), same pattern as Phase 2a. The downstream steps (traces regen,
retrain, Probe 3) are CPU-cheap and run locally.

Usage:
    python scripts/mine_hard_negatives.py \\
        --hard-neg-k 14 --out data/training/strm_relevance/hard_negatives.json
    # smoke (tiny, CPU-feasible):
    python scripts/mine_hard_negatives.py --max-queries 8 --limit-docs 2000 --out /tmp/hn.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_relevance_data import (  # noqa: E402
    GOLD_CATEGORIES,
    build_doc_index,
    load_questions,
    open_docs_table,
)

DEFAULT_QUESTIONS_PARQUET = "scripts/_scratch/erag/data/questions/test.parquet"
DEFAULT_DOCS_PARQUET = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_OUTPUT = "data/training/strm_relevance/hard_negatives.json"


def _resolve_device(device: str) -> str:
    """Resolve the SentenceTransformer device string."""
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _build_st(device: str):
    """Construct SentenceTransformer directly (bypasses the _STEmbedder wrapper
    so we can pass batch_size + show_progress_bar + keep float32 arrays)."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    from src.config import config as _config
    st = SentenceTransformer(_config.embedding_model, device=device)
    # bge-small-en-v1.5 max_seq_length is 512 by default; pin it explicitly so
    # long docs truncate deterministically rather than relying on the model card.
    try:
        st.max_seq_length = 512
    except Exception:
        pass
    return st


def _encode(st, texts: list[str], batch_size: int, desc: str) -> np.ndarray:
    """Batched encode -> [N, 384] float32, L2-normalized."""
    arr = st.encode(texts, batch_size=batch_size, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=False)
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _collect_doc_strings(docs_tbl, doc_ids: list[str]) -> list[str]:
    """Read ``title + "\\n" + content`` per doc id (memory-mapped, one-row slice
    per lookup). Returns the string list aligned with ``doc_ids`` order."""
    # Reuse the same row-index map build_doc_index produced (caller passes the
    # ordered id list; we re-derive row indices from the parquet row order, which
    # build_doc_index guarantees matches all_doc_ids order).
    out: list[str] = []
    n = len(doc_ids)
    for i in range(n):
        row = docs_tbl.slice(i, 1)
        title = row.column("title").to_pylist()[0]
        content = row.column("content").to_pylist()[0]
        t = str(title or "")
        c = str(content or "")
        text = (t + "\n" + c) if t else c
        out.append(text or doc_ids[i])
    return out


def mine(*, questions_path: str, docs_path: str, output: str,
         max_queries: int, hard_neg_k: int, batch_size: int, device: str,
         limit_docs: int, exclude_all_gold: bool, seed: int) -> dict:
    """Bulk-embed docs + questions, FAISS-mine hard negatives per query.

    Returns a stats dict (for logging). Writes the ``{query_id: [doc_id, ...]}``
    JSON to ``output``.
    """
    questions = load_questions(questions_path, GOLD_CATEGORIES, max_queries)
    if not questions:
        print(f"ERROR: no gold-bearing questions loaded from {questions_path}",
              file=sys.stderr)
        return {}
    all_gold = {d for q in questions for d in q["expected_doc_ids"]}
    print(f"  {len(questions)} questions, {len(all_gold)} unique gold doc_ids",
          flush=True)

    print(f"Indexing documents parquet {docs_path} (doc_id column only)",
          flush=True)
    doc_idx, all_doc_ids = build_doc_index(docs_path)
    if limit_docs > 0 and limit_docs < len(all_doc_ids):
        # Deterministic subset for smoke runs (first N docs -- not random, so the
        # smoke is reproducible; coverage will be lower but the shape is tested).
        all_doc_ids = all_doc_ids[:limit_docs]
        doc_idx = {d: i for i, d in enumerate(all_doc_ids)}
    print(f"  {len(all_doc_ids)} docs to embed", flush=True)

    dev = _resolve_device(device)
    print(f"Memory-mapping documents parquet title+content (lazy)", flush=True)
    docs_tbl = open_docs_table(docs_path)
    st = _build_st(dev)
    print(f"  embedder: {st._model_name if hasattr(st, '_model_name') else 'bge-small'} "
          f"on {dev}, max_seq_length={getattr(st, 'max_seq_length', '?')}", flush=True)

    t0 = time.time()
    print(f"Embedding {len(all_doc_ids)} docs (mining embed: raw text, single string)...",
          flush=True)
    doc_strings = _collect_doc_strings(docs_tbl, all_doc_ids)
    doc_vecs = _encode(st, doc_strings, batch_size, "docs")
    print(f"  doc_vecs {doc_vecs.shape} in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    q_texts = [q["question"] for q in questions]
    query_vecs = _encode(st, q_texts, batch_size, "questions")
    print(f"  query_vecs {query_vecs.shape} in {time.time() - t0:.1f}s", flush=True)

    print(f"Building faiss.IndexFlatIP(384) over {len(all_doc_ids)} docs...",
          flush=True)
    try:
        import faiss  # type: ignore
    except ImportError as e:
        print(f"ERROR: faiss not installed ({e}); pip install faiss-cpu (or faiss-gpu)",
              file=sys.stderr)
        return {}
    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)
    print(f"  index built, ntotal={index.ntotal}", flush=True)

    # Search deep enough that dropping gold still leaves hard_neg_k candidates.
    # margin guards against queries whose gold docs cluster at the very top.
    rng = np.random.default_rng(seed)
    hard_neg_map: dict[str, list[str]] = {}
    n_full = 0
    n_partial = 0
    n_missing = 0
    t0 = time.time()
    for qi, q in enumerate(questions):
        gold = set(q["expected_doc_ids"])
        exclusion = all_gold if exclude_all_gold else gold
        k_search = hard_neg_k + len(gold) + 10
        k_search = min(k_search, len(all_doc_ids))
        qv = np.ascontiguousarray(query_vecs[qi:qi + 1], dtype="float32")
        _, idxs = index.search(qv, k_search)
        picked: list[str] = []
        for raw_i in idxs[0].tolist():
            if raw_i < 0:
                continue
            doc_id = all_doc_ids[raw_i]
            if doc_id in exclusion:
                continue
            picked.append(doc_id)
            if len(picked) >= hard_neg_k:
                break
        if len(picked) >= hard_neg_k:
            n_full += 1
        elif picked:
            n_partial += 1
            # Backfill with random non-gold (rare -- only if the pool near the
            # query is gold-saturated; keeps the record shape stable for the
            # generator).
            need = hard_neg_k - len(picked)
            pool = [d for d in all_doc_ids if d not in exclusion
                    and d not in picked]
            if pool:
                backfill = list(rng.choice(pool, size=min(need, len(pool)),
                                           replace=False))
                picked.extend(backfill)
        else:
            n_missing += 1
            pool = [d for d in all_doc_ids if d not in exclusion]
            if pool:
                picked = list(rng.choice(pool, size=min(hard_neg_k, len(pool)),
                                          replace=False))
        hard_neg_map[q["question_id"]] = picked
        if (qi + 1) % 50 == 0:
            print(f"  mined {qi + 1}/{len(questions)} queries "
                  f"({time.time() - t0:.1f}s)", flush=True)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hard_neg_map, f, ensure_ascii=False, indent=2)

    stats = {
        "n_queries": len(questions), "n_docs_embedded": len(all_doc_ids),
        "hard_neg_k": hard_neg_k, "exclude_all_gold": exclude_all_gold,
        "n_full": n_full, "n_partial": n_partial, "n_missing": n_missing,
        "device": dev, "output": str(out_path),
    }
    print(f"DONE. mined {len(hard_neg_map)} queries -> {out_path}", flush=True)
    print(f"  full={n_full} partial={n_partial} missing={n_missing} "
          f"(k={hard_neg_k}, excl_all_gold={exclude_all_gold})", flush=True)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(
        description="Mine hard negatives for the STRM 2a relevance head (GPU job)")
    p.add_argument("--questions-parquet", default=DEFAULT_QUESTIONS_PARQUET)
    p.add_argument("--docs-parquet", default=DEFAULT_DOCS_PARQUET)
    p.add_argument("--out", default=DEFAULT_OUTPUT)
    p.add_argument("--max-queries", type=int, default=480)
    p.add_argument("--hard-neg-k", type=int, default=14,
                   help="hard negatives per query (default matches --neg-per-query)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto", help="cpu|cuda|auto")
    p.add_argument("--limit-docs", type=int, default=0,
                   help="smoke: embed only the first N docs (0 = all 511K)")
    p.add_argument("--exclude-all-gold", action="store_true",
                   help="also exclude docs gold for OTHER queries (default off)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    qpath = Path(args.questions_parquet)
    dpath = Path(args.docs_parquet)
    for pth, label in ((qpath, "questions"), (dpath, "documents")):
        if not pth.exists():
            print(f"ERROR: {label} not found at {pth}", file=sys.stderr)
            return 1

    stats = mine(
        questions_path=str(qpath), docs_path=str(dpath), output=args.out,
        max_queries=args.max_queries, hard_neg_k=args.hard_neg_k,
        batch_size=args.batch_size, device=args.device,
        limit_docs=args.limit_docs, exclude_all_gold=args.exclude_all_gold,
        seed=args.seed,
    )
    if not stats:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())