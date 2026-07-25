"""Probe: does raw bge cosine(query, doc) clear the 2a top-3 gate on ERAG?

Isolates WHERE the relevance signal is lost. The 2a head reads the backbone's
256-d slot readout y_t (a lossy projection of the doc's 384-d bge embedding
through the frozen routing-trained backbone). If bge cosine ALREADY clears the
gate, the signal exists in bge space and the backbone projection destroys it
-> the head needs the raw doc bge, not y_t. If bge cosine ALSO fails, ERAG gold
docs are not bge-retrievable by the question (a data/task issue, not a head
issue).

Re-embeds the questions_meta.jsonl candidates with bge-small (no backbone) and
computes per-query top-3 recall + Wilson CI on the same split the trainer uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.generate_relevance_data import (  # noqa: E402
    build_doc_index,
    get_doc,
    load_questions,
    open_docs_table,
    embed_doc,
)
from src.ingestion.chunker import HierarchicalChunker  # noqa: E402
from src.ingestion.parsers import MarkdownParser  # noqa: E402
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    _split_queries,
    _wilson_ci95,
)
from src.subconscious.training.routing_training import build_embedder  # noqa: E402


def _cos_top3_recall(qvec, doc_vecs, labels):
    # qvec [384], doc_vecs [K,384], labels [K] (1=gold)
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    sims = d @ q
    gold_idx = [i for i, l in enumerate(labels) if l == 1]
    n_gold = len(gold_idx)
    if n_gold == 0:
        return None
    k_top = min(3, len(sims))
    top = set(np.argsort(-sims)[:k_top].tolist())
    n_in = sum(1 for i in gold_idx if i in top)
    return n_in / n_gold, n_in == n_gold


def main() -> int:
    qpath = "scripts/_scratch/erag/data/questions/test.parquet"
    dpath = "scripts/_scratch/erag/data/documents/test.parquet"
    questions = load_questions(qpath, [
        "basic", "semantic", "intra_document_reasoning", "project_related",
        "conflicting", "completeness", "miscellaneous",
    ], 80)
    # NOTE: 'constrained' intentionally omitted to match a quick re-run; add back
    # if you want the full 8. Actually include all 8 for parity:
    from scripts.generate_relevance_data import GOLD_CATEGORIES
    questions = load_questions(qpath, GOLD_CATEGORIES, 80)
    print(f"loaded {len(questions)} questions")
    doc_idx, all_ids = build_doc_index(dpath)
    tbl = open_docs_table(dpath)
    embedder = build_embedder("on-demand")
    parser = MarkdownParser()
    chunker = HierarchicalChunker()
    rng = np.random.default_rng(0)
    gold_all = {d for q in questions for d in q["expected_doc_ids"]}
    non_gold = [d for d in all_ids if d not in gold_all]

    # rebuild the SAME candidate plans the generator built (seed 0, neg=14)
    plans = []
    for q in questions:
        gold = q["expected_doc_ids"]
        neg_ids = list(rng.choice(non_gold, size=min(14, len(non_gold)), replace=False))
        cand = list(gold) + [d for d in neg_ids if d not in gold]
        rng.shuffle(cand)
        plans.append((q, cand))

    # embed each candidate doc's RAW bge (mean-pool of sections) -- the input to
    # wm.step, NOT the backbone readout y_t.
    cache = {}
    def dvec(doc_id):
        if doc_id in cache:
            return cache[doc_id]
        tc = get_doc(tbl, doc_idx, doc_id)
        if tc is None:
            return None
        v = embed_doc(doc_id, tc[0], tc[1], parser, chunker, embedder, "cpu")
        arr = v.squeeze(0).numpy().astype(np.float32)
        cache[doc_id] = arr
        return arr

    records = []
    for q, cand in plans:
        vecs = []
        labels = []
        for d in cand:
            v = dvec(d)
            if v is None:
                continue
            vecs.append(v)
            labels.append(1 if d in set(q["expected_doc_ids"]) else 0)
        if not vecs or sum(labels) == 0:
            continue
        qv = np.asarray(embedder.encode([q["question"]])[0], dtype=np.float32)
        records.append((qv, np.stack(vecs), labels))

    # same split as trainer
    train_idx, val_idx = _split_queries(len(records),
                                       RelevanceTrainingConfig().val_fraction, 0)
    val = [records[i] for i in val_idx]
    recalls = []
    hits = 0
    for qv, dv, lab in val:
        r = _cos_top3_recall(qv, dv, lab)
        if r is None:
            continue
        recalls.append(r[0])
        if r[1]:
            hits += 1
    mean_top3 = sum(recalls) / len(recalls)
    hit_rate = hits / len(recalls)
    ci = _wilson_ci95(hit_rate, len(recalls))
    print(f"\nbge-cosine baseline on {len(val)} val queries:")
    print(f"  mean_top3_recall = {mean_top3:.3f}  (gate 0.6)")
    print(f"  hit_rate = {hit_rate:.2f}  Wilson ci=[{ci[0]:.2f},{ci[1]:.2f}]  (gate low>0.5)")
    print(f"  -> {'GO' if mean_top3>=0.6 and ci[0]>0.5 else 'NO-GO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())