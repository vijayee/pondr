"""Tests for the STRM Phase 2a relevance-head data generator.

CPU-only. No real ERAG parquet, no 19.5M backbone checkpoint, no bge-small
download: the parquet helpers are exercised on tiny synthetic parquets, and
``build_records`` runs end-to-end on a fresh random-init ``JGSBackbone`` (the
shapes + label logic are what we're validating, not the trained weights) with a
deterministic stub embedder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_relevance_data import (  # noqa: E402
    GOLD_CATEGORIES,
    build_doc_index,
    build_records,
    embed_doc,
    get_doc,
    load_questions,
    open_docs_table,
)
from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.backbone import JGSBackbone  # noqa: E402
from src.ingestion.chunker import HierarchicalChunker  # noqa: E402
from src.ingestion.parsers import MarkdownParser  # noqa: E402


# ── synthetic parquet helpers ──

def _write_questions_parquet(path: Path, rows: list[dict]) -> None:
    tbl = pa.table({
        "question_id": [r["question_id"] for r in rows],
        "question_type": [r["question_type"] for r in rows],
        "question": [r["question"] for r in rows],
        "expected_doc_ids": [r["expected_doc_ids"] for r in rows],
        "gold_answer": [r.get("gold_answer", "") for r in rows],
        "answer_facts": [r.get("answer_facts", []) for r in rows],
    })
    pq.write_table(tbl, path)


def _write_docs_parquet(path: Path, rows: list[dict]) -> None:
    tbl = pa.table({
        "doc_id": [r["doc_id"] for r in rows],
        "source_type": [r.get("source_type", "confluence") for r in rows],
        "title": [r.get("title", "") for r in rows],
        "content": [r.get("content", "") for r in rows],
    })
    pq.write_table(tbl, path)


class _StubEmbedder:
    """Deterministic 384-d embedder: a per-text hash -> vector.

    Different texts get different vectors (so the SSM steps produce
    distinguishable y_t slots); the same text always maps to the same vector
    (deterministic). NOT bge-small -- the values are meaningless, only the
    shapes + label logic are under test.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(h)
            v = rng.standard_normal(384).astype(np.float32)
            out.append(v.tolist())
        return out


# ── load_questions ──

def test_load_questions_filters_category_drops_empty_gold_caps(tmp_path):
    rows = [
        {"question_id": "qst_0001", "question_type": "basic",
         "question": "q1", "expected_doc_ids": ["d1"]},
        {"question_id": "qst_0002", "question_type": "semantic",
         "question": "q2", "expected_doc_ids": ["d2", "d3"]},
        # empty gold -> dropped
        {"question_id": "qst_0003", "question_type": "basic",
         "question": "q3", "expected_doc_ids": []},
        # excluded category (no ground truth) -> dropped
        {"question_id": "qst_0004", "question_type": "high_level",
         "question": "q4", "expected_doc_ids": ["d4"]},
        {"question_id": "qst_0005", "question_type": "miscellaneous",
         "question": "q5", "expected_doc_ids": ["d5"]},
    ]
    p = tmp_path / "q.parquet"
    _write_questions_parquet(p, rows)
    qs = load_questions(str(p), GOLD_CATEGORIES, max_queries=480)
    ids = [q["question_id"] for q in qs]
    assert ids == ["qst_0001", "qst_0002", "qst_0005"]   # 0003 + 0004 dropped
    # max_queries cap
    assert len(load_questions(str(p), GOLD_CATEGORIES, max_queries=2)) == 2
    # expected_doc_ids kept as plain str
    assert qs[1]["expected_doc_ids"] == ["d2", "d3"]
    assert all(isinstance(d, str) for q in qs for d in q["expected_doc_ids"])


# ── doc index + get_doc ──

def test_build_doc_index_and_get_doc(tmp_path):
    rows = [
        {"doc_id": "d1", "title": "t1", "content": "alpha", "source_type": "slack"},
        {"doc_id": "d2", "title": "t2", "content": "beta", "source_type": "jira"},
        {"doc_id": "d3", "title": "t3", "content": "gamma", "source_type": "github"},
    ]
    p = tmp_path / "docs.parquet"
    _write_docs_parquet(p, rows)
    idx, all_ids = build_doc_index(str(p), {"d1"})
    assert all_ids == ["d1", "d2", "d3"]
    assert idx["d2"] == 1
    tbl = open_docs_table(str(p))
    assert get_doc(tbl, idx, "d2") == ("t2", "beta")
    assert get_doc(tbl, idx, "missing") is None     # unknown id -> None


# ── embed_doc ──

def test_embed_doc_mean_pools_sections_to_one_vector(tmp_path):
    embedder = _StubEmbedder()
    parser = MarkdownParser()
    chunker = HierarchicalChunker()
    # markdown with two headings -> two sections -> mean of two section vectors
    content = "# Heading A\nbody A para one\n\n# Heading B\nbody B para two"
    v = embed_doc("d1", "Doc Title", content, parser, chunker, embedder,
                  torch.device("cpu"))
    assert v.shape == (1, 384)
    assert v.dtype == torch.float32
    # the doc vector is the MEAN of the two section vectors (verifies pooling,
    # not just "first section"): reconstruct the same mean from the stub.
    parsed = parser.parse_text("Doc Title\n" + content, source_path="d1")
    parsed = chunker.chunk(parsed)
    sec_texts = [(s.heading + "\n" + s.content) if s.heading else s.content
                 for s in parsed.sections]
    vecs = np.asarray(embedder.encode(sec_texts), dtype=np.float32)
    expected = vecs.mean(axis=0)
    assert np.allclose(v.squeeze(0).numpy(), expected, atol=1e-5)


def test_embed_doc_cold_start_fallback_for_empty_content():
    embedder = _StubEmbedder()
    v = embed_doc("d1", "", "   \n   ", MarkdownParser(), HierarchicalChunker(),
                  embedder, torch.device("cpu"))
    assert v.shape == (1, 384)   # falls back to embedding the title/doc_id


# ── build_records end-to-end (fresh random backbone, stub embedder) ──

def _synthetic_questions() -> list[dict]:
    # shape mirrors load_questions() output (incl. the "category" key that
    # build_records reads), not the raw parquet row shape.
    return [
        {"question_id": "qst_0001", "category": "basic",
         "question": "question one about alpha",
         "expected_doc_ids": ["d1"]},
        {"question_id": "qst_0002", "category": "semantic",
         "question": "question two about beta gamma",
         "expected_doc_ids": ["d2", "d3"]},
        {"question_id": "qst_0003", "category": "basic",
         "question": "question three about delta",
         "expected_doc_ids": ["d4"]},
    ]


def _synthetic_docs() -> list[dict]:
    return [
        {"doc_id": "d1", "title": "alpha doc", "content": "alpha alpha alpha"},
        {"doc_id": "d2", "title": "beta doc", "content": "beta beta beta"},
        {"doc_id": "d3", "title": "gamma doc", "content": "gamma gamma gamma"},
        {"doc_id": "d4", "title": "delta doc", "content": "delta delta delta"},
        {"doc_id": "d5", "title": "epsilon doc", "content": "epsilon epsilon epsilon"},
        {"doc_id": "d6", "title": "zeta doc", "content": "zeta zeta zeta"},
    ]


def test_build_records_shapes_labels_and_no_leakage(tmp_path):
    docs = _synthetic_docs()
    dp = tmp_path / "docs.parquet"
    _write_docs_parquet(dp, docs)
    questions = _synthetic_questions()
    gold_ids = {d for q in questions for d in q["expected_doc_ids"]}
    idx, all_ids = build_doc_index(str(dp), gold_ids)
    tbl = open_docs_table(str(dp))

    backbone = JGSBackbone(BackboneConfig())   # fresh random init, CPU
    for p in backbone.parameters():
        p.requires_grad = False
    embedder = _StubEmbedder()
    parser = MarkdownParser()
    chunker = HierarchicalChunker()

    records, stats = build_records(
        questions, tbl, idx, all_ids, backbone, embedder, parser, chunker,
        neg_per_query=2, device=torch.device("cpu"), seed=0,
    )
    assert len(records) == 3
    assert stats["n_queries"] == 3
    # per-query shapes + label consistency
    by_q = {r["query_id"]: r for r in records}
    r1 = by_q["qst_0001"]
    assert r1["query_emb"].shape == (384,)
    K = r1["slots_y"].shape[0]
    assert r1["slots_y"].shape == (K, 256)
    assert r1["labels"].shape == (K,)
    # exactly the gold docs are labeled 1
    gold_set = set(r1["expected_doc_ids"])
    for sid, lab in zip(r1["source_ids"], r1["labels"].tolist()):
        assert lab == (1 if sid in gold_set else 0)
    assert int(r1["labels"].sum()) == len(r1["expected_doc_ids"])
    # source_ids are plain str (not np.str_) -- the replay logger matches on them
    assert all(isinstance(s, str) for s in r1["source_ids"])
    # slots_y finite (the random backbone still produces finite outputs)
    assert bool(torch.isfinite(r1["slots_y"]).all())

    # NO LEAKAGE: query 2's gold {d2,d3} must both be present and labeled 1,
    # and a doc that is gold for q1 (d1) must NOT be labeled 1 for q2.
    r2 = by_q["qst_0002"]
    g2 = set(r2["expected_doc_ids"])
    assert int(r2["labels"].sum()) == 2
    assert g2.issubset(set(r2["source_ids"]))
    for sid, lab in zip(r2["source_ids"], r2["labels"].tolist()):
        assert lab == (1 if sid in g2 else 0)
    # d1 is gold for q1 only; if it appears in q2's candidates (as a negative)
    # it must be labeled 0 there.
    if "d1" in r2["source_ids"]:
        i = r2["source_ids"].index("d1")
        assert r2["labels"].tolist()[i] == 0

    # pos/neg counts: each query contributes #gold positives
    total_pos = sum(int(r["labels"].sum()) for r in records)
    assert total_pos == 1 + 2 + 1
    assert stats["n_pos"] == total_pos
    assert stats["n_neg"] == sum(r["labels"].shape[0] for r in records) - total_pos
    # unique docs cached <= total candidate slots (caching dedups)
    assert stats["n_unique_docs"] <= sum(r["slots_y"].shape[0] for r in records)


def test_build_records_neg_per_query_bounds_candidate_count(tmp_path):
    docs = _synthetic_docs()
    dp = tmp_path / "docs.parquet"
    _write_docs_parquet(dp, docs)
    questions = _synthetic_questions()[:1]   # one query, gold d1
    gold_ids = {"d1"}
    idx, all_ids = build_doc_index(str(dp), gold_ids)
    tbl = open_docs_table(str(dp))
    backbone = JGSBackbone(BackboneConfig())
    for p in backbone.parameters():
        p.requires_grad = False
    records, _ = build_records(
        questions, tbl, idx, all_ids, backbone, _StubEmbedder(),
        MarkdownParser(), HierarchicalChunker(),
        neg_per_query=2, device=torch.device("cpu"), seed=1,
    )
    r = records[0]
    # 1 gold + 2 negatives = 3 candidates (all 6 docs available, so 2 negatives fit)
    assert r["slots_y"].shape[0] == 3
    assert int(r["labels"].sum()) == 1