"""Tests for the ingestion package + the forgetting exemption (task #17).

Offline: uses installed ``wavedb`` (CPU) + ``torch``; no GLiNER/Bonsai. Covers:

* parsers (markdown headings -> sections; plain-text paragraph split);
* HierarchicalChunker leaf sizing (oversized sub-split, tiny merge, child_of);
* the UnifiedIngestionPipeline structure-only path (no extractors) +
  upsert-by-source_path (re-ingest reuses the doc id);
* the forgetting exemption: the three decay sweeps + the retrieval-boost
  hook skip doc-owned edges by the ``doc_`` prefix;
* the salience-compute script loads by unit type (``doc_`` -> ``ingested_at``)
  so a corpus with documents does not crash.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import torch

from src.config import ConsolidationConfig
from src.gnn import Consolidator
from src.gnn.consolidate import KNOWN_PREDICATES, _FORGET_PREDICATES
from src.gnn.graph_loader import PREDICATE_VOCAB
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.parsers import MarkdownParser, PlainTextParser, detect_type
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.document import Document, DocumentSection
from src.memory.edge_meta import edge_meta_key, record_archived_edge_op
from src.memory.store import HippocampalStore, _b2s


# ── parsers ──

def test_markdown_parser_sections_from_headings():
    md = "# Title\n\nIntro para.\n\n## A\n\nBody A.\n\n### A1\n\nDeep.\n"
    parsed = MarkdownParser().parse_text(md, source_path="x.md")
    assert parsed.title == "Title"
    levels = [s.level for s in parsed.sections]
    assert levels == [1, 2, 3], f"heading levels wrong: {levels}"
    assert parsed.sections[1].heading == "A"
    assert "Body A" in parsed.sections[1].content


def test_detect_type_by_extension():
    assert detect_type("foo.md") == "markdown"
    assert detect_type("foo.txt") == "text"
    assert detect_type("foo.unknown") == "text"  # fallback


def test_plain_text_parser_paragraph_split():
    text = "Para one about WaveDB.\n\nPara two about HBTrie.\n\nPara three."
    parsed = PlainTextParser().parse_text(text, source_path="x.txt")
    assert len(parsed.sections) == 3
    assert all(s.level == 1 for s in parsed.sections)
    assert "WaveDB" in parsed.sections[0].content


# ── chunker ──

def test_chunker_leaf_sizing_subsplit(tmp_path):
    """An oversized section sub-splits on paragraph boundaries into <=cap leaves.

    A single paragraph larger than the cap stays whole by design (the chunker
    does not split mid-paragraph in this slice), so the test feeds MANY small
    paragraphs so the packer can actually produce cap-sized leaves.
    """
    para = "A short paragraph. " * 4              # ~20 tokens
    body = "\n\n".join([para] * 20)               # 20 paras, ~400 tokens total
    md = "# H\n\n" + body + "\n"
    parsed = MarkdownParser().parse_text(md, source_path="x.md")
    chunker = HierarchicalChunker(max_section_tokens=100, min_section_tokens=1)
    out = chunker.chunk(parsed)
    assert len(out.sections) > 1, "oversized section not sub-split"
    for s in out.sections:
        assert len(s.content) // 4 <= 100 + 5, f"sub-split leaf exceeds cap: {len(s.content)//4}"


def test_chunker_tiny_merge_into_parent():
    """A section below min_section_tokens merges into its parent's body."""
    big = "Parent body big enough to clear the min floor. " * 30  # ~210 tokens
    md = "# Parent\n\n" + big + "\n\n## Tiny\n\nsmall.\n"
    parsed = MarkdownParser().parse_text(md, source_path="x.md")
    chunker = HierarchicalChunker(max_section_tokens=512, min_section_tokens=64)
    out = chunker.chunk(parsed)
    headings = [s.heading for s in out.sections]
    assert "Tiny" not in headings, "tiny section not merged away"
    assert len(out.sections) == 1
    assert "small" in out.sections[0].content, "tiny body not absorbed into parent"


def test_chunker_child_of_links_reflect_tree():
    md = "# Root\n\nroot body. " * 30 + "\n\n## Child\n\nchild body. " * 20 + "\n"
    # build a clean doc with two >64-token sections
    root_body = "Root section body long enough to clear the floor. " * 6
    child_body = "Child section body long enough to clear the floor. " * 6
    md = f"# Root\n\n{root_body}\n\n## Child\n\n{child_body}\n"
    parsed = MarkdownParser().parse_text(md, source_path="x.md")
    out = HierarchicalChunker(min_section_tokens=64).chunk(parsed)
    assert len(out.sections) == 2
    # section 1 (Child) parent_index points at section 0 (Root).
    assert out.sections[1].parent_index == 0
    assert out.sections[0].parent_index is None


# ── pipeline (structure-only) + upsert ──

def _doc_from_sections(doc_id, source_path, sections):
    secs = [DocumentSection(id=f"{doc_id}_sec_{i:03d}", heading=h,
                            level=l, content=c, parent_section=p)
            for i, (h, l, c, p) in enumerate(sections)]
    return Document(id=doc_id, source_type="markdown", source_path=source_path,
                    title="T", ingested_at="2026-07-11T00:00:00", sections=secs)


def test_pipeline_structure_only_ingest_and_upsert(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    pipe = UnifiedIngestionPipeline(store)  # no extractors -> structure-only

    # Write a markdown source to disk.
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nbody body body body body body body. " * 10, encoding="utf-8")
    doc_id, created = pipe.ingest(str(src), extractor=None, relation_extractor=None)
    assert created is True
    assert doc_id.startswith("doc_")
    assert store.default_document_ids() == [doc_id]

    # Re-ingest the SAME source -> UPDATE, reuse doc_id, no duplicate.
    doc_id2, created2 = pipe.ingest(str(src), extractor=None, relation_extractor=None)
    assert created2 is False
    assert doc_id2 == doc_id, "re-ingest did not reuse the doc id"
    assert store.default_document_ids() == [doc_id], "upsert created a duplicate"
    store.close()


# ── forgetting exemption ──

def _one_edge_data(s_id: str, o_id: str, pred_idx: int = 0):
    """A 1-edge subgraph: nodes [s, o], one edge with predicate KNOWN[pred_idx]."""
    data = SimpleNamespace(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        node_id=[s_id, o_id],
        edge_attr=torch.zeros(1, PREDICATE_VOCAB),
    )
    data.edge_attr[0, pred_idx] = 1.0
    out = {"salience": torch.tensor([0.0, 0.0])}  # low -> prunable/archivable
    return data, out


def _base_report():
    return {
        "dry_run": True, "trained": False, "subgraphs_scored": 0,
        "abstracts": [], "edges_proposed": [], "edges_accepted": [],
        "edges_unverified": [], "anomalies": [], "ontology_proposed": [],
        "pruned": [], "verifier_calls": 0, "verifier_accepted": 0,
        "score_distributions": {
            "ontology": [0] * 100, "linkpred": [0] * 100,
            "salience_endpoint": [0] * 100,
        },
        "forgetting": {"edges_seen": 0, "boosted": 0, "archived": [], "ltp": 0,
                       "reconsolidated": [], "ontology_deprecated": [],
                       "deep_archived": []},
    }


def test_forget_exempt_step_forget_skips_doc_edges(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    cons = Consolidator(store, dry_run=True, allow_untrained_apply=False,
                        config=ConsolidationConfig(prune_salience_below=-100.0))

    # Doc-owned (doc, has_entity, E:Alice) edge: skipped.
    data, out = _one_edge_data("doc_000001", "E:Alice")
    rep = _base_report()
    cons._step_forget(data, out, rep)
    assert rep["forgetting"]["edges_seen"] == 0, "doc edge not skipped by _step_forget"

    # Control: episode edge is processed (has_entity is in _FORGET_PREDICATES).
    data, out = _one_edge_data("ep_000001", "E:Alice")
    rep = _base_report()
    cons._step_forget(data, out, rep)
    assert rep["forgetting"]["edges_seen"] >= 1, "episode edge unexpectedly skipped"
    store.close()


def test_forget_exempt_step_prune_skips_doc_edges(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    cons = Consolidator(store, dry_run=True, allow_untrained_apply=False,
                        config=ConsolidationConfig(prune_salience_below=0.15))

    # Doc edge with both endpoints below threshold: not pruned.
    data, out = _one_edge_data("doc_000001", "doc_000002")
    rep = _base_report()
    cons._step_prune(data, out, rep)
    assert rep["pruned"] == [], "doc edge pruned by _step_prune"

    # Control: episode edge with both endpoints below threshold: pruned.
    data, out = _one_edge_data("ep_000001", "E:Alice")
    rep = _base_report()
    cons._step_prune(data, out, rep)
    assert len(rep["pruned"]) == 1, "episode edge not pruned"
    store.close()


def test_forget_exempt_step_deep_archive_skips_doc_edges(tmp_path):
    store = HippocampalStore(str(tmp_path / "mem"))
    cons = Consolidator(store, dry_run=True, allow_untrained_apply=False,
                        config=ConsolidationConfig(deep_archive_days=1))
    cons._deep_archive_candidates = []

    # Plant two aged archived-edge index entries: one doc-owned, one episode.
    old = "2020-01-01T00:00:00Z"
    store.db.batch_sync([
        record_archived_edge_op("doc_000001", "has_entity", "E:Alice", old),
        record_archived_edge_op("ep_000001", "has_entity", "E:Bob", old),
    ])
    rep = _base_report()
    cons._step_deep_archive(rep)
    subjects = [d["subject"] for d in rep["forgetting"]["deep_archived"]]
    assert "doc_000001" not in subjects, "doc edge deep-archived (not exempt)"
    assert "ep_000001" in subjects, "episode edge not deep-archived"
    store.close()


def test_retrieval_boost_skips_doc_edges(tmp_path):
    """The retrieval-boost hook writes no sidecar for a doc-owned result."""
    store = HippocampalStore(str(tmp_path / "mem"))
    from src.retrieval.graph_traversal import GraphTraversal
    trav = GraphTraversal(store)
    # A doc result (defensive: Phase 1 retrieve() returns episodes only, but
    # _apply_retrieval_boost must skip a doc id if one ever appears).
    results = [{"episode_id": "doc_000001",
                "entities": ["Alice"], "topics": [], "tones": []}]
    trav._apply_retrieval_boost(results, ["Alice"], [], [], "important")
    assert not _b2s(store.db.get_sync(
        edge_meta_key("doc_000001", "has_entity", "E:Alice"))), \
        "sidecar written for a doc edge"
    store.close()


# ── salience-compute loads by unit type ──

def test_salience_compute_handles_doc_ids(tmp_path):
    """compute_entity_salience does not crash on a doc id in the has_entity scan."""
    store = HippocampalStore(str(tmp_path / "mem"))
    # A document with an entity (emits (doc, has_entity, E:Alice)).
    store.encode_document(_doc_from_sections(
        "doc_000001", "src.md",
        [("Intro", 1, "body body body body body body body body. " * 10, None)]))
    # Patch doc-level entities by re-encoding with an entity on the doc.
    from src.memory.document import Document as D
    secs = [DocumentSection(id="doc_000001_sec_000", heading="Intro", level=1,
                            content="body " * 60, entities=["Alice"])]
    d = D(id="doc_000001", source_type="markdown", source_path="src.md",
          title="T", ingested_at="2026-07-11T00:00:00", sections=secs,
          entities=["Alice"])
    store.encode_document(d, update=True)
    store.close()

    # Run the script as a subprocess against the store; it must exit 0
    # (a doc id in the has_entity POS scan no longer crashes get_episode).
    res = subprocess.run(
        [sys.executable, "scripts/compute_entity_salience.py",
         "--db", str(tmp_path / "mem")],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"salience compute crashed:\n{res.stderr}"
    assert "Alice" in res.stdout, "doc entity not counted"