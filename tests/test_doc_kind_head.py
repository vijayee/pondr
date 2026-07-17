"""Offline tests for the DocKindHead (Phase 3c Sec 7.11 deferred step).

All CPU-runnable against ``ReferenceSSM`` + the deterministic ``stub`` embedder
(no ``sentence_transformers`` download, no Bonsai, no WaveDB on the head path).
The pipeline-integration test uses a tmp_path WaveDB store (mirrors
``tests/test_contradiction.py``). The real-backbone train test is gated on the
Phase 2a checkpoint existing locally so the suite runs on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.ingestion.doc_kind import (
    BackboneDocKindTagger,
    BonsaiDocKindTagger,
    build_doc_kind_tagger,
    join_section_texts,
)
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.store import HippocampalStore
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig, INSTANCE_CONFIGS
from src.subconscious.doc_kind_head import DocKindHead
from src.subconscious.training.doc_kind_training import (
    DocKindHeadTrainingConfig,
    load_doc_kind_pairs,
    train_doc_kind_head_supervised,
)
from src.subconscious.training.routing_training import build_embedder, load_backbone

BACKBONE_PATH = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"


def _head() -> DocKindHead:
    bb = JGSBackbone(BackboneConfig())
    return DocKindHead(bb)


def _secs(n: int = 3) -> list[torch.Tensor]:
    """n distinct [1, 384] section embeddings (random, CPU)."""
    return [torch.randn(1, 384) for _ in range(n)]


# ── contract / shape ──

def test_forward_returns_five_logit_heads():
    head = _head()
    logits = head.forward(_secs(3))
    assert logits.shape == (1, len(DocKindHead.LABELS))


def test_classify_returns_label_in_vocab_for_multisection_doc():
    head = _head()
    embedder = build_embedder("stub")
    label = head.classify(["# Status\n\ndep is green as of 2026-03-31.",
                           "# Update\n\nwe switched to postgres."], embedder)
    assert label in DocKindHead.LABELS


def test_classify_returns_none_for_empty_sections():
    head = _head()
    embedder = build_embedder("stub")
    assert head.classify([], embedder) is None
    # whitespace-only sections are filtered -> None
    assert head.classify(["   ", "\n\n"], embedder) is None


def test_head_parameters_exclude_backbone():
    bb = JGSBackbone(BackboneConfig())
    head = DocKindHead(bb)
    bb_params = sum(p.numel() for p in bb.parameters())
    head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    # The backbone is stored via object.__setattr__ (not a submodule), so the
    # head's param set must NOT include the ~19.5M backbone params.
    assert head_trainable < bb_params
    assert all(p.requires_grad for p in bb.parameters())


def test_backbone_frozen_after_load_backbone():
    if not Path(BACKBONE_PATH).exists():
        pytest.skip("Phase 2a backbone checkpoint not present locally")
    bb = load_backbone(BACKBONE_PATH, BackboneConfig(), device="cpu")
    assert all(not p.requires_grad for p in bb.parameters())
    assert sum(p.numel() for p in bb.parameters()) == 19_518_016


# ── loader ──

def test_load_doc_kind_head_round_trips(tmp_path):
    bb = JGSBackbone(BackboneConfig())
    head = DocKindHead(bb)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"head": head.state_dict(), "labels": list(DocKindHead.LABELS),
                "val_accuracy": 0.75, "epoch": 4}, ckpt_path)
    from src.subconscious.training.routing_training import load_doc_kind_head
    out = load_doc_kind_head(str(ckpt_path), bb, device="cpu")
    assert out.training is False  # eval mode
    # The loaded classifier head matches the saved one.
    assert torch.equal(out.head[0].weight, head.head[0].weight)
    # Lean checkpoint: no backbone keys, param count far below the 19.5M backbone.
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["head"]
    assert not any("backbone" in k for k in sd)
    assert sum(v.numel() for v in sd.values()) < 19_000_000


def test_load_doc_kind_head_rejects_label_mismatch(tmp_path):
    bb = JGSBackbone(BackboneConfig())
    head = DocKindHead(bb)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"head": head.state_dict(),
                "labels": ["other", "plan", "reference", "decision_update",
                           "point_in_time_snapshot"],  # WRONG order
                "val_accuracy": 0.5, "epoch": 0}, ckpt_path)
    from src.subconscious.training.routing_training import load_doc_kind_head
    with pytest.raises(RuntimeError, match="label-order mismatch"):
        load_doc_kind_head(str(ckpt_path), bb, device="cpu")


# ── supervised training (stub embedder; loss decreases via overfitting) ──

def test_supervised_step_runs_and_decreases_loss(tmp_path):
    head = _head()
    bb = head.backbone
    for p in bb.parameters():
        p.requires_grad = False
    embedder = build_embedder("stub")
    # 5 docs, one per class (distinct section text -> distinct stub embeddings
    # -> the head can overfit them).
    train = [
        {"doc_id": f"d{i}", "section_texts": [f"# sec {i}\n\nbody {i} {label}"],
         "label": label}
        for i, label in enumerate(DocKindHead.LABELS)
    ] * 3  # 15 docs (5 distinct, repeated) so the head can overfit
    val = train[:2]
    cfg = DocKindHeadTrainingConfig(epochs=10, device="cpu", embedder_source="stub",
                                    checkpoint_dir=str(tmp_path / "head"))

    losses = []

    def cb(epoch, tl, va):
        losses.append(tl)

    train_doc_kind_head_supervised(head, bb, train, val, embedder, cfg, progress_cb=cb)
    assert len(losses) == 10
    assert all(l == l for l in losses)  # no NaN
    # Overfitting the 5 distinct docs -> loss drops meaningfully.
    assert losses[-1] < losses[0]
    # Lean checkpoint: head.state_dict() excludes the backbone.
    ckpt = torch.load(tmp_path / "head" / "best.pt", map_location="cpu",
                      weights_only=False)
    sd = ckpt["head"]
    assert ckpt["labels"] == list(DocKindHead.LABELS)
    assert not any("backbone" in k for k in sd)
    assert sum(v.numel() for v in sd.values()) < 19_000_000


def test_load_doc_kind_pairs_drops_malformed(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(
        '{"doc_id":"d1","section_texts":["# s\\n\\nb1"],"label":"plan"}\n'
        'not-json\n'
        '{"doc_id":"d2","section_texts":[],"label":"plan"}\n'              # empty sections
        '{"doc_id":"d3","section_texts":["ok"],"label":"nonsense"}\n'      # out-of-vocab label
        '{"doc_id":"d4","section_texts":["ok"],"label":"reference"}\n',
        encoding="utf-8",
    )
    recs = load_doc_kind_pairs(str(p))
    assert len(recs) == 2
    assert {r["doc_id"] for r in recs} == {"d1", "d4"}


# ── tagger adapters ──

def test_join_section_texts_matches_pipeline_doc_text():
    """The Bonsai adapter's join is byte-identical to ``_doc_text``."""
    # Build a parsed-like object with .sections carrying heading/content.
    class _S:
        def __init__(self, heading, content):
            self.heading = heading
            self.content = content

    class _Parsed:
        def __init__(self, secs):
            self.sections = secs

    parsed = _Parsed([_S("Status", "dep is green"), _S(None, "footer text")])
    section_texts = [
        (s.heading + "\n" + s.content) if s.heading else s.content
        for s in parsed.sections
    ]
    assert join_section_texts(section_texts) == UnifiedIngestionPipeline._doc_text(parsed)


def test_bonsai_adapter_joins_and_delegates():
    received: list[str] = []

    class _Decider:
        def classify_doc_kind(self, text):
            received.append(text)
            return "plan"

    tagger = BonsaiDocKindTagger(_Decider())
    assert tagger.classify_doc_kind(["# s1\n\nbody1", "body2"]) == "plan"
    # The decider received the joined text (one call).
    assert len(received) == 1
    assert received[0] == "# s1\n\nbody1\n\nbody2"
    # Empty section list -> None (no decider call).
    assert tagger.classify_doc_kind([]) is None
    assert len(received) == 1


def test_backbone_tagger_writes_label_at_ingest(tmp_path):
    """The BackboneDocKindTagger wired as doc_kind_tagger tags a doc at ingest."""
    head = _head()
    embedder = build_embedder("stub")
    tagger = BackboneDocKindTagger(head, embedder)
    src = tmp_path / "status.md"
    src.write_text("# Q1 status\n\ndep is green as of 2026-03-31.", encoding="utf-8")
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        pipe = UnifiedIngestionPipeline(store)
        doc_id, _ = pipe.ingest(str(src), extractor=None, relation_extractor=None,
                                doc_kind_tagger=tagger)
        # An untrained head still returns SOME label for a non-empty doc (argmax
        # of random logits) -- the point is the WIRING: the label is written.
        assert store.document_kind(doc_id) in DocKindHead.LABELS
    finally:
        store.close()


def test_build_doc_kind_tagger_prefers_head_then_bonsai_then_none(tmp_path):
    """build_doc_kind_tagger: head checkpoint > Bonsai decider > None."""
    # No head checkpoint, no decider -> None (cold-start).
    assert build_doc_kind_tagger(
        head_path=str(tmp_path / "absent.pt"), bonsai_decider=None, verbose=False
    ) is None
    # No head checkpoint, decider present -> BonsaiDocKindTagger.
    class _Decider:
        def classify_doc_kind(self, text):
            return "reference"
    t = build_doc_kind_tagger(
        head_path=str(tmp_path / "absent.pt"), bonsai_decider=_Decider(), verbose=False
    )
    assert isinstance(t, BonsaiDocKindTagger)