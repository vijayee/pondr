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
    TEMPORAL_FEAT_DIM,
    build_doc_kind_tagger,
    extract_temporal_features,
    join_section_texts,
)
from src.ingestion.pipeline import UnifiedIngestionPipeline
from src.memory.store import HippocampalStore
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig, INSTANCE_CONFIGS
from src.subconscious.doc_kind_head import DocKindHead
from src.subconscious.training.doc_kind_training import (
    DocKindHeadTrainingConfig,
    evaluate_doc_kind_per_class,
    load_doc_kind_pairs,
    severity_doc_kind_loss,
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


# ── per-class eval scorecard (Phase 0) ──

def test_evaluate_doc_kind_per_class_shapes_and_unsafe_cell():
    """The scorecard reports the confusion matrix + unsafe_cell from a
    controlled (monkeypatched) forward, so the ship-gate metric is honest."""
    head = _head()
    labels = list(DocKindHead.LABELS)
    snap, dec = labels.index("point_in_time_snapshot"), labels.index("decision_update")
    plan = labels.index("plan")

    # 4 val docs: (true, pred). One snapshot mispredicted as decision_update
    # (the unsafe cell), one snapshot correct, one decision_update correct,
    # one plan correct.
    cases = [(snap, dec), (snap, snap), (dec, dec), (plan, plan)]

    def _logits(pred):
        out = torch.zeros(1, len(labels))
        out[0, pred] = 10.0
        return out

    val = [{"label": labels[true]} for true, _ in cases]
    val_embs = [[torch.randn(1, 384)] for _ in cases]
    # Monkeypatch forward to ignore inputs and return the controlled logits.
    preds_iter = iter(cases)
    head.forward = lambda embs: _logits(next(preds_iter)[1])

    pc = evaluate_doc_kind_per_class(head, val, val_embs)
    assert pc["acc"] == 0.75                      # 3 of 4 correct
    assert pc["top2_acc"] == 1.0                  # every true is the top or 2nd
    assert pc["unsafe_cell"] == 1                 # the one snap->dec confusion
    assert pc["snapshot_n"] == 2
    assert pc["snapshot_recall"] == 0.5          # 1 of 2 snapshots correct
    assert pc["decision_update_recall"] == 1.0
    assert pc["confusion"][snap][dec] == 1
    # confusion is 5x5; rows=true, cols=pred.
    assert len(pc["confusion"]) == 5 and len(pc["confusion"][0]) == 5
    # Wilson CI on 1/2 -> a real interval inside [0,1], not a point.
    lo, hi = pc["snapshot_recall_ci95"]
    assert 0.0 <= lo < hi <= 1.0
    # recall_per_class is keyed by label string.
    assert set(pc["recall_per_class"]) == set(labels)


def test_evaluate_doc_kind_per_class_wilson_ci_at_zero_n():
    """Empty snapshot class -> degenerate CI [0,1] (honest: no data)."""
    head = _head()
    labels = list(DocKindHead.LABELS)
    dec = labels.index("decision_update")
    # Only decision_update docs; no snapshots in val. forward returns a fixed
    # dec logit for every doc.
    dec_logits = 10.0 * torch.nn.functional.one_hot(
        torch.tensor(dec), len(labels)).float().unsqueeze(0)
    head.forward = lambda embs: dec_logits
    val = [{"label": "decision_update"}, {"label": "decision_update"}]
    val_embs = [[torch.randn(1, 384)], [torch.randn(1, 384)]]
    pc = evaluate_doc_kind_per_class(head, val, val_embs)
    assert pc["snapshot_n"] == 0
    assert pc["snapshot_recall"] == 0.0
    assert pc["snapshot_recall_ci95"] == [0.0, 1.0]
    assert pc["decision_update_recall"] == 1.0


# ── severity-weighted loss (Phase 2) ──

def test_severity_loss_penalizes_unsafe_direction_more_than_correct():
    """When truth is snapshot, the loss for predicting decision_update (unsafe)
    must be MUCH higher than for predicting snapshot (correct), and higher than
    the plain CE alone. This is the sign-trap guard: ``-logp[dec]`` would
    reward correctness, so the term is ``penalty * p(dec)`` instead."""
    w = torch.ones(len(DocKindHead.LABELS))
    snap = DocKindHead.LABELS.index("point_in_time_snapshot")
    logits_bad = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]])    # argmax=dec (unsafe)
    logits_good = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])   # argmax=snap (correct)

    ce_bad = severity_doc_kind_loss(logits_bad, snap, w, 0.0).item()
    ce_good = severity_doc_kind_loss(logits_good, snap, w, 0.0).item()
    sev_bad = severity_doc_kind_loss(logits_bad, snap, w, 5.0).item()
    sev_good = severity_doc_kind_loss(logits_good, snap, w, 5.0).item()

    # CE alone: the unsafe case is already costlier than correct (low p(snap)).
    assert ce_bad > ce_good
    # Severity widens the gap: extra penalty on the unsafe case, ~0 on correct.
    assert sev_bad - ce_bad > 4.0          # ~ penalty * p(dec) ~ 5*0.97
    assert sev_good - ce_good < 0.1        # ~ penalty * p(dec) ~ 5*0.007
    assert sev_bad > sev_good               # never invert (the trap)


def test_severity_loss_inert_for_non_snapshot_target_and_zero_penalty():
    """The penalty only applies when truth is snapshot AND penalty > 0; for any
    other class, or penalty=0, it equals plain class-weighted CE."""
    w = torch.ones(len(DocKindHead.LABELS))
    dec = DocKindHead.LABELS.index("decision_update")
    snap = DocKindHead.LABELS.index("point_in_time_snapshot")
    logits = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]])

    # decision_update truth: penalty must NOT change the loss (reverse direction
    # is only annoying, not unsafe).
    a = severity_doc_kind_loss(logits, dec, w, 0.0).item()
    b = severity_doc_kind_loss(logits, dec, w, 5.0).item()
    assert a == b
    # snapshot truth with penalty=0 recovers plain CE.
    c = severity_doc_kind_loss(logits, snap, w, 0.0).item()
    ref = (-torch.nn.functional.log_softmax(logits, dim=-1)[0, snap]).item()
    assert abs(c - ref) < 1e-6


def test_severity_loss_drives_unsafe_cell_down_in_overfit(tmp_path):
    """A/B smoke: two heads on the same tiny corpus that forces the unsafe
    confusion. The penalty>0 head ends with a smaller (or equal) unsafe_cell
    than the penalty=0 head -- it trains AWAY from snap->dec, not toward it."""
    # A corpus where snapshots outnumber decision_updates but the stub
    # embeddings are similar enough that plain CE can leak snap->dec. Repeat
    # to let both heads overfit.
    snap_txt = [f"# status\n\nstate as of 2026-0{m}-01 is green" for m in range(1, 8)]
    dec_txt = [f"# decision\n\nwe decided to switch to postgres on 2026-0{m}-01"
               for m in range(1, 4)]
    train = ([{"doc_id": f"s{i}", "section_texts": [snap_txt[i]], "label": "point_in_time_snapshot"}
              for i in range(len(snap_txt))] +
             [{"doc_id": f"d{i}", "section_texts": [dec_txt[i]], "label": "decision_update"}
              for i in range(len(dec_txt))])
    val = train

    def _run(penalty):
        head = _head()
        bb = head.backbone
        for p in bb.parameters():
            p.requires_grad = False
        embedder = build_embedder("stub")
        cfg = DocKindHeadTrainingConfig(epochs=12, device="cpu", embedder_source="stub",
                                        checkpoint_dir=str(tmp_path / f"h{penalty}"),
                                        unsafe_confusion_penalty=penalty,
                                        accum_steps=1)
        train_doc_kind_head_supervised(head, bb, train, val, embedder, cfg,
                                        progress_cb=None)
        from src.subconscious.training.doc_kind_training import evaluate_doc_kind_per_class
        # Re-embed val with the SAME embedder the trainer cached.
        from src.subconscious.training.doc_kind_training import _embed_sections
        val_embs = [_embed_sections(embedder, r["section_texts"], torch.device("cpu"))
                    for r in val]
        return evaluate_doc_kind_per_class(head, val, val_embs)["unsafe_cell"]

    base = _run(0.0)
    sev = _run(5.0)
    assert sev <= base, f"severity loss made unsafe_cell WORSE: {sev} > {base}"


# ── labeler confidence gate (Phase 1) ──

def test_filter_labeled_results_rejects_low_confidence_and_persists_fields():
    """The confidence gate is the root-cause fix for the v1 teacher noise: a
    low-confidence label is rejected (not written); a confident one carries its
    confidence + reason into the record."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_label_doc_kind_corpus",
        Path(__file__).resolve().parent.parent / "scripts" / "label_doc_kind_corpus.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _R:
        def __init__(self, response, error=None):
            self.response = response
            self.error = error

    docs = [("d1", "t1", ["# s\n\nbody1"]),
            ("d2", "t2", ["# s\n\nbody2"]),
            ("d3", "t3", ["# s\n\nbody3"]),
            ("d4", "t4", ["# s\n\nbody4"]),
            ("d5", "t5", ["# s\n\nbody5"])]
    results = [
        _R({"doc_kind": "point_in_time_snapshot", "confidence": 0.92, "reason": "status as of date"}),  # OK
        _R({"doc_kind": "decision_update", "confidence": 0.4, "reason": "guess"}),    # REJECT low conf
        _R({"doc_kind": "incident", "confidence": 0.9}),                               # REJECT OOV
        _R({"doc_kind": "plan", "confidence": 0.85, "reason": "roadmap"}),             # OK
        _R({}, error="timeout"),                                                      # FAIL
    ]
    records, counts, failures, low_conf, verdicts = mod.filter_labeled_results(
        docs, results, min_confidence=0.7)

    assert len(records) == 2
    assert {r["doc_id"] for r in records} == {"d1", "d4"}
    assert counts["point_in_time_snapshot"] == 1
    assert counts["plan"] == 1
    assert counts["decision_update"] == 0
    # low-conf + OOV + failure each counted.
    assert low_conf == 1
    assert failures == 2
    # The verdicts are the authoritative per-row outcome (one per doc, in order).
    assert len(verdicts) == len(docs)
    statuses = {v[0]: v[1] for v in verdicts}
    assert statuses["d1"] == "OK"
    assert statuses["d2"] == "REJECT_LOWCONF"
    assert statuses["d3"] == "REJECT_OOV"
    assert statuses["d4"] == "OK"
    assert statuses["d5"] == "FAIL"
    # The written record carries confidence + reason (for auditing).
    r0 = next(r for r in records if r["doc_id"] == "d1")
    assert r0["confidence"] == 0.92
    assert r0["reason"] == "status as of date"
    assert r0["label"] == "point_in_time_snapshot"
    assert r0["section_texts"] == ["# s\n\nbody1"]


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


# ── temporal feature (Phase 4) ──

def test_extract_temporal_features_shape_and_signatures():
    """The feature vector is 6 dims in [0,1] and separates the clear cases on the
    load-bearing dims (as_of for snapshot, decision for decision_update, plan for
    plan; reference carries no date)."""
    snap = ["# Status as of 2026-03-31\n\ndep is green."]
    dec = ["# Decision 2026-03-31\n\nwe switched to postgres."]
    plan = ["# Roadmap\n\nwe will migrate to k8s by 2026-06-30."]
    ref = ["# Overview\n\nthe system has three layers with no date."]
    snap_f = extract_temporal_features(snap)
    dec_f = extract_temporal_features(dec)
    plan_f = extract_temporal_features(plan)
    ref_f = extract_temporal_features(ref)
    for fv in (snap_f, dec_f, plan_f, ref_f):
        assert len(fv) == TEMPORAL_FEAT_DIM == 6
        assert all(0.0 <= x <= 1.0 for x in fv)
    # dim layout: [has_date, has_as_of, has_decision, n_dates_norm,
    #             first_date_in_heading, has_plan]
    assert snap_f == [1.0, 1.0, 0.0, 1/3, 1.0, 0.0]   # as-of + date in heading
    assert dec_f == [1.0, 0.0, 1.0, 1/3, 1.0, 0.0]   # decision phrase + heading
    assert plan_f[0] == 1.0 and plan_f[5] == 1.0     # has a date + plan phrase
    assert ref_f == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # no date, no signal


def test_extract_temporal_features_empty_is_all_zero():
    assert extract_temporal_features([]) == [0.0] * TEMPORAL_FEAT_DIM
    assert extract_temporal_features([""]) == [0.0] * TEMPORAL_FEAT_DIM


def test_head_feat_dim_widens_linear_and_forward_shape():
    """A feat-trained head's first Linear is wider (256+k -> 128) and forward
    accepts the feat tensor; feat=None falls back to zeros (backward-compat)."""
    bb = JGSBackbone(BackboneConfig())
    head0 = DocKindHead(bb, feat_dim=0)
    headk = DocKindHead(bb, feat_dim=TEMPORAL_FEAT_DIM)
    # The first Linear's in_features grows by feat_dim.
    assert head0.head[0].in_features == 256
    assert headk.head[0].in_features == 256 + TEMPORAL_FEAT_DIM
    assert headk.feat_dim == TEMPORAL_FEAT_DIM
    secs = _secs(3)
    # feat=None -> zeros fallback, still [1,5].
    assert headk.forward(secs).shape == (1, len(DocKindHead.LABELS))
    # feat supplied -> [1,5] (concat path).
    feat = torch.zeros(1, TEMPORAL_FEAT_DIM)
    assert headk.forward(secs, feat=feat).shape == (1, len(DocKindHead.LABELS))
    # A feat-less head ignores feat (still [1,5]).
    assert head0.forward(secs, feat=feat).shape == (1, len(DocKindHead.LABELS))


def test_head_feat_changes_logits():
    """A non-zero feat actually changes the output (the concat path is live, not
    silently dropped)."""
    bb = JGSBackbone(BackboneConfig())
    head = DocKindHead(bb, feat_dim=TEMPORAL_FEAT_DIM)
    head.eval()
    secs = _secs(3)
    with torch.no_grad():
        base = head.forward(secs)                       # feat=None -> zeros
        feat = torch.ones(1, TEMPORAL_FEAT_DIM)          # non-zero
        fed = head.forward(secs, feat=feat)
    assert not torch.allclose(base, fed)


def test_load_doc_kind_head_reads_feat_dim(tmp_path):
    """The loader widens the head to match the checkpoint's feat_dim; a feat-trained
    checkpoint round-trips into a feat-trained head with matching Linear weights."""
    bb = JGSBackbone(BackboneConfig())
    head = DocKindHead(bb, feat_dim=TEMPORAL_FEAT_DIM)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"head": head.state_dict(), "labels": list(DocKindHead.LABELS),
                "val_accuracy": 0.6, "epoch": 3, "feat_dim": TEMPORAL_FEAT_DIM},
               ckpt_path)
    from src.subconscious.training.routing_training import load_doc_kind_head
    out = load_doc_kind_head(str(ckpt_path), bb, device="cpu")
    assert out.feat_dim == TEMPORAL_FEAT_DIM
    assert out.head[0].in_features == 256 + TEMPORAL_FEAT_DIM
    assert torch.equal(out.head[0].weight, head.head[0].weight)


def test_load_doc_kind_head_rejects_feat_dim_mismatch(tmp_path):
    """A feat-trained checkpoint loaded into a feat-less head is a shape mismatch
    (the loader reads feat_dim from the ckpt, so it actually constructs the RIGHT
    head -- this test guards a future change that drops the read). To force a
    mismatch we hand-write a ckpt whose head.state_dict has the feat-less shape
    but whose feat_dim key claims feat-trained."""
    bb = JGSBackbone(BackboneConfig())
    featless = DocKindHead(bb, feat_dim=0)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"head": featless.head.state_dict(),   # Linear in_features=256
                "labels": list(DocKindHead.LABELS),
                "val_accuracy": 0.5, "epoch": 0,
                "feat_dim": TEMPORAL_FEAT_DIM},         # claims feat-trained
               ckpt_path)
    from src.subconscious.training.routing_training import load_doc_kind_head
    with pytest.raises(RuntimeError, match="mismatch"):
        load_doc_kind_head(str(ckpt_path), bb, device="cpu")


def test_backbone_tagger_passes_feat_when_head_has_feat_dim():
    """The serve tagger computes the temporal feature from section_texts and
    passes it into classify when head.feat_dim>0 (no train/serve skew)."""
    bb = JGSBackbone(BackboneConfig())
    head = DocKindHead(bb, feat_dim=TEMPORAL_FEAT_DIM)
    embedder = build_embedder("stub")
    tagger = BackboneDocKindTagger(head, embedder)
    seen: list = []

    real_classify = head.classify

    def spy(section_texts, embedder_arg, feat=None):
        seen.append(feat)
        return real_classify(section_texts, embedder_arg, feat=feat)
    head.classify = spy

    label = tagger.classify_doc_kind(["# Q1 status\n\ngreen as of 2026-03-31."])
    assert label in DocKindHead.LABELS
    # A feat tensor WAS passed (not None) and has the right shape.
    assert len(seen) == 1 and seen[0] is not None
    assert seen[0].shape == (1, TEMPORAL_FEAT_DIM)


def test_backbone_tagger_no_feat_when_head_feat_dim_zero():
    """A feat-less head gets feat=None from the tagger (the original path)."""
    head = _head()   # feat_dim=0
    embedder = build_embedder("stub")
    tagger = BackboneDocKindTagger(head, embedder)
    seen: list = []
    real_classify = head.classify
    head.classify = lambda section_texts, embedder_arg, feat=None: (
        seen.append(feat) or real_classify(section_texts, embedder_arg, feat=feat))
    tagger.classify_doc_kind(["# s\n\nbody"])
    assert seen == [None]


def test_supervised_training_temporal_feature_runs_and_persists_feat_dim(tmp_path):
    """temporal_feature=True trains without error, passes feats in train+eval
    (no skew), and the checkpoint carries feat_dim so the loader widens the head."""
    from src.subconscious.training.doc_kind_training import _embed_sections
    bb = JGSBackbone(BackboneConfig())
    head = DocKindHead(bb, feat_dim=TEMPORAL_FEAT_DIM)
    for p in bb.parameters():
        p.requires_grad = False
    embedder = build_embedder("stub")
    train = [
        {"doc_id": "s1", "section_texts": ["# status\n\ngreen as of 2026-03-31"],
         "label": "point_in_time_snapshot"},
        {"doc_id": "d1", "section_texts": ["# decision\n\nwe switched to postgres"],
         "label": "decision_update"},
        {"doc_id": "p1", "section_texts": ["# roadmap\n\nwe will migrate next quarter"],
         "label": "plan"},
    ] * 3
    val = train[:2]
    cfg = DocKindHeadTrainingConfig(epochs=4, device="cpu", embedder_source="stub",
                                    checkpoint_dir=str(tmp_path / "feathead"),
                                    temporal_feature=True, accum_steps=1)
    result = train_doc_kind_head_supervised(head, bb, train, val, embedder, cfg,
                                            progress_cb=None)
    assert result["best_per_class"] is not None
    ckpt = torch.load(tmp_path / "feathead" / "best.pt", map_location="cpu",
                      weights_only=False)
    assert ckpt["feat_dim"] == TEMPORAL_FEAT_DIM
    # The loader round-trips the feat-trained head.
    from src.subconscious.training.routing_training import load_doc_kind_head
    out = load_doc_kind_head(str(tmp_path / "feathead" / "best.pt"), bb, device="cpu")
    assert out.feat_dim == TEMPORAL_FEAT_DIM


def test_supervised_training_rejects_feat_flag_head_mismatch(tmp_path):
    """temporal_feature=True with a feat-less head (feat_dim=0) is a hard error
    (would silently drop the signal / skew)."""
    bb = JGSBackbone(BackboneConfig())
    head = DocKindHead(bb, feat_dim=0)   # feat-less, but flag on
    for p in bb.parameters():
        p.requires_grad = False
    embedder = build_embedder("stub")
    train = [{"doc_id": "s1", "section_texts": ["# s\n\nbody"],
              "label": "point_in_time_snapshot"}] * 3
    val = train[:1]
    cfg = DocKindHeadTrainingConfig(epochs=1, device="cpu", embedder_source="stub",
                                    checkpoint_dir=str(tmp_path / "bad"),
                                    temporal_feature=True, accum_steps=1)
    with pytest.raises(RuntimeError, match="temporal_feature=True but head.feat_dim"):
        train_doc_kind_head_supervised(head, bb, train, val, embedder, cfg,
                                       progress_cb=None)


def test_gate_score_prefers_safe_then_guard_min():
    """Gate-aware selection: a safe (unsafe<=1) epoch beats an unsafe one even at
    lower acc; among safe epochs the one with the higher min(snap,dec) wins."""
    from src.subconscious.training.doc_kind_training import _gate_score
    unsafe_high = {"unsafe_cell": 3, "snapshot_recall": 0.9,
                   "decision_update_recall": 0.9, "acc": 0.80}
    safe_low = {"unsafe_cell": 0, "snapshot_recall": 0.5,
                "decision_update_recall": 0.5, "acc": 0.40}
    # safe beats unsafe despite lower acc.
    assert _gate_score(safe_low) > _gate_score(unsafe_high)
    # among safe epochs, the one with the stronger guard class wins.
    safe_weak = {"unsafe_cell": 0, "snapshot_recall": 0.4,
                 "decision_update_recall": 0.9, "acc": 0.70}
    safe_strong = {"unsafe_cell": 0, "snapshot_recall": 0.8,
                   "decision_update_recall": 0.8, "acc": 0.60}
    assert _gate_score(safe_strong) > _gate_score(safe_weak)