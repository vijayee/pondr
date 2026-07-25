"""PROBE (uncommitted): true scorecard of the v2 feat-on head vs CLEAN val labels.

Loads the existing v2 abon head on CPU, predicts on the 76 clean-val docs
(panel-majority labels), and scores via evaluate_doc_kind_per_class (one-hot
dummy-head trick from the Bonsai probe). Compares the clean-label scorecard to
the original flash-label scorecard (0.43 acc / unsafe 0 / snap 0.75 / dec 0.33)
-- this is the TRUE dec/snap recall, with flash's over-assignment bias removed.

GPU-free (CPU). Not committed (scripts/_scratch/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from src.ingestion.doc_kind import extract_temporal_features
from src.subconscious.configs import BackboneConfig
from src.subconscious.doc_kind_head import DocKindHead
from src.subconscious.training.doc_kind_training import evaluate_doc_kind_per_class
from src.subconscious.training.routing_training import build_embedder, load_backbone, load_doc_kind_head

CLEAN_VAL = "data/training/doc_kind_head/pairs_clean_val.jsonl"
FLASH_VAL = "data/training/doc_kind_head/pairs_v3_val.jsonl"   # original flash labels
HEAD = "data/training/doc_kind_head_abon/best.pt"
BACKBONE = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"


def _preds(head, val, embedder):
    preds = []
    with torch.no_grad():
        for rec in val:
            sec = rec["section_texts"]
            feat = None
            if getattr(head, "feat_dim", 0) > 0:
                feat = torch.tensor(extract_temporal_features(sec), dtype=torch.float32).unsqueeze(0)
            preds.append(head.classify(sec, embedder, feat=feat) or "other")
    return preds


def _scorecard(val, preds):
    labels = list(DocKindHead.LABELS)
    it = iter(labels.index(p) if p in labels else labels.index("other") for p in preds)

    class _Dummy:
        def eval(self): return self
        def forward(self, embs, feat=None):
            i = next(it)
            out = torch.zeros(1, len(labels)); out[0, i] = 10.0; return out

    embs = [[torch.zeros(1, 384)] for _ in val]
    return evaluate_doc_kind_per_class(_Dummy(), val, embs, val_feats=None)


def main() -> int:
    clean_val = [json.loads(l) for l in open(CLEAN_VAL, encoding="utf-8") if l.strip()]
    flash_val = [json.loads(l) for l in open(FLASH_VAL, encoding="utf-8") if l.strip()]
    print(f"clean val: {len(clean_val)} (panel labels), flash val: {len(flash_val)}",
          flush=True)

    print("loading backbone + head (CPU) ...", flush=True)
    backbone = load_backbone(BACKBONE, BackboneConfig(), device="cpu")
    head = load_doc_kind_head(HEAD, backbone, device="cpu")
    head.eval()
    print(f"  feat_dim={getattr(head, 'feat_dim', 0)}", flush=True)
    embedder = build_embedder("on-demand")

    print(f"predicting on {len(clean_val)} val docs (CPU) ...", flush=True)
    preds = _preds(head, clean_val, embedder)

    clean_pc = _scorecard(clean_val, preds)
    flash_pc = _scorecard(flash_val, preds)   # SAME preds, flash labels -> the 0.33 repro

    def _show(name, pc):
        rc = pc["recall_per_class"]
        print(f"\n=== {name} ===", flush=True)
        print(f"  acc={pc['acc']:.3f} unsafe={pc['unsafe_cell']} "
              f"snap_r={rc['point_in_time_snapshot']:.2f} "
              f"dec_r={rc['decision_update']:.2f} plan_r={rc['plan']:.2f} "
              f"snap_n={pc['snapshot_n']} CI={pc['snapshot_recall_ci95']}", flush=True)
        print(f"  confusion: {pc['confusion']}", flush=True)

    _show("CLEAN val (panel-majority labels) -- TRUE scorecard", clean_pc)
    _show("FLASH val (original labels) -- the 0.33 we saw", flash_pc)

    rc_c = clean_pc["recall_per_class"]; rc_f = flash_pc["recall_per_class"]
    print(f"\n=== DELTA (clean - flash) ===", flush=True)
    print(f"  acc:   {flash_pc['acc']:.2f} -> {clean_pc['acc']:.2f}", flush=True)
    print(f"  unsafe:{flash_pc['unsafe_cell']} -> {clean_pc['unsafe_cell']}", flush=True)
    print(f"  snap_r: {rc_f['point_in_time_snapshot']:.2f} -> "
          f"{rc_c['point_in_time_snapshot']:.2f}", flush=True)
    print(f"  dec_r:  {rc_f['decision_update']:.2f} -> "
          f"{rc_c['decision_update']:.2f}", flush=True)
    print(f"  snap CI: {flash_pc['snapshot_recall_ci95']} -> "
          f"{clean_pc['snapshot_recall_ci95']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())