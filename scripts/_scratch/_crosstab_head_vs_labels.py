"""PROBE (uncommitted): cross-tab head predictions vs flash labels vs glm labels.

The audit found the independent teacher (glm) disagrees with flash on 12/24 of
the decision_update val docs -- and glm looks right (flash over-assigns
decision_update to support threads / status reports / design questions / POC
kickoffs). This probe loads the v2 feat-on head (abon/best.pt) on CPU (GPU-free)
and asks: on the docs flash labeled decision_update but glm called non-dec, does
the HEAD also call non-dec? If so, the head is RIGHT and flash's val label is
the error -> the head's dec 0.33 is a noise floor, not an arch ceiling.

Reports dec recall against (a) the original flash labels and (b) glm labels, so
we see how much of the "missed" dec is actually flash mislabeling.

Not committed (scripts/_scratch/). CPU-only (GPU stays free).
"""
from __future__ import annotations

import argparse
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
from src.subconscious.training.routing_training import build_embedder, load_backbone
from src.subconscious.training.routing_training import load_doc_kind_head  # noqa: F401 (may not exist)

VAL = "data/training/doc_kind_head/pairs_v3_val.jsonl"
AUDIT = "data/training/doc_kind_head/val_audit_report.jsonl"
HEAD = "data/training/doc_kind_head_abon/best.pt"
BACKBONE = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--head", default=HEAD)
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    val = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]
    audit = {r["doc_id"]: r for r in
             (json.loads(l) for l in open(args.audit, encoding="utf-8") if l.strip())}
    print(f"val: {len(val)} docs, audit rows: {len(audit)}", flush=True)

    print(f"loading backbone (CPU) from {args.backbone} ...", flush=True)
    backbone = load_backbone(args.backbone, BackboneConfig(), device=args.device)
    print(f"loading head from {args.head} ...", flush=True)
    head = load_doc_kind_head(args.head, backbone, device=args.device)
    head.eval()
    print(f"  head feat_dim={getattr(head, 'feat_dim', 0)}", flush=True)

    print(f"building bge-small embedder (on-demand, CPU) ...", flush=True)
    embedder = build_embedder("on-demand")

    labels = list(DocKindHead.LABELS)
    print(f"predicting on {len(val)} val docs (CPU, may take a few min) ...", flush=True)
    preds: dict[str, str] = {}
    with torch.no_grad():
        for i, rec in enumerate(val):
            sec = rec["section_texts"]
            feat = None
            if getattr(head, "feat_dim", 0) > 0:
                feat = torch.tensor(extract_temporal_features(sec), dtype=torch.float32).unsqueeze(0)
            pred = head.classify(sec, embedder, feat=feat)
            preds[rec.get("doc_id", f"r{i}")] = pred or "other"
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(val)} ...", flush=True)

    # Cross-tab: flash_label (val), head_pred, glm_label (audit teacher).
    print(f"\n=== DEC RECALL: head vs flash labels vs glm labels ===", flush=True)
    # (a) against flash labels (the original scorecard truth)
    dec_flash = [r for r in val if r["label"] == "decision_update"]
    dec_flash_correct = sum(1 for r in dec_flash
                            if preds[r.get("doc_id", "?")] == "decision_update")
    # (b) against glm labels (independent teacher)
    dec_glm = [r for r in val
               if audit.get(r.get("doc_id", {}), {}).get("teacher_label") == "decision_update"]
    dec_glm_correct = sum(1 for r in dec_glm
                          if preds[r.get("doc_id", "?")] == "decision_update")
    print(f"  vs FLASH labels: dec recall = {dec_flash_correct}/{len(dec_flash)} "
          f"= {dec_flash_correct/len(dec_flash):.2f}  (the 0.33 we saw)", flush=True)
    print(f"  vs GLM labels:   dec recall = {dec_glm_correct}/{len(dec_glm)} "
          f"= {dec_glm_correct/max(1,len(dec_glm)):.2f}  (cleaner truth)", flush=True)

    # The key table: of the 24 flash-dec docs, head pred vs glm pred.
    print(f"\n=== flash=decision_update docs: head vs glm ===", flush=True)
    head_right_glm_right = 0    # both call non-dec -> flash was wrong, head right
    head_wrong_glm_right = 0    # glm non-dec, head dec -> head fell for flash's label
    for r in dec_flash:
        did = r.get("doc_id", "?")
        hp = preds.get(did, "?")
        ar = audit.get(did, {})
        glm = ar.get("teacher_label")
        if glm is not None and glm != "decision_update" and hp != "decision_update":
            head_right_glm_right += 1
        elif glm is not None and glm != "decision_update" and hp == "decision_update":
            head_wrong_glm_right += 1
    n_glm_nondec = sum(1 for r in dec_flash
                       if audit.get(r.get("doc_id", "?"), {}).get("teacher_label") is not None
                       and audit.get(r.get("doc_id", "?"), {}).get("teacher_label") != "decision_update")
    print(f"  of the {len(dec_flash)} flash-dec docs, glm called {n_glm_nondec} non-dec.", flush=True)
    print(f"  head ALSO called non-dec on {head_right_glm_right} of those -> head RIGHT (flash label was noise)",
          flush=True)
    print(f"  head called dec on {head_wrong_glm_right} of those -> head fell for flash's (wrong) label",
          flush=True)

    # Full per-doc dump for the dec docs.
    print(f"\n=== per-doc (flash=decision_update) ===", flush=True)
    print(f"{'doc_id':<40} {'flash':<18} {'head':<22} {'glm':<22}", flush=True)
    for r in dec_flash:
        did = r.get("doc_id", "?")
        hp = preds.get(did, "?")
        glm = audit.get(did, {}).get("teacher_label", "?")
        print(f"{did[:38]:<40} {r['label'][:16]:<18} {str(hp)[:20]:<22} {str(glm)[:20]:<22}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())