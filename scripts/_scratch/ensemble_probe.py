"""Ensemble eval probe for the attention penalty heads (NOT committed -- scratch).

Loads the four attention penalty-head checkpoints (pen=0/1/2/5, all
attention + temporal feature on, trained on the clean 261-train/76-val split)
on the shared frozen JGSBackbone, runs all of them on the 76-doc clean val,
combines via (a) logit-average and (b) majority-vote, and measures the FULL
strict ship gate for each combination/subset:
  unsafe_cell <= 1 AND snapshot_recall >= 0.70 AND decision_update_recall >= 0.70
  AND val_acc >= 0.55 AND snapshot_recall Wilson-CI95 lower bound >= 0.50.

Reuses ``evaluate_doc_kind_per_class`` via an ``EnsembleHead`` wrapper so the
metrics (confusion, Wilson CI, unsafe_cell) are EXACTLY the served-path metrics
(no reimplementation drift). The wrapper exposes ``forward(embs, feat)`` +
``eval()``; for majority-vote it returns a one-hot-ish logit (voted class=1.0,
others=-1e9) so the eval's argmax == the voted class, with tie-break by the
logit-average argmax.

Sanity: the single-head subsets (e.g. {2}) must reproduce the known per-head
scorecards (pen=2 ep20: snap=12/17=0.706, dec=12/17=0.706, unsafe=1, acc=0.553,
CI_lo=0.469). If they don't, the embedder/feat path diverged -- stop.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.doc_kind_head import DocKindHead
from src.subconscious.training.doc_kind_training import (
    DocKindHeadTrainingConfig,
    _embed_sections,
    evaluate_doc_kind_per_class,
    load_doc_kind_pairs,
)
from src.subconscious.training.routing_training import build_embedder, load_backbone, load_doc_kind_head

import math

CKPTS = {
    "pen5": "data/training/doc_kind_head_attn/best.pt",
    "pen0": "data/training/doc_kind_head_attn_ce0/best.pt",
    "pen1": "data/training/doc_kind_head_attn_ce1/best.pt",
    "pen2": "data/training/doc_kind_head_attn_ce2/best.pt",
}
VAL_PATH = "data/training/doc_kind_head/pairs_clean_val.jsonl"
BACKBONE_PATH = DocKindHeadTrainingConfig().backbone_path


def wilson_lo(k, n, z=1.96):
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half)


class EnsembleHead:
    """Wraps N DocKindHeads; forward returns combined logits."""

    def __init__(self, heads, rule="avg"):
        self.heads = heads
        self.rule = rule

    def eval(self):
        for h in self.heads:
            h.eval()

    def parameters(self):
        return iter(self.heads[0].parameters())  # for device detection only

    def forward(self, embs, feat=None):
        logits = [h.forward(embs, feat=feat) for h in self.heads]  # each [1,5]
        stacked = torch.stack(logits, dim=0)  # [H,1,5]
        if self.rule == "avg":
            return stacked.mean(dim=0)  # [1,5]
        if self.rule == "vote":
            n_heads = len(self.heads)
            avg = stacked.mean(dim=0)  # [1,5] for tie-break
            out = torch.full_like(logits[0], -1e9)
            for i in range(stacked.shape[1]):
                votes = torch.tensor([int(l[i].argmax().item()) for l in logits])
                counts = torch.bincount(votes, minlength=5)
                top = int(counts.argmax().item())
                if counts[top] > n_heads / 2:  # strict majority
                    pred = top
                else:  # tie -> logit-average argmax
                    pred = int(avg[i].argmax().item())
                out[i, pred] = 1.0
            return out
        raise ValueError(self.rule)


def gate_verdict(pc):
    snap_k = round(pc["snapshot_recall"] * pc["snapshot_n"])
    ci_lo = wilson_lo(snap_k, pc["snapshot_n"])
    checks = {
        "unsafe<=1": pc["unsafe_cell"] <= 1,
        "snap>=0.70": pc["snapshot_recall"] >= 0.70,
        "dec>=0.70": pc["decision_update_recall"] >= 0.70,
        "acc>=0.55": pc["acc"] >= 0.55,
        "CI_lo>=0.50": ci_lo >= 0.50,
    }
    passed = all(checks.values())
    return checks, passed, ci_lo


def main():
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_str)
    print(f"device={dev_str}", flush=True)

    print(f"Loading frozen backbone from {BACKBONE_PATH}", flush=True)
    backbone = load_backbone(str(BACKBONE_PATH), BackboneConfig(), device=dev_str)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    print(f"Loading val docs from {VAL_PATH}", flush=True)
    val = load_doc_kind_pairs(VAL_PATH)
    print(f"  {len(val)} val docs", flush=True)

    print("Building embedder (on-demand, bge-small) ...", flush=True)
    embedder = build_embedder("on-demand")

    print("Embedding val sections + computing temporal feats ...", flush=True)
    val_embs = [_embed_sections(embedder, rec["section_texts"], dev) for rec in val]
    from src.ingestion.doc_kind import extract_temporal_features
    val_feats = [torch.tensor(extract_temporal_features(rec["section_texts"]),
                             dtype=torch.float32, device=dev).unsqueeze(0)
                 for rec in val]

    # Load all 4 heads on the shared backbone.
    heads = {}
    for name, path in CKPTS.items():
        print(f"  loading {name}: {path}", flush=True)
        heads[name] = load_doc_kind_head(str(path), backbone, device=dev_str)
    print(flush=True)

    # Sanity: pen2 alone must reproduce the known scorecard.
    print("=== SANITY: single-head subsets (must match known per-head scorecards) ===")
    for name in ["pen5", "pen0", "pen1", "pen2"]:
        ens = EnsembleHead([heads[name]], rule="avg")
        pc = evaluate_doc_kind_per_class(ens, val, val_embs, val_feats=val_feats)
        _, passed, ci_lo = gate_verdict(pc)
        print(f"  {name:5s}: snap={pc['snapshot_recall']:.3f}({round(pc['snapshot_recall']*pc['snapshot_n'])}/{pc['snapshot_n']}) "
              f"dec={pc['decision_update_recall']:.3f} unsafe={pc['unsafe_cell']} acc={pc['acc']:.3f} CI_lo={ci_lo:.3f}")
    print(flush=True)

    # Sweep subsets (size >= 2) and both combine rules.
    names = ["pen0", "pen1", "pen2", "pen5"]
    subsets = []
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            subsets.append(list(combo))

    print("=== ENSEMBLE SWEEP (subset | rule -> snap dec unsafe acc CI_lo | GATE) ===")
    results = []
    for subset in subsets:
        for rule in ["avg", "vote"]:
            ens = EnsembleHead([heads[n] for n in subset], rule=rule)
            pc = evaluate_doc_kind_per_class(ens, val, val_embs, val_feats=val_feats)
            checks, passed, ci_lo = gate_verdict(pc)
            tag = "GATE-CLEAR" if passed else ""
            label = f"{'+'.join(subset)}|{rule}"
            print(f"  {label:22s} snap={pc['snapshot_recall']:.3f}({round(pc['snapshot_recall']*pc['snapshot_n'])}/{pc['snapshot_n']}) "
                  f"dec={pc['decision_update_recall']:.3f} unsafe={pc['unsafe_cell']} acc={pc['acc']:.3f} CI_lo={ci_lo:.3f} {tag}")
            if passed:
                snap_k = round(pc["snapshot_recall"] * pc["snapshot_n"])
                results.append((label, pc, checks, ci_lo))
    print(flush=True)

    if results:
        print("=== GATE-CLEARING ENSEMBLES ===")
        for label, pc, checks, ci_lo in results:
            print(f"  {label}: {checks}")
            print(f"    confusion={pc['confusion']}")
            print(f"    recall_per_class={pc['recall_per_class']}")
    else:
        print("=== NO ensemble combination cleared the full strict gate ===")
        # Report the best near-miss (max criteria passed, then max min(snap,dec)).
        print("Best near-miss combos (by # gate criteria passed, then min(snap,dec)):")
        near = []
        for subset in subsets:
            for rule in ["avg", "vote"]:
                ens = EnsembleHead([heads[n] for n in subset], rule=rule)
                pc = evaluate_doc_kind_per_class(ens, val, val_embs, val_feats=val_feats)
                checks, passed, ci_lo = gate_verdict(pc)
                n_pass = sum(checks.values())
                near.append((n_pass, min(pc["snapshot_recall"], pc["decision_update_recall"]),
                             f"{'+'.join(subset)}|{rule}", pc, checks, ci_lo))
        near.sort(key=lambda x: (-x[0], -x[1]))
        for n_pass, msd, label, pc, checks, ci_lo in near[:6]:
            print(f"  {label:22s} pass={n_pass}/5 min(snap,dec)={msd:.3f} "
                  f"snap={pc['snapshot_recall']:.3f} dec={pc['decision_update_recall']:.3f} "
                  f"unsafe={pc['unsafe_cell']} acc={pc['acc']:.3f} CI_lo={ci_lo:.3f} {checks}")


if __name__ == "__main__":
    main()