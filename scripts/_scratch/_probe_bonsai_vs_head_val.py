"""PROBE (uncommitted): Bonsai zero-shot vs the feat-on DocKindHead on the SAME
76-doc v2 val split.

Reproduces scripts/train_doc_kind_head.py's exact seed-0 split (dedup by doc_id
-> random.Random(0).shuffle -> first 76 = val), runs Bonsai classify_doc_kind
on each val doc, and scores with the SAME evaluate_doc_kind_per_class (via a
one-hot-logits dummy head so unsafe_cell / Wilson CI are computed identically).
Compares against the head's feat-on best-epoch scorecard in
data/training/doc_kind_head_abon/train_log.json.

Not committed (scripts/_scratch/). Bonsai must be up at localhost:8080/v1.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.gnn.bonsai_decider import BonsaiDecider
from src.ingestion.doc_kind import join_section_texts
from src.subconscious.doc_kind_head import DocKindHead
from src.subconscious.training.doc_kind_training import (
    evaluate_doc_kind_per_class,
    load_doc_kind_pairs,
)

PAIRS = "data/training/doc_kind_head/pairs_v2.jsonl"
HEAD_LOG = sys.argv[1] if len(sys.argv) > 1 else \
    "data/training/doc_kind_head_abon/train_log.json"
SEED = 0
VAL_FRACTION = 0.2


def _split(records):
    # Dedup by doc_id (or joined section_texts) -- identical to the trainer.
    seen, unique = set(), []
    for rec in records:
        did = rec.get("doc_id") or "\n".join(rec["section_texts"])
        if did in seen:
            continue
        seen.add(did)
        unique.append(rec)
    rng = random.Random(SEED)
    idx = list(range(len(unique)))
    rng.shuffle(idx)
    n_val = max(1, int(len(unique) * VAL_FRACTION))
    return [unique[i] for i in idx[:n_val]]


def _scorecard_from_preds(val, preds):
    """Score (val, pred-labels) with the SAME metric defs as the head's eval.

    A dummy head whose forward returns one-hot logits at the predicted index ->
    evaluate_doc_kind_per_class computes acc / confusion / unsafe_cell / Wilson
    CI identically to the head's scorecard. None pred (Bonsai HTTP/parse/OOV
    failure) -> 'other' (the cold-start contract the guard writes at ingest)."""
    labels = list(DocKindHead.LABELS)
    idx_iter = iter(labels.index(p) if p in labels else labels.index("other")
                    for p in preds)

    class _Dummy:
        # Plain object (not nn.Module): evaluate_doc_kind_per_class only needs
        # .eval() (no-op) + .forward(embs) returning [1, C] logits.
        def eval(self):
            return self

        def forward(self, embs, feat=None):
            i = next(idx_iter)
            out = torch.zeros(1, len(labels))
            out[0, i] = 10.0
            return out

    dummy = _Dummy()
    val_embs = [[torch.zeros(1, 384)] for _ in val]
    return evaluate_doc_kind_per_class(dummy, val, val_embs, val_feats=None)


def main() -> int:
    records = load_doc_kind_pairs(PAIRS)
    val = _split(records)
    print(f"val split: {len(val)} docs (seed {SEED}, frac {VAL_FRACTION})")

    decider = BonsaiDecider()
    if not decider.health_check(timeout=5.0):
        print("ERROR: Bonsai not up at localhost:8080/v1", file=sys.stderr)
        return 1
    print("Bonsai up. Running zero-shot classify_doc_kind on val (one HTTP/doc) ...")

    preds, none_count = [], 0
    for k, rec in enumerate(val):
        text = join_section_texts(rec["section_texts"])
        pred = decider.classify_doc_kind(text)
        if pred is None:
            none_count += 1
        preds.append(pred)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(val)} ...", flush=True)

    print(f"  Bonsai returned None (HTTP/parse/OOV -> cold-start 'other') on "
          f"{none_count}/{len(val)} docs")
    bonsai = _scorecard_from_preds(val, preds)

    head_log = json.load(open(HEAD_LOG, encoding="utf-8"))
    head = head_log["best_per_class"]
    head_epoch = head_log.get("best_epoch", -1)

    labels = list(DocKindHead.LABELS)
    snap = labels.index("point_in_time_snapshot")
    dec = labels.index("decision_update")

    def _row(name, pc):
        ci = pc["snapshot_recall_ci95"]
        return (name, pc["acc"], pc["unsafe_cell"],
                pc["recall_per_class"]["point_in_time_snapshot"],
                pc["recall_per_class"]["decision_update"],
                pc["snapshot_n"], ci[0], ci[1])

    rows = [_row("Bonsai 0-shot", bonsai),
            _row(f"Head feat-on (e{head_epoch})", head)]
    print("\n=== VAL SCORECARD (same 76-doc split) ===")
    print(f"{'tagger':<22} {'acc':>6} {'unsafe':>7} {'snap_r':>7} {'dec_r':>6} "
          f"{'snap_n':>7} {'CI95lo':>7} {'CI95hi':>7}")
    for name, acc, unsafe, sr, dr, sn, lo, hi in rows:
        print(f"{name:<22} {acc:>6.2f} {unsafe:>7} {sr:>7.2f} {dr:>6.2f} "
              f"{sn:>7} {lo:>7.2f} {hi:>7.2f}")

    print("\n=== Bonsai confusion (rows=true, cols=pred; 0=snap 1=dec 2=plan "
          "3=ref 4=other) ===")
    for r in bonsai["confusion"]:
        print(" ", r)
    print("=== Head feat-on confusion ===")
    for r in head["confusion"]:
        print(" ", r)

    # Head-to-head on the two guard-relevant axes.
    bsafe = bonsai["unsafe_cell"]
    hsafe = head["unsafe_cell"]
    print(f"\nunsafe_cell (lower=safe):  Bonsai={bsafe}  Head={hsafe}  "
          f"-> {'HEAD safer' if hsafe < bsafe else 'BONSAI safer' if bsafe < hsafe else 'tie'}")
    print(f"snapshot_recall:          Bonsai={bonsai['recall_per_class']['point_in_time_snapshot']:.2f}  "
          f"Head={head['recall_per_class']['point_in_time_snapshot']:.2f}")
    print(f"decision_update_recall:    Bonsai={bonsai['recall_per_class']['decision_update']:.2f}  "
          f"Head={head['recall_per_class']['decision_update']:.2f}")
    print(f"acc:                       Bonsai={bonsai['acc']:.2f}  Head={head['acc']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())