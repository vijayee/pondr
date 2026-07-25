"""End-to-end serve-path gate check for the shipped ensemble (NOT committed).

Exercises the REAL serve entrypoint -- ``build_doc_kind_tagger(ensemble_paths=
...) -> EnsembleBackboneDocKindTagger.classify_doc_kind`` -- on every doc in the
76-doc clean val, builds the confusion matrix, and checks the FULL strict ship
gate. This is NOT the eval-probe combine (which averaged ``head.forward`` logits
inside ``EnsembleHead``); this is the served object re-embedding + recomputing
the temporal feature + averaging + argmax per doc. If the served scorecard
matches the probe (snap 0.765 / dec 0.765 / unsafe 0 / acc 0.632 / CI_lo 0.527),
the serve path has zero drift from the measured gate.

Confusion layout (LABELS order):
  [point_in_time_snapshot, decision_update, plan, reference, other]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.doc_kind import (
    DEFAULT_DOC_KIND_ENSEMBLE_PATHS,
    EnsembleBackboneDocKindTagger,
    build_doc_kind_tagger,
)
from src.subconscious.doc_kind_head import DocKindHead
from src.subconscious.training.doc_kind_training import load_doc_kind_pairs

VAL_PATH = "data/training/doc_kind_head/pairs_clean_val.jsonl"
LABELS = DocKindHead.LABELS


def wilson_lo(k, n, z=1.96):
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half)


def main():
    val = load_doc_kind_pairs(VAL_PATH)
    print(f"val docs: {len(val)}", flush=True)
    print(f"ensemble paths: {list(DEFAULT_DOC_KIND_ENSEMBLE_PATHS)}", flush=True)
    tagger = build_doc_kind_tagger(
        ensemble_paths=list(DEFAULT_DOC_KIND_ENSEMBLE_PATHS),
        device="auto", verbose=True,
    )
    if tagger is None:
        print("FAIL: tagger is None (ensemble ckpts absent?)", flush=True)
        return 1
    print(f"tagger: {type(tagger).__name__}", flush=True)
    assert isinstance(tagger, EnsembleBackboneDocKindTagger)

    # Confusion[true][pred].
    conf = [[0] * len(LABELS) for _ in range(len(LABELS))]
    unsafe = 0  # snap -> dec
    for rec in val:
        gold = rec["label"]
        pred = tagger.classify_doc_kind(rec["section_texts"])
        if pred is None:
            pred = "other"
        ti = LABELS.index(gold)
        pi = LABELS.index(pred)
        conf[ti][pi] += 1
        if gold == "point_in_time_snapshot" and pred == "decision_update":
            unsafe += 1

    n_per = [sum(row) for row in conf]
    recall = [conf[i][i] / n_per[i] if n_per[i] else 0.0 for i in range(len(LABELS))]
    snap = recall[LABELS.index("point_in_time_snapshot")]
    dec = recall[LABELS.index("decision_update")]
    snap_n = n_per[LABELS.index("point_in_time_snapshot")]
    snap_k = conf[LABELS.index("point_in_time_snapshot")][LABELS.index("point_in_time_snapshot")]
    ci_lo = wilson_lo(snap_k, snap_n)
    total = sum(n_per)
    acc = sum(conf[i][i] for i in range(len(LABELS))) / total if total else 0.0

    print(f"\nconfusion (rows=true, cols=pred; order={list(LABELS)}):")
    for i, row in enumerate(conf):
        print(f"  {LABELS[i]:24s} {row}")
    print(f"\nsnap={snap:.3f}({snap_k}/{snap_n}) dec={dec:.3f} unsafe={unsafe} "
          f"acc={acc:.3f} CI_lo={ci_lo:.3f}")
    checks = {
        "unsafe<=1": unsafe <= 1,
        "snap>=0.70": snap >= 0.70,
        "dec>=0.70": dec >= 0.70,
        "acc>=0.55": acc >= 0.55,
        "CI_lo>=0.50": ci_lo >= 0.50,
    }
    print(f"gate: {checks}")
    print(f"GATE {'PASS' if all(checks.values()) else 'FAIL'}", flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())