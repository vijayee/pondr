#!/usr/bin/env python3
"""STRM text2x DocKindHead penalty-frontier SWEEP eval. Uncommitted scratch.

The retrained ce0_text2x (penalty 0.0) + ce2_text2x (penalty 2.0) ensemble
FAILED the strict gate on text2x (snap 9/17, dec 11/17, unsafe 2 vs the
phase2a 13-13-0). The ce0/ce2 pair was tuned for PHASE2A's feature distribution;
text2x's features need a different frontier pair. This script trains no new
heads -- it sweeps over the available text2x-trained penalty heads and scores
EVERY single head + EVERY logit-averaged pair on the 76-doc clean val to find
the pair that clears snap>=0.70 / dec>=0.70 / unsafe<=1 / acc>=0.55 / CI_lo>=0.50.

Mechanism (no reimplementation): ``build_doc_kind_tagger(ensemble_paths=[h])``
builds a 1-head EnsembleBackboneDocKindTagger (= that head's argmax); with
``ensemble_paths=[hA, hB]`` it logit-averages the two heads -- the exact combine
the strict gate was measured on (src/ingestion/doc_kind.py:240-296). So singles
and pairs use the SAME served path; the scorecard IS the served scorecard.

Scorecard copied verbatim from scripts/_scratch/ensemble_serve_gate.py:35-96.

Read-only over the text2x-trained checkpoints. Writes
data/training/strm_relevance/text2x_dockind_sweep_summary.json.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/_scratch/ -> repo root
sys.path.insert(0, str(ROOT))

from src.subconscious.doc_kind_head import DocKindHead  # noqa: E402
from src.subconscious.training.doc_kind_training import load_doc_kind_pairs  # noqa: E402
from src.ingestion.doc_kind import build_doc_kind_tagger  # noqa: E402

TEXT2X_BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full_finetuned_text2x.pt"
DOC_KIND_VAL = "data/training/doc_kind_head/pairs_clean_val.jsonl"
OUT_JSON = "data/training/strm_relevance/text2x_dockind_sweep2_summary.json"
LABELS = DocKindHead.LABELS

# The text2x-trained penalty-frontier heads. ce0..ce3 from the first sweep;
# ce0_5/ce1_5/ce2_5/ce4 from the EXPANDED sweep (user chose "expand penalty
# sweep for a pair" after no pair cleared among ce0..ce3).
HEADS: dict[str, str] = {
    "pen0.0": "data/training/doc_kind_head_attn_ce0_text2x/best.pt",
    "pen0.5": "data/training/doc_kind_head_attn_ce0_5_text2x/best.pt",
    "pen1.0": "data/training/doc_kind_head_attn_ce1_text2x/best.pt",
    "pen1.5": "data/training/doc_kind_head_attn_ce1_5_text2x/best.pt",
    "pen2.0": "data/training/doc_kind_head_attn_ce2_text2x/best.pt",
    "pen2.5": "data/training/doc_kind_head_attn_ce2_5_text2x/best.pt",
    "pen3.0": "data/training/doc_kind_head_attn_ce3_text2x/best.pt",
    "pen4.0": "data/training/doc_kind_head_attn_ce4_text2x/best.pt",
}


def _check_paths() -> None:
    missing = [f"{k}: {v}" for k, v in HEADS.items() if not Path(v).exists()]
    if not Path(TEXT2X_BACKBONE).exists():
        missing.append(f"text2x backbone: {TEXT2X_BACKBONE}")
    if not Path(DOC_KIND_VAL).exists():
        missing.append(f"doc-kind val: {DOC_KIND_VAL}")
    if missing:
        print("ERROR: missing checkpoints/inputs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)


def wilson_lo(k, n, z=1.96):
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half)


def doc_kind_scorecard(tagger, val):
    conf = [[0] * len(LABELS) for _ in range(len(LABELS))]
    unsafe = 0
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
    checks = {
        "unsafe<=1": unsafe <= 1,
        "snap>=0.70": snap >= 0.70,
        "dec>=0.70": dec >= 0.70,
        "acc>=0.55": acc >= 0.55,
        "CI_lo>=0.50": ci_lo >= 0.50,
    }
    return {
        "snap": snap, "snap_k": snap_k, "snap_n": snap_n,
        "dec": dec, "unsafe": unsafe, "acc": acc, "ci_lo": ci_lo,
        "checks": checks, "gate_pass": all(checks.values()),
    }


def score(paths, val):
    """Build the served tagger for the given head path(s) over the text2x
    backbone and score it on the 76-doc val. A 1-element list = single head;
    a 2-element list = logit-averaged pair."""
    tagger = build_doc_kind_tagger(
        ensemble_paths=list(paths),
        backbone_path=TEXT2X_BACKBONE, device="cpu",
        embedder_source="on-demand", verbose=False,
    )
    if tagger is None:
        return None
    return doc_kind_scorecard(tagger, val)


def fmt_row(label, sc):
    if sc is None:
        return f"{label:>14} |  tagger None"
    return (f"{label:>14} | snap={sc['snap']:.3f}({sc['snap_k']}/{sc['snap_n']}) "
            f"dec={sc['dec']:.3f} unsafe={sc['unsafe']} acc={sc['acc']:.3f} "
            f"CI_lo={sc['ci_lo']:.3f} -> {'PASS' if sc['gate_pass'] else 'FAIL'}")


def main() -> int:
    _check_paths()
    print("=== STRM text2x DocKindHead penalty-frontier SWEEP ===", flush=True)
    val = load_doc_kind_pairs(DOC_KIND_VAL)
    print(f"val docs: {len(val)}", flush=True)

    results: dict[str, dict] = {}

    # ── singles ──
    print("\n--- singles ---", flush=True)
    for k, p in HEADS.items():
        sc = score([p], val)
        results[k] = sc
        print(fmt_row(k, sc), flush=True)

    # ── pairs (logit-averaged) ──
    print("\n--- pairs (logit-avg) ---", flush=True)
    pairs = list(itertools.combinations(HEADS.keys(), 2))
    passing_pairs = []
    for a, b in pairs:
        label = f"{a}+{b}"
        sc = score([HEADS[a], HEADS[b]], val)
        results[label] = sc
        print(fmt_row(label, sc), flush=True)
        if sc is not None and sc["gate_pass"]:
            passing_pairs.append(label)

    # ── verdict ──
    # Rank clearing pairs by snap desc, then dec desc, then unsafe asc, then
    # acc desc -- so the SHIP PAIR is the best clearing pair, not just the first.
    def _rank_key(label):
        sc = results[label]
        return (-(sc["snap"]), -(sc["dec"]), sc["unsafe"], -(sc["acc"]))
    passing_pairs = sorted(passing_pairs, key=_rank_key)
    print("\n=== VERDICT ===", flush=True)
    print(f"gate-clearing pairs: {passing_pairs if passing_pairs else 'NONE'}", flush=True)
    if passing_pairs:
        print(f"SHIP PAIR: {passing_pairs[0]} (best clearing pair by "
              f"snap>dec>unsafe>acc)", flush=True)
    else:
        print("No pair clears the strict gate on text2x across the expanded "
              "penalty frontier {0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0}. Options: "
              "more epochs, finer penalty grid, accept DocKindHead stays on "
              "phase2a (hybrid), or abandon text2x-for-dockind.", flush=True)

    out = {k: (v if v is None else {kk: vv for kk, vv in v.items()})
           for k, v in results.items()}
    out_path = Path(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "backbone": TEXT2X_BACKBONE,
        "heads": HEADS,
        "passing_pairs": passing_pairs,
        "results": out,
    }, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0 if passing_pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())