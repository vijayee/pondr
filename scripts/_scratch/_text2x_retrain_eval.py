#!/usr/bin/env python3
"""STRM text2x retrain RECOVERY eval (DeepSeek consult #4 fork A). Uncommitted scratch.

Forgetting-validation (scripts/_scratch/_forget_validation.py) showed the
INTEGRATION GAP blocks a direct swap: the existing Phase 2a-trained live heads
collapse on the text2x STRM backbone (gate 0.826 -> 0.258; DocKindHead
13-13-0 -> 0-0-1) because the heads expect Phase 2a features and the text2x
backbone lives in a near-orthogonal subspace (proxy angular 0.92). The
fine-tune forgetting itself was a wash (text2x vs parent gate delta +0.00000).

Fork (A) -- chosen by the user -- is to RETRAIN the live heads on the frozen
text2x backbone so the heads learn the new subspace. This script measures
whether that retrain RECOVERS the live baselines on the text2x backbone:

  (1) Phase 2b RetrievalGate  -- retrained gate (data/pod_runs/phase2b_text2x/
      best.pt) over the text2x backbone, evaluated on the SAME Phase 2b val
      split (seed 0, 0.2, dedup-by-query) -> comparable to baseline 0.82586.
  (2) DocKindHead ensemble    -- retrained pen0+pen2 pair (logit-avg) over the
      text2x backbone, scored on the 76-doc clean val -> comparable to
      snap/dec 0.765 / unsafe 0 / acc 0.632 / CI_lo 0.527.

It ALSO runs the Phase 2a column (existing heads over the Phase 2a backbone) as
a self-validation control: it must reproduce 0.82586 + 13-13-0, or this eval is
wrong (do not trust the text2x column).

Read-only over existing + newly-trained checkpoints: no live engine touched, no
trained artifacts modified. Writes
data/training/strm_relevance/text2x_retrain_eval_summary.json.

Acceptance: gate_recovered = text2x_gate_acc >= 0.80 AND >> 0.258 floor;
dk_recovered = text2x ensemble strict-gate PASS (snap>=0.70, dec>=0.70,
unsafe<=1, acc>=0.55, CI_lo>=0.50). ship_ready = both.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/_scratch/ -> repo root
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.doc_kind_head import DocKindHead  # noqa: E402
from src.subconscious.training.routing_training import (  # noqa: E402
    _embed_all,
    build_embedder,
    evaluate_routing,
    load_backbone,
    load_routing_pairs,
    load_retrieval_gate,
)
from src.subconscious.training.doc_kind_training import load_doc_kind_pairs  # noqa: E402
from src.ingestion.doc_kind import (  # noqa: E402
    DEFAULT_DOC_KIND_ENSEMBLE_PATHS,
    build_doc_kind_tagger,
)

# ── the two columns ──
PHASE2A_BACKBONE = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
TEXT2X_BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full_finetuned_text2x.pt"

# Existing (Phase 2a-trained) heads -> self-validation control.
OLD_GATE = "data/pod_runs/phase2b/best.pt"
OLD_ENSEMBLE = list(DEFAULT_DOC_KIND_ENSEMBLE_PATHS)

# NEW text2x-retrained heads -> the recovery measurement.
NEW_GATE = "data/pod_runs/phase2b_text2x/best.pt"
NEW_ENSEMBLE = [
    "data/training/doc_kind_head_attn_ce0_text2x/best.pt",
    "data/training/doc_kind_head_attn_ce2_text2x/best.pt",
]

ROUTING_PAIRS = "data/training/jepa/routing_pairs.jsonl"
DOC_KIND_VAL = "data/training/doc_kind_head/pairs_clean_val.jsonl"
OUT_JSON = "data/training/strm_relevance/text2x_retrain_eval_summary.json"

GATE_BASELINE = 0.82586387434555
N_VAL_EXPECTED = 191
GATE_FLOOR = 0.258  # text2x under the OLD (Phase 2a-trained) heads -- the floor to beat
GATE_RECOVERY_MIN = 0.80  # recovery threshold (baseline is 0.826)
LABELS = DocKindHead.LABELS


def _check_paths() -> None:
    """Loud exit on any missing checkpoint (mirror train_retrieval_gate.py:64-70)."""
    needed = [
        ("phase2a backbone", PHASE2A_BACKBONE),
        ("text2x backbone", TEXT2X_BACKBONE),
        ("old gate", OLD_GATE),
        ("new gate", NEW_GATE),
        ("routing pairs", ROUTING_PAIRS),
        ("doc-kind val", DOC_KIND_VAL),
    ]
    for en in (OLD_ENSEMBLE, NEW_ENSEMBLE):
        for p in en:
            needed.append(("ensemble head", p))
    missing = [f"{label}: {p}" for label, p in needed if not Path(p).exists()]
    if missing:
        print("ERROR: missing checkpoints/inputs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)


def reconstruct_val_split() -> list[dict]:
    """Reproduce train_retrieval_gate.py:73-109 EXACTLY (mirror
    _forget_validation.py:104-130). Asserts n_val == N_VAL_EXPECTED."""
    records = load_routing_pairs(ROUTING_PAIRS)
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        q = rec["query"]
        if q in seen:
            continue
        seen.add(q)
        unique.append(rec)
    rng = random.Random(0)
    idx = list(range(len(unique)))
    rng.shuffle(idx)
    n_val = max(1, int(len(unique) * 0.2))
    val_data = [unique[i] for i in idx[:n_val]]
    if n_val != N_VAL_EXPECTED:
        print(f"WARNING: reconstructed n_val={n_val} != expected {N_VAL_EXPECTED} "
              f"(unique pairs={len(unique)}). The control column will NOT "
              f"reproduce 0.826 -- the val split diverged.", file=sys.stderr)
    print(f"[split] routing pairs: {len(records)} -> {len(unique)} unique -> "
          f"{n_val} val (expected {N_VAL_EXPECTED})", flush=True)
    return val_data


# ── Harness A: Phase 2b RetrievalGate ──
def run_gate(backbone_path: str, gate_ckpt: str, val_data, val_emb, dev, label: str) -> float:
    """Load a gate ckpt over the given backbone, eval routing accuracy."""
    print(f"[gate] {label}: backbone={backbone_path} gate={gate_ckpt}", flush=True)
    backbone = load_backbone(backbone_path, BackboneConfig(), device="cpu")
    gate = load_retrieval_gate(gate_ckpt, backbone, device="cpu")
    acc = evaluate_routing(gate, val_data, val_emb, dev)
    print(f"[gate] {label}: val_acc={acc:.5f} (baseline {GATE_BASELINE:.5f}, "
          f"delta {acc - GATE_BASELINE:+.5f}, floor {GATE_FLOOR})", flush=True)
    del backbone, gate
    return float(acc)


# ── Harness B: DocKindHead ensemble ──
# Scorecard logic copied verbatim from scripts/_scratch/ensemble_serve_gate.py:35-96.
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
        "confusion": conf,
    }


def run_doc_kind(backbone_path: str, ensemble_paths: list[str], val, label: str):
    print(f"[dockind] {label}: backbone={backbone_path} ensemble={ensemble_paths}",
          flush=True)
    tagger = build_doc_kind_tagger(
        ensemble_paths=list(ensemble_paths),
        backbone_path=backbone_path, device="cpu",
        embedder_source="on-demand", verbose=False,
    )
    if tagger is None:
        print(f"[dockind] {label}: tagger None (ensemble ckpts absent?)", file=sys.stderr)
        return None
    sc = doc_kind_scorecard(tagger, val)
    print(f"[dockind] {label}: snap={sc['snap']:.3f}({sc['snap_k']}/{sc['snap_n']}) "
          f"dec={sc['dec']:.3f} unsafe={sc['unsafe']} acc={sc['acc']:.3f} "
          f"CI_lo={sc['ci_lo']:.3f} -> {'PASS' if sc['gate_pass'] else 'FAIL'}",
          flush=True)
    return sc


def main() -> int:
    _check_paths()
    dev = torch.device("cpu")
    print("=== STRM text2x retrain RECOVERY eval ===", flush=True)

    # ── Harness A setup: reconstruct val split + embed once ──
    val_data = reconstruct_val_split()
    print("[embed] building on-demand embedder (bge-small) ...", flush=True)
    embedder = build_embedder("on-demand")
    val_emb = _embed_all(embedder, [q["query"] for q in val_data], dev)
    print(f"[embed] val_emb={tuple(val_emb.shape)}", flush=True)

    # ── Harness A: gate ──
    print("\n--- Harness A: Phase 2b RetrievalGate (baseline 0.826) ---", flush=True)
    ctrl_gate_acc = run_gate(PHASE2A_BACKBONE, OLD_GATE, val_data, val_emb, dev,
                             "control: phase2a-old-heads")
    new_gate_acc = run_gate(TEXT2X_BACKBONE, NEW_GATE, val_data, val_emb, dev,
                            "RETRAIN: text2x-new-heads")

    # ── Harness B: DocKindHead ensemble ──
    print("\n--- Harness B: DocKindHead ensemble (baseline 13/13/0/0.632/0.527) ---",
          flush=True)
    dk_val = load_doc_kind_pairs(DOC_KIND_VAL)
    print(f"[dockind] val docs: {len(dk_val)}", flush=True)
    ctrl_dk = run_doc_kind(PHASE2A_BACKBONE, OLD_ENSEMBLE, dk_val,
                           "control: phase2a-old-heads")
    new_dk = run_doc_kind(TEXT2X_BACKBONE, NEW_ENSEMBLE, dk_val,
                          "RETRAIN: text2x-new-heads")

    # ── verdicts ──
    gate_repro = abs(ctrl_gate_acc - GATE_BASELINE) < 0.02
    dk_ctrl_pass = bool(ctrl_dk["gate_pass"]) if ctrl_dk else False
    harness_validated = gate_repro and dk_ctrl_pass

    gate_recovered = (new_gate_acc >= GATE_RECOVERY_MIN) and (new_gate_acc > GATE_FLOOR + 0.10)
    dk_recovered = bool(new_dk["gate_pass"]) if new_dk else False
    ship_ready = harness_validated and gate_recovered and dk_recovered

    print("\n=== VERDICT ===", flush=True)
    print(f"harness_validated={harness_validated}  "
          f"(gate_repro={gate_repro}: control={ctrl_gate_acc:.5f} vs {GATE_BASELINE:.5f}; "
          f"dk_ctrl_pass={dk_ctrl_pass})", flush=True)
    print(f"gate_recovered={gate_recovered}  (text2x gate={new_gate_acc:.5f}; "
          f"need>={GATE_RECOVERY_MIN} and >> floor {GATE_FLOOR}; "
          f"old-heads-on-text2x floor was 0.25752)", flush=True)
    print(f"dk_recovered={dk_recovered}  (text2x ensemble strict-gate "
          f"{'PASS' if dk_recovered else 'FAIL'})", flush=True)
    print(f"ship_ready={ship_ready}", flush=True)
    if not harness_validated:
        print("!! HARNESS NOT VALIDATED -- the control column did NOT reproduce "
              "the baselines. The eval is wrong; do NOT trust the text2x column.",
              file=sys.stderr)

    # ── summary table ──
    print("\n=== MATRIX ===", flush=True)
    hdr = (f"{'column':>28} | {'gate_acc':>9} | {'snap':>6} | {'dec':>6} | "
           f"{'unsafe':>6} | {'acc':>6} | {'CI_lo':>6} | {'dk_gate':>7}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    def _row(label, g, dk):
        if dk is None:
            print(f"{label:>28} | {g:.5f} |   n/a  |   n/a  |   n/a  |   n/a  | "
                  f"  n/a  |   n/a  ", flush=True)
        else:
            print(f"{label:>28} | {g:.5f} | {dk['snap']:.3f} | {dk['dec']:.3f} | "
                  f"{dk['unsafe']:>6} | {dk['acc']:.3f} | {dk['ci_lo']:.3f} | "
                  f"{'PASS' if dk['gate_pass'] else 'FAIL':>7}", flush=True)
    _row("control phase2a-old-heads", ctrl_gate_acc, ctrl_dk)
    _row("RETRAIN text2x-new-heads", new_gate_acc, new_dk)

    # ── write JSON ──
    summary = {
        "backbones": {"phase2a": PHASE2A_BACKBONE, "text2x": TEXT2X_BACKBONE},
        "gate_ckpts": {"old": OLD_GATE, "new": NEW_GATE},
        "ensemble_paths": {"old": OLD_ENSEMBLE, "new": NEW_ENSEMBLE},
        "baselines": {"gate": GATE_BASELINE, "n_val_expected": N_VAL_EXPECTED,
                      "gate_floor": GATE_FLOOR, "gate_recovery_min": GATE_RECOVERY_MIN},
        "gate": {"control_phase2a": ctrl_gate_acc, "retrain_text2x": new_gate_acc},
        "doc_kind": {
            "control_phase2a": ({k: v for k, v in ctrl_dk.items() if k != "confusion"}
                                if ctrl_dk else None),
            "retrain_text2x": ({k: v for k, v in new_dk.items() if k != "confusion"}
                               if new_dk else None),
        },
        "verdicts": {
            "harness_validated": harness_validated,
            "gate_repro": gate_repro,
            "dk_ctrl_pass": dk_ctrl_pass,
            "gate_recovered": gate_recovered,
            "dk_recovered": dk_recovered,
            "ship_ready": ship_ready,
        },
    }
    out_path = Path(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0 if harness_validated else 2


if __name__ == "__main__":
    raise SystemExit(main())