#!/usr/bin/env python3
"""STRM forgetting-validation (DeepSeek consult #4 gate). Uncommitted scratch.

Question: does the text2x flat_last fine-tune (commit 6756e94, all 19.5M params)
break the LIVE heads that were trained against the Phase 2a backbone? The live
engine (build_ponder/serve_ponder) defaults to the Phase 2a backbone_final.pt;
the Phase 2b RetrievalGate (val 0.826) and the DocKindHead ensemble (snap/dec
13/17, unsafe 0, acc 0.632, CI_lo 0.527) were trained on that backbone's feature
distribution. Wiring a fine-tuned STRM backbone would swap the backbone under
those heads without retraining them.

Two questions, separated by a 4-backbone matrix:
  (1) Fine-tune FORGETTING  = text2x vs its PARENT backbone_v2_full.pt
      (both loaded under the SAME trained heads; only the backbone varies).
  (2) INTEGRATION GAP       = STRM backbones vs Phase 2a (the heads' training
      backbone). If parent itself is far below 0.826 / 13-13-0, the STRM
      backbone was NEVER drop-in compatible -> wiring ANY STRM backbone needs
      the joint multi-task retrain, regardless of forgetting.

The Phase 2a column is ALSO the harness-validation control: it must reproduce
0.826 / 13-13-0, or this eval script is wrong (do not trust the other columns).

Plus a HEAD-INDEPENDENT feature-shift proxy: capture gate.forward's output
[N,256] (the literal input to the trained routing heads -- retrieval_gate.py
:107,115) under each backbone and measure pairwise L2 + 1-cosine. Same gate
checkpoint loaded over each backbone -> the output difference is driven PURELY
by the backbone. Control: phase2a-vs-phase2a = 0 (identical input).

ERAG Decider+guards 200/200 is backbone-independent (BonsaiDecider has no
backbone ref) -> skipped. The STRM z_logit gate is the objective the fine-tune
OPTIMIZED -> already measured (text2x passes), not a forgetting check.

Read-only over existing checkpoints: no live engine touched, no trained
artifacts modified. Writes data/training/strm_relevance/forget_validation_summary.json.
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

# ── the 4-backbone matrix ──
BACKBONES: dict[str, str] = {
    # heads' TRAINING backbone -> baseline-repro control (must reproduce 0.826 /
    # 13-13-0). If it does not, the eval is wrong.
    "phase2a": "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt",
    # text2x fine-tune PARENT -> the forgetting baseline (text2x vs this).
    "parent": "data/training/strm_backbone_relevance/backbone_v2_full.pt",
    # the candidate.
    "text2x": "data/training/strm_backbone_relevance/backbone_v2_full_finetuned_text2x.pt",
    # the other candidate -> ship tie-breaker (does Stage 3 forget less?).
    "stage3": "data/training/strm_backbone_relevance/backbone_v2_full_finetuned.pt",
}
GATE_CKPT = "data/pod_runs/phase2b/best.pt"                  # trained Phase 2b gate (excludes backbone)
ROUTING_PAIRS = "data/training/jepa/routing_pairs.jsonl"     # Phase 2b val split source
DOC_KIND_VAL = "data/training/doc_kind_head/pairs_clean_val.jsonl"  # 76-doc DocKindHead val
OUT_JSON = "data/training/strm_relevance/forget_validation_summary.json"

# Baselines the Phase 2a column MUST reproduce (harness-validation gate).
GATE_BASELINE = 0.82586387434555
N_VAL_EXPECTED = 191  # seed 0, val_fraction 0.2, dedup-by-query (train_log.json)
LABELS = DocKindHead.LABELS  # [point_in_time_snapshot, decision_update, plan, reference, other]


def _check_paths() -> None:
    """Loud exit on any missing checkpoint (mirror train_retrieval_gate.py:64-70)."""
    missing = []
    for name, p in BACKBONES.items():
        if not Path(p).exists():
            missing.append(f"{name}: {p}")
    for p in (GATE_CKPT, ROUTING_PAIRS, DOC_KIND_VAL):
        if not Path(p).exists():
            missing.append(p)
    if missing:
        print("ERROR: missing checkpoints/inputs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)


def reconstruct_val_split() -> list[dict]:
    """Reproduce train_retrieval_gate.py:73-109 EXACTLY: load -> dedup by query
    (keep first) -> random.Random(0).shuffle -> n_val = max(1, int(len*0.2)) ->
    val = first n_val. Asserts n_val == N_VAL_EXPECTED (else the split diverged
    and the Phase 2a column will not reproduce 0.826)."""
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
              f"(unique pairs={len(unique)}). The Phase 2a column will NOT "
              f"reproduce 0.826 -- the val split diverged from the baseline run.",
              file=sys.stderr)
    print(f"[split] routing pairs: {len(records)} -> {len(unique)} unique -> "
          f"{n_val} val (expected {N_VAL_EXPECTED})", flush=True)
    return val_data


# ── Harness A: Phase 2b RetrievalGate ──
def run_gate(val_data, val_emb, dev):
    """Per backbone: load gate ckpt over the backbone, eval routing accuracy,
    and capture gate.forward output [N,256] (the proxy feature). Returns
    {"acc": float, "output": Tensor [N,256] cpu}."""
    rows = {}
    for name, path in BACKBONES.items():
        print(f"[gate] backbone={name} ...", flush=True)
        backbone = load_backbone(path, BackboneConfig(), device="cpu")
        gate = load_retrieval_gate(GATE_CKPT, backbone, device="cpu")
        acc = evaluate_routing(gate, val_data, val_emb, dev)
        # Capture the proxy feature in a separate forward (gate.forward's 3rd
        # return = the trained routing heads' input [N,256]). Same gate ckpt over
        # each backbone -> output diff driven purely by the backbone.
        with torch.no_grad():
            gate.reset_state(len(val_data), device=dev, dtype=torch.float32)
            _, _, output = gate.forward(val_emb.to(dev))
        rows[name] = {"acc": float(acc), "output": output.detach().cpu()}
        print(f"[gate] {name}: val_acc={acc:.5f} (baseline {GATE_BASELINE:.5f}, "
              f"delta {acc - GATE_BASELINE:+.5f})", flush=True)
        del backbone, gate
    return rows


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


def run_doc_kind(val):
    rows = {}
    for name, path in BACKBONES.items():
        print(f"[dockind] backbone={name} ...", flush=True)
        tagger = build_doc_kind_tagger(
            ensemble_paths=list(DEFAULT_DOC_KIND_ENSEMBLE_PATHS),
            backbone_path=path, device="cpu",
            embedder_source="on-demand", verbose=False,
        )
        if tagger is None:
            print(f"[dockind] {name}: tagger None (ensemble ckpts absent?)", file=sys.stderr)
            rows[name] = None
            continue
        sc = doc_kind_scorecard(tagger, val)
        rows[name] = sc
        print(f"[dockind] {name}: snap={sc['snap']:.3f}({sc['snap_k']}/{sc['snap_n']}) "
              f"dec={sc['dec']:.3f} unsafe={sc['unsafe']} acc={sc['acc']:.3f} "
              f"CI_lo={sc['ci_lo']:.3f} -> {'PASS' if sc['gate_pass'] else 'FAIL'}",
              flush=True)
    return rows


# ── Feature-shift proxy ──
def proxy_shifts(gate_rows):
    """Pairwise mean L2 + mean angular distance (1 - cosine) on the captured
    gate outputs [N,256]. Includes the phase2a-vs-phase2a control (must be 0)."""
    names = list(BACKBONES.keys())
    out = {n: gate_rows[n]["output"] for n in names}
    shifts = {}
    for a in names:
        for b in names:
            oa, ob = out[a], out[b]
            l2 = (oa - ob).norm(dim=-1).mean().item()
            ang = (1.0 - torch.nn.functional.cosine_similarity(oa, ob, dim=-1)).mean().item()
            shifts[f"{a}->{b}"] = {"l2": l2, "angular": ang}
    return shifts


def main() -> int:
    _check_paths()
    dev = torch.device("cpu")
    print("=== STRM forgetting-validation ===", flush=True)
    print(f"backbones: {list(BACKBONES.keys())}", flush=True)

    # ── Harness A setup: reconstruct val split + embed once ──
    val_data = reconstruct_val_split()
    print("[embed] building on-demand embedder (bge-small) ...", flush=True)
    embedder = build_embedder("on-demand")
    val_emb = _embed_all(embedder, [q["query"] for q in val_data], dev)
    print(f"[embed] val_emb={tuple(val_emb.shape)}", flush=True)

    # ── Harness A: Phase 2b gate ──
    print("\n--- Harness A: Phase 2b RetrievalGate (baseline 0.826) ---", flush=True)
    gate_rows = run_gate(val_data, val_emb, dev)

    # ── Harness B: DocKindHead ensemble ──
    print("\n--- Harness B: DocKindHead ensemble (baseline 13/13/0/0.632/0.527) ---",
          flush=True)
    dk_val = load_doc_kind_pairs(DOC_KIND_VAL)
    print(f"[dockind] val docs: {len(dk_val)}", flush=True)
    dk_rows = run_doc_kind(dk_val)

    # ── Feature-shift proxy ──
    print("\n--- Feature-shift proxy (gate.forward output [N,256]) ---", flush=True)
    shifts = proxy_shifts(gate_rows)
    for k, v in shifts.items():
        print(f"  {k:>16}: L2={v['l2']:.4f}  angular={v['angular']:.4f}", flush=True)

    # ── verdicts ──
    p2a_acc = gate_rows["phase2a"]["acc"]
    # Harness validated iff phase2a reproduces gate (within 0.02) AND DocKindHead
    # passes the strict ship gate (snap/dec/unsafe/acc/CI_lo).
    gate_repro = abs(p2a_acc - GATE_BASELINE) < 0.02
    dk_p2a_pass = bool(dk_rows["phase2a"]["gate_pass"]) if dk_rows["phase2a"] else False
    harness_validated = gate_repro and dk_p2a_pass
    # Proxy control: phase2a-vs-phase2a must be ~0.
    ctrl = shifts["phase2a->phase2a"]
    proxy_control_ok = (ctrl["l2"] < 1e-5 and ctrl["angular"] < 1e-5)

    # Pure fine-tune forgetting = text2x vs parent.
    forget_acc_delta = gate_rows["text2x"]["acc"] - gate_rows["parent"]["acc"]
    forget_proxy = shifts["parent->text2x"]

    # Integration gap = parent vs phase2a (was the STRM backbone ever drop-in?).
    integ_acc_delta = gate_rows["parent"]["acc"] - p2a_acc

    print("\n=== VERDICT ===", flush=True)
    print(f"harness_validated={harness_validated}  "
          f"(gate_repro={gate_repro}: phase2a={p2a_acc:.5f} vs {GATE_BASELINE:.5f}; "
          f"dk_p2a_pass={dk_p2a_pass})", flush=True)
    print(f"proxy_control_ok={proxy_control_ok}  (phase2a->phase2a L2={ctrl['l2']:.2e} "
          f"angular={ctrl['angular']:.2e})", flush=True)
    print(f"forgetting text2x-vs-parent: gate_acc_delta={forget_acc_delta:+.5f}  "
          f"proxy L2={forget_proxy['l2']:.4f} angular={forget_proxy['angular']:.4f}",
          flush=True)
    print(f"integration gap parent-vs-phase2a: gate_acc_delta={integ_acc_delta:+.5f}",
          flush=True)
    if not harness_validated:
        print("!! HARNESS NOT VALIDATED -- the Phase 2a column did NOT reproduce "
              "the baselines. The eval is wrong; do NOT trust the other columns.",
              file=sys.stderr)

    # ── summary table ──
    print("\n=== MATRIX ===", flush=True)
    hdr = f"{'backbone':>8} | {'gate_acc':>9} | {'snap':>6} | {'dec':>6} | {'unsafe':>6} | {'acc':>6} | {'CI_lo':>6} | {'dk_gate':>7}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for name in BACKBONES:
        g = gate_rows[name]["acc"]
        dk = dk_rows[name]
        if dk is None:
            print(f"{name:>8} | {g:.5f} |   n/a  |   n/a  |   n/a  |   n/a  |   n/a  |   n/a ",
                  flush=True)
        else:
            print(f"{name:>8} | {g:.5f} | {dk['snap']:.3f} | {dk['dec']:.3f} | "
                  f"{dk['unsafe']:>6} | {dk['acc']:.3f} | {dk['ci_lo']:.3f} | "
                  f"{'PASS' if dk['gate_pass'] else 'FAIL':>7}", flush=True)

    # ── write JSON ──
    summary = {
        "backbones": BACKBONES,
        "baselines": {"gate": GATE_BASELINE, "n_val_expected": N_VAL_EXPECTED},
        "gate": {n: {"acc": gate_rows[n]["acc"]} for n in BACKBONES},
        "doc_kind": {n: ({k: v for k, v in dk_rows[n].items() if k != "confusion"}
                          if dk_rows[n] else None) for n in BACKBONES},
        "proxy": shifts,
        "verdicts": {
            "harness_validated": harness_validated,
            "gate_repro": gate_repro,
            "dk_p2a_pass": dk_p2a_pass,
            "proxy_control_ok": proxy_control_ok,
            "forget_text2x_vs_parent_acc_delta": forget_acc_delta,
            "forget_text2x_vs_parent_proxy": forget_proxy,
            "integration_gap_parent_vs_phase2a_acc_delta": integ_acc_delta,
        },
    }
    out_path = Path(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0 if harness_validated else 2


if __name__ == "__main__":
    raise SystemExit(main())