"""Task #43: train-on-lmsys / eval-Onyx regularized z_logit TRANSFER probe.

Task #41 ([[pondr-strm-task41-serve-zrgate-saturation]]) found the flat-readout
CompositeZHead FAILS the SERVE z_r gate (saturation) AND does not robustly clear
the z_logit gate held-out on 114 Onyx serve turns -- 934K-2.5M params on ~91
train turns overfits, and ~23 val turns make the per-source z_logit gap median
noisy. The user's chosen framing (2026-07-21): ``lmsys/lmsys-chat-1m`` is a
SUPPLEMENT to escape the overfit, then RE-GATE on real Onyx (lmsys is a means,
not a new target distribution). This probe is that test.

Train a ``CompositeZHead`` (``StateReadout`` flat_last + ``ZRelevanceHead``) on
the lmsys serve-like traces (task #42, ``generate_lmsys_serve_traces.py`` --
thousands of turns, ~66x+ the Onyx train set), optionally regularized (heavier
``weight_decay``, smaller readout), via the SAME ``fit_relevance`` task #41 used.
Then EVAL on the existing Onyx serve traces
(``traces_serve_identity_hraw.pt``, 114 turns) measuring the per-source z_logit
gap (2.0 gate) ON ONYX -- the TRANSFER test: does a head regularized on lots of
serve-like lmsys data clear z_logit on real Onyx serve (the task #41 lever firmed
up)?

  GO  = Onyx z_logit gap median >= 2.0, ROBUST across --seeds (>= 2/3 pass).
        -> the flat readout IS the ship lever; the overfit was the blocker, and
           the signal transfers lmsys -> Onyx. Re-run the live SERVE gate.
  NO-GO = Onyx z_logit < 2.0 across seeds.
        -> either the lever is genuinely weak (the z_i bilinear has no decisive
           margin on serve even with data) OR it doesn't transfer
           (conversational context retrieval != ingested-document recall). Either
           way, do NOT ship on this evidence; report which.

Also reports (a) the z_r gap (0.2 gate) on Onyx, (b) an lmsys HELD-OUT sanity
(does the head learn lmsys at all -- a NO here means the training itself failed,
not the transfer), and (c) an lmsys ALL-TURNS ceiling.

Reuses ``probe_serve_composite_zrgate`` helpers (``_load_serve_traces``,
``_zr_per_slot``, ``_zr_and_logit_gaps``) so the metric is byte-identical to task
#41 -- the only difference is WHERE the head is trained (lmsys) vs evaluated
(Onyx). Offline: no backbone, no WM, no embedder, no live probe. CPU-fine for the
small composite; --device cuda trains faster on the larger lmsys set.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))        # sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import probe_serve_composite_zrgate as p41  # noqa: E402
from src.subconscious.state_readout import (  # noqa: E402
    CompositeZHead,
    load_composite_z_head,
)
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    _split_queries,
    fit_relevance,
)

DEFAULT_LMSYS = "data/training/strm_relevance/traces_lmsys_serve_hraw.pt"
DEFAULT_ONYX = "data/training/strm_relevance/traces_serve_identity_hraw.pt"
DEFAULT_OUT = "data/training/strm_relevance/lmsys_to_onyx_zlogit.json"
DEFAULT_CKPT_ROOT = "data/training/strm_state_readout/lmsys_to_onyx"

# z_logit gate threshold (pre-sigmoid per-source gap median). z_r gate is 0.2
# (kept for reporting; task #41 showed z_r saturates on serve).
ZLOGIT_GATE = 2.0
ZR_GATE = 0.2


def _to_device(traces: list[dict], device: str) -> list[dict]:
    """Move every tensor field in each record to the train/eval device. fit_relevance
    does not move inputs to the head's device itself (task #41 ran on CPU so it
    never surfaced); this keeps the head + inputs on the same device."""
    if device == "cpu":
        return traces
    dev = torch.device(device)
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in rec.items()}
            for rec in traces]


def _train_reg(name: str, traces: list[dict], hidden: int | None,
               weight_decay: float, epochs: int, seed: int, device: str,
               ckpt_dir: Path) -> dict:
    """Train one regularized CompositeZHead on the lmsys traces via fit_relevance.
    Like p41._train_composite but exposes ``weight_decay`` (the regularization
    knob) and writes best.pt (gate-selected on the lmsys val split) to ckpt_dir."""
    dim_in = int(traces[0]["slots_h_raw"].shape[1])
    head = CompositeZHead(dim_in=dim_in, hidden=hidden)
    arch = f"MLP-{hidden}" if hidden else "Linear"
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\ntraining {name} seed={seed} ({arch} {dim_in}->384, {n_params:,} params, "
          f"wd={weight_decay}, {epochs} epochs) -> {ckpt_dir}", flush=True)
    cfg = RelevanceTrainingConfig(
        epochs=epochs, seed=seed, device=device,
        checkpoint_dir=str(ckpt_dir),
        slot_signal_field="slots_h_raw",
        weight_decay=weight_decay,
    )
    result = fit_relevance(traces, cfg, head=head)
    return {"name": name, "arch": arch, "hidden": hidden, "dim_in": dim_in,
            "n_params": n_params, "weight_decay": weight_decay,
            "best_epoch": result["best_epoch"],
            "lmsys_train_top3": result["best_pc"]["mean_top3_recall"],
            "lmsys_train_ci": result["best_pc"]["hit_ci95"],
            "lmsys_train_go": result["go"], "ckpt": ckpt_dir / "best.pt"}


def _run_one(name: str, lmsys: list[dict], onyx: list[dict], hidden: int | None,
             weight_decay: float, epochs: int, seed: int, device: str,
             ckpt_root: Path) -> dict:
    """Train on lmsys (one seed), eval on Onyx + lmsys held-out + lmsys ceiling."""
    ckpt_dir = ckpt_root / f"{name.replace(' ','')}_s{seed}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    r = _train_reg(name, lmsys, hidden, weight_decay, epochs, seed, device, ckpt_dir)
    composite = load_composite_z_head(str(r["ckpt"]), device=device,
                                      map_location=device)
    # lmsys held-out sanity (replicate fit_relevance's val split for this seed).
    _, val_idx = _split_queries(len(lmsys), RelevanceTrainingConfig().val_fraction,
                                seed)
    r["lmsys_heldout"] = p41._zr_and_logit_gaps(
        composite, [lmsys[i] for i in val_idx], device)
    r["lmsys_allturns"] = p41._zr_and_logit_gaps(composite, lmsys, device)
    # The TRANSFER eval: z_r + z_logit gaps on the REAL Onyx serve traces.
    r["onyx"] = p41._zr_and_logit_gaps(composite, onyx, device)
    r["seed"] = seed
    ho = r["lmsys_heldout"]
    on = r["onyx"]
    print(f"  lmsys held-out z_logit={ho['z_logit']['median']:.3f} "
          f"(n_ge_2.0={ho['z_logit']['n_ge_gate']}/{ho['z_logit']['n_eligible']})  "
          f"ONYX z_logit={on['z_logit']['median']:.3f} "
          f"(n_ge_2.0={on['z_logit']['n_ge_gate']}/{on['z_logit']['n_eligible']}, "
          f"{'PASS' if on['z_logit']['median'] is not None and on['z_logit']['median'] >= ZLOGIT_GATE else 'fail'})  "
          f"ONYX z_r={on['z_r']['median']:.4f}", flush=True)
    return r


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task #43: train-on-lmsys / eval-Onyx regularized z_logit "
                    "TRANSFER probe (the task #41 lever firmed up with more data).")
    p.add_argument("--lmsys", default=DEFAULT_LMSYS,
                   help="lmsys serve-like traces (generate_lmsys_serve_traces.py).")
    p.add_argument("--onyx", default=DEFAULT_ONYX,
                   help="Onyx serve traces (the transfer target; task #41's set).")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated train seeds (robustness sweep).")
    p.add_argument("--readout", default="mlp128",
                   choices=["linear", "mlp64", "mlp128"],
                   help="StateReadout arch (smaller = more regularized).")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight_decay (heavier = more regularized; try 0.1).")
    p.add_argument("--device", default="cpu",
                   help="train+eval device (cuda trains faster on the lmsys set).")
    p.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    if not Path(args.lmsys).exists():
        print(f"ERROR: lmsys traces not found at {args.lmsys}\n"
              f"  run: python scripts/generate_lmsys_serve_traces.py",
              file=sys.stderr)
        return 1
    if not Path(args.onyx).exists():
        print(f"ERROR: onyx traces not found at {args.onyx}", file=sys.stderr)
        return 1

    lmsys = p41._load_serve_traces(args.lmsys)
    onyx = p41._load_serve_traces(args.onyx)
    # Move tensors to the train/eval device (fit_relevance does not move inputs
    # itself; the gap eval moves per-record, but pre-moving avoids repeated H2D).
    lmsys = _to_device(lmsys, args.device)
    onyx = _to_device(onyx, args.device)
    if len(lmsys) < 50:
        print(f"ERROR: only {len(lmsys)} lmsys records (need >=50 for a real "
              f"regularization test)", file=sys.stderr)
        return 1
    if len(onyx) < 5:
        print(f"ERROR: only {len(onyx)} onyx records", file=sys.stderr)
        return 1
    lk = sorted(r["slots_h_raw"].shape[0] for r in lmsys)
    ok = sorted(r["slots_h_raw"].shape[0] for r in onyx)
    print(f"lmsys: {len(lmsys)} turns (K min/med/max={lk[0]}/{lk[len(lk)//2]}/{lk[-1]}), "
          f"dim_in={lmsys[0]['slots_h_raw'].shape[1]}", flush=True)
    print(f"onyx:  {len(onyx)} turns (K min/med/max={ok[0]}/{ok[len(ok)//2]}/{ok[-1]})  "
          f"[transfer target]", flush=True)

    hidden = {"linear": None, "mlp64": 64, "mlp128": 128}[args.readout]
    name = args.readout
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ckpt_root = Path(args.ckpt_root)

    per_seed = []
    for seed in seeds:
        per_seed.append(_run_one(name, lmsys, onyx, hidden, args.weight_decay,
                                 args.epochs, seed, args.device, ckpt_root))

    # ── aggregate across seeds ──
    onyx_zlogit_meds = [r["onyx"]["z_logit"]["median"] for r in per_seed
                        if r["onyx"]["z_logit"]["median"] is not None]
    onyx_zr_meds = [r["onyx"]["z_r"]["median"] for r in per_seed
                    if r["onyx"]["z_r"]["median"] is not None]
    lmsys_ho_zlogit_meds = [r["lmsys_heldout"]["z_logit"]["median"] for r in per_seed
                            if r["lmsys_heldout"]["z_logit"]["median"] is not None]
    n_onyx_pass = sum(1 for m in onyx_zlogit_meds if m >= ZLOGIT_GATE)
    # Robust = at least 2 seeds pass AND a majority pass (>=2/3 for 3 seeds).
    robust_pass = n_onyx_pass >= 2 and n_onyx_pass * 2 >= len(seeds)

    print("\n" + "=" * 78)
    print("VERDICT (task #43: train-on-lmsys / eval-Onyx z_logit TRANSFER probe)")
    print("=" * 78)
    print(f"  readout={args.readout}  weight_decay={args.weight_decay}  "
          f"seeds={seeds}  z_logit gate={ZLOGIT_GATE}")
    print(f"  lmsys train turns: {len(lmsys)}  (~{len(lmsys)//91}x the Onyx train set)")
    print()
    for r in per_seed:
        on = r["onyx"]; ho = r["lmsys_heldout"]
        print(f"  seed {r['seed']} (best ep {r['best_epoch']}): "
              f"lmsys_train_top3={r['lmsys_train_top3']:.3f}  "
              f"lmsys_heldout z_logit={ho['z_logit']['median']:.3f}  "
              f"ONYX z_logit={on['z_logit']['median']:.3f} "
              f"({'PASS' if on['z_logit']['median'] and on['z_logit']['median']>=ZLOGIT_GATE else 'fail'})  "
              f"ONYX z_r={on['z_r']['median']:.4f}")
    print()
    if onyx_zlogit_meds:
        print(f"  ONYX z_logit median across seeds: "
              f"{statistics.median(onyx_zlogit_meds):.3f}  "
              f"(per-seed: {['%.3f'%m for m in onyx_zlogit_meds]})")
        print(f"  ONYX z_logit passes {n_onyx_pass}/{len(onyx_zlogit_meds)} seeds  "
              f"-> {'ROBUST PASS' if robust_pass else 'NOT robust'}")
    if lmsys_ho_zlogit_meds:
        print(f"  lmsys held-out z_logit median: "
              f"{statistics.median(lmsys_ho_zlogit_meds):.3f}  "
              f"(sanity: head learns lmsys -> "
              f"{'YES' if statistics.median(lmsys_ho_zlogit_meds) >= ZLOGIT_GATE else 'weak'})")
    print()
    if robust_pass:
        print("  -> TRANSFER GO: a flat-readout composite regularized on lots of lmsys")
        print("     serve-like data clears the z_logit gate (>= 2.0) on REAL Onyx serve,")
        print("     robust across seeds. The task #41 overfit was the blocker, and the")
        print("     signal TRANSFERS (conversational context retrieval -> Onyx doc recall).")
        print("     The flat readout IS the ship lever. NEXT: re-run the live SERVE gate")
        print("     (probe_strm_selectivity_real.py with the composite wired in).")
    elif n_onyx_pass > 0:
        print("  -> PARTIAL: some seeds pass Onyx z_logit but not robustly. The signal is")
        print("     real but unstable -- try heavier regularization (--weight-decay 0.1,")
        print("     --readout linear) or more lmsys data before calling the lever.")
    elif lmsys_ho_zlogit_meds and statistics.median(lmsys_ho_zlogit_meds) >= ZLOGIT_GATE:
        print("  -> TRANSFER FAIL (lmsys sanity PASS): the head LEARNS lmsys (held-out")
        print("     z_logit >= 2.0 on lmsys) but does NOT transfer to Onyx. The lever is")
        print("     sound in-distribution but conversational context retrieval != Onyx")
        print("     ingested-document recall -- a distribution gap lmsys can't bridge.")
        print("     Needs real Onyx transcripts (or accept lmsys as the gate).")
    else:
        print("  -> TRANSFER FAIL (lmsys sanity also weak): the head doesn't decisively")
        print("     clear z_logit even on lmsys held-out. The z_i bilinear has no robust")
        print("     decisive margin on serve-like data at ALL -- the lever is genuinely")
        print("     weak, not just overfit. Do NOT ship; reconsider the cross-slot")
        print("     trajectory Transformer or a de-saturating margin loss.")
    print("=" * 78)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "readout": args.readout, "weight_decay": args.weight_decay,
            "epochs": args.epochs, "seeds": seeds,
            "n_lmsys": len(lmsys), "n_onyx": len(onyx),
            "onyx_zlogit_median_across_seeds": (statistics.median(onyx_zlogit_meds)
                                                if onyx_zlogit_meds else None),
            "n_onyx_zlogit_pass": n_onyx_pass, "robust_pass": robust_pass,
            "per_seed": [{"seed": seeds[i], "best_epoch": r["best_epoch"],
                          "lmsys_train_top3": r["lmsys_train_top3"],
                          "lmsys_heldout": r["lmsys_heldout"],
                          "onyx": r["onyx"]} for i, r in enumerate(per_seed)],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())