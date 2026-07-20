"""Train the RecoverabilityHead (STRM Phase 2b) -- closed-form ridge fit.

Thin CLI over
``src.subconscious.training.recoverability_training.fit_recoverability``.
Loads the Phase 0a traces, fits the label-generating decoder ``D`` and the
head ``P`` (both closed-form ridge), evaluates P's AUC + the k-baseline on
the held-out val chains, and writes ``best.pt`` == ``final.pt`` +
``train_log.json`` to the checkpoint dir. No backbone, no embedder, no epoch
loop -- the head is the ridge solution baked into an ``nn.Linear(1920, 1)``.

Usage:
    python scripts/train_recoverability_head.py
    python scripts/train_recoverability_head.py --traces data/probe/recoverability/traces.pt

Regenerate the traces first if missing:
    python scripts/generate_strm_traces.py --max-chains 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.recoverability_training import (  # noqa: E402
    RecoverabilityTrainingConfig,
    fit_recoverability,
)
from src.subconscious.training.strm_traces import load_traces  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the STRM RecoverabilityHead (closed-form ridge)")
    p.add_argument("--traces",
                   default=RecoverabilityTrainingConfig().traces_path,
                   help="Phase 0a trace file (per-chain {inputs, states})")
    p.add_argument("--output",
                   default=RecoverabilityTrainingConfig().checkpoint_dir,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
    p.add_argument("--k-max", type=int,
                   default=RecoverabilityTrainingConfig().k_max,
                   help="max lookback horizon k = t - i for (i,t) pairs")
    p.add_argument("--lam", type=float,
                   default=RecoverabilityTrainingConfig().lam,
                   help="ridge penalty (on standardized features) for D and P")
    p.add_argument("--gate-auc", type=float,
                   default=RecoverabilityTrainingConfig().gate_auc,
                   help="GO gate: P val AUC must clear this AND beat k-baseline")
    p.add_argument("--val-fraction", type=float,
                   default=RecoverabilityTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int,
                   default=RecoverabilityTrainingConfig().seed)
    args = p.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: traces not found at {traces_path}. Generate them first:\n"
              f"  python scripts/generate_strm_traces.py --max-chains 400 "
              f"--output {traces_path}", file=sys.stderr)
        return 1

    print(f"Loading STRM traces -> {traces_path}", flush=True)
    traces = load_traces(traces_path)
    n_chains = len(traces)
    if n_chains < 5:
        print(f"ERROR: only {n_chains} chains -- need >=5 to fit + eval. "
              f"Regenerate with --max-chains 400.", file=sys.stderr)
        return 1
    lens = sorted(t["states"].shape[0] for t in traces)
    print(f"  {n_chains} chains (len min/med/max="
          f"{lens[0]}/{lens[n_chains // 2]}/{lens[-1]})", flush=True)

    cfg = RecoverabilityTrainingConfig(
        traces_path=str(traces_path), k_max=args.k_max, lam=args.lam,
        gate_auc=args.gate_auc, val_fraction=args.val_fraction,
        seed=args.seed, checkpoint_dir=args.output,
    )
    print(f"Fitting (closed-form ridge D + P, lam={cfg.lam}, "
          f"k_max={cfg.k_max}, val_fraction={cfg.val_fraction})", flush=True)
    result = fit_recoverability(traces, cfg)

    final_ckpt = Path(cfg.checkpoint_dir) / "final.pt"
    if not final_ckpt.exists():
        print(f"ERROR: head checkpoint not written at {final_ckpt}",
              file=sys.stderr)
        return 1
    print(f"DONE. val AUC={result['ridge_auc']:.4f}  "
          f"k-baseline={result['k_auc']:.4f}  "
          f"-> {'GO' if result['go'] else 'NO-GO'}", flush=True)
    print(f"  final.pt at {final_ckpt}", flush=True)
    print(f"  Next: Phase 4 wires this head's predict() into the salience "
          f"trigger (which past anchor is forgotten enough to surface).",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())