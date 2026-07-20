"""Train the LatentDynamicsHead (STRM Phase 2c) -- closed-form ridge fit.

Thin CLI over
``src.subconscious.training.latent_dynamics_training.fit_latent_dynamics``.
Loads the Phase 0a traces, ridge-fits ``z_{t+1} = A z_t + b`` on the train
chains' consecutive transitions, evaluates R^2 + surprise-AUC on the held-out
val chains, and writes ``best.pt`` == ``final.pt`` + ``train_log.json`` to
the checkpoint dir. No backbone, no embedder, no epoch loop -- the head is the
closed-form solution baked into an ``nn.Linear(384, 384)``.

Usage:
    python scripts/train_latent_dynamics_head.py
    python scripts/train_latent_dynamics_head.py --traces data/probe/recoverability/traces.pt

Regenerate the traces first if missing:
    python scripts/generate_strm_traces.py --max-chains 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.latent_dynamics_training import (  # noqa: E402
    LatentDynamicsTrainingConfig,
    fit_latent_dynamics,
)
from src.subconscious.training.strm_traces import load_traces  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Train the STRM LatentDynamicsHead (closed-form ridge)")
    p.add_argument("--traces", default=LatentDynamicsTrainingConfig().traces_path,
                   help="Phase 0a trace file (per-chain {inputs, states})")
    p.add_argument("--output", default=LatentDynamicsTrainingConfig().checkpoint_dir,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
    p.add_argument("--lam", type=float, default=LatentDynamicsTrainingConfig().lam,
                   help="ridge penalty (on standardized features)")
    p.add_argument("--r2-gate", type=float, default=LatentDynamicsTrainingConfig().r2_gate,
                   help="GO gate: linear R^2 over the constant-mean baseline")
    p.add_argument("--surprise-auc-gate", type=float,
                   default=LatentDynamicsTrainingConfig().surprise_auc_gate,
                   help="GO gate: correct-vs-mismatched L2-residual surprise-AUC")
    p.add_argument("--val-fraction", type=float,
                   default=LatentDynamicsTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int, default=LatentDynamicsTrainingConfig().seed)
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

    cfg = LatentDynamicsTrainingConfig(
        traces_path=str(traces_path), lam=args.lam,
        r2_gate=args.r2_gate, surprise_auc_gate=args.surprise_auc_gate,
        val_fraction=args.val_fraction, seed=args.seed,
        checkpoint_dir=args.output,
    )
    print(f"Fitting (closed-form ridge, lam={cfg.lam}, val_fraction={cfg.val_fraction})",
          flush=True)
    result = fit_latent_dynamics(traces, cfg)

    final_ckpt = Path(cfg.checkpoint_dir) / "final.pt"
    if not final_ckpt.exists():
        print(f"ERROR: head checkpoint not written at {final_ckpt}", file=sys.stderr)
        return 1
    print(f"DONE. R^2={result['r2']:.4f}  surprise-AUC={result['surprise_auc']:.4f}  "
          f"-> {'GO' if result['go'] else 'NO-GO'}", flush=True)
    print(f"  final.pt at {final_ckpt}", flush=True)
    print(f"  Next: Phase 4 wires this head's surprise() into the salience trigger.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())