"""Train the RelevanceHead (STRM Phase 2a) -- per-slot BCE on ERAG-Bench traces.

Thin CLI over
``src.subconscious.training.relevance_training.fit_relevance``. Loads the
Phase 2a traces (from ``scripts/generate_relevance_data.py``), splits queries
80/20, trains the per-slot BCE head, evaluates per-query top-3 recall + Wilson
95% CI each epoch, and writes ``best.pt`` (gate-selected) + ``final.pt`` +
``train_log.json`` to the checkpoint dir. No backbone, no embedder at train
time -- the y_t slots + query_emb are precomputed in the traces.

Usage:
    python scripts/train_relevance_head.py
    python scripts/train_relevance_head.py --traces data/training/strm_relevance/traces.pt

Generate the traces first if missing:
    python scripts/generate_relevance_data.py --max-queries 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    fit_relevance,
    load_relevance_traces,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the STRM RelevanceHead (per-slot BCE, ERAG-Bench traces)")
    p.add_argument("--traces",
                   default=RelevanceTrainingConfig().traces_path,
                   help="Phase 2a trace file (per-query {query_emb, slots_y, labels})")
    p.add_argument("--output",
                   default=RelevanceTrainingConfig().checkpoint_dir,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
    p.add_argument("--epochs", type=int,
                   default=RelevanceTrainingConfig().epochs)
    p.add_argument("--lr", type=float,
                   default=RelevanceTrainingConfig().learning_rate)
    p.add_argument("--weight-decay", type=float,
                   default=RelevanceTrainingConfig().weight_decay)
    p.add_argument("--accum-steps", type=int,
                   default=RelevanceTrainingConfig().accum_steps)
    p.add_argument("--pos-weight-cap", type=float,
                   default=RelevanceTrainingConfig().pos_weight_cap)
    p.add_argument("--gate-top3", type=float,
                   default=RelevanceTrainingConfig().gate_top3,
                   help="GO gate: mean per-query top-3 recall must clear this")
    p.add_argument("--gate-wilson-low", type=float,
                   default=RelevanceTrainingConfig().gate_wilson_low,
                   help="GO gate: Wilson 95%% CI lower bound on the per-query "
                        "full-recall hit rate must clear this")
    p.add_argument("--val-fraction", type=float,
                   default=RelevanceTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int,
                   default=RelevanceTrainingConfig().seed)
    args = p.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: traces not found at {traces_path}. Generate them first:\n"
              f"  python scripts/generate_relevance_data.py --max-queries 80 "
              f"--output {traces_path}", file=sys.stderr)
        return 1

    print(f"Loading relevance traces -> {traces_path}", flush=True)
    traces = load_relevance_traces(str(traces_path))
    n = len(traces)
    if n < 3:
        print(f"ERROR: only {n} usable queries -- need >=3 to split + fit. "
              f"Regenerate with --max-queries 80.", file=sys.stderr)
        return 1
    ks = sorted(rec["slots_y"].shape[0] for rec in traces)
    print(f"  {n} queries (K min/med/max={ks[0]}/{ks[n // 2]}/{ks[-1]})", flush=True)

    cfg = RelevanceTrainingConfig(
        traces_path=str(traces_path), epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, accum_steps=args.accum_steps,
        pos_weight_cap=args.pos_weight_cap, gate_top3=args.gate_top3,
        gate_wilson_low=args.gate_wilson_low, val_fraction=args.val_fraction,
        seed=args.seed, checkpoint_dir=args.output,
    )
    print(f"Training (per-slot BCE, {cfg.epochs} epochs, lr={cfg.learning_rate}, "
          f"accum={cfg.accum_steps}, val_fraction={cfg.val_fraction})", flush=True)
    result = fit_relevance(traces, cfg)

    best = Path(cfg.checkpoint_dir) / "best.pt"
    if not best.exists():
        print(f"ERROR: head checkpoint not written at {best}", file=sys.stderr)
        return 1
    print(f"DONE. best epoch={result['best_epoch']} "
          f"top3={result['best_pc']['mean_top3_recall']:.4f} "
          f"hit={result['best_pc']['hit_rate']:.2f} "
          f"-> {'GO' if result['go'] else 'NO-GO'}", flush=True)
    print(f"  best.pt at {best}", flush=True)
    print(f"  Next: Phase 3 context-builder consumes r_i as the slot-selection "
          f"bias; Phase 4 salience trigger combines it with recoverability.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())