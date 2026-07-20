"""Train the ContextBuilder (STRM Phase 3 -- learned PresentationGate).

Thin CLI over
``src.subconscious.training.context_builder_training.fit_context_builder``.
Reuses the Phase 2a ERAG traces (from ``scripts/generate_relevance_data.py``)
and the FROZEN shipped 2a relevance head (computes ``r_i`` as a constant input
feature). Splits queries 80/20 (the SAME split the 2a trainer used, same seed),
trains the per-slot BCE + Plackett-Luce builder, evaluates gold-coverage vs the
heuristic PresentationGate at equal per-query ``m`` each epoch, and writes
``best.pt`` (gate-selected) + ``final.pt`` + ``train_log.json`` to the
checkpoint dir. No backbone, no embedder at train time.

Usage:
    python scripts/train_context_builder.py
    python scripts/train_context_builder.py \
        --traces data/training/strm_relevance/traces.pt \
        --relevance-head data/training/strm_relevance/best.pt

Prereqs:
    python scripts/generate_relevance_data.py --max-queries 80
    python scripts/train_relevance_head.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.context_builder_training import (  # noqa: E402
    ContextBuilderTrainingConfig,
    fit_context_builder,
)
from src.subconscious.training.relevance_training import (  # noqa: E402
    load_relevance_traces,
)
from src.subconscious.relevance_head import load_relevance_head  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the STRM ContextBuilder (learned PresentationGate, "
                    "BCE + Plackett-Luce, ERAG traces, frozen 2a r_i bias)")
    p.add_argument("--traces",
                   default=ContextBuilderTrainingConfig().traces_path,
                   help="Phase 2a trace file (reused: per-query {query_emb, "
                        "slots_y, slots_doc_emb, source_ids, labels})")
    p.add_argument("--relevance-head",
                   default=ContextBuilderTrainingConfig().relevance_head_path,
                   help="Frozen shipped 2a relevance head checkpoint "
                        "(computes r_i as a constant input feature)")
    p.add_argument("--output",
                   default=ContextBuilderTrainingConfig().checkpoint_dir,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
    p.add_argument("--epochs", type=int,
                   default=ContextBuilderTrainingConfig().epochs)
    p.add_argument("--lr", type=float,
                   default=ContextBuilderTrainingConfig().learning_rate)
    p.add_argument("--weight-decay", type=float,
                   default=ContextBuilderTrainingConfig().weight_decay)
    p.add_argument("--accum-steps", type=int,
                   default=ContextBuilderTrainingConfig().accum_steps)
    p.add_argument("--pos-weight-cap", type=float,
                   default=ContextBuilderTrainingConfig().pos_weight_cap)
    p.add_argument("--pl-weight", type=float,
                   default=ContextBuilderTrainingConfig().pl_weight,
                   help="Listwise Plackett-Luce auxiliary weight (default 0.1)")
    p.add_argument("--val-fraction", type=float,
                   default=ContextBuilderTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int,
                   default=ContextBuilderTrainingConfig().seed)
    p.add_argument("--device",
                   default=ContextBuilderTrainingConfig().device,
                   help="torch device (cpu | cuda)")
    args = p.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: traces not found at {traces_path}. Generate them first:\n"
              f"  python scripts/generate_relevance_data.py --max-queries 80 "
              f"--output {traces_path}", file=sys.stderr)
        return 1
    rel_path = Path(args.relevance_head)
    if not rel_path.exists():
        print(f"ERROR: frozen relevance head not found at {rel_path}. Train it "
              f"first:\n  python scripts/train_relevance_head.py", file=sys.stderr)
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

    print(f"Loading frozen 2a relevance head -> {rel_path}", flush=True)
    frozen_head = load_relevance_head(str(rel_path), device="cpu")

    cfg = ContextBuilderTrainingConfig(
        traces_path=str(traces_path), relevance_head_path=str(rel_path),
        epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay,
        accum_steps=args.accum_steps, pos_weight_cap=args.pos_weight_cap,
        pl_weight=args.pl_weight, val_fraction=args.val_fraction,
        seed=args.seed, device=args.device, checkpoint_dir=args.output,
    )
    print(f"Training (BCE + {cfg.pl_weight}-weighted PL, {cfg.epochs} epochs, "
          f"lr={cfg.learning_rate}, accum={cfg.accum_steps}, "
          f"val_fraction={cfg.val_fraction}, device={cfg.device})", flush=True)
    result = fit_context_builder(traces, frozen_head, cfg)

    best = Path(cfg.checkpoint_dir) / "best.pt"
    if not best.exists():
        print(f"ERROR: builder checkpoint not written at {best}", file=sys.stderr)
        return 1
    print(f"DONE. best epoch={result['best_epoch']} "
          f"cov_learn={result['best_pc']['mean_cov_learn']:.4f} "
          f"cov_heur={result['best_pc']['mean_cov_heur']:.4f} "
          f"cov_r={result['best_pc']['mean_cov_r_only']:.4f} "
          f"lambda_r={result['best_pc']['lambda_r']:.3f} "
          f"-> {'GO' if result['go'] else 'NO-GO'}", flush=True)
    print(f"  best.pt at {best}", flush=True)
    print(f"  Next: serve wiring (--strm-context-builder, default off); flip the "
          f"default once the gate holds with margin.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())