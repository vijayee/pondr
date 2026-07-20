"""Train the GraduationHeadV2 (STRM Phase 2d) -- v2 beats the v1 r_i proxy.

Thin CLI over
``src.subconscious.training.graduation_training.fit_graduation``. Loads the
LABELED replay log (``replay_labeled.jsonl`` from
``scripts/generate_graduation_labels.py``), trains the v2 MLP classifier on
the precomputed per-slot features (pooled WM state 1536 + slot readout 256 +
llm_signal one-hot 5), and writes ``best.pt`` + ``final.pt`` +
``train_log.json`` to the checkpoint dir. The gate: v2 AUC must beat the v1
``integral(r_i dt)`` proxy's AUC on the held-out val split AND clear the
chance floor. No backbone, no embedder at train time -- the features are
precomputed in the replay log.

The TRAINING RUN is deferred until enough sessions have been logged with
``--strm-graduation-logging`` that ``replay_labeled.jsonl`` has labeled slots
across sessions. The CODE here + the synthetic tests land now.

Usage:
    python scripts/train_graduation_head.py
    python scripts/train_graduation_head.py \\
        --replay data/training/strm_graduation/replay_labeled.jsonl

Prereq: generate the labels first:
    python scripts/fetch_onyx_sessions.py --limit 50   # populate sessions.jsonl
    python scripts/generate_graduation_labels.py        # replay.jsonl -> labeled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.graduation_training import (  # noqa: E402
    GraduationTrainingConfig,
    fit_graduation,
    load_replay_labeled,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the STRM GraduationHeadV2 (v2 beats the v1 r_i proxy)")
    p.add_argument("--replay",
                   default=GraduationTrainingConfig().replay_path,
                   help="replay_labeled.jsonl (labeled replay log)")
    p.add_argument("--output",
                   default=GraduationTrainingConfig().checkpoint_dir,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
    p.add_argument("--epochs", type=int,
                   default=GraduationTrainingConfig().epochs)
    p.add_argument("--lr", type=float,
                   default=GraduationTrainingConfig().learning_rate)
    p.add_argument("--weight-decay", type=float,
                   default=GraduationTrainingConfig().weight_decay)
    p.add_argument("--accum-steps", type=int,
                   default=GraduationTrainingConfig().accum_steps)
    p.add_argument("--pos-weight-cap", type=float,
                   default=GraduationTrainingConfig().pos_weight_cap,
                   help="cap on the BCE pos_weight (positives are rare; an "
                        "uncapped weight destabilizes the MLP)")
    p.add_argument("--gate-auc-min", type=float,
                   default=GraduationTrainingConfig().gate_auc_min,
                   help="GO gate: v2 AUC must beat v1 AUC AND clear this floor")
    p.add_argument("--min-val-n", type=int,
                   default=GraduationTrainingConfig().min_val_n,
                   help="minimum val slots for the AUC gate to be meaningful")
    p.add_argument("--v1-dt", type=float,
                   default=GraduationTrainingConfig().v1_dt,
                   help="v1 proxy time-step (r_i stream integral)")
    p.add_argument("--val-fraction", type=float,
                   default=GraduationTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int,
                   default=GraduationTrainingConfig().seed)
    args = p.parse_args()

    replay_path = Path(args.replay)
    if not replay_path.exists():
        print(f"ERROR: labeled replay not found at {replay_path}.\n"
              f"  Generate it first:\n"
              f"    python scripts/generate_graduation_labels.py",
              file=sys.stderr)
        return 1
    print(f"Loading labeled replay -> {replay_path}", flush=True)
    records = load_replay_labeled(str(replay_path))
    if not records:
        print(f"ERROR: no labeled records in {replay_path}. Need >= "
              f"{2 * args.min_val_n} labeled slots across sessions; run more "
              f"serve sessions with --strm-graduation-logging.",
              file=sys.stderr)
        return 1
    n_pos = sum(1 for r in records if r["later_needed"])
    print(f"  {len(records)} labeled slots ({n_pos} positive)", flush=True)

    cfg = GraduationTrainingConfig(
        replay_path=str(replay_path), checkpoint_dir=args.output,
        epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay,
        accum_steps=args.accum_steps, pos_weight_cap=args.pos_weight_cap,
        gate_auc_min=args.gate_auc_min, min_val_n=args.min_val_n,
        v1_dt=args.v1_dt, val_fraction=args.val_fraction, seed=args.seed,
    )
    print(f"Training v2 MLP (epochs={cfg.epochs}, lr={cfg.learning_rate}, "
          f"pos_weight_cap={cfg.pos_weight_cap}, v1_dt={cfg.v1_dt})",
          flush=True)
    result = fit_graduation(records, cfg)

    final_ckpt = Path(cfg.checkpoint_dir) / "final.pt"
    if not final_ckpt.exists():
        print(f"ERROR: head checkpoint not written at {final_ckpt}",
              file=sys.stderr)
        return 1
    print(f"DONE. v2_auc={result['best_v2_auc']:.4f}  "
          f"v1_auc={result['best_v1_auc']:.4f}  "
          f"-> {'GO' if result['go'] else 'NO-GO'}", flush=True)
    print(f"  best.pt at {Path(cfg.checkpoint_dir) / 'best.pt'}", flush=True)
    print(f"  Next: Phase 4 wires this head's predict() into the LTM-promotion "
          f"path (a slot graduates when v2 P(later_needed) clears threshold).",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())