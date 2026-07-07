"""Pre-train the shared JGS backbone (Phase 2a).

Thin CLI over ``src.subconscious.training.pretrain.pretrain_backbone``. Run on
the pod against the ``sequences.jsonl`` produced by
``scripts/extract_backbone_sequences.py``. Defaults match
``BackboneTrainingConfig``; override the SSM backend here (``reference`` for
CPU/GPU dev, ``mamba3-cuda`` for the official kernels on the pod).

Usage (pod, real Mamba3):
    python scripts/train_backbone.py \
        --pairs data/training/backbone/sequences.jsonl \
        --backend mamba3-cuda --device cuda --dtype bfloat16 \
        --checkpoint-dir checkpoints/backbone

Usage (dev / GPU fallback):
    python scripts/train_backbone.py --backend reference --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.configs import BackboneConfig, BackboneTrainingConfig  # noqa: E402
from src.subconscious.training.pretrain import pretrain_backbone  # noqa: E402


def _progress(step: int, train_loss: float, val_loss: float) -> None:
    print(f"[step {step:>5}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-train the JGS backbone on follows-chain sequences")
    p.add_argument("--pairs", default=BackboneTrainingConfig().pairs_path,
                   help="Path to sequences.jsonl from extract_backbone_sequences.py")
    p.add_argument("--backend", default=BackboneConfig().ssm_backend,
                   choices=["reference", "mamba3-pytorch", "mamba3-cuda"],
                   help="SSM backend (reference = pure PyTorch, runs anywhere)")
    p.add_argument("--checkpoint-dir", default=BackboneTrainingConfig().checkpoint_dir)
    p.add_argument("--total-steps", type=int, default=BackboneTrainingConfig().total_steps)
    p.add_argument("--batch-size", type=int, default=BackboneTrainingConfig().batch_size)
    p.add_argument("--device", default=BackboneTrainingConfig().device, help="auto|cpu|cuda")
    p.add_argument("--dtype", default=BackboneTrainingConfig().dtype, help="float32|float16|bfloat16")
    p.add_argument("--val-fraction", type=float, default=BackboneTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int, default=BackboneTrainingConfig().seed)
    args = p.parse_args()

    backbone = BackboneConfig(ssm_backend=args.backend)
    cfg = BackboneTrainingConfig(
        backbone=backbone,
        pairs_path=args.pairs,
        checkpoint_dir=args.checkpoint_dir,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Backbone: d_model={backbone.d_model} n_layers={backbone.n_layers} "
          f"backend={backbone.ssm_backend} (~{cfg.total_steps} steps, batch {cfg.batch_size}, "
          f"{cfg.dtype} on {cfg.device})", flush=True)

    bb = pretrain_backbone(cfg, args.pairs, progress_cb=_progress)

    final_ckpt = Path(cfg.checkpoint_dir) / "backbone_final.pt"
    if not final_ckpt.exists():
        print(f"ERROR: expected checkpoint not found at {final_ckpt}", file=sys.stderr)
        return 1
    n_params = sum(p.numel() for p in bb.parameters())
    print(f"DONE. backbone_final.pt at {final_ckpt}  ({n_params:,} params)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())