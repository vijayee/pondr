"""Train the Retrieval Gate JGS instance (Phase 2b).

Thin CLI over
``src.subconscious.training.routing_training.train_retrieval_gate_supervised``.
Loads the frozen Phase 2a backbone, the Oracle JEPA routing pairs (Phase 1d /
generated this phase), builds the Retrieval Gate, and trains it supervised.
The outcome-based (REINFORCE) stage is wired in the library
(``OutcomeBasedTrainer``) but is driven by live pipeline signals, not this
script — this script produces the supervised checkpoint.

Usage (local CPU, real embeddings):
    python scripts/train_retrieval_gate.py \\
        --pairs data/training/jepa/routing_pairs.jsonl \\
        --embed-source on-demand --device auto --dtype float32

Usage (offline smoke, no model download):
    python scripts/train_retrieval_gate.py --embed-source stub --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.retrieval_gate import RetrievalGate  # noqa: E402
from src.subconscious.training.routing_training import (  # noqa: E402
    RetrievalGateTrainingConfig,
    build_embedder,
    load_backbone,
    load_routing_pairs,
    train_retrieval_gate_supervised,
)


def _progress(epoch: int, train_loss: float, val_acc: float) -> None:
    print(f"[epoch {epoch:>3}] train_loss={train_loss:.4f} val_acc={val_acc:.4f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Train the Retrieval Gate JGS instance")
    p.add_argument("--pairs", default="data/training/jepa/routing_pairs.jsonl",
                   help="Oracle JEPA routing pairs JSONL")
    p.add_argument("--backbone", default=RetrievalGateTrainingConfig().backbone_path,
                   help="Phase 2a backbone checkpoint (backbone_final.pt)")
    p.add_argument("--output", default=RetrievalGateTrainingConfig().checkpoint_dir,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
    p.add_argument("--embed-source", default="on-demand", choices=["on-demand", "stub"],
                   help="on-demand = real bge-small; stub = deterministic hash (smoke only)")
    p.add_argument("--epochs", type=int, default=RetrievalGateTrainingConfig().epochs)
    p.add_argument("--batch-size", type=int, default=RetrievalGateTrainingConfig().batch_size)
    p.add_argument("--lr", type=float, default=RetrievalGateTrainingConfig().learning_rate)
    p.add_argument("--val-fraction", type=float, default=RetrievalGateTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int, default=RetrievalGateTrainingConfig().seed)
    p.add_argument("--device", default=RetrievalGateTrainingConfig().device, help="auto|cpu|cuda")
    p.add_argument("--dtype", default=RetrievalGateTrainingConfig().dtype,
                   help="float32 (bf16/autocast still unfixed in the 2a path)")
    args = p.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: routing pairs not found at {pairs_path}", file=sys.stderr)
        return 1
    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"ERROR: backbone checkpoint not found at {backbone_path}", file=sys.stderr)
        return 1

    print(f"Loading routing pairs from {pairs_path}", flush=True)
    records = load_routing_pairs(str(pairs_path))
    if len(records) < 10:
        print(f"ERROR: only {len(records)} routing pairs — need >=10 to train", file=sys.stderr)
        return 1

    # Dedup by query before splitting. The Oracle cache replays identical query
    # strings (the generator's vocab can produce duplicates), so the raw file may
    # contain the same (query, route) pair many times. Duplicates add no training
    # signal (just class imbalance) and — critically — would let the same query
    # land in BOTH train and val on a random split (leakage). Dedup by query
    # keeps the first occurrence; the split then cannot leak. Always safe.
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        q = rec["query"]
        if q in seen:
            continue
        seen.add(q)
        unique.append(rec)
    if len(unique) < len(records):
        print(f"  dedup: {len(records)} -> {len(unique)} unique queries "
              f"({len(records) - len(unique)} duplicates dropped)", flush=True)
    records = unique
    if len(records) < 10:
        print(f"ERROR: only {len(records)} unique routing pairs — need >=10 to train",
              file=sys.stderr)
        return 1

    # Train/val split (deterministic via seed) on the deduped set.
    import random
    rng = random.Random(args.seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * args.val_fraction))
    val_idx = set(idx[:n_val])
    train_data = [records[i] for i in idx[n_val:]]
    val_data = [records[i] for i in idx[:n_val]]
    print(f"  {len(train_data)} train / {len(val_data)} val (unique)", flush=True)

    print(f"Loading frozen backbone from {backbone_path}", flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    n_bb = sum(p.numel() for p in backbone.parameters())
    print(f"  backbone: {n_bb:,} params (frozen)", flush=True)

    print(f"Building embedder (source={args.embed_source})", flush=True)
    embedder = build_embedder(args.embed_source)

    gate = RetrievalGate(backbone)
    n_gate = sum(p.numel() for p in gate.parameters() if p.requires_grad)
    print(f"  gate trainable params: {n_gate:,} (backbone excluded)", flush=True)

    cfg = RetrievalGateTrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        val_fraction=args.val_fraction, seed=args.seed, device=args.device,
        dtype=args.dtype, checkpoint_dir=args.output,
        backbone_path=str(backbone_path),
    )
    print(f"Training: {cfg.epochs} epochs, batch {cfg.batch_size}, lr {cfg.learning_rate}, "
          f"{cfg.dtype} on {cfg.device}", flush=True)

    result = train_retrieval_gate_supervised(
        gate, backbone, train_data, val_data, embedder, cfg, progress_cb=_progress,
    )

    final_ckpt = Path(cfg.checkpoint_dir) / "final.pt"
    if not final_ckpt.exists():
        print(f"ERROR: gate checkpoint not written at {final_ckpt}", file=sys.stderr)
        return 1
    print(f"DONE. best_val={result['best_val']:.4f}  final.pt at {final_ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())