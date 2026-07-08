"""Train the Phase 3a 5-head GNN on the regenerated labels (Task 4a).

Thin CLI over ``src.gnn.train.train_gnn``. Pod-ready + CPU-testable: float32,
one subgraph per step, ASCII-only output. The compact corpus DB (the generation
snapshot) is opened read-only here -- the loader's BFS is the same walk the
label generator used, so a training example and its labels are over the same
node/edge set by construction (zero skew). See ``src/gnn/train.py`` + ADR 010
for the store-backed pod path (Path A) + the ``--head`` joint/per-head design.

Usage (pod, the real run -- Task 4b):
    python scripts/train_gnn.py \
        --db data/compact_corpus.db \
        --labels data/training/gnn/ \
        --head all --device cuda --epochs 50

    # optional: refine one head on top of the frozen joint backbone
    python scripts/train_gnn.py --db ... --labels ... \
        --head salience --backbone-checkpoint data/pod_runs/phase3a/all.pt \
        --device cuda --epochs 20

Usage (CPU dev smoke):
    python scripts/train_gnn.py --db <tmp db> --labels <tmp labels> \
        --head all --epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402  (hard dep via src.gnn.*; module-top, NOT local -- a local
#            import inside main() bit run_consolidation.py: a module-level helper
#            referenced torch before main ran. Keep it top-level here.)

from src.gnn.train import GNNTrainConfig, HEAD_CHOICES, train_gnn  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402


def _progress(step: int, train_loss: float, val_loss: float) -> None:
    # ASCII-only (Windows cp1252 crashes on U+2192/U+2500 in print). val_loss is
    # NaN until the first epoch's validation completes.
    if val_loss != val_loss:  # NaN check
        print(f"[epoch end step {step:>5}] train_loss={train_loss:.4f}", flush=True)
    else:
        print(f"[epoch end step {step:>5}] train_loss={train_loss:.4f} "
              f"val={val_loss:.4f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Train the Phase 3a 5-head GNN on regenerated labels")
    p.add_argument("--db", required=True,
                   help="Path to the compact corpus WaveDB store (the generation snapshot; "
                        "opened read-only during training)")
    p.add_argument("--labels", default="data/training/gnn/",
                   help="Dir with the *_labels.jsonl files + quality_report.json from Task 3")
    p.add_argument("--head", choices=list(HEAD_CHOICES), default="all",
                   help="all = joint multi-task run (saves all.pt + per-head .pt); "
                        "<one> = train that head only (cluster trains the diffpool head)")
    p.add_argument("--checkpoint-dir", default=GNNTrainConfig().checkpoint_dir,
                   help="Where to write {head}.pt (+ sidecar .meta.json)")
    p.add_argument("--backbone-checkpoint", default=None,
                   help="Load a full GNN checkpoint and freeze the GAT backbone; only "
                        "meaningful with --head <one> (refine that head on shared features)")
    p.add_argument("--epochs", type=int, default=GNNTrainConfig().epochs)
    p.add_argument("--lr", type=float, default=GNNTrainConfig().lr)
    p.add_argument("--device", default=GNNTrainConfig().device, help="auto|cpu|cuda")
    p.add_argument("--dtype", default=GNNTrainConfig().dtype,
                   help="float32 (bf16/autocast still unfixed in the 2a SSM path; "
                        "the GNN is independent of it but kept fp32 for the cold start)")
    p.add_argument("--val-fraction", type=float, default=GNNTrainConfig().val_fraction)
    p.add_argument("--seed", type=int, default=GNNTrainConfig().seed)
    p.add_argument("--ogb-pretrain", action="store_true",
                   help="Pod-only OGB pretrain-then-transfer (DEFERRED -- raises a clear "
                        "error; direct-train is the cold-start fallback). Lazy import.")
    args = p.parse_args()

    cfg = GNNTrainConfig(
        epochs=args.epochs, lr=args.lr, device=args.device, dtype=args.dtype,
        val_fraction=args.val_fraction, seed=args.seed,
        checkpoint_dir=args.checkpoint_dir, head=args.head,
        backbone_checkpoint=args.backbone_checkpoint, ogb_pretrain=args.ogb_pretrain,
    )

    print(f"GNN: hidden={cfg.hidden_dim} layers={cfg.num_layers} heads={cfg.num_heads} "
          f"head={cfg.head} (~{cfg.epochs} epochs, {cfg.dtype} on {cfg.device})",
          flush=True)

    store = HippocampalStore(args.db)
    try:
        summary = train_gnn(cfg, store, Path(args.labels), progress_cb=_progress)
    finally:
        store.close()

    # Verify the checkpoint(s) exist + are strict-loadable (mirrors the
    # consolidation loader, which does torch.load + load_state_dict on a bare
    # state_dict).
    ok = True
    for ckpt in summary["checkpoints"]:
        path = Path(ckpt)
        if not path.exists():
            print(f"ERROR: expected checkpoint not found at {path}", file=sys.stderr)
            ok = False
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        n_params = sum(v.numel() for v in state.values())
        print(f"  saved {path.name}  ({n_params:,} params, strict-loadable)", flush=True)
    if not ok:
        return 1

    for note in summary["notes"]:
        print(f"  NOTE: {note}", flush=True)
    print(f"DONE. head={summary['head']} steps={summary['steps']} "
          f"params={summary['param_count']:,} "
          f"wall={summary['wall_clock_s']:.1f}s "
          f"checkpoints={len(summary['checkpoints'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())