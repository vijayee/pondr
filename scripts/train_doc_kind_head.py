"""Train the DocKindHead JGS instance (Phase 3c Sec 7.11 deferred step).

Thin CLI over
``src.subconscious.training.doc_kind_training.train_doc_kind_head_supervised``.
Loads the frozen Phase 2a backbone, the exported doc-kind pairs (section_texts
+ the zero-shot doc_kind label Sec 7.11 wrote at ingest), builds the DocKindHead,
and trains it supervised. The trained head replaces the ingest HTTP tagger with
a local forward pass (``scripts/ingest_document.py`` prefers it automatically
once ``data/training/doc_kind_head/best.pt`` exists).

Usage (export from a live store, then train on real bge-small):
    python scripts/train_doc_kind_head.py \\
        --export-from-db ./data/memory_db \\
        --embed-source on-demand --device auto --dtype float32

Usage (offline smoke, no model download, pre-existing pairs):
    python scripts/train_doc_kind_head.py --embed-source stub --epochs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.doc_kind_head import DocKindHead  # noqa: E402
from src.subconscious.training.doc_kind_training import (  # noqa: E402
    DocKindHeadTrainingConfig,
    export_doc_kind_pairs,
    load_doc_kind_pairs,
    train_doc_kind_head_supervised,
)
from src.subconscious.training.routing_training import build_embedder, load_backbone  # noqa: E402


def _progress(epoch: int, train_loss: float, val_acc: float) -> None:
    print(f"[epoch {epoch:>3}] train_loss={train_loss:.4f} val_acc={val_acc:.4f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Train the DocKindHead JGS instance")
    p.add_argument("--pairs", default=DocKindHeadTrainingConfig().pairs_path,
                   help="doc-kind pairs JSONL (section_texts + label)")
    p.add_argument("--export-from-db", default=None,
                   help="if set, export pairs from this HippocampalStore DB path before training")
    p.add_argument("--db", default=None,
                   help="memory (hot) WaveDB store directory for --export-from-db")
    p.add_argument("--backbone", default=DocKindHeadTrainingConfig().backbone_path,
                   help="Phase 2a backbone checkpoint (backbone_final.pt)")
    p.add_argument("--output", default=DocKindHeadTrainingConfig().checkpoint_dir,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
    p.add_argument("--embed-source", default="on-demand", choices=["on-demand", "stub"],
                   help="on-demand = real bge-small; stub = deterministic hash (smoke only)")
    p.add_argument("--epochs", type=int, default=DocKindHeadTrainingConfig().epochs)
    p.add_argument("--lr", type=float, default=DocKindHeadTrainingConfig().learning_rate)
    p.add_argument("--val-fraction", type=float, default=DocKindHeadTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int, default=DocKindHeadTrainingConfig().seed)
    p.add_argument("--device", default=DocKindHeadTrainingConfig().device, help="auto|cpu|cuda")
    p.add_argument("--dtype", default=DocKindHeadTrainingConfig().dtype,
                   help="float32 (bf16/autocast still unfixed in the 2a path)")
    p.add_argument("--unsafe-penalty", type=float,
                   default=DocKindHeadTrainingConfig().unsafe_confusion_penalty,
                   help="severity-weighted loss knob: extra penalty on the "
                        "snapshot->decision_update confusion (0.0 = plain CE, "
                        "A/B baseline; default 5.0). The reverse direction stays "
                        "on the base CE term (extra ask_user, not unsafe).")
    p.add_argument("--temporal-feature", action="store_true",
                   help="Phase 4: concatenate a doc-level temporal feature "
                        "(date/as-of/decision/plan signal) with the pooled "
                        "embedding before the head -- attacks the mean-pool blind "
                        "spot. Off = the original embedding-only head (A/B).")
    args = p.parse_args()

    # Optional: export pairs from a live store first.
    if args.export_from_db is not None:
        from src.memory.store import HippocampalStore
        db_path = args.db or "./data/memory_db"
        print(f"Exporting doc-kind pairs from {db_path} -> {args.pairs}", flush=True)
        store = HippocampalStore(db_path)
        try:
            n = export_doc_kind_pairs(store, args.pairs)
        finally:
            store.close()
        if n < 10:
            print(f"ERROR: only {n} docs exported -- need >=10 to train. "
                  f"Ingest more docs (with --extract so doc_kind is tagged) first.",
                  file=sys.stderr)
            return 1

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: doc-kind pairs not found at {pairs_path} "
              f"(pass --export-from-db to create it)", file=sys.stderr)
        return 1
    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"ERROR: backbone checkpoint not found at {backbone_path}", file=sys.stderr)
        return 1

    print(f"Loading doc-kind pairs from {pairs_path}", flush=True)
    records = load_doc_kind_pairs(str(pairs_path))
    if len(records) < 10:
        print(f"ERROR: only {len(records)} doc-kind pairs -- need >=10 to train",
              file=sys.stderr)
        return 1

    # Dedup by doc_id (a re-ingest writes the same id; duplicates add no signal
    # and could leak across the train/val split). Keep first occurrence.
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        did = rec.get("doc_id") or "\n".join(rec["section_texts"])
        if did in seen:
            continue
        seen.add(did)
        unique.append(rec)
    if len(unique) < len(records):
        print(f"  dedup: {len(records)} -> {len(unique)} unique docs "
              f"({len(records) - len(unique)} duplicates dropped)", flush=True)
    records = unique
    if len(records) < 10:
        print(f"ERROR: only {len(records)} unique docs -- need >=10 to train",
              file=sys.stderr)
        return 1

    # Train/val split (deterministic via seed).
    import random
    rng = random.Random(args.seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * args.val_fraction))
    val_data = [records[i] for i in idx[:n_val]]
    train_data = [records[i] for i in idx[n_val:]]
    print(f"  {len(train_data)} train / {len(val_data)} val (unique)", flush=True)

    print(f"Loading frozen backbone from {backbone_path}", flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    n_bb = sum(p.numel() for p in backbone.parameters())
    print(f"  backbone: {n_bb:,} params (frozen)", flush=True)

    print(f"Building embedder (source={args.embed_source})", flush=True)
    embedder = build_embedder(args.embed_source)

    # Phase 4: --temporal-feature widens the head's first Linear to accept the
    # temporal feature vector (feat_dim=TEMPORAL_FEAT_DIM). Off = feat_dim=0
    # (the original embedding-only head, the A/B baseline).
    if args.temporal_feature:
        from src.ingestion.doc_kind import TEMPORAL_FEAT_DIM
        feat_dim = TEMPORAL_FEAT_DIM
        print(f"  temporal feature ON: head.feat_dim={feat_dim}", flush=True)
    else:
        feat_dim = 0
    head = DocKindHead(backbone, feat_dim=feat_dim)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  head trainable params: {n_head:,} (backbone excluded)", flush=True)

    cfg = DocKindHeadTrainingConfig(
        epochs=args.epochs, learning_rate=args.lr,
        val_fraction=args.val_fraction, seed=args.seed, device=args.device,
        dtype=args.dtype, checkpoint_dir=args.output,
        backbone_path=str(backbone_path), embedder_source=args.embed_source,
        pairs_path=str(pairs_path),
        unsafe_confusion_penalty=args.unsafe_penalty,
        temporal_feature=args.temporal_feature,
    )
    print(f"Training: {cfg.epochs} epochs, lr {cfg.learning_rate}, "
          f"{cfg.dtype} on {cfg.device}, "
          f"unsafe_penalty={cfg.unsafe_confusion_penalty} "
          f"temporal_feature={cfg.temporal_feature}", flush=True)

    result = train_doc_kind_head_supervised(
        head, backbone, train_data, val_data, embedder, cfg, progress_cb=_progress,
    )

    final_ckpt = Path(cfg.checkpoint_dir) / "final.pt"
    if not final_ckpt.exists():
        print(f"ERROR: head checkpoint not written at {final_ckpt}", file=sys.stderr)
        return 1
    print(f"DONE. best_val={result['best_val']:.4f}  final.pt at {final_ckpt}", flush=True)
    print(f"Next: scripts/ingest_document.py --source <path> will now use this "
          f"head as the doc-kind tagger (no :8080 call).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())