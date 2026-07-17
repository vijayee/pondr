"""Reproduce the v2 seed-0 split and build the v3 train/val files.

The Bonsai-vs-head probe measures the head on the 76-doc v2 val split (seed 0,
val_fraction 0.2). To make a v3 retrain (v2 + synthetic) comparable to that probe
BEFORE/AFTER, the val must be the EXACT same 76 real docs -- synthetic must NOT
leak into val. This script reproduces the trainer's split logic verbatim
(dedup by doc_id -> random.Random(seed).shuffle -> first n_val = val) and writes:

  pairs_v3_train.jsonl = 304 real v2 train + ALL synthetic (synthetic is TRAIN only)
  pairs_v3_val.jsonl   = 76 real v2 val (zero synthetic)

Feed both to scripts/train_doc_kind_head.py --train ... --val ... so the trainer
skips its internal seed-split and trains on exactly this train set while scoring
on the fixed real val.

One-time offline data prep (NOT prod traffic). ``data/`` is gitignored.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(path: str) -> list[dict]:
    recs: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def _dedup(records: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for rec in records:
        did = rec.get("doc_id") or "\n".join(rec["section_texts"])
        if did in seen:
            continue
        seen.add(did)
        unique.append(rec)
    return unique


def _dist(records: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in records:
        out[rec["label"]] = out.get(rec["label"], 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the v3 doc-kind train/val split.")
    ap.add_argument("--v2-pairs", default="data/training/doc_kind_head/pairs_v2.jsonl",
                    help="v2 real corpus (the seed-0 split source)")
    ap.add_argument("--synth-pairs", default="data/training/doc_kind_head/pairs_synth.jsonl",
                    help="synthetic pairs (TRAIN only)")
    ap.add_argument("--out-train", default="data/training/doc_kind_head/pairs_v3_train.jsonl")
    ap.add_argument("--out-val", default="data/training/doc_kind_head/pairs_v3_val.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()

    v2 = _dedup(_load(args.v2_pairs))
    print(f"v2 unique: {len(v2)} (label dist {_dist(v2)})", flush=True)

    # Reproduce the trainer's split verbatim.
    rng = random.Random(args.seed)
    idx = list(range(len(v2)))
    rng.shuffle(idx)
    n_val = max(1, int(len(v2) * args.val_fraction))
    val = [v2[i] for i in idx[:n_val]]
    real_train = [v2[i] for i in idx[n_val:]]
    print(f"  seed-{args.seed} split: {len(real_train)} real train / {len(val)} real val",
          flush=True)

    synth: list[dict] = []
    if Path(args.synth_pairs).exists():
        synth = _dedup(_load(args.synth_pairs))
        # Defensive: drop any synthetic whose doc_id collides with a real doc_id
        # (synth ids are "synth-...", so this should be a no-op).
        real_ids = {r.get("doc_id") for r in v2}
        synth = [r for r in synth if r.get("doc_id") not in real_ids]
        print(f"synthetic unique: {len(synth)} (label dist {_dist(synth)})",
              flush=True)
    else:
        print(f"  WARNING: no synthetic at {args.synth_pairs} -- v3 train = real train only",
              file=sys.stderr)

    train = real_train + synth
    print(f"v3 train: {len(train)} = {len(real_train)} real + {len(synth)} synth "
          f"(label dist {_dist(train)})", flush=True)
    print(f"v3 val:   {len(val)} real (zero synthetic; label dist {_dist(val)})",
          flush=True)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for rec in train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(args.out_val, "w", encoding="utf-8") as f:
        for rec in val:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {args.out_train} ({len(train)}) and {args.out_val} ({len(val)})",
          flush=True)
    print("next: python scripts/train_doc_kind_head.py --train <out-train> "
          "--val <out-val> --temporal-feature --unsafe-penalty 5.0 --epochs 80 "
          "--device auto", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())