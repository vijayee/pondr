"""Extract subgraphs for oracle labeling (Phase 1d GNN training prep).

Thin runner over ``OracleLabelingPipeline.extract_subgraph``: samples episode
ids from a store as subgraph centers, extracts each subgraph (BFS over the
memory graph), and writes them to a JSONL file — one ``{center, radius, nodes,
edges}`` record per line. No live oracle/Bonsai calls (infra only in 1b); a
future phase consumes these subgraphs + labels them.

    python scripts/generate_training_data.py --db /workspace/volumes/hippo/memory_db \\
        --out data/training_subgraphs.jsonl --n 200 --radius 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import HippocampalStore  # noqa: E402
from src.training.oracle_labeling import OracleLabelingPipeline, sample_episode_centers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract subgraphs for oracle labeling.")
    parser.add_argument("--db", required=True, help="WaveDB store path (ingested corpus).")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--n", type=int, help="Max episode centers to sample (all if omitted).")
    parser.add_argument("--radius", type=int, default=3, help="BFS hop radius.")
    args = parser.parse_args()

    store = HippocampalStore(args.db)
    try:
        pipe = OracleLabelingPipeline(store)
        centers = sample_episode_centers(store, n=args.n)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for cid in centers:
                sub = pipe.extract_subgraph(cid, radius=args.radius)
                f.write(json.dumps(sub, ensure_ascii=False) + "\n")
        print(f"Wrote {len(centers)} subgraph(s) → {args.out}")
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())