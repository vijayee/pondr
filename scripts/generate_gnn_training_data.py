"""Generate GNN training data from a populated WaveDB.

Usage (validate slice):
    python scripts/generate_gnn_training_data.py \\
        --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \\
        --output data/training/gnn/ --num-subgraphs 10

Scale-up is a CLI-arg change (e.g. ``--num-subgraphs 4000 --resume``).

Adapted from ``docs/Phase 1d.md`` §5 to the REAL WaveDB API: subgraphs come
from ``OracleLabelingPipeline.extract_subgraph`` (BFS over the memory graph,
correct API), episode centers from ``sample_episode_centers`` (content/ep
scan), and the current ontology is the shipped ``SEED_ONTOLOGY`` (there is no
``subClassOf`` in the conversational schema, so the doc's
``get_current_ontology`` graph query is replaced). Episode nodes in the
salience task are content-enriched via ``GraphTraversal.hydrate_episode`` so
the Oracle can judge importance from summaries, not just node ids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.ontology import SEED_ONTOLOGY  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402
from src.retrieval.graph_traversal import GraphTraversal  # noqa: E402
from src.training.generator_common import (  # noqa: E402
    add_oracle_args,
    make_oracle,
    run_batches,
    write_jsonl,
    write_report,
)
from src.training.oracle_labeling import OracleLabelingPipeline, sample_episode_centers  # noqa: E402
from src.training.prompts import (  # noqa: E402
    gnn_anomaly_prompt,
    gnn_cluster_prompt,
    gnn_link_prediction_prompt,
    gnn_ontology_prompt,
    gnn_salience_prompt,
)

# task -> (prompt_fn, output_file). The ontology task also needs the ontology.
TASKS = [
    ("salience", gnn_salience_prompt, "salience_labels.jsonl"),
    ("clusters", gnn_cluster_prompt, "cluster_labels.jsonl"),
    ("link_prediction", gnn_link_prediction_prompt, "link_prediction_labels.jsonl"),
    ("anomalies", gnn_anomaly_prompt, "anomaly_labels.jsonl"),
    ("ontology", gnn_ontology_prompt, "ontology_labels.jsonl"),
]


def _enrich_subgraph(pipe: OracleLabelingPipeline, traversal: GraphTraversal,
                     center: str, radius: int) -> dict:
    """Extract a subgraph and enrich its episode nodes with hydrated content.

    The structural tasks (cluster/link/anomaly) only need ids+types+edges, but
    feeding content makes every task's labels better and is cheap (one
    ``hydrate_episode`` per episode node, bounded by the subgraph radius).
    """
    sub = pipe.extract_subgraph(center, radius=radius)
    for node in sub["nodes"]:
        if node.get("type") == "episode":
            hy = traversal.hydrate_episode(node["id"])
            node["summary"] = hy.get("summary", "")
            node["entities"] = hy.get("entities", [])
            node["topics"] = hy.get("topics", [])
            node["tones"] = hy.get("tones", [])
            node["decisions"] = hy.get("decisions", [])
            node["timestamp"] = hy.get("timestamp", "")
    return sub


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GNN training data")
    parser.add_argument("--db", default="data/pod_runs/phase1b_scale/ingest_db_dialogsum",
                        help="WaveDB store path (ingested corpus)")
    parser.add_argument("--output", default="data/training/gnn/", help="Output directory")
    parser.add_argument("--num-subgraphs", type=int, default=10,
                        help="Number of subgraphs to label (validate-slice default 10)")
    parser.add_argument("--subgraph-radius", type=int, default=1,
                        help="BFS radius (default 1: on a dense real corpus radius>=2 "
                             "fans out to ~5000 nodes / 90s+ per subgraph — unusable. "
                             "Raise only on a small/sparse graph.)")
    parser.add_argument("--resume", action="store_true", help="Resume from per-task checkpoints")
    add_oracle_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = HippocampalStore(args.db)
    try:
        pipe = OracleLabelingPipeline(store)
        traversal = GraphTraversal(store)
        oracle = make_oracle(args, output_dir)

        # ── 1. Extract subgraphs (deterministic; not Oracle-dependent) ──
        print("Extracting subgraphs from memory graph...")
        centers = sample_episode_centers(store, n=args.num_subgraphs)
        subgraphs = []
        for c in centers:
            sg = _enrich_subgraph(pipe, traversal, c, args.subgraph_radius)
            if len(sg["nodes"]) >= 3:  # minimum viable subgraph
                subgraphs.append(sg)
        print(f"Extracted {len(subgraphs)} subgraphs (radius={args.subgraph_radius})")

        ontology_json = json.dumps(SEED_ONTOLOGY, ensure_ascii=False)
        all_stats: dict = {}
        start_time = time.time()

        for task_name, prompt_fn, output_file in TASKS:
            print(f"\n{'=' * 60}\nGenerating {task_name} labels...\n{'=' * 60}")

            def build_prompt(sg, _idx, _pf=prompt_fn, _tn=task_name):
                if _tn == "ontology":
                    return _pf(json.dumps(sg, ensure_ascii=False), ontology_json)
                return _pf(json.dumps(sg, ensure_ascii=False))

            def to_record(sg, result, _idx):
                return {"subgraph_id": sg["center"], "labels": result.response,
                        "cost": result.cost}

            records, stats = run_batches(
                oracle, subgraphs, build_prompt, to_record,
                output_dir, task_name, args.oracle_batch_size, args.resume,
                progress_label="subgraphs",
            )
            write_jsonl(output_dir / output_file, records)
            all_stats[task_name] = stats
            print(f"  {task_name}: {stats.get('labeled', 0)} labels, "
                  f"${stats.get('total_cost', 0.0):.4f}")

        report = {
            "total_subgraphs": len(subgraphs),
            "radius": args.subgraph_radius,
            "tasks": all_stats,
            "oracle_stats": oracle.get_stats(),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        write_report(output_dir / "quality_report.json", report)

        print(f"\n{'=' * 60}\nGNN training data generation complete.")
        print(f"Total Oracle calls: {oracle.total_calls}  tokens: {oracle.total_tokens}")
        print(f"Output: {output_dir}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())