"""Generate Bonsai training data (query planning + relation extraction pairs).

Usage (validate slice):
    python scripts/generate_bonsai_training_data.py \\
        --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \\
        --output data/training/bonsai/ --num-query-pairs 10 --num-relation-pairs 10

Adapted from ``docs/Phase 1d.md`` §6 to the REAL WaveDB API: episode content +
graph fields come from ``GraphTraversal.hydrate_episode`` (NOT ``ep.entities``
on a content-only ``get_episode`` result), and episode ids come from
``sample_episode_centers`` (content/ep scan). ``_generate_hypothetical_questions``
is reused verbatim (pure logic).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import HippocampalStore  # noqa: E402
from src.retrieval.graph_traversal import GraphTraversal  # noqa: E402
from src.training.generator_common import (  # noqa: E402
    add_oracle_args,
    make_oracle,
    run_batches,
    write_jsonl,
    write_report,
)
from src.training.oracle_labeling import sample_episode_centers  # noqa: E402
from src.training.prompts import (  # noqa: E402
    bonsai_query_planning_prompt,
    bonsai_relation_extraction_prompt,
)


def _all_episodes(store, traversal, count, rng):
    """Return ``count`` random text-bearing hydrated episodes.

    Shuffles the (cheap) episode-id list first, then hydrates lazily until
    ``count`` text-bearing episodes are collected — so we pay ~``count``
    ``hydrate_episode`` calls, not ``max(count, 50)``. Each episode yields up
    to 3 query-planning questions, so callers requesting ``N`` pairs should
    ask for ~``N`` episodes here and slice the flattened pairs.
    """
    ids = sample_episode_centers(store, n=None)
    rng.shuffle(ids)
    out = []
    for eid in ids:
        hy = traversal.hydrate_episode(eid)
        if hy.get("text"):
            out.append(hy)
        if len(out) >= count:
            break
    return out


def _generate_hypothetical_questions(episode: dict) -> list[str]:
    """Generate hypothetical questions a user might ask about this episode.

    Pure logic — no Oracle, no graph reads. Reused verbatim from the doc.
    """
    questions: list[str] = []
    for entity in episode.get("entities", [])[:2]:
        questions.append(f"What did {entity} say?")
        questions.append(f"What was {entity}'s opinion?")
    for topic in episode.get("topics", [])[:1]:
        questions.append(f"What did we discuss about {topic}?")
    for tone in episode.get("tones", [])[:1]:
        questions.append(f"What was I {tone} about?")
    if episode.get("decisions"):
        questions.append("What did we decide?")
        questions.append("Why did we make that decision?")
    questions.append("What happened after this conversation?")
    return questions[:3]


def _build_query_items(episodes):
    """Flatten (episode, question) pairs for the query-planning task."""
    items = []
    for ep in episodes:
        for q in _generate_hypothetical_questions(ep):
            items.append({"episode": ep, "question": q})
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Bonsai training data")
    parser.add_argument("--db", default="data/pod_runs/phase1b_scale/ingest_db_dialogsum",
                        help="WaveDB store path (ingested corpus)")
    parser.add_argument("--output", default="data/training/bonsai/", help="Output directory")
    parser.add_argument("--num-query-pairs", type=int, default=10)
    parser.add_argument("--num-relation-pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument("--resume", action="store_true", help="Resume from per-task checkpoints")
    add_oracle_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    store = HippocampalStore(args.db)
    try:
        traversal = GraphTraversal(store)
        oracle = make_oracle(args, output_dir)
        start_time = time.time()

        # ── 1. Query planning pairs ──
        print("Generating query planning pairs...")
        # Each episode yields up to 3 questions; fetch num_query_pairs episodes
        # (generous — some yield fewer) and slice the flattened pairs.
        q_eps = _all_episodes(store, traversal, args.num_query_pairs, rng)
        q_items = _build_query_items(q_eps)[: args.num_query_pairs]

        def qp_prompt(item, _idx):
            return bonsai_query_planning_prompt(item["episode"]["text"], item["question"])

        def qp_record(item, result, _idx):
            return {"conversation_id": item["episode"]["episode_id"],
                    "conversation_text": item["episode"]["text"],
                    "training_pair": result.response}

        qp_records, qp_stats = run_batches(
            oracle, q_items, qp_prompt, qp_record,
            output_dir, "query_planning", args.oracle_batch_size, args.resume,
            progress_label="pairs",
        )
        write_jsonl(output_dir / "query_planning_pairs.jsonl", qp_records)
        print(f"  Generated {len(qp_records)} query planning pairs")

        # ── 2. Relation extraction pairs ──
        print("\nGenerating relation extraction pairs...")
        r_eps = _all_episodes(store, traversal, args.num_relation_pairs, rng)
        r_items = [{"episode": ep} for ep in r_eps]

        def re_prompt(item, _idx):
            return bonsai_relation_extraction_prompt(item["episode"]["text"])

        def re_record(item, result, _idx):
            rels = result.response.get("relations", []) if isinstance(result.response, dict) else []
            return {"conversation_id": item["episode"]["episode_id"],
                    "conversation_text": item["episode"]["text"],
                    "relations": rels}

        re_records, re_stats = run_batches(
            oracle, r_items, re_prompt, re_record,
            output_dir, "relation_extraction", args.oracle_batch_size, args.resume,
            progress_label="episodes",
        )
        write_jsonl(output_dir / "relation_extraction_pairs.jsonl", re_records)
        print(f"  Generated {len(re_records)} relation extraction pairs")

        report = {
            "query_planning_pairs": len(qp_records),
            "relation_extraction_pairs": len(re_records),
            "query_planning_stats": qp_stats,
            "relation_extraction_stats": re_stats,
            "oracle_stats": oracle.get_stats(),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        write_report(output_dir / "quality_report.json", report)

        print(f"\nBonsai training data generation complete.")
        print(f"Total Oracle calls: {oracle.total_calls}  tokens: {oracle.total_tokens}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())