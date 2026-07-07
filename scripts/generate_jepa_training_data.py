"""Generate JEPA routing training data.

Usage (validate slice):
    python scripts/generate_jepa_training_data.py \\
        --output data/training/jepa/ --num-pairs 20

Synthetic query templates — no WaveDB corpus read. Ported from
``docs/Phase 1d.md`` §7 with imports fixed to use the shared generator helpers.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.generator_common import (  # noqa: E402
    add_oracle_args,
    make_oracle,
    run_batches,
    write_jsonl,
    write_report,
)
from src.training.prompts import jepa_routing_prompt  # noqa: E402
from src.subconscious.routing import PATHWAYS as _PATHWAY_VOCAB  # noqa: E402

AVAILABLE_DOMAINS = """
- database: WaveDB, Postgres, HBTrie, SQL, configuration, performance
- coding: Python, Rust, Dart, tree-sitter, AST parsing, code review
- robotics: actuators, sensors, inverse kinematics, control policies
- economics: Spark Ledger, monetary theory, QE, zk-SNARKs
- ai_architecture: neural networks, cognitive systems, memory models
- personal: user preferences, relationships, emotional patterns
"""

AVAILABLE_PATHWAYS = """
- ssm_direct: Answer from working memory. No retrieval needed.
- graph_retrieve: Query the memory graph. Standard retrieval.
- process_exec: Execute a stored process.
- tool_plan: Plan a multi-step tool use strategy.
- conscious_deliberation: Engage System 2 for complex reasoning.
"""

# (template, expected_pathways) — expected pathways are not passed to the
# Oracle; they document the routing intent for later quality analysis.
_TEMPLATES = [
    ("What is {entity}?", ["ssm_direct"]),
    ("When did we discuss {topic}?", ["graph_retrieve"]),
    ("Who was involved in {topic}?", ["graph_retrieve"]),
    ("What did {entity} say about {topic}?", ["graph_retrieve"]),
    ("Why was I {tone} about {topic}?", ["graph_retrieve"]),
    ("What happened after {event}?", ["graph_retrieve"]),
    ("Review this code for security issues", ["process_exec"]),
    ("Deploy the latest changes", ["process_exec"]),
    ("Run the test suite and report failures", ["process_exec"]),
    ("Why did we choose {entity_a} over {entity_b}?", ["conscious_deliberation"]),
    ("What are the implications of {decision}?", ["conscious_deliberation"]),
    ("Design a new approach for {problem}", ["conscious_deliberation"]),
    ("Compare {domain_a} performance with {domain_b} reliability", ["graph_retrieve"]),
    ("How does {domain_a} architecture influence {domain_b} design?", ["conscious_deliberation"]),
    ("Plan the sequence of tool calls needed to {task}", ["tool_plan"]),
    ("Outline a multi-step tool strategy for {objective}", ["tool_plan"]),
]

# Vocabularies sized so the cross-product yields thousands of UNIQUE queries
# (the first 5k run had only ~177 unique → the gate memorized strings instead
# of generalizing). Each slot is in-domain (database/coding/robotics/
# economics/ai_architecture/personal) so the Oracle routes plausibly.
_ENTITIES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
    "WaveDB", "Postgres", "MongoDB", "Redis", "SQLite", "DuckDB",
    "Python", "Rust", "Go", "TypeScript", "C++", "Julia",
    "HBTrie", "Bonsai", "FAISS", "Triton", "TVM",
    "the WAL", "the MVCC layer", "the query planner", "the replication log",
    "the embedding model", "the SSM kernel", "the JEPA head", "the gate",
    "llama-server", "Ollama", "Docker", "k8s", "nginx",
    "the optimizer", "the compactor", "the checkpoint", "the index",
]
_TOPICS = [
    "database_design", "configuration", "performance", "security", "api_design",
    "concurrency", "replication", "indexing", "caching", "batching",
    "serialization", "compaction", "memory_pressure", "latency", "throughput",
    "crash_recovery", "schema_migration", "vector_search", "graph_traversal",
    "encoding", "retrieval", "routing", "context_window", "tokenization",
    "embedding_quality",
]
_TONES = ["frustrated", "excited", "curious", "skeptical", "confident",
          "confused", "optimistic", "annoyed", "impressed", "worried"]
_EVENTS = [
    "morphisms", "the optimizer", "the refactor", "the deployment",
    "the migration", "the outage", "the benchmark", "the rollout",
    "the rollback", "the release", "the compaction", "the schema change",
]
_DECISIONS = [
    "using WaveDB", "the DEBOUNCED choice", "the cost-based optimizer",
    "the sub-block packing fix", "the MVCC redesign", "dropping FAISS",
    "the SSM reference backend", "the chunk-size change", "the rebrand",
    "the 2a skeleton", "the oracle labeling", "the retrieval gate",
    "the local Bonsai endpoint", "the compact corpus",
]
_PROBLEMS = [
    "sync mode configuration", "async performance", "encryption API",
    "write amplification", "graph query segfault", "reopen data loss",
    "scan corruption at scale", "WAL truncation", "Bonsai over-extraction",
    "GLiNER CPU bottleneck", "MVCC read amplification", "sub-block packing",
]
_TASKS = [
    "deploy the service", "migrate the database", "audit the dependencies",
    "rotate the credentials", "rebuild the search index", "purge the queue",
    "scale the workers", "snapshot the volume", "retrain the gate",
    "compact the corpus", "rotate the logs", "restart the endpoint",
    "validate the checkpoint", "provision a pod", "backfill the embeddings",
    "warm the cache", "drain the node", "canary the release",
    "roll back the migration", "re-provision the GPU",
]
_OBJECTIVES = [
    "backing up the corpus", "scaling the cluster", "validating the release",
    "provisioning a new node", "diagnosing the segfault", "reducing write amp",
    "improving retrieval recall", "shrinking the DB footprint",
    "automating the rollout", "auditing the access logs", "rebalancing shards",
    "estimating model size", "planning the tool chain", "delegating to a bigger model",
    "recovering the lost keys",
]
# Cross-domain comparison templates ("Compare {domain_a} performance with
# {domain_b} reliability", "How does {domain_a} architecture influence
# {domain_b} design?") only need a handful of contrasting domains; the full
# 6-domain set lives in AVAILABLE_DOMAINS above and is labeled by the Oracle.
_DOMAINS = ["database", "robotics", "economics"]


def _generate_diverse_queries(num_queries: int, seed: int = 0) -> list[dict]:
    """Generate diverse query patterns for routing training.

    Returns a list of ``{"query", "expected_pathways"}`` dicts. ``seed`` makes
    the validate slice reproducible; scale runs pass a different seed.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(num_queries):
        template, expected = rng.choice(_TEMPLATES)
        query = template.format(
            entity=rng.choice(_ENTITIES),
            entity_a=rng.choice(_ENTITIES),
            entity_b=rng.choice(_ENTITIES),
            topic=rng.choice(_TOPICS),
            tone=rng.choice(_TONES),
            event=rng.choice(_EVENTS),
            decision=rng.choice(_DECISIONS),
            problem=rng.choice(_PROBLEMS),
            task=rng.choice(_TASKS),
            objective=rng.choice(_OBJECTIVES),
            domain_a=rng.choice(_DOMAINS),
            domain_b=rng.choice(_DOMAINS),
        )
        out.append({"query": query, "expected_pathways": expected})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JEPA routing data")
    parser.add_argument("--output", default="data/training/jepa/", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    add_oracle_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle = make_oracle(args, output_dir)
    start_time = time.time()
    items = _generate_diverse_queries(args.num_pairs, seed=args.seed)
    print(f"Generating {len(items)} JEPA routing pairs...")

    def build_prompt(item, _idx):
        return jepa_routing_prompt(item["query"], AVAILABLE_DOMAINS, AVAILABLE_PATHWAYS)

    def to_record(item, result, _idx):
        # The Oracle is a reliable labeler for domains/skills/model_size/
        # deliberation, but its PATHWAY label collapses to graph_retrieve
        # (it re-labels most tool_plan/process_exec/ssm_direct intents as
        # "needs retrieval") — a gate trained on those labels routes
        # everything to graph_retrieve (verified). The template's expected
        # pathway IS the routing intent by construction ("Plan the sequence
        # of tool calls…" = tool_plan), so it is cleaner pathway ground
        # truth. Use it as the training target; keep the Oracle's choice
        # as oracle_pathway for audit. (Other fields stay Oracle-labeled.)
        route = dict(result.response) if isinstance(result.response, dict) else {}
        exp = item["expected_pathways"] or []
        if exp and exp[0] in _PATHWAY_VOCAB:
            route["oracle_pathway"] = route.get("pathway")
            route["pathway"] = exp[0]
        return {"query": item["query"], "route": route,
                "expected_pathways": item["expected_pathways"], "cost": result.cost}

    records, stats = run_batches(
        oracle, items, build_prompt, to_record,
        output_dir, "routing", args.oracle_batch_size, args.resume,
        progress_label="queries", max_workers=args.oracle_max_workers,
    )
    write_jsonl(output_dir / "routing_pairs.jsonl", records)

    # Pathway distribution for the quality report.
    from collections import Counter
    pathways = Counter()
    for r in records:
        route = r.get("route", {}) if isinstance(r.get("route"), dict) else {}
        pathways[route.get("pathway", "unknown")] += 1

    report = {
        "routing_pairs": len(records),
        "pathway_distribution": dict(pathways),
        "stats": stats,
        "oracle_stats": oracle.get_stats(),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    write_report(output_dir / "quality_report.json", report)
    print(f"\nGenerated {len(records)} routing pairs. Pathways: {dict(pathways)}")
    print(f"Total Oracle calls: {oracle.total_calls}  tokens: {oracle.total_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())