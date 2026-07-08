"""Generate GNN training data from a populated WaveDB.

Usage (validate slice, radius 1 — small subgraphs, one Oracle call each):
    python scripts/generate_gnn_training_data.py \\
        --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \\
        --output data/training/gnn/ --num-subgraphs 10 --sharding off

Scale-up is a CLI-arg change (e.g. ``--num-subgraphs 300 --subgraph-radius 3
--sharding auto``). At radius 3 on the dense 5002-ep corpus a subgraph is
~10K nodes / 4 MB JSON — one Oracle call blows past ``oracle_max_tokens``, so
``--sharding auto`` (on when radius>=2 or node count > --shard-threshold)
routes salience/link/ontology through ``src/gnn/sharded_labeling.py``: the full
subgraph is kept for what the GNN trains on, the *labeling* is sharded across
many calls, and ``recombine_*`` merges shards back into the existing per-
subgraph JSONL schemas (so ``train_gnn.py`` + ``validators.py`` are unchanged).
See ``docs/Phase 3a Task 3 - sharded labeling design.md``.

Two heads are Oracle-FREE here (spec §2):
- **anomaly** — the corpus is anomaly-free, so labels come from
  ``anomaly_injector`` + ``anomaly_rules`` (inject -> detect -> label), zero
  Oracle calls. The per-subgraph random-subset policy (which types to inject)
  is the generator's job; ``build_anomaly_labels`` records ``(seed, types)`` so
  the trainer reproduces the corrupted graph deterministically.
- **cluster** (DiffPool) — self-supervised by default (empty labels, 0 Oracle
  calls); ``--oracle-cluster-supervision`` adds one episode-only weak call.

A new Bonsai dataset — **anomaly_decision_pairs** (spec §2.5) — is generated
from the injected anomalies: for each flagged candidate the Oracle demonstrates
a fix/ask_user/dismiss decision + Hippo action + reasoning, from the SAME
retrieve-then-prompt context Bonsai will have at deploy (the radius-1
neighborhood of the flagged node within the extracted+corrupted subgraph —
works for injected/synthetic nodes that aren't in the store). Oracle is the
teacher; Bonsai is the student; the Oracle is NOT in the deploy loop. Written
to ``--bonsai-output`` (default ``data/training/bonsai/``), validated by
``validate_bonsai``.

Adapted from ``docs/Phase 1d.md`` §5 to the REAL WaveDB API: subgraphs come
from ``OracleLabelingPipeline.extract_subgraph`` (BFS over the memory graph),
episode centers from ``sample_episode_centers`` (content/ep scan), and the
current ontology is the shipped ``SEED_ONTOLOGY``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gnn.anomaly_injector import inject_anomalies  # noqa: E402
from src.gnn.anomaly_rules import (  # noqa: E402
    ANOMALY_TYPES,
    _is_node_id,
    enrich_subgraph as enrich_anomaly_subgraph,
)
from src.gnn.sharded_labeling import (  # noqa: E402
    DEFAULT_MAX_CANDIDATE_PAIRS,
    DEFAULT_SHARD_SIZE,
    build_anomaly_labels,
    build_cluster_episode_prompt,
    build_link_pred_shard_prompt,
    build_link_pred_shards,
    build_ontology_shard_prompt,
    build_ontology_shards,
    build_salience_shard_prompt,
    episode_only_context,
    recombine_link_pred,
    recombine_ontology,
    recombine_salience,
    shard_nodes,
    to_shard_record,
)
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
from src.training.oracle_labeling import (  # noqa: E402
    OracleLabelingPipeline,
    sample_episode_centers,
)
from src.training.prompts import (  # noqa: E402
    bonsai_anomaly_decision_prompt,
    gnn_link_prediction_prompt,
    gnn_ontology_prompt,
    gnn_salience_prompt,
)

# Oracle-labelable GNN tasks (cluster + anomaly are handled separately — cluster
# is self-supervised by default, anomaly is Oracle-FREE injection). Each entry:
# (task_name, one_call_prompt_fn, output_file, needs_ontology).
ONECALL_TASKS = [
    ("salience", gnn_salience_prompt, "salience_labels.jsonl", False),
    ("link_prediction", gnn_link_prediction_prompt, "link_prediction_labels.jsonl", False),
    ("ontology", gnn_ontology_prompt, "ontology_labels.jsonl", True),
]


def _hydrate_episodes(traversal: GraphTraversal, sub: dict) -> dict:
    """Enrich an extracted subgraph's episode nodes with hydrated content
    (summary/entities/topics/tones/decisions/timestamp) so the salience/link
    shard prompts can judge importance from summaries, not just node ids.

    Mutates ``sub`` in place (the structural edges are untouched — only node
    fields are added). The anomaly path uses a separate enrichment
    (``anomaly_rules.enrich_subgraph``) that also surfaces data/extra edges.
    """
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


def _local_neighborhood(subgraph: dict, node: str) -> dict:
    """Radius-1 neighborhood of ``node`` within ``subgraph`` — the retrieve-then-
    prompt context Bonsai will have at deploy (spec §2.5).

    At deploy the consolidator extracts a subgraph from the store around a query
    center, detects anomalies within it, and routes each flagged node to Bonsai
    with its local neighborhood. Here we reproduce that: the flagged node + its
    direct graph neighbors + the edges among them, taken from the
    extracted+corrupted subgraph in memory. This works for injected/synthetic
    nodes (e.g. ``ep_000001_dup``, ``ep_iso_...``, ``M:0003``) that exist in the
    corrupted subgraph but NOT in the store — a fresh store BFS would return
    nothing for them. Data edges (literal objects) are excluded.
    """
    node_ids = {node}
    for e in subgraph["edges"]:
        s, o = e["subject"], e["object"]
        if s == node and _is_node_id(o):
            node_ids.add(o)
        elif o == node and _is_node_id(s):
            node_ids.add(s)
    nodes = [n for n in subgraph["nodes"] if n["id"] in node_ids]
    edges = [
        e for e in subgraph["edges"]
        if e["subject"] in node_ids and _is_node_id(e["object"]) and e["object"] in node_ids
    ]
    return {"center": node, "nodes": nodes, "edges": edges}


def _run_onecall(oracle, task_name, prompt_fn, output_file, subgraphs, output_dir,
                 args, ontology_json: str, needs_ontology: bool) -> tuple[list, dict]:
    """The radius-1 small-subgraph path: one Oracle call per subgraph with the
    full-subgraph prompt (the pre-sharding behavior)."""
    def build_prompt(sg, _idx):
        sg_json = json.dumps(sg, ensure_ascii=False)
        return (prompt_fn(sg_json, ontology_json) if needs_ontology
                else prompt_fn(sg_json))

    def to_record(sg, result, _idx):
        return {"subgraph_id": sg["center"], "labels": result.response, "cost": result.cost}

    records, stats = run_batches(
        oracle, subgraphs, build_prompt, to_record,
        output_dir, task_name, args.oracle_batch_size, args.resume,
        progress_label="subgraphs", max_workers=args.oracle_max_workers,
    )
    write_jsonl(output_dir / output_file, records)
    return records, stats


def _run_sharded(oracle, task_name, output_file, subgraphs, output_dir, args,
                 build_shards, build_shard_prompt, ontology_json) -> tuple[list, dict]:
    """The radius>=2 path: build shards per subgraph, run Oracle over all shards,
    recombine shard records back into one per-subgraph record in the existing
    schema (so ``train_gnn.py`` + ``validators.py`` are unchanged — spec §4)."""
    all_shards: list[dict] = []
    for sg in subgraphs:
        all_shards.extend(build_shards(sg))
    total_shards = len(all_shards)

    def build_prompt(shard, _idx):
        return (build_shard_prompt(shard, ontology_json)
                if task_name == "ontology" else build_shard_prompt(shard))

    shard_records, stats = run_batches(
        oracle, all_shards, build_prompt, to_shard_record,
        output_dir, task_name, args.oracle_batch_size, args.resume,
        progress_label="shards", max_workers=args.oracle_max_workers,
    )
    subgraphs_by_id = {sg["center"]: sg for sg in subgraphs}
    # salience's recombine needs the hydrated subgraphs (to compute edge scores
    # from the structural edge set); link/ontology recombine from shard records
    # alone. Branch by task — the three recombine_* signatures differ.
    if task_name == "salience":
        records = recombine_salience(shard_records, subgraphs_by_id)
    elif task_name == "link_prediction":
        records = recombine_link_pred(shard_records)
    else:
        records = recombine_ontology(shard_records)
    write_jsonl(output_dir / output_file, records)
    stats["total_shards"] = total_shards
    return records, stats


def _run_cluster(oracle, subgraphs, output_dir, args) -> tuple[list, dict]:
    """DiffPool is self-supervised by default (empty ``clusters`` labels, 0
    Oracle calls — the head trains from structure via ``DiffPoolHead.loss``).
    ``--oracle-cluster-supervision`` adds one episode-only weak Oracle call per
    subgraph (spec §2) using the purpose-built ``build_cluster_episode_prompt``
    over ``episode_only_context`` (small — fits in one call, no sharding)."""
    if not args.oracle_cluster_supervision:
        records = [
            {"subgraph_id": sg["center"], "labels": {"clusters": []}, "cost": 0.0}
            for sg in subgraphs
        ]
        write_jsonl(output_dir / "cluster_labels.jsonl", records)
        return records, {"labeled": 0, "total_cost": 0.0, "self_supervised": True}

    ctxs = [episode_only_context(sg) for sg in subgraphs]

    def build_prompt(ctx, _idx):
        return build_cluster_episode_prompt(ctx)

    def to_record(ctx, result, _idx):
        return {"subgraph_id": ctx["center"], "labels": result.response, "cost": result.cost}

    records, stats = run_batches(
        oracle, ctxs, build_prompt, to_record,
        output_dir, "cluster", args.oracle_batch_size, args.resume,
        progress_label="subgraphs", max_workers=args.oracle_max_workers,
    )
    write_jsonl(output_dir / "cluster_labels.jsonl", records)
    return records, stats


def _run_anomaly(subgraphs, store, output_dir, args) -> tuple[list[dict], list[dict], dict]:
    """Oracle-FREE anomaly labels (spec §2): per subgraph, enrich with the
    data/extra edges the detectors need, inject a SEEDED RANDOM SUBSET of the 9
    types, and label via ``build_anomaly_labels``. Zero Oracle calls.

    Also returns ``decision_items`` — the flagged candidates (9 head types +
    identity_drift review-flags) with their retrieve-then-prompt context, built
    from the corrupted subgraph in memory (one at a time, so memory is bounded).
    The per-subgraph random-subset policy is the generator's job (spec §2.5):
    ``k = randint(0, 9)`` so ~1/10 of subgraphs stay clean (true negatives) and
    the rest get a varied corruption cocktail (each type's signature learned in
    isolation and in combination, not only the all-9 pattern).

    Returns ``(anomaly_records, decision_items, injection_stats)`` where
    ``injection_stats`` audits the cocktail (clean subgraph count + per-type
    planted counts) into ``quality_report.json``.
    """
    rng = random.Random(args.anomaly_seed)
    anomaly_records: list[dict] = []
    decision_items: list[dict] = []
    types_seen: dict[str, int] = {}
    clean_count = 0
    for idx, sg in enumerate(subgraphs):
        enriched = enrich_anomaly_subgraph(store, copy.deepcopy(sg))
        k = rng.randint(0, len(ANOMALY_TYPES))
        chosen = rng.sample(ANOMALY_TYPES, k)
        seed = args.anomaly_seed + idx
        rec = build_anomaly_labels(enriched, seed=seed, types=chosen)
        anomaly_records.append(rec)
        if not chosen:
            clean_count += 1
        for t in chosen:
            # Counts REQUESTED types, not what the injector actually planted —
            # an injector silently skips a type whose precondition isn't met
            # (e.g. stale_abstraction on a subgraph with no M: node). The actual
            # planted set is in rec["labels"]["planted"]; this audits the cocktail
            # we ASKED for across the run.
            types_seen[t] = types_seen.get(t, 0) + 1

        # Build anomaly_decision_pairs items from the SAME injected anomalies
        # the head trained on (data-consistency win — spec §2.5 "we planted the
        # drift"). The context is the radius-1 neighborhood of the flagged node
        # within the corrupted subgraph (retrieve-then-prompt, deploy-faithful).
        if args.skip_anomaly_decision_pairs:
            continue
        # Inject ONCE per subgraph to reproduce the corrupted graph the head's
        # labels were computed on (``build_anomaly_labels`` injects its own
        # internal copy for the LABEL record — the single source of truth for
        # the labels; this second inject only yields the corrupted graph for
        # the decision-pair CONTEXT, never labels). One inject per subgraph,
        # reused across all its flagged candidates — not per-finding.
        corrupted, _ = inject_anomalies(enriched, seed=seed, types=chosen)
        # Spec §2.5: identity_drift is the PRIMARY Bonsai decider target (a
        # review-flag, not a head label). A naive ``anomalies + identity_drift``
        # capped at N would let the 9 injected-type findings crowd identity_drift
        # out entirely on a dense subgraph — silently losing the main signal. So
        # split the per-subgraph budget: identity_drift gets up to half (it
        # deliberately over-fires, so this is plenty), the 9-type findings get
        # the rest. Both stay represented; the cap still bounds Oracle cost.
        cap = args.max_decision_pairs_per_subgraph
        id_drift = list(rec["labels"]["identity_drift"])
        anoms = list(rec["labels"]["anomalies"])
        if cap <= 0:
            selected = []  # cap 0 = no pairs this subgraph (--skip-* skips all)
        else:
            id_budget = min(len(id_drift), max(1, cap // 2)) if id_drift else 0
            an_budget = max(0, cap - id_budget)
            selected = id_drift[:id_budget] + anoms[:an_budget]
        for f in selected:
            flagged_node = f["node"]
            anomaly_type = f["type"]
            if not flagged_node or not _is_node_id(flagged_node):
                continue  # flagged subject must be a graph node id
            decision_items.append({
                "subgraph_id": rec["subgraph_id"],
                "flagged_entity": flagged_node,
                "anomaly_type": anomaly_type,
                "retrieved_context": _local_neighborhood(corrupted, flagged_node),
            })

    write_jsonl(output_dir / "anomaly_labels.jsonl", anomaly_records)
    injection_stats = {
        "subgraphs": len(subgraphs),
        "clean_true_negatives": clean_count,
        "types_requested": types_seen,  # type -> # subgraphs it was requested in
    }
    return anomaly_records, decision_items, injection_stats


def _run_anomaly_decision(oracle, items, bonsai_output_dir, args) -> tuple[list, dict]:
    """Generate the Bonsai anomaly-DECISION pairs (spec §2.5): retrieve-then-
    prompt the Oracle for a fix/ask_user/dismiss decision + Hippo action +
    reasoning on each flagged candidate. Cached by prompt hash ($0 on local
    Ollama); written to ``--bonsai-output``."""
    if not items:
        write_jsonl(bonsai_output_dir / "anomaly_decision_pairs.jsonl", [])
        return [], {"labeled": 0, "total_cost": 0.0, "skipped": True}

    def build_prompt(item, _idx):
        return bonsai_anomaly_decision_prompt(
            item["flagged_entity"], item["retrieved_context"], item["anomaly_type"]
        )

    def to_record(item, result, _idx):
        resp = result.response or {}
        return {
            "flagged_entity": item["flagged_entity"],
            "retrieved_context": item["retrieved_context"],
            "anomaly_type": item["anomaly_type"],
            "decision": resp.get("decision"),
            "action": resp.get("action"),
            "reasoning": resp.get("reasoning"),
        }

    records, stats = run_batches(
        oracle, items, build_prompt, to_record,
        bonsai_output_dir, "anomaly_decision", args.oracle_batch_size, args.resume,
        progress_label="flagged candidates", max_workers=args.oracle_max_workers,
    )
    write_jsonl(bonsai_output_dir / "anomaly_decision_pairs.jsonl", records)
    return records, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GNN training data")
    parser.add_argument("--db", default="data/pod_runs/phase1b_scale/ingest_db_dialogsum",
                        help="WaveDB store path (ingested corpus)")
    parser.add_argument("--output", default="data/training/gnn/", help="GNN output directory")
    parser.add_argument("--bonsai-output", default="data/training/bonsai/",
                        help="Bonsai output directory (anomaly_decision_pairs.jsonl)")
    parser.add_argument("--num-subgraphs", type=int, default=10,
                        help="Number of subgraphs to label (validate-slice default 10)")
    parser.add_argument("--subgraph-radius", type=int, default=3,
                        help="BFS radius (Phase 3a default 3, matching the §5 regen-at-scale "
                             "target. radius>=2 on a dense corpus fans out to ~5000+ nodes per "
                             "subgraph -> --sharding auto routes through sharded_labeling.")
    parser.add_argument("--resume", action="store_true", help="Resume from per-task checkpoints")
    parser.add_argument(
        "--sharding", choices=("auto", "on", "off"), default="auto",
        help="auto = sharded when radius>=2 OR max subgraph node count > --shard-threshold "
             "(spec §5); on = always shard salience/link/ontology; off = old one-call path "
             "(radius-1 dev slice). Anomaly is injection-based and cluster self-supervised in "
             "ALL modes.")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help="Salience node-shard / link+ontology candidate-pair shard size")
    parser.add_argument("--max-candidate-pairs", type=int, default=DEFAULT_MAX_CANDIDATE_PAIRS,
                        help="Cap for link/ontology candidate pairs per subgraph")
    parser.add_argument("--shard-threshold", type=int, default=200,
                        help="auto-sharding trigger: max subgraph node count above this shards")
    parser.add_argument("--oracle-cluster-supervision", action="store_true",
                        help="Add one episode-only Oracle call per subgraph for weak DiffPool "
                             "supervision (off by default — cluster is self-supervised).")
    parser.add_argument("--anomaly-seed", type=int, default=0,
                        help="Base seed for the per-subgraph anomaly injection")
    parser.add_argument("--max-decision-pairs-per-subgraph", type=int, default=10,
                        help="Cap on anomaly_decision_pairs flagged candidates per subgraph "
                             "(bounds Oracle cost on dense subgraphs)")
    parser.add_argument("--skip-anomaly-decision-pairs", action="store_true",
                        help="Skip the anomaly_decision_pairs Bonsai task (the anomaly head "
                             "labels are still written).")
    add_oracle_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output)
    bonsai_output_dir = Path(args.bonsai_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    bonsai_output_dir.mkdir(parents=True, exist_ok=True)

    store = HippocampalStore(args.db)
    try:
        pipe = OracleLabelingPipeline(store)
        traversal = GraphTraversal(store)
        oracle = make_oracle(args, output_dir)

        # ── 1. Extract subgraphs (deterministic; not Oracle-dependent) ──
        print("Extracting subgraphs from memory graph...")
        centers = sample_episode_centers(store, n=args.num_subgraphs)
        subgraphs: list[dict] = []
        for c in centers:
            sg = pipe.extract_subgraph(c, radius=args.subgraph_radius)
            if len(sg["nodes"]) >= 3:  # minimum viable subgraph
                subgraphs.append(sg)
        # Hydrate episode content for the salience/link shard prompts (mutates
        # in place; the anomaly path enriches a separate deepcopy).
        for sg in subgraphs:
            _hydrate_episodes(traversal, sg)
        max_nodes = max((len(sg["nodes"]) for sg in subgraphs), default=0)
        print(f"Extracted {len(subgraphs)} subgraphs (radius={args.subgraph_radius}, "
              f"max nodes={max_nodes})")

        sharded = args.sharding == "on" or (
            args.sharding == "auto"
            and (args.subgraph_radius >= 2 or max_nodes > args.shard_threshold)
        )
        print(f"Sharding: {args.sharding} -> {'sharded' if sharded else 'one-call'} "
              f"(salience/link/ontology); anomaly=injection; cluster=self-supervised")

        ontology_json = json.dumps(SEED_ONTOLOGY, ensure_ascii=False)
        all_stats: dict = {}
        start_time = time.time()

        # ── 2. Oracle-labelable heads: salience / link / ontology ──
        for task_name, prompt_fn, output_file, needs_ontology in ONECALL_TASKS:
            print(f"\n{'=' * 60}\nGenerating {task_name} labels...\n{'=' * 60}")
            if sharded:
                if task_name == "salience":
                    build_shards = lambda sg, _ss=args.shard_size: shard_nodes(sg, shard_size=_ss)
                    records, stats = _run_sharded(
                        oracle, task_name, output_file, subgraphs, output_dir, args,
                        build_shards, build_salience_shard_prompt, ontology_json)
                elif task_name == "link_prediction":
                    build_shards = lambda sg, _ss=args.shard_size, _mp=args.max_candidate_pairs: \
                        build_link_pred_shards(sg, shard_size=_ss, max_candidate_pairs=_mp)
                    records, stats = _run_sharded(
                        oracle, task_name, output_file, subgraphs, output_dir, args,
                        build_shards, build_link_pred_shard_prompt, ontology_json)
                else:  # ontology
                    build_shards = lambda sg, _ss=args.shard_size, _mp=args.max_candidate_pairs: \
                        build_ontology_shards(sg, shard_size=_ss, max_candidate_pairs=_mp)
                    records, stats = _run_sharded(
                        oracle, task_name, output_file, subgraphs, output_dir, args,
                        build_shards, build_ontology_shard_prompt, ontology_json)
            else:
                records, stats = _run_onecall(
                    oracle, task_name, prompt_fn, output_file, subgraphs, output_dir,
                    args, ontology_json, needs_ontology)
            all_stats[task_name] = stats
            print(f"  {task_name}: {stats.get('labeled', 0)} labels, "
                  f"${stats.get('total_cost', 0.0):.4f}")

        # ── 3. Cluster (DiffPool) — self-supervised or weakly supervised ──
        print(f"\n{'=' * 60}\nGenerating cluster labels...\n{'=' * 60}")
        _, cluster_stats = _run_cluster(oracle, subgraphs, output_dir, args)
        all_stats["cluster"] = cluster_stats
        print(f"  cluster: {'self-supervised (0 Oracle calls)' if not args.oracle_cluster_supervision else str(cluster_stats.get('labeled', 0)) + ' labels'}")

        # ── 4. Anomaly — Oracle-FREE injection labeling + decision items ──
        print(f"\n{'=' * 60}\nGenerating anomaly labels (injection, 0 Oracle)...\n{'=' * 60}")
        anomaly_records, decision_items, injection_stats = _run_anomaly(
            subgraphs, store, output_dir, args)
        all_stats["anomaly"] = {
            "labeled": len(anomaly_records), "total_cost": 0.0,
            "oracle_calls": 0, "decision_items": len(decision_items),
            "injection": injection_stats,
        }
        print(f"  anomaly: {len(anomaly_records)} records (0 Oracle calls), "
              f"{len(decision_items)} decision candidates")

        # ── 5. anomaly_decision_pairs — Bonsai distillation data (spec §2.5) ──
        print(f"\n{'=' * 60}\nGenerating anomaly_decision_pairs (Bonsai)...\n{'=' * 60}")
        _, decision_stats = _run_anomaly_decision(oracle, decision_items, bonsai_output_dir, args)
        all_stats["anomaly_decision"] = decision_stats
        print(f"  anomaly_decision: {decision_stats.get('labeled', 0)} pairs, "
              f"${decision_stats.get('total_cost', 0.0):.4f}")

        report = {
            "total_subgraphs": len(subgraphs),
            "radius": args.subgraph_radius,
            "sharding": {"mode": args.sharding, "active": sharded,
                         "shard_size": args.shard_size,
                         "max_candidate_pairs": args.max_candidate_pairs,
                         "shard_threshold": args.shard_threshold, "max_nodes": max_nodes},
            "tasks": all_stats,
            "oracle_stats": oracle.get_stats(),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        write_report(output_dir / "quality_report.json", report)

        print(f"\n{'=' * 60}\nGNN training data generation complete.")
        print(f"Total Oracle calls: {oracle.total_calls}  tokens: {oracle.total_tokens}")
        print(f"GNN output: {output_dir}")
        print(f"Bonsai output: {bonsai_output_dir}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())