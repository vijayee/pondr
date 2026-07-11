#!/usr/bin/env python
"""Nightly dream-state consolidation entrypoint (Phase 3a Task 6).

Runs one ``Consolidator`` pass over the memory graph. Dry-run by default
(prints a report, mutates nothing); ``--apply`` writes abstractions, accepted
edges, and pruned-edge archives to the store.

Bonsai verification of medium-confidence edge proposals is optional: pass
``--verify`` to enable the Oracle/Bonsai-backed verifier (requires the local
Bonsai / Ollama endpoint). Without ``--verify``, proposals in the "propose"
band are recorded as unverified and NOT accepted (honest, not faked).

Mirrors the pod-ready shape of ``scripts/train_backbone.py`` — but the
consolidation loop runs anywhere the store + a (possibly untrained) model
live, including CPU dev. A trained GNN checkpoint (Task 4) is loaded with
``--checkpoint``; without it the loop runs an untrained model and the report's
``trained`` flag is False.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

# Ensure ``src`` is importable when run as a bare script (pytest conftest
# normally handles this; the script may run outside the test harness).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402  (hard dep via src.gnn.*; imported here so _load_model's torch.load resolves at module scope)

from src.config import ConsolidationConfig, Phase3aConfig, config  # noqa: E402
from src.gnn.consolidate import Consolidator  # noqa: E402
from src.gnn.model import GNNModel  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402


def _load_model(checkpoint: str, device: str) -> GNNModel:
    model = GNNModel(
        hidden_dim=128, num_heads=4, num_layers=3,
        predicate_vocab_size=32, num_clusters=16,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _build_verifier() -> "object | None":
    """Build a Bonsai-backed verifier against the local endpoint.

    Lazy: only constructed when ``--verify`` is passed. Uses the 1d Oracle
    HTTP client pattern (requests, OpenAI-compatible /chat/completions) against
    ``config.oracle_endpoint`` — the same endpoint the label generator uses.
    Returns a callable ``verifier(proposal: dict) -> bool``.
    """
    from src.training.oracle_labeling import OracleClient, OracleConfig
    from src.training.prompts import gnn_link_prediction_prompt

    client = OracleClient(OracleConfig())

    def verifier(proposal: dict) -> bool:
        # Reuse the link-prediction prompt shape: feed the single proposed edge
        # as a one-edge subgraph and ask the Oracle whether the edge should
        # exist. A ``predicted_edges`` entry for this pair ⇒ accept.
        sub = {"center": proposal["subject"], "nodes": [
            {"id": proposal["subject"], "type": "node", "depth": 0},
            {"id": proposal["object"], "type": "node", "depth": 1}],
            "edges": []}
        prompt = gnn_link_prediction_prompt(json.dumps(sub, ensure_ascii=False))
        try:
            result = client.generate(prompt)
        except Exception as e:  # pragma: no cover — network path
            logging.warning("verifier call failed: %s", e)
            return False
        preds = result.response.get("predicted_edges", []) or []
        for pe in preds:
            if (pe.get("subject") == proposal["subject"]
                    and pe.get("object") == proposal["object"]):
                return True
        return False

    return verifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3a dream-state consolidation")
    parser.add_argument("--db", default=config.db_path, help="WaveDB store path")
    parser.add_argument("--checkpoint", default=None,
                        help="Trained GNN checkpoint (without it the model is untrained)")
    parser.add_argument("--apply", action="store_true",
                        help="Mutate the store (default: dry-run, no writes)")
    parser.add_argument("--force-untrained", action="store_true",
                        help="Allow --apply without --checkpoint (an untrained model's "
                             "random salience prunes ~every edge — destructive; use only "
                             "to smoke the apply path on a throwaway store)")
    parser.add_argument("--verify", action="store_true",
                        help="Enable Bonsai-backed verification of medium-confidence edges")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on subgraphs scored (dev slice; full run leaves unset)")
    parser.add_argument("--centers", default=None,
                        help="Comma-separated episode ids to use as subgraph centers")
    parser.add_argument("--device", default="cpu", help="torch device (cpu | cuda)")
    parser.add_argument("--report", default=None,
                        help="Write the JSON report to this path")
    # ── Consolidation knobs (override ConsolidationConfig defaults). All default
    # to None so only explicitly-passed flags override -- ConsolidationConfig stays
    # the source of truth, and Consolidator(config=ConsolidationConfig(...)) (the
    # path the tests use) is unaffected. Threshold knobs (accept/bonsai/prune) are
    # also sweepable from one run via the report's score_distributions histograms.
    parser.add_argument("--accept-threshold", type=float, default=None,
                        help="Auto-accept edges/ontology proposals above this (default 0.85)")
    parser.add_argument("--bonsai-propose-threshold", type=float, default=None,
                        help="Propose to Bonsai between this and accept (default 0.60)")
    parser.add_argument("--prune-salience-below", type=float, default=None,
                        help="Archive edges where BOTH endpoints below this (default 0.15)")
    parser.add_argument("--ontology-strategy", default=None,
                        choices=["all", "topk", "rotation"],
                        help="Entity x class candidate selection (default all; "
                             "all=score every pair chunked, topk=embedding prefilter, "
                             "rotation=legacy budget slice)")
    parser.add_argument("--ontology-topk", type=int, default=None,
                        help="Classes per entity for --ontology-strategy topk (default 10)")
    parser.add_argument("--ontology-budget", type=int, default=None,
                        help="Cap for --ontology-strategy rotation (default 16)")
    parser.add_argument("--linkpred-budget", type=int, default=None,
                        help="Candidate non-edge pairs scored per subgraph (default 16)")
    parser.add_argument("--collect-bar", type=float, default=None,
                        help="Histogram collects scores >= this (default 0.0; bins are tiny)")
    # ── Anomaly head subgraph bound (the giant-subgraph fix). The anomaly step
    # runs a SECOND bounded forward (radius-2 + fanout-cap) so the anomaly head
    # is SERVED on the same bounded subgraph it TRAINED on (train/serve parity;
    # serving it on the radius-3 giant would let duplicate_episode dominate
    # again). The other 4 steps stay on the radius-3 subgraph. Defaults (None)
    # inherit ConsolidationConfig (radius=2, cap=64); 0 cap = uncapped = the
    # prior giant, for comparison.
    parser.add_argument("--anomaly-radius", type=int, default=None,
                        help="BFS radius for the anomaly step's second forward (default 2)")
    parser.add_argument("--anomaly-fanout-cap", type=int, default=None,
                        help="Per-node fanout cap for the anomaly step's subgraph (default 64; "
                             "0 = uncapped = the prior 10,680-node-giant behavior)")
    # ── Phase 3b forgetting knobs. ``--forget/--no-forget`` toggles the master
    # gate (``config.forgetting_enabled``, read by the consolidator); the three
    # thresholds override ConsolidationConfig defaults. LTP thresholds
    # (reconsolidation_count>=3 across >=15 days) are canonical constants in
    # ``src/memory/forgetting.py`` (the pure decay module), NOT CLI knobs -- the
    # worked-example fidelity gate (step 1) pinned them; exposing them invites
    # breaking the 0.010->0.0060->0.0018 repro. The deep-archive tier (>365d
    # physical remove) is deferred (no consumer yet) so has no flag.
    parser.add_argument("--forget", action=argparse.BooleanOptionalAction, default=None,
                        help="Toggle the forgetting system master gate (default on; --no-forget "
                             "makes the system behave as if forgetting were never deployed)")
    parser.add_argument("--utility-prune-below", type=float, default=None,
                        help="Soft-archive a current edge whose composed utility_score drops "
                             "below this (default 0.1; archived, NOT deleted)")
    parser.add_argument("--ontology-decay-days", type=int, default=None,
                        help="Deprecate discovered classes unseen for this many days (default 30; "
                             "seed classes are never eligible)")
    parser.add_argument("--anomaly-resolve-threshold", type=float, default=None,
                        help="contradictory_state score at/above which the resolver auto-"
                             "supersedes (default 0.8; below = record-only)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Build the ConsolidationConfig override from the non-None knobs only.
    cfg = Phase3aConfig().consolidation
    overrides = {
        "accept_threshold": args.accept_threshold,
        "bonsai_propose_threshold": args.bonsai_propose_threshold,
        "prune_salience_below": args.prune_salience_below,
        "ontology_strategy": args.ontology_strategy,
        "ontology_topk": args.ontology_topk,
        "ontology_candidate_budget": args.ontology_budget,
        "linkpred_candidate_budget": args.linkpred_budget,
        "score_collect_bar": args.collect_bar,
        "anomaly_subgraph_radius": args.anomaly_radius,
        # Phase 3b forgetting thresholds (ConsolidationConfig fields).
        "utility_prune_below": args.utility_prune_below,
        "ontology_decay_days": args.ontology_decay_days,
        "anomaly_resolve_threshold": args.anomaly_resolve_threshold,
    }
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    # Phase 3b master gate: the consolidator reads ``config.forgetting_enabled``
    # from the global singleton, so override it here (process-scoped) when the
    # flag is explicitly passed. Default (None) leaves the config default (True).
    if args.forget is not None:
        config.forgetting_enabled = args.forget
    # Anomaly fanout-cap is handled separately: 0 = uncapped (None) is a LEGITIMATE
    # override (the prior giant behavior, for comparison), so it must NOT be
    # dropped by the "is not None" filter above. Only override when the flag was
    # explicitly passed (default None -> inherit ConsolidationConfig.anomaly_fanout_cap).
    if args.anomaly_fanout_cap is not None:
        cfg = replace(cfg, anomaly_fanout_cap=(
            args.anomaly_fanout_cap if args.anomaly_fanout_cap > 0 else None))

    store = HippocampalStore(args.db)
    try:
        model = _load_model(args.checkpoint, args.device) if args.checkpoint else None
        verifier = _build_verifier() if args.verify else None
        centers = args.centers.split(",") if args.centers else None

        cons = Consolidator(
            store, model=model, verifier=verifier, config=cfg,
            dry_run=(not args.apply), device=args.device,
            allow_untrained_apply=args.force_untrained,
        )
        report = cons.run(centers=centers, limit=args.limit)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())