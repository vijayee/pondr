#!/usr/bin/env python
"""Recall-before/after consolidation eval (the missing #6 gate).

The audit's missing item: no end-to-end evaluation of what an apply does to
RECALL on a real corpus before it touches one. This script runs the eval on a
DISPOSABLE COPY of the DB (the real store is never opened for writing):

1. copy the DB directory to a temp sibling,
2. sample probe episodes (by default 32, seeded -- deterministic),
3. apply one Consolidator pass -- trained checkpoint REQUIRED -- on the copy,
4. re-run the semantic-vector probe per episode and check the probe's content
   is STILL reachable: the source episode in the hits, OR an applied
   abstractor M-node (`abstracted_from` contains the probe -- the M-node is
   what the default candidate set replaced the source with).

Probe semantics: hits are filtered to DEFAULT candidate semantics -- an
abstracted source (`content/ep/{eid}/abstracted = 1`) no longer counts as its
own hit (the graph leg + FAISS leg exclude `abstracted` episodes; the
WaveDB-layer full-index search does not, so the probe applies that filter
itself to model what the retrieval path does with the result).

Retention: fraction of probes still reachable post-apply. A probe whose source
was not in its own top-L hits EVEN pre-apply (degenerate: empty summary /
crowded embedding space) is excluded from the denominator and reported.

Exit code: 0 iff retention >= --min-retention (default 1.0: EVERY reachable
probe stays reachable). Use it as CI-style evidence before --dream-apply /
run_consolidation --apply on a real corpus. Note: with
``ConsolidationConfig.apply_gate_enabled`` (default) an apply without a wired
decider is REFUSED by the gate, so pass ``--decide`` with the Bonsai server up
-- without it this script trivially reports nothing mutated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Phase3aConfig  # noqa: E402
from src.gnn.consolidate import Consolidator  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402
from src.retrieval.wavedb_vector_store import WavedbVectorStore  # noqa: E402
from src.retrieval.vector_search import _sentence_transformers_embedder  # noqa: E402
from src.subconscious.dream_worker import _load_trained_model  # noqa: E402


def _copy_db(db_path: str) -> Path:
    """Copy the DB (a WaveDB directory, or a single file) to a temp path."""
    src = Path(db_path)
    if not src.exists():
        raise SystemExit(f"db not found: {db_path}")
    root = Path(tempfile.mkdtemp(prefix="consol_gate_eval_"))
    dst = root / src.name
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True, help="WaveDB store path (read-only: "
                   "the eval copies it; the real DB is never written)")
    p.add_argument("--checkpoint", required=True,
                   help="Trained combined GNN checkpoint (the eval refuses an "
                        "untrained model -- random salience prunes ~everything)")
    p.add_argument("--decide", action="store_true",
                   help="Wire the Bonsai decider (REQUIRED for an apply that "
                        "creates abstracts; without it the eval gate refuses "
                        "the apply and retention is trivially 1.0). Requires "
                        "the local Bonsai server.")
    p.add_argument("--probes", type=int, default=32,
                   help="Number of probe episodes to sample (default 32)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--depth", type=int, default=20,
                   help="Vector top-L per probe (default 20 -- deep enough "
                        "that rank loss shows up before slot loss)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-retention", type=float, default=1.0,
                   help="Exit 0 iff retention >= this (default 1.0)")
    p.add_argument("--limit", type=int, default=None,
                   help="Consolidator subgraph limit (a bounded smoke run)")
    p.add_argument("--apply-gate-max-prunes", type=int, default=None)
    p.add_argument("--apply-gate-max-abstracts", type=int, default=None)
    args = p.parse_args()

    copy_path = _copy_db(args.db)
    print(f"[eval] DB copy at {copy_path}")

    store = HippocampalStore(str(copy_path))
    results: dict = {"db": args.db, "copy": str(copy_path), "probes": [],
                     "excluded_probes": []}
    try:
        # 1. Probe sampling: default-candidate episodes with a summary.
        ids = store.default_episode_ids(include_abstracted=False)
        with_summary = []
        for eid in ids:
            ep = store.get_episode(eid)
            if ep is not None and (ep.summary or "").strip():
                with_summary.append(eid)
        if len(with_summary) > args.probes:
            with_summary = random.Random(args.seed).sample(
                with_summary, args.probes)
        if not with_summary:
            raise SystemExit("no episodes with summaries to probe")
        print(f"[eval] {len(with_summary)} probe(s)")

        embedder = _sentence_transformers_embedder()
        backend = WavedbVectorStore(store, embedder=embedder)

        def _hits(probe_summaries: list[str]) -> dict[str, set[str]]:
            """Full-index top-L per probe (batched encode, one model load)."""
            vecs = embedder.encode(probe_summaries)
            out: dict[str, set[str]] = {}
            for eid, vec in zip(with_summary, vecs):
                hits = backend.search_by_vector(
                    [float(x) for x in vec], k=args.depth)
                out[eid] = {h for h, _ in hits}
            return out

        pre = _hits([store.get_episode(eid).summary for eid in with_summary])

        # 2. One apply pass on the copy (the eval gate inside run() governs).
        cfg = Phase3aConfig().consolidation
        if args.apply_gate_max_prunes is not None:
            cfg = replace(cfg, apply_max_prunes=args.apply_gate_max_prunes)
        if args.apply_gate_max_abstracts is not None:
            cfg = replace(cfg, apply_max_abstracts=args.apply_gate_max_abstracts)
        decider = None
        if args.decide:
            from src.gnn.bonsai_decider import BonsaiDecider
            decider = BonsaiDecider()
        model = _load_trained_model(args.checkpoint, args.device)
        cons = Consolidator(store, model=model, config=cfg,
                            dry_run=False, device=args.device, decider=decider)
        report = cons.run(limit=args.limit)
        results["report"] = {k: report[k] for k in (
            "dry_run", "trained", "subgraphs_scored", "abstracts",
            "edges_accepted", "pruned", "apply_skipped",
            "abstracts_applied",)}
        skipped = report.get("apply_skipped")

        # 3. Abstractor map: only M-nodes applied THIS pass exist on the copy.
        abstractor_of: dict[str, str] = {}
        for a in report["abstracts_applied"]:
            for eid in a["episodes"]:
                abstractor_of[eid] = a["mid"]

        post = _hits([store.get_episode(eid).summary for eid in with_summary])

        # 4. Verdicts: reachability = source hit (still default-candidate) or
        # the applied abstractor M-node in the hits. Retention counts AFFECTED
        # probes only -- an unaffected probe is not evidence the apply is safe
        # (folding it into the denominator dilutes a 1-lost signal to ~0.1
        # lost on a big corpus, defeating the gate).
        kept = failed = excluded = unaffected = 0
        for eid in with_summary:
            abstracted = (store.db.get_sync(f"content/ep/{eid}/abstracted")
                          or b"") not in (b"", None)
            if not abstracted:
                # Never abstracted: nothing changed for it.
                results["probes"].append(
                    {"eid": eid, "affected": False, "reachable": True})
                unaffected += 1
                continue
            pre_ok = eid in pre[eid]  # degenerate probe guard
            hit_ids = post[eid]
            reachable = (eid in hit_ids
                         or abstractor_of.get(eid, None) in hit_ids)
            row = {"eid": eid, "affected": True, "reachable": reachable,
                   "pre_top_ok": pre_ok,
                   "abstractor": abstractor_of.get(eid)}
            results["probes"].append(row)
            if not pre_ok:
                results["probes"][-1]["excluded"] = "degenerate pre-hit"
                results["excluded_probes"].append(eid)
                excluded += 1
            elif reachable:
                kept += 1
            else:
                failed += 1
        results["counts"] = {"probes": len(with_summary),
                             "unaffected": unaffected,
                             "affected": kept + failed + excluded,
                             "kept": kept, "failed": failed,
                             "degenerate_excluded": excluded}
        # Retention denominator: affected probes that were reachable
        # pre-apply (degenerate probes are excluded from the metric).
        denom = kept + failed
        results["retention"] = (kept / denom) if denom else 1.0
        ok = results["retain_threshold"] = (
            results["retention"] >= args.min_retention)
        print(json.dumps({k: results[k] for k in (
            "counts", "retention", "retain_threshold")}, indent=2))
        if skipped:
            print(f"[eval] apply was SKIPPED ({skipped}) -- the numbers above "
                  "measure an unchanged store; pass --decide (Bonsai up) for a "
                  "real eval.")
        for row in results["probes"]:
            if row.get("affected") and not row.get("reachable", True):
                print(f"[eval] LOST {row['eid']} (abstractor "
                      f"{row.get('abstractor')})")
        return 0 if ok else 1
    finally:
        store.close()
        if os.environ.get("EVAL_KEEP_COPY"):
            print(f"[eval] copy KEPT at {copy_path} (EVAL_KEEP_COPY)")
        else:
            shutil.rmtree(copy_path.parent, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())