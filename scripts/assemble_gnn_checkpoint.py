"""Assemble one combined GNN checkpoint from per-head retrained sources.

Problem this solves
-------------------
``scripts/run_consolidation.py:_load_model`` loads a SINGLE state_dict into the full
``GNNModel`` (``torch.load`` + ``load_state_dict``). But the best weights for each head
currently live in DIFFERENT files:

  * backbone (input_proj + GAT layers), salience, link_prediction -- the #125 joint
    run ``all.pt`` (these heads were not broken and were not retrained).
  * diffpool -- ``diffpool_retrain/cluster.pt`` (refined; the #125 diffpool collapsed
    to 1 cluster, fixed + retrained).
  * anomaly -- ``anomaly_retrain/anomaly.pt`` (refined; the #125 anomaly collapsed to
    predict-0, fixed + retrained).
  * ontology + the taxonomy encoder -- ``ontology_trained.pt`` (the two-encoder pair
    classifier trained separately; ``all.pt`` predates the taxonomy encoder and has NO
    ``taxonomy.*`` keys, and its single-encoder ``ontology.*`` weights are untrained).

So pointing consolidation at ``all.pt`` silently runs the OLD broken diffpool/anomaly
and a random taxonomy encoder; pointing it at any one retrained file fixes one head but
leaves the others broken. This script overlays the right head weights from each source
into one strict-loadable ``all_fixed.pt`` so consolidation runs all 5 heads in their
fixed/trained state.

Architecture note: ``all.pt`` is the pre-two-encoder model (48 keys); the retrained
files + ``ontology_trained.pt`` are the current model (72 keys -- 24 ``taxonomy.*``
added by the ontology two-encoder fix). The backbone/salience/linkpred/diffpool/anomaly/
ontology.net keys are identical in name + shape across old and new (verified), so the
overlay is a clean per-prefix replacement. The final assembled state has all 72 keys.

Usage (defaults pick the Phase 3a production sources)::

    python scripts/assemble_gnn_checkpoint.py
    python scripts/assemble_gnn_checkpoint.py --output data/pod_runs/phase3a/all_fixed.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402  (hard dep via src.gnn; module-top, NOT local -- see train_gnn.py)

from src.gnn.model import GNNModel  # noqa: E402

# Head-prefix -> source checkpoint. A prefix of "taxonomy." is grouped with ontology
# (the taxonomy encoder is the ontology head's class-side encoder; it only has trained
# weights in the ontology run). Keys are taken from the base for anything not overlaid.
DEFAULT_BACKBONE = "data/pod_runs/phase3a/all.pt"
DEFAULT_DIFFPOOL = "data/pod_runs/phase3a/diffpool_retrain/cluster.pt"
DEFAULT_ANOMALY = "data/pod_runs/phase3a/anomaly_retrain/anomaly.pt"
DEFAULT_ONTOLOGY = "data/pod_runs/phase3a/ontology_trained.pt"
DEFAULT_OUTPUT = "data/pod_runs/phase3a/all_fixed.pt"


def _load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _overlay(combined: dict, source: dict, prefixes: tuple[str, ...],
             label: str) -> list[str]:
    """Copy every key starting with one of ``prefixes`` from ``source`` into
    ``combined``. Returns the keys overlaid (for the provenance report). Raises if a
    prefix matches NO key in ``source`` (a silent empty overlay would leave stale
    weights from the base -- fail loud instead)."""
    overlaid: list[str] = []
    for prefix in prefixes:
        matched = [k for k in source if k.startswith(prefix)]
        if not matched:
            avail = sorted({k.split(".")[0] for k in source})
            raise KeyError(
                f"source {label!r} has no keys with prefix {prefix!r} -- cannot overlay; "
                f"available prefixes: {avail}")
        for k in matched:
            combined[k] = source[k].clone()
            overlaid.append(k)
    return overlaid


def main() -> int:
    p = argparse.ArgumentParser(
        description="Assemble one combined GNN checkpoint from per-head retrained sources.")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="Base state_dict: provides input_proj/layers/salience/linkpred "
                        "(the #125 frozen backbone + unbroken heads).")
    p.add_argument("--diffpool", default=DEFAULT_DIFFPOOL,
                   help="State_dict providing diffpool.* (refined cluster head).")
    p.add_argument("--anomaly", default=DEFAULT_ANOMALY,
                   help="State_dict providing anomaly.* (refined anomaly head).")
    p.add_argument("--ontology", default=DEFAULT_ONTOLOGY,
                   help="State_dict providing ontology.* + taxonomy.* (trained two-encoder).")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Where to write the combined strict-loadable checkpoint.")
    args = p.parse_args()

    # 1. Start from the backbone base (everything not overlaid stays from here).
    combined = _load(args.backbone)
    print(f"base   {args.backbone}: {len(combined)} keys", flush=True)

    # 2. Overlay each head from its best source.
    provenance: dict[str, list[str]] = {}
    provenance["diffpool (from " + args.diffpool + ")"] = _overlay(
        combined, _load(args.diffpool), ("diffpool.",), args.diffpool)
    provenance["anomaly (from " + args.anomaly + ")"] = _overlay(
        combined, _load(args.anomaly), ("anomaly.",), args.anomaly)
    provenance["ontology + taxonomy (from " + args.ontology + ")"] = _overlay(
        combined, _load(args.ontology), ("ontology.", "taxonomy."), args.ontology)
    for label, keys in provenance.items():
        print(f"overlay  {label}: {len(keys)} keys", flush=True)

    # 3. Verify the assembled state strict-loads into the current GNNModel (the same
    #    contract run_consolidation._load_model relies on). A failure here means the
    #    sources are architecturally inconsistent -- do NOT write the checkpoint.
    model = GNNModel(hidden_dim=128, num_heads=4, num_layers=3,
                     predicate_vocab_size=32, num_clusters=16)
    missing, unexpected = model.load_state_dict(combined, strict=True)
    assert not missing, f"assembled state missing keys: {missing}"
    assert not unexpected, f"assembled state has unexpected keys: {unexpected}"
    param_count = sum(v.numel() for v in combined.values())

    # 4. Provenance report: which head-prefix groups came from which source (base vs
    #    overlaid), so a reader can audit exactly what is in the combined checkpoint.
    print("\nprovenance (per head-prefix group):", flush=True)
    groups = sorted({k.split(".")[0] for k in combined})
    src_for = {"diffpool": args.diffpool, "anomaly": args.anomaly,
               "ontology": args.ontology, "taxonomy": args.ontology}
    for g in groups:
        keys = [k for k in combined if k.startswith(g + ".")]
        src = src_for.get(g, f"base({args.backbone})")
        print(f"  {g:12s} {len(keys):2d} keys  <- {src}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in combined.items()}, out)
    print(f"\nwrote {out}  ({len(combined)} keys, {param_count:,} params, strict-loadable)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())