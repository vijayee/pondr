"""Generate synthetic code-aware training data.

Usage (validate slice):
    python scripts/generate_code_aware_data.py \\
        --output data/training/code_aware/ --num-examples 10

Synthetic — no WaveDB corpus read. For each example we pick a code domain,
carve a focused ontology fragment (the class + its subclasses + the properties
touching it) out of ``SEED_ONTOLOGY``, and ask the Oracle to synthesize a
training example. Ported from ``docs/Phase 1d.md`` §9 with the
``get_current_ontology`` graph query replaced by the shipped ``SEED_ONTOLOGY``
(the conversational schema has no ``subClassOf`` graph edges to query).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.ontology import SEED_ONTOLOGY  # noqa: E402
from src.training.generator_common import (  # noqa: E402
    add_oracle_args,
    make_oracle,
    run_batches,
    write_jsonl,
    write_report,
)
from src.training.prompts import code_aware_synthetic_prompt  # noqa: E402

# Domains we synthesize examples for. Each is a top-level code class in
# ``CODE_CLASSES`` (merged into ``SEED_ONTOLOGY["classes"]``).
DOMAINS = [
    "CodeArtifact",
    "VersionControl",
    "Issue",
    "Test",
    "Architecture",
    "API",
    "Data",
    "Configuration",
    "Infrastructure",
    "Deployment",
    "Observability",
    "Quality",
]


def _ontology_fragment(ontology: dict, domain: str) -> str:
    """Carve the ``domain`` class + its subclasses + touching properties.

    Returns a JSON string the Oracle can ground synthetic examples in, instead
    of dumping the whole (large) merged ontology into every prompt.
    """
    classes = ontology.get("classes", {})
    node = classes.get(domain)
    if node is None:
        # Fall back to the full class list if the domain isn't a class itself.
        fragment_classes = {domain: {"subclasses": []}}
    else:
        fragment_classes = {domain: node}
        # Include one hop of subclasses as their own entries so the Oracle sees
        # the children's children too.
        for sub in node.get("subclasses", []):
            sub_node = classes.get(sub)
            if sub_node is not None:
                fragment_classes[sub] = sub_node

    # Properties whose domain or range mentions the domain or one of its subs.
    relevant_keys = {domain, *(node.get("subclasses", []) if node else [])}
    fragment_props = {}
    for pname, spec in ontology.get("properties", {}).items():
        dom = spec.get("domain", "")
        rng = spec.get("range", "")
        if dom in relevant_keys or rng in relevant_keys:
            fragment_props[pname] = spec

    return json.dumps(
        {"classes": fragment_classes, "properties": fragment_props},
        ensure_ascii=False,
    )


def _build_items(num_examples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    for _ in range(num_examples):
        domain = rng.choice(DOMAINS)
        fragment = _ontology_fragment(SEED_ONTOLOGY, domain)
        items.append({"domain": domain, "code_ontology_fragment": fragment})
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate code-aware training data")
    parser.add_argument("--output", default="data/training/code_aware/",
                        help="Output directory")
    parser.add_argument("--num-examples", type=int, default=10,
                        help="Number of synthetic examples (validate-slice default 10)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    add_oracle_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle = make_oracle(args, output_dir)
    start_time = time.time()
    items = _build_items(args.num_examples, seed=args.seed)
    print(f"Generating {len(items)} code-aware synthetic examples...")

    def build_prompt(item, _idx):
        return code_aware_synthetic_prompt(item["domain"], item["code_ontology_fragment"])

    def to_record(item, result, _idx):
        return {"domain": item["domain"], "label": result.response, "cost": result.cost}

    records, stats = run_batches(
        oracle, items, build_prompt, to_record,
        output_dir, "code_aware", args.oracle_batch_size, args.resume,
        progress_label="examples",
    )
    write_jsonl(output_dir / "code_aware_examples.jsonl", records)

    from collections import Counter
    domain_counts = Counter(r["domain"] for r in records)
    report = {
        "examples": len(records),
        "domain_distribution": dict(domain_counts),
        "stats": stats,
        "oracle_stats": oracle.get_stats(),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    write_report(output_dir / "quality_report.json", report)
    print(f"\nGenerated {len(records)} code-aware examples. Domains: {dict(domain_counts)}")
    print(f"Total Oracle calls: {oracle.total_calls}  tokens: {oracle.total_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())