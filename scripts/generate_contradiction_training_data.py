#!/usr/bin/env python
"""Stage B: generate LoRA fine-tune training pairs for the Bonsai contradiction
path (Phase 3c Sec 7 / plan mellow-jumping-token.md).

Two task types, one per axis the zero-shot eval (Sec 7.1-7.8) showed the 8B
fails: (1) EXTRACTION -- emit the ``has_state(Entity, Value)`` predicate the
production encoder lifts (8B ignores the schema zero-shot, 0/13 strict); (2)
ADJUDICATION -- discriminate real conflicts (fix) from non-conflicts (dismiss)
instead of rubber-stamping (8B false-fixes 3/3 negatives zero-shot).

Labels are PLANTED/STRUCTURAL -- we design the pair so we already know the
gold answer; NO model judges any pair (the eval proved model judges rubber-
stamp). The generator (DeepSeek via Ollama, through OracleClient) is used ONLY
as a paraphraser: it writes the natural-language INPUT doc for an extraction
pair from a planted spec (entity, value, style, optional person/decision).
Adjudication pairs are fully structural (the deploy decider takes a structured
flag + provenance, not raw docs) -- no generator calls.

Each record is a standard chat-message pair (``{"messages": [{"role":"user",
"content": <deploy-time prompt>}, {"role":"assistant", "content": <gold JSON>]}``).
The user turn is the EXACT deploy-time prompt the production extractor/decider
sends (``BONSAI_RELATION_PROMPT`` from src/encoding/bonsai_relations.py and
``bonsai_contradiction_decision_prompt`` from src/training/prompts.py), so the
fine-tune targets the schema the product actually sends at runtime. The
assistant turn is clean JSON (no fences) so the model learns to emit parseable
output.

Output (gitignored, regenerable): ``data/training/bonsai/contradiction_pairs.jsonl``
+ a ``quality_report.json`` summary. Smoke first with small --num-*, inspect,
then scale.

    python scripts/generate_contradiction_training_data.py \
        --oracle-model deepseek-v4-flash:cloud --oracle-endpoint http://localhost:11434/v1 \
        --num-extraction 200 --num-adjudication 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoding.bonsai_relations import BONSAI_RELATION_PROMPT  # noqa: E402
from src.training.generator_common import (  # noqa: E402
    add_oracle_args, make_oracle, write_jsonl, write_report, run_batches,
)
from src.training.prompts import bonsai_contradiction_decision_prompt  # noqa: E402

# ── spec pool ───────────────────────────────────────────────────────────
# (canonical_entity, [plausible values]). The canonical name is what the gold
# has_state uses; the generator is told to refer to the entity by this name (a
# natural possessive like "the team's <entity>" is allowed). Entities span the
# fixture's domains + extra domains for generalization.
ENTITIES = [
    ("database", ["MySQL", "Postgres", "SQLite", "DuckDB"]),
    ("data layer", ["MySQL", "Postgres"]),
    ("deployment target", ["staging", "production"]),
    ("framework", ["React", "Svelte", "Vue", "Solid"]),
    ("hosting provider", ["AWS", "GCP", "Azure"]),
    ("CI runner", ["GitHub Actions", "self-hosted Jenkins", "CircleCI"]),
    ("secrets store", ["Vault", "AWS Secrets Manager", "Doppler"]),
    ("package manager", ["npm", "pnpm", "yarn", "bun"]),
    ("region", ["us-east-1", "eu-west-1", "ap-southeast-2"]),
    ("cache", ["redis", "memcached", "Valkey"]),
    ("queue", ["Kafka", "RabbitMQ", "SQS"]),
    ("search backend", ["Elasticsearch", "OpenSearch", "Typesense"]),
    ("monitoring stack", ["Datadog", "Prometheus", "Grafana Cloud"]),
    ("ticket status", ["open", "closed", "in progress"]),
    ("build status", ["green", "red", "flaky"]),
    ("language runtime", ["Python 3.11", "Python 3.12", "Go 1.22"]),
    ("orchestrator", ["Kubernetes", "Nomad", "ECS"]),
    ("load balancer", ["ALB", "HAProxy", "Cloudflare"]),
]

DOC_STYLES = [
    "team Slack/Teams channel message",
    "engineering design doc section",
    "meeting minutes / decision log entry",
    "email to the engineering team",
    "GitHub PR description",
    "README / runbook section",
    "retrospective note",
    "incident postmortem section",
]

# People for optional decides(...) relations -- keeps the full relation schema
# active so the LoRA does not collapse to has_state-only.
PEOPLE = ["Alice", "Bob", "the platform team", "Priya", "the infra group", "Sam"]

# Adjudication conflict types and their gold decision + action.
# - real: same entity, different values, same ongoing scope, newer supersedes
#   older -> fix + supersede_assertion (the only action the dispatcher auto-applies)
# - complementary_temporal: time-qualified values at different dates (N14 shape)
#   -> dismiss (point-in-time facts don't conflict)
# - same_value: same entity, same value -> dismiss (no collision)
# - different_entity: different entities, same value -> dismiss (no shared entity)
CONFLICT_TYPES = {
    "real":                  {"decision": "fix",     "action": "supersede_assertion"},
    "complementary_temporal":{"decision": "dismiss", "action": "no_action"},
    "same_value":            {"decision": "dismiss", "action": "no_action"},
    "different_entity":      {"decision": "dismiss", "action": "no_action"},
}
# Adjudication-type sampling weights (real conflicts dominate so recall stays
# high; the three negative shapes are weighted toward the load-bearing N14).
CONFLICT_WEIGHTS = {
    "real": 50,
    "complementary_temporal": 22,
    "same_value": 14,
    "different_entity": 14,
}


def _gen_extraction_prompt(spec: dict) -> str:
    """Prompt the GENERATOR (DeepSeek) to write one realistic input doc.

    The generator is a paraphraser, NOT a judge: it is told the planted facts
    (entity, value, optional person/decision) and the doc style, and writes
    natural prose. It must NOT use the literal token "has_state" (that is the
    gold predicate, not prose) and must keep the canonical entity name.
    """
    person_clause = ""
    if spec.get("person"):
        person_clause = (
            f" Also have {spec['person']} make or announce this decision "
            f"(e.g. \"{spec['person']} decided/announced that ...\")."
        )
    return f"""Write a single short {spec['style']} (2-5 sentences) in which an
engineering team establishes that the {spec['entity']} is now, or has been
chosen as, {spec['value']}. Vary the phrasing naturally for the {spec['style']}
format -- do NOT use the literal token "has_state". Refer to the entity as
"{spec['entity']}" (a natural possessive like "the team's {spec['entity']}" is
fine). Sound like real engineering writing, not a template.{person_clause}

Return ONLY valid JSON:
{{"body": "<the doc body text, no heading markdown, just the prose>"}}"""


# ── extraction pairs (need the generator) ──────────────────────────────

def _build_extraction_specs(n: int, rng: random.Random) -> list[dict]:
    specs: list[dict] = []
    for _ in range(n):
        entity, values = rng.choice(ENTITIES)
        value = rng.choice(values)
        style = rng.choice(DOC_STYLES)
        spec = {"entity": entity, "value": value, "style": style}
        # ~40% also include a person making the decision -> gold has decides(...) too
        if rng.random() < 0.4:
            spec["person"] = rng.choice(PEOPLE)
        specs.append(spec)
    return specs


def _extraction_gold(spec: dict) -> dict:
    """Construct the gold extraction JSON structurally from the planted spec.

    Always includes has_state(entity, value); optionally decides(person, decision).
    This keeps the full relation schema active so the LoRA does not collapse
    every doc to has_state-only (which would risk forgetting decides/explains).
    """
    rels = [{"subject": spec["entity"], "predicate": "has_state", "object": spec["value"]}]
    if spec.get("person"):
        decision = f"adopt {spec['value']} for {spec['entity']}"
        rels.append({"subject": spec["person"], "predicate": "decides", "object": decision})
    return {"relations": rels}


def _extraction_to_record(spec: dict, result, idx: int) -> dict:
    """Build a chat-message training record from the generator's doc."""
    body = ""
    if getattr(result, "error", None) is None and isinstance(result.response, dict):
        body = str(result.response.get("body", "") or "").strip()
    if not body:
        # failure sentinel -> skip (run_batches counts it as failed; we drop it
        # from the JSONL so the trainer never sees an empty input).
        return {}
    user = BONSAI_RELATION_PROMPT.format(text=body)
    gold = _extraction_gold(spec)
    assistant = json.dumps(gold, ensure_ascii=False)
    return {
        "id": f"ext_{idx:04d}",
        "task": "extraction",
        "spec": spec,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


# ── adjudication pairs (fully structural, no generator) ────────────────

def _build_adjudication_specs(n: int, rng: random.Random) -> list[dict]:
    types = list(CONFLICT_TYPES.keys())
    weights = [CONFLICT_WEIGHTS[t] for t in types]
    specs: list[dict] = []
    for _ in range(n):
        ctype = rng.choices(types, weights=weights, k=1)[0]
        entity, values = rng.choice(ENTITIES)
        if ctype == "different_entity":
            # two different but confusable entities, same value
            entity2, values2 = rng.choice(ENTITIES)
            while entity2 == entity:
                entity2, values2 = rng.choice(ENTITIES)
            value = rng.choice(list(set(values) & set(values2) or values))
            specs.append({"conflict_type": ctype, "entity": entity,
                          "entity2": entity2, "value": value})
        elif ctype == "same_value":
            value = rng.choice(values)
            specs.append({"conflict_type": ctype, "entity": entity,
                          "old_value": value, "new_value": value})
        elif ctype == "complementary_temporal":
            old_v, new_v = rng.sample(values, 2) if len(values) >= 2 else (values[0], values[0])
            specs.append({"conflict_type": ctype, "entity": entity,
                          "old_value": old_v, "new_value": new_v,
                          "old_date": "2026-01-15", "new_date": "2026-07-15"})
        else:  # real
            old_v, new_v = rng.sample(values, 2) if len(values) >= 2 else (values[0], values[0])
            specs.append({"conflict_type": ctype, "entity": entity,
                          "old_value": old_v, "new_value": new_v,
                          "old_date": "2026-07-14", "new_date": "2026-07-15"})
    return specs


def _adjudication_context(spec: dict) -> dict:
    """Build the structured flag + retrieved context the deploy decider sees.

    Mirrors bonsai_contradiction_decision_prompt input: flagged_entity +
    state_values (value, asserted_by, asserted_at) + surrounding ctx. The
    episode summaries carry the distinguishing signal (point-in-time vs
    going-forward) so the decider can learn to discriminate.
    """
    ctype = spec["conflict_type"]
    entity = spec["entity"]
    if ctype == "different_entity":
        # Two different entities; the "flag" is still on one, but ctx shows the
        # other entity owns the second value -> dismiss (no shared entity).
        e2 = spec["entity2"]
        state_values = [
            {"value": spec["value"], "asserted_by": f"docs/{entity.replace(' ', '-')}-v1.md",
             "asserted_at": "2026-07-14"},
            {"value": spec["value"], "asserted_by": f"docs/{e2.replace(' ', '-')}-v1.md",
             "asserted_at": "2026-07-15"},
        ]
        ctx = {"states": [
            {"entity": entity, "value": spec["value"], "source": "v1"},
            {"entity": e2, "value": spec["value"], "source": "v1"},
        ]}
        return {"flagged_entity": entity, "state_values": state_values, "ctx": ctx}

    if ctype == "same_value":
        old_v = new_v = spec["old_value"]
        ep_old = f"{entity} is {old_v} per the v1 plan."
        ep_new = f"{entity} is {new_v} per the v2 plan."
    elif ctype == "complementary_temporal":
        old_v, new_v = spec["old_value"], spec["new_value"]
        ep_old = f"As of {spec['old_date']}, {entity} was {old_v}."
        ep_new = f"As of {spec['new_date']}, {entity} is {new_v}."
    else:  # real
        old_v, new_v = spec["old_value"], spec["new_value"]
        ep_old = f"Going forward, {entity} is {old_v}."
        ep_new = f"Going forward, {entity} is {new_v} (supersedes the prior plan)."

    state_values = [
        {"value": old_v, "asserted_by": f"docs/{entity.replace(' ', '-')}-v1.md",
         "asserted_at": spec.get("old_date", "2026-07-14")},
        {"value": new_v, "asserted_by": f"docs/{entity.replace(' ', '-')}-v2.md",
         "asserted_at": spec.get("new_date", "2026-07-15")},
    ]
    ctx = {"episodes": [{"summary": ep_old}, {"summary": ep_new}]}
    return {"flagged_entity": entity, "state_values": state_values, "ctx": ctx}


def _adjudication_gold(spec: dict) -> dict:
    g = CONFLICT_TYPES[spec["conflict_type"]]
    ctype = spec["conflict_type"]
    if ctype == "real":
        reasoning = (f"Same {spec['entity']} carries two different live values; the "
                    f"newer ({spec['new_value']}) supersedes the older ({spec['old_value']}).")
    elif ctype == "complementary_temporal":
        reasoning = (f"Values are time-qualified at different dates ({spec['old_date']} "
                    f"vs {spec['new_date']}); both are true at their respective times "
                    f"-- not a contradiction.")
    elif ctype == "same_value":
        reasoning = (f"Both assertions give the same value ({spec['old_value']}); no "
                    f"collision, no contradiction.")
    else:  # different_entity
        reasoning = (f"The two values belong to different entities ({spec['entity']} vs "
                    f"{spec['entity2']}); no shared entity, no contradiction.")
    return {"decision": g["decision"], "action": g["action"], "reasoning": reasoning}


def _build_adjudication_records(specs: list[dict]) -> list[dict]:
    records = []
    for i, spec in enumerate(specs):
        env = _adjudication_context(spec)
        user = bonsai_contradiction_decision_prompt(env["flagged_entity"], {
            "state_values": env["state_values"], **env["ctx"],
        })
        gold = _adjudication_gold(spec)
        records.append({
            "id": f"adj_{i:04d}",
            "task": "adjudication",
            "spec": spec,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
            ],
        })
    return records


# ── main ───────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Generate Bonsai contradiction LoRA training pairs.")
    p.add_argument("--output", default="data/training/bonsai/",
                   help="Output directory (default: %(default)s)")
    p.add_argument("--num-extraction", type=int, default=200,
                   help="Extraction pairs to generate (needs the generator)")
    p.add_argument("--num-adjudication", type=int, default=200,
                   help="Adjudication pairs to generate (structural, no generator)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default %(default)s)")
    p.add_argument("--report", action="store_true",
                   help="Write quality_report.json summary")
    p.add_argument("--resume", action="store_true",
                   help="Resume extraction generation from per-task checkpoint (cached oracle calls are free)")
    add_oracle_args(p)
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ── 1. Adjudication pairs (structural, no oracle) ──
    print("Generating adjudication pairs (structural)...")
    adj_specs = _build_adjudication_specs(args.num_adjudication, rng)
    adj_records = _build_adjudication_records(adj_specs)
    by_type = {}
    for s in adj_specs:
        by_type[s["conflict_type"]] = by_type.get(s["conflict_type"], 0) + 1
    print(f"  {len(adj_records)} adjudication pairs: {by_type}")

    # ── 2. Extraction pairs (need the generator) ──
    print("\nGenerating extraction pairs (via generator)...")
    oracle = make_oracle(args, output_dir)
    ext_specs = _build_extraction_specs(args.num_extraction, rng)
    ext_records, ext_stats = run_batches(
        oracle, ext_specs,
        build_prompt=lambda it, i: _gen_extraction_prompt(it),
        to_record=_extraction_to_record,
        output_dir=output_dir, task_name="contradiction_extraction",
        batch_size=args.oracle_batch_size, resume=args.resume,
        progress_label="extraction pairs", max_workers=args.oracle_max_workers,
    )
    # Drop empty (failed) extraction records -- the trainer must not see empty inputs.
    ext_records = [r for r in ext_records if r]
    print(f"  {len(ext_records)} extraction pairs kept "
          f"({ext_stats.get('failed', 0)} failed/dropped)")

    # ── 3. Write JSONL ──
    all_records = ext_records + adj_records
    out_file = output_dir / "contradiction_pairs.jsonl"
    write_jsonl(out_file, all_records)

    # ── 4. Summary ──
    n_ext = len(ext_records)
    n_adj = len(adj_records)
    summary = {
        "total_pairs": n_ext + n_adj,
        "extraction_pairs": n_ext,
        "adjudication_pairs": n_adj,
        "adjudication_by_type": by_type,
        "oracle_stats": oracle.get_stats(),
        "extraction_stats": ext_stats,
        "output": str(out_file),
    }
    print("\n" + "=" * 60)
    print(f"TOTAL {summary['total_pairs']} pairs -> {out_file}")
    print(f"  extraction:   {n_ext}")
    print(f"  adjudication: {n_adj}  {by_type}")
    print(f"  oracle: {oracle.get_stats()}")
    if args.report:
        write_report(output_dir / "contradiction_quality_report.json", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())