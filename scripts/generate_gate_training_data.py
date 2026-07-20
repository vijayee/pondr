"""Generate gate training data (Uncertainty Detector, Aspirational Model,
Self-Model, Common Sense Resolver).

Usage (validate slice):
    python scripts/generate_gate_training_data.py \\
        --db data/pod_runs/phase1b_scale/ingest_db_dialogsum \\
        --output data/training/gates/ --num-examples 40

Adapted from ``docs/Phase 1d.md`` §8 to the REAL WaveDB API: episodes come
from ``sample_episode_centers`` + ``GraphTraversal.hydrate_episode``; related
episodes via the public ``episodes_by_entity`` / ``episodes_by_topic`` aliases
(the doc's ``in_("in_episode")`` was the wrong direction). ``--seed`` makes the
validate slice reproducible.

The Common Sense Resolver (4th gate, Phase 4d) produces deliberately-ambiguous
query variants from a polyseme vocabulary so the Oracle labels the three CSR
branches: not-ambiguous / context-resolves / genuinely-ambiguous.
"""

from __future__ import annotations

import argparse
import json
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
    aspirational_model_prompt,
    common_sense_resolver_prompt,
    self_model_prompt,
    uncertainty_detector_prompt,
)


def _all_episodes(store, traversal, count, rng):
    """Return ``count`` random hydrated episodes.

    Shuffles the (cheap) episode-id list first, then hydrates lazily until
    ``count`` summary-bearing episodes are collected — so we pay ~``count``
    ``hydrate_episode`` calls (each is several graph queries), not hundreds.
    """
    ids = sample_episode_centers(store, n=None)
    rng.shuffle(ids)
    out = []
    for eid in ids:
        hy = traversal.hydrate_episode(eid)
        if hy.get("summary"):
            out.append(hy)
        if len(out) >= count:
            break
    return out


def _get_related_episodes(traversal, episode_id: str, entities, limit=5) -> list[dict]:
    """Episodes sharing an entity with ``episode_id`` (excluding it)."""
    related: set[str] = set()
    for entity in (entities or [])[:3]:
        for eid in traversal.episodes_by_entity(entity):
            related.add(eid)
    related.discard(episode_id)
    out = []
    for rid in list(related)[:limit]:
        hy = traversal.hydrate_episode(rid)
        if hy.get("summary"):
            out.append({"id": hy["episode_id"], "summary": hy["summary"]})
    return out


def _build_uncertainty_inputs(store, traversal, count, rng):
    episodes = _all_episodes(store, traversal, count, rng)
    inputs = []
    for ep in episodes:
        related = _get_related_episodes(traversal, ep["episode_id"], ep.get("entities"), limit=5)
        context = "\n".join(r["summary"] for r in related) or "(no related episodes)"
        query = _generate_query(ep)
        roll = rng.random()
        if roll < 0.3:
            retrieval = "No results found."
        elif roll < 0.5:
            retrieval = f"Found {len(related)} partially relevant episodes."
        else:
            retrieval = f"Found {len(related)} relevant episodes:\n" + \
                        "\n".join(r["summary"] for r in related[:3])
        inputs.append({"context": context, "query": query, "retrieval_results": retrieval})
    return inputs


def _build_aspirational_inputs(store, traversal, count, rng):
    episodes = _all_episodes(store, traversal, count, rng)
    inputs = []
    for ep in episodes:
        topics = ep.get("topics", [])
        entities = ep.get("entities", [])
        goal_context = f"Recent topics: {', '.join(topics)}. Recent entities: {', '.join(entities[:5])}."
        actions = [
            f"Encode this episode about {topics[0] if topics else 'this topic'}",
            f"Set a reminder to follow up on {entities[0] if entities else 'this'}",
            f"Explore more about {topics[0] if topics else 'this'}",
            "Skip encoding this routine conversation",
        ]
        inputs.append({"goal_context": goal_context, "candidate_action": rng.choice(actions)})
    return inputs


def _build_self_model_inputs(store, traversal, count, rng):
    episodes = _all_episodes(store, traversal, count, rng)
    inputs = []
    for ep in episodes:
        topic = (ep.get("topics") or ["unknown"])[0]
        episode_count = len(traversal.episodes_by_topic(topic))
        if episode_count > 10:
            knowledge_state = f"Dense knowledge: {episode_count} episodes about {topic}."
        elif episode_count > 3:
            knowledge_state = f"Moderate knowledge: {episode_count} episodes about {topic}."
        else:
            knowledge_state = f"Sparse knowledge: {episode_count} episodes about {topic}."
        query = (f"What is the exact {topic} configuration we used?"
                 if rng.random() < 0.4 else f"What did we discuss about {topic}?")
        inputs.append({"knowledge_state": knowledge_state, "query": query})
    return inputs


def _generate_query(episode: dict) -> str:
    if episode.get("entities"):
        return f"What did {episode['entities'][0]} say?"
    if episode.get("topics"):
        return f"What did we discuss about {episode['topics'][0]}?"
    return "What was this conversation about?"


# A small polyseme vocabulary for building deliberately-ambiguous CSR inputs.
# Each entry maps a surface word to its plausible readings; the generator picks
# a word + a branch (not-ambiguous / context-resolves / genuinely-ambiguous) and
# the Oracle labels which CSR check applies. Extended freely; the point is to
# exercise all three branches of ``common_sense_resolver_prompt``.
_POLYSEMES: dict[str, list[str]] = {
    "bank": ["a financial institution", "the side of a river"],
    "pitch": ["a sales presentation", "a playing field", "the slope of a roof"],
    "match": ["a game or contest", "a small fire-starting stick"],
    "rock": ["a stone", "a musical genre"],
    "sentence": ["a grammatical unit", "a prison term"],
    "current": ["happening right now", "an electrical flow"],
    "frame": ["a picture border", "a structural support"],
    "draft": ["a written document", "a current of air"],
}


def _build_common_sense_inputs(store, traversal, count, rng):
    """Build deliberately-ambiguous CSR inputs (Phase 4d, the 4th gate).

    For each sampled episode, build an ambiguous query from a polyseme and a
    disambiguating context (the episode + related episodes), exercising all
    three CSR branches so the Oracle labels a balanced mix:
      ~40% NOT AMBIGUOUS       -- one candidate only (a clear reading).
      ~40% CONTEXT RESOLVES    -- 2-3 candidates, context present (the Oracle
                                  checks whether the episode summary supports
                                  one reading).
      ~20% GENUINELY AMBIGUOUS  -- 2 candidates, context stripped so neither
                                  reading is favored (should_ask_clarification).
    """
    episodes = _all_episodes(store, traversal, count, rng)
    inputs = []
    for ep in episodes:
        related = _get_related_episodes(traversal, ep["episode_id"],
                                        ep.get("entities"), limit=3)
        ctx_lines = [f"[{ep['episode_id']}] {ep.get('summary', '')}"]
        ctx_lines += [f"[{r['id']}] {r['summary']}" for r in related]
        retrieved_context = "\n".join(ctx_lines) or "(no related episodes)"

        word = rng.choice(list(_POLYSEMES))
        readings = _POLYSEMES[word]
        roll = rng.random()
        if roll < 0.4:
            # NOT AMBIGUOUS: a single candidate -> the only sensible reading.
            input_text = f"What did we decide about {word}?"
            candidates = json.dumps([readings[0]], ensure_ascii=False)
        elif roll < 0.8:
            # CONTEXT RESOLVES: the full candidate set + the disambiguating
            # context -- the Oracle checks whether the episode summary favors
            # one reading.
            input_text = f"What did we decide about {word}?"
            candidates = json.dumps(readings, ensure_ascii=False)
        else:
            # GENUINELY AMBIGUOUS: two candidates, context stripped so neither
            # reading is favored -> the honest CSR answer is ask_clarification.
            input_text = f"I need to get to the {word}."
            retrieved_context = "(no related context)"
            candidates = json.dumps(readings[:2], ensure_ascii=False)
        inputs.append({
            "input_text": input_text,
            "retrieved_context": retrieved_context,
            "candidate_interpretations": candidates,
        })
    return inputs


GATES = [
    ("uncertainty_detector", uncertainty_detector_prompt, _build_uncertainty_inputs),
    ("aspirational_model", aspirational_model_prompt, _build_aspirational_inputs),
    ("self_model", self_model_prompt, _build_self_model_inputs),
    ("common_sense_resolver", common_sense_resolver_prompt, _build_common_sense_inputs),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate gate training data")
    parser.add_argument("--db", default="data/pod_runs/phase1b_scale/ingest_db_dialogsum",
                        help="WaveDB store path (ingested corpus)")
    parser.add_argument("--output", default="data/training/gates/", help="Output directory")
    parser.add_argument("--num-examples", type=int, default=40,
                        help="Total examples across all 4 gates (validate-slice default 40)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from per-gate checkpoints")
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
        per_gate = max(1, args.num_examples // len(GATES))
        all_stats: dict = {}

        for gate_name, prompt_fn, input_builder in GATES:
            print(f"\nGenerating {gate_name} training data...")
            inputs = input_builder(store, traversal, per_gate, rng)

            def build_prompt(inp, _idx, _pf=prompt_fn):
                return _pf(**inp)

            def to_record(inp, result, _idx):
                return {"input": inp, "label": result.response, "cost": result.cost}

            records, stats = run_batches(
                oracle, inputs, build_prompt, to_record,
                output_dir, gate_name, args.oracle_batch_size, args.resume,
                progress_label="examples",
            )
            write_jsonl(output_dir / f"{gate_name}.jsonl", records)
            all_stats[gate_name] = {"examples": len(records), **stats}
            print(f"  {gate_name}: {len(records)} examples")

        report = {
            "per_gate": all_stats,
            "oracle_stats": oracle.get_stats(),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        write_report(output_dir / "quality_report.json", report)
        print(f"\nGate training data generation complete.")
        print(f"Total Oracle calls: {oracle.total_calls}  tokens: {oracle.total_tokens}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())