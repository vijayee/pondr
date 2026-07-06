"""Process a conversation corpus through the hippocampal encoding pipeline.

Reads a JSONL corpus (one conversation per line, ``{"id", "turns": [[user, assistant], ...]}``),
runs each turn through the encoder (GLiNER + Bonsai → Episode → atomic WaveDB
batch), and writes the resulting memory store to ``--db``.

On RunPod, point ``--db`` at a path on a mounted network volume so the store
survives pod stop/start and can be synced back to local. The WaveDB files
under that directory are the Phase 1a deliverable.

Per-conversation errors are isolated: a single failed conversation (model
hiccup, unparseable Bonsai JSON) is logged with its line number and the
conversation id, then the run continues. The exit code is non-zero if any
conversation failed, so CI/cron can detect partial failures.

Usage:
    python scripts/process_corpus.py --input data/sample_conversations.jsonl \\
        --db /workspace/volumes/hippo/memory_db
    python scripts/process_corpus.py --input data/corpus.jsonl --limit 100

Optional ``--extractions`` writes a JSONL sidecar of per-episode extraction
results (entities/topics/tones/decisions/relations) for Step 10 quality
measurement, without re-running the models.

``--report`` writes a JSON ingestion-quality report at the end of the run:
conversation/episode counts, per-axis extraction totals + means, failure
count + list, and processing rate. ``--resume`` checkpoints progress to
``{db}/.checkpoint.json`` (the set of processed conversation ids, flushed
every ``--progress-every`` conversations and at shutdown) so a crashed or
stopped run can resume without re-encoding conversations already in the store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make ``src.*`` importable whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoding.encoder import HippocampalEncoder  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402

_CHECKPOINT_NAME = ".checkpoint.json"


def _load_checkpoint(db_path: str) -> dict:
    """Load the resume checkpoint (processed conversation ids + counters)."""
    cp_file = Path(db_path) / _CHECKPOINT_NAME
    if not cp_file.exists():
        return {"processed_ids": [], "episodes": 0, "failures": []}
    try:
        with open(cp_file, encoding="utf-8") as f:
            cp = json.load(f)
        cp.setdefault("processed_ids", [])
        cp.setdefault("episodes", 0)
        cp.setdefault("failures", [])
        cp["processed_ids"] = list(cp["processed_ids"])
        return cp
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARN: checkpoint {cp_file} unreadable ({e}); starting fresh", file=sys.stderr)
        return {"processed_ids": [], "episodes": 0, "failures": []}


def _save_checkpoint(db_path: str, cp: dict) -> None:
    """Atomically write the checkpoint (temp file + rename)."""
    cp_file = Path(db_path) / _CHECKPOINT_NAME
    tmp = cp_file.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False)
    os.replace(tmp, cp_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a conversation corpus through the encoding pipeline.")
    parser.add_argument("--input", required=True, help="JSONL file with conversations (one per line).")
    parser.add_argument(
        "--user",
        required=True,
        help="User handle (the agent's owner / a persona). Each conversation "
        "becomes one session (S:NNNN) under U:<user>. Scopes the global chat "
        "history so cross-chat recall is a first-class query.",
    )
    parser.add_argument(
        "--db",
        default="./data/memory_db",
        help="WaveDB store path. On RunPod, use a network-volume mount so the "
        "store persists across pod restarts and can be synced back to local.",
    )
    parser.add_argument("--limit", type=int, help="Max conversations to process.")
    parser.add_argument(
        "--extractions",
        help="Optional JSONL path to dump per-episode extraction results "
        "(for Step 10 quality measurement).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a progress line every N conversations.",
    )
    parser.add_argument(
        "--report",
        help="Write a JSON ingestion-quality report to this path at end of run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from {db}/.checkpoint.json — skip conversations already "
        "in the store. Checkpoint is flushed every --progress-every convs.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2

    # Parent dir for the store must exist; WaveDB creates the db files itself
    # but not nested parent directories.
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    extractions_file = None
    if args.extractions:
        Path(args.extractions).parent.mkdir(parents=True, exist_ok=True)
        extractions_file = open(args.extractions, "w", encoding="utf-8")

    checkpoint = _load_checkpoint(args.db) if args.resume else None
    done_ids: set[str] = set(checkpoint["processed_ids"]) if checkpoint else set()
    skipped_resumed = 0

    processed = 0
    episodes = 0 if not checkpoint else int(checkpoint.get("episodes", 0))
    failures: list[str] = list(checkpoint["failures"]) if checkpoint else []
    # Per-axis extraction totals across all episodes this run + checkpoint.
    axis_totals = {"entities": 0, "topics": 0, "tones": 0, "decisions": 0}
    store = None
    start_time = time.time()

    try:
        # Construct inside the try so a failure to load the GLiNER/Bonsai models
        # (e.g. running off-pod) still closes the store via the finally below
        # rather than leaking the WaveDB handle + WAL files.
        store = HippocampalStore(args.db)
        encoder = HippocampalEncoder(store, user_id=args.user)

        with open(in_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if args.limit and processed >= args.limit:
                    break

                try:
                    conv = json.loads(line)
                except json.JSONDecodeError as e:
                    failures.append(f"line {lineno}: bad JSON: {e}")
                    print(f"[fail] line {lineno}: bad JSON: {e}", file=sys.stderr)
                    continue

                turns = conv.get("turns", [])
                if not turns:
                    continue

                conv_id = conv.get("id", f"line_{lineno}")
                if conv_id in done_ids:
                    skipped_resumed += 1
                    continue

                try:
                    eps = encoder.encode_conversation(turns)
                except Exception as e:  # model error, unparseable Bonsai JSON, etc.
                    # Per the plan's process instruction, report the exact error.
                    failures.append(f"{conv_id} (line {lineno}): {e}")
                    print(f"[fail] {conv_id} (line {lineno}): {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    continue

                processed += 1
                episodes += len(eps)
                done_ids.add(conv_id)
                for ep in eps:
                    axis_totals["entities"] += len(ep.entities or [])
                    axis_totals["topics"] += len(ep.topics or [])
                    axis_totals["tones"] += len(ep.tones or [])
                    axis_totals["decisions"] += len(ep.decisions or [])

                if extractions_file is not None:
                    for ep in eps:
                        extractions_file.write(json.dumps({
                            "conversation_id": conv_id,
                            "user_id": ep.user_id,
                            "session_id": ep.session_id,
                            "episode_id": ep.id,
                            "follows": ep.follows,
                            "entities": ep.entities,
                            "topics": ep.topics,
                            "tones": ep.tones,
                            "decisions": ep.decisions,
                            "relations": ep.relations,
                        }) + "\n")

                if processed % max(args.progress_every, 1) == 0:
                    print(f"Processed {processed} conversations ({episodes} episodes)")
                    if args.resume:
                        _save_checkpoint(args.db, {
                            "processed_ids": sorted(done_ids),
                            "episodes": episodes,
                            "failures": failures,
                        })
    finally:
        if extractions_file is not None:
            extractions_file.close()
        if store is not None:
            store.close()
        if args.resume:
            _save_checkpoint(args.db, {
                "processed_ids": sorted(done_ids),
                "episodes": episodes,
                "failures": failures,
            })

    elapsed = time.time() - start_time
    print(f"\nDone. {processed} conversations, {episodes} episodes stored in {args.db}.")
    if skipped_resumed:
        print(f"Skipped {skipped_resumed} already-processed conversation(s) (resume).")

    if args.report:
        report = {
            "input": str(in_path),
            "db": args.db,
            "user": args.user,
            "conversations_processed": processed,
            "conversations_skipped_resume": skipped_resumed,
            "episodes_total": episodes,
            "mean_episodes_per_conv": (episodes / processed) if processed else 0.0,
            "failures": len(failures),
            "failure_list": failures,
            "axis_totals": axis_totals,
            "mean_per_episode": {
                k: (v / episodes) if episodes else 0.0 for k, v in axis_totals.items()
            },
            "elapsed_seconds": elapsed,
            "rate_conversations_per_sec": (processed / elapsed) if elapsed else 0.0,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote ingestion report → {args.report}")

    if failures:
        print(f"\n{len(failures)} conversation(s) failed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())