"""Shared helpers for the Phase 1d training-data generator scripts.

Factored out so the six generators (``scripts/generate_*_training_data.py``)
share one implementation of: Oracle-client construction from CLI args,
per-task checkpoint save/load (resume support), JSONL writing, and the
``quality_report.json`` summary. One place to de-wonk the resume / IO logic.

Generators are invoked as ``python scripts/generate_<task>_training_data.py``
with ``sys.path.insert(0, repo_root)`` (see the existing
``scripts/generate_training_data.py``), so they import this as
``from src.training.generator_common import ...``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from ..config import config as _config
from .oracle_labeling import OracleClient, OracleConfig


def add_oracle_args(parser) -> None:
    """Add ``--oracle-*`` overrides to an argparse parser.

    Defaults come from the Hippo ``Config`` (``oracle_*`` fields, env-
    overridden) so the flags are only needed to swap models per-run.
    """
    parser.add_argument("--oracle-model", default=_config.oracle_model,
                        help="Oracle model name (default: %(default)s)")
    parser.add_argument("--oracle-endpoint", default=_config.oracle_endpoint,
                        help="Oracle OpenAI-compatible endpoint (default: %(default)s)")
    parser.add_argument("--oracle-temperature", type=float, default=_config.oracle_temperature)
    parser.add_argument("--oracle-max-tokens", type=int, default=_config.oracle_max_tokens,
                        help="Max output tokens (default: %(default)s; raise if a task truncates)")
    parser.add_argument("--oracle-batch-size", type=int, default=10,
                        help="Oracle calls per batch (progress/checkpoint granularity)")
    parser.add_argument("--oracle-timeout", type=float, default=_config.oracle_timeout)
    parser.add_argument("--oracle-max-workers", type=int, default=1,
                        help="Concurrent Oracle calls per batch (default 1 = sequential; "
                             ">1 dispatches across a thread pool so network-bound calls "
                             "overlap. Token cost is unchanged; only wall-clock shrinks. "
                             "The cache + stat counters are lock-guarded so this is safe.")


def make_oracle(args, output_dir: Path) -> OracleClient:
    """Build an ``OracleClient`` from CLI args, with an on-disk cache under ``output_dir``.

    The cache (``.oracle_cache.json``) makes resumes cheap: identical prompts
    are not re-sent to the Oracle across runs.
    """
    cfg = OracleConfig(
        model=args.oracle_model,
        endpoint=args.oracle_endpoint,
        temperature=args.oracle_temperature,
        max_tokens=args.oracle_max_tokens,
        timeout=args.oracle_timeout,
        cache_path=output_dir / ".oracle_cache.json",
    )
    return OracleClient(cfg)


def load_checkpoint(path: Path) -> tuple[int, list]:
    """Load a per-task checkpoint → ``(last_index, results)``; ``(0, [])`` if absent/corrupt."""
    if not path.exists():
        return 0, []
    try:
        with open(path, encoding="utf-8") as f:
            cp = json.load(f)
        return int(cp.get("last_index", 0)), list(cp.get("results", []))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0, []


def save_checkpoint(path: Path, results: list, last_index: int) -> None:
    """Persist a per-task checkpoint (index + results-so-far) for ``--resume``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_index": last_index, "results": results}, f)


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Overwrite ``path`` with one JSON record per line (UTF-8, ASCII preserved)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to ``path`` (used when streaming without a final rewrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_report(path: Path, report: dict) -> None:
    """Write a ``quality_report.json`` summary for one generator run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def run_batches(
    oracle: OracleClient,
    items: list,
    build_prompt: Any,
    to_record: Any,
    output_dir: Path,
    task_name: str,
    batch_size: int,
    resume: bool,
    extra_stats: Optional[dict] = None,
    progress_label: str = "items",
    max_workers: int = 1,
) -> tuple[list, dict]:
    """Drive the per-task Oracle batch loop with checkpointing.

    - ``items``: the full ordered list of inputs for this task.
    - ``build_prompt(item, idx) -> str``: render the Oracle prompt for one item.
    - ``to_record(item, result, idx) -> dict``: build the JSONL record from the
      Oracle result.
    - Checkpoints to ``output_dir / f"{task_name}_checkpoint.json"`` each batch.
    - Returns ``(records, stats)`` where stats is a ``Counter``-style dict.

    On ``resume``, skips items[:start_idx] and reuses checkpointed records.
    """
    from collections import Counter

    stats: Counter = Counter()
    checkpoint_path = output_dir / f"{task_name}_checkpoint.json"
    start_idx, records = (load_checkpoint(checkpoint_path) if resume else (0, []))
    if start_idx:
        print(f"  {task_name}: resuming from index {start_idx} ({len(records)} cached records)")

    start_time = time.time()
    for i in range(start_idx, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompts = [build_prompt(it, i + j) for j, it in enumerate(batch)]
        batch_results = oracle.generate_batch(prompts, max_workers=max_workers)
        for j, (it, result) in enumerate(zip(batch, batch_results)):
            records.append(to_record(it, result, i + j))
            stats["labeled"] += 1
            stats["total_cost"] += result.cost
            stats["total_input_tokens"] += result.input_tokens
            stats["total_output_tokens"] += result.output_tokens
            if result.cached:
                stats["cached"] += 1
        save_checkpoint(checkpoint_path, records, i + len(batch))
        oracle.flush_cache()
        done = min(i + batch_size, len(items))
        print(f"  {task_name}: {done}/{len(items)} {progress_label} "
              f"(calls={oracle.total_calls} tokens={oracle.total_tokens} "
              f"${oracle.total_cost:.4f})")

    stats["elapsed_seconds"] = round(time.time() - start_time, 2)
    if extra_stats:
        stats.update(extra_stats)
    return records, dict(stats)