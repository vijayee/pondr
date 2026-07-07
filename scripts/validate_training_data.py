"""Validate Phase 1d generated training data.

Usage:
    python scripts/validate_training_data.py --data-dir data/training/

Reads every ``*.jsonl`` the generators wrote under ``--data-dir`` and reports,
per dataset, whether each line parses and has the expected top-level keys (and
nested label keys where applicable). Pure JSON-shape checks — no Oracle /
WaveDB dependency — so this is safe to run on a partial or completed run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.validators import (  # noqa: E402
    total_size_mb,
    validate_all,
)


def _print_group(name: str, results: dict) -> tuple[int, int]:
    """Print one dataset group; return ``(files_ok, files_total)``."""
    if not results:
        print(f"\n{name}: (no output directory)")
        return 0, 0
    ok_count = 0
    total = 0
    for item, res in results.items():
        total += 1
        path = res.get("path", "?")
        short = Path(path).name if path != "?" else item
        if res.get("missing_file"):
            mark = "MISSING"
            detail = "no file"
        elif res["ok"]:
            mark = "OK"
            ok_count += 1
            detail = f"{res['lines']} lines"
        else:
            mark = "FAIL"
            parts = []
            if res["parse_errors"]:
                parts.append(f"{len(res['parse_errors'])} parse errors")
            if res["missing_keys"]:
                parts.append(f"{res['missing_keys']} missing-key records")
            if res["label_missing"]:
                parts.append(f"{res['label_missing']} label-missing records")
            detail = "; ".join(parts) or "unknown"
            if res["parse_errors"]:
                detail += " | e.g. " + res["parse_errors"][0]
        print(f"  [{mark}] {name}/{short}: {detail}")
    return ok_count, total


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1d training data")
    parser.add_argument("--data-dir", default="data/training/",
                        help="Root of generated training-data outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        return 1

    results = validate_all(data_dir)
    print(f"Validating Phase 1d training data under {data_dir}\n")

    total_ok = 0
    total_files = 0
    for group, res in results.items():
        ok, n = _print_group(group, res)
        total_ok += ok
        total_files += n

    size_mb = total_size_mb(data_dir)
    print(f"\n{'=' * 60}")
    print(f"Files OK: {total_ok}/{total_files}   total size: {size_mb:.2f} MB")
    return 0 if (total_files > 0 and total_ok == total_files) else 1


if __name__ == "__main__":
    sys.exit(main())