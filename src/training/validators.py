"""Quality validation for Phase 1d generated training data.

Reusable, side-effect-free checks used by ``scripts/validate_training_data.py``
and by the offline test suite. Each validator reads a JSONL file (or accepts an
iterable of parsed dicts for tests), checks every line is parseable and has the
expected top-level keys, and returns a summary dict. No Oracle / WaveDB
dependency — pure JSON-shape checks.

The expected shapes (from ``docs/Phase 1d.md`` and the generator outputs):

- ``gnn/{task}_labels.jsonl`` — ``{"subgraph_id", "labels": {...}, "cost"}``
  where ``labels`` carries task-specific keys.
- ``bonsai/query_planning_pairs.jsonl`` —
  ``{"conversation_id", "conversation_text", "training_pair": {"question","query","reasoning"}}``.
- ``bonsai/relation_extraction_pairs.jsonl`` —
  ``{"conversation_id", "conversation_text", "relations": [...]}``.
- ``jepa/routing_pairs.jsonl`` — ``{"query", "route": {...}}``.
- ``gates/{gate}.jsonl`` — ``{"input": {...}, "label": {...}}``.
- ``code_aware/code_aware_examples.jsonl`` —
  ``{"domain", "label": {"conversation", "extracted_entities", ...}, "cost"}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


# ── per-dataset expected shapes ──

# GNN label sub-keys per task (the "labels" payload the Oracle returns).
GNN_LABEL_KEYS: dict[str, set[str]] = {
    "salience": {"node_scores"},
    "cluster": {"clusters"},
    "link_prediction": {"predicted_edges"},
    "anomaly": {"anomalies"},
    "ontology": {"suggested_edges", "misclassified"},
}

# Top-level keys each JSONL record must have, per output file stem.
RECORD_KEYS: dict[str, set[str]] = {
    # GNN
    **{
        f"{task}_labels": {"subgraph_id", "labels"}
        for task in GNN_LABEL_KEYS
    },
    # Bonsai
    "query_planning_pairs": {"conversation_id", "training_pair"},
    "relation_extraction_pairs": {"conversation_id", "relations"},
    # JEPA
    "routing_pairs": {"query", "route"},
    # Gates
    "uncertainty_detector": {"input", "label"},
    "aspirational_model": {"input", "label"},
    "self_model": {"input", "label"},
    # Code-aware: generator wraps the Oracle response under "label".
    "code_aware_examples": {"domain", "label"},
}

# Nested label sub-keys for the code-aware "label" payload (the Oracle is told
# to return at least ``conversation`` + ``extracted_entities``).
CODE_AWARE_LABEL_KEYS = {"conversation", "extracted_entities"}


def iter_jsonl(path: Path) -> Iterable[tuple[Optional[dict], Optional[str]]]:
    """Yield ``(parsed_dict, None)`` per line, or ``(None, error_msg)`` on a parse error.

    Blank lines are skipped. Errors include the 1-based line number.
    """
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line), None
            except json.JSONDecodeError as e:
                yield None, f"line {lineno}: {e.msg}"


def validate_file(
    path: Path,
    required_keys: set[str],
    label_keys: Optional[set[str]] = None,
) -> dict:
    """Validate one JSONL file's shape.

    ``required_keys``: top-level keys every record must have.
    ``label_keys``: optional keys that must be present in a nested ``labels``
    or ``route``/``label`` sub-dict (GNN ``labels``, etc.).

    Returns ``{"path", "lines", "parse_errors", "missing_keys",
    "label_missing", "ok"}``.
    """
    parse_errors: list[str] = []
    missing: int = 0
    label_missing: int = 0
    count: int = 0
    if not path.exists():
        return {"path": str(path), "lines": 0, "parse_errors": [],
                "missing_keys": 0, "label_missing": 0, "ok": False, "missing_file": True}

    for obj, err in iter_jsonl(path):
        if err is not None:
            parse_errors.append(err)
            continue
        count += 1
        if not isinstance(obj, dict) or not required_keys <= obj.keys():
            missing += 1
            continue
        if label_keys:
            # GNN records nest under "labels"; gates under "label"; JEPA
            # under "route". Pick whichever is present.
            nested = obj.get("labels") or obj.get("label") or obj.get("route") or {}
            if not isinstance(nested, dict) or not label_keys <= nested.keys():
                # For GNN ontology, "misclassified" may be absent if the Oracle
                # found nothing — accept suggested_edges alone as a fallback.
                if label_keys == GNN_LABEL_KEYS["ontology"]:
                    if isinstance(nested, dict) and "suggested_edges" in nested:
                        continue
                label_missing += 1

    return {
        "path": str(path),
        "lines": count,
        "parse_errors": parse_errors,
        "missing_keys": missing,
        "label_missing": label_missing,
        "ok": count > 0 and not parse_errors and missing == 0 and label_missing == 0,
        "missing_file": False,
    }


def validate_gnn(gnn_dir: Path) -> dict:
    """Validate all five GNN label files."""
    out: dict[str, dict] = {}
    for task, label_keys in GNN_LABEL_KEYS.items():
        out[task] = validate_file(
            gnn_dir / f"{task}_labels.jsonl",
            required_keys=RECORD_KEYS[f"{task}_labels"],
            label_keys=label_keys,
        )
    return out


def validate_bonsai(bonsai_dir: Path) -> dict:
    return {
        "query_planning": validate_file(
            bonsai_dir / "query_planning_pairs.jsonl",
            required_keys=RECORD_KEYS["query_planning_pairs"],
        ),
        "relation_extraction": validate_file(
            bonsai_dir / "relation_extraction_pairs.jsonl",
            required_keys=RECORD_KEYS["relation_extraction_pairs"],
        ),
    }


def validate_jepa(jepa_dir: Path) -> dict:
    return {
        "routing": validate_file(
            jepa_dir / "routing_pairs.jsonl",
            required_keys=RECORD_KEYS["routing_pairs"],
        ),
    }


def validate_gates(gates_dir: Path) -> dict:
    return {
        gate: validate_file(
            gates_dir / f"{gate}.jsonl",
            required_keys=RECORD_KEYS[gate],
        )
        for gate in ("uncertainty_detector", "aspirational_model", "self_model")
    }


def validate_code_aware(code_dir: Path) -> dict:
    return {
        "code_aware_examples": validate_file(
            code_dir / "code_aware_examples.jsonl",
            required_keys=RECORD_KEYS["code_aware_examples"],
            label_keys=CODE_AWARE_LABEL_KEYS,
        ),
    }


def validate_all(data_dir: Path) -> dict:
    """Validate every Phase 1d dataset under ``data_dir``.

    Returns a nested dict keyed by dataset group. Missing files/dirs are
    reported as ``{"ok": False, "missing_file": True}`` rather than raising,
    so a partial run validates cleanly.
    """
    data_dir = Path(data_dir)
    return {
        "gnn": validate_gnn(data_dir / "gnn") if (data_dir / "gnn").exists() else {},
        "bonsai": validate_bonsai(data_dir / "bonsai") if (data_dir / "bonsai").exists() else {},
        "jepa": validate_jepa(data_dir / "jepa") if (data_dir / "jepa").exists() else {},
        "gates": validate_gates(data_dir / "gates") if (data_dir / "gates").exists() else {},
        "code_aware": validate_code_aware(data_dir / "code_aware") if (data_dir / "code_aware").exists() else {},
    }


def total_size_mb(data_dir: Path) -> float:
    """Total size of all ``*.jsonl`` under ``data_dir``, in MB."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return 0.0
    total = sum(f.stat().st_size for f in data_dir.rglob("*.jsonl") if f.is_file())
    return total / 1024 / 1024