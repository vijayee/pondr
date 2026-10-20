"""Layer-3 sandbox: a repairable verification world for the expansion loop.

WHAT THIS IS: a prepared fixture (track C) for the non-reach -> new-intent
mechanism, not an organic-data claim. Each session materializes a real dataset
on disk and hides ONE verification rule the agent cannot see:

    header     -- one file begins with a header row that is not data
    exclude    -- one file is excluded from the manifest
    name_hash  -- the checksum convention includes sorted file names
    dedupe     -- duplicate rows across files count once

The agent submits a manifest spec {files, rows, hash_mode}. The verifier is
mechanical: it compares the spec to the values computed from the files. On a
mismatch it returns a reason that NAMES the violated rule -- that reason is the
evidence the expansion arm gets and the no-evidence control does not.

The harness (scripts/icm/run_layer3_expansion.py) asks:
    does the agent, after a VIOLATED outcome, form a new intent and reach
    (EXPANSION), versus repeating the same spec (RETRY) or retrying without
    the reason (NO-EVIDENCE)?

The world is deterministic per (session, rule); no model judgment is used for
the outcome.
"""

from __future__ import annotations

import os
import random

RULES = ("header", "exclude", "name_hash", "dedupe")
HASH_MODES = ("content", "content+names")
EXCLUDED_FILE = "part-03.csv"
HEADER_FILE = "part-01.csv"
DUPLICATE_ROWS = 3


class Layer3Env:
    """One sandbox per run; sessions are subdirectories with real files."""

    def __init__(self, root: str, seed: int = 11):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.seed = seed
        self._truth: dict[str, dict] = {}
        self._meta: dict[str, dict] = {}

    def materialize(self, session_id: str, rule: str) -> dict:
        if rule not in RULES:
            raise KeyError(f"unknown rule {rule!r}")
        session_dir = os.path.join(self.root, session_id)
        if os.path.isdir(session_dir):
            for fn in os.listdir(session_dir):
                os.remove(os.path.join(session_dir, fn))
        os.makedirs(session_dir, exist_ok=True)
        rng = random.Random(f"{self.seed}|{session_id}|{rule}")
        base = rng.randint(18, 26)
        counts = [base + rng.randint(-3, 3) for _ in range(4)]
        files: list[list[str]] = []
        for fi, n in enumerate(counts):
            rows = [f"row-{session_id}-{fi}-{r}={rng.randint(0, 999)}"
                    for r in range(n)]
            if rule == "header" and fi == 1:
                rows = ["id,value"] + rows
            files.append(rows)
        if rule == "dedupe":
            # exact duplicate rows across two files
            files[2].extend(files[0][:DUPLICATE_ROWS])
        for fi, rows in enumerate(files):
            with open(os.path.join(session_dir, f"part-{fi:02d}.csv"),
                      "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(rows) + "\n")
        names = [f"part-{i:02d}.csv" for i in range(4)]
        line_counts = []
        for name in names:
            with open(os.path.join(session_dir, name), encoding="utf-8") as f:
                line_counts.append(sum(1 for _ in f))
        data_rows = sum(counts)
        if rule == "exclude":
            data_rows -= counts[3]
        if rule == "dedupe":
            data_rows -= DUPLICATE_ROWS
        truth = {
            "rule": rule,
            "files": 4 - (1 if rule == "exclude" else 0),
            "rows": data_rows,
            "hash_mode": "content+names" if rule == "name_hash" else "content",
            "excluded": EXCLUDED_FILE if rule == "exclude" else None,
            "excluded_lines": counts[3] if rule == "exclude" else 0,
            "header": HEADER_FILE if rule == "header" else None,
            "duplicates": DUPLICATE_ROWS if rule == "dedupe" else 0,
        }
        self._truth[session_id] = truth
        self._meta[session_id] = {
            "files": names,
            "line_counts": line_counts,
            "total_lines": sum(line_counts),
        }
        return dict(self._meta[session_id])

    def meta(self, session_id: str) -> dict:
        return dict(self._meta[session_id])

    def correct(self, session_id: str) -> dict:
        return dict(self._truth[session_id])

    def check(self, session_id: str, spec) -> dict:
        truth = self._truth[session_id]
        if not isinstance(spec, dict):
            return {"ok": False, "reason": "invalid manifest: not a JSON object"}
        try:
            files = int(spec.get("files"))
            rows = int(spec.get("rows"))
        except (TypeError, ValueError):
            return {"ok": False, "reason": "invalid manifest: files/rows must be integers"}
        mode = str(spec.get("hash_mode", ""))
        if mode not in HASH_MODES:
            return {"ok": False,
                    "reason": "invalid manifest: hash_mode must be 'content' or 'content+names'"}
        # report ALL mismatches: one expansion should be able to fix the
        # manifest in a single step (the verifier lists its defects)
        errors = []
        if files != truth["files"]:
            if truth["rule"] == "exclude":
                errors.append(
                    f"files mismatch: expected {truth['files']}, got {files}; "
                    f"{truth['excluded']} is excluded from the manifest "
                    f"(its {truth['excluded_lines']} lines are not counted).")
            else:
                errors.append(
                    f"files mismatch: expected {truth['files']}, got {files}.")
        if rows != truth["rows"]:
            if truth["rule"] == "header":
                errors.append(
                    f"rows mismatch: expected {truth['rows']}, got {rows}; "
                    f"{truth['header']} has a header row that is not data.")
            elif truth["rule"] == "dedupe":
                errors.append(
                    f"rows mismatch: expected {truth['rows']}, got {rows}; "
                    f"{truth['duplicates']} duplicate rows across files count once.")
            elif truth["rule"] == "exclude":
                errors.append(
                    f"rows mismatch: expected {truth['rows']}, got {rows}; "
                    f"the {truth['excluded_lines']} rows of the excluded file "
                    f"are not counted.")
            else:
                errors.append(
                    f"rows mismatch: expected {truth['rows']}, got {rows}.")
        if mode != truth["hash_mode"]:
            if truth["hash_mode"] == "content+names":
                errors.append(
                    "hash_mode mismatch: expected 'content+names'; the checksum "
                    "includes sorted file names.")
            else:
                errors.append(
                    "hash_mode mismatch: expected 'content'; the checksum does "
                    "not include file names.")
        if errors:
            return {"ok": False, "reason": " | ".join(errors)}
        return {"ok": True, "reason": "ok"}


def reason_keywords(reason: str) -> tuple[str, ...]:
    """Deterministic keywords for the explanation-alignment check: the
    explanation must address the failure the verifier actually reported, which
    may be a different field than the hidden rule (e.g. a row-count error under
    the name_hash rule)."""
    r = str(reason).lower()
    keys: list[str] = []
    if "header" in r:
        keys += ["header"]
    if "exclud" in r or "skip" in r:
        keys += ["exclud", "skip"]
    if "duplicate" in r:
        keys += ["duplicate", "unique", "dedup"]
    if "hash_mode" in r:
        keys += ["hash", "checksum", "name"]
    if "files mismatch" in r:
        keys += ["file"]
    if "rows mismatch" in r:
        keys += ["row", "line", "count"]
    return tuple(dict.fromkeys(keys)) or ("row", "file", "hash")
