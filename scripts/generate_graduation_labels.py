"""Label the STRM 2d v2 graduation replay log -> replay_labeled.jsonl.

The v2 graduation head is trained on ``later_needed`` labels: after a WM ring
slot was compressed out, did a LATER turn in the same session reference its
content (the "would-have-been-needed" signal)? The replay logger
(``src/orchestrator.py``'s ``_write_graduation_replay``, gated on
``strm_graduation_logging``) wrote one record per ring slot per turn to
``data/training/strm_graduation/replay.jsonl`` with ``later_needed: null``.
This script fills that field and writes ``replay_labeled.jsonl``.

v1 labeling heuristic (string/source_id overlap, per the STRM plan): a slot is
``later_needed=1`` if its ``source_id`` re-appears in a LATER turn of the same
session AFTER a ring gap (the slot was compressed out then re-recalled --
re-appearance in the ring happens via ``working_memory.inject`` of a retrieved
episode, so re-appearance IS a salience recall). Consecutive-turn presence
(the slot still sitting in the ring, no eviction) does NOT count -- the slot
must be absent for >= 1 turn then come back. This is deterministic, runs on
the replay log alone (no embedder, no store), and matches the serve semantics
of the v1 ``integral(r_i dt)`` proxy (both are within-session signals).

Slots with ``source_id == None`` (the raw query step, None-provenance recalls)
cannot be matched and stay ``later_needed: null`` (the trainer drops null
labels). A future refinement (the plan's "salience recall OR consumer
search_memory/expand") would additionally mark slots whose ``source_id`` a
later turn's consumer tool referenced -- left for v2 of the labeler once the
replay log records consumer-tool calls; the re-appearance signal is the
primary one and ships now.

Usage:
    python scripts/generate_graduation_labels.py \\
        --replay data/training/strm_graduation/replay.jsonl \\
        --output data/training/strm_graduation/replay_labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

DEFAULT_REPLAY = "data/training/strm_graduation/replay.jsonl"
DEFAULT_OUTPUT = "data/training/strm_graduation/replay_labeled.jsonl"


def load_replay(path: str) -> list[dict]:
    """Load replay.jsonl -> list of records (drop unparseable lines).

    Each record is one ring slot at one turn (see ``_write_graduation_replay``).
    Drops lines that fail to parse OR lack ``turn_id``/``session_id`` -- a
    malformed record would silently mis-order the re-appearance analysis.
    """
    out: list[dict] = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if not isinstance(rec, dict) or "turn_id" not in rec or "session_id" not in rec:
                dropped += 1
                continue
            out.append(rec)
    if dropped:
        print(f"  load_replay: dropped {dropped} unparseable/malformed records",
              file=sys.stderr)
    return out


def label_later_needed(records: list[dict]) -> list[dict]:
    """Fill ``later_needed`` per record from the re-appearance heuristic.

    For each session, build the ordered set of ``turn_id``s at which each
    ``source_id`` appears. A slot occurrence at turn ``t`` (with a non-null
    ``source_id``) is ``later_needed=1`` iff its ``source_id`` appears at some
    later turn ``t2`` with a gap -- i.e. there is a turn ``t1`` in ``(t, t2)``
    where the source_id is ABSENT (it was compressed out then re-recalled).
    Consecutive presence (no gap) is the slot still sitting in the ring, not a
    re-recall, so it does not count. Records with ``source_id == None`` keep
    ``later_needed: null`` (unmatchable; the trainer drops them).

    Returns a NEW list (the input records are not mutated); each output
    record is a shallow copy with ``later_needed`` set.
    """
    # turns_per_session: session_id -> sorted unique turn_ids present in the log.
    # appearance: (session_id, source_id) -> sorted list of turn_ids it appears at.
    turns_per_session: dict[str, list[int]] = defaultdict(list)
    appearance: dict[tuple[str, str], list[int]] = defaultdict(list)
    seen_turns: dict[str, set[int]] = defaultdict(set)
    for rec in records:
        sid = rec["session_id"]
        tid = rec["turn_id"]
        if tid not in seen_turns[sid]:
            seen_turns[sid].add(tid)
            turns_per_session[sid].append(tid)
        src = rec.get("source_id")
        if src is not None:
            appearance[(sid, str(src))].append(tid)
    for sid in turns_per_session:
        turns_per_session[sid].sort()
    for key in appearance:
        appearance[key] = sorted(set(appearance[key]))

    # A source_id is "absent at turn u" if u is a present turn of the session
    # and the source_id does not appear at u.
    def _later_needed(session_id: str, source_id: str, turn_id: int) -> Optional[bool]:
        sess_turns = turns_per_session.get(session_id, [])
        if not sess_turns:
            return None
        appears_at = appearance.get((session_id, source_id), [])
        appears_set = set(appears_at)
        # A slot occurrence at turn_id is "later needed" iff its source_id
        # re-appears at some later turn u AND the slot was absent for at least
        # one session turn strictly between turn_id and u (compressed out then
        # re-recalled -- consecutive presence is just the ring still holding it).
        for u in appears_at:
            if u <= turn_id:
                continue
            if any((v not in appears_set) for v in sess_turns if turn_id < v < u):
                return True
        return False

    out: list[dict] = []
    for rec in records:
        src = rec.get("source_id")
        ln = _later_needed(rec["session_id"], str(src), rec["turn_id"]) \
            if src is not None else None
        new = dict(rec)
        new["later_needed"] = ln
        out.append(new)
    return out


def write_labeled(records: list[dict], output: str) -> dict:
    """Write replay_labeled.jsonl (one record per line). Returns label stats."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = neg = null = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            ln = rec.get("later_needed")
            if ln is True:
                pos += 1
            elif ln is False:
                neg += 1
            else:
                null += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"positive": pos, "negative": neg, "null": null, "total": len(records)}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Label the STRM 2d replay log -> replay_labeled.jsonl")
    p.add_argument("--replay", default=DEFAULT_REPLAY,
                   help="replay.jsonl from the orchestrator's graduation logger")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="labeled JSONL output (v2 graduation training input)")
    args = p.parse_args()

    replay_path = Path(args.replay)
    if not replay_path.exists():
        print(f"ERROR: replay log not found at {replay_path}", file=sys.stderr)
        return 1
    records = load_replay(args.replay)
    if not records:
        print(f"ERROR: no records in {replay_path}", file=sys.stderr)
        return 1
    print(f"  loaded {len(records)} replay records", flush=True)
    labeled = label_later_needed(records)
    stats = write_labeled(labeled, args.output)
    print(f"DONE. wrote {stats['total']} records -> {args.output}", flush=True)
    print(f"  later_needed: positive={stats['positive']} "
          f"negative={stats['negative']} null={stats['null']}", flush=True)
    if stats["positive"] == 0:
        print("  WARNING: zero positive labels -- the v2 head has nothing to "
              "learn. Run more sessions with --strm-graduation-logging until a "
              "source_id re-appears after a ring gap.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())