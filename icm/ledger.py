"""Expectation-ledger storage on WaveDB -- Phase 0 data model.

Layout (single-node, embedded; recall path never needs the network):

    KV subtree "icm":
        session/<sid>            SessionRecord
        intent/<iid>             IntentRecord
        expectation/<eid>        ExpectationRecord (resolution filled at outcome time)
        outcome/<oid>            OutcomeRecord
        index/ids                {"session": [...], "intent": [...], "expectation": [...], "outcome": [...]}
    Graph layer "icm" (structural links, SPO/POS/OSP/PSO quadruple writes):
        intent:<iid>   is_a            Intent
        intent:<iid>   has_expectation expectation:<eid>
        expectation:<eid> belongs_to   intent:<iid>
        expectation:<eid> has_congruence confirmed|violated
        outcome:<oid>  resolves        expectation:<eid>

Single-writer assumption: the ledger maintains the `index/ids` record by
read-modify-write, which is safe for one instrumentation process. Concurrent
writers are explicitly out of scope for Phase 0.

Persistence contract: everything sync-API (`*Sync` on subtree / root). Bytes
are canonical JSON (ensure_ascii, sort_keys) so re-reads are byte-identical.
Close order is subtree -> graph -> db (WaveDB refuses close with open subtrees).
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid

import wavedb

from icm.congruence import (Congruence, RESIDUE_VALUES, evaluate_congruence,  # noqa: E402
                            validate_branches)

HORIZON_KINDS = ("step", "time", "manual")
STATUS_PENDING, STATUS_RESOLVED, STATUS_EXPIRED = "pending", "resolved", "expired"
RECORD_KINDS = ("session", "intent", "expectation", "outcome", "episode",
                "icapture")
ID_KEYS = {"session": "session_id", "intent": "intent_id",
           "expectation": "exp_id", "outcome": "outcome_id",
           "episode": "episode_id", "icapture": "id"}


class LedgerError(ValueError):
    pass


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _parse(raw: bytes, key: str) -> dict:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:  # corrupted record is a hard error, not a fallback
        raise LedgerError(f"corrupt record at {key}: {e}") from e


class ICMLedger:
    """Sync-only expectation ledger on WaveDB (KV subtree + graph links)."""

    def __init__(self, db_path: str, quiet: bool = True):
        os.makedirs(db_path, exist_ok=True)
        # quiet=False streams WaveDB's C-layer logs (WAL recovery, compaction,
        # eviction) -- diagnostics runs should stay LOUD: the recovery lines
        # ("Read N manifest entries / Found N WAL files") are the state
        # transition record when data loss needs forensics.
        wavedb.set_quiet(quiet)
        self._db = wavedb.WaveDB(db_path)
        self._store = self._db.open_subtree("icm")
        self._graph = wavedb.GraphLayer("icm", self._db)
        self._ids: dict[str, list[str]] = {k: [] for k in RECORD_KINDS}
        self._extra_closers: list = []   # layers that must close BEFORE db
        raw = self._get_raw("index/ids")
        if raw is not None:
            try:
                self._ids = json.loads(raw)
            except Exception as e:
                raise LedgerError(f"corrupt id index: {e}") from e

    def register_close(self, closer) -> None:
        """Layer close contracts run BEFORE db.close (subtrees-first order)."""
        self._extra_closers.append(closer)

    # ---- low-level ----

    def _get_raw(self, key: str):
        # missing keys return None (no raise) on wavedb 0.2.12
        return self._db.get_sync(key)

    def _put_raw(self, key: str, value: bytes) -> None:
        self._db.put_sync(key, value)

    def rec_key(self, kind: str, rec_id: str) -> str:
        return f"{kind}/{rec_id}"

    def _write_record(self, kind: str, rec: dict) -> None:
        rec_id = rec[ID_KEYS[kind]]
        key = self.rec_key(kind, rec_id)
        if rec_id not in self._ids[kind]:
            self._ids[kind].append(rec_id)
            self._put_raw("index/ids", _canonical(self._ids))
        self._put_raw(key, _canonical(rec))

    # ---- reads ----

    def get(self, kind: str, rec_id: str):
        raw = self._get_raw(self.rec_key(kind, rec_id))
        return None if raw is None else _parse(raw, self.rec_key(kind, rec_id))

    def ids(self, kind: str) -> list[str]:
        return list(self._ids[kind])

    def records(self, kind: str):
        for rec_id in self._ids[kind]:
            rec = self.get(kind, rec_id)
            if rec is None:
                raise LedgerError(f"index lists {kind}/{rec_id} but record is missing")
            yield rec

    def count(self, kind: str) -> int:
        return len(self._ids[kind])

    def delete_record(self, kind: str, rec_id: str) -> None:
        """Remove a record's KV key and registry entry (registry is the
        enumeration source, so both must go, or records() raises)."""
        if rec_id not in self._ids[kind]:
            raise LedgerError(f"{kind}/{rec_id} not in registry")
        self._ids[kind].remove(rec_id)
        self._put_raw("index/ids", _canonical(self._ids))
        self._db.del_sync(self.rec_key(kind, rec_id))

    def put(self, kind: str, rec: dict) -> None:
        """Direct record write with id/key/registry invariants enforced."""
        rec_id = rec.get(ID_KEYS[kind])
        if not rec_id or not isinstance(rec_id, str):
            raise LedgerError(f"{kind} record missing string {ID_KEYS[kind]}")
        existing = self.get(kind, rec_id)
        merged = {**rec}
        if existing:
            merged = {**existing, **rec}
        merged["updated_ts"] = time.time()
        self._write_record(kind, merged)

    # ---- domain construction ----

    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    def put_session(self, session_id: str, meta: dict | None = None) -> dict:
        rec = {"session_id": session_id, "meta": meta or {}, "created_ts": time.time()}
        self.put("session", rec)
        return rec

    def put_intent(self, intent_id: str, session_id: str, aim: str,
                   created_step: int = 0, slots: dict | None = None) -> dict:
        """slots = S-P-O decomposition (docs/Intent_Subject_Predicate_Object_Schema.md):
        {subject, attitude, relation, object, residue, propositions, slot_version}.
        residue == "underspecified" marks an OPEN intent -- an object that does
        not yet exist, i.e. a prospect to be checked, not a malformed record."""
        if not aim:
            raise LedgerError("intent aim must be non-empty")
        if slots is not None:
            residue = slots.get("residue")
            if residue is not None and residue not in RESIDUE_VALUES[1:]:
                raise LedgerError(f"residue {residue!r} not in {RESIDUE_VALUES[1:]}")
        rec = {"intent_id": intent_id, "session_id": session_id, "aim": aim,
               "status": "active", "created_step": created_step, "created_ts": time.time()}
        if slots is not None:
            rec["slots"] = dict(slots)
            rec["open_object"] = slots.get("residue") == "underspecified"
        self.put("intent", rec)
        self._graph.insert_sync(f"intent:{intent_id}", "is_a", "Intent")
        return rec

    def put_expectation(self, exp_id: str, session_id: str, intent_id: str, statement: str,
                        branches: dict, horizon: dict, context_ref: dict,
                        recall_refs: list[str] | None = None,
                        created_step: int = 0) -> dict:
        try:
            validate_branches(branches)
        except ValueError as e:
            raise LedgerError(str(e)) from e
        kind = horizon.get("kind")
        if kind not in HORIZON_KINDS:
            raise LedgerError(f"horizon.kind must be one of {HORIZON_KINDS}")
        if kind == "step" and not isinstance(horizon.get("deadline_step"), int):
            raise LedgerError("horizon.kind='step' requires integer deadline_step")
        if kind == "time" and not isinstance(horizon.get("deadline_ts"), (int, float)):
            raise LedgerError("horizon.kind='time' requires numeric deadline_ts")
        if self.get("intent", intent_id) is None:
            raise LedgerError(f"expectation references unknown intent {intent_id}")
        rec = {
            "exp_id": exp_id, "session_id": session_id, "intent_id": intent_id,
            "statement": statement, "branches": branches, "horizon": horizon,
            "context_ref": context_ref, "recall_refs": recall_refs or [],
            "created_ts": time.time(), "created_step": created_step,
            "status": STATUS_PENDING, "resolution": None,
        }
        self.put("expectation", rec)
        self._graph.insert_sync(f"intent:{intent_id}", "has_expectation", f"expectation:{exp_id}")
        self._graph.insert_sync(f"expectation:{exp_id}", "belongs_to", f"intent:{intent_id}")
        return rec

    def put_outcome(self, outcome_id: str, exp_id: str, outcome_text: str,
                    observed_branch: str, source: str = "env",
                    resolved_step: int = 0) -> dict:
        """Record an observation, resolve its expectation, and stamp congruence."""
        exp = self.get("expectation", exp_id)
        if exp is None:
            raise LedgerError(f"outcome references unknown expectation {exp_id}")
        if exp["status"] == STATUS_RESOLVED:
            raise LedgerError(f"expectation {exp_id} already resolved -- outcomes are final")
        if exp["status"] == STATUS_EXPIRED:
            raise LedgerError(f"expectation {exp_id} expired -- past-horizon outcomes are rejected")
        result = evaluate_congruence(exp["branches"], observed_branch)
        rec = {
            "outcome_id": outcome_id, "exp_id": exp_id, "session_id": exp["session_id"],
            "outcome_text": outcome_text, "observed_branch": observed_branch,
            "source": source, "resolved_ts": time.time(), "resolved_step": resolved_step,
            "congruence": result.to_dict(),
        }
        self.put("outcome", rec)
        exp["status"] = STATUS_RESOLVED
        exp["resolution"] = {
            "outcome_id": outcome_id, "observed_branch": observed_branch,
            "congruence": result.congruence.value, "rule_version": result.rule_version,
            "resolved_ts": rec["resolved_ts"], "resolved_step": resolved_step,
        }
        self.put("expectation", exp)
        self._graph.insert_sync(f"outcome:{outcome_id}", "resolves", f"expectation:{exp_id}")
        self._graph.insert_sync(f"expectation:{exp_id}", "has_congruence", result.congruence.value)
        return rec

    def expire_pending(self, now_ts: float | None = None, current_step: int | None = None) -> int:
        """Mark past-horizon pending expectations expired (NO congruence label).

        Expiry is not a violated expectation -- fabricating a label for "nothing
        was observed" would poison the training signal, so expired records keep
        resolution=None and gate coverage simply excludes them.
        """
        now_ts = now_ts if now_ts is not None else time.time()
        n = 0
        for exp in list(self.records("expectation")):
            if exp["status"] != STATUS_PENDING:
                continue
            hz = exp["horizon"]
            due = (hz["kind"] == "step" and current_step is not None
                   and current_step > hz["deadline_step"])
            due = due or (hz["kind"] == "time" and now_ts > hz["deadline_ts"])
            if not due:
                continue
            exp["status"] = STATUS_EXPIRED
            exp["expired_ts"] = now_ts
            self.put("expectation", exp)
            n += 1
        return n

    # ---- graph helpers (tests + later traversal) ----

    def graph_out(self, vertex_id: str, predicate: str) -> list[str]:
        res = self._graph.query().vertex(vertex_id).out(predicate).execute_sync()
        try:
            return list(res.vertices) if hasattr(res, "vertices") else []
        finally:
            res.close()

    # ---- lifecycle ----

    def close(self) -> None:
        for closer in self._extra_closers:
            closer()
        self._extra_closers.clear()
        self._store.close()
        self._graph.close()
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()