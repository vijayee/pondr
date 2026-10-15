"""Hierarchical intent capture -- the agent-intent corpus's recording layer.

Built per the agreed protocol (2026-10-15, addendum 7):

  HIERARCHY:  goal (root intent, structured S/P-O-lite) -> steps (ONE LINE:
              purpose + expected branch; retrieval-bearing steps only)
              -> retrieval events (query + retrieved ids) -> outcomes.
  THE LAW:    labels are ALWAYS computed by the congruence rule from declared
              branches + the observed branch.  A capture caller may never
              state a label; capture.observe() computes and stores it.
  HONESTY:    no fabrication -- if the outcome is genuinely in-between the
              declared branches, observe accepts "other" (the rule's max-
              surprise label) rather than forcing a branch.
  BACKING:    records persist through the ICMLedger (WaveDB) as kind
              "icapture" payloads, plus a canonical JSONL mirror for the
              ablation harness.

Usage:
    cap = IntentCapture.open(ledger)
    g = cap.goal("measure the reversal tail", desired=">=20 verified", ...)
    s = cap.step("run the screen", desired="...", feared=None, ...)
    e = cap.retrieval("verify q400", retrieved=[...])
    cap.outcome(e, "desired")          # label computed here
    cap.close()                        # emits the JSONL mirror
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid

from icm.congruence import evaluate_congruence, CongruenceError, BRANCHES, Congruence  # noqa


class IntentCapture:
    """Hierarchical capture over an ICMLedger. Labels computed by the rule."""

    def __init__(self, ledger, mirror_dir=None):
        self.ledger = ledger
        self.mirror_dir = mirror_dir or os.path.join(
            "data", "_scratch", "icm", "capture")
        self._goals = {}
        self._steps = {}
        self._events = {}
        self._lines = []
        self._closed = False

    # -- goals ---------------------------------------------------------
    def goal(self, text: str, desired: str | None = None,
             feared: str | None = None, default: str | None = None) -> str:
        gid = f"goal-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        br = {"desired": desired, "feared": feared, "default": default}
        if not any(br.get(b) is not None for b in BRANCHES):
            raise ValueError("a goal predicts at least one branch (declare "
                             "desired/feared/default -- never a bare goal)")
        rec = {"kind": "icapture", "level": "goal", "parent": None,
               "id": gid, "text": text, "branches": br,
               "declared_at": time.time()}
        self._store(gid, rec)
        self._goals[gid] = rec
        return gid

    # -- steps (one-liners; parent = a goal) --------------------------------
    def step(self, parent: str, line: str, desired: str | None = None,
             feared: str | None = None, default: str | None = None) -> str:
        if parent not in self._goals:
            raise ValueError(f"step parent {parent!r} is not an open goal")
        sid = f"{parent}-s{len([k for k in self._steps if k.startswith(parent)]) + 1:02d}"
        br = {"desired": desired, "feared": feared, "default": default}
        # steps may omit branches entirely (branchless steps still link the
        # hierarchy; their utility is measured against the PARENT's branches)
        rec = {"kind": "icapture", "level": "step", "parent": parent,
               "id": sid, "text": line, "branches": br,
               "declared_at": time.time()}
        self._store(sid, rec)
        self._steps[sid] = rec
        return sid

    # -- retrieval events ----------------------------------------------------
    def retrieval(self, step: str, query: str, retrieved: list[str],
                  k: int | None = None) -> str:
        if step not in self._steps and step not in self._goals:
            raise ValueError(f"retrieval step {step!r} not open")
        eid = f"retr-{len(self._events) + 1:04d}-" + uuid.uuid4().hex[:6]
        rec = {"kind": "icapture", "level": "retrieval", "parent": step,
               "id": eid, "query": query, "retrieved": list(retrieved),
               "k": k, "declared_at": time.time()}
        self._store(eid, rec)
        self._events[eid] = rec
        return eid

    # -- outcomes: the rule computes the label; nobody states one -------------
    def outcome(self, retrieval_id: str, observed_branch: str,
                note: str = "") -> str:
        ev = self._events.get(retrieval_id)
        if ev is None:
            raise ValueError(f"unknown retrieval {retrieval_id!r}")
        # the operative prediction = the STEP's branches, falling back to the
        # parent goal's (branchless steps inherit the goal's prediction)
        step = self._steps.get(ev["parent"]) or self._goals.get(ev["parent"])
        br = step["branches"]
        if not any(br.get(b) is not None for b in BRANCHES):
            br = self._goals.get(step["parent"], step)["branches"]
        if not any(br.get(b) is not None for b in BRANCHES):
            raise ValueError("outcome has no declared prediction to check "
                             "against -- declare branches on the step or goal")
        result = evaluate_congruence(br, observed_branch)  # NEVER a stated label
        oid = f"outc-{len(self._events):04d}"
        rec = {"kind": "icapture", "level": "outcome", "parent": retrieval_id,
               "id": oid, "observed": observed_branch,
               "label": result.congruence.value,        # computed
               "rule_version": result.rule_version,
               "reason": result.reason, "note": note,
               "declared_at": time.time()}
        self._store(oid, rec)
        return oid

    # -- persistence -----------------------------------------------------------
    def _store(self, rid: str, rec: dict):
        self.ledger.put("icapture", rec)
        self._lines.append(rec)

    def close(self) -> str:
        if self._closed:
            return self.mirror_path()
        os.makedirs(self.mirror_dir, exist_ok=True)
        path = self.mirror_path()
        with io.open(path, "a", encoding="utf-8", newline="\n") as f:
            for rec in self._lines:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._closed = True
        return path

    def mirror_path(self) -> str:
        return os.path.join(self.mirror_dir, "session_capture.jsonl")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()