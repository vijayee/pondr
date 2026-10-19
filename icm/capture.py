"""Hierarchical intent capture -- the agent-intent corpus's recording layer.

Built per the agreed protocol (2026-10-15, addendum 7) and extended to the
six-field intent record in docs/ICM_intent_store_spec.md sec 2 (2026-10-18):

  HIERARCHY:  goal (root intent, structured S/P-O-lite) -> steps (ONE LINE:
              purpose + expected branch; retrieval-bearing steps only)
              -> retrieval events (query + retrieved ids) -> outcomes.
  INTENT:     every goal/step carries the intent's QUALIFIER (state/version
              marker), its REQUIREMENTS (a LIST of S-P-O triples -- measured
              4.63 per real issue; fewer is legitimate and never padded), and
              an EPOCH (declared run number/timestamp, falling back to capture
              time). A non-reach (violated) outcome can trigger a NEW intent
              (triggered_by), which records the expansion loop.
  KEYS:       when an IntentKeyer is supplied, the judge's write-time key is
              stored beside the intent (intent_key + key_status) and copied
              onto each retrieval event, so a memory unit carries the intent
              key under which it was written. The judge never blocks a write.
  THE LAW:    labels are ALWAYS computed by the congruence rule from declared
              branches + the observed branch.  A capture caller may never
              state a label; capture.observe() computes and stores it.
  HONESTY:    no fabrication -- if the outcome is genuinely in-between the
              declared branches, observe accepts "other" (the rule's max-
              surprise label) rather than forcing a branch. Requirements are
              recorded as declared; the capture layer never synthesises extra
              triples to reach the expected 3-5.
  BACKING:    records persist through the ICMLedger (WaveDB) as kind
              "icapture" payloads, plus a canonical JSONL mirror for the
              ablation harness.

Usage:
    cap = IntentCapture(ledger)
    g = cap.goal("measure the reversal tail", desired=">=20 verified",
                 qualifier="latest run", requirements=[...], epoch=14)
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

SCHEMA_VERSION = 2
REQUIREMENT_FIELDS = ("subject", "attitude", "relation", "object", "qualifier")


def normalize_requirements(requirements) -> list[dict]:
    """Validate/normalise the intent's requirement list.

    A dict is accepted as a single requirement (for callers migrating from the
    one-triple form) and is always stored as a LIST. Each item must state a
    relation and an object; subject defaults to SELF per the S-P-O schema.
    Counts are recorded, never padded -- a one-requirement intent stays one.
    """
    if requirements is None:
        return []
    if isinstance(requirements, dict):
        requirements = [requirements]
    if not isinstance(requirements, (list, tuple)):
        raise ValueError("requirements must be a list of S-P-O triples")
    out = []
    for i, item in enumerate(requirements):
        if not isinstance(item, dict):
            raise ValueError(f"requirement {i} is not a dict")
        relation = str(item.get("relation") or "").strip()
        obj = str(item.get("object") or "").strip()
        if not relation or not obj:
            raise ValueError(f"requirement {i} needs non-empty relation and object")
        out.append({
            "subject": str(item.get("subject") or "SELF").strip() or "SELF",
            "attitude": str(item.get("attitude") or "").strip(),
            "relation": relation,
            "object": obj,
            "qualifier": str(item.get("qualifier") or "").strip(),
        })
    return out


def _normalize_epoch(epoch):
    """Epoch = declared run number or timestamp. None means capture time."""
    if epoch is None:
        return None
    if isinstance(epoch, bool):
        raise ValueError("epoch must be a run number, timestamp or label, not bool")
    if isinstance(epoch, (dict, list, tuple, set)):
        raise ValueError("epoch must be a scalar run number, timestamp or label")
    if isinstance(epoch, (int, float)):
        return epoch
    text = str(epoch).strip()
    if not text:
        raise ValueError("epoch must be non-empty when declared")
    return text


class IntentCapture:
    """Hierarchical capture over an ICMLedger. Labels computed by the rule."""

    def __init__(self, ledger, mirror_dir=None, keyer=None):
        self.ledger = ledger
        self.mirror_dir = mirror_dir or os.path.join(
            "data", "_scratch", "icm", "capture")
        # keyer: optional icm.intent_keys.IntentKeyer. Duck-typed so callers
        # can inject a stub in tests; None disables write-time keying.
        self.keyer = keyer
        self._goals = {}
        self._steps = {}
        self._events = {}
        self._outcomes = {}
        self._lines = []
        self._closed = False
        self._key_errors = 0

    # -- intent fields ---------------------------------------------------
    def _key_fields(self, text: str, qualifier: str | None) -> dict:
        """Write-time judge keying. A judge failure NEVER blocks the write."""
        if self.keyer is None:
            return {"intent_key": None, "key_status": "disabled",
                    "key_abstained": None}
        try:
            res = self.keyer.key_for(text, qualifier or "")
        except Exception:
            # transport/parse failures are counted and surfaced, not hidden;
            # capture continues so the ledger keeps the declared intent.
            self._key_errors += 1
            return {"intent_key": None, "key_status": "error",
                    "key_abstained": None}
        if res.key is None:
            if res.status == "error":
                self._key_errors += 1
            return {"intent_key": None, "key_status": res.status,
                    "key_abstained": None}
        return {"intent_key": res.key, "key_status": res.status,
                "key_abstained": bool(res.key.get("abstained"))}

    def _expand_from(self, triggered_by: str | None) -> str | None:
        """Validate an expansion link: only a NON-REACH event starts a new intent.

        The causal direction is user-set (memory: intent is prior to outcomes).
        Non-reach generates the next intent; a confirmed outcome does not.
        """
        if triggered_by is None:
            return None
        rec = self._outcomes.get(triggered_by) or self.ledger.get(
            "icapture", triggered_by)
        if rec is None:
            raise ValueError(f"triggered_by {triggered_by!r} is unknown")
        if rec.get("level") != "outcome":
            raise ValueError("triggered_by must reference an outcome record")
        if rec.get("label") != "violated":
            raise ValueError("expansion is only recorded after a NON-REACH "
                             f"event; {triggered_by!r} has label "
                             f"{rec.get('label')!r}")
        return triggered_by

    def _intent_record(self, level: str, parent, rid: str, text: str, br: dict,
                       qualifier, requirements, epoch, triggered_by,
                       now: float, prediction: str = "branches") -> dict:
        reqs = normalize_requirements(requirements)
        ep = _normalize_epoch(epoch)
        declared_q = str(qualifier or "").strip()
        return {
            "kind": "icapture", "level": level, "parent": parent, "id": rid,
            "text": text, "branches": br, "declared_at": now,
            "schema_version": SCHEMA_VERSION,
            "prediction": prediction,
            "qualifier": declared_q,
            "requirements": reqs,
            "requirement_count": len(reqs),
            "epoch": ep if ep is not None else now,
            "epoch_source": "declared" if ep is not None else "capture-time",
            "triggered_by": self._expand_from(triggered_by),
            **self._key_fields(text, declared_q),
        }

    # -- goals ---------------------------------------------------------
    def goal(self, text: str, desired: str | None = None,
             feared: str | None = None, default: str | None = None,
             qualifier: str | None = None, requirements=None,
             epoch=None, triggered_by: str | None = None) -> str:
        gid = f"goal-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        br = {"desired": desired, "feared": feared, "default": default}
        if not any(br.get(b) is not None for b in BRANCHES):
            raise ValueError("a goal predicts at least one branch (declare "
                             "desired/feared/default -- never a bare goal)")
        now = time.time()
        rec = self._intent_record("goal", None, gid, text, br, qualifier,
                                  requirements, epoch, triggered_by, now)
        self._store(gid, rec)
        self._goals[gid] = rec
        return gid

    # -- declared intents without a prediction (live-session capture) --------
    def intent(self, text: str, level: str = "goal", parent: str | None = None,
               qualifier: str | None = None, requirements=None,
               epoch=None, triggered_by: str | None = None) -> str:
        """Declare an intent that predicts NO branch.

        Live work often declares what it is aiming at without predicting an
        outcome. Such an intent can never produce a congruence label -- the
        outcome() rule refuses it by construction -- so recording it does not
        weaken the label law; it keeps the declaration and its write-time key
        without fabricating a prediction. Use goal()/step() when a prediction
        exists.
        """
        br = {"desired": None, "feared": None, "default": None}
        now = time.time()
        if level == "goal":
            if parent is not None:
                raise ValueError("a goal-level intent has no parent")
            rid = f"goal-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            rec = self._intent_record("goal", None, rid, text, br, qualifier,
                                      requirements, epoch, triggered_by, now,
                                      prediction="none")
            self._store(rid, rec)
            self._goals[rid] = rec
            return rid
        if level == "step":
            if parent not in self._goals:
                raise ValueError(f"step parent {parent!r} is not an open goal")
            rid = (f"{parent}-s{len([k for k in self._steps if k.startswith(parent)]) + 1:02d}")
            rec = self._intent_record("step", parent, rid, text, br, qualifier,
                                      requirements, epoch, triggered_by, now,
                                      prediction="none")
            self._store(rid, rec)
            self._steps[rid] = rec
            return rid
        raise ValueError("level must be 'goal' or 'step'")

    # -- steps (one-liners; parent = a goal) --------------------------------
    def step(self, parent: str, line: str, desired: str | None = None,
             feared: str | None = None, default: str | None = None,
             qualifier: str | None = None, requirements=None,
             epoch=None, triggered_by: str | None = None) -> str:
        if parent not in self._goals:
            raise ValueError(f"step parent {parent!r} is not an open goal")
        sid = f"{parent}-s{len([k for k in self._steps if k.startswith(parent)]) + 1:02d}"
        br = {"desired": desired, "feared": feared, "default": default}
        now = time.time()
        # steps may omit branches entirely (branchless steps still link the
        # hierarchy; their utility is measured against the PARENT's branches)
        rec = self._intent_record("step", parent, sid, line, br, qualifier,
                                  requirements, epoch, triggered_by, now)
        self._store(sid, rec)
        self._steps[sid] = rec
        return sid

    # -- retrieval events ----------------------------------------------------
    def retrieval(self, step: str, query: str, retrieved: list[str],
                  k: int | None = None) -> str:
        if step not in self._steps and step not in self._goals:
            raise ValueError(f"retrieval step {step!r} not open")
        parent = self._steps.get(step) or self._goals.get(step)
        # operative intent = the step's, falling back to its goal when the
        # step carries no key (branchless/disabled keying still links memory)
        op = parent
        if parent.get("level") == "step" and parent.get("intent_key") is None:
            op = self._goals.get(parent.get("parent"), parent)
        eid = f"retr-{len(self._events) + 1:04d}-" + uuid.uuid4().hex[:6]
        rec = {"kind": "icapture", "level": "retrieval", "parent": step,
               "id": eid, "query": query, "retrieved": list(retrieved),
               "memory_units": list(retrieved), "k": k,
               "intent_id": op["id"],
               "intent_key": op.get("intent_key"),
               "key_status": op.get("key_status"),
               "epoch": op.get("epoch"),
               "qualifier": op.get("qualifier"),
               "declared_at": time.time()}
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
        # uuid suffix: two captures against one ledger must not overwrite each
        # other's outcomes (the old len()-only id collided across sessions)
        oid = f"outc-{len(self._events):04d}-" + uuid.uuid4().hex[:6]
        rec = {"kind": "icapture", "level": "outcome", "parent": retrieval_id,
               "id": oid, "observed": observed_branch,
               "label": result.congruence.value,        # computed
               "rule_version": result.rule_version,
               "reason": result.reason, "note": note,
               "declared_at": time.time()}
        self._store(oid, rec)
        self._outcomes[oid] = rec
        return oid

    # -- accounting ------------------------------------------------------------
    def key_summary(self) -> dict:
        """Keying status per intent; `errors` counts failed judge calls."""
        counts: dict[str, int] = {}
        for rec in list(self._goals.values()) + list(self._steps.values()):
            status = rec.get("key_status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {"counts": counts, "errors": self._key_errors}

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
