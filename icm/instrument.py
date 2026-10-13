"""Instrumentation harness for Phase 0 -- session wrapper + adapter contract.

The harness wraps an agent run with a small, explicit record layer:

    ICMSession(ledger, session_id)
        .begin(meta)                  -> SessionRecord
        .declare_intent(aim)          -> IntentRecord (durable intent node)
        .step()                       -> advance the logical step counter
        .expect(intent_id, ...)       -> ExpectationRecord (prospective memory entry)
        .resolve(exp_id, ...)         -> OutcomeRecord + congruence stamp
        .expire_pending()             -> mark past-horizon pendings (no label)
        .finish()                     -> close intents, flush

Adapter contract for REAL agents (to be wired later, unchanged by Phase 0):
    An ExpectationPolicy receives the agent event stream (dicts: tool_call,
    tool_result, assistant_message, ...) and proposes ExpectationDecl /
    OutcomeDecl. Phase 0's synthetic pilot implements the policy deterministically;
    the eventual Bonsai/serve integration just supplies another policy object --
    the harvest path (ledger + rule + gate) is policy-agnostic by construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ExpectationDecl:
    intent_id: str
    statement: str
    branches: dict
    horizon: dict
    recall_refs: list[str] = field(default_factory=list)


@dataclass
class OutcomeDecl:
    exp_id: str
    outcome_text: str
    observed_branch: str
    source: str = "env"


class ICMSession:
    """Writes session-scoped records into an ICMLedger with a logical clock."""

    def __init__(self, ledger, session_id: str):
        self.ledger = ledger
        self.session_id = session_id
        self._step = 0
        self._open_intents: list[str] = []
        self._begun = False

    def begin(self, meta: dict | None = None) -> dict:
        rec = self.ledger.put_session(self.session_id, meta)
        self._begun = True
        return rec

    def step(self) -> int:
        """Advance the logical clock (one step = one observable event)."""
        if not self._begun:
            raise RuntimeError("session not begun")
        self._step += 1
        return self._step

    def declare_intent(self, aim: str, intent_id: str | None = None) -> str:
        intent_id = intent_id or self.ledger.new_id("int")
        self.ledger.put_intent(intent_id, self.session_id, aim, created_step=self._step)
        self._open_intents.append(intent_id)
        return intent_id

    def expect(self, decl: ExpectationDecl, exp_id: str | None = None) -> str:
        exp_id = exp_id or self.ledger.new_id("exp")
        self.ledger.put_expectation(
            exp_id, self.session_id, decl.intent_id, decl.statement,
            decl.branches, decl.horizon,
            context_ref={"step": self._step, "digest": f"{self.session_id}@{self._step}"},
            recall_refs=decl.recall_refs, created_step=self._step,
        )
        return exp_id

    def resolve(self, decl: OutcomeDecl, outcome_id: str | None = None) -> str:
        outcome_id = outcome_id or self.ledger.new_id("out")
        self.ledger.put_outcome(outcome_id, decl.exp_id, decl.outcome_text,
                                decl.observed_branch, decl.source,
                                resolved_step=self._step)
        return outcome_id

    def expire_pending(self, now_ts: float | None = None) -> int:
        return self.ledger.expire_pending(now_ts=now_ts, current_step=self._step)

    def finish(self, close_status: str = "done") -> None:
        for intent_id in self._open_intents:
            rec = self.ledger.get("intent", intent_id)
            if rec and rec.get("status") == "active":
                rec["status"] = close_status
                self.ledger.put("intent", rec)


def events_to_outcomes(events: list[dict]) -> list[OutcomeDecl]:
    """Reference extraction for the simplest real-agent shape: a tool_result
    event that names an expectation to resolve.

    Expected event dict:
        {"type": "tool_result", "exp_id": ..., "observed_branch": ...,
         "outcome_text": ...}
    Returns only well-formed resolutions; malformed events are dropped (Phase 0
    harvesters must never fabricate, but partial logging beats crashing a run).
    """
    out = []
    for e in events:
        if not isinstance(e, dict) or e.get("type") != "tool_result":
            continue
        if not e.get("exp_id") or not e.get("observed_branch"):
            continue
        out.append(OutcomeDecl(str(e["exp_id"]), str(e.get("outcome_text", "")),
                               str(e["observed_branch"]), "tool_result"))
    return out