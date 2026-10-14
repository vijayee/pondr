"""ICM -- Intent-Conditioned Memory research experiment (separate from the
Ponder Engine's fade architecture; shares only the WaveDB substrate).

Phase 0 deliverables live here:
    congruence  -- deterministic congruence rule (the training signal)
    ledger      -- expectation/outcome/intent records on WaveDB KV + graph
    instrument  -- session harness + agent-adapter contract
    pilot       -- deterministic synthetic harvest pilot
    gate        -- Phase-0 GO/NO-GO metrics (coverage/non-vacuity/determinism/recovery)
"""

from icm.congruence import (  # noqa: F401
    BRANCHES,
    RULE_VERSION,
    Congruence,
    CongruenceError,
    CongruenceResult,
    evaluate_congruence,
)
from icm.ledger import ICMLedger, LedgerError  # noqa: F401
from icm.instrument import ExpectationDecl, ICMSession, OutcomeDecl, events_to_outcomes  # noqa: F401
from icm.gate import compute_gate_metrics  # noqa: F401
from icm.pilot import run_pilot  # noqa: F401
from icm.episodic import EpisodicStore, FORGET_RULE_VERSION, gist_of  # noqa: F401

__all__ = [
    "BRANCHES", "RULE_VERSION", "Congruence", "CongruenceError", "CongruenceResult",
    "evaluate_congruence", "ICMLedger", "LedgerError", "ExpectationDecl", "ICMSession",
    "OutcomeDecl", "events_to_outcomes", "compute_gate_metrics", "run_pilot",
    "EpisodicStore", "FORGET_RULE_VERSION", "gist_of",
]