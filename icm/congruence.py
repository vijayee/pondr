"""Deterministic congruence rule for the ICM expectation ledger (Phase 0).

Congruence axis ONLY: did reality match the prediction? Valence (good/bad) is
carried on the OutcomeRecord's observed_branch and deliberately kept out of
the congruence label -- the plan's sec 5.5 distinction (a desired-and-expected
outcome teaches nothing; the unexpected, regardless of valence, is the signal).

Pure function + version tag, so stored labels can be recomputed at any time
(gate metric g3 re-evaluates every resolved expectation and must reproduce
byte-identical labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

RULE_VERSION = "icm-congruence/1"

# The fixed branch vocabulary. "other" is NEVER a predicted branch -- it exists
# only on the observation side and always means "something unpredicted happened".
BRANCHES = ("desired", "feared", "default")


class Congruence(Enum):
    CONFIRMED = "confirmed"
    VIOLATED = "violated"


class CongruenceError(ValueError):
    pass


@dataclass(frozen=True)
class CongruenceResult:
    congruence: Congruence
    predicted_branches: tuple[str, ...]
    observed_branch: str
    rule_version: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "congruence": self.congruence.value,
            "predicted_branches": list(self.predicted_branches),
            "observed_branch": self.observed_branch,
            "rule_version": self.rule_version,
            "reason": self.reason,
        }


def validate_branches(branches: dict) -> tuple[str, ...]:
    """Predicted branch names from a branches dict; requires at least one."""
    if not isinstance(branches, dict):
        raise CongruenceError("branches must be a dict with keys desired/feared/default")
    predicted = tuple(b for b in BRANCHES if branches.get(b) is not None)
    if not predicted:
        raise CongruenceError("expectation must predict at least one branch "
                              "(non-null desired/feared/default)")
    for key in branches:
        if key not in BRANCHES:
            raise CongruenceError(f"unknown branch key {key!r}")
    return predicted


def evaluate_congruence(branches: dict, observed_branch: str) -> CongruenceResult:
    """Classify an observation against the predicted branch set.

    predicted, observed -> verdict:
      "other" observed          -> VIOLATED (unpredicted outcome; max surprise)
      predicted branch observed -> CONFIRMED (matched prediction)
      unpredicted named branch  -> VIOLATED (missed prediction)
    Raises CongruenceError on invalid inputs -- the rule may never silently
    guess, because its output is the training signal.
    """
    predicted = validate_branches(branches)
    if observed_branch == "other":
        return CongruenceResult(Congruence.VIOLATED, predicted, "other", RULE_VERSION,
                                "observed unpredicted branch 'other'")
    if observed_branch not in BRANCHES:
        raise CongruenceError(f"observed branch {observed_branch!r} not in {BRANCHES + ('other',)}")
    if branches.get(observed_branch) is not None:
        return CongruenceResult(Congruence.CONFIRMED, predicted, observed_branch, RULE_VERSION,
                                f"observed predicted branch {observed_branch!r}")
    return CongruenceResult(Congruence.VIOLATED, predicted, observed_branch, RULE_VERSION,
                            f"observed unpredicted branch {observed_branch!r}")