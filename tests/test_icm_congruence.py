"""Congruence rule unit tests (ICM Phase 0)."""

import pytest

from icm.congruence import Congruence, CongruenceError, RULE_VERSION, evaluate_congruence

B = {"desired": "green", "feared": None, "default": "stalls"}


def test_confirm_predicted_branch():
    r = evaluate_congruence(B, "desired")
    assert r.congruence is Congruence.CONFIRMED
    assert r.rule_version == RULE_VERSION
    assert r.predicted_branches == ("desired", "default")


def test_confirm_second_predicted_branch():
    assert evaluate_congruence(B, "default").congruence is Congruence.CONFIRMED


def test_violated_other_is_max_surprise():
    r = evaluate_congruence(B, "other")
    assert r.congruence is Congruence.VIOLATED
    assert r.reason  # every label carries a deterministic reason string


def test_violated_unpredicted_named_branch():
    # predicted desired+default, observed feared -> missed prediction
    r = evaluate_congruence(B, "feared")
    assert r.congruence is Congruence.VIOLATED


def test_no_predicted_branch_rejected():
    with pytest.raises(CongruenceError):
        evaluate_congruence({"desired": None, "feared": None, "default": None}, "desired")


def test_unknown_branch_key_rejected():
    with pytest.raises(CongruenceError):
        evaluate_congruence({"desired": "x", "weird": "y"}, "desired")


def test_invalid_observed_rejected():
    with pytest.raises(CongruenceError):
        evaluate_congruence(B, "excellent")


def test_valence_is_not_in_the_label():
    # a feared outcome that was predicted is CONFIRMED (congruence, not valence)
    b = {"desired": "fine", "feared": "crash", "default": None}
    assert evaluate_congruence(b, "feared").congruence is Congruence.CONFIRMED


def test_determinism_byte_stable():
    a = evaluate_congruence(B, "other").to_dict()
    c = evaluate_congruence(B, "other").to_dict()
    assert a == c