"""Smoke test for the Common Sense Resolver gate input builder.

The 4th gate (Phase 4d) was added to ``scripts/generate_gate_training_data.py``.
The other three input builders (uncertainty / aspirational / self-model) are
exercised only by the deferred validate-slice Oracle run; this test covers the
NEW ``_build_common_sense_inputs`` path offline so a regression in its branch
logic, polyseme selection, or candidate JSON serialization is caught without
needing a store + Oracle.

It monkeypatches the two store-dependent helpers (``_all_episodes`` +
``_get_related_episodes``) with canned data, then asserts:
  * every produced input has exactly the 3 slots the CSR prompt signature takes
    (``input_text`` / ``retrieved_context`` / ``candidate_interpretations``);
  * ``candidate_interpretations`` is a JSON-encoded list (parses);
  * across enough samples, all three branches fire (not-ambiguous yields 1
    candidate; context-resolves yields the full set; genuinely-ambiguous yields
    2 candidates + stripped context) -- so the Oracle gets a balanced mix.
"""

from __future__ import annotations

import json
import random

import scripts.generate_gate_training_data as gen


def _fake_episodes(n: int) -> list[dict]:
    return [
        {"episode_id": f"ep_{i:03d}", "summary": f"episode {i} summary",
         "entities": ["Alice"], "topics": ["Storage"]}
        for i in range(n)
    ]


def _patch_builders(monkeypatch, episodes):
    monkeypatch.setattr(gen, "_all_episodes", lambda store, trav, c, rng: episodes[:c])
    monkeypatch.setattr(
        gen, "_get_related_episodes",
        lambda trav, eid, ents, limit=5: [{"id": "ep_rel", "summary": "related"}],
    )


def test_build_common_sense_inputs_shape(monkeypatch):
    _patch_builders(monkeypatch, _fake_episodes(20))
    rng = random.Random(0)
    inputs = gen._build_common_sense_inputs(store=None, traversal=None, count=20, rng=rng)
    assert len(inputs) == 20
    for inp in inputs:
        assert set(inp) == {"input_text", "retrieved_context", "candidate_interpretations"}
        cands = json.loads(inp["candidate_interpretations"])
        assert isinstance(cands, list) and cands, "candidates must be a non-empty JSON list"
        assert inp["input_text"]
        assert inp["retrieved_context"]


def test_build_common_sense_inputs_branches_all_fire(monkeypatch):
    """Across enough samples, all 3 CSR branches (1 / 2+ / 2 candidates w/
    stripped context) are represented so the Oracle labels a balanced mix."""
    _patch_builders(monkeypatch, _fake_episodes(200))
    rng = random.Random(42)
    inputs = gen._build_common_sense_inputs(store=None, traversal=None, count=200, rng=rng)

    n_single, n_full, n_stripped = 0, 0, 0
    for inp in inputs:
        cands = json.loads(inp["candidate_interpretations"])
        if len(cands) == 1:
            n_single += 1
        elif inp["retrieved_context"] == "(no related context)":
            n_stripped += 1
        else:
            n_full += 1
    # Each branch fires at least once (the rolls are 0.4 / 0.4 / 0.2 weighted;
    # 200 samples makes a zero-count a real bug, not RNG).
    assert n_single > 0, "NOT AMBIGUOUS branch (1 candidate) never fired"
    assert n_full > 0, "CONTEXT RESOLVES branch (full candidate set) never fired"
    assert n_stripped > 0, "GENUINELY AMBIGUOUS branch (stripped context) never fired"


def test_build_common_sense_inputs_genuinely_ambiguous_has_two_candidates(monkeypatch):
    """The genuinely-ambiguous branch always yields exactly 2 candidates."""
    _patch_builders(monkeypatch, _fake_episodes(200))
    rng = random.Random(7)
    inputs = gen._build_common_sense_inputs(store=None, traversal=None, count=200, rng=rng)
    stripped = [inp for inp in inputs if inp["retrieved_context"] == "(no related context)"]
    assert stripped, "no stripped-context inputs -- branch never fired"
    for inp in stripped:
        cands = json.loads(inp["candidate_interpretations"])
        assert len(cands) == 2, "genuinely-ambiguous branch must yield exactly 2 candidates"