"""Tests for write-time judge keying (icm/intent_keys.py).

Pins the pre-registered behaviour: one call per unique intent (cache by text
hash + declared qualifier), a requirement LIST and an epoch reference in every
key, four explicit statuses (judged/cached/abstained/error), no placeholder key
on failure, and pairwise opposition only when explicitly requested.
"""

from __future__ import annotations

import json

from icm.intent_keys import IntentKeyer, parse_key

GOOD = {
    "requirements": [
        {"subject": "SELF", "attitude": "report", "relation": "report",
         "object": "checksum of dataset alpha"},
        {"subject": "index", "attitude": "verify", "relation": "match",
         "object": "row count"},
        {"subject": "log", "attitude": "avoid", "relation": "overwrite",
         "object": "the archived snapshot"},
    ],
    "qualifier": "",
    "epoch_ref": {"kind": "ordinal", "value": "first"},
}


class StubTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


def test_key_for_returns_requirement_list_and_epoch_ref(tmp_path):
    transport = StubTransport([json.dumps(GOOD)])
    keyer = IntentKeyer(cache_path=str(tmp_path / "cache.jsonl"), ask=transport)
    res = keyer.key_for("Report the checksum for dataset alpha at the first load.")
    assert res.status == "judged" and res.ok
    key = res.key
    assert key["requirement_count"] == 3
    assert key["epoch_ref"] == {"kind": "ordinal", "value": "EARLIEST"}
    assert key["declared_qualifier"] == ""
    assert len(transport.prompts) == 1


def test_cache_hit_costs_no_call_and_survives_a_new_instance(tmp_path):
    cache = str(tmp_path / "cache.jsonl")
    t1 = StubTransport([json.dumps(GOOD)])
    k1 = IntentKeyer(cache_path=cache, ask=t1)
    assert k1.key_for("report the checksum").status == "judged"
    # same text, different whitespace -> same normalised hash -> cache hit
    assert k1.key_for("report   the\nchecksum").status == "cached"
    assert len(t1.prompts) == 1
    t2 = StubTransport([])
    k2 = IntentKeyer(cache_path=cache, ask=t2)
    res = k2.key_for("report the checksum")
    assert res.status == "cached" and res.key["requirement_count"] == 3
    assert t2.prompts == []


def test_declared_qualifier_partitions_the_cache(tmp_path):
    t = StubTransport([json.dumps(GOOD), json.dumps(GOOD)])
    k = IntentKeyer(cache_path=str(tmp_path / "cache.jsonl"), ask=t)
    assert k.key_for("report it", "first ingest").status == "judged"
    assert k.key_for("report it", "later ingest").status == "judged"
    assert len(t.prompts) == 2
    assert k.key_for("report it", "first ingest").status == "cached"
    assert len(t.prompts) == 2


def test_empty_response_is_error_with_no_placeholder_key(tmp_path):
    k = IntentKeyer(cache_path=str(tmp_path / "cache.jsonl"),
                    ask=StubTransport([""]))
    res = k.key_for("anything")
    assert res.status == "error" and res.key is None
    assert not res.ok


def test_unparseable_response_is_error(tmp_path):
    k = IntentKeyer(cache_path=str(tmp_path / "cache.jsonl"),
                    ask=StubTransport(["I cannot help with that."]))
    res = k.key_for("anything")
    assert res.status == "error" and res.key is None


def test_abstained_is_a_key_with_nothing_in_it(tmp_path):
    abstain = {"requirements": [], "qualifier": "",
               "epoch_ref": {"kind": "none", "value": ""}}
    k = IntentKeyer(cache_path=str(tmp_path / "cache.jsonl"),
                    ask=StubTransport([json.dumps(abstain)]))
    res = k.key_for("do the thing")
    assert res.status == "abstained"
    assert res.key is not None and res.key["abstained"] is True


def test_parse_key_tolerates_prose_and_partial_items():
    raw = ("Here is the key:\n" + json.dumps({
        "requirements": [
            {"relation": "report", "object": "value"},
            {"relation": "", "object": "dropped"},
            {"relation": "verify", "object": "count"},
        ],
        "epoch_ref": "latest",
    }) + "\nDone.")
    key = parse_key(raw, "stub", "h")
    assert [r["object"] for r in key["requirements"]] == ["value", "count"]
    assert key["requirements"][0]["subject"] == "SELF"
    assert key["epoch_ref"]["value"] == "LATEST"


def test_cache_is_partitioned_by_model(tmp_path):
    """A model switch must not serve the previous model's cached keys."""
    cache = str(tmp_path / "cache.jsonl")
    t1 = StubTransport([json.dumps(GOOD)])
    k1 = IntentKeyer(cache_path=cache, ask=t1, model="judge-a")
    assert k1.key_for("report it").status == "judged"
    t2 = StubTransport([json.dumps(GOOD)])
    k2 = IntentKeyer(cache_path=cache, ask=t2, model="judge-b")
    assert k2.key_for("report it").status == "judged"
    assert len(t2.prompts) == 1
    assert k2.key_for("report it").status == "cached"


def test_relation_between_is_explicit_and_cached(tmp_path):
    t = StubTransport(["OPPOSED", "SAME"])
    k = IntentKeyer(cache_path=str(tmp_path / "cache.jsonl"), ask=t)
    assert k.relation_between("report the archived sha",
                             "report the current sha") == "opposed"
    assert k.relation_between("report the archived sha",
                             "report the current sha") == "opposed"  # cached
    assert len(t.prompts) == 1
    assert k.relation_between("a different", "pair") == "same"
    assert len(t.prompts) == 2


def test_relation_cache_is_namespaced_away_from_keys(tmp_path):
    """A cached relation string must never be returned as an intent key."""
    cache = str(tmp_path / "cache.jsonl")
    k = IntentKeyer(cache_path=cache,
                    ask=StubTransport(["OPPOSED", json.dumps(GOOD)]))
    assert k.relation_between("x intent", "y intent") == "opposed"
    res = k.key_for("x intent")
    assert res.status == "judged" and isinstance(res.key, dict)
