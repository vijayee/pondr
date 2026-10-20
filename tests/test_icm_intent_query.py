"""Tests for the two-stage query path (icm/intent_query.py).

Pre-registered in the module docstring: no score blending, deterministic
ranking, explicit fallbacks, and reproduction of the addendum-23 mechanism
(entity/type filter, then EARLIEST / MIDDLE / LATEST ordering by epoch) -- the
stage-2 order must pick the correct record where dense@1 picks a wrong one.

The world here is a MECHANICS fixture, not a research construction: it exists
to pin the implementation's ordering and fallback behaviour. The claim itself
is still regressed by the filesystem-verified 60-trial harness
(scripts/icm/twobytwo_stage2.py).
"""

from __future__ import annotations

import numpy as np

from icm.intent_keys import KeyResult
from icm.intent_query import TwoStageRetriever, content_tokens, entities_of

VOCAB = ["alpha", "beta", "checksum", "timeout", "first", "latest", "middle",
         "report", "dataset", "load", "2026", "dated", "run", "value"]


def embed(texts):
    rows = []
    for text in texts:
        low = str(text).lower()
        v = np.array([1.0 if w in low else 0.0 for w in VOCAB])
        n = float(np.linalg.norm(v))
        rows.append(v / n if n else v)
    return np.array(rows)


def key(epoch_kind="ordinal", epoch_value="EARLIEST", entity="alpha",
        qualifier=""):
    return {
        "requirements": [{"subject": "SELF", "attitude": "report",
                          "relation": "report", "object": f"checksum of {entity}"}],
        "qualifier": qualifier,
        "declared_qualifier": "",
        "epoch_ref": {"kind": epoch_kind, "value": epoch_value},
        "abstained": False,
    }


def world():
    return [
        # the dense trap: e2 shares the most surface vocabulary with the query
        {"id": "alpha-e2", "type": "checksum", "entity": "alpha", "epoch": "2026-04-15",
         "text": "dataset alpha: checksum 333, dated 2026-04-15, run 14 "
                 "(report of the load)"},
        {"id": "alpha-e1", "type": "checksum", "entity": "alpha", "epoch": "2026-03-20",
         "text": "dataset alpha: checksum 222, dated 2026-03-20, run 13"},
        {"id": "alpha-e0", "type": "checksum", "entity": "alpha", "epoch": "2026-03-01",
         "text": "dataset alpha: checksum 111, dated 2026-03-01, run 12"},
        # decoys: same artifact kind, other entity; and other kind
        {"id": "beta-e0", "type": "checksum", "entity": "beta", "epoch": "2026-03-01",
         "text": "dataset beta: checksum 999, report of the first load, "
                 "dated 2026-03-01, run 12"},
        {"id": "alpha-timeout", "type": "timeout", "entity": "alpha", "epoch": "2026-04-15",
         "text": "dataset alpha: timeout 800, report of the load, dated 2026-04-15"},
    ]


QUERY = "report the checksum for dataset alpha at the first load"
META = {"type": "checksum", "entity": "alpha"}


def test_entities_and_stopwords():
    toks = entities_of(key())
    assert "alpha" in toks
    assert "checksum" in toks          # artifact noun, not generic
    assert not ({"dataset", "report", "of", "the"} & toks)
    assert content_tokens("data/_scratch/alpha.txt") >= {"alpha"}


def test_dense_is_fooled_and_two_stage_picks_the_earliest():
    r = TwoStageRetriever(embed)
    res = r.query(QUERY, world(), query_key=key("ordinal", "EARLIEST"), query_meta=META)
    # the fixture has teeth: dense@1 is the WRONG record
    assert res["stage1_ids"][0] == "alpha-e2"
    assert res["fallback"] is None and res["stage2_applied"] is True
    assert res["top"][0]["id"] == "alpha-e0"
    assert res["epoch_ref"] == {"kind": "ordinal", "value": "EARLIEST"}


def test_latest_and_middle_ordering():
    r = TwoStageRetriever(embed)
    latest = r.query(QUERY, world(), query_key=key("ordinal", "LATEST"), query_meta=META)
    assert latest["top"][0]["id"] == "alpha-e2"
    middle = r.query(QUERY, world(), query_key=key("ordinal", "MIDDLE"), query_meta=META)
    assert middle["top"][0]["id"] == "alpha-e1"


def test_entity_and_type_filters_exclude_decoys():
    r = TwoStageRetriever(embed)
    res = r.query(QUERY, world(), query_key=key(), query_meta=META)
    assert set(res["filtered_ids"]) == {"alpha-e0", "alpha-e1", "alpha-e2"}
    # filtered-out candidates are still ranked (behind), never dropped
    assert {c["id"] for c in res["ranked"]} == {c["id"] for c in world()}


def test_no_key_falls_back_to_dense_and_says_so():
    r = TwoStageRetriever(embed)                      # no keyer, no key
    res = r.query(QUERY, world(), query_meta=META)
    assert res["fallback"] == "no_key" and res["stage2_applied"] is False
    assert [c["id"] for c in res["ranked"]] == res["stage1_ids"]
    assert res["top"][0]["id"] == res["stage1_ids"][0]


def test_empty_filter_falls_back_to_dense_and_says_so():
    r = TwoStageRetriever(embed)
    res = r.query(QUERY, world(), query_key=key(entity="zeta"),
                  query_meta={"type": "checksum"})
    assert res["fallback"] == "empty_filter"
    assert [c["id"] for c in res["ranked"]] == res["stage1_ids"]


def test_no_epoch_reference_keeps_dense_order_within_the_filtered_set():
    r = TwoStageRetriever(embed)
    res = r.query(QUERY, world(), query_key=key("none", ""), query_meta=META)
    assert res["fallback"] == "no_epoch" and res["stage2_applied"] is False
    filtered_in_dense_order = [i for i in res["stage1_ids"]
                               if i in set(res["filtered_ids"])]
    assert [c["id"] for c in res["top"][:3]] == filtered_in_dense_order[:3]


def test_absolute_epoch_reference_orders_by_distance():
    r = TwoStageRetriever(embed)
    res = r.query(QUERY, world(), query_key=key("absolute", "2026-03-20"),
                  query_meta=META)
    assert res["top"][0]["id"] == "alpha-e1"


def test_latest_keeps_missing_epochs_last():
    r = TwoStageRetriever(embed)
    pool = world() + [{"id": "alpha-undated", "type": "checksum",
                       "entity": "alpha",
                       "text": "dataset alpha: checksum 444, report of a load"}]
    res = r.query(QUERY, pool, query_key=key("ordinal", "LATEST"),
                  query_meta=META)
    ids = [c["id"] for c in res["ranked"]]
    assert ids[0] == "alpha-e2"
    # filtered candidates rank first; within them the undated record is last
    n = len(res["filtered_ids"])
    assert ids[n - 1] == "alpha-undated"
    assert set(ids[:n]) == {"alpha-e2", "alpha-e1", "alpha-e0", "alpha-undated"}


def test_absolute_reference_in_another_epoch_domain_is_no_epoch():
    r = TwoStageRetriever(embed)
    numeric = [
        {"id": "n-12", "type": "checksum", "entity": "alpha", "epoch": 12,
         "text": "dataset alpha: checksum 111, run 12"},
        {"id": "n-14", "type": "checksum", "entity": "alpha", "epoch": 14,
         "text": "dataset alpha: checksum 333, run 14"},
    ]
    # the key names a DATE while the pool carries RUN NUMBERS -> ordering those
    # numbers against a timestamp is meaningless, so fall back and say so
    res = r.query(QUERY, numeric, query_key=key("absolute", "2026-03-20"),
                  query_meta=META)
    assert res["fallback"] == "no_epoch"


def test_ignore_key_entities_keeps_only_explicit_entities():
    meta = {"type": "checksum", "ignore_key_entities": True}
    assert entities_of(key(), meta) == set()
    assert entities_of(key(), {**meta, "entity": "alpha"}) == {"alpha"}


def test_pre_change_version_is_not_filtered_out_by_feature_tokens():
    """The correct pre-change version may lack the feature the key names."""
    r = TwoStageRetriever(embed)
    pool = [
        {"id": "v-before", "type": "src/gate.py", "epoch": "2026-01-01",
         "text": "dataset alpha: checksum 111, dated 2026-01-01"},
        {"id": "v-after", "type": "src/gate.py", "epoch": "2026-02-01",
         "text": "dataset alpha: timeout 800, checksum 222, report load"},
    ]
    qkey = {"requirements": [{"subject": "SELF", "attitude": "restore",
                              "relation": "restore", "object": "the timeout 800"}],
            "qualifier": "", "epoch_ref": {"kind": "ordinal", "value": "EARLIEST"},
            "abstained": False}
    res = r.query("restore the first version of src/gate.py",
                  pool, query_key=qkey,
                  query_meta={"type": "src/gate.py", "ignore_key_entities": True})
    assert set(res["filtered_ids"]) == {"v-before", "v-after"}
    assert res["top"][0]["id"] == "v-before"


def test_keyer_is_used_when_no_key_is_supplied():
    class StubKeyer:
        def __init__(self):
            self.calls = []

        def key_for(self, text, declared_qualifier=""):
            self.calls.append((text, declared_qualifier))
            return KeyResult(key(), "judged")

    keyer = StubKeyer()
    r = TwoStageRetriever(embed, keyer=keyer)
    res = r.query(QUERY, world(), query_meta={**META, "qualifier": "first ingest"})
    assert res["key_status"] == "judged"
    assert keyer.calls == [(QUERY, "first ingest")]
    assert res["top"][0]["id"] == "alpha-e0"


def test_ranking_is_deterministic():
    r = TwoStageRetriever(embed)
    a = r.query(QUERY, world(), query_key=key(), query_meta=META)
    b = r.query(QUERY, world(), query_key=key(), query_meta=META)
    assert [c["id"] for c in a["ranked"]] == [c["id"] for c in b["ranked"]]
    assert a["stage1_ids"] == b["stage1_ids"]
