"""Tests for the R8 Jev decision-model backend (``src/gnn/jev_decider.py``).

Offline throughout: HTTP is monkeypatched at the module's ``requests`` seam
(the same plain-``requests`` pattern ``BonsaiDecider`` uses), so every test
runs with no Ollama and no Bonsai server. The contracts pinned:

- Wire shape: POST ``{jev_endpoint}/v1/systemone`` with
  ``{model, state, questions}``; ``answers`` parsed; Jev's
  ``{input_tokens, output_tokens}`` usage MAPPED onto the OpenAI-style keys
  the ``@record_llm_call`` decorator reads.
- Cold-start: any HTTP / parse / malformed-answer failure -> ``None`` (or a
  dropped verdict) -- the same defer contract as a down Bonsai.
- ``decide_contradiction``'s deterministic pre-filter short-circuits BEFORE
  any HTTP (byte-identical guards, zero Jev calls on non-conflicts).
- Chunking: ``judge_dedup_pairs`` chunks at 64 questions; ANY failed chunk
  fails the WHOLE call (all-or-nothing defer, never partial application).
- ``make_decider`` factory: ``"bonsai"`` -> plain ``BonsaiDecider``
  (byte-identical default); a Jev model name -> ``JevDecider`` (a subclass,
  so every ``BonsaiDecider`` DI seam holds).

One LIVE test is gated on ``DECISION_BACKEND`` env (the operator pulled a Jev
model) -- with the default env it skips, mirroring the Bonsai live-gate
pattern.
"""

from __future__ import annotations

import json
import os

import pytest
import requests as _real_requests

from src.gnn.bonsai_decider import BonsaiDecider
from src.gnn.jev_decider import JevDecider, make_decider


# ── HTTP mock harness ──────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _make_dec(backend="nimble", **kw) -> JevDecider:
    kw.setdefault("jev_endpoint", "http://ollama.test:11434")
    return JevDecider(jev_model=backend, **kw)


def _choice_ans(opt, p):
    return {"type": "choice", "choice": opt,
            "probabilities": {opt: p, "other": 1.0 - p}}


# ── judge_dedup_pairs ──────────────────────────────────────────────────────

def _cands(n):
    return [{"eid": f"ep_{i:03d}", "summary": f"cand {i}",
             "entities": [], "topics": []} for i in range(n)]


# The adapter POSTs a pre-serialized body (``data=``) rather than ``json=``:
# its cold-start contract requires a JSON-dump failure to return None instead
# of raising, which ``requests``'s own json= path cannot do.
def _fake_post_capturer(seen, body):
    def fake_post(url, data=None, timeout=None, headers=None, **kw):
        seen["url"] = url
        seen["payload"] = json.loads(data)
        return _Resp(body=body)
    return fake_post


def test_judge_dedup_pairs_shapes_and_eids(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.gnn.jev_decider.requests.post", _fake_post_capturer(seen, {
        "answers": {
            "pair_0": _choice_ans("update", 0.9),
            "pair_1": _choice_ans("store", 0.8),
        }}))
    dec = _make_dec()
    out = dec.judge_dedup_pairs("new ep summary", ["alice"], ["t"],
                                _cands(2))
    assert seen["url"] == "http://ollama.test:11434/v1/systemone"
    payload = seen["payload"]
    assert payload["model"] == "nimble"
    assert payload["state"]["new_episode"]["summary"] == "new ep summary"
    assert payload["questions"]["pair_1"]["type"] == "choice"
    assert set(payload["questions"]["pair_0"]["criteria"]) == \
        {"store", "update", "merge", "skip"}
    assert out == [
        {"eid": "ep_000", "action": "update",
         "reason": "jev dedup=update p=0.900"},
        {"eid": "ep_001", "action": "store",
         "reason": "jev dedup=store p=0.800"},
    ]


def test_judge_dedup_pairs_malformed_verdicts_dropped(monkeypatch):
    body = {"answers": {
        "pair_0": _choice_ans("update", 0.9),
        "pair_1": _choice_ans("detonate", 0.9),  # out-of-vocab -> drop
        "pair_2": {"type": "choice"},            # no choice -> drop
    }}
    monkeypatch.setattr("src.gnn.jev_decider.requests.post",
                        lambda *a, **kw: _Resp(body=body))
    dec = _make_dec()
    out = dec.judge_dedup_pairs("s", [], [], _cands(3))
    assert [v["eid"] for v in out] == ["ep_000"]


def test_judge_dedup_pairs_chunks_over_64(monkeypatch):
    counts = []

    def fake_post(url, data=None, timeout=None, **kw):
        questions = json.loads(data)["questions"]
        counts.append(len(questions))
        return _Resp(body={"answers": {
            f"pair_{i}": _choice_ans("skip", 0.95)
            for i in range(len(questions))
        }})

    monkeypatch.setattr("src.gnn.jev_decider.requests.post", fake_post)
    dec = _make_dec()
    out = dec.judge_dedup_pairs("s", [], [], _cands(70))
    assert counts == [64, 6]
    assert len(out) == 70
    assert all(v["action"] == "skip" for v in out)


def test_judge_dedup_pairs_failed_chunk_fails_whole_call(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise _real_requests.ConnectionError("dropped")
        return _Resp(body={"answers": {"pair_0": _choice_ans("store", 0.9)}})

    monkeypatch.setattr("src.gnn.jev_decider.requests.post", fake_post)
    dec = _make_dec()
    out = dec.judge_dedup_pairs("s", [], [], _cands(65))
    assert out is None  # all-or-nothing: never a partial application


def test_judge_dedup_pairs_empty_candidates_none(monkeypatch):
    monkeypatch.setattr("src.gnn.jev_decider.requests.post",
                        lambda *a, **kw: pytest.fail("no HTTP on empty"))
    dec = _make_dec()
    assert dec.judge_dedup_pairs("s", [], [], []) is None


# ── verify_fidelity ────────────────────────────────────────────────────────

@pytest.mark.parametrize("p,expected", [(0.9, True), (0.5, True), (0.3, False)])
def test_verify_fidelity_threshold(monkeypatch, p, expected):
    monkeypatch.setattr(
        "src.gnn.jev_decider.requests.post",
        lambda *a, **kw: _Resp(body={"answers": {
            "corrupt": {"type": "noul", "noul": p}}}))
    dec = _make_dec()
    out = dec.verify_fidelity("blurb", "narrative")
    assert out["corruption"] is expected


def test_verify_fidelity_lower_threshold_conservative(monkeypatch):
    # threshold 0.4: a p=0.45 verdict counts as corrupt (a misjudged "clean"
    # OVERWRITES the verbatim; a misjudged "corrupt" only defers). With the
    # default 0.5 the same answer would be clean.
    monkeypatch.setattr(
        "src.gnn.jev_decider.requests.post",
        lambda *a, **kw: _Resp(body={"answers": {
            "corrupt": {"type": "noul", "noul": 0.45}}}))
    dec = _make_dec(corruption_threshold=0.4)
    assert dec.verify_fidelity("b", "n")["corruption"] is True
    dec = _make_dec()  # default 0.5: the same answer is clean
    assert dec.verify_fidelity("b", "n")["corruption"] is False


def test_verify_fidelity_missing_answer_none(monkeypatch):
    monkeypatch.setattr(
        "src.gnn.jev_decider.requests.post",
        lambda *a, **kw: _Resp(body={"answers": {}}))
    dec = _make_dec()
    assert dec.verify_fidelity("b", "n") is None


def test_verify_fidelity_http_failure_none(monkeypatch):
    monkeypatch.setattr(
        "src.gnn.jev_decider.requests.post",
        lambda *a, **kw: exec("raise _e",
                              {"_e": _real_requests.ConnectionError("down")}))
    dec = _make_dec()
    assert dec.verify_fidelity("b", "n") is None


# ── decide_anomaly / decide_contradiction ──────────────────────────────────

def test_decide_anomaly_three_way(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.gnn.jev_decider.requests.post", _fake_post_capturer(seen, {
        "answers": {"decision": _choice_ans("ask_user", 0.77)}}))
    dec = _make_dec()
    out = dec.decide_anomaly({"node": "E:alice", "evidence": {}},
                             {"neighbors": []})
    assert out == {
        "decision": "ask_user",
        "action": "no_action",  # never a fabricated supersede_assertion
        "reasoning": "jev decide_anomaly=ask_user p=0.770",
    }
    assert seen["payload"]["state"]["flag"]["entity"] == "E:alice"


def test_decide_anomaly_http_failure_none(monkeypatch):
    monkeypatch.setattr("src.gnn.jev_decider.requests.post",
                        lambda *a, **kw: _Resp(status_code=503))
    dec = _make_dec()
    assert dec.decide_anomaly({"node": "x"}, {}) is None


def test_decide_anomaly_unserializable_state_none_no_http(monkeypatch):
    # retrieved_context is radius-1 graph material -- if it carries a
    # non-JSON-serializable value, the dump fails and the call returns None
    # (never a raised TypeError violating the cold-start contract).
    monkeypatch.setattr("src.gnn.jev_decider.requests.post",
                        lambda *a, **kw: pytest.fail("no HTTP on dump fail"))
    dec = _make_dec()
    assert dec.decide_anomaly({"node": "x"}, {"neighbors": [{1, 2}]}) is None


def test_decide_contradiction_prefilter_short_circuits_no_http(monkeypatch):
    # Guard 1 (equal values): a dismiss, decided BEFORE any HTTP. The mock
    # raises if called -- zero Jev calls on a non-conflict.
    monkeypatch.setattr("src.gnn.jev_decider.requests.post",
                        lambda *a, **kw: pytest.fail(
                            "deterministic pre-filter must not hit HTTP"))
    dec = _make_dec()
    out = dec.decide_contradiction(
        {"node": "E:x"},
        {"state_values": [{"value": "red", "doc_kind": "", "source_path": ""},
                          {"value": "red"}]})
    assert out["decision"] == "dismiss"


def test_decide_contradiction_prefilter_complementary_temporal(monkeypatch):
    # Guard 2 (both point_in_time_snapshot): ask_user, still no HTTP.
    monkeypatch.setattr("src.gnn.jev_decider.requests.post",
                        lambda *a, **kw: pytest.fail(
                            "deterministic pre-filter must not hit HTTP"))
    dec = _make_dec()
    out = dec.decide_contradiction(
        {"node": "E:x"},
        {"state_values": [
            {"value": "red", "doc_kind": "point_in_time_snapshot"},
            {"value": "blue", "doc_kind": "point_in_time_snapshot"}]})
    assert out["decision"] == "ask_user"


def test_decide_contradiction_real_conflict_hits_jev(monkeypatch):
    monkeypatch.setattr(
        "src.gnn.jev_decider.requests.post",
        lambda *a, **kw: _Resp(body={"answers": {
            "decision": _choice_ans("fix", 0.66)}}))
    dec = _make_dec()
    out = dec.decide_contradiction(
        {"node": "E:x"},
        {"state_values": [{"value": "red", "doc_kind": "decision_update"},
                          {"value": "blue", "doc_kind": "decision_update"}]})
    assert out["decision"] == "fix" and out["action"] == "no_action"


# ── classify_doc_kind ──────────────────────────────────────────────────────

def test_classify_doc_kind_labels(monkeypatch):
    monkeypatch.setattr(
        "src.gnn.jev_decider.requests.post",
        lambda *a, **kw: _Resp(body={"answers": {
            "doc_kind": _choice_ans("decision_update", 0.9)}}))
    dec = _make_dec()
    assert dec.classify_doc_kind("doc text") == "decision_update"


def test_classify_doc_kind_off_vocab_none(monkeypatch):
    monkeypatch.setattr(
        "src.gnn.jev_decider.requests.post",
        lambda *a, **kw: _Resp(body={"answers": {
            "doc_kind": _choice_ans("poetry", 1.0)}}))
    dec = _make_dec()
    assert dec.classify_doc_kind("doc text") is None


# ── Jev plumbing: usage mapping + pause_gate + health ──────────────────────

def test_usage_mapped_to_openai_style_keys(monkeypatch):
    monkeypatch.setattr(
        "src.gnn.jev_decider.requests.post",
        lambda *a, **kw: _Resp(body={
            "answers": {"corrupt": {"type": "noul", "noul": 0.1}},
            "usage": {"input_tokens": 10, "output_tokens": 2}}))
    dec = _make_dec()
    dec.verify_fidelity("b", "n")
    # The @record_llm_call decorator reads prompt_tokens/completion_tokens.
    assert dec._last_usage == {
        "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


def test_pause_gate_checked_before_http(monkeypatch):
    order = []

    def fake_gate():
        order.append("gate")

    def fake_post(url, json=None, timeout=None, **kw):
        order.append("http")
        return _Resp(body={"answers": {"corrupt": {"type": "noul", "noul": 0.1}}})

    monkeypatch.setattr("src.gnn.jev_decider.requests.post", fake_post)
    dec = _make_dec()
    dec.pause_gate = fake_gate
    dec.verify_fidelity("b", "n")
    assert order == ["gate", "http"]


def test_health_checks(monkeypatch):
    monkeypatch.setattr("src.gnn.jev_decider.requests.get",
                        lambda *a, **kw: _Resp(status_code=200))
    assert _make_dec().jev_health_check() is True
    monkeypatch.setattr("src.gnn.jev_decider.requests.get",
                        lambda *a, **kw: _Resp(status_code=500))
    assert _make_dec().jev_health_check() is False
    monkeypatch.setattr("src.gnn.jev_decider.requests.get",
                        lambda *a, **kw: exec(
                            "raise _e", {"_e": _real_requests.Timeout("x")}))
    assert _make_dec().jev_health_check() is False


# ── make_decider factory ───────────────────────────────────────────────────

def test_make_decider_bonsai_is_the_default(monkeypatch):
    from src.config import config as _cfg
    monkeypatch.setattr(_cfg, "decision_backend", "bonsai", raising=False)
    assert type(make_decider("bonsai")) is BonsaiDecider
    assert type(make_decider(None)) is BonsaiDecider  # config default governs


def test_make_decider_jev_name_routes_to_subclass():
    dec = make_decider("nimble")
    assert isinstance(dec, JevDecider)
    assert isinstance(dec, BonsaiDecider)  # every BonsaiDecider DI seam holds
    assert dec.jev_model == "nimble"


# ── live gate: only when the operator set DECISION_BACKEND ────────────────

_live_backend = os.getenv("DECISION_BACKEND", "bonsai")


@pytest.mark.skipif(_live_backend == "bonsai",
                    reason="live Jev backend not configured "
                           "(set DECISION_BACKEND=<ollama model name>)")
def test_live_jev_decision_roundtrip():
    dec = _make_dec(backend=_live_backend)
    assert dec.jev_health_check(), "DECISION_BACKEND set but Ollama down"
    out = dec.verify_fidelity(
        "Alice lives in Paris and works at Acme.",
        "Alice lives in PARIS and works at Acme (paraphrased city).")
    # A valid roundtrip returns the 2-field dict (corruption True/False per
    # the model) -- never raises, never a fabricated third shape.
    if out is not None:
        assert set(out) == {"corruption", "reason"}