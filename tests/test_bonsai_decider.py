"""Tests for ``BonsaiDecider`` (the deploy-time consolidation decider).

Offline: ``requests.post`` / ``requests.get`` are monkeypatched so NO live
Bonsai server is needed. Verifies the three decisions (gist / verify_typing /
decide_anomaly) + health_check + the JSON-recovery parse path
(fenced/bare/truncated). The live dogfood test lives in
``test_bonsai_decider_live.py`` (skip via ``GET /v1/models``).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.gnn.bonsai_decider import BonsaiDecider


# ── helpers: fake requests responses ──────────────────────────────────────

class _FakeResp:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)
        self._body = body

    def json(self):
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body


def _chat_response(content: str) -> _FakeResp:
    """A chat-completion response whose message.content is ``content``."""
    return _FakeResp(200, {"choices": [{"message": {"content": content}}]})


@pytest.fixture
def decider():
    return BonsaiDecider(endpoint="http://bogus:8080/v1")


# ── health_check ──────────────────────────────────────────────────────────

def test_health_check_true_on_200(decider):
    with patch("src.gnn.bonsai_decider.requests.get",
               return_value=_FakeResp(200, {"data": []})):
        assert decider.health_check() is True


def test_health_check_false_on_error(decider):
    import requests
    with patch("src.gnn.bonsai_decider.requests.get",
               side_effect=requests.ConnectionError("down")):
        assert decider.health_check() is False


def test_health_check_false_on_non_200(decider):
    with patch("src.gnn.bonsai_decider.requests.get",
               return_value=_FakeResp(503, "unavailable")):
        assert decider.health_check() is False


# ── gist ──────────────────────────────────────────────────────────────────

def test_gist_returns_stripped_string(decider):
    content = json.dumps({"gist": "  Alice and Bob discussed the DB.  "})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        g = decider.gist([{"id": "ep_1", "summary": "s1"}])
    assert g == "Alice and Bob discussed the DB."


def test_gist_strips_control_chars(decider):
    # a literal BEL + vertical tab inside the gist are C0 controls that must
    # be stripped (newline/tab are kept -- legitimate prose).
    content = json.dumps({"gist": "line1\x07\x0b\nline2"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        g = decider.gist([{"id": "ep_1", "summary": "s1"}])
    assert "\x07" not in g and "\x0b" not in g
    assert "\n" in g  # newline preserved


def test_gist_empty_sources_returns_none(decider):
    assert decider.gist([]) is None


def test_gist_none_on_http_failure(decider):
    import requests
    with patch("src.gnn.bonsai_decider.requests.post",
               side_effect=requests.ConnectionError("down")):
        assert decider.gist([{"id": "ep_1", "summary": "s1"}]) is None


def test_gist_none_on_non_json_content(decider):
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response("not json at all")):
        assert decider.gist([{"id": "ep_1", "summary": "s1"}]) is None


def test_gist_none_on_missing_gist_field(decider):
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(json.dumps({"other": "x"}))):
        assert decider.gist([{"id": "ep_1", "summary": "s1"}]) is None


def test_gist_strips_fences(decider):
    body = "```json\n" + json.dumps({"gist": "fenced gist"}) + "\n```"
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(body)):
        assert decider.gist([{"id": "ep_1", "summary": "s1"}]) == "fenced gist"


def test_gist_salvages_outermost_span(decider):
    # trailing prose after the JSON object -- the outermost-span fallback
    # carves the object out.
    body = json.dumps({"gist": "salvaged"}) + " -- here is some trailing prose."
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(body)):
        assert decider.gist([{"id": "ep_1", "summary": "s1"}]) == "salvaged"


# ── verify_typing ─────────────────────────────────────────────────────────

def test_verify_typing_accept(decider):
    content = json.dumps({"accept": True, "new_class": None, "parent": None,
                          "reasoning": "yes"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        v = decider.verify_typing("E:Alice", "Person", {"entity": "E:Alice"})
    assert v["accept"] is True
    assert v["new_class"] is None and v["parent"] is None


def test_verify_typing_new_class_with_parent(decider):
    content = json.dumps({"accept": True, "new_class": "DBEngineer",
                          "parent": "Person", "reasoning": "narrower"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        v = decider.verify_typing("E:Alice", "Person", {"entity": "E:Alice"})
    assert v["accept"] is True
    assert v["new_class"] == "DBEngineer" and v["parent"] == "Person"


def test_verify_typing_new_class_without_parent_rejected(decider):
    # a new class without a parent would orphan -> the decider normalizes to a
    # rejection so the caller records nothing.
    content = json.dumps({"accept": True, "new_class": "DBEngineer",
                          "parent": None, "reasoning": "orphan"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        v = decider.verify_typing("E:Alice", "Person", {"entity": "E:Alice"})
    assert v["accept"] is False


def test_verify_typing_reject(decider):
    content = json.dumps({"accept": False, "new_class": None, "parent": None,
                          "reasoning": "no"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        v = decider.verify_typing("E:Alice", "Person", {"entity": "E:Alice"})
    assert v["accept"] is False


def test_verify_typing_null_string_normalized(decider):
    # the model sometimes emits the string "null" instead of JSON null.
    content = json.dumps({"accept": True, "new_class": "null", "parent": "null",
                          "reasoning": "x"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        v = decider.verify_typing("E:Alice", "Person", {"entity": "E:Alice"})
    assert v["new_class"] is None and v["parent"] is None
    assert v["accept"] is True


def test_verify_typing_none_on_missing_accept(decider):
    # no ``accept`` field -> can't validate a verdict -> None (record-only).
    content = json.dumps({"nope": "missing accept"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        v = decider.verify_typing("E:Alice", "Person", {"entity": "E:Alice"})
    assert v is None


# ── decide_anomaly ────────────────────────────────────────────────────────

def test_decide_anomaly_fix(decider):
    content = json.dumps({"decision": "fix", "action": "supersede_episode",
                          "reasoning": "drift"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        d = decider.decide_anomaly({"node": "E:Alice", "type": "identity_drift"},
                                   {"entity": "E:Alice"})
    assert d["decision"] == "fix"
    assert d["action"] == "supersede_episode"


def test_decide_anomaly_ask_user(decider):
    content = json.dumps({"decision": "ask_user", "action": "ask",
                          "reasoning": "ambiguous"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        d = decider.decide_anomaly({"node": "E:Alice", "type": "identity_drift"},
                                   {"entity": "E:Alice"})
    assert d["decision"] == "ask_user"


def test_decide_anomaly_invalid_decision_returns_none(decider):
    content = json.dumps({"decision": "explode", "action": "x", "reasoning": "y"})
    with patch("src.gnn.bonsai_decider.requests.post",
               return_value=_chat_response(content)):
        d = decider.decide_anomaly({"node": "E:Alice", "type": "identity_drift"},
                                   {"entity": "E:Alice"})
    assert d is None


def test_decide_anomaly_none_on_http_failure(decider):
    import requests
    with patch("src.gnn.bonsai_decider.requests.post",
               side_effect=requests.ConnectionError("down")):
        d = decider.decide_anomaly({"node": "E:Alice", "type": "identity_drift"},
                                   {"entity": "E:Alice"})
    assert d is None


# ── _parse_json_object unit tests ─────────────────────────────────────────

def test_parse_json_object_clean():
    assert BonsaiDecider._parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_fenced():
    assert BonsaiDecider._parse_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_parse_json_object_with_trailing_prose():
    assert BonsaiDecider._parse_json_object('{"a": 1} trailing') == {"a": 1}


def test_parse_json_object_garbage_returns_none():
    assert BonsaiDecider._parse_json_object("not json") is None


def test_parse_json_object_list_returns_none():
    # a bare list is not a dict decision object.
    assert BonsaiDecider._parse_json_object("[1, 2, 3]") is None