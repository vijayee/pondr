"""Tests for scripts/fetch_onyx_sessions.py (STRM Phase 2d Onyx fetch).

No real Onyx server, no real API key, no ``responses`` dep. The HTTP seam is a
``get_json(url, params)`` callable injected into ``list_sessions`` /
``fetch_session_detail`` / ``fetch_all`` -- the tests pass a fake that returns
canned JSON by URL, so pagination + thread reconstruction + the failed-session
skip are exercised without any network. The env-var auth guard + the SECRET-
scrubbing of the HTTP error message are also covered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fetch_onyx_sessions import (  # noqa: E402
    fetch_all,
    fetch_session_detail,
    list_sessions,
    make_get_json,
    reconstruct_messages,
    require_env,
    session_has_error,
    session_record,
    write_jsonl,
)


# ── fake get_json: returns canned JSON by URL + params ──

def _fake_get_json(pages=None, details=None):
    """Build a fake get_json. ``pages`` is the list of /chat/search page
    responses (in order); ``details`` is ``{session_id: detail_dict}``."""
    calls = {"search_pages": [], "details": []}

    def get_json(url, params=None):
        if url.startswith("/chat/search"):
            idx = len(calls["search_pages"])
            calls["search_pages"].append(params or {})
            if pages is None or idx >= len(pages):
                return {"groups": [], "has_more": False, "next_page": None}
            return pages[idx]
        if url.startswith("/chat/get-chat-session/"):
            sid = url.rsplit("/", 1)[-1]
            calls["details"].append(sid)
            if details is None or sid not in details:
                raise RuntimeError(f"no fake detail for {sid}")
            return details[sid]
        raise AssertionError(f"fake get_json got unexpected url {url}")

    get_json.calls = calls
    return get_json


# ── require_env ──

def test_require_env_refuses_when_key_missing(monkeypatch):
    monkeypatch.setenv("ONYX_BASE_URL", "http://192.168.1.198")
    monkeypatch.delenv("ONYX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ONYX_API_KEY"):
        require_env()


def test_require_env_refuses_when_base_missing(monkeypatch):
    monkeypatch.setenv("ONYX_API_KEY", "some-secret-key")
    monkeypatch.delenv("ONYX_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="ONYX_BASE_URL"):
        require_env()


def test_require_env_returns_base_and_key(monkeypatch):
    monkeypatch.setenv("ONYX_BASE_URL", "http://192.168.1.198/")
    monkeypatch.setenv("ONYX_API_KEY", "some-secret-key")
    base, key = require_env()
    assert base == "http://192.168.1.198"   # trailing slash stripped
    assert key == "some-secret-key"


# ── list_sessions pagination ──

def test_list_sessions_paginates_until_has_more_false():
    pages = [
        {"groups": [{"title": "g1", "chats": [{"id": "s1"}, {"id": "s2"}]}],
         "has_more": True, "next_page": 2},
        {"groups": [{"title": "g2", "chats": [{"id": "s3"}]}],
         "has_more": False, "next_page": None},
    ]
    gj = _fake_get_json(pages=pages)
    out = list_sessions(gj, page_size=50)
    assert [c["id"] for c in out] == ["s1", "s2", "s3"]
    # two search calls (page 1, page 2)
    assert len(gj.calls["search_pages"]) == 2
    assert gj.calls["search_pages"][0]["page"] == 1
    assert gj.calls["search_pages"][1]["page"] == 2


def test_list_sessions_respects_limit():
    pages = [
        {"groups": [{"title": "g", "chats": [{"id": f"s{i}"} for i in range(50)]}],
         "has_more": True, "next_page": 2},
    ]
    gj = _fake_get_json(pages=pages)
    out = list_sessions(gj, page_size=50, limit=3)
    assert [c["id"] for c in out] == ["s0", "s1", "s2"]


def test_list_sessions_passes_query_filter():
    pages = [{"groups": [], "has_more": False, "next_page": None}]
    gj = _fake_get_json(pages=pages)
    list_sessions(gj, query="deploy", page_size=10)
    assert gj.calls["search_pages"][0]["query"] == "deploy"


def test_list_sessions_advances_past_a_stuck_next_page():
    # has_more=True but next_page == current page (a misbehaving server) -- the
    # belt-and-suspenders guard advances the page so we do not loop forever.
    pages = [
        {"groups": [{"title": "g", "chats": [{"id": "s1"}]}],
         "has_more": True, "next_page": 1},     # stuck at page 1
        {"groups": [], "has_more": False, "next_page": None},
    ]
    gj = _fake_get_json(pages=pages)
    out = list_sessions(gj, page_size=10)
    assert [c["id"] for c in out] == ["s1"]
    assert len(gj.calls["search_pages"]) == 2   # advanced, then stopped


# ── reconstruct_messages ──

def test_reconstruct_messages_sorts_chronologically_and_maps_fields():
    msgs = [
        {"message_id": 3, "parent_message": 2, "message_type": "assistant",
         "message": "third", "time_sent": "2026-07-20T03:00:00Z",
         "citations": None, "error": None},
        {"message_id": 1, "parent_message": None, "message_type": "user",
         "message": "first", "time_sent": "2026-07-20T01:00:00Z",
         "citations": None, "error": None},
        {"message_id": 2, "parent_message": 1, "message_type": "assistant",
         "message": "second", "time_sent": "2026-07-20T02:00:00Z",
         "citations": {1: "docA"}, "error": None},
    ]
    out = reconstruct_messages(msgs)
    assert [m["message_id"] for m in out] == [1, 2, 3]      # sorted by time_sent
    assert out[0]["role"] == "user"
    assert out[0]["text"] == "first"
    assert out[0]["parent_message"] is None
    assert out[1]["parent_message"] == 1                     # preserved
    assert out[1]["citations"] == {1: "docA"}               # preserved verbatim
    assert out[2]["role"] == "assistant"


def test_reconstruct_messages_empty_or_none():
    assert reconstruct_messages([]) == []
    assert reconstruct_messages(None) == []                 # type: ignore[arg-type]


def test_reconstruct_messages_guards_non_list_and_non_dict_items():
    # a malformed detail (messages is a string / contains a non-dict) yields an
    # empty / filtered thread, not a crash.
    assert reconstruct_messages("not a list") == []          # type: ignore[arg-type]
    out = reconstruct_messages([
        {"message_id": 1, "message_type": "user", "message": "ok",
         "time_sent": "t1", "parent_message": None, "citations": None},
        "not a dict",                                        # type: ignore[list-item]
        42,                                                  # type: ignore[list-item]
    ])
    assert len(out) == 1
    assert out[0]["message_id"] == 1


# ── session_has_error + session_record ──

def test_session_has_error_detects_errored_message():
    detail = {"messages": [{"error": None}, {"error": "timeout"}]}
    assert session_has_error(detail) is True
    assert session_has_error({"messages": [{"error": None}]}) is False
    assert session_has_error({"messages": []}) is False
    assert session_has_error({}) is False


def test_session_record_prefers_detail_fields_then_summary_fallbacks():
    summary = {"id": "abc", "name": "summary name",
               "time_created": "2026-07-19T00:00:00Z"}
    detail = {"chat_session_id": "abc", "description": "detail desc",
              "time_created": "2026-07-20T00:00:00Z",
              "messages": [{"message_id": 1, "parent_message": None,
                            "message_type": "user", "message": "hi",
                            "time_sent": "2026-07-20T01:00:00Z",
                            "citations": None, "error": None}]}
    rec = session_record(summary, detail)
    assert rec["session_id"] == "abc"                       # detail's chat_session_id
    assert rec["name"] == "detail desc"                     # detail's description
    assert rec["time_created"] == "2026-07-20T00:00:00Z"    # detail's time
    assert rec["has_error"] is False
    assert len(rec["messages"]) == 1


def test_session_record_falls_back_to_summary_when_detail_lacks_fields():
    summary = {"id": "abc", "name": "summary name",
               "time_created": "2026-07-19T00:00:00Z"}
    detail = {"messages": []}                               # no chat_session_id/description
    rec = session_record(summary, detail)
    assert rec["session_id"] == "abc"                       # summary id fallback
    assert rec["name"] == "summary name"                    # summary name fallback
    assert rec["time_created"] == "2026-07-19T00:00:00Z"    # summary time fallback


# ── fetch_all ──

def test_fetch_all_skips_errored_sessions_unless_include_failed():
    pages = [{"groups": [{"title": "g", "chats": [{"id": "ok"}, {"id": "bad"}]}],
              "has_more": False, "next_page": None}]
    details = {
        "ok": {"chat_session_id": "ok", "description": "ok session",
               "time_created": "t1", "messages": [
                   {"message_id": 1, "parent_message": None,
                    "message_type": "user", "message": "hi",
                    "time_sent": "t1", "citations": None, "error": None}]},
        "bad": {"chat_session_id": "bad", "description": "bad session",
                "time_created": "t2", "messages": [
                    {"message_id": 1, "parent_message": None,
                     "message_type": "assistant", "message": "oops",
                     "time_sent": "t2", "citations": None, "error": "boom"}]},
    }
    gj = _fake_get_json(pages=pages, details=details)
    kept = fetch_all(gj, query=None, page_size=50, limit=None, include_failed=False)
    assert [r["session_id"] for r in kept] == ["ok"]         # bad skipped

    gj2 = _fake_get_json(pages=pages, details=details)
    kept2 = fetch_all(gj2, query=None, page_size=50, limit=None, include_failed=True)
    assert [r["session_id"] for r in kept2] == ["ok", "bad"]
    assert kept2[1]["has_error"] is True                     # still marked


def test_fetch_all_skips_a_session_whose_detail_fetch_raises():
    pages = [{"groups": [{"title": "g", "chats": [{"id": "s1"}, {"id": "s2"}]}],
              "has_more": False, "next_page": None}]
    details = {"s1": {"chat_session_id": "s1", "description": "d1",
                      "time_created": "t1", "messages": []}}
    # s2 not in details -> _fake_get_json raises for it
    gj = _fake_get_json(pages=pages, details=details)
    kept = fetch_all(gj, query=None, page_size=50, limit=None, include_failed=False)
    assert [r["session_id"] for r in kept] == ["s1"]         # s2 skipped, not abort


def test_fetch_all_skips_a_session_with_a_malformed_detail():
    # a detail that is not a dict (e.g. the API returned a list) makes
    # session_record raise -> the wider per-session guard skips it, not aborts.
    pages = [{"groups": [{"title": "g", "chats": [{"id": "s1"}, {"id": "s2"}]}],
              "has_more": False, "next_page": None}]
    details = {
        "s1": {"chat_session_id": "s1", "description": "d1",
               "time_created": "t1", "messages": []},
        "s2": ["not", "a", "dict"],                          # malformed
    }
    gj = _fake_get_json(pages=pages, details=details)
    kept = fetch_all(gj, query=None, page_size=50, limit=None, include_failed=False)
    assert [r["session_id"] for r in kept] == ["s1"]         # s2 skipped, not abort


# ── write_jsonl ──

def test_write_jsonl_one_record_per_line(tmp_path):
    records = [
        {"session_id": "s1", "name": "n1", "time_created": "t1",
         "has_error": False, "messages": [{"message_id": 1, "role": "user"}]},
        {"session_id": "s2", "name": "n2", "time_created": "t2",
         "has_error": True, "messages": []},
    ]
    out = tmp_path / "sessions.jsonl"
    write_jsonl(records, str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["session_id"] == "s1"
    assert json.loads(lines[1])["has_error"] is True


# ── SECURITY: the HTTP error message scrubs the Bearer key ──

def test_make_get_json_error_message_scrubs_the_api_key(monkeypatch):
    """The Onyx API key is a SECRET. The RuntimeError raised on a non-2xx
    response must NOT contain the key (the Authorization header is scrubbed --
    only status + url + body print, and the url carries no key)."""
    import requests as _requests

    secret = "SUPER-SECRET-KEY-DO-NOT-LEAK"

    class _FakeResp:
        ok = False
        status_code = 500
        text = "internal server error"

        def json(self):
            return {}

    # monkeypatch the Session.get used inside make_get_json's closure so no
    # real network call is made; the fake response triggers the error path.
    def _fake_get(self, url, params=None, timeout=None):
        return _FakeResp()

    monkeypatch.setattr(_requests.Session, "get", _fake_get)
    get_json = make_get_json("http://192.168.1.198", secret)
    with pytest.raises(RuntimeError) as ei:
        get_json("/chat/search", {"page": 1})
    msg = str(ei.value)
    assert secret not in msg                      # the key MUST NOT appear
    assert "Bearer" not in msg                    # the header MUST NOT appear
    assert "500" in msg                           # status is surfaced
    assert "/chat/search" in msg                  # url is surfaced (no secret in it)