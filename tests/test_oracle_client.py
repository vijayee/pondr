"""Offline tests for the Phase 1d ``OracleClient``.

No live Ollama calls in the default suite — ``_post`` is stubbed to assert
retry, on-disk cache hit/miss, fence-strip, outermost-span parsing, and
truncation handling. The live path is gated behind a localhost:11434 probe and
``pytest.skip``s when the model isn't pulled.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import requests

from src.training.oracle_labeling import (
    OracleClient,
    OracleConfig,
)


def _resp(content: str, inp: int = 10, out: int = 5, status: int = 200):
    return status, {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": inp, "completion_tokens": out},
    }


def _client(tmp_path: Path, **overrides) -> OracleClient:
    cfg = OracleConfig(
        model="stub",
        endpoint="http://stub/v1",
        max_tokens=overrides.pop("max_tokens", 4096),
        max_retries=overrides.pop("max_retries", 3),
        retry_delay=overrides.pop("retry_delay", 0.0),
        cache_path=tmp_path / ".oracle_cache.json",
    )
    return OracleClient(cfg)


# ── generate: parse paths ──


def test_generate_parses_clean_json(tmp_path):
    c = _client(tmp_path)
    c._post = lambda payload: _resp('{"labels": {"x": 1}}')
    r = c.generate("p")
    assert r.response == {"labels": {"x": 1}}
    assert r.input_tokens == 10 and r.output_tokens == 5
    assert r.cached is False
    assert c.total_calls == 1 and c.total_tokens == 15


def test_generate_strips_code_fence(tmp_path):
    c = _client(tmp_path)
    c._post = lambda payload: _resp('```json\n{"k": "v"}\n```')
    assert c.generate("p").response == {"k": "v"}


def test_generate_outermost_span_ignores_prose(tmp_path):
    c = _client(tmp_path)
    c._post = lambda payload: _resp('Sure! {"answer": 42} hope that helps')
    assert c.generate("p").response == {"answer": 42}


def test_generate_outermost_span_when_clean(tmp_path):
    """When the outermost span IS valid JSON, it wins."""
    c = _client(tmp_path)
    c._post = lambda payload: _resp('envelope {"inner": {"k": 1}} trailing')
    assert c.generate("p").response == {"inner": {"k": 1}}


def test_generate_truncated_response_does_not_salvage_inner_fragment(tmp_path):
    """A truncated outer object must NOT silently return a nested fragment.

    Regression guard for the code-aware slice: a response cut mid-payload used
    to salvage the first balanced inner dict (e.g. one extracted_relations
    entry) as the label — a silent wrong label. It must now raise loudly.
    """
    c = _client(tmp_path, max_tokens=4096)
    # Truncated: outer { never closes; one complete inner relation object.
    content = '{"conversation": "User: ...", "extracted_relations": [{"s": "a", "p": "b", "o": "c"} , {"s": "x'
    c._post = lambda payload: _resp(content, out=4096)
    with pytest.raises(RuntimeError, match="truncated at max_tokens"):
        c.generate("p")


def test_generate_truncation_hint_only_when_cap_hit(tmp_path):
    """Parse failure with output BELOW max_tokens is the plain unparseable error."""
    c = _client(tmp_path, max_tokens=4096)
    c._post = lambda payload: _resp("totally not json at all", out=5)
    with pytest.raises(RuntimeError, match="unparseable JSON"):
        c.generate("p")


def test_generate_unparseable_raises_runtimeerror(tmp_path):
    c = _client(tmp_path)
    c._post = lambda payload: _resp("totally not json at all")
    with pytest.raises(RuntimeError, match="unparseable JSON"):
        c.generate("p")


def test_generate_missing_content_raises_runtimeerror(tmp_path):
    c = _client(tmp_path)
    c._post = lambda payload: (200, {"choices": []})
    with pytest.raises(RuntimeError, match="missing choices"):
        c.generate("p")


# ── retry / non-retryable ──


def test_generate_retries_connection_error_then_succeeds(tmp_path):
    c = _client(tmp_path, max_retries=3, retry_delay=0.0)
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("boom")
        return _resp('{"ok": true}')

    c._post = flaky
    assert c.generate("p").response == {"ok": True}
    assert calls["n"] == 3


def test_generate_4xx_is_non_retryable(tmp_path):
    c = _client(tmp_path, max_retries=3, retry_delay=0.0)
    calls = {"n": 0}

    def bad(payload):
        calls["n"] += 1
        return (400, {"error": "bad request"})

    c._post = bad
    with pytest.raises(RuntimeError, match="HTTP 400"):
        c.generate("p")
    assert calls["n"] == 1  # not retried


def test_generate_5xx_retries_then_raises(tmp_path):
    c = _client(tmp_path, max_retries=2, retry_delay=0.0)
    c._post = lambda payload: (503, {"error": "down"})
    with pytest.raises(RuntimeError, match="failed after 2 retries"):
        c.generate("p")


# ── cache ──


def test_cache_hit_skips_http_and_marks_cached(tmp_path):
    c = _client(tmp_path)
    posted = {"n": 0}

    def post(payload):
        posted["n"] += 1
        return _resp('{"v": 1}')

    c._post = post
    c.generate("same prompt")
    c.generate("same prompt")
    assert posted["n"] == 1  # second call served from cache
    assert c.generate("same prompt").cached is True


def test_cache_persists_to_disk_across_instances(tmp_path):
    c1 = _client(tmp_path)
    c1._post = lambda payload: _resp('{"v": 1}')
    c1.generate("persist me")
    c1.flush_cache()
    assert (tmp_path / ".oracle_cache.json").exists()

    c2 = _client(tmp_path)  # loads cache on init
    posted = {"n": 0}

    def post2(payload):
        posted["n"] += 1
        return _resp('{"v": 2}')

    c2._post = post2
    r = c2.generate("persist me")
    assert posted["n"] == 0  # served from disk cache, no HTTP
    assert r.cached is True
    assert r.response == {"v": 1}  # original cached payload


def test_flush_cache_keeps_prior_records_on_resumed_run(tmp_path):
    """Regression: flush_cache must not drop cache-loaded records from disk.

    A resumed run loads prior records (cached=True), generates new ones, then
    flushes. The cache file must contain BOTH — replacing the file with only
    the new records would lose the prior ones and defeat cross-run resume.
    """
    c1 = _client(tmp_path)
    c1._post = lambda payload: _resp('{"v": 1}')
    c1.generate("old prompt")
    c1.flush_cache()

    c2 = _client(tmp_path)  # loads "old prompt" (cached=True)
    c2._post = lambda payload: _resp('{"v": 2}')
    c2.generate("new prompt")
    c2.flush_cache()

    c3 = _client(tmp_path)  # should find BOTH in the cache file
    posted = {"n": 0}

    def post3(payload):
        posted["n"] += 1
        return _resp('{"v": 3}')

    c3._post = post3
    assert c3.generate("old prompt").response == {"v": 1}
    assert c3.generate("new prompt").response == {"v": 2}
    assert posted["n"] == 0  # both served from disk cache


def test_get_stats_reports_usage(tmp_path):
    c = _client(tmp_path)
    c._post = lambda payload: _resp('{"v": 1}')
    c.generate("a")
    c.generate("a")  # cached
    c.generate("b")
    stats = c.get_stats()
    assert stats["total_calls"] == 2  # only non-cached calls counted
    assert stats["cache_size"] == 2
    assert stats["total_tokens"] == 30  # 2 calls * 15 tokens


# ── native Ollama /api/chat path (think=False for qwen3) ──


def _native_resp(content: str, inp: int = 10, out: int = 5, status: int = 200):
    return status, {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": inp,
        "eval_count": out,
        "done": True,
    }


def test_generate_native_chat_think_false_builds_native_payload(tmp_path):
    """think=False builds an /api/chat payload (think/format/options), not OpenAI."""
    captured = {}

    def fake_post(payload):
        captured["payload"] = payload
        return _native_resp('{"predicted_edges": [], "negative_edges": []}')

    c = _client(tmp_path, max_tokens=2048)
    c.config.think = False
    c._post = fake_post
    r = c.generate("score these pairs")
    assert r.error is None
    assert r.response == {"predicted_edges": [], "negative_edges": []}
    # native payload shape, NOT the OpenAI shape
    p = captured["payload"]
    assert p["think"] is False
    assert p["format"] == "json"
    assert p["stream"] is False
    assert "response_format" not in p
    assert p["options"]["num_predict"] == 2048
    assert "max_tokens" not in p


def test_generate_native_chat_parses_native_response_shape(tmp_path):
    """_extract reads message.content + eval_count from the native /api/chat shape."""
    c = _client(tmp_path)
    c.config.think = False
    c._post = lambda payload: _native_resp('{"k": 7}', inp=42, out=9)
    r = c.generate("x")
    assert r.response == {"k": 7}
    assert r.input_tokens == 42
    assert r.output_tokens == 9


def test_generate_native_chat_missing_content_raises(tmp_path):
    """A native response with no message.content surfaces a clear RuntimeError."""
    c = _client(tmp_path)
    c.config.think = False
    c._post = lambda payload: (200, {"message": {}, "done": True})
    with pytest.raises(RuntimeError, match="missing message.content"):
        c.generate("x")


# ── live path (gated) ──


def _ollama_reachable() -> bool:
    try:
        s = socket.create_connection(("localhost", 11434), timeout=1.0)
        s.close()
        return True
    except OSError:
        return False


def test_live_oracle_smoke(tmp_path):
    """One real call against local Ollama — skipped if not running/pulled."""
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at localhost:11434")
    cfg = OracleConfig(
        model=OracleConfig().model,
        endpoint="http://localhost:11434/v1",
        timeout=60.0,
        cache_path=tmp_path / ".oracle_cache.json",
    )
    c = OracleClient(cfg)
    try:
        r = c.generate('Return ONLY JSON: {"status": "ok"}')
        assert isinstance(r.response, dict)
        assert c.total_calls == 1
        assert c.total_tokens > 0
    except RuntimeError as e:
        # Model not pulled / misconfigured — surface, don't fail the suite.
        pytest.skip(f"live Oracle unavailable: {e}")