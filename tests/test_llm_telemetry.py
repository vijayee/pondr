"""Tests for ``src/observability/llm_telemetry.py`` -- the A5 JSONL sink +
``@record_llm_call`` decorator. The contract: never raises, never changes the
return / signature, flag-off = zero overhead + no file, flag-on = one JSONL line
per call carrying the ``_last_usage`` side-channel tokens + latency."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.config import config
from src.observability import llm_telemetry as mod
from src.observability.llm_telemetry import (
    LLMTelemetry,
    configure_telemetry,
    record_llm_call,
)


@pytest.fixture
def telemetry_on(tmp_path):
    """Enable telemetry + point the module sink at a tmp JSONL file. Restores
    the config flag + the null singleton on teardown."""
    path = str(tmp_path / "tel.jsonl")
    saved_flag = config.llm_telemetry_enabled
    saved_sink = mod._telemetry
    config.llm_telemetry_enabled = True
    configure_telemetry(path)
    yield path
    config.llm_telemetry_enabled = saved_flag
    mod._telemetry = saved_sink


# -- 1. record() writes a JSONL line with the right shape --
def test_record_writes_jsonl_line(telemetry_on) -> None:
    mod._telemetry.record("gist", 12.5, 100, 20, True, anchor_id=7)
    lines = Path(telemetry_on).read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["task"] == "gist"
    assert rec["latency_ms"] == 12.5
    assert rec["in_tok"] == 100
    assert rec["out_tok"] == 20
    assert rec["ok"] is True
    assert rec["anchor_id"] == 7
    assert "ts" in rec


# -- 2. flag OFF -> decorator passthrough, no file, return unchanged --
def test_flag_off_is_passthrough_no_file(tmp_path) -> None:
    saved_flag = config.llm_telemetry_enabled
    saved_sink = mod._telemetry
    config.llm_telemetry_enabled = False
    # Point the sink somewhere so a bug would write; flag-off must still NOT write.
    configure_telemetry(str(tmp_path / "off.jsonl"))
    try:

        class _Obj:
            @record_llm_call("t")
            def echo(self, x):
                return x * 3

        obj = _Obj()
        assert obj.echo(7) == 21  # return unchanged
        assert not (tmp_path / "off.jsonl").exists()  # NO file written
    finally:
        config.llm_telemetry_enabled = saved_flag
        mod._telemetry = saved_sink


# -- 3. decorator captures the _last_usage side-channel --
def test_decorator_captures_side_channel_usage(telemetry_on) -> None:

    class _Obj:
        @record_llm_call("verify_fidelity")
        def go(self):
            self._last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
            return "ok"

    _Obj().go()
    rec = json.loads(Path(telemetry_on).read_text(encoding="utf-8").strip())
    assert rec["in_tok"] == 10
    assert rec["out_tok"] == 5
    assert rec["task"] == "verify_fidelity"
    assert rec["ok"] is True


# -- 4. missing usage (_last_usage None) -> null tokens, no crash --
def test_decorator_missing_usage_null_tokens(telemetry_on) -> None:

    class _Obj:
        @record_llm_call("classify")
        def go(self):
            # No _last_usage set -> getattr returns None -> null tokens.
            return "label"

    _Obj().go()
    rec = json.loads(Path(telemetry_on).read_text(encoding="utf-8").strip())
    assert rec["in_tok"] is None
    assert rec["out_tok"] is None
    assert rec["ok"] is True


# -- 5. never raises: fn raises -> decorator re-raises, recording still fires --
def test_decorator_reraises_and_records_failure(telemetry_on) -> None:

    class _Boom(Exception):
        pass

    class _Obj:
        @record_llm_call("decide")
        def go(self):
            self._last_usage = {"prompt_tokens": 1, "completion_tokens": 1}
            raise _Boom("nope")

    with pytest.raises(_Boom):
        _Obj().go()
    rec = json.loads(Path(telemetry_on).read_text(encoding="utf-8").strip())
    assert rec["ok"] is False
    assert rec["in_tok"] == 1  # side-channel was set before the raise


# -- 6. thread-safety: concurrent records -> no interleaved lines --
def test_record_is_thread_safe(telemetry_on) -> None:
    n = 50

    def _writer(i):
        mod._telemetry.record(f"t{i}", float(i), i, i * 2, True, idx=i)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = Path(telemetry_on).read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == n  # exactly n lines, no interleaving / loss
    # Every line is valid JSON (no half-written lines).
    for ln in lines:
        json.loads(ln)


# -- 7. configure_telemetry swaps the sink; null singleton never writes --
def test_null_sink_never_writes(tmp_path) -> None:
    sink = LLMTelemetry()  # no path -> null sink
    sink.record("t", 1.0, 1, 1, True)
    # No file created (no path to write to).
    assert not list(tmp_path.glob("*"))


# -- 8. extra kwargs land in the record --
def test_extra_kwargs_in_record(telemetry_on) -> None:
    mod._telemetry.record("compaction_drift", None, None, None, True,
                          anchor_id=42, drift=0.83, count=2)
    rec = json.loads(Path(telemetry_on).read_text(encoding="utf-8").strip())
    assert rec["task"] == "compaction_drift"
    assert rec["drift"] == 0.83
    assert rec["count"] == 2
    assert rec["anchor_id"] == 42
    assert rec["in_tok"] is None