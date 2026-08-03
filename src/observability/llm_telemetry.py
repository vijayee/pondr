"""A5: non-invasive per-LLM-call telemetry (Tencent-survey Phase 1 item 5).

A JSONL sink + decorator that records one line per Bonsai judge call -- the
cheap observability layer Pondr lacked. ``BonsaiDecider._post_json`` funnels all
9 judge calls (verify_fidelity, author_scene, judge_dedup_pairs, verify_typing,
decide_anomaly, decide_contradiction, classify_doc_kind, gist,
consolidate_gist); the OpenAI-compatible ``llama-server`` response carries
``outer["usage"] = {prompt_tokens, completion_tokens, total_tokens}`` which was
DISCARDED. The decorator surfaces it alongside latency, per task.

Adapted from Tencent's ``MetricTrackingRunner.run`` (``metric-tracking-runner.ts:
179-324``): the inner call is awaited OUTSIDE any recording try/except (errors
propagate to the caller unchanged), then the report is emitted in a ``finally``
block wrapped in its OWN try/except -- **never throws, never changes the return
value, never changes the signature** (``functools.wraps``). Tencent's sink is
Kafka/OTel; Pondr adapts to a JSONL file sink (the ~1500 lines of
Kafka/ClickHouse/OTLP/Langfuse exporter glue are SKIPPED per the survey).

The field vocabulary mirrors ``OracleClient`` (``src/training/oracle_labeling.py``
-- ``input_tokens`` / ``output_tokens`` / latency) so the two telemetry streams
read together; the JSONL keys are ``in_tok`` / ``out_tok`` / ``latency_ms`` for
brevity.

Byte-identical OFF: the decorator reads ``config.llm_telemetry_enabled`` at call
time (master-config style 1, mirrors ``dedup_enabled`` / ``hybrid_retrieval``).
Flag off -> direct ``return fn(...)`` BEFORE timing -> zero overhead, no file,
no behavior change. ``configure_telemetry`` (called by ``build_ponder`` when the
flag is on) swaps the module singleton to a path-backed sink; the default null
sink (no path) records nothing, so flag-on-without-configure is a silent no-op,
not a crash.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = ["LLMTelemetry", "configure_telemetry", "record_llm_call"]


class LLMTelemetry:
    """Append-only JSONL of per-LLM-call telemetry.

    Thread-safe (one ``threading.Lock`` guards the append); never raises (every
    op in try/except). A null sink (``path=None``) records nothing -- the flag-on
    / not-configured case is a silent no-op, not a crash. The parent directory
    is created best-effort on construction.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        self._lock = threading.Lock()
        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:  # noqa: BLE001 - best-effort; record() no-ops on a missing dir
                self._path = None

    def record(self, task: str, latency_ms: Optional[float],
               input_tokens: Optional[int], output_tokens: Optional[int],
               ok: bool = True, **extra: Any) -> None:
        """Append one JSONL line. Never raises -- a telemetry hiccup must never
        break the call it wraps. Null sink (no path) -> no-op."""
        if self._path is None:
            return
        try:
            rec = {
                "ts": time.time(),
                "task": task,
                "latency_ms": latency_ms,
                "in_tok": input_tokens,
                "out_tok": output_tokens,
                "ok": ok,
            }
            if extra:
                rec.update(extra)
            line = json.dumps(rec, separators=(",", ":"))
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:  # noqa: BLE001 - never raises (the contract)
            pass


# Module singleton. Null sink (no path) until ``configure_telemetry`` swaps it.
_telemetry = LLMTelemetry()


def configure_telemetry(path: str) -> None:
    """Point the module sink at a JSONL ``path`` (called by ``build_ponder``
    when ``llm_telemetry`` is on). Replacing the global is safe -- the decorator
    reads ``_telemetry`` at record time (``finally``), so a mid-run reconfigure
    takes effect on the next call."""
    global _telemetry
    _telemetry = LLMTelemetry(path)


def record_llm_call(task_name: str) -> Callable[[Callable], Callable]:
    """Decorator factory: wrap an LLM-judge method with one JSONL record.

    * **Off** (``config.llm_telemetry_enabled`` False) -> direct passthrough,
      zero overhead (one config-flag read before the call; no timing, no file).
    * **On** -> time the call (``time.perf_counter``), read the side-channel
      ``self._last_usage`` (set by ``_post_json`` on its success path), append
      one record in ``finally``. Recording is wrapped in its OWN try/except ->
      never raises, never mutates the return.
    * ``functools.wraps`` -> signature + docstring preserved.

    The inner call is awaited OUTSIDE the recording try/except (Tencent's
    pattern): the ``except Exception: ok = False; raise`` re-raises the original
    error unchanged, THEN ``finally`` records ``ok=False``. A cold-start
    ``None`` return (the judge's fallback) records ``ok=True`` with null tokens
    (``_last_usage`` is None when ``_post_json`` returned None, or when a
    deterministic pre-filter short-circuited before it -- e.g.
    ``decide_contradiction``'s ``_deterministic_non_conflict``).
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            from ..config import config as _master_config
            if not _master_config.llm_telemetry_enabled:
                return fn(self, *args, **kwargs)
            t0 = time.perf_counter()
            ok = True
            try:
                return fn(self, *args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                try:
                    usage = getattr(self, "_last_usage", None) or {}
                    _telemetry.record(
                        task_name,
                        (time.perf_counter() - t0) * 1000.0,
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        ok,
                    )
                except Exception:  # noqa: BLE001 - never raises (the contract)
                    pass

        return wrapper

    return deco