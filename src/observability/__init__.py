"""Observability layer: non-invasive per-LLM-call telemetry + memory-system
signals. ``llm_telemetry`` records one JSONL line per Bonsai judge call
(input/output tokens + latency); ``drift`` lives under ``subconscious`` (it is
a memory-system signal, not a generic observability util). See A5
([[pondr-tencent-agent-memory-survey]] Phase 1 item 5)."""