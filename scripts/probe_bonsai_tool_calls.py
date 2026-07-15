"""Probe: does the local Bonsai server (Prism-fork llama-server) support
OpenAI-style tool-calling -- i.e. does it accept a ``tools`` param and return
``choices[0].message.tool_calls``?

This is the empirical check for risk R-bonsai-tool-call-support (the MAIN risk
of the Phase 2c+ feedback-salience slice). The slice's self-chat path emits
``record_feedback`` as a Bonsai tool call; when Bonsai returns no ``tool_calls``
(unsupported server/model), a structured JSON-array fallback fires. This probe
tells you which path live self-chat will actually take.

Run AFTER starting the local Bonsai server (memory hippo-bonsai-local-server:
clone PrismML-Eng/Bonsai-demo, prebuilt CUDA12.4 binary, cold-start ~18s/shape).
Pre-warm before pytest.

    python scripts/probe_bonsai_tool_calls.py

Exit code 0 = server reachable (prints the verdict); 1 = server unreachable.
"""

from __future__ import annotations

import json
import sys

import requests

from src.config import config
from src.tools import SELF_CHAT_TOOLS, TOOL_SCHEMAS


def _post(endpoint: str, model: str, messages: list[dict],
          tools=None, tool_choice=None) -> dict:
    payload: dict = {"model": model, "messages": messages, "temperature": 0.0,
                     "max_tokens": 512}
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    r = requests.post(endpoint.rstrip("/") + "/chat/completions",
                      json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    endpoint = config.bonsai_endpoint
    model = config.generation_model
    print(f"[probe] endpoint={endpoint} model={model}")

    # 1. Reachability.
    try:
        up = requests.get(endpoint.rstrip("/") + "/models", timeout=5.0)
        print(f"[probe] GET /models -> {up.status_code}")
    except Exception as e:  # noqa: BLE001
        print(f"[probe] SERVER UNREACHABLE: {type(e).__name__}: {e}")
        print("[probe] Start the local Bonsai server first (see memory "
              "hippo-bonsai-local-server).")
        return 1

    # A prompt engineered to elicit a record_feedback call: it names the exact
    # tool + the cited unit id + asks for a 1-5 rating, so a tool-capable server
    # has every reason to emit the call.
    messages = [
        {"role": "system", "content": "You rate memory units for usefulness."},
        {"role": "user", "content": (
            "You just answered using context unit ep_001. Call the "
            "record_feedback tool with unit_id='ep_001' and rating=5 to report "
            "that it was essential. Reply with ONLY the tool call."
        )},
    ]

    verdicts = []

    # 2. tool_choice='auto' (what self-chat uses).
    print("\n[probe] === A. tools + tool_choice='auto' (self-chat path) ===")
    try:
        resp = _post(endpoint, model, messages, tools=SELF_CHAT_TOOLS,
                     tool_choice="auto")
        msg = resp.get("choices", [{}])[0].get("message", {})
        finish = resp.get("choices", [{}])[0].get("finish_reason")
        tcs = msg.get("tool_calls")
        print(f"  finish_reason: {finish}")
        print(f"  content: {msg.get('content', '')!r}")
        print(f"  tool_calls: {json.dumps(tcs, indent=2) if tcs else None}")
        verdicts.append(("auto", bool(tcs), tcs))
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {type(e).__name__}: {e}")
        verdicts.append(("auto", False, f"error: {e}"))

    # 3. tool_choice='required' (forces a call if the server honors it).
    print("\n[probe] === B. tools + tool_choice='required' ===")
    try:
        resp = _post(endpoint, model, messages, tools=SELF_CHAT_TOOLS,
                     tool_choice="required")
        msg = resp.get("choices", [{}])[0].get("message", {})
        finish = resp.get("choices", [{}])[0].get("finish_reason")
        tcs = msg.get("tool_calls")
        print(f"  finish_reason: {finish}")
        print(f"  tool_calls: {json.dumps(tcs, indent=2) if tcs else None}")
        verdicts.append(("required", bool(tcs), tcs))
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {type(e).__name__}: {e}")
        verdicts.append(("required", False, f"error: {e}"))

    # 4. Full schema set (what the external consumer uses).
    print("\n[probe] === C. full TOOL_SCHEMAS (record_feedback+expand+search_memory) ===")
    try:
        resp = _post(endpoint, model, messages, tools=TOOL_SCHEMAS,
                     tool_choice="auto")
        msg = resp.get("choices", [{}])[0].get("message", {})
        tcs = msg.get("tool_calls")
        print(f"  tool_calls: {json.dumps(tcs, indent=2) if tcs else None}")
        verdicts.append(("full", bool(tcs), tcs))
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {type(e).__name__}: {e}")
        verdicts.append(("full", False, f"error: {e}"))

    # Verdict.
    print("\n[probe] === VERDICT ===")
    any_supported = any(v for _, v, _ in verdicts if isinstance(v, bool))
    if any_supported:
        print("  Bonsai SUPPORTS tool-calling on at least one path -> self-chat "
              "will use the record_feedback tool-call path.")
        for name, ok, _ in verdicts:
            if isinstance(ok, bool):
                print(f"    {name}: {'SUPPORTED' if ok else 'no tool_calls'}")
    else:
        print("  Bonsai did NOT return tool_calls on any path -> self-chat "
              "falls back to the structured JSON-array rating call. The "
              "feedback loop still works; only the emission mechanism differs.")
        for name, _, detail in verdicts:
            print(f"    {name}: {detail if not isinstance(detail, bool) else 'no tool_calls'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())