"""Live dogfood: build_ponder serves a real query on the TRAINED models + live-encodes.

The end-to-end proof that the SSM/JEPA consistency gap is closed: the trained
Phase 2a backbone + Phase 2b gate (loaded from disk by ``build_ponder``) serve a
live query against the local 8B Bonsai, the self-chat tool loop runs on the
trained gate+backbone, and -- with ``live_encode=True`` -- the exchange is
persisted as an episode (GLiNER-on-device, CUDA with an OOM-safe CPU fallback).

Skips gracefully when the full live stack is unavailable: Bonsai down, the
trained checkpoints missing, or the heavy deps (bge sentence-transformers +
gliner/gliner2) / their model files not present. Pre-warm Bonsai first
(PTX-JIT cold-start ~18s/shape; see memory ``hippo-bonsai-local-server``).

``end_state="synthesize"`` is passed explicitly so the synthesize path (LLM
call -> self-chat loop -> live-encode persist) runs deterministically regardless
of the trained gate's end-state prediction; the gate still does retrieval
routing (route_text) over the trained backbone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src.config import config
from src.runtime import DEFAULT_BACKBONE_PATH, DEFAULT_GATE_PATH, build_ponder

REPO_ROOT = Path(__file__).resolve().parent.parent


def _have(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


@pytest.fixture(scope="module")
def ponder_live(tmp_path_factory):
    # Bonsai up?
    url = config.bonsai_endpoint.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable: {e}")

    if not (_have(DEFAULT_BACKBONE_PATH) and _have(DEFAULT_GATE_PATH)):
        pytest.skip("trained checkpoints not present locally")

    # Heavy deps + model files (bge + gliner/gliner2). build_ponder with
    # live_encode=True constructs HippocampalEncoder -> GLiNERExtractor, and
    # embedder_source="on-demand" loads bge-small. Any of these missing ->
    # skip (the full live stack isn't set up), rather than hard-fail.
    tmp = tmp_path_factory.mktemp("serve_ponder")
    try:
        orch = build_ponder(
            str(tmp / "memory_db"),
            backbone_path=DEFAULT_BACKBONE_PATH,
            gate_path=DEFAULT_GATE_PATH,
            embedder_source="on-demand",
            device="auto",
            gliner_device="auto",
            live_encode=True,
        )
    except Exception as e:  # noqa: BLE001 - live setup unavailable -> skip
        pytest.skip(f"live build_ponder unavailable (deps/models?): {e}")
    yield orch
    try:
        orch.store.close()
    except Exception:  # noqa: BLE001
        pass


def test_serve_ponder_live_query_runs_on_trained_models(ponder_live):
    orch = ponder_live
    # Force the synthesize path so the LLM call + self-chat loop + live-encode
    # persist all run deterministically (the trained gate still routes retrieval).
    res = orch.query("What did we decide about the memory store, in short?",
                     end_state="synthesize")
    assert res["end_state_plan"].end_state == "synthesize"
    assert isinstance(res.get("response"), str) and res["response"].strip()
    # The self-chat tool loop ran on the trained gate + backbone.
    assert "loop_tool_messages" in res
    assert "loop_collected" in res
    assert "loop_exhausted" in res
    names = [c.get("name") for c in res["loop_collected"]]
    assert all(n in ("expand", "search_memory", "record_feedback") for n in names)
    assert "tool_calls" not in (res["response"] or "")
    # Live-encode persisted the exchange as an episode (GLiNER-on-device ran).
    assert res.get("persisted_episode_id"), "live-encode did not persist an episode"