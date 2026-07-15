"""Live dogfood tests for ``BonsaiDecider`` against the local Bonsai server.

Skipped automatically when the endpoint (``config.bonsai_endpoint`` /
``localhost:8080/v1``) is unreachable, via the ``GET /v1/models`` probe (the
established skip guard -- mirrors ``test_bonsai_relations.py``). Run with the
8B Bonsai server up (see memory ``hippo-bonsai-local-server``); pre-warm to
avoid PTX-JIT cold-start stalls.
"""

from __future__ import annotations

import pytest
import requests

from src.config import config
from src.gnn.bonsai_decider import BonsaiDecider


@pytest.fixture(scope="module")
def decider_live():
    url = config.bonsai_endpoint.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable: {e}")
    return BonsaiDecider()


def test_health_check_live(decider_live):
    assert decider_live.health_check() is True


def test_gist_live(decider_live):
    sources = [
        {"id": "ep_1", "summary": "Alice decided to use WaveDB for the vector store.",
         "text": "We picked WaveDB because the in-DB vector layer avoids a sidecar."},
        {"id": "ep_2", "summary": "Alice and Bob benchmarked WaveDB vs FAISS.",
         "text": "FLAT/COSINE was exact and fast enough at this scale."},
    ]
    g = decider_live.gist(sources)
    assert g is not None, "Bonsai returned no gist"
    assert isinstance(g, str) and len(g) > 10
    # Control chars stripped (defensive -- the gist is stored).
    assert all(ord(c) >= 0x20 or c in "\n\t" for c in g)


def test_decide_anomaly_live(decider_live):
    flag = {"node": "E:Alice", "type": "identity_drift",
            "evidence": "disjoint topic neighborhoods"}
    ctx = {"entity": "E:Alice", "states": ["active", "retired"],
           "episodes": [{"id": "ep_1", "summary": "coding work",
                         "timestamp": "2026-01-01"},
                        {"id": "ep_2", "summary": "parenting chat",
                         "timestamp": "2026-01-02"}],
           "topics": ["db", "family"], "instance_of": ["Person"]}
    d = decider_live.decide_anomaly(flag, ctx)
    assert d is not None, "Bonsai returned no anomaly decision"
    assert d["decision"] in ("fix", "ask_user", "dismiss")
    assert isinstance(d["action"], str)