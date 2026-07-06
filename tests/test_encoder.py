"""Integration tests for the HippocampalEncoder pipeline.

End-to-end: GLiNER extract → Bonsai relations → Episode → atomic WaveDB write.
Requires the GLiNER models and the Bonsai llama-server endpoint (RunPod GPU
pod). Skipped when ``gliner``/``gliner2`` aren't importable so the offline
suite stays green; the live Bonsai endpoint is checked per-test and skipped if
down.
"""

import pytest

pytest.importorskip("gliner")
pytest.importorskip("gliner2")

import requests  # noqa: E402

from src.config import config  # noqa: E402
from src.encoding.encoder import HippocampalEncoder  # noqa: E402
from src.memory.store import HippocampalStore  # noqa: E402


@pytest.fixture
def live_encoder(tmp_path):
    """Encoder backed by a real store + live extractors, scoped to a test user.

    Skips the whole suite if the Bonsai endpoint is unreachable — the GLiNER
    importorskip above already gated on the GLiNER packages.
    """
    url = config.bonsai_endpoint.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable: {e}")
    store = HippocampalStore(str(tmp_path / "enc_db"))
    yield HippocampalEncoder(store, user_id="test_user")
    store.close()


def test_encode_single_turn(live_encoder):
    """Encoder processes a single conversation turn end-to-end (one session)."""
    episodes = live_encoder.encode_conversation([
        ("I'm frustrated with the WAL config. Why are there three modes?",
         "IMMEDIATE is safest but slowest, DEBOUNCED is the sweet spot, ASYNC is fastest."),
    ])
    ep = episodes[0]
    assert ep.id.startswith("ep_")
    assert ep.user_id == "test_user"
    assert ep.session_id is not None and ep.session_id.startswith("S:")
    assert "configuration" in ep.topics, ep.topics

    loaded = live_encoder.store.get_episode(ep.id)
    assert loaded is not None
    assert loaded.summary == ep.summary
    # The episode is listed under its session.
    assert ep.id in live_encoder.store.list_session_episodes(ep.session_id)


def test_encode_conversation_chain(live_encoder):
    """Multiple turns form a follows chain within one session."""
    turns = [
        ("What's HBTrie?", "HBTrie is a hierarchical B+tree..."),
        ("How does it compare to B+tree?", "Each level is itself a B+tree..."),
        ("I'll use it then.", "Great choice."),
    ]
    episodes = live_encoder.encode_conversation(turns)

    assert len(episodes) == 3
    assert episodes[0].follows is None
    assert episodes[1].follows == episodes[0].id
    assert episodes[2].follows == episodes[1].id
    # All three share one session.
    assert len({e.session_id for e in episodes}) == 1


def test_encode_sample_conversations(live_encoder):
    """All 20 sample conversations encode without errors."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "sample_conversations.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)
            episodes = live_encoder.encode_conversation(conv["turns"])
            assert len(episodes) == len(conv["turns"])