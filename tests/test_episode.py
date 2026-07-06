"""Unit tests for the Episode data model.

These run offline — no GLiNER/Bonsai/WaveDB-GPU needed (only the dataclass).
"""

from src.memory.episode import Episode


def test_episode_creation():
    """Episode can be created with minimal fields."""
    ep = Episode(
        id="ep_001",
        timestamp="2026-07-03T10:00:00",
        summary="Test",
        full_text="User: Hi\nAssistant: Hello",
    )
    assert ep.id == "ep_001"
    assert ep.state == "current"
    assert ep.salience == 0.5
    # validity_start defaults to the episode timestamp (a fact's validity
    # begins when it was encoded).
    assert ep.validity_start == "2026-07-03T10:00:00"


def test_episode_from_extraction():
    """Episode.from_extraction creates a valid episode from extraction results."""
    ep = Episode.from_extraction(
        episode_id="ep_001",
        user_message="Hello",
        assistant_response="Hi there!",
        extracted={"entities": ["User"], "topics": ["test"], "tones": ["neutral"], "decisions": []},
        relations=[],
        timestamp="2026-07-03T10:00:00",
    )
    assert ep.full_text == "User: Hello\nAssistant: Hi there!"
    assert "Hi there!" in ep.summary
    assert ep.entities == ["User"]
    assert ep.topics == ["test"]
    assert ep.tones == ["neutral"]
    assert ep.relations == []
    assert ep.follows is None


def test_episode_from_extraction_follows_chain():
    """from_extraction threads `follows` for the conversation chain."""
    ep = Episode.from_extraction(
        episode_id="ep_002",
        user_message="Next",
        assistant_response="reply",
        extracted={"entities": [], "topics": [], "tones": [], "decisions": []},
        relations=[],
        follows="ep_001",
        timestamp="2026-07-03T10:00:01",
    )
    assert ep.follows == "ep_001"


def test_episode_summary_truncation():
    """Long assistant responses are truncated to 200 chars + ellipsis."""
    long_response = "x" * 500
    ep = Episode.from_extraction(
        episode_id="ep_001",
        user_message="q",
        assistant_response=long_response,
        extracted={"entities": [], "topics": [], "tones": [], "decisions": []},
        relations=[],
        timestamp="2026-07-03T10:00:00",
    )
    assert ep.summary == "x" * 200 + "..."