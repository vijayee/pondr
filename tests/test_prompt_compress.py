"""Offline tests for the Phase 2c prompt compression (Task 5).

Verifies: short prompts pass through byte-identical; long prompts compress to
≤ bonsai_max_input; the WM preamble appears; key spans are extracted; the hard
cap is enforced. No GLiNER (cheap path is the default), no Bonsai.
"""

from __future__ import annotations

import torch

from src.config import Phase2cConfig
from src.retrieval.prompt_compress import compress_prompt_for_planning
from src.subconscious.state_serializer import JGSSnapshot


def _wm(meta=None) -> JGSSnapshot:
    return JGSSnapshot(
        state_tensors=[torch.zeros(1, 16, 384) for _ in range(4)],
        input_count=1, timestamp=0.0, metadata=meta or {},
    )


def _cfg(short=500, max_input=2000):
    cfg = Phase2cConfig()
    cfg.prompt_compression.short_prompt_threshold = short
    cfg.prompt_compression.bonsai_max_input = max_input
    return cfg


# ── short prompts pass through ──

def test_short_prompt_passes_byte_identical():
    p = "What did Alice say about Postgres?"
    assert compress_prompt_for_planning(p, config=_cfg()) == p


def test_short_at_threshold_passes():
    p = "x" * 500
    assert compress_prompt_for_planning(p, config=_cfg()) == p


# ── long prompts compress ──

def test_long_prompt_capped_to_max_input():
    p = "Alice talked to Bob about Postgres and MySQL and Redis. " * 200  # ~10k chars
    cfg = _cfg(max_input=2000)
    out = compress_prompt_for_planning(p, config=cfg)
    assert len(out) <= 2000


def test_long_prompt_includes_key_spans():
    p = "Alice talked to Bob about Postgres and MySQL and Redis. " * 200
    out = compress_prompt_for_planning(p, config=_cfg())
    # TitleCase spans (Alice, Bob, Postgres, MySQL, Redis) are extracted.
    assert "Alice" in out or "Postgres" in out
    assert "Key entities:" in out


def test_long_prompt_includes_wm_preamble_when_wm_given():
    p = "What did we decide about the database migration? " * 200
    wm = _wm(meta={"active_domains": ["database"], "last_query_type": "factual"})
    out = compress_prompt_for_planning(p, working_memory=wm, config=_cfg())
    assert "Active domains: database" in out
    assert "Recent focus: factual" in out


def test_no_wm_no_preamble():
    p = "Some long prompt without entities here. " * 200
    out = compress_prompt_for_planning(p, config=_cfg())
    assert "Active domains" not in out


def test_truncation_marker_present_when_prompt_exceeds_budget():
    # No TitleCase spans to extract → spans block empty → whole budget for raw.
    p = "lowercase only text no entities here at all. " * 300
    cfg = _cfg(max_input=400)
    out = compress_prompt_for_planning(p, config=cfg)
    assert len(out) <= 400
    assert "[...truncated]" in out


def test_no_gliner_falls_back_to_cheap_spans():
    # use_gliner=True but gliner not installed → must not raise, falls back.
    p = "Alice and Bob discussed Postgres. " * 200
    out = compress_prompt_for_planning(p, config=_cfg(), use_gliner=True)
    assert len(out) <= 2000
    assert "Postgres" in out or "Alice" in out


def test_bonsai_never_receives_over_max_input():
    p = "x" * 5000
    cfg = _cfg(max_input=1000)
    out = compress_prompt_for_planning(p, config=cfg)
    assert len(out) <= 1000


# ── planning accuracy guard ──

def test_compressed_preserves_entity_signal():
    """The compressed prompt must still let a planner find the key entity.

    A regression guard against over-compression (docs/Phase 2c.md §7.3): the
    entity the planner needs must survive compression.
    """
    p = ("Alice and Bob had a long discussion about migrating the Postgres "
         "database to a new cluster. " * 200)
    out = compress_prompt_for_planning(p, config=_cfg())
    # At least one of the planning-relevant entities survives.
    assert any(e in out for e in ("Alice", "Bob", "Postgres"))