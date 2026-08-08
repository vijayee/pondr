"""The collapse (exp #4 follow-on -- [[pondr-mamba3-cross-turn-carry-result]],
[[pondr-fade-architecture-router]]).

The keystone premise of the fade architecture ("Mamba3 has within-context memory
but ZERO cross-context memory -> need the bge-EWMA leg + WaveDB sidechannels to
carry across turns") is FALSIFIED by exp #4: ONE ``InferenceParams`` kept alive
across N turns carries a needle losslessly to the ~2048-token training-ctx
ceiling. So within the window, SSM-A (the bge EWMA ``VectorCarrySSM`` +
``BlurbStore`` + cosine router + R1/R3/R4 cascade) is REDUNDANT -- Mamba3's
carried state IS the within-window cross-turn memory.

The collapse ships additive/default-OFF (``FadeConfig.collapse``): when on, the
SSM-A ingest/recall path is SUPPRESSED and the carried Mamba3 is the sole
within-window memory (beyond the window is the orchestrator's WaveDB retriever --
a separate path this memory never owned). The cosine router is DROPPED, not
redesigned: it exists only because SSM-A's state is a bge vector you can
``cos(state, bge(anchor))``; with SSM-A gone there is nothing to route -- the
within-window recall is UNCONDITIONAL carry. SSM-A stays intact for the OFF
baseline (collapse disables, does not delete) so dual-SSM vs collapse is A/B-able.

These tests pin the wiring without torch/HF, reusing the stub ``CarryVoice`` /
embedder from ``tests.test_fade_voice_carry.py``:

- collapse ON + carry ON: ``ingest`` does NOT touch ``ssm_a`` / ``blurbs`` / ``ring``
  but DOES call ``voice.ingest_turn``; ``recall`` returns exactly ONE
  ``REGIME_CARRY`` recall (anchor_id=-1) after an ingest, ``[]`` before any ingest
  (turn 1).
- collapse ON: ``fading_anchors()`` returns ``[]`` (consolidation inert -- the
  ``blurbs`` store is empty, so there is nothing to gist).
- collapse ON with no ``CarryVoice`` (``voice=None``): ``recall`` returns ``[]``,
  no crash.
- collapse OFF: byte-identical to today (the SSM-A regime recalls + prepended
  carry when carry on).
- ``format_fade_block`` on a collapse-style ``[carry]``-only list renders the
  single ``[carry, in-context]`` line and is never dropped by the A4 cascade.
"""

from __future__ import annotations

import numpy as np

from src.subconscious.fade import (
    FadeConfig,
    FadeMemory,
    REGIME_CARRY,
    REGIME_GIST,
    REGIME_VERBATIM,
    format_fade_block,
)
from tests.test_fade_voice_carry import _StubCarryVoice, _StubEmbedder, _cfg


# ---------------------------------------------------------------------------
# collapse ON: ingest suppresses the SSM-A path, fires carry only
# ---------------------------------------------------------------------------

def test_collapse_ingest_suppresses_ssm_a_but_fires_carry():
    """collapse ON: ``ingest`` does NOT step SSM-A, does NOT add a blurb, does NOT
    push to the ring -- but DOES forward the turn to ``voice.ingest_turn`` (the
    carried state is the sole within-window memory). Returns -1 (the carry
    sentinel; there is no SSM-A anchor_id under collapse)."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(collapse=True, voice_carry=True),
                     _StubEmbedder(), voice)
    aid = mem.ingest("User: the passphrase is SQUEAKY-RAVEN-42\nAssistant: noted")
    # the carry leg fired.
    assert voice.ingest_calls == [("User: the passphrase is SQUEAKY-RAVEN-42\n"
                                  "Assistant: noted", 2048)]
    assert voice._carry_seqlen > 0
    # the SSM-A leg did NOT fire: no vector stepped, no blurb stored, no ring slot.
    assert len(mem.blurbs) == 0
    assert np.all(mem.ssm_a.state() == 0)
    assert len(mem.ring) == 0
    assert mem._next_id == 0       # the id counter never advanced
    assert aid == -1               # the carry sentinel


def test_collapse_recall_returns_single_carry_after_ingest():
    """collapse ON: after a turn is ingested, ``recall`` returns EXACTLY ONE
    recall -- the ``REGIME_CARRY`` carry (anchor_id=-1, cos=1.0). No SSM-A regime
    recalls (verbatim/gist/forgotten) are present -- the regime path is skipped."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(collapse=True, voice_carry=True),
                     _StubEmbedder(), voice)
    mem.ingest("User: the passphrase is SQUEAKY-RAVEN-42\nAssistant: noted")
    results = mem.recall("what was the passphrase", top_k=5)
    assert len(results) == 1
    assert results[0].regime == REGIME_CARRY
    assert results[0].anchor_id == -1
    assert results[0].cos == 1.0
    assert "SQUEAKY-RAVEN-42" in results[0].content
    # no SSM-A regime recalls leaked through.
    assert not any(r.regime in (REGIME_VERBATIM, REGIME_GIST)
                   for r in results)


def test_collapse_recall_turn_one_is_empty():
    """collapse ON, turn 1 (no ingest yet -> ``_carry_seqlen == 0``): ``recall``
    returns ``[]``. The carry guard skips (nothing carried yet), and the SSM-A
    path is suppressed -- so there is no fallback regime recall either."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(collapse=True, voice_carry=True),
                     _StubEmbedder(), voice)
    assert mem.recall("anything", top_k=5) == []
    assert voice.recall_calls == []      # the carry guard never cued


def test_collapse_multi_turn_stays_single_carry():
    """collapse ON across several turns: ``recall`` is still exactly ONE carry
    recall (the carried state accumulates the conversation; recall surfaces one
    continuation, not one-per-turn). The SSM-A store stays empty throughout."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(collapse=True, voice_carry=True),
                     _StubEmbedder(), voice)
    mem.ingest("turn one: the plate is ABC-9921")
    mem.ingest("turn two: filler about the weather")
    mem.ingest("turn three: more filler")
    assert len(mem.blurbs) == 0           # SSM-A still empty after 3 turns
    results = mem.recall("what was the plate", top_k=5)
    assert len(results) == 1
    assert results[0].regime == REGIME_CARRY
    # the stub surfaces the last-ingested turn (it does not decode a real
    # continuation); the point under test is that recall stays ONE carry entry
    # across multiple turns, not one-per-turn.
    assert results[0].content == "turn three: more filler"


# ---------------------------------------------------------------------------
# collapse ON: consolidation is inert (fading_anchors == [])
# ---------------------------------------------------------------------------

def test_collapse_fading_anchors_is_empty():
    """collapse ON: ``fading_anchors()`` returns ``[]`` because the ``blurbs``
    store is never populated (the SSM-A ingest path is skipped). The
    consolidation worker has nothing to gist -- it is inert under collapse. This
    is the property that makes collapse safe to ship without touching the
    consolidation worker: it iterates an empty ``blurbs._ids`` naturally."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(collapse=True, voice_carry=True),
                     _StubEmbedder(), voice)
    for i in range(6):
        mem.ingest(f"turn {i}")
    # the SSM-A store is empty -> fading_anchors has nothing to enumerate.
    assert len(mem.blurbs) == 0
    assert mem.fading_anchors(epsilon=0.03, max_depth=3, max_per_tick=8) == []


# ---------------------------------------------------------------------------
# collapse ON with no CarryVoice: no crash, no carry
# ---------------------------------------------------------------------------

def test_collapse_no_carry_voice_returns_empty_no_crash():
    """collapse ON with ``voice=None`` (no ``CarryVoice``): the ``hasattr`` guard
    skips the carry leg, and the SSM-A leg is suppressed -> ``recall`` returns
    ``[]`` with no crash. (``build_ponder`` separately requires the mamba3 backend
    under collapse; this pins the FadeMemory-level no-crash contract.)"""
    mem = FadeMemory(_cfg(collapse=True, voice_carry=True),
                     _StubEmbedder(), None)
    mem.ingest("User: hi\nAssistant: hello")   # no ingest_turn -> no-op, no raise
    assert mem.recall("anything", top_k=5) == []
    assert len(mem.blurbs) == 0                 # SSM-A still suppressed


# ---------------------------------------------------------------------------
# collapse OFF: byte-identical to today
# ---------------------------------------------------------------------------

def test_collapse_off_byte_identical_to_default():
    """collapse OFF (explicit ``collapse=False``) and the ``FadeConfig`` default
    produce identical ``recall()`` output for the same ingest sequence. Pins the
    default-OFF contract: the collapse is invisible when off (the SSM-A regime
    path runs unchanged, carry rides alongside as today)."""
    def _run(cfg):
        mem = FadeMemory(cfg, _StubEmbedder(), _StubCarryVoice())
        mem.ingest("docA:0")
        mem.ingest("docB:0")
        for i in range(6):
            mem.ingest(f"doc{i+1}:0")
        return mem.recall("docA:0", top_k=4)

    cfg_off = _cfg(collapse=False, voice_carry=True)
    cfg_default = _cfg(voice_carry=True)            # collapse defaults False
    assert cfg_default.collapse is False
    off = _run(cfg_off)
    default = _run(cfg_default)
    assert off == default
    # the SSM-A path ran (anchors were created) in both.
    assert len(_run(cfg_off)) > 0


def test_collapse_off_runs_ssm_a_path():
    """collapse OFF: the SSM-A ingest path runs as today -- anchors are created,
    SSM-A steps, the ring fills. (Negative control for the collapse-ON
    suppression tests above.)"""
    mem = FadeMemory(_cfg(collapse=False, voice_carry=True),
                     _StubEmbedder(), _StubCarryVoice())
    mem.ingest("docA:0")
    assert mem._next_id == 1
    assert len(mem.blurbs) == 1
    assert np.any(mem.ssm_a.state() != 0)
    assert len(mem.ring) == 1


# ---------------------------------------------------------------------------
# format_fade_block: the collapse-style [carry]-only list
# ---------------------------------------------------------------------------

def _r(eid, regime, content, *, cos_q=0.0, cos=0.5, name=None) -> dict:
    names = {REGIME_VERBATIM: "verbatim", REGIME_GIST: "gist",
             REGIME_CARRY: "carry"}
    return {"anchor_id": eid, "regime": regime,
            "regime_name": name or names.get(regime, "?"),
            "cos": cos, "cos_q": cos_q, "content": content, "blurb": None}


def test_format_fade_block_collapse_only_carry():
    """A collapse-style recall list (one ``[carry]`` entry, no R1/R3) renders as
    a single ``[carry, in-context]`` line under the ``[FADE MEMORY]`` header."""
    recalls = [_r(-1, REGIME_CARRY, "the plate was ABC-9921", cos=1.0)]
    block = format_fade_block(recalls)
    assert "[FADE MEMORY" in block
    assert "[carry, in-context] the plate was ABC-9921" in block
    # no regime lines leaked (there are none to render).
    assert "[verbatim, recent]" not in block
    assert "[gist, fading]" not in block


def test_cascade_keeps_carry_only_list_within_budget():
    """The A4 budget cascade never drops the single carry line from a
    collapse-style list: carry has the lowest drop-priority (regime -> -1), and a
    one-entry list fits any non-zero budget. The collapsed recall is never
    silently truncated away."""
    long = "x" * 400
    recalls = [_r(-1, REGIME_CARRY, f"carry-needle {long}", cos=1.0)]
    block = format_fade_block(recalls, max_tokens=300)
    assert "carry-needle" in block
    assert "[carry, in-context]" in block


# ---------------------------------------------------------------------------
# end-to-end: collapse recall flows into the fade block
# ---------------------------------------------------------------------------

def test_collapse_recall_flows_into_fade_block():
    """The single ``REGIME_CARRY`` recall ``FadeMemory.recall`` produces under
    collapse flows through the orchestrator-seam dict shape into
    ``format_fade_block`` unchanged (the orchestrator builds ``fade_recalls`` from
    the same attributes -- a collapse recall is just ``[{regime:5}]``)."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(collapse=True, voice_carry=True,
                          fade_block_max_tokens=4000),
                     _StubEmbedder(), voice)
    mem.ingest("User: the passphrase is SQUEAKY-RAVEN-42\nAssistant: noted")
    recalls = mem.recall("what was the passphrase", top_k=5)
    dict_recalls = [
        {"anchor_id": r.anchor_id, "regime": r.regime,
         "regime_name": {REGIME_CARRY: "carry",
                         REGIME_VERBATIM: "verbatim",
                         REGIME_GIST: "gist"}[r.regime],
         "cos": r.cos, "cos_q": r.cos_q, "content": r.content, "blurb": r.blurb}
        for r in recalls
    ]
    block = format_fade_block(dict_recalls, max_tokens=4000)
    assert "[carry, in-context]" in block
    assert "SQUEAKY-RAVEN-42" in block