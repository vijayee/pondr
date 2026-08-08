"""Cross-turn Mamba3 carry wiring (exp #4, [[pondr-mamba3-cross-turn-carry-result]]).

The probe kept ONE ``InferenceParams`` alive across N forward calls (turns) and
showed Mamba3 retains a needle losslessly across the per-call boundary -- the
fade architecture's keystone premise ("Mamba3 has zero cross-context memory")
is FALSIFIED. This change makes that finding real in serve: a per-conversation
carried state on ``Mamba3Voice``, authored by ``FadeMemory.ingest`` (the sole
writer) and read by ``FadeMemory.recall`` (clone-into-throwaway, read-only),
shipped additive and default-OFF.

These tests pin the wiring without torch/HF:

- ``ingest`` calls ``voice.ingest_turn`` with the turn text (carry on).
- ``recall`` prepends a ``REGIME_CARRY`` ``Recall`` (anchor_id=-1, cos=1.0)
  whose ``content`` is the decoded continuation, AFTER at least one ingest;
  skipped on turn 1 (nothing carried yet).
- ``voice_carry=False`` -> no ``ingest_turn`` / ``recall_from_carry`` calls and
  ``recall()`` byte-identical to today.
- a non-``CarryVoice`` (``voice=None`` or a plain ``Voice``) with carry on is a
  no-op (the ``hasattr`` guard) -> byte-identical to carry off.
- ``recall_from_carry`` is read-only: it does NOT advance the carried state
  (``ingest_turn`` is the sole writer).
- ``reset`` clears the carried state (no bleed into the next session).
- ``format_fade_block`` renders the ``[carry, in-context]`` line and the A4
  cascade NEVER drops carry before R3 (carry = lowest drop-priority).

The thing under test is the REAL ``FadeMemory`` carry hooks; the voice is a stub
``CarryVoice`` (in-memory, no torch) that records its calls.
"""

from __future__ import annotations

import hashlib

import pytest
import torch  # the real-Mamba3Voice carry tests compare cache tensors (clone/equal)

from src.subconscious.fade import (
    FadeConfig,
    FadeMemory,
    Mamba3Voice,
    REGIME_CARRY,
    REGIME_FORGOTTEN,
    REGIME_GIST,
    REGIME_VERBATIM,
    format_fade_block,
)
from tests.test_fade import _StubHFTokenizer, _StubMamba3Out


# -- deterministic embedder (SHA256 stretch -> normalized, 384-d) ---------------

class _StubEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            buf = bytearray()
            h = hashlib.sha256(t.encode("utf-8")).digest()
            counter = 0
            while len(buf) < self.dim:
                buf += hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
                counter += 1
            vec = [(b / 127.5 - 1.0) for b in buf[: self.dim]]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


# -- a stub CarryVoice (in-memory, no torch/HF) --------------------------------

class _StubCarryVoice:
    """A ``CarryVoice`` double: records calls, keeps an in-memory token count.

    Mirrors ``Mamba3Voice``'s carry contract:

    - ``ingest_turn`` is the SOLE writer (advances ``_carry_seqlen``).
    - ``recall_from_carry`` is READ-ONLY (does NOT advance ``_carry_seqlen``;
      it surfaces the last-ingested needle so a test can assert the content
      flows through the wiring).
    - ``reset_carry`` clears the state (``_carry_seqlen`` -> 0).
    - ``expand`` is the plain-``Voice`` R3 path (returns ``blurb + ' [expanded]'``).

    ``FadeMemory`` guards carry on ``hasattr(self.voice, "ingest_turn")`` and
    reads ``getattr(self.voice, "_carry_seqlen", 0)``, so the stub exposes the
    same ``_carry_seqlen`` attribute the real ``Mamba3Voice`` does.
    """

    def __init__(self) -> None:
        self.ingest_calls: list[tuple[str, int]] = []
        self.recall_calls: list[tuple[str, int]] = []
        self.expand_calls: list[str] = []
        self._carry_seqlen = 0          # the read-key FadeMemory.recall checks
        self._carried_text = ""         # last ingested turn (for recall content)

    # -- CarryVoice (the carry leg) ----------------------------------------
    def ingest_turn(self, text: str, max_tokens: int = 2048) -> int:
        self.ingest_calls.append((text, max_tokens))
        # One "token" per character keeps the budget math transparent; respect
        # the max_tokens cap the way Mamba3Voice does (stop advancing at budget).
        room = max_tokens - self._carry_seqlen
        n = min(len(text), max(room, 0))
        self._carry_seqlen += n
        self._carried_text = text        # the needle a recall surfaces
        return n

    def recall_from_carry(self, cue: str, max_new_tokens: int = 64) -> str:
        self.recall_calls.append((cue, max_new_tokens))
        # READ-ONLY: do NOT advance ``_carry_seqlen``. Return the carried text
        # so the test can assert the decoded continuation reaches the Recall.
        if self._carry_seqlen == 0 or not self._carried_text:
            return ""
        return self._carried_text

    def reset_carry(self) -> None:
        self._carry_seqlen = 0
        self._carried_text = ""

    # -- Voice (the R3 expansion leg) --------------------------------------
    def expand(self, blurb: str, max_new_tokens: int) -> str:
        self.expand_calls.append(blurb)
        return blurb + " [expanded]"


class _PlainVoice:
    """A plain ``Voice`` (expand only) -- NOT a ``CarryVoice``. Used to prove the
    ``hasattr`` guard makes carry a no-op for non-carry voices."""

    def __init__(self) -> None:
        self.expand_calls: list[str] = []

    def expand(self, blurb: str, max_new_tokens: int) -> str:
        self.expand_calls.append(blurb)
        return blurb + " [expanded]"


# -- helpers -------------------------------------------------------------------

def _cfg(**kw) -> FadeConfig:
    """A fade config with a low ``cos_gist`` for the synthetic embedder (whose
    cross-doc floor is ~0.01, unlike real bge's ~0.37)."""
    return FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=8,
                      **kw)


# ---------------------------------------------------------------------------
# ingest -> ingest_turn
# ---------------------------------------------------------------------------

def test_ingest_calls_ingest_turn_when_carry_on():
    """``voice_carry=True`` + a ``CarryVoice`` -> each ``ingest`` forwards the
    turn text to ``ingest_turn`` with ``voice_carry_max_tokens``."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(voice_carry=True, voice_carry_max_tokens=2048),
                     _StubEmbedder(), voice)
    mem.ingest("User: hi\nAssistant: hello")
    assert voice.ingest_calls == [("User: hi\nAssistant: hello", 2048)]
    # the carried state advanced (turn-1 recall will now fire).
    assert voice._carry_seqlen > 0


def test_ingest_does_not_call_ingest_turn_when_carry_off():
    """``voice_carry=False`` (the default) -> ``ingest`` never touches the carry
    leg. The SSM-A ingest + blurb + ring path is unchanged."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(), _StubEmbedder(), voice)   # voice_carry defaults False
    mem.ingest("User: hi\nAssistant: hello")
    assert voice.ingest_calls == []


def test_ingest_carry_failure_is_swallowed():
    """A carry failure is swallowed so the SSM-A ingest that just succeeded is
    not masked (the ``try/except`` in ``FadeMemory.ingest``)."""
    class _BoomVoice(_StubCarryVoice):
        def ingest_turn(self, text, max_tokens=2048):
            raise RuntimeError("carry exploded")
    voice = _BoomVoice()
    mem = FadeMemory(_cfg(voice_carry=True), _StubEmbedder(), voice)
    # No raise; the SSM-A ingest completes and the anchor is registered.
    aid = mem.ingest("docA:0")
    assert aid == 0
    assert mem.blurbs.text(0) is not None


# ---------------------------------------------------------------------------
# recall -> prepended REGIME_CARRY Recall
# ---------------------------------------------------------------------------

def test_recall_prepends_carry_recall_after_ingest():
    """After a turn is ingested, ``recall`` prepends a ``REGIME_CARRY`` recall
    (anchor_id=-1, cos=1.0) whose content is the carried continuation. It sits
    ABOVE the SSM-A regime recalls (within-window recent recall ranks first)."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(voice_carry=True), _StubEmbedder(), voice)
    mem.ingest("User: the passphrase is SQUEAKY-RAVEN-42\nAssistant: noted")
    results = mem.recall("what was the passphrase", top_k=3)
    # the carried recall is first.
    assert results[0].regime == REGIME_CARRY
    assert results[0].anchor_id == -1
    assert results[0].cos == pytest.approx(1.0)
    assert "SQUEAKY-RAVEN-42" in results[0].content
    # recall_from_carry was cued with the query text and the recall-token budget.
    assert voice.recall_calls == [("what was the passphrase",
                                   mem.cfg.voice_carry_recall_tokens)]
    # the rest are the SSM-A regime recalls (unchanged, still present).
    assert any(r.regime in (REGIME_VERBATIM, REGIME_GIST, REGIME_FORGOTTEN)
               for r in results[1:])


def test_recall_no_carry_on_turn_one():
    """Turn-1 recall (no ingest yet -> ``_carry_seqlen == 0``) -> no carried
    recall. The SSM-A regime recalls are the only output."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(voice_carry=True), _StubEmbedder(), voice)
    results = mem.recall("anything", top_k=3)
    assert voice.recall_calls == []
    assert not any(r.regime == REGIME_CARRY for r in results)


def test_recall_carry_off_has_no_carry_recall():
    """``voice_carry=False`` -> ``recall`` never calls ``recall_from_carry`` and
    emits no ``REGIME_CARRY`` recall."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(), _StubEmbedder(), voice)   # carry off
    mem.ingest("User: hi\nAssistant: hello")
    results = mem.recall("anything", top_k=3)
    assert voice.recall_calls == []
    assert not any(r.regime == REGIME_CARRY for r in results)


def test_recall_non_carry_voice_is_noop():
    """``voice_carry=True`` but the voice is NOT a ``CarryVoice`` (no
    ``ingest_turn``) -> the ``hasattr`` guard makes carry a no-op: no crash, no
    carry recall. ``voice=None`` likewise."""
    # a plain Voice (expand only) with carry on.
    plain = _PlainVoice()
    mem = FadeMemory(_cfg(voice_carry=True), _StubEmbedder(), plain)
    mem.ingest("User: hi\nAssistant: hello")   # no ingest_turn -> no-op
    results = mem.recall("anything", top_k=3)
    assert not any(r.regime == REGIME_CARRY for r in results)

    # voice=None with carry on (the Phase-A serve path).
    mem_none = FadeMemory(_cfg(voice_carry=True), _StubEmbedder(), None)
    mem_none.ingest("User: hi\nAssistant: hello")
    res_none = mem_none.recall("anything", top_k=3)
    assert not any(r.regime == REGIME_CARRY for r in res_none)


# ---------------------------------------------------------------------------
# the read-only invariant
# ---------------------------------------------------------------------------

def test_recall_from_carry_does_not_mutate_carried_state():
    """``recall_from_carry`` is READ-ONLY: calling it does NOT advance
    ``_carry_seqlen`` (``ingest_turn`` is the sole writer). This is the clone-
    into-throwaway invariant that keeps recall->ingest ordering clean."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(voice_carry=True), _StubEmbedder(), voice)
    mem.ingest("turn one")
    seqlen_after_ingest = voice._carry_seqlen
    assert seqlen_after_ingest > 0
    # recall cues the carried state but must not advance it.
    mem.recall("cue", top_k=3)
    assert voice._carry_seqlen == seqlen_after_ingest
    # a second recall still does not advance.
    mem.recall("cue again", top_k=3)
    assert voice._carry_seqlen == seqlen_after_ingest
    # only ingest advances.
    mem.ingest("turn two")
    assert voice._carry_seqlen > seqlen_after_ingest


# ---------------------------------------------------------------------------
# reset clears carry
# ---------------------------------------------------------------------------

def test_reset_clears_carry():
    """``FadeMemory.reset`` calls ``reset_carry`` -> ``_carry_seqlen`` is 0 and a
    subsequent recall emits no carried recall (no bleed into the next session)."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(voice_carry=True), _StubEmbedder(), voice)
    mem.ingest("User: hi\nAssistant: hello")
    assert voice._carry_seqlen > 0
    assert any(r.regime == REGIME_CARRY for r in mem.recall("hi", top_k=3))
    mem.reset()
    assert voice._carry_seqlen == 0
    # post-reset recall: no carried recall (turn-1 of a fresh session).
    assert not any(r.regime == REGIME_CARRY for r in mem.recall("hi", top_k=3))


def test_reset_non_carry_voice_is_noop():
    """``reset`` with a non-carry voice (no ``reset_carry``) does not crash (the
    ``hasattr`` guard)."""
    mem = FadeMemory(_cfg(voice_carry=True), _StubEmbedder(), _PlainVoice())
    mem.ingest("docA:0")
    mem.reset()   # no raise
    # the SSM-A store is reset (the existing reset behavior still holds).
    assert len(mem.blurbs) == 0


# ---------------------------------------------------------------------------
# byte-identical when off
# ---------------------------------------------------------------------------

def test_recall_byte_identical_carry_off_vs_default():
    """``voice_carry=False`` (explicit) and the ``FadeConfig`` default produce
    identical ``recall()`` output for the same ingest sequence. Pins the
    default-OFF contract: the carry leg is invisible when off."""
    emb = _StubEmbedder()
    cfg_off = _cfg(voice_carry=False)
    cfg_default = _cfg()   # voice_carry defaults False
    assert cfg_default.voice_carry is False

    def _run(cfg):
        mem = FadeMemory(cfg, _StubEmbedder(), _StubCarryVoice())
        mem.ingest("docA:0")
        mem.ingest("docB:0")
        for i in range(6):
            mem.ingest(f"doc{i+1}:0")
        return mem.recall("docA:0", top_k=4)

    off = _run(cfg_off)
    default = _run(cfg_default)
    assert off == default
    assert not any(r.regime == REGIME_CARRY for r in off)


def test_carry_on_non_carry_voice_byte_identical_to_off():
    """``voice_carry=True`` on a NON-``CarryVoice`` is byte-identical to carry
    off: the ``hasattr`` guard skips both ingest and recall carry hooks, so the
    SSM-A regime path is untouched. This is the contract that makes the feature
    safe to ship default-OFF behind an additive flag."""
    emb = _StubEmbedder()

    def _run(voice, carry: bool):
        mem = FadeMemory(_cfg(voice_carry=carry), emb, voice)
        mem.ingest("docA:0")
        mem.ingest("docB:0")
        for i in range(6):
            mem.ingest(f"doc{i+1}:0")
        return mem.recall("docA:0", top_k=4)

    off = _run(_PlainVoice(), False)
    on_noncarry = _run(_PlainVoice(), True)
    assert on_noncarry == off   # the guard makes carry-on a no-op


# ---------------------------------------------------------------------------
# format_fade_block: the carry line + the A4 cascade drop priority
# ---------------------------------------------------------------------------

def _r(eid, regime, content, *, cos_q=0.0, cos=0.5, name=None) -> dict:
    names = {REGIME_VERBATIM: "verbatim", REGIME_GIST: "gist",
             REGIME_FORGOTTEN: "forgotten", REGIME_CARRY: "carry"}
    return {"anchor_id": eid, "regime": regime,
            "regime_name": name or names.get(regime, "?"),
            "cos": cos, "cos_q": cos_q, "content": content, "blurb": None}


def test_format_fade_block_renders_carry_line():
    """A ``REGIME_CARRY`` recall renders under the ``[carry, in-context]`` label,
    alongside the R1/R3 lines."""
    recalls = [
        _r(-1, REGIME_CARRY, "the passphrase was SQUEAKY-RAVEN-42", cos=1.0),
        _r(0, REGIME_VERBATIM, "postgres WAL tuning notes", cos_q=0.9),
        _r(1, REGIME_GIST, "learning rate schedule blurb", cos_q=0.6),
    ]
    block = format_fade_block(recalls)
    assert "[FADE MEMORY" in block
    assert "[carry, in-context] the passphrase was SQUEAKY-RAVEN-42" in block
    assert "[verbatim, recent] postgres WAL tuning notes" in block
    assert "[gist, fading] learning rate schedule blurb" in block


def test_cascade_never_drops_carry_before_r3():
    """The A4 budget cascade gives carry the LOWEST drop-priority (drop-key
    regime -> -1): a tight budget drops BOTH R3 gists before the single carry
    line. Carry is the most valuable within-window recall -- it survives while
    any R3/R1 remains."""
    long = "x" * 200
    recalls = [
        _r(-1, REGIME_CARRY, f"carry-needle {long}", cos=1.0),
        _r(0, REGIME_GIST, f"gist-A {long}", cos_q=0.9),
        _r(1, REGIME_GIST, f"gist-B {long}", cos_q=0.8),
        _r(2, REGIME_VERBATIM, f"verb-C {long}", cos_q=0.7),
    ]
    # Tight budget: header + 2 long lines fit, 4 do not. The cascade drops the
    # two R3 gists first (highest regime), keeping carry + R1.
    block = format_fade_block(recalls, max_tokens=160)
    assert "carry-needle" in block        # carry kept (lowest drop-priority)
    assert "verb-C" in block               # R1 kept (regime < R3)
    assert "gist-A" not in block           # R3 dropped before carry
    assert "gist-B" not in block


def test_carry_survives_when_only_carry_and_r1_remain():
    """Carry is never dropped while any R1 remains: a budget that fits carry + R1
    but not R3 keeps both carry and R1, drops R3. (Carry vs R1: carry has lower
    drop-priority than even R1 -- regime -1 < 1.)"""
    long = "z" * 200
    recalls = [
        _r(-1, REGIME_CARRY, f"carry {long}", cos=1.0),
        _r(0, REGIME_GIST, f"gist {long}", cos_q=0.9),
        _r(1, REGIME_VERBATIM, f"verb {long}", cos_q=0.6),
    ]
    # Budget fits exactly carry + R1 (both long): drops the R3 gist only.
    block = format_fade_block(recalls, max_tokens=160)
    assert "[carry, in-context]" in block    # carry kept (lowest drop-priority)
    assert "[verbatim, recent]" in block      # R1 kept
    assert block.count("[gist, fading]") == 0  # the R3 gist dropped, not carry
    # (the header prose contains the word "gist" -- assert on the line marker,
    # not the bare word.)


# ---------------------------------------------------------------------------
# end-to-end: the carried recall flows through the [FADE MEMORY] block
# ---------------------------------------------------------------------------

def test_carry_recall_flows_into_fade_block():
    """The synthetic ``REGIME_CARRY`` recall ``FadeMemory.recall`` produces flows
    through the orchestrator-seam dict shape into ``format_fade_block`` unchanged
    (the orchestrator builds ``fade_recalls`` from the same attributes)."""
    voice = _StubCarryVoice()
    mem = FadeMemory(_cfg(voice_carry=True, fade_block_max_tokens=4000),
                     _StubEmbedder(), voice)
    mem.ingest("User: the passphrase is SQUEAKY-RAVEN-42\nAssistant: noted")
    recalls = mem.recall("what was the passphrase", top_k=3)
    # the orchestrator builds this dict from each Recall (orch:941-964).
    dict_recalls = [
        {"anchor_id": r.anchor_id, "regime": r.regime,
         "regime_name": {REGIME_VERBATIM: "verbatim", REGIME_GIST: "gist",
                          REGIME_FORGOTTEN: "forgotten",
                          REGIME_CARRY: "carry"}[r.regime],
         "cos": r.cos, "cos_q": r.cos_q, "content": r.content, "blurb": r.blurb}
        for r in recalls
    ]
    block = format_fade_block(dict_recalls, max_tokens=4000)
    assert "[carry, in-context]" in block
    assert "SQUEAKY-RAVEN-42" in block


# ---------------------------------------------------------------------------
# the REAL Mamba3Voice carry methods (stub model, no torch/HF/CUDA)
# ---------------------------------------------------------------------------
# The stub-``_StubCarryVoice`` tests above pin the FadeMemory WIRING (hooks fire,
# ``hasattr`` guards, byte-identical-when-off). These pin the real
# ``Mamba3Voice.ingest_turn`` / ``recall_from_carry`` / ``reset_carry`` against a
# stub model whose cache is a real tensor dict (so the clone-into-throwaway
# read-only invariant is actually exercised -- a non-cloned recall would mutate
# the carried cache in-place and the assertions below would fail).

class _StubCarryModel:
    """A ``MambaLMHeadModel`` stand-in whose ``allocate_inference_cache`` returns a
    real per-layer 4-tensor dict (the angle/ssm/k/v shape the real
    ``Mamba3._get_states_from_cache`` unpacks) and whose ``__call__`` writes back
    into the cache IN-PLACE (mimicking the real ``.copy_()`` write-back). This is
    the crucial difference from ``_StubMamba3Model`` (which returns ``{}``): a
    recall that failed to clone would stamp the carried cache, and the read-only
    test below would catch it. ``inference_params`` is read for the write-back."""

    def __init__(self, vocab: int = 20) -> None:
        self.vocab = vocab
        self.calls = 0

    def __call__(self, cur, inference_params=None):
        import torch

        self.calls += 1
        # Mimic the real Mamba3 state write-back: stamp the call count into every
        # cache tensor in-place. The real model does ``.copy_()`` of the new
        # recurrent state into ``key_value_memory_dict[layer]``; this stand-in
        # fills with the call count so a mutation is observable.
        if inference_params is not None and inference_params.key_value_memory_dict:
            for cache in inference_params.key_value_memory_dict.values():
                for t in cache:
                    t.fill_(float(self.calls))
        seq = cur.shape[1]
        nxt = 10 + (self.calls % 5)        # deterministic rotating argmax
        logits = torch.full((1, seq, self.vocab), -1e4)
        logits[0, -1, nxt] = 0.0
        return _StubMamba3Out(logits)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kw):
        import torch

        # one layer, 4 state tensors (the real 4-tuple: angle_dt, ssm, k, v).
        return {0: [torch.zeros(2, 2) for _ in range(4)]}


def test_mamba3_voice_ingest_turn_advances_carried_state():
    """``ingest_turn`` forwards the turn and advances ``_carry_seqlen`` /
    ``_carry_inf.seqlen_offset`` (the sole writer). The stub model stamps the
    cache with its call count, so the carried cache is observably written."""
    voice = Mamba3Voice(_StubCarryModel(), _StubHFTokenizer(),
                        device="cpu", temperature=0.0)
    n = voice.ingest_turn("turn one", max_tokens=2048)
    assert n > 0                              # tokens consumed (stub tok -> 3 ids)
    assert voice._carry_seqlen == n
    assert voice._carry_inf.seqlen_offset == n
    # the carried cache was written (the model stamped it with call count 1).
    assert float(voice._carry_inf.key_value_memory_dict[0][0][0, 0]) == 1.0


def test_mamba3_voice_recall_from_carry_is_read_only():
    """``recall_from_carry`` is READ-ONLY: it clones the carried cache into a
    throwaway, so the model's in-place write-back lands in the CLONES, not the
    carried state. After recall: ``_carry_seqlen`` / ``_carry_inf.seqlen_offset``
    are unchanged, and the carried cache tensor is byte-identical to its
    post-ingest value. WITHOUT the clone (the bug the clone prevents) the stub
    model's ``fill_`` would stamp the carried cache and this test would fail."""
    voice = Mamba3Voice(_StubCarryModel(), _StubHFTokenizer(),
                        device="cpu", temperature=0.0)
    voice.ingest_turn("turn one", max_tokens=2048)
    carried_stamp = voice._carry_inf.key_value_memory_dict[0][0].clone()
    seqlen = voice._carry_seqlen
    offset = voice._carry_inf.seqlen_offset
    assert float(carried_stamp[0, 0]) == 1.0   # ingest stamped it (call 1)

    # recall decodes a continuation from the carried state.
    out = voice.recall_from_carry("the cue", max_new_tokens=4)
    assert out                                 # decoded something (greedy)
    # the carried state is UNCHANGED (recall used the throwaway clone).
    assert voice._carry_seqlen == seqlen
    assert voice._carry_inf.seqlen_offset == offset
    assert torch.equal(voice._carry_inf.key_value_memory_dict[0][0],
                       carried_stamp)
    # the model WAS called by recall (the throwaway got stamped), but the carried
    # cache kept the ingest value -> the clone isolated the throwaway.
    assert float(voice._carry_inf.key_value_memory_dict[0][0][0, 0]) == 1.0


def test_mamba3_voice_carry_saturated_recall_returns_empty():
    """When the carried state is saturated (``_carry_seqlen`` reached the budget),
    a recall whose cue+decode would exceed the slab returns ``""`` (the ``gen``
    clamp guards the cue forward too). Saturation is the CORRECT stop: beyond the
    ~2048-token window is WaveDB's job (exp #4)."""
    voice = Mamba3Voice(_StubCarryModel(), _StubHFTokenizer(),
                        device="cpu", temperature=0.0)
    # Tiny budget so saturation is reachable in few turns (each turn = 3 ids;
    # _carry_max = 8 + 64 = 72). Ingest until it stops advancing.
    for _ in range(40):
        if voice.ingest_turn("x", max_tokens=8) == 0:
            break
    assert voice._carry_seqlen > 0
    assert voice._carry_seqlen <= voice._carry_max   # never exceeds the slab
    # Once saturated, recall's gen clamps to <= 0 -> "" (no cue forward, no OOB).
    # Drive to full saturation: keep ingesting until ingest returns 0.
    for _ in range(40):
        voice.ingest_turn("x", max_tokens=8)
    assert voice.ingest_turn("x", max_tokens=8) == 0   # confirmed saturated
    assert voice.recall_from_carry("cue", max_new_tokens=64) == ""


def test_mamba3_voice_reset_carry_clears_state():
    """``reset_carry`` drops the carried InferenceParams; the next ``ingest_turn``
    re-allocates fresh (seqlen 0, a new cache). No bleed into the next session."""
    voice = Mamba3Voice(_StubCarryModel(), _StubHFTokenizer(),
                        device="cpu", temperature=0.0)
    voice.ingest_turn("turn one", max_tokens=2048)
    assert voice._carry_inf is not None
    assert voice._carry_seqlen > 0
    first_inf = voice._carry_inf
    voice.reset_carry()
    assert voice._carry_inf is None
    assert voice._carry_seqlen == 0
    assert voice._carry_max == 0
    # re-ingest allocates a FRESH InferenceParams (not the dropped one).
    voice.ingest_turn("turn two", max_tokens=2048)
    assert voice._carry_inf is not None
    assert voice._carry_inf is not first_inf
    assert voice._carry_seqlen > 0
    # the fresh cache was zero-initialized, then re-ingest stamped it with the
    # model's CURRENT call count (the model persists across reset; only the
    # InferenceParams is dropped). The dropped stamp (1.0 from call 1) did not
    # bleed into the new cache -- the new one carries call 2's stamp.
    assert float(voice._carry_inf.key_value_memory_dict[0][0][0, 0]) == 2.0


def test_mamba3_voice_carry_turn_one_recall_returns_empty():
    """Turn-1 recall (no ingest yet) -> ``""`` (the ``_carry_seqlen == 0`` guard).
    Mirrors ``test_recall_no_carry_on_turn_one`` but on the real Mamba3Voice."""
    voice = Mamba3Voice(_StubCarryModel(), _StubHFTokenizer(),
                        device="cpu", temperature=0.0)
    assert voice._carry_inf is None
    assert voice.recall_from_carry("anything", max_new_tokens=4) == ""