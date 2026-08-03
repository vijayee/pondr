"""A4: regime-cascade drop order (R4 -> R3 -> R1) for the ``[FADE MEMORY]`` block.

Tencent-survey A4 ([[pondr-tencent-agent-memory-survey]]): when the fade block
overflows a token budget, drop the most-faded first -- R3 (gist) before R1
(verbatim), R4 already skipped at render -- and within a regime drop the
least-prompt-relevant (lowest ``cos_q``) first. The block sits OUTSIDE the
chunker's context budget (prepended to ``user_content`` after the chunked
context is built), so ``format_fade_block`` is the only place it is bounded.

These tests pin: byte-identical when ``max_tokens=0`` or the block fits; the
R3-before-R1 cascade; the within-regime lowest-cos_q tiebreak; R4 never counted;
``cos_q`` surfaced on ``Recall`` from ``BlurbStore.retrieve``; and the
orchestrator passing the budget through at the inject seam.
"""

from __future__ import annotations

import hashlib

from src.config import Phase2cConfig
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.subconscious.backbone import JGSBackbone
from src.subconscious.configs import BackboneConfig
from src.subconscious.fade import (
    FadeConfig,
    FadeMemory,
    REGIME_FORGOTTEN,
    REGIME_GIST,
    REGIME_VERBATIM,
    format_fade_block,
)


# ── deterministic embedder (SHA256 stretch -> normalized, 384-d) ──────────────

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


def _r(eid, regime, content, *, cos_q=0.0, cos=0.5, name=None) -> dict:
    """A recall dict in the orchestrator-seam shape (with the A4 ``cos_q`` key)."""
    names = {REGIME_VERBATIM: "verbatim", REGIME_GIST: "gist",
             REGIME_FORGOTTEN: "forgotten"}
    return {"anchor_id": eid, "regime": regime,
            "regime_name": name or names.get(regime, "?"),
            "cos": cos, "cos_q": cos_q, "content": content, "blurb": None}


# ─────────────────────────────────────────────────────────────────────────────
# format_fade_block: byte-identical + drop cascade
# ─────────────────────────────────────────────────────────────────────────────

def test_no_budget_byte_identical():
    """``max_tokens=0`` (default) -> the block is unchanged from pre-A4. Direct
    callers that do not pass ``max_tokens`` get byte-identical output."""
    recalls = [
        _r(0, REGIME_VERBATIM, "postgres WAL tuning notes", cos_q=0.9),
        _r(1, REGIME_GIST, "learning rate schedule blurb", cos_q=0.6),
        _r(2, REGIME_FORGOTTEN, "[forgotten]", cos_q=0.1),
    ]
    default = format_fade_block(recalls)
    explicit = format_fade_block(recalls, max_tokens=0)
    assert default == explicit
    assert "[FADE MEMORY" in default
    assert "[verbatim, recent] postgres WAL tuning notes" in default
    assert "[gist, fading] learning rate schedule blurb" in default
    # R4 never rendered.
    assert "forgotten" not in default


def test_block_fits_no_drop():
    """A generous budget -> no drop -> byte-identical to the no-budget block."""
    recalls = [
        _r(0, REGIME_VERBATIM, "verbatim note one", cos_q=0.9),
        _r(1, REGIME_GIST, "gist note two", cos_q=0.5),
    ]
    assert format_fade_block(recalls, max_tokens=4000) == format_fade_block(recalls)


def test_drop_cascade_r3_before_r1():
    """Tight budget that fits only the 2 R1 -> both R3 dropped first, both R1
    kept. The cascade drops most-faded first (R3=3 > R1=1)."""
    long = "x" * 200
    recalls = [
        _r(0, REGIME_GIST, f"gist-A {long}", cos_q=0.9),
        _r(1, REGIME_GIST, f"gist-B {long}", cos_q=0.8),
        _r(2, REGIME_VERBATIM, f"verb-C {long}", cos_q=0.7),
        _r(3, REGIME_VERBATIM, f"verb-D {long}", cos_q=0.6),
    ]
    # header(~47 tok) + 4 lines(~52 tok each) ~ 255 tok; +2 R1 only ~ 151 tok.
    block = format_fade_block(recalls, max_tokens=160)
    assert block.count("[gist, fading]") == 0   # both R3 dropped
    assert block.count("[verbatim, recent]") == 2  # both R1 kept
    assert "gist-A" not in block and "gist-B" not in block
    assert "verb-C" in block and "verb-D" in block


def test_within_regime_lowest_cos_q_dropped_first():
    """2 R3 (cos_q 0.9, 0.5) + 1 R1, budget fits 1 R3 + the R1 -> the cos_q=0.5
    R3 is dropped, the cos_q=0.9 R3 kept (lowest prompt-relevance drops first
    within a regime)."""
    long = "y" * 200
    recalls = [
        _r(0, REGIME_GIST, f"keep-high {long}", cos_q=0.9),
        _r(1, REGIME_GIST, f"drop-low {long}", cos_q=0.5),
        _r(2, REGIME_VERBATIM, f"verb {long}", cos_q=0.6),
    ]
    # header(47) + 3 lines(~52) ~ 203; 2 lines (1 R3 + 1 R1) ~ 151.
    block = format_fade_block(recalls, max_tokens=160)
    assert "keep-high" in block       # cos_q=0.9 R3 kept
    assert "drop-low" not in block    # cos_q=0.5 R3 dropped (lowest cos_q in R3)
    assert "verb" in block            # R1 kept (regime wins over R3)


def test_tight_budget_drops_all_returns_empty():
    """A budget too tight for any recall (even one) -> ``""``, NOT a header-only
    block. The ``no empty header`` contract holds under budget pressure too."""
    recalls = [
        _r(0, REGIME_GIST, "gist " + "x" * 200, cos_q=0.9),
        _r(1, REGIME_VERBATIM, "verb " + "y" * 200, cos_q=0.8),
    ]
    # header alone ~47 tokens -> a 5-token budget drops everything.
    assert format_fade_block(recalls, max_tokens=5) == ""


def test_r4_skipped_and_not_counted():
    """R4 is excluded from the rendered items entirely -- it never counts toward
    the budget and is never a drop victim. A budget that fits header+R3+R1 keeps
    both; R4 is absent from the output."""
    recalls = [
        _r(0, REGIME_FORGOTTEN, "[forgotten]", cos_q=0.1),
        _r(1, REGIME_GIST, "gist note", cos_q=0.8),
        _r(2, REGIME_VERBATIM, "verb note", cos_q=0.9),
    ]
    # Budget generous enough for header + R3 + R1 (the only content-bearing
    # items). R4 is not in items, so it cannot push them out.
    block = format_fade_block(recalls, max_tokens=4000)
    assert block == format_fade_block(recalls)  # no drop
    assert "forgotten" not in block
    assert "gist note" in block and "verb note" in block


# ─────────────────────────────────────────────────────────────────────────────
# cos_q surfaced from BlurbStore.retrieve
# ─────────────────────────────────────────────────────────────────────────────

def test_cos_q_surfaced_on_recall():
    """``FadeMemory.recall`` surfaces ``cos_q`` (cos(bge(prompt), bge(anchor)),
    computed by ``BlurbStore.retrieve`` and discarded pre-A4) onto each
    ``Recall``. The best-relevance anchor has the highest ``cos_q``."""
    emb = _StubEmbedder()
    mem = FadeMemory(FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.01,
                                ring_capacity=8), emb, voice=None)
    a = mem.ingest("alpha")
    b = mem.ingest("beta")
    results = mem.recall("alpha", top_k=5)
    assert len(results) >= 2
    # The matching anchor is retrieved best-first with cos_q ~ 1.0.
    top = results[0]
    assert top.anchor_id == a
    assert top.cos_q == 1.0 or abs(top.cos_q - 1.0) < 1e-5
    # cos_q is relevance-ordered (best first), and every recall carries it.
    cos_qs = [r.cos_q for r in results]
    assert cos_qs == sorted(cos_qs, reverse=True)
    assert all(isinstance(q, float) for q in cos_qs)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator inject seam passes the budget through
# ─────────────────────────────────────────────────────────────────────────────

def _ep(eid) -> Episode:
    return Episode(id=eid, timestamp="2026-08-01T10:00:00",
                   summary=f"summary {eid}", full_text=f"User: u{eid}\nAssistant: a{eid}",
                   entities=[], topics=["storage"], tones=[], decisions=[])


def _orchestrator(tmp_path, fade, *, db_subdir, plan=None):
    store = HippocampalStore(str(tmp_path / db_subdir))
    store.encode_episode(_ep("ep_001"))
    retriever = HippocampalRetriever(store, planner=_StubPlanner(plan or {}),
                                     embedder=_StubEmbedder())
    backbone = JGSBackbone(BackboneConfig())
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(tmp_path / db_subdir / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone,
        embedder=_StubEmbedder(), mode_a=_StubModeA("LLM SAID THIS"), config=cfg,
        user_id="victor", fade_memory=fade, fade_inject=True,
    )
    return orch


class _StubPlanner:
    def __init__(self, plan): self._plan = plan
    def plan(self, prompt, conversation_history=None): return self._plan


class _StubModeA:
    def __init__(self, reply): self.reply = reply; self.calls = []
    def _complete(self, messages, tools=None, tool_choice=None):
        self.calls.append(messages); return self.reply, None


def _user_msg_contents(calls):
    for msgs in calls:
        for m in msgs:
            if m.get("role") == "user":
                yield m["content"]


def _fade_with_budget(budget: int, long_a: str, long_b: str) -> FadeMemory:
    """A FadeMemory with 2 pre-ingested R1 anchors (distinct long blurbs) and a
    set ``fade_block_max_tokens``."""
    fade = FadeMemory(
        FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.01, ring_capacity=8,
                   fade_block_max_tokens=budget),
        _StubEmbedder(), voice=None)
    fade.ingest("alpha", blurb_text=long_a)
    fade.ingest("beta", blurb_text=long_b)
    return fade


def test_orchestrator_inject_applies_budget(tmp_path):
    """The orchestrator passes ``cfg.fade_block_max_tokens`` to
    ``format_fade_block``: with budget=0 / generous the block keeps both R1
    recalls; with a tight budget one is dropped. ``cos_q`` rides on the
    observability dict."""
    long_a = "A" * 200
    long_b = "B" * 200

    # budget=0 -> no drop -> both R1 lines present.
    orch_off = _orchestrator(tmp_path, _fade_with_budget(0, long_a, long_b),
                             db_subdir="off")
    orch_off.query("alpha")
    user_off = list(_user_msg_contents(orch_off.mode_a.calls))
    block_off = next(c for c in user_off if "[FADE MEMORY" in c)
    assert block_off.count("[verbatim, recent]") == 2
    # cos_q surfaced on the observability dict (not LLM-facing, but present).
    # Re-query to read result["fade_recalls"] -- the first query already recalls.
    res_off = orch_off.query("alpha")
    assert all("cos_q" in r for r in res_off["fade_recalls"])

    # generous budget -> byte-identical block to budget=0.
    orch_gen = _orchestrator(tmp_path, _fade_with_budget(4000, long_a, long_b),
                             db_subdir="gen")
    orch_gen.query("alpha")
    block_gen = next(c for c in _user_msg_contents(orch_gen.mode_a.calls)
                     if "[FADE MEMORY" in c)
    assert block_gen == block_off

    # tight budget -> one R1 dropped (the cascade). header(47) + 2 long
    # lines(~52 each) ~ 151 tok; 1 line ~ 99 tok; budget=120 sits between.
    orch_tight = _orchestrator(tmp_path, _fade_with_budget(120, long_a, long_b),
                               db_subdir="tight")
    orch_tight.query("alpha")
    block_tight = next(c for c in _user_msg_contents(orch_tight.mode_a.calls)
                       if "[FADE MEMORY" in c)
    assert block_tight.count("[verbatim, recent]") == 1
    assert block_tight != block_off  # the drop changed the block

    orch_off.store.close()
    orch_gen.store.close()
    orch_tight.store.close()