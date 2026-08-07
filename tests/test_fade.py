"""Tests for ``src/subconscious/fade.py`` -- the Stage-1 dual-SSM fade memory.

CPU, self-contained. The embedder and the voice (SSM-B) are injected seams, so
the tests use a synthetic doc-themed embedder (mimics bge: same-doc chunks
correlated, cross-doc near-orthogonal) and a stub voice that records its calls.
The thing under test -- ``VectorCarrySSM`` (SSM-A), ``BlurbStore``, the free
cosine router, and the 4-regime dispatch -- is the REAL code.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.subconscious.fade import (
    BlurbStore,
    FadeConfig,
    FadeMemory,
    Recall,
    REGIME_FILL,
    REGIME_FORGOTTEN,
    REGIME_GIST,
    REGIME_VERBATIM,
    VectorCarrySSM,
    format_fade_block,
)


# --------------------------------------------------------------------- doubles
def _stable_seed(s: str) -> int:
    """Stable 64-bit seed from a string (PYTHONHASHSEED-independent)."""
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "big")


class _StubEmbedder:
    """Mimics bge-small: same-doc chunks correlated, cross-doc near-orthogonal.

    Text format ``<doc>:<chunk>`` selects the doc theme. Each doc gets a fixed
    random unit ``base``; each chunk is
    ``normalize(within_doc*base + (1-within_doc)*noise)`` so ``cos(chunk, base) =
    within_doc`` and same-doc ``cos ~ within_doc**2``, cross-doc
    ``~ (1-within_doc)**2``. Deterministic (stable hash). No model loaded.
    """

    def __init__(self, dim: int = 384, within_doc: float = 0.9, seed: int = 0) -> None:
        self.dim = dim
        self.within_doc = within_doc
        self.seed = seed
        self._base: dict[str, np.ndarray] = {}

    def _doc_base(self, doc: str) -> np.ndarray:
        if doc not in self._base:
            rng = np.random.default_rng(self.seed ^ _stable_seed(doc))
            v = rng.standard_normal(self.dim).astype(np.float64)
            self._base[doc] = v / np.linalg.norm(v)
        return self._base[doc]

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            doc = t.partition(":")[0]
            base = self._doc_base(doc)
            rng = np.random.default_rng(self.seed ^ _stable_seed(t))
            noise = rng.standard_normal(self.dim).astype(np.float64)
            noise /= np.linalg.norm(noise)
            v = self.within_doc * base + (1.0 - self.within_doc) * noise
            v = v / np.linalg.norm(v)
            out.append([float(x) for x in v.tolist()])
        return out


class _StubVoice:
    """Records expand() calls; returns ``blurb + ' [expanded]'``."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def expand(self, blurb: str, max_new_tokens: int) -> str:
        self.calls.append(blurb)
        return blurb + " [expanded]"


# ----------------------------------------------------------------- VectorCarrySSM
def test_ssm_step_math() -> None:
    ssm = VectorCarrySSM(dim=4, decay=0.5, write_gate=1.0)
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    ssm.step(a)
    assert np.allclose(ssm.state(), a)
    ssm.step(b)
    # state = decay*prev + write_gate*new = 0.5*a + 1.0*b
    assert np.allclose(ssm.state(), [0.5, 1.0, 0.0, 0.0])
    q = ssm.query()
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)


def test_ssm_reset_zeros_state() -> None:
    ssm = VectorCarrySSM(dim=4, decay=0.9)
    ssm.step(np.ones(4, dtype=np.float32))
    ssm.reset()
    assert np.all(ssm.state() == 0)


def test_ssm_query_zero_state_is_safe() -> None:
    # All-zero state -> norm 0 -> guarded to 1 -> zero query (no divide-by-zero
    # NaN). The guard prevents NaN, not unit-norm from a zero vector.
    ssm = VectorCarrySSM(dim=4, decay=0.9)
    q = ssm.query()
    assert not np.isnan(q).any()
    assert np.all(q == 0.0)


def test_ssm_rejects_wrong_dim() -> None:
    ssm = VectorCarrySSM(dim=4, decay=0.9)
    with pytest.raises(ValueError):
        ssm.step(np.ones(3, dtype=np.float32))


# --------------------------------------------------------------------- BlurbStore
def test_blurb_store_retrieve_top1_is_self() -> None:
    store = BlurbStore(dim=4)
    store.add(0, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "chunk0")
    store.add(1, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), "chunk1")
    hits = store.retrieve(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert hits[0][0] == 0
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[0][2] == "chunk0"


def test_blurb_store_lookups() -> None:
    store = BlurbStore(dim=4)
    store.add(7, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "seven")
    assert store.text(7) == "seven"
    assert np.allclose(store.vector(7), [1.0, 0.0, 0.0, 0.0])
    assert store.text(99) is None
    assert store.vector(99) is None
    assert len(store) == 1


def test_blurb_store_empty_retrieve() -> None:
    store = BlurbStore(dim=4)
    assert store.retrieve(np.zeros(4, dtype=np.float32), k=5) == []


# ------------------------------------------------------------- the free router
def test_recoverability_recent_higher_than_older() -> None:
    # The fade: an older anchor has lower cos(state, anchor) than a recent one.
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=64)
    mem = FadeMemory(cfg, emb, _StubVoice())
    old = mem.ingest("docA:0")
    for i in range(8):
        mem.ingest(f"doc{i+1}:0")          # cross-doc fades `old`
    recent = mem.ingest("docA:1")        # same doc, just ingested
    cos_old = mem._recoverability(old)
    cos_recent = mem._recoverability(recent)
    # The fade direction: the recent anchor has higher cos(state, anchor) than
    # the older one. (A just-ingested anchor mid-stream is NOT cos~1.0 -- the
    # state is a blend, so cos ~ 1/||state||; only N=0 with no history is 1.0,
    # which test_regime1_verbatim_immediate covers.)
    assert cos_recent > cos_old


# ----------------------------------------------------------- regime dispatch
def test_regime1_verbatim_immediate_no_voice() -> None:
    emb = _StubEmbedder()
    voice = _StubVoice()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=8)
    mem = FadeMemory(cfg, emb, voice)
    aid = mem.ingest("docA:0")
    r = mem.recall_anchor(aid)
    assert r.regime == REGIME_VERBATIM
    assert r.content == "docA:0"
    assert r.cos == pytest.approx(1.0, abs=1e-5)
    assert voice.calls == []              # verbatim does not invoke the voice


def test_regime_sweep_hits_all_three_and_voice_for_gist() -> None:
    # Stream cross-doc chunks past an anchor; the regime must transition
    # R1 (in ring) -> R3 (gist, faded-but-retrievable) -> R4 (forgotten).
    emb = _StubEmbedder()
    voice = _StubVoice()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=4)
    mem = FadeMemory(cfg, emb, voice)
    aid = mem.ingest("docA:0")
    r0 = mem.recall_anchor(aid)
    assert r0.regime == REGIME_VERBATIM          # N=0, in ring
    seen = {r0.regime}
    first_r3: Recall | None = None
    for i in range(24):
        mem.ingest(f"doc{i+1}:0")
        r = mem.recall_anchor(aid)
        seen.add(r.regime)
        if r.regime == REGIME_GIST and first_r3 is None:
            first_r3 = r
    assert REGIME_VERBATIM in seen
    assert REGIME_GIST in seen
    assert REGIME_FORGOTTEN in seen
    # R3 expands the retrieved blurb via the voice; R1 and R4 do not.
    assert first_r3 is not None
    assert first_r3.content.endswith("[expanded]")
    assert first_r3.blurb is not None


def test_regime4_forgotten_no_confabulation() -> None:
    # Fast decay + many cross-doc -> well below cos_gist -> R4, and the voice
    # is NOT called (no confabulation -- the graceful tip-of-tongue floor).
    emb = _StubEmbedder()
    voice = _StubVoice()
    cfg = FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.20, ring_capacity=2)
    mem = FadeMemory(cfg, emb, voice)
    aid = mem.ingest("docA:0")
    for i in range(30):
        mem.ingest(f"doc{i+1}:0")
    r = mem.recall_anchor(aid)
    assert r.regime == REGIME_FORGOTTEN
    assert r.content == "[forgotten]"
    assert voice.calls == []


def test_voice_none_passes_blurb_verbatim() -> None:
    # ``voice=None`` (the Phase-A serve path, no token-LM loaded) -> Regime 3
    # returns the retrieved blurb VERBATIM (the built-in passthrough), with no
    # ``[expanded]`` suffix. Sweep cross-doc chunks past the anchor (mirrors
    # test_regime_sweep_hits_all_three_and_voice_for_gist) to land in R3, then
    # assert the passthrough content.
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=4)
    mem = FadeMemory(cfg, emb, None)           # no voice
    aid = mem.ingest("docA:0")
    first_r3: Recall | None = None
    for i in range(24):
        mem.ingest(f"doc{i+1}:0")              # cross-doc fades `aid`
        r = mem.recall_anchor(aid)
        if r.regime == REGIME_GIST and first_r3 is None:
            first_r3 = r
    assert first_r3 is not None, "never reached R3 (gist)"
    assert first_r3.content == first_r3.blurb   # passthrough: content IS the blurb
    assert "[expanded]" not in first_r3.content  # no voice was called
    assert first_r3.blurb is not None            # a blurb was retrieved (the gist)
    # Anchor-locked (the R3 content-drift fix, docs/fade-serve-validation-result.md):
    # R3 returns the anchor's OWN blurb ("docA:0"), NOT a state-closest cross-doc
    # chunk. The state is the recoverability signal, not the retrieval key.
    assert first_r3.blurb == "docA:0"


def test_r3_is_anchor_locked_not_state_closest() -> None:
    # Regression guard for the R3 content-drift found in the serve REPL
    # validation (docs/fade-serve-validation-result.md): across a mixed-domain
    # session the faded state is dominated by the most-recent chunk, so a
    # state-closest blurb retrieves WRONG-TOPIC content (the most-recent
    # cross-doc blurb, not the recalled anchor's). The fix makes R3
    # anchor-locked: retrieve the anchor's OWN blurb. WITHOUT the fix this
    # test fails -- ``r.blurb`` is the most-recent cross-doc chunk ("doc24:0"
    # or similar), not "docA:0".
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=4)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0")
    last_cross = ""
    r3_blurbs: set[str] = set()
    for i in range(24):
        last_cross = f"doc{i+1}:0"
        mem.ingest(last_cross)              # cross-doc fades `aid`, drifts state
        r = mem.recall_anchor(aid)
        if r.regime == REGIME_GIST:
            r3_blurbs.add(r.blurb)
    assert r3_blurbs, "never reached R3 (gist)"
    # Every R3 recall returns the anchor's OWN blurb, never a cross-doc chunk.
    assert all(b == "docA:0" for b in r3_blurbs)
    assert last_cross != "docA:0"          # sanity: a cross-doc blurb existed to drift to


def test_regime2_off_medium_e_is_gist() -> None:
    # Default (regime2_enabled=False): medium-e falls through to R3 (gist).
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.15, ring_capacity=4,
                     regime2_enabled=False)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0")
    saw_gist = False
    for i in range(24):
        mem.ingest(f"doc{i+1}:0")
        r = mem.recall_anchor(aid)
        if r.regime == REGIME_GIST:
            saw_gist = True
    assert saw_gist


def test_regime2_on_intercepts_medium_e() -> None:
    # With regime2_enabled=True, medium-e -> R2 (fill), so R3 (gist) is never
    # reached (R2 intercepts the band). R2 must fire (the band is non-empty).
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.15, ring_capacity=4,
                     regime2_enabled=True)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0")
    saw_fill = False
    saw_gist = False
    for i in range(24):
        mem.ingest(f"doc{i+1}:0")
        r = mem.recall_anchor(aid)
        if r.regime == REGIME_FILL:
            saw_fill = True
        if r.regime == REGIME_GIST:
            saw_gist = True
    assert saw_fill
    assert not saw_gist                  # R2 intercepts before R3


# ------------------------------------------------------------- ring eviction
def test_ring_eviction_cos_ring_branch_returns_verbatim() -> None:
    # An anchor evicted from the ring but still state-fresh (same-doc, slow
    # decay) returns verbatim via the cos_ring branch -- the ring is not the
    # only R1 path; the state being fresh is enough.
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.99, cos_ring=0.60, cos_gist=0.20, ring_capacity=3)
    mem = FadeMemory(cfg, emb, _StubVoice())
    a = mem.ingest("docA:0")
    for i in range(5):
        mem.ingest(f"docA:{i+1}")        # same-doc, slow decay -> a stays fresh
    assert a not in mem.ring             # evicted (cap 3)
    cos_a = mem._recoverability(a)
    assert cos_a >= cfg.cos_ring         # state still fresh despite eviction
    r = mem.recall_anchor(a)
    assert r.regime == REGIME_VERBATIM
    assert r.content == "docA:0"


# ------------------------------------------------------------- query-driven recall
def test_recall_query_routes_relevant_anchors() -> None:
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=4)
    mem = FadeMemory(cfg, emb, _StubVoice())
    a = mem.ingest("docA:0")
    mem.ingest("docA:1")                 # same-doc sibling
    for i in range(6):
        mem.ingest(f"doc{i+1}:0")          # cross-doc pushes docA out of the state
    # Query about docA -> retrieves docA anchors (a is top, cos(query,a)=1.0).
    results = mem.recall("docA:0", top_k=4)
    ids = [r.anchor_id for r in results]
    assert a in ids
    assert ids[0] == a                   # best-relevance first
    # a is old (faded over 7 steps) -> gist or forgotten, NOT verbatim.
    regimes = {r.anchor_id: r.regime for r in results}
    assert regimes[a] in (REGIME_GIST, REGIME_FORGOTTEN)


# ------------------------------------------------------------- lifecycle
def test_reset_clears_everything() -> None:
    emb = _StubEmbedder()
    mem = FadeMemory(FadeConfig(decay=0.9, ring_capacity=4), emb, _StubVoice())
    mem.ingest("docA:0")
    assert len(mem.blurbs) == 1
    assert np.any(mem.ssm_a.state() != 0)
    mem.reset()
    assert len(mem.blurbs) == 0
    assert np.all(mem.ssm_a.state() == 0)
    assert len(mem.ring) == 0
    aid = mem.ingest("docA:0")           # fresh id after reset
    assert aid == 0


def test_recall_anchor_unknown_returns_none() -> None:
    mem = FadeMemory(FadeConfig(), _StubEmbedder(), _StubVoice())
    assert mem.recall_anchor(999) is None


def test_format_fade_block_renders_r1_r3_skips_r4() -> None:
    # Phase B: ``format_fade_block`` renders the LLM-facing ``[FADE MEMORY]`` block.
    # It lists content-bearing recalls (R1 verbatim + R3 gist), regime-framed, and
    # SKIPS R4 (forgotten is a SIGNAL for a future long-term-memory pull, not LLM
    # content). Returns ``""`` when no content-bearing recalls are present (so an
    # all-R4 turn omits the block entirely -- byte-identical to flag-off). Takes the
    # dict shape the orchestrator RECALL seam builds.
    recalls = [
        {"anchor_id": 0, "regime": REGIME_VERBATIM, "regime_name": "verbatim",
         "cos": 0.99, "content": "postgres WAL tuning notes", "blurb": None},
        {"anchor_id": 1, "regime": REGIME_GIST, "regime_name": "gist",
         "cos": 0.55, "content": "learning rate schedule blurb", "blurb": "learning rate schedule blurb"},
        {"anchor_id": 2, "regime": REGIME_FORGOTTEN, "regime_name": "forgotten",
         "cos": 0.10, "content": "[forgotten]", "blurb": None},
    ]
    block = format_fade_block(recalls)
    assert "[FADE MEMORY" in block
    assert "[verbatim, recent] postgres WAL tuning notes" in block
    assert "[gist, fading] learning rate schedule blurb" in block
    # R4 is a signal, not content -- it must NOT appear in the LLM block.
    assert "forgotten" not in block
    assert "[forgotten]" not in block
    # R2 (off, but possible) renders under its own label if it ever appears.
    r2_block = format_fade_block([
        {"anchor_id": 3, "regime": REGIME_FILL, "regime_name": "fill",
         "cos": 0.7, "content": "reconstructed fill", "blurb": None},
    ])
    assert "[fill, reconstructed] reconstructed fill" in r2_block

    # All-R4 -> no content-bearing recalls -> empty block (omitted entirely).
    assert format_fade_block([
        {"anchor_id": 9, "regime": REGIME_FORGOTTEN, "regime_name": "forgotten",
         "cos": 0.05, "content": "[forgotten]", "blurb": None},
    ]) == ""
    # Empty recall list -> empty block.
    assert format_fade_block([]) == ""
    # Empty-content R1/R3 are dropped (no bare header line).
    assert format_fade_block([
        {"anchor_id": 4, "regime": REGIME_VERBATIM, "regime_name": "verbatim",
         "cos": 0.99, "content": "   ", "blurb": None},
    ]) == ""


# ----------------------------------------------------------- blurb_text override
def test_ingest_blurb_text_override() -> None:
    # The embed handle (``chunk_text``) and the recalled blurb (``blurb_text``)
    # may differ -- the production code-ingestion design: embed a prose summary,
    # recall the raw source. The bge vector MUST key off ``chunk_text`` (the
    # handle), and the stored blurb MUST be ``blurb_text[:blurb_chars]``.
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=8,
                     blurb_chars=40)
    mem = FadeMemory(cfg, emb, _StubVoice())
    handle = "docA:0"                       # the embed handle (short, distinct doc)
    raw_src = "x" * 200                     # the recalled content (long, different)
    aid = mem.ingest(handle, blurb_text=raw_src)
    # 1. The stored blurb is the raw source, truncated to blurb_chars -- NOT the
    #    handle. This is what R1/R3 recall returns.
    assert mem.blurbs.text(aid) == raw_src[: cfg.blurb_chars]
    assert mem.blurbs.text(aid) != handle
    # 2. The keyed vector is the embed of the HANDLE (``chunk_text``), not the
    #    raw source -- the recoverability router keys off the handle's bge.
    v = mem.blurbs.vector(aid)
    assert v is not None
    handle_vec = np.asarray(emb.encode([handle])[0], dtype=np.float32)
    assert np.allclose(v, handle_vec, atol=1e-5)
    # 3. R1 recall returns the raw source blurb (the override), verbatim.
    r = mem.recall_anchor(aid)
    assert r.regime == REGIME_VERBATIM
    assert r.content == raw_src[: cfg.blurb_chars]


def test_ingest_default_blurb_unchanged() -> None:
    # Regression guard for the backward-compat: one-arg ingest (the only form
    # every existing caller uses) stores ``chunk_text[:blurb_chars]`` as the
    # blurb and keys the vector off ``chunk_text`` -- byte-identical to the
    # pre-override behavior. The ``blurb_text`` extension must NOT change this.
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.9, cos_ring=0.95, cos_gist=0.20, ring_capacity=8,
                     blurb_chars=25)
    mem = FadeMemory(cfg, emb, _StubVoice())
    chunk = "docA:0 the rest of this chunk is longer than blurb_chars"
    aid = mem.ingest(chunk)                 # one arg -> blurb_text defaults None
    assert mem.blurbs.text(aid) == chunk[: cfg.blurb_chars]
    v = mem.blurbs.vector(aid)
    assert v is not None
    chunk_vec = np.asarray(emb.encode([chunk])[0], dtype=np.float32)
    assert np.allclose(v, chunk_vec, atol=1e-5)
    r = mem.recall_anchor(aid)
    assert r.regime == REGIME_VERBATIM
    assert r.content == chunk[: cfg.blurb_chars]


# ---------------------------------------------------- gist-on-forgetting (Phase C)
class _StubGist:
    """Minimal double for ``StructuredGist`` -- only the three attributes
    ``consolidate`` reads (``narrative`` / ``facts`` / ``state_assertions``).
    Keeps test_fade.py from importing the gister's Bonsai-client chain."""

    def __init__(self, narrative: str, facts=None, state_assertions=None) -> None:
        self.narrative = narrative
        self.facts = facts or []
        self.state_assertions = state_assertions or []


def _fade_anchor_to_r4(mem: FadeMemory, aid: int, n: int = 24) -> None:
    """Stream cross-doc chunks past ``aid`` until it routes R4 (forgotten)."""
    for i in range(n):
        mem.ingest(f"zz{i}:0")
        if mem.recall_anchor(aid).regime == REGIME_FORGOTTEN:
            return
    # Some seeds need a few more; keep going until R4 or the cap.
    for i in range(n, n + 24):
        mem.ingest(f"zz{i}:0")


def test_blurb_store_update_rekeys() -> None:
    # update() replaces the row's vec + text, stores the facts sidecar, and
    # invalidates the retrieve matrix (the NEW vec is top-1 for the new text).
    emb = _StubEmbedder()
    store = BlurbStore(dim=emb.dim)
    v0 = np.asarray(emb.encode(["docA:0"])[0], dtype=np.float32)
    store.add(0, v0, "docA:0")
    new_vec = np.asarray(emb.encode(["docB:9"])[0], dtype=np.float32)
    store.update(0, new_vec, "docB:9 gist", facts=[{"p": "has_state", "o": "v"}])
    assert store.text(0) == "docB:9 gist"
    assert np.allclose(store.vector(0), new_vec / np.linalg.norm(new_vec), atol=1e-5)
    assert store.facts(0) == [{"p": "has_state", "o": "v"}]
    # retrieve keyed on the NEW text returns this row top-1 (matrix was rebuilt).
    q = np.asarray(emb.encode(["docB:9"])[0], dtype=np.float32)
    hits = store.retrieve(q, k=1)
    assert hits[0][0] == 0


def test_blurb_store_update_unknown_raises() -> None:
    store = BlurbStore(dim=4)
    with pytest.raises(KeyError):
        store.update(999, np.zeros(4, dtype=np.float32), "x")


def test_consolidate_jumps_to_r1() -> None:
    # A faded-to-R4 anchor, consolidated with a same-doc narrative, jumps back
    # to R1 (verbatim) -- the gist is now the most-recent SSM write. count=1,
    # prior_gist stored, facts sidecar staged.
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.20, ring_capacity=2)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0")
    _fade_anchor_to_r4(mem, aid)
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN
    gist = _StubGist("docA:0 a tight gist of doc a", facts=[{"p": "k", "o": "v"}])
    cos_new = mem.consolidate(aid, gist)
    assert cos_new >= cfg.cos_ring or cos_new > 0.6   # back to fresh/recallable
    r = mem.recall_anchor(aid)
    assert r.regime == REGIME_VERBATIM
    assert r.content == gist.narrative[: cfg.blurb_chars]
    assert mem.consolidation_count(aid) == 1
    assert mem.prior_gist(aid) == gist.narrative
    assert mem.blurbs.facts(aid) == [{"p": "k", "o": "v"}]


def test_consolidate_refreshes_ring_no_dup() -> None:
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.20, ring_capacity=4)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0")
    _fade_anchor_to_r4(mem, aid)
    # `aid` was evicted from the ring by the cross-doc flood; consolidate
    # re-appends it exactly once at the tail.
    assert aid not in mem.ring
    mem.consolidate(aid, _StubGist("docA:0 the gist"))
    assert mem.ring.count(aid) == 1
    assert mem.ring[-1] == aid


def test_fading_anchors_filters() -> None:
    # Only cos < cos_gist+epsilon, not in ring, under max_depth, blurb present.
    # Most-faded-first; per-tick cap respected. ``fresh`` is ingested LAST so the
    # ring (cap 4) holds it -> excluded from the sweep.
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.20, ring_capacity=4)
    mem = FadeMemory(cfg, emb, _StubVoice())
    a = mem.ingest("docA:0")
    b = mem.ingest("docB:0")
    _fade_anchor_to_r4(mem, a)
    _fade_anchor_to_r4(mem, b)
    fresh = mem.ingest("docC:0")   # last -> in ring, high cos -> excluded
    assert fresh in mem.ring
    ids = mem.fading_anchors(epsilon=0.03, max_depth=3, max_per_tick=8)
    assert fresh not in ids
    assert set(ids) <= {a, b}
    # most-faded-first: sorted ascending by cos; just assert both eligible.
    assert a in ids and b in ids
    # cap respected.
    ids_cap = mem.fading_anchors(epsilon=0.03, max_depth=3, max_per_tick=1)
    assert len(ids_cap) <= 1


def test_consolidate_gist_of_gist() -> None:
    # Two consolidations: the second's prior_gist is the first narrative; count
    # climbs to 2; at max_depth the anchor is excluded from fading_anchors
    # (stays R4 -- the forgotten -> long-term-pull floor).
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.20, ring_capacity=2)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0")
    _fade_anchor_to_r4(mem, aid)
    g1 = "docA:0 first-level gist"
    mem.consolidate(aid, _StubGist(g1))
    assert mem.consolidation_count(aid) == 1
    assert mem.prior_gist(aid) == g1
    # Fade it again, then consolidate a second time (gist-of-gist).
    _fade_anchor_to_r4(mem, aid)
    g2 = "docA:0 zeroed second-level gist"
    mem.consolidate(aid, _StubGist(g2))
    assert mem.consolidation_count(aid) == 2
    assert mem.prior_gist(aid) == g2
    # At max_depth=2 the anchor is now excluded from the sweep.
    _fade_anchor_to_r4(mem, aid)
    assert aid not in mem.fading_anchors(epsilon=0.03, max_depth=2, max_per_tick=8)
    # But re-eligible at a higher cap (the cap gates, not blocks forever).
    assert aid in mem.fading_anchors(epsilon=0.03, max_depth=3, max_per_tick=8)


def test_consolidate_skip_when_narrative_none() -> None:
    # The gister returns None (cold-start: Bonsai down) -> the WORKER skips the
    # consolidation (the anchor stays R4). Here we assert the memory-level
    # invariant the worker relies on: a None gist never reaches consolidate, so
    # the anchor's count/blurb are untouched. (The worker's None-skip is tested
    # directly in test_consolidation.py.)
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.20, ring_capacity=2)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0")
    _fade_anchor_to_r4(mem, aid)
    before_text = mem.blurbs.text(aid)
    before_count = mem.consolidation_count(aid)
    # Simulate the worker's None-skip: do NOT call consolidate.
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN
    assert mem.blurbs.text(aid) == before_text
    assert mem.consolidation_count(aid) == before_count == 0

# --------------------------------------------------------------- Mamba3 voice
class _StubMamba3Out:
    """Mimics the HF-style ``model(input_ids)`` return: an object with ``.logits``."""

    def __init__(self, logits):
        self.logits = logits


class _StubMamba3Model:
    """Stand-in for ``MambaLMHeadModel``: argmax of the last position rotates each
    call so the greedy continuation is a deterministic token sequence. Records
    the prefix length it saw each call (forward-per-token -> it must grow)."""

    def __init__(self, vocab: int = 20):
        self.vocab = vocab
        self.calls = 0
        self.seen_seq: list[int] = []

    def __call__(self, cur):
        import torch

        self.calls += 1
        seq = cur.shape[1]
        self.seen_seq.append(seq)
        nxt = 10 + (self.calls % 5)      # deterministic rotating argmax
        logits = torch.full((1, seq, self.vocab), -1e4)
        logits[0, -1, nxt] = 0.0
        return _StubMamba3Out(logits)


class _StubHFTokenizer:
    """Stand-in for the Llama-3.1 ``AutoTokenizer``: encode -> fixed ids,
    eos -> 0 (never hit), decode -> a readable token-per-id string."""

    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3]                 # ignores text; non-empty -> proceeds

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


def test_mamba3_voice_expand_greedy_continuation():
    """Mamba3Voice.expand primes on the blurb, then forward-per-tokens: each
    call sees the growing prefix (the quadratic re-forward, since the CuTe
    step() kernel is unavailable). Greedy argmax -> deterministic continuation."""
    from src.subconscious.fade import Mamba3Voice

    model = _StubMamba3Model()
    voice = Mamba3Voice(model, _StubHFTokenizer(), device="cpu", temperature=0.0)
    out = voice.expand("the blurb", 4)
    assert out == "t11 t12 t13 t14"          # calls 1..4 -> argmax 11..14
    assert model.calls == 4
    # forward-per-token: the prefix the model sees grows by one each step.
    assert model.seen_seq == [3, 4, 5, 6]


def test_mamba3_voice_empty_blurb_passthrough():
    """An empty/whitespace blurb returns unchanged and never calls the model
    (the guard before encode -> forward)."""
    from src.subconscious.fade import Mamba3Voice

    model = _StubMamba3Model()
    voice = Mamba3Voice(model, _StubHFTokenizer(), device="cpu", temperature=0.0)
    assert voice.expand("   ", 4) == "   "
    assert model.calls == 0


def test_mamba3_voice_satisfies_voice_protocol():
    """Mamba3Voice has the ``expand(blurb, max_new_tokens) -> str`` contract the
    ``FadeMemory`` Regime-3 path calls (duck-typed, like ``_StubVoice``)."""
    from src.subconscious.fade import Mamba3Voice, Voice

    voice = Mamba3Voice(_StubMamba3Model(), _StubHFTokenizer(), device="cpu")
    assert callable(getattr(voice, "expand", None))
    # Voice is a Protocol (structural); the call site does not isinstance-check,
    # but the signature must match.
    assert voice.expand("x", 1) != "x"


def test_bundled_tcc_helper_is_safe():
    """``_bundled_tcc`` returns a path when triton-windows is installed and
    ``None`` otherwise -- never raises. Used by ``load_mamba3_voice`` to
    self-set ``CC`` for the TinyCC Triton JIT path."""
    from src.subconscious.fade import _bundled_tcc

    tcc = _bundled_tcc()
    assert tcc is None or isinstance(tcc, str)
