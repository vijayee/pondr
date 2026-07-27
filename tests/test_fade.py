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