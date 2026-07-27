"""The fade memory -- dual-SSM recall that degrades from verbatim to gist.

Implements the Stage-1 fade architecture (``docs/fade-architecture.md``, task #34):
an SSM memory that recalls recent content verbatim and degrades to the gist of
older content as the state compresses over a streamed session. The fade EMERGES
from real state degradation (the SSM-A vector channel decaying under the
recurrence), not a policy switch on wall-clock age.

The two legs (validated by the two no-training gating probes, both PASS):
  - SSM-A (the fade leg): a 384-d bge-space vector-carry channel. Each ingested
    chunk writes its frozen ``bge(chunk)`` vector into the channel; the recurrence
    decays older vectors as newer ones stream in. The faded state is read as a
    retrieval query. THE FADE LIVES HERE -- in the vector's graceful decay
    (probe #32: exact -> sibling/gist -> unrelated, tunable by decay). No text is
    decoded from the state (decoding is the 3x-disproven path).
  - SSM-B (the voice leg): the token-LM ``SSMLanguageModel``. Given a retrieved
    blurb (a short text excerpt), EXPANDS it into fuller recall via continuation
    (probe #31: the token-LM is the voice, not a fade substrate -- only ~1-2
    tokens of verbatim then prior-only).

The router (the keystone): because SSM-A's state IS a blend of bge vectors (it
lives in the same 384-d space as the anchors), the recoverability signal is FREE
-- ``e(i,t) = 1 - cos(state_t, bge(anchor_i))`` is both the retrieval score AND
the forgetting signal. No ridge decoder, no trained head. This is the Sec-6.1
recovery-decoder recipe (``e = ||D(state) - anchor||^2``) realized directly by
cosine, because the channel is already in the anchor's space -- the payoff of the
bge-space channel decision (Option 2). Probe #32 showed this cosine tracks the
fade: 1.0 (N=0) -> 0.95 (N=1) -> ~0.8 plateau (the tip-of-tongue floor).

The four regimes (selected per-anchor by ``e(i,t)`` at recall time):
  1. low e -- still in the state -> VERBATIM from the ring. The ring is a recency
     window that gives true verbatim independent of the state's fade (the
     architecture's "the ring gives true verbatim for the ring window regardless").
     Gate: anchor in the ring, OR ``cos >= cos_ring`` (state still fresh).
  2. medium e -- degraded but residual -> Regime 3 behavior (retrieve + expand).
     The Transformer fill-holes readout (``CrossSlotTransformerZHead``) is a
     deprioritized Stage-2 layer (probe #31: the token-LM state residual is thin);
     ``regime2_enabled=False`` by default, so medium-e falls through to Regime 3.
  3. high e, vector still retrievable (``cos >= cos_gist``) -- gone from state,
     address survives -> the faded state retrieves a (possibly sibling) blurb ->
     SSM-B expands it. Fuzzy gist.
  4. high e, vector too faded (``cos < cos_gist``) -> TRULY FORGOTTEN -> the system
     says "forgotten" rather than confabulating. The graceful floor
     (metacognition / tip-of-tongue).

Isolated module: no orchestrator/runtime/serve IMPORTS (the wiring flows one way --
``runtime.build_ponder`` constructs and injects a ``FadeMemory``; this module never
imports the serve path). The voice (SSM-B) and the embedder (bge) are injected
(``Voice`` / ``Embedder`` protocols) so the unit test runs CPU-only with synthetic
vectors and a test-double voice; production wires the real token-LM
(``load_token_lm_voice``) + the shared bge embedder (reused from ``build_ponder``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

# Regime labels (integers so they serialize trivially).
REGIME_VERBATIM = 1     # ring / state-fresh -> exact text
REGIME_FILL = 2         # Transformer fill-holes (Stage 2, deprioritized)
REGIME_GIST = 3         # faded state retrieves a blurb -> SSM-B expands
REGIME_FORGOTTEN = 4    # tip-of-tongue floor -> "forgotten" marker

# Human-readable regime names (kept here next to the labels so callers -- the
# orchestrator, the eval scripts -- do not each redefine the mapping; the eval
# scripts' own ``REGIME_NAME`` mirror this).
REGIME_NAME: dict[int, str] = {
    REGIME_VERBATIM: "verbatim",
    REGIME_FILL: "fill",
    REGIME_GIST: "gist",
    REGIME_FORGOTTEN: "forgotten",
}


def format_fade_block(recalls: list[dict]) -> str:
    """Render fade recalls as a ``[FADE MEMORY]`` block for the LLM context.

    Takes the dict shape the orchestrator's RECALL seam builds
    (``{anchor_id, regime, regime_name, cos, content, blurb}``). Lists only
    CONTENT-BEARING recalls:

    - R1 (verbatim): the recent exact text.
    - R3 (gist): the anchor's blurb (verbatim when ``voice is None`` -- Phase B;
      a paraphrase once the token-LM voice leg is wired).

    R4 (forgotten) is intentionally SKIPPED: it is a SIGNAL for a future long-term-
    memory pull (mechanism TBD -- explicit tool call vs background fulfillment), not
    LLM-facing content. The forgotten material is simply absent from the block -- the
    gradient "exact -> gist -> (gone)" reads naturally, like a human who does not
    enumerate what they have forgotten. R4 stays in ``fade_recalls`` (observability +
    the signal the future pull consumes).

    Returns ``""`` when no content-bearing recalls are present (so the block is
    omitted entirely -- no empty header). Bounded by the seam's ``top_k`` /
    ``blurb_chars`` caps (the recalls arrive already capped).
    """
    lines: list[str] = []
    for r in recalls:
        regime = r.get("regime")
        if regime == REGIME_FORGOTTEN:
            continue  # R4 is a signal, not LLM content (see docstring).
        content = (r.get("content") or "").strip()
        if not content:
            continue
        if regime == REGIME_VERBATIM:
            lines.append(f"[verbatim, recent] {content}")
        elif regime == REGIME_GIST:
            lines.append(f"[gist, fading] {content}")
        elif regime == REGIME_FILL:
            lines.append(f"[fill, reconstructed] {content}")
        else:
            lines.append(f"[{r.get('regime_name') or regime}] {content}")
    if not lines:
        return ""
    header = ("[FADE MEMORY -- your fading working memory of this conversation]\n"
              "(Recent exchanges are recalled exactly; older ones are given as a "
              "fading gist. What has fully faded is not listed.)")
    return header + "\n" + "\n".join(lines)


class Embedder(Protocol):
    """``texts -> list of dim-d float vectors`` (bge-small-en-v1.5, 384-d, frozen)."""

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Voice(Protocol):
    """SSM-B: expand a retrieved blurb into fuller recall via continuation."""

    def expand(self, blurb: str, max_new_tokens: int) -> str: ...


@dataclass
class FadeConfig:
    """Fade-memory hyperparameters. Defaults calibrated from probe #32 and the
    cross-domain eval (``scripts/eval_fade_cross_domain.py``; see ``cos_gist``).

    ``decay``: the SSM-A EWMA decay (the fade timescale). 0.99 -> ~8-16-step
    gist window (probe #32). ``cos_ring``/``cos_gist``: the regime boundaries on
    the free cosine router. ``ring_capacity``: the recency verbatim window.
    ``blurb_chars``: max chars of chunk text stored as the blurb (a gist-sized
    excerpt -- the content seed SSM-B expands). ``expand_tokens``: SSM-B
    continuation length for a gist. ``regime2_enabled``: the Transformer
    fill-holes readout (Stage 2, deprioritized per probe #31).
    """

    decay: float = 0.99
    write_gate: float = 1.0
    cos_ring: float = 0.95      # cos >= this (or anchor in ring) -> Regime 1
    # cos >= this (and not R1) -> Regime 3 (gist); below it -> Regime 4 (forgotten).
    # Calibrated for REAL bge-small-en-v1.5 by the cross-domain eval
    # (scripts/eval_fade_cross_domain.py): bge-small has a HIGH cosine floor --
    # same-domain ~0.6, cross-domain ~0.37 -- so the threshold must sit BETWEEN
    # them to separate "same topic -> fuzzy gist (R3)" from "different topic ->
    # forgotten (R4)". 0.40 (probe #32's 0.30 was calibrated on the synthetic test
    # embedder whose cross-doc floor is ~0.01, far below real bge's ~0.37, so 0.30
    # never reached R4 on real bge). The unit tests override this (their synthetic
    # embedder has a ~0.01 cross-doc floor, so they use 0.15-0.20).
    cos_gist: float = 0.40
    ring_capacity: int = 32     # recent anchors kept verbatim
    blurb_chars: int = 600      # max chars of chunk text stored as the blurb
    expand_tokens: int = 64     # SSM-B continuation length for a gist
    regime2_enabled: bool = False  # Transformer fill-holes (Stage 2, off)


# ----------------------------------------------------------------------- SSM-A
class VectorCarrySSM:
    """SSM-A: a 384-d bge-space vector-carry channel (the fade leg).

    The no-training stand-in validated by probe #32: an exponentially-weighted
    moving average of chunk bge vectors,

        state_p = decay * state_{p-1} + write_gate * bge(chunk_p),   state_0 = 0

    so the anchor at stream position ``i`` contributes ``decay**N *
    bge(chunk_i)`` to ``state_{i+N}``. The fade is NOT "the vector shrinks"
    (cosine retrieval is scale-invariant -- the probe's read-only control stayed
    100% exact forever); it is the recurrence OVERWRITING the slot with newer
    chunk vectors, so the query drifts anchor -> recent-chunk blend -> unrelated.

    Structured to upgrade to a trained ``SelectiveSSM`` with selective gating
    (write only important chunks, preserve others) -- the probe's "exact only at
    N=0" is the EWMA's conservative lower bound; selective gating should EXTEND
    the exact window. That upgrade is a Stage-2 follow-on (a subclass overriding
    ``step``); the EWMA is the validated substrate for Stage 1.
    """

    def __init__(self, dim: int, decay: float, write_gate: float = 1.0) -> None:
        self.dim = int(dim)
        self.decay = float(decay)
        self.write_gate = float(write_gate)
        self._state = np.zeros(self.dim, dtype=np.float32)

    def step(self, bge_vec: np.ndarray) -> np.ndarray:
        """Write one chunk's bge vector into the channel; return the new state."""
        v = np.asarray(bge_vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"bge_vec length {v.shape[0]} != dim {self.dim}")
        self._state = self.decay * self._state + self.write_gate * v
        return self._state

    def state(self) -> np.ndarray:
        """The raw (unnormalized) ``dim``-d channel state at the current step."""
        return self._state

    def query(self) -> np.ndarray:
        """The state L2-normalized as a retrieval query.

        float64 norm for underflow-safety at extreme ``decay**N`` (probe #32's
        control fix: a float32 norm squares ~1e-29 components into 0 at
        decay=0.5, N>=96, leaving the query un-normalized). The FADE path (state
        O(1)) is unaffected; the safety matters for the read-only control and for
        very long fast-decay streams.
        """
        s64 = self._state.astype(np.float64)
        nrm = float(np.linalg.norm(s64))
        if nrm == 0.0:
            nrm = 1.0
        return (s64 / nrm).astype(np.float32)

    def reset(self) -> None:
        self._state = np.zeros(self.dim, dtype=np.float32)


# ------------------------------------------------------------------ blurb store
class BlurbStore:
    """External store of ``(bge(chunk), blurb_text)`` keyed by anchor_id.

    The bridge: blurbs are stored at ingestion (the chunk's text, gist-sized),
    keyed by the chunk's bge vector. Retrieval is cosine against the stored bge
    vectors -- the same ``corpus @ q`` the probe validated. ``add`` L2-normalizes
    the bge (cosine is scale-invariant, so the stored vector and the SSM-A query
    share a convention); ``retrieve(q)`` returns the top-k ``(anchor_id, cos,
    text)``. In-memory numpy: the architecture's "external store" is satisfied by
    keeping the blurb OUT of the SSM state (it is not decoded from the state);
    persistence to WaveDB is a follow-on, not a Stage-1 need.
    """

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self._ids: list[int] = []
        self._vecs: list[np.ndarray] = []     # L2-normalized [dim] fp32
        self._texts: list[str] = []
        self._index: dict[int, int] = {}      # anchor_id -> row
        self._matrix: Optional[np.ndarray] = None  # [N, dim] rebuilt lazily

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v64 = np.asarray(v, dtype=np.float64).reshape(-1)
        nrm = float(np.linalg.norm(v64))
        if nrm == 0.0:
            nrm = 1.0
        return (v64 / nrm).astype(np.float32)

    def add(self, anchor_id: int, bge_vec: np.ndarray, text: str) -> None:
        """Store one chunk's blurb keyed by its (normalized) bge vector."""
        vn = self._normalize(bge_vec)
        if vn.shape[0] != self.dim:
            raise ValueError(f"bge_vec length {vn.shape[0]} != dim {self.dim}")
        self._index[anchor_id] = len(self._ids)
        self._ids.append(anchor_id)
        self._vecs.append(vn)
        self._texts.append(text)
        self._matrix = None  # invalidate the cached retrieval matrix

    def _mat(self) -> Optional[np.ndarray]:
        if self._matrix is None and self._vecs:
            self._matrix = np.stack(self._vecs, axis=0)  # [N, dim]
        return self._matrix

    def retrieve(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float, str]]:
        """Top-k ``(anchor_id, cos, text)`` by cosine of ``query`` against the
        stored bge vectors, best first. ``query`` is L2-normalized here."""
        mat = self._mat()
        if mat is None or len(self._ids) == 0:
            return []
        q = self._normalize(query)
        sims = mat @ q  # [N] cosine (both normalized)
        n = len(sims)
        k = max(0, min(k, n))
        if k == 0:
            return []
        # argpartition for top-k, then sort those k by score descending.
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]
        return [(self._ids[i], float(sims[i]), self._texts[i]) for i in order]

    def vector(self, anchor_id: int) -> Optional[np.ndarray]:
        """The stored L2-normalized bge for ``anchor_id`` (for the router's
        per-anchor cosine). None if not present."""
        i = self._index.get(anchor_id)
        return None if i is None else self._vecs[i]

    def text(self, anchor_id: int) -> Optional[str]:
        i = self._index.get(anchor_id)
        return None if i is None else self._texts[i]

    def __len__(self) -> int:
        return len(self._ids)


# --------------------------------------------------------------------- the voice
class TokenLMVoice:
    """SSM-B: the token-LM expands a retrieved blurb via continuation.

    Wraps the trained ``SSMLanguageModel`` (``pondr-token-lm-ssm-result``). Given
    a blurb (a short text excerpt), encodes it as a prompt and autoregressively
    continues via ``generate`` -- which primes the recurrent state with the
    prompt in one forward pass, then steps, so the state actually carries the
    blurb (the point of the LM build). Research substrate; production swaps in
    the serving LLM as the voice.
    """

    def __init__(self, model, tokenizer, device: str = "cpu",
                 temperature: float = 0.7, top_k: int = 40) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = float(temperature)
        self.top_k = int(top_k)

    def expand(self, blurb: str, max_new_tokens: int) -> str:
        import torch  # lazy: keeps the module importable without torch for tests

        ids = self.tokenizer.encode(blurb)
        if not ids:
            return blurb
        prompt = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(prompt, max_new_tokens,
                                  self.temperature, self.top_k)  # [1, seq+new]
        new_ids = out[0].tolist()[len(ids):]
        return self.tokenizer.decode(new_ids)


def load_token_lm_voice(checkpoint_path: str, tokenizer_path: str,
                         device: str = "cpu", temperature: float = 0.7,
                         top_k: int = 40) -> TokenLMVoice:
    """Load the trained token-LM as the SSM-B voice. Lazy torch import.

    ``checkpoint_path`` / ``tokenizer_path`` are the token-LM ckpt + tokenizer
    (``data/token_lm/...``; see ``pondr-token-lm-ssm-result``). The model is
    frozen (``eval()``) -- the voice expands, it does not train."""
    import torch

    from .token_lm import LMConfig, SSMLanguageModel
    from .tokenizer_ import train_or_load_tokenizer

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LMConfig(**ckpt["config"])
    model = SSMLanguageModel(cfg).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = train_or_load_tokenizer(iter([]), tokenizer_path, vocab_size=cfg.vocab)
    return TokenLMVoice(model, tok, device, temperature, top_k)


def bge_embedder() -> Embedder:
    """The frozen bge-small-en-v1.5 embedder (384-d). Lazy import."""
    from src.retrieval.vector_search import _sentence_transformers_embedder

    return _sentence_transformers_embedder()


# ----------------------------------------------------------------- the recall
@dataclass
class Recall:
    """One anchor's recall result at the current stream step."""

    anchor_id: int
    regime: int                 # REGIME_* label
    cos: float                  # cos(state_t, bge(anchor_i)) -- the free router
    content: str                # verbatim text (R1) / expanded blurb (R3) / marker (R4)
    blurb: Optional[str] = None  # the retrieved blurb before expansion (R3 only)


# ----------------------------------------------------------------- FadeMemory
class FadeMemory:
    """The dual-SSM fade memory (Stage 1).

    Holds SSM-A (``VectorCarrySSM``), the blurb store, the ring (recency verbatim
    window), and an injected voice (SSM-B). The router is the free cosine
    ``e(i,t) = 1 - cos(state_t, bge(anchor_i))`` -- no trained head, no ridge fit
    (the channel is in bge space). ``ingest`` streams a chunk; ``recall_anchor``
    routes one anchor to its regime; ``recall`` routes a query's relevant past
    anchors. ``voice`` is optional: when ``None``, Regime 3 returns the retrieved
    blurb verbatim (the built-in passthrough -- no separate stub voice class), so
    the memory runs without loading the token-LM; production wires the real
    token-LM (``load_token_lm_voice``) for continuation expansion.
    """

    def __init__(self, cfg: FadeConfig, embedder: Embedder,
                 voice: Optional[Voice] = None, dim: int = 384) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.voice = voice
        self.dim = int(dim)
        self.ssm_a = VectorCarrySSM(self.dim, cfg.decay, cfg.write_gate)
        self.blurbs = BlurbStore(self.dim)
        self.ring: deque[int] = deque(maxlen=cfg.ring_capacity)
        self._next_id = 0

    # -- ingestion -----------------------------------------------------
    def ingest(self, chunk_text: str) -> int:
        """Encode the chunk, step SSM-A, store the blurb, push to the ring.

        Returns the anchor_id (the chunk's position in the stream). The bge
        vector is the chunk's OWN frozen embedding (no teacher LLM -- bge is
        frozen); it is what SSM-A carries and what the blurb is keyed by."""
        anchor_id = self._next_id
        self._next_id += 1
        vec = self._encode_one(chunk_text)         # [dim] bge (unnormalized ok)
        self.ssm_a.step(vec)                        # the fade leg advances
        blurb = chunk_text[: self.cfg.blurb_chars]
        self.blurbs.add(anchor_id, vec, blurb)      # keyed by bge
        self.ring.append(anchor_id)                 # recency verbatim window
        return anchor_id

    def _encode_one(self, text: str) -> np.ndarray:
        raw = self.embedder.encode([text])[0]
        return np.asarray(raw, dtype=np.float32)

    # -- the free router ----------------------------------------------
    def _recoverability(self, anchor_id: int) -> Optional[float]:
        """``cos(state_t, bge(anchor_i))`` -- the free recoverability signal.

        High = the anchor is still in the state (low e -> Regime 1); low = faded
        (high e -> Regime 3/4). None if the anchor is not in the store."""
        v = self.blurbs.vector(anchor_id)
        if v is None:
            return None
        q = self.ssm_a.query()  # L2-normalized state
        return float(np.dot(q, v))  # both normalized -> cosine

    # -- per-anchor regime dispatch -----------------------------------
    def recall_anchor(self, anchor_id: int) -> Optional[Recall]:
        """Route one anchor by its recoverability -> the form the recall takes.

        Returns None if the anchor is unknown. The ring provides true verbatim
        for the recency window independent of the state's fade; the cosine
        router handles everything older (gist / forgotten)."""
        cos_i = self._recoverability(anchor_id)
        if cos_i is None:
            return None
        cfg = self.cfg
        in_ring = anchor_id in self.ring
        # Regime 1: recency window (ring) OR state still fresh.
        if in_ring or cos_i >= cfg.cos_ring:
            return Recall(anchor_id, REGIME_VERBATIM, cos_i,
                          content=self._verbatim(anchor_id))
        # Regime 2 (Transformer fill-holes) is deprioritized (probe #31: thin).
        # When enabled, medium-e is labeled FILL; without the Stage-2
        # ``CrossSlotTransformerZHead`` wired, the honest degraded behavior is the
        # same retrieve+expand Regime 3 does (the label is the dispatch decision;
        # the Transformer readout is the Stage-2 differentiator that replaces
        # this). When disabled (default), medium-e falls through to Regime 3.
        if cfg.regime2_enabled and cos_i >= cfg.cos_gist:
            return self._retrieve_and_expand(anchor_id, cos_i, REGIME_FILL)
        # Regime 3: faded but the address survives -> retrieve + expand.
        if cos_i >= cfg.cos_gist:
            return self._retrieve_and_expand(anchor_id, cos_i, REGIME_GIST)
        # Regime 4: tip-of-tongue floor -> forgotten (do not confabulate).
        return Recall(anchor_id, REGIME_FORGOTTEN, cos_i, content="[forgotten]")

    def _verbatim(self, anchor_id: int) -> str:
        # The ring is the preferred verbatim source; the blurb store holds the
        # same text for every anchor (the source of truth), so an anchor evicted
        # from the ring but still state-fresh (cos >= cos_ring) still returns its
        # exact text.
        return self.blurbs.text(anchor_id) or ""

    def _retrieve_and_expand(self, anchor_id: int, cos_i: float,
                             regime: int) -> Recall:
        # Regime 3 (and the degraded Regime 2): the anchor's ADDRESS survives
        # (cos_i >= cos_gist = the state still carries a recoverable trace of
        # it), so retrieve the anchor's OWN blurb and let SSM-B expand it. The
        # state's role is the recoverability SIGNAL (cos -> regime), NOT the
        # retrieval key: a state-closest blurb is dominated by the most-recent
        # chunk and drifts to wrong-topic content across a mixed-domain
        # session (the R3 content-drift found in the serve REPL validation --
        # docs/fade-serve-validation-result.md; the within-domain "state
        # retrieves a sibling" of probe #32 only holds when the state is still
        # dominated by the anchor's domain). The fade is expressed through the
        # SSM-B expansion (paraphrase) + the regime label (lower confidence),
        # not through retrieving a sibling. If the anchor's blurb is gone,
        # degrade to R4. (Mirrors ``_verbatim`` -- the blurb store is the source
        # of truth, addressed by anchor_id.)
        blurb = self.blurbs.text(anchor_id)
        if blurb is None:
            return Recall(anchor_id, REGIME_FORGOTTEN, cos_i, content="[forgotten]")
        # ``voice is None`` -> the built-in passthrough: return the anchor's
        # blurb verbatim (no token-LM loaded). Production wires ``TokenLMVoice``
        # for continuation expansion; tests + the Phase-A serve path run
        # passthrough.
        if self.voice is None:
            return Recall(anchor_id, regime, cos_i, content=blurb, blurb=blurb)
        expanded = self.voice.expand(blurb, self.cfg.expand_tokens)
        return Recall(anchor_id, regime, cos_i, content=expanded, blurb=blurb)

    # -- query-driven recall ------------------------------------------
    def recall(self, query_text: str, top_k: int = 5) -> list[Recall]:
        """Recall past content relevant to ``query_text``, each in the form its
        recency dictates.

        Step 1: retrieve candidate anchors by the query's bge vs the blurb store
        (relevance -- standard semantic search). Step 2: route each candidate by
        the free cosine router (recency -> verbatim / gist / forgotten). Returns
        the routed recalls, best-relevance first."""
        q = self._encode_one(query_text)
        candidates = self.blurbs.retrieve(q, k=top_k)  # (anchor_id, cos_q, text)
        out: list[Recall] = []
        for anchor_id, _, _ in candidates:
            r = self.recall_anchor(anchor_id)
            if r is not None:
                out.append(r)
        return out

    # -- lifecycle -----------------------------------------------------
    def reset(self) -> None:
        """Zero the state, clear the blurb store and ring (a fresh session)."""
        self.ssm_a.reset()
        self.blurbs = BlurbStore(self.dim)
        self.ring.clear()
        self._next_id = 0