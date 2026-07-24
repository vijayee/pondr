"""STRM Phase 4 Step 4: the salience trigger.

Salience decides, from the WM ring + the live recurrent state, which anchors
(ring slots with provenance) the engine is *about to forget* and should
proactively recall from LTM before the user asks. The signal is internal --
state-conditioned, not prompt-triggered -- which is the whole point of Phase 4
(retrieval today is externally triggered only; ``retrieval_gate.py`` embeds the
prompt, never the state).

Definition (spec ``docs/STRM-implementation-plan.md:418-452`` + proposal §5
step 9)::

    salient(anchor) = (rec_i < theta) AND (r_i > phi) AND (surprise_i < surprise_cap)

Per anchor (a ring slot with ``source_id`` + ``text``):

* ``r_i`` -- frozen 2a relevance head via ``relevance_score.score_ring_slots``
  (reused, byte-identical to the graduation-logger / context-builder path).
  High = the slot is relevant to the current query -> contributes to salience.
* ``rec_i`` -- RECOVERABILITY score from the 2b head. The head's raw
  ``predict`` is a FORGETTING score (high = more forgotten; it was ridge-fit to
  the reconstruction error ``e = ||D(state) - anchor||^2``). We NEGATE it so
  ``rec_i`` is recoverability: LOW = likely forgotten = SALIENT. This honors
  the spec's ``rec_i < theta`` sign and the "low recoverability = forgotten"
  wording. ``theta`` is a low percentile of the val recoverability
  distribution -> only the most-forgotten anchors clear it.
* ``surprise_i`` -- 2c head's ``surprise(project(prev_state), project(state))``
  -- one scalar PER TURN (the transition surprise), applied to every anchor
  this turn. HIGH surprise = the turn is novel/unexpected -> SUPPRESS salience
  (do not pre-empt a novel turn with a proactive recall; proposal §5 step 9).
  ``surprise_cap`` is a high percentile of the val surprise distribution ->
  only very-surprising turns suppress.

Thresholds are NOT magic numbers: ``theta`` / ``phi`` / ``surprise_cap`` are
percentiles on the 2b / 2a / 2c val-score distributions, persisted in a
``thresholds.json`` sidecar (written by ``scripts/compute_salience_thresholds.py``)
with their percentile basis documented. See the plan's de-wonk note #5.

This module is pure (no orchestrator state). The orchestrator pre-retrieval
hook (Step 4) calls ``compute_salience`` and stashes the anchors for Step 5
(state-conditioned retrieval + pin-tagged re-inject) and Step 6 (freshness
watermark + stale-uncertain signal). Flag-off -> the hook never runs -> the
serve path is byte-identical to pre-Step-4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .latent_dynamics_head import LatentDynamicsHead
from .recoverability_head import RecoverabilityHead, pool_state_tensors
from .relevance_score import score_ring_slots_with_doc_embs


@dataclass(frozen=True)
class SalienceThresholds:
    """The three salience gate thresholds + their percentile basis.

    All three are percentiles on the named head's val-score distribution
    (computed by ``scripts/compute_salience_thresholds.py``). ``basis`` is a
    human-readable provenance string so a reader of the sidecar knows where each
    number came from without re-running the script.
    """

    theta: float               # recoverability percentile (LOW = forgotten = salient)
    phi: float                 # relevance percentile (HIGH = relevant = salient)
    surprise_cap: float        # surprise percentile (HIGH = suppress)
    theta_percentile: float    # the percentile p in [0,100] that produced theta
    phi_percentile: float
    surprise_cap_percentile: float
    basis: str = ""
    n_recoverability: int = 0  # val samples the percentile was taken over
    n_relevance: int = 0
    n_latent_dynamics: int = 0


def load_salience_thresholds(path: str) -> SalienceThresholds:
    """Load a ``thresholds.json`` sidecar -> ``SalienceThresholds``.

    The sidecar is written by ``scripts/compute_salience_thresholds.py`` and
    carries the three thresholds, their percentile basis, and the val-sample
    counts. Missing optional fields default to 0 / empty (the percentile fields
    are required -- a sidecar without them is a mis-wire).
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return SalienceThresholds(
        theta=float(d["theta"]),
        phi=float(d["phi"]),
        surprise_cap=float(d["surprise_cap"]),
        theta_percentile=float(d["theta_percentile"]),
        phi_percentile=float(d["phi_percentile"]),
        surprise_cap_percentile=float(d["surprise_cap_percentile"]),
        basis=str(d.get("basis", "")),
        n_recoverability=int(d.get("n_recoverability", 0)),
        n_relevance=int(d.get("n_relevance", 0)),
        n_latent_dynamics=int(d.get("n_latent_dynamics", 0)),
    )


def percentile_threshold(values, p: float) -> float:
    """The ``p``-th percentile of ``values`` (p in [0, 100]).

    A thin ``np.percentile`` wrapper that accepts a tensor or array and returns
    a python float. Used by the thresholds script and by tests that build a
    sidecar inline.
    """
    arr = values.detach().cpu().numpy() if isinstance(values, Tensor) else np.asarray(values)
    return float(np.percentile(arr.astype(np.float64), p))


@dataclass
class SalienceAnchor:
    """One ring slot scored for salience. Carried to Step 5 (retrieval) + Step 6
    (freshness watermark / stale-uncertain signal)."""

    slot_index: int
    source_id: Optional[str]
    text: Optional[str]
    r_i: Optional[float]            # relevance; None if unscoreable (no text/head)
    rec_i: Optional[float]          # recoverability (negated forgetting); low = forgotten
    surprise_i: Optional[float]     # per-turn transition surprise; high = suppress
    age: int                        # ring-position proxy (0 = newest); refined in Step 6
    salient: bool
    # The 384-d bge doc vector (re-embedded slot text) used as the 2b anchor AND
    # the Step 5 retrieval query (state-conditioned: this anchor only fires
    # because the state flagged it as being-forgotten). None for unscoreable
    # slots (no text) -- and those are never salient, so a salient anchor always
    # carries one. NOT the turn-level z_t projection: project(state_tensors) is
    # one vector per turn, so per-anchor retrievals on it would be identical
    # (dedup would collapse them and the budget cap would be theater). The
    # anchor's own doc vector gives per-anchor diversity (different forgotten
    # anchors recall different LTM episodes) -- the deferred Step 7 eval decides
    # whether this state-conditioned query shape beats fixed-interval RAG.
    doc_emb: Optional[Tensor] = None


# Step 5 budget cap: at most this many salience-fired retrievals per turn (the
# proactive-recall budget). A proactive recall should fire rarely -- the salience
# AND is already selective, and this caps the worst-case per-turn retrieval
# blast. The deferred Step 7 eval measures the actual per-turn count (surfaced
# in the result dict) to tune this against fixed-interval RAG at equal budget.
SALIENCE_RETRIEVAL_BUDGET = 3


def _decide_salience(
    r_i: Optional[float],
    rec_i: Optional[float],
    surprise_i: Optional[float],
    thresholds: SalienceThresholds,
) -> bool:
    """The pure sign logic. ``salient = (rec_i < theta) AND (r_i > phi) AND
    (surprise_i < surprise_cap)``. A None score (unscoreable slot -- no text, no
    head) is NEVER salient: an anchor we cannot score is not one we can
    responsibly pre-empt the user with. Pinned by the sign tests so Step 5 wires
    the comparison the right way around (low recoverability -> salient, high
    surprise -> suppress, high relevance -> salient)."""
    if r_i is None or rec_i is None or surprise_i is None:
        return False
    return (rec_i < thresholds.theta) and (r_i > thresholds.phi) and (surprise_i < thresholds.surprise_cap)


def compute_salience(
    ring_slots: list,
    state_tensors: list[Tensor],
    prev_state_tensors: list[Tensor],
    working_memory,
    relevance_head,
    recoverability_head: Optional[RecoverabilityHead],
    latent_dynamics_head: Optional[LatentDynamicsHead],
    embedder,
    query_emb: Tensor,
    thresholds: SalienceThresholds,
) -> list[SalienceAnchor]:
    """Score every ring slot for salience. Pure (no orchestrator state).

    Args:
        ring_slots: ``working_memory.ring_buffer()`` (oldest-first). Slots with
            no ``source_id`` / ``text`` get ``r_i = None`` (via
            ``score_ring_slots``) and are never salient.
        state_tensors: the POST-query-step live WM state (``state_tensors()``)
            -- the 2b head reads it as "what's in mind right now".
        prev_state_tensors: the PRE-query-step state (cloned BEFORE
            ``working_memory.update(prompt_emb)``) -- the 2c surprise term needs
            both states (``surprise(z_t, z_{t+1})``).
        working_memory: passed for ``score_ring_slots`` (it re-embeds slot text
            via ``working_memory.embed`` -- the canonical r_i path, reused).
        relevance_head / recoverability_head / latent_dynamics_head: the three
            STRM read-out heads. Any None -> the corresponding score is None for
            every anchor -> no anchor is salient (the trigger is all-three-AND,
            so a missing head disarms it).
        embedder: the bge embedder, passed through to
            ``score_ring_slots_with_doc_embs`` as a None-guard (scoring is
            skipped when no embedder is wired -> r_i = None -> not salient).
        query_emb: ``[1, 384]`` or ``[384]`` -- the current prompt embedding,
            for the 2a relevance scoring.
        thresholds: ``SalienceThresholds`` (percentiles from the val sidecar).

    Returns:
        One ``SalienceAnchor`` per ring slot (oldest-first), each carrying its
        scores + the ``salient`` decision. Empty list if the ring is empty.
    """
    n = len(ring_slots)
    if n == 0:
        return []

    # r_i per slot AND the re-embedded 384-d doc vector per slot, in ONE pass
    # (the canonical 2a path; None for unscoreable slots -- no text / no head).
    # The 384-d doc vector doubles as the 2b anchor: the recoverability head was
    # trained on (state_pooled 1536, anchor u_i 384) where u_i is the episode's
    # INPUT embedding, NOT the slot's 256-d step output ``y``. The slot stores
    # only ``y`` (256-d) + ``text``, so the re-embedded text (384-d) is the
    # faithful at-serve proxy for the original u_i. Slots with no text get
    # ``doc_embs[i] = None`` -> rec_i = None -> not salient (unscoreable).
    _slots, r_is, doc_embs = score_ring_slots_with_doc_embs(
        working_memory, relevance_head, embedder, query_emb, slots=ring_slots,
    )

    # rec_i per slot: recoverability = NEGATED forgetting score (low = forgotten).
    rec_is: list[Optional[float]] = [None] * n
    if recoverability_head is not None and state_tensors is not None:
        head_dev = next(recoverability_head.parameters()).device
        state_pooled = pool_state_tensors(state_tensors).to(head_dev)  # [1, 1536]
        with torch.no_grad():
            for i, slot in enumerate(ring_slots):
                anchor = doc_embs[i]
                if anchor is None:
                    continue  # unscoreable slot -> rec_i stays None -> not salient
                anchor = anchor.to(torch.float32).squeeze(0).reshape(1, -1).to(head_dev)  # [1, 384]
                forgetting = recoverability_head.predict(state_pooled, anchor)  # [1,1]
                rec_is[i] = -float(forgetting.detach())

    # surprise_i: ONE scalar per turn (the transition surprise), applied to all
    # anchors. high surprise -> suppress.
    surprise_i: Optional[float] = None
    if latent_dynamics_head is not None and prev_state_tensors is not None and state_tensors is not None:
        ld_dev = next(latent_dynamics_head.parameters()).device
        with torch.no_grad():
            z_t = latent_dynamics_head.project(prev_state_tensors).to(ld_dev)       # [1, 384]
            z_tp1 = latent_dynamics_head.project(state_tensors).to(ld_dev)          # [1, 384]
            surprise_i = float(latent_dynamics_head.surprise(z_t, z_tp1).detach())

    anchors: list[SalienceAnchor] = []
    for i, slot in enumerate(ring_slots):
        # age: ring-position proxy (0 = newest slot). Refined to real turn
        # timestamps in Step 6's freshness watermark.
        age = (n - 1) - i
        salient = _decide_salience(r_is[i], rec_is[i], surprise_i, thresholds)
        anchors.append(SalienceAnchor(
            slot_index=i,
            source_id=slot.source_id,
            text=slot.text,
            r_i=r_is[i],
            rec_i=rec_is[i],
            surprise_i=surprise_i,
            age=age,
            salient=salient,
            doc_emb=doc_embs[i],
        ))
    return anchors


def salient_anchors(anchors: list[SalienceAnchor]) -> list[SalienceAnchor]:
    """Convenience: the subset of ``anchors`` with ``salient=True`` (oldest-first)."""
    return [a for a in anchors if a.salient]


# ── cosine+age salience (the OOD-immune trigger) ──────────────────────────────
#
# The learned-head AND (rec_i<theta)&(r_i>phi)&(surprise_i<surprise_cap) was the
# Phase 4 ship-eval blocker: ALL THREE heads are OOD-fragile at serve (see
# pondr-strm-salience-three-heads-ood.md). surprise blocks 100% of turns (serve
# surprise ~100-280x above the val-percentile cap); rec_i never crosses theta
# (and is run-to-run unstable); r_i saturates near 1.0 and never discriminates
# (the yt_sidepath OOD offset pins it high). The permissive upper-bound smoke
# ``works`` only because it sets all three thresholds to +/-inf -- i.e. it
# ignores the heads and fires on any provenance-bearing slot, which trivially
# hits the right fact in a synthetic one-fact ring but does NOT test targeting.
#
# The fix: drop the three learned heads for the salience DECISION and use two
# frozen, OOD-immune signals -- bge cosine (a fixed encoder, never OOD) for
# TARGETING (which slot is relevant to the query) + ring age (a deterministic
# forgotten-proxy, no learned head) for the "old enough to bother recalling"
# gate. The Step 5 retrieval + Step 6 signal path is shared unchanged (the
# anchors carry ``doc_emb`` so ``retrieve_by_embedding`` works identically).
#
# Validated on the synthetic ship bank: cross-fact cosine discriminates 6/6
# (own fact is the max over the other 5, median gap +0.263) -- the targeting the
# saturated 2a head cannot do. The real-serve bar (cosine vs OTHER RECALLED
# episodes, not unrelated fillers) is tighter and is the real ship gate (the
# documented real-onyx held-out follow-on); this mode is the synthetic clearing.


@dataclass(frozen=True)
class CosineAgeSalienceConfig:
    """The frozen-signal salience config (no learned heads).

    ``salient = (bge_cos(query_emb, slot_doc_emb) > cos_phi) AND
    (age >= age_threshold)`` where ``age`` is the ring-position proxy (0 = newest
    slot). ``cos_phi`` is a serve-calibrated cosine floor (NOT a val percentile
    of a learned head -- bge is frozen so this is distribution-stable). Slots
    with no text/provenance are unscoreable -> never salient (same rule as the
    learned path). OOD-immune: no 2a/2b/2c head is read.
    """

    cos_phi: float            # min bge_cos(query, slot_text) to be salient
    age_threshold: int         # min ring-age (turns since written proxy) to be salient
    basis: str = ""


def compute_salience_cosine_age(
    ring_slots: list,
    working_memory,
    embedder,
    query_emb: Tensor,
    config: CosineAgeSalienceConfig,
) -> list[SalienceAnchor]:
    """Score every ring slot with frozen bge cosine + ring age (no learned heads).

    Mirrors ``compute_salience``'s shape so the orchestrator's Step 5/6 path is
    reused unchanged: one ``SalienceAnchor`` per ring slot (oldest-first) with
    ``r_i`` = the bge cosine score (carried to the consumer signal), ``rec_i``
    / ``surprise_i`` = None (not used), ``doc_emb`` = the re-embedded slot text
    (the Step 5 retrieval query), and ``salient`` = the cosine+age decision.
    Slots with no text get ``r_i = None`` -> not salient (unscoreable), matching
    the learned path. Empty list if the ring is empty.
    """
    n = len(ring_slots)
    if n == 0:
        return []

    r_is: list[Optional[float]] = [None] * n
    doc_embs: list = [None] * n
    if embedder is not None:
        idx_text = [(i, s.text) for i, s in enumerate(ring_slots)
                    if s.text is not None and str(s.text).strip()]
        if idx_text:
            # Re-embed each slot's text (the SAME 384-d bge path the learned r_i
            # uses for its doc vector); this doubles as the Step 5 query.
            doc_emb_tensors = working_memory.embed([t for _, t in idx_text])
            q = query_emb.detach().to(torch.float32).reshape(-1)
            q_norm = torch.norm(q) + 1e-12
            for j, (i, _) in enumerate(idx_text):
                d = doc_emb_tensors[j].to(torch.float32).reshape(-1)
                cos = float(torch.dot(q, d) / (q_norm * (torch.norm(d) + 1e-12)))
                r_is[i] = cos
                doc_embs[i] = doc_emb_tensors[j]

    anchors: list[SalienceAnchor] = []
    for i, slot in enumerate(ring_slots):
        age = (n - 1) - i  # ring-position proxy (0 = newest); oldest slot = n-1
        salient = (r_is[i] is not None
                   and r_is[i] > config.cos_phi
                   and age >= config.age_threshold)
        anchors.append(SalienceAnchor(
            slot_index=i,
            source_id=slot.source_id,
            text=slot.text,
            r_i=r_is[i],
            rec_i=None,
            surprise_i=None,
            age=age,
            salient=salient,
            doc_emb=doc_embs[i],
        ))
    return anchors


def format_salience_gap(signals: list[dict]) -> str:
    """Build the consumer-facing gap statement for ``stale_uncertain`` signals.

    Proposal sec 5: do not lie by omission. When the salience trigger fired
    retrieval for a young anchor and got nothing back (the episode may be known
    but not yet fully ingested by Thread 2's async-distill worker), surface a
    STATED gap so the consumer can wait / re-ask / proceed with eyes open rather
    than being silently suppressed. Returns ``""`` when there are no
    ``stale_uncertain`` signals (the byte-identical flag-off case).
    """
    stale = [s for s in signals if s.get("kind") == "stale_uncertain"]
    if not stale:
        return ""
    texts = [s.get("text") for s in stale if s.get("text")]
    if texts:
        joined = "; ".join(texts)
        return (f"I may know this but have not finished ingesting it: {joined}.")
    return "I may know this but have not finished ingesting it."