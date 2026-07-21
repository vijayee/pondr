"""STRM Probe 4a: 2a relevance-head SELECTIVITY on REAL serve data (no LLM).

The truer ship gate after the OOD finding. Probe 3 ([[pondr-strm-probe3-cost-
parity]]) showed the 2a relevance head SATURATES at serve on SYNTHETIC
fact-summaries (r_i ~ 0.9998 for both probe and filler turns -> probe-minus-
filler gap ~ 1e-4; gate needs >= 0.2). Decomposing the shipped head's logits
on real ERAG-trace tensors ([[pondr-strm-probe3-hardneg-retrain]]) showed it
DISCRIMINATES CLEANLY on its training distribution (gold r_i 0.974 vs neg
0.034 vs gold-with-unrelated-query 0.014). So the synthetic Probe 3 facts may
simply be OUT OF DISTRIBUTION for an ERAG-trained projection. Whether the head
discriminates on the ACTUAL serve distribution (real Onyx transcripts) is
untested -- that is this probe.

WHAT. Replays the real local chat transcripts (``docs/*.json``, Onyx export
shape) through the TRAINED backbone with the WM ring ON + the trained 2a
relevance head loaded -- salience OFF (we are measuring the head, not the
trigger), no Bonsai, no GLiNER, no Onyx, no secrets -- and captures per-turn
per-slot ``r_i`` on the REAL recalled episodes the retriever injects into the
ring every turn (``orchestrator.py`` injects retrieved episodes with
``source_id`` + ``text`` at :624-647 regardless of salience). Then asks two
questions of the captured r_i:

  1. DISTRIBUTION -- does r_i vary at all on real serve data? If the head
     saturates (Probe 3 finding), r_i ~ 0.9998 for everything -> tiny std, a
     near-degenerate percentile range. If it discriminates on real data, r_i
     spreads across [0, 1] -> a healthy std + wide p10/p90 range. This needs
     NO probe/filler labeling: a near-constant r_i is the saturation signature.
  2. SELECTIVITY GAP -- per recalled episode E, across the turns E is in the
     ring, label the turn whose user-text is most bge-cosine-similar to E's
     text as the "probe" turn (the turn E is most relevant to) and the rest as
     "fillers"; gap = probe_r - mean_filler_r. Gate: min gap >= 0.2 (matches
     Probe 3's _selectivity gate). This mirrors Probe 3 but on REAL slots +
     REAL queries, with cosine (not hand-authoring) picking probe vs filler.

VERDICT. If the distribution is healthy AND the selectivity gap >= 0.2 on real
data, the synthetic Probe 3 NO-GO was a scenario artifact (OOD), and the ship
decision should move to Probe 4b (answer-quality LLM-judge). If r_i still
saturates on real data, the head is genuinely broken on the serve
distribution -> train on serve-distribution data, not ERAG.

This script uses NO secrets (the transcripts are already on disk). The
relevance-head checkpoint is a trained artifact, not a secret. Probe-only --
no src changes, no shipped artifacts.

Usage:
    python scripts/probe_strm_selectivity_real.py
    python scripts/probe_strm_selectivity_real.py --transcripts docs/a.json \\
        --ring-capacity 16 --max-turns 0 --out report.json
    # salience-ON (permissive) variant for comparison (does arming change the
    # slot population / r_i distribution?):
    python scripts/probe_strm_selectivity_real.py --salience permissive \\
        --out report_salience.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import Phase2cConfig  # noqa: E402
from src.orchestrator import PonderOrchestrator  # noqa: E402
from src.retrieval.query_planner import BonsaiQueryPlanner  # noqa: E402
from src.retrieval.retriever import HippocampalRetriever  # noqa: E402
from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.context_builder import load_context_builder  # noqa: E402
from src.subconscious.cross_slot_transformer import (  # noqa: E402
    load_cross_slot_transformer,
)
from src.subconscious.latent_dynamics_head import (  # noqa: E402
    LatentDynamicsHead,
    load_latent_dynamics_head,
)
from src.subconscious.recoverability_head import load_recoverability_head  # noqa: E402
from src.subconscious.relevance_head import load_relevance_head  # noqa: E402
from src.subconscious.salience import (  # noqa: E402
    SalienceThresholds,
    load_salience_thresholds,
)
from src.subconscious.state_readout import load_composite_z_head  # noqa: E402
from src.subconscious.training.routing_training import build_embedder, load_backbone  # noqa: E402
from src.subconscious.z_relevance_head import load_z_relevance_head  # noqa: E402

# Reuse the committed transcript-replay helpers verbatim (the same loader +
# episode builder the 2d v2 harness uses -> the ring slots we score are the
# SAME shape the live deploy produces).
from scripts.replay_chat_to_graduation import (  # noqa: E402
    _encode_best_effort,
    _iso,
    _pair_turns,
    build_episode,
    load_transcript_threads,
)

DEFAULT_BACKBONE_PATH = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
DEFAULT_RELEVANCE_HEAD = "data/training/strm_relevance/best.pt"
# The shipped Phase 3 ContextBuilder (the small Transformer that attends over
# the WM ring of y_t with the 2a r_i as an additive bias). Loaded when present
# so the probe can ALSO capture the transformer's per-slot score s_i -- the
# DeepSeek-Hole-1 test of whether restoring the transformer to the relevance-
# locator role would discriminate on real serve data where the 2a bilinear head
# saturates. Optional: if the checkpoint is absent the probe falls back to the
# r_i-only run (byte-identical to the prior behavior).
DEFAULT_CONTEXT_BUILDER = "data/training/strm_context_builder/best.pt"
# Only needed when ``--salience permissive`` (arming the trigger requires all
# three heads + thresholds, matching ``_salience_armed`` in the orchestrator).
DEFAULT_RECOVERABILITY_HEAD = "data/training/strm_recoverability/best.pt"
DEFAULT_LATENT_DYNAMICS_HEAD = "data/training/strm_latent_dynamics/best.pt"
# The Phase B ``h_t`` probe head (a ZRelevanceHead trained on slots_z). Loaded
# when present so the probe ALSO scores each ring slot with the z_i head -- the
# serve half of GATE 1: does the projected SSM state z_i = project(slot.h) carry
# query-relevance signal the 2a y_t readout did NOT? Optional: absent -> the
# r_i / s_i run is byte-identical to the prior behavior.
DEFAULT_Z_RELEVANCE_HEAD = "data/training/strm_z_relevance/best.pt"
DEFAULT_THRESHOLDS = "data/training/strm_salience/thresholds.json"
DEFAULT_TRANSCRIPTS = (
    "docs/The_Ponder_Engine_Chat.json",
    "docs/The _Ponder_Engine_Coding_Chat.json",
)


class _StubModeA:
    """No LLM round-trip. The probe measures r_i, not synthesis."""

    def _complete(self, messages, tools=None, tool_choice=None):
        return ("[probe-stub-response]", None)


def _permissive_thresholds() -> SalienceThresholds:
    """Every SCORED anchor is salient (the AND passes for any non-None scores).
    Only used when ``--salience permissive`` (so salience-fired pin-tagged
    episodes join the ring alongside the prompt-driven ones)."""
    return SalienceThresholds(
        theta=1e18, phi=-1e18, surprise_cap=1e18,
        theta_percentile=0.0, phi_percentile=100.0, surprise_cap_percentile=100.0,
        basis="permissive-upper-bound", n_recoverability=0, n_relevance=0, n_latent_dynamics=0,
    )


def _resolve_thresholds(salience: str) -> Optional[SalienceThresholds]:
    """Permissive -> the upper-bound sidecar (every scored anchor fires). Real
    -> the shipped thresholds.json. Off -> None (salience not armed)."""
    if salience == "permissive":
        return _permissive_thresholds()
    if salience == "real":
        return load_salience_thresholds(DEFAULT_THRESHOLDS)
    return None


def _cosine(a, b) -> float:
    """Cosine of two 1-d tensors (bge embeddings)."""
    a = a.to(torch.float32).reshape(-1)
    b = b.to(torch.float32).reshape(-1)
    na = a.norm().item() or 1.0
    nb = b.norm().item() or 1.0
    return float(torch.dot(a, b).item() / (na * nb))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _build_serve_trace(slots, emit_idx, out_slots, emit_embs, query_emb,
                       user_text, ld_head, emit_raw_state: bool = False) -> Optional[dict]:
    """Build ONE ``fit_relevance``-format trace record from the scored ring slots
    of a single serve turn -- for retraining the z-head on SERVE-distribution
    data (task #38: the z-head trained on ERAG candidate traces failed the SERVE
    gate ([[pondr-strm-task33-gate-train-go-serve-fail]]) because the train/serve
    z_i distribution differs; this emits the serve distribution so the retrained
    head sees the SAME persistent-state z_i + recalled-episode slots the SERVE
    gate scores).

    For each text-bearing slot ``i`` in ``emit_idx`` (the slots that made it into
    ``out_slots``): ``slots_z[i] = ld_head.project(slots[i].h).squeeze(0)`` [384]
    -- the z_i the ZRelevanceHead is trained on. ``ld_head.project`` is
    parameter-free, so an UNTRAINED ``LatentDynamicsHead`` here matches the
    generator's ``slots_z`` and the trainer's ``z_from_states`` exactly (the
    same train/serve z_i identity the ``--identity-instance`` flag pins).
    ``slots_y[i] = slots[i].y`` [256] -- ``ZRelevanceHead`` ignores ``slot_y``
    but ``fit_relevance`` requires the field. ``query_emb`` = bge(user_text).
    ``slots_doc_emb[i] = emit_embs[j]`` [384] -- the bge of the slot's text (the
    doc baseline for the across-slot std normalizer + the bge top-3 baseline).

    The LABEL is the bge cosine between the slot text and this turn's query --
    the SAME signal the SERVE gate's probe/filler split uses (the probe turn =
    max-cos turn per source). The top-1-cos slot is gold (1), the rest negative
    (0); the continuous cos is also stored under ``cos`` for an optional
    multi-positive / regression variant. Turns with <3 text-bearing slots are
    skipped (top-3 recall is degenerate when K<3). Returns None when skipped.

    ``emit_raw_state`` (task #39): also store ``slots_h_raw`` [K',4,16,384] fp16
    -- the raw per-layer SSM state per slot, the material for the Phase-0a-style
    serve-state representation probe (z_chan_last / z_flat_last / z_flat_all vs
    the mean-pool z_mean_last). fp16 keeps it small (~78KB/slot-turn)."""
    K = len(emit_idx)
    if K < 3 or ld_head is None:
        return None
    # Drop any text-bearing slot whose Phase-A state capture (slot.h) is missing
    # -- the probe's own z-scoring guards this (``getattr(slots[i], "h", None)``);
    # a None here would be a Phase A regression, so skip the slot rather than
    # crash. Keeps slots_z / slots_y / labels / cos / source_ids aligned.
    kept = [(j, i) for j, i in enumerate(range(K))  # j = out_slots idx, i = emit_idx[j]
            if getattr(slots[emit_idx[i]], "h", None) is not None]
    if len(kept) < 3:
        return None
    cos_vals = torch.tensor([out_slots[j]["cos"] for j, _ in kept],
                            dtype=torch.float32)              # [K']
    labels = torch.zeros(len(kept), dtype=torch.float32)
    labels[int(cos_vals.argmax().item())] = 1.0               # top-1-cos = gold
    slots_z = torch.stack([
        ld_head.project(slots[emit_idx[i]].h).squeeze(0).to(torch.float32)
        for _, i in kept
    ])                                                        # [K', 384]
    slots_y = torch.stack([
        slots[emit_idx[i]].y.to(torch.float32).squeeze(0).reshape(-1)
        for _, i in kept
    ])                                                        # [K', 256]
    slots_doc_emb = torch.stack([
        emit_embs[j].to(torch.float32).squeeze(0).reshape(-1)
        for j, _ in kept
    ])                                                        # [K', 384]
    rec = {
        "query_emb": query_emb.to(torch.float32).squeeze(0).reshape(-1),  # [384]
        "slots_y": slots_y, "slots_z": slots_z, "labels": labels,
        "source_ids": [out_slots[j]["source_id"] for j, _ in kept],
        "question": user_text, "cos": cos_vals,
        "slots_doc_emb": slots_doc_emb,
    }
    if emit_raw_state:
        # Per-slot per-layer state [4,16,384] fp16. slot.h is a list of 4
        # per-layer tensors; reshape(16,384) handles both [1,16,384] and
        # [16,384] storage (6144 elements either way).
        rec["slots_h_raw"] = torch.stack([
            torch.stack([
                layer.detach().to("cpu").to(torch.float16).reshape(16, 384)
                for layer in slots[emit_idx[i]].h
            ])
            for _, i in kept
        ])                                                    # [K',4,16,384] fp16
    return rec


def _capture_turn(orch: PonderOrchestrator, user_text: str,
                  context_builder=None, z_head=None, ld_head=None,
                  emit_trace: bool = False, ld_head_for_emit=None,
                  emit_raw_state: bool = False, z_head_arch="composite") -> dict:
    """Score the current WM ring against this turn's query. Returns one record:
    ``{user_text, slots: [{source_id, text, r_i, logit, s_i, s_i_pure, z_r,
    z_logit, cos}]}`` where ``cos`` is the bge cosine between the slot text and
    THIS turn's user_text (the probe/filler label signal); ``logit`` is the
    pre-sigmoid 2a relevance logit (the decisive signal under ``--ablate-yt``
    when the sigmoid is saturated); ``s_i`` / ``s_i_pure`` are the
    ContextBuilder's transformer scores (as-shipped with the 2a r_i bias, and
    pure with that bias zeroed -- the DeepSeek-Hole-1 test), present only when a
    builder is passed; and ``z_r`` / ``z_logit`` are the Phase B z_i-head's
    sigmoid + pre-sigmoid score (``z_i = project(slot.h)``, the ``h_t`` analog of
    the 2a ``y_t`` test -- the serve half of GATE 1), present only when a z_head
    is passed. Only text-bearing slots are scored."""
    prompt_emb = orch.working_memory.embed([user_text])[0]
    slots = orch.working_memory.ring_buffer()
    (slots, r_is, logits, s_is, s_is_pure, z_logits, z_rs) = _score_ring_with_logits(
        orch.working_memory, orch.relevance_head, orch.embedder, prompt_emb,
        slots, context_builder=context_builder, z_head=z_head, ld_head=ld_head,
        z_head_arch=z_head_arch)
    # Embed each scored slot's text once for the cosine probe/filler label.
    out_slots = []
    emit_idx: list[int] = []
    emit_embs: list = []  # slot text bge, aligned with out_slots (task #39 doc baseline)
    for i, (s, r, lg, si, sip, zlg, zr) in enumerate(zip(
            slots, r_is, logits, s_is, s_is_pure, z_logits, z_rs)):
        if r is None or not s.text:
            continue
        slot_emb = orch.working_memory.embed([s.text])[0]
        rec = {
            "source_id": str(s.source_id) if s.source_id is not None else None,
            "text": s.text,
            "r_i": r,
            "logit": lg,
            "cos": _cosine(prompt_emb, slot_emb),
        }
        if si is not None:
            rec["s_i"] = si
        if sip is not None:
            rec["s_i_pure"] = sip
        if zlg is not None:
            rec["z_logit"] = zlg
        if zr is not None:
            rec["z_r"] = zr
        out_slots.append(rec)
        emit_idx.append(i)
        emit_embs.append(slot_emb)
    out = {"user_text": user_text, "slots": out_slots}
    # Task #38: when emitting serve-distribution training traces, attach the
    # fit_relevance-format record built from this turn's ring. ``_analyze`` does
    # not read ``_trace``, so the probe's measurement path is unaffected; the
    # key is present ONLY when ``emit_trace`` is set (default off -> byte-
    # identical to the prior probe output). Task #39: ``emit_raw_state`` also
    # stores the per-slot per-layer SSM state (``slots_h_raw``) so the offline
    # serve-state representation probe can compute z_chan_last / z_flat_last /
    # z_flat_all -- still only when ``emit_trace`` is set.
    if emit_trace:
        out["_trace"] = _build_serve_trace(slots, emit_idx, out_slots, emit_embs,
                                           prompt_emb, user_text, ld_head_for_emit,
                                           emit_raw_state=emit_raw_state)
    return out


def _score_ring_with_logits(working_memory, relevance_head, embedder,
                            prompt_emb, slots, context_builder=None,
                            z_head=None, ld_head=None, z_head_arch="composite"):
    """Mirror ``relevance_score._score`` but return the pre-sigmoid LOGIT too
    (so the ablation can measure the bilinear gap that the saturated sigmoid
    hides), and -- when a ContextBuilder is supplied -- the transformer's per-
    slot score ``s_i`` in TWO variants:

      * ``s_i``       -- AS-SHIPPED: the 2a ``r_i`` is the additive bias
        (``lambda_r * r``), i.e. the score a rewired salience gate would
        actually consume if the shipped builder were wired into the gate.
      * ``s_i_pure``  -- the ``r`` bias ZEROED, so ``s_i_pure = (q . h)*scale +
        bias`` -- the pure cross-slot-attention score over ``y_t`` + doc-identity
        against the query. This isolates DeepSeek Hole 1: does the transformer
        attending over the WM state readouts ``y_t`` discriminate on real serve
        data, INDEPENDENT of the saturated 2a ``r_i`` bias? (The constant
        ``bias`` cancels in the probe-minus-filler selectivity gap, so the
        ``s_i_pure`` gap == the cross-slot attention gap == the test of whether
        ``y_t`` carries enough signal for the transformer to locate relevance.)

    AND -- when a ``z_head`` (ZRelevanceHead) is supplied (with ``ld_head`` for
    ``project(slot.h)``) -- the Phase B z_i-head's per-slot ``z_logit`` +
    ``z_r``: ``z_i = ld_head.project(slot.h)`` (last layer, mean over d_state,
    384-d), scored ``bilinear(proj_z(z_i), proj_q(query))``. This is the ``h_t``
    analog of the 2a ``y_t`` test -- the serve half of GATE 1: does the projected
    SSM recurrent state carry query-relevance signal the ``y_t`` readout did NOT?
    ``ld_head.project`` is parameter-free, so an untrained ``LatentDynamicsHead``
    here matches the generator's ``slots_z`` exactly (train/serve z_i identity).

    Returns ``(slots, r_is, logits, s_is, s_is_pure, z_logits, z_rs)``, each
    length-``len(slots)`` with ``None`` at unscored positions (matching
    ``score_ring_slots``); the transformer lists stay ``[None]*n`` when no
    builder is passed, and the z lists stay ``[None]*n`` when no z_head is
    passed (or a scored slot lacks ``slot.h``)."""
    n = len(slots)
    r_is: list = [None] * n
    logits: list = [None] * n
    s_is: list = [None] * n
    s_is_pure: list = [None] * n
    z_logits: list = [None] * n
    z_rs: list = [None] * n
    if relevance_head is None or embedder is None:
        return slots, r_is, logits, s_is, s_is_pure, z_logits, z_rs
    idx_text = [(i, s.text) for i, s in enumerate(slots)
                if s.text is not None and str(s.text).strip()]
    if not idx_text:
        return slots, r_is, logits, s_is, s_is_pure, z_logits, z_rs
    head_dev = next(relevance_head.parameters()).device
    doc_emb_tensors = working_memory.embed([t for _, t in idx_text])
    ys = torch.cat([slots[i].y.to(torch.float32).squeeze(0).reshape(1, -1)
                    for i, _ in idx_text], dim=0).to(head_dev)
    ds = torch.cat([e.to(torch.float32).squeeze(0).reshape(1, -1)
                    for e in doc_emb_tensors], dim=0).to(head_dev)
    q = prompt_emb.to(torch.float32).squeeze(0).reshape(1, -1).to(head_dev)
    with torch.no_grad():
        lg = relevance_head.logits(ys, ds, q)          # [K', 1]
        r = torch.sigmoid(lg)                           # [K', 1]
        if context_builder is not None:
            r_flat = r.reshape(-1)                      # [K'] -- the 2a bias
            # as-shipped transformer score (2a r_i as the additive bias).
            s = context_builder.logits(ys, ds, q, r_flat)              # [K']
            # pure transformer score: r bias zeroed -> (q . h)*scale + bias only.
            s_pure = context_builder.logits(ys, ds, q, torch.zeros_like(r_flat))
        # Phase B: z_i = project(slot.h) scored by the z_head. slot.h is present
        # for every ring slot when ring_capacity>0 (Phase A captures it); a None
        # here would mean Phase A regressed -> that slot is z-unscored (None).
        # Task #46 (acceptance): ``z_head_arch`` selects the scoring branch.
        # ``composite`` (default, byte-identical to pre-#46) = the mean-pool
        # ``ld_head.project(h)`` [384] scored by the bare ZRelevanceHead bilinear
        # (task #41's ``data/training/strm_z_relevance/best.pt`` format).
        # ``transformer`` = Head B (CrossSlotTransformerZHead): score the raw
        # LAST-layer state flattened [6144] (the SAME ``slots_h_raw`` last-layer
        # representation Head B trained on in probe_head_to_head_onyx), so the
        # live SERVE gate measures the state-trajectory Transformer the head-to-
        # head picked. ``composite-raw`` = Head A from the head-to-head
        # (CompositeZHead mlp128: StateReadout [6144->384] + ZRelevanceHead
        # bilinear) scored on the SAME raw flat_last [6144] as Head B -- the
        # apples-to-apples Head A vs Head B comparison on the live ring (Head A
        # differs from Head B ONLY in scoring: pointwise bilinear vs cross-slot
        # attention; both read the same raw state). ``ld_head`` is unused in the
        # transformer / composite-raw branches (they read the raw state, not the
        # mean-pool projection) but is kept in the signature so the composite
        # path is unchanged.
        if z_head is not None and (
                ld_head is not None or z_head_arch in ("transformer", "composite-raw")):
            z_rows = []
            z_mask: list[bool] = []
            for i, _ in idx_text:
                h = getattr(slots[i], "h", None)
                if h is None:
                    z_rows.append(None)
                    z_mask.append(False)
                elif z_head_arch in ("transformer", "composite-raw"):
                    # Last SSM layer [1,16,384] or [16,384] flattened -> [6144]
                    # (16 d_state x 384 d_model). Matches ``_load_serve_traces``
                    # last-layer flatten + Head B's ``StateReadout`` input dim.
                    z_rows.append(h[-1].detach().to(torch.float32).reshape(-1))
                    z_mask.append(True)
                else:
                    z_i = ld_head.project(h).squeeze(0).to(torch.float32)  # [384]
                    z_rows.append(z_i)
                    z_mask.append(True)
            if any(z_mask):
                z_stack = torch.stack([z for z in z_rows if z is not None]).to(head_dev)
                # z_head.logits IGNORES slot_y (``del slot_y`` -- the pure-z_i
                # test), so pass a zero dummy of the right shape rather than a
                # slice of ``ys`` (which would misalign with z_stack when some
                # slots lack h -- harmless today since slot_y is unused, but the
                # dummy makes the "ignored" intent explicit and avoids a latent
                # mis-wire if a future head ever reads slot_y).
                slot_y_dummy = torch.zeros(z_stack.shape[0], z_head.slot_dim,
                                           device=head_dev, dtype=torch.float32)
                z_lg = z_head.logits(slot_y_dummy, z_stack, q)             # [Kz, 1]
                z_r = torch.sigmoid(z_lg)                                  # [Kz, 1]
                k = 0
                for j, ok in enumerate(z_mask):
                    if not ok:
                        continue
                    zi = idx_text[j][0]
                    z_logits[zi] = float(z_lg[k].item())
                    z_rs[zi] = float(z_r[k].item())
                    k += 1
    for j, (i, _) in enumerate(idx_text):
        logits[i] = float(lg[j].item())
        r_is[i] = float(r[j].item())
        if context_builder is not None:
            s_is[i] = float(s[j].item())
            s_is_pure[i] = float(s_pure[j].item())
    return slots, r_is, logits, s_is, s_is_pure, z_logits, z_rs


def _ablate_yt_sidepath(head) -> None:
    """Zero the ``yt_sidepath`` final layer so ``logits = bilinear + bias`` (the
    WM-state readout term removed). The shipped head learned a large ~-8.5
    ``yt`` offset that centers the sigmoid; with it zeroed, r_i = sigmoid(bilinear
    + bias) still saturates high for everything (bilinear is large-positive), so
    the DEcisive ablation signal is the pre-sigmoid LOGIT gap (bias cancels ->
    logit gap == bilinear gap == the real discrimination). Zeroing weight+bias of
    the final Linear makes the sidepath output 0 for any input."""
    final = head.yt_sidepath[2]   # Sequential: [Linear(256,64), GELU, Linear(64,1)]
    final.weight.data.zero_()
    final.bias.data.zero_()
    print("[ablate] zeroed yt_sidepath final layer "
          f"(weight norm was {final.weight.norm().item():.4f}, bias {final.bias.item():.4f})",
          flush=True)


def replay_and_capture(
    *, transcripts: list[str], backbone_path: str, rel_head_path: str,
    ring_capacity: int, max_turns: int, device: str, salience: str,
    user_id: str, rec_head_path: str, ld_head_path: str, ablate_yt: bool,
    context_builder_path: Optional[str] = None,
    z_relevance_head_path: Optional[str] = None,
    identity_instance: bool = False,
    emit_traces: Optional[str] = None,
    emit_raw_state: bool = False,
    z_head_arch: str = "composite",
) -> tuple[list[dict], dict]:
    """Replay every transcript turn through the orchestrator and capture per-
    turn per-slot r_i (and, when a ContextBuilder checkpoint is supplied, the
    transformer's s_i / s_i_pure; when a z_relevance_head checkpoint is
    supplied, the z_i-head's z_r / z_logit) on the real recalled ring slots.
    Returns (turn_records, run_stats)."""
    tmpdir = tempfile.mkdtemp(prefix="pondr_probe4a_")
    turn_records: list[dict] = []
    store = None
    try:
        db_path = str(Path(tmpdir) / "db")
        from src.memory.store import HippocampalStore  # noqa: E402
        store = HippocampalStore(db_path)
        embedder = build_embedder("on-demand")
        backbone = load_backbone(str(backbone_path), BackboneConfig(), device=device)
        relevance_head = load_relevance_head(str(rel_head_path), device=device)
        if ablate_yt:
            _ablate_yt_sidepath(relevance_head)
        # The ContextBuilder is OPTIONAL -- load it when a checkpoint path is
        # supplied and present so the probe also captures the transformer's s_i
        # (the DeepSeek-Hole-1 test). Absent path -> r_i-only run (the prior
        # behavior). The builder is query-conditioned + reads y_t, so it is
        # independent of the salience arming below; load it in every salience
        # mode.
        context_builder = None
        if context_builder_path and Path(context_builder_path).exists():
            context_builder = load_context_builder(str(context_builder_path),
                                                   device=device)
            print(f"[probe] ContextBuilder loaded: {context_builder_path} "
                  f"(capturing s_i + s_i_pure alongside r_i)", flush=True)
        elif context_builder_path:
            print(f"[probe] ContextBuilder checkpoint not found at "
                  f"{context_builder_path} -- r_i-only run.", file=sys.stderr)
        # The Phase B z_i head (ZRelevanceHead) is OPTIONAL -- load it when a
        # checkpoint path is supplied and present so the probe also captures the
        # z_i-head's z_r / z_logit (the serve half of GATE 1). The project step
        # uses an UNTRAINED LatentDynamicsHead (project is parameter-free) so
        # serve z_i EXACTLY matches the generator's slots_z the head was trained
        # on. Absent path -> z capture skipped (byte-identical to prior runs).
        z_head = None
        ld_head_for_z: Optional[LatentDynamicsHead] = None
        if z_relevance_head_path and Path(z_relevance_head_path).exists():
            if z_head_arch == "transformer":
                # Task #46 acceptance: Head B (CrossSlotTransformerZHead) -- the
                # cross-slot Transformer the head-to-head (task #45) picked. Scores
                # the raw last-layer state [6144] (NOT the mean-pool projection),
                # so ld_head_for_z stays None -- the transformer branch of
                # ``_score_ring_with_logits`` builds h_raw from slot.h directly.
                z_head = load_cross_slot_transformer(str(z_relevance_head_path),
                                                     device=device)
                print(f"[probe] CrossSlotTransformerZHead (Head B) loaded: "
                      f"{z_relevance_head_path} (capturing z_r + z_logit via "
                      f"cross-slot attention on the live recalled ring)", flush=True)
            elif z_head_arch == "composite-raw":
                # Task #46 acceptance: Head A from the head-to-head
                # (CompositeZHead mlp128: readout [6144->384] + bilinear), scored
                # on the SAME raw flat_last [6144] as Head B -- the apples-to-
                # apples Head A vs Head B comparison on the live ring. Loads via
                # ``load_composite_z_head`` (NOT ``load_z_relevance_head`` -- the
                # head-to-head's bilinear checkpoint is a CompositeZHead with
                # ``readout.*`` + ``z_head.*`` keys + doc_dim=6144, not the bare
                # ZRelevanceHead task #41's ``strm_z_relevance/best.pt`` is).
                # ld_head_for_z stays None -- the composite-raw branch feeds the
                # raw state, not the mean-pool projection.
                z_head = load_composite_z_head(str(z_relevance_head_path),
                                               device=device)
                print(f"[probe] CompositeZHead (Head A, composite-raw) loaded: "
                      f"{z_relevance_head_path} (capturing z_r + z_logit via "
                      f"pointwise bilinear on the live recalled ring)", flush=True)
            else:
                z_head = load_z_relevance_head(str(z_relevance_head_path), device=device)
                ld_head_for_z = LatentDynamicsHead()  # untrained; project is parameter-free
                print(f"[probe] ZRelevanceHead loaded: {z_relevance_head_path} "
                      f"(capturing z_r + z_logit alongside r_i)", flush=True)
        elif z_relevance_head_path:
            print(f"[probe] ZRelevanceHead checkpoint not found at "
                  f"{z_relevance_head_path} -- z capture skipped.", file=sys.stderr)
        # Task #38: an untrained LatentDynamicsHead for EMITTING serve-distribution
        # training traces (project is parameter-free, so this matches the
        # generator/trainer z_i exactly). Independent of z_head -- emit uses the
        # raw z_i, z_head scoring uses z_head.logits. Created only when
        # emit_traces is set; None otherwise (the probe path is byte-identical).
        ld_head_for_emit: Optional[LatentDynamicsHead] = None
        if emit_traces:
            ld_head_for_emit = LatentDynamicsHead()
            extra = " + slots_h_raw (per-layer state)" if emit_raw_state else ""
            print(f"[probe] emit-traces ON -> {emit_traces} (capturing per-turn "
                  f"slots_z + slots_y + top-1-cos labels{extra} for z-head retrain"
                  f"{' / serve-state probe' if emit_raw_state else ''})",
                  flush=True)
        trace_records: list[dict] = []
        # Salience arming needs all three heads + thresholds (``_salience_armed``
        # in the orchestrator). OFF (the default) measures the head alone on the
        # prompt-driven recalled slots; permissive/real arm the trigger so
        # salience-fired pin-tagged episodes join the ring (the comparison
        # variant). The relevance head is loaded in ALL cases (r_i is what we
        # measure); only ``strm_salience`` + the other two heads + thresholds
        # differ.
        strm_salience = salience != "off"
        recoverability_head = None
        latent_dynamics_head = None
        if strm_salience:
            recoverability_head = load_recoverability_head(str(rec_head_path),
                                                           device=device)
            latent_dynamics_head = load_latent_dynamics_head(str(ld_head_path),
                                                              device=device)
        thresholds = _resolve_thresholds(salience)
        planner = BonsaiQueryPlanner(endpoint=None)  # None -> rule-based fallback
        retriever = HippocampalRetriever(
            store, planner=planner, auto_load_index=True,
            retrieval_gate=None, embedder=embedder,
        )
        cfg = Phase2cConfig()
        cfg.session.state_dir = str(Path(tmpdir) / "sessions")
        orch = PonderOrchestrator(
            store=store, retriever=retriever, backbone=backbone, embedder=embedder,
            mode_a=_StubModeA(), config=cfg, user_id=user_id, encoder=None,
            relevance_head=relevance_head, ring_capacity=ring_capacity,
            recoverability_head=recoverability_head,
            latent_dynamics_head=latent_dynamics_head,
            strm_salience=strm_salience, salience_thresholds=thresholds,
            identity_instance=identity_instance,
        )

        total_queries = 0
        total_encoded = 0
        total_skipped = 0
        epoch_base = 0.0
        for tpath in transcripts:
            session_id, turns = load_transcript_threads(tpath)
            pairs = _pair_turns(turns)
            if max_turns > 0:
                pairs = pairs[:max_turns]
            print(f"[replay] {tpath} session={session_id} -> {len(pairs)} user turns",
                  flush=True)
            if not pairs:
                continue
            orch.user_id = session_id
            orch.working_memory.reset()
            history: list[dict] = []
            # Seed: encode turn 0 so query 1 has memory to recall.
            u0, a0 = pairs[0]
            ep0 = build_episode(
                f"{session_id}__ep0000", u0, a0, timestamp=_iso(epoch_base, 0),
                user_id=user_id, session_id=session_id, embedder=embedder)
            if _encode_best_effort(store, ep0, session_id, 0):
                total_encoded += 1
            else:
                total_skipped += 1
            history.append({"role": "user", "content": u0})
            history.append({"role": "assistant", "content": a0})
            for i in range(1, len(pairs)):
                u, a = pairs[i]
                try:
                    orch.query(u, conversation_history=list(history),
                                auto_persist=False, signal="routine")
                except Exception as e:  # noqa: BLE001 - one bad turn must not kill the run
                    print(f"  [query-fail] session={session_id} turn={i}: {e}",
                          file=sys.stderr)
                # Score the ring NOW (after the query step + recalled-episode
                # injects populate text-bearing slots). prompt_emb is re-derived
                # from the user text (deterministic -- same embed call query()
                # uses internally).
                rec = _capture_turn(orch, u, context_builder=context_builder,
                                    z_head=z_head, ld_head=ld_head_for_z,
                                    emit_trace=emit_traces is not None,
                                    ld_head_for_emit=ld_head_for_emit,
                                    emit_raw_state=emit_raw_state,
                                    z_head_arch=z_head_arch)
                if emit_traces:
                    # Pop the trace BEFORE the record enters turn_records so
                    # _analyze (the measurement path) never sees it.
                    trace_rec = rec.pop("_trace", None)
                    if trace_rec is not None:
                        trace_records.append(trace_rec)
                rec["session_id"] = session_id
                rec["turn_index"] = i
                turn_records.append(rec)
                ep = build_episode(
                    f"{session_id}__ep{i:04d}", u, a,
                    timestamp=_iso(epoch_base, i), user_id=user_id,
                    session_id=session_id, embedder=embedder)
                if _encode_best_effort(store, ep, session_id, i):
                    total_encoded += 1
                else:
                    total_skipped += 1
                history.append({"role": "user", "content": u})
                history.append({"role": "assistant", "content": a})
                total_queries += 1
                if (i + 1) % 20 == 0:
                    print(f"  replayed {i + 1}/{len(pairs)} turns "
                          f"(encoded={total_encoded} skipped={total_skipped})",
                          flush=True)
            epoch_base += 1e6
        if emit_traces:
            emit_path = Path(emit_traces)
            emit_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(trace_records, emit_path)
            print(f"[probe] wrote {len(trace_records)} serve-distribution trace "
                  f"records -> {emit_path}", flush=True)
        run_stats = {
            "n_turns": total_queries, "n_encoded": total_encoded,
            "n_skipped": total_skipped, "ring_capacity": ring_capacity,
            "salience": salience, "device": device,
            "context_builder": context_builder is not None,
            "z_relevance_head": z_head is not None,
            "z_head_arch": z_head_arch,
            "emit_traces": emit_traces,
            "n_trace_records": len(trace_records) if emit_traces else None,
            "emit_raw_state": emit_raw_state and emit_traces is not None,
        }
        return turn_records, run_stats
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)


def _analyze(turn_records: list[dict]) -> dict:
    """Compute (1) the r_i + logit + s_i + s_i_pure distributions across all
    scored (slot, turn) pairs and (2) the per-source selectivity gap (probe minus
    mean filler) for each. The logit gap is the decisive signal under
    ``--ablate-yt`` (the shipped head's ~-8.5 yt offset centers the sigmoid, so
    r_i saturates even when the bilinear term discriminates; the logit gap, bias
    cancels, == the bilinear gap). The ``s_i_pure`` gap is the DeepSeek-Hole-1
    test: it is the cross-slot-attention score over y_t with the saturated r_i
    bias removed, so its gap == whether the transformer can locate relevance from
    the WM state readouts alone. ``s_i`` (as-shipped, with the r_i bias) tests
    what a rewired salience gate would actually consume."""
    # (1) distributions across every scored slot occurrence.
    all_r = [s["r_i"] for rec in turn_records for s in rec["slots"]]
    all_lg = [s["logit"] for rec in turn_records for s in rec["slots"]
              if s.get("logit") is not None]
    all_s = [s["s_i"] for rec in turn_records for s in rec["slots"]
             if s.get("s_i") is not None]
    all_sp = [s["s_i_pure"] for rec in turn_records for s in rec["slots"]
              if s.get("s_i_pure") is not None]
    all_zr = [s["z_r"] for rec in turn_records for s in rec["slots"]
              if s.get("z_r") is not None]
    all_zlg = [s["z_logit"] for rec in turn_records for s in rec["slots"]
               if s.get("z_logit") is not None]

    def _dist(vals: list[float]) -> dict:
        return {
            "n": len(vals),
            "min": min(vals) if vals else None,
            "p10": _percentile(vals, 0.10),
            "p50": _percentile(vals, 0.50),
            "p90": _percentile(vals, 0.90),
            "max": max(vals) if vals else None,
            "mean": statistics.fmean(vals) if vals else None,
            "stdev": statistics.pstdev(vals) if len(vals) >= 2 else 0.0,
        }

    dist = {
        "n_scored": len(all_r),
        "min": min(all_r) if all_r else None,
        "p10": _percentile(all_r, 0.10),
        "p50": _percentile(all_r, 0.50),
        "p90": _percentile(all_r, 0.90),
        "max": max(all_r) if all_r else None,
        "mean": statistics.fmean(all_r) if all_r else None,
        "stdev": statistics.pstdev(all_r) if len(all_r) >= 2 else 0.0,
        "frac_ge_0p99": (sum(1 for r in all_r if r >= 0.99) / len(all_r)) if all_r else 0.0,
        "logit_min": min(all_lg) if all_lg else None,
        "logit_p10": _percentile(all_lg, 0.10),
        "logit_p50": _percentile(all_lg, 0.50),
        "logit_p90": _percentile(all_lg, 0.90),
        "logit_max": max(all_lg) if all_lg else None,
        "logit_stdev": statistics.pstdev(all_lg) if len(all_lg) >= 2 else 0.0,
        "s_i": _dist(all_s),
        "s_i_pure": _dist(all_sp),
        # Phase B: the z_i-head's sigmoid (z_r) + pre-sigmoid (z_logit). z_r is
        # the GATE 1 metric (serve selectivity gap median >= 0.2, the 2a r_i
        # scale); z_logit (unbounded) uses the >= 2.0 logit gate like the 2a
        # logit / s_i. frac_ge_0p99 on z_r mirrors the 2a saturation signature.
        "z_r": {**_dist(all_zr),
                "frac_ge_0p99": (sum(1 for r in all_zr if r >= 0.99) / len(all_zr))
                                if all_zr else 0.0},
        "z_logit": _dist(all_zlg),
    }
    # (2) per-source selectivity for r_i, logit, s_i, s_i_pure. Group scored
    # occurrences by source_id; for each source seen on >= 3 turns, the probe
    # turn = max-cos turn, fillers = the remaining turns; gap = probe -
    # mean(filler). Report min + median gap (the r_i gate is min gap >= 0.2,
    # matching Probe 3). s_i / s_i_pure are unbounded logits like the 2a logit,
    # so they use the same >= 2.0 logit gap gate.
    by_source: dict[str, list[dict]] = {}
    for rec in turn_records:
        for s in rec["slots"]:
            sid = s["source_id"]
            if sid is None:
                continue
            by_source.setdefault(sid, []).append({
                "turn_index": rec["turn_index"], "r_i": s["r_i"],
                "logit": s.get("logit"), "cos": s["cos"],
                "s_i": s.get("s_i"), "s_i_pure": s.get("s_i_pure"),
                "z_r": s.get("z_r"), "z_logit": s.get("z_logit"),
            })

    r_gaps: list[float] = []
    lg_gaps: list[float] = []
    s_gaps: list[float] = []
    sp_gaps: list[float] = []
    zr_gaps: list[float] = []
    zlg_gaps: list[float] = []
    per_source_examples: list[dict] = []
    for sid, occs in by_source.items():
        if len(occs) < 3:
            continue
        occs_sorted = sorted(occs, key=lambda o: o["cos"], reverse=True)
        probe = occs_sorted[0]
        fillers = occs_sorted[1:]
        mean_filler_r = statistics.fmean(o["r_i"] for o in fillers)
        r_gap = probe["r_i"] - mean_filler_r
        r_gaps.append(r_gap)
        lg_gap = None
        if probe["logit"] is not None and all(o["logit"] is not None for o in fillers):
            lg_gap = probe["logit"] - statistics.fmean(o["logit"] for o in fillers)
            lg_gaps.append(lg_gap)
        s_gap = None
        if probe["s_i"] is not None and all(o["s_i"] is not None for o in fillers):
            s_gap = probe["s_i"] - statistics.fmean(o["s_i"] for o in fillers)
            s_gaps.append(s_gap)
        sp_gap = None
        if probe["s_i_pure"] is not None and all(o["s_i_pure"] is not None for o in fillers):
            sp_gap = probe["s_i_pure"] - statistics.fmean(o["s_i_pure"] for o in fillers)
            sp_gaps.append(sp_gap)
        # Phase B GATE 1: the z_i-head's selectivity gap (probe z minus mean
        # filler z). z_r uses the 0.2 sigmoid gap gate (the 2a r_i scale, the
        # GATE 1 metric); z_logit uses the 2.0 unbounded-logit gate.
        zr_gap = None
        if probe["z_r"] is not None and all(o["z_r"] is not None for o in fillers):
            zr_gap = probe["z_r"] - statistics.fmean(o["z_r"] for o in fillers)
            zr_gaps.append(zr_gap)
        zlg_gap = None
        if probe["z_logit"] is not None and all(o["z_logit"] is not None for o in fillers):
            zlg_gap = probe["z_logit"] - statistics.fmean(o["z_logit"] for o in fillers)
            zlg_gaps.append(zlg_gap)
        if len(per_source_examples) < 12:
            per_source_examples.append({
                "source_id": sid, "n_turns": len(occs),
                "probe_cos": probe["cos"], "probe_r_i": probe["r_i"],
                "mean_filler_r_i": mean_filler_r, "r_gap": r_gap,
                "probe_logit": probe["logit"], "logit_gap": lg_gap,
                "probe_s_i": probe["s_i"], "s_gap": s_gap,
                "probe_s_i_pure": probe["s_i_pure"], "s_pure_gap": sp_gap,
                "probe_z_r": probe["z_r"], "z_r_gap": zr_gap,
                "probe_z_logit": probe["z_logit"], "z_logit_gap": zlg_gap,
            })

    def _gap_stats(gaps: list[float], thr: float):
        return {
            "min": min(gaps) if gaps else None,
            "median": statistics.median(gaps) if gaps else None,
            "mean": statistics.fmean(gaps) if gaps else None,
            "n_ge_thr": sum(1 for g in gaps if g >= thr),
            "n_eligible": len(gaps),
            "gate_median_ge_thr": (statistics.median(gaps) >= thr) if gaps else False,
        }

    selectivity = {
        "n_sources_total": len(by_source),
        "n_sources_eligible": len(r_gaps),
        "min_r_gap": min(r_gaps) if r_gaps else None,
        "median_r_gap": statistics.median(r_gaps) if r_gaps else None,
        "mean_r_gap": statistics.fmean(r_gaps) if r_gaps else None,
        "n_r_gap_ge_0p2": sum(1 for g in r_gaps if g >= 0.2),
        "gate_min_r_gap_ge_0p2": (min(r_gaps) >= 0.2) if r_gaps else False,
        "min_logit_gap": min(lg_gaps) if lg_gaps else None,
        "median_logit_gap": statistics.median(lg_gaps) if lg_gaps else None,
        "mean_logit_gap": statistics.fmean(lg_gaps) if lg_gaps else None,
        "n_logit_gap_ge_2": sum(1 for g in lg_gaps if g >= 2.0),
        "gate_median_logit_gap_ge_2": (statistics.median(lg_gaps) >= 2.0) if lg_gaps else False,
        "s_i": _gap_stats(s_gaps, 2.0),
        "s_i_pure": _gap_stats(sp_gaps, 2.0),
        # Phase B GATE 1: z_r gap (sigmoid scale, thr 0.2 -- the primary gate)
        # + z_logit gap (unbounded, thr 2.0 -- the ablation-scale view).
        "z_r": _gap_stats(zr_gaps, 0.2),
        "z_logit": _gap_stats(zlg_gaps, 2.0),
        "examples": per_source_examples,
    }
    return {"distribution": dist, "selectivity": selectivity}


def _main() -> int:
    p = argparse.ArgumentParser(
        description="STRM Probe 4a: 2a relevance-head selectivity on REAL serve data (no LLM)")
    p.add_argument("--transcripts", nargs="+", default=list(DEFAULT_TRANSCRIPTS))
    p.add_argument("--backbone", default=DEFAULT_BACKBONE_PATH)
    p.add_argument("--relevance-head", default=DEFAULT_RELEVANCE_HEAD)
    p.add_argument("--recoverability-head", default=DEFAULT_RECOVERABILITY_HEAD,
                    help="2b head (only needed when --salience != off)")
    p.add_argument("--latent-dynamics-head", default=DEFAULT_LATENT_DYNAMICS_HEAD,
                    help="2c head (only needed when --salience != off)")
    p.add_argument("--ring-capacity", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=0, help="cap user turns per session (0=all)")
    p.add_argument("--device", default="auto", help="backbone+head device: auto|cpu|cuda")
    p.add_argument("--salience", choices=("off", "permissive", "real"), default="off",
                   help="off=measure the head alone on prompt-driven slots; "
                        "permissive=arm with upper-bound thresholds; "
                        "real=arm with the shipped thresholds.json")
    p.add_argument("--ablate-yt", action="store_true",
                   help="zero the yt_sidepath so logits = bilinear + bias; the "
                        "LOGIT gap (reported alongside r_i) is the decisive signal")
    p.add_argument("--context-builder", default=DEFAULT_CONTEXT_BUILDER,
                   help="ContextBuilder checkpoint to ALSO capture the "
                        "transformer's s_i / s_i_pure (the DeepSeek-Hole-1 test). "
                        "Default = the shipped Phase 3 builder; pass '' to skip.")
    p.add_argument("--z-relevance-head", default=DEFAULT_Z_RELEVANCE_HEAD,
                   help="ZRelevanceHead checkpoint (Phase B h_t probe) to ALSO "
                        "capture the z_i-head's z_r / z_logit (the serve half of "
                        "GATE 1: does z_i = project(slot.h) carry query-relevance "
                        "signal y_t did NOT). Default = the trained z-head; pass "
                        "'' to skip.")
    p.add_argument("--z-head-arch",
                   choices=("composite", "transformer", "composite-raw"),
                   default="composite",
                   help="Task #46 acceptance: which arch the --z-relevance-head "
                        "checkpoint is + how the live z-scoring path scores it. "
                        "composite (default) = the mean-pool ZRelevanceHead bilinear "
                        "(byte-identical to pre-#46; task #41's bare-ZRelevanceHead "
                        "checkpoint format). transformer = Head B "
                        "(CrossSlotTransformerZHead, task #45 winner): scores the "
                        "raw last-layer state [6144] via cross-slot attention -- the "
                        "live SERVE gate the head-to-head pointed at. composite-raw = "
                        "Head A from the head-to-head (CompositeZHead mlp128: readout "
                        "[6144->384] + bilinear) scored on the SAME raw flat_last "
                        "[6144] as Head B -- the apples-to-apples Head A vs Head B "
                        "comparison on the live ring. Use transformer / composite-raw "
                        "with --backbone backbone_v2_full.pt --identity-instance so "
                        "the state space matches the head-to-head training.")
    p.add_argument("--user-id", default="pondr")
    p.add_argument("--out", default="", help="write the JSON report to this path")
    p.add_argument("--identity-instance", action="store_true",
                   help="Phase 1 gate re-run (task #33): drive the orchestrator's "
                        "WorkingMemory with identity input_proj + zero state_lora -- "
                        "the direct-SSM path the from-scratch relevance trainer "
                        "optimizes. Use when --backbone points at a backbone trained "
                        "under that path (the new from-scratch ckpt) so the SERVE gate "
                        "measures the SAME path the z-head was trained on. Default off "
                        "-> random instance projections (byte-identical to pre-task-#33).")
    p.add_argument("--emit-traces", default="",
                   help="Task #38: write a fit_relevance-format trace file "
                        "(per-turn slots_z + slots_y + top-1-cos labels) for "
                        "retraining the z-head on SERVE-distribution data. The "
                        "trace generator IS this probe's exact replay path "
                        "(persistent WM + recalled-episode ring), so train/serve "
                        "distributions match by construction. Use with "
                        "--identity-instance + --backbone <new from-scratch ckpt>. "
                        "Default '' -> off, byte-identical to the prior probe.")
    p.add_argument("--emit-raw-state", action="store_true",
                   help="Task #39: with --emit-traces, ALSO store the per-slot "
                       "per-layer SSM state (slots_h_raw [K,4,16,384] fp16) + the "
                       "slot text bge (slots_doc_emb) so the offline serve-state "
                       "representation probe can compute z_mean_last / z_chan_last / "
                       "z_flat_last / z_flat_all and compare across-slot variance + "
                       "bge top-3 per representation (the fork diagnostic: mean-pool-"
                       "kills-signal -> state-trajectory Transformer, vs all-reps-flat "
                       "-> backbone-on-serve retrain). No effect without --emit-traces. "
                       "Default off -> byte-identical to --emit-traces alone.")
    args = p.parse_args()

    if not Path(args.backbone).exists():
        print(f"ERROR: backbone not found at {args.backbone}", file=sys.stderr)
        return 1
    if not Path(args.relevance_head).exists():
        print(f"ERROR: relevance-head checkpoint not found at {args.relevance_head}",
              file=sys.stderr)
        return 1
    if args.salience != "off":
        for label, hp in (("recoverability-head", args.recoverability_head),
                          ("latent-dynamics-head", args.latent_dynamics_head)):
            if not Path(hp).exists():
                print(f"ERROR: {label} not found at {hp} (required when --salience != off)",
                      file=sys.stderr)
                return 1
        if args.salience == "real" and not Path(DEFAULT_THRESHOLDS).exists():
            print(f"ERROR: thresholds not found at {DEFAULT_THRESHOLDS} "
                  f"(required for --salience real)", file=sys.stderr)
            return 1
    for t in args.transcripts:
        if not Path(t).exists():
            print(f"ERROR: transcript not found at {t}", file=sys.stderr)
            return 1

    turn_records, run_stats = replay_and_capture(
        transcripts=args.transcripts, backbone_path=args.backbone,
        rel_head_path=args.relevance_head, ring_capacity=args.ring_capacity,
        max_turns=args.max_turns, device=args.device, salience=args.salience,
        user_id=args.user_id, rec_head_path=args.recoverability_head,
        ld_head_path=args.latent_dynamics_head, ablate_yt=args.ablate_yt,
        context_builder_path=args.context_builder or None,
        z_relevance_head_path=args.z_relevance_head or None,
        identity_instance=args.identity_instance,
        emit_traces=args.emit_traces or None,
        emit_raw_state=args.emit_raw_state,
        z_head_arch=args.z_head_arch)
    analysis = _analyze(turn_records)
    report = {"run": run_stats, **analysis, "n_turn_records": len(turn_records)}

    print("=" * 72)
    print(f"STRM Probe 4a -- 2a relevance-head selectivity on REAL serve data"
          f"{' [ABLATE yt_sidepath]' if args.ablate_yt else ''}")
    print(f"  transcripts={args.transcripts} ring={run_stats['ring_capacity']} "
          f"salience={run_stats['salience']} turns={run_stats['n_turns']} "
          f"(encoded={run_stats['n_encoded']} skipped={run_stats['n_skipped']})")
    print("-" * 72)
    d = report["distribution"]
    print(f"  r_i DISTRIBUTION on real recalled slots ({d['n_scored']} scored):")
    print(f"    min={d['min']:.4f}  p10={d['p10']:.4f}  p50={d['p50']:.4f}  "
          f"p90={d['p90']:.4f}  max={d['max']:.4f}")
    print(f"    mean={d['mean']:.4f}  stdev={d['stdev']:.4f}  frac>=0.99={d['frac_ge_0p99']:.2%}")
    print(f"  LOGIT distribution (pre-sigmoid; the ablation signal):")
    print(f"    min={d['logit_min']:.3f}  p10={d['logit_p10']:.3f}  "
          f"p50={d['logit_p50']:.3f}  p90={d['logit_p90']:.3f}  max={d['logit_max']:.3f}  "
          f"stdev={d['logit_stdev']:.3f}")
    print(f"    (Probe 3 synthetic saturation: r_i ~ 0.9998, stdev ~ 0 -> near-constant)")
    s = report["selectivity"]
    print(f"  SELECTIVITY (probe turn = max bge-cosine turn vs this slot; "
          f"{s['n_sources_eligible']} eligible of {s['n_sources_total']}):")
    if s["min_r_gap"] is not None:
        print(f"    r_i  gap: min={s['min_r_gap']:+.4f} median={s['median_r_gap']:+.4f} "
              f"mean={s['mean_r_gap']:+.4f}  n>=0.2={s['n_r_gap_ge_0p2']}/"
              f"{s['n_sources_eligible']}  "
              f"gate(min>=0.2): {'PASS' if s['gate_min_r_gap_ge_0p2'] else 'FAIL'}")
    else:
        print("    r_i  gap: (no source seen on >=3 turns -- run longer / raise --ring-capacity)")
    if s["min_logit_gap"] is not None:
        print(f"    logit gap: min={s['min_logit_gap']:+.3f} median={s['median_logit_gap']:+.3f} "
              f"mean={s['mean_logit_gap']:+.3f}  n>=2.0={s['n_logit_gap_ge_2']}/"
              f"{s['n_sources_eligible']}  "
              f"gate(median>=2.0): {'PASS' if s['gate_median_logit_gap_ge_2'] else 'FAIL'}")

    def _print_transformer(label: str, dist_key: str, sel_key: str) -> None:
        dd = d.get(dist_key)
        ss = s.get(sel_key)
        if not dd or dd["n"] == 0 or not ss or ss["n_eligible"] == 0:
            print(f"    {label}: (not captured -- no ContextBuilder loaded)")
            return
        print(f"  {label} distribution ({dd['n']} scored):")
        print(f"    min={dd['min']:+.3f}  p10={dd['p10']:+.3f}  p50={dd['p50']:+.3f}  "
              f"p90={dd['p90']:+.3f}  max={dd['max']:+.3f}  stdev={dd['stdev']:.3f}")
        print(f"    {label} gap: min={ss['min']:+.3f} median={ss['median']:+.3f} "
              f"mean={ss['mean']:+.3f}  n>=2.0={ss['n_ge_thr']}/{ss['n_eligible']}  "
              f"gate(median>=2.0): {'PASS' if ss['gate_median_ge_thr'] else 'FAIL'}")

    _print_transformer("s_i      (as-shipped, 2a r_i bias)", "s_i", "s_i")
    _print_transformer("s_i_pure (r bias zeroed -- Hole 1)", "s_i_pure", "s_i_pure")

    # Phase B GATE 1: the z_i-head's selectivity. z_r is the primary gate metric
    # (sigmoid scale, median gap >= 0.2 in >= 3/4 runs vs 2a's ~0); z_logit is
    # the unbounded ablation-scale view (median >= 2.0).
    zd = d.get("z_r")
    zs = s.get("z_r")
    zld = d.get("z_logit")
    zls = s.get("z_logit")
    if zd and zs and zs["n_eligible"] > 0:
        print(f"  z_r   (Phase B h_t probe, GATE 1) distribution ({zd['n']} scored):")
        print(f"    min={zd['min']:.4f}  p10={zd['p10']:.4f}  p50={zd['p50']:.4f}  "
              f"p90={zd['p90']:.4f}  max={zd['max']:.4f}  "
              f"stdev={zd['stdev']:.4f}  frac>=0.99={zd['frac_ge_0p99']:.2%}")
        print(f"    z_r   gap: min={zs['min']:+.4f} median={zs['median']:+.4f} "
              f"mean={zs['mean']:+.4f}  n>=0.2={zs['n_ge_thr']}/{zs['n_eligible']}  "
              f"gate(median>=0.2): {'PASS' if zs['gate_median_ge_thr'] else 'FAIL'}")
        if zld and zls and zls["n_eligible"] > 0:
            print(f"  z_logit (pre-sigmoid) distribution ({zld['n']} scored):")
            print(f"    min={zld['min']:+.3f}  p10={zld['p10']:+.3f}  "
                  f"p50={zld['p50']:+.3f}  p90={zld['p90']:+.3f}  max={zld['max']:+.3f}  "
                  f"stdev={zld['stdev']:.3f}")
            print(f"    z_logit gap: min={zls['min']:+.3f} median={zls['median']:+.3f} "
                  f"mean={zls['mean']:+.3f}  n>=2.0={zls['n_ge_thr']}/{zls['n_eligible']}  "
                  f"gate(median>=2.0): {'PASS' if zls['gate_median_ge_thr'] else 'FAIL'}")
    else:
        print("  z_r (Phase B h_t probe): (not captured -- no ZRelevanceHead loaded "
              "or no source seen on >=3 turns)")
    print("-" * 72)
    print("  per-source examples:")
    for ex in s["examples"]:
        lg = ex.get("logit_gap")
        lg_s = f"{lg:+.3f}" if lg is not None else "  n/a"
        sp = ex.get("s_pure_gap")
        sp_s = f"{sp:+.3f}" if sp is not None else "  n/a"
        zr = ex.get("z_r_gap")
        zr_s = f"{zr:+.4f}" if zr is not None else "  n/a"
        print(f"    {ex['source_id'][:24]:<24} n={ex['n_turns']:>2} "
              f"cos={ex['probe_cos']:.3f} r_gap={ex['r_gap']:+.4f} "
              f"lg_gap={lg_s} s_pure_gap={sp_s} z_r_gap={zr_s}")
    print("=" * 72)
    print("VERDICT (ablate-yt): a healthy LOGIT gap (median >= 2.0) => the bilinear")
    print("  term discriminates on real serve data and yt_sidepath was the culprit;")
    print("  retrain with yt_sidepath zeroed/regularized. A ~0 logit gap => the bilinear")
    print("  proj itself collapses on serve -> train on serve-distribution data.")
    print("VERDICT (no ablate): healthy r_i dist (stdev>0) + min r_gap>=0.2 => head")
    print("  discriminates on real serve data => synthetic Probe 3 NO-GO was OOD.")
    print("  saturated r_i (stdev~0, frac>=0.99~100%) => head broken on serve.")
    print("VERDICT (s_i_pure -- DeepSeek Hole 1): a healthy s_i_pure gap (median >= 2.0)")
    print("  => the Transformer attending over y_t discriminates on real serve data")
    print("  INDEPENDENT of the saturated r_i bias -> restoring it to the relevance-")
    print("  locator role is viable (then address labels/cost/two-pass in the retrain).")
    print("  A ~0 s_i_pure gap => y_t carries insufficient signal for the transformer")
    print("  to locate relevance -> must store actual SSM states, not just y_t readouts.")
    print("VERDICT (z_r -- Phase B GATE 1, the h_t test): a healthy z_r gap "
          "(median >= 0.2)")
    print("  => the projected SSM state z_i carries serve-relevance signal the y_t")
    print("  readout did NOT -> the state-trajectory-transformer premise SURVIVES;")
    print("  proceed to the transformer rewire (Phases C-F). A ~0 z_r gap (like 2a's")
    print("  r_i ~0) => h_t does NOT carry the signal either -> STOP, do not build the")
    print("  transformer; the vision's state-locator is dead on h_t too (likely a")
    print("  backbone problem -- routing-trained, not relevance-trained -- not a")
    print("  readout problem). Run >= 4 times; GATE 1 = median >= 0.2 in >= 3/4 runs.")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())