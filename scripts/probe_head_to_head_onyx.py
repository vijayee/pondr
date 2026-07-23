"""Task #45: the DeepSeek-v4-pro head-to-head on REAL Onyx serve traces.

Task #44 ([[pondr-strm-task44-contrastive-partial-desat-transfer-fail]]) showed
the contrastive InfoNCE loss PARTIALLY de-saturates the per-slot z_i bilinear
(lmsys held-out z_logit 0.931, ~3x BCE) but the MEDIAN stays sub-gate (arch
margin bounded ~1.0) AND lmsys->Onyx transfer is WORSE under contrastive
(0.048 vs BCE 0.258 -- bias-invariance removes the bias as a distribution-shift
absorber). The flat-readout z_i bilinear is NOT the ship lever even with the
loss fix. DeepSeek-v4-pro (consulted 2026-07-21) diagnosed the root cause
mechanistically: a POINTWISE bilinear scores each slot INDEPENDENTLY against the
query, so it can only produce an ABSOLUTE relevance (sim-to-query). The 2.0
z_logit gate is a RELATIVE margin (gold logit - mean filler logit), and on
serve the fillers are topically close -> their absolute sims are ALL high ->
the bilinear's absolute score cannot push a 2.0 relative gap. A CROSS-SLOT
attention head can implement relative scoring (each slot's logit attends to
the query AND to all other slots -> score ~ sim-to-query - mean sim of all
slots, which DeepSeek said is the mechanism by which it escapes the margin
bound the pointwise bilinear cannot). This probe is the decisive A/B test.

**Head A** = the current ``CompositeZHead`` (``StateReadout`` mlp128 [6144->384]
+ ``ZRelevanceHead`` bilinear ``proj_z(z_i).proj_q(q)/sqrt(P) + bias``), the
task #44 arch. **Head B** = a minimal cross-slot Transformer: the SAME
``StateReadout`` mlp128 -> per-slot z_i [K,384] (so the ONLY difference from
Head A is the cross-slot attention vs the pointwise bilinear -- a win/loss is
cleanly attributable to the cross-slot mechanism, not the readout), + a learned
positional embedding, + the query as a [CLS] token prepended, + a 2-layer /
4-head / hidden-256 / FFN-512 Transformer encoder, + a per-slot logit head on
each slot's encoder output. SAME contrastive InfoNCE loss (T=1.0), SAME frozen
``backbone_v2_full.pt`` SSM (the traces already carry its ``slots_h_raw``),
SAME eval (``p41._zr_and_logit_gaps`` -> per-source z_logit gap, 2.0 gate).

**Data.** ``traces_onyx_serve_hraw.pt`` (task #45, 1012 turns from 76 REAL
Onyx sessions fetched via cookie auth -- in-distribution, no lmsys transfer
confound). Split by SESSION (not by query): held-out = ENTIRE unseen
conversations, the true "generalizes to new Onyx" test DeepSeek specified.
3-seed robustness.

**Decision rule (DeepSeek):**
  * Head A clears the 2.0 z_logit gate on held-out Onyx (robust, >= 2/3 seeds)
    -> SHIP the bilinear; the loss was the only blocker.
  * Head A fails BUT Head B clears -> the cross-slot Transformer IS the lever;
    invest there next (scale + wire into the live serve probe).
  * NEITHER clears -> ABANDON the state-trajectory-locator (option C): the
    state path has been tested 5 ways and saturates; the bge 2a head (0.889
    train) already works.

**Isolation (binding constraint).** Standalone diagnostic; Head B
(``CrossSlotTransformerZHead``) lives in ``src/subconscious/cross_slot_transformer.py``
(promoted there for the task #46 live-SERVE-gate acceptance test so
``probe_strm_selectivity_real.py --z-head-arch transformer`` can load + score
with it). The class is byte-identical to the version this probe trained under,
so the existing ``best.pt`` checkpoints load unchanged. It is imported ONLY
by this probe + the live probe -- ``build_ponder`` / ``serve_ponder`` /
``DEFAULT_BACKBONE_PATH`` are never touched, so every existing head + the 2b
gate stay byte-identical by construction. Reuses ``contrastive_loss`` +
``_to_device`` from ``probe_contrastive_zlogit`` (task #44, committed) and
``p41._zr_and_logit_gaps`` + ``p41._load_serve_traces`` (task #41). Does NOT
call ``fit_relevance``. No live wiring, no HF upload (diagnostic; the Onyx
traces are PRIVATE chat data, local + gitignored, never uploaded per user
directive).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))        # sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import probe_serve_composite_zrgate as p41  # noqa: E402
import probe_contrastive_zlogit as p44  # noqa: E402
import probe_strm_selectivity_real as p_sel  # noqa: E402  # Stage 0: doc_kind_map helpers
from src.subconscious.cross_slot_transformer import (  # noqa: E402
    CrossSlotTransformerZHead,
)
from src.subconscious.state_readout import CompositeZHead  # noqa: E402
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    _gate_score,
    _split_queries,
    evaluate_relevance,
)

DEFAULT_ONYX = "data/training/strm_relevance/traces_onyx_serve_hraw.pt"
DEFAULT_OUT = "data/training/strm_relevance/head_to_head_onyx.json"
DEFAULT_CKPT_ROOT = "data/training/strm_state_readout/head_to_head_onyx"

ZLOGIT_GATE = 2.0

# D0.4a (task #47): the 2 live-transcript sessions (docs/*.json replayed by
# probe_strm_selectivity_real.py) are IN the 53-session Onyx training traces, so
# the prior head-to-head's held-out number was partly in-sample on them. Force
# them OUT of train and into a dedicated live-eval bucket for EVERY seed so the
# conversation-ring z_logit gap on them is genuinely held-out -- the decisive
# H3 (overfit) vs H2 (content-shift) diagnostic (see the approved plan). The
# live probe replays these 2 transcripts; their chat_session_ids (full Onyx
# UUIDs, opaque -- no PII) are confirmed present in traces_onyx_serve_hraw.pt.
# ``--live-eval-sessions ""`` disables the hold-out -> byte-identical to pre-#47.
DEFAULT_LIVE_EVAL_SESSION_IDS = (
    "682afdd9-e8ea-4258-a329-65f67b5d27d5",  # docs/The _Ponder_Engine_Coding_Chat.json
    "69e17901-9c6c-4375-a6f1-736e95e1d316",  # docs/The_Ponder_Engine_Chat.json
)


# Head B (CrossSlotTransformerZHead) is imported from src/subconscious/ -- the
# acceptance-test promotion (task #45 follow-up): the probe's local class was
# lifted verbatim into ``src/subconscious/cross_slot_transformer.py`` so the
# live SERVE gate (``probe_strm_selectivity_real.py --z-head-arch transformer``)
# can load + score with it. Same submodule names -> the probe's existing
# best.pt checkpoints load via ``load_cross_slot_transformer`` unchanged.


def _build_head(arch: str, dim_in: int, hidden: int | None,
                n_slot_types: int = 0, learnable_temp: bool = False,
                dropout: float = 0.0, n_doc_kinds: int = 0,
                per_kind_full: bool = False,
                per_kind_bodies: bool = False) -> nn.Module:
    """Construct Head A (bilinear composite) or Head B (cross-slot Transformer).

    The Phase-1b/1d knobs (``n_slot_types`` slot-type embedding, ``learnable_temp``
    logit temperature, ``dropout`` readout regularization) are Head B ONLY -- Head
    A (``CompositeZHead``) ignores them. Defaults 0/False/0.0 = the task #45 arch
    (byte-identical; the existing best.pt strict-loads via ``_load_head``).

    Phase 1f-7 Stage 1 ``n_doc_kinds`` (Head A ONLY): the per-doc-kind readout.
    0 = byte-identical shared readout. Head B (transformer) ignores it (its
    ``n_slot_types`` conv/retrieved channel is ORTHOGONAL to doc-kind; do NOT
    repurpose it).

    Phase 1f-7 Stage 1 REDESIGN ``per_kind_full`` (Head A ONLY, MoE on
    non-overlapping data): N independent full readouts (no shared body) instead of
    the shared body + per-kind heads. The trainer routes ALL slots of a record
    through the GOLD's readout (by-gold-kind) so each readout trains only on its
    own kind's gold. Default False = the shared-body arch (byte-identical).

    Phase 1f-7 Stage 2 #6 ``per_kind_bodies`` (Head A ONLY, the architectural
    robust fix): N independent 6144->hidden ReLU bodies (one per kind) feeding ONE
    shared hidden->384 head. PER-SLOT routing (same as shared-body, NOT by-gold) --
    the shared head makes per-slot gold-vs-filler logits comparable. Default False
    = the shared-body arch (byte-identical). Mutually exclusive with
    ``per_kind_full`` (different readout archs)."""
    if arch == "bilinear":
        return CompositeZHead(dim_in=dim_in, hidden=hidden, n_doc_kinds=n_doc_kinds,
                               per_kind_full=per_kind_full,
                               per_kind_bodies=per_kind_bodies)
    if arch == "transformer":
        return CrossSlotTransformerZHead(dim_in=dim_in, hidden=hidden,
                                         n_slot_types=n_slot_types,
                                         learnable_temp=learnable_temp,
                                         dropout=dropout)
    raise ValueError(f"unknown arch {arch!r}")


def _load_head(arch: str, ckpt_path: str, dim_in: int, hidden: int | None,
               device: str, n_slot_types: int = 0, learnable_temp: bool = False,
               dropout: float = 0.0, n_doc_kinds: int = 0,
               per_kind_full: bool = False,
               per_kind_bodies: bool = False) -> nn.Module:
    """Reload a trained head from its checkpoint (best.pt or final.pt).

    The Phase-1b/1d knobs are read from the checkpoint (a Phase-1 head wrote
    them); a task-#45 checkpoint omits them -> the defaults 0/False/0.0 rebuild
    the byte-identical arch + strict-load. ``n_doc_kinds`` (Phase 1f-7) is read
    from the ckpt (default 0) for Head A; Head B ignores it. ``per_kind_full``
    (Stage 1 redesign MoE) + ``per_kind_bodies`` (Stage 2 #6 per-kind bodies +
    shared head) are read from the ckpt (default False). ``_build_head`` is used
    for Head A (bilinear knobs threaded) + Head B (slot-type knobs threaded)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["head"] if isinstance(ckpt, dict) and "head" in ckpt else ckpt
    if isinstance(ckpt, dict) and "head" in ckpt:
        n_slot_types = int(ckpt.get("n_slot_types", n_slot_types))
        learnable_temp = bool(ckpt.get("learnable_temp", learnable_temp))
        dropout = float(ckpt.get("dropout", dropout))
        n_doc_kinds = int(ckpt.get("n_doc_kinds", n_doc_kinds))
        per_kind_full = bool(ckpt.get("per_kind_full", per_kind_full))
        per_kind_bodies = bool(ckpt.get("per_kind_bodies", per_kind_bodies))
    head = _build_head(arch, dim_in, hidden, n_slot_types=n_slot_types,
                       learnable_temp=learnable_temp, dropout=dropout,
                       n_doc_kinds=n_doc_kinds, per_kind_full=per_kind_full,
                       per_kind_bodies=per_kind_bodies)
    head.load_state_dict(sd)
    dev = torch.device(device)
    return head.to(dev).eval()


# ── session-level split (held-out = unseen conversations) ──

def _session_of(rec: dict) -> str:
    """The session a record belongs to (all source_ids in a record share the
    same Onyx session UUID prefix, ``{session_id}#{msg_idx}``)."""
    return str(rec["source_ids"][0]).split("#", 1)[0]


def _self_slot_idx(rec: dict) -> int | None:
    """Index of the JUST-ADDED current-prompt slot in a mixed-ring record, or
    None if there is no conversation slot.

    The Phase 1c generator scores the ring AFTER ``orch.query(u)`` adds the
    prompt via ``working_memory.update`` (``strm_ring_text`` ON), so the
    current prompt is in the ring with ``source_id = "{session}#msg{counter}"``
    and trivially matches the query (cos ~= 1.0). The self-slot is the
    ``#msg`` slot with the HIGHEST numeric suffix (the orchestrator's
    per-query counter is monotonic, so the largest = the just-added prompt).
    Found [[pondr-strm-phase1d-self-match-rootcause]]: 100% of mixed-ring
    records have argmax cos >= 0.999 and 89% have the self-slot AS the argmax
    gold, so the model is trained to rank the query's own just-typed message
    highest -- a trivial identity task misaligned with the gate (the self-slot
    has a unique per-turn source_id -> <3 occurrences -> gate-ineligible).
    """
    msg_slots = [(i, str(s)) for i, s in enumerate(rec["source_ids"])
                 if "#" in str(s) and "__ep" not in str(s)]
    if not msg_slots:
        return None

    def _suffix(sid: str) -> int:
        try:
            return int(sid.rsplit("#msg", 1)[-1])
        except ValueError:
            return -1
    return max(msg_slots, key=lambda x: _suffix(x[1]))[0]


def _drop_self_slot(rec: dict, multi_positive_margin: float = 0.0) -> dict | None:
    """Return a copy of ``rec`` with the self-slot (current prompt) removed and
    labels re-derived over the REMAINING slots (the real "most relevant PRIOR
    message/episode" target, matching task #45's prior-only gold).

    ``multi_positive_margin`` > 0 -> multi-positive InfoNCE gold: every
    remaining slot within ``max_cos - margin`` is a positive (DeepSeek's fix
    for near-tied prior-message / retrieved-episode duplicates -- both pulled
    up, not forced into opposition). ``0.0`` -> single top-1-cos gold.
    Returns None if fewer than 3 slots remain (the contrastive loss needs >=3
    to form a meaningful filler pool; dropped + counted, never silently
    kept)."""
    si = _self_slot_idx(rec)
    keep = [j for j in range(len(rec["source_ids"])) if j != si]
    if len(keep) < 3:
        return None
    out = dict(rec)
    out["source_ids"] = [rec["source_ids"][j] for j in keep]
    out["slot_types"] = rec["slot_types"][keep]
    out["cos"] = rec["cos"][keep]
    for k in ("slots_h_raw", "slots_y", "slots_z", "slots_doc_emb",
              "slot_doc_kinds", "slots_pre_state", "slots_step_input"):
        if k in rec:
            out[k] = rec[k][keep]
    cos = out["cos"].to(torch.float32)
    labels = torch.zeros(len(keep), dtype=torch.float32)
    if multi_positive_margin > 0.0:
        mx = float(cos.max())
        labels[cos >= mx - multi_positive_margin] = 1.0
        if float(labels.sum()) == 0.0:  # guard: margin too tight -> top-1
            labels[int(cos.argmax().item())] = 1.0
    else:
        labels[int(cos.argmax().item())] = 1.0
    out["labels"] = labels
    return out


def _gold_source_id(rec: dict) -> str | None:
    """The source_id of a record's GOLD slot after ``--drop-self-slot`` (single
    top-1-cos gold: ``labels`` has one 1.0 at the argmax-cos prior slot). Stage 0
    uses this to test whether the record's retrieval target is a CODE doc."""
    idx = int(rec["labels"].argmax())
    sids = rec["source_ids"]
    if idx >= len(sids):
        return None
    return str(sids[idx])


def _gold_doc_kind(rec: dict, doc_kind_map: dict[str, str] | None) -> str | None:
    """'code' / 'text' / None for a record's gold slot (None for a conv gold or
    when no doc_kind_map). Reuses ``p_sel._doc_kind_for_source`` so the
    source_id -> doc_id -> kind resolution matches the live probe exactly."""
    sid = _gold_source_id(rec)
    if sid is None:
        return None
    return p_sel._doc_kind_for_source(sid, doc_kind_map)


def _build_doc_kind_map_cached(doc_store: str, onyx_path: str) -> dict[str, str]:
    """Build {doc_id: 'code'|'text'} from the persisted doc store, cached to JSON
    next to the traces so a later run with the store moved/missing still works
    (the doc_id -> kind mapping is stable for a frozen corpus). Stage 0 only."""
    cache_path = Path(onyx_path).parent / (Path(onyx_path).stem + ".doc_kind_map.json")
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt cache -> rebuild below
            pass
    if not doc_store or not Path(doc_store).exists():
        raise SystemExit(f"--code-only-gold needs the doc store (or a cached "
                         f"doc_kind_map); not found: {doc_store!r} and no cache "
                         f"at {cache_path}")
    from src.memory.store import HippocampalStore  # noqa: E402
    store = HippocampalStore(str(doc_store))
    try:
        dk = p_sel._build_doc_kind_map(store)
    finally:
        # Match probe_strm_selectivity_real.py: the store holds a WaveDB handle;
        # close it so the build does not leak the DB across the trainer process.
        try:
            store.close()
        except Exception:  # noqa: BLE001 - best-effort close
            pass
    cache_path.write_text(json.dumps(dk, indent=2), encoding="utf-8")
    return dk


def _filter_code_gold(records: list[dict],
                      doc_kind_map: dict[str, str] | None) -> list[dict]:
    """Stage 0: keep only records whose gold slot is a CODE doc (the code-doc
    retrieval task). Records whose gold is a conv slot or a text doc are dropped
    (counted + reported)."""
    kept, n_conv, n_text, n_none = [], 0, 0, 0
    for r in records:
        k = _gold_doc_kind(r, doc_kind_map)
        if k == "code":
            kept.append(r)
        elif k == "text":
            n_text += 1
        elif k is None:
            n_none += 1  # conv gold or unmapped
    print(f"  --code-only-gold: kept {len(kept)} code-gold records "
          f"(dropped {n_text} text-gold, {n_none} conv/unmapped) of "
          f"{len(records)}", flush=True)
    return kept


def _gold_cos_gap(rec: dict) -> float | None:
    """Baseline well-posedness / h-norm guard for Stage 0 (S0.4): the INPUT bge-
    embedding cos gap = gold_slot_cos - mean(filler_cos). The gold slot is the
    argmax-cos prior slot (labels re-derived over prior slots by
    ``--drop-self-slot``), so this is positive by construction -- the QUESTION is
    how large. A near-zero median on the code-gold records means the bge
    embedding itself barely separates the gold code doc from fillers -> the task
    is ill-posed at the embedding layer (and a z_logit FAIL is not evidence
    about ``h``). A healthy positive + a z_logit FAIL points at ``h``/readout
    (Stage 1/2), not the embedding. Guards against a trivial h-norm false
    positive too: if the head's z_logit gap tracks this cos gap, the readout
    added nothing over the embedding."""
    cos = rec.get("cos")
    labels = rec.get("labels")
    if cos is None or labels is None:
        return None
    cos = cos.to(torch.float32)
    labels = labels.to(torch.float32)
    gold = labels > 0
    if gold.sum() == 0 or (~gold).sum() == 0:
        return None
    g = float(cos[gold].mean())
    f = float(cos[~gold].mean())
    return g - f


def _gold_cos_baseline(records: list[dict]) -> dict:
    """Median + n of the per-record gold-cos gap over ``records`` (Stage 0)."""
    gaps = [g for g in (_gold_cos_gap(r) for r in records) if g is not None]
    if not gaps:
        return {"median": None, "n": 0}
    return {"median": float(statistics.median(gaps)), "n": len(gaps)}


# Stage 1 doc-kind indexing for the per-doc-kind readout. Reuses the live
# probe's source_id -> kind resolution so train/serve align. conv/unmapped -> 0,
# text-doc -> 1, code-doc -> 2 (the ``n_doc_kinds=3`` channel; the transformer's
# n_slot_types conv/retrieved channel is ORTHOGONAL -- do NOT conflate them).
DOC_KIND_CONV = 0
DOC_KIND_TEXT = 1
DOC_KIND_CODE = 2


def _slot_doc_kinds(rec: dict, doc_kind_map: dict[str, str] | None) -> torch.Tensor:
    """A ``[K] long`` doc-kind index per slot (conv=0 / text=1 / code=2) for the
    per-doc-kind readout. Built from ``source_ids`` + the doc_kind_map (Stage 1
    S1.5). When ``doc_kind_map is None`` returns all-0 (byte-identical: the
    shared readout is the only head, so routing is a no-op)."""
    sids = rec["source_ids"]
    if doc_kind_map is None:
        return torch.zeros(len(sids), dtype=torch.long)
    out = torch.zeros(len(sids), dtype=torch.long)
    for i, sid in enumerate(sids):
        k = p_sel._doc_kind_for_source(str(sid), doc_kind_map)
        if k == "text":
            out[i] = DOC_KIND_TEXT
        elif k == "code":
            out[i] = DOC_KIND_CODE
        else:
            out[i] = DOC_KIND_CONV
    return out


def _session_split(records: list[dict], val_fraction: float, seed: int,
                    live_eval_sessions: frozenset[str] = frozenset()):
    """Split records by SESSION so held-out turns are ENTIRE unseen
    conversations (the true generalization test; a query-split would leak a
    session's turns into both halves). Returns
    ``(train, val, live_eval, val_sessions, live_eval_sessions)``.

    ``live_eval_sessions`` (D0.4a) are forced OUT of the train pool AND out of
    the random held-out draw, into a dedicated ``live_eval`` bucket evaluated
    separately per seed. The random held-out is drawn from the REMAINING
    sessions, so a live-eval session is never in train (clean held-out) and
    never double-counted in the random held-out. Empty ``live_eval_sessions``
    -> ``live_eval`` is empty -> byte-identical to pre-#47 (train/val split over
    ALL sessions, no separate bucket)."""
    sessions = sorted({_session_of(r) for r in records})
    # Keep only live-eval ids actually present in the traces (silently drop a
    # stale id -- e.g. a transcript no longer in the training set).
    live_eval_sessions = {s for s in live_eval_sessions if s in sessions}
    rng = random.Random(seed)
    pool = [s for s in sessions if s not in live_eval_sessions]
    shuffled = pool[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_sessions = set(shuffled[:n_val])
    train = [r for r in records if _session_of(r) not in val_sessions
             and _session_of(r) not in live_eval_sessions]
    val = [r for r in records if _session_of(r) in val_sessions]
    live_eval = [r for r in records if _session_of(r) in live_eval_sessions]
    return train, val, live_eval, sorted(val_sessions), sorted(live_eval_sessions)


# ── the contrastive training loop (generic over Head A / Head B) ──

def margin_ranking_loss(logits: Tensor, gold_mask: Tensor, margin: float,
                        hard_negative: bool = False,
                        neg_mask: Tensor | None = None) -> Tensor:
    """Pairwise hinge margin loss that DIRECTLY optimizes the z_logit gate.

    ``L = mean_{g in gold, f in filler} relu(margin - (logit_g - logit_f))``.

    The live gate is the z_logit gap = (top-gold logit) - (mean filler logit); a
    per-pair hinge of ``margin`` is a TIGHTER surrogate -- if every gold logit
    exceeds every filler logit by ``margin``, the gate gap is >= ``margin`` by
    construction. Unlike InfoNCE (which a constant function can satisfy when the
    softmax target is near-uniform), the hinge is only zero when the gap is
    realized, so it cannot collapse to a flat logit vector -- the failure mode
    DeepSeek diagnosed for the transformer ([[pondr-strm-phase1f-deepseek-live-inversion-diagnosis]],
    ranked experiment #1).

    Bias-invariant for the bilinear composite: its scalar ``bias`` adds the same
    constant to ``logit_g`` and ``logit_f`` -> cancels in the diff. For the
    transformer the per-slot logits are attention-produced; the diff is still the
    optimized quantity regardless.

    ``hard_negative`` (DeepSeek's "hardest negatives from the ring"): instead of
    all gold x filler pairs, score only the HARDEST filler (max filler logit) per
    gold -- ``mean_{g} relu(margin - (logit_g - max_filler_logit))``. Focuses the
    gradient on the most-violated pair per gold (the filler the head is most
    tempted to rank above the gold) instead of averaging over easy, already-
    satisfied pairs. 0 if no gold or no filler.

    ``neg_mask`` (Phase 1f-7 Stage 2 #3, code hard-neg mining, DeepSeek's
    highest-leverage step): an optional boolean mask over slots restricting which
    non-gold slots are ELIGIBLE fillers. When provided, ``f`` is drawn from
    ``(~gold_mask) & neg_mask`` instead of ``~gold_mask``. For a code-gold record
    passing ``neg_mask = (slot_doc_kinds == DOC_KIND_CODE)``, the hardest filler is
    forced to be a CODE slot -- the hinge pushes the code gold above the highest-
    scoring code FILLER specifically, teaching code-vs-code separation (the exact
    1f-6 code mis-ranking failure mode: code gold and code fillers both score
    high -> no gap). AdamW-clean (a loss-STRUCTURE change, not gradient
    weighting). ``None`` = all non-gold slots eligible = byte-identical.
    """
    if gold_mask.sum() == 0:
        # Grad-safe zero: ``logits.new_zeros(())`` has no grad_fn, so
        # ``loss.backward()`` in the train loop would raise ("element 0 of
        # tensors does not require grad"). ``logits.sum() * 0.0`` returns
        # exactly 0.0 with a grad_fn that propagates a ZERO gradient -- the
        # empty-filler record contributes nothing but the backward pass
        # succeeds. Phase 1f-7 Stage 2 #3 (code hard-neg): the
        # ``f.numel() == 0`` branch below is now reachable for the ~3/143
        # code-gold records whose ring has NO other code filler (the mask
        # empties the pool). In 1f-6 (no neg_mask) this path is never hit
        # (every record has >=1 gold and >=2 fillers) -> byte-identical.
        return logits.sum() * 0.0
    g = logits[gold_mask]                       # [n_gold]
    fill_mask = ~gold_mask
    if neg_mask is not None:
        fill_mask = fill_mask & neg_mask
    f = logits[fill_mask]                        # [n_fill]
    if f.numel() == 0:
        return logits.sum() * 0.0
    if hard_negative:
        hardest = f.max()
        return torch.relu(margin - (g - hardest)).mean()
    # All gold x filler pairs: [n_gold, n_fill].
    return torch.relu(margin - (g.unsqueeze(1) - f.unsqueeze(0))).mean()


def _train_head(arch: str, train: list[dict], val: list[dict], hidden: int | None,
                temperature: float, epochs: int, lr: float, weight_decay: float,
                accum_steps: int, seed: int, device: str, ckpt_dir: Path,
                n_slot_types: int = 0, learnable_temp: bool = False,
                dropout: float = 0.0, label_smoothing: float = 0.0,
                cosine_schedule: bool = False,
                margin_loss: float = 0.0, hard_negative: bool = False,
                n_doc_kinds: int = 0, kind_head_wd: float = 0.0,
                per_kind_full: bool = False,
                per_kind_bodies: bool = False,
                class_balanced_gold: bool = False,
                per_kind_loss_weight: bool = False,
                code_hard_neg: bool = False,
                no_replacement_sampler: bool = False,
                sqrt_freq_sampler: bool = False) -> dict:
    """Train one head (bilinear OR transformer) on the train sessions with the
    SAME contrastive InfoNCE loss. Mirrors ``p44._train_contrastive`` but is
    generic over the head arch. Uses ``evaluate_relevance`` on the train-internal
    query-val split for the per-epoch TRAIN top-3 gate + best-ckpt selection
    (this is a TRAINING signal, not the final held-out eval). Writes best.pt +
    final.pt; the caller chooses which to score (``--select-ckpt``).

    Phase 1d knobs (all default to the byte-identical task-#45 path):
    ``n_slot_types``/``learnable_temp``/``dropout`` build a Phase-1 Head B;
    ``label_smoothing`` softens the InfoNCE target (``p44.contrastive_loss``);
    ``cosine_schedule`` anneals the lr. When all are 0/False the loop is
    byte-identical to pre-#52 (no slot_types kwarg, hard loss, constant lr).

    Phase 1f-5 margin-loss knob (default 0.0 = byte-identical, uses
    ``p44.contrastive_loss``): when ``margin_loss`` > 0 the loop uses
    ``margin_ranking_loss`` (pairwise hinge, directly optimizes the z_logit gate)
    INSTEAD of InfoNCE; ``hard_negative`` restricts the hinge to the hardest
    filler per gold. DeepSeek's ranked experiment #1 for the transformer's live
    flatness ([[pondr-strm-phase1f-deepseek-live-inversion-diagnosis]]). When
    active, ``temperature``/``label_smoothing`` are unused by the loss (the head's
    learnable_temp is still constructed but the margin loss ignores T)."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    dim_in = int(train[0]["slots_h_raw"].shape[1])
    head = _build_head(arch, dim_in, hidden, n_slot_types=n_slot_types,
                       learnable_temp=learnable_temp, dropout=dropout,
                       n_doc_kinds=n_doc_kinds, per_kind_full=per_kind_full,
                       per_kind_bodies=per_kind_bodies).to(dev)
    use_slot_types = getattr(head, "n_slot_types", 0) > 0
    use_doc_kinds = getattr(head, "n_doc_kinds", 0) > 0
    use_moe = bool(getattr(head, "per_kind_full", False))
    use_per_kind_bodies = bool(getattr(head, "per_kind_bodies", False))
    n_params = sum(p.numel() for p in head.parameters())
    arch_name = (f"MLP-{hidden}" if hidden else "Linear") if arch == "bilinear" \
        else f"Transformer({'MLP-'+str(hidden) if hidden else 'Linear'} readout)"
    if use_doc_kinds:
        arch_name = f"PerKind-{arch_name}({n_doc_kinds} kinds)"
        if use_moe:
            arch_name = f"MoE-{arch_name}(by-gold train, per-slot serve)"
        elif use_per_kind_bodies:
            arch_name = (f"PerKindBodies-{arch_name}"
                         f"(per-slot train+serve, shared head)")
    loss_tag = (f"MARGIN(m={margin_loss}, hard-neg={hard_negative})"
                if margin_loss > 0.0 else f"CONTRASTIVE(T={temperature})")
    print(f"\ntraining {arch} seed={seed} {loss_tag} ({arch_name} {dim_in}->384, "
          f"{n_params:,} params, wd={weight_decay}, {epochs} epochs, "
          f"slot_types={use_slot_types}, doc_kinds={use_doc_kinds}, "
          f"dropout={dropout}, label_smoothing={label_smoothing}, "
          f"cosine={cosine_schedule}) -> {ckpt_dir}", flush=True)

    # Phase 1f-7: heavier weight decay on the per-kind heads (the ~400 code
    # slots overfit; kind_head_wd applies ONLY to the per-kind params, the
    # shared body + z_head keep the base weight_decay). Param groups by name
    # prefix; AdamW applies per-group weight_decay. The per-kind params are
    # ``readout.kind_heads.*`` (shared-body arch) OR ``readout.kind_readouts.*``
    # (MoE per_kind_full arch) OR ``readout.kind_bodies.*`` (per_kind_bodies
    # arch; the shared ``readout.head.*`` stays in base_params -- over-
    # regularizing the cross-kind head would hurt comparability). Match any.
    # For the MoE / per_kind_bodies runs kind_head_wd=0 (default) so this block
    # is a no-op (base wd on all params = the Stage 1 over-regularization fix).
    if use_doc_kinds and kind_head_wd > 0.0:
        kind_params, base_params = [], []
        for n, p in head.named_parameters():
            if (n.startswith("readout.kind_heads.")
                    or n.startswith("readout.kind_readouts.")
                    or n.startswith("readout.kind_bodies.")):
                kind_params.append(p)
            else:
                base_params.append(p)
        optimizer = torch.optim.AdamW(
            [{"params": base_params, "weight_decay": weight_decay},
             {"params": kind_params, "weight_decay": kind_head_wd}],
            lr=lr)
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if cosine_schedule else None)
    # Per-epoch TRAIN gate: query-split WITHIN train (mirrors _train_contrastive).
    train_idx, valq_idx = _split_queries(len(train),
                                         RelevanceTrainingConfig().val_fraction, seed)
    train_q = [train[i] for i in train_idx]
    valq = [train[i] for i in valq_idx]
    print(f"  train sessions -> {len(train_q)} train / {len(valq)} query-val "
          f"(held-out: {len(val)} turns from {len({_session_of(r) for r in val})} "
          f"unseen sessions)", flush=True)

    # Phase 1f-7 Stage 2 (DeepSeek #1 / #4 / #5): class-balanced per-record
    # sampling weights on the SHARED readout. Upweight minority gold kinds
    # (code/text) and downweight conv so each kind contributes ~equally per
    # epoch -- fixes the conv-majority-dominated shared-body root cause WITHOUT
    # per-kind decomposition, preserving the cross-kind logit space. Base weights
    # = 1 / count(gold_doc_kind) over train_q. Three variants share this machinery:
    #   - --class-balanced-gold (#1): WeightedRandomSampler(replacement=True,
    #     num_samples=len(train_q)) -- WITH replacement (code ~2x/epoch, some conv
    #     records unseen/epoch). Passed the gate s2-only (1/3, extreme variance).
    #   - --no-replacement-sampler (#4, DeepSeek ladder #2): WeightedRandomSampler
    #     (replacement=False, num_samples=len(train_q)) -- a WEIGHTED SHUFFLE: every
    #     record seen EXACTLY once/epoch in a weighted order. Removes the
    #     replacement-sampling noise that drove #1's extreme seed variance (s0/s1
    #     inverted, s2 passed both) while keeping the per-epoch balance that let s2
    #     pass BOTH buckets + the per-step loss scale 1.0 (AdamW-clean). DeepSeek's
    #     ranked variance-tamer for the #1-sampler.
    #   - --sqrt-freq-sampler (#5, DeepSeek (a)): same no-replacement weighted
    #     shuffle as #4 but with weights = sqrt(1/count) instead of 1/count -- a
    #     MILDER upweight (code ~1.9x vs conv instead of 3.6x, text ~1.2x instead
    #     of 2.4x). DeepSeek's medium-leverage fallback: reduces the conv-starvation
    #     that hurt s1/text in #1/#4 while keeping enough code gradient to (hopefully)
    #     push s2 code (+0.722 in #4) over the 2.0 threshold AND avoid s1's flat
    #     collapse. Still AdamW-clean (per-step loss scale 1.0). Hypothesis: #4 came
    #     closest (code median +0.722, s0 passed both) -- the 3.6x upweight was just
    #     aggressive enough to starve s1; sqrt tempers that without losing the code
    #     lift. Default off = byte-identical 1f-6.
    # The epoch loop uses a WeightedRandomSampler(sampler_weights, len(train_q),
    # replacement=cb_replace) instead of a uniform shuffle. Default off (all flags
    # False) = uniform shuffle = byte-identical 1f-6. valq (held-out) is NEVER
    # reweighted. ``cb_replace`` is only read when sampler_weights is set (one of
    # the three flags is on), so it is inert by default.
    sampler_weights = None
    cb_replace = True
    if class_balanced_gold or no_replacement_sampler or sqrt_freq_sampler:
        cb_counts = {DOC_KIND_CONV: 0, DOC_KIND_TEXT: 0, DOC_KIND_CODE: 0}
        for r in train_q:
            cb_counts[int(r.get("gold_doc_kind", DOC_KIND_CONV))] += 1
        inv = {k: (1.0 / cb_counts[k] if cb_counts[k] > 0 else 0.0)
               for k in cb_counts}
        # #5 sqrt-freq: milder upweight -- sqrt of the inverse frequency. code
        # ~sqrt(1/102)=0.099 vs conv ~sqrt(1/364)=0.052 -> ~1.9x (vs 3.6x at
        # power 1.0). Keeps the no-replacement weighted shuffle of #4.
        if sqrt_freq_sampler:
            inv = {k: (inv[k] ** 0.5) for k in cb_counts}
        w = [inv[int(r.get("gold_doc_kind", DOC_KIND_CONV))] for r in train_q]
        sampler_weights = torch.tensor(w, dtype=torch.double)
        cb_replace = not (no_replacement_sampler or sqrt_freq_sampler)
        if sqrt_freq_sampler:
            flag_name = "--sqrt-freq-sampler"
            weight_desc = "sqrt-inverse-freq"
        elif no_replacement_sampler:
            flag_name = "--no-replacement-sampler"
            weight_desc = "inverse-freq"
        else:
            flag_name = "--class-balanced-gold"
            weight_desc = "inverse-freq"
        repl_desc = ("replacement=False (weighted shuffle, each record "
                     "once/epoch)" if (no_replacement_sampler or sqrt_freq_sampler)
                     else "replacement=True (w/ replacement)")
        print(f"  {flag_name}: train gold-kind counts "
              f"conv={cb_counts[DOC_KIND_CONV]}/text={cb_counts[DOC_KIND_TEXT]}/"
              f"code={cb_counts[DOC_KIND_CODE]} -> per-record {weight_desc} weight "
              f"conv={inv[DOC_KIND_CONV]:.4f}/text={inv[DOC_KIND_TEXT]:.4f}/"
              f"code={inv[DOC_KIND_CODE]:.4f} (WeightedRandomSampler, "
              f"{len(train_q)} samples/epoch, {repl_desc})", flush=True)

    # Phase 1f-7 Stage 2 (DeepSeek #2): per-kind inverse-frequency LOSS weighting
    # on the SHARED readout. Same per-kind BALANCE as the sampler (each kind
    # contributes equally to the loss) but UNIFORM sampling: every record is
    # seen exactly once/epoch, NO replacement. Per-record loss weight =
    # 1/count(gold_doc_kind), normalized to mean 1.0 so the effective lr/loss
    # scale is unchanged -- only the per-record gradient RATIO shifts (code
    # ~3.6x conv per record).
    #
    # RESULT (recorded negative, [[pondr-strm-phase1f7-stage2-lossweight-fail]]):
    # this COLLAPSES training under AdamW -- all 3 seeds/both archs stuck at the
    # margin-loss ceiling (train_loss ~2.495, top3 ~0.27) from ~epoch 5, r_pos
    # -> 0.03 (degenerate anti-correlated ranking). Root: per-record loss
    # SCALING fights AdamW's adaptive moments -- the running first/second
    # moments MIX gradients scaled by different per-record weights (0.57x conv
    # to 2.0x code), and the eps + bias-correction terms break the per-step
    # scale invariance AdamW normally provides, so the optimizer is pushed to a
    # degenerate fixed point. This is WORSE than the sampler (#1), which kept
    # each step's loss at scale 1.0 (only the record MIX changed) and so trained
    # normally in-sample (top3 ~0.65) -- the sampler is the AdamW-clean
    # rebalancing mechanism; loss-weighting is not. Default off
    # (per_kind_loss_weight=False) = unweighted loss = byte-identical 1f-6.
    record_loss_weights = None
    if per_kind_loss_weight:
        lw_counts = {DOC_KIND_CONV: 0, DOC_KIND_TEXT: 0, DOC_KIND_CODE: 0}
        for r in train_q:
            lw_counts[int(r.get("gold_doc_kind", DOC_KIND_CONV))] += 1
        inv_lw = {k: (1.0 / lw_counts[k] if lw_counts[k] > 0 else 0.0)
                  for k in lw_counts}
        raw = torch.tensor(
            [inv_lw[int(r.get("gold_doc_kind", DOC_KIND_CONV))] for r in train_q],
            dtype=torch.double)
        mean_w = float(raw.mean()) if raw.numel() else 1.0
        if mean_w <= 0.0:
            mean_w = 1.0
        record_loss_weights = (raw / mean_w).to(torch.float32)
        print(f"  --per-kind-loss-weight: train gold-kind counts "
              f"conv={lw_counts[DOC_KIND_CONV]}/text={lw_counts[DOC_KIND_TEXT]}/"
              f"code={lw_counts[DOC_KIND_CODE]} -> normalized per-record loss "
              f"weight (mean=1.0) conv={inv_lw[DOC_KIND_CONV]/mean_w:.3f}/"
              f"text={inv_lw[DOC_KIND_TEXT]/mean_w:.3f}/"
              f"code={inv_lw[DOC_KIND_CODE]/mean_w:.3f} (uniform sampling, no "
              f"replacement -> preserves text 30.16)", flush=True)

    rng = random.Random(seed)
    best_score: tuple | None = None
    best_pc: dict | None = None
    best_epoch = -1
    last_pc: dict | None = None
    last_go = False
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accum = max(1, accum_steps)
    ck_slot_dim = int(head.slot_dim)
    ck_doc_dim = int(head.doc_dim)
    ck_query_dim = int(head.query_dim)
    ck_proj_dim = int(head.proj_dim)

    def _save_ckpt(path: Path, pc: dict, epoch: int, go: bool) -> None:
        torch.save({"head": head.state_dict(), "arch": arch,
                    "slot_dim": ck_slot_dim, "doc_dim": ck_doc_dim,
                    "query_dim": ck_query_dim, "proj_dim": ck_proj_dim,
                    "hidden": hidden,
                    "n_slot_types": int(getattr(head, "n_slot_types", 0)),
                    "n_doc_kinds": int(getattr(head, "n_doc_kinds", 0)),
                    "per_kind_full": bool(getattr(head, "per_kind_full", False)),
                    "per_kind_bodies": bool(getattr(head, "per_kind_bodies", False)),
                    "learnable_temp": bool(getattr(head, "learnable_temp", False)),
                    "dropout": float(getattr(head, "dropout", 0.0)),
                    "label_smoothing": float(label_smoothing),
                    "top3_recall": pc["mean_top3_recall"],
                    "hit_rate": pc["hit_rate"], "hit_ci95": pc["hit_ci95"],
                    "go": go, "epoch": epoch,
                    "loss": "margin" if margin_loss > 0.0 else "contrast",
                    "margin": float(margin_loss),
                    "hard_negative": bool(hard_negative)}, path)

    for epoch in range(epochs):
        head.train()
        if sampler_weights is not None:
            # Class-balanced: sample len(train_q) records, weighted by inverse
            # gold-kind frequency. ``cb_replace`` selects WITH replacement (#1
            # sampler) vs WITHOUT (#4 weighted shuffle -- each record once/epoch).
            # Per-epoch generator seeded by (seed, epoch) for determinism
            # (reproducible across reruns).
            gen = torch.Generator().manual_seed(seed * 100003 + epoch)
            sampler = torch.utils.data.WeightedRandomSampler(
                sampler_weights, num_samples=len(train_q),
                replacement=cb_replace, generator=gen)
            order = list(sampler)
        else:
            order = list(range(len(train_q)))
            rng.shuffle(order)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for k, qi in enumerate(order):
            rec = train_q[qi]
            z_flat = rec["slots_h_raw"].to(dev).to(torch.float32)      # [K,6144]
            q = rec["query_emb"].to(dev).to(torch.float32)            # [384]
            labels = rec["labels"].to(dev).to(torch.float32)          # [K]
            K = z_flat.shape[0]
            slot_y = torch.zeros(K, ck_slot_dim, device=dev)
            # Phase 1d: thread slot_types into the Phase-1 head (n_slot_types>0);
            # Head A + the task-#45 Head B (n_slot_types=0) take the no-kwarg path
            # -> byte-identical. ``rec["slot_types"]`` is present on the mixed-ring
            # traces; old conv-only traces default to all-0 (loader) -- unused
            # when the head has no slot-type embedding.
            if use_slot_types:
                st = rec.get("slot_types")
                if st is None:
                    st = torch.zeros(K, dtype=torch.long)
                logits = head.logits(slot_y, z_flat, q,
                                     slot_types=st.to(dev).long()).squeeze(-1)
            elif use_doc_kinds:
                # Phase 1f-7: per-doc-kind bilinear. Two routing regimes:
                # - MoE per_kind_full (Stage 1 redesign): route ALL slots through
                #   the GOLD's readout (by-gold-kind) so the gold-vs-filler margin
                #   loss is WITHIN one readout's logit space (well-defined, exactly
                #   Stage 0's setup). Gradient flows into one readout per record =
                #   each readout trains only on its own kind's gold (non-overlapping
                #   data, no cross-kind competition). ``gold_doc_kind`` is stamped
                #   on the record in ``main`` AFTER --drop-self-slot (the gold
                #   index changes when the self-slot is dropped).
                # - shared-body (Stage 1): per-slot routing via the stamped
                #   ``slot_doc_kinds`` (each slot -> its own kind head).
                if use_moe:
                    gk = int(rec.get("gold_doc_kind", DOC_KIND_CONV))
                    dk = torch.full((K,), gk, dtype=torch.long)
                else:
                    dk = rec.get("slot_doc_kinds")
                    if dk is None:
                        dk = torch.zeros(K, dtype=torch.long)
                logits = head.logits(slot_y, z_flat, q,
                                     slot_doc_kinds=dk.to(dev).long()).squeeze(-1)
            else:
                logits = head.logits(slot_y, z_flat, q).squeeze(-1)   # [K]
            gold = labels > 0
            # Phase 1f-7 Stage 2 #3 (DeepSeek #1): code hard-neg mining. For
            # code-gold records, RESTRICT the margin-hinge negatives to CODE
            # filler slots (neg_mask = slot_doc_kinds == DOC_KIND_CODE). This
            # forces the hardest negative to be a code slot, directly teaching
            # code-vs-code separation -- the 1f-6 code mis-rank failure (code
            # gold + code fillers both scoring high -> no gap). AdamW-CLEAN: a
            # loss-STRUCTURE change (which negatives are in the hinge), NOT a
            # gradient-weight change -- each step's loss stays at scale 1.0, so
            # AdamW's running moments behave normally (unlike #2 loss-weighting,
            # which collapsed). Text/conv-gold records keep an unrestricted
            # hinge (neg_mask=None) so the text 30.16 direction is untouched.
            # Only valid for the shared readout (n_doc_kinds=0): per-slot kinds
            # are stamped for the mask even though the head does not route on
            # them (slot_doc_kinds is stamped in main when code_hard_neg is on).
            neg_mask = None
            if code_hard_neg and int(rec.get("gold_doc_kind", DOC_KIND_CONV)) == DOC_KIND_CODE:
                sdk = rec.get("slot_doc_kinds")
                if sdk is not None:
                    # ``slot_doc_kinds`` is stamped in ``main`` AFTER
                    # ``_to_device`` -> it stays on CPU even when
                    # ``args.device`` is CUDA (the MoE/n_doc_kinds>0 path hides
                    # this by ``dk.to(dev)`` at the logits call, which does not
                    # touch the stored field). Build the mask on the logits'
                    # device so the ``& gold_mask`` + ``logits[fill_mask]``
                    # index stay co-located.
                    neg_mask = (sdk.to(logits.device) == DOC_KIND_CODE)
            if margin_loss > 0.0:
                loss = margin_ranking_loss(logits, gold, margin_loss,
                                           hard_negative=hard_negative,
                                           neg_mask=neg_mask) / accum
            else:
                loss = p44.contrastive_loss(logits, gold, temperature,
                                            label_smoothing=label_smoothing) / accum
            # Phase 1f-7 Stage 2 #2: scale this record's loss by its normalized
            # inverse-frequency weight (uniform sampling preserved). Applied
            # AFTER the /accum so the per-step gradient keeps the per-record ratio
            # and the accumulated-batch mean stays 1.0. ``qi`` is the train_q index
            # the sampler/shuffle selected for this step -> matches the weight
            # tensor's index space. Byte-identical when record_loss_weights is None.
            if record_loss_weights is not None:
                loss = loss * float(record_loss_weights[qi])
            loss.backward()
            total_loss += float(loss.item()) * accum
            n_steps += 1
            if (k + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        if n_steps % accum != 0:
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / max(n_steps, 1)
        pc = evaluate_relevance(head, valq, slot_signal_field="slots_h_raw")
        last_pc = pc
        last_go = (pc["mean_top3_recall"] >= RelevanceTrainingConfig().gate_top3
                   and pc["hit_ci95"][0] > RelevanceTrainingConfig().gate_wilson_low)
        ci = pc["hit_ci95"]
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} "
              f"top3={pc['mean_top3_recall']:.3f} hit={pc['hit_rate']:.2f} "
              f"ci=[{ci[0]:.2f},{ci[1]:.2f}] r_pos={pc['mean_r_positive']:.3f} "
              f"{'GO' if last_go else 'no-go'}", flush=True)

        score = _gate_score(pc, RelevanceTrainingConfig())
        if best_score is None or score > best_score:
            best_score = score
            best_pc = pc
            best_epoch = epoch
            _save_ckpt(ckpt_dir / "best.pt", pc, epoch, last_go)

    if last_pc is not None:
        _save_ckpt(ckpt_dir / "final.pt", last_pc, epochs - 1, last_go)
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "arch": arch,
                   "loss": "margin" if margin_loss > 0.0 else "contrast",
                   "margin": float(margin_loss),
                   "hard_negative": bool(hard_negative),
                   "temperature": temperature, "seed": seed,
                   "n_train": len(train_q), "n_valq": len(valq),
                   "n_heldout": len(val), "n_slot_types": n_slot_types,
                   "learnable_temp": learnable_temp, "dropout": dropout,
                   "label_smoothing": label_smoothing,
                   "cosine_schedule": cosine_schedule,
                   "best_scorecard": best_pc}, f, indent=2)

    return {"arch": arch, "arch_name": arch_name, "hidden": hidden,
            "dim_in": dim_in, "n_params": n_params, "best_epoch": best_epoch,
            "train_top3": best_pc["mean_top3_recall"] if best_pc else 0.0,
            "train_go": last_go,
            "ckpt_best": ckpt_dir / "best.pt", "ckpt_final": ckpt_dir / "final.pt"}


class _EnsembleHead(nn.Module):
    """Logit-averaging ensemble of N same-arch heads (Phase 1d, the
    [[jgs-head-multi-gate-best-practice-hypothesis]] pattern: averaging the
    per-seed logits at EVAL recovers both frontier ends the high-variance
    best-ckpt selection trades off). Exposes ``n_slot_types``/``slot_dim`` so
    ``p41._zr_per_slot``'s guarded ``slot_types`` forwarding works unchanged; the
    ``logits`` call averages the member logits (the softmax/sigmoid of an average
    is more stable than averaging probabilities, and the 2.0 z_logit gate is on
    the pre-sigmoid logit, so averaging logits is the gate-faithful aggregation).
    """

    def __init__(self, heads: list[nn.Module]) -> None:
        super().__init__()
        assert heads, "_EnsembleHead needs >=1 head"
        self.heads = nn.ModuleList(heads)
        h0 = heads[0]
        self.n_slot_types = int(getattr(h0, "n_slot_types", 0))
        # Phase 1f-7: mirror n_doc_kinds so a per-kind ensemble forwards the
        # slot_doc_kinds kwarg to its members (the live eval path gates on this
        # attribute; without it a per-kind ensemble would take the no-kwarg path
        # and the member heads would raise "requires doc_kinds").
        self.n_doc_kinds = int(getattr(h0, "n_doc_kinds", 0))
        # Phase 1f-7 Stage 1 redesign: mirror per_kind_full so the ensemble's
        # arch_name + ckpt stamp reflect the MoE arch (completeness; the
        # ensemble is not on the critical path).
        self.per_kind_full = bool(getattr(h0, "per_kind_full", False))
        # Phase 1f-7 Stage 2 #6: mirror per_kind_bodies (completeness; same
        # per-slot routing as the shared-body arch, already forwarded above).
        self.per_kind_bodies = bool(getattr(h0, "per_kind_bodies", False))
        self.slot_dim = int(getattr(h0, "slot_dim", 0))
        self.query_dim = int(getattr(h0, "query_dim", 0))
        self.doc_dim = int(getattr(h0, "doc_dim", 0))
        self.proj_dim = int(getattr(h0, "proj_dim", 0))

    def logits(self, slot_y, slot_signal, query_emb, slot_types=None,
               slot_doc_kinds=None):
        acc = None
        for h in self.heads:
            # Forward only the kwargs the member head declares (a mixed ensemble
            # is not supported -- all members share h0's n_slot_types/n_doc_kinds).
            kw = {}
            if self.n_slot_types > 0 and slot_types is not None:
                kw["slot_types"] = slot_types
            if self.n_doc_kinds > 0 and slot_doc_kinds is not None:
                kw["slot_doc_kinds"] = slot_doc_kinds
            lg = h.logits(slot_y, slot_signal, query_emb, **kw)
            acc = lg if acc is None else acc + lg
        assert acc is not None
        return acc / len(self.heads)


def _run_arch(arch: str, train: list[dict], val: list[dict],
              live_eval: list[dict], hidden: int | None,
              temperature: float, weight_decay: float, epochs: int, lr: float,
              accum_steps: int, seed: int, device: str, ckpt_root: Path,
              n_slot_types: int = 0, learnable_temp: bool = False,
              dropout: float = 0.0, label_smoothing: float = 0.0,
              cosine_schedule: bool = False,
              select_ckpt: str = "best",
              margin_loss: float = 0.0, hard_negative: bool = False,
              n_doc_kinds: int = 0, kind_head_wd: float = 0.0,
              per_kind_full: bool = False,
              per_kind_bodies: bool = False,
              class_balanced_gold: bool = False,
              per_kind_loss_weight: bool = False,
              code_hard_neg: bool = False,
              no_replacement_sampler: bool = False,
              sqrt_freq_sampler: bool = False) -> dict:
    """Train one arch (one seed), eval on held-out sessions + all-turns ceiling
    + (D0.4a) the live-eval bucket (the 2 live-transcript sessions, held OUT of
    train for every seed -> the genuinely-held-out conversation-ring gap).

    ``select_ckpt`` (Phase 1d, the [[pondr-strm-task47-d04a-selection-variance-cuda]]
    de-wonk fix): ``"best"`` (default, byte-identical to pre-#52) scores the
    valq-gate-selected best.pt; ``"final"`` scores the final-epoch ckpt, which
    sidesteps the high-variance valq draw that selected an undertrained best.pt
    in D0.4a. The returned ``head`` is the selected ckpt (used by the ensemble
    in ``main``)."""
    ckpt_dir = ckpt_root / f"{arch}_s{seed}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    r = _train_head(arch, train, val, hidden, temperature, epochs, lr,
                    weight_decay, accum_steps, seed, device, ckpt_dir,
                    n_slot_types=n_slot_types, learnable_temp=learnable_temp,
                    dropout=dropout, label_smoothing=label_smoothing,
                    cosine_schedule=cosine_schedule,
                    margin_loss=margin_loss, hard_negative=hard_negative,
                    n_doc_kinds=n_doc_kinds, kind_head_wd=kind_head_wd,
                    per_kind_full=per_kind_full,
                    per_kind_bodies=per_kind_bodies,
                    class_balanced_gold=class_balanced_gold,
                    per_kind_loss_weight=per_kind_loss_weight,
                    code_hard_neg=code_hard_neg,
                    no_replacement_sampler=no_replacement_sampler,
                    sqrt_freq_sampler=sqrt_freq_sampler)
    ckpt_path = r["ckpt_final"] if select_ckpt == "final" else r["ckpt_best"]
    r["select_ckpt"] = select_ckpt
    r["ckpt"] = ckpt_path
    # De-wonk (Phase 1d): the SCORED ckpt's epoch, not best.pt's. ``--select-ckpt
    # final`` scores final.pt (saved at ``epochs - 1``); ``--select-ckpt best``
    # scores best.pt (saved at ``best_epoch``). The verdict print + JSON report
    # the epoch of the ckpt actually scored, so a "best ep 1" line never misleads
    # when the scored ckpt is the final-epoch one.
    r["scored_epoch"] = (epochs - 1) if select_ckpt == "final" else r["best_epoch"]
    head = _load_head(arch, str(ckpt_path), r["dim_in"], hidden, device,
                      n_doc_kinds=n_doc_kinds, per_kind_full=per_kind_full,
                      per_kind_bodies=per_kind_bodies)
    r["head"] = head
    # The decisive eval: z_r + z_logit gaps on HELD-OUT sessions (unseen convs).
    r["heldout"] = p41._zr_and_logit_gaps(head, val, device)
    # All-turns ceiling (in-sample upper bound -- if even this fails, no signal).
    r["allturns"] = p41._zr_and_logit_gaps(head, train + val, device)
    # D0.4a: the live-transcript sessions, held OUT of train this seed -> the
    # genuinely-held-out conversation-ring z_logit gap (H3 vs H2 diagnostic).
    r["live_eval"] = (p41._zr_and_logit_gaps(head, live_eval, device)
                      if live_eval else None)
    r["seed"] = seed
    ho = r["heldout"]
    on = r["allturns"]
    le = r["live_eval"]
    le_str = (f"  LIVE-EVAL z_logit={le['z_logit']['median']:.3f} "
              f"({'PASS' if le['z_logit']['median'] is not None and le['z_logit']['median']>=ZLOGIT_GATE else 'fail'}, "
              f"n={le['z_logit']['n_eligible']})" if le else "  LIVE-EVAL n/a")
    print(f"  [{arch} s{seed} {select_ckpt}] HELD-OUT z_logit={ho['z_logit']['median']:.3f} "
          f"(n_ge_2.0={ho['z_logit']['n_ge_gate']}/{ho['z_logit']['n_eligible']}, "
          f"{'PASS' if ho['z_logit']['median'] is not None and ho['z_logit']['median']>=ZLOGIT_GATE else 'fail'})  "
          f"z_r={ho['z_r']['median']:.4f}  |  ALL-TURNS z_logit={on['z_logit']['median']:.3f}"
          f"{le_str}", flush=True)
    return r


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task #45: Head A (bilinear) vs Head B (cross-slot "
                    "Transformer) head-to-head on REAL Onyx serve traces.")
    p.add_argument("--onyx", default=DEFAULT_ONYX,
                   help="Onyx serve traces (generate_onyx_serve_traces.py).")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated train seeds (robustness sweep).")
    p.add_argument("--readout", default="mlp128",
                   choices=["linear", "mlp64", "mlp128"],
                   help="StateReadout arch (shared by Head A + Head B so the "
                        "win is attributable to the cross-slot mechanism).")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="fraction of SESSIONS held out (unseen conversations).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--device", default="cpu",
                   help="train+eval device (cuda trains much faster).")
    p.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    # ── Phase 1d knobs (all default to the byte-identical task-#45 path) ──
    p.add_argument("--n-slot-types", type=int, default=0,
                   help="Phase 1b: Head B slot-type embedding size (2 = conv vs "
                        "retrieved; 0 = NO embedding = byte-identical to task #45).")
    p.add_argument("--learnable-temp", action="store_true",
                   help="Phase 1b: Head B learnable logit temperature (default "
                        "off = byte-identical to task #45).")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Phase 1d: readout dropout (0.0 = no-op = byte-identical; "
                        "0.1 for the Phase 1 retrain).")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Phase 1d: soft-target InfoNCE smoothing (0.0 = hard-mask "
                        "= byte-identical; 0.05 for the Phase 1 retrain).")
    p.add_argument("--cosine-schedule", action="store_true",
                   help="Phase 1d: cosine-anneal the lr over --epochs (default off "
                        "= constant lr = byte-identical).")
    p.add_argument("--select-ckpt", choices=["best", "final"], default="best",
                   help="Phase 1d: which ckpt to score + ensemble. 'best' (default) "
                        "= the valq-gate-selected best.pt (byte-identical to "
                        "pre-#52); 'final' = the final-epoch ckpt (sidesteps the "
                        "high-variance valq draw -- the D0.4a de-wonk).")
    p.add_argument("--ensemble", action="store_true",
                   help="Phase 1d: also eval a logit-averaging ensemble of the "
                        "per-seed selected ckpts (the jgs-head multi-gate pattern).")
    p.add_argument("--drop-self-slot", action="store_true",
                   help="Phase 1d root-cause fix: remove the JUST-ADDED current-"
                        "prompt slot (cos ~= 1.0, the trivial self-match gold "
                        "89 pct of records) from every ring and re-derive gold over "
                        "the remaining PRIOR slots -- the real relevance target. "
                        "OFF (default) = use stored labels = byte-identical to the "
                        "collapsed run. The head locates PRIOR memory, not the "
                        "query itself; the self-slot is gate-ineligible anyway "
                        "(unique per-turn source_id < 3 occurrences).")
    p.add_argument("--multi-positive-margin", type=float, default=0.0,
                   help="Phase 1d: when >0 (with --drop-self-slot or alone), use "
                        "multi-positive InfoNCE gold -- every slot within "
                        "max_cos - margin is a positive (DeepSeek's fix for "
                        "near-tied prior-message / retrieved-episode duplicates). "
                        "0.0 = single top-1-cos gold = byte-identical.")
    p.add_argument("--margin-loss", type=float, default=0.0,
                   help="Phase 1f-5: when >0, replace InfoNCE with a pairwise "
                        "hinge margin loss that DIRECTLY optimizes the z_logit "
                        "gate -- mean_{gold,filler} relu(margin - (logit_gold - "
                        "logit_filler)). DeepSeek's ranked experiment #1 for the "
                        "transformer's live flatness (a constant function can "
                        "satisfy InfoNCE under a near-uniform target; the hinge "
                        "is only zero when the gap is realized, so it cannot "
                        "collapse to a flat logit vector). 0.0 = InfoNCE = "
                        "byte-identical. When active, --temperature / "
                        "--label-smoothing are unused by the loss (the head's "
                        "learnable_temp is still constructed). DeepSeek suggests "
                        "margin 2.0-3.0 (gate is 2.0); 2.5 is the default-ish pick.")
    p.add_argument("--hard-negative", action="store_true",
                   help="Phase 1f-5: with --margin-loss >0, restrict the hinge to "
                        "the HARDEST filler (max filler logit) per gold instead of "
                        "all gold x filler pairs -- focuses the gradient on the "
                        "most-violated pair (DeepSeek's 'hardest negatives from "
                        "the ring'). No-op when --margin-loss is 0 (byte-"
                        "identical).")
    p.add_argument("--live-eval-sessions", default=",".join(DEFAULT_LIVE_EVAL_SESSION_IDS),
                   help="comma-separated Onyx session UUIDs to force OUT of train "
                        "and into a dedicated live-eval bucket every seed (D0.4a; "
                        "default = the 2 live-transcript sessions). Empty string "
                        "disables -> byte-identical to pre-#47.")
    p.add_argument("--code-only-gold", action="store_true",
                   help="Stage 0 diagnostic: filter train/val/live-eval to records "
                        "whose GOLD slot is a CODE doc (the code-doc retrieval task). "
                        "Default off = byte-identical. Needs --doc-store (or a cached "
                        "doc_kind_map next to the traces).")
    p.add_argument("--doc-store", default=None,
                   help="Path to the persisted doc store used to build the "
                        "doc_id -> kind map for --code-only-gold. Default = none.")
    p.add_argument("--n-doc-kinds", type=int, default=0,
                   help="Phase 1f-7 Stage 1: per-doc-kind readout channel size "
                        "(0 = byte-identical shared readout = old final.pt "
                        "strict-loads; 3 = conv/text/code routed by a doc_kinds "
                        "tensor built from slot.source_id + --doc-store). Orthogonal "
                        "to --n-slot-types (transformer conv/retrieved path "
                        "unchanged). Needs --doc-store.")
    p.add_argument("--kind-head-wd", type=float, default=0.0,
                   help="Phase 1f-7 Stage 1: separate weight decay on the per-kind "
                        "readout heads (combats the ~400-code-slot overfit; the "
                        "shared MLP body keeps --weight-decay). 0.0 = same wd as "
                        "the base group = byte-identical optimizer split.")
    p.add_argument("--per-kind-data-isolation", action="store_true",
                   help="Phase 1f-7 Stage 1 REDESIGN -- MoE on non-overlapping data: "
                        "per_kind_full readout (N INDEPENDENT 6144->128->384 "
                        "readouts, one per kind, NO shared body) + by-GOLD-kind train "
                        "routing (ALL slots of a record route to the GOLD slot's "
                        "readout, so the margin loss is within one head = well-defined "
                        "and gradient flows only into that readout = each readout "
                        "trains ONLY on its kind's gold, mirroring the Stage 0 code-"
                        "only win per kind). Implies --n-doc-kinds 3. Serve routes "
                        "per-slot (unchanged). Default off = byte-identical shared "
                        "readout (old final.pt strict-loads).")
    p.add_argument("--per-kind-bodies", action="store_true",
                   help="Phase 1f-7 Stage 2 #6 (DeepSeek ladder step 3, the architectural "
                        "robust fix): per_kind_bodies readout -- N INDEPENDENT 6144->128 "
                        "ReLU bodies (one per kind = escapes the conv-majority pull that "
                        "flattens the shared body for 4/6 #5 seeds) feeding ONE SHARED "
                        "128->384 head (the single shared final projection preserves the "
                        "cross-kind logit comparability the MoE per_kind_full FAIL proved "
                        "is required -- per-kind *heads* break the cross-kind gate; a "
                        "shared head preserves it). PER-SLOT routing (same as the "
                        "shared-body arch, NOT MoE by-gold) -- the shared head makes "
                        "per-slot gold-vs-filler logits comparable. Implies --n-doc-kinds "
                        "3. COMPOSES with --sqrt-freq-sampler (orthogonal: the sampler "
                        "rebalances which records are seen; per-kind-bodies changes the "
                        "body arch -- DeepSeek's explicit recommendation: train jointly "
                        "from scratch with the sqrt-inverse-freq no-replacement sampler). "
                        "Mutually exclusive with --per-kind-data-isolation (different "
                        "readout archs both implying n_doc_kinds=3). Requires "
                        "--drop-self-slot + --doc-store. Default off = byte-identical "
                        "shared readout (old final.pt strict-loads).")
    p.add_argument("--class-balanced-gold", action="store_true",
                   help="Phase 1f-7 Stage 2 (DeepSeek consult #1, the cross-kind-gate "
                        "fix): inverse-frequency per-record sampling on the SHARED "
                        "readout (n_doc_kinds=0). Upweights minority gold kinds (code "
                        "143 / text 229) and downweights conv (562) so each kind "
                        "contributes ~equally per epoch -- fixes the conv-majority-"
                        "dominated shared-body root cause WITHOUT per-kind "
                        "decomposition, preserving the cross-kind logit space that "
                        "MoE/Stage 1 broke (the all-turns 0.000 ceiling). Uses a "
                        "WeightedRandomSampler over train records weighted by 1/count "
                        "of the record's gold-doc-kind. Requires --drop-self-slot + "
                        "--doc-store (stamps gold_doc_kind). Default off = uniform "
                        "shuffle = byte-identical 1f-6.")
    p.add_argument("--per-kind-loss-weight", action="store_true",
                   help="Phase 1f-7 Stage 2 (DeepSeek consult #2): per-record inverse-"
                        "frequency LOSS weighting on the SHARED readout (n_doc_kinds=0). "
                        "Same per-kind BALANCE as --class-balanced-gold (each kind "
                        "contributes equally to the loss) but UNIFORM sampling -- every "
                        "record seen once/epoch, no replacement. Per-record loss weight = "
                        "1/count(gold_doc_kind), normalized to mean 1.0. RECORDED NEGATIVE "
                        "RESULT: this COLLAPSES training under AdamW (all seeds/both archs "
                        "stuck at the margin-loss ceiling, top3 ~0.27, r_pos -> 0.03 from "
                        "~epoch 5) -- per-record loss scaling fights AdamW's adaptive "
                        "moments (running-moment mixing + eps break the per-step scale "
                        "invariance). The sampler (#1, scale-1.0 steps) is the AdamW-clean "
                        "mechanism; this flag is kept for reproducibility of the negative "
                        "result, not for production. Requires --drop-self-slot + --doc-store "
                        "(stamps gold_doc_kind). Default off = uniform shuffle + unweighted "
                        "loss = byte-identical 1f-6.")
    p.add_argument("--code-hard-neg", action="store_true",
                   help="Phase 1f-7 Stage 2 #3 (DeepSeek reconsult, highest-leverage): "
                        "for code-gold records, force the hardest negative to be a CODE "
                        "filler (restrict the margin-hinge fillers to code-kind non-gold "
                        "slots). Teaches code-vs-code separation -- the exact 1f-6 code "
                        "mis-ranking failure mode (code gold and code fillers both score "
                        "high -> no gap). AdamW-CLEAN (a loss-STRUCTURE change, not "
                        "gradient weighting -- unlike --per-kind-loss-weight which "
                        "collapsed under AdamW). Leaves text-gold and conv-gold records "
                        "untouched (their hard-neg is still the hardest filler overall) "
                        "-> preserves text 30.16. Implies --hard-negative; requires "
                        "--margin-loss > 0, --drop-self-slot, --doc-store (stamps "
                        "slot_doc_kinds + gold_doc_kind). Runs on the SHARED readout "
                        "(n_doc_kinds=0, unbalanced -- no sampler). Default off = "
                        "byte-identical 1f-6.")
    p.add_argument("--no-replacement-sampler", action="store_true",
                   help="Phase 1f-7 Stage 2 #4 (DeepSeek reconsult ladder step 2): "
                        "the same class-balanced inverse-frequency gold-kind weights as "
                        "--class-balanced-gold, but WeightedRandomSampler(replacement="
                        "False, num_samples=len(train_q)) -- a WEIGHTED SHUFFLE: every "
                        "record is seen EXACTLY once/epoch (no record unseen, no minority "
                        "record ~2x/epoch), in an order weighted by inverse gold-kind "
                        "frequency (minority code/text records tend earlier each epoch). "
                        "Removes the replacement-sampling noise that drove #1's extreme "
                        "seed variance (s0/s1 inverted, s2 passed both -> 1/3) while "
                        "keeping the per-epoch balance that let s2 pass BOTH buckets + "
                        "the per-step loss scale 1.0 (AdamW-clean). DeepSeek's ranked "
                        "variance-tamer for the #1-sampler. Requires --drop-self-slot + "
                        "--doc-store (stamps gold_doc_kind). Default off = byte-identical "
                        "1f-6.")
    p.add_argument("--sqrt-freq-sampler", action="store_true",
                   help="Phase 1f-7 Stage 2 #5 (DeepSeek (a), medium-leverage fallback): "
                        "the same no-replacement weighted shuffle as --no-replacement-"
                        "sampler (#4), but with weights = sqrt(1/count(gold_doc_kind)) "
                        "instead of 1/count -- a MILDER upweight (code ~1.9x vs conv "
                        "instead of 3.6x, text ~1.2x instead of 2.4x). #4 came closest "
                        "to passing (code median +0.722, s0 passed BOTH buckets, no -19 "
                        "inversion) but the 3.6x upweight starved s1 (FLAT ~0) and left "
                        "s2 code +0.722 just under 2.0. Hypothesis: sqrt tempers the "
                        "conv-starvation that flattened s1 while keeping enough code "
                        "gradient to push s2 code over 2.0 -- DeepSeek's ranked (a) "
                        "fallback when the full-inverse-freq upweight is too aggressive. "
                        "Still AdamW-clean (per-step loss scale 1.0; WeightedRandomSampler"
                        "(replacement=False, num_samples=len(train_q)) = weighted shuffle, "
                        "each record once/epoch). Requires --drop-self-slot + --doc-store "
                        "(stamps gold_doc_kind). Default off = byte-identical 1f-6.")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    if not Path(args.onyx).exists():
        print(f"ERROR: onyx traces not found at {args.onyx}\n"
              f"  run: python scripts/generate_onyx_serve_traces.py",
              file=sys.stderr)
        return 1

    records = p41._load_serve_traces(args.onyx)
    records = p44._to_device(records, args.device)
    if len(records) < 100:
        print(f"ERROR: only {len(records)} onyx records (need >=100 for a real "
              f"session-split head-to-head)", file=sys.stderr)
        return 1
    ok = sorted(r["slots_h_raw"].shape[0] for r in records)
    print(f"onyx: {len(records)} turns (K min/med/max={ok[0]}/{ok[len(ok)//2]}/{ok[-1]}), "
          f"dim_in={records[0]['slots_h_raw'].shape[1]}", flush=True)

    # Phase 1f-7 Stage 1: per-doc-kind readout channel. Build the doc_id -> kind
    # map ONCE (shared with --code-only-gold below) and stamp a per-slot
    # ``slot_doc_kinds`` [K] long tensor on EVERY record BEFORE --drop-self-slot
    # so the keep-list slice in _drop_self_slot keeps it aligned with
    # slots_h_raw / slot_types / source_ids. conv/unmapped -> 0 (DOC_KIND_CONV),
    # text-doc -> 1 (DOC_KIND_TEXT), code-doc -> 2 (DOC_KIND_CODE). Default off
    # (n_doc_kinds=0) = byte-identical (no stamped field, shared readout).
    #
    # Phase 1f-7 Stage 2: the gold-kind mechanisms are MUTUALLY EXCLUSIVE -- pick
    # ONE per run (they stamp overlapping fields and combining double-rebalances,
    # mixes arches, or layers loss changes untested together). --per-kind-data-
    # isolation is a different ARCH (MoE n_doc_kinds=3); --class-balanced-gold /
    # --per-kind-loss-weight are data/loss variants on the SHARED readout
    # (n_doc_kinds=0); --code-hard-neg is a loss-STRUCTURE variant on the SHARED
    # readout (DeepSeek's ladder step 1, run ALONE on unbalanced 1f-6). NOTE:
    # --per-kind-bodies (Stage 2 #6) is NOT in this mutex -- it COMPOSES with
    # --sqrt-freq-sampler (orthogonal: arch vs sampling); its arch-mutex with
    # --per-kind-data-isolation is enforced separately below.
    n_rebalance = sum([args.per_kind_data_isolation, args.class_balanced_gold,
                      args.per_kind_loss_weight, args.code_hard_neg,
                      args.no_replacement_sampler, args.sqrt_freq_sampler])
    if n_rebalance > 1:
        print("ERROR: --per-kind-data-isolation / --class-balanced-gold / "
              "--per-kind-loss-weight / --code-hard-neg / --no-replacement-sampler "
              "/ --sqrt-freq-sampler are mutually exclusive (alternative variants "
              "of the gold-kind fix; pick one per run).", file=sys.stderr)
        return 1
    # Phase 1f-7 Stage 2 #6 (--per-kind-bodies) arch-mutex: it implies
    # --n-doc-kinds 3 with a DIFFERENT readout arch than --per-kind-data-isolation
    # (MoE), so combining them is contradictory. It COMPOSES with the sampler/
    # loss variants above (n_rebalance is unaffected by per_kind_bodies, so
    # --per-kind-bodies + --sqrt-freq-sampler gives n_rebalance=1 = allowed).
    if args.per_kind_bodies and args.per_kind_data_isolation:
        print("ERROR: --per-kind-bodies and --per-kind-data-isolation are mutually "
              "exclusive (both imply --n-doc-kinds 3 with DIFFERENT readout archs: "
              "per-kind bodies + shared head vs MoE per_kind_full). Pick one.",
              file=sys.stderr)
        return 1
    if args.per_kind_bodies and not args.drop_self_slot:
        print("ERROR: --per-kind-bodies requires --drop-self-slot (gold = prior "
              "doc, never the cos~1.0 self-slot, which is conv=0 and would route "
              "every record through the conv body).", file=sys.stderr)
        return 1

    # Phase 1f-7 Stage 2 #3 (--code-hard-neg): implies --hard-negative (the
    # code-restricted filler only matters under the hardest-filler hinge) and
    # requires --margin-loss > 0 (it is a margin-loss technique), --drop-self-slot
    # (gold = prior doc, never the self-slot), and --doc-store (the per-slot
    # kind map for the code-filler mask + the gold-kind stamp).
    if args.code_hard_neg:
        if args.margin_loss <= 0.0:
            print("ERROR: --code-hard-neg requires --margin-loss > 0 (it restricts "
                  "the margin-hinge fillers; the InfoNCE path has no filler mask).",
                  file=sys.stderr)
            return 1
        if not args.hard_negative:
            args.hard_negative = True
            print("  --code-hard-neg: implied --hard-negative (the code-restricted "
                  "filler is the hardest CODE slot)", flush=True)
        if not args.drop_self_slot:
            print("ERROR: --code-hard-neg requires --drop-self-slot (the gold slot "
                  "must be a prior doc, never the cos~1.0 self-slot).",
                  file=sys.stderr)
            return 1
        if not args.doc_store:
            print("ERROR: --code-hard-neg requires --doc-store (the doc_id -> kind "
                  "map for the per-slot code-filler mask + the gold-kind stamp).",
                  file=sys.stderr)
            return 1

    # Phase 1f-7 Stage 1 REDESIGN (--per-kind-data-isolation): the MoE arch
    # needs n_doc_kinds=3 (one full readout per kind); imply it when the flag is
    # on so the user doesn't have to pass both. --doc-store is still required
    # (the per-slot kinds used at serve + the gold-kind stamp both need the map).
    if args.per_kind_data_isolation and args.n_doc_kinds == 0:
        args.n_doc_kinds = 3
        print("  --per-kind-data-isolation: implied --n-doc-kinds 3 "
              "(MoE per_kind_full readout, one full 6144->128->384 per kind)",
              flush=True)
    # Phase 1f-7 Stage 2 #6 (--per-kind-bodies): per-kind bodies + shared head
    # also needs n_doc_kinds=3 (one body per kind); imply it. --doc-store is
    # required (the per-slot kinds used at train AND serve routing both need the
    # map; enforced by the n_doc_kinds>0 requires --doc-store guard below).
    if args.per_kind_bodies and args.n_doc_kinds == 0:
        args.n_doc_kinds = 3
        print("  --per-kind-bodies: implied --n-doc-kinds 3 "
              "(per_kind_bodies readout: N bodies -> shared head)",
              flush=True)
    doc_kind_map = None
    # Phase 1f-7 Stage 2 #3: --code-hard-neg needs the per-slot kind map to build
    # the code-filler mask even on the SHARED readout (n_doc_kinds=0, where the head
    # ignores slot_doc_kinds but the loss code reads it). Stamp slot_doc_kinds in
    # that case too (BEFORE --drop-self-slot, same as the n_doc_kinds>0 path).
    if args.n_doc_kinds > 0 or args.code_hard_neg:
        if not args.doc_store:
            print("ERROR: --n-doc-kinds > 0 / --code-hard-neg requires --doc-store "
                  "(the doc_id -> kind map source).", file=sys.stderr)
            return 1
        doc_kind_map = _build_doc_kind_map_cached(args.doc_store, args.onyx)
        n_stamped = 0
        for rec in records:
            rec["slot_doc_kinds"] = _slot_doc_kinds(rec, doc_kind_map)
            n_stamped += 1
        if args.n_doc_kinds > 0:
            print(f"  --n-doc-kinds {args.n_doc_kinds}: stamped slot_doc_kinds on "
                  f"{n_stamped} records (conv=0/text=1/code=2; "
                  f"--kind-head-wd {args.kind_head_wd})", flush=True)
        else:
            print(f"  --code-hard-neg: stamped slot_doc_kinds on {n_stamped} "
                  f"records (conv=0/text=1/code=2; SHARED readout n_doc_kinds=0, "
                  f"kinds used ONLY for the code-filler mask, not routed)", flush=True)

    # Phase 1d root-cause fix: the mixed-ring generator scores the ring AFTER
    # orch.query adds the current prompt, so the self-slot (cos ~= 1.0) is the
    # trivial gold 89% of the time -- misaligned with the gate (which scores
    # prior messages/episodes). When --drop-self-slot is set, remove the
    # self-slot from every ring and re-derive gold over the remaining prior
    # slots (single top-1-cos, or multi-positive within --multi-positive-margin
    # for near-tied prior-message/episode duplicates). Both flags off ->
    # byte-identical to the collapsed run (no re-derivation; stored labels used).
    if args.multi_positive_margin > 0.0 and not args.drop_self_slot:
        print("NOTE: --multi-positive-margin without --drop-self-slot keeps the "
              "self-slot (cos 1.0) as a positive -- does not fix the self-match "
              "root cause; ignoring. Use WITH --drop-self-slot.", file=sys.stderr)
        args.multi_positive_margin = 0.0
    if args.drop_self_slot:
        before = len(records)
        cleaned = [_drop_self_slot(r, args.multi_positive_margin)
                   for r in records]
        n_drop = sum(1 for r in records if _self_slot_idx(r) is not None)
        records = [r for r in cleaned if r is not None]
        n_lost = before - len(records)
        print(f"  --drop-self-slot: removed the self-slot from {n_drop}/{before} "
              f"records; {n_lost} records dropped (<3 slots remained), "
              f"{len(records)} kept"
              + (f" | --multi-positive-margin {args.multi_positive_margin} "
                 "(multi-positive InfoNCE gold)" if args.multi_positive_margin > 0
                 else " | single top-1-cos gold"), flush=True)

    # Phase 1f-7 Stage 1 REDESIGN + Stage 2: stamp the GOLD slot's doc-kind index
    # on every record. Used by SIX mechanisms that all need the gold's kind:
    # - --per-kind-data-isolation: by-GOLD-kind train routing (MoE per_kind_full)
    #   = within-one-head margin loss = non-overlapping data per kind.
    # - --class-balanced-gold: inverse-frequency per-record SAMPLING weights
    #   (upweight minority code/text gold, downweight conv) on the SHARED readout
    #   = fixes the conv-majority-dominated shared-body root cause WITHOUT per-kind
    #   decomposition, preserving the cross-kind logit space (DeepSeek #1).
    # - --per-kind-loss-weight: per-record LOSS weights (DeepSeek #2, refuted).
    # - --code-hard-neg: identify CODE-gold records whose margin-hinge fillers are
    #   restricted to code slots (DeepSeek #3, AdamW-clean).
    # - --no-replacement-sampler: same inverse-freq weights as #1 but a weighted
    #   SHUFFLE (replacement=False); DeepSeek ladder step 2 variance-tamer (#4).
    # - --sqrt-freq-sampler: same weighted shuffle as #4 but sqrt(1/count) weights
    #   = MILDER upweight; DeepSeek (a) medium-leverage fallback (#5).
    # Gold = labels.argmax() -> source_ids[idx] -> _gold_doc_kind; mapped
    # code->2 / text->1 / conv/None->0. MUST run AFTER --drop-self-slot (gold =
    # prior doc, never the cos~1.0 self-slot, which is conv=0 and would route/
    # weight every record as conv). doc_kind_map built from --doc-store. Default
    # off = no stamped field = byte-identical.
    if args.per_kind_data_isolation or args.class_balanced_gold \
            or args.per_kind_loss_weight or args.code_hard_neg \
            or args.no_replacement_sampler or args.sqrt_freq_sampler:
        if not args.drop_self_slot:
            print("ERROR: --per-kind-data-isolation/--class-balanced-gold/"
                  "--per-kind-loss-weight/--code-hard-neg/--no-replacement-sampler"
                  "/--sqrt-freq-sampler require --drop-self-slot (the gold slot "
                  "must be a prior doc, never the cos~1.0 self-slot, which is "
                  "conv=0 and would route/weight every record as conv).",
                  file=sys.stderr)
            return 1
        if not args.doc_store:
            print("ERROR: --per-kind-data-isolation/--class-balanced-gold/"
                  "--per-kind-loss-weight/--code-hard-neg/--no-replacement-sampler"
                  "/--sqrt-freq-sampler require --doc-store (the doc_id -> kind "
                  "map for the gold-kind stamp).", file=sys.stderr)
            return 1
        if doc_kind_map is None:
            doc_kind_map = _build_doc_kind_map_cached(args.doc_store, args.onyx)
        gold_counts = {DOC_KIND_CONV: 0, DOC_KIND_TEXT: 0, DOC_KIND_CODE: 0}
        for rec in records:
            gk_str = _gold_doc_kind(rec, doc_kind_map)
            if gk_str == "code":
                gk = DOC_KIND_CODE
            elif gk_str == "text":
                gk = DOC_KIND_TEXT
            else:
                gk = DOC_KIND_CONV
            rec["gold_doc_kind"] = gk
            gold_counts[gk] += 1
        if args.per_kind_data_isolation:
            tag = "--per-kind-data-isolation: by-gold routing"
            suffix = (" (each readout trains ONLY on its kind's gold = "
                      "non-overlapping)")
        elif args.class_balanced_gold:
            tag = "--class-balanced-gold: inverse-freq sampling"
            suffix = (" (upweight code/text, downweight conv -> shared-readout "
                      "fix)")
        elif args.per_kind_loss_weight:
            tag = "--per-kind-loss-weight: inverse-freq loss weight"
            suffix = (" (uniform sampling, per-record loss weight -> shared-"
                      "readout fix)")
        elif args.code_hard_neg:
            tag = "--code-hard-neg: code-restricted hard-neg"
            suffix = (" (code-gold records: hardest filler forced to a CODE slot "
                      "-> code-vs-code separation, AdamW-clean)")
        elif args.sqrt_freq_sampler:
            tag = "--sqrt-freq-sampler: milder weighted-shuffle sampling"
            suffix = (" (sqrt-inverse-freq weights, replacement=False -> each "
                      "record once/epoch, milder upweight than #4, AdamW-clean)")
        else:
            tag = "--no-replacement-sampler: weighted-shuffle sampling"
            suffix = (" (inverse-freq weights, replacement=False -> each record "
                      "seen once/epoch, variance-tamer for #1, AdamW-clean)")
        print(f"  {tag}: stamped gold_doc_kind on {len(records)} records -> "
              f"conv={gold_counts[DOC_KIND_CONV]} / text={gold_counts[DOC_KIND_TEXT]} "
              f"/ code={gold_counts[DOC_KIND_CODE]}{suffix}", flush=True)
    # Stage 0 diagnostic: filter to records whose GOLD slot is a CODE doc. Gold
    # is the argmax-labels prior slot, so this MUST run AFTER --drop-self-slot
    # (the self-slot is never a doc) and requires --drop-self-slot. The doc_id ->
    # kind map is built from the persisted doc store (cached to JSON next to the
    # traces; the mapping is stable for a frozen corpus). Reuses the map built
    # above for --n-doc-kinds (if any). Default off = byte-identical to the 1f-6
    # retrain.
    if args.code_only_gold:
        if not args.drop_self_slot:
            print("ERROR: --code-only-gold requires --drop-self-slot (gold is a "
                  "prior doc slot, never the self-slot).", file=sys.stderr)
            return 1
        if doc_kind_map is None:
            doc_kind_map = _build_doc_kind_map_cached(args.doc_store, args.onyx)
        before = len(records)
        records = _filter_code_gold(records, doc_kind_map)
        if len(records) < 100:
            print(f"ERROR: only {len(records)} code-gold records after filter "
                  f"(need >=100 for a real session-split); aborting.",
                  file=sys.stderr)
            return 1
        if before != len(records):
            print(f"  --code-only-gold: {before} -> {len(records)} records", flush=True)
        # S0.4 well-posedness / h-norm guard: is the gold code doc even
        # distinguishable from fillers by the bge embedding? Near-zero -> the task
        # is ill-posed at the embedding layer (a z_logit FAIL says nothing about
        # h); healthy positive + a z_logit FAIL points at h/readout.
        gcb = _gold_cos_baseline(records)
        if gcb["median"] is not None:
            print(f"  --code-only-gold gold-cos gap (gold-filler, bge): median="
                  f"{gcb['median']:+.4f} over {gcb['n']} records", flush=True)

    hidden = {"linear": None, "mlp64": 64, "mlp128": 128}[args.readout]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ckpt_root = Path(args.ckpt_root)
    live_eval_ids = frozenset(
        s.strip() for s in args.live_eval_sessions.split(",") if s.strip())
    if live_eval_ids:
        present = {_session_of(r) for r in records}
        missing = [s for s in live_eval_ids if s not in present]
        if missing:
            print(f"NOTE: --live-eval-sessions not in traces (dropped): {missing}",
                  file=sys.stderr)
        print(f"live-eval hold-out: {sorted(live_eval_ids)} ({len(live_eval_ids)} "
              f"sessions forced OUT of train every seed -> genuinely held-out)",
              flush=True)

    # Session split is seed-dependent (which sessions are held out varies) --
    # re-split per seed so each seed sees a fresh generalization test. The
    # live-eval sessions are held OUT of train for EVERY seed (D0.4a).
    per_arch = {"bilinear": [], "transformer": []}
    for seed in seeds:
        train, val, live_eval, val_sessions, live_sessions = _session_split(
            records, args.val_fraction, seed, live_eval_ids)
        # De-wonk: assert the live-eval sessions never leaked into train.
        train_sessions = {_session_of(r) for r in train}
        leaked = [s for s in live_sessions if s in train_sessions]
        assert not leaked, f"live-eval session(s) leaked into train: {leaked}"
        print(f"\n=== seed {seed}: {len(train)} train / {len(val)} held-out turns "
              f"({len(val_sessions)} unseen sessions) / {len(live_eval)} live-eval "
              f"turns ({len(live_sessions)} live sessions) ===", flush=True)
        for arch in ("bilinear", "transformer"):
            per_arch[arch].append(_run_arch(
                arch, train, val, live_eval, hidden, args.temperature,
                args.weight_decay, args.epochs, args.lr, args.accum_steps, seed,
                args.device, ckpt_root, n_slot_types=args.n_slot_types,
                learnable_temp=args.learnable_temp, dropout=args.dropout,
                label_smoothing=args.label_smoothing,
                cosine_schedule=args.cosine_schedule,
                select_ckpt=args.select_ckpt,
                margin_loss=args.margin_loss,
                hard_negative=args.hard_negative,
                n_doc_kinds=args.n_doc_kinds,
                kind_head_wd=args.kind_head_wd,
                per_kind_full=args.per_kind_data_isolation,
                per_kind_bodies=args.per_kind_bodies,
                class_balanced_gold=args.class_balanced_gold,
                per_kind_loss_weight=args.per_kind_loss_weight,
                code_hard_neg=args.code_hard_neg,
                no_replacement_sampler=args.no_replacement_sampler,
                sqrt_freq_sampler=args.sqrt_freq_sampler))

    # ── Phase 1d ensemble (logit-avg of the per-seed selected ckpts) ──
    # The live_eval bucket is the same 2 live-transcript sessions held OUT of
    # EVERY seed's train (seed-independent), so it is a CLEAN held-out for the
    # cross-seed ensemble (the per-seed random held-out val is NOT -- a seed-0
    # head saw seed-N's val sessions in train). Eval the ensemble ONLY on
    # live_eval + report; the per-arch per-seed held-out numbers above stand.
    ensemble = {}
    if args.ensemble:
        if len(seeds) < 2:
            print(f"  [--ensemble] need >=2 seeds to ensemble; have {len(seeds)} "
                  f"-> ensemble skipped (per-seed numbers above stand).",
                  flush=True)
        for arch in ("bilinear", "transformer"):
            rows = per_arch[arch]
            heads = [r["head"] for r in rows if r.get("head") is not None]
            if len(heads) < 2:
                continue
            ens = _EnsembleHead(heads).to(args.device).eval()
            # ``live_eval`` is the last seed's bucket (identical across seeds).
            ens_le = (p41._zr_and_logit_gaps(ens, live_eval, args.device)
                      if live_eval else None)
            ensemble[arch] = {"n_members": len(heads),
                             "live_eval": ens_le,
                             "z_logit_median": (ens_le["z_logit"]["median"]
                                                if ens_le else None)}
            if ens_le and ens_le["z_logit"]["median"] is not None:
                m = ens_le["z_logit"]["median"]
                print(f"  [{arch} ENSEMBLE x{len(heads)} {args.select_ckpt}] "
                      f"LIVE-EVAL z_logit={m:.3f} "
                      f"({'PASS' if m >= ZLOGIT_GATE else 'fail'}, "
                      f"n={ens_le['z_logit']['n_eligible']})", flush=True)

    # ── aggregate + DeepSeek decision rule ──
    def _med(rows, key, sub):
        vals = [r[key][sub]["median"] for r in rows
                if r[key][sub]["median"] is not None]
        return statistics.median(vals) if vals else None

    def _npass(rows, key, sub):
        vals = [r[key][sub]["median"] for r in rows
                if r[key][sub]["median"] is not None]
        return sum(1 for m in vals if m is not None and m >= ZLOGIT_GATE)

    a_held = _med(per_arch["bilinear"], "heldout", "z_logit")
    b_held = _med(per_arch["transformer"], "heldout", "z_logit")
    a_pass = _npass(per_arch["bilinear"], "heldout", "z_logit")
    b_pass = _npass(per_arch["transformer"], "heldout", "z_logit")
    a_robust = a_pass >= 2 and a_pass * 2 >= len(seeds)
    b_robust = b_pass >= 2 and b_pass * 2 >= len(seeds)

    # D0.4a: live-eval (the 2 live-transcript sessions, held OUT of train every
    # seed) = Head B on its TRAINED distribution (conversation rings), genuinely
    # held-out. PASS -> live-gate failure is H2 (content shift); FAIL -> H3
    # (overfit) is dominant/co-dominant. Only meaningful for the live-eval
    # sessions actually present (empty -> None).
    def _le_med(rows):
        vals = [r["live_eval"]["z_logit"]["median"] for r in rows
                if r["live_eval"] and r["live_eval"]["z_logit"]["median"] is not None]
        return statistics.median(vals) if vals else None

    def _le_npass(rows):
        vals = [r["live_eval"]["z_logit"]["median"] for r in rows
                if r["live_eval"] and r["live_eval"]["z_logit"]["median"] is not None]
        return sum(1 for m in vals if m is not None and m >= ZLOGIT_GATE)

    b_live = _le_med(per_arch["transformer"])
    b_live_pass = _le_npass(per_arch["transformer"])
    b_live_robust = (b_live_pass >= 2 and b_live_pass * 2 >= len(seeds))

    print("\n" + "=" * 80)
    print("VERDICT (task #45: Head A bilinear vs Head B cross-slot Transformer)")
    print("=" * 80)
    print(f"  readout={args.readout}  T={args.temperature}  wd={args.weight_decay}  "
          f"seeds={seeds}  val_fraction={args.val_fraction} (session split)  "
          f"z_logit gate={ZLOGIT_GATE}")
    print(f"  traces: {len(records)} real Onyx serve turns "
          f"(task #44 lmsys->Onyx transfer was 0.048; this is in-distribution)")
    print()
    for arch, rows in (("bilinear (A)", per_arch["bilinear"]),
                       ("transformer (B)", per_arch["transformer"])):
        for r in rows:
            ho = r["heldout"]
            print(f"  {arch} s{r['seed']} ({r['select_ckpt']} ep {r['scored_epoch']}, "
                  f"train_top3={r['train_top3']:.3f}): "
                  f"HELD-OUT z_logit={ho['z_logit']['median']:.3f} "
                  f"({'PASS' if ho['z_logit']['median'] and ho['z_logit']['median']>=ZLOGIT_GATE else 'fail'})  "
                  f"z_r={ho['z_r']['median']:.4f}")
        hm = _med(rows, "heldout", "z_logit")
        am = _med(rows, "allturns", "z_logit")
        n = _npass(rows, "heldout", "z_logit")
        print(f"  -> {arch}: held-out z_logit median={hm if hm is not None else 'n/a':>5} "
              f"({n}/{len(seeds)} pass)  all-turns={am if am is not None else 'n/a':>5}")
    print()
    print(f"  Head A (bilinear):    held-out z_logit {a_held if a_held is not None else 'n/a'} "
          f"-> {a_pass}/{len(seeds)} pass -> {'ROBUST PASS' if a_robust else 'NOT robust'}")
    print(f"  Head B (transformer): held-out z_logit {b_held if b_held is not None else 'n/a'} "
          f"-> {b_pass}/{len(seeds)} pass -> {'ROBUST PASS' if b_robust else 'NOT robust'}")
    if b_live is not None:
        print(f"  Head B LIVE-EVAL (held-out live transcripts, conv ring): z_logit "
              f"{b_live:.3f} -> {b_live_pass}/{len(seeds)} pass -> "
              f"{'ROBUST PASS' if b_live_robust else 'NOT robust'}")
    print()
    print("  DECISION RULE (DeepSeek):")
    if a_robust:
        print("  -> SHIP THE BILINEAR (Head A). The contrastive loss on real Onyx")
        print("     clears the 2.0 gate held-out -> the loss was the only blocker;")
        print("     the flat-readout z_i bilinear IS the ship arch. NEXT: re-run")
        print("     the live SERVE gate (probe_strm_selectivity_real.py wired in).")
    elif b_robust:
        print("  -> CROSS-SLOT TRANSFORMER IS THE LEVER (Head B). The pointwise")
        print("     bilinear (Head A) fails the gate but cross-slot attention clears")
        print("     it -> DeepSeek's relative-scoring mechanism was right: attention")
        print("     escapes the pointwise margin bound. NEXT: scale Head B + wire")
        print("     into the live serve probe; tune depth/heads/temperature.")
    else:
        a_all = _med(per_arch["bilinear"], "allturns", "z_logit")
        b_all = _med(per_arch["transformer"], "allturns", "z_logit")
        print("  -> NEITHER CLEARS (option C: abandon the state-trajectory-locator).")
        print(f"     Head A held-out {a_held} / all-turns ceiling {a_all}; "
              f"Head B held-out {b_held} / all-turns ceiling {b_all}.")
        if b_all is not None and (a_all is None or b_all > a_all):
            print("     NOTE: Head B lifts the all-turns ceiling over Head A -> the")
            print("     cross-slot mechanism shows a SIGNAL even if it doesn't clear")
            print("     the 2.0 gate held-out; a larger trace set or a deeper/wider")
            print("     Transformer might clear it. Flag for user judgment (B-pilot).")
        else:
            print("     The state path has now been tested 5 ways (mean-pool, flat")
            print("     BCE, flat BCE + 55x lmsys, flat contrastive, cross-slot")
            print("     Transformer) and saturates on serve. The bge 2a head (0.889")
            print("     train) already works; accept the SSM state does not beat bge")
            print("     for relevance and stop investing in the state-trajectory lever.")
    print()
    print("  D0.4a DIAGNOSTIC (task #47): Head B on its TRAINED distribution (the")
    print(f"  live-transcript conversation rings, held OUT of train every seed):")
    if b_live is None:
        print("  -> LIVE-EVAL n/a (no live-eval sessions in traces; pass "
              "--live-eval-sessions to enable).")
    elif b_live_robust:
        print(f"  -> LIVE-EVAL PASS ({b_live_pass}/{len(seeds)}, median {b_live:.3f}). "
              "Head B generalizes on conversation rings held-out -> the live-gate")
        print("     failure is H2 (content shift: live ring = retrieved-doc slots, a")
        print("     different task). Phase 1 = full-ring retrain incl. documents.")
    else:
        print(f"  -> LIVE-EVAL FAIL ({b_live_pass}/{len(seeds)}, median {b_live:.3f}). "
              "Head B does NOT generalize on conversation rings held-out -> H3")
        print("     (overfit) is dominant/co-dominant. Phase 1 = full-ring retrain "
              "+ heavy regularization (Path C stack).")
    print("=" * 80)

    # ── Stage 0 decision (code-only bilinear diagnostic) ──
    # When --code-only-gold, the decisive question is: does the bilinear head,
    # trained ONLY on code-doc-gold records, clear the 2.0 z_logit gate on
    # HELD-OUT code-doc-gold sessions? PASS = the code query-relevant signal IS
    # in h and a shared readout is the only compromise -> Stage 1 (per-doc-kind
    # readout). FAIL = the signal is not in h (the Phase 1 backbone never saw a
    # code-vs-text signal) -> Stage 2 (backbone retrain). Exception: held-out FAIL
    # but all-turns ceiling PASS -> overfit, not absent -> Stage 1 with heavier
    # regularization. The gold-cos baseline (printed above) guards against an
    # embedding-ill-posed false negative and an h-norm false positive.
    stage0 = None
    if args.code_only_gold:
        a_ho = a_held
        a_ceiling = _med(per_arch["bilinear"], "allturns", "z_logit")
        a_pass_s0 = a_pass
        n_seeds = len(seeds)
        robust = a_pass_s0 >= 2 and a_pass_s0 * 2 >= n_seeds
        ceiling_pass = (a_ceiling is not None and a_ceiling >= ZLOGIT_GATE)
        if robust:
            verdict = ("PASS -> Stage 1 (per-doc-kind readout): the code query-"
                       "relevant signal IS in h; the shared readout is the only "
                       "compromise.")
        elif ceiling_pass:
            verdict = ("FAIL held-out but all-turns ceiling PASS -> overfit, not "
                       "absent -> Stage 1 with heavier regularization (kind-head wd, "
                       "dropout).")
        else:
            verdict = ("FAIL -> Stage 2 (code-aware backbone retrain): the code "
                       "query-relevant signal is NOT in h (Phase 1 backbone never "
                       "saw a code-vs-text signal).")
        stage0 = {"heldout_zlogit_median": a_ho, "allturns_ceiling": a_ceiling,
                  "pass": a_pass_s0, "n_seeds": n_seeds, "robust": robust,
                  "ceiling_pass": ceiling_pass, "verdict": verdict,
                  "gold_cos_baseline": _gold_cos_baseline(records)}
        print("\n" + "=" * 80)
        print("STAGE 0 VERDICT (1f-7 code-only bilinear diagnostic)")
        print("=" * 80)
        print(f"  bilinear held-out z_logit median={a_ho if a_ho is not None else 'n/a'} "
              f"({a_pass_s0}/{n_seeds} pass)  all-turns ceiling="
              f"{a_ceiling if a_ceiling is not None else 'n/a'}")
        print(f"  -> {verdict}")
        print("=" * 80)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "readout": args.readout, "temperature": args.temperature,
            "weight_decay": args.weight_decay, "epochs": args.epochs,
            "seeds": seeds, "val_fraction": args.val_fraction,
            "n_records": len(records),
            "live_eval_sessions": sorted(live_eval_ids),
            "stage0": stage0,
            "bilinear_heldout_zlogit_median": a_held,
            "transformer_heldout_zlogit_median": b_held,
            "bilinear_pass": a_pass, "transformer_pass": b_pass,
            "bilinear_robust": a_robust, "transformer_robust": b_robust,
            "transformer_live_eval_zlogit_median": b_live,
            "transformer_live_eval_pass": b_live_pass,
            "transformer_live_eval_robust": b_live_robust,
            "phase1d": {"n_slot_types": args.n_slot_types,
                        "learnable_temp": args.learnable_temp,
                        "dropout": args.dropout,
                        "label_smoothing": args.label_smoothing,
                        "cosine_schedule": args.cosine_schedule,
                        "select_ckpt": args.select_ckpt,
                        "ensemble": args.ensemble},
            "phase1f5": {"margin_loss": args.margin_loss,
                         "hard_negative": args.hard_negative},
            "ensemble": {arch: {"n_members": e["n_members"],
                                "z_logit_median": e["z_logit_median"]}
                         for arch, e in ensemble.items()},
            "per_seed": {
                "bilinear": [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                              "select_ckpt": r["select_ckpt"],
                              "scored_epoch": r["scored_epoch"],
                              "train_top3": r["train_top3"],
                              "heldout": r["heldout"], "allturns": r["allturns"],
                              "live_eval": r["live_eval"]}
                             for r in per_arch["bilinear"]],
                "transformer": [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                                 "select_ckpt": r["select_ckpt"],
                                 "scored_epoch": r["scored_epoch"],
                                 "train_top3": r["train_top3"],
                                 "heldout": r["heldout"], "allturns": r["allturns"],
                                 "live_eval": r["live_eval"]}
                                for r in per_arch["transformer"]]},
        }, indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())