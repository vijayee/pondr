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
                dropout: float = 0.0) -> nn.Module:
    """Construct Head A (bilinear composite) or Head B (cross-slot Transformer).

    The Phase-1b/1d knobs (``n_slot_types`` slot-type embedding, ``learnable_temp``
    logit temperature, ``dropout`` readout regularization) are Head B ONLY -- Head
    A (``CompositeZHead``) ignores them. Defaults 0/False/0.0 = the task #45 arch
    (byte-identical; the existing best.pt strict-loads via ``_load_head``)."""
    if arch == "bilinear":
        return CompositeZHead(dim_in=dim_in, hidden=hidden)
    if arch == "transformer":
        return CrossSlotTransformerZHead(dim_in=dim_in, hidden=hidden,
                                         n_slot_types=n_slot_types,
                                         learnable_temp=learnable_temp,
                                         dropout=dropout)
    raise ValueError(f"unknown arch {arch!r}")


def _load_head(arch: str, ckpt_path: str, dim_in: int, hidden: int | None,
               device: str, n_slot_types: int = 0, learnable_temp: bool = False,
               dropout: float = 0.0) -> nn.Module:
    """Reload a trained head from its checkpoint (best.pt or final.pt).

    The Phase-1b/1d knobs are read from the checkpoint (a Phase-1 head wrote
    them); a task-#45 checkpoint omits them -> the defaults 0/False/0.0 rebuild
    the byte-identical arch + strict-load. ``_build_head`` is used for Head A
    (knobs ignored) + Head B (knobs threaded into the ctor)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["head"] if isinstance(ckpt, dict) and "head" in ckpt else ckpt
    if isinstance(ckpt, dict) and "head" in ckpt:
        n_slot_types = int(ckpt.get("n_slot_types", n_slot_types))
        learnable_temp = bool(ckpt.get("learnable_temp", learnable_temp))
        dropout = float(ckpt.get("dropout", dropout))
    head = _build_head(arch, dim_in, hidden, n_slot_types=n_slot_types,
                       learnable_temp=learnable_temp, dropout=dropout)
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
    for k in ("slots_h_raw", "slots_y", "slots_z", "slots_doc_emb"):
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
                        hard_negative: bool = False) -> Tensor:
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
    """
    if gold_mask.sum() == 0:
        return logits.new_zeros(())
    g = logits[gold_mask]                       # [n_gold]
    f = logits[~gold_mask]                      # [n_fill]
    if f.numel() == 0:
        return logits.new_zeros(())
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
                margin_loss: float = 0.0, hard_negative: bool = False) -> dict:
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
                       learnable_temp=learnable_temp, dropout=dropout).to(dev)
    use_slot_types = getattr(head, "n_slot_types", 0) > 0
    n_params = sum(p.numel() for p in head.parameters())
    arch_name = (f"MLP-{hidden}" if hidden else "Linear") if arch == "bilinear" \
        else f"Transformer({'MLP-'+str(hidden) if hidden else 'Linear'} readout)"
    loss_tag = (f"MARGIN(m={margin_loss}, hard-neg={hard_negative})"
                if margin_loss > 0.0 else f"CONTRASTIVE(T={temperature})")
    print(f"\ntraining {arch} seed={seed} {loss_tag} ({arch_name} {dim_in}->384, "
          f"{n_params:,} params, wd={weight_decay}, {epochs} epochs, "
          f"slot_types={use_slot_types}, dropout={dropout}, "
          f"label_smoothing={label_smoothing}, cosine={cosine_schedule}) -> "
          f"{ckpt_dir}", flush=True)

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
            else:
                logits = head.logits(slot_y, z_flat, q).squeeze(-1)   # [K]
            gold = labels > 0
            if margin_loss > 0.0:
                loss = margin_ranking_loss(logits, gold, margin_loss,
                                           hard_negative=hard_negative) / accum
            else:
                loss = p44.contrastive_loss(logits, gold, temperature,
                                            label_smoothing=label_smoothing) / accum
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
        self.slot_dim = int(getattr(h0, "slot_dim", 0))
        self.query_dim = int(getattr(h0, "query_dim", 0))
        self.doc_dim = int(getattr(h0, "doc_dim", 0))
        self.proj_dim = int(getattr(h0, "proj_dim", 0))

    def logits(self, slot_y, slot_signal, query_emb, slot_types=None):
        acc = None
        for h in self.heads:
            if self.n_slot_types > 0 and slot_types is not None:
                lg = h.logits(slot_y, slot_signal, query_emb, slot_types=slot_types)
            else:
                lg = h.logits(slot_y, slot_signal, query_emb)
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
              margin_loss: float = 0.0, hard_negative: bool = False) -> dict:
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
                    margin_loss=margin_loss, hard_negative=hard_negative)
    ckpt_path = r["ckpt_final"] if select_ckpt == "final" else r["ckpt_best"]
    r["select_ckpt"] = select_ckpt
    r["ckpt"] = ckpt_path
    # De-wonk (Phase 1d): the SCORED ckpt's epoch, not best.pt's. ``--select-ckpt
    # final`` scores final.pt (saved at ``epochs - 1``); ``--select-ckpt best``
    # scores best.pt (saved at ``best_epoch``). The verdict print + JSON report
    # the epoch of the ckpt actually scored, so a "best ep 1" line never misleads
    # when the scored ckpt is the final-epoch one.
    r["scored_epoch"] = (epochs - 1) if select_ckpt == "final" else r["best_epoch"]
    head = _load_head(arch, str(ckpt_path), r["dim_in"], hidden, device)
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
                hard_negative=args.hard_negative))

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

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "readout": args.readout, "temperature": args.temperature,
            "weight_decay": args.weight_decay, "epochs": args.epochs,
            "seeds": seeds, "val_fraction": args.val_fraction,
            "n_records": len(records),
            "live_eval_sessions": sorted(live_eval_ids),
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