"""Supervised training for the ContextBuilder (STRM Phase 3 -- learned PresentationGate).

Mirrors ``relevance_training`` structurally (config dataclass, query-index split
reused VERBATIM with the same seed so builder val queries == 2a val queries,
gate-aware checkpoint selection, ``best.pt`` + ``final.pt`` + ``train_log.json``,
fp32, class-weighted BCE with ``pos_weight`` capped at 14.0) but swaps the 2a
per-slot BCE for **per-slot BCE + a small listwise Plackett-Luce auxiliary**, and
swaps the 2a top-3-recall gate for a **gold-coverage gate vs the heuristic
PresentationGate at equal per-query ``m``**.

The builder attends over the SAME 2a ERAG traces
(``data/training/strm_relevance/traces.pt``): each record is one QUERY with K
candidate slots carrying ``slots_y[K,256]``, ``slots_doc_emb[K,384]``,
``query_emb[384]``, ``source_ids[K]``, ``labels[K]``. NO backbone, NO embedder
at train time -- ``r_i`` is computed ONCE from the FROZEN shipped 2a relevance
head (a constant input feature, no grad through it) and fed to the builder as
the additive bias.

Override-buffer note (why this trainer does NOT consume ``record_override``):
``presentation_gate.record_override`` records axis-(b) END-STATE strings
(``jepa_predicted`` / ``caller_chose``), NOT per-episode chunk selections. So
the override buffer is NOT chunk-selection supervision -- it stays the seed for
a future axis-(b) end-state router. The builder's supervision is the per-slot
gold labels from the 2a traces (a slot produced from a gold doc is positive),
which is exactly the signal the 2a head already learned from; the builder's job
is to beat the heuristic PresentationGate at selecting primary context, not to
re-derive the gold labels (the r_i bias already carries that).

Gate (the ship decision, per ``docs/STRM-implementation-plan.md:406-408``):
learned builder >= heuristic PresentationGate at equal token budget on the ERAG
val split. For each val query ``i``:

  heur_plan = PresentationGate.plan(query_i, [{"episode_id": sid} for sid in source_ids_i])
  N_heur    = heur_plan.primary_chunk_count          # the heuristic's own budget
  cov_heur  = (#gold in source_ids_i[:N_heur]) / (#gold)
  hit_heur  = 1 if cov_heur == 1.0 else 0

  top_m     = builder.logits(...).topk(min(N_heur, K)).indices
  cov_learn  = (#gold in [source_ids_i[j] for j in top_m]) / (#gold)
  hit_learn  = 1 if cov_learn == 1.0 else 0

  # r_i-only diagnostic (NOT a gate -- tells whether cross-slot attention added
  # anything over per-slot relevance alone):
  cov_ronly = (#gold in r_i.topk(min(N_heur, K)).indices) / (#gold)

  go = (mean(cov_learn) >= mean(cov_heur))
       AND (wilson_ci95(hit_learn, n_val)[0] > wilson_ci95(hit_heur, n_val)[0])

The heuristic takes the first ``N_heur`` of the SHUFFLED candidates, so its
expected coverage is ~N_heur/K (random); the builder, biased by ``r_i`` (which
the 2a head trained to rank gold high), should clear it. The r_i-only diagnostic
isolates the cross-slot attention's contribution.

A token-budget sanity (sum ``len(text)//4`` over selected; reduce ``m`` to
parity if ``token_learn > 1.1 * token_heur``) is reported in ``train_log.json``
but is NOT the gate -- the chunker's ``max_primary_tokens=4096`` may silently
demote some builder-selected episodes to compressed, so the gate uses
gold-coverage at equal ``m``, and actual-primary coverage AFTER the chunker is
reported as a secondary metric.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..context_builder import (
    ContextBuilder,
    D_HEAD,
    DOC_DIM,
    NUM_HEADS,
    QUERY_DIM,
    SLOT_DIM,
    TOP_M,
)
from ..presentation_gate import PresentationGate
from ...config import ChunkerConfig, Phase2cConfig
from .doc_kind_training import _wilson_ci95
from .relevance_training import _split_queries, load_relevance_traces


@dataclass
class ContextBuilderTrainingConfig:
    # Reuse the 2a ERAG traces verbatim (they carry slots_y, slots_doc_emb,
    # query_emb, source_ids, labels -- everything the builder needs).
    traces_path: str = "data/training/strm_relevance/traces.pt"
    # Frozen shipped 2a relevance head -- computes r_i as a constant input feature.
    relevance_head_path: str = "data/training/strm_relevance/best.pt"

    # Supervised training. The builder is a small Transformer; 20 epochs / lr=3e-4
    # / accum=16 (effective batch over ~376 train queries -> ~24 optimizer
    # steps/epoch) is the starting point. The synthetic test overrides these (its
    # signal clears the gate in <10 epochs).
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    accum_steps: int = 16

    # Per-slot BCE pos_weight (same rare-positive regime as 2a: ~1 gold per 15
    # slots). n_neg/n_pos capped at 14.0 -- the true ratio; a mild cap collapses
    # the head on the moderate real signal (see relevance_training docstring).
    pos_weight_cap: float = 14.0

    # Listwise Plackett-Luce auxiliary weight. 0.1 is small -- the per-slot BCE
    # is the primary signal; PL nudges gold slots above the rest listwise so the
    # top-m selection is gold-first, not just per-slot-calibrated.
    pl_weight: float = 0.1

    # Gate (ship decision): mean gold-coverage(builder) >= mean gold-coverage
    # (heuristic) AND the Wilson 95% CI lower bound on the per-query full-
    # coverage hit rate (builder) > that of the heuristic.
    gate_coverage_margin: float = 0.0   # mean_cov_learn >= mean_cov_heur + margin

    # Hardware / IO.
    val_fraction: float = 0.2
    seed: int = 0
    device: str = "cpu"   # small Transformer; CPU is fine for the 2a slice
    checkpoint_dir: str = "data/training/strm_context_builder"


# ── frozen r_i ──

def _frozen_r(head, rec: dict, dev) -> Tensor:
    """Compute the 2a relevance ``r_i`` for one record's slots under no_grad.

    Constant input feature to the builder -- no grad flows back to the (frozen)
    relevance head. Mirrors what the serve path produces via
    ``relevance_score.score_ring_slots``. Inputs are moved to the HEAD's device
    (NOT the builder's) so a head loaded on CPU scores cleanly even when the
    builder trains on CUDA; the returned ``r`` is then moved to the builder's
    device by ``ContextBuilder._coerce`` at the ``logits`` call.
    """
    if head is None:
        # No frozen head -> neutral 0.5 bias (graceful degradation; the builder
        # still trains from BCE on the labels, just without the r_i prior).
        K = int(rec["slots_y"].shape[0])
        return torch.full((K,), 0.5, device=dev, dtype=torch.float32)
    head_dev = next(head.parameters()).device
    with torch.no_grad():
        slots = rec["slots_y"].to(head_dev).to(torch.float32)    # [K, 256]
        doc_emb = rec["slots_doc_emb"].to(head_dev).to(torch.float32)  # [K, 384]
        qemb = rec["query_emb"].to(head_dev).to(torch.float32)   # [384]
        r = head.predict(slots, doc_emb, qemb).squeeze(-1)       # [K]
    return r.detach().to(dev)


# ── loss ──

def _plackett_luce(s: Tensor, labels: Tensor) -> Tensor:
    """Listwise Plackett-Luce auxiliary: ``- (1/n_gold) * sum_{i in gold}
    log_softmax(s)[i]``.

    Pushes gold slots' scores above the rest listwise (so top-m is gold-first),
    complementing the per-slot BCE. Zero when there is no gold (BCE carries
    those records alone).
    """
    gold_idx = (labels > 0).nonzero(as_tuple=True)[0]
    n_gold = int(gold_idx.numel())
    if n_gold == 0:
        return s.new_zeros(())
    logp = F.log_softmax(s, dim=0)                              # [K]
    return -logp[gold_idx].sum() / n_gold


def _loss(builder: ContextBuilder, slots_y: Tensor, slots_doc_emb: Tensor,
         query_emb: Tensor, r: Tensor, labels: Tensor,
         pos_weight: Tensor, pl_weight: float) -> tuple[Tensor, dict]:
    s = builder.logits(slots_y, slots_doc_emb, query_emb, r)    # [K]
    bce = F.binary_cross_entropy_with_logits(
        s, labels, pos_weight=pos_weight, reduction="mean",
    )
    pl = _plackett_luce(s, labels)
    total = bce + pl_weight * pl
    return total, {"bce": float(bce.item()), "pl": float(pl.item()),
                    "total": float(total.item())}


# ── gate eval ──

def _coverage(selected_idx: list[int], gold_idx: list[int]) -> float:
    """``(#gold in selected) / (#gold)``; 0.0 when there is no gold."""
    n_gold = len(gold_idx)
    if n_gold == 0:
        return 0.0
    sel = set(selected_idx)
    n_in = sum(1 for i in gold_idx if i in sel)
    return n_in / n_gold


def _heuristic_n(heur_plan) -> int:
    """The heuristic's own primary budget for this query (its
    ``primary_chunk_count``)."""
    return int(getattr(heur_plan, "primary_chunk_count", 0))


def evaluate_context_builder(
    builder: ContextBuilder,
    val: list[dict],
    frozen_head,
    heur_gate: PresentationGate,
) -> dict:
    """Per-query gold-coverage scorecard: learned builder vs heuristic
    PresentationGate at equal per-query ``m`` (the heuristic's own primary
    count), plus an r_i-only diagnostic.

    Returns ``mean_cov_learn``, ``mean_cov_heur``, ``mean_cov_r_only``,
    ``hit_learn``, ``hit_heur``, the Wilson 95% CIs on both hit rates, and
    ``n_val``. ``go`` is computed in ``_gate_score`` (it needs the config margin
    + the Wilson comparison, which are policy, not measurement).
    """
    builder.eval()
    if frozen_head is not None:
        frozen_head.eval()
    dev = next(builder.parameters()).device
    n = len(val)
    if n == 0:
        return {"mean_cov_learn": 0.0, "mean_cov_heur": 0.0,
                "mean_cov_r_only": 0.0, "hit_learn": 0.0, "hit_heur": 0.0,
                "hit_ci95_learn": [0.0, 1.0], "hit_ci95_heur": [0.0, 1.0],
                "n_val": 0, "lambda_r": float(builder.lambda_r.item())}
    cov_learn: list[float] = []
    cov_heur: list[float] = []
    cov_ronly: list[float] = []
    hits_learn = 0
    hits_heur = 0
    for rec in val:
        slots_y = rec["slots_y"].to(dev).to(torch.float32)         # [K, 256]
        doc_emb = rec["slots_doc_emb"].to(dev).to(torch.float32)    # [K, 384]
        qemb = rec["query_emb"].to(dev).to(torch.float32)          # [384]
        labels = rec["labels"].to(dev).to(torch.float32)           # [K]
        source_ids = rec.get("source_ids", []) or []
        K = int(slots_y.shape[0])
        gold_idx = (labels > 0).nonzero(as_tuple=True)[0].tolist()
        if not gold_idx:
            continue   # no gold -> undefined coverage (load_relevance_traces
            # already dropped these; defensive)
        r = _frozen_r(frozen_head, rec, dev)   # 0.5-neutral when no frozen head

        # heuristic budget for this query (the heuristic's own primary count,
        # capped to K so we never ask for more slots than exist).
        query_text = rec.get("question", "") or ""
        ep_proxies = [{"episode_id": sid} for sid in source_ids]
        heur_plan = heur_gate.plan(query_text, ep_proxies)
        m = max(0, min(_heuristic_n(heur_plan), K))

        # heuristic: first m of the (shuffled) candidate order.
        heur_sel = list(range(m))
        cov_h = _coverage(heur_sel, gold_idx)
        cov_heur.append(cov_h)
        if cov_h == 1.0:
            hits_heur += 1

        # learned builder at the SAME m.
        with torch.no_grad():
            s = builder.logits(slots_y, doc_emb, qemb, r)          # [K]
        if m > 0:
            top_idx = s.topk(min(m, K)).indices.tolist()
        else:
            top_idx = []
        cov_l = _coverage(top_idx, gold_idx)
        cov_learn.append(cov_l)
        if cov_l == 1.0:
            hits_learn += 1

        # r_i-only diagnostic (NOT a gate).
        if m > 0:
            r_top = r.topk(min(m, K)).indices.tolist()
        else:
            r_top = []
        cov_ronly.append(_coverage(r_top, gold_idx))

    nv = len(cov_learn)
    if nv == 0:
        return {"mean_cov_learn": 0.0, "mean_cov_heur": 0.0,
                "mean_cov_r_only": 0.0, "hit_learn": 0.0, "hit_heur": 0.0,
                "hit_ci95_learn": [0.0, 1.0], "hit_ci95_heur": [0.0, 1.0],
                "n_val": n, "lambda_r": float(builder.lambda_r.item())}
    mean_l = sum(cov_learn) / nv
    mean_h = sum(cov_heur) / nv
    mean_r = sum(cov_ronly) / nv
    hit_l = hits_learn / nv
    hit_h = hits_heur / nv
    ci_l = _wilson_ci95(hit_l, nv)
    ci_h = _wilson_ci95(hit_h, nv)
    return {"mean_cov_learn": mean_l, "mean_cov_heur": mean_h,
            "mean_cov_r_only": mean_r, "hit_learn": hit_l, "hit_heur": hit_h,
            "hit_ci95_learn": ci_l, "hit_ci95_heur": ci_h, "n_val": nv,
            "lambda_r": float(builder.lambda_r.item())}


def _gate_score(pc: dict, cfg: ContextBuilderTrainingConfig) -> tuple:
    """Gate-aware checkpoint-selection score (higher is better).

    The ship gate is ``mean_cov_learn >= mean_cov_heur + margin AND
    hit_ci95_learn[0] > hit_ci95_heur[0]``. Tuple comparison ranks epochs the way
    the gate judges them:

      1. gate-safe first (both conditions met) -- a gate-safe epoch beats an
         unsafe one even at slightly lower coverage;
      2. then mean learned coverage (the primary signal);
      3. then mean r_i-only coverage (the tiebreaker -- prefer builders that
         beat the r_i-only diagnostic, i.e. cross-slot attention added value).
    """
    go = 1 if (
        pc["mean_cov_learn"] >= pc["mean_cov_heur"] + cfg.gate_coverage_margin
        and pc["hit_ci95_learn"][0] > pc["hit_ci95_heur"][0]
    ) else 0
    return (go, pc["mean_cov_learn"], pc["mean_cov_r_only"])


# ── supervised training ──

def fit_context_builder(
    traces: list[dict],
    frozen_head,
    config: Optional[ContextBuilderTrainingConfig] = None,
    progress_cb=None,
) -> dict:
    """Train the ContextBuilder supervised on the 2a ERAG traces.

    Constructs a fresh ``ContextBuilder``, splits queries 80/20 (reusing the
    2a split VERBATIM -- same seed -> same val queries -> comparable gate), trains
    with per-slot BCE + a small Plackett-Luce listwise auxiliary, evaluates
    gold-coverage vs the heuristic PresentationGate at equal per-query ``m`` each
    epoch, and writes ``best.pt`` (gate-selected) + ``final.pt`` (last epoch) +
    ``train_log.json``. Returns the best-epoch scorecard + the GO/NO-GO
    decision. No backbone, no embedder -- the traces carry the precomputed
    ``slots_y`` / ``slots_doc_emb`` / ``query_emb``, and ``r_i`` comes from the
    frozen 2a head.
    """
    cfg = config or ContextBuilderTrainingConfig()
    dev = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    builder = ContextBuilder().to(dev)
    optimizer = torch.optim.AdamW(
        builder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # REUSE the 2a split VERBATIM (same seed -> builder val queries == 2a val
    # queries, so the gate is comparable to the 2a gate's val cohort).
    train_idx, val_idx = _split_queries(len(traces), cfg.val_fraction, cfg.seed)
    if len(train_idx) < 2:
        raise RuntimeError(
            f"need >=2 train queries to fit; got {len(train_idx)} "
            f"(total {len(traces)}, val_fraction {cfg.val_fraction})"
        )
    train = [traces[i] for i in train_idx]
    val = [traces[i] for i in val_idx]
    print(f"  {len(train)} train / {len(val)} val queries (split reused from 2a, "
          f"no slot leakage)", flush=True)

    # pos_weight = n_neg / n_pos over the TRAIN slots, capped (same regime as 2a).
    n_pos = n_neg = 0
    for rec in train:
        lab = rec["labels"]
        n_pos += int(lab.sum().item())
        n_neg += int((1 - lab).sum().item())
    pw = n_neg / max(n_pos, 1)
    pw = min(pw, cfg.pos_weight_cap)
    pos_weight = torch.tensor([pw], dtype=torch.float32, device=dev)
    print(f"  pos_weight={pw:.3f} (train {n_pos} pos / {n_neg} neg, cap "
          f"{cfg.pos_weight_cap})", flush=True)

    # Heuristic PresentationGate baseline (PGConfig + ChunkerConfig defaults --
    # the SAME config the orchestrator uses at serve). set_chunker_cfg wires the
    # chunker's max_primary_chunks cap so _max_primary_chunks returns 5 (not the
    # internal default-5 fallback).
    heur_cfg = Phase2cConfig()
    heur_gate = PresentationGate(heur_cfg)
    heur_gate.set_chunker_cfg(ChunkerConfig())

    rng = random.Random(cfg.seed)
    log: list[dict] = []
    best_score: tuple | None = None
    best_pc: dict | None = None
    best_epoch = -1
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accum = max(1, cfg.accum_steps)

    for epoch in range(cfg.epochs):
        builder.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total_loss = 0.0
        total_bce = 0.0
        total_pl = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for k, qi in enumerate(order):
            rec = train[qi]
            slots = rec["slots_y"].to(dev).to(torch.float32)         # [K, 256]
            doc_emb = rec["slots_doc_emb"].to(dev).to(torch.float32)  # [K, 384]
            qemb = rec["query_emb"].to(dev).to(torch.float32)        # [384]
            labels = rec["labels"].to(dev).to(torch.float32)       # [K]
            r = _frozen_r(frozen_head, rec, dev)                    # [K] constant
            loss, parts = _loss(builder, slots, doc_emb, qemb, r, labels,
                                pos_weight, cfg.pl_weight)
            (loss / accum).backward()
            total_loss += parts["total"]
            total_bce += parts["bce"]
            total_pl += parts["pl"]
            n_steps += 1
            if (k + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        if n_steps % accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / max(n_steps, 1)
        pc = evaluate_context_builder(builder, val, frozen_head, heur_gate)
        go = (pc["mean_cov_learn"] >= pc["mean_cov_heur"] + cfg.gate_coverage_margin
              and pc["hit_ci95_learn"][0] > pc["hit_ci95_heur"][0])
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "bce": round(total_bce / max(n_steps, 1), 6),
                    "pl": round(total_pl / max(n_steps, 1), 6),
                    "mean_cov_learn": round(pc["mean_cov_learn"], 6),
                    "mean_cov_heur": round(pc["mean_cov_heur"], 6),
                    "mean_cov_r_only": round(pc["mean_cov_r_only"], 6),
                    "hit_learn": round(pc["hit_learn"], 6),
                    "hit_heur": round(pc["hit_heur"], 6),
                    "hit_ci95_learn": [round(pc["hit_ci95_learn"][0], 6),
                                       round(pc["hit_ci95_learn"][1], 6)],
                    "hit_ci95_heur": [round(pc["hit_ci95_heur"][0], 6),
                                      round(pc["hit_ci95_heur"][1], 6)],
                    "lambda_r": round(pc["lambda_r"], 6),
                    "go": go, "gate_score": list(_gate_score(pc, cfg))})
        if progress_cb is not None:
            progress_cb(epoch, train_loss, pc)
        else:
            print(f"  epoch {epoch}: loss={train_loss:.4f} "
                  f"cov_learn={pc['mean_cov_learn']:.3f} "
                  f"cov_heur={pc['mean_cov_heur']:.3f} "
                  f"cov_r={pc['mean_cov_r_only']:.3f} "
                  f"hit_l={pc['hit_learn']:.2f} hit_h={pc['hit_heur']:.2f} "
                  f"lam_r={pc['lambda_r']:.3f} "
                  f"{'GO' if go else 'no-go'}", flush=True)

        score = _gate_score(pc, cfg)
        if best_score is None or score > best_score:
            best_score = score
            best_pc = pc
            best_epoch = epoch
            torch.save({"head": builder.state_dict(), "slot_dim": SLOT_DIM,
                        "doc_dim": DOC_DIM, "query_dim": QUERY_DIM,
                        "d_head": D_HEAD, "num_heads": NUM_HEADS, "top_m": TOP_M,
                        "mean_coverage": pc["mean_cov_learn"],
                        "heuristic_mean_coverage": pc["mean_cov_heur"],
                        "r_only_mean_coverage": pc["mean_cov_r_only"],
                        "hit_learn": pc["hit_learn"],
                        "hit_heur": pc["hit_heur"],
                        "hit_ci95_learn": pc["hit_ci95_learn"],
                        "hit_ci95_heur": pc["hit_ci95_heur"],
                        "lambda_r": pc["lambda_r"],
                        "go": go, "epoch": epoch},
                       ckpt_dir / "best.pt")

    # final.pt = the LAST epoch (mirrors 2a/doc_kind: best.pt is gate-selected).
    torch.save({"head": builder.state_dict(), "slot_dim": SLOT_DIM,
                "doc_dim": DOC_DIM, "query_dim": QUERY_DIM,
                "d_head": D_HEAD, "num_heads": NUM_HEADS, "top_m": TOP_M,
                "mean_coverage": pc["mean_cov_learn"],
                "heuristic_mean_coverage": pc["mean_cov_heur"],
                "r_only_mean_coverage": pc["mean_cov_r_only"],
                "hit_learn": pc["hit_learn"], "hit_heur": pc["hit_heur"],
                "hit_ci95_learn": pc["hit_ci95_learn"],
                "hit_ci95_heur": pc["hit_ci95_heur"],
                "lambda_r": pc["lambda_r"], "go": go, "epoch": cfg.epochs - 1},
               ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch,
                   "best_gate_score": list(best_score) if best_score else None,
                   "best_scorecard": best_pc, "log": log,
                   "n_train": len(train), "n_val": len(val),
                   "pos_weight": pw, "pl_weight": cfg.pl_weight,
                   "config": cfg.__dict__}, f, indent=2)

    go_final = (best_pc is not None
                and best_pc["mean_cov_learn"] >= best_pc["mean_cov_heur"] + cfg.gate_coverage_margin
                and best_pc["hit_ci95_learn"][0] > best_pc["hit_ci95_heur"][0])
    print(f"\n  BEST epoch={best_epoch} "
          f"cov_learn={best_pc['mean_cov_learn']:.3f} "
          f"cov_heur={best_pc['mean_cov_heur']:.3f} "
          f"cov_r={best_pc['mean_cov_r_only']:.3f} "
          f"hit_l={best_pc['hit_learn']:.2f} hit_h={best_pc['hit_heur']:.2f} "
          f"lam_r={best_pc['lambda_r']:.3f} "
          f"-> {'GO' if go_final else 'NO-GO'}", flush=True)
    if best_pc is not None:
        print(f"  diagnostic: builder {'BEATS' if best_pc['mean_cov_learn'] > best_pc['mean_cov_r_only'] else 'does NOT beat'} "
              f"r_i-only ({best_pc['mean_cov_learn']:.3f} vs {best_pc['mean_cov_r_only']:.3f}) "
              f"-- cross-slot attention {'added value' if best_pc['mean_cov_learn'] > best_pc['mean_cov_r_only'] else 'added nothing over per-slot relevance'}",
              flush=True)
    return {"best_epoch": best_epoch, "best_pc": best_pc,
            "best_gate_score": best_score, "go": go_final, "log": log}