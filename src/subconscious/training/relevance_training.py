"""Supervised training for the RelevanceHead (STRM Phase 2a).

Mirrors ``doc_kind_training`` structurally (config dataclass, gate-aware
checkpoint selection, ``best.pt`` + ``final.pt`` + ``train_log.json``, fp32,
class-weighted loss with ``pos_weight`` capped at 14.0) but swaps 5-class CE for **per-slot BCE**:
each training example is one QUERY with K candidate slots, and the head scores
each slot's relevance ``r_i in [0,1]`` against the query. A slot produced from
a gold doc is positive; a slot from a sampled non-gold doc is negative.

Reads the precomputed traces from ``scripts/generate_relevance_data.py``
(``data/training/strm_relevance/traces.pt``): a list of records
``{query_emb[384], slots_y[K,256], source_ids[K], labels[K]}``. NO backbone,
NO embedder at train time -- the y_t slots + query_emb are precomputed, so the
trainer is a small MLP fitter (CPU-fine).

Split is by QUERY (80/20) -- a slot from one query never appears in both splits
(no slot leakage). Positives are rare (~1 gold per 15 slots), so the BCE loss
carries a ``pos_weight`` (n_neg/n_pos) capped at ``pos_weight_cap`` (default
14.0 -- the true 14:1 ratio; a mild 3.0 cap collapses the head to "low
relevance everywhere" on the moderate real signal, see the config docstring).

Gate (the ship decision): per-query **top-3 recall** -- for each val query,
score its K slots -> ``r_i``; top-3 recall = (# gold slots in the top-3 by
``r_i``) / (# gold slots). Aggregate the mean over val queries. GO requires
``mean_top3_recall >= gate_top3`` (0.6) AND a Wilson 95% CI lower bound on the
per-query full-recall hit rate > ``gate_wilson_low`` (0.5) -- i.e. most queries
recover ALL their gold in the top-3, not just a high mean dragged up by a few
easy queries. The Wilson interval (reused from ``doc_kind_training._wilson_ci95``)
stays inside [0,1] and is non-degenerate at the small val-query counts we run.
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

from ..relevance_head import DOC_DIM, PROJ_DIM, QUERY_DIM, SLOT_DIM, RelevanceHead
from .doc_kind_training import _wilson_ci95


@dataclass
class RelevanceTrainingConfig:
    # Trace file written by scripts/generate_relevance_data.py.
    traces_path: str = "data/training/strm_relevance/traces.pt"

    # Supervised training. The default epochs=120 / lr=1e-3 are what clears the
    # real ERAG gate: at 30 epochs / lr=3e-4 the shared-projection head is still
    # under-trained (loss decreasing ~0.009/epoch, top-3 recall climbing
    # monotonically to only 0.571 -- NO-GO). The real signal is moderate (gold
    # bge cosine ~0.9 vs negatives that are other docs with non-trivial cosine,
    # not the ~0 of the synthetic), so it needs the longer run + the larger step
    # to converge; at 120 epochs / lr=1e-3 it clears decisively (best epoch ~76,
    # top-3 recall 0.889, Wilson ci_lo 0.77). The synthetic test overrides these
    # (its signal is strong enough to clear in <20 epochs).
    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    accum_steps: int = 4    # gradient accumulation over N queries (effective
    #   batch). Small (4, not doc_kind's 16) because the 2a slice is small (~376
    #   train queries): accum=4 gives ~94 optimizer steps/epoch.

    # Per-slot BCE pos_weight (positives are ~1 gold per 15 slots). The weight
    # is n_neg/n_pos, CAPPED at pos_weight_cap. The real ERAG signal in
    # slots_doc_emb is moderate (gold bge cosine ~0.9 vs negatives that are
    # other docs with non-trivial cosine, not the ~0 of the synthetic), so a
    # mild 3.0 cap lets the 14 negatives (weight 1 each) out-pull the single
    # gold (weight 3) 14:3 and the head collapses to "low relevance everywhere"
    # (r_pos DROPS). The default 14.0 matches the true 14:1 ratio and holds the
    # gold up. (The synthetic test passes an explicit pos_weight_cap=3.0 -- its
    # signal is strong enough that the mild cap suffices, exercising the
    # machinery without the heavy class-weight.) A cap still guards against a
    # pathological split inflating the ratio.
    pos_weight_cap: float = 14.0

    # Gate (ship decision): mean per-query top-3 recall >= gate_top3 AND the
    # Wilson 95% CI lower bound on the per-query full-recall hit rate >
    # gate_wilson_low.
    gate_top3: float = 0.6
    gate_wilson_low: float = 0.5

    # Hardware / IO.
    val_fraction: float = 0.2
    seed: int = 0
    device: str = "cpu"   # the trainer is a small MLP fitter; CPU is fine
    checkpoint_dir: str = "data/training/strm_relevance"


# ── data ──

def load_relevance_traces(path: str) -> list[dict]:
    """Load the 2a relevance traces from the Step 1 generator's ``traces.pt``.

    Each record is ``{query_id, question, category, expected_doc_ids,
    query_emb[384], slots_y[K,256], slots_doc_emb[K,384], source_ids[K],
    labels[K]}``. Drops records with no slots or no labels (a record with zero
    gold would have an all-zero label vector and an undefined top-3 recall --
    the generator already filters these, but a defensive drop keeps the trainer
    honest). Requires ``slots_doc_emb`` (the raw bge doc embedding the head
    fuses alongside ``y_t`` -- without it the head has no relevance signal; see
    ``relevance_head``'s docstring); a stale trace file without it is rejected
    with a "regenerate" pointer.
    """
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(records, list):
        raise RuntimeError(
            f"relevance traces at {path} is not a list (got {type(records).__name__})"
        )
    out: list[dict] = []
    for r in records:
        slots = r.get("slots_y")
        labels = r.get("labels")
        qemb = r.get("query_emb")
        doc_emb = r.get("slots_doc_emb")
        if not isinstance(slots, Tensor) or not isinstance(labels, Tensor):
            continue
        if not isinstance(qemb, Tensor):
            continue
        if not isinstance(doc_emb, Tensor):
            raise RuntimeError(
                f"relevance trace at {path} lacks slots_doc_emb (the raw bge doc "
                f"embedding the head fuses with y_t). Regenerate with "
                f"scripts/generate_relevance_data.py --retrace."
            )
        if slots.shape[0] == 0 or labels.shape[0] == 0:
            continue
        if int(labels.sum().item()) == 0:
            continue   # no gold -> undefined top-3 recall
        out.append(r)
    if not out:
        raise RuntimeError(
            f"no usable relevance traces at {path} (need >=1 gold slot per query)"
        )
    return out


def _split_queries(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split query indices into train/val (disjoint, no slot leakage)."""
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    if n_val >= n:
        n_val = max(1, n // 5)
    val = sorted(idx[:n_val])
    train = sorted(idx[n_val:])
    return train, val


# ── eval ──

def _score_slots(head: RelevanceHead, slots_y: Tensor, slots_doc_emb: Tensor,
                query_emb: Tensor) -> Tensor:
    """``r_i`` for each slot -> ``[K]`` (sigmoid of the bilinear logit).

    Uses ``head.logits`` (the pre-sigmoid score) + sigmoid so the trainer's
    logit path and this eval path share one obvious code site.
    """
    with torch.no_grad():
        r = torch.sigmoid(head.logits(slots_y, slots_doc_emb,
                                      query_emb)).squeeze(-1)    # [K]
    return r


def evaluate_relevance(
    head: RelevanceHead,
    val: list[dict],
) -> dict:
    """Per-query top-3 recall scorecard for the RelevanceHead.

    For each val query: score its K slots -> ``r_i``; the top-3 by ``r_i`` are
    the slots the head would surface. ``top3_recall = (# gold in top-3) /
    (# gold)``. ``full_recall_hit = 1`` iff ALL gold slots are in the top-3
    (the per-query Bernoulli for the Wilson CI). Returns the mean top-3 recall,
    the hit rate, the Wilson 95% CI on the hit rate, and the mean ``r_i`` over
    positive slots (a tiebreaker -- prefer heads that score positives high).
    """
    head.eval()
    n = len(val)
    if n == 0:
        return {"mean_top3_recall": 0.0, "hit_rate": 0.0,
                "hit_ci95": [0.0, 1.0], "mean_r_positive": 0.0, "n_val": 0}
    recalls: list[float] = []
    hits = 0
    r_pos: list[float] = []
    for rec in val:
        slots = rec["slots_y"]
        doc_emb = rec["slots_doc_emb"]
        labels = rec["labels"]
        qemb = rec["query_emb"]
        r = _score_slots(head, slots, doc_emb, qemb)              # [K]
        gold_idx = (labels > 0).nonzero(as_tuple=True)[0].tolist()
        n_gold = len(gold_idx)
        if n_gold == 0:
            continue   # load_relevance_traces already dropped these; defensive
        k_top = min(3, r.shape[0])
        top_idx = set(r.topk(k_top).indices.tolist())
        n_gold_in_top = sum(1 for i in gold_idx if i in top_idx)
        recalls.append(n_gold_in_top / n_gold)
        if n_gold_in_top == n_gold:
            hits += 1
        for i in gold_idx:
            r_pos.append(float(r[i].item()))
    if not recalls:
        return {"mean_top3_recall": 0.0, "hit_rate": 0.0,
                "hit_ci95": [0.0, 1.0], "mean_r_positive": 0.0, "n_val": n}
    mean_top3 = sum(recalls) / len(recalls)
    hit_rate = hits / len(recalls)
    ci = _wilson_ci95(hit_rate, len(recalls))
    mean_r_pos = sum(r_pos) / len(r_pos) if r_pos else 0.0
    return {"mean_top3_recall": mean_top3, "hit_rate": hit_rate,
            "hit_ci95": ci, "mean_r_positive": mean_r_pos, "n_val": n}


def _gate_score(pc: dict, cfg: RelevanceTrainingConfig) -> tuple:
    """Gate-aware checkpoint-selection score (higher is better).

    The ship gate is ``mean_top3_recall >= gate_top3 AND hit_ci95[0] >
    gate_wilson_low``. Selecting by best-recall alone misses gate-good epochs
    when the Wilson CI is the binding constraint (a slightly-lower-recall epoch
    with a tighter CI can be the safer ship). Tuple comparison ranks epochs the
    way the gate judges them:

      1. gate-safe first (both conditions met) -- a gate-safe epoch beats an
         unsafe one even at lower recall;
      2. then mean top-3 recall (the primary signal);
      3. then mean ``r_i`` over positives (the tiebreaker -- prefer heads that
         score positives high, so the surfaced slots are confidently relevant).
    """
    go = 1 if (pc["mean_top3_recall"] >= cfg.gate_top3
               and pc["hit_ci95"][0] > cfg.gate_wilson_low) else 0
    return (go, pc["mean_top3_recall"], pc["mean_r_positive"])


# ── supervised training ──

def fit_relevance(
    traces: list[dict],
    config: Optional[RelevanceTrainingConfig] = None,
    progress_cb=None,
) -> dict:
    """Train the RelevanceHead supervised on the ERAG-Bench traces.

    Constructs a fresh ``RelevanceHead``, splits queries 80/20 (no slot
    leakage), trains with per-slot BCE (pos_weight capped), evaluates per-query
    top-3 recall + Wilson CI each epoch, and writes ``best.pt`` (gate-selected)
    + ``final.pt`` (last epoch) + ``train_log.json``. Returns the best-epoch
    scorecard + the GO/NO-GO decision. No backbone, no embedder -- the traces
    carry the precomputed ``slots_y`` + ``query_emb``.
    """
    cfg = config or RelevanceTrainingConfig()
    dev = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    head = RelevanceHead().to(dev)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    train_idx, val_idx = _split_queries(len(traces), cfg.val_fraction, cfg.seed)
    if len(train_idx) < 2:
        raise RuntimeError(
            f"need >=2 train queries to fit; got {len(train_idx)} "
            f"(total {len(traces)}, val_fraction {cfg.val_fraction})"
        )
    train = [traces[i] for i in train_idx]
    val = [traces[i] for i in val_idx]
    print(f"  {len(train)} train / {len(val)} val queries (split by query, no "
          f"slot leakage)", flush=True)

    # pos_weight = n_neg / n_pos over the TRAIN slots, capped at 14.0 (the true
    # 14:1 ratio). A mild 3.0 cap collapses the head on the moderate real
    # signal (r_pos drops); 14.0 holds the gold up.
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

    rng = random.Random(cfg.seed)
    log: list[dict] = []
    best_score: tuple | None = None
    best_pc: dict | None = None
    best_epoch = -1
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accum = max(1, cfg.accum_steps)

    for epoch in range(cfg.epochs):
        head.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for k, qi in enumerate(order):
            rec = train[qi]
            slots = rec["slots_y"].to(dev).to(torch.float32)        # [K, 256]
            doc_emb = rec["slots_doc_emb"].to(dev).to(torch.float32)  # [K, 384]
            qemb = rec["query_emb"].to(dev).to(torch.float32)       # [384]
            labels = rec["labels"].to(dev).to(torch.float32)        # [K]
            logits = head.logits(slots, doc_emb, qemb).squeeze(-1)  # [K]
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight, reduction="mean",
            ) / accum
            loss.backward()
            total_loss += float(loss.item()) * accum
            n_steps += 1
            if (k + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        if n_steps % accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / max(n_steps, 1)
        pc = evaluate_relevance(head, val)
        go = (pc["mean_top3_recall"] >= cfg.gate_top3
              and pc["hit_ci95"][0] > cfg.gate_wilson_low)
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "mean_top3_recall": round(pc["mean_top3_recall"], 6),
                    "hit_rate": round(pc["hit_rate"], 6),
                    "hit_ci95": [round(pc["hit_ci95"][0], 6),
                                 round(pc["hit_ci95"][1], 6)],
                    "mean_r_positive": round(pc["mean_r_positive"], 6),
                    "go": go,
                    "gate_score": list(_gate_score(pc, cfg))})
        if progress_cb is not None:
            progress_cb(epoch, train_loss, pc)
        else:
            ci = pc["hit_ci95"]
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} "
                  f"top3={pc['mean_top3_recall']:.3f} hit={pc['hit_rate']:.2f} "
                  f"ci=[{ci[0]:.2f},{ci[1]:.2f}] "
                  f"r_pos={pc['mean_r_positive']:.3f} "
                  f"{'GO' if go else 'no-go'}", flush=True)

        score = _gate_score(pc, cfg)
        if best_score is None or score > best_score:
            best_score = score
            best_pc = pc
            best_epoch = epoch
            torch.save({"head": head.state_dict(), "slot_dim": SLOT_DIM,
                        "doc_dim": DOC_DIM, "query_dim": QUERY_DIM,
                        "proj_dim": PROJ_DIM,
                        "top3_recall": pc["mean_top3_recall"],
                        "hit_rate": pc["hit_rate"],
                        "hit_ci95": pc["hit_ci95"],
                        "go": go, "epoch": epoch},
                       ckpt_dir / "best.pt")

    # final.pt = the LAST epoch (mirrors doc_kind: best.pt is gate-selected,
    # final.pt is the last epoch for inspection).
    torch.save({"head": head.state_dict(), "slot_dim": SLOT_DIM,
                "doc_dim": DOC_DIM, "query_dim": QUERY_DIM,
                "proj_dim": PROJ_DIM,
                "top3_recall": pc["mean_top3_recall"], "hit_rate": pc["hit_rate"],
                "hit_ci95": pc["hit_ci95"], "go": go, "epoch": cfg.epochs - 1},
               ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch,
                   "best_gate_score": list(best_score) if best_score else None,
                   "best_scorecard": best_pc, "log": log,
                   "n_train": len(train), "n_val": len(val),
                   "pos_weight": pw, "config": cfg.__dict__}, f, indent=2)

    go_final = (best_pc is not None and best_pc["mean_top3_recall"] >= cfg.gate_top3
                and best_pc["hit_ci95"][0] > cfg.gate_wilson_low)
    print(f"\n  BEST epoch={best_epoch} top3={best_pc['mean_top3_recall']:.3f} "
          f"hit={best_pc['hit_rate']:.2f} "
          f"ci=[{best_pc['hit_ci95'][0]:.2f},{best_pc['hit_ci95'][1]:.2f}] "
          f"-> {'GO' if go_final else 'NO-GO'}", flush=True)
    return {"best_epoch": best_epoch, "best_pc": best_pc,
            "best_gate_score": best_score, "go": go_final, "log": log}