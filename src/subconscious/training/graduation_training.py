"""Supervised training for the GraduationHeadV2 (STRM Phase 2d).

Trains the v2 ``later_needed`` classifier on the LABELED replay log
(``replay_labeled.jsonl`` from ``scripts/generate_graduation_labels.py``).
Each replay record is one WM ring slot at one turn: the v2 head's three
features (the pooled WM state ``state_t_pooled`` 1536-d, the slot readout
``slot_y_t`` 256-d, the ``llm_signal`` one-hot 5-d) are all PRECOMPUTED in the
log, so training needs NO backbone and NO embedder -- it reads JSONL + trains
the MLP, mirroring how the 2a relevance trainer reads ``traces.pt`` (no
backbone at train time). The label is the ``later_needed`` 0/1 the label
generator filled (null labels are dropped).

Gate -- the v2 head must BEAT the v1 ``integral(r_i dt)`` proxy on the same
held-out slots. The v1 proxy is parameter-free: a slot's v1 score is the
cumulative sum of its ``r_i`` stream up to the turn (the integral so far, the
serve semantics -- when a slot is about to be evicted, ``graduation_score``
integrates its relevance so far). Both v2 (head ``predict``) and v1
(cumulative ``r_i``) produce a per-record score; the gate is that the v2
AUC (rank-based, threshold-free) beats the v1 AUC AND clears a chance floor
(``gate_auc_min``), with enough val slots (``min_val_n``). A Wilson 95% CI on
the v2 best-F1 recall is reported in the log for honesty (the gate itself is
AUC so a small-n snapshot recall does not flip the decision).

Checkpoint: ``{"head": state_dict, "state_dim_pooled": int, "slot_dim": int,
"llm_signal_dim": int, "hidden_dim": int, "v2_auc": float, "v1_auc": float,
"go": bool, "epoch": int}`` + ``train_log.json``. The training RUN is deferred
until ``replay_labeled.jsonl`` has enough labeled slots; the CODE here +
``scripts/train_graduation_head.py`` + the synthetic tests land now.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..graduation_head import (
    GraduationHeadV2, LLM_SIGNAL_DIM, SLOT_DIM, encode_llm_signal,
)
from ..recoverability_head import STATE_DIM_POOLED


@dataclass
class GraduationTrainingConfig:
    # Supervised training of the v2 MLP classifier (no backbone, no embedder).
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    accum_steps: int = 4       # gradient accumulation (effective batch)

    # Per-slot BCE with pos_weight = n_neg/n_pos (positives -- re-recalled
    # slots -- are rare). CAPPED: an uncapped weight on a heavily-imbalanced
    # replay destabilizes the MLP (same lesson as the doc_kind class-weight
    # cap). A mild cap keeps the rare-positive signal without letting one
    # gradient dominate; the synthetic (balanced) is unaffected.
    pos_weight_cap: float = 14.0

    # Gate: v2 AUC must beat v1 AUC AND clear the chance floor, with enough
    # val slots for the AUC to be meaningful.
    gate_auc_min: float = 0.5
    min_val_n: int = 8

    # v1 proxy dt (the relevance stream's time-step). The v1 score is
    # cumulative sum(r_i * dt); dt=1.0 (the replay default, one r_i per turn).
    v1_dt: float = 1.0

    # Splits + IO.
    val_fraction: float = 0.2
    seed: int = 0
    dtype: str = "float32"
    device: str = "auto"
    checkpoint_dir: str = "data/training/strm_graduation"
    replay_path: str = "data/training/strm_graduation/replay_labeled.jsonl"


# ── data ──

def load_replay_labeled(path: str) -> list[dict]:
    """Load replay_labeled.jsonl -> list of records, DROPPING null labels.

    Each record is one ring slot at one turn (see the orchestrator's
    ``_write_graduation_replay`` + ``scripts/generate_graduation_labels.py``).
    A record with ``later_needed`` not a bool (null / unlabelable, e.g. a
    None-source_id slot the labeler could not match) is dropped -- the v2
    head has no target for it. Also drops records missing any of the three
    feature fields (a malformed record would silently feed zeros). Reports
    counts.
    """
    out: list[dict] = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if not isinstance(rec, dict):
                dropped += 1
                continue
            if not isinstance(rec.get("later_needed"), bool):
                dropped += 1
                continue
            if ("slot_y_t" not in rec or "state_t_pooled" not in rec
                    or "llm_signal" not in rec
                    or "r_i" not in rec or "turn_id" not in rec
                    or "session_id" not in rec or "source_id" not in rec):
                dropped += 1
                continue
            out.append(rec)
    if dropped:
        print(f"  load_replay_labeled: dropped {dropped} unparseable/null/"
              f"malformed records")
    return out


def _wilson_ci95(p: float, n: int) -> list[float]:
    """Wilson score 95% interval for a binomial proportion.

    Honest small-n CI (mirrors ``doc_kind_training._wilson_ci95``). Returns
    ``[low, high]``; ``[0.0, 1.0]`` when ``n <= 0``.
    """
    if n <= 0:
        return [0.0, 1.0]
    z = 1.96
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based (Mann-Whitney) AUC -- threshold-free, tie-aware.

    ``scores`` are the model's per-record predictions; ``labels`` are 0/1.
    AUC = P(score(pos) > score(neg)) over all pos/neg pairs, 0.5 = chance. With
    average ranks for ties. Returns 0.5 when either class is empty (chance --
    the gate then fails on the chance floor, not a crash).
    """
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks (1-indexed, ties share the mean rank).
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0     # 1-indexed mean over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(ranks[idx] for idx in range(len(labels)) if labels[idx] == 1)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def v1_scores_per_record(records: list[dict], dt: float = 1.0) -> list[float]:
    """The v1 proxy's per-record graduation score = cumulative ``sum(r_i dt)``.

    For each ``source_id`` within a session, the v1 score at a turn is the
    cumulative sum of ``r_i * dt`` over that source_id's appearances up to AND
    including the turn (the ``integral so far`` -- the serve semantics of
    ``GraduationProxyV1`` when a slot is about to be evicted). Records whose
    ``r_i`` is null score 0.0 for that step (a slot never scored is not
    graduated). Returns one score per record, aligned with ``records``.
    """
    # group: (session_id, source_id) -> list of (turn_id, record_index)
    group: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for i, rec in enumerate(records):
        src = rec.get("source_id")
        if src is None:
            continue
        key = (rec["session_id"], str(src))
        group.setdefault(key, []).append((rec["turn_id"], i))
    out = [0.0] * len(records)
    for key, items in group.items():
        items.sort(key=lambda t: t[0])          # chronological by turn_id
        running = 0.0
        for _turn_id, idx in items:
            r = records[idx].get("r_i")
            if r is not None:
                try:
                    running += float(r) * dt
                except (TypeError, ValueError):
                    pass
            out[idx] = running
    return out


def _build_tensors(records: list[dict], device: torch.device) -> tuple:
    """Build (state_pooled, slot_y, llm_signal_onehot, labels) tensors [N, D]."""
    n = len(records)
    state = torch.zeros(n, STATE_DIM_POOLED, dtype=torch.float32, device=device)
    slot_y = torch.zeros(n, SLOT_DIM, dtype=torch.float32, device=device)
    sig = torch.zeros(n, LLM_SIGNAL_DIM, dtype=torch.float32, device=device)
    labels = torch.zeros(n, dtype=torch.float32, device=device)
    for i, rec in enumerate(records):
        state[i] = torch.tensor(rec["state_t_pooled"], dtype=torch.float32, device=device)
        slot_y[i] = torch.tensor(rec["slot_y_t"], dtype=torch.float32, device=device)
        sig[i] = encode_llm_signal(rec["llm_signal"])
        labels[i] = 1.0 if rec["later_needed"] else 0.0
    return state, slot_y, sig, labels


def _v2_scores(head: GraduationHeadV2, state: Tensor, slot_y: Tensor,
               sig: Tensor) -> list[float]:
    head.eval()
    with torch.no_grad():
        p = head.predict(state, slot_y, sig).squeeze(-1)   # [N]
    return p.detach().cpu().tolist()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _gate_score(v2_auc: float, v1_auc: float, gate_auc_min: float) -> tuple:
    """Checkpoint-selection score (higher is better): (go, v2_auc).

    ``go`` = v2 beats v1 AND v2 clears the chance floor. Then rank by v2_auc.
    Tuple comparison keeps a gate-passing epoch above a gate-failing one even
    at similar AUC (mirrors the doc_kind ``_gate_score`` lexicographic shape).
    """
    go = 1 if (v2_auc > v1_auc and v2_auc >= gate_auc_min) else 0
    return (go, v2_auc)


def fit_graduation(
    records: list[dict],
    config: Optional[GraduationTrainingConfig] = None,
    progress_cb=None,
) -> dict:
    """Train GraduationHeadV2 on the labeled replay; gate = v2 beats v1 proxy.

    Splits by ``session_id`` (no session leakage across train/val). Trains the
    MLP with class-weighted BCE (AdamW + gradient accumulation). Per epoch:
    v2 AUC (head) vs v1 AUC (cumulative ``r_i``) on the val split, the gate
    (v2 beats v1 + clears the chance floor), and a Wilson CI on v2 best-F1
    recall for honesty. Saves ``best.pt`` (gate-selected), ``final.pt``, and
    ``train_log.json``. Returns ``{"best_v2_auc", "best_v1_auc", "go", "log"}``.

    The v1 AUC is computed over the FULL record set's v1 scores restricted to
    the val split -- the v1 proxy is parameter-free (no train leakage), so
    its score is a deterministic function of the replay data.
    """
    cfg = config or GraduationTrainingConfig()
    dev = _resolve_device(cfg.device)
    if not records:
        raise RuntimeError("fit_graduation: no records (need replay_labeled.jsonl)")

    # Split by session_id (no slot leakage across sessions).
    rng = random.Random(cfg.seed)
    sessions = sorted({r["session_id"] for r in records})
    rng.shuffle(sessions)
    n_val_sess = max(1, int(len(sessions) * cfg.val_fraction)) if len(sessions) > 1 else 0
    val_sessions = set(sessions[:n_val_sess])
    train_recs = [r for r in records if r["session_id"] not in val_sessions]
    val_recs = [r for r in records if r["session_id"] in val_sessions]
    if not val_recs or len(val_recs) < cfg.min_val_n:
        # Too few val slots to trust the AUC: fall back to a simple tail split
        # (chronological) so the trainer still runs on small synthetic data.
        # No session boundary -- acceptable for the synthetic gate; the real
        # run accumulates enough sessions for the session split to take hold.
        order = sorted(range(len(records)), key=lambda i: records[i]["turn_id"])
        cut = max(cfg.min_val_n, int(len(records) * cfg.val_fraction))
        if cut >= len(records):
            cut = len(records) // 2 if len(records) >= 2 * cfg.min_val_n else 0
        val_idx = set(order[len(records) - cut:]) if cut else set()
        val_recs = [records[i] for i in sorted(val_idx)]
        train_recs = [records[i] for i in range(len(records)) if i not in val_idx]
    if not train_recs or not val_recs:
        raise RuntimeError(
            f"fit_graduation: split produced empty train ({len(train_recs)}) "
            f"or val ({len(val_recs)}) -- need >= {2 * cfg.min_val_n} labeled "
            f"records across sessions"
        )

    train_state, train_y, train_sig, train_labels = _build_tensors(train_recs, dev)
    val_state, val_y, val_sig, val_labels = _build_tensors(val_recs, dev)

    # v1 proxy scores (parameter-free; computed once over all records, then
    # sliced to val). The v1 AUC is the baseline the v2 head must beat.
    v1_all = v1_scores_per_record(records, dt=cfg.v1_dt)
    # Map v1_all back to val_recs by index: rebuild val v1 from val_recs' ids.
    # records was the full list; val_recs are elements of it -- find their
    # positions to slice v1_all correctly.
    val_id_set = {id(r) for r in val_recs}
    v1_val = [v1_all[i] for i, r in enumerate(records) if id(r) in val_id_set]
    val_label_list = [int(rec["later_needed"]) for rec in val_recs]
    v1_auc = _auc(v1_val, val_label_list)

    head = GraduationHeadV2().to(dev)
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    n_pos = float(int(train_labels.sum().item()))
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = min(pos_weight, cfg.pos_weight_cap)
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=dev)
    print(f"  train: {len(train_recs)} slots ({int(n_pos)} pos / {int(n_neg)} neg), "
          f"pos_weight={pos_weight:.2f} (cap {cfg.pos_weight_cap})", flush=True)
    print(f"  val: {len(val_recs)} slots -- v1 proxy AUC = {v1_auc:.4f}", flush=True)

    accum = max(1, cfg.accum_steps)
    n_train = len(train_recs)
    log: list[dict] = []
    best_score: tuple | None = None
    best_v2_auc = 0.0
    best_v1_auc = v1_auc
    best_epoch = -1
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        head.train()
        order = list(range(n_train))
        rng.shuffle(order)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for k, i in enumerate(order):
            s = train_state[i:i + 1]
            y = train_y[i:i + 1]
            g = train_sig[i:i + 1]
            tgt = train_labels[i:i + 1]
            logit = head.mlp(torch.cat([s, y, g], dim=1))   # [1, 1]
            loss = F.binary_cross_entropy_with_logits(
                logit.squeeze(-1), tgt, pos_weight=pw) / accum
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
        v2_scores = _v2_scores(head, val_state, val_y, val_sig)
        v2_auc = _auc(v2_scores, val_label_list)

        # Wilson CI on v2 best-F1 recall (honesty metric; the gate is AUC).
        best_f1 = 0.0
        best_f1_recall = 0.0
        n_pos_val = sum(val_label_list)
        n_neg_val = len(val_label_list) - n_pos_val
        if n_pos_val > 0 and n_neg_val > 0:
            for thr in [round(0.05 * k, 2) for k in range(1, 20)]:
                tp = sum(1 for s, y in zip(v2_scores, val_label_list)
                         if s >= thr and y == 1)
                fp = sum(1 for s, y in zip(v2_scores, val_label_list)
                         if s >= thr and y == 0)
                fn = n_pos_val - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / n_pos_val
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_recall = rec
        ci = _wilson_ci95(best_f1_recall, n_pos_val)
        score = _gate_score(v2_auc, v1_auc, cfg.gate_auc_min)
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "v2_auc": round(v2_auc, 6), "v1_auc": round(v1_auc, 6),
                    "go": bool(score[0]), "best_f1": round(best_f1, 6),
                    "best_f1_recall": round(best_f1_recall, 6),
                    "best_f1_recall_ci95": [round(c, 4) for c in ci],
                    "gate_score": list(score)})
        if progress_cb is not None:
            progress_cb(epoch, train_loss, v2_auc, v1_auc)
        else:
            print(f"  epoch {epoch}: loss={train_loss:.4f} v2_auc={v2_auc:.4f} "
                  f"v1_auc={v1_auc:.4f} go={bool(score[0])} f1={best_f1:.3f}",
                  flush=True)

        if best_score is None or score > best_score:
            best_score = score
            best_v2_auc = v2_auc
            best_v1_auc = v1_auc
            best_epoch = epoch
            torch.save({"head": head.state_dict(),
                        "state_dim_pooled": STATE_DIM_POOLED,
                        "slot_dim": SLOT_DIM,
                        "llm_signal_dim": LLM_SIGNAL_DIM,
                        "hidden_dim": head.hidden_dim,
                        "v2_auc": v2_auc, "v1_auc": v1_auc,
                        "go": bool(score[0]), "epoch": epoch},
                       ckpt_dir / "best.pt")

    torch.save({"head": head.state_dict(),
                "state_dim_pooled": STATE_DIM_POOLED,
                "slot_dim": SLOT_DIM,
                "llm_signal_dim": LLM_SIGNAL_DIM,
                "hidden_dim": head.hidden_dim,
                "v2_auc": best_v2_auc, "v1_auc": best_v1_auc,
                "go": bool(best_score[0]) if best_score else False,
                "epoch": cfg.epochs - 1},
               ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_v2_auc": best_v2_auc, "best_v1_auc": best_v1_auc,
                   "go": bool(best_score[0]) if best_score else False,
                   "best_epoch": best_epoch,
                   "best_gate_score": list(best_score) if best_score else None,
                   "n_train": n_train, "n_val": len(val_recs),
                   "v1_dt": cfg.v1_dt, "config": cfg.__dict__,
                   "log": log}, f, indent=2)
    go = bool(best_score[0]) if best_score else False
    print(f"\n  BEST epoch={best_epoch} v2_auc={best_v2_auc:.4f} "
          f"v1_auc={best_v1_auc:.4f} go={go}", flush=True)
    return {"best_v2_auc": best_v2_auc, "best_v1_auc": best_v1_auc,
            "go": go, "log": log, "best_epoch": best_epoch,
            "best_gate_score": best_score}


def evaluate_v1_vs_v2(
    records: list[dict], head: GraduationHeadV2, dt: float = 1.0,
) -> dict:
    """Standalone AUC comparison (no training) -- the gate on a loaded head.

    Scores every record with the v2 head AND the v1 proxy (cumulative ``r_i``)
    and returns ``{"v2_auc", "v1_auc", "go"}``. Used by tests / a future
    smoke-check that loads a trained ``best.pt`` and re-confirms the gate on
    a held-out replay slice without retraining.
    """
    dev = next(head.parameters()).device
    state, slot_y, sig, _labels = _build_tensors(records, dev)
    labels = [1 if r["later_needed"] else 0 for r in records]
    v2 = _v2_scores(head, state, slot_y, sig)
    v1 = v1_scores_per_record(records, dt=dt)
    v2_auc = _auc(v2, labels)
    v1_auc = _auc(v1, labels)
    return {"v2_auc": v2_auc, "v1_auc": v1_auc,
            "go": bool(v2_auc > v1_auc and v2_auc >= 0.5)}