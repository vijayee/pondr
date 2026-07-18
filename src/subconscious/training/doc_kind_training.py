"""Supervised training for the DocKindHead (Phase 3c Sec 7.11 deferred step).

Mirrors ``routing_training.train_retrieval_gate_supervised``: freeze the shared
backbone (Phase 2a weights), train the instance-owned params (input/output
projections + LoRA, state_lora, decomposed gate) and the 5-class classifier head
on the zero-shot doc-kind labels Sec 7.11 wrote to ``content/doc/{doc_id}/
doc_kind``. CE loss, class-weighted so the head can't win by predicting only the
majority class (``other`` / ``decision_update`` dominate a typical store).

Each training example is one DOCUMENT -- a variable-length sequence of section
embeddings. The head processes one doc at a time (``reset_state(1)`` + an inject
loop + pool), so the "batch" is inherently 1; we step the optimizer per doc (SGD
with class-weighted CE). Section embeddings are computed once per doc up front
(the embedder is the expensive part) and cached by doc index.

The checkpoint is ``{"head": state_dict, "labels": DocKindHead.LABELS,
"val_accuracy": float, "epoch": int, "feat_dim": int}`` + a ``train_log.json``
(mirrors the gate trainer's save; ``feat_dim`` is the Phase 4 temporal-feature
width, 0 for a feat-less head). ``load_doc_kind_head`` validates the persisted
label order and reads ``feat_dim`` to widen the Linear on load.
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

from ...ingestion.doc_kind import TEMPORAL_FEAT_DIM, extract_temporal_features
from ..backbone import JGSBackbone
from ..doc_kind_head import DocKindHead
from .routing_training import _resolve_device, _resolve_dtype


@dataclass
class DocKindHeadTrainingConfig:
    # Architecture is fixed by INSTANCE_CONFIGS["doc_kind"] + BackboneConfig.

    # Supervised training.
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    accum_steps: int = 16   # gradient accumulation over N docs (effective batch)

    # Severity-weighted loss (Phase 2): an EXTRA penalty on the unsafe confusion
    # direction (snapshot truth -> decision_update prediction). That confusion
    # bypasses the complementary-temporal guard and wrong-supersedes, so it is
    # trained away from harder than other mislabels. The reverse direction
    # (decision_update -> snapshot) only triggers an extra ask_user (annoying, not
    # unsafe), so it stays on the base CE term. 0.0 recovers the plain class-
    # weighted CE (A/B baseline). Default 5.0 ~ the class-weight cap (a load-
    # bearing knob, not a switch: too small and the 6/13 confusion persists; too
    # large and the head over-foregrounds snapshot vs decision_update and the
    # decision_update recall collapses -- the ship gate checks BOTH guard
    # classes, so the two are coupled).
    unsafe_confusion_penalty: float = 5.0

    # Phase 4: concatenate a doc-level temporal feature vector with the pooled
    # embedding before the head (attacks the mean-pool blind spot -- the pool
    # discards which section carries the date that distinguishes a snapshot "as
    # of T" from a decision "made on T"). When True the trainer (a) expects the
    # head to have been constructed with ``feat_dim == TEMPORAL_FEAT_DIM``, (b)
    # computes ``extract_temporal_features`` per doc, caches them, and passes
    # them into ``forward`` in BOTH the train loop and ``evaluate_doc_kind_per_class``
    # (else reported val != served behavior -- skew), and (c) persists
    # ``feat_dim`` in the checkpoint so the loader widens the Linear on load.
    # The A/B is the arch change itself (feature-on vs feature-off); default
    # False keeps the original head for the Phase 3 baseline.
    temporal_feature: bool = False

    # Phase 5: attention-over-sections readout. When True the head's section
    # reduction is a learned additive attention (vs the equal-weight mean-pool),
    # letting it FIND the date-bearing section instead of averaging it away --
    # attacks root cause #3 (decision_update separability ceiling under
    # mean-pool + frozen backbone), which more data (v3), cleaner labels (v4),
    # and the severity loss all failed to break. ORTHOGONAL to
    # ``temporal_feature`` (feat adds a doc-level regex signal; attention finds
    # the section) -- the two compose. When True the trainer (a) expects the
    # head to have been constructed with ``attention_readout=True``, and (b)
    # persists ``attention`` in the checkpoint so the loader builds the
    # attention modules on load. Default False keeps the mean-pool head for A/B.
    attention_readout: bool = False

    # Hardware / IO.
    dtype: str = "float32"   # gate training is fp32-only; same here
    device: str = "auto"
    val_fraction: float = 0.2
    seed: int = 0
    embedder_source: str = "on-demand"   # "stub" for offline tests

    backbone_path: str = (
        "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
    )
    checkpoint_dir: str = "data/training/doc_kind_head"
    pairs_path: str = "data/training/doc_kind_head/pairs.jsonl"


# ── data ──

def load_doc_kind_pairs(path: str) -> list[dict]:
    """Load doc-kind training pairs from JSONL.

    Each record is ``{"doc_id", "section_texts": [str, ...], "label": str}``.
    Drops records that fail to parse OR whose ``label`` is not in
    ``DocKindHead.LABELS`` OR whose ``section_texts`` is empty/non-list -- a
    malformed record would silently degrade to a wrong label, so drop it
    (honest, mirrors ``load_routing_pairs``). Reports a count.
    """
    records: list[dict] = []
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
            st = rec.get("section_texts")
            label = rec.get("label")
            if (isinstance(st, list) and st
                    and all(isinstance(s, str) for s in st)
                    and label in DocKindHead.LABELS):
                records.append(rec)
            else:
                dropped += 1
    if dropped:
        print(f"  load_doc_kind_pairs: dropped {dropped} unparseable/malformed records")
    return records


def export_doc_kind_pairs(store, path: str) -> int:
    """Export ``(section_texts, doc_kind)`` per doc from ``store`` to JSONL.

    Iterates ``store.default_document_ids()`` -> ``get_document(doc_id,
    load_bodies=True)``. Skips docs whose ``doc_kind`` is not in
    ``DocKindHead.LABELS`` (defensive -- should not happen post-7.11, but a
    pre-7.11 doc with the empty-key ``"other"`` default IS in the labels, so it
    is kept) and docs with no sections. Writes one JSONL record per doc. Returns
    the count written. The caller opens/closes the store.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    label_counts: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for doc_id in store.default_document_ids():
            doc = store.get_document(doc_id, load_bodies=True)
            if doc is None:
                continue
            kind = doc.doc_kind or "other"
            if kind not in DocKindHead.LABELS:
                continue
            section_texts = [
                (s.heading + "\n" + s.content) if s.heading else s.content
                for s in doc.sections
            ]
            section_texts = [s for s in section_texts if s and s.strip()]
            if not section_texts:
                continue
            f.write(json.dumps({
                "doc_id": doc_id,
                "section_texts": section_texts,
                "label": kind,
            }, ensure_ascii=False) + "\n")
            label_counts[kind] = label_counts.get(kind, 0) + 1
            n += 1
    print(f"  export_doc_kind_pairs: wrote {n} docs to {path}")
    print(f"  label distribution: {dict(sorted(label_counts.items()))}")
    return n


# ── embed + eval ──

def _embed_sections(embedder, section_texts: list[str], device: torch.device) -> list[Tensor]:
    """Embed each section text -> a list of ``[1, 384]`` float32 tensors on device."""
    vecs = embedder.encode(section_texts)
    return [
        torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        for v in vecs
    ]


def _doc_label_index(rec: dict) -> int:
    return DocKindHead.LABELS.index(rec["label"])


def evaluate_doc_kind(
    head: DocKindHead,
    val: list[dict],
    val_embs: list[list[Tensor]],
) -> float:
    """Classification accuracy on held-out docs (exact-match argmax)."""
    head.eval()
    n = len(val)
    if n == 0:
        return 0.0
    correct = 0
    with torch.no_grad():
        for rec, embs in zip(val, val_embs):
            logits = head.forward(embs)
            pred = int(logits.argmax(dim=-1).item())
            if pred == _doc_label_index(rec):
                correct += 1
    return correct / n


def _wilson_ci95(p: float, n: int) -> list[float]:
    """Wilson score 95% interval for a binomial proportion.

    Honest small-n CI for a per-class recall (a count / its class total). With
    n~13 snapshots the point estimate is nearly meaningless on its own; the
    Wilson interval stays inside [0,1] and is non-degenerate at p=0 or p=1.
    Returns ``[low, high]``; ``[0.0, 1.0]`` when ``n <= 0``.
    """
    if n <= 0:
        return [0.0, 1.0]
    z = 1.96
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def evaluate_doc_kind_per_class(
    head: DocKindHead,
    val: list[dict],
    val_embs: list[list[Tensor]],
    val_feats: Optional[list[Tensor]] = None,
) -> dict:
    """Per-class eval: the ship-decision scorecard for the DocKindHead.

    Returns accuracy + top-2 (LENIENT: true label in top-2 logits -- a fairer
    metric when a 2nd-stage guard could recover the right label), per-class
    recall, the full 5x5 confusion matrix (rows=true, cols=pred), the
    ``unsafe_cell`` (snapshot true -> decision_update pred -- the ship blocker;
    this is the confusion that bypasses the complementary-temporal guard and
    wrong-supersedes), and a Wilson 95% CI on the small snapshot class so the
    gate isn't fooled by small-n noise.

    ``val_feats`` (Phase 4): the per-doc temporal feature tensors. When supplied
    they are passed into ``forward(feat=...)`` so the scorecard reflects the
    SERVED path (a feat-trained head fed zeros would under-report -- train/serve
    skew). ``None`` -> ``forward(embs)`` (the head falls back to zeros if it has
    a feat_dim, or ignores feat if it doesn't) -- keeps the no-feature baseline
    and the monkeypatched-forward unit tests working.
    """
    head.eval()
    labels = list(DocKindHead.LABELS)
    n_classes = len(labels)
    n = len(val)
    if n == 0:
        zero = {lab: 0.0 for lab in labels}
        return {"acc": 0.0, "top2_acc": 0.0, "recall_per_class": zero,
                "confusion": [[0] * n_classes for _ in range(n_classes)],
                "unsafe_cell": 0, "snapshot_recall": 0.0,
                "decision_update_recall": 0.0, "snapshot_n": 0,
                "snapshot_recall_ci95": [0.0, 1.0]}
    confusion = [[0] * n_classes for _ in range(n_classes)]
    correct = 0
    top2 = 0
    with torch.no_grad():
        for i, (rec, embs) in enumerate(zip(val, val_embs)):
            if val_feats is not None:
                logits = head.forward(embs, feat=val_feats[i])   # [1, C]
            else:
                logits = head.forward(embs)                       # [1, C]
            true = _doc_label_index(rec)
            pred = int(logits.argmax(dim=-1).item())
            confusion[true][pred] += 1
            if pred == true:
                correct += 1
            top2_idx = logits.topk(min(2, n_classes), dim=-1).indices[0].tolist()
            if true in top2_idx:
                top2 += 1
    acc = correct / n
    top2_acc = top2 / n
    row_totals = [sum(confusion[i]) for i in range(n_classes)]
    recall_per_class = {
        labels[i]: (confusion[i][i] / row_totals[i]) if row_totals[i] else 0.0
        for i in range(n_classes)
    }
    snap = labels.index("point_in_time_snapshot")
    dec = labels.index("decision_update")
    snapshot_n = row_totals[snap]
    snapshot_recall = recall_per_class["point_in_time_snapshot"]
    return {
        "acc": acc,
        "top2_acc": top2_acc,
        "recall_per_class": recall_per_class,
        "confusion": confusion,
        "unsafe_cell": confusion[snap][dec],
        "snapshot_recall": snapshot_recall,
        "decision_update_recall": recall_per_class["decision_update"],
        "snapshot_n": snapshot_n,
        "snapshot_recall_ci95": _wilson_ci95(snapshot_recall, snapshot_n),
    }


# ── supervised training ──


# Canonical label indices for the severity loss (LABELS is a fixed order, so
# these are stable across runs; module-level so the test + the trainer agree).
_SNAP_IDX = DocKindHead.LABELS.index("point_in_time_snapshot")
_DEC_IDX = DocKindHead.LABELS.index("decision_update")


def severity_doc_kind_loss(
    logits: Tensor,
    target_idx: int,
    class_weight: Tensor,
    unsafe_confusion_penalty: float,
    accum: int = 1,
) -> Tensor:
    """Class-weighted CE + an asymmetric penalty on the unsafe confusion.

    Every class pays the class-weighted cross-entropy term. When the truth is
    ``point_in_time_snapshot`` an EXTRA term penalizes the probability mass the
    model assigns to ``decision_update`` -- the ship-blocking confusion (a
    snapshot misread as a decision_update bypasses the complementary-temporal
    guard and wrong-supersedes). The penalty is ``penalty * p(decision_update)``
    (an expected-cost term), NOT ``penalty * -logp[dec]``: ``-logp[dec]`` is LARGE
    when ``p(dec)`` is SMALL, i.e. it would reward the model for being CORRECT
    and vanish exactly when the model is wrong -- a sign trap that silently
    trains the head to be MORE unsafe. ``p(dec)`` is large when the model is
    actually confusing (the case we want to push away from) and ~0 when the
    model already predicts snapshot (no constant bias). The CE term already
    pulls ``p(snap)`` up; this selectively pushes ``p(dec)`` down extra hard.
    The reverse direction (decision_update -> snapshot) only triggers an extra
    ask_user (annoying, not unsafe), so it stays on the base CE term.
    ``unsafe_confusion_penalty=0.0`` recovers the plain class-weighted CE
    (A/B baseline). ``accum`` divides for gradient accumulation (the caller sums
    N per-doc losses and steps once).

    ``logits`` is ``[1, C]``; returns a scalar.
    """
    logp = F.log_softmax(logits, dim=-1)               # [1, C]
    ce = -class_weight[target_idx] * logp[0, target_idx]
    if unsafe_confusion_penalty > 0.0 and target_idx == _SNAP_IDX:
        # p(decision_update) = exp(logp[0, DEC]); large when the model is
        # confusing (pushes p(dec) down), ~0 when it already predicts snapshot.
        ce = ce + unsafe_confusion_penalty * torch.exp(logp[0, _DEC_IDX])
    return ce / max(1, accum)


def _gate_score(pc: dict) -> tuple:
    """Gate-aware checkpoint-selection score (higher is better).

    The ship gate is ``unsafe_cell <= 1 AND snapshot_recall >= 0.70 AND
    decision_update_recall >= 0.70 AND acc >= 0.55``. Selecting the checkpoint by
    best-val_acc alone misses gate-good epochs (a slightly-lower-acc epoch can
    have a far better unsafe_cell / guard-class balance -- the selection flaw
    Phase 3 surfaced: best-val_acc epoch had unsafe=2 while a later epoch had
    unsafe=0). This score ranks epochs the way the gate judges them:

      1. safety first -- ``unsafe_cell <= 1`` (a wrong-supersede is the ship
         blocker, so a safe epoch beats an unsafe epoch even at lower acc);
      2. then the binding guard class -- ``min(snapshot_recall,
         decision_update_recall)`` (the gate requires BOTH, so the weaker one
         is the lever -- maximizing it lifts the gate);
      3. then overall ``acc`` (the tiebreaker).

    Tuple comparison does exactly this lexicographic order. We do NOT collapse
    it to a weighted scalar -- the gate is a conjunction, and a scalar would let
    a high-acc unsafe epoch beat a safe one.
    """
    safe = 1 if pc["unsafe_cell"] <= 1 else 0
    guard_min = min(pc["snapshot_recall"], pc["decision_update_recall"])
    return (safe, guard_min, pc["acc"])


def train_doc_kind_head_supervised(
    head: DocKindHead,
    backbone: JGSBackbone,
    train_data: list[dict],
    val_data: list[dict],
    embedder,
    config: Optional[DocKindHeadTrainingConfig] = None,
    device: Optional[torch.device] = None,
    progress_cb=None,
) -> dict:
    """Train the DocKindHead supervised on the exported doc-kind pairs.

    Backbone is frozen (caller passes the already-frozen ``load_backbone``
    result; this re-freezes for safety). Trains head params only. Embeds each
    doc's sections once up front (the embedder is the expensive part) and caches
    by index. Each doc is one forward (reset + inject loop + pool + head); the
    optimizer steps per doc (SGD) with class-weighted CE. Checkpoints best +
    final as ``head.state_dict()``, writes a per-epoch ``train_log.json``.
    Returns ``{"best_val", "log", "best_per_class"}`` where ``best_per_class``
    is the scorecard (acc, per-class recall, unsafe_cell, snapshot recall CI95)
    at the best-val checkpoint.
    """
    cfg = config or DocKindHeadTrainingConfig()
    dev = device or _resolve_device(cfg.device)
    _resolve_dtype(cfg.dtype)   # float32 always; warns if a non-fp32 dtype requested

    # Freeze the shared backbone (load_backbone already froze it; belt+suspenders).
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    head = head.to(dev)          # dtype is always float32
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # Embed each doc's sections once up front (cache by index). The embedder is
    # the expensive part; the frozen backbone makes the forward cheap.
    print(f"  embedding {len(train_data)} train + {len(val_data)} val docs ...")
    train_embs = [_embed_sections(embedder, rec["section_texts"], dev) for rec in train_data]
    val_embs = [_embed_sections(embedder, rec["section_texts"], dev) for rec in val_data]

    # Phase 4: temporal feature. The head's mean-pool discards which section
    # carries the date that distinguishes a snapshot from a decision_update; the
    # feature re-injects that signal. The head must have been constructed with
    # ``feat_dim == TEMPORAL_FEAT_DIM`` when ``cfg.temporal_feature`` is on -- a
    # mismatch (feat flag on but feat-less head, or vice versa) would silently
    # feed zeros / drop the signal, so it is a hard error. Feats are cached by
    # doc index (pure regex, cheap, but run once for determinism) and passed
    # into forward in BOTH train and eval (else reported val != served -- skew).
    use_feat = bool(cfg.temporal_feature)
    if use_feat and head.feat_dim != TEMPORAL_FEAT_DIM:
        raise RuntimeError(
            f"temporal_feature=True but head.feat_dim={head.feat_dim} != "
            f"TEMPORAL_FEAT_DIM={TEMPORAL_FEAT_DIM} -- construct the head with "
            f"feat_dim=TEMPORAL_FEAT_DIM (or set temporal_feature=False)"
        )
    if not use_feat and head.feat_dim > 0:
        raise RuntimeError(
            f"temporal_feature=False but head.feat_dim={head.feat_dim} > 0 -- "
            f"a feat-trained head trained without passing feats would skew; set "
            f"temporal_feature=True or construct a feat-less head"
        )
    if use_feat:
        print(f"  temporal feature ON (feat_dim={head.feat_dim}): computing "
              f"per-doc temporal features ...")

        def _feats(recs: list[dict]) -> list[Tensor]:
            out = []
            for rec in recs:
                fv = extract_temporal_features(rec["section_texts"])
                out.append(torch.tensor(fv, dtype=torch.float32,
                                        device=dev).unsqueeze(0))   # [1, k]
            return out
        train_feats = _feats(train_data)
        val_feats = _feats(val_data)
    else:
        train_feats = None
        val_feats = None

    # Phase 5 attention-over-sections: cross-check cfg vs head construction (a
    # flag/head mismatch would silently train the wrong readout). Mirrors the
    # temporal-feature guards above.
    use_attn = bool(cfg.attention_readout)
    if use_attn and not head.attention_readout:
        raise RuntimeError(
            f"attention_readout=True but head.attention_readout=False -- "
            f"construct the head with attention_readout=True (or set "
            f"attention_readout=False)"
        )
    if not use_attn and head.attention_readout:
        raise RuntimeError(
            f"attention_readout=False but head.attention_readout=True -- a "
            f"head built with the attention readout must be trained with "
            f"attention_readout=True (set it or construct a mean-pool head)"
        )
    if use_attn:
        print(f"  attention readout ON (attn_dim={head.ATTN_DIM}): "
              f"additive attention over per-section step outputs")

    # Inverse-frequency class weights so the head can't collapse to the majority
    # class (``other`` / ``decision_update`` typically dominate a real store).
    # CAPPED at 3.0: an uncapped inverse-freq weight on a 3-example class
    # (e.g. reference -> 11.2x) destabilizes per-doc SGD -- the head thrashes
    # trying to fit the rare class and mode-collapses on the rest. A mild cap
    # keeps the minority signal without letting one class dominate the gradient.
    train_labels = [_doc_label_index(r) for r in train_data]
    n_classes = len(DocKindHead.LABELS)
    counts = [0] * n_classes
    for c in train_labels:
        counts[c] += 1
    total = max(sum(counts), 1)
    smooth = 1.0
    _CAP = 3.0
    weights = [min(total / (n_classes * (counts[c] + smooth)), _CAP) for c in range(n_classes)]
    class_weight = torch.tensor(weights, dtype=torch.float32, device=dev)
    print(f"  class weights (capped {_CAP}): {[round(float(x), 2) for x in class_weight]}")
    print(f"  train label counts: {dict(zip(DocKindHead.LABELS, counts))}")

    rng = random.Random(cfg.seed)
    n_train = len(train_data)
    log: list[dict] = []
    best_val = 0.0           # max val_acc across epochs (reported in the log)
    best_score: tuple | None = None   # gate-aware score of the checkpointed best
    best_per_class: dict | None = None
    best_epoch = -1
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Severity-weighted loss knob (Phase 2); the helper computes class-weighted
    # CE + the asymmetric unsafe-direction penalty. 0.0 -> plain CE (A/B).
    unsafe_pen_weight = float(cfg.unsafe_confusion_penalty)

    # Gradient accumulation: each doc is an independent SSM forward (per-doc
    # recurrent state, reset per doc), so we can't batch the variable-length
    # section sequence the way the gate trainer batches single-step queries.
    # Instead we accumulate loss/accum across N docs and step once -- this is
    # mini-batch SGD in effect (the head params see the MEAN gradient over N
    # docs), which is far less noisy than per-doc SGD on 582k params (per-doc
    # SGD mode-collapsed to a single class; the gate trainer uses batch=32).
    accum = max(1, cfg.accum_steps)

    for epoch in range(cfg.epochs):
        head.train()
        order = list(range(n_train))
        rng.shuffle(order)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for k, i in enumerate(order):
            if train_feats is not None:
                logits = head.forward(train_embs[i], feat=train_feats[i])   # [1, 5]
            else:
                logits = head.forward(train_embs[i])                        # [1, 5]
            target_idx = _doc_label_index(train_data[i])
            loss = severity_doc_kind_loss(
                logits, target_idx, class_weight,
                unsafe_pen_weight, accum=accum,
            )
            loss.backward()
            total_loss += float(loss.item()) * accum
            n_steps += 1
            if (k + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        # tail step for any leftover docs < accum
        if n_steps % accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / max(n_steps, 1)
        pc = evaluate_doc_kind_per_class(head, val_data, val_embs, val_feats=val_feats)
        val_acc = pc["acc"]
        if val_acc > best_val:
            best_val = val_acc
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "val_accuracy": round(val_acc, 6),
                    "unsafe_cell": pc["unsafe_cell"],
                    "snapshot_recall": round(pc["snapshot_recall"], 6),
                    "decision_update_recall": round(pc["decision_update_recall"], 6),
                    "gate_score": list(_gate_score(pc))})
        if progress_cb is not None:
            progress_cb(epoch, train_loss, val_acc)
        else:
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
                  f"unsafe={pc['unsafe_cell']} "
                  f"snap_r={pc['snapshot_recall']:.2f} "
                  f"dec_r={pc['decision_update_recall']:.2f}")

        # Gate-aware checkpoint selection (Phase 4): keep the epoch that best
        # satisfies the SHIP gate (safe first, then the binding guard class,
        # then acc) -- NOT the best-val_acc epoch (a lower-acc epoch can be far
        # more gate-safe; the selection flaw Phase 3 surfaced). best.pt is the
        # head we'd actually ship-evaluate; feat_dim + attention are persisted so
        # the loader rebuilds the head's Linear width + attention modules on load.
        score = _gate_score(pc)
        if best_score is None or score > best_score:
            best_score = score
            best_per_class = pc
            best_epoch = epoch
            torch.save({"head": head.state_dict(), "labels": list(DocKindHead.LABELS),
                        "val_accuracy": val_acc, "epoch": epoch,
                        "feat_dim": head.feat_dim,
                        "attention": head.attention_readout},
                       ckpt_dir / "best.pt")

    torch.save({"head": head.state_dict(), "labels": list(DocKindHead.LABELS),
                "val_accuracy": best_val, "epoch": cfg.epochs - 1,
                "feat_dim": head.feat_dim,
                "attention": head.attention_readout},
               ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_val": best_val, "log": log,
                   "n_train": n_train, "n_val": len(val_data),
                   "label_counts": dict(zip(DocKindHead.LABELS, counts)),
                   "best_per_class": best_per_class,
                   "best_epoch": best_epoch,
                   "best_gate_score": list(best_score) if best_score else None,
                   "config": cfg.__dict__}, f, indent=2)
    # Print the best-checkpoint scorecard so the ship gate is visible at the end.
    print(f"\n  BEST epoch scorecard (epoch={best_epoch} gate_score={best_score}):")
    print(f"    acc={best_per_class['acc']:.4f} top2={best_per_class['top2_acc']:.4f}")
    print(f"    unsafe_cell(snapshot->decision_update)={best_per_class['unsafe_cell']}")
    _recall_str = ", ".join(f"{k}:{v:.2f}" for k, v in best_per_class["recall_per_class"].items())
    print(f"    recall_per_class={{{_recall_str}}}")
    _ci = best_per_class["snapshot_recall_ci95"]
    print(f"    snapshot_n={best_per_class['snapshot_n']} "
          f"snap_recall_ci95=[{_ci[0]:.2f}, {_ci[1]:.2f}]")
    return {"best_val": best_val, "log": log, "best_per_class": best_per_class,
            "best_epoch": best_epoch, "best_gate_score": best_score}