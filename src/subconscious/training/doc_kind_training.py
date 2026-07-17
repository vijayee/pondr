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
"val_accuracy": float, "epoch": int}`` + a ``train_log.json`` (mirrors the gate
trainer's save). ``load_doc_kind_head`` validates the persisted label order.
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


# ── supervised training ──

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
    Returns ``{"best_val", "log"}``.
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
    best_val = 0.0
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            logits = head.forward(train_embs[i])           # [1, 5]
            target = torch.tensor([_doc_label_index(train_data[i])],
                                   dtype=torch.long, device=dev)
            loss = F.cross_entropy(logits, target, weight=class_weight) / accum
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
        val_acc = evaluate_doc_kind(head, val_data, val_embs)
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "val_accuracy": round(val_acc, 6)})
        if progress_cb is not None:
            progress_cb(epoch, train_loss, val_acc)
        else:
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save({"head": head.state_dict(), "labels": list(DocKindHead.LABELS),
                        "val_accuracy": best_val, "epoch": epoch},
                       ckpt_dir / "best.pt")

    torch.save({"head": head.state_dict(), "labels": list(DocKindHead.LABELS),
                "val_accuracy": best_val, "epoch": cfg.epochs - 1},
               ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_val": best_val, "log": log,
                   "n_train": n_train, "n_val": len(val_data),
                   "label_counts": dict(zip(DocKindHead.LABELS, counts)),
                   "config": cfg.__dict__}, f, indent=2)
    return {"best_val": best_val, "log": log}