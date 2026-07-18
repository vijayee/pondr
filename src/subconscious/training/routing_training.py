"""Supervised + outcome-based training for the Retrieval Gate (Phase 2b).

Two stages, matching ``docs/Phase 2b.md`` §3 (corrected — see the doc's §0):

1. **Supervised** on Oracle JEPA routing pairs (``train_retrieval_gate_supervised``):
   freeze the shared backbone (Phase 2a weights), train the instance-owned
   params (input/output projections + LoRA, decomposed gate) and the five
   routing heads. Five losses: domain (multi-label BCE), pathway (CE), skill
   (multi-label BCE, auxiliary ×0.5), model_size (CE), deliberation (BCE).
2. **Outcome-based** (``OutcomeBasedTrainer``): REINFORCE on recorded
   (embedding, context, decision, outcome) tuples from the live pipeline —
   the personalization stage. No-op until the replay buffer has ≥ min_buffer
   outcomes (the live signals aren't wired yet; exercised in tests for now).

The shared backbone is the Phase 2a full-corpus retrain at
``data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt`` (better
than the bounded 1,108-pair slice — see ``hippo-phase-2a-status``).

bf16/autocast is still unfixed in the 2a path, so the gate trains in **float32**
(the instance params + heads are small — fp32 is fine and matches the doc's
``RetrievalGateTrainingConfig.dtype="float32"``).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..backbone import JGSBackbone
from ..configs import BackboneConfig, INSTANCE_CONFIGS, InstanceConfig
from ..doc_kind_head import DocKindHead
from ..gate import GateContext
from ..retrieval_gate import RetrievalGate
from ..routing import (
    AVAILABLE_DOMAINS, META_SKILLS, MODEL_SIZES, PATHWAYS,
    Embedder, RoutingDecision, RoutingOutcome, RoutingReplayEntry,
)
from .replay_buffer import ReplayBuffer


# ── device / dtype resolution (mirrors training/pretrain.py) ──

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str) -> torch.dtype:
    # Gate training is float32-only. The bf16/autocast dtype-mix bug in the 2a
    # path is unfixed (deferred), and the gate is small enough that fp32 is
    # fine, so every requested dtype resolves to float32. We WARN on a non-fp32
    # request rather than silently downgrading, so a user who passes --dtype bf16
    # expecting a speedup doesn't misread their own wall-clock.
    import sys
    if name not in ("float32", "fp32", "auto"):
        print(f"  NOTE: dtype '{name}' requested but bf16/autocast is unfixed in the "
              f"2a path — training runs float32 (no speedup).", file=sys.stderr)
    return torch.float32


# ── backbone loading ──

def load_backbone(
    path: str,
    config: Optional[BackboneConfig] = None,
    device: str = "auto",
    map_location: str = "cpu",
) -> JGSBackbone:
    """Load the shared Phase 2a backbone checkpoint → frozen ``JGSBackbone``.

    The checkpoint is ``{"backbone": state_dict, "step": n}`` (see
    ``training/pretrain.py:_save_checkpoint``). Loads strict, moves to the
    resolved device, freezes every param (the gate trains its own heads +
    instance params; the shared backbone is fixed).
    """
    cfg = config or BackboneConfig()
    # weights_only=False: the checkpoint is the user's own Phase 2a output (a
    # plain {"backbone": state_dict, "step": n} dict, no code). Safe here; do
    # NOT load arbitrary .pt files this way.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    sd = ckpt["backbone"] if isinstance(ckpt, dict) and "backbone" in ckpt else ckpt
    backbone = JGSBackbone(cfg)
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    # Surface a loud mismatch rather than silently training on a partial load.
    if missing or unexpected:
        raise RuntimeError(
            f"backbone checkpoint {path} mismatch: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    dev = _resolve_device(device)
    backbone = backbone.to(dev)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone


def load_retrieval_gate(
    path: str,
    backbone: JGSBackbone,
    config: Optional[InstanceConfig] = None,
    device: str = "auto",
    map_location: str = "cpu",
) -> RetrievalGate:
    """Load a trained Phase 2b RetrievalGate checkpoint onto ``backbone``.

    The checkpoint is ``{"gate": state_dict, "val_accuracy": float,
    "epoch": int}`` (see ``train_retrieval_gate_supervised``'s save). The
    gate's ``state_dict()`` EXCLUDES the shared backbone (it is stored via
    ``object.__setattr__`` on ``JGSInstance``, not registered as a submodule),
    so loading ``ckpt["gate"]`` restores only the instance-owned params
    (input/output projections + LoRA, decomposed gate) and the five routing
    heads -- the already-frozen ``backbone`` passed in is reused, NOT
    reloaded. Loads strict (raises on any missing/unexpected key, mirroring
    ``load_backbone``), moves to the resolved device, eval mode.

    This is the runtime loader -- it pairs with ``load_backbone`` so a serving
    entrypoint can stand up the TRAINED gate (val 0.826) on the TRAINED
    backbone, instead of the fresh untrained instances every test construction
    uses.
    """
    cfg = config or INSTANCE_CONFIGS["retrieval_gate"]
    gate = RetrievalGate(backbone, cfg)
    # weights_only=False: the checkpoint is the user's own Phase 2b output (a
    # plain {"gate": sd, ...} dict, no code). Safe here; do NOT load arbitrary
    # .pt files this way -- same contract as load_backbone.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    sd = ckpt["gate"] if isinstance(ckpt, dict) and "gate" in ckpt else ckpt
    missing, unexpected = gate.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"gate checkpoint {path} mismatch: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    dev = _resolve_device(device)
    gate = gate.to(dev)
    gate.eval()
    return gate


def load_doc_kind_head(
    path: str,
    backbone: JGSBackbone,
    config: Optional[InstanceConfig] = None,
    device: str = "auto",
    map_location: str = "cpu",
) -> DocKindHead:
    """Load a trained DocKindHead checkpoint onto ``backbone`` (frozen).

    Pairs with ``load_backbone`` exactly as ``load_retrieval_gate`` does. The
    checkpoint is ``{"head": state_dict, "labels": [...], "val_accuracy": float,
    "epoch": int, "feat_dim": int}`` (see ``train_doc_kind_head_supervised``'s
    save; ``feat_dim`` is absent on pre-Phase-4 checkpoints -> defaults to 0).
    The head's ``state_dict()`` EXCLUDES the shared backbone (stored via
    ``object.__setattr__`` on ``JGSInstance``), so loading ``ckpt["head"]``
    restores only the instance-owned params (input/output projections + LoRA,
    state_lora, decomposed gate) and the 5-class classifier head -- the
    already-frozen ``backbone`` passed in is reused, NOT reloaded. ``feat_dim``
    is read BEFORE constructing the head so the classifier Linear's in_features
    matches the checkpoint (a feat-trained ckpt widens the Linear; a feat-less
    ckpt keeps it at 256). Loads strict (raises on any missing/unexpected key,
    mirroring ``load_backbone`` / ``load_retrieval_gate``), validates the
    persisted label order against ``DocKindHead.LABELS`` (a mismatch is a hard
    error -- the logits would map to the wrong classes), moves to the resolved
    device, eval mode.
    """
    cfg = config or INSTANCE_CONFIGS["doc_kind"]
    # weights_only=False: the checkpoint is the user's own training output (a
    # plain {"head": sd, "labels": [...], ...} dict, no code). Safe here; do
    # NOT load arbitrary .pt files this way -- same contract as load_backbone.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    # feat_dim (Phase 4): widen the head's first Linear to accept the temporal
    # feature. Read BEFORE constructing the head so the Linear shape matches the
    # checkpoint (a feat-trained ckpt into a feat-less head, or vice versa, is a
    # shape mismatch -> load_state_dict would silently mis-wire the Linear).
    # Default 0 = pre-Phase-4 checkpoint (no feature) -> backward-compatible.
    feat_dim = int(ckpt.get("feat_dim", 0)) if isinstance(ckpt, dict) else 0
    if feat_dim < 0:
        raise RuntimeError(
            f"doc-kind head checkpoint {path} has negative feat_dim={feat_dim}"
        )
    # attention (Phase 5): build the attention readout modules so the head's
    # state_dict matches (attn_key + attn_query). Read BEFORE constructing the
    # head, like feat_dim. Default False = pre-Phase-5 checkpoint (mean-pool) ->
    # backward-compatible. A mismatch (attention ckpt into a mean-pool head or
    # vice versa) is caught by the strict state_dict check below.
    attention = bool(ckpt.get("attention", False)) if isinstance(ckpt, dict) else False
    head = DocKindHead(backbone, cfg, feat_dim=feat_dim,
                       attention_readout=attention)
    sd = ckpt["head"] if isinstance(ckpt, dict) and "head" in ckpt else ckpt
    labels = ckpt.get("labels") if isinstance(ckpt, dict) else None
    if labels is not None and list(labels) != list(DocKindHead.LABELS):
        raise RuntimeError(
            f"doc-kind head checkpoint {path} label-order mismatch: "
            f"ckpt={list(labels)} != head={list(DocKindHead.LABELS)} -- "
            f"the logits would map to the wrong classes"
        )
    missing, unexpected = head.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"doc-kind head checkpoint {path} mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    dev = _resolve_device(device)
    head = head.to(dev)
    head.eval()
    return head


# ── config ──

@dataclass
class RetrievalGateTrainingConfig:
    # Architecture (mirrors INSTANCE_CONFIGS["retrieval_gate"] + BackboneConfig).
    d_model: int = 384
    d_state: int = 16
    lora_rank: int = 4

    # Supervised training (Oracle pairs).
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    skill_loss_weight: float = 0.5     # skills are auxiliary

    # Outcome-based training (REINFORCE personalization).
    online_lr: float = 1e-5
    replay_buffer_capacity: int = 1000
    outcome_batch_size: int = 32
    min_buffer: int = 50               # don't train until this many outcomes

    # Hardware / IO.
    dtype: str = "float32"
    device: str = "auto"
    val_fraction: float = 0.2
    seed: int = 0
    backbone_path: str = "data/pod_runs/phase2a_full/checkpoints/backbone/backbone_final.pt"
    checkpoint_dir: str = "data/training/routing_gate"


# ── data + embedder ──

def load_routing_pairs(path: str) -> list[dict]:
    """Load Oracle JEPA routing pairs from JSONL.

    Each record is ``{"query", "route": {domains, pathway, meta_skills,
    model_size, needs_deliberation, confidence, reasoning}, ...}`` (see
    ``scripts/generate_jepa_training_data.py``). Drops records that fail to
    parse OR whose ``route`` is missing the required keys (``pathway``,
    ``model_size``, ``domains``) — the Oracle occasionally returns a slightly-off
    schema (``route.data.pathway`` nesting, ``route.domain`` singular); keeping
    those would let ``_routing_targets`` silently degrade them to default
    (graph_retrieve / no-domains) labels, i.e. *wrong* labels. Dropping is
    honest. ``meta_skills`` may be empty/absent (auxiliary). Reports a count.
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
            route = rec.get("route")
            # Validate the fields _routing_targets reads. ``needs_deliberation``
            # MUST be a real bool: a string "false" is truthy in Python → would
            # silently flip the deliberation label to True. ``meta_skills`` MUST
            # be a list: a string would iterate char-by-char (none match the
            # vocab → silent zero labels). Drop both, same as the other fields.
            if (isinstance(rec.get("query"), str) and isinstance(route, dict)
                    and isinstance(route.get("domains"), list)
                    and route.get("pathway") in PATHWAYS
                    and route.get("model_size") in MODEL_SIZES
                    and isinstance(route.get("meta_skills", []), list)
                    and isinstance(route.get("needs_deliberation", False), bool)):
                records.append(rec)
            else:
                dropped += 1
    if dropped:
        print(f"  load_routing_pairs: dropped {dropped} unparseable/malformed records")
    return records


class _StubEmbedder:
    """Deterministic hash embedder for offline tests / no-model-download smoke.

    Maps text → 384-dim float vector via SHA256 of the text, expanded to 384 and
    L2-normalized. Shape-correct and deterministic, NOT semantically meaningful
    — use ``build_embedder("on-demand")`` for real training (the routing pairs
    are query strings; real embeddings make the gate actually learn routing).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Expand the 32-byte hash to dim bytes by repeated hashing, then map
            # bytes to [-1, 1] and normalize.
            buf = bytearray()
            counter = 0
            while len(buf) < self.dim:
                buf += hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
                counter += 1
            vec = [(b / 127.5 - 1.0) for b in buf[:self.dim]]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


def build_embedder(source: str) -> Embedder:
    """Construct an ``Embedder`` by source name.

    ``"on-demand"`` → the real local bge-small-en-v1.5 sentence-transformer
    (384-dim, matching ``config.embedding_model`` and the backbone's
    ``d_model=384``). Lazy-imported so this module imports without
    ``sentence_transformers`` installed. ``"stub"`` → the deterministic hash
    embedder (shape-only dev smoke — NOT for real training).
    """
    if source == "stub":
        return _StubEmbedder(dim=384)
    if source == "on-demand":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised only when installed
            raise ImportError(
                "build_embedder('on-demand') requires sentence_transformers: "
                "pip install sentence-transformers"
            ) from e
        from ...config import config as _config  # local bge-small model name
        st = SentenceTransformer(_config.embedding_model)

        class _STEmbedder:
            def encode(self, texts: list[str]) -> list[list[float]]:
                # SentenceTransformer.encode returns an ndarray; convert to plain
                # lists so it satisfies the Embedder Protocol's return type.
                import numpy as np
                arr = st.encode(texts, convert_to_numpy=True)
                return [list(map(float, row)) for row in np.asarray(arr)]

        return _STEmbedder()
    raise ValueError(f"unknown embedder source: {source!r}")


# ── loss + targets ──

def _safe_index(value: str, vocab: list[str]) -> int:
    """Vocab index of ``value``; 0 if absent (malformed Oracle output → class 0)."""
    return vocab.index(value) if value in vocab else 0


def _routing_targets(batch: list[dict], device: torch.device) -> dict[str, Tensor]:
    """Build the 5 target tensors for one supervised batch from Oracle routes."""
    b = len(batch)
    dom_t = torch.zeros(b, len(AVAILABLE_DOMAINS), device=device)
    skill_t = torch.zeros(b, len(META_SKILLS), device=device)
    path_t = torch.zeros(b, dtype=torch.long, device=device)
    size_t = torch.zeros(b, dtype=torch.long, device=device)
    delib_t = torch.zeros(b, 1, device=device)
    for i, ex in enumerate(batch):
        route = ex["route"]
        for d in route.get("domains", []):
            if d in AVAILABLE_DOMAINS:
                dom_t[i, AVAILABLE_DOMAINS.index(d)] = 1.0
        for s in route.get("meta_skills", []):
            if s in META_SKILLS:
                skill_t[i, META_SKILLS.index(s)] = 1.0
        path_t[i] = _safe_index(route.get("pathway", "graph_retrieve"), PATHWAYS)
        size_t[i] = _safe_index(route.get("model_size", "3B"), MODEL_SIZES)
        delib_t[i, 0] = 1.0 if route.get("needs_deliberation", False) else 0.0
    return {"domain": dom_t, "skill": skill_t, "pathway": path_t,
            "model_size": size_t, "deliberation": delib_t}


def _routing_loss(logits: dict[str, Tensor], targets: dict[str, Tensor],
                  skill_weight: float, pathway_weight: Optional[Tensor] = None,
                  size_weight: Optional[Tensor] = None) -> Tensor:
    """Five-head supervised loss (domain BCE / pathway CE / skill BCE aux /
    model_size CE / deliberation BCE). Returns a scalar.

    ``pathway_weight`` / ``size_weight`` (optional, ``[vocab]`` tensors) are
    inverse-frequency class weights passed to the two cross-entropy heads —
    without them the gate collapses to the majority class (``graph_retrieve`` is
    ~61% of pairs, ``3B`` ~82%), routing everything to the majority and making
    the 0.86 val_acc a majority-class baseline rather than learning. See
    ``_class_weights``.
    """
    domain_loss = F.binary_cross_entropy_with_logits(logits["domain"], targets["domain"])
    pathway_loss = F.cross_entropy(logits["pathway"], targets["pathway"],
                                   weight=pathway_weight)
    skill_loss = F.binary_cross_entropy_with_logits(logits["skill"], targets["skill"])
    size_loss = F.cross_entropy(logits["model_size"], targets["model_size"],
                                weight=size_weight)
    delib_loss = F.binary_cross_entropy_with_logits(logits["deliberation"], targets["deliberation"])
    return domain_loss + pathway_loss + skill_weight * skill_loss + size_loss + delib_loss


def _class_weights(labels: list[int], n_classes: int, device: torch.device,
                   smooth: float = 1.0) -> Tensor:
    """Inverse-frequency class weights for a CE loss.

    ``w[c] = total / (n_classes * (count[c] + smooth))`` (standard inverse-freq
    with +1 smoothing so a class with 0 examples doesn't divide by zero).
    Minority classes get larger weight so the gate can't win by predicting only
    the majority. Returns a ``[n_classes]`` tensor on ``device``.
    """
    counts = [0] * n_classes
    for c in labels:
        if 0 <= c < n_classes:
            counts[c] += 1
    total = max(sum(counts), 1)
    w = [total / (n_classes * (counts[c] + smooth)) for c in range(n_classes)]
    return torch.tensor(w, dtype=torch.float32, device=device)


def _train_labels(train_data: list[dict]) -> tuple[list[int], list[int]]:
    """Pathway + model_size label indices for every training record (for weights)."""
    path_labels = [_safe_index(r["route"].get("pathway", "graph_retrieve"), PATHWAYS)
                   for r in train_data]
    size_labels = [_safe_index(r["route"].get("model_size", "3B"), MODEL_SIZES)
                   for r in train_data]
    return path_labels, size_labels


# ── supervised training ──

def _embed_all(embedder: Embedder, queries: list[str], device: torch.device) -> Tensor:
    """Embed a list of query strings → ``[N, input_dim]`` float32 tensor on device."""
    vecs = embedder.encode(queries)
    return torch.tensor(vecs, dtype=torch.float32, device=device)


def evaluate_routing(
    gate: RetrievalGate,
    val: list[dict],
    val_embeddings: Tensor,
    device: torch.device,
) -> float:
    """Routing accuracy on held-out pairs (doc §3.2 weighted metric).

    pathway exact (0.35) + domain F1 (0.30) + model_size exact (0.20)
    + deliberation exact (0.15). Domain is scored as **per-example F1** between
    the Oracle domains and the predicted domains (not subset-recall): pure
    recall is gameable — a gate that fires ALL domains always satisfies
    ``oracle ⊆ predicted`` and scores 1.0, so an untrained random init can win
    ``best.pt`` (observed: epoch-0 random scored 0.99 on recall and clobbered
    every trained epoch). F1 penalizes over-prediction, so the metric reflects
    training and ``best.pt`` is a real trained checkpoint. Empty/empty = 1.0;
    empty/non-empty = 0.0; otherwise harmonic mean of precision and recall.
    Runs the whole val set in ONE batched forward.
    """
    gate.eval()
    n = len(val)
    if n == 0:
        return 0.0
    with torch.no_grad():
        gate.reset_state(n, device=device, dtype=torch.float32)
        logits, gate_decision, _ = gate.forward(val_embeddings.to(device))
        decisions = gate.decode_batch(logits, gate_decision)
    cp = cd = cs = cdl = 0
    for ex, dec in zip(val, decisions):
        route = ex["route"]
        oracle = set(d for d in route.get("domains", []) if d in AVAILABLE_DOMAINS)
        pred = set(dec.domains)
        if not oracle and not pred:
            cd += 1.0                       # both empty — correct
        elif not oracle or not pred:
            cd += 0.0                       # one empty, other not — wrong
        else:
            tp = len(oracle & pred)
            prec = tp / len(pred)
            rec = tp / len(oracle)
            cd += (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if dec.pathway == route.get("pathway"):
            cp += 1
        if dec.model_size == route.get("model_size"):
            cs += 1
        if dec.needs_deliberation == route.get("needs_deliberation", False):
            cdl += 1
    return (0.35 * cp + 0.30 * cd + 0.20 * cs + 0.15 * cdl) / n


def train_retrieval_gate_supervised(
    gate: RetrievalGate,
    backbone: JGSBackbone,
    train_data: list[dict],
    val_data: list[dict],
    embedder: Embedder,
    config: Optional[RetrievalGateTrainingConfig] = None,
    device: Optional[torch.device] = None,
    progress_cb=None,
) -> dict:
    """Train the Retrieval Gate supervised on Oracle routing pairs.

    Backbone is frozen (caller passes the already-frozen ``load_backbone``
    result; this also re-freezes for safety). Trains gate params only. Embeds
    all queries once up front (the embedder is the expensive part), resets the
    instance state per batch (each query is independent in 2b — no cross-query
    memory yet), checkpoints best + final as ``gate.state_dict()``, writes a
    per-epoch ``train_log.json``. Returns ``{"best_val", "log"}``.
    """
    cfg = config or RetrievalGateTrainingConfig()
    dev = device or _resolve_device(cfg.device)
    _resolve_dtype(cfg.dtype)   # float32 always; warns if a non-fp32 dtype requested

    # Freeze the shared backbone (load_backbone already froze it; belt+suspenders).
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    gate = gate.to(dev)          # dtype is always float32 (see _resolve_dtype)
    optimizer = torch.optim.AdamW(
        [p for p in gate.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # Embed all queries once (embedder calls are the expensive part).
    train_emb = _embed_all(embedder, [ex["query"] for ex in train_data], dev)
    val_emb = _embed_all(embedder, [ex["query"] for ex in val_data], dev)

    # Inverse-frequency class weights for the two CE heads. Without these the
    # gate collapses to the majority class (graph_retrieve ~61%, 3B ~82%) and
    # routes everything to the majority — a non-functional gate. The weights
    # make a minority-class example worth ~10-50x a majority one in the loss.
    path_labels, size_labels = _train_labels(train_data)
    pathway_weight = _class_weights(path_labels, len(PATHWAYS), dev)
    size_weight = _class_weights(size_labels, len(MODEL_SIZES), dev)
    print(f"  class weights — pathway: {[round(float(x),2) for x in pathway_weight]}")
    print(f"  class weights — size:    {[round(float(x),2) for x in size_weight]}")

    rng = random.Random(cfg.seed)
    n_train = len(train_data)
    log: list[dict] = []
    best_val = 0.0
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        gate.train()
        order = list(range(n_train))
        rng.shuffle(order)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            batch = [train_data[i] for i in idx]
            embs = train_emb[idx].to(dev)          # [B, 384] (float32)
            optimizer.zero_grad()
            gate.reset_state(len(idx), device=dev, dtype=embs.dtype)
            logits, _gd, _out = gate.forward(embs)
            targets = _routing_targets(batch, dev)
            loss = _routing_loss(logits, targets, cfg.skill_loss_weight,
                                 pathway_weight=pathway_weight, size_weight=size_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_acc = evaluate_routing(gate, val_data, val_emb, dev)
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "val_accuracy": round(val_acc, 6)})
        if progress_cb is not None:
            progress_cb(epoch, train_loss, val_acc)
        else:
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save({"gate": gate.state_dict(), "val_accuracy": best_val,
                        "epoch": epoch}, ckpt_dir / "best.pt")

    torch.save({"gate": gate.state_dict(), "val_accuracy": best_val,
                "epoch": cfg.epochs - 1}, ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_val": best_val, "log": log,
                   "n_train": n_train, "n_val": len(val_data),
                   "config": cfg.__dict__}, f, indent=2)
    return {"best_val": best_val, "log": log}


# ── outcome-based trainer (REINFORCE) ──

def _reinforce_loss(logits: dict[str, Tensor], decision: RoutingDecision,
                     reward: float) -> Tensor:
    """REINFORCE policy-gradient term for one recorded decision.

    ``-reward · log p(chosen)`` summed across the five heads: domain + skill
    (sigmoid, per chosen label), pathway + model_size (softmax, chosen class),
    deliberation (sigmoid on the chosen bool). Gradients accumulate into the
    returned scalar; the caller sums across the batch and backprops once.
    """
    r = torch.tensor(float(reward), dtype=logits["domain"].dtype, device=logits["domain"].device)
    terms: list[Tensor] = []

    # Domain: reinforce each chosen domain (sigmoid).
    dom_logits = logits["domain"][0]
    for d in decision.domains:
        if d in AVAILABLE_DOMAINS:
            i = AVAILABLE_DOMAINS.index(d)
            terms.append(-r * F.logsigmoid(dom_logits[i]))

    # Skill: reinforce each chosen skill (sigmoid). Empty list → no term.
    skill_logits = logits["skill"][0]
    for s in decision.meta_skills:
        if s in META_SKILLS:
            i = META_SKILLS.index(s)
            terms.append(-r * F.logsigmoid(skill_logits[i]))

    # Pathway (softmax).
    if decision.pathway in PATHWAYS:
        i = PATHWAYS.index(decision.pathway)
        terms.append(-r * F.log_softmax(logits["pathway"][0], dim=-1)[i])

    # Model size (softmax).
    if decision.model_size in MODEL_SIZES:
        i = MODEL_SIZES.index(decision.model_size)
        terms.append(-r * F.log_softmax(logits["model_size"][0], dim=-1)[i])

    # Deliberation (sigmoid on the chosen bool).
    dl = logits["deliberation"][0, 0]
    if decision.needs_deliberation:
        terms.append(-r * F.logsigmoid(dl))
    else:
        terms.append(-r * F.logsigmoid(-dl))

    if not terms:
        return torch.zeros((), dtype=logits["domain"].dtype, device=logits["domain"].device)
    return torch.stack(terms).sum()


class OutcomeBasedTrainer:
    """Fine-tunes the Retrieval Gate from recorded routing outcomes (REINFORCE).

    Stores ``(embedding, context, decision, outcome)`` in a ``ReplayBuffer``,
    then on ``train_from_outcomes`` re-runs ``gate.forward`` on each sampled
    entry to recover fresh logits and applies a policy-gradient step weighted
    by the outcome reward. Lower LR than supervised (online fine-tuning). No-op
    until the buffer holds ≥ ``min_buffer`` outcomes (the live user-feedback /
    efficiency / delegation signals aren't wired into the pipeline yet — this
    is exercised in tests with synthetic outcomes for now).
    """

    def __init__(self, gate: RetrievalGate, config: Optional[RetrievalGateTrainingConfig] = None):
        self.gate = gate
        cfg = config or RetrievalGateTrainingConfig()
        self.config = cfg
        # Freeze the shared backbone (the supervised trainer does too, but a
        # caller can construct this trainer on an un-frozen gate — e.g. the
        # tests). The optimizer below only steps gate params, so without this
        # freeze backbone grads would accumulate across REINFORCE steps (never
        # cleared by optimizer.zero_grad, never applied) — a slow memory leak on
        # the 19.5M backbone params in a long personalization loop.
        for p in gate.backbone.parameters():
            p.requires_grad = False
        self.buffer = ReplayBuffer(capacity=cfg.replay_buffer_capacity)
        self.optimizer = torch.optim.AdamW(
            [p for p in gate.parameters() if p.requires_grad],
            lr=cfg.online_lr, weight_decay=cfg.weight_decay,
        )

    def record_outcome(self, embedding: Tensor, context: Optional[GateContext],
                       decision: RoutingDecision, outcome: RoutingOutcome) -> None:
        """Record a (state, decision, outcome) tuple for later training."""
        self.buffer.push(RoutingReplayEntry(
            embedding=embedding, context=context, decision=decision,
            outcome=outcome, filled=True,
        ))

    def train_from_outcomes(self, batch_size: Optional[int] = None) -> float:
        """One REINFORCE step over a replay sample. Returns the loss (0 if too few)."""
        bs = batch_size or self.config.outcome_batch_size
        if len(self.buffer) < self.config.min_buffer:
            return 0.0
        batch = self.buffer.sample(bs)
        self.gate.train()         # eval mode would silence any future dropout
        self.optimizer.zero_grad()
        total = torch.zeros((), device=next(self.gate.parameters()).device,
                            dtype=next(self.gate.parameters()).dtype)
        for entry in batch:
            if not entry.filled or entry.outcome is None:
                continue
            # Re-run the forward to recover fresh logits (the decision only
            # carries discrete choices). State resets per entry — each recorded
            # decision was independent.
            self.gate.reset_state(1, device=entry.embedding.device,
                                  dtype=entry.embedding.dtype)
            logits, _gd, _out = self.gate.forward(entry.embedding, entry.context)
            total = total + _reinforce_loss(logits, entry.decision, entry.outcome.reward())
        if total.requires_grad:
            total.backward()
            self.optimizer.step()
        return float(total.item())