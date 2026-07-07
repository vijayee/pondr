"""Backbone pre-training on ``follows``-chain state-transition pairs.

Loads ``sequences.jsonl`` (see ``scripts/extract_backbone_sequences.py``),
groups pairs into per-conversation turn sequences, and pre-trains the shared
JGS backbone with a JEPA-style objective: an online predictor predicts the
next embedding, and an EMA target backbone provides a stable representation of
the next embedding to contrast against (BYOL/JEPA-style — the EMA target is the
anti-collapse mechanism alongside batch negatives).

Runs on CPU (dev) or a single modest GPU (pod). Right-sized for a few thousand
pairs: a few thousand steps, small batch. See ``docs/Phase 2a.md`` §0.3/§0.4.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ..configs import BackboneTrainingConfig, BackboneConfig
from ..backbone import JGSBackbone
from .jepa_loss import step_loss


def load_pairs(path: str) -> list[dict]:
    """Load (state_t, state_{t+1}) records from a JSONL file."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def _group_chains(pairs: list[dict]) -> list[list[list[float]]]:
    """Reconstruct per-chain embedding sequences from forward pairs.

    Each forward pair is ``(emb_t, emb_{t+1})`` with a ``chain_id`` and
    ``position``. Reverse pairs are ignored for sequence reconstruction (they
    double the data only for pair-wise training, not for chain sequences).
    """
    by_chain: dict[str, dict[int, list[list[float]]]] = defaultdict(dict)
    for p in pairs:
        if p.get("type") != "forward":
            continue
        cid = p["chain_id"]
        pos = p["position"]
        by_chain[cid][pos] = (p["state_t"], p["state_t_plus_1"])

    chains: list[list[list[float]]] = []
    for cid, steps in by_chain.items():
        if not steps:
            continue
        max_pos = max(steps)
        # Build sequence emb_0..emb_{max_pos+1} if contiguous.
        seq: list[list[float]] = []
        ok = True
        for pos in range(max_pos + 1):
            if pos not in steps:
                ok = False
                break
            if pos == 0:
                seq.append(steps[pos][0])  # emb_0
            seq.append(steps[pos][1])      # emb_{pos+1}
        if ok and len(seq) >= 2:
            chains.append(seq)
    return chains


class BackboneDataset(Dataset):
    """Yields per-chain embedding sequences as tensors ``[seq, 384]``."""

    def __init__(self, pairs: list[dict]):
        self.chains = [torch.tensor(c, dtype=torch.float32) for c in _group_chains(pairs)
                       if len(c) >= 2]

    @classmethod
    def from_chains(cls, chains: list[list[list[float]]]) -> "BackboneDataset":
        """Build a dataset from already-grouped, already-split chains.

        Used by ``pretrain_backbone`` so the train/val split happens once (on
        the chain list) rather than re-parsing pairs per split.
        """
        ds = cls.__new__(cls)
        ds.chains = [torch.tensor(c, dtype=torch.float32) for c in chains if len(c) >= 2]
        return ds

    def __len__(self) -> int:
        return len(self.chains)

    def __getitem__(self, idx: int) -> Tensor:
        return self.chains[idx]


def _collate(batch: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad chains to the same length; return (padded [B,S,384], mask [B,S])."""
    max_len = max(c.shape[0] for c in batch)
    dim = batch[0].shape[1]
    padded = torch.zeros(len(batch), max_len, dim, dtype=batch[0].dtype)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, c in enumerate(batch):
        padded[i, : c.shape[0]] = c
        mask[i, : c.shape[0]] = True
    return padded, mask


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bfloat16":
        # CPU supports bf16; CUDA supports it on Ampere+.
        return torch.bfloat16
    if name == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    return torch.float32


def _cosine_warmup_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _sample_negatives(padded: Tensor, mask: Tensor, n: int) -> Tensor:
    """Sample ``n`` valid embeddings from the batch as JEPA negatives."""
    valid = padded[mask]  # [V, dim]
    if valid.shape[0] == 0:
        return padded.new_zeros(n, padded.shape[-1])
    idx = torch.randint(0, valid.shape[0], (min(n, valid.shape[0]),), device=valid.device)
    return valid[idx]


@torch.no_grad()
def _update_ema(target: JGSBackbone, online: JGSBackbone, decay: float) -> None:
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def pretrain_backbone(config: Optional[BackboneTrainingConfig] = None,
                      pairs_path: Optional[str] = None,
                      progress_cb=None) -> JGSBackbone:
    """Pre-train the shared JGS backbone. Returns the trained backbone.

    ``progress_cb(step, train_loss, val_loss)`` is called every log interval if
    provided (used by tests + the CLI script).
    """
    config = config or BackboneTrainingConfig()
    pairs_path = pairs_path or config.pairs_path
    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype, device)

    pairs = load_pairs(pairs_path)
    if not pairs:
        raise RuntimeError(f"no training pairs found at {pairs_path}")
    chains = [c for c in _group_chains(pairs) if len(c) >= 2]
    n_val = max(1, int(len(chains) * config.val_fraction)) if chains else 0
    val_chains = chains[:n_val]
    train_chains = chains[n_val:] if len(chains) > n_val else chains

    train_ds = BackboneDataset.from_chains(train_chains)
    val_ds = BackboneDataset.from_chains(val_chains)

    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                        collate_fn=_collate, drop_last=False)

    backbone = JGSBackbone(config.backbone).to(device=device, dtype=dtype)
    target = copy.deepcopy(backbone).to(device=device, dtype=dtype)
    for p in target.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in backbone.parameters() if p.requires_grad],
        lr=config.learning_rate, betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    step = 0
    epoch = 0
    while step < config.total_steps:
        for padded, mask in loader:
            if step >= config.total_steps:
                break
            padded = padded.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            for _ in range(config.gradient_accumulation):
                # Autocast only matters for low-precision dtypes; in float32 it
                # is a no-op, so disable it to keep the CPU float32 dev path clean.
                with torch.amp.autocast(device_type=device.type, dtype=dtype,
                                         enabled=(dtype != torch.float32)):
                    online_pred, _, _ = backbone.forward_seq(padded)
                    with torch.no_grad():
                        target_pred, _, _ = target.forward_seq(padded)
                    # online_pred[:, t] predicts emb_{t+1}; target_pred[:, t] is
                    # the EMA target's representation of emb_{t+1}.
                    pred = online_pred[:, :-1, :]
                    tgt = target_pred[:, :-1, :]
                    tgt_mask = mask[:, 1:]
                    negatives = _sample_negatives(padded, mask, config.num_negative_samples)
                    loss = step_loss(pred, tgt, tgt_mask, negatives, config.temperature)
                (loss / config.gradient_accumulation).backward()
                total_loss += loss.detach()
            optimizer.step()
            _update_ema(target, backbone, config.target_ema_decay)

            # LR schedule
            lr = _cosine_warmup_lr(step, config.warmup_steps, config.total_steps, config.learning_rate)
            for g in optimizer.param_groups:
                g["lr"] = lr

            if progress_cb is not None or step % 100 == 0:
                vl = _validate(backbone, target, val_ds, config, device, dtype)
                if progress_cb is not None:
                    progress_cb(step, float(total_loss.item()) / max(config.gradient_accumulation, 1), vl)
                else:
                    print(f"step {step}: train_loss={total_loss.item():.4f} val_loss={vl:.4f} lr={lr:.2e}")

            if config.checkpoint_every and step > 0 and step % config.checkpoint_every == 0:
                _save_checkpoint(backbone, optimizer, step, config.checkpoint_dir)
            step += 1
        epoch += 1
        if epoch > 1000:  # safety
            break

    _save_checkpoint(backbone, optimizer, step, config.checkpoint_dir, final=True)
    return backbone


@torch.no_grad()
def _validate(backbone, target, val_ds, config, device, dtype) -> float:
    if len(val_ds) == 0:
        return 0.0
    loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=_collate)
    total, count = 0.0, 0
    for padded, mask in loader:
        padded = padded.to(device)
        mask = mask.to(device)
        online_pred, _, _ = backbone.forward_seq(padded)
        target_pred, _, _ = target.forward_seq(padded)
        pred = online_pred[:, :-1, :]
        tgt = target_pred[:, :-1, :]
        tgt_mask = mask[:, 1:]
        negatives = _sample_negatives(padded, mask, config.num_negative_samples)
        loss = step_loss(pred, tgt, tgt_mask, negatives, config.temperature)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def _save_checkpoint(backbone, optimizer, step, ckpt_dir, final=False) -> None:
    name = "backbone_final.pt" if final else f"checkpoint_{step}.pt"
    path = os.path.join(ckpt_dir, name)
    torch.save({"backbone": backbone.state_dict(), "step": step}, path)