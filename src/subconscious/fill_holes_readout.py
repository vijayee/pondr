"""R2 fill-holes readout: recover a faded anchor's bge from MEMORY (Stage 2).

The "fill the holes" recovery path for FADE Regime 2 (task #35). When SSM-A's
state is degraded below ``cos_gist`` (the band R3 declares "[forgotten]"), this
readout reconstructs the anchor's bge ADDRESS from *memory* -- the degraded
state + the recent ring context -- and, if it matches the stored bge above
``cos_reconstruct``, the anchor's own blurb is recalled (anchor-locked, no
drift -- the 9455795 invariant). R2 splits the old R4 band into R2 (recoverable)
+ R4 (truly forgotten) by the RECOVERY TEST, not a fixed threshold.

## What it reads, what it outputs

Input (one record, the single-record contract mirroring
``CrossSlotTransformerZHead``):

  * ``state`` -- the raw SSM-A state ``[dim]`` (``ssm_a.state()``), the degraded
    bge-vector channel. This is the [CLS] token (position 0).
  * ``ring_bges`` -- ``[K, dim]`` (``blurbs.vector(aid)`` for ``aid`` in the
    ring), the recent context slots (positions 1..K).

Output: ``[dim]`` L2-normalized -- the reconstructed anchor bge (an ADDRESS in
the frozen bge space, NOT content). The loss is JEPA-fade InfoNCE
(``jepa_infonce_loss``): the prediction must be closer to the anchor's true bge
than to in-batch negatives.

## Why a Transformer (and not the closed form)

SSM-A is a LINEAR EWMA, so un-fade is closed-form arithmetic
(``recovered = (state - sum_j decay**(T-j)*wg*bge_j) / decay**N``). With ALL
interferers (the full blurb store -- the RECORD) it is EXACT, but that uses the
record (defeats the fade: always recovers, R4 never fires). With RING-ONLY
interferers it is the linear baseline -- and ``scripts/probe_r2_band.py`` showed
that baseline FAILS in the R2 band (top-1 = 0.0): the evicted interferers NEWER
than the anchor are AMPLIFIED by the division (``decay**(anchor-j)`` with
``j > anchor`` -> ``(1/decay)**(j-anchor) > 1``) and swamp the signal. The
Transformer earns its keep by approximating the MISSING evicted interferers
nonlinearly from the ring context -- e.g. estimating the stream's centroid
(the dominant interferer bias) from the ring and subtracting it, a learned
operation the linear closed-form does not do. Whether that is POSSIBLE
cross-domain (where the evicted interferers are uncorrelated with the ring) is
the Stage-3 gate's question, not assumed here.

## Anti-collapse

SSM-A is FROZEN (the readout is the only trained part). No LM-prior auxiliary
(the token-LM anti-collapse term from ``jepa_gist`` is N/A -- there is no
token-LM here). InfoNCE itself prevents collapse: a constant prediction is
equally far from every negative AND the positive, failing the objective
(``jepa_gist.py:89-94``). The frozen bge target cannot collapse.

Self-contained: no STRM ``StateReadout`` / ``z_relevance_head`` deps (those are
6144-d STRM-state dims; this reads 384-d bge space). The architectural template
is ``CrossSlotTransformerZHead`` (``cross_slot_transformer.py:45``), mirrored
with two changes: (1) the output head is ``Linear(d_model, dim)`` -> 384-d
reconstruction, not ``Linear(d_model, 1)`` per-slot logit; (2) the [CLS] output
(position 0) is read, not the slot outputs (positions 1..K).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class FillHolesConfig:
    """Architecture hyperparameters for ``FillHolesReadout``.

    ``dim`` is the bge space (384 for bge-small-en-v1.5); ``d_model`` is the
    Transformer's internal width (kept = dim so the state/slot projections are
    square). ``max_pos`` must exceed the largest ring (``ring_capacity + 1`` for
    the [CLS] + K slots). The defaults mirror ``CrossSlotTransformerZHead`` (2
    layers, 4 heads, FFN 512, norm-first GELU, batch-first)."""
    dim: int = 384
    d_model: int = 384
    n_heads: int = 4
    n_layers: int = 2
    ffn: int = 512
    max_pos: int = 64          # > ring_capacity (32) + 1
    dropout: float = 0.0


class FillHolesReadout(nn.Module):
    """Transformer cross-attention readout: (state + ring) -> recovered bge.

    ``forward(state, ring_bges)`` returns the L2-normalized recovered anchor bge
    ``[dim]`` (or ``[B, dim]`` when batched). Single-record: ``state`` is
    ``[dim]``, ``ring_bges`` is ``[K, dim]``. Batched: ``state`` is ``[B, dim]``,
    ``ring_bges`` is ``[B, K, dim]`` (K constant across the batch -- true for
    evicted anchors, where the ring is full at ``ring_capacity``).

    The [CLS] token (position 0) carries the projected degraded state; the slot
    tokens (positions 1..K) carry the projected ring bges. The encoder
    cross-attends; the [CLS] output is read by ``out_head`` -> the recovered
    bge. Positional embeddings + a learned [CLS] token are added (mirrors
    ``CrossSlotTransformerZHead``)."""

    def __init__(self, cfg: FillHolesConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FillHolesConfig()
        d = self.cfg.d_model
        self.dim = self.cfg.dim
        # Project the degraded state (raw ssm_a.state()) and the ring slot bges
        # into the Transformer's d_model space. Square projections (dim == d_model
        # by default) keep it simple; a non-square d_model would need these.
        self.state_proj = nn.Linear(self.cfg.dim, d)
        self.slot_proj = nn.Linear(self.cfg.dim, d)
        # Learned [CLS] token (position 0) added to the projected state, so the
        # encoder's query token carries BOTH a learned slot and the live state.
        self.cls_token = nn.Parameter(torch.randn(1, d) * 0.02)
        # Positional embeddings for positions 0..max_pos-1 (0 = [CLS]).
        self.pos_emb = nn.Parameter(torch.randn(self.cfg.max_pos, d) * 0.02)
        self.dropout = nn.Dropout(self.cfg.dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.ffn, batch_first=True,
            activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.n_layers)
        # Output head: [CLS] output -> recovered bge (384-d reconstruction).
        self.out_head = nn.Linear(d, self.cfg.dim)

    def forward(self, state: Tensor, ring_bges: Tensor) -> Tensor:
        """Reconstruct the anchor bge from the degraded state + the ring.

        ``state``: ``[dim]`` (single record) or ``[B, dim]`` (batched) -- the
        raw SSM-A state. ``ring_bges``: ``[K, dim]`` (single) or ``[B, K, dim]``
        (batched) -- the ring slots' bge vectors. Returns ``[dim]`` (single) or
        ``[B, dim]`` (batched), L2-normalized."""
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)                 # [1, dim]
            ring_bges = ring_bges.unsqueeze(0)         # [1, K, dim]
        B, K, _ = ring_bges.shape
        assert K + 1 < self.cfg.max_pos, (
            f"K+1={K + 1} exceeds max_pos={self.cfg.max_pos}; raise max_pos")
        cls = self.state_proj(state) + self.cls_token  # [B, d]
        slots = self.slot_proj(ring_bges)              # [B, K, d]
        # Build the [1+K] sequence: [CLS] at position 0, slots at 1..K.
        seq = torch.cat([cls.unsqueeze(1), slots], dim=1)   # [B, 1+K, d]
        seq = seq + self.pos_emb[:K + 1].unsqueeze(0)       # broadcast positions
        seq = self.dropout(seq)
        out = self.encoder(seq)                             # [B, 1+K, d]
        cls_out = out[:, 0, :]                              # [B, d] (the [CLS])
        recovered = self.out_head(cls_out)                  # [B, dim]
        recovered = F.normalize(recovered, p=2, dim=-1)     # unit length for cosine
        return recovered[0] if single else recovered

    # --------------------------------------------------------------- checkpoint
    def checkpoint(self, step: int) -> dict:
        """The checkpoint shape ``train_r2_readout`` writes / the loader reads."""
        return {"readout": self.state_dict(), "config": asdict(self.cfg),
                "step": int(step)}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: str = "auto") -> "FillHolesReadout":
        """Build a readout from a checkpoint dict and load it (strict)."""
        cfg = FillHolesConfig(**ckpt["config"])
        readout = cls(cfg)
        missing, unexpected = readout.load_state_dict(ckpt["readout"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"FillHolesReadout state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}")
        dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                           else device if device != "auto" else "cpu")
        return readout.to(dev).eval()


def load_fill_holes_readout(path: str, device: str = "auto",
                            map_location: str = "cpu") -> FillHolesReadout:
    """Load a trained ``FillHolesReadout`` checkpoint -> ready-to-serve readout.

    The checkpoint is the shape ``FillHolesReadout.checkpoint`` writes
    (``{"readout": state_dict, "config": FillHolesConfig dict, "step": int}``).
    Mirrors ``load_cross_slot_transformer``'s strict-mismatch discipline."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return FillHolesReadout.from_checkpoint(ckpt, device=device)