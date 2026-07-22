"""Task #45 acceptance: the cross-slot Transformer relevance head (Head B).

The head-to-head (``scripts/probe_head_to_head_onyx.py``, committed 836b3eb,
[[pondr-strm-task45-cross-slot-transformer-beats-bilinear]]) showed the
cross-slot Transformer CLEARS the 2.0 z_logit gate held-out on REAL Onyx
(2/3 seeds, median 2.614) where the pointwise bilinear
(``CompositeZHead`` = ``StateReadout`` + ``ZRelevanceHead``) FAILS (0/3, median
0.200). DeepSeek-v4-pro's mechanistic diagnosis: a pointwise bilinear scores
each slot INDEPENDENTLY vs the query (an ABSOLUTE sim-to-query), so on serve --
where the fillers are topically close and ALL sims are high -- it cannot
produce the 2.0 RELATIVE margin the gate demands. Cross-slot attention lets
each slot's logit depend on the query AND every other slot -> a RELATIVE
score (sim-to-query attenuated by the candidate pool), the mechanism that
escapes the pointwise margin bound.

This module promotes Head B from a LOCAL nn.Module in the probe to a
``src/subconscious/`` head so the live SERVE gate
(``probe_strm_selectivity_real.py``) can load + score with it -- the
acceptance test. It is byte-identical to the probe's class (same submodule
names -> the probe's checkpoints load via ``load_cross_slot_transformer``
without retraining), so the task #45 verdict carries over unchanged.

**Isolation (binding constraint).** Local-only until the live gate passes.
The live orchestrator (``build_ponder`` / ``serve_ponder`` /
``DEFAULT_BACKBONE_PATH``) is never touched; the existing 19.5M backbone +
all five downstream heads + the 2b gate stay byte-identical. This module is
imported ONLY by the probe scripts (``probe_strm_selectivity_real.py`` via
``--z-head-arch transformer``; ``probe_head_to_head_onyx.py`` for the class).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.subconscious.state_readout import DEFAULT_DIM_IN, StateReadout
from src.subconscious.z_relevance_head import (  # noqa: E402
    PROJ_DIM as Z_PROJ_DIM,
    QUERY_DIM as Z_QUERY_DIM,
    SLOT_DIM as Z_SLOT_DIM,
    Z_DIM,
)


class CrossSlotTransformerZHead(nn.Module):
    """Cross-slot attention relevance head -- the DeepSeek option B.

    Same per-slot readout as ``CompositeZHead`` (``StateReadout`` mlp128
    [dim_in -> 384]), so the ONLY difference from Head A is the SCORING: the
    bilinear's pointwise ``proj_z(z_i).proj_q(q)`` is replaced by a Transformer
    encoder that cross-attends the query (prepended as a [CLS] token) against
    all K slots, then a per-slot logit head reads each slot's encoder output.
    The attention lets slot k's logit depend on the query AND on every other
    slot -> a RELATIVE score (sim-to-query attenuated by the candidate pool),
    the mechanism DeepSeek identified as escaping the pointwise margin bound.

    Single-record interface (matches ``CompositeZHead.logits``): one record's K
    slots at a time, no batching/padding. ``logits(slot_y, slot_signal, q)``
    returns ``[K, 1]`` (so ``.squeeze(-1)`` -> ``[K]``, the contract
    ``p41._zr_per_slot`` + the contrastive loop assume). ``slot_y`` is accepted
    and ignored (the pure-z_i test, same as the composite). Exposes
    ``slot_dim``/``query_dim``/``proj_dim``/``doc_dim`` so the contrastive loop's
    checkpoint-dim reads (modeled on ``_train_contrastive``) are shape-consistent.
    """

    def __init__(self, dim_in: int = DEFAULT_DIM_IN, hidden: int | None = 128,
                 d_model: int = Z_DIM, n_heads: int = 4, n_layers: int = 2,
                 ffn: int = 512, max_pos: int = 64,
                 n_slot_types: int = 0, learnable_temp: bool = False) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.readout = StateReadout(dim_in=self.dim_in, dim_out=d_model, hidden=hidden)
        self.d_model = int(d_model)
        self.max_pos = int(max_pos)
        # Learned positional embedding for the K slot tokens (positions 1..K; the
        # query [CLS] takes position 0 -- a learned token, not the query emb).
        self.pos_emb = nn.Parameter(torch.randn(self.max_pos, d_model) * 0.02)
        # Learned [CLS] query token; the actual query emb is projected + added so
        # the encoder's query token carries BOTH a learned slot and the live query.
        self.cls_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.query_proj = nn.Linear(d_model, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.logit_head = nn.Linear(d_model, 1)
        # Phase 1b (task #50): a slot-type embedding so the cross-slot attention
        # can condition on whether a slot is a CONVERSATION message vs a RETRIEVED
        # doc-episode (the live production ring mixes both; Head B trained on
        # conversation-only rings -> H2 content-shift). ``n_slot_types=0`` (default)
        # = NO embedding = byte-identical to the task #45 arch (the existing
        # best.pt strict-loads). When >0, ``slot_types`` (a [K] long tensor) MUST
        # be passed to ``logits``; the embedding is summed into ``z``.
        self.n_slot_types = int(n_slot_types)
        if self.n_slot_types > 0:
            self.slot_type_emb = nn.Embedding(self.n_slot_types, d_model)
        # Phase 1b: a learnable temperature on the logit head, addressing the
        # s1 -2.508 collapse + the z_r sigmoid compression (poorly-scaled logits).
        # Score = ``logit_head(slot_out) / softplus(logit_temp)``. ``learnable_temp
        # =False`` (default) = NO temp param = byte-identical to the task #45 arch.
        # Init temp so softplus(temp) ~= 1.0 (temp ~= 0.5414) -> a no-op at init.
        self.learnable_temp = bool(learnable_temp)
        if self.learnable_temp:
            self.logit_temp = nn.Parameter(torch.tensor(0.5414))
        # Mirror CompositeZHead's checkpoint dims (the contrastive loop reads
        # these; doc_dim = the readout input dim so a loader can rebuild it).
        self.slot_dim = Z_SLOT_DIM
        self.query_dim = Z_QUERY_DIM
        self.proj_dim = Z_PROJ_DIM
        self.doc_dim = self.dim_in

    def logits(self, slot_y: Tensor, slot_signal: Tensor,
               query_emb: Tensor, slot_types: Tensor | None = None) -> Tensor:
        """Pre-sigmoid relevance logit per slot -> ``[K, 1]``.

        ``slot_signal`` is the raw flattened state ``[K, dim_in]`` (or
        ``[dim_in]``); ``slot_y`` accepted + ignored (pure-z_i test).
        ``slot_types`` (Phase 1b) is an optional ``[K]`` long tensor of slot-type
        ids (0=conversation, 1=retrieved); added as a type embedding into ``z``
        when ``n_slot_types > 0``. ``None`` -> no type embedding (byte-identical
        to the task #45 arch; required only if ``n_slot_types > 0``).
        """
        z = self.readout(slot_signal)                       # [K, d_model]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        K = z.shape[0]
        assert K < self.max_pos, f"K={K} exceeds max_pos={self.max_pos}"
        z = z + self.pos_emb[1:K + 1]                        # [K, d_model]
        if self.n_slot_types > 0:
            assert slot_types is not None, (
                "n_slot_types>0 requires slot_types [K] passed to logits()")
            st = slot_types.to(torch.long).reshape(-1).to(z.device)[:K]
            z = z + self.slot_type_emb(st)                   # [K, d_model]
        # The query arrives as ``[d_model]`` (the head-to-head's
        # ``rec["query_emb"]``) OR ``[1, d_model]`` (the live probe's
        # ``prompt_emb`` batch-of-1). Single-record contract -> flatten to
        # exactly ``[d_model]`` so ``cls_token[0] + q`` stays [d_model] and the
        # ``[CLS]`` token cat is 2-dim throughout.
        q = self.query_proj(query_emb.to(torch.float32).reshape(-1))  # [d_model]
        cls = self.cls_token[0] + q                          # [d_model] (pos 0)
        seq = torch.cat([cls.unsqueeze(0), z], dim=0).unsqueeze(0)  # [1, 1+K, d]
        out = self.encoder(seq)                              # [1, 1+K, d_model]
        slot_out = out[0, 1:, :]                             # [K, d_model]
        logit = self.logit_head(slot_out)                    # [K, 1]
        if self.learnable_temp:
            logit = logit / F.softplus(self.logit_temp)       # scale-aware logit
        return logit                                         # [K, 1]

    def predict(self, slot_y, slot_signal, query_emb):
        return torch.sigmoid(self.logits(slot_y, slot_signal, query_emb))

    forward = predict

    @classmethod
    def from_state_dict(cls, sd: dict, dim_in: int,
                        hidden: int | None = None,
                        n_slot_types: int = 0,
                        learnable_temp: bool = False) -> "CrossSlotTransformerZHead":
        """Build a Head B from a raw state_dict and load it (strict).

        ``hidden`` is the ``StateReadout`` hidden width (None -> Linear readout).
        The encoder/readout arch is fixed by the ctor defaults (2 layers / 4
        heads / FFN-512 / max_pos=64) -- the checkpoint-carried arch knobs are
        ``dim_in`` + ``hidden`` + (Phase 1b) ``n_slot_types`` + ``learnable_temp``.
        Old checkpoints (task #45) carry neither Phase-1b knob -> both default to
        0/False -> NO new params -> strict load is byte-identical. Strict load: a
        key mismatch is a hard error (a silent partial load would score with
        random weights).
        """
        head = cls(dim_in=dim_in, hidden=hidden, n_slot_types=n_slot_types,
                   learnable_temp=learnable_temp)
        missing, unexpected = head.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"CrossSlotTransformerZHead state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        head.eval()
        return head


def load_cross_slot_transformer(
    path: str,
    device: str = "auto",
    map_location: str = "cpu",
) -> CrossSlotTransformerZHead:
    """Load a trained CrossSlotTransformerZHead checkpoint -> ready-to-serve head.

    The checkpoint is the shape ``probe_head_to_head_onyx._train_head`` writes
    (``{"head": state_dict, "arch": "transformer", "slot_dim", "doc_dim"``
    (``=dim_in``), ``"query_dim", "proj_dim", "hidden", ...}``). ``doc_dim`` is
    the readout input dim; ``hidden`` is the readout's MLP width (None = Linear).
    Mirrors ``load_composite_z_head`` so the probe can swap arches via one flag.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "head" in ckpt:
        dim_in = int(ckpt.get("doc_dim", DEFAULT_DIM_IN))
        hidden = ckpt.get("hidden", None)
        if hidden is not None:
            hidden = int(hidden)
        n_slot_types = int(ckpt.get("n_slot_types", 0))           # Phase 1b
        learnable_temp = bool(ckpt.get("learnable_temp", False))    # Phase 1b
        sd = ckpt["head"]
    else:
        # Bare state_dict fallback (not produced by the head-to-head; defensive).
        dim_in = DEFAULT_DIM_IN
        hidden = None
        n_slot_types = 0
        learnable_temp = False
        sd = ckpt
    head = CrossSlotTransformerZHead.from_state_dict(
        sd, dim_in=dim_in, hidden=hidden,
        n_slot_types=n_slot_types, learnable_temp=learnable_temp)
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return head.to(dev).eval()