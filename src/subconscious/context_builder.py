"""ContextBuilder: learned ``PresentationGate`` selector/reranker (STRM Phase 3).

The Phase 3 context-builder is the learned ``PresentationGate`` the codebase has
been reserving replay buffers for since Phase 2c. It attends over the WM ring
buffer of recent step outputs ``y_t`` with the shipped 2a relevance head's
``r_i`` as an ADDITIVE attention bias, and emits a DISCRETE top-m selection of
which retrieved episodes become primary context. The continuous ``attn@V`` is
an internal training surrogate only; the consumer-facing output is bounded text
of the selected episodes (same shape as the heuristic ``PresentationGate``'s
primary/compressed split) -- ``predict`` returns top-m SLOT INDICES, which the
orchestrator maps back to retrieved episodes via ``source_id``.

Design deviation from the STRM spec (documented up front, matching the shipped
2a/2b/2c read-out heads): the spec says ``INSTANCE_CONFIGS["strm_context_builder"]``
and a ``JGSInstance`` modeling. The shipped 2a relevance / 2b recoverability /
2c latent-dynamics heads are plain ``nn.Module`` because they READ the WM ring
rather than STEP the SSM; the context-builder is the same -- it attends over
existing ring slots, it does not step the SSM. So ``ContextBuilder`` is a plain
``nn.Module``, NOT registered in ``INSTANCE_CONFIGS``. Documented in
``relevance_head.py:11-22``.

Architecture -- cross-slot self-attention with a relevance bias, NOT a per-slot
rescore. A naive "``q . W_k(y_t) + lambda_r * r_i``" with no cross-slot mixing is
just a re-scored relevance head and adds nothing over the 2a head alone. The
``cross_attn`` layer (multi-head self-attention over the K slots) is what makes
the "cross-slot attention" claim real: each slot's key/value is contextualized
by the OTHER slots before the query scores it. ``W_doc`` fuses the slot's
re-embedded doc identity (the SAME 384-d bge vector the 2a head consumed) so the
builder is not blind to doc semantics when ``y_t`` (256-d) carries little.

    k_i = W_k(y_i) + W_doc(doc_i)                 # [K, d_head]
    v_i = W_v(y_i)                               # [K, d_head]
    h   = cross_attn(k, k, v)[0]                  # [K, d_head]  cross-slot
    q   = W_q(query)                              # [1, d_head]
    s_i = (q * h_i).sum(-1) * scale + lambda_r * r_i + bias   # [K]

``lambda_r`` is a learnable scalar (init 1.0) weighting the frozen 2a ``r_i``
bias -- so the shipped relevance signal is a LIVE input, not a frozen constant
(de-wonk risk: a builder that silently drives ``lambda_r -> 0`` is just a
per-slot rescore and the 2a head added nothing; the trainer logs the final
``lambda_r`` and the serve-path test asserts it stayed nonzero). No positional
encoding anywhere -- ERAG candidates are shuffled, so the builder must be
permutation-equivariant (a test asserts slot-permutation -> top-m permutes
correspondingly).

Training is per-slot BCE + a small listwise Plackett-Luce auxiliary on the 2a
ERAG traces (``training/context_builder_training.py``); ``r_i`` is computed at
train time from the FROZEN shipped 2a head as a constant input feature. The
gate is gold-coverage of the top-m selection vs the heuristic ``PresentationGate``
at equal per-query ``m`` on the val split, with a Wilson 95% CI lower bound.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

# Ring slot y_t = step OUTPUT (output_proj readout), 256-d -- matches
# INSTANCE_CONFIGS["working_memory"].output_dim + the 2a RelevanceHead.SLOT_DIM.
SLOT_DIM = 256
# The slot's re-embedded bge-small doc vector, 384-d -- the SAME vector the 2a
# head consumed (produced by relevance_score.score_ring_slots_with_doc_embs).
DOC_DIM = 384
# The query embedding is the RAW bge-small vector, 384-d.
QUERY_DIM = 384
# Attention head width. q/k/v project to d_head; MultiheadAttention splits it
# across num_heads.
D_HEAD = 128
NUM_HEADS = 4
# Default number of primary slots selected at serve. The gate adapts m per
# query (= heuristic N_heur); the checkpoint stores the serve-time fixed m.
TOP_M = 5
# lambda_r init: the shipped r_i starts as a unit-weight live bias. A trained
# checkpoint that drives this to 0 has discarded the 2a signal -- the trainer
# logs the final value and de-wonk flags a collapse.
LAMBDA_R_INIT = 1.0


class ContextBuilder(nn.Module):
    """Cross-slot attention selector with a 2a-relevance bias.

    A small Transformer selector/reranker over the WM ring slots. ``logits``
    returns a per-slot score ``s`` (pre-sigmoid, for BCEWithLogits at training);
    ``predict`` returns the top-m slot indices (the discrete selection the
    orchestrator maps to retrieved episodes via ``source_id``). The query is an
    input, not a parameter -- the SAME slots score differently against different
    queries.
    """

    def __init__(
        self,
        slot_dim: int = SLOT_DIM,
        doc_dim: int = DOC_DIM,
        query_dim: int = QUERY_DIM,
        d_head: int = D_HEAD,
        num_heads: int = NUM_HEADS,
        top_m: int = TOP_M,
        lambda_init: float = LAMBDA_R_INIT,
    ) -> None:
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.doc_dim = int(doc_dim)
        self.query_dim = int(query_dim)
        self.d_head = int(d_head)
        self.num_heads = int(num_heads)
        self.top_m = int(top_m)
        if d_head % num_heads != 0:
            raise ValueError(
                f"ContextBuilder: d_head ({d_head}) must be divisible by "
                f"num_heads ({num_heads}) for MultiheadAttention"
            )
        self.W_q = nn.Linear(self.query_dim, self.d_head)
        self.W_k = nn.Linear(self.slot_dim, self.d_head)
        self.W_v = nn.Linear(self.slot_dim, self.d_head)
        self.W_doc = nn.Linear(self.doc_dim, self.d_head)
        # batch_first so the call site passes [1, K, d_head]; cross-slot
        # self-attention -- query=key=value=k, so each slot's representation is
        # contextualized by the other slots (the whole point vs a per-slot rescore).
        self.cross_attn = nn.MultiheadAttention(
            self.d_head, self.num_heads, batch_first=True,
        )
        self.lambda_r = nn.Parameter(torch.tensor(float(lambda_init)))
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.bias = nn.Parameter(torch.zeros(1))

    # ── prediction ──

    def _coerce(
        self,
        slots_y: Tensor,
        slots_doc_emb: Tensor,
        query_emb: Tensor,
        r: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Coerce the four inputs to 2-D and move them to the builder's device.

        Serve pattern: one query, K ring slots. ``slots_y`` [K, slot_dim],
        ``slots_doc_emb`` [K, doc_dim], ``query_emb`` [query_dim] or
        [1, query_dim], ``r`` [K]. Moving to the builder's device here (a no-op
        when already aligned) avoids a mixed-device cross_attn crash at serve
        where the WM ring sits on the backbone device and the bge query sits on
        CPU -- mirrors the device-move the orchestrator's r_i helper does for
        the 2a head, so the serve path does not have to remember to move.
        """
        dev = next(self.parameters()).device
        s = slots_y.to(torch.float32).to(dev)
        d = slots_doc_emb.to(torch.float32).to(dev)
        q = query_emb.to(torch.float32).to(dev)
        r = r.to(torch.float32).to(dev)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if d.dim() == 1:
            d = d.unsqueeze(0)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        rr = r
        if rr.dim() == 0:
            rr = rr.unsqueeze(0)
        if s.shape[0] != d.shape[0]:
            raise ValueError(
                f"ContextBuilder: slots_y rows {s.shape[0]} != slots_doc_emb "
                f"rows {d.shape[0]} -- the two slot tensors must be aligned"
            )
        if s.shape[0] != rr.shape[0]:
            raise ValueError(
                f"ContextBuilder: slots_y rows {s.shape[0]} != r length "
                f"{rr.shape[0]} -- r is one relevance per slot"
            )
        if s.shape[1] != self.slot_dim:
            raise ValueError(
                f"ContextBuilder: slot_y dim {s.shape[1]} != "
                f"self.slot_dim {self.slot_dim}"
            )
        if d.shape[1] != self.doc_dim:
            raise ValueError(
                f"ContextBuilder: slot_doc_emb dim {d.shape[1]} != "
                f"self.doc_dim {self.doc_dim}"
            )
        if q.shape[-1] != self.query_dim:
            raise ValueError(
                f"ContextBuilder: query dim {q.shape[-1]} != "
                f"self.query_dim {self.query_dim}"
            )
        return s, d, q, rr

    def logits(
        self,
        slots_y: Tensor,
        slots_doc_emb: Tensor,
        query_emb: Tensor,
        r: Tensor,
    ) -> Tensor:
        """Per-slot pre-sigmoid score ``s`` -> ``[K]`` (for BCEWithLogits).

        ``r`` is the per-slot 2a relevance ``r_i`` (the frozen head's output) --
        a constant input feature at train time, a live bias at serve.
        """
        s, d, q, rr = self._coerce(slots_y, slots_doc_emb, query_emb, r)
        k = self.W_k(s) + self.W_doc(d)                 # [K, d_head]
        v = self.W_v(s)                                 # [K, d_head]
        # cross-slot self-attention: [1, K, d_head] in -> [1, K, d_head] out.
        h, _ = self.cross_attn(k.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        h = h.squeeze(0)                                # [K, d_head]
        qh = self.W_q(q)                                # [1, d_head]
        sim = (h * qh).sum(-1) * self.scale             # [K]
        return sim + self.lambda_r * rr + self.bias.squeeze().expand_as(sim)

    def predict(
        self,
        slots_y: Tensor,
        slots_doc_emb: Tensor,
        query_emb: Tensor,
        r: Tensor,
        m: Optional[int] = None,
    ) -> tuple[list[int], Tensor]:
        """Top-m slot indices + their scores. ``m`` defaults to ``self.top_m``;
        clamped to ``K`` so a ring smaller than ``top_m`` does not crash.

        Returns ``(idx_list, scores)`` where ``idx_list`` is a Python list of
        ints (length ``min(m, K)``) in descending-score order and ``scores`` is
        the corresponding ``[min(m, K)]`` tensor.
        """
        s = self.logits(slots_y, slots_doc_emb, query_emb, r)   # [K]
        k = s.shape[0]
        mm = int(min(m if m is not None else self.top_m, k))
        if mm <= 0:
            return [], s.new_zeros(0)
        top = s.topk(mm)                                 # descending by default
        idx_list = [int(i) for i in top.indices.tolist()]
        return idx_list, top.values

    forward = logits

    # ── load ──

    @classmethod
    def from_state_dict(
        cls,
        sd: dict,
        slot_dim: int = SLOT_DIM,
        doc_dim: int = DOC_DIM,
        query_dim: int = QUERY_DIM,
        d_head: int = D_HEAD,
        num_heads: int = NUM_HEADS,
        top_m: int = TOP_M,
        lambda_init: float = LAMBDA_R_INIT,
    ) -> "ContextBuilder":
        """Build a builder and load a raw ``state_dict`` into it.

        Strict: a shape mismatch from a different slot_dim/doc_dim/query_dim/
        d_head/num_heads is a hard error, not a silent mis-wire.
        """
        builder = cls(
            slot_dim=slot_dim, doc_dim=doc_dim, query_dim=query_dim,
            d_head=d_head, num_heads=num_heads, top_m=top_m,
            lambda_init=lambda_init,
        )
        missing, unexpected = builder.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"context builder state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        builder.eval()
        return builder


def load_context_builder(
    path: str,
    device: str = "auto",
    map_location: str = "cpu",
) -> ContextBuilder:
    """Load a trained ContextBuilder checkpoint -> ready-to-serve module.

    The checkpoint is ``{"head": state_dict, "slot_dim": int, "doc_dim": int,
    "query_dim": int, "d_head": int, "num_heads": int, "top_m": int,
    "mean_coverage": float, "heuristic_mean_coverage": float, "go": bool, ...}``
    (see ``training.context_builder_training.fit_context_builder``). Like the
    2a/2b/2c loaders this takes NO backbone -- the builder reads ring slots + a
    query at serve. Dims are read before construction so a checkpoint fit on
    different dims loads into a matching builder. Moves to the resolved device,
    eval.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        slot_dim = int(ckpt.get("slot_dim", SLOT_DIM))
        doc_dim = int(ckpt.get("doc_dim", DOC_DIM))
        query_dim = int(ckpt.get("query_dim", QUERY_DIM))
        d_head = int(ckpt.get("d_head", D_HEAD))
        num_heads = int(ckpt.get("num_heads", NUM_HEADS))
        top_m = int(ckpt.get("top_m", TOP_M))
        sd = ckpt["head"] if "head" in ckpt else ckpt
    else:
        slot_dim, doc_dim, query_dim = SLOT_DIM, DOC_DIM, QUERY_DIM
        d_head, num_heads, top_m = D_HEAD, NUM_HEADS, TOP_M
        sd = ckpt
    builder = ContextBuilder.from_state_dict(
        sd, slot_dim=slot_dim, doc_dim=doc_dim, query_dim=query_dim,
        d_head=d_head, num_heads=num_heads, top_m=top_m,
    )
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return builder.to(dev).eval()