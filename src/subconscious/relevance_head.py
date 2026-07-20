"""RelevanceHead: per-slot query-conditioned relevance ``r_i in [0,1]``.

STRM Phase 2a -- the relevance read-out. Given a query and the WM ring's
per-slot step outputs ``y_t`` (the learned ``output_proj`` readout, 256-d --
the same signal ``RetrievalGate`` / ``DocKindHead`` classify on) PLUS the slot's
raw bge-small doc embedding (384-d, the step INPUT -- the doc's semantic
identity), it scores each slot's relevance to the query as a scalar
``r_i in [0,1]``. Phase 3's context-builder consumes ``r_i`` as the bias term
that selects which recent slots to surface in the prompt; Phase 4's salience
trigger combines it with the 2b recoverability signal.

Design deviation from the STRM spec (documented up front): the spec says
``RelevanceHead(JGSInstance)`` "modeled on DocKindHead". DocKindHead is a
``JGSInstance`` because it STEPS the SSM at serve (it ingests a fresh doc's
sections). The relevance head scores slots ALREADY in the WM ring against a
query -- it does NOT step the SSM. It is structurally a read-out head like the
shipped 2b recoverability + 2c latent-dynamics heads, which are plain
``nn.Module`` (NOT ``JGSInstance``, NOT registered in ``INSTANCE_CONFIGS``).
For consistency with that shipped pattern and to avoid a pointless SSM-coupling,
``RelevanceHead`` is a plain ``nn.Module`` reading ``(slot_y_t, slot_doc_emb,
query_emb)``.

Architecture -- LEARNED COSINE via a SHARED projection, NOT a concat MLP and
NOT a dual-tower. Relevance is a similarity/ranking task (``doc`` close to
``query``), not a classification task. An additive MLP ``w_d . doc + w_q . query
+ b`` CANNOT represent cosine (a multiplicative ``doc . query`` interaction):
on the real ERAG slice a ``[y_t; doc_emb; query_emb] = 1024 -> 128 -> 1`` MLP
tops out at top-3 recall 0.60 over 30 epochs / 376 train queries, while raw bge
cosine clears the gate at 1.00 (see ``scripts/_scratch/_probe_relevance_bge_baseline``).
A dual-tower (separate ``doc_proj``/``query_proj``) CAN represent cosine but
optimizes poorly: the two towers have a rotation gauge-freedom
(``doc_proj=R.A, query_proj=R.B`` leaves the dot ``A.B`` invariant for any
rotation ``R``), so on the similarity synthetic it plateaus at top-3 recall
0.58. Because the doc and the query are BOTH bge-small vectors in the SAME
384-d space, a SINGLE shared projection is better-conditioned (no gauge
freedom) and represents cosine exactly (``proj = I``). The head is therefore:

    score = (proj(doc_emb) . proj(query_emb)) / sqrt(P)
            + yt_sidepath(y_t) + bias
    r_i   = sigmoid(score)

``proj`` (384 -> P=128) maps the doc and the query into one shared space whose
dot product is the learned relevance similarity; with ``proj ~ I`` this is
bge cosine (which clears the gate at 1.00). ``yt_sidepath`` (a small
``Linear(256,64) -> GELU -> Linear(64,1)``) reads the WM recurrent readout
``y_t`` and adds a scalar -- this honors the STRM intent (the head IS a
WM-ring-slot read-out head, reading the recurrent state) and lets a future
backbone fine-tune (one that makes ``y_t`` carry relevance) activate that path;
on the frozen routing-trained backbone ``y_t`` carries no relevance signal, so
this sidepath learns near-zero and the shared-projection term carries the
signal.

Training is supervised BCE on the ERAG-Bench traces
(``training/relevance_training.py``): a slot produced from a gold doc is
positive, a slot from a sampled non-gold doc is negative. The gate is per-query
top-3 recall + a Wilson 95% CI lower bound.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# The ring slot y_t = the step OUTPUT (output_proj readout), 256-d. Matches
# INSTANCE_CONFIGS["working_memory"].output_dim (configs.py).
SLOT_DIM = 256
# The slot's RAW bge-small doc embedding (the step INPUT), 384-d -- the doc's
# semantic identity. This is where the relevance signal lives (the frozen
# routing-trained backbone's y_t readout does not preserve it).
DOC_DIM = 384
# The query embedding is the RAW bge-small vector, 384-d.
QUERY_DIM = 384
# The bilinear projection width (doc and query both project to P-d, then dot).
PROJ_DIM = 128


class RelevanceHead(nn.Module):
    """Per-slot ``r_i = sigmoid(shared_proj(doc, query) + yt_sidepath(y_t))``.

    A single ``proj`` (384 -> P) maps the slot's doc embedding and the query
    embedding into one shared space; their dot product (scaled by
    ``1/sqrt(P)``) is the learned relevance similarity -- with ``proj ~ I`` this
    is bge cosine. ``yt_sidepath`` adds a scalar read of the WM recurrent state
    ``y_t`` (dormant on the frozen routing-trained backbone; activates under a
    future backbone fine-tune). The head is query-conditioned: the SAME slot
    scores differently against different queries (the query is an input, not a
    parameter).
    """

    def __init__(
        self,
        slot_dim: int = SLOT_DIM,
        doc_dim: int = DOC_DIM,
        query_dim: int = QUERY_DIM,
        proj_dim: int = PROJ_DIM,
    ) -> None:
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.doc_dim = int(doc_dim)
        self.query_dim = int(query_dim)
        self.proj_dim = int(proj_dim)
        # doc and query are both bge-small vectors in the SAME 384-d space, so
        # one shared projection (not a dual-tower) -- better-conditioned, no
        # rotation gauge-freedom, represents cosine exactly at proj ~ I.
        if doc_dim != query_dim:
            raise ValueError(
                f"RelevanceHead: shared projection requires doc_dim == query_dim "
                f"(got doc_dim={doc_dim} query_dim={query_dim})"
            )
        self.proj = nn.Linear(self.doc_dim, self.proj_dim)
        self.yt_sidepath = nn.Sequential(
            nn.Linear(self.slot_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.scale = 1.0 / math.sqrt(self.proj_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    # ── prediction ──

    def _broadcast(self, slot_y: Tensor, slot_doc_emb: Tensor,
                   query_emb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Coerce the three inputs to a common 2-D batch, broadcasting the
        single-row side (the serve pattern: one query, K ring slots)."""
        s = slot_y.to(torch.float32)
        d = slot_doc_emb.to(torch.float32)
        q = query_emb.to(torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if d.dim() == 1:
            d = d.unsqueeze(0)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        bsz = max(s.shape[0], d.shape[0], q.shape[0])
        for name, t in (("slot_y", s), ("slot_doc_emb", d), ("query_emb", q)):
            if t.shape[0] not in (1, bsz):
                raise ValueError(
                    f"RelevanceHead: {name} batch {t.shape[0]} is incompatible "
                    f"with the others (max {bsz}) -- pass one query "
                    f"([query_dim]) with K slots ([K, slot_dim]/[K, doc_dim]) "
                    f"to broadcast"
                )
        s = s.expand(bsz, -1)
        d = d.expand(bsz, -1)
        q = q.expand(bsz, -1)
        if s.shape[1] != self.slot_dim:
            raise ValueError(
                f"RelevanceHead: slot_y dim {s.shape[1]} != "
                f"self.slot_dim {self.slot_dim}"
            )
        if d.shape[1] != self.doc_dim:
            raise ValueError(
                f"RelevanceHead: slot_doc_emb dim {d.shape[1]} != "
                f"self.doc_dim {self.doc_dim}"
            )
        if q.shape[1] != self.query_dim:
            raise ValueError(
                f"RelevanceHead: query dim {q.shape[1]} != "
                f"self.query_dim {self.query_dim}"
            )
        return s, d, q

    def logits(self, slot_y: Tensor, slot_doc_emb: Tensor,
               query_emb: Tensor) -> Tensor:
        """Pre-sigmoid relevance logit -> ``[batch, 1]`` (for BCEWithLogits)."""
        s, d, q = self._broadcast(slot_y, slot_doc_emb, query_emb)
        dp = self.proj(d)                    # [B, P]
        qp = self.proj(q)                    # [B, P]
        sim = (dp * qp).sum(-1, keepdim=True) * self.scale   # [B, 1]
        yt = self.yt_sidepath(s)             # [B, 1]
        return sim + yt + self.bias

    def predict(self, slot_y: Tensor, slot_doc_emb: Tensor,
                query_emb: Tensor) -> Tensor:
        """``r_i = sigmoid(logits(...))`` -> ``[batch, 1]``, each row in
        ``[0, 1]``. The natural serve pattern is ONE query and K ring slots:
        pass ``slot_y`` as ``[K, 256]``, ``slot_doc_emb`` as ``[K, 384]``, and
        ``query_emb`` as ``[384]`` (1-D) and the query is broadcast one row
        per slot."""
        return torch.sigmoid(self.logits(slot_y, slot_doc_emb, query_emb))

    forward = predict

    # ── load ──

    @classmethod
    def from_state_dict(
        cls,
        sd: dict,
        slot_dim: int = SLOT_DIM,
        doc_dim: int = DOC_DIM,
        query_dim: int = QUERY_DIM,
        proj_dim: int = PROJ_DIM,
    ) -> "RelevanceHead":
        """Build a head and load a raw ``state_dict`` into it.

        Strict: a shape mismatch from a different slot_dim/doc_dim/query_dim/
        proj_dim is a hard error, not a silent mis-wire.
        """
        head = cls(slot_dim=slot_dim, doc_dim=doc_dim, query_dim=query_dim,
                   proj_dim=proj_dim)
        missing, unexpected = head.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"relevance head state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        head.eval()
        return head


def load_relevance_head(
    path: str,
    device: str = "auto",
    map_location: str = "cpu",
) -> RelevanceHead:
    """Load a trained RelevanceHead checkpoint -> ready-to-serve module.

    The checkpoint is ``{"head": state_dict, "slot_dim": int, "doc_dim": int,
    "query_dim": int, "proj_dim": int, "top3_recall": float, "go": bool,
    ...}`` (see ``relevance_training.fit_relevance``). Like the 2b/2c loaders
    this takes NO backbone -- the head reads a ring slot + a query at serve.
    ``slot_dim``/``doc_dim``/``query_dim``/``proj_dim`` are read before
    construction so a checkpoint fit on different dims loads into a matching
    head. Moves to the resolved device, eval.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        slot_dim = int(ckpt.get("slot_dim", SLOT_DIM))
        doc_dim = int(ckpt.get("doc_dim", DOC_DIM))
        query_dim = int(ckpt.get("query_dim", QUERY_DIM))
        proj_dim = int(ckpt.get("proj_dim", PROJ_DIM))
        sd = ckpt["head"] if "head" in ckpt else ckpt
    else:
        slot_dim, doc_dim, query_dim, proj_dim = SLOT_DIM, DOC_DIM, QUERY_DIM, PROJ_DIM
        sd = ckpt
    head = RelevanceHead.from_state_dict(
        sd, slot_dim=slot_dim, doc_dim=doc_dim, query_dim=query_dim,
        proj_dim=proj_dim,
    )
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return head.to(dev).eval()