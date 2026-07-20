"""ZRelevanceHead: probe head for the Phase B ``h_t`` relevance gate (STRM rewire).

The state-trajectory-transformer rewire (see
``docs/strm-transformer-relocator-plan.md`` and
[[pondr-strm-transformer-relocator-drift]]) hinges on ONE unverified premise:
that the projected SSM recurrent state ``z_i = LatentDynamicsHead.project(slot.h)``
(last layer, mean over ``d_state`` -> 384-d) carries query-relevance signal that
the 2a head's ``y_t`` readout did NOT. Probe 4a killed the ``y_t`` path
(``s_i_pure`` gap ~0); the ``h_t`` path was never tested. This head IS that test,
and it is the cheap GATE 1 before any transformer build.

It is a pure-``z_i`` bilinear, the ``z_i`` analog of the 2a ``RelevanceHead``'s
bilinear term with two deliberate differences:

  1. **The slot signal is ``z_i`` (SSM state, 384-d), NOT ``doc_emb`` (bge).**
     The 2a head's signal lives in the bge doc embedding (top-3 recall 0.889 on
     ERAG); ``z_i`` has no bge in it -- it is the frozen routing-trained
     backbone's recurrent state. So this head's train task is strictly harder
     than 2a's: align the SSM state space to query-relevance. If ``z_i`` carries
     NO relevance signal (the backbone is routing-trained, not relevance-trained)
     this head fails the TRAIN gate too -- a cheap NO-GO with no serve probe
     needed. If it clears train, the SERVE probe (Probe 4a harness extended with
     the z_i head) is still required (train success != serve transfer, as 2a
     showed: 0.889 train, saturates serve).

  2. **There is NO ``yt_sidepath`` -- the ``y_t`` path is removed entirely.**
     Probe 4a's ablation showed ``yt_sidepath`` (OOD at serve) drowns the bilinear
     signal; this head drops it so the test is pure. ``slot_y`` is accepted by
     ``logits`` and IGNORED -- the ignored ``slot_y`` IS the point of the
     "pure ``z_i``" test. It is accepted only for signature compatibility with
     ``fit_relevance``'s ``head.logits(slots, slot_signal, query)`` call so the
     2a trainer is reused verbatim (only the slot-signal FIELD changes,
     ``slots_doc_emb`` -> ``slots_z``).

DUAL projection, not shared. ``z_i`` (SSM state) and the query (bge) live in
DIFFERENT spaces, unlike 2a's doc+query (both bge, shared projection, no rotation
gauge-freedom). A single shared projection over different spaces constrains the
bilinear to PSD (``M = W^T W``); a DUAL projection (``proj_z``, ``proj_q``) gives
a general bilinear ``z_i^T (W_z^T W_q) query`` -- the most-expressive instance,
chosen so a NO-GO is a property of the SIGNAL, not the arch. (A false NO-GO from
an arch constraint would wrongly kill the state-trajectory path.)

::

    score = (proj_z(z_i) . proj_q(query)) / sqrt(P) + bias
    z_r_i = sigmoid(score)

This is a PROBE head, not the final transformer. Phase B GATE 1: the z_i-head's
serve selectivity gap (probe turn minus mean filler, on real Onyx transcripts)
median >= 0.2 in >= 3/4 runs vs 2a's ~0 on the same slots. GO -> ``h_t`` carries
the signal, proceed to the transformer rewire (Phases C-F). NO-GO -> ``h_t`` does
NOT carry it either; stop, do not build the transformer, document that the
vision's state-locator is dead on ``h_t`` too (likely a backbone problem, not a
readout problem).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# z_i = LatentDynamicsHead.project(slot.h): last SSM layer, mean over d_state=16
# -> [384]. Matches latent_dynamics_head.STATE_DIM (the 0b-validated "last" rep).
Z_DIM = 384
# The query embedding is the raw bge-small vector, 384-d (same as 2a's QUERY_DIM).
QUERY_DIM = 384
# The bilinear projection width (z and query both project to P-d, then dot).
PROJ_DIM = 128
# slot_y (the WM recurrent readout, 256-d) is ACCEPTED and IGNORED -- exposed so
# the shared trainer's checkpoint dims are shape-consistent. Matches 2a SLOT_DIM.
SLOT_DIM = 256


class ZRelevanceHead(nn.Module):
    """Pure-``z_i`` bilinear probe: ``score = (proj_z(z) . proj_q(q))/sqrt(P) + bias``.

    A dual projection (``proj_z`` 384->P, ``proj_q`` 384->P) maps the SSM state
    ``z_i`` and the bge query into a shared P-d space whose dot product (scaled
    by ``1/sqrt(P)``) is the learned cross-space relevance similarity. No
    ``yt_sidepath``: the ``y_t`` path is dropped (the pure-``z_i`` test).
    ``slot_y`` is accepted by :meth:`logits` and ignored -- signature-compatible
    with ``fit_relevance``'s 3-arg ``head.logits(slots, slot_signal, query)`` call
    so the 2a trainer is reused verbatim. The head is query-conditioned: the SAME
    slot scores differently against different queries.
    """

    def __init__(
        self,
        z_dim: int = Z_DIM,
        query_dim: int = QUERY_DIM,
        proj_dim: int = PROJ_DIM,
        slot_dim: int = SLOT_DIM,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.query_dim = int(query_dim)
        self.proj_dim = int(proj_dim)
        self.slot_dim = int(slot_dim)
        # Exposed as ``doc_dim`` so the shared trainer's checkpoint dim keys are
        # shape-consistent and the loader can reconstruct (doc_dim == z_dim).
        self.doc_dim = int(z_dim)
        self.proj_z = nn.Linear(self.z_dim, self.proj_dim)
        self.proj_q = nn.Linear(self.query_dim, self.proj_dim)
        self.scale = 1.0 / math.sqrt(self.proj_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    # ── prediction ──

    def _broadcast(self, slot_z: Tensor, query_emb: Tensor) -> tuple[Tensor, Tensor]:
        """Coerce z_i + query to a common 2-D batch, broadcasting the single-row
        side (the serve pattern: one query, K ring slots)."""
        z = slot_z.to(torch.float32)
        q = query_emb.to(torch.float32)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        bsz = max(z.shape[0], q.shape[0])
        for name, t in (("slot_z", z), ("query_emb", q)):
            if t.shape[0] not in (1, bsz):
                raise ValueError(
                    f"ZRelevanceHead: {name} batch {t.shape[0]} is incompatible "
                    f"with the other (max {bsz}) -- pass one query "
                    f"([query_dim]) with K slots ([K, z_dim]) to broadcast"
                )
        z = z.expand(bsz, -1)
        q = q.expand(bsz, -1)
        if z.shape[1] != self.z_dim:
            raise ValueError(
                f"ZRelevanceHead: slot_z dim {z.shape[1]} != self.z_dim {self.z_dim}"
            )
        if q.shape[1] != self.query_dim:
            raise ValueError(
                f"ZRelevanceHead: query dim {q.shape[1]} != "
                f"self.query_dim {self.query_dim}"
            )
        return z, q

    def logits(self, slot_y: Tensor, slot_z: Tensor,
               query_emb: Tensor) -> Tensor:
        """Pre-sigmoid z_i relevance logit -> ``[batch, 1]`` (for BCEWithLogits).

        ``slot_y`` is IGNORED by design -- the pure-``z_i`` test drops the
        ``y_t`` path. It is accepted only for signature compatibility with the 2a
        trainer (``head.logits(slots, slot_signal, query)``), so ``fit_relevance``
        is reused verbatim with the slot-signal field swapped to ``slots_z``.
        """
        del slot_y  # unused by design (the pure-z_i test drops the y_t path)
        z, q = self._broadcast(slot_z, query_emb)
        zp = self.proj_z(z)                              # [B, P]
        qp = self.proj_q(q)                              # [B, P]
        sim = (zp * qp).sum(-1, keepdim=True) * self.scale   # [B, 1]
        return sim + self.bias

    def predict(self, slot_y: Tensor, slot_z: Tensor,
                query_emb: Tensor) -> Tensor:
        """``z_r_i = sigmoid(logits(...))`` -> ``[batch, 1]``. The natural serve
        pattern is ONE query and K ring slots: pass ``slot_y`` as ``[K, 256]``
        (ignored), ``slot_z`` as ``[K, 384]``, ``query_emb`` as ``[384]`` (1-D)."""
        return torch.sigmoid(self.logits(slot_y, slot_z, query_emb))

    forward = predict

    # ── load ──

    @classmethod
    def from_state_dict(
        cls,
        sd: dict,
        z_dim: int = Z_DIM,
        query_dim: int = QUERY_DIM,
        proj_dim: int = PROJ_DIM,
        slot_dim: int = SLOT_DIM,
    ) -> "ZRelevanceHead":
        """Build a head and load a raw ``state_dict`` into it.

        Strict: a shape mismatch from a different z_dim/query_dim/proj_dim is a
        hard error, not a silent mis-wire.
        """
        head = cls(z_dim=z_dim, query_dim=query_dim, proj_dim=proj_dim,
                   slot_dim=slot_dim)
        missing, unexpected = head.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"z-relevance head state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        head.eval()
        return head


def load_z_relevance_head(
    path: str,
    device: str = "auto",
    map_location: str = "cpu",
) -> ZRelevanceHead:
    """Load a trained ZRelevanceHead checkpoint -> ready-to-serve probe head.

    The checkpoint shape is the same as ``load_relevance_head``'s
    (``{"head": state_dict, "slot_dim", "doc_dim" (=z_dim), "query_dim",
    "proj_dim", ...}`` -- see ``relevance_training.fit_relevance``). ``doc_dim``
    is read as ``z_dim`` (the head saves its z_dim under the ``doc_dim`` key for
    shape-consistency with the shared trainer). Moves to the resolved device, eval.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        z_dim = int(ckpt.get("doc_dim", Z_DIM))     # saved as doc_dim (== z_dim)
        query_dim = int(ckpt.get("query_dim", QUERY_DIM))
        proj_dim = int(ckpt.get("proj_dim", PROJ_DIM))
        slot_dim = int(ckpt.get("slot_dim", SLOT_DIM))
        sd = ckpt["head"] if "head" in ckpt else ckpt
    else:
        z_dim, query_dim, proj_dim, slot_dim = Z_DIM, QUERY_DIM, PROJ_DIM, SLOT_DIM
        sd = ckpt
    head = ZRelevanceHead.from_state_dict(
        sd, z_dim=z_dim, query_dim=query_dim, proj_dim=proj_dim, slot_dim=slot_dim,
    )
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return head.to(dev).eval()