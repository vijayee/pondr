"""Phase 0b: a LEARNED readout over the raw SSM recurrent state.

Phase 0a (``scripts/probe_state_signal_distribution.py``, commit 62fcbdc,
[[pondr-strm-phase0a-state-signal-readout]]) showed the SSM recurrent state is
NOT collapsed: per-channel and flat representations vary 0.45-0.76x as much as
the doc embeddings across docs, while the FIXED mean-pool
``LatentDynamicsHead.project`` (mean over the 16 ``d_state`` channels) cancels
the opposing-sign signal to a near-constant 0.15x. The state carries
doc-identity VARIANCE; no single channel is pre-aligned with the bge query
(per-channel top-3 ~0.20-0.25), so a learned readout must MIX channels to find
a query-relevant direction. That is this module's job.

``StateReadout`` maps a flattened raw state (default ``z_flat_last`` [6144] =
the last SSM layer's 16 ``d_state`` channels x 384 ``d_model``) to a 384-d
``z_i`` that the existing ``ZRelevanceHead`` scores against the query. It is
either a single ``nn.Linear(dim_in, 384)`` (the strong "a linear readout
suffices" test) or a 2-layer MLP (``hidden`` set) if the linear form can't
recover the nonlinear ``slot.h = g*W_B(doc_bge)`` encoding.

``CompositeZHead`` wraps ``StateReadout + ZRelevanceHead`` and exposes the
3-arg ``logits(slot_y, slot_signal, query)`` signature
``fit_relevance`` calls, so the existing trainer
(``src/subconscious/training/relevance_training.py``) trains BOTH modules
end-to-end with only ``head=CompositeZHead(...)`` + ``slot_signal_field=
"slots_h_raw"`` -- no trainer change. The composite saves its ``dim_in`` under
``doc_dim`` (mirroring ``ZRelevanceHead``'s z_dim-as-doc_dim convention) so the
loader can reconstruct the readout input shape; the Linear-vs-MLP arch is
inferred from the state_dict keys (no extra checkpoint field needed).

This is the Phase 0b GATE: train the composite on the ERAG relevance labels
(the same labels 2a hit 0.889 / the mean-pool z-head hit 0.285) and check the
TRAIN gate (``mean_top3_recall >= 0.6``, ``hit_ci95[0] > 0.5``). GO -> a learned
readout recovers the signal the mean-pool destroyed -> re-run the SERVE gate
(``probe_strm_selectivity_real.py``) as acceptance. NO-GO -> even a learned
readout can't align the state to the query -> fall through to Phase 1 (backbone
fine-tune).

Local-only until the gate passes; the live orchestrator (``build_ponder`` /
``serve_ponder``) is never touched (binding constraint: don't break existing
functionality).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.subconscious.z_relevance_head import (  # noqa: E402
    PROJ_DIM as Z_PROJ_DIM,
    QUERY_DIM as Z_QUERY_DIM,
    SLOT_DIM as Z_SLOT_DIM,
    Z_DIM,
    ZRelevanceHead,
)

# Default raw-state input dim: the last SSM layer flattened, 16 d_state x 384
# d_model = 6144. (All 4 layers flattened = 24576 is the ``flat_all`` option;
# pass dim_in=24576 to the composite to use it.)
DEFAULT_DIM_IN = 16 * 384


class StateReadout(nn.Module):
    """Learned ``raw_state [dim_in] -> z_i [384]`` projection.

    ``hidden=None`` -> a single ``nn.Linear(dim_in, dim_out)`` (the "linear
    readout suffices" test). ``hidden=int`` -> a 2-layer MLP
    ``Linear(dim_in, hidden) -> ReLU -> Linear(hidden, dim_out)`` for the
    nonlinear ``g*W_B(doc_bge)`` encoding. ``forward`` casts to fp32 (the raw
    state is fp16 from Phase A's ``slot.h`` capture).
    """

    def __init__(self, dim_in: int, dim_out: int = Z_DIM, hidden: int | None = None) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.hidden = int(hidden) if hidden is not None else None
        if hidden is None:
            self.net = nn.Linear(self.dim_in, self.dim_out)
        else:
            self.net = nn.Sequential(
                nn.Linear(self.dim_in, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.dim_out),
            )

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)            # [B, dim_out]

    @classmethod
    def from_state_dict(cls, sd: dict, dim_in: int, dim_out: int = Z_DIM) -> "StateReadout":
        """Build a readout and load a raw ``state_dict`` (keys prefixed ``net.``).

        Linear-vs-MLP is inferred from the keys: ``net.2.weight`` present -> MLP
        with ``hidden = sd["net.0.weight"].shape[0]``; else Linear. Strict on
        missing/unexpected keys (a mis-wire is a hard error, not a silent load).
        Accepts either bare ``net.*`` keys (a standalone readout) or
        ``readout.net.*`` keys (a composite's readout slice) -- the
        ``readout.`` prefix is stripped BEFORE the MLP inference so the
        ``net.2.weight`` check sees the de-prefixed keys.
        """
        prefix = "readout."
        own = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        hidden = None
        if "net.2.weight" in own:
            hidden = int(own["net.0.weight"].shape[0])
        ro = cls(dim_in=dim_in, dim_out=dim_out, hidden=hidden)
        missing, unexpected = ro.load_state_dict(own, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"StateReadout state_dict mismatch: "
                f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
            )
        ro.eval()
        return ro


class CompositeZHead(nn.Module):
    """``StateReadout -> ZRelevanceHead`` composite, trained end-to-end.

    ``logits(slot_y, slot_signal, query)``: ``z_i = readout(slot_signal)`` then
    ``ZRelevanceHead.logits(slot_y, z_i, query)``. ``slot_signal`` is the raw
    flattened state (``slots_h_raw``, ``[K, dim_in]``); ``slot_y`` is accepted
    and ignored (the pure-z_i test, unchanged from ``ZRelevanceHead``). The
    composite exposes ``slot_dim``/``query_dim``/``proj_dim`` from the wrapped
    z-head and ``doc_dim = dim_in`` (the readout input dim) so
    ``fit_relevance``'s checkpoint-dim reads are shape-consistent and the loader
    can reconstruct the readout.
    """

    def __init__(self, dim_in: int = DEFAULT_DIM_IN, hidden: int | None = None) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.readout = StateReadout(dim_in=self.dim_in, hidden=hidden)
        self.z_head = ZRelevanceHead()
        # Mirror the z-head's checkpoint dims; doc_dim = the readout input dim
        # (so the loader rebuilds the readout with the right dim_in).
        self.slot_dim = self.z_head.slot_dim
        self.query_dim = self.z_head.query_dim
        self.proj_dim = self.z_head.proj_dim
        self.doc_dim = self.dim_in

    def logits(self, slot_y: Tensor, slot_signal: Tensor, query_emb: Tensor) -> Tensor:
        """Pre-sigmoid relevance logit -> ``[batch, 1]``.

        ``slot_signal`` is the raw flattened state ``[K, dim_in]`` (or
        ``[dim_in]``); ``slot_y`` is accepted and ignored (pure-z_i test).
        """
        z_i = self.readout(slot_signal)                 # [K, 384] or [1, 384]
        return self.z_head.logits(slot_y, z_i, query_emb)

    def predict(self, slot_y: Tensor, slot_signal: Tensor, query_emb: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(slot_y, slot_signal, query_emb))

    forward = predict

    @classmethod
    def from_state_dict(
        cls,
        sd: dict,
        dim_in: int,
        z_dim: int = Z_DIM,
        query_dim: int = Z_QUERY_DIM,
        proj_dim: int = Z_PROJ_DIM,
        slot_dim: int = Z_SLOT_DIM,
    ) -> "CompositeZHead":
        """Build a composite from a raw composite ``state_dict`` and load it.

        Keys are ``readout.*`` and ``z_head.*``. The readout's Linear-vs-MLP arch
        is inferred from ``readout.net.*`` keys. Strict on missing/unexpected.
        """
        # Infer readout hidden-ness from the readout.net keys.
        hidden = None
        if "readout.net.2.weight" in sd:
            hidden = int(sd["readout.net.0.weight"].shape[0])
        head = cls(dim_in=dim_in, hidden=hidden)
        # Rebind the z_head dims to the checkpoint's if they differ (defensive).
        head.z_head = ZRelevanceHead.from_state_dict(
            {k[len("z_head."):]: v for k, v in sd.items() if k.startswith("z_head.")},
            z_dim=z_dim, query_dim=query_dim, proj_dim=proj_dim, slot_dim=slot_dim,
        )
        readout_sd = {k: v for k, v in sd.items() if k.startswith("readout.")}
        head.readout = StateReadout.from_state_dict(readout_sd, dim_in=dim_in, dim_out=z_dim)
        head.eval()
        return head


def load_composite_z_head(
    path: str,
    device: str = "auto",
    map_location: str = "cpu",
) -> CompositeZHead:
    """Load a trained CompositeZHead checkpoint -> ready-to-serve head.

    The checkpoint is the shape ``fit_relevance`` writes
    (``{"head": composite_state_dict, "slot_dim", "doc_dim" (=dim_in),
    "query_dim", "proj_dim", ...}``). ``doc_dim`` is the readout input dim.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "head" in ckpt:
        dim_in = int(ckpt.get("doc_dim", DEFAULT_DIM_IN))
        query_dim = int(ckpt.get("query_dim", Z_QUERY_DIM))
        proj_dim = int(ckpt.get("proj_dim", Z_PROJ_DIM))
        slot_dim = int(ckpt.get("slot_dim", Z_SLOT_DIM))
        sd = ckpt["head"]
    else:
        # Bare state_dict fallback (not produced by fit_relevance; defensive).
        dim_in = DEFAULT_DIM_IN
        query_dim, proj_dim, slot_dim = Z_QUERY_DIM, Z_PROJ_DIM, Z_SLOT_DIM
        sd = ckpt
    head = CompositeZHead.from_state_dict(
        sd, dim_in=dim_in, query_dim=query_dim, proj_dim=proj_dim, slot_dim=slot_dim,
    )
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    return head.to(dev).eval()