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

    Phase 1f-7 per-doc-kind readout (``n_doc_kinds > 0``): a SHARED
    ``body = Linear(dim_in, hidden) -> ReLU`` (learned once on the full mixed
    ring) feeds ``kind_heads = ModuleList([Linear(hidden, dim_out) for _ in
    range(n_doc_kinds)])``, routed by a ``doc_kinds: [B] long`` tensor. Each
    kind gets its own query-direction rotation; the shared body controls
    overfit on the ~400 code slots. ``n_doc_kinds=0`` (default) builds the
    ORIGINAL single-shared ``net = Sequential(Linear, ReLU, Linear)`` so the
    state_dict keys (``net.0.*`` / ``net.2.*``) are byte-identical and old
    ``best.pt``/``final.pt`` strict-load -- the binding check. The per-kind
    arch uses a SEPARATE key namespace (``body.0.*`` / ``kind_heads.{k}.*``)
    so the two never collide.

    Phase 1f-7 Stage 1 REDESIGN -- MoE on non-overlapping data
    (``per_kind_full=True``): N INDEPENDENT full readouts
    ``kind_readouts = ModuleList([Sequential(Linear(dim_in, hidden), ReLU,
    Linear(hidden, dim_out)) for _ in range(n_doc_kinds)])`` -- NO shared body,
    so the code readout never shares params with the conv-majority body (the
    Stage 1 FAIL root cause: the shared body was conv-dominated). Each kind's
    readout trains ONLY on its own kind's gold (the trainer routes ALL slots of
    a record through the GOLD's readout, so gradient flows into one readout per
    record = non-overlapping data, mirroring the Stage 0 code-only win per
    kind). Separate key namespace ``kind_readouts.{k}.0.*`` / ``.{k}.2.*`` so
    ``from_state_dict`` can distinguish it from the shared-body arch. Serve
    routes per-SLOT (unchanged) -- the live gate groups by source_id (one kind
    per source) so the probe-vs-filler gap is within one readout = well-defined.
    """

    def __init__(self, dim_in: int, dim_out: int = Z_DIM, hidden: int | None = None,
                 n_doc_kinds: int = 0, per_kind_full: bool = False) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.hidden = int(hidden) if hidden is not None else None
        self.n_doc_kinds = int(n_doc_kinds)
        self.per_kind_full = bool(per_kind_full)
        if self.per_kind_full:
            # MoE: N independent full readouts, no shared body. Needs both
            # n_doc_kinds>0 and hidden (each readout is the Stage 0 mlp128).
            if self.n_doc_kinds <= 0:
                raise ValueError("per_kind_full requires n_doc_kinds>0")
            if hidden is None:
                raise ValueError("per_kind_full requires hidden (per-kind MLP body)")
            self.kind_readouts = nn.ModuleList(
                [nn.Sequential(nn.Linear(self.dim_in, self.hidden),
                               nn.ReLU(),
                               nn.Linear(self.hidden, self.dim_out))
                 for _ in range(self.n_doc_kinds)]
            )
        elif self.n_doc_kinds > 0 and hidden is None:
            # The shared-body per-kind arch needs a body to feed the heads;
            # a pure-linear per-kind readout is degenerate (no shared depth).
            raise ValueError("n_doc_kinds>0 requires hidden (shared MLP body)")
        elif self.n_doc_kinds > 0:
            # Shared body 6144 -> hidden (ReLU); per-kind head hidden -> dim_out.
            self.body = nn.Sequential(
                nn.Linear(self.dim_in, self.hidden),
                nn.ReLU(),
            )
            self.kind_heads = nn.ModuleList(
                [nn.Linear(self.hidden, self.dim_out) for _ in range(self.n_doc_kinds)]
            )
        elif hidden is None:
            self.net = nn.Linear(self.dim_in, self.dim_out)
        else:
            self.net = nn.Sequential(
                nn.Linear(self.dim_in, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.dim_out),
            )

    def forward(self, x: Tensor, doc_kinds: Tensor | None = None) -> Tensor:
        x = x.to(torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.n_doc_kinds <= 0:
            return self.net(x)            # [B, dim_out] (byte-identical path)
        if doc_kinds is None:
            raise RuntimeError(
                "StateReadout(n_doc_kinds>0).forward requires a doc_kinds tensor"
            )
        dk = doc_kinds.to(x.device).long()
        if self.per_kind_full:
            # MoE: route each row to its kind's FULL readout (no shared body).
            # new_zeros (not new_empty) so any out-of-[0,n) kind yields zeros
            # rather than garbage -- a robustness guard.
            out = x.new_zeros(x.shape[0], self.dim_out)
            for k in range(self.n_doc_kinds):
                mask = dk == k
                if mask.any():
                    out[mask] = self.kind_readouts[k](x[mask])
            return out                    # [B, dim_out]
        # Shared-body per-doc-kind: body once, then route each row to its head.
        h = self.body(x)                  # [B, hidden] (ReLU)
        out = x.new_empty(x.shape[0], self.dim_out)
        for k in range(self.n_doc_kinds):
            mask = dk == k
            if mask.any():
                out[mask] = self.kind_heads[k](h[mask])
        return out                        # [B, dim_out]

    @classmethod
    def from_state_dict(cls, sd: dict, dim_in: int, dim_out: int = Z_DIM) -> "StateReadout":
        """Build a readout and load a raw ``state_dict`` (keys prefixed ``net.``).

        Linear-vs-MLP is inferred from the keys: ``net.2.weight`` present -> MLP
        with ``hidden = sd["net.0.weight"].shape[0]``; else Linear. The
        per-doc-kind arch is inferred from ``kind_heads.{k}.weight`` keys
        (n_doc_kinds = the count of distinct kind_heads.{k}.weight). The MoE
        ``per_kind_full`` arch is inferred from ``kind_readouts.{k}.0.weight``
        keys (detected FIRST, since its ``kind_readouts.*`` namespace is
        distinct from the shared-body ``kind_heads.*``). Strict on
        missing/unexpected keys (a mis-wire is a hard error, not a silent load).
        Accepts either bare ``net.*`` keys (a standalone readout) or
        ``readout.net.*`` keys (a composite's readout slice) -- the
        ``readout.`` prefix is stripped BEFORE the key inference so the
        ``net.2.weight`` / ``kind_heads.0.weight`` checks see the de-prefixed
        keys.
        """
        prefix = "readout."
        own = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        # MoE per_kind_full FIRST (kind_readouts.* namespace is distinct from
        # the shared-body kind_heads.* namespace, so order is unambiguous).
        readout_keys = [k for k in own
                        if k.startswith("kind_readouts.") and k.endswith(".0.weight")]
        if readout_keys:
            per_kind_full = True
            n_doc_kinds = len({k.split(".")[1] for k in readout_keys})
            hidden = int(own["kind_readouts.0.0.weight"].shape[0])
        else:
            per_kind_full = False
            n_doc_kinds = 0
            kind_keys = [k for k in own if k.startswith("kind_heads.") and k.endswith(".weight")]
            if kind_keys:
                n_doc_kinds = len({k.split(".")[1] for k in kind_keys})
            hidden = None
            if n_doc_kinds > 0:
                hidden = int(own["body.0.weight"].shape[0])
            elif "net.2.weight" in own:
                hidden = int(own["net.0.weight"].shape[0])
        ro = cls(dim_in=dim_in, dim_out=dim_out, hidden=hidden,
                 n_doc_kinds=n_doc_kinds, per_kind_full=per_kind_full)
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

    def __init__(self, dim_in: int = DEFAULT_DIM_IN, hidden: int | None = None,
                 n_doc_kinds: int = 0, per_kind_full: bool = False) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.n_doc_kinds = int(n_doc_kinds)
        self.per_kind_full = bool(per_kind_full)
        self.readout = StateReadout(dim_in=self.dim_in, hidden=hidden,
                                    n_doc_kinds=self.n_doc_kinds,
                                    per_kind_full=self.per_kind_full)
        self.z_head = ZRelevanceHead()
        # Mirror the z-head's checkpoint dims; doc_dim = the readout input dim
        # (so the loader rebuilds the readout with the right dim_in).
        self.slot_dim = self.z_head.slot_dim
        self.query_dim = self.z_head.query_dim
        self.proj_dim = self.z_head.proj_dim
        self.doc_dim = self.dim_in

    def logits(self, slot_y: Tensor, slot_signal: Tensor, query_emb: Tensor,
               slot_doc_kinds: Tensor | None = None) -> Tensor:
        """Pre-sigmoid relevance logit -> ``[batch, 1]``.

        ``slot_signal`` is the raw flattened state ``[K, dim_in]`` (or
        ``[dim_in]``); ``slot_y`` is accepted and ignored (pure-z_i test).
        ``slot_doc_kinds`` routes each slot to its per-kind readout head when
        ``n_doc_kinds>0`` (Phase 1f-7); ignored when ``n_doc_kinds=0``.
        """
        z_i = self.readout(slot_signal, doc_kinds=slot_doc_kinds)  # [K, 384] or [1, 384]
        return self.z_head.logits(slot_y, z_i, query_emb)

    def predict(self, slot_y: Tensor, slot_signal: Tensor, query_emb: Tensor,
                slot_doc_kinds: Tensor | None = None) -> Tensor:
        return torch.sigmoid(self.logits(slot_y, slot_signal, query_emb, slot_doc_kinds))

    forward = predict

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
        is inferred from ``readout.net.*`` keys; the per-doc-kind arch is
        inferred from ``readout.kind_heads.{k}.weight`` keys (n_doc_kinds = the
        count of distinct kinds). Strict on missing/unexpected.
        """
        # Infer readout hidden-ness + per-doc-kind from the readout keys.
        # MoE per_kind_full FIRST (kind_readouts.* namespace is distinct from
        # the shared-body kind_heads.* namespace).
        readout_keys = [k for k in sd
                        if k.startswith("readout.kind_readouts.")
                        and k.endswith(".0.weight")]
        if readout_keys:
            per_kind_full = True
            n_doc_kinds = len({k.split(".")[2] for k in readout_keys})
            hidden = int(sd["readout.kind_readouts.0.0.weight"].shape[0])
        else:
            per_kind_full = False
            n_doc_kinds = 0
            kind_keys = [k for k in sd
                        if k.startswith("readout.kind_heads.") and k.endswith(".weight")]
            if kind_keys:
                n_doc_kinds = len({k.split(".")[2] for k in kind_keys})
            hidden = None
            if n_doc_kinds > 0:
                hidden = int(sd["readout.body.0.weight"].shape[0])
            elif "readout.net.2.weight" in sd:
                hidden = int(sd["readout.net.0.weight"].shape[0])
        head = cls(dim_in=dim_in, hidden=hidden, n_doc_kinds=n_doc_kinds,
                   per_kind_full=per_kind_full)
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