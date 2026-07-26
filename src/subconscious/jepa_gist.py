"""JEPA-latent gist objective: predict the bge latent of the gist from the SSM state.

The pivot from the failed summary-CE retrain (``docs/gist-retrain-result.md``). That
retrain proved the plain summary objective does NOT shape gist-recoverable content
into the SSM state: the likelihood-swap discrimination margin was -0.003 nats
(unchanged from the frozen probe's 0.000), the decoder learned to IGNORE the state
(state-vs-zero gap -0.037), and the encoder collapsed (continuation val ppl 175 ->
1419, 8.1x). The failure mode is pinned: **summary-CE lets the decoder SHORTCUT to
the marginal summary distribution** (most summary tokens are generic structure) and
ignore the state; the encoder collapses its prior WITHOUT gaining gist.

This module implements the prescribed fix: predict the *latent* of the gist, not the
tokens. A lossy latent-space prediction is intrinsically gist-shaped and has no
generic-token prior to shortcut to; the contrastive in-batch negatives make
doc-specificity the explicit objective (``jepa_loss.jepa_contrastive_loss`` -- the
prediction must be closer to the true gist's bge embedding than to other docs', so
the state is the only source of doc-specific signal). This is the same objective
family as the planned Stage-2 JEPA-fade (predict the latent of OLDER content from
the state, with recency as the prediction horizon) -- so Stage 1 is the first half
of Stage 2, not a throwaway.

Architecture (encoder + latent head, state-seeded):

    encoder = SSMLanguageModel          # the token-LM (text-as-state); frozen for the
                                       # probe-style path, UNFROZEN for the retrain
    doc_ids -> encoder.forward -> (logits, enc_states)   # per-layer [b, d_state, d_model_enc]
    pool(enc_states) -> [b, n_layers * d_model_enc]      # mean over d_state, concat layers
    predictor : MLP(pool -> latent_dim=384) -> L2-normalize
    target = bge_small.encode(gist_text)                 # 384-d, FROZEN, L2-normalized
    loss = jepa_contrastive_loss(pred, target, negatives) + lm_prior_weight * next_token_CE

The encoder is the token-LM (d_model=256, 6 layers, d_state=16) -- NOT the bge
``JGSBackbone`` (d_model=384). The fade vision is over a TOKEN stream ("text as
state"; verbatim-recall-of-recent-tokens -> gist-of-older-tokens); the bge backbone
operates on pre-embedded chunks (no token stream, no verbatim recall) and is the
wrong foundation for Stage 2. The cost of the token-LM encoder is a small predictor
projection 256*6 -> 384; the existing ``JEPAPredictor``/EMA machinery in
``backbone.py``/``pretrain.py`` was built for the 384-d backbone and is NOT reused --
only the dim-agnostic ``jepa_loss`` is reused.

The **LM-prior auxiliary** (next-token CE on the doc, weight
``--lm-prior-weight``) is the anti-collapse fix the failed retrain lacked: it
directly penalizes the continuation-prior blowup that went 8.1x last time. No EMA
target encoder -- the bge target is already frozen (cannot collapse), and the
contrastive in-batch negatives are the anti-collapse mechanism (a constant
prediction is equally far from all negatives and the positive, failing the
objective).

Standalone: this does NOT touch the bge ``WorkingMemory`` / ``JGSBackbone`` /
``serve_ponder`` path, and it leaves the failed token-decode ``gist_readout`` path
in place (the harness of record for the probe). Wiring into the orchestrator is a
gated follow-on, only after the gate PASSES.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .token_lm import LMConfig, SSMLanguageModel
from .tokenizer_ import TokenizerWrapper


@dataclass
class LatentPredictorConfig:
    """Architecture hyperparameters for the gist-latent predictor head.

    The predictor pools the encoder's per-layer recurrent states (mean over the
    ``d_state`` channel per layer, all ``n_layers`` concatenated ->
    ``[b, n_layers * d_model_enc]``) and maps them to a ``latent_dim``-d L2-
    normalized vector in the frozen teacher's (bge-small) meaning space. ``pool``
    is currently fixed to ``"mean_d_state_concat_layers"`` (the 0a-validated
    representation, generalized from 4x384 to ``n_layers * d_model_enc``); the
    field is kept for provenance and a future alternative pool.
    """

    latent_dim: int = 384
    hidden: int = 512
    n_mlp_layers: int = 3
    pool: str = "mean_d_state_concat_layers"


def pool_encoder_states(enc_states: list[Tensor]) -> Tensor:
    """Map per-layer encoder state -> the pooled state representation.

    ``enc_states``: list of ``n_enc`` tensors ``[b, d_state, d_model_enc]`` (from
    ``SSMLanguageModel.forward``). Mean over the ``d_state`` axis per layer and
    concatenate all layers -> ``[b, n_enc * d_model_enc]``. This generalizes the
    0a-validated ``recoverability_head.pool_state_tensors`` representation (which
    was hardcoded to 4 layers x 384) to the token-LM encoder's
    ``n_layers * d_model_enc``; it is the richer state feature (all layers, not
    just the last) the predictor reads.
    """
    if not enc_states:
        raise ValueError("pool_encoder_states called with no state tensors")
    per_layer = []
    for st in enc_states:
        # Preserve the input dtype (bf16 on GPU, fp32 on CPU). The mean over
        # d_state is numerically safe in bf16, and preserving dtype keeps the
        # pooled tensor compatible with the predictor's MLP weights OUTSIDE
        # autocast (a forced fp32 cast here would hand a bf16 Linear a fp32
        # input and raise a dtype-mismatch once autocast is off -- which is
        # exactly the eval path and the trainer's post-training val).
        s = st
        if s.dim() == 3:
            per_layer.append(s.mean(dim=1))        # [b, d_model_enc]
        elif s.dim() == 2:
            # [d_state, d_model_enc] (batch dim squeezed) -> mean over d_state
            per_layer.append(s.mean(dim=0).unsqueeze(0))
        else:
            raise ValueError(f"unexpected state tensor dim {s.dim()}")
    return torch.cat(per_layer, dim=-1)  # [b, n_enc * d_model_enc]


class LatentPredictor(nn.Module):
    """MLP head: pooled encoder state -> L2-normalized gist latent (bge space).

    ``forward(enc_states)`` pools the per-layer encoder states and maps them to a
    ``[b, latent_dim]`` L2-normalized vector. The output is in the frozen bge-small
    meaning space (the JEPA target), so the contrastive loss
    (``jepa_loss.jepa_contrastive_loss``) and the frozen bge targets are
    dimensionally compatible with no extra projection.
    """

    def __init__(self, encoder_cfg: LMConfig, cfg: LatentPredictorConfig):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg
        in_dim = encoder_cfg.n_layers * encoder_cfg.d_model
        hidden = cfg.hidden
        out_dim = cfg.latent_dim
        n = cfg.n_mlp_layers
        if n < 2:
            raise ValueError("n_mlp_layers must be >= 2 (input + output)")
        # n Linear layers with GELU between: [in, hidden], [hidden, hidden] x (n-2),
        # [hidden, out]. For n=3: Linear(in, hidden), Linear(hidden, hidden),
        # Linear(hidden, out).
        layers: list[nn.Module] = []
        prev = in_dim
        for i in range(n):
            cur = hidden if i < n - 1 else out_dim
            layers.append(nn.Linear(prev, cur))
            if i < n - 1:
                layers.append(nn.GELU())
            prev = cur
        self.mlp = nn.Sequential(*layers)

    def forward(self, enc_states: list[Tensor]) -> Tensor:
        pooled = pool_encoder_states(enc_states)  # [b, n_enc * d_model_enc]
        z = self.mlp(pooled)                       # [b, latent_dim]
        return F.normalize(z, p=2, dim=-1)         # unit length for cosine


class JEPAGistModel(nn.Module):
    """Encoder + latent predictor: doc -> SSM state -> gist latent (bge space).

    ``encode`` runs the encoder over a doc and returns its final per-layer
    recurrent state. ``predict_latent`` pools that state and maps it to a
    gist latent. ``forward`` does both in one call AND returns the encoder's
    next-token logits (for the LM-prior auxiliary that prevents the encoder
    collapse seen in the failed retrain).

    The encoder is frozen by default (the probe-style / eval path); the retrain
    path passes ``freeze_encoder=False`` so the JEPA-latent contrastive loss +
    LM-prior auxiliary flow back through the predictor -> encoder recurrence and
    reshape the continuation-shaped state into a gist-shaped one.
    """

    def __init__(self, encoder: SSMLanguageModel, cfg: LatentPredictorConfig,
                 freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.predictor = LatentPredictor(encoder.config, cfg)
        # Freeze the encoder unless the caller explicitly opts out. The eval path
        # keeps the default frozen (asserted again in ``load_jepa_gist``); the
        # retrain path passes ``freeze_encoder=False`` so the JEPA-latent loss can
        # flow back through the predictor -> encoder recurrence and reshape the
        # state.
        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    # ------------------------------------------------------------------ encode
    def encode(self, doc_ids: Tensor, no_grad: bool = True) -> list[Tensor]:
        """Run the encoder over ``doc_ids`` -> final per-layer states.

        ``no_grad=True`` (default, the eval / gate path): run under
        ``torch.no_grad()`` with the encoder in eval mode and return DETACHED
        states. ``no_grad=False`` (the retrain path, encoder UNFROZEN): run in
        the graph with the encoder in train mode and return the LIVE states so
        gradient from the JEPA-latent loss flows back through the predictor ->
        encoder recurrence. The encoder's params MUST have ``requires_grad=True``
        for this to do anything (the trainer thaws them; the loader always
        re-freezes for eval).
        """
        if no_grad:
            with torch.no_grad():
                self.encoder.eval()
                _, states = self.encoder.forward(doc_ids)
                return [s.detach() for s in states]
        self.encoder.train()
        _, states = self.encoder.forward(doc_ids)
        return states

    # ------------------------------------------------------------- predict latent
    def predict_latent(self, enc_states: list[Tensor]) -> Tensor:
        """Pool the encoder states -> L2-normalized gist latent ``[b, latent_dim]``."""
        return self.predictor(enc_states)

    # --------------------------------------------------------- full forward (train)
    def forward(
        self,
        doc_ids: Tensor,
        no_grad: bool = False,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """One-call training forward: doc -> (gist latent, LM logits, enc states).

        Returns ``(pred_latent [b, latent_dim], lm_logits [b, seq, vocab],
        enc_states)``. The encoder's next-token ``lm_logits`` are returned so the
        trainer can compute the LM-prior auxiliary (next-token CE on the doc) in
        the same forward -- this is the anti-collapse term that preserves the
        continuation prior. ``no_grad=True`` runs the whole thing under
        ``no_grad`` with the encoder in eval mode (used by the val metric, not
        the train loop).
        """
        if no_grad:
            with torch.no_grad():
                self.encoder.eval()
                logits, states = self.encoder.forward(doc_ids)
                pred = self.predictor([s.detach() for s in states])
                return pred, logits.detach(), [s.detach() for s in states]
        self.encoder.train()
        logits, states = self.encoder.forward(doc_ids)
        pred = self.predictor(states)
        return pred, logits, states

    # --------------------------------------------------------------- param count
    def trainable_parameters(self) -> int:
        """Count only the trainable params (predictor; + encoder if thawed)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- load
def load_jepa_gist(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path = "",
    device: str = "auto",
    dtype: str = "bfloat16",
) -> tuple[JEPAGistModel, TokenizerWrapper]:
    """Load a trained JEPA-gist model + its encoder + tokenizer.

    Two load paths, selected by the checkpoint contents (mirrors
    ``load_gist_readout``):

      * **Retrain checkpoint** (contains an ``"encoder"`` key, written with
        ``save_encoder=True``): the encoder is restored strict from the JEPA
        checkpoint itself.
      * **Frozen-encoder checkpoint** (no ``"encoder"`` key): the encoder is
        loaded strict from the separate ``encoder_checkpoint`` (the token-LM
        ckpt). NOT supported here -- the JEPA path always retrains the encoder
        (``--train-encoder``), so the retrain ckpt carries it. If you need the
        frozen-encoder probe path, use ``load_gist_readout`` instead.

    Both paths freeze the encoder for EVAL (eval never trains; training-thaw is
    the trainer's job, not the loader's). Mirrors ``load_gist_readout``'s
    strict-mismatch discipline: each state_dict is loaded strict; any
    missing/unexpected key raises.

    Returns ``(model, tokenizer)``. The checkpoint format (written by
    ``scripts/train_jepa_gist.py``)::

        {"predictor": <LatentPredictor state_dict>,
         "latent_config": <LatentPredictorConfig dict>,
         "encoder_config": <LMConfig dict>,
         "encoder_ref": <encoder ckpt basename, provenance only>,
         "step": <int>,
         "encoder": <SSMLanguageModel state_dict>  # ONLY if save_encoder=True
        }
    """
    from .tokenizer_ import train_or_load_tokenizer  # local import keeps top clean

    dev = _resolve_device(device)
    dt = _resolve_dtype(dtype, dev)

    # ---- tokenizer (the cache is canonical; corpus arg unused on load).
    tok = train_or_load_tokenizer(iter([]), tokenizer_path, vocab_size=4096)

    # ---- JEPA checkpoint (read once; carries the predictor + maybe the encoder).
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    latent_cfg = LatentPredictorConfig(**ckpt["latent_config"])

    # ---- encoder: from the JEPA ckpt (retrain path -- the only supported path).
    if "encoder" not in ckpt:
        raise RuntimeError(
            f"jepa checkpoint {checkpoint_path} has no 'encoder' key; the JEPA "
            f"path always retrains the encoder (--train-encoder), so the ckpt "
            f"must carry it. For a frozen-encoder probe use load_gist_readout."
        )
    encoder_cfg = LMConfig(**ckpt["encoder_config"])
    encoder = SSMLanguageModel(encoder_cfg).to(device=dev, dtype=dt)
    enc_sd = ckpt["encoder"]
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"jepa checkpoint {checkpoint_path} encoder mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )

    # ---- predictor + model: build with the encoder, load the predictor strict.
    model = JEPAGistModel(encoder, latent_cfg, freeze_encoder=True).to(
        device=dev, dtype=dt
    )
    pred_sd = ckpt["predictor"]
    missing, unexpected = model.predictor.load_state_dict(pred_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"jepa checkpoint {checkpoint_path} predictor mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    # Re-freeze after .to() and the predictor load for EVAL (eval never trains).
    model._freeze_encoder()
    assert all(not p.requires_grad for p in model.encoder.parameters()), \
        "encoder must be frozen after load"
    model.eval()
    return model, tok


# --------------------------------------------------------------- device/dtype
# Local copies of the helpers in scripts/train_token_lm.py (kept here so the
# loader has no script-side import dependency; identical resolution to
# gist_readout.py).
def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    return torch.float32


def save_jepa_checkpoint(
    path: str | Path,
    model: JEPAGistModel,
    step: int,
    encoder_ref: str,
    save_encoder: bool = False,
) -> None:
    """Write the JEPA-gist checkpoint.

    ``save_encoder=True`` (the retrain path, always the case for JEPA) ALSO writes
    the retrained encoder state_dict under ``"encoder"`` so the eval loader can
    restore the whole retrained model from one checkpoint with no separate
    encoder file.
    """
    payload = {
        "predictor": model.predictor.state_dict(),
        "latent_config": asdict(model.cfg),
        "encoder_config": asdict(model.encoder.config),
        "encoder_ref": encoder_ref,
        "step": step,
    }
    if save_encoder:
        payload["encoder"] = model.encoder.state_dict()
    torch.save(payload, path)