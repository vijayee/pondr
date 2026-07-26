"""Frozen-encoder state->gist readout (the §3.3 content probe, done right).

Post-mortem §6: **state shape is set by the OBJECTIVE, not the backbone.** The bge
backbone was trained on a next-EMBEDDING *identity* objective, so its state became
identity-shaped -- the §3.3 content probe (``pondr-jst-content-probe-result``)
FAILED to recover any doc-specific content from it (permutation ~= corpus-mean ~=
main: swapping states changed nothing). The token-LM (``token_lm.py``, ``3b11aeb``)
was then trained on a token CE (continuation) objective, so its state is
*continuation*-shaped -- it can continue a doc, not summarize one.

The load-bearing open question this module exists to answer: does a content
objective shape **gist-recoverable content** into the SSM state? Per the
post-mortem's own methodology (§3 item 1: *"probe the EXACT property you will
serve, not a proxy; frozen backbone, no retrain"*), we freeze the token-LM as an
**encoder**, train **only a small new decoder** that reads the encoder's final
recurrent state and generates a gist, and gate on whether the decoded gist is
faithful to the ingested doc **on held-out docs** with the §3.3 swap control
(``scripts/eval_gist_readout.py``).

Why the decoder reads ``states`` and not ``x``: the encoder's ``lm_head`` reads the
block output ``x`` (the continuation path -- that is why the token-LM continues
instead of summarizing). The only doc-specific signal available to the decoder at
generate time is the encoder's final recurrent **state** (passed through a small
per-layer projection as the decoder's initial state). So if the decoder produces a
doc-specific gist, the content MUST have come through the state -- the property
under test. A swap of states between two docs must therefore swap the decoded gist
(``swap-follows-state``); §3.3 failed exactly here (swap ~= main).

Architecture (encoder-decoder, state-seeded):

    encoder = SSMLanguageModel          # FROZEN, loaded from the token-LM ckpt
    doc_ids -> encoder.forward -> enc_states   # per-layer [b, d_state, d_model_enc]
    state_proj[i] : Linear(d_model_enc, d_model_dec)   # per decoder layer, trainable
    enc_states[-n_dec:] -> projected -> dec_states     # seed the decoder's state
    decoder = SSMLanguageModel(small)   # trainable; fresh, NOT tied to the encoder
    dec_states + BOS -> decoder.step (autoregressive) -> gist token ids

The decoder is intentionally small (the encoder is the expensive part and it is
frozen); the trainable surface is the decoder + the per-layer state projection.

Standalone: this does NOT touch the bge ``WorkingMemory`` / ``JGSBackbone`` /
``serve_ponder`` path. Wiring any of this into the orchestrator is a gated
follow-on, only after the probe PASSES.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from .token_lm import LMConfig, SSMLanguageModel
from .tokenizer_ import TokenizerWrapper


@dataclass
class GistConfig:
    """Architecture hyperparameters for the gist decoder.

    The decoder is a fresh, SMALL ``SSMLanguageModel`` (smaller than the encoder)
    whose initial recurrent state is a per-layer projection of the encoder's final
    state. ``vocab`` MUST equal the encoder's vocab (shared tokenizer); the
    special-token ids MUST match too. ``d_state`` defaults to the encoder's
    ``d_state`` so the per-channel projection only crosses ``d_model``.
    """

    vocab: int = 4096
    d_model_dec: int = 96
    n_layers_dec: int = 2
    d_state: int = 16
    gist_seq_len: int = 64
    tie_head: bool = True
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class GistDecoder(nn.Module):
    """Small state-seeded decoder that reads the encoder's recurrent state.

    Holds a fresh ``SSMLanguageModel`` (the decoder) and a per-layer
    ``state_proj`` (``Linear(d_model_enc, d_model_dec)``) that maps each seeded
    encoder layer's state into the corresponding decoder layer's initial state.
    Decoder layer ``i`` is seeded from encoder layer ``n_enc - n_dec + i`` (the
    last ``n_dec`` encoder layers -- the most-processed states).

    The decoder's own ``lm_head`` reads its own block output ``x`` during
    autoregressive generation; that is correct and intended -- the DOC CONTENT
    enters only through the seeded initial state (the projection), which is the
    property under test. Nothing about the source doc is fed to the decoder at
    generate time except its final recurrent state.
    """

    def __init__(self, encoder_cfg: LMConfig, gist_cfg: GistConfig):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.gist_cfg = gist_cfg
        dec_lm_cfg = LMConfig(
            vocab=gist_cfg.vocab,
            d_model=gist_cfg.d_model_dec,
            n_layers=gist_cfg.n_layers_dec,
            d_state=gist_cfg.d_state,
            seq_len=gist_cfg.gist_seq_len,
            tie_head=gist_cfg.tie_head,
            dropout=gist_cfg.dropout,
            pad_token_id=gist_cfg.pad_token_id,
            bos_token_id=gist_cfg.bos_token_id,
            eos_token_id=gist_cfg.eos_token_id,
        )
        self.decoder = SSMLanguageModel(dec_lm_cfg)
        # Per-layer projection of the encoder state into the decoder's initial
        # state. Applied per channel on the last dim: [b, d_state, d_model_enc]
        # -> [b, d_state, d_model_dec]. One Linear per decoder layer, trainable.
        self.state_proj = nn.ModuleList(
            [
                nn.Linear(encoder_cfg.d_model, gist_cfg.d_model_dec)
                for _ in range(gist_cfg.n_layers_dec)
            ]
        )

    # ----------------------------------------------------------- state -> state
    def project_states(self, enc_states: list[Tensor]) -> list[Tensor]:
        """Map the encoder's final per-layer states to the decoder's initial states.

        ``enc_states``: list of ``n_enc`` tensors ``[b, d_state, d_model_enc]``
        (from ``SSMLanguageModel.forward``). Returns a list of ``n_dec`` tensors
        ``[b, d_state, d_model_dec]``, decoder layer ``i`` projected from encoder
        layer ``n_enc - n_dec + i``.
        """
        n_enc = len(enc_states)
        n_dec = self.gist_cfg.n_layers_dec
        if n_enc < n_dec:
            raise ValueError(
                f"encoder has {n_enc} layers but decoder needs {n_dec} seeded states"
            )
        dec_states: list[Tensor] = []
        for i in range(n_dec):
            src = enc_states[n_enc - n_dec + i]  # [b, d_state, d_model_enc]
            dec_states.append(self.state_proj[i](src))  # [b, d_state, d_model_dec]
        return dec_states

    # --------------------------------------------------- teacher-forced forward
    def forward(
        self,
        gist_ids: Tensor,
        enc_states: list[Tensor],
    ) -> Tensor:
        """Teacher-forced forward for training.

        ``gist_ids``: ``[batch, gist_len]`` (``[BOS, t1, ..., tN, EOS, PAD...]``).
        ``enc_states``: the frozen encoder's final per-layer states. Returns
        ``logits [batch, gist_len, vocab]`` (predict ``gist_ids[:, t+1]`` from
        position ``t``; the caller shifts + masks PAD/EOS). The decoder is seeded
        with the projected encoder state, so the only doc-specific signal is the
        state.
        """
        dec_states = self.project_states(enc_states)
        logits, _ = self.decoder.forward(gist_ids, states=dec_states)
        return logits  # [b, gist_len, vocab]

    # ------------------------------------------------------- autoregressive gen
    @torch.no_grad()
    def generate(
        self,
        enc_states: list[Tensor],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Autoregressive decode from the seeded state.

        Returns generated token ids ``[batch, n]`` (``n <= max_new_tokens``,
        stopping at ``eos``). The decoder is primed by stepping ``BOS`` through
        the (projected) encoder state -- NOT by a prompt ``forward`` -- so the
        recurrence starts from the doc state and the first token is read straight
        out of it.
        """
        self.eval()
        device = self.decoder.token_emb.weight.device
        dec_states = self.project_states(enc_states)
        batch = dec_states[0].shape[0]
        bos = torch.full((batch,), self.gist_cfg.bos_token_id,
                         dtype=torch.long, device=device)
        logits, dec_states = self.decoder.step(bos, dec_states)  # [b, vocab]
        out: list[Tensor] = []
        eos_id = self.gist_cfg.eos_token_id
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                next_id = logits.argmax(dim=-1)  # [b]
            else:
                logits_t = logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0 and top_k < self.gist_cfg.vocab:
                    kth = torch.topk(logits_t, k=top_k, dim=-1)
                    thresh = kth.values[:, -1:].expand_as(logits_t)
                    logits_t = torch.where(
                        logits_t < thresh,
                        torch.full_like(logits_t, float("-inf")),
                        logits_t,
                    )
                probs = torch.softmax(logits_t, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [b]
            out.append(next_id)
            finished = finished | (next_id == eos_id)
            if bool(finished.all()):
                break
            logits, dec_states = self.decoder.step(next_id, dec_states)
        if not out:
            return torch.empty(batch, 0, dtype=torch.long, device=device)
        return torch.stack(out, dim=1)  # [b, n]


class GistReadoutModel(nn.Module):
    """Frozen encoder + trained decoder: text in -> gist text out (the probe).

    ``encode`` runs the frozen token-LM over a doc and returns its final
    per-layer recurrent state. ``generate_gist`` projects that state into the
    decoder's initial state and autoregressively decodes a gist. The encoder is
    frozen for real (``requires_grad=False``); only the decoder + state
    projection train.
    """

    def __init__(self, encoder: SSMLanguageModel, gist_cfg: GistConfig):
        super().__init__()
        self.encoder = encoder
        self.gist_cfg = gist_cfg
        self.decoder = GistDecoder(encoder.config, gist_cfg)
        # Freeze the encoder for real. Asserted again in load_gist_readout; kept
        # here so a model built in code (tests) is correct without the loader.
        self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, doc_ids: Tensor) -> list[Tensor]:
        """Run the frozen encoder over ``doc_ids`` -> final per-layer states.

        ``doc_ids``: ``[batch, seq]`` long tensor (caller truncates to the
        encoder's ``seq_len``; longer docs are truncated, not chunked, for the
        single-doc probe). Returns a list of ``n_enc`` tensors
        ``[batch, d_state, d_model_enc]`` -- the recurrent state that is the only
        doc-specific signal the decoder sees.
        """
        self.encoder.eval()
        _, states = self.encoder.forward(doc_ids)
        # Detach so no graph is carried from the frozen encoder into the decoder
        # backward (the encoder is frozen, but detach makes the boundary explicit
        # and avoids any accidental grad accumulation into encoder params).
        return [s.detach() for s in states]

    # ----------------------------------------------------------- decode -> text
    @torch.no_grad()
    def generate_gist(
        self,
        enc_states: list[Tensor],
        tokenizer: TokenizerWrapper,
        max_new_tokens: int = 48,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> str:
        """Decode an encoder state into a gist string (single doc, batch[0])."""
        ids = self.decoder.generate(
            enc_states,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return tokenizer.decode(ids[0].tolist())

    # --------------------------------------------------------------- param count
    def trainable_parameters(self) -> int:
        """Count only the trainable params (decoder + projection; encoder frozen)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- load
def load_gist_readout(
    checkpoint_path: str | Path,
    encoder_checkpoint: str | Path,
    tokenizer_path: str | Path,
    device: str = "auto",
    dtype: str = "bfloat16",
) -> tuple[GistReadoutModel, TokenizerWrapper]:
    """Load a trained gist readout + its frozen encoder + tokenizer.

    Mirrors ``load_backbone``'s strict-mismatch discipline: the encoder checkpoint
    and the decoder checkpoint are each loaded strict; any missing/unexpected key
    raises (never silently train on a partial load). The encoder is moved to the
    resolved device/dtype and frozen for real (``requires_grad=False`` asserted).

    Returns ``(model, tokenizer)``. The checkpoint format (written by
    ``scripts/train_gist_readout.py``) is::

        {"decoder": <GistDecoder state_dict>,
         "gist_config": <GistConfig dict>,
         "encoder_config": <LMConfig dict>,
         "encoder_ref": <encoder ckpt basename, provenance only>,
         "step": <int>}
    """
    from .tokenizer_ import train_or_load_tokenizer  # local import keeps top clean

    dev = _resolve_device(device)
    dt = _resolve_dtype(dtype, dev)

    # ---- tokenizer (the cache is canonical; corpus arg unused on load).
    tok = train_or_load_tokenizer(iter([]), tokenizer_path, vocab_size=4096)

    # ---- encoder: load the token-LM checkpoint strict, freeze for real.
    enc_ckpt = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
    enc_cfg_dict = enc_ckpt["config"] if isinstance(enc_ckpt, dict) and "config" in enc_ckpt else None
    if enc_cfg_dict is None:
        raise RuntimeError(
            f"encoder checkpoint {encoder_checkpoint} has no 'config'; "
            f"expected a token-LM checkpoint written by scripts/train_token_lm.py"
        )
    encoder_cfg = LMConfig(**enc_cfg_dict)
    encoder = SSMLanguageModel(encoder_cfg).to(device=dev, dtype=dt)
    enc_sd = enc_ckpt["model"] if "model" in enc_ckpt else enc_ckpt
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder checkpoint {encoder_checkpoint} mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # ---- decoder + readout: load the gist checkpoint strict.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gist_cfg = GistConfig(**ckpt["gist_config"])
    model = GistReadoutModel(encoder, gist_cfg).to(device=dev, dtype=dt)
    dec_sd = ckpt["decoder"]
    missing, unexpected = model.decoder.load_state_dict(dec_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"gist checkpoint {checkpoint_path} decoder mismatch: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    # Re-freeze after .to() (moving a module does not flip requires_grad, but the
    # decoder load + any future edits should not be able to thaw the encoder; this
    # is the load-time assertion the plan requires).
    model._freeze_encoder()
    assert all(not p.requires_grad for p in model.encoder.parameters()), \
        "encoder must be frozen after load"
    model.eval()
    return model, tok


# --------------------------------------------------------------- device/dtype
# Local copies of the helpers in scripts/train_token_lm.py (kept here so the
# loader has no script-side import dependency; the token-LM path uses the same
# resolution).
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


def save_gist_checkpoint(
    path: str | Path,
    model: GistReadoutModel,
    step: int,
    encoder_ref: str,
) -> None:
    """Write the gist checkpoint (decoder + configs; encoder referenced, not copied)."""
    torch.save(
        {
            "decoder": model.decoder.state_dict(),
            "gist_config": asdict(model.gist_cfg),
            "encoder_config": asdict(model.encoder.config),
            "encoder_ref": encoder_ref,
            "step": step,
        },
        path,
    )