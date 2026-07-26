"""Token-level language model whose sequence mixer is the owned ``SelectiveSSM``.

This is the content-objective backbone the §3.3 content probe prescribed as the
next step (post-mortem §6: *state shape is set by the OBJECTIVE, not the
backbone*). The bge backbone was trained on a next-EMBEDDING *identity* objective,
so its state became identity-shaped -- content was falsified out of it (see
``pondr-jst-content-probe-result``). To get language in and out that makes sense
you must train a TOKEN objective (vocab head + next-token CE) into a block. This
is that model.

Stack (Mamba3-style I/O, owned block):

    tokenizer (BPE, ``tokenizer_.py``)          # text <-> token ids
    token_emb  = nn.Embedding(vocab, d_model)    # ids -> vectors
    layers     = ModuleList([SelectiveSSM ...])  # recurrent sequence mixer
    lm_head    = nn.Linear(d_model, vocab)       # vectors -> vocab logits (tied
                                                 #   to token_emb.weight)
    loss       = next-token cross-entropy

The SSM block is just the sequence mixer; text-out comes from the tokenizer +
vocab head + token objective, not from the block. Both ``forward()`` (sequence,
the training path) and ``step()`` (single token, the generation + live-serve
path) are real -- ``SelectiveSSM.step`` is not a stub, so ``generate`` actually
runs the recurrent decode, it does not fall back to ``forward``.

Standalone: this does NOT touch the bge ``WorkingMemory`` / ``JGSBackbone`` /
``serve_ponder`` path. Wiring the LM into the live orchestrator is a gated
follow-on (see plan). Param budget at the defaults: 6 layers of ~2.1M (block) +
tied emb/head 4096*256 = ~1.05M ~= ~13.7M.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .configs import BackboneConfig
from .ssm import make_ssm


@dataclass
class LMConfig:
    """Architecture hyperparameters for the token-level LM-SSM.

    Right-sized to be a *first working* content model, not a large LM: d_model
    256, 6 layers, d_state 16, vocab 4096, seq_len 256. ~14M params with tied
    emb/head. All knobs are tunable for a later scale-up once the small model
    generates coherently.
    """

    vocab: int = 4096
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    seq_len: int = 256
    # Tying the LM head to the token embedding is standard for small LMs (halves
    # the params in the I/O wrapper, and the vocab*d_model block dominates
    # anyway). Set False to learn a separate head.
    tie_head: bool = True
    # Dropout off by default -- the model is small and the corpus modest; turn on
    # if train/val gap widens.
    dropout: float = 0.0
    # Special token ids (the tokenizer wrapper fills these in). Kept on the
    # config so the model is self-contained for generation/padding.
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


class SSMLanguageModel(nn.Module):
    """A small token-level LM whose sequence mixer is ``SelectiveSSM``.

    State is per-layer: a list of ``[batch, d_state, d_model]`` tensors, one per
    SSM layer. ``forward`` threads the sequence through all layers (training);
    ``step`` advances a single token through all layers (generation / live
    serve). ``generate`` autoregressively samples via ``step`` -- the recurrent
    state is the memory that keeps a long prefix on-topic, the point of the
    build.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab, config.d_model)
        # Each layer is an owned SelectiveSSM built through the same factory the
        # rest of the codebase uses (so a backend swap is one line).
        block_cfg = BackboneConfig(
            d_model=config.d_model,
            d_state=config.d_state,
            ssm_backend="selective",
        )
        self.layers = nn.ModuleList(
            [make_ssm("selective", block_cfg) for _ in range(config.n_layers)]
        )
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab, bias=False)
        if config.tie_head:
            # Tied weights: the output projection reuses the input embedding
            # matrix. Standard for small LMs; halves the I/O param count.
            self.lm_head.weight = self.token_emb.weight
        # Init: small normal for embeddings/heads (stable softmax); the SSM
        # block carries its own Mamba-style init (A_log, D=1).
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        if not config.tie_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ helpers
    def _layer_states(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        states: Optional[list[Tensor]] = None,
    ) -> list[Tensor]:
        """Return one fresh zero state per layer, or the provided states."""
        if states is not None:
            assert len(states) == len(self.layers)
            return states
        return [layer.init_state(batch, device, dtype) for layer in self.layers]

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Tensor,
        states: Optional[list[Tensor]] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Sequence path (training).

        ``input_ids``: ``[batch, seq]`` long tensor. Returns
        ``(logits [batch, seq, vocab], final_states)`` where ``final_states`` is
        a list of per-layer ``[batch, d_state, d_model]`` tensors -- pass them
        back into ``step`` to continue generation from the end of the prompt.
        """
        batch, seq = input_ids.shape
        device = input_ids.device
        # Compute in float32 for the softmax/CE path regardless of the SSM's
        # bf16 forward; the embedding lookup is index-gather so dtype is set
        # here from the model parameter dtype.
        dtype = self.token_emb.weight.dtype
        x = self.token_emb(input_ids)  # [b, seq, d_model]
        x = self.drop(x)
        cur_states = self._layer_states(batch, device, dtype, states)
        new_states: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            x, st = layer.forward(x, cur_states[i])
            new_states.append(st)
        x = self.drop(x)
        logits = self.lm_head(x)  # [b, seq, vocab]
        return logits, new_states

    # -------------------------------------------------------------------- step
    def step(
        self,
        token_id: Tensor,
        states: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        """Single-token recurrent step (generation + live serve).

        ``token_id``: ``[batch]`` long tensor. ``states``: per-layer states from
        the previous step or from the prompt's ``forward``. Returns
        ``(logits [batch, vocab], new_states)``. This is the path the content
        probe's "state carries context" property is exercised on -- and it is
        real, not a stub, because ``SelectiveSSM.step`` is real.
        """
        batch = token_id.shape[0]
        device = token_id.device
        dtype = self.token_emb.weight.dtype
        x = self.token_emb(token_id).to(dtype)  # [b, d_model]
        x = self.drop(x)
        new_states: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            x, st = layer.step(x, states[i])
            new_states.append(st)
        x = self.drop(x)
        logits = self.lm_head(x)  # [b, vocab]
        return logits, new_states

    # ---------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Autoregressive sampling via ``step``.

        ``prompt_ids``: ``[batch, seq]`` long tensor (the prefix). Returns
        ``[batch, seq + max_new_tokens]`` -- the prompt concatenated with the
        sampled continuation. ``temperature=0`` = greedy argmax; ``>0`` samples
        (with optional ``top_k`` truncation). The recurrent state is primed by
        one ``forward`` over the prompt, then each new token goes through
        ``step`` -- so the state actually carries the prefix (the point of the
        build), it is not re-fed the whole window each step.
        """
        self.eval()
        device = prompt_ids.device
        batch, seq = prompt_ids.shape
        # Prime the recurrent state with the prompt in one sequence pass.
        logits, states = self.forward(prompt_ids)
        next_logits = logits[:, -1, :]  # [b, vocab]
        out = [prompt_ids]
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                next_id = next_logits.argmax(dim=-1)  # [b]
            else:
                logits_t = next_logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0 and top_k < self.config.vocab:
                    # Keep only the top_k logits; set the rest to -inf so they
                    # are never sampled.
                    kth = torch.topk(logits_t, k=top_k, dim=-1)
                    thresh = kth.values[:, -1:].expand_as(logits_t)
                    logits_t = torch.where(
                        logits_t < thresh,
                        torch.full_like(logits_t, float("-inf")),
                        logits_t,
                    )
                probs = torch.softmax(logits_t, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Both greedy (argmax) and sampling yield ``[batch]``; stack a token
            # axis for the cat below.
            out.append(next_id.unsqueeze(-1))
            next_logits, states = self.step(next_id, states)
        return torch.cat(out, dim=1)

    # ------------------------------------------------------------- param count
    def num_parameters(self) -> int:
        """Total trainable parameter count (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())