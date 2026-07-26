"""Unit tests for the frozen-encoder state->gist readout (``gist_readout.py``).

CPU-runnable, self-contained (no skip, no external checkpoint, no teacher LLM).
Exercises the three things the probe depends on before any real training runs:

1. Shape/roundtrip: ``encode`` -> ``generate_gist`` returns a ``str``; the encoder
   is frozen for real and the decoder is trainable; the decoder reads the encoder
   STATE (projected to the right shape), not the encoder block output ``x``.
2. Swap-control mechanics: on a toy (doc -> doc-specific gist token) corpus, the
   decoded gist follows the STATE -- swapping states swaps the decoded gist. This
   is the load-bearing §3.3 gate, verified deterministically with a random (frozen)
   encoder: the encoder's state differs between two distinct docs, so the decoder
   can learn to read doc identity out of the state, and swapping the state swaps
   the output.
3. Teacher-forced forward shape + that the decoder is the only trainable surface.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.subconscious.gist_readout import (
    GistConfig,
    GistReadoutModel,
)
from src.subconscious.token_lm import LMConfig, SSMLanguageModel


# Special-token ids are fixed by the tokenizer wrapper (PAD=0, BOS=1, EOS=2).
PAD, BOS, EOS = 0, 1, 2


def _enc_cfg(**kw) -> LMConfig:
    base = dict(vocab=32, d_model=16, n_layers=2, d_state=4, seq_len=8)
    base.update(kw)
    return LMConfig(**base)


def _gist_cfg(**kw) -> GistConfig:
    base = dict(
        vocab=32, d_model_dec=12, n_layers_dec=1, d_state=4,
        gist_seq_len=6, tie_head=True, dropout=0.0,
        pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS,
    )
    base.update(kw)
    return GistConfig(**base)


def _build(**kw) -> GistReadoutModel:
    enc = SSMLanguageModel(_enc_cfg())
    return GistReadoutModel(enc, _gist_cfg(**kw))


# --------------------------------------------------------------------- shapes
def test_encode_generate_roundtrip_returns_str():
    m = _build()
    doc = torch.randint(3, 32, (1, 6))
    states = m.encode(doc)
    # encode returns one state per encoder layer, shape [b, d_state, d_model_enc].
    assert len(states) == m.encoder.config.n_layers
    assert states[0].shape == (1, 4, 16)

    # A trivial stand-in tokenizer: map ids -> chars. generate_gist needs a
    # TokenizerWrapper; for the shape/roundtrip test we exercise the decoder
    # directly and decode ints ourselves (the str contract is the tokenizer's).
    ids = m.decoder.generate(states, max_new_tokens=5, temperature=0.0)
    assert ids.shape[0] == 1
    assert ids.shape[1] <= 5
    assert ((ids >= 0) & (ids < 32)).all()


def test_encoder_frozen_decoder_trainable():
    m = _build()
    assert all(not p.requires_grad for p in m.encoder.parameters()), \
        "encoder must be frozen"
    assert any(p.requires_grad for p in m.decoder.parameters()), \
        "decoder must be trainable"
    # The trainable surface is ONLY the decoder (state_proj + the small decoder LM):
    # every trainable param belongs to the decoder, and its count matches
    # ``trainable_parameters``.
    dec_n = sum(p.numel() for p in m.decoder.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable == dec_n, "the only trainable params should be the decoder's"
    assert m.trainable_parameters() == sum(
        p.numel() for p in m.parameters() if p.requires_grad
    )


def test_decoder_reads_states_not_x():
    """The decoder's initial state is a projection of the encoder STATE, not the
    encoder block output ``x``. Verify by shape: project_states output matches the
    decoder layer's own ``init_state`` shape, and the decoder forward accepts it.
    """
    m = _build()
    doc = torch.randint(3, 32, (2, 5))
    enc_states = m.encode(doc)
    dec_states = m.decoder.project_states(enc_states)
    # One projected state per decoder layer.
    assert len(dec_states) == m.gist_cfg.n_layers_dec
    # Shape matches the decoder's own init_state -- i.e. it is a valid seed for the
    # decoder layers (the encoder x is [b, seq, d_model_enc]; the state is
    # [b, d_state, d_model_dec]). This is the "reads states, not x" check.
    ref = m.decoder.decoder.layers[0].init_state(2, dec_states[0].device, dec_states[0].dtype)
    assert dec_states[0].shape == ref.shape

    # Teacher-forced forward with the seeded state produces vocab logits.
    gist_ids = torch.tensor([[BOS, 5, 6, EOS, PAD, PAD], [BOS, 7, 8, EOS, PAD, PAD]])
    logits = m.decoder.forward(gist_ids, enc_states)
    assert logits.shape == (2, 6, 32)


def test_decoder_grad_does_not_reach_encoder():
    """A backward through the decoder must not produce grads in the frozen encoder
    (the encode path is detached + requires_grad=False)."""
    m = _build()
    doc = torch.randint(3, 32, (2, 5))
    enc_states = m.encode(doc)
    gist_ids = torch.tensor([[BOS, 5, 6, EOS, PAD, PAD], [BOS, 7, 8, EOS, PAD, PAD]])
    logits = m.decoder.forward(gist_ids, enc_states)
    targets = gist_ids[:, 1:]
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, 32),
        targets.reshape(-1),
        ignore_index=PAD,
    )
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in m.encoder.parameters()), \
        "encoder params must receive no grad"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.decoder.parameters()), \
        "decoder params must receive grad"


# ------------------------------------------------------- swap-control mechanics
def _train_toy_readout(m: GistReadoutModel, pairs, steps=400, lr=5e-3):
    """Train ONLY the decoder on (doc_ids, gist_ids) pairs until it recovers the
    doc-specific gist token. Tiny CPU problem; converges in a few hundred steps."""
    optim = torch.optim.AdamW(m.decoder.parameters(), lr=lr)
    m.encoder.eval()
    for _ in range(steps):
        for doc_ids, gist_ids in pairs:
            enc_states = m.encode(doc_ids)
            logits = m.decoder.forward(gist_ids, enc_states)
            targets = gist_ids[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, m.gist_cfg.vocab),
                targets.reshape(-1),
                ignore_index=PAD,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()


def test_swap_control_follows_state():
    """Toy corpus: doc A -> gist token 10, doc B -> gist token 20. After training
    the decoder, decoding from state_A yields 10 and from state_B yields 20.
    Swapping states (decoding B's state in A's "slot") yields 20 -- the gist
    follows the STATE, the §3.3 gate's load-bearing property. A random frozen
    encoder suffices: distinct docs -> distinct states -> a learnable readout.
    """
    torch.manual_seed(0)
    m = _build(vocab=32, d_model_dec=16, n_layers_dec=2, gist_seq_len=4)
    # Doc A and B are distinct token streams (content tokens >= 3,避开 PAD/BOS/EOS).
    doc_a = torch.tensor([[BOS, 3, 3, 3, EOS]])
    doc_b = torch.tensor([[BOS, 4, 4, 4, EOS]])
    # Gist targets: doc-specific token (10 for A, 20 for B), EOS-terminated.
    gist_a = torch.tensor([[BOS, 10, EOS, PAD]])
    gist_b = torch.tensor([[BOS, 20, EOS, PAD]])
    _train_toy_readout(m, [(doc_a, gist_a), (doc_b, gist_b)], steps=600)

    state_a = m.encode(doc_a)
    state_b = m.encode(doc_b)
    out_a = m.decoder.generate(state_a, max_new_tokens=3, temperature=0.0)[0].tolist()
    out_b = m.decoder.generate(state_b, max_new_tokens=3, temperature=0.0)[0].tolist()

    # Main fidelity: each state decodes to its own doc-specific gist token.
    assert 10 in out_a, f"state_A should decode to gist token 10, got {out_a}"
    assert 20 in out_b, f"state_B should decode to gist token 20, got {out_b}"

    # Swap control: put B's state in A's slot (no slot/prompt exists -- only BOS,
    # so this is literally decoding state_b); the gist follows the state -> 20.
    # And vice versa. This is the inversion §3.3 failed (swap ~= main there).
    out_swap = m.decoder.generate(state_b, max_new_tokens=3, temperature=0.0)[0].tolist()
    assert 20 in out_swap, f"swapped (B's state) must follow the state -> 20, got {out_swap}"
    # Distinct states produce distinct gists (swapping CHANGES the output): the
    # literal anti-§3.3 property.
    assert out_a != out_b, "swapping states must change the decoded gist (§3.3 failed here)"