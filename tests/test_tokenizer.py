"""Unit tests for the BPE tokenizer wrapper (``src/subconscious/tokenizer_.py``).

Trains a tiny BPE on an in-memory corpus (no ERAG dependency); exercises
encode/decode round-trip, BOS/EOS wrapping, padding/truncation, special-token
ids, and the train-or-load cache path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.subconscious.tokenizer_ import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    TokenizerWrapper,
    train_or_load_tokenizer,
    train_tokenizer,
)

# A tiny but varied corpus -- enough for a 256-vocab BPE to learn real tokens.
CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog was lazy and the fox was quick",
    "a stitch in time saves nine",
    "nine stitches save time and the dog too",
    "the brown fox and the lazy dog meet the quick fox",
    "time flies when the quick fox jumps over the lazy dog",
] * 30


def test_train_tokenizer_vocab_and_specials():
    tok = train_tokenizer(CORPUS, vocab_size=256)
    assert tok.vocab_size <= 256
    # Special tokens are present at the fixed ids.
    assert tok.tok.token_to_id("<pad>") == PAD_ID
    assert tok.tok.token_to_id("<bos>") == BOS_ID
    assert tok.tok.token_to_id("<eos>") == EOS_ID


def test_encode_wraps_bos_eos():
    tok = train_tokenizer(CORPUS, vocab_size=256)
    ids = tok.encode("the quick fox")
    # BOS ... EOS wrapping (single-sequence template).
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert len(ids) >= 3  # bos + at least one content token + eos


def test_encode_decode_roundtrip():
    tok = train_tokenizer(CORPUS, vocab_size=256)
    text = "the quick brown fox"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    # The content survives; specials are stripped.
    assert decoded.strip() == text


def test_encode_batch_pads_and_truncates():
    tok = train_tokenizer(CORPUS, vocab_size=256)
    batch = tok.encode_batch(["the quick fox", "the dog"], max_length=8)
    assert all(len(b) == 8 for b in batch)
    assert all(b[0] == BOS_ID for b in batch)
    # Padding id fills the tail.
    flat = [i for b in batch for i in b]
    assert PAD_ID in flat


def test_decode_batch_strips_specials():
    tok = train_tokenizer(CORPUS, vocab_size=256)
    texts = ["the quick fox", "the lazy dog"]
    batch_ids = tok.encode_batch(texts, max_length=10)
    decoded = tok.decode_batch(batch_ids)
    assert decoded[0].strip() == "the quick fox"
    assert decoded[1].strip() == "the lazy dog"


def test_train_or_load_caches():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "tok.json"
        tok1 = train_or_load_tokenizer(CORPUS, cache, vocab_size=256)
        assert cache.exists()
        # Second call loads from cache (corpus NOT consumed -- pass a generator
        # that would blow up if iterated, to prove the cache path is taken).
        def explode():
            raise AssertionError("corpus should not be consumed on cache hit")
            yield  # makes `explode` a generator function; raise fires on iter
        tok2 = train_or_load_tokenizer(explode(), cache, vocab_size=256)
        assert tok2.vocab_size == tok1.vocab_size
        # Same tokenizer content (encode the same string identically).
        assert tok2.encode("the quick fox") == tok1.encode("the quick fox")


def test_encode_resets_padding_state_from_prior_batch():
    """Regression: ``encode_batch(max_length=...)`` mutates the shared tokenizer's
    padding/truncation state. A subsequent ``encode()`` must NOT inherit that
    state (it must return the raw, unpadded/untruncated ids)."""
    tok = train_tokenizer(CORPUS, vocab_size=256)
    # Prime the shared tokenizer with padding + truncation.
    tok.encode_batch(["the quick fox", "the dog"], max_length=8)
    raw = tok.encode("the quick brown fox")
    # Raw (no padding): BOS + content tokens + EOS, length driven by the text,
    # NOT 8. If padding leaked in, this would be exactly 8 (and tail would be
    # PAD_ID=0).
    assert raw[0] == BOS_ID and raw[-1] == EOS_ID
    assert len(raw) != 8 or raw[-2] != PAD_ID, "padding leaked into encode()"
    # And a long string is not truncated to 8.
    long_ids = tok.encode(" ".join(["the quick fox"] * 20))
    assert len(long_ids) > 8, "truncation leaked into encode()"


def test_model_uses_tokenizer_vocab_size():
    """The LM's vocab dimension must match the tokenizer's actual vocab, not a
    hard-coded 4096. This is the wiring that prevents a vocab-mismatch crash
    at train time."""
    from src.subconscious.token_lm import LMConfig, SSMLanguageModel

    tok = train_tokenizer(CORPUS, vocab_size=256)
    cfg = LMConfig(vocab=tok.vocab_size, d_model=16, n_layers=1, d_state=4, seq_len=8)
    m = SSMLanguageModel(cfg)
    ids = torch.tensor([tok.encode("the quick fox")])
    logits, _ = m.forward(ids)
    assert logits.shape[-1] == tok.vocab_size