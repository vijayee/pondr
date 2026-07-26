"""BPE tokenizer for the token-level LM-SSM.

Wraps the HF ``tokenizers`` library (installed, no third-party repo dep) to
train a small BPE on the public ERAG corpus and expose a plain
encode/decode/encode_batch surface. Right-sized vocab (4096) keeps the LM's
embedding table small (no 50k-vocab GPT-2 emb bloat); the small LM is the
content-objective proof, not a general-purpose LM.

The trained tokenizer is cached as JSON so re-runs are free. Trained on public
ERAG text only -- no onyx, no private transcripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


# Special-token ids are fixed (the LMConfig mirrors them). Keeping them here as
# the single source of truth so the tokenizer and the model config never drift.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class TokenizerWrapper:
    """Thin wrapper over a trained HF ``Tokenizer``.

    Exposes ``encode`` (str -> list[int], BOS/EOS wrapped), ``decode``
    (list[int] -> str, specials stripped), ``encode_batch`` (with optional
    padding/truncation), and ``vocab_size``. The underlying ``Tokenizer`` is
    available at ``.tok`` for callers that need the raw API.
    """

    def __init__(self, tok: Tokenizer, vocab: int):
        self.tok = tok
        self._vocab = vocab

    # ----------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return self._vocab

    # ----------------------------------------------------------------- encode
    def encode(self, text: str) -> list[int]:
        """``text`` -> token ids, with ``<bos>`` ... ``<eos>`` wrapping.

        Resets any padding/truncation state a prior ``encode_batch(max_length=)``
        may have left on the shared tokenizer (which would otherwise silently
        pad/truncate single-string encodes). Returns the raw, unpadded ids."""
        self.tok.no_padding()
        self.tok.no_truncation()
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str], max_length: Optional[int] = None) -> list[list[int]]:
        """Batch encode. If ``max_length`` is set, the tokenizer truncates
        (keeping BOS, dropping the tail before EOS) and pads to ``max_length``
        with ``<pad>`` (id 0). Returns a list of equal-length id lists."""
        # Re-configure padding/truncation around the requested length. The HF
        # tokenizer mutates state; this is cheap and keeps the call
        # self-contained.
        if max_length is not None:
            self.tok.enable_truncation(max_length=max_length)
            self.tok.enable_padding(length=max_length, pad_id=PAD_ID, pad_token=PAD_TOKEN)
        else:
            self.tok.no_padding()
            self.tok.no_truncation()
        return [enc.ids for enc in self.tok.encode_batch(texts)]

    # ----------------------------------------------------------------- decode
    def decode(self, ids: Iterable[int]) -> str:
        """ids -> str, with special tokens (``<bos>``/``<eos>``/``<pad>``)
        stripped."""
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def decode_batch(self, batch: Iterable[Iterable[int]]) -> list[str]:
        return [self.tok.decode(list(ids), skip_special_tokens=True) for ids in batch]


def _build_bpe(vocab_size: int) -> tuple[Tokenizer, BpeTrainer]:
    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tok.pre_tokenizer = Whitespace()
    # Wrap every encoded sequence with <bos> ... <eos> (single-sequence
    # template). Decode strips the specials via skip_special_tokens.
    tok.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, BOS_ID), (EOS_TOKEN, EOS_ID)],
    )
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    return tok, trainer


def train_tokenizer(
    corpus: Iterable[str],
    vocab_size: int = 4096,
) -> TokenizerWrapper:
    """Train a BPE tokenizer on ``corpus`` (an iterable of text strings).

    Returns a ``TokenizerWrapper``. The corpus is consumed once as an
    iterator (streamable, so the 511k-doc ERAG set fits without loading all
    text into memory at once).
    """
    tok, trainer = _build_bpe(vocab_size)
    tok.train_from_iterator(corpus, trainer)
    actual = tok.get_vocab_size(with_added_tokens=False)
    return TokenizerWrapper(tok, actual)


def train_or_load_tokenizer(
    corpus: Iterable[str],
    cache_path: str | Path,
    vocab_size: int = 4096,
) -> TokenizerWrapper:
    """Load from ``cache_path`` if it exists; else train on ``corpus`` and save.

    The cache is the tokenizer JSON. The corpus is ONLY consumed if the cache
    is missing. Vocab size is read from the cached tokenizer on load (so a
    re-run with a different ``vocab_size`` requires deleting the cache).
    """
    cache = Path(cache_path)
    if cache.exists():
        tok = Tokenizer.from_file(str(cache))
        return TokenizerWrapper(tok, tok.get_vocab_size(with_added_tokens=False))
    tok, trainer = _build_bpe(vocab_size)
    tok.train_from_iterator(corpus, trainer)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(cache))
    actual = tok.get_vocab_size(with_added_tokens=False)
    return TokenizerWrapper(tok, actual)