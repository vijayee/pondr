"""Cross-encoder re-ranker for the retrieval pipeline.

Targets the LongMemEval multi-session failure mode (the (2) lever): relevant
cross-session episodes are surfaced by RRF recall but buried at rank 4-20 under
off-topic hits, so the answer LLM counts/synthesizes from a noisy top-K. A
dedicated cross-encoder (``BAAI/bge-reranker-v2-m3``) re-scores each candidate's
FULL text against the query and re-orders -- a DIFFERENT signal than the
DeepSeek same-model re-rank (fix (2)) that scored 38/50 (added nothing). The
cross-encoder scores query-doc interaction directly (not bi-embedding cosine),
so it can demote a lexically-similar-but-off-topic hit that graph+vector+BM25
all ranked high.

Flag-gated (``config.rerank_enabled``), default OFF, byte-identical when off
(the call site is a guarded no-op: ``if config.rerank_enabled and self.reranker
is not None``).

Implementation: uses ``transformers`` directly (``AutoModelForSequenceClassification``
+ ``AutoTokenizer``), NOT ``sentence_transformers.CrossEncoder``. The harness
eval repo ships a ``datasets/`` benchmark package that shadows HuggingFace
``datasets`` at import time, which ``sentence_transformers`` imports -- so
``sentence_transformers`` is unimportable in the harness env (the A/B experiment
runs there). ``transformers`` alone has no ``datasets`` import, so this module
works in BOTH the Pondr env (live serve) and the harness env (the experiment).
bge-reranker-v2-m3 is a standard 1-label sequence-classification transformer;
the score is the raw logit (monotonic -> rank-equivalent to sigmoid). Lazy-
imported so this module imports without ``transformers`` installed (tests +
offline tools). No new dependency: ``transformers`` is already a Pondr dep
(``build_embedder`` pulls it transitively, and the encoder uses it directly).

Device handling mirrors ``build_embedder`` / GLiNER (``[[gliner-gpu-config-task]]``):
``device="auto"`` -> CUDA if available, else CPU. The CUDA move is OOM-safe
(catches the runtime error and falls back to CPU). The model is loaded ONCE
(lazily, on the first ``rerank`` call) and reused; ``rerank`` is stateless
after that first load.
"""

from __future__ import annotations

from typing import Any, Optional


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` -> ``"cuda"`` if available, else ``"cpu"``.

    Mirrors the GLiNER ``device="auto"`` convention (CUDA w/ CPU fallback). A
    concrete device string (``"cpu"`` / ``"cuda"`` / ``"cuda:1"``) is passed
    through unchanged so callers can pin a device.
    """
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _result_text(r: dict) -> str:
    """The text we re-rank a result against the query by.

    Prefers ``text`` (the episode's full_text -- the most discriminative
    surface; the same field BM25 indexes), then ``source_evidence`` (the harness
    adapter's name for the same full_text -- the saved-search result shape),
    then ``summary`` / ``memory`` (the gist -- for a hydrated graph hit whose
    full_text was truncated, or the harness adapter's gist field), then ``""``
    so an empty result never breaks the scorer (``CrossEncoder.predict`` on an
    empty string returns a low score, which is the honest outcome for a
    contentless candidate). Handling BOTH the retriever's internal dict shape
    (``text``/``summary``) and the harness-mapped shape (``source_evidence``/
    ``memory``) lets the same reranker re-rank live pipeline results AND saved
    search-results files (the A/B experiment reuses saved searches).
    """
    return (
        r.get("text")
        or r.get("source_evidence")
        or r.get("summary")
        or r.get("memory")
        or ""
    )


class CrossEncoderReranker:
    """A cross-encoder re-ranker over ``transformers`` sequence-classification.

    bge-reranker-v2-m3 is a 1-label ``AutoModelForSequenceClassification``: the
    relevance score for a (query, doc) pair is the model's single output logit
    (raw, monotonic -> rank-equivalent to sigmoid). Constructed ONCE (model +
    tokenizer loaded + moved to device) and reused across calls. ``rerank``
    scores each result's text vs the query and returns a NEW list in score-desc
    order. The input list is never mutated. On ANY error (model load failure,
    predict failure, OOM) ``rerank`` returns the input list UNCHANGED -- the
    failure-fallback is "graceful no-op" (retrieval stays correct, just un-re-
    ranked), matching the codebase convention that an optional stage never
    degrades the baseline.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        # Lazily loaded so the module import is side-effect-free; ``_model`` /
        # ``_tokenizer`` are None until the first ``rerank``.
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: Optional[str] = None

    # ── lazy load ───────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load tokenizer + model on first use (CUDA w/ CPU fallback).

        Raises are caught at the ``rerank`` boundary so a load failure degrades
        to the no-op fallback rather than crashing retrieval. Loaded once and
        cached on ``self._model`` / ``self._tokenizer``.
        """
        if self._model is not None:
            return
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification, AutoTokenizer,
        )
        resolved = _resolve_device(self.device)
        try:
            self._load_on(resolved, AutoModelForSequenceClassification, AutoTokenizer)
        except (OSError, RuntimeError, ValueError, EnvironmentError):
            # CUDA OOM / unavailable after the auto check, or a corrupt load ->
            # retry on CPU (the honest fallback; the model still works, just
            # slower). If THIS fails too, ``rerank`` catches it -> no-op. Skip
            # the retry when we already resolved to CPU -- an identical retry
            # would just re-fail the same way and delay the no-op fallback.
            if resolved == "cpu":
                raise
            self._load_on("cpu", AutoModelForSequenceClassification, AutoTokenizer)

    def _load_on(self, device: str, model_cls, tok_cls) -> None:
        tok = tok_cls.from_pretrained(self.model_name)
        model = model_cls.from_pretrained(self.model_name)
        model.eval()
        model.to(device)
        self._tokenizer = tok
        self._model = model
        self._device = device

    # ── public ──────────────────────────────────────────────────────────────

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Score (query, text) pairs with the loaded model, batched.

        Returns one float per pair (the raw logit). Pairs are tokenized as
        paired sequence-classification input (query, text) with padding +
        truncation to ``max_length``; the model's single output logit per pair
        is the relevance score. Batched by ``batch_size`` to bound GPU memory.
        """
        import torch  # type: ignore
        tok = self._tokenizer
        model = self._model
        device = self._device
        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = tok(
                [query] * len(batch_texts), batch_texts,
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits.squeeze(-1)
            # ``logits`` may be shape () (single pair), (n,) or (n, 1) ->
            # ``tolist`` flattens all to plain floats.
            scores.extend(float(s) for s in logits.tolist())
        return scores

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Re-order ``results`` by cross-encoder(query, result_text) score, desc.

        Returns a NEW list; ``results`` is not mutated. On any failure (load,
        predict, OOM) returns ``results`` unchanged (the graceful no-op). When
        ``top_k`` is set, returns at most that many (after sorting); ``None``
        preserves the input length (re-order only, no truncation -- the
        retriever's ``default_retrieval_limit`` already capped the candidate
        set, so re-ranking does not need to re-cap unless the caller asks).

        The score is stamped onto each returned dict as ``rerank_score`` (a
        float) so a caller/test can inspect the new ordering; the original
        ``score`` field is preserved (the cross-encoder score is a DIFFERENT
        signal than the graph/vector/BM25 score and should not overwrite it).
        """
        if not results:
            return list(results)
        try:
            self._ensure_loaded()
            texts = [_result_text(r) for r in results]
            scores = self._score_pairs(query, texts)
        except Exception:  # noqa: BLE001 - any failure -> graceful no-op
            # Load failure, predict failure, OOM, or transformers missing.
            # Return the input unchanged; retrieval stays correct, just un-re-
            # ranked.
            return list(results)

        # Sort by score desc. Python's sort is stable, so equal scores retain
        # their original (input) order even under ``reverse=True`` -- no
        # re-ordering noise for ties, no explicit tiebreak needed.
        order = sorted(
            range(len(results)),
            key=lambda i: scores[i],
            reverse=True,
        )
        out: list[dict] = []
        for i in order:
            d = dict(results[i])  # shallow copy so we don't mutate the input
            d["rerank_score"] = scores[i]
            out.append(d)
        if top_k is not None:
            out = out[:max(0, int(top_k))]
        return out