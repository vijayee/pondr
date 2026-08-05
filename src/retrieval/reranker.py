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
is not None``). No new dependency: ``sentence_transformers`` is already a Pondr
dep (``build_embedder`` lazy-imports ``SentenceTransformer``; ``CrossEncoder``
ships in the same package). Lazy-imported here too so this module imports
without the package installed (tests + offline tools).

Device handling mirrors ``build_embedder`` / GLiNER (``[[gliner-gpu-config-task]]``):
``device="auto"`` -> CUDA if available, else CPU. The CUDA move is OOM-safe
(catches the runtime error and falls back to CPU). ``CrossEncoder`` is loaded
ONCE in the ctor and reused; ``rerank`` is stateless after that.
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
    """A cross-encoder re-ranker wrapping ``sentence_transformers.CrossEncoder``.

    Constructed ONCE (model loaded + moved to device) and reused across calls.
    ``rerank`` scores each result's text vs the query and returns a NEW list in
    score-desc order. The input list is never mutated. On ANY error (model load
    failure, predict failure, OOM) ``rerank`` returns the input list UNCHANGED
    -- the failure-fallback is "graceful no-op" (retrieval stays correct, just
    un-re-ranked), matching the codebase convention that an optional stage never
    degrades the baseline.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._ce: Optional[Any] = None  # lazily loaded so import is side-effect-free

    # ── lazy load ───────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> Any:
        """Load the CrossEncoder on first use (CUDA w/ CPU fallback).

        Raises are caught at the ``rerank`` boundary so a load failure degrades
        to the no-op fallback rather than crashing retrieval. Loaded once and
        cached on ``self._ce``.
        """
        if self._ce is not None:
            return self._ce
        from sentence_transformers import CrossEncoder  # type: ignore
        resolved = _resolve_device(self.device)
        try:
            self._ce = CrossEncoder(
                self.model_name, max_length=self.max_length, device=resolved,
            )
        except (OSError, RuntimeError, ValueError):
            # CUDA OOM / unavailable after the auto check, or a corrupt load ->
            # retry on CPU (the honest fallback; the model still works, just
            # slower). If THIS fails too, ``rerank`` catches it -> no-op. Skip
            # the retry when we already resolved to CPU -- an identical retry
            # would just re-fail the same way and delay the no-op fallback.
            if resolved == "cpu":
                raise
            self._ce = CrossEncoder(
                self.model_name, max_length=self.max_length, device="cpu",
            )
        return self._ce

    # ── public ──────────────────────────────────────────────────────────────

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
            ce = self._ensure_loaded()
            pairs = [(query, _result_text(r)) for r in results]
            scores = ce.predict(pairs)
            # CrossEncoder.predict returns an ndarray; coerce to plain floats so
            # the stamped ``rerank_score`` is JSON-serializable (the harness
            # round-trips result dicts through json).
            try:
                import numpy as np  # type: ignore
                scores = [float(s) for s in np.asarray(scores).reshape(-1)]
            except ImportError:
                scores = [float(s) for s in scores]
        except Exception:  # noqa: BLE001 - any failure -> graceful no-op
            # Load failure, predict failure, OOM, or numpy missing. Return the
            # input unchanged; retrieval stays correct, just un-re-ranked.
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