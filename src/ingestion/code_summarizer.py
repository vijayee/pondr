"""LLM prose summaries for code sections (STRM 1f-6).

A code section's *embedding handle* -- the string the document retriever's
cosine surfacing AND the STRM backbone's inject-time ``y_t`` (-> state ``h``)
locate it by -- is computed from the raw source today, so code slots mis-rank
against prose queries. The 1f-5 doc-kind split showed code-doc z_logit median
-0.801 vs text-doc 3.969 ([[pondr-strm-phase1f5-margin-dockind-result]]). This
module generates a 1-2 sentence natural-language description of each code
section at ingest time so the embedding can be computed from prose instead,
while the actual source stays as the recalled content (the ``text`` field is
untouched; only a parallel ``sec.summary`` / ``embed_text`` handle changes --
see the 1f-6 plan).

Never raises: a down server, non-200, non-JSON, or missing content routes to
``None`` *per section* so the pipeline falls back to embedding the raw source
(byte-identical to a not-wired ingest). Mirrors
``src/gnn/bonsai_decider.py: BonsaiDecider._post_json``'s cold-start-safe
philosophy and reuses ``src/training/oracle_labeling.py: OracleClient`` for the
cache + retry + stats plumbing (re-ingesting the same repo = cache hits). The
JSON envelope ``{"summary": "..."}`` keeps the call inside OracleClient's
``response_format: json_object`` path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from src.training.oracle_labeling import OracleClient, OracleConfig
from src.training.prompts import code_section_summary_prompt

# The synthetic module-root section heading emitted by CodeParser for the
# imports / module docstring / top-level statements span (see
# ``src/ingestion/code_parser.py``). It is skipped here: it rarely matches a
# prose query, summarizing it wastes a call, and leaving it ``None`` makes its
# embedding fall back to raw content (byte-identical to today).
_MODULE_ROOT_HEADING = "<module>"


class CodeSectionSummarizer:
    """Generate a 1-2 sentence prose summary per code section; never raises.

    Constructible offline (the Oracle connection is lazy). ``summarize_sections``
    is the only public method and is cold-start safe: any failure class returns
    ``None`` for that section (or for the whole batch on a hard failure) so the
    ingestion pipeline never aborts and degrades to embedding the raw source.
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash:cloud",
        endpoint: Optional[str] = None,
        temperature: float = 0.1,
        timeout: float = 60.0,
        max_tokens: int = 320,
        max_workers: int = 4,
        cache_path: Optional[str] = None,
    ) -> None:
        cfg = OracleConfig(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if endpoint:
            cfg.endpoint = endpoint
        if cache_path:
            cfg.cache_path = Path(cache_path)
        self._client = OracleClient(cfg)
        self._max_workers = max_workers

    def summarize_sections(
        self, items: list[tuple[str, str]]
    ) -> list[Optional[str]]:
        """Return one prose summary per ``(heading, content)`` item, or ``None``.

        Never raises. The ``<module>`` root section and empty/whitespace
        content are skipped -> ``None`` (their embeddings fall back to raw
        content). On a hard batch failure (server down) the whole result is
        all-``None`` and a single ``[code-summarizer] ...`` line is logged to
        stderr so a silently-degraded run is visible. Per-section parse
        failures are counted and logged once.
        """
        n = len(items)
        results: list[Optional[str]] = [None] * n
        if n == 0:
            return results

        prompts: list[str] = []
        idx_map: list[int] = []
        for i, (heading, content) in enumerate(items):
            if heading == _MODULE_ROOT_HEADING:
                continue
            if not content or not content.strip():
                continue
            prompts.append(code_section_summary_prompt(heading, content))
            idx_map.append(i)
        if not prompts:
            return results

        try:
            oracle_results = self._client.generate_batch(
                prompts,
                response_format="json_object",
                max_workers=self._max_workers,
            )
        except Exception as e:  # noqa: BLE001 - never-raise contract
            print(f"[code-summarizer] batch failed: {e}", file=sys.stderr)
            return results  # all None -> pipeline falls back to raw content

        # generate_batch promises one result per prompt (sentinel on failure),
        # but guard a misalignment defensively rather than risk an index error.
        if len(oracle_results) != len(prompts):
            print(
                f"[code-summarizer] batch length mismatch "
                f"({len(oracle_results)} != {len(prompts)}); skipping",
                file=sys.stderr,
            )
            return results

        n_failed = 0
        for j, ores in enumerate(oracle_results):
            i = idx_map[j]
            if ores is None or ores.error or not isinstance(ores.response, dict):
                n_failed += 1
                continue
            summary = ores.response.get("summary")
            if isinstance(summary, str):
                summary = summary.strip()
            if not summary:
                n_failed += 1
                continue
            results[i] = summary

        if n_failed:
            print(
                f"[code-summarizer] {n_failed}/{len(prompts)} sections returned "
                f"no summary (degraded to raw-content embedding)",
                file=sys.stderr,
            )
        return results

    def get_stats(self) -> dict:
        """Oracle usage stats (calls, cache hits, tokens) for run reporting."""
        return self._client.get_stats()