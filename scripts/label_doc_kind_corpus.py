"""Label a doc-kind training corpus with the DeepSeek-flash Oracle.

One-time OFFLINE data prep for the DocKindHead (Phase 3c Sec 7.11 deferred
step). Samples N docs from the EnterpriseRAG-Bench documents parquet, parses
each into the SAME post-chunker ``section_texts`` the ingestion pipeline
produces at serve time (MarkdownParser -> HierarchicalChunker -> per-section
text, so there is no train/serve skew), and asks DeepSeek-flash to label each
doc's semantic kind. Writes ``pairs.jsonl`` (``{"doc_id", "section_texts",
"label"}``) consumable directly by ``scripts/train_doc_kind_head.py --pairs``.

This is NOT production traffic. The trained head is what ships (a local
forward pass at ingest, no cloud); the Oracle only touches the corpus here,
once, to produce ground-truth labels stronger than the 8B zero-shot labels
Sec 7.11 writes at ingest (a different model family than Bonsai -> the head
learns a boundary that is not just "approximate 8B"). One call per doc.

Usage (full ~300-doc pass):
    python scripts/label_doc_kind_corpus.py --n 300 \\
        --parquet scripts/_scratch/erag/data/documents/test.parquet \\
        --out data/training/doc_kind_head/pairs.jsonl

Usage (smoke, 3 docs, verbose):
    python scripts/label_doc_kind_corpus.py --n 3 --verbose

The Oracle endpoint + model default to the Hippo config (localhost:11434/v1
proxy -> DeepSeek cloud); --model overrides (default deepseek-v4-flash:cloud,
the preferred teacher). A prompt-hash cache (--cache) makes a re-run skip
already-labeled docs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# UTF-8 stdout -- doc titles/content can be non-ASCII and cp1252 would crash
# (mirrors run_consolidation.py / ingest_document.py).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 -- reconfigure is best-effort on some shells
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# The 5 Sec 7.11 kinds, with definitions the Oracle uses to label
# consistently. Order is DocKindHead.LABELS (the head's canonical logit order);
# the label STRING is what we keep, so order only matters for the prompt
# presentation. Definitions mirror the guard semantics: point_in_time_snapshot
# -> complementary (ask_user), decision_update -> real conflict.
_LABELS = (
    "point_in_time_snapshot",
    "decision_update",
    "plan",
    "reference",
    "other",
)

_PROMPT = """You are labeling an enterprise document with exactly one of five semantic kinds. Read the document text and return the single best-fitting kind, plus a confidence and a short reason.

- point_in_time_snapshot: a record of state AS OF a date. A status report, quarterly snapshot, dashboard export, health check, "as of 2026-03-31 the deploy is green". Time-bound; a later snapshot supersedes it.
- decision_update: a record of a decision or change and its rationale. An ADR, a policy change, "we switched from X to Y", an architecture decision. A later decision_update supersedes an earlier one.
- plan: a description of intended future work. A roadmap, proposal, sprint plan, OKRs, project plan. Forward-looking, not yet executed.
- reference: evergreen reference material. Runbooks, manuals, how-to guides, architecture references, glossaries, onboarding docs. Not time-bound, not a decision.
- other: anything that does not clearly fit the above. If no kind clearly fits, return "other".

Key distinction -- be careful: a STATUS report ("state as of date") is point_in_time_snapshot; a DECISION record ("we decided to switch") is decision_update. They are easy to confuse. If the doc records an observation of current state, label it point_in_time_snapshot; if it records a choice that was made, label it decision_update.

Return JSON with exactly three keys:
{"doc_kind": "<one of the five labels above>", "confidence": <0.0-1.0 float>, "reason": "<at most 12 words explaining the choice>"}

`confidence` is how clearly the document fits the chosen kind (1.0 = unambiguous, 0.5 = plausible but uncertain). Keep `reason` to at most 12 words.

Document:
"""


def _build_prompt(title: str, doc_text: str) -> str:
    # Title is metadata the parquet carries separately; include it as context
    # for the labeler (the label is a doc-level property; this does not skew
    # serve-time -- the head trains on section_texts, not this prompt).
    header = f"Title: {title}\n\n" if title else ""
    return _PROMPT + header + doc_text


def _sample_row_indices(num_rows: int, n: int, seed: int) -> set[int]:
    """Deterministic uniform sample of ``n`` row indices from ``[0, num_rows)``."""
    rng = random.Random(seed)
    n = min(n, num_rows)
    return set(rng.sample(range(num_rows), n))


def _section_texts(content: str, chunker, parser) -> list[str]:
    """Parse + chunk ``content`` -> the per-section texts the pipeline emits.

    Mirrors ``UnifiedIngestionPipeline``: MarkdownParser.parse_text ->
    HierarchicalChunker.chunk -> (heading + "\\n" + content) per section. This
    is the exact post-chunk shape the head sees at serve time (no skew).
    """
    parsed = parser.parse_text(content)
    parsed = chunker.chunk(parsed)
    out = []
    for s in parsed.sections:
        text = (s.heading + "\n" + s.content) if s.heading else s.content
        if text and text.strip():
            out.append(text)
    return out


def filter_labeled_results(
    docs: list[tuple[str, str, list[str]]],
    results: list,
    min_confidence: float,
    labels: tuple[str, ...] = _LABELS,
) -> tuple[list[dict], dict[str, int], int, int, list[tuple[str, str, str]]]:
    """Filter index-aligned Oracle results into writeable JSONL records.

    The v1 corpus's 6/13 unsafe snapshot->decision_update confusion was teacher
    noise; the confidence gate is the fix. A failure sentinel (``result.error``
    set), an out-of-vocab label, OR a label with ``confidence < min_confidence``
    is rejected (counted, NOT written) -- a wrong/uncertain label would silently
    degrade the head. Returns ``(records, label_counts, failures, low_conf,
    verdicts)`` where each ``record`` is the JSONL payload (doc_id, section_texts,
    label, confidence, reason) and each ``verdict`` is ``(doc_id, status, line)``
    with ``status`` in ``{OK, FAIL, REJECT_OOV, REJECT_LOWCONF}`` -- the
    authoritative per-row outcome the caller prints in --verbose mode (so the
    verbose log can never drift from the filter logic). Rejected rows are
    retried on a re-run (the cache skips the OK docs). Pure function -- no I/O --
    so the confidence gate is unit-testable without an Oracle call.
    """
    label_counts: dict[str, int] = {k: 0 for k in labels}
    records: list[dict] = []
    verdicts: list[tuple[str, str, str]] = []
    failures = 0
    low_conf = 0
    for (doc_id, _title, sec_texts), result in zip(docs, results):
        if result.error:
            failures += 1
            verdicts.append((doc_id, "FAIL", f"error: {result.error}"))
            continue
        kind = result.response.get("doc_kind")
        if kind not in labels:
            failures += 1
            verdicts.append((doc_id, "REJECT_OOV", f"out-of-vocab label {kind!r}"))
            continue
        conf = result.response.get("confidence")
        if not isinstance(conf, (int, float)) or conf < min_confidence:
            low_conf += 1
            verdicts.append((doc_id, "REJECT_LOWCONF",
                            f"low confidence {conf!r} (gate {min_confidence})"))
            continue
        label_counts[kind] += 1
        records.append({
            "doc_id": doc_id,
            "section_texts": sec_texts,
            "label": kind,
            "confidence": round(float(conf), 3),
            "reason": str(result.response.get("reason") or "")[:200],
        })
        verdicts.append((doc_id, "OK", f"{kind} (conf={float(conf):.2f})"))
    return records, label_counts, failures, low_conf, verdicts


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Label a doc-kind training corpus with the DeepSeek-flash Oracle.",
    )
    ap.add_argument("--parquet",
                    default="scripts/_scratch/erag/data/documents/test.parquet",
                    help="EnterpriseRAG-Bench documents parquet (doc_id,title,content)")
    ap.add_argument("--n", type=int, default=300,
                    help="number of docs to sample + label (default 300)")
    ap.add_argument("--out", default="data/training/doc_kind_head/pairs.jsonl",
                    help="output pairs JSONL (consumed by train_doc_kind_head.py --pairs)")
    ap.add_argument("--model", default="deepseek-v4-flash:cloud",
                    help="Oracle model (default deepseek-v4-flash:cloud)")
    ap.add_argument("--endpoint", default=None,
                    help="Oracle endpoint (default: config.oracle_endpoint)")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="concurrent Oracle calls (default 4)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed (deterministic sample + train/val split)")
    ap.add_argument("--cache", default="data/training/doc_kind_head/oracle_cache.json",
                    help="prompt-hash cache for resume (re-run skips labeled docs)")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="cap on doc text sent to the Oracle (mirrors _BONSAI_TEXT_CAP)")
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="Oracle output token cap (default 768; the JSON now carries confidence+reason)")
    ap.add_argument("--min-confidence", type=float, default=0.7,
                    help="reject labels with confidence below this (0.0=keep all; default 0.7)")
    ap.add_argument("--verbose", action="store_true",
                    help="print each doc_id + label as it completes")
    args = ap.parse_args()

    from src.config import config as _config
    from src.ingestion.chunker import HierarchicalChunker
    from src.ingestion.doc_kind import join_section_texts
    from src.ingestion.parsers import MarkdownParser
    from src.subconscious.doc_kind_head import DocKindHead
    from src.training.oracle_labeling import OracleClient, OracleConfig

    # Sanity: the prompt's label set must match the head's canonical LABELS,
    # or the exported pairs would train the head on a mismatched vocabulary.
    assert _LABELS == DocKindHead.LABELS, (
        f"label set drift: prompt {_LABELS} != head {DocKindHead.LABELS}"
    )

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"ERROR: parquet not found at {parquet_path}", file=sys.stderr)
        return 1

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    num_rows = pf.metadata.num_rows
    targets = _sample_row_indices(num_rows, args.n, args.seed)
    print(f"parquet: {num_rows} rows; sampling {len(targets)} docs (seed={args.seed})",
          flush=True)
    if not targets:
        print("ERROR: empty sample (--n 0?)", file=sys.stderr)
        return 1

    # Stream batches so the 1.4 GB / 512k-row content column never loads whole.
    ic = _config.ingestion
    chunker = HierarchicalChunker(
        max_section_tokens=ic.max_section_tokens,
        min_section_tokens=ic.min_section_tokens,
        semantic_split_threshold=ic.semantic_split_threshold,
    )
    parser = MarkdownParser()

    docs: list[tuple[str, str, list[str]]] = []  # (doc_id, title, section_texts)
    last_target = max(targets)
    global_idx = 0
    done = False
    for batch in pf.iter_batches(batch_size=8192, columns=["doc_id", "title", "content"]):
        doc_ids = batch.column("doc_id").to_pylist()
        titles = batch.column("title").to_pylist()
        contents = batch.column("content").to_pylist()
        for i in range(len(doc_ids)):
            if global_idx in targets:
                content = contents[i] or ""
                sec_texts = _section_texts(content, chunker, parser)
                if sec_texts:  # skip docs that parse to no sections
                    docs.append((str(doc_ids[i]), str(titles[i] or ""), sec_texts))
            if global_idx >= last_target:
                done = True
                break
            global_idx += 1
        if done:
            break

    print(f"parsed {len(docs)} docs with >=1 section "
          f"({len(targets) - len(docs)} parsed to zero sections, skipped)", flush=True)
    if not docs:
        print("ERROR: no usable docs (all parsed to zero sections). "
              "Check the parquet content column.", file=sys.stderr)
        return 1

    # Build one prompt per doc (join_section_texts = byte-identical to the
    # Bonsai tagger's input, capped at max-chars).
    prompts = [_build_prompt(title, join_section_texts(sec_texts, cap=args.max_chars))
               for (_did, title, sec_texts) in docs]

    cache_path = Path(args.cache) if args.cache else None
    oracle_cfg = OracleConfig(
        model=args.model,
        endpoint=args.endpoint or _config.oracle_endpoint,
        temperature=0.1,           # near-deterministic for label consistency
        max_tokens=args.max_tokens,  # one JSON object (doc_kind+confidence+reason); think OFF
        batch_delay=0.0,
        cache_path=cache_path,
        # deepseek-v4-flash is a REASONING model: under the OpenAI /v1 path the
        # reasoning CoT shares the max_tokens budget with content, so a small
        # cap is eaten by reasoning and content comes back EMPTY (truncation).
        # think=False routes through Ollama's native /api/chat which honors the
        # flag -- flash then emits the JSON object directly (no CoT), so output
        # tokens are cheap. The v2 prompt also asks for confidence + a <=12-word
        # reason (~30-50 tokens); 768 is plenty. Verified on a runbook: 9 tokens
        # for the v1 {doc_kind}-only object.
        think=False,
    )
    # Validate the cache dir exists before the client tries to flush to it.
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    client = OracleClient(oracle_cfg)
    print(f"labeling {len(docs)} docs with {args.model} "
          f"(max_workers={args.max_workers}, cache={cache_path})", flush=True)

    # generate_batch returns results in INPUT ORDER (one per prompt), with a
    # failure sentinel (response={}, error set) at any index that exhausted
    # retries -- so results align 1:1 with `docs`/`prompts` by position.
    results = client.generate_batch(
        prompts, response_format="json_object",
        max_workers=args.max_workers,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # filter_labeled_results is pure (no I/O); the write loop just serializes.
    # verdicts is the authoritative per-row outcome -- print it in verbose mode
    # so the diagnostic log can never drift from the filter logic.
    records, label_counts, failures, low_conf, verdicts = filter_labeled_results(
        docs, results, args.min_confidence)
    if args.verbose:
        for doc_id, status, line in verdicts:
            print(f"  {status:<13} {doc_id}: {line}", flush=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    stats = client.get_stats()
    print(flush=True)
    print(f"wrote {written} pairs to {out_path}", flush=True)
    print(f"label distribution: {dict(sorted(label_counts.items()))}", flush=True)
    print(f"oracle: {stats['total_calls']} calls, {stats['cached_calls']} cached, "
          f"{stats['total_tokens']} tokens, ${stats['total_cost']}", flush=True)
    print(f"failures/rejections: {failures} (low-confidence skipped: {low_conf}, "
          f"gate={args.min_confidence})", flush=True)
    if failures or low_conf:
        print("note: skipped rows are not written; re-run (cache skips the OK docs) "
              "to retry them", flush=True)
    if written < 10:
        print(f"ERROR: only {written} usable pairs -- need >=10 to train. "
              f"Raise --n, lower --min-confidence, or re-run to retry failures.",
              file=sys.stderr)
        return 1
    print(f"next: python scripts/train_doc_kind_head.py --pairs {out_path} "
          f"--embed-source on-demand --device auto", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())