"""Generate targeted synthetic doc-kind training pairs (Phase 3c retrain lever).

The v2 head's binding failures are all "X -> snapshot" boundary confusions
(dec->snap 8/24, plan->snap 11/24) and snapshots WITHOUT the "as of" phrase
(snap->plan/other 5/20) -- the temporal feature keys on "as of" + heading-date,
so snapshots phrased differently and decisions/plans that carry a date leak to
snapshot. Real EnterpriseRAG-Bench docs under-represent these boundary cases
(the parquet is mostly whatever it is). This script GENERATES them to spec.

Pipeline per example:
  1. DeepSeek-flash generates a markdown doc to a boundary-targeted SPEC
     (returned as {"title","markdown"} JSON so it works through the existing
     json_object Oracle path).
  2. The markdown is parsed+chunked into the SAME section_texts the ingestion
     pipeline emits at serve (MarkdownParser -> HierarchicalChunker) -- no skew.
  3. The SAME DeepSeek-flash labeler prompt (label_doc_kind_corpus._PROMPT) re-
     labels the chunked text BLIND (it sees only the doc, not the spec). The
     record is KEPT only if the labeler agrees with the spec label AND
     confidence >= --min-confidence. This catches generator drift (a "decision
     that resembles a snapshot" spec that the model actually wrote as a
     snapshot) -- the same confidence-gate discipline as v2 real labeling.

Output: pairs JSONL ({"doc_id","section_texts","label","confidence","reason",
"spec"}) consumable by scripts/train_doc_kind_head.py (directly via --pairs, or
concatenated with the real v2 train by scripts/prep_doc_kind_v3_split.py).
Synthetic docs are tagged ``doc_id="synth-<spec_key>-<i>"`` so they can be
excluded from val (the trainer gets a fixed REAL val file -- synthetic is TRAIN
only; the val stays the 76 real v2 docs so before/after is measured on the same
distribution).

NOT production traffic. One-time offline data prep (mirrors
label_doc_kind_corpus.py). Bonsai/Oracle are NOT the trained head.

Usage:
    python scripts/generate_doc_kind_synthetic.py \\
        --out data/training/doc_kind_head/pairs_synth.jsonl \\
        --cache data/training/doc_kind_head/oracle_cache_synth.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# UTF-8 stdout -- generated doc text can be non-ASCII and cp1252 would crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 -- reconfigure is best-effort on some shells
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Reuse the labeler's prompt + label set + chunker helper so the verify step
# is byte-identical to the v2 real-labeling path (no drift between synthetic
# and real label semantics). Loaded via importlib (scripts/ is not a package).
_labeler_path = Path(__file__).resolve().parent / "label_doc_kind_corpus.py"
_spec = importlib.util.spec_from_file_location("_label_doc_kind_corpus", _labeler_path)
_labeler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_labeler)
_LABELS = _labeler._LABELS
_LABEL_PROMPT = _labeler._PROMPT            # the verify prompt (blind re-label)
_section_texts = _labeler._section_texts    # parse+chunk -> serve-shape section_texts


# Boundary-targeted generation specs. ``instruction`` is the spec the generator
# follows; ``label`` is the KNOWN-correct label (the spec prescribes it). The
# verify step re-labels blind and keeps only agreements, so a spec that the
# model mis-rendered is rejected. Counts chosen to over-weight the binding
# confusions (X->snap) and the as-of-less snapshots.
SPECS = [
    # --- the binding confusions (X -> snapshot) ---
    {"key": "snap_no_asof", "label": "point_in_time_snapshot", "count": 45,
     "instruction": (
         "a point-in-time snapshot recording observed state at a specific date. "
         "CRITICAL: do NOT use the phrases 'as of', 'as at', or 'as-of' anywhere. "
         "Instead phrase it as a status report, quarterly status, dashboard "
         "snapshot, or health check, with a date in a heading like "
         "'# Q1 2026 status' or in the body like 'current state on 2026-03-31'. "
         "It describes WHAT IS (observed state), NOT a decision and NOT a plan.")},
    {"key": "dec_like_snap", "label": "decision_update", "count": 55,
     "instruction": (
         "a decision_update that RESEMBLES a snapshot: it is dated and written in "
         "a status-report style, BUT it records a DECISION or CHANGE that was made "
         "(a switch, migration, adoption, approval, policy change, or ADR). It "
         "must convey a CHOICE that happened on a date, not just observed state. "
         "Example framing: 'On 2026-03-31 we switched from X to Y' or "
         "'Decision: adopt postgres for the metadata store'. The decision/change "
         "verb is what distinguishes it from a mere status snapshot.")},
    {"key": "plan_like_snap", "label": "plan", "count": 55,
     "instruction": (
         "a plan describing intended future work, that carries a date and some "
         "status-like framing BUT is clearly a PLAN/roadmap, not a snapshot of "
         "state and not a decision. Forward-looking, not-yet-executed. Use plan "
         "language: roadmap, will, going to, sprint, OKRs, milestone, targeting, "
         "next quarter. The date is a target/timeline, not an 'as of' state date.")},
    {"key": "dec_like_plan", "label": "decision_update", "count": 25,
     "instruction": (
         "a decision_update that is forward-looking BUT records a DECISION "
         "already made (past-tense choice affecting future work), e.g. 'we "
         "decided to adopt k8s for the migration'. Distinguish from a plan by "
         "the decision being MADE (past tense), not proposed or intended.")},
    # --- clean reinforcement (anchor the easy cases too, balanced) ---
    {"key": "snap_clean", "label": "point_in_time_snapshot", "count": 25,
     "instruction": (
         "a clear point-in-time snapshot using 'as of <date>' phrasing. A record "
         "of observed state, time-bound; a later snapshot supersedes it.")},
    {"key": "dec_clean", "label": "decision_update", "count": 25,
     "instruction": (
         "a clear decision_update / ADR. Records a decision made and its "
         "rationale; a later decision_update supersedes an earlier one.")},
    {"key": "plan_clean", "label": "plan", "count": 25,
     "instruction": (
         "a clear plan / roadmap / sprint plan. Intended future work, "
         "forward-looking, not yet executed.")},
    {"key": "other", "label": "other", "count": 20,
     "instruction": (
         "an enterprise doc that does not clearly fit snapshot/decision/plan/"
         "reference -- e.g. meeting notes, a changelog entry, an incident "
         "timeline, or an announcement.")},
    {"key": "reference", "label": "reference", "count": 15,
     "instruction": (
         "an evergreen reference doc -- a runbook, manual, how-to guide, "
         "architecture reference, or glossary. Not time-bound, not a decision.")},
]

_GEN_PROMPT = """You are generating a realistic enterprise document for a training corpus. Generate exactly ONE document matching this spec:

Spec: {instruction}

Requirements:
- Realistic, messy enterprise prose (not a toy example), the kind a real engineering org would write.
- 2-4 sections, each with a ## markdown heading, plus a short intro paragraph.
- ~200-400 words total.
- The document must UNAMBIGUOUSLY match the spec's semantic kind.

Return JSON with exactly two keys:
{{"title": "<a short document title>", "markdown": "<the full document in markdown, starting with a # title heading then ## section headings and body>"}}

Return ONLY the JSON object, no preamble or explanation.
"""


def _build_gen_prompt(instruction: str) -> str:
    return _GEN_PROMPT.format(instruction=instruction)


def _build_verify_prompt(sec_texts: list[str]) -> str:
    # Blind re-label: the verifier sees only the chunked doc text, NOT the spec
    # (so it can catch a spec the generator mis-rendered). Byte-identical prompt
    # to the v2 real-labeling path (no title context -- synthetic has none).
    from src.ingestion.doc_kind import join_section_texts
    return _LABEL_PROMPT + join_section_texts(sec_texts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate targeted synthetic doc-kind training pairs.",
    )
    ap.add_argument("--out", default="data/training/doc_kind_head/pairs_synth.jsonl",
                    help="output pairs JSONL (synthetic, TRAIN only)")
    ap.add_argument("--model", default="deepseek-v4-flash:cloud",
                    help="Oracle model (default deepseek-v4-flash:cloud)")
    ap.add_argument("--endpoint", default=None,
                    help="Oracle endpoint (default: config.oracle_endpoint)")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="concurrent Oracle calls (default 4)")
    ap.add_argument("--cache", default="data/training/doc_kind_head/oracle_cache_synth.json",
                    help="prompt-hash cache for resume")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="Oracle output cap (generation needs room for a doc; default 1024)")
    ap.add_argument("--min-confidence", type=float, default=0.7,
                    help="verify gate: keep only if labeler conf >= this (default 0.7)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale factor for spec counts (default 1.0)")
    ap.add_argument("--verbose", action="store_true",
                    help="print each keep/reject as it completes")
    args = ap.parse_args()

    from src.config import config as _config
    from src.ingestion.chunker import HierarchicalChunker
    from src.ingestion.parsers import MarkdownParser
    from src.subconscious.doc_kind_head import DocKindHead
    from src.training.oracle_labeling import OracleClient, OracleConfig

    assert _LABELS == DocKindHead.LABELS, (
        f"label set drift: labeler {_LABELS} != head {DocKindHead.LABELS}"
    )

    ic = _config.ingestion
    chunker = HierarchicalChunker(
        max_section_tokens=ic.max_section_tokens,
        min_section_tokens=ic.min_section_tokens,
        semantic_split_threshold=ic.semantic_split_threshold,
    )
    parser = MarkdownParser()

    cache_path = Path(args.cache) if args.cache else None
    # One client for both generate + verify (shared cache; think=False so flash
    # emits JSON directly without eating the token budget on CoT).
    oracle_cfg = OracleConfig(
        model=args.model,
        endpoint=args.endpoint or _config.oracle_endpoint,
        temperature=0.7,           # generation wants variety; verify is 0.1 below
        max_tokens=args.max_tokens,
        batch_delay=0.0,
        cache_path=cache_path,
        think=False,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    client = OracleClient(oracle_cfg)

    # Build the generation work list (spec, i) and their prompts.
    work: list[tuple[str, str, str]] = []   # (spec_key, spec_label, gen_prompt)
    for spec in SPECS:
        n = max(1, int(round(spec["count"] * args.scale)))
        for i in range(n):
            work.append((spec["key"], spec["label"], _build_gen_prompt(spec["instruction"])))
    print(f"generating {len(work)} synthetic docs across {len(SPECS)} specs "
          f"(scale={args.scale}, model={args.model})", flush=True)

    # --- phase 1: generate markdown docs ---
    gen_prompts = [w[2] for w in work]
    gen_results = client.generate_batch(
        gen_prompts, response_format="json_object", max_workers=args.max_workers,
    )

    # Chunk each generated doc; collect (work_idx, spec_key, spec_label, sec_texts).
    chunked: list[tuple[str, str, list[str]]] = []   # (spec_key, spec_label, sec_texts)
    gen_fail = 0
    for (spec_key, spec_label, _), res in zip(work, gen_results):
        if res.error or not isinstance(res.response, dict):
            gen_fail += 1
            continue
        md = res.response.get("markdown") or res.response.get("body") or ""
        title = str(res.response.get("title") or "").strip()
        if title and not md.lstrip().startswith("#"):
            md = f"# {title}\n\n{md}"
        sec = _section_texts(md, chunker, parser)
        if not sec:
            gen_fail += 1
            continue
        chunked.append((spec_key, spec_label, sec))
    print(f"phase 1 (generate): {len(chunked)} docs chunked, {gen_fail} failed/empty",
          flush=True)
    if not chunked:
        print("ERROR: no generated docs chunked successfully", file=sys.stderr)
        return 1

    # --- phase 2: blind verify (re-label with the v2 labeler prompt) ---
    verify_prompts = [_build_verify_prompt(sec) for (_k, _l, sec) in chunked]
    # Use a near-deterministic temperature for the verify pass by making a
    # second client (same cache dir; different temperature). Shared cache is
    # keyed on prompt hash, so generate + verify prompts never collide.
    verify_cfg = OracleConfig(
        model=args.model,
        endpoint=args.endpoint or _config.oracle_endpoint,
        temperature=0.1,           # near-deterministic, mirrors v2 real labeling
        max_tokens=768,
        batch_delay=0.0,
        cache_path=cache_path,
        think=False,
    )
    verify_client = OracleClient(verify_cfg)
    print(f"phase 2 (verify): re-labeling {len(chunked)} docs blind "
          f"(keep if labeler==spec AND conf>={args.min_confidence})", flush=True)
    verify_results = verify_client.generate_batch(
        verify_prompts, response_format="json_object", max_workers=args.max_workers,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    rejected_drift = 0
    rejected_lowconf = 0
    label_counts: dict[str, int] = {k: 0 for k in _LABELS}
    spec_kept: dict[str, int] = {s["key"]: 0 for s in SPECS}
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, ((spec_key, spec_label, sec), res) in enumerate(zip(chunked, verify_results)):
            if res.error or not isinstance(res.response, dict):
                rejected_drift += 1
                if args.verbose:
                    print(f"  REJECT(err)     synth-{spec_key}-{idx}: {res.error}", flush=True)
                continue
            vk = res.response.get("doc_kind")
            conf = res.response.get("confidence")
            if vk != spec_label:
                rejected_drift += 1
                if args.verbose:
                    print(f"  REJECT(drift)    synth-{spec_key}-{idx}: spec={spec_label} "
                          f"verifier={vk} conf={conf}", flush=True)
                continue
            if not isinstance(conf, (int, float)) or conf < args.min_confidence:
                rejected_lowconf += 1
                if args.verbose:
                    print(f"  REJECT(lowconf) synth-{spec_key}-{idx}: spec={spec_label} "
                          f"conf={conf} (gate {args.min_confidence})", flush=True)
                continue
            doc_id = f"synth-{spec_key}-{idx}"
            f.write(json.dumps({
                "doc_id": doc_id,
                "section_texts": sec,
                "label": spec_label,
                "confidence": round(float(conf), 3),
                "reason": str(res.response.get("reason") or "")[:200],
                "spec": spec_key,
            }, ensure_ascii=False) + "\n")
            kept += 1
            label_counts[spec_label] += 1
            spec_kept[spec_key] += 1
            if args.verbose:
                print(f"  KEEP            {doc_id}: {spec_label} conf={float(conf):.2f}",
                      flush=True)

    stats = client.get_stats()
    print(flush=True)
    print(f"wrote {kept} synthetic pairs to {out_path}", flush=True)
    print(f"label distribution: {dict(sorted(label_counts.items()))}", flush=True)
    print(f"per-spec kept: {dict(sorted(spec_kept.items()))}", flush=True)
    print(f"rejected: {rejected_drift} drift (spec!=verifier or error) + "
          f"{rejected_lowconf} low-conf (gate {args.min_confidence})", flush=True)
    print(f"oracle (gen client): {stats['total_calls']} calls, "
          f"{stats['cached_calls']} cached, {stats['total_tokens']} tokens, "
          f"${stats['total_cost']}", flush=True)
    if kept < 10:
        print(f"ERROR: only {kept} usable synthetic pairs. Re-run (cache skips "
              f"done docs) or raise --scale.", file=sys.stderr)
        return 1
    print(f"next: concat with v2 train (synthetic is TRAIN only; val stays the "
          f"76 real v2 docs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())