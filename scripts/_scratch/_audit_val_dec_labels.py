"""PROBE (uncommitted): audit the v2 val labels with an INDEPENDENT teacher.

The v2 val labels (and the dec recall 0.33 ceiling) come from DeepSeek-flash. If
flash mislabeled real snapshots/plans as decision_update, the head is being
scored against bad labels and dec recall is artificially capped -- NOT an arch
ceiling. This probe re-labels the 76 val docs with a DIFFERENT model family
(default glm-5.2:cloud, configurable) using the SAME v2 labeler prompt, then
reports per-class disagreement + a focus on the 24 decision_update val docs.

GPU-free (cloud Oracle only). If teacher says many of the 24 dec val docs are
non-dec -> label noise -> clean the val, re-score, maybe ship v2. If teacher
agrees with flash on nearly all 24 -> the dec docs really are decisions the
head can't separate -> arch ceiling -> attention-over-sections is the lever.

Not committed (scripts/_scratch/). Oracle must be up at localhost:11434/v1.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Reuse the v2 labeler prompt + label set (same prompt, different model = the
# isolated variable). Loaded via importlib (scripts/ is not a package).
_labeler_path = Path(__file__).resolve().parent.parent / "label_doc_kind_corpus.py"
_spec = importlib.util.spec_from_file_location("_label_doc_kind_corpus", _labeler_path)
_labeler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_labeler)
_LABELS = _labeler._LABELS
_LABEL_PROMPT = _labeler._PROMPT

VAL = "data/training/doc_kind_head/pairs_v3_val.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit v2 val labels with an independent teacher.")
    ap.add_argument("--val", default=VAL, help="val JSONL (has v2-flash labels)")
    ap.add_argument("--teacher", default="glm-5.2:cloud",
                    help="independent teacher model (default glm-5.2:cloud, "
                         "different family from DeepSeek-flash)")
    ap.add_argument("--cache", default="data/training/doc_kind_head/oracle_cache_audit.json",
                    help="prompt-hash cache (resume)")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--report", default="data/training/doc_kind_head/val_audit_report.jsonl",
                    help="per-doc report JSONL")
    args = ap.parse_args()

    from src.config import config as _config
    from src.ingestion.doc_kind import join_section_texts
    from src.training.oracle_labeling import OracleClient, OracleConfig

    val = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]
    print(f"val: {len(val)} docs (label dist "
          f"{dict(sorted({l: sum(1 for r in val if r['label']==l) for l in _LABELS}.items()))})",
          flush=True)

    # Independent teacher: same prompt, different family, near-deterministic.
    cfg = OracleConfig(
        model=args.teacher,
        endpoint=_config.oracle_endpoint,
        temperature=0.1,
        max_tokens=512,
        batch_delay=0.0,
        cache_path=Path(args.cache),
        think=None,            # default OpenAI /v1 + json_object path (NOT qwen3)
    )
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    client = OracleClient(cfg)
    print(f"re-labeling {len(val)} val docs with independent teacher "
          f"({args.teacher}, same v2 prompt)...", flush=True)

    prompts = [_LABEL_PROMPT + join_section_texts(r["section_texts"]) for r in val]
    results = client.generate_batch(prompts, response_format="json_object",
                                    max_workers=args.max_workers)

    # Per-doc report + disagreement tally.
    rows = []
    disagree_per_class: dict[str, int] = {l: 0 for l in _LABELS}
    total_per_class: dict[str, int] = {l: 0 for l in _LABELS}
    teacher_label_counts: dict[str, int] = {l: 0 for l in _LABELS}
    n_err = 0
    for rec, res in zip(val, results):
        flash_label = rec["label"]
        total_per_class[flash_label] += 1
        if res.error or not isinstance(res.response, dict):
            n_err += 1
            teacher_label, tconf = None, None
        else:
            teacher_label = res.response.get("doc_kind")
            tconf = res.response.get("confidence")
            if teacher_label not in _LABELS:
                teacher_label = None   # OOV -> treat as no-opinion (don't count as agreement)
            else:
                teacher_label_counts[teacher_label] += 1
        agree = (teacher_label == flash_label)
        if not agree:
            disagree_per_class[flash_label] += 1
        text = join_section_texts(rec["section_texts"])
        rows.append({
            "doc_id": rec.get("doc_id", "?"),
            "flash_label": flash_label,
            "flash_conf": rec.get("confidence"),
            "teacher_label": teacher_label,
            "teacher_conf": tconf,
            "agree": agree,
            "text_head": text[:280],
        })

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== TEACHER ({args.teacher}) vs FLASH (v2) on {len(val)} val docs ===",
          flush=True)
    print(f"errors/OOV: {n_err}", flush=True)
    print(f"teacher label dist: {dict(sorted(teacher_label_counts.items()))}", flush=True)
    print(f"\nper-class disagreement (flash -> teacher differs):", flush=True)
    for l in _LABELS:
        if total_per_class[l]:
            d = disagree_per_class[l]
            t = total_per_class[l]
            print(f"  {l:<24} {d}/{t} ({d/t*100:.0f}%) relabeled by teacher", flush=True)

    # Focus: the 24 decision_update val docs -- how many does the teacher call NON-dec?
    dec_rows = [r for r in rows if r["flash_label"] == "decision_update"]
    dec_relabel = [r for r in dec_rows if not r["agree"]]
    print(f"\n=== FOCUS: decision_update val docs ({len(dec_rows)} total) ===", flush=True)
    print(f"teacher calls NON-decision_update on {len(dec_relabel)}/{len(dec_rows)} "
          f"of the docs flash labeled decision_update:", flush=True)
    for r in dec_relabel:
        print(f"  [{r['doc_id']}] flash=decision_update(conf {r['flash_conf']}) "
              f"-> teacher={r['teacher_label']}(conf {r['teacher_conf']})", flush=True)
        print(f"    text: {r['text_head'][:160]}...", flush=True)

    # Interpretation hint.
    dec_disagree_rate = (len(dec_relabel) / len(dec_rows)) if dec_rows else 0
    print(f"\n=== INTERPRETATION HINT ===", flush=True)
    if dec_disagree_rate >= 0.25:
        print(f"dec disagreement {dec_disagree_rate:.0%} is HIGH -- likely label noise in "
              f"the val dec set. The head's dec 0.33 is partly a noise floor. Next: "
              f"clean the val dec labels (relabel with a stronger panel) and re-score the "
              f"head against clean labels -- may ship v2 without new training data.",
              flush=True)
    else:
        print(f"dec disagreement {dec_disagree_rate:.0%} is LOW -- the val dec labels look "
              f"clean. The 16/24 the head misses really are decision_updates it can't "
              f"separate from snap/plan. Arch ceiling CONFIRMED -> attention-over-sections "
              f"is the lever, NOT more data.", flush=True)

    stats = client.get_stats()
    print(f"\noracle: {stats['total_calls']} calls, {stats['cached_calls']} cached, "
          f"{stats['total_tokens']} tokens", flush=True)
    print(f"report: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())