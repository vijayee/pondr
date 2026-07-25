"""Matched 1f-5b baseline gate: OLD store + OLD ckpts + force_rule_based=True.

Isolates whether the 1f-6 text-doc lift came from the prose summaries or from
the planner-path change (endpoint=None server-first -> force_rule_based). Runs
the SAME gate as _run_1f6_gate.py but against the 1f-5b artifacts so the ONLY
difference vs 1f-6 is the store content (prose-summary embeddings vs raw code).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "data/training/strm_state_readout/phase1f_margin_doc_ring"
BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full.pt"
DOC_STORE = "data/training/strm_relevance/doc_corpus_store"  # OLD 1f-5b store
OUT_DIR = ROOT / "data/training/strm_relevance"
ZLOGIT_GATE = 2.0


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _extract(report):
    zst = report.get("selectivity", {}).get("z_logit_by_slot_type", {})
    out = {}
    for b in ("full", "conv", "retrieved", "retrieved_text", "retrieved_code"):
        out[b] = zst.get(b, {}).get("median")
    return out


def _fmt(v):
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "  n/a "


def main():
    rows = []
    for s in (0, 1, 2):
        ckpt = CKPT_ROOT / f"bilinear_s{s}" / "final.pt"
        if not ckpt.exists():
            print(f"[1f5b] MISSING {ckpt}", flush=True)
            continue
        out_json = OUT_DIR / f"serve_gate_1f5b_matched_s{s}.json"
        cmd = [
            sys.executable, "scripts/probe_strm_selectivity_real.py",
            "--z-head-arch", "composite-raw",
            "--z-relevance-head", str(ckpt),
            "--backbone", BACKBONE,
            "--strm-ring-text", "--identity-instance",
            "--doc-store", DOC_STORE,
            "--device", "cuda",
            "--out", str(out_json),
        ]
        print(f"[1f5b] === seed {s} ===", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        (OUT_DIR / f"serve_gate_1f5b_matched_s{s}.log").write_text(r.stdout)
        if r.returncode != 0:
            print(f"[1f5b] seed {s} FAILED rc={r.returncode}", flush=True)
            continue
        report = json.loads(out_json.read_text())
        ext = _extract(report)
        ext["seed"] = s
        rows.append(ext)
        print(f"[1f5b] seed {s}: {ext}", flush=True)

    print("\n=== 1f-5b MATCHED BASELINE (old store + old ckpts + force_rule_based) ===", flush=True)
    print(f"{'seed':>4} | {'full':>8} | {'conv':>8} | {'retrieved':>9} | "
          f"{'ret_text':>9} | {'ret_code':>9}", flush=True)
    for r in rows:
        print(f"{r['seed']:>4} | {_fmt(r['full']):>8} | {_fmt(r['conv']):>8} | "
              f"{_fmt(r['retrieved']):>9} | {_fmt(r['retrieved_text']):>9} | "
              f"{_fmt(r['retrieved_code']):>9}", flush=True)
    for b in ("retrieved_text", "retrieved_code", "retrieved", "full", "conv"):
        vals = [r[b] for r in rows if r[b] is not None]
        m = _median(vals) if vals else None
        p = sum(1 for v in vals if v is not None and v >= ZLOGIT_GATE)
        print(f"  {b:>16}: median={_fmt(m)}  pass={p}/3", flush=True)
    (OUT_DIR / "serve_gate_1f5b_matched_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()