"""Run the 1f-6 live gate for all 3 bilinear seeds + collect doc-kind medians.

Invokes probe_strm_selectivity_real.py per seed against the summarized doc
store, then reads each --out JSON and prints + writes a combined summary:
per-seed full/conv/retrieved/retrieved_text/retrieved_code z_logit medians +
the 3-seed pass verdict against the 2.0 gate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "data/training/strm_state_readout/phase1f6_margin_doc_ring"
BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full.pt"
DOC_STORE = "data/training/strm_relevance/doc_corpus_store_summarized"
OUT_DIR = ROOT / "data/training/strm_relevance"
ZLOGIT_GATE = 2.0


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _extract(report: dict) -> dict:
    """Pull the per-bucket z_logit medians from a gate report JSON.

    Shape (probe_strm_selectivity_real.py:1310):
        report["selectivity"]["z_logit_by_slot_type"][<bucket>]["median"]
        + ["gate_median_ge_thr"]  (per-seed median >= 2.0)
    """
    zst = report.get("selectivity", {}).get("z_logit_by_slot_type", {})
    out = {}
    for bucket in ("full", "conv", "retrieved", "retrieved_text", "retrieved_code"):
        b = zst.get(bucket, {})
        out[bucket] = b.get("median")
        out[f"{bucket}_gate_pass"] = b.get("gate_median_ge_thr")
    return out


def main():
    rows = []
    for s in (0, 1, 2):
        ckpt = CKPT_ROOT / f"bilinear_s{s}" / "final.pt"
        if not ckpt.exists():
            print(f"[gate] MISSING {ckpt} -- skip seed {s}", flush=True)
            continue
        out_json = OUT_DIR / f"serve_gate_1f6_s{s}.json"
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
        print(f"[gate] === seed {s} ===", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        (OUT_DIR / f"serve_gate_1f6_s{s}.log").write_text(r.stdout)
        print(r.stdout[-1500:], flush=True)
        if r.returncode != 0:
            print(f"[gate] seed {s} FAILED rc={r.returncode}", flush=True)
            continue
        report = json.loads(out_json.read_text())
        ext = _extract(report)
        ext["seed"] = s
        rows.append(ext)
        print(f"[gate] seed {s} medians: {ext}", flush=True)

    print("\n=== 1f-6 LIVE GATE SUMMARY (bilinear, summarized store) ===", flush=True)
    print(f"{'seed':>4} | {'full':>8} | {'conv':>8} | {'retrieved':>9} | "
          f"{'ret_text':>9} | {'ret_code':>9}", flush=True)
    for r in rows:
        print(f"{r['seed']:>4} | "
              f"{_fmt(r['full']):>8} | {_fmt(r['conv']):>8} | "
              f"{_fmt(r['retrieved']):>9} | {_fmt(r['retrieved_text']):>9} | "
              f"{_fmt(r['retrieved_code']):>9}", flush=True)

    def _med3(bucket):
        vals = [r[bucket] for r in rows if r[bucket] is not None]
        return _median(vals) if vals else None

    verdict = {}
    for bucket in ("retrieved_text", "retrieved_code", "retrieved", "full", "conv"):
        m = _med3(bucket)
        passes = sum(1 for r in rows if r[bucket] is not None and r[bucket] >= ZLOGIT_GATE)
        verdict[bucket] = {"median_3seed": m, "passes_2_of_3": passes >= 2, "n_pass": passes}
    print("\n=== 3-seed medians + pass verdict (gate >= 2.0, need >=2/3) ===", flush=True)
    for bucket, v in verdict.items():
        print(f"  {bucket:>16}: median={_fmt(v['median_3seed'])}  "
              f"pass={v['n_pass']}/3  -> {'PASS' if v['passes_2_of_3'] else 'FAIL'}",
              flush=True)

    pass_crit = (verdict["retrieved_text"]["passes_2_of_3"]
                 and verdict["retrieved_code"]["passes_2_of_3"]
                 and verdict["retrieved"]["passes_2_of_3"])
    print(f"\n=== 1f-6 PASS CRITERION (ret_text>=2.0 AND ret_code>=2.0 AND "
          f"retrieved>=2.0, each in >=2/3): {'PASS' if pass_crit else 'FAIL'} ===",
          flush=True)

    (OUT_DIR / "serve_gate_1f6_summary.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict,
                    "pass_criterion": pass_crit, "gate": ZLOGIT_GATE}, indent=2))
    print(f"\nwrote {OUT_DIR / 'serve_gate_1f6_summary.json'}", flush=True)


def _fmt(v):
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "  n/a "


if __name__ == "__main__":
    main()