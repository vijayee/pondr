"""Run the 1f-7 Stage 1 live gate for all per-kind bilinear seeds + collect
doc-kind medians. Mirrors _run_1f6_gate.py but points at the Stage 1 per-kind
checkpoints (n_doc_kinds inferred by load_composite_z_head from the
kind_heads.{k}.weight keys, OR per_kind_full kind_readouts, OR n_doc_kinds=0
shared readout; the live probe builds z_slot_doc_kinds from --doc-store).
PASS = retrieved_code >= 2.0 in >= 2/3 seeds AND retrieved_text >= 2.0
(no regression on text) -- the Stage 1 plan S1.10 criterion, generalized to
N seeds (need = ceil(2*N/3) passes).

Seed list is configurable via --seeds (comma-separated, default 0,1,2).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "data/training/strm_state_readout/head_to_head_onyx"
BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full.pt"
DOC_STORE = "data/training/strm_relevance/doc_corpus_store_summarized"
OUT_DIR = ROOT / "data/training/strm_relevance"
ZLOGIT_GATE = 2.0
DEFAULT_SEEDS = (0, 1, 2)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _extract(report: dict) -> dict:
    zst = report.get("selectivity", {}).get("z_logit_by_slot_type", {})
    out = {}
    for bucket in ("full", "conv", "retrieved", "retrieved_text", "retrieved_code"):
        b = zst.get(bucket, {})
        out[bucket] = b.get("median")
        out[f"{bucket}_gate_pass"] = b.get("gate_median_ge_thr")
    return out


def _fmt(v):
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "  n/a "


def _parse_seeds(argv: list[str]) -> tuple[int, ...]:
    for i, a in enumerate(argv):
        if a == "--seeds" and i + 1 < len(argv):
            return tuple(int(x) for x in argv[i + 1].split(",") if x.strip() != "")
    return DEFAULT_SEEDS


def _parse_backbone(argv: list[str]) -> str:
    """--backbone PATH overrides the frozen backbone the live probe re-encodes
    serve data with. Default = BACKBONE (byte-identical when omitted). Stage 3
    fine-tune passes the fine-tuned backbone here."""
    for i, a in enumerate(argv):
        if a == "--backbone" and i + 1 < len(argv):
            return argv[i + 1]
    return BACKBONE


def _parse_opt(argv: list[str], flag: str, default: str) -> str:
    """Generic --flag VALUE override with a byte-identical default. Used for
    --ckpt-root (which bilinear_s{s}/final.pt to gate on) and --summary (the
    aggregate summary json path). Defaults preserve the Stage 3 behavior."""
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return default


def main():
    seeds = _parse_seeds(sys.argv[1:])
    backbone = _parse_backbone(sys.argv[1:])
    ckpt_root_str = _parse_opt(sys.argv[1:], "--ckpt-root", str(CKPT_ROOT))
    summary_str = _parse_opt(sys.argv[1:], "--summary",
                             str(OUT_DIR / "serve_gate_1f7_stage1_summary.json"))
    ckpt_root = Path(ckpt_root_str)
    summary_path = Path(summary_str)
    n_seeds = len(seeds)
    need = math.ceil(2 * n_seeds / 3)  # >=2/3 majority threshold
    print(f"[gate] backbone={backbone} | seeds={seeds} | need={need}/{n_seeds} | "
          f"ckpt_root={ckpt_root}", flush=True)
    rows = []
    for s in seeds:
        ckpt = ckpt_root / f"bilinear_s{s}" / "final.pt"
        if not ckpt.exists():
            print(f"[gate] MISSING {ckpt} -- skip seed {s}", flush=True)
            continue
        out_json = OUT_DIR / f"serve_gate_1f7_stage1_s{s}.json"
        cmd = [
            sys.executable, "scripts/probe_strm_selectivity_real.py",
            "--z-head-arch", "composite-raw",
            "--z-relevance-head", str(ckpt),
            "--backbone", backbone,
            "--strm-ring-text", "--identity-instance",
            "--doc-store", DOC_STORE,
            "--device", "cuda",
            "--out", str(out_json),
        ]
        print(f"[gate] === seed {s} ({ckpt.name}) ===", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        (OUT_DIR / f"serve_gate_1f7_stage1_s{s}.log").write_text(r.stdout)
        print(r.stdout[-1500:], flush=True)
        if r.returncode != 0:
            print(f"[gate] seed {s} FAILED rc={r.returncode}", flush=True)
            continue
        report = json.loads(out_json.read_text())
        ext = _extract(report)
        ext["seed"] = s
        rows.append(ext)
        print(f"[gate] seed {s} medians: {ext}", flush=True)

    if not rows:
        print("[gate] no seeds completed", flush=True)
        return 1

    n_eval = len(rows)
    print(f"\n=== 1f-7 Stage 1 LIVE GATE SUMMARY (per-kind bilinear, "
          f"{n_eval}/{n_seeds} seeds evaluated) ===", flush=True)
    print(f"{'seed':>4} | {'full':>8} | {'conv':>8} | {'retrieved':>9} | "
          f"{'ret_text':>9} | {'ret_code':>9}", flush=True)
    for r in rows:
        print(f"{r['seed']:>4} | "
              f"{_fmt(r['full']):>8} | {_fmt(r['conv']):>8} | "
              f"{_fmt(r['retrieved']):>9} | {_fmt(r['retrieved_text']):>9} | "
              f"{_fmt(r['retrieved_code']):>9}", flush=True)

    def _med(bucket):
        vals = [r[bucket] for r in rows if r[bucket] is not None]
        return _median(vals) if vals else None

    verdict = {}
    for bucket in ("retrieved_text", "retrieved_code", "retrieved", "full", "conv"):
        m = _med(bucket)
        passes = sum(1 for r in rows if r[bucket] is not None and r[bucket] >= ZLOGIT_GATE)
        # passes_2_of_3 kept as the key name for back-compat with readers of the
        # summary json; semantically now "passes the >=2/3-of-N majority criterion".
        verdict[bucket] = {"median": m, "passes_2_of_3": passes >= need,
                           "n_pass": passes, "n_seeds": n_eval, "need": need}
    print(f"\n=== {n_eval}-seed medians + pass verdict "
          f"(gate >= {ZLOGIT_GATE}, need >={need}/{n_eval}) ===", flush=True)
    for bucket, v in verdict.items():
        print(f"  {bucket:>16}: median={_fmt(v['median'])}  "
              f"pass={v['n_pass']}/{n_eval}  -> {'PASS' if v['passes_2_of_3'] else 'FAIL'}",
              flush=True)

    pass_crit = (verdict["retrieved_text"]["passes_2_of_3"]
                 and verdict["retrieved_code"]["passes_2_of_3"])
    print(f"\n=== Stage 1 PASS (ret_code>=2.0 AND ret_text>=2.0, each in >={need}/{n_eval}): "
          f"{'PASS -> ship' if pass_crit else 'FAIL -> Stage 2'} ===", flush=True)

    summary_path.write_text(
        json.dumps({"rows": rows, "verdict": verdict,
                    "pass_criterion": pass_crit, "gate": ZLOGIT_GATE,
                    "n_seeds": n_eval, "need": need, "backbone": backbone,
                    "ckpt_root": str(ckpt_root)},
                   indent=2))
    print(f"\nwrote {summary_path}", flush=True)
    return 0 if pass_crit else 2


if __name__ == "__main__":
    raise SystemExit(main())