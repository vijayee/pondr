"""Multi-seed gate runner for the STRM Phase 4 ship-deciding eval
(``scripts/eval_strm_ship_decision.py``). UNTRACKED scratch runner (matches the
existing untracked ``_run_1f*_gate.py`` siblings per commit-at-will).

SHIP CRITERION (the literal plan gate, ``docs/STRM-implementation-plan.md:550``):
  STRM+recall answers more factual questions correctly than fixed-interval
  refresh at equal recall budget, in >= 2/3 seeds.

  pass_crit = accuracy_passes_2_of_3 AND coverage_passes_2_of_3
    - accuracy: tier2.strm_beats_fixed_accuracy (the literal gate)
    - coverage: tier1.strm_beats_fixed (the mechanism precondition -- if STRM
      does not surface the right fact more often than fixed at equal budget,
      the accuracy win is not attributable to the salience decision)
  need = ceil(2 * n_seeds / 3)  (=2 for 3 seeds). Exit 0 = SHIP, 2 = HOLD.

Each seed is a distinct conversation ordering (the eval shuffles fact/filler
order by seed -- see its header); the >=2/3 bar tests robustness across
orderings, not a single lucky draw.

Run (needs Bonsai 8B on :8080 + DeepSeek-flash on :11434 for Tier 2):
  PYTHONPATH=. python scripts/_run_strm_ship_gate.py --seeds 0,1,2
Tier 1 only (offline, fast):
  PYTHONPATH=. python scripts/_run_strm_ship_gate.py --seeds 0,1,2 --tier1-only
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/training/strm_relevance"
DEFAULT_SEEDS = (0, 1, 2)


def _parse_seeds(argv: list[str]) -> tuple[int, ...]:
    for i, a in enumerate(argv):
        if a == "--seeds" and i + 1 < len(argv):
            return tuple(int(x) for x in argv[i + 1].split(",") if x.strip() != "")
    return DEFAULT_SEEDS


def _parse_opt(argv: list[str], flag: str, default: str) -> str:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return default


def _parse_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def _extract(report: dict) -> dict:
    t1 = report.get("tier1_coverage", {}) or {}
    t2 = report.get("tier2_accuracy", {}) or {}
    s = t2.get("strm", {}) or {}
    f = t2.get("fixed", {}) or {}
    o = t2.get("off", {}) or {}
    return {
        "seed": report.get("seed"),
        "seed_pass": bool(report.get("seed_pass")),
        "coverage_pass": bool(report.get("coverage_pass")),
        "skipped_accuracy": bool(report.get("skipped_accuracy")),
        "cov_strm_beats_fixed": bool(t1.get("strm_beats_fixed")),
        "cov_budget_parity": bool(t1.get("budget_parity")),
        "acc_strm_beats_fixed": bool(t2.get("strm_beats_fixed_accuracy")),
        "acc_budget_parity": bool(t2.get("budget_parity")),
        "strm_acc": s.get("acc"),
        "fixed_acc": f.get("acc"),
        "off_acc": o.get("acc"),
        "strm_ci": s.get("ci95"),
        "fixed_ci": f.get("ci95"),
    }


def _fmt(v) -> str:
    if v is None:
        return "  n/a "
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> int:
    seeds = _parse_seeds(sys.argv[1:])
    out_dir = Path(_parse_opt(sys.argv[1:], "--out-dir", str(OUT_DIR)))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(_parse_opt(sys.argv[1:], "--summary",
                                   str(out_dir / "ship_gate_summary.json")))
    tier1_only = _parse_flag(sys.argv[1:], "--tier1-only")
    no_skip = _parse_flag(sys.argv[1:], "--no-skip-accuracy-if-coverage-fails")
    facts = _parse_opt(sys.argv[1:], "--facts", "6")
    horizon = _parse_opt(sys.argv[1:], "--horizon", "40")
    ring_cap = _parse_opt(sys.argv[1:], "--ring-cap", "0")
    theta = _parse_opt(sys.argv[1:], "--theta", "-0.04")
    salience_mode = _parse_opt(sys.argv[1:], "--salience-mode", "learned")
    cos_phi = _parse_opt(sys.argv[1:], "--cos-phi", "0.6")
    age_threshold = _parse_opt(sys.argv[1:], "--age-threshold", "3")
    n_seeds = len(seeds)
    need = math.ceil(2 * n_seeds / 3)
    print(f"[gate] seeds={seeds} | need={need}/{n_seeds} | tier1_only={tier1_only} | "
          f"facts={facts} horizon={horizon} ring-cap={ring_cap} theta={theta} | "
          f"salience-mode={salience_mode} cos-phi={cos_phi} age-threshold={age_threshold}",
          flush=True)
    rows = []
    for s in seeds:
        out_json = out_dir / f"ship_s{s}.json"
        cmd = [
            sys.executable, "scripts/eval_strm_ship_decision.py",
            "--facts", facts, "--horizon", horizon, "--ring-cap", ring_cap,
            "--theta", theta, "--seed", str(s), "--out", str(out_json),
            "--salience-mode", salience_mode, "--cos-phi", cos_phi,
            "--age-threshold", age_threshold,
        ]
        if tier1_only:
            cmd.append("--tier1-only")
        if no_skip:
            cmd.append("--no-skip-accuracy-if-coverage-fails")
        print(f"[gate] === seed {s} ===", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        (out_dir / f"ship_s{s}.log").write_text(r.stdout)
        print(r.stdout[-1800:], flush=True)
        if r.returncode != 0 or not out_json.exists():
            print(f"[gate] seed {s} FAILED rc={r.returncode}", flush=True)
            continue
        report = json.loads(out_json.read_text(encoding="utf-8"))
        ext = _extract(report)
        rows.append(ext)
        print(f"[gate] seed {s}: {ext}", flush=True)

    if not rows:
        print("[gate] no seeds completed", flush=True)
        return 1

    n_eval = len(rows)
    print(f"\n=== STRM SHIP GATE SUMMARY ({n_eval}/{n_seeds} seeds evaluated) ===",
          flush=True)
    print(f"{'seed':>4} | {'cov':>5} | {'acc':>5} | {'strm':>6} | {'fixed':>6} | "
          f"{'off':>6} | {'seed_pass':>9}", flush=True)
    for r in rows:
        print(f"{r['seed']:>4} | "
              f"{'PASS' if r['cov_strm_beats_fixed'] else 'fail':>5} | "
              f"{'PASS' if r['acc_strm_beats_fixed'] else 'fail':>5} | "
              f"{_fmt(r['strm_acc']):>6} | {_fmt(r['fixed_acc']):>6} | "
              f"{_fmt(r['off_acc']):>6} | {str(r['seed_pass']):>9}", flush=True)

    def _passes(key):
        return sum(1 for r in rows if r[key])

    cov_passes = _passes("cov_strm_beats_fixed")
    acc_passes = _passes("acc_strm_beats_fixed")
    verdict = {
        "coverage": {"n_pass": cov_passes, "n_seeds": n_eval, "need": need,
                     "passes_2_of_3": cov_passes >= need},
        "accuracy": {"n_pass": acc_passes, "n_seeds": n_eval, "need": need,
                     "passes_2_of_3": acc_passes >= need,
                     "n_skipped": sum(1 for r in rows if r["skipped_accuracy"])},
    }
    print(f"\n=== verdict (need >= {need}/{n_eval} on each) ===", flush=True)
    print(f"  coverage (mechanism precondition): {cov_passes}/{n_eval} -> "
          f"{'PASS' if verdict['coverage']['passes_2_of_3'] else 'FAIL'}", flush=True)
    print(f"  accuracy (literal plan gate):      {acc_passes}/{n_eval} -> "
          f"{'PASS' if verdict['accuracy']['passes_2_of_3'] else 'FAIL'}", flush=True)

    pass_crit = (verdict["coverage"]["passes_2_of_3"]
                 and verdict["accuracy"]["passes_2_of_3"])
    print(f"\n=== SHIP DECISION (accuracy >=2/3 AND coverage >=2/3): "
          f"{'SHIP STRM default-on' if pass_crit else 'HOLD -- stay opt-in'} ===",
          flush=True)

    summary_path.write_text(json.dumps(
        {"rows": rows, "verdict": verdict, "pass_criterion": pass_crit,
         "n_seeds": n_eval, "need": need, "tier1_only": tier1_only,
         "facts": facts, "horizon": horizon, "ring_cap": ring_cap, "theta": theta,
         "salience_mode": salience_mode, "cos_phi": cos_phi,
         "age_threshold": age_threshold},
        indent=2))
    print(f"\nwrote {summary_path}", flush=True)
    return 0 if pass_crit else 2


if __name__ == "__main__":
    raise SystemExit(main())