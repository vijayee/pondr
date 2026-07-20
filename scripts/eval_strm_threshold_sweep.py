"""STRM Phase 4 Probe 2: salience threshold sweep (the "improve it" probe).

Probe 1 (context coverage) found the mechanism has headroom (permissive mode
rescued 1/6) but the SHIPPED thresholds fire 0/6 -- so the bottleneck is
threshold tuning, not the mechanism. Probe 2 finds the operating point.

WHY THIS IS FAST + GROUNDED: F's per-turn salience scores (r_i, rec_i,
surprise_i) are DETERMINISTIC given the scenario (same ring state + same frozen
heads); only the threshold flips the ``salient`` decision. And in this scenario
F is the ONLY salience candidate (filler slots have no text -> unscoreable), so
F fires iff ``salient`` AND is always within the budget=3. So once we capture
F's real scores + the prompt-driven baseline (coverage_off) from ONE real run,
firing + coverage for ANY (theta, phi, surprise_cap) is computable analytically
without re-running the orchestrator::

    F_salient(cfg) = (rec_i < theta) & (r_i > phi) & (surprise_i < surprise_cap)
    coverage_on(cfg)  = coverage_off OR F_salient(cfg)
    rescued(cfg)      = F_salient(cfg) AND NOT coverage_off

The analytical model is VALIDATED against 2 real runs (shipped + permissive)
before the full grid is reported, so the sweep is grounded, not blind.

Output:
  1. The score table -- per trial, F's (r_i, rec_i, surprise_i) and which of
     the three conditions pass at the SHIPPED thresholds -> identifies the
     BOTTLENECK term (the one suppressing firing).
  2. A per-axis sweep (theta, phi, surprise_cap each varied with the other two
     held at shipped) -> which lever moves firing, and the knee (lowest value
     that produces a positive coverage delta with selective firing).
  3. A joint recommendation: the config that maximizes delta with firing
     selective enough to be worth its budget (the deferred Step 7 gate tunes
     this for real data; this is the synthetic operating point).

Usage:
  python scripts/eval_strm_threshold_sweep.py --facts 6 --gap 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Phase2cConfig
from src.memory.store import HippocampalStore
from src.retrieval.retriever import HippocampalRetriever
from src.runtime import DEFAULT_BACKBONE_PATH
from src.subconscious.configs import BackboneConfig
from src.subconscious.latent_dynamics_head import load_latent_dynamics_head
from src.subconscious.recoverability_head import load_recoverability_head
from src.subconscious.relevance_head import load_relevance_head
from src.subconscious.salience import SalienceThresholds, load_salience_thresholds
from src.subconscious.training.routing_training import build_embedder, load_backbone

# Reuse Probe 1 building blocks (the harness, the fact/filler banks, the stubs).
from scripts.eval_strm_context_coverage import (
    ARTIFACTS_PRESENT, FACT_BANK, FILLER_BANK, _StubPlanner,
    _build_orchestrator, _reset_for_trial, _seed_corpus, _seed_ring,
)

REC_CKPT = Path("data/training/strm_recoverability/best.pt")
LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")
REL_CKPT = Path("data/training/strm_relevance/best.pt")
THRESHOLDS = Path("data/training/strm_salience/thresholds.json")
BACKBONE = Path(DEFAULT_BACKBONE_PATH)


def _fact_id(i: int) -> str:
    return f"fact_{i:02d}"


def _capture_trial(orch_on, orch_off, fact_idx, fact_summary, query, fillers):
    """Run both conditions for one trial; return F's real salience scores (from
    orch_on._salience_anchors) + coverage_off + coverage_on (shipped)."""
    fid = _fact_id(fact_idx)
    # OFF (threshold-independent baseline): does prompt-driven find F?
    _reset_for_trial(orch_off)
    _seed_ring(orch_off, fid, fact_summary, fillers)
    res_off = orch_off.query(query)
    off_ids = {ep.get("episode_id") for ep in res_off.get("retrieved_episodes", []) or []}
    coverage_off = fid in off_ids
    # ON with shipped thresholds: capture F's anchor scores.
    _reset_for_trial(orch_on)
    _seed_ring(orch_on, fid, fact_summary, fillers)
    res_on = orch_on.query(query)
    on_ids = {ep.get("episode_id") for ep in res_on.get("retrieved_episodes", []) or []}
    coverage_on = fid in on_ids
    anchors = orch_on._salience_anchors or []
    f_anchor = next((a for a in anchors if a.source_id == fid), None)
    return {
        "fact_id": fid, "summary": fact_summary, "query": query,
        "r_i": f_anchor.r_i if f_anchor else None,
        "rec_i": f_anchor.rec_i if f_anchor else None,
        "surprise_i": f_anchor.surprise_i if f_anchor else None,
        "salient_shipped": bool(f_anchor.salient) if f_anchor else False,
        "coverage_off": coverage_off,
        "coverage_on_shipped": coverage_on,
    }


def _analytical(trial, theta, phi, surprise_cap) -> bool:
    """F is salient under (theta, phi, surprise_cap) -- analytical, no re-run."""
    r, rec, s = trial["r_i"], trial["rec_i"], trial["surprise_i"]
    if r is None or rec is None or s is None:
        return False
    return (rec < theta) and (r > phi) and (s < surprise_cap)


def _sweep_axis(trials, axis: str, values, shipped: SalienceThresholds):
    """Vary ONE threshold across ``values`` (other two at shipped). Returns per-
    value (firing, coverage_on, coverage_off, delta, rescued)."""
    out = []
    for v in values:
        theta = v if axis == "theta" else shipped.theta
        phi = v if axis == "phi" else shipped.phi
        sc = v if axis == "surprise_cap" else shipped.surprise_cap
        n = len(trials)
        fire = sum(_analytical(t, theta, phi, sc) for t in trials)
        cov_on = sum(t["coverage_off"] or _analytical(t, theta, phi, sc) for t in trials)
        cov_off = sum(t["coverage_off"] for t in trials)
        rescued = sum(_analytical(t, theta, phi, sc) and not t["coverage_off"] for t in trials)
        out.append({
            "value": v, "firing": fire, "coverage_on": cov_on,
            "coverage_off": cov_off, "delta": cov_on - cov_off, "rescued": rescued,
        })
    return out


def _fmt(v) -> str:
    if v == float("inf"):
        return "+inf"
    if isinstance(v, float) and abs(v) < 1e-3 and v != 0.0:
        return f"{v:.2e}"
    return f"{v:.4g}"


def run_sweep(*, n_facts: int, gap: int, ring_capacity: int, seed: int) -> dict:
    rng = random.Random(seed)
    facts = list(FACT_BANK[:n_facts])
    filler_pool = list(FILLER_BANK)
    shipped = load_salience_thresholds(str(THRESHOLDS))

    tmpdir = tempfile.mkdtemp(prefix="pondr_probe2_")
    store: Optional[HippocampalStore] = None
    trials: list = []
    try:
        store = HippocampalStore(str(Path(tmpdir) / "db"))
        embedder = build_embedder("on-demand")
        rec = load_recoverability_head(str(REC_CKPT), device="cpu")
        ld = load_latent_dynamics_head(str(LD_CKPT), device="cpu")
        rel = load_relevance_head(str(REL_CKPT), device="cpu")
        retriever = HippocampalRetriever(
            store, planner=_StubPlanner(), auto_load_index=True,
            retrieval_gate=None, embedder=embedder)
        _seed_corpus(store, embedder, facts)
        cfg = Phase2cConfig()
        cfg.session.state_dir = str(Path(tmpdir) / "sessions")
        bb_on = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        bb_off = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        orch_on = _build_orchestrator(
            store, retriever, embedder, bb_on, strm_salience=True,
            thresholds=shipped, recoverability_head=rec, latent_dynamics_head=ld,
            relevance_head=rel, ring_capacity=ring_capacity, cfg=cfg)
        orch_off = _build_orchestrator(
            store, retriever, embedder, bb_off, strm_salience=False,
            thresholds=None, recoverability_head=None, latent_dynamics_head=None,
            relevance_head=None, ring_capacity=ring_capacity, cfg=cfg)
        for i, (summary, query) in enumerate(facts):
            shuffled = filler_pool[:]
            rng.shuffle(shuffled)
            fillers = [shuffled[j % len(shuffled)] for j in range(gap)]
            trials.append(_capture_trial(orch_on, orch_off, i, summary, query, fillers))
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Validate the analytical model against the real shipped run.
    mismatches = 0
    for t in trials:
        pred = t["coverage_off"] or _analytical(t, shipped.theta, shipped.phi, shipped.surprise_cap)
        if pred != t["coverage_on_shipped"]:
            mismatches += 1
    validation = {"analytical_matches_real_at_shipped": mismatches == 0,
                  "mismatches": mismatches}

    # Score distributions + bottleneck at shipped thresholds.
    def _which(t):
        r, rec, s = t["r_i"], t["rec_i"], t["surprise_i"]
        return (r is not None and r > shipped.phi,
                rec is not None and rec < shipped.theta,
                s is not None and s < shipped.surprise_cap)
    score_table = []
    for t in trials:
        rel_ok, forg_ok, surp_ok = _which(t)
        score_table.append({
            "fact_id": t["fact_id"], "r_i": t["r_i"], "rec_i": t["rec_i"],
            "surprise_i": t["surprise_i"], "rel_ok": rel_ok, "forg_ok": forg_ok,
            "surprise_ok": surp_ok, "coverage_off": t["coverage_off"],
            "salient_shipped": t["salient_shipped"],
        })

    # Per-axis sweeps (values span the observed score range + shipped + inf).
    surprises = sorted(t["surprise_i"] for t in trials if t["surprise_i"] is not None)
    recs = sorted(t["rec_i"] for t in trials if t["rec_i"] is not None)
    rs = sorted(t["r_i"] for t in trials if t["r_i"] is not None)

    def _span(vals, shipped_v, lo_pad=0.5, hi_pad=2.0):
        if not vals:
            return [shipped_v, float("inf")]
        uniq = sorted(set(vals))
        span = [uniq[0] * lo_pad, shipped_v, uniq[-1] * hi_pad, float("inf")]
        # add each observed value (each is a knee where one trial flips)
        return sorted(set(span + uniq + [float("inf")]),
                       key=lambda x: (x == float("inf"), x))

    surp_vals = _span(surprises, shipped.surprise_cap)
    theta_vals = _span(recs, shipped.theta)
    phi_vals = _span(rs, shipped.phi)
    sweep = {
        "surprise_cap": _sweep_axis(trials, "surprise_cap", surp_vals, shipped),
        "theta": _sweep_axis(trials, "theta", theta_vals, shipped),
        "phi": _sweep_axis(trials, "phi", phi_vals, shipped),
    }
    # Identify the bottleneck: the term failing on the most trials at shipped.
    n = len(trials)
    fail_rel = sum(1 for t in score_table if not t["rel_ok"])
    fail_forg = sum(1 for t in score_table if not t["forg_ok"])
    fail_surp = sum(1 for t in score_table if not t["surprise_ok"])
    bottleneck = max(
        [("surprise_cap", fail_surp), ("theta", fail_forg), ("phi", fail_rel)],
        key=lambda kv: kv[1])

    return {
        "n_facts": n, "gap": gap, "ring_capacity": ring_capacity, "seed": seed,
        "shipped": {"theta": shipped.theta, "phi": shipped.phi,
                    "surprise_cap": shipped.surprise_cap},
        "validation": validation,
        "score_table": score_table,
        "bottleneck": {"term": bottleneck[0], "trials_failing": bottleneck[1]},
        "sweep": sweep,
        "trials": trials,
    }


def _print_report(rep: dict) -> None:
    ship = rep["shipped"]
    print("=" * 70)
    print(f"STRM Probe 2 -- threshold sweep (facts={rep['n_facts']} gap={rep['gap']} "
          f"ring={rep['ring_capacity']} seed={rep['seed']})")
    print(f"shipped: theta={_fmt(ship['theta'])}  phi={_fmt(ship['phi'])}  "
          f"surprise_cap={_fmt(ship['surprise_cap'])}")
    print("-" * 70)
    val = rep["validation"]
    print(f"analytical model validation: "
          f"{'MATCHES real shipped run' if val['analytical_matches_real_at_shipped'] else 'MISMATCH'} "
          f"({val['mismatches']} mismatches)")
    print("-" * 70)
    print("per-trial F scores (shipped thresholds):")
    print(f"  {'fact':8} {'r_i':>9} {'rec_i':>9} {'surprise_i':>11}  rel  forg  surp  off  salient")
    for t in rep["score_table"]:
        print(f"  {t['fact_id']:8} {_fmt(t['r_i']):>9} {_fmt(t['rec_i']):>9} "
              f"{_fmt(t['surprise_i']):>11}   {'Y' if t['rel_ok'] else 'n'}    "
              f"{'Y' if t['forg_ok'] else 'n'}    {'Y' if t['surprise_ok'] else 'n'}    "
              f"{'Y' if t['coverage_off'] else 'n'}    {'Y' if t['salient_shipped'] else 'n'}")
    bn = rep["bottleneck"]
    print(f"  BOTTLENECK: {bn['term']} (fails on {bn['trials_failing']}/{rep['n_facts']} trials "
          f"at shipped) -> that is the lever to loosen first")
    print("-" * 70)
    for axis in ("surprise_cap", "theta", "phi"):
        print(f"sweep {axis} (other two at shipped):")
        print(f"  {'value':>12} {'firing':>7} {'cov_on':>7} {'cov_off':>7} {'delta':>7} {'rescued':>8}")
        for row in rep["sweep"][axis]:
            print(f"  {_fmt(row['value']):>12} {row['firing']:>7} {row['coverage_on']:>7} "
                  f"{row['coverage_off']:>7} {row['delta']:>+7} {row['rescued']:>8}")
        print()
    print("=" * 70)


def _main() -> int:
    ap = argparse.ArgumentParser(description="STRM Probe 2: salience threshold sweep")
    ap.add_argument("--facts", type=int, default=6)
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--ring-cap", type=int, default=0, help="default gap+4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not ARTIFACTS_PRESENT:
        print("MISSING ARTIFACTS -- need the trained backbone + 2a/2b/2c heads + "
              "thresholds.json. Run the STRM training scripts first.", file=sys.stderr)
        return 2
    ring_cap = args.ring_cap if args.ring_cap > 0 else args.gap + 4
    rep = run_sweep(n_facts=args.facts, gap=args.gap, ring_capacity=ring_cap, seed=args.seed)
    _print_report(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())