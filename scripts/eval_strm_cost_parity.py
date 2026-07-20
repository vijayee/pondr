"""STRM Phase 4 Probe 3: cost parity -- salience-decided vs fixed-interval
proactive recall, at EQUAL proactive budget. + a selectivity diagnostic.

Probe 2 found the bottleneck is a miscalibrated theta AND that the 2a relevance
head saturates (~1.0) so phi is never a lever on the single-query scenario.
Probe 3 asks the cost-parity question that decides whether salience is worth
integrating, on a MULTI-TURN scenario where the relevance gate actually has to
WORK (discriminate fact-relevant turns from filler turns) for salience to time
its recalls.

WHY THIS IS THE RIGHT QUESTION: salience's value proposition is not "more
retrieval" -- it is "retrieve the RIGHT thing at the RIGHT time." A fixed-
interval policy that proactively recalls every N turns spends the same budget;
salience spends it only when a forgotten anchor is RELEVANT to the current
query. IF the relevance head discriminates, salience concentrates its budget on
fact-relevant turns and beats fixed-interval at equal budget. IF it does not
discriminate, salience fires on filler turns too, wastes its budget, and is
dominated by fixed-interval. Probe 3 tests exactly this.

SCENARIO (isolates the proactive mechanism): prompt-driven retrieval is disabled
(``retrieve -> []``) so the ONLY path a fact reaches context is a proactive
recall. Seed M facts into the WM ring at turn 0; run a horizon of T turns where
each turn steps the WM state with a query embedding (aging the facts); a few
"fact-relevant" turns query an associative prompt about ONE fact, the rest are
unrelated filler. The facts decay in-ring as the state drifts.

Three conditions:
  OFF   -- no proactive. Coverage on fact-relevant turns = 0 (retrieve is []).
  STRM  -- salience ON with a RETUNED theta (serve-distribution, from Probe 2).
           Salience fires on turns where a fact anchor is forgotten + relevant
           to the query -> proactive recall -> fact in context. Cost = total
           salience signals over the horizon.
  FIXED -- salience OFF; every N turns a manual proactive recall ROUND-ROBINS
           over the M seeded facts (uniform spread, no relevance signal -- the
           fair "schedule-decided, no-relevance" baseline). N tuned so FIXED's
           call count <= STRM's call count.

METRIC: on each fact-relevant turn, was the TARGET fact proactively recalled?
  STRM  : a recall signal with anchor_source_id == target.
  FIXED : the manual proactive this turn recalled the target.
  OFF   : never.

SELECTIVITY DIAGNOSTIC: for each fact, compare its 2a relevance score r_i on its
PROBE turn vs its r_i on FILLER turns. If r_i is ~equal (both near-saturated),
the relevance head does NOT discriminate and the salience trigger cannot time
its recalls -- the root cause of any cost-parity failure, surfaced in the
harness output (not just ad-hoc).

GATE: STRM beats FIXED at equal-or-lower budget (strm_beats_fixed). If STRM is
DOMINATED (higher cost AND lower coverage than FIXED), the mechanism fails the
cost-parity gate -- exactly the ship-deciding signal the user asked for.

Usage:
  python scripts/eval_strm_cost_parity.py --facts 4 --horizon 30 --ring-cap 32 \
      --theta -0.04 --seed 0
"""

from __future__ import annotations

import argparse
import json
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

from scripts.eval_strm_context_coverage import (
    ARTIFACTS_PRESENT, FACT_BANK, _StubPlanner, _build_orchestrator,
    _reset_for_trial, _seed_corpus,
)

REC_CKPT = Path("data/training/strm_recoverability/best.pt")
LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")
REL_CKPT = Path("data/training/strm_relevance/best.pt")
THRESHOLDS = Path("data/training/strm_salience/thresholds.json")
BACKBONE = Path(DEFAULT_BACKBONE_PATH)

# Unrelated filler prompts (step the WM state, no fact relevance -- IF the 2a
# head discriminates, these score LOW r_i on every fact anchor).
FILLER_QUERIES = [
    "Tell me about the weather this week.",
    "What is a good recipe for dinner?",
    "Describe a hiking trail you know.",
    "How does a guitar produce sound?",
    "Explain marathon training basics.",
    "What grows well in a summer garden?",
    "How do train schedules work?",
    "What color should I paint a kitchen?",
    "Tell me about Pacific coral reefs.",
    "Where are local yoga classes held?",
    "What makes a good novel?",
    "Where do hawks usually nest?",
    "How are ferries scheduled?",
    "What is a violin recital like?",
]


def _disable_prompt_driven(orch):
    """Isolate the proactive mechanism: prompt-driven retrieve is a no-op so the
    ONLY way a fact reaches context is a proactive (salience / manual) recall."""
    orch.retriever.retrieve = lambda *a, **k: []
    orch.retriever.retrieve_with_routing = lambda *a, **k: {
        "supported": True, "results": [], "route": None}


def _seed_ring_facts(orch, facts):
    """Seed all M facts into the WM ring at turn 0 (provenance on each)."""
    for i, (summary, _q) in enumerate(facts):
        fid = f"fact_{i:02d}"
        emb = orch.working_memory.embed([summary])[0]
        orch.working_memory.inject(emb, source_id=fid, text=summary)


def _ring_fact_ids(orch) -> list:
    """source_ids of fact slots currently in the ring (oldest-first)."""
    return [s.source_id for s in orch.working_memory.ring_buffer()
            if s.source_id is not None]


def _run_strm(orch, schedule, facts) -> dict:
    """STRM condition: salience ON. Returns per-turn records + total cost + the
    per-turn anchor scores (for the selectivity diagnostic)."""
    _reset_for_trial(orch)
    _disable_prompt_driven(orch)
    _seed_ring_facts(orch, facts)
    records = []
    per_turn_scores = []  # [{turn, target, anchors: {fid: (r_i, rec_i, salient)}}]
    cost = 0
    for t, (query, target) in enumerate(schedule):
        res = orch.query(query)
        sigs = res.get("salience_signals", []) or []
        cost += len(sigs)
        target_recalled = (target is not None and any(
            s.get("kind") == "recall" and s.get("anchor_source_id") == target
            for s in sigs))
        anchors = orch._salience_anchors or []
        scores = {a.source_id: (a.r_i, a.rec_i, bool(a.salient))
                  for a in anchors if a.source_id is not None}
        per_turn_scores.append({"turn": t, "target": target, "anchors": scores})
        records.append({"turn": t, "target": target,
                        "target_recalled": target_recalled,
                        "salience_fired": len(sigs)})
    return {"records": records, "cost": cost, "per_turn_scores": per_turn_scores}


def _run_off(orch, schedule, facts) -> dict:
    """OFF condition: no proactive. Coverage is always False (retrieve is [])."""
    _reset_for_trial(orch)
    _disable_prompt_driven(orch)
    _seed_ring_facts(orch, facts)
    records = []
    for t, (query, target) in enumerate(schedule):
        orch.query(query)
        records.append({"turn": t, "target": target, "target_recalled": False})
    return {"records": records, "cost": 0}


def _run_fixed(orch, schedule, N, facts) -> dict:
    """FIXED condition: salience OFF + manual proactive every N turns. N tuned
    to match STRM's cost. The manual proactive ROUND-ROBINS over the M seeded
    facts (uniform spread, no relevance signal) -- the fair "schedule-decided,
    no-relevance" baseline. Coverage on a fact-relevant turn = the manual
    proactive THIS turn recalled the target (only possible on a scheduled turn
    whose round-robin index happens to land on the target)."""
    _reset_for_trial(orch)
    _disable_prompt_driven(orch)
    _seed_ring_facts(orch, facts)
    records = []
    cost = 0
    manual_count = 0
    for t, (query, target) in enumerate(schedule):
        recalled_this_turn = False
        if N > 0 and t % N == 0 and t > 0:
            # Round-robin over the seeded facts (uniform spread, no relevance).
            fact_ids = [sid for sid in _ring_fact_ids(orch)
                        if sid and sid.startswith("fact_")]
            if fact_ids:
                fid = fact_ids[manual_count % len(fact_ids)]
                manual_count += 1
                slot = next((s for s in orch.working_memory.ring_buffer()
                             if s.source_id == fid), None)
                if slot is not None and slot.text is not None:
                    emb = orch.working_memory.embed([slot.text])[0]
                    hits = orch.retriever.retrieve_by_embedding(emb)
                    cost += 1
                    hit_ids = {h.get("episode_id") for h in hits}
                    if hits:
                        top = hits[0]
                        top_text = top.get("summary", "") or top.get("text", "")
                        top_emb = orch.working_memory.embed([top_text])[0]
                        orch.working_memory.inject(
                            top_emb, source_id=top.get("episode_id"),
                            text=top_text, pin=True)
                    recalled_this_turn = target is not None and target in hit_ids
        orch.query(query)
        records.append({"turn": t, "target": target,
                        "target_recalled": recalled_this_turn})
    return {"records": records, "cost": cost}


def _selectivity(per_turn_scores, fact_ids) -> dict:
    """Does the 2a relevance head discriminate fact-relevant turns from filler?
    For each fact: r_i on its PROBE turn vs mean r_i on FILLER turns. If the gap
    is small (both near-saturated), the relevance gate is non-functional and
    salience cannot time its recalls -- the root cause of cost-parity failure."""
    per_fact = {}
    for fid in fact_ids:
        probe_r, filler_rs = None, []
        for row in per_turn_scores:
            sc = row["anchors"].get(fid)
            if sc is None:
                continue
            r_i = sc[0]
            if row["target"] == fid:
                probe_r = r_i
            elif row["target"] is None:
                filler_rs.append(r_i)
        mean_filler = sum(filler_rs) / len(filler_rs) if filler_rs else None
        gap = (probe_r - mean_filler) if (probe_r is not None and mean_filler is not None) else None
        per_fact[fid] = {"probe_r_i": probe_r,
                         "mean_filler_r_i": mean_filler,
                         "n_filler": len(filler_rs), "gap": gap}
    # The head discriminates iff every fact's probe r_i exceeds its filler mean
    # by a meaningful margin (>= 0.2). Saturated-near-1.0 on both = no selectivity.
    gaps = [f["gap"] for f in per_fact.values() if f["gap"] is not None]
    discriminates = bool(gaps and all(g >= 0.2 for g in gaps))
    return {"per_fact": per_fact, "discriminates": discriminates,
            "min_gap": min(gaps) if gaps else None}


def run_probe(*, n_facts: int, horizon: int, ring_capacity: int,
              theta: float, seed: int) -> dict:
    facts = list(FACT_BANK[:n_facts])
    fact_ids = [f"fact_{i:02d}" for i in range(n_facts)]
    shipped = load_salience_thresholds(str(THRESHOLDS))
    # Retune theta to the serve distribution (Probe 2); keep phi + surprise_cap.
    retuned = SalienceThresholds(
        theta=theta, phi=shipped.phi, surprise_cap=shipped.surprise_cap,
        theta_percentile=50.0, phi_percentile=shipped.phi_percentile,
        surprise_cap_percentile=shipped.surprise_cap_percentile,
        basis=f"retuned theta={theta} (serve dist, Probe 2); phi/surprise shipped",
        n_recoverability=shipped.n_recoverability, n_relevance=shipped.n_relevance,
        n_latent_dynamics=shipped.n_latent_dynamics)

    # Schedule: fact i is probed at turn probe_start + i*probe_step (after
    # decay); the rest are filler. Each fact probed once.
    probe_start = max(8, horizon // 4)
    probe_step = (max(4, (horizon - probe_start - 2) // max(1, n_facts - 1))
                  if n_facts > 1 else 1)
    fact_turns = {}
    for i in range(n_facts):
        t = min(probe_start + i * probe_step, horizon - 1)
        fact_turns[t] = i
    filler_pool = list(FILLER_QUERIES)
    schedule = []
    for t in range(horizon):
        if t in fact_turns:
            fi = fact_turns[t]
            schedule.append((facts[fi][1], f"fact_{fi:02d}"))  # associative query
        else:
            schedule.append((filler_pool[t % len(filler_pool)], None))

    tmpdir = tempfile.mkdtemp(prefix="pondr_probe3_")
    store: Optional[HippocampalStore] = None
    try:
        store = HippocampalStore(str(Path(tmpdir) / "db"))
        embedder = build_embedder("on-demand")
        rec = load_recoverability_head(str(REC_CKPT), device="cpu")
        ld = load_latent_dynamics_head(str(LD_CKPT), device="cpu")
        rel = load_relevance_head(str(REL_CKPT), device="cpu")
        retriever = HippocampalRetriever(
            store, planner=_StubPlanner(), auto_load_index=True,
            retrieval_gate=None, embedder=embedder)
        # Corpus: the M facts (distractors unnecessary -- retrieve is [] so
        # prompt-driven never runs; only retrieve_by_embedding on fact slots).
        _seed_corpus(store, embedder, facts)
        cfg = Phase2cConfig()
        cfg.session.state_dir = str(Path(tmpdir) / "sessions")
        common = dict(store=store, retriever=retriever, embedder=embedder,
                      ring_capacity=ring_capacity, cfg=cfg)

        bb_strm = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        orch_strm = _build_orchestrator(
            backbone=bb_strm, strm_salience=True, thresholds=retuned,
            recoverability_head=rec, latent_dynamics_head=ld, relevance_head=rel,
            **common)

        bb_off = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        orch_off = _build_orchestrator(
            backbone=bb_off, strm_salience=False, thresholds=None,
            recoverability_head=None, latent_dynamics_head=None,
            relevance_head=None, **common)

        bb_fix = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        orch_fix = _build_orchestrator(
            backbone=bb_fix, strm_salience=False, thresholds=None,
            recoverability_head=None, latent_dynamics_head=None,
            relevance_head=None, **common)

        # Run STRM first to get its cost, then tune FIXED's N so FIXED's cost
        # is <= STRM's (FIXED never spends MORE budget than STRM -- the fair
        # "equal-or-lower budget" bar for the gate).
        strm = _run_strm(orch_strm, schedule, facts)
        off = _run_off(orch_off, schedule, facts)
        target_cost = max(strm["cost"], 1)
        N = max(1, horizon // target_cost)
        fixed = _run_fixed(orch_fix, schedule, N, facts)
        selectivity = _selectivity(strm["per_turn_scores"], fact_ids)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)

    def _coverage(rep):
        recs = [r for r in rep["records"] if r["target"] is not None]
        return sum(1 for r in recs if r["target_recalled"]), len(recs)

    s_hit, s_n = _coverage(strm)
    f_hit, f_n = _coverage(fixed)
    o_hit, o_n = _coverage(off)
    s_cost, f_cost = strm["cost"], fixed["cost"]
    return {
        "n_facts": n_facts, "horizon": horizon, "ring_capacity": ring_capacity,
        "seed": seed, "theta_retuned": theta, "shipped_theta": shipped.theta,
        "shipped_phi": shipped.phi, "shipped_surprise_cap": shipped.surprise_cap,
        "fact_probe_turns": fact_turns,
        "strm": {"coverage": s_hit, "n_probes": s_n, "cost": s_cost},
        "fixed": {"coverage": f_hit, "n_probes": f_n, "cost": f_cost, "N": N},
        "off": {"coverage": o_hit, "n_probes": o_n, "cost": 0},
        "budget_parity": f_cost <= s_cost,
        "strm_beats_fixed": (s_hit > f_hit) and (s_cost <= f_cost),
        "strm_dominated_by_fixed": (s_cost > f_cost) and (s_hit < f_hit),
        "strm_beats_off": s_hit > o_hit,
        "selectivity": selectivity,
        "records_strm": strm["records"],
        "records_fixed": fixed["records"],
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    if v == float("inf"):
        return "+inf"
    if isinstance(v, float) and abs(v) < 1e-3 and v != 0.0:
        return f"{v:.2e}"
    return f"{v:.4g}"


def _print_report(rep: dict) -> None:
    print("=" * 72)
    print(f"STRM Probe 3 -- cost parity (facts={rep['n_facts']} horizon={rep['horizon']} "
          f"ring={rep['ring_capacity']} seed={rep['seed']})")
    print(f"retuned theta={rep['theta_retuned']} (shipped theta={_fmt(rep['shipped_theta'])}); "
          f"phi={_fmt(rep['shipped_phi'])} surprise_cap={_fmt(rep['shipped_surprise_cap'])}")
    print(f"fact probe turns: {rep['fact_probe_turns']}")
    print("-" * 72)
    print(f"  {'cond':6} {'coverage':>10} {'cost(calls)':>12}")
    for name, c in (("OFF", rep["off"]), ("STRM", rep["strm"]), ("FIXED", rep["fixed"])):
        print(f"  {name:6} {c['coverage']:>4}/{c['n_probes']:<5} {c['cost']:>12}")
    print("-" * 72)
    parity = "YES" if rep["budget_parity"] else "NO"
    print(f"  budget parity (FIXED cost <= STRM cost): {parity}  "
          f"(STRM={rep['strm']['cost']}, FIXED={rep['fixed']['cost']}, N={rep['fixed']['N']})")
    verdict = ("STRM BEATS FIXED at equal-or-lower budget" if rep["strm_beats_fixed"]
               else "STRM DOMINATED by FIXED (higher cost, lower coverage)" if rep["strm_dominated_by_fixed"]
               else "STRM does NOT beat FIXED at equal budget")
    print(f"  cost-parity gate: {verdict}")
    print(f"  STRM beats OFF (proactive helps at all): {rep['strm_beats_off']}")
    print("-" * 72)
    sel = rep["selectivity"]
    print(f"  selectivity (does the 2a relevance head discriminate?): "
          f"{'YES' if sel['discriminates'] else 'NO -- relevance gate non-functional'} "
          f"(min probe-minus-filler r_i gap = {_fmt(sel['min_gap'])}; gate needs >= 0.2)")
    print(f"    {'fact':8} {'probe_r':>9} {'filler_r':>9} {'gap':>9} {'n_filler':>8}")
    for fid, f in sel["per_fact"].items():
        print(f"    {fid:8} {_fmt(f['probe_r_i']):>9} {_fmt(f['mean_filler_r_i']):>9} "
              f"{_fmt(f['gap']):>9} {f['n_filler']:>8}")
    print("-" * 72)
    print("  per-probe (fact-relevant turns):")
    print(f"    {'turn':>4} {'target':8} {'STRM':>6} {'FIXED':>6}")
    sr = {r["turn"]: r for r in rep["records_strm"] if r["target"] is not None}
    fr = {r["turn"]: r for r in rep["records_fixed"] if r["target"] is not None}
    for t in sorted(set(sr) | set(fr)):
        s = sr.get(t, {})
        f = fr.get(t, {})
        print(f"    {t:>4} {s.get('target','-'):8} "
              f"{'Y' if s.get('target_recalled') else 'n':>6} "
              f"{'Y' if f.get('target_recalled') else 'n':>6}")
    print("=" * 72)


def _main() -> int:
    ap = argparse.ArgumentParser(description="STRM Probe 3: cost parity vs fixed-interval RAG")
    ap.add_argument("--facts", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--ring-cap", type=int, default=32)
    ap.add_argument("--theta", type=float, default=-0.04,
                    help="retuned recoverability threshold (serve dist, Probe 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not ARTIFACTS_PRESENT:
        print("MISSING ARTIFACTS -- need the trained backbone + 2a/2b/2c heads + "
              "thresholds.json. Run the STRM training scripts first.", file=sys.stderr)
        return 2
    rep = run_probe(n_facts=args.facts, horizon=args.horizon,
                    ring_capacity=args.ring_cap, theta=args.theta, seed=args.seed)
    _print_report(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())