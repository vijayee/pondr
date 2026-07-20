"""STRM Phase 4 Probe 1: context-coverage ON vs OFF.

The first concrete slice of the deferred Step 7 ship-deciding eval. Measures
whether ``--strm-salience`` surfaces a seeded fact into the retrieved context at
query time (after an aging gap) MORE than the flag-off baseline -- a direct
"value at generating context" metric that needs NO LLM judge. Answer-quality
(the ERAG LLM-judge) is deferred to the full Step 7 eval; this probe only
measures context *coverage* + the salience firing rate.

Scenario per trial (one target fact F out of an M-fact corpus):
  1. SEED: inject F into the WM ring (``source_id=F.id``, ``text=F.summary``) --
     simulates "F was discussed a few turns ago and is sitting in working
     memory". The corpus (F + M-1 distractor episodes) is pre-encoded into the
     store with real bge summary embeddings.
  2. AGE: inject K filler slots (unrelated text, no source_id -> unscoreable ->
     never salient). Each filler is one SSM step that drifts the recurrent state
     away from F, so F becomes the oldest, most-forgotten ring slot.
  3. QUERY: ``query(Q_F)`` where Q_F is ASSOCIATIVELY related to F (the 2a head
     should score it relevant) but NOT a lexical/cosine near-duplicate (so the
     prompt-driven vector search may rank a distractor above F and miss it).

Metric at the query turn:
  fact_in_context: F.episode_id in ``retrieved_episodes``.
  salience fired on F: a ``recall`` signal with ``anchor_source_id == F.id``.
  rescued: ON found F AND OFF did not (the salience contribution).

Two conditions, same corpus + same scenario: ``--strm-salience`` ON vs OFF.
Aggregate over the fact bank: coverage_on, coverage_off, delta, firing stats.

Real stack (honest measurement): trained Phase 2a backbone + real bge-small +
the three STRM heads (2a/2b/2c) + the real thresholds sidecar + ring ON. No
Bonsai (a stub planner returns an empty plan -> graph traversal finds no
candidates -> the retriever's semantic fallback runs REAL vector search; the
salience path uses ``retrieve_by_embedding``, also real vector search). No LLM
(stub mode_a). No gate (the non-routing ``retrieve()`` path). No encoder (no
live-encode; the store is read-only across the two conditions).

Two threshold modes:
  real       -- the shipped ``thresholds.json`` sidecar. The salience AND is
                deliberately selective (theta/phi/surprise_cap are val
                percentiles), so this reports the HONEST today-firing rate
                (which may be low -- that is the finding).
  permissive -- theta=+inf, phi=-inf, surprise_cap=+inf so every SCORED anchor
                is salient. This is the UPPER BOUND on the mechanism's value:
                "if salience fired on every forgotten-relevant anchor, what is
                the coverage delta?" Isolates the retrieval/merge/pin mechanism
                from the threshold-tuning question.

Usage:
  python scripts/eval_strm_context_coverage.py --facts 6 --gap 10 \
      --ring-cap 16 --thresholds real --seed 0 --out report.json
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

# Allow ``python scripts/...`` from the repo root without an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import Phase2cConfig
from src.memory.episode import Episode
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.runtime import DEFAULT_BACKBONE_PATH
from src.subconscious.configs import BackboneConfig
from src.subconscious.latent_dynamics_head import load_latent_dynamics_head
from src.subconscious.recoverability_head import load_recoverability_head
from src.subconscious.relevance_head import load_relevance_head
from src.subconscious.salience import SalienceThresholds, load_salience_thresholds
from src.subconscious.training.routing_training import build_embedder, load_backbone

# ── artifacts (gated; the run skips with a clear message if any are absent) ──

REC_CKPT = Path("data/training/strm_recoverability/best.pt")
LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")
REL_CKPT = Path("data/training/strm_relevance/best.pt")
THRESHOLDS = Path("data/training/strm_salience/thresholds.json")
BACKBONE = Path(DEFAULT_BACKBONE_PATH)

ARTIFACTS_PRESENT = all(p.exists() for p in (REC_CKPT, LD_CKPT, REL_CKPT, THRESHOLDS, BACKBONE))


# ── stubs (no Bonsai, no LLM) ──

class _StubPlanner:
    """Returns an EMPTY plan so graph traversal finds no candidates and the
    retriever falls back to REAL vector search (the semantic fallback). This
    isolates the salience delta to the vector-search path both conditions
    share, with no Bonsai dependency."""

    def plan(self, prompt: str, conversation_history=None) -> dict:
        return {"entities": [], "topics": [], "tones": [], "entity_mode": "union"}


class _StubModeA:
    """No LLM round-trip. The probe measures retrieval coverage, not synthesis."""

    def __init__(self, reply: str = "SYNTH") -> None:
        self.reply = reply

    def _complete(self, messages, tools=None, tool_choice=None):
        return self.reply, None


# ── fact bank: ASSOCIATIVE (fact, query) pairs ──
# Each query is RELATED to its fact (so 2a relevance scores r_i high) but NOT a
# cosine near-duplicate (so prompt-driven vector search can rank a distractor
# above the fact and miss it -- the gap salience closes).

FACT_BANK = [
    ("Alice chose Postgres for the audit log because it needs ACID compliance.",
     "What are we using to meet our compliance record-keeping requirements?"),
    ("The deploy pipeline failed because the CUDA driver was 535 and sm_120 needs 610 or newer.",
     "Why did the GPU build break on the new hardware?"),
    ("Bob scheduled the architecture review for Thursday at 2pm.",
     "When is the design sync happening?"),
    ("We capped the proactive retrieval budget at 3 to bound per-turn latency.",
     "How do we keep proactive recall cheap?"),
    ("The vector index moved from FAISS to the in-DB WaveDB VectorLayer.",
     "Where do the summary embeddings live now?"),
    ("Carol said the ontology bug was a disconnected class DAG, fixed with a two-encoder classifier.",
     "What was the root cause of the resolution failures?"),
    ("The forgetting curve flattens after about eight turns without re-exposure.",
     "How long until a memory stabilizes on its own?"),
    ("Dave pushed the migration cut-over to the end of the sprint to avoid a Friday deploy.",
     "When will the migration go live?"),
    ("The salience trigger suppresses proactive recall on high-surprise turns.",
     "When does the engine hold back a pre-emptive fetch?"),
    ("We picked cosine over dot product so magnitude does not dominate the ranking.",
     "Why did we normalize the similarity score?"),
]

# Fillers: unrelated text whose embeddings drift the WM state away from the
# target fact. No source_id / no text on the ring slot -> unscoreable -> never
# salient (so the ONLY salience candidate is the seeded fact F).
FILLER_BANK = [
    "The weather forecast predicts rain for the weekend.",
    "My favorite recipe uses thyme and roasted garlic.",
    "The hiking trail closes at sunset in winter.",
    "She bought a new acoustic guitar last Tuesday.",
    "The marathon route was changed due to road work.",
    "Their garden has tomatoes, basil, and marigolds.",
    "The train to the coast leaves at 7:15 on weekdays.",
    "He painted the kitchen a soft sage green.",
    "The documentary covered Pacific coral reefs.",
    "Yoga classes moved to the community center.",
    "The bookstore ordered extra copies of the new novel.",
    "A hawk nested on the chapel roof this spring.",
    "The ferry was cancelled by a small craft advisory.",
    "Her violin recital moved to the old town hall.",
]


def _permissive_thresholds() -> SalienceThresholds:
    """Every SCORED anchor is salient (the AND passes for any non-None scores).
    Upper bound on the mechanism's value, isolating retrieval/merge/pin from
    threshold tuning."""
    return SalienceThresholds(
        theta=1e18, phi=-1e18, surprise_cap=1e18,
        theta_percentile=0.0, phi_percentile=100.0, surprise_cap_percentile=100.0,
        basis="permissive-upper-bound", n_recoverability=0, n_relevance=0, n_latent_dynamics=0,
    )


def _seed_corpus(store: HippocampalStore, embedder, facts: list[tuple[str, str]]) -> None:
    """Pre-encode the M fact episodes with real bge summary embeddings and
    persist them. The store indexes ``episode.summary_embedding`` into the
    vector layer (it does NOT auto-embed), so the caller must set it."""
    for i, (summary, _query) in enumerate(facts):
        ep = Episode(
            id=f"fact_{i:02d}",
            timestamp="2026-07-01T10:00:00",
            summary=summary,
            full_text=f"User: {summary}\nAssistant: noted.",
            entities=[], topics=[], tones=[], decisions=[],
        )
        ep.summary_embedding = embedder.encode([summary])[0]
        store.encode_episode(ep)


def _reset_for_trial(orch: PonderOrchestrator) -> None:
    """Reset the WM ring + the salience bookkeeping so each trial is
    independent (no cross-trial state leak). The store + retriever are shared
    and read-only across trials."""
    orch.working_memory.reset()
    orch._salience_turn_count = 0
    orch._source_entry_turn = {}
    orch._salience_anchors = None
    orch._salience_fired_episodes = None
    orch._salience_signals = None


def _seed_ring(orch: PonderOrchestrator, fact_id: str, fact_summary: str,
               fillers: list[str]) -> None:
    """SEED F into the ring (with provenance) then AGE with K filler steps
    (no provenance -> unscoreable -> never salient). F ends up the oldest
    ring slot, the only salience candidate."""
    f_emb = orch.working_memory.embed([fact_summary])[0]
    orch.working_memory.inject(f_emb, source_id=fact_id, text=fact_summary)
    for filler_text in fillers:
        # text=None -> the slot is unscoreable (no doc_emb) -> never salient.
        f_emb_filler = orch.working_memory.embed([filler_text])[0]
        orch.working_memory.inject(f_emb_filler, source_id=None, text=None)


def _run_condition(orch: PonderOrchestrator, fact_id: str, fact_summary: str,
                   query: str, fillers: list[str]) -> dict:
    """One condition (ON or OFF) of one trial. Returns the per-turn
    measurement: was F in retrieved_episodes, did salience fire on F, the
    salience retrieval count + signal breakdown."""
    _reset_for_trial(orch)
    _seed_ring(orch, fact_id, fact_summary, fillers)
    res = orch.query(query)
    retrieved = res.get("retrieved_episodes", []) or []
    retrieved_ids = {ep.get("episode_id") for ep in retrieved}
    signals = res.get("salience_signals", []) or []
    fired_on_f = [s for s in signals if s.get("anchor_source_id") == fact_id]
    return {
        "fact_in_context": fact_id in retrieved_ids,
        "salience_fired_on_fact": any(s.get("kind") == "recall" for s in fired_on_f),
        "salience_retrieval_count": res.get("salience_retrieval_count"),
        "n_signals": len(signals),
        "signal_kinds": {k: sum(1 for s in signals if s.get("kind") == k)
                         for k in ("recall", "stale_uncertain")},
    }


def _run_trial(fact_idx: int, fact_summary: str, query: str, fillers: list[str],
               orch_on: PonderOrchestrator, orch_off: PonderOrchestrator) -> dict:
    """Run one (fact, query) pair under both conditions and compute the delta."""
    fact_id = f"fact_{fact_idx:02d}"
    on = _run_condition(orch_on, fact_id, fact_summary, query, fillers)
    off = _run_condition(orch_off, fact_id, fact_summary, query, fillers)
    return {
        "fact_id": fact_id,
        "summary": fact_summary,
        "query": query,
        "on": on,
        "off": off,
        "rescued": on["fact_in_context"] and not off["fact_in_context"],
    }


def _build_orchestrator(store: HippocampalStore, retriever: HippocampalRetriever,
                        embedder, backbone, *, strm_salience: bool,
                        thresholds: Optional[SalienceThresholds],
                        recoverability_head, latent_dynamics_head, relevance_head,
                        ring_capacity: int, cfg: Phase2cConfig) -> PonderOrchestrator:
    return PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone, embedder=embedder,
        mode_a=_StubModeA(), config=cfg, user_id="probe",
        ring_capacity=ring_capacity,
        recoverability_head=recoverability_head,
        latent_dynamics_head=latent_dynamics_head,
        relevance_head=relevance_head,
        strm_salience=strm_salience, salience_thresholds=thresholds,
    )


def run_eval(*, n_facts: int, gap: int, ring_capacity: int,
             thresholds_mode: str, seed: int) -> dict:
    """Run the ON-vs-OFF coverage probe over ``n_facts`` trials and aggregate."""
    rng = random.Random(seed)
    facts = list(FACT_BANK[:n_facts])
    if len(facts) < n_facts:
        raise ValueError(f"fact bank has {len(FACT_BANK)} pairs; --facts {n_facts} too large")
    # Per-trial fillers: distinct fillers up to the pool size, then cycled, so
    # any --gap works (rng.sample would raise on gap > pool). Shuffled per trial
    # so the state drift differs across facts (no one ordering dominates).
    filler_pool = list(FILLER_BANK)

    thresholds = (load_salience_thresholds(str(THRESHOLDS))
                  if thresholds_mode == "real" else _permissive_thresholds())

    tmpdir = tempfile.mkdtemp(prefix="pondr_probe1_")
    store: Optional[HippocampalStore] = None
    trials: list = []
    try:
        db_path = str(Path(tmpdir) / "db")
        store = HippocampalStore(db_path)
        embedder = build_embedder("on-demand")
        rec = load_recoverability_head(str(REC_CKPT), device="cpu")
        ld = load_latent_dynamics_head(str(LD_CKPT), device="cpu")
        rel = load_relevance_head(str(REL_CKPT), device="cpu")
        retriever = HippocampalRetriever(
            store, planner=_StubPlanner(), auto_load_index=True,
            retrieval_gate=None, embedder=embedder,
        )
        _seed_corpus(store, embedder, facts)

        cfg = Phase2cConfig()
        cfg.session.state_dir = str(Path(tmpdir) / "sessions")
        backbone_on = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        backbone_off = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")
        orch_on = _build_orchestrator(
            store, retriever, embedder, backbone_on, strm_salience=True,
            thresholds=thresholds, recoverability_head=rec, latent_dynamics_head=ld,
            relevance_head=rel, ring_capacity=ring_capacity, cfg=cfg)
        orch_off = _build_orchestrator(
            store, retriever, embedder, backbone_off, strm_salience=False,
            thresholds=None, recoverability_head=None, latent_dynamics_head=None,
            relevance_head=None, ring_capacity=ring_capacity, cfg=cfg)

        for i, (summary, query) in enumerate(facts):
            shuffled = filler_pool[:]
            rng.shuffle(shuffled)
            fillers = [shuffled[j % len(shuffled)] for j in range(gap)]
            trials.append(_run_trial(i, summary, query, fillers, orch_on, orch_off))
    finally:
        # Close the store BEFORE rmtree so the wavedb file locks release. Both
        # are best-effort: a construction failure mid-build must not leak the
        # temp dir, and on Windows the locks may lag the close.
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)

    n = len(trials)
    cov_on = sum(1 for t in trials if t["on"]["fact_in_context"]) / n
    cov_off = sum(1 for t in trials if t["off"]["fact_in_context"]) / n
    rescued = sum(1 for t in trials if t["rescued"])
    fired_on_fact = sum(1 for t in trials if t["on"]["salience_fired_on_fact"])
    avg_salience_count = (sum(t["on"]["salience_retrieval_count"] or 0 for t in trials) / n)
    return {
        "n_facts": n, "gap": gap, "ring_capacity": ring_capacity,
        "thresholds": thresholds_mode, "seed": seed,
        "coverage_on": round(cov_on, 4),
        "coverage_off": round(cov_off, 4),
        "coverage_delta": round(cov_on - cov_off, 4),
        "rescued": rescued,
        "salience_fired_on_fact": fired_on_fact,
        "avg_salience_retrieval_count": round(avg_salience_count, 4),
        "trials": trials,
    }


def _main() -> int:
    ap = argparse.ArgumentParser(description="STRM Phase 4 Probe 1: context-coverage ON vs OFF")
    ap.add_argument("--facts", type=int, default=6, help="number of (fact,query) trials (default 6)")
    ap.add_argument("--gap", type=int, default=10, help="filler steps between seed and query (default 10)")
    ap.add_argument("--ring-cap", type=int, default=0,
                    help="WM ring capacity K (default gap+4)")
    ap.add_argument("--thresholds", choices=("real", "permissive"), default="real",
                    help="real=shipped sidecar (honest firing rate); "
                         "permissive=upper bound (every scored anchor salient)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="write the JSON report to this path")
    args = ap.parse_args()

    if not ARTIFACTS_PRESENT:
        print("MISSING ARTIFACTS -- need the trained backbone + 2a/2b/2c heads + "
              "thresholds.json. Run the STRM training scripts first.", file=sys.stderr)
        return 2

    ring_cap = args.ring_cap if args.ring_cap > 0 else args.gap + 4
    if ring_cap < args.gap + 2:
        print(f"[warn] ring-cap {ring_cap} < gap+2 ({args.gap + 2}); the seeded "
              f"fact may be evicted before the query. Bumping to gap+4.",
              file=sys.stderr)
        ring_cap = args.gap + 4

    report = run_eval(n_facts=args.facts, gap=args.gap, ring_capacity=ring_cap,
                      thresholds_mode=args.thresholds, seed=args.seed)

    print("=" * 64)
    print(f"STRM Probe 1 -- context coverage (facts={report['n_facts']} "
          f"gap={report['gap']} ring={report['ring_capacity']} "
          f"thresholds={report['thresholds']} seed={report['seed']})")
    print("-" * 64)
    print(f"  coverage ON  : {report['coverage_on']:.2%}")
    print(f"  coverage OFF : {report['coverage_off']:.2%}")
    print(f"  delta        : {report['coverage_delta']:+.2%}")
    print(f"  rescued (ON found, OFF missed): {report['rescued']}/{report['n_facts']}")
    print(f"  salience fired on the fact    : {report['salience_fired_on_fact']}/{report['n_facts']}")
    print(f"  avg salience retrieval count  : {report['avg_salience_retrieval_count']:.2f}")
    print("-" * 64)
    for t in report["trials"]:
        on, off = t["on"], t["off"]
        print(f"  {t['fact_id']}: ON={'Y' if on['fact_in_context'] else 'n'} "
              f"OFF={'Y' if off['fact_in_context'] else 'n'} "
              f"salience={'recall' if on['salience_fired_on_fact'] else '-'} "
              f"(signals={on['n_signals']} {on['signal_kinds']})  {t['query'][:48]}")
    print("=" * 64)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())