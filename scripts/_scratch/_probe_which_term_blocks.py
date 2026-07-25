"""Diagnostic: which salience AND term blocks firing under shipped thresholds?

Reuses eval_strm_context_coverage's setup (real backbone + 2a/2b/2c heads +
shipped thresholds sidecar), seeds one fact, ages with fillers, queries the
probe question, then reads orch._salience_anchors to print each anchor's
(r_i, rec_i, surprise_i, salient) and which of the three AND terms pass.

GOAL: DeepSeek's hypothesis is surprise_cap=5.55e-05 is so tiny it blocks every
turn (cost=0/hit=0). This confirms or refutes that by showing the actual
per-turn scores vs the thresholds. UNTRACKED scratch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from src.config import Phase2cConfig
from src.memory.store import HippocampalStore
from src.orchestrator import PonderOrchestrator
from src.retrieval.retriever import HippocampalRetriever
from src.runtime import DEFAULT_BACKBONE_PATH
from src.subconscious.configs import BackboneConfig
from src.subconscious.latent_dynamics_head import load_latent_dynamics_head
from src.subconscious.recoverability_head import load_recoverability_head
from src.subconscious.relevance_head import load_relevance_head
from src.subconscious.salience import load_salience_thresholds
from src.subconscious.training.routing_training import build_embedder

# Reuse the eval's fact bank + fillers + stubs (single source of truth).
from scripts.eval_strm_context_coverage import (
    FACT_BANK, FILLER_BANK, _StubPlanner, _StubModeA,
    _seed_corpus, _seed_ring, _reset_for_trial,
)

REC_CKPT = Path("data/training/strm_relevance/best.pt")          # 2a lives here? no
# The 2a relevance head ckpt:
REL_CKPT = Path("data/training/strm_relevance/best.pt")
# Probe 1 uses these paths -- mirror them exactly:
REC_CKPT = Path("data/training/strm_recoverability/best.pt")
LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")
THRESHOLDS = Path("data/training/strm_salience/thresholds.json")
BACKBONE = Path(DEFAULT_BACKBONE_PATH)


def _fmt(v):
    if v is None:
        return " None "
    return f"{v:+.5f}"


def main() -> int:
    thresholds = load_salience_thresholds(str(THRESHOLDS))
    print(f"shipped thresholds: theta={thresholds.theta:+.5f} "
          f"phi={thresholds.phi:+.5f} surprise_cap={thresholds.surprise_cap:.3e}")
    print("  AND = (rec_i < theta) & (r_i > phi) & (surprise_i < surprise_cap)")
    print()

    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="pondr_which_")
    store = None
    try:
        db_path = str(Path(tmpdir) / "db")
        store = HippocampalStore(db_path)
        embedder = build_embedder("on-demand")
        rec = load_recoverability_head(str(REC_CKPT), device="cpu")
        ld = load_latent_dynamics_head(str(LD_CKPT), device="cpu")
        rel = load_relevance_head(str(REL_CKPT), device="cpu")
        retriever = HippocampalRetriever(
            store, planner=_StubPlanner(), auto_load_index=True, embedder=embedder,
        )
        # Seed the corpus with the first 6 facts (matches the eval).
        facts = list(FACT_BANK[:6])
        _seed_corpus(store, embedder, facts)

        cfg = Phase2cConfig()
        cfg.session.state_dir = str(Path(tmpdir) / "sessions")
        backbone = load_backbone_local(str(BACKBONE))
        orch = PonderOrchestrator(
            store=store, retriever=retriever, backbone=backbone, embedder=embedder,
            mode_a=_StubModeA(), config=cfg, user_id="probe",
            ring_capacity=16,
            recoverability_head=rec, latent_dynamics_head=ld, relevance_head=rel,
            strm_salience=True, salience_thresholds=thresholds,
        )

        # Trial: fact_00, age with 10 fillers, query the fact_00 probe question.
        fact_idx = 0
        fact_summary, query = facts[fact_idx]
        fact_id = f"fact_{fact_idx:02d}"
        fillers = list(FILLER_BANK[:10])
        _reset_for_trial(orch)
        _seed_ring(orch, fact_id, fact_summary, fillers)
        print(f"trial: {fact_id}  query={query[:60]!r}")
        print(f"ring slots after seed+age: {len(orch.working_memory.ring_buffer())}")
        res = orch.query(query)
        anchors = orch._salience_anchors
        if anchors is None:
            print("NO salience anchors stashed (salience did not run?)")
            return 2
        print(f"\n{'slot':>4} {'src':>9} {'r_i':>9} {'rec_i':>10} {'surprise':>11} "
              f"{'r>phi':>6} {'rec<th':>6} {'surp<cap':>7} {'salient':>7}")
        n_pass_r = n_pass_rec = n_pass_surp = n_salient = 0
        for a in anchors:
            r = a.r_i; rec = a.rec_i; surp = a.surprise_i
            pr = r is not None and r > thresholds.phi
            prec = rec is not None and rec < thresholds.theta
            psurp = surp is not None and surp < thresholds.surprise_cap
            n_pass_r += int(pr and a.source_id is not None)
            n_pass_rec += int(prec and a.source_id is not None)
            n_pass_surp += int(psurp and a.source_id is not None)
            n_salient += int(a.salient)
            src = a.source_id or "-"
            print(f"{a.slot_index:>4} {src:>9} {_fmt(r):>9} {_fmt(rec):>10} "
                  f"{_fmt(surp):>11} {str(pr):>6} {str(prec):>6} {str(psurp):>7} "
                  f"{str(a.salient):>7}")
        print(f"\namong provenance-bearing slots: r>phi={n_pass_r}  rec<theta={n_pass_rec}  "
              f"surp<cap={n_pass_surp}  salient={n_salient}")
        print(f"salience_retrieval_count={res.get('salience_retrieval_count')} "
              f"n_signals={len(res.get('salience_signals',[]) or [])}")
    finally:
        if store is not None:
            try: store.close()
            except Exception: pass
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


def load_backbone_local(path):
    from src.subconscious.training.routing_training import load_backbone
    return load_backbone(path, BackboneConfig(), device="cpu")


if __name__ == "__main__":
    raise SystemExit(main())