"""Sweep: aging depth x {shipped thresholds vs surprise disabled}.

Shows whether disabling the surprise term (surprise_cap=+inf) alone unblocks
salience firing at realistic long-horizon aging (the fact must age enough for
rec_i < theta to pass). UNTRACKED scratch.
"""
from __future__ import annotations

import sys
import tempfile, shutil
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from src.subconscious.training.routing_training import build_embedder, load_backbone
from scripts.eval_strm_context_coverage import (
    FACT_BANK, FILLER_BANK, _StubPlanner, _StubModeA,
    _seed_corpus, _seed_ring, _reset_for_trial,
)

REC_CKPT = Path("data/training/strm_recoverability/best.pt")
LD_CKPT = Path("data/training/strm_latent_dynamics/best.pt")
REL_CKPT = Path("data/training/strm_relevance/best.pt")
THRESHOLDS = Path("data/training/strm_salience/thresholds.json")
BACKBONE = Path(DEFAULT_BACKBONE_PATH)


def _fact_anchor(orch):
    for a in (orch._salience_anchors or []):
        if a.source_id == "fact_00":
            return a
    return None


def main() -> int:
    base = load_salience_thresholds(str(THRESHOLDS))
    no_surprise = replace(base, surprise_cap=1e18)  # disable surprise term
    print(f"shipped: theta={base.theta:+.4f} phi={base.phi:+.4f} "
          f"surprise_cap={base.surprise_cap:.3e}")
    print(f"no_surp: theta={no_surprise.theta:+.4f} phi={no_surprise.phi:+.4f} "
          f"surprise_cap=+inf")
    print()

    tmpdir = tempfile.mkdtemp(prefix="pondr_sweep_")
    store = None
    try:
        store = HippocampalStore(str(Path(tmpdir) / "db"))
        embedder = build_embedder("on-demand")
        rec = load_recoverability_head(str(REC_CKPT), device="cpu")
        ld = load_latent_dynamics_head(str(LD_CKPT), device="cpu")
        rel = load_relevance_head(str(REL_CKPT), device="cpu")
        retriever = HippocampalRetriever(
            store, planner=_StubPlanner(), auto_load_index=True, embedder=embedder)
        facts = list(FACT_BANK[:6])
        _seed_corpus(store, embedder, facts)
        cfg = Phase2cConfig()
        cfg.session.state_dir = str(Path(tmpdir) / "sessions")
        backbone = load_backbone(str(BACKBONE), BackboneConfig(), device="cpu")

        fact_summary, query = facts[0]
        filler_pool = list(FILLER_BANK)

        def run(n_fill, ring_cap, thr, label):
            orch = PonderOrchestrator(
                store=store, retriever=retriever, backbone=backbone, embedder=embedder,
                mode_a=_StubModeA(), config=cfg, user_id="probe", ring_capacity=ring_cap,
                recoverability_head=rec, latent_dynamics_head=ld, relevance_head=rel,
                strm_salience=True, salience_thresholds=thr,
            )
            _reset_for_trial(orch)
            fillers = [filler_pool[j % len(filler_pool)] for j in range(n_fill)]
            _seed_ring(orch, "fact_00", fact_summary, fillers)
            res = orch.query(query)
            a = _fact_anchor(orch)
            if a is None:
                print(f"  {label:18s} n_fill={n_fill:2d} ring={ring_cap:2d} "
                      f"-> fact_00 EVICTED (no anchor)")
                return
            pr = a.r_i is not None and a.r_i > thr.phi
            prec = a.rec_i is not None and a.rec_i < thr.theta
            psurp = a.surprise_i is not None and a.surprise_i < thr.surprise_cap
            print(f"  {label:18s} n_fill={n_fill:2d} ring={ring_cap:2d} "
                  f"r={a.r_i:+.3f} rec={a.rec_i:+.4f} surp={a.surprise_i:+.5f} "
                  f"| r>phi={pr!s:5} rec<th={prec!s:5} surp<cap={psurp!s:5} "
                  f"salient={a.salient}")

        print("fact_00 probe, varying aging (fillers) + ring cap:")
        for n_fill, ring_cap in [(10, 16), (20, 40), (30, 40), (38, 40)]:
            run(n_fill, ring_cap, base, "shipped")
            run(n_fill, ring_cap, no_surprise, "no_surprise")
            print()
    finally:
        if store is not None:
            try: store.close()
            except Exception: pass
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())