"""STRM Phase 4 Step 4: compute the salience thresholds sidecar.

The three salience gate thresholds -- ``theta`` (recoverability), ``phi``
(relevance), ``surprise_cap`` (surprise) -- are percentiles on the 2b / 2a / 2c
val-score distributions, NOT magic numbers (plan de-wonk note #5). This script
re-derives the per-sample val scores each head produces on its OWN val split
(the same split it was trained/evaluated on) and writes
``data/training/strm_salience/thresholds.json`` with the three thresholds +
their percentile basis + the val-sample counts.

Re-derivation (not a re-fit): the 2b / 2c heads are closed-form ridge and the
2a head is frozen -- we load the trained checkpoints and run them over their
val splits. This avoids modifying the committed trainers and keeps the
thresholds consistent with the shipped heads.

Percentile defaults (the operating point the salience AND is tuned to):
  theta_p            = 30   (bottom-30% recoverability = most-forgotten -> salient)
  phi_p              = 70   (top-30% relevance -> salient)
  surprise_cap_p     = 80   (only top-20% surprising turns SUPPRESS salience)

The AND of these is deliberately selective: a proactive recall should fire
rarely (only for a forgotten-AND-relevant anchor on a non-novel turn). The
deferred Step 7 eval is what decides whether to flip ``--strm-salience`` default
on; if it NO-GOs, these percentiles are the first knob to tune.

Run:  python scripts/compute_salience_thresholds.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.latent_dynamics_head import load_latent_dynamics_head
from src.subconscious.recoverability_head import load_recoverability_head
from src.subconscious.relevance_head import load_relevance_head
from src.subconscious.salience import percentile_threshold
from src.subconscious.training.strm_traces import (
    load_traces,
    sample_recoverability_pairs,
    sample_transitions,
    split_chains,
    state_rep_last,
)


# 2b/2c share the Phase 0a SSM-state traces; 2a has its own ERAG-Bench traces.
DEFAULT_RECOVERY_TRACES = "data/probe/recoverability/traces.pt"
DEFAULT_RELEVANCE_TRACES = "data/training/strm_relevance/traces.pt"
DEFAULT_REC_CKPT = "data/training/strm_recoverability/best.pt"
DEFAULT_LD_CKPT = "data/training/strm_latent_dynamics/best.pt"
DEFAULT_REL_CKPT = "data/training/strm_relevance/best.pt"
DEFAULT_OUT = "data/training/strm_salience/thresholds.json"


def _split_queries(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Replicate ``relevance_training._split_queries`` exactly (same seed/shuffle)
    so the 2a percentile is over the SAME val queries the head was evaluated on."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    if n_val >= n:
        n_val = max(1, n // 5)
    val = sorted(idx[:n_val])
    train = sorted(idx[n_val:])
    return train, val


def recoverability_val_scores(traces_path: str, ckpt_path: str) -> np.ndarray:
    """2b: per-anchor RECOVERABILITY (negated forgetting) on the 2b val split.

    Reuses the trained head's ``predict(state_pooled, anchor)`` over the val
    (state_t, anchor u_i) pairs. Returns a 1-D array of recoverability scores
    (low = forgotten)."""
    traces = load_traces(traces_path)
    train_idx, val_idx = split_chains(len(traces), val_fraction=0.2, seed=0)
    va = [traces[i] for i in val_idx]
    _S, U, _K = sample_recoverability_pairs(va, k_max=8, state_rep="pooled")
    # S is the pooled state [N,1536]; U is the anchor input embedding [N,384].
    head = load_recoverability_head(ckpt_path, device="cpu")
    head.eval()
    with torch.no_grad():
        S_t = torch.from_numpy(_S).to(torch.float32) if not torch.is_tensor(_S) else _S.to(torch.float32)
        U_t = torch.from_numpy(U).to(torch.float32) if not torch.is_tensor(U) else U.to(torch.float32)
        forgetting = head.predict(S_t, U_t).squeeze(-1).detach().cpu().numpy()
    return -forgetting  # recoverability (low = forgotten)


def latent_dynamics_val_scores(traces_path: str, ckpt_path: str) -> np.ndarray:
    """2c: per-transition surprise (L2 residual) on the 2c val split.

    Reuses the trained head's ``surprise(z_t, z_{t+1})`` over the val
    consecutive-transition pairs (already projected to the last rep, 384-d)."""
    traces = load_traces(traces_path)
    train_idx, val_idx = split_chains(len(traces), val_fraction=0.2, seed=0)
    va = [traces[i] for i in val_idx]
    z_va = [state_rep_last(tr["states"]) for tr in va]
    Zt, Ztp1 = sample_transitions(z_va)
    head = load_latent_dynamics_head(ckpt_path, device="cpu")
    head.eval()
    with torch.no_grad():
        Zt_t = torch.from_numpy(Zt).to(torch.float32)
        Ztp1_t = torch.from_numpy(Ztp1).to(torch.float32)
        surp = head.surprise(Zt_t, Ztp1_t).detach().cpu().numpy()
    return surp


def relevance_val_scores(traces_path: str, ckpt_path: str) -> np.ndarray:
    """2a: per-slot r_i on the 2a val split (the SAME queries the head was
    evaluated on). Collects every val slot's r_i into one distribution."""
    traces = torch.load(traces_path, weights_only=False)
    _train, val = _split_queries(len(traces), val_fraction=0.2, seed=0)
    head = load_relevance_head(ckpt_path, device="cpu")
    head.eval()
    ris = []
    with torch.no_grad():
        for i in val:
            tr = traces[i]
            sy = tr["slots_y"].to(torch.float32)            # [K, 256]
            sd = tr["slots_doc_emb"].to(torch.float32)      # [K, 384]
            q = tr["query_emb"].to(torch.float32)           # [384]
            r = head.predict(sy, sd, q).squeeze(-1).detach().cpu().numpy()  # [K]
            ris.append(r)
    return np.concatenate(ris)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recovery-traces", default=DEFAULT_RECOVERY_TRACES)
    ap.add_argument("--relevance-traces", default=DEFAULT_RELEVANCE_TRACES)
    ap.add_argument("--rec-ckpt", default=DEFAULT_REC_CKPT)
    ap.add_argument("--ld-ckpt", default=DEFAULT_LD_CKPT)
    ap.add_argument("--rel-ckpt", default=DEFAULT_REL_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--theta-p", type=float, default=30.0)
    ap.add_argument("--phi-p", type=float, default=70.0)
    ap.add_argument("--surprise-cap-p", type=float, default=80.0)
    args = ap.parse_args()

    for p in (args.rec_ckpt, args.ld_ckpt, args.rel_ckpt):
        if not Path(p).exists():
            raise SystemExit(f"missing trained checkpoint: {p} (run the train_*_head.py CLIs first)")

    print("[salience thresholds] re-deriving per-sample val scores from trained heads...")
    rec = recoverability_val_scores(args.recovery_traces, args.rec_ckpt)
    surp = latent_dynamics_val_scores(args.recovery_traces, args.ld_ckpt)
    rel = relevance_val_scores(args.relevance_traces, args.rel_ckpt)
    print(f"  2b recoverability: n={len(rec)}  range=[{rec.min():.4f}, {rec.max():.4f}]")
    print(f"  2c surprise:       n={len(surp)}  range=[{surp.min():.4f}, {surp.max():.4f}]")
    print(f"  2a relevance:      n={len(rel)}   range=[{rel.min():.4f}, {rel.max():.4f}]")

    theta = percentile_threshold(rec, args.theta_p)
    phi = percentile_threshold(rel, args.phi_p)
    surprise_cap = percentile_threshold(surp, args.surprise_cap_p)
    print(f"  theta           = {theta:.6f}  (p={args.theta_p}, recoverability; low=forgotten=salient)")
    print(f"  phi             = {phi:.6f}  (p={args.phi_p}, relevance; high=relevant=salient)")
    print(f"  surprise_cap    = {surprise_cap:.6f}  (p={args.surprise_cap_p}, surprise; high=suppress)")

    basis = (
        "theta = p{tp:.0f} of 2b val recoverability (negated forgetting, "
        "n={nrec}); phi = p{pp:.0f} of 2a val r_i (n={nrel}); surprise_cap = "
        "p{sp:.0f} of 2c val surprise (n={nsurp}). The salience AND "
        "(rec_i<theta)&(r_i>phi)&(surprise_i<surprise_cap) is deliberately "
        "selective; tune these percentiles first if Step 7 NO-GOs."
    ).format(tp=args.theta_p, pp=args.phi_p, sp=args.surprise_cap_p,
             nrec=len(rec), nrel=len(rel), nsurp=len(surp))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "theta": theta,
        "phi": phi,
        "surprise_cap": surprise_cap,
        "theta_percentile": args.theta_p,
        "phi_percentile": args.phi_p,
        "surprise_cap_percentile": args.surprise_cap_p,
        "basis": basis,
        "n_recoverability": int(len(rec)),
        "n_relevance": int(len(rel)),
        "n_latent_dynamics": int(len(surp)),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())