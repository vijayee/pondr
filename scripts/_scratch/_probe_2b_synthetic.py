"""SCRATCH (never committed): probe the 2b synthetic trace + fit numbers.

Builds synthetic traces whose pooled last-layer state is a leaky sum of past
inputs (c=0.7), runs fit_recoverability, and prints the AUCs so the test
assertions can be set from real numbers (not guessed -- avoids the 2c
common-mode false-NO-GO trap).
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.subconscious.recoverability_head import STATE_DIM_POOLED, ANCHOR_DIM
from src.subconscious.training.recoverability_training import (
    RecoverabilityTrainingConfig, fit_recoverability,
)


def _synthetic(n_chains=20, length=30, active=32, c=0.7, n_anchors=5, seed=0):
    """Pooled last layer = leaky sum of SPARSE anchors; other 3 layers zero.

    Most u_t are zero; a few random anchors are nonzero. state_last(t) =
    sum_j c^{t-j} u_j, so each anchor u_i is encoded with weight c^k (k=t-i)
    PLUS interference from later anchors between i and t. The decoder D,
    fit across all pairs, recovers u_i with error that grows with k AND with
    interference -- and interference varies WITHIN k (depends on whether a
    later anchor landed close to t), which P reads from state_t but k alone
    cannot. That is the signal P must beat the k-baseline on.
    """
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(n_chains):
        us = [np.zeros(ANCHOR_DIM, dtype=np.float32) for _ in range(length)]
        anchor_pos = sorted(rng.choice(length, size=min(n_anchors, length),
                                      replace=False).tolist())
        for p in anchor_pos:
            u = np.zeros(ANCHOR_DIM, dtype=np.float32)
            # NON-NEGATIVE inputs so ||u_i||^2 is rankable by a linear functional
            # (a linear P with positive weights on u_i ranks ||u_i||^2 monotonically).
            u[:active] = rng.uniform(0.0, 1.0, active).astype(np.float32)
            us[p] = u
        state_last = np.zeros(ANCHOR_DIM, dtype=np.float32)
        states, inputs = [], []
        for t in range(length):
            state_last = c * state_last + us[t]
            layer = np.zeros((4, 16, ANCHOR_DIM), dtype=np.float32)
            layer[-1] = np.tile(state_last, (16, 1))
            states.append(layer)
            inputs.append(us[t].copy())
        states_t = torch.from_numpy(np.stack(states))
        inputs_t = torch.from_numpy(np.stack(inputs))
        traces.append({"inputs": inputs_t, "states": states_t})
    return traces


if __name__ == "__main__":
    traces = _synthetic(n_chains=20, length=30, active=32, c=0.7, n_anchors=5, seed=0)
    cfg = RecoverabilityTrainingConfig(
        k_max=6, lam=10.0, gate_auc=0.75, val_fraction=0.2,
        seed=0, checkpoint_dir="data/probe/strm_2b_scratch",
    )
    r = fit_recoverability(traces, cfg)
    print(f"\nRESULT: ridge_auc={r['ridge_auc']:.4f} k_auc={r['k_auc']:.4f} "
          f"go={r['go']}")
    print(f"decay: {r['decay']}")
    import json
    log = json.load(open("data/probe/strm_2b_scratch/train_log.json"))
    print(f"train AUC={log['ridge_auc_train']:.4f}  val_pos_frac={log['val_pos_frac']:.3f}")