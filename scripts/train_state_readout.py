"""Train the Phase 0b learned StateReadout + ZRelevanceHead (STRM composite).

Thin CLI over
``src.subconscious.training.relevance_training.fit_relevance`` -- the SAME
trainer the 2a ``RelevanceHead`` and the Phase B ``ZRelevanceHead`` use, with
two swaps:

  1. ``slot_signal_field="slots_h_raw"`` -- the head's per-slot input is the RAW
     flattened SSM recurrent state (emitted by ``scripts/generate_relevance_data.py
     --emit-raw-state`` as ``slots_h_raw`` [K, 6144] or [K, 24576]), NOT the
     fixed mean-pool ``slots_z``. Phase 0a (``scripts/probe_state_signal_distribution.py``,
     [[pondr-strm-phase0a-state-signal-readout]]) showed the mean-pool cancels
     opposing-sign channel signal to near-constant while the flattened state
     varies 0.45-0.76x -- the signal is in the state, killed by the mean-pool.
     This tests whether a LEARNED readout recovers it.
  2. ``head=CompositeZHead(dim_in, hidden)`` -- ``StateReadout -> ZRelevanceHead``
     (``src/subconscious/state_readout.py``). The readout maps the flattened
     state to a 384-d ``z_i`` the existing z-head scores against the query; the
     two train end-to-end. ``fit_relevance`` is reused verbatim (the composite
     exposes the ``slot_dim``/``doc_dim``/``query_dim``/``proj_dim`` attrs the
     checkpoint reads, with ``doc_dim = dim_in`` so the loader rebuilds the
     readout).

GATE 0b (the TRAIN half of the Phase B gate, re-run on the learned readout):

  * ``mean_top3_recall >= 0.6`` AND ``hit_ci95[0] > 0.5`` (same as 2a / the
    Phase B z-head). The mean-pool z-head hit **0.285** (== random) on these
    labels; 2a hit **0.889**. GO -> a learned readout recovers the signal the
    mean-pool destroyed -> re-run the SERVE gate
    (``scripts/probe_strm_selectivity_real.py``) as acceptance. NO-GO -> even a
    learned readout can't align the state to the query -> fall through to Phase 1
    (warm-started backbone fine-tune).

Generate the raw-state traces first (``--emit-raw-state`` is OFF by default so
the default traces lack ``slots_h_raw`` -- regenerate):

    python scripts/generate_relevance_data.py --retrace --emit-raw-state
    # add --raw-state-rep flat_all for the 24576-d variant

Usage:
    python scripts/train_state_readout.py
    python scripts/train_state_readout.py --traces data/training/strm_relevance/traces.pt \\
        --output data/training/strm_state_readout --hidden 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.state_readout import DEFAULT_DIM_IN, CompositeZHead  # noqa: E402
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    fit_relevance,
    load_relevance_traces,
)

DEFAULT_OUTPUT = "data/training/strm_state_readout"
SLOT_SIGNAL_FIELD = "slots_h_raw"


def _infer_dim_in(traces: list[dict]) -> int:
    """Read the raw-state width from the first trace's ``slots_h_raw``.

    The generator emits [K, 6144] (``flat_last``) or [K, 24576] (``flat_all``);
    the composite's ``StateReadout`` must be built with the matching ``dim_in``.
    """
    for rec in traces:
        h_raw = rec.get(SLOT_SIGNAL_FIELD)
        if h_raw is not None and hasattr(h_raw, "shape") and h_raw.dim() >= 2:
            return int(h_raw.shape[1])
    return DEFAULT_DIM_IN


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the STRM Phase 0b learned StateReadout + ZRelevanceHead "
                    "(per-slot BCE on raw flattened SSM state, ERAG traces)")
    p.add_argument("--traces",
                   default=RelevanceTrainingConfig().traces_path,
                   help="Phase 2a trace file (must carry slots_h_raw -- regenerate "
                        "with scripts/generate_relevance_data.py --emit-raw-state)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
    p.add_argument("--hidden", type=int, default=None,
                   help="MLP hidden dim for the StateReadout (default: None -> a "
                        "single Linear, the 'linear readout suffices' test; set e.g. "
                        "1024 for a 2-layer MLP if the linear form fails the gate)")
    p.add_argument("--epochs", type=int,
                   default=RelevanceTrainingConfig().epochs)
    p.add_argument("--lr", type=float,
                   default=RelevanceTrainingConfig().learning_rate)
    p.add_argument("--weight-decay", type=float,
                   default=RelevanceTrainingConfig().weight_decay)
    p.add_argument("--accum-steps", type=int,
                   default=RelevanceTrainingConfig().accum_steps)
    p.add_argument("--pos-weight-cap", type=float,
                   default=RelevanceTrainingConfig().pos_weight_cap)
    p.add_argument("--gate-top3", type=float,
                   default=RelevanceTrainingConfig().gate_top3,
                   help="TRAIN gate: mean per-query top-3 recall must clear this")
    p.add_argument("--gate-wilson-low", type=float,
                   default=RelevanceTrainingConfig().gate_wilson_low,
                   help="TRAIN gate: Wilson 95%% CI lower bound on the per-query "
                        "full-recall hit rate must clear this")
    p.add_argument("--val-fraction", type=float,
                   default=RelevanceTrainingConfig().val_fraction)
    p.add_argument("--seed", type=int,
                   default=RelevanceTrainingConfig().seed)
    args = p.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: traces not found at {traces_path}. Generate them first:\n"
              f"  python scripts/generate_relevance_data.py --retrace "
              f"--emit-raw-state --output {traces_path}", file=sys.stderr)
        return 1

    print(f"Loading relevance traces (slot_signal_field={SLOT_SIGNAL_FIELD}) "
          f"-> {traces_path}", flush=True)
    try:
        traces = load_relevance_traces(str(traces_path),
                                       slot_signal_field=SLOT_SIGNAL_FIELD)
    except RuntimeError as e:
        print(f"ERROR: {e}\n  Regenerate with scripts/generate_relevance_data.py "
              f"--retrace --emit-raw-state.", file=sys.stderr)
        return 1
    n = len(traces)
    if n < 3:
        print(f"ERROR: only {n} usable queries -- need >=3 to split + fit. "
              f"Regenerate with --max-queries 80.", file=sys.stderr)
        return 1
    dim_in = _infer_dim_in(traces)
    ks = sorted(rec[SLOT_SIGNAL_FIELD].shape[0] for rec in traces)
    print(f"  {n} queries (K min/med/max={ks[0]}/{ks[n // 2]}/{ks[-1]}), "
          f"dim_in={dim_in}, hidden={args.hidden}", flush=True)

    cfg = RelevanceTrainingConfig(
        traces_path=str(traces_path), epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, accum_steps=args.accum_steps,
        pos_weight_cap=args.pos_weight_cap, gate_top3=args.gate_top3,
        gate_wilson_low=args.gate_wilson_low, val_fraction=args.val_fraction,
        seed=args.seed, checkpoint_dir=args.output,
        slot_signal_field=SLOT_SIGNAL_FIELD,
    )
    head = CompositeZHead(dim_in=dim_in, hidden=args.hidden)
    n_params = sum(p.numel() for p in head.parameters())
    arch = "MLP" if args.hidden else "Linear"
    print(f"Training CompositeZHead ({arch} readout {dim_in}->384, "
          f"{n_params:,} params, {cfg.epochs} epochs, lr={cfg.learning_rate}, "
          f"accum={cfg.accum_steps}, val_fraction={cfg.val_fraction})", flush=True)
    result = fit_relevance(traces, cfg, head=head)

    best = Path(cfg.checkpoint_dir) / "best.pt"
    if not best.exists():
        print(f"ERROR: head checkpoint not written at {best}", file=sys.stderr)
        return 1
    print(f"DONE. best epoch={result['best_epoch']} "
          f"top3={result['best_pc']['mean_top3_recall']:.4f} "
          f"hit={result['best_pc']['hit_rate']:.2f} "
          f"-> {'GO' if result['go'] else 'NO-GO'}", flush=True)
    print(f"  best.pt at {best}", flush=True)
    print(f"  TRAIN gate {'PASS' if result['go'] else 'FAIL'}: a learned {arch} "
          f"readout {'RECOVERS' if result['go'] else 'does NOT recover'} the "
          f"state signal the mean-pool destroyed (z-head baseline 0.285, 2a 0.889).",
          flush=True)
    if result["go"]:
        print(f"  Next (serve half -- acceptance, task #33): re-run "
              f"scripts/probe_strm_selectivity_real.py with the composite head "
              f"({best}) -- the --state-readout-head flag is wired in that task.",
              flush=True)
    else:
        print(f"  Next (fall-through): Phase 1 warm-started backbone fine-tune "
              f"(see plans/reflective-greeting-beacon.md).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())