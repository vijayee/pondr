"""Train the ZRelevanceHead (STRM Phase B ``h_t`` probe) -- per-slot BCE on ERAG.

Thin CLI over
``src.subconscious.training.relevance_training.fit_relevance`` -- the SAME
trainer the 2a ``RelevanceHead`` uses, with two swaps:

  1. ``slot_signal_field="slots_z"`` -- the head's per-slot input is ``z_i =
     LatentDynamicsHead.project(slot.h)`` (the projected SSM recurrent state,
     emitted by ``scripts/generate_relevance_data.py`` as ``slots_z``), NOT the
     2a ``slots_doc_emb`` (raw bge). This isolates the ONE variable the Phase B
     gate tests: does the SSM state carry query-relevance signal the ``y_t``
     readout did NOT?
  2. ``head=ZRelevanceHead()`` -- a dual-projection bilinear over ``(z_i,
     query)`` with NO ``yt_sidepath`` (the pure-``z_i`` test drops the ``y_t``
     path). It is signature-compatible with the 2a trainer (``slot_y`` accepted
     and ignored), so ``fit_relevance`` is reused verbatim.

CHEAP GATE 1 (this script is the train half; the serve half is
``scripts/probe_strm_selectivity_real.py --z-relevance-head``):

  * TRAIN gate: ``mean_top3_recall >= 0.6`` + Wilson CI on the ERAG val split
    (same as 2a). FAIL -> ``z_i`` carries NO relevance signal even on the
    training distribution -> NO-GO, no serve probe needed (the frozen
    routing-trained backbone's recurrent state does not encode query-relevance).
  * SERVE gate (Probe 4a harness): the z_i-head's selectivity gap median
    >= 0.2 in >= 3/4 runs on real Onyx transcripts vs 2a's ~0 on the same slots.
    GO -> ``h_t`` carries serve-relevant signal -> proceed to the transformer
    rewire (Phases C-F). NO-GO -> stop, do not build the transformer.

Generate the traces first (they must carry ``slots_z`` -- the generator emits it
whenever the ring is ON, which it is by default):

    python scripts/generate_relevance_data.py --max-queries 80

Usage:
    python scripts/train_z_relevance_head.py
    python scripts/train_z_relevance_head.py --traces data/training/strm_relevance/traces.pt \\
        --output data/training/strm_z_relevance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    fit_relevance,
    load_relevance_traces,
)
from src.subconscious.z_relevance_head import ZRelevanceHead  # noqa: E402

DEFAULT_OUTPUT = "data/training/strm_z_relevance"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train the STRM ZRelevanceHead (Phase B h_t probe; per-slot BCE, ERAG traces)")
    p.add_argument("--traces",
                   default=RelevanceTrainingConfig().traces_path,
                   help="Phase 2a trace file (must carry slots_z -- regenerate if stale)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Checkpoint output dir (best.pt + final.pt + train_log.json)")
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
              f"  python scripts/generate_relevance_data.py --max-queries 80 "
              f"--output {traces_path}", file=sys.stderr)
        return 1

    print(f"Loading relevance traces (slot_signal_field=slots_z) -> {traces_path}",
          flush=True)
    try:
        traces = load_relevance_traces(str(traces_path), slot_signal_field="slots_z")
    except RuntimeError as e:
        print(f"ERROR: {e}\n  Regenerate with scripts/generate_relevance_data.py "
              f"--retrace (Phase B emits slots_z).", file=sys.stderr)
        return 1
    n = len(traces)
    if n < 3:
        print(f"ERROR: only {n} usable queries -- need >=3 to split + fit. "
              f"Regenerate with --max-queries 80.", file=sys.stderr)
        return 1
    ks = sorted(rec["slots_z"].shape[0] for rec in traces)
    print(f"  {n} queries (K min/med/max={ks[0]}/{ks[n // 2]}/{ks[-1]})", flush=True)

    cfg = RelevanceTrainingConfig(
        traces_path=str(traces_path), epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, accum_steps=args.accum_steps,
        pos_weight_cap=args.pos_weight_cap, gate_top3=args.gate_top3,
        gate_wilson_low=args.gate_wilson_low, val_fraction=args.val_fraction,
        seed=args.seed, checkpoint_dir=args.output,
        slot_signal_field="slots_z",
    )
    head = ZRelevanceHead()
    print(f"Training ZRelevanceHead (per-slot BCE on slots_z, {cfg.epochs} epochs, "
          f"lr={cfg.learning_rate}, accum={cfg.accum_steps}, "
          f"val_fraction={cfg.val_fraction})", flush=True)
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
    print(f"  TRAIN gate {'PASS' if result['go'] else 'FAIL'}: z_i carries "
          f"{'SOME' if result['go'] else 'NO'} train-time relevance signal.",
          flush=True)
    print(f"  Next (serve half of GATE 1): "
          f"python scripts/probe_strm_selectivity_real.py "
          f"--z-relevance-head {best}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())