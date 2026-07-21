"""Task #39: WHERE is the SERVE state-signal loss -- the mean-pool readout, or
the backbone's persistent-serve state? The fork diagnostic for task #38.

Task #38 ([[pondr-strm-task38-serve-retrain-weak-signal]]) retrained the z-head
on SERVE-distribution traces and got TRAIN-on-serve NO-GO (top3 0.522, plateau
0.478 = 2.3x random) + SERVE z_r gap +0.0373 (improved from -0.0066 but far below
the 0.2 gate). Diagnosis: the bottleneck is the backbone's persistent-serve-state
``z_i`` (OOD vs the candidate-sequence states the backbone was trained on), NOT
the head -- the head can only read what ``z_i`` carries. Two open levers, this
probe forks between them:

  A. MEAN-POOL KILLS SERVE SIGNAL (the [[pondr-strm-phase0a-state-signal-readout]]
     finding, re-tested on the NEW backbone's SERVE state). ``z_i =
     LatentDynamicsHead.project(slot.h)`` means over the 16 ``d_state`` channels
     of the last layer; if those channels carry opposing-sign signal at serve,
     the mean cancels while a richer rep (``z_flat_last`` [16,384] or
     ``z_flat_all`` [4,16,384]) varies. -> the STATE-TRAJECTORY TRANSFORMER (the
     ORIGINAL STRM vision [[pondr-strm-transformer-relocator-drift]] -- attend
     over the full per-layer state sequence, not the mean-pool) is the right
     lever, NOT a backbone retrain.

  B. PERSISTENT-STATE DEGENERATE: if ALL representations (mean/flat_last/flat_all)
     are weak at serve -- the state barely varies across the recalled ring slots
     -- the backbone's persistent-serve state itself is degenerate. -> BACKBONE-
     ON-SERVE RETRAIN (the Phase 1 trainer on serve-distribution persistent-state
     traces, not ERAG candidate traces) is the lever.

This probe is OFFLINE: it reads the serve traces emitted by
``probe_strm_selectivity_real.py --emit-traces --emit-raw-state`` (which carry
``slots_h_raw`` [K,4,16,384] fp16 + ``slots_doc_emb`` [K,384] + ``query_emb``
[384] + top-1-cos ``labels`` [K] per turn -- the EXACT serve distribution the
SERVE gate scores). No backbone, no WorkingMemory, no embedder -- just the
captured state. It materializes four representations per turn and measures, per
representation:

  - across-slot std (std over the K recalled slots, mean over D dims; median over
    turns), normalized by the doc (slot-text-bge) across-slot std -> the VARIANCE
    test. This is the DECISIVE metric for the fork.
  - bge(query, rep) top-3 recall (cosine vs the 384-d query) where dimensionally
    computable (rep dim == 384): the doc baseline (ceiling -- the label is
    top-1-cos, so a cos-ranker trivially ranks gold #1), the mean-pool, and EACH
    of the 16 last-layer channels individually. Secondary color.

FORK DECISION (the metric that picks the next lever):
  - If ``z_flat_last`` / ``z_flat_all`` / max-channel std ratio >> ``z_mean_last``
    std ratio (the mean-pool is much flatter than the richer reps) -> FORK A:
    mean-pool kills serve signal -> state-trajectory Transformer.
  - If ALL std ratios are low (mean-pool ≈ flat ≈ channel, all near the doc
    baseline or below the GO threshold) -> FORK B: persistent state degenerate
    -> backbone-on-serve retrain.

Reuses ``_cos_top3_recall`` / ``_mean_top3`` / ``_median`` (mirrored from
``probe_state_signal_distribution.py``) + ``_split_queries`` / ``_wilson_ci95``
from the relevance training path. No new model code, no training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    _split_queries,
    _wilson_ci95,
)


# The serve traces carry slot.h as fp16 [K,4,16,384]; the last SSM layer is the
# 4th (index -1). ld_head.project = mean over d_state (16) of the last layer.
LAST_LAYER = -1
D_STATE = 16
D_MODEL = 384
N_LAYERS = 4

# The Phase 0a GO threshold for "the state varies across slots" (materially above
# the 0.068x mean-pool collapse floor). Reused here as the FORK B trigger: if NO
# rep clears it, the persistent-serve state is degenerate.
GO_STD_RATIO_THRESH = 0.15


def _cos_top3_recall(qvec, doc_vecs, labels):
    """qvec [D], doc_vecs [K,D], labels [K] (1=gold) -> (recall, hit) or None."""
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    sims = d @ q
    gold_idx = [i for i, l in enumerate(labels) if l == 1]
    n_gold = len(gold_idx)
    if n_gold == 0:
        return None
    k_top = min(3, len(sims))
    top = set(np.argsort(-sims)[:k_top].tolist())
    n_in = sum(1 for i in gold_idx if i in top)
    return n_in / n_gold, n_in == n_gold


def _mean_top3(records, rep_key):
    """Mean top-3 recall + Wilson CI over the val split for one representation.

    ``rep_key`` names a per-slot np.ndarray field on each record; the field's last
    axis must match the query dim (384) for a cosine to be defined."""
    cfg = RelevanceTrainingConfig()
    train_idx, val_idx = _split_queries(len(records), cfg.val_fraction, cfg.seed)
    val = [records[i] for i in val_idx]
    recalls, hits = [], 0
    for rec in val:
        r = _cos_top3_recall(rec["query"], rec[rep_key], rec["labels"])
        if r is None:
            continue
        recalls.append(r[0])
        if r[1]:
            hits += 1
    if not recalls:
        return None
    mean_top3 = sum(recalls) / len(recalls)
    hit_rate = hits / len(recalls)
    ci = _wilson_ci95(hit_rate, len(recalls))
    return {"mean_top3": mean_top3, "hit_rate": hit_rate, "ci": ci, "n": len(recalls)}


def _median(xs):
    if not xs:
        return float("nan")
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def _materialize(h_raw: np.ndarray) -> dict:
    """slots_h_raw [K,4,16,384] fp16 -> the four representations (numpy fp32).

    ``z_mean_last`` reproduces ``LatentDynamicsHead.project`` (mean over d_state
    of the last layer); ``z_chan_last`` is the raw last-layer per-channel state;
    ``z_flat_last`` / ``z_flat_all`` are its flattenings."""
    K = h_raw.shape[0]
    last = h_raw[:, LAST_LAYER, :, :].astype(np.float32)        # [K,16,384]
    z_mean = last.mean(axis=1)                                  # [K,384]
    z_flat_last = last.reshape(K, D_STATE * D_MODEL)            # [K,6144]
    z_flat_all = h_raw.astype(np.float32).reshape(K, N_LAYERS * D_STATE * D_MODEL)
    return {"z_mean": z_mean, "z_chan": last, "z_flat_last": z_flat_last,
            "z_flat_all": z_flat_all}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task #39: serve-state representation probe -- fork between "
                    "mean-pool-kills-signal (-> Transformer) and persistent-state-"
                    "degenerate (-> backbone-on-serve retrain). Offline.")
    p.add_argument("--traces", required=True,
                   help="serve traces emitted by probe_strm_selectivity_real.py "
                        "with --emit-traces --emit-raw-state (must carry "
                        "slots_h_raw).")
    p.add_argument("--out", default="", help="write the JSON report to this path")
    args = p.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: traces not found at {traces_path}", file=sys.stderr)
        return 1

    print(f"Loading serve traces from {traces_path}", flush=True)
    trace_records = torch.load(traces_path, weights_only=False)
    print(f"  {len(trace_records)} raw trace records", flush=True)

    # Keep only records that carry slots_h_raw (i.e. were emitted with
    # --emit-raw-state). A record without it is a Phase A state-capture miss
    # (slot.h was None for too many slots) -- skip, do not crash.
    def _to_np(t, dtype):
        return t.detach().cpu().numpy().astype(dtype) if isinstance(t, torch.Tensor) \
            else np.asarray(t, dtype=dtype)

    records: list[dict] = []
    skipped = 0
    for rec in trace_records:
        h = rec.get("slots_h_raw")
        if h is None:
            skipped += 1
            continue
        reps = _materialize(_to_np(h, np.float32))
        records.append({
            "query": _to_np(rec["query_emb"], np.float32),
            "labels": _to_np(rec["labels"], np.int64),
            "doc": _to_np(rec["slots_doc_emb"], np.float32),
            **reps,
        })
    if not records:
        print("ERROR: no records carry slots_h_raw -- re-emit with "
              "--emit-raw-state.", file=sys.stderr)
        return 1
    print(f"  {len(records)} records carry slots_h_raw ({skipped} skipped: no "
          f"raw state)", flush=True)

    # ── Analysis: across-slot std per representation ──
    doc_stds = [float(r["doc"].std(axis=0).mean()) for r in records]
    doc_std_med = _median(doc_stds)
    zmean_stds = [float(r["z_mean"].std(axis=0).mean()) for r in records]
    zmean_std_med = _median(zmean_stds)
    zflat_last_stds = [float(r["z_flat_last"].std(axis=0).mean()) for r in records]
    zflat_last_std_med = _median(zflat_last_stds)
    zflat_all_stds = [float(r["z_flat_all"].std(axis=0).mean()) for r in records]
    zflat_all_std_med = _median(zflat_all_stds)

    # Per-channel std (z_chan [K,16,384]): std over K -> [16,384]; mean over 384
    # -> [16] per-channel std; report the MAX and MEDIAN channel (median over
    # turns of the per-turn max-channel std).
    per_q_max_chan, per_q_med_chan = [], []
    for r in records:
        chan_std = r["z_chan"].std(axis=0).mean(axis=1)  # [16]
        per_q_max_chan.append(float(chan_std.max()))
        per_q_med_chan.append(float(np.median(chan_std)))
    max_chan_std_med = _median(per_q_max_chan)
    med_chan_std_med = _median(per_q_med_chan)

    def ratio(x):
        return x / doc_std_med if doc_std_med > 0 else float("nan")

    # ── Top-3 recall (secondary; the label is top-1-cos, so doc = ceiling ~1.0) ──
    doc_top3 = _mean_top3(records, "doc")
    zmean_top3 = _mean_top3(records, "z_mean")
    chan_top3 = []
    for cc in range(D_STATE):
        chan_records = [{
            "query": r["query"], "labels": r["labels"],
            f"chan{cc}": r["z_chan"][:, cc, :],  # [K,384]
        } for r in records]
        t = _mean_top3(chan_records, f"chan{cc}")
        if t is not None:
            chan_top3.append((cc, t["mean_top3"]))
    chan_top3_sorted = sorted(chan_top3, key=lambda kv: kv[1], reverse=True)
    best_chan_idx, best_chan_top3 = chan_top3_sorted[0] if chan_top3 else (-1, float("nan"))
    median_chan_top3 = _median([t for _, t in chan_top3])

    # ── Report ──
    print("\n" + "=" * 72)
    print("TASK #39: serve-state signal distribution across representations")
    print("=" * 72)
    print(f"  turns: {len(records)}   (doc_std baseline median = {doc_std_med:.5f})")
    print(f"  (serve: recalled ring slots are topically close to the query, so the")
    print(f"   doc-std normalizer is LOWER than ERAG's -- the RATIO is what matters)")
    print()
    print("Across-slot std (median over turns; std over K slots, mean over D dims):")
    print(f"  doc         [K,384]   std = {doc_std_med:.5f}   ratio = {ratio(doc_std_med):.3f}x   (the normalizer)")
    print(f"  z_mean_last [K,384]   std = {zmean_std_med:.5f}   ratio = {ratio(zmean_std_med):.3f}x   "
          f"(the mean-pool the z-head reads)")
    print(f"  z_chan_last max-chan  std = {max_chan_std_med:.5f}   ratio = {ratio(max_chan_std_med):.3f}x   "
          f"(median channel ratio = {ratio(med_chan_std_med):.3f}x)")
    print(f"  z_flat_last [K,6144]  std = {zflat_last_std_med:.5f}   ratio = {ratio(zflat_last_std_med):.3f}x")
    print(f"  z_flat_all  [K,24576] std = {zflat_all_std_med:.5f}   ratio = {ratio(zflat_all_std_med):.3f}x")
    print()
    print("Top-3 recall (cosine vs the 384-d query, val split; label = top-1-cos):")
    if doc_top3:
        print(f"  doc baseline  top-3 = {doc_top3['mean_top3']:.3f}  "
              f"(hit {doc_top3['hit_rate']:.2f}, n={doc_top3['n']})  "
              f"(~1.0 ceiling: the label IS argmax-cos)")
    if zmean_top3:
        print(f"  z_mean_last   top-3 = {zmean_top3['mean_top3']:.3f}  "
              f"(hit {zmean_top3['hit_rate']:.2f}, ci [{zmean_top3['ci'][0]:.2f},"
              f"{zmean_top3['ci'][1]:.2f}], n={zmean_top3['n']})  "
              f"(task #38 TRAIN-on-serve plateau was ~0.478)")
    print(f"  z_chan_last   best channel (c{best_chan_idx}) top-3 = {best_chan_top3:.3f}   "
          f"(median channel top-3 = {median_chan_top3:.3f})")
    if len(chan_top3_sorted) >= 4:
        top4 = ", ".join(f"c{cc}={t:.3f}" for cc, t in chan_top3_sorted[:4])
        print(f"                top-4 channels: {top4}")
    print()

    # ── FORK decision ──
    # The decisive comparison is the mean-pool std ratio vs the richer-rep std
    # ratios, all normalized by the SAME serve doc baseline.
    r_mean = ratio(zmean_std_med)
    r_chan = ratio(max_chan_std_med)
    r_flat_last = ratio(zflat_last_std_med)
    r_flat_all = ratio(zflat_all_std_med)

    # FORK A: a richer rep carries materially more across-slot variance than the
    # mean-pool (the mean-pool cancelled it). Require the richer rep to BOTH
    # clear the GO threshold AND exceed the mean-pool by a clear margin.
    RICHER_MARGIN = 2.0  # richer rep std ratio >= 2x the mean-pool ratio
    richer_varies = ((r_flat_last >= GO_STD_RATIO_THRESH and r_flat_last >= RICHER_MARGIN * r_mean)
                     or (r_flat_all >= GO_STD_RATIO_THRESH and r_flat_all >= RICHER_MARGIN * r_mean)
                     or (r_chan >= GO_STD_RATIO_THRESH and r_chan >= RICHER_MARGIN * r_mean))

    # FORK B: NO representation varies across the recalled slots (every std ratio
    # below the GO threshold) -> the persistent-serve state is degenerate.
    all_flat = (r_mean < GO_STD_RATIO_THRESH and r_chan < GO_STD_RATIO_THRESH
                and r_flat_last < GO_STD_RATIO_THRESH and r_flat_all < GO_STD_RATIO_THRESH)

    print("-" * 72)
    print("FORK DECISION (task #39)")
    print("-" * 72)
    print(f"  mean-pool  std ratio = {r_mean:.3f}x")
    print(f"  max-chan   std ratio = {r_chan:.3f}x   ({r_chan / r_mean:.2f}x the mean-pool)"
          if r_mean > 0 else f"  max-chan   std ratio = {r_chan:.3f}x")
    print(f"  flat_last  std ratio = {r_flat_last:.3f}x   ({r_flat_last / r_mean:.2f}x the mean-pool)"
          if r_mean > 0 else f"  flat_last  std ratio = {r_flat_last:.3f}x")
    print(f"  flat_all   std ratio = {r_flat_all:.3f}x   ({r_flat_all / r_mean:.2f}x the mean-pool)"
          if r_mean > 0 else f"  flat_all   std ratio = {r_flat_all:.3f}x")
    print(f"  GO_STD_RATIO_THRESH = {GO_STD_RATIO_THRESH}x; RICHER_MARGIN = {RICHER_MARGIN}x")
    print(f"  richer_varies = {richer_varies}   all_flat = {all_flat}")
    print()
    if all_flat and not richer_varies:
        print("  -> FORK B: PERSISTENT-STATE DEGENERATE. The state barely varies across")
        print("     the recalled ring slots in EVERY representation (mean/chan/flat). The")
        print("     backbone's persistent-serve state is degenerate -- the new backbone was")
        print("     trained on CANDIDATE-SEQUENCE states (fresh, wm.reset() per query) and")
        print("     its persistent-serve state carries no per-slot contrast. Lever =")
        print("     BACKBONE-ON-SERVE RETRAIN (Phase 1 trainer on serve-distribution")
        print("     persistent-state traces, not ERAG candidate traces). Bigger (pod, hours).")
    elif richer_varies:
        print("  -> FORK A: MEAN-POOL KILLS SERVE SIGNAL. A richer representation")
        print("     (z_flat_last / z_flat_all / a channel) carries materially more across-")
        print("     slot variance than the mean-pool -- the mean over d_state cancelled")
        print("     opposing-sign serve signal (the [[pondr-strm-phase0a-state-signal-")
        print("     readout]] finding, re-confirmed on the NEW backbone's SERVE state).")
        print("     Lever = STATE-TRAJECTORY TRANSFORMER (the ORIGINAL STRM vision --")
        print("     attend over the full per-layer state sequence, not the mean-pool).")
        print("     NOT a backbone retrain: the state varies, the readout destroyed it.")
        if not np.isnan(best_chan_top3) and best_chan_top3 >= 0.5:
            print(f"     BONUS: channel c{best_chan_idx} is already query-aligned (top-3")
            print(f"     {best_chan_top3:.3f}) -- the Transformer has a strong input.")
        else:
            print(f"     NOTE: no single channel is pre-aligned with the bge query (best top-3")
            print(f"     {best_chan_top3:.3f} ~ random) -- the Transformer must MIX channels /")
            print(f"     state steps to find a query-relevant direction (its core job).")
    else:
        # Ambiguous: richer reps vary somewhat but not decisively above the
        # mean-pool, and the state isn't fully flat. The cheaper lever first.
        print("  -> AMBIGUOUS: the state varies in some representations but no richer rep")
        print(f"     clears {GO_STD_RATIO_THRESH}x AND beats the mean-pool by {RICHER_MARGIN}x, and not all reps are")
        print(f"     flat either. The cheaper lever first: try a small STATE-TRAJECTORY")
        print(f"     TRANSFORMER readout (FORK A's lever) on the captured slots_h_raw -- if a")
        print(f"     learned attention over the per-layer state recovers serve top-3 above the")
        print(f"     task #38 plateau (0.478), the mean-pool was the bottleneck. If not, fall")
        print(f"     through to backbone-on-serve retrain (FORK B).")
    print("-" * 72)

    if args.out:
        import json
        report = {
            "n_records": len(records), "n_skipped_no_raw": skipped,
            "doc_std_median": doc_std_med,
            "std_ratio": {"z_mean": r_mean, "max_chan": r_chan,
                          "flat_last": r_flat_last, "flat_all": r_flat_all},
            "top3": {
                "doc": doc_top3, "z_mean": zmean_top3,
                "best_chan": {"idx": best_chan_idx, "top3": best_chan_top3},
                "median_chan_top3": median_chan_top3,
            },
            "fork": {"richer_varies": richer_varies, "all_flat": all_flat},
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str),
                                  encoding="utf-8")
        print(f"\nwrote report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())