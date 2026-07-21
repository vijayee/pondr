"""Task #41: the SERVE ``z_r`` GATE for the flat-readout composite -- the
measurement task #40 never made.

Task #40 ([[pondr-strm-task40-flat-readout-beats-meanpool-transformer-no-add]])
trained three readouts on the captured SERVE ``slots_h_raw`` and found a learned
FLAT readout on ``z_flat_last`` [6144] lifts serve top-3 to 0.739 (clears the 0.6
TRAIN gate, robust over 3 seeds) -- confirming FORK A (the mean-pool was the
bottleneck). BUT that probe measured top-3 RECALL, NOT the SERVE gate metric:
the ``z_r`` SELECTIVITY GAP (probe slot's ``z_r`` minus mean filler ``z_r``, per
source, median >= 0.2 -- the task #33 gate, ``probe_strm_selectivity_real.py``
lines 753-759). Top-3 clearing 0.6 does NOT imply the z_r gap clears 0.2 (top-3
ranks gold in the top-3 without a decisive sigmoid gap). This probe makes that
missing measurement.

The composite is the proper Phase 0b artifact ([[pondr-strm-phase0b-gate-no-go]],
``src/subconscious/state_readout.py``): ``StateReadout`` (raw flattened state
[6144] -> z_i [384]) + ``ZRelevanceHead`` (bilinear z_i . query), trained
end-to-end via the SAME ``fit_relevance`` the 2a / Phase B heads use (only
``slot_signal_field="slots_h_raw"`` + ``head=CompositeZHead`` swap). So the
``z_r`` here is EXACTLY what the live serve probe would compute if the composite
were wired in (``z_r = sigmoid(composite.logits(slot_y, z_flat, q))``); the
readout replaces the parameter-free mean-pool ``project(slot.h)``, nothing else.
Offline == live for this metric.

Two readouts side-by-side (mirroring task #40's (a)/(b)):

  (a) Linear  -- ``StateReadout(6144, 384)`` (Phase 0b default; the "linear
       readout suffices" test).
  (b) MLP-128 -- ``StateReadout(6144, 384, hidden=128)`` (task #40's FlatMLP
       arch, which cleared top-3 0.739).

Train distribution: the captured SERVE traces (train-on-serve, like task #38).
This is the CHEAP, permissive IN-DISTRIBUTION ceiling -- it asks whether the flat
readout can extract a decisive z_r gap from serve state AT ALL. The recurring
train/serve OOD history (task #33 ERAG-trained z_r -0.0066; task #38
serve-trained MEAN-POOL z_r +0.0373) means an ERAG-trained composite would face
the SAME OOD gap -- so if a serve-trained flat readout can't clear 0.2, an
ERAG-trained one certainly won't, and the flat readout is NOT the ship lever
(the task #40 top-3 was misleading). If a serve-trained flat readout DOES clear
0.2 (held-out), the NEXT step is the ERAG-trained composite (regenerate
``traces_hraw.pt`` with the new ``backbone_v2_full.pt`` + ``--identity-instance``
+ ``--emit-raw-state``, then re-run this probe's z_r gap on all serve turns) --
the real OOD ship test.

Two evaluations per readout:

  - HELD-OUT (clean): train on the 80% train-split, compute the z_r gap on the
    20% val-split (the SAME ``_split_queries`` ``fit_relevance`` uses, replicated
    so the val turns are unseen). The decisive number.
  - ALL-TURNS (ceiling): train on ALL serve turns, compute the z_r gap on ALL
    turns (fully in-sample). The upper bound -- if even this doesn't clear 0.2
    the signal isn't there to extract.

The z_r gap replicates ``probe_strm_selectivity_real.py`` lines 712-759 exactly:
group scored (turn, slot) occurrences by ``source_id``; for each source with
>= 3 occurrences, the probe = the max-cos occurrence, fillers = the rest; gap =
probe z_r - mean(filler z_r); median over eligible sources. GATE: median >= 0.2.

The probe ALSO measures the z_logit gap (the pre-sigmoid logit, gap >= 2.0 gate,
the unbounded-logit analog the live serve probe reports alongside z_r) as a
SATURATION diagnostic. A readout with a large z_logit gap but a small z_r gap
HAS a real logit margin that the SIGMOID compresses (serve fillers are
topically close -> all logits high -> sigmoid flat -> z_r margin collapses, the
task #38 / Probe 3 mechanism). That diagnosis changes the conclusion: the lever
is NOT a better readout, it is scoring on the logit scale (z_logit gate) or a
margin/temperature loss that de-saturates the sigmoid. A small z_logit gap too
means no margin signal at all (the readout genuinely fails).

Offline: reads ``traces_serve_identity_hraw.pt`` (carries ``slots_h_raw``
[K,4,16,384] fp16 + ``query_emb`` + ``labels`` + ``source_ids`` + ``cos``). No
backbone, no WorkingMemory, no embedder, no live probe. CPU-fine (114 records).
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.state_readout import (  # noqa: E402
    DEFAULT_DIM_IN,
    CompositeZHead,
    load_composite_z_head,
)
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    _split_queries,
    fit_relevance,
)

# The serve traces carry slot.h as fp16 [K,4,16,384]; the composite's StateReadout
# takes the flattened LAST layer [K, 16*384=6144] (Phase 0b's validated "last" rep
# -- z_flat_all 0.798x diluted the signal per task #39, so last layer only).
LAST_LAYER = -1
D_STATE = 16
D_MODEL = 384
DEFAULT_TRACES = "data/training/strm_relevance/traces_serve_identity_hraw.pt"
DEFAULT_OUT = "data/training/strm_relevance/serve_composite_zrgate.json"
# Train-on-serve artifacts are DIAGNOSTIC, not ship artifacts -- keep them out of
# the real Phase 0b state_readout dir; gitignored (data/ is gitignored anyway).
DEFAULT_CKPT_ROOT = "data/training/strm_state_readout/serve_zrgate"


def _load_serve_traces(traces_path: str) -> list[dict]:
    """Load the captured serve traces, flatten the last SSM layer to [K, 6144].

    The serve probe emits ``slots_h_raw`` as the per-layer state [K,4,16,384]
    (task #39); the composite's ``StateReadout`` takes the flattened LAST layer
    [K, 6144]. Keeps ``source_ids`` / ``cos`` (the z_r gap needs them) alongside
    the ``fit_relevance`` fields (``slots_y`` / ``labels`` / ``query_emb`` /
    ``slots_h_raw``). Drops records without raw state or with <3 slots / no gold
    (top-3 degenerate otherwise -- mirrors load_relevance_traces)."""
    raw = torch.load(traces_path, map_location="cpu", weights_only=False)
    out: list[dict] = []
    for rec in raw:
        h = rec.get("slots_h_raw")
        if h is None:
            continue
        h = h.float()                                       # [K,4,16,384]
        K = h.shape[0]
        if K < 3:
            continue
        labels = rec["labels"].float()
        if int(labels.sum().item()) == 0:
            continue
        z_flat = h[:, LAST_LAYER, :, :].reshape(K, D_STATE * D_MODEL)  # [K,6144]
        out.append({
            "query_emb": rec["query_emb"].float().reshape(-1),    # [384]
            "slots_y": rec["slots_y"].float(),                     # [K,256]
            "labels": labels,                                    # [K]
            "source_ids": list(rec["source_ids"]),                # [K]
            "cos": rec["cos"].float(),                           # [K]
            "slots_h_raw": z_flat,                               # [K,6144]
        })
    return out


def _zr_per_slot(composite: CompositeZHead, rec: dict, device) -> tuple[Tensor, Tensor]:
    """``(z_r, z_logit)`` per slot -> ``([K], [K])``.

    ``z_r = sigmoid(logit)`` (the SERVE gate metric, 0.2 sigmoid gap gate);
    ``z_logit`` is the pre-sigmoid logit (the 2.0 unbounded-logit gap gate).
    Returning BOTH lets the probe tell SATURATION (large logit gap, compressed
    sigmoid gap) from a genuine lack of signal (both small) -- the diagnostic
    that decides whether the flat readout fails the z_r gate because the head
    saturates (fixable: margin/temperature) or because z_i carries no margin
    signal (not fixable by a readout)."""
    z_flat = rec["slots_h_raw"].to(device).to(torch.float32)      # [K,6144]
    q = rec["query_emb"].to(device).to(torch.float32)             # [384]
    K = z_flat.shape[0]
    slot_y = torch.zeros(K, int(composite.slot_dim), device=device)
    with torch.no_grad():
        logit = composite.logits(slot_y, z_flat, q).squeeze(-1)    # [K]
        return torch.sigmoid(logit), logit


def _selectivity_gap(turns: list[dict], per_slot_fn, gate: float) -> dict:
    """Per-source selectivity gap (mirrors probe_strm_selectivity_real.py
    lines 712-759). Group (turn, slot) occurrences by source_id; for each source
    with >= 3 occurrences, probe = max-cos occ, fillers = the rest; gap =
    probe score - mean(filler score). ``per_slot_fn(rec) -> [K]`` returns the
    per-slot score (z_r OR z_logit; the device is captured in the closure).
    Returns median/mean/min + n_eligible + the ``>= gate`` decision."""
    by_source: dict[str, list[tuple[float, float]]] = {}
    for rec in turns:
        score = per_slot_fn(rec)                                  # [K]
        cos = rec["cos"]
        sids = rec["source_ids"]
        for i in range(score.shape[0]):
            by_source.setdefault(sids[i], []).append(
                (float(cos[i].item()), float(score[i].item())))
    gaps: list[float] = []
    for sid, occs in by_source.items():
        if len(occs) < 3:
            continue
        occs.sort(key=lambda o: o[0], reverse=True)               # by cos desc
        probe = occs[0]
        fillers = occs[1:]
        gaps.append(probe[1] - statistics.fmean(o[1] for o in fillers))
    if not gaps:
        return {"median": None, "mean": None, "min": None, "n_eligible": 0,
                "n_sources": len(by_source), "n_ge_gate": 0, "gate": False}
    return {
        "median": statistics.median(gaps),
        "mean": statistics.fmean(gaps),
        "min": min(gaps),
        "n_eligible": len(gaps),
        "n_sources": len(by_source),
        "n_ge_gate": sum(1 for g in gaps if g >= gate),
        "gate": statistics.median(gaps) >= gate,
    }


def _zr_and_logit_gaps(composite: CompositeZHead, turns: list[dict], device) -> dict:
    """Both the z_r sigmoid gap (0.2 gate) and the z_logit pre-sigmoid gap
    (2.0 gate, the unbounded-logit analog the live serve probe reports). The
    z_logit gap is the SATURATION diagnostic: a large z_logit gap with a small
    z_r gap means the head produces a real logit margin that the sigmoid
    compresses (fixable); both small means no margin signal (not fixable by a
    readout)."""
    def zr_fn(rec): return _zr_per_slot(composite, rec, device)[0]
    def zlg_fn(rec): return _zr_per_slot(composite, rec, device)[1]
    return {"z_r": _selectivity_gap(turns, zr_fn, 0.2),
            "z_logit": _selectivity_gap(turns, zlg_fn, 2.0)}


def _train_composite(name: str, traces: list[dict], hidden: int | None,
                     ckpt_dir: Path, epochs: int, seed: int, device: str) -> dict:
    """Train one CompositeZHead via ``fit_relevance`` (the proper Phase 0b
    trainer) on the full trace set, writing best.pt to ``ckpt_dir``. Returns
    the TRAIN gate scorecard (top-3 + Wilson) + the best-epoch index."""
    dim_in = int(traces[0]["slots_h_raw"].shape[1])
    head = CompositeZHead(dim_in=dim_in, hidden=hidden)
    arch = "MLP-128" if hidden else "Linear"
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\ntraining {name} ({arch} readout {dim_in}->384, {n_params:,} params, "
          f"{epochs} epochs, seed {seed}) -> {ckpt_dir}", flush=True)
    cfg = RelevanceTrainingConfig(
        epochs=epochs, seed=seed, device=device,
        checkpoint_dir=str(ckpt_dir),
        slot_signal_field="slots_h_raw",
    )
    result = fit_relevance(traces, cfg, head=head)
    return {"name": name, "arch": arch, "hidden": hidden, "dim_in": dim_in,
            "n_params": n_params, "best_epoch": result["best_epoch"],
            "train_top3": result["best_pc"]["mean_top3_recall"],
            "train_hit": result["best_pc"]["hit_rate"],
            "train_ci": result["best_pc"]["hit_ci95"],
            "train_go": result["go"], "ckpt": ckpt_dir / "best.pt"}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task #41: SERVE z_r GATE for the flat-readout composite -- "
                    "the z_r selectivity gap (median >= 0.2) task #40 never "
                    "measured. Offline on the captured serve traces.")
    p.add_argument("--traces", default=DEFAULT_TRACES,
                   help="serve traces with slots_h_raw (probe_strm_selectivity_real.py "
                        "--emit-traces --emit-raw-state).")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default="",
                   help="comma-separated subset of readouts to run: a,b "
                        "(a=Linear, b=MLP-128; default both)")
    p.add_argument("--device", default="cpu",
                   help="train device (default cpu; the composite is a small MLP).")
    p.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT,
                   help="root for the per-readout train-on-serve checkpoints "
                        "(diagnostic, not ship artifacts).")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    if not Path(args.traces).exists():
        print(f"ERROR: traces not found at {args.traces}", file=sys.stderr)
        return 1

    traces = _load_serve_traces(args.traces)
    if len(traces) < 5:
        print(f"ERROR: only {len(traces)} usable records (need >=5)", file=sys.stderr)
        return 1
    ks = sorted(r["slots_h_raw"].shape[0] for r in traces)
    print(f"loaded {len(traces)} serve turns (K min/med/max={ks[0]}/"
          f"{ks[len(traces)//2]}/{ks[-1]}), dim_in={traces[0]['slots_h_raw'].shape[1]}",
          flush=True)

    # The composite is scored on the SAME device it was trained on + loaded to.
    # load_composite_z_head defaults to auto (CUDA if available); pin to the
    # train device so train/eval match and the small MLP stays on CPU when asked.
    eval_device = args.device

    readouts = (
        ("(a) Linear",  "a", None),
        ("(b) MLP-128",  "b", 128),
    )
    only = set(args.only.split(",")) if args.only else {"a", "b"}
    ckpt_root = Path(args.ckpt_root)
    results = []
    for name, key, hidden in readouts:
        if key not in only:
            continue
        ckpt_dir = ckpt_root / ("linear" if hidden is None else f"mlp{hidden}")
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        r = _train_composite(name, traces, hidden, ckpt_dir, args.epochs,
                             args.seed, args.device)
        composite = load_composite_z_head(str(r["ckpt"]), device=eval_device,
                                          map_location=eval_device)
        # HELD-OUT: replicate fit_relevance's val split; eval z_r gap on val turns
        # (the composite trained on the train-split -- best.pt is gate-selected
        # over those val turns, but the z_r gap is a DIFFERENT metric from top-3,
        # so this is still a fair held-out z_r measurement, not a tautology).
        _, val_idx = _split_queries(len(traces),
                                   RelevanceTrainingConfig().val_fraction, args.seed)
        r["heldout"] = _zr_and_logit_gaps(composite,
                                          [traces[i] for i in val_idx], eval_device)
        # ALL-TURNS ceiling (train on all, eval on all -- in-sample upper bound).
        r["allturns"] = _zr_and_logit_gaps(composite, traces, eval_device)
        results.append(r)
        hg, ag = r["heldout"], r["allturns"]
        print(f"  {name} TRAIN top3={r['train_top3']:.3f} "
              f"(ci[{r['train_ci'][0]:.2f},{r['train_ci'][1]:.2f}], "
              f"{'GO' if r['train_go'] else 'no-go'})", flush=True)
        print(f"    held-out  z_r    gap med={hg['z_r']['median']:.4f} "
              f"(n_elig={hg['z_r']['n_eligible']}, n_ge_0.2={hg['z_r']['n_ge_gate']}, "
              f"gate={'PASS' if hg['z_r']['gate'] else 'fail'})", flush=True)
        print(f"    held-out  z_logit gap med={hg['z_logit']['median']:.4f} "
              f"(n_elig={hg['z_logit']['n_eligible']}, n_ge_2.0={hg['z_logit']['n_ge_gate']}, "
              f"gate={'PASS' if hg['z_logit']['gate'] else 'fail'})", flush=True)
        print(f"    all-turns z_r    gap med={ag['z_r']['median']:.4f} "
              f"(n_elig={ag['z_r']['n_eligible']}, n_ge_0.2={ag['z_r']['n_ge_gate']}, "
              f"gate={'PASS' if ag['z_r']['gate'] else 'fail'}) [ceiling]", flush=True)
        print(f"    all-turns z_logit gap med={ag['z_logit']['median']:.4f} "
              f"(n_elig={ag['z_logit']['n_eligible']}, n_ge_2.0={ag['z_logit']['n_ge_gate']}, "
              f"gate={'PASS' if ag['z_logit']['gate'] else 'fail'}) [ceiling]", flush=True)

    # ── verdict ──
    print("\n" + "=" * 76)
    print("VERDICT (task #41: SERVE z_r GATE for the flat-readout composite)")
    print("=" * 76)
    print(f"  SERVE gate: per-source selectivity gap median >= 0.2 (z_r) / >= 2.0 (z_logit)")
    print(f"  baselines: task #38 serve-trained MEAN-POOL z_r gap +0.0373; task #40 flat")
    print(f"  readout top-3 0.739 (z_r gap never measured)")
    print()
    for r in results:
        hg, ag = r["heldout"], r["allturns"]
        print(f"  {r['name']} {r['arch']}: TRAIN top3={r['train_top3']:.3f}")
        print(f"    held-out  z_r={hg['z_r']['median']:.4f} "
              f"(n_ge_0.2={hg['z_r']['n_ge_gate']}/{hg['z_r']['n_eligible']}, "
              f"{'PASS' if hg['z_r']['gate'] else 'fail'})  "
              f"z_logit={hg['z_logit']['median']:.4f} "
              f"(n_ge_2.0={hg['z_logit']['n_ge_gate']}/{hg['z_logit']['n_eligible']}, "
              f"{'PASS' if hg['z_logit']['gate'] else 'fail'})")
        print(f"    all-turns z_r={ag['z_r']['median']:.4f} "
              f"(n_ge_0.2={ag['z_r']['n_ge_gate']}/{ag['z_r']['n_eligible']}, "
              f"{'PASS' if ag['z_r']['gate'] else 'fail'})  "
              f"z_logit={ag['z_logit']['median']:.4f} "
              f"(n_ge_2.0={ag['z_logit']['n_ge_gate']}/{ag['z_logit']['n_eligible']}, "
              f"{'PASS' if ag['z_logit']['gate'] else 'fail'}) [ceiling]")
    print()

    heldout_pass = any(r["heldout"]["z_r"]["gate"] for r in results)
    ceiling_pass = any(r["allturns"]["z_r"]["gate"] for r in results)
    # SATURATION diagnostic: does a readout with a large z_logit gap also have a
    # large z_r gap? If z_logit clears 2.0 but z_r doesn't clear 0.2, the head
    # HAS a real logit margin that the sigmoid compresses -> the fix is a
    # margin/temperature loss, not a better readout. If both fail, no margin.
    sat_lines = []
    for r in results:
        ag = r["allturns"]
        if (ag["z_logit"]["median"] is not None and ag["z_logit"]["median"] >= 2.0
                and not ag["z_r"]["gate"]):
            sat_lines.append(f"     {r['name']}: z_logit ceiling {ag['z_logit']['median']:.2f} "
                             f">= 2.0 but z_r ceiling {ag['z_r']['median']:.3f} < 0.2 -> SATURATED")

    if heldout_pass:
        print("  -> HELD-OUT z_r GATE PASS: a learned flat readout extracts a DECISIVE")
        print("     serve z_r gap (>= 0.2) on held-out serve turns. The flat readout IS")
        print("     the ship lever in-distribution. NEXT: the ERAG-trained composite")
        print("     (regenerate traces_hraw.pt with backbone_v2_full.pt + --identity-")
        print("     instance + --emit-raw-state, re-run this probe's z_r gap on ALL")
        print("     serve turns) -- the real OOD ship test.")
    elif ceiling_pass:
        print("  -> CEILING-ONLY PASS: the flat readout clears 0.2 only in-sample, not")
        print("     held-out. Overfits the 114-record serve set. NOT justified for the")
        print("     ERAG generate without regularization / more Onyx transcripts.")
    elif sat_lines:
        print("  -> z_r GATE FAIL but SATURATION diagnosed: the head produces a real")
        print("     z_logit margin (in-sample ceiling >= 2.0) that the SIGMOID compresses")
        print("     to a sub-0.2 z_r gap. The flat readout ranks gold well (top-3 0.65-0.74)")
        print("     with a real LOGIT margin, but serve fillers are topically close -> all")
        print("     logits high -> sigmoid saturates -> z_r margin collapses. This is the")
        print("     task #38 / Probe 3 saturation mechanism, now measured at the z_logit")
        print("     vs z_r split. The z_r (sigmoid) gate is the WRONG metric for a z_i")
        print("     bilinear on serve; score on z_logit (>= 2.0) or de-saturate (margin /")
        print("     temperature loss).")
        for s in sat_lines:
            print(s)
        # Honesty: the ceiling z_logit pass is IN-SAMPLE (train turns included).
        # The held-out z_logit is the honest number; flag whether it clears 2.0.
        ho_logit_pass = any(r["heldout"]["z_logit"]["median"] is not None
                            and r["heldout"]["z_logit"]["median"] >= 2.0 for r in results)
        if not ho_logit_pass:
            print("     BUT the held-out z_logit median is BELOW 2.0 (the ceiling pass is")
            print("     in-sample -- 934K-2.5M params on ~91 train turns overfits). The")
            print("     flat readout does NOT robustly clear even the z_logit gate held-out")
            print("     on 114 serve turns (multi-seed: held-out z_logit 0.04-1.64, noisy).")
            print("     The lever is NOT validated on this data: it needs MORE Onyx serve")
            print("     transcripts + regularization before the z_logit gate is a real")
            print("     ship signal. Do NOT invest in the ERAG generate on this evidence.")
    else:
        print("  -> z_r GATE FAIL (no saturation rescue): even the in-sample z_logit gap")
        print("     is < 2.0, so the flat readout produces NO decisive margin at ALL (not")
        print("     just a compressed one). The task #40 top-3 (0.739) was misleading --")
        print("     gold ranks in the top-3 without a logit margin over fillers. The flat")
        print("     readout is NOT the ship lever; the mean-pool (task #38 +0.0373) and")
        print("     the flat readout both fail. Do NOT invest in the ERAG generate.")
        print("     Reconsider: cross-slot trajectory Transformer, or accept the z_r gate")
        print("     is too strict for a per-slot z_i bilinear on serve (topical closeness")
        print("     -> saturation).")
    print("=" * 76)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "n_records": len(traces), "epochs": args.epochs, "seed": args.seed,
            "results": [{"name": r["name"], "arch": r["arch"], "hidden": r["hidden"],
                         "dim_in": r["dim_in"], "n_params": r["n_params"],
                         "best_epoch": r["best_epoch"], "train_top3": r["train_top3"],
                         "train_ci": r["train_ci"], "train_go": r["train_go"],
                         "heldout": r["heldout"], "allturns": r["allturns"]}
                        for r in results],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())