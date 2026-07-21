"""Task #44: contrastive InfoNCE margin loss retrain + eval z_logit on lmsys/Onyx.

The decisive test of the task #43 diagnosis
([[pondr-strm-task43-lmsys-transfer-zlogit-fail]]): 55x more serve-like data did
NOT raise the z_logit margin (stayed ~0.3 held-out) because saturation is
INTRINSIC to the per-slot z_i bilinear on serve-like data, not a data-size
artifact. The diagnosed root cause: ``fit_relevance``'s per-slot BCE
(``F.binary_cross_entropy_with_logits(logits, labels, pos_weight=...)``) pushes
each gold->1 / filler->0 INDEPENDENTLY -- there is NO inter-slot contrast, so on
topically-close data all logits stay high together -> small margins. The fix
must be loss-level: a CONTRASTIVE (InfoNCE) margin loss that directly pushes
gold's logit DECISIVELY above the fillers'. This probe is that test.

**Why contrastive de-saturates (the mechanism).** The composite's ``bias`` is a
SINGLE scalar broadcast to every slot (``z_relevance_head.py:107,155``). Per-slot
BCE can "cheat" via that bias: push it down so ALL sigmoids -> 0, minimizing the
filler loss (14 fillers, weight 1 each) while dragging gold down too (the
pos_weight-upweighted gold loss is still net-cheaper to let collapse than to
hold up against 14 similar fillers) -- the head reports "low relevance
everywhere" and the gold-filler MARGIN stays small. The contrastive loss
``L = logsumexp(logits/T) - logsumexp(logits_gold/T)`` is BIAS-INVARIANT (the bias
adds the same constant to every slot's logit, so it cancels in the
logsumexp-all minus logsumexp-gold). The head CANNOT cheat via the bias -- it is
FORCED to separate gold from fillers through the bilinear term
``(proj_z(z).proj_q(q))/sqrt(P)`` alone. That is the precise mechanism by which
contrastive attacks the diagnosed root cause: the missing inter-slot contrast,
and the bias-collapse cheat BCE permits.

**Multi-positive InfoNCE.** ``L = logsumexp(logits_all/T) - logsumexp(logits_gold/T)``
(the backbone's ``relevance_loss`` form, ``train_backbone_relevance.py:162``).
On serve traces (single gold = top-1-cos) this is standard 1-of-K InfoNCE
``logsumexp(all/T) - gold_logit/T``; the multi-positive form also handles ERAG
multi-gold if we ever train there. Temperature T sets the margin scale: the
contrastive optimum pushes ``gold_logit - mean(filler_logit) ~ T * log(K_neg)``,
so T=1.0 -> ~2.0 for K~8 -- matching the 2.0 z_logit gate scale directly.

**Apples-to-apples with task #43.** SAME arch (``CompositeZHead`` mlp128),
SAME data (the lmsys serve traces, ``traces_lmsys_serve_hraw.pt``), SAME eval
(``p41._zr_and_logit_gaps`` -> the per-source z_logit gap on lmsys held-out +
real Onyx). The ONLY thing that changes is the LOSS: BCE -> contrastive. So a
PASS here (Onyx z_logit >= 2.0 robust) where task #43 FAILED (0.258) is a clean
attribution of the win to the contrastive loss, not to data/arch/eval.

**Isolation (the binding constraint).** Standalone script with its OWN training
loop -- reuses ``evaluate_relevance`` / ``_gate_score`` / ``_split_queries`` /
the checkpoint format + ``load_composite_z_head`` (all loss-agnostic), but does
NOT call ``fit_relevance``. So ``fit_relevance`` (the shared 2a / Phase B trainer)
is UNTOUCHED -> every existing head + the 2b gate stay byte-identical. No live
wiring, no HF upload (diagnostic, gitignored under data/). ``--loss bce``
reproduces task #43 (it calls ``fit_relevance``) as the within-script control.

GO/NO-GO (same gates as task #43):
  GO  = Onyx z_logit gap median >= 2.0, ROBUST across --seeds (>= 2/3 pass).
        -> the contrastive margin loss IS the de-saturating lever; the flat
           readout z_i bilinear is the ship arch once the loss is fixed. Re-run
           the live SERVE gate (probe_strm_selectivity_real.py) next.
  NO-GO = Onyx z_logit < 2.0 across seeds.
        -> either the z_i bilinear genuinely cannot separate gold from
           topically-close fillers (ranking 0.61 is real but the margin is
           arch-bounded, not loss-bounded), OR the lmsys->Onyx transfer gap
           remains. Report which; the cross-slot trajectory Transformer (option
           2) or accepting the saturation (option 3) follow.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))        # sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import probe_serve_composite_zrgate as p41  # noqa: E402
from src.subconscious.state_readout import (  # noqa: E402
    CompositeZHead,
    load_composite_z_head,
)
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    _gate_score,
    _split_queries,
    evaluate_relevance,
    fit_relevance,
)

DEFAULT_LMSYS = "data/training/strm_relevance/traces_lmsys_serve_hraw.pt"
DEFAULT_ONYX = "data/training/strm_relevance/traces_serve_identity_hraw.pt"
DEFAULT_OUT = "data/training/strm_relevance/contrastive_zlogit.json"
DEFAULT_CKPT_ROOT = "data/training/strm_state_readout/contrastive_zlogit"

# z_logit gate threshold (pre-sigmoid per-source gap median). The z_r gate (0.2)
# is decided inside p41._zr_and_logit_gaps (hardcoded there); we report z_r as a
# diagnostic only -- the decisive gate is z_logit (task #41 showed z_r saturates
# on serve under BCE; the contrastive is bias-invariant so it MAY also lift z_r,
# but the ship decision rides on z_logit, matching task #43).
ZLOGIT_GATE = 2.0


def _to_device(traces: list[dict], device: str) -> list[dict]:
    """Move every tensor field in each record to the train/eval device. Mirrors
    task #43's helper (fit_relevance + evaluate_relevance do not move inputs to
    the head's device themselves; this keeps the head + inputs co-located)."""
    if device == "cpu":
        return traces
    dev = torch.device(device)
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in rec.items()}
            for rec in traces]


# ── the contrastive (InfoNCE) margin loss ──

def contrastive_loss(logits: Tensor, gold_mask: Tensor,
                     temperature: float) -> Tensor:
    """Multi-positive InfoNCE on the composite's per-slot logits.

    ``L = logsumexp(logits/T) - logsumexp(logits_gold/T)`` (0 if no gold).

    Bias-invariant by construction: the composite's ``bias`` is a single scalar
    broadcast to every slot, so it adds the same constant to all logits and
    CANCELS in (logsumexp(all) - logsumexp(gold)). The head cannot "cheat" by
    collapsing all sigmoids toward 0 (the per-slot BCE saturation mechanism,
    [[pondr-strm-task41-serve-zrgate-saturation]] / [[pondr-strm-probe3-cost-parity]]);
    it is forced to separate gold from fillers via the bilinear term alone.

    ``logits`` ``[K]`` (the composite's pre-sigmoid logit per slot, already
    includes the bilinear scale ``1/sqrt(P)`` + bias); ``gold_mask`` ``[K]`` bool;
    ``temperature`` divides the logits (small T -> sharper softmax -> larger
    required margin). The contrastive optimum pushes
    ``gold_logit - mean(filler_logit) ~ T*log(K_neg)`` -- T=1.0 -> ~2.0 for K~8,
    on the 2.0 z_logit gate scale.
    """
    if gold_mask.sum() == 0:
        return logits.new_zeros(())
    return (torch.logsumexp(logits / temperature, dim=0)
            - torch.logsumexp(logits[gold_mask] / temperature, dim=0))


# ── contrastive training loop (mirrors fit_relevance's loop, loss swapped) ──

def _train_contrastive(name: str, traces: list[dict], hidden: int | None,
                      temperature: float, epochs: int, lr: float,
                      weight_decay: float, accum_steps: int, seed: int,
                      device: str, ckpt_dir: Path) -> dict:
    """Train one CompositeZHead on the lmsys traces with the contrastive loss.

    Mirrors ``fit_relevance``'s loop + gate-aware checkpoint selection
    (``_gate_score``) + checkpoint format (so ``load_composite_z_head`` reads
    it), but swaps the per-slot BCE for ``contrastive_loss``. Reuses
    ``evaluate_relevance`` (loss-agnostic -- it only calls ``head.logits`` +
    top-3 + Wilson) for the per-epoch TRAIN top-3 gate. Writes best.pt
    (gate-selected on the lmsys val split) to ``ckpt_dir``."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    dim_in = int(traces[0]["slots_h_raw"].shape[1])
    head = CompositeZHead(dim_in=dim_in, hidden=hidden).to(dev)
    arch = f"MLP-{hidden}" if hidden else "Linear"
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\ntraining {name} seed={seed} CONTRASTIVE ({arch} {dim_in}->384, "
          f"{n_params:,} params, T={temperature}, wd={weight_decay}, {epochs} "
          f"epochs) -> {ckpt_dir}", flush=True)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx, val_idx = _split_queries(len(traces),
                                        RelevanceTrainingConfig().val_fraction,
                                        seed)
    train = [traces[i] for i in train_idx]
    val = [traces[i] for i in val_idx]
    print(f"  {len(train)} train / {len(val)} val queries (split by query, no "
          f"slot leakage)", flush=True)

    rng = random.Random(seed)
    best_score: tuple | None = None
    best_pc: dict | None = None
    best_epoch = -1
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accum = max(1, accum_steps)
    ck_slot_dim = int(head.slot_dim)
    ck_doc_dim = int(head.doc_dim)
    ck_query_dim = int(head.query_dim)
    ck_proj_dim = int(head.proj_dim)
    last_pc: dict | None = None

    for epoch in range(epochs):
        head.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for k, qi in enumerate(order):
            rec = train[qi]
            z_flat = rec["slots_h_raw"].to(dev).to(torch.float32)      # [K,6144]
            q = rec["query_emb"].to(dev).to(torch.float32)            # [384]
            labels = rec["labels"].to(dev).to(torch.float32)          # [K]
            K = z_flat.shape[0]
            # slot_y is IGNORED by the composite (pure-z_i test); zeros match
            # p41._zr_per_slot so train + eval see the same ignored input.
            slot_y = torch.zeros(K, ck_slot_dim, device=dev)
            logits = head.logits(slot_y, z_flat, q).squeeze(-1)       # [K]
            gold = labels > 0
            loss = contrastive_loss(logits, gold, temperature) / accum
            loss.backward()
            total_loss += float(loss.item()) * accum
            n_steps += 1
            if (k + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        if n_steps % accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / max(n_steps, 1)
        pc = evaluate_relevance(head, val, slot_signal_field="slots_h_raw")
        last_pc = pc
        go = (pc["mean_top3_recall"] >= RelevanceTrainingConfig().gate_top3
              and pc["hit_ci95"][0] > RelevanceTrainingConfig().gate_wilson_low)
        ci = pc["hit_ci95"]
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} "
              f"top3={pc['mean_top3_recall']:.3f} hit={pc['hit_rate']:.2f} "
              f"ci=[{ci[0]:.2f},{ci[1]:.2f}] r_pos={pc['mean_r_positive']:.3f} "
              f"{'GO' if go else 'no-go'}", flush=True)

        score = _gate_score(pc, RelevanceTrainingConfig())
        if best_score is None or score > best_score:
            best_score = score
            best_pc = pc
            best_epoch = epoch
            torch.save({"head": head.state_dict(), "slot_dim": ck_slot_dim,
                        "doc_dim": ck_doc_dim, "query_dim": ck_query_dim,
                        "proj_dim": ck_proj_dim,
                        "top3_recall": pc["mean_top3_recall"],
                        "hit_rate": pc["hit_rate"], "hit_ci95": pc["hit_ci95"],
                        "go": go, "epoch": epoch, "loss": "contrast"},
                       ckpt_dir / "best.pt")

    # final.pt = the LAST epoch (mirrors fit_relevance; best.pt is gate-selected).
    if last_pc is not None:
        torch.save({"head": head.state_dict(), "slot_dim": ck_slot_dim,
                    "doc_dim": ck_doc_dim, "query_dim": ck_query_dim,
                    "proj_dim": ck_proj_dim,
                    "top3_recall": last_pc["mean_top3_recall"],
                    "hit_rate": last_pc["hit_rate"],
                    "hit_ci95": last_pc["hit_ci95"], "go": go, "epoch": epochs - 1,
                    "loss": "contrast"}, ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "loss": "contrast",
                   "temperature": temperature, "seed": seed,
                   "n_train": len(train), "n_val": len(val),
                   "best_scorecard": best_pc}, f, indent=2)

    return {"name": name, "arch": arch, "hidden": hidden, "dim_in": dim_in,
            "n_params": n_params, "temperature": temperature,
            "best_epoch": best_epoch,
            "lmsys_train_top3": best_pc["mean_top3_recall"] if best_pc else 0.0,
            "lmsys_train_ci": best_pc["hit_ci95"] if best_pc else [0.0, 1.0],
            "lmsys_train_go": (best_pc is not None
                               and best_pc["mean_top3_recall"] >= 0.6
                               and best_pc["hit_ci95"][0] > 0.5),
            "ckpt": ckpt_dir / "best.pt"}


def _train_bce_control(name: str, traces: list[dict], hidden: int | None,
                       weight_decay: float, epochs: int, seed: int, device: str,
                       ckpt_dir: Path) -> dict:
    """BCE control: reproduce task #43 exactly via ``fit_relevance``. Proves the
    script's eval pipeline matches task #43's numbers (the contrastive result is
    comparable to a same-script BCE baseline, not just the historical number)."""
    dim_in = int(traces[0]["slots_h_raw"].shape[1])
    head = CompositeZHead(dim_in=dim_in, hidden=hidden)
    arch = f"MLP-{hidden}" if hidden else "Linear"
    print(f"\ntraining {name} seed={seed} BCE control ({arch}, wd={weight_decay}, "
          f"{epochs} epochs) -> {ckpt_dir}", flush=True)
    cfg = RelevanceTrainingConfig(
        epochs=epochs, seed=seed, device=device,
        checkpoint_dir=str(ckpt_dir), slot_signal_field="slots_h_raw",
        weight_decay=weight_decay)
    result = fit_relevance(traces, cfg, head=head)
    return {"name": name, "arch": arch, "hidden": hidden, "dim_in": dim_in,
            "temperature": None,
            "best_epoch": result["best_epoch"],
            "lmsys_train_top3": result["best_pc"]["mean_top3_recall"],
            "lmsys_train_ci": result["best_pc"]["hit_ci95"],
            "lmsys_train_go": result["go"], "ckpt": ckpt_dir / "best.pt"}


def _run_one(name: str, lmsys: list[dict], onyx: list[dict], hidden: int | None,
             loss: str, temperature: float, weight_decay: float, epochs: int,
             lr: float, accum_steps: int, seed: int, device: str,
             ckpt_root: Path) -> dict:
    """Train one head (contrastive OR bce) on lmsys, eval on Onyx + lmsys held-out."""
    ckpt_dir = ckpt_root / f"{name}_s{seed}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if loss == "contrast":
        r = _train_contrastive(name, lmsys, hidden, temperature, epochs, lr,
                               weight_decay, accum_steps, seed, device, ckpt_dir)
    else:
        r = _train_bce_control(name, lmsys, hidden, weight_decay, epochs, seed,
                               device, ckpt_dir)
    composite = load_composite_z_head(str(r["ckpt"]), device=device,
                                      map_location=device)
    # lmsys held-out sanity (replicate fit_relevance's val split for this seed).
    _, val_idx = _split_queries(len(lmsys), RelevanceTrainingConfig().val_fraction,
                                seed)
    r["lmsys_heldout"] = p41._zr_and_logit_gaps(
        composite, [lmsys[i] for i in val_idx], device)
    # The TRANSFER eval: z_r + z_logit gaps on the REAL Onyx serve traces.
    r["onyx"] = p41._zr_and_logit_gaps(composite, onyx, device)
    r["seed"] = seed
    r["loss"] = loss
    ho = r["lmsys_heldout"]
    on = r["onyx"]
    print(f"  lmsys held-out z_logit={ho['z_logit']['median']:.3f} "
          f"(n_ge_2.0={ho['z_logit']['n_ge_gate']}/{ho['z_logit']['n_eligible']})  "
          f"ONYX z_logit={on['z_logit']['median']:.3f} "
          f"(n_ge_2.0={on['z_logit']['n_ge_gate']}/{on['z_logit']['n_eligible']}, "
          f"{'PASS' if on['z_logit']['median'] is not None and on['z_logit']['median'] >= ZLOGIT_GATE else 'fail'})  "
          f"ONYX z_r={on['z_r']['median']:.4f}", flush=True)
    return r


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task #44: contrastive InfoNCE margin loss retrain + "
                    "eval-Onyx z_logit (the task #43 saturation fix).")
    p.add_argument("--lmsys", default=DEFAULT_LMSYS,
                   help="lmsys serve-like traces (generate_lmsys_serve_traces.py).")
    p.add_argument("--onyx", default=DEFAULT_ONYX,
                   help="Onyx serve traces (the transfer target; task #41/#43's set).")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated train seeds (robustness sweep).")
    p.add_argument("--loss", default="contrast", choices=["contrast", "bce"],
                   help="loss: contrast (InfoNCE margin, the fix) or bce "
                        "(reproduces task #43 via fit_relevance, the control).")
    p.add_argument("--readout", default="mlp128",
                   choices=["linear", "mlp64", "mlp128"],
                   help="StateReadout arch (same knob as task #43).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="contrastive softmax temperature T (margin ~ T*log(K); "
                        "1.0 -> ~2.0 for K~8, on the 2.0 gate scale).")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW lr (defaults to fit_relevance's 1e-3).")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight_decay (same knob as task #43).")
    p.add_argument("--accum-steps", type=int, default=4,
                   help="gradient accumulation (fit_relevance's default).")
    p.add_argument("--device", default="cpu",
                   help="train+eval device (cuda trains faster on the lmsys set).")
    p.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    if not Path(args.lmsys).exists():
        print(f"ERROR: lmsys traces not found at {args.lmsys}\n"
              f"  run: python scripts/generate_lmsys_serve_traces.py",
              file=sys.stderr)
        return 1
    if not Path(args.onyx).exists():
        print(f"ERROR: onyx traces not found at {args.onyx}", file=sys.stderr)
        return 1

    lmsys = p41._load_serve_traces(args.lmsys)
    onyx = p41._load_serve_traces(args.onyx)
    # Move tensors to the train/eval device (fit_relevance + evaluate_relevance
    # do not move inputs themselves; pre-moving avoids repeated H2D + keeps the
    # contrastive loop's head + inputs co-located).
    lmsys = _to_device(lmsys, args.device)
    onyx = _to_device(onyx, args.device)
    if len(lmsys) < 50:
        print(f"ERROR: only {len(lmsys)} lmsys records (need >=50)", file=sys.stderr)
        return 1
    if len(onyx) < 5:
        print(f"ERROR: only {len(onyx)} onyx records", file=sys.stderr)
        return 1
    lk = sorted(r["slots_h_raw"].shape[0] for r in lmsys)
    ok = sorted(r["slots_h_raw"].shape[0] for r in onyx)
    print(f"lmsys: {len(lmsys)} turns (K min/med/max={lk[0]}/{lk[len(lk)//2]}/{lk[-1]}), "
          f"dim_in={lmsys[0]['slots_h_raw'].shape[1]}", flush=True)
    print(f"onyx:  {len(onyx)} turns (K min/med/max={ok[0]}/{ok[len(ok)//2]}/{ok[-1]})  "
          f"[transfer target]", flush=True)

    hidden = {"linear": None, "mlp64": 64, "mlp128": 128}[args.readout]
    name = args.readout
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ckpt_root = Path(args.ckpt_root)

    per_seed = []
    for seed in seeds:
        per_seed.append(_run_one(
            name, lmsys, onyx, hidden, args.loss, args.temperature,
            args.weight_decay, args.epochs, args.lr, args.accum_steps, seed,
            args.device, ckpt_root))

    # ── aggregate across seeds ──
    onyx_zlogit_meds = [r["onyx"]["z_logit"]["median"] for r in per_seed
                        if r["onyx"]["z_logit"]["median"] is not None]
    onyx_zr_meds = [r["onyx"]["z_r"]["median"] for r in per_seed
                    if r["onyx"]["z_r"]["median"] is not None]
    lmsys_ho_zlogit_meds = [r["lmsys_heldout"]["z_logit"]["median"] for r in per_seed
                            if r["lmsys_heldout"]["z_logit"]["median"] is not None]
    n_onyx_pass = sum(1 for m in onyx_zlogit_meds if m >= ZLOGIT_GATE)
    # Robust = at least 2 seeds pass AND a majority pass (>=2/3 for 3 seeds).
    robust_pass = n_onyx_pass >= 2 and n_onyx_pass * 2 >= len(seeds)

    print("\n" + "=" * 78)
    print(f"VERDICT (task #44: {args.loss.upper()} loss, "
          f"{'T='+str(args.temperature) if args.loss == 'contrast' else ''})")
    print("=" * 78)
    print(f"  readout={args.readout}  weight_decay={args.weight_decay}  "
          f"seeds={seeds}  z_logit gate={ZLOGIT_GATE}")
    print(f"  lmsys train turns: {len(lmsys)}  (~{len(lmsys)//91}x the Onyx train set)")
    print(f"  baseline (task #43 BCE): Onyx z_logit 0.258 (0/3 pass), lmsys held-out 0.326")
    print()
    for r in per_seed:
        on = r["onyx"]; ho = r["lmsys_heldout"]
        ts = f" T={r['temperature']}" if r.get("temperature") is not None else ""
        print(f"  seed {r['seed']} ({r['loss']}{ts}, best ep {r['best_epoch']}): "
              f"lmsys_train_top3={r['lmsys_train_top3']:.3f}  "
              f"lmsys_heldout z_logit={ho['z_logit']['median']:.3f}  "
              f"ONYX z_logit={on['z_logit']['median']:.3f} "
              f"({'PASS' if on['z_logit']['median'] and on['z_logit']['median']>=ZLOGIT_GATE else 'fail'})  "
              f"ONYX z_r={on['z_r']['median']:.4f}")
    print()
    if onyx_zlogit_meds:
        print(f"  ONYX z_logit median across seeds: "
              f"{statistics.median(onyx_zlogit_meds):.3f}  "
              f"(per-seed: {['%.3f'%m for m in onyx_zlogit_meds]})")
        print(f"  ONYX z_logit passes {n_onyx_pass}/{len(onyx_zlogit_meds)} seeds  "
              f"-> {'ROBUST PASS' if robust_pass else 'NOT robust'}")
    if lmsys_ho_zlogit_meds:
        print(f"  lmsys held-out z_logit median: "
              f"{statistics.median(lmsys_ho_zlogit_meds):.3f}  "
              f"(task #43 BCE was 0.326 -> "
              f"{'DE-SATURATED' if statistics.median(lmsys_ho_zlogit_meds) >= ZLOGIT_GATE else 'still weak'})")
    print()
    if robust_pass:
        print("  -> CONTRASTIVE GO: the InfoNCE margin loss clears the z_logit gate")
        print("     (>= 2.0) on REAL Onyx serve, robust across seeds, where BCE (task #43)")
        print("     failed (0.258). The de-saturation diagnosis was correct: the per-slot")
        print("     BCE's missing inter-slot contrast (and the bias-collapse cheat) was")
        print("     the root cause; the contrastive is bias-invariant -> forced a real")
        print("     bilinear margin. The flat-readout z_i bilinear IS the ship arch once")
        print("     the loss is fixed. NEXT: re-run the live SERVE gate")
        print("     (probe_strm_selectivity_real.py with the composite wired in).")
    elif n_onyx_pass > 0:
        print("  -> PARTIAL: some seeds pass Onyx z_logit but not robustly. The contrastive")
        print("     de-saturates the margin (more than BCE) but not decisively -- sweep T")
        print("     (smaller T = sharper margin) or train longer before calling the lever.")
    elif lmsys_ho_zlogit_meds and statistics.median(lmsys_ho_zlogit_meds) >= ZLOGIT_GATE:
        print("  -> TRANSFER FAIL (lmsys de-saturated): the contrastive DOES de-saturate")
        print("     lmsys held-out (z_logit >= 2.0, vs BCE's 0.326) but does NOT transfer")
        print("     to Onyx. The loss fix worked in-distribution; the lmsys->Onyx transfer")
        print("     gap remains (conversational context retrieval != Onyx doc recall).")
        print("     Needs real Onyx transcripts, OR accept lmsys as the gate dist.")
    else:
        print("  -> CONTRASTIVE NO-GO: the InfoNCE margin loss does NOT clear z_logit even")
        print("     on lmsys held-out (BCE-equivalent ~0.3). The z_i bilinear genuinely")
        print("     CANNOT separate gold from topically-close fillers with a decisive")
        print("     margin -- the ranking (top-3 0.61) is real but the margin is")
        print("     ARCH-BOUNDED, not loss-bounded. The flat-readout lever is dead;")
        print("     reconsider the cross-slot trajectory Transformer (option 2) or accept")
        print("     the z_i bilinear saturates on serve (option 3).")
    print("=" * 78)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "loss": args.loss, "readout": args.readout,
            "temperature": args.temperature, "weight_decay": args.weight_decay,
            "epochs": args.epochs, "seeds": seeds,
            "n_lmsys": len(lmsys), "n_onyx": len(onyx),
            "onyx_zlogit_median_across_seeds": (statistics.median(onyx_zlogit_meds)
                                                if onyx_zlogit_meds else None),
            "lmsys_heldout_zlogit_median_across_seeds": (statistics.median(lmsys_ho_zlogit_meds)
                                                        if lmsys_ho_zlogit_meds else None),
            "n_onyx_zlogit_pass": n_onyx_pass, "robust_pass": robust_pass,
            "per_seed": [{"seed": seeds[i], "loss": r["loss"],
                          "temperature": r.get("temperature"),
                          "best_epoch": r["best_epoch"],
                          "lmsys_train_top3": r["lmsys_train_top3"],
                          "lmsys_heldout": r["lmsys_heldout"],
                          "onyx": r["onyx"]} for i, r in enumerate(per_seed)],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())