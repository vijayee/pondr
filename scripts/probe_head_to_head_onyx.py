"""Task #45: the DeepSeek-v4-pro head-to-head on REAL Onyx serve traces.

Task #44 ([[pondr-strm-task44-contrastive-partial-desat-transfer-fail]]) showed
the contrastive InfoNCE loss PARTIALLY de-saturates the per-slot z_i bilinear
(lmsys held-out z_logit 0.931, ~3x BCE) but the MEDIAN stays sub-gate (arch
margin bounded ~1.0) AND lmsys->Onyx transfer is WORSE under contrastive
(0.048 vs BCE 0.258 -- bias-invariance removes the bias as a distribution-shift
absorber). The flat-readout z_i bilinear is NOT the ship lever even with the
loss fix. DeepSeek-v4-pro (consulted 2026-07-21) diagnosed the root cause
mechanistically: a POINTWISE bilinear scores each slot INDEPENDENTLY against the
query, so it can only produce an ABSOLUTE relevance (sim-to-query). The 2.0
z_logit gate is a RELATIVE margin (gold logit - mean filler logit), and on
serve the fillers are topically close -> their absolute sims are ALL high ->
the bilinear's absolute score cannot push a 2.0 relative gap. A CROSS-SLOT
attention head can implement relative scoring (each slot's logit attends to
the query AND to all other slots -> score ~ sim-to-query - mean sim of all
slots, which DeepSeek said is the mechanism by which it escapes the margin
bound the pointwise bilinear cannot). This probe is the decisive A/B test.

**Head A** = the current ``CompositeZHead`` (``StateReadout`` mlp128 [6144->384]
+ ``ZRelevanceHead`` bilinear ``proj_z(z_i).proj_q(q)/sqrt(P) + bias``), the
task #44 arch. **Head B** = a minimal cross-slot Transformer: the SAME
``StateReadout`` mlp128 -> per-slot z_i [K,384] (so the ONLY difference from
Head A is the cross-slot attention vs the pointwise bilinear -- a win/loss is
cleanly attributable to the cross-slot mechanism, not the readout), + a learned
positional embedding, + the query as a [CLS] token prepended, + a 2-layer /
4-head / hidden-256 / FFN-512 Transformer encoder, + a per-slot logit head on
each slot's encoder output. SAME contrastive InfoNCE loss (T=1.0), SAME frozen
``backbone_v2_full.pt`` SSM (the traces already carry its ``slots_h_raw``),
SAME eval (``p41._zr_and_logit_gaps`` -> per-source z_logit gap, 2.0 gate).

**Data.** ``traces_onyx_serve_hraw.pt`` (task #45, 1012 turns from 76 REAL
Onyx sessions fetched via cookie auth -- in-distribution, no lmsys transfer
confound). Split by SESSION (not by query): held-out = ENTIRE unseen
conversations, the true "generalizes to new Onyx" test DeepSeek specified.
3-seed robustness.

**Decision rule (DeepSeek):**
  * Head A clears the 2.0 z_logit gate on held-out Onyx (robust, >= 2/3 seeds)
    -> SHIP the bilinear; the loss was the only blocker.
  * Head A fails BUT Head B clears -> the cross-slot Transformer IS the lever;
    invest there next (scale + wire into the live serve probe).
  * NEITHER clears -> ABANDON the state-trajectory-locator (option C): the
    state path has been tested 5 ways and saturates; the bge 2a head (0.889
    train) already works.

**Isolation (binding constraint).** Standalone script; Head B is a LOCAL
nn.Module defined here (zero ``src/`` changes -> every existing head + the 2b
gate stay byte-identical by construction). Reuses ``contrastive_loss`` +
``_to_device`` from ``probe_contrastive_zlogit`` (task #44, committed) and
``p41._zr_and_logit_gaps`` + ``p41._load_serve_traces`` (task #41). Does NOT
call ``fit_relevance``. No live wiring, no HF upload (diagnostic; the Onyx
traces are PRIVATE chat data, local + gitignored, never uploaded per user
directive).
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))        # sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import probe_serve_composite_zrgate as p41  # noqa: E402
import probe_contrastive_zlogit as p44  # noqa: E402
from src.subconscious.state_readout import (  # noqa: E402
    DEFAULT_DIM_IN,
    CompositeZHead,
    StateReadout,
)
from src.subconscious.training.relevance_training import (  # noqa: E402
    RelevanceTrainingConfig,
    _gate_score,
    _split_queries,
    evaluate_relevance,
)
from src.subconscious.z_relevance_head import (  # noqa: E402
    PROJ_DIM as Z_PROJ_DIM,
    QUERY_DIM as Z_QUERY_DIM,
    SLOT_DIM as Z_SLOT_DIM,
    Z_DIM,
)

DEFAULT_ONYX = "data/training/strm_relevance/traces_onyx_serve_hraw.pt"
DEFAULT_OUT = "data/training/strm_relevance/head_to_head_onyx.json"
DEFAULT_CKPT_ROOT = "data/training/strm_state_readout/head_to_head_onyx"

ZLOGIT_GATE = 2.0


# ── Head B: the cross-slot Transformer (local module, zero src/ changes) ──

class CrossSlotTransformerZHead(nn.Module):
    """Cross-slot attention relevance head -- the DeepSeek option B.

    Same per-slot readout as ``CompositeZHead`` (``StateReadout`` mlp128
    [dim_in -> 384]), so the ONLY difference from Head A is the SCORING: the
    bilinear's pointwise ``proj_z(z_i).proj_q(q)`` is replaced by a Transformer
    encoder that cross-attends the query (prepended as a [CLS] token) against
    all K slots, then a per-slot logit head reads each slot's encoder output.
    The attention lets slot k's logit depend on the query AND on every other
    slot -> a RELATIVE score (sim-to-query attenuated by the candidate pool),
    the mechanism DeepSeek identified as escaping the pointwise margin bound.

    Single-record interface (matches ``CompositeZHead.logits``): one record's K
    slots at a time, no batching/padding. ``logits(slot_y, slot_signal, q)``
    returns ``[K, 1]`` (so ``.squeeze(-1)`` -> ``[K]``, the contract
    ``p41._zr_per_slot`` + the contrastive loop assume). ``slot_y`` is accepted
    and ignored (the pure-z_i test, same as the composite). Exposes
    ``slot_dim``/``query_dim``/``proj_dim``/``doc_dim`` so the contrastive loop's
    checkpoint-dim reads (modeled on ``_train_contrastive``) are shape-consistent.
    """

    def __init__(self, dim_in: int = DEFAULT_DIM_IN, hidden: int | None = 128,
                 d_model: int = Z_DIM, n_heads: int = 4, n_layers: int = 2,
                 ffn: int = 512, max_pos: int = 64) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.readout = StateReadout(dim_in=self.dim_in, dim_out=d_model, hidden=hidden)
        self.d_model = int(d_model)
        self.max_pos = int(max_pos)
        # Learned positional embedding for the K slot tokens (positions 1..K; the
        # query [CLS] takes position 0 -- a learned token, not the query emb).
        self.pos_emb = nn.Parameter(torch.randn(self.max_pos, d_model) * 0.02)
        # Learned [CLS] query token; the actual query emb is projected + added so
        # the encoder's query token carries BOTH a learned slot and the live query.
        self.cls_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.query_proj = nn.Linear(d_model, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.logit_head = nn.Linear(d_model, 1)
        # Mirror CompositeZHead's checkpoint dims (the contrastive loop reads
        # these; doc_dim = the readout input dim so a loader can rebuild it).
        self.slot_dim = Z_SLOT_DIM
        self.query_dim = Z_QUERY_DIM
        self.proj_dim = Z_PROJ_DIM
        self.doc_dim = self.dim_in

    def logits(self, slot_y: torch.Tensor, slot_signal: torch.Tensor,
               query_emb: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid relevance logit per slot -> ``[K, 1]``.

        ``slot_signal`` is the raw flattened state ``[K, dim_in]`` (or
        ``[dim_in]``); ``slot_y`` accepted + ignored (pure-z_i test).
        """
        z = self.readout(slot_signal)                       # [K, d_model]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        K = z.shape[0]
        assert K < self.max_pos, f"K={K} exceeds max_pos={self.max_pos}"
        z = z + self.pos_emb[1:K + 1]                        # [K, d_model]
        q = self.query_proj(query_emb.to(torch.float32))     # [d_model]
        cls = self.cls_token[0] + q                          # [d_model] (pos 0)
        seq = torch.cat([cls.unsqueeze(0), z], dim=0).unsqueeze(0)  # [1, 1+K, d]
        out = self.encoder(seq)                              # [1, 1+K, d_model]
        slot_out = out[0, 1:, :]                             # [K, d_model]
        return self.logit_head(slot_out)                     # [K, 1]

    def predict(self, slot_y, slot_signal, query_emb):
        return torch.sigmoid(self.logits(slot_y, slot_signal, query_emb))

    forward = predict


def _build_head(arch: str, dim_in: int, hidden: int | None) -> nn.Module:
    """Construct Head A (bilinear composite) or Head B (cross-slot Transformer)."""
    if arch == "bilinear":
        return CompositeZHead(dim_in=dim_in, hidden=hidden)
    if arch == "transformer":
        return CrossSlotTransformerZHead(dim_in=dim_in, hidden=hidden)
    raise ValueError(f"unknown arch {arch!r}")


def _load_head(arch: str, ckpt_path: str, dim_in: int, hidden: int | None,
               device: str) -> nn.Module:
    """Reload a trained head from its checkpoint (best.pt)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["head"] if isinstance(ckpt, dict) and "head" in ckpt else ckpt
    head = _build_head(arch, dim_in, hidden)
    head.load_state_dict(sd)
    dev = torch.device(device)
    return head.to(dev).eval()


# ── session-level split (held-out = unseen conversations) ──

def _session_of(rec: dict) -> str:
    """The session a record belongs to (all source_ids in a record share the
    same Onyx session UUID prefix, ``{session_id}#{msg_idx}``)."""
    return str(rec["source_ids"][0]).split("#", 1)[0]


def _session_split(records: list[dict], val_fraction: float, seed: int):
    """Split records by SESSION so held-out turns are ENTIRE unseen
    conversations (the true generalization test; a query-split would leak a
    session's turns into both halves). Returns (train, val, val_sessions)."""
    sessions = sorted({_session_of(r) for r in records})
    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_sessions = set(shuffled[:n_val])
    train = [r for r in records if _session_of(r) not in val_sessions]
    val = [r for r in records if _session_of(r) in val_sessions]
    return train, val, sorted(val_sessions)


# ── the contrastive training loop (generic over Head A / Head B) ──

def _train_head(arch: str, train: list[dict], val: list[dict], hidden: int | None,
                temperature: float, epochs: int, lr: float, weight_decay: float,
                accum_steps: int, seed: int, device: str, ckpt_dir: Path) -> dict:
    """Train one head (bilinear OR transformer) on the train sessions with the
    SAME contrastive InfoNCE loss. Mirrors ``p44._train_contrastive`` but is
    generic over the head arch. Uses ``evaluate_relevance`` on the train-internal
    query-val split for the per-epoch TRAIN top-3 gate + best-ckpt selection
    (this is a TRAINING signal, not the final held-out eval). Writes best.pt."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    dim_in = int(train[0]["slots_h_raw"].shape[1])
    head = _build_head(arch, dim_in, hidden).to(dev)
    n_params = sum(p.numel() for p in head.parameters())
    arch_name = (f"MLP-{hidden}" if hidden else "Linear") if arch == "bilinear" \
        else f"Transformer({'MLP-'+str(hidden) if hidden else 'Linear'} readout)"
    print(f"\ntraining {arch} seed={seed} CONTRASTIVE ({arch_name} {dim_in}->384, "
          f"{n_params:,} params, T={temperature}, wd={weight_decay}, {epochs} "
          f"epochs) -> {ckpt_dir}", flush=True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    # Per-epoch TRAIN gate: query-split WITHIN train (mirrors _train_contrastive).
    train_idx, valq_idx = _split_queries(len(train),
                                         RelevanceTrainingConfig().val_fraction, seed)
    train_q = [train[i] for i in train_idx]
    valq = [train[i] for i in valq_idx]
    print(f"  train sessions -> {len(train_q)} train / {len(valq)} query-val "
          f"(held-out: {len(val)} turns from {len({_session_of(r) for r in val})} "
          f"unseen sessions)", flush=True)

    rng = random.Random(seed)
    best_score: tuple | None = None
    best_pc: dict | None = None
    best_epoch = -1
    last_pc: dict | None = None
    last_go = False
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accum = max(1, accum_steps)
    ck_slot_dim = int(head.slot_dim)
    ck_doc_dim = int(head.doc_dim)
    ck_query_dim = int(head.query_dim)
    ck_proj_dim = int(head.proj_dim)

    for epoch in range(epochs):
        head.train()
        order = list(range(len(train_q)))
        rng.shuffle(order)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for k, qi in enumerate(order):
            rec = train_q[qi]
            z_flat = rec["slots_h_raw"].to(dev).to(torch.float32)      # [K,6144]
            q = rec["query_emb"].to(dev).to(torch.float32)            # [384]
            labels = rec["labels"].to(dev).to(torch.float32)          # [K]
            K = z_flat.shape[0]
            slot_y = torch.zeros(K, ck_slot_dim, device=dev)
            logits = head.logits(slot_y, z_flat, q).squeeze(-1)       # [K]
            gold = labels > 0
            loss = p44.contrastive_loss(logits, gold, temperature) / accum
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
        pc = evaluate_relevance(head, valq, slot_signal_field="slots_h_raw")
        last_pc = pc
        last_go = (pc["mean_top3_recall"] >= RelevanceTrainingConfig().gate_top3
                   and pc["hit_ci95"][0] > RelevanceTrainingConfig().gate_wilson_low)
        ci = pc["hit_ci95"]
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} "
              f"top3={pc['mean_top3_recall']:.3f} hit={pc['hit_rate']:.2f} "
              f"ci=[{ci[0]:.2f},{ci[1]:.2f}] r_pos={pc['mean_r_positive']:.3f} "
              f"{'GO' if last_go else 'no-go'}", flush=True)

        score = _gate_score(pc, RelevanceTrainingConfig())
        if best_score is None or score > best_score:
            best_score = score
            best_pc = pc
            best_epoch = epoch
            torch.save({"head": head.state_dict(), "arch": arch,
                        "slot_dim": ck_slot_dim, "doc_dim": ck_doc_dim,
                        "query_dim": ck_query_dim, "proj_dim": ck_proj_dim,
                        "hidden": hidden, "top3_recall": pc["mean_top3_recall"],
                        "hit_rate": pc["hit_rate"], "hit_ci95": pc["hit_ci95"],
                        "go": last_go, "epoch": epoch, "loss": "contrast"},
                       ckpt_dir / "best.pt")

    if last_pc is not None:
        torch.save({"head": head.state_dict(), "arch": arch,
                    "slot_dim": ck_slot_dim, "doc_dim": ck_doc_dim,
                    "query_dim": ck_query_dim, "proj_dim": ck_proj_dim,
                    "hidden": hidden, "top3_recall": last_pc["mean_top3_recall"],
                    "hit_rate": last_pc["hit_rate"], "hit_ci95": last_pc["hit_ci95"],
                    "go": last_go, "epoch": epochs - 1, "loss": "contrast"},
                   ckpt_dir / "final.pt")
    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "arch": arch, "loss": "contrast",
                   "temperature": temperature, "seed": seed,
                   "n_train": len(train_q), "n_valq": len(valq),
                   "n_heldout": len(val), "best_scorecard": best_pc}, f, indent=2)

    return {"arch": arch, "arch_name": arch_name, "hidden": hidden,
            "dim_in": dim_in, "n_params": n_params, "best_epoch": best_epoch,
            "train_top3": best_pc["mean_top3_recall"] if best_pc else 0.0,
            "train_go": last_go, "ckpt": ckpt_dir / "best.pt"}


def _run_arch(arch: str, train: list[dict], val: list[dict], hidden: int | None,
              temperature: float, weight_decay: float, epochs: int, lr: float,
              accum_steps: int, seed: int, device: str, ckpt_root: Path) -> dict:
    """Train one arch (one seed), eval on held-out sessions + all-turns ceiling."""
    ckpt_dir = ckpt_root / f"{arch}_s{seed}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    r = _train_head(arch, train, val, hidden, temperature, epochs, lr,
                    weight_decay, accum_steps, seed, device, ckpt_dir)
    head = _load_head(arch, str(r["ckpt"]), r["dim_in"], hidden, device)
    # The decisive eval: z_r + z_logit gaps on HELD-OUT sessions (unseen convs).
    r["heldout"] = p41._zr_and_logit_gaps(head, val, device)
    # All-turns ceiling (in-sample upper bound -- if even this fails, no signal).
    r["allturns"] = p41._zr_and_logit_gaps(head, train + val, device)
    r["seed"] = seed
    ho = r["heldout"]
    on = r["allturns"]
    print(f"  [{arch} s{seed}] HELD-OUT z_logit={ho['z_logit']['median']:.3f} "
          f"(n_ge_2.0={ho['z_logit']['n_ge_gate']}/{ho['z_logit']['n_eligible']}, "
          f"{'PASS' if ho['z_logit']['median'] is not None and ho['z_logit']['median']>=ZLOGIT_GATE else 'fail'})  "
          f"z_r={ho['z_r']['median']:.4f}  |  ALL-TURNS z_logit={on['z_logit']['median']:.3f}",
          flush=True)
    return r


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task #45: Head A (bilinear) vs Head B (cross-slot "
                    "Transformer) head-to-head on REAL Onyx serve traces.")
    p.add_argument("--onyx", default=DEFAULT_ONYX,
                   help="Onyx serve traces (generate_onyx_serve_traces.py).")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated train seeds (robustness sweep).")
    p.add_argument("--readout", default="mlp128",
                   choices=["linear", "mlp64", "mlp128"],
                   help="StateReadout arch (shared by Head A + Head B so the "
                        "win is attributable to the cross-slot mechanism).")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="fraction of SESSIONS held out (unseen conversations).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--device", default="cpu",
                   help="train+eval device (cuda trains much faster).")
    p.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    if not Path(args.onyx).exists():
        print(f"ERROR: onyx traces not found at {args.onyx}\n"
              f"  run: python scripts/generate_onyx_serve_traces.py",
              file=sys.stderr)
        return 1

    records = p41._load_serve_traces(args.onyx)
    records = p44._to_device(records, args.device)
    if len(records) < 100:
        print(f"ERROR: only {len(records)} onyx records (need >=100 for a real "
              f"session-split head-to-head)", file=sys.stderr)
        return 1
    ok = sorted(r["slots_h_raw"].shape[0] for r in records)
    print(f"onyx: {len(records)} turns (K min/med/max={ok[0]}/{ok[len(ok)//2]}/{ok[-1]}), "
          f"dim_in={records[0]['slots_h_raw'].shape[1]}", flush=True)

    hidden = {"linear": None, "mlp64": 64, "mlp128": 128}[args.readout]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ckpt_root = Path(args.ckpt_root)

    # Session split is seed-dependent (which sessions are held out varies) --
    # re-split per seed so each seed sees a fresh generalization test.
    per_arch = {"bilinear": [], "transformer": []}
    for seed in seeds:
        train, val, val_sessions = _session_split(records, args.val_fraction, seed)
        print(f"\n=== seed {seed}: {len(train)} train / {len(val)} held-out turns "
              f"({len(val_sessions)} unseen sessions) ===", flush=True)
        for arch in ("bilinear", "transformer"):
            per_arch[arch].append(_run_arch(
                arch, train, val, hidden, args.temperature, args.weight_decay,
                args.epochs, args.lr, args.accum_steps, seed, args.device, ckpt_root))

    # ── aggregate + DeepSeek decision rule ──
    def _med(rows, key, sub):
        vals = [r[key][sub]["median"] for r in rows
                if r[key][sub]["median"] is not None]
        return statistics.median(vals) if vals else None

    def _npass(rows, key, sub):
        vals = [r[key][sub]["median"] for r in rows
                if r[key][sub]["median"] is not None]
        return sum(1 for m in vals if m is not None and m >= ZLOGIT_GATE)

    a_held = _med(per_arch["bilinear"], "heldout", "z_logit")
    b_held = _med(per_arch["transformer"], "heldout", "z_logit")
    a_pass = _npass(per_arch["bilinear"], "heldout", "z_logit")
    b_pass = _npass(per_arch["transformer"], "heldout", "z_logit")
    a_robust = a_pass >= 2 and a_pass * 2 >= len(seeds)
    b_robust = b_pass >= 2 and b_pass * 2 >= len(seeds)

    print("\n" + "=" * 80)
    print("VERDICT (task #45: Head A bilinear vs Head B cross-slot Transformer)")
    print("=" * 80)
    print(f"  readout={args.readout}  T={args.temperature}  wd={args.weight_decay}  "
          f"seeds={seeds}  val_fraction={args.val_fraction} (session split)  "
          f"z_logit gate={ZLOGIT_GATE}")
    print(f"  traces: {len(records)} real Onyx serve turns "
          f"(task #44 lmsys->Onyx transfer was 0.048; this is in-distribution)")
    print()
    for arch, rows in (("bilinear (A)", per_arch["bilinear"]),
                       ("transformer (B)", per_arch["transformer"])):
        for r in rows:
            ho = r["heldout"]
            print(f"  {arch} s{r['seed']} (best ep {r['best_epoch']}, "
                  f"train_top3={r['train_top3']:.3f}): "
                  f"HELD-OUT z_logit={ho['z_logit']['median']:.3f} "
                  f"({'PASS' if ho['z_logit']['median'] and ho['z_logit']['median']>=ZLOGIT_GATE else 'fail'})  "
                  f"z_r={ho['z_r']['median']:.4f}")
        hm = _med(rows, "heldout", "z_logit")
        am = _med(rows, "allturns", "z_logit")
        n = _npass(rows, "heldout", "z_logit")
        print(f"  -> {arch}: held-out z_logit median={hm if hm is not None else 'n/a':>5} "
              f"({n}/{len(seeds)} pass)  all-turns={am if am is not None else 'n/a':>5}")
    print()
    print(f"  Head A (bilinear):    held-out z_logit {a_held if a_held is not None else 'n/a'} "
          f"-> {a_pass}/{len(seeds)} pass -> {'ROBUST PASS' if a_robust else 'NOT robust'}")
    print(f"  Head B (transformer): held-out z_logit {b_held if b_held is not None else 'n/a'} "
          f"-> {b_pass}/{len(seeds)} pass -> {'ROBUST PASS' if b_robust else 'NOT robust'}")
    print()
    print("  DECISION RULE (DeepSeek):")
    if a_robust:
        print("  -> SHIP THE BILINEAR (Head A). The contrastive loss on real Onyx")
        print("     clears the 2.0 gate held-out -> the loss was the only blocker;")
        print("     the flat-readout z_i bilinear IS the ship arch. NEXT: re-run")
        print("     the live SERVE gate (probe_strm_selectivity_real.py wired in).")
    elif b_robust:
        print("  -> CROSS-SLOT TRANSFORMER IS THE LEVER (Head B). The pointwise")
        print("     bilinear (Head A) fails the gate but cross-slot attention clears")
        print("     it -> DeepSeek's relative-scoring mechanism was right: attention")
        print("     escapes the pointwise margin bound. NEXT: scale Head B + wire")
        print("     into the live serve probe; tune depth/heads/temperature.")
    else:
        a_all = _med(per_arch["bilinear"], "allturns", "z_logit")
        b_all = _med(per_arch["transformer"], "allturns", "z_logit")
        print("  -> NEITHER CLEARS (option C: abandon the state-trajectory-locator).")
        print(f"     Head A held-out {a_held} / all-turns ceiling {a_all}; "
              f"Head B held-out {b_held} / all-turns ceiling {b_all}.")
        if b_all is not None and (a_all is None or b_all > a_all):
            print("     NOTE: Head B lifts the all-turns ceiling over Head A -> the")
            print("     cross-slot mechanism shows a SIGNAL even if it doesn't clear")
            print("     the 2.0 gate held-out; a larger trace set or a deeper/wider")
            print("     Transformer might clear it. Flag for user judgment (B-pilot).")
        else:
            print("     The state path has now been tested 5 ways (mean-pool, flat")
            print("     BCE, flat BCE + 55x lmsys, flat contrastive, cross-slot")
            print("     Transformer) and saturates on serve. The bge 2a head (0.889")
            print("     train) already works; accept the SSM state does not beat bge")
            print("     for relevance and stop investing in the state-trajectory lever.")
    print("=" * 80)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "readout": args.readout, "temperature": args.temperature,
            "weight_decay": args.weight_decay, "epochs": args.epochs,
            "seeds": seeds, "val_fraction": args.val_fraction,
            "n_records": len(records),
            "bilinear_heldout_zlogit_median": a_held,
            "transformer_heldout_zlogit_median": b_held,
            "bilinear_pass": a_pass, "transformer_pass": b_pass,
            "bilinear_robust": a_robust, "transformer_robust": b_robust,
            "per_seed": {
                "bilinear": [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                              "train_top3": r["train_top3"],
                              "heldout": r["heldout"], "allturns": r["allturns"]}
                             for r in per_arch["bilinear"]],
                "transformer": [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                                 "train_top3": r["train_top3"],
                                 "heldout": r["heldout"], "allturns": r["allturns"]}
                                for r in per_arch["transformer"]]},
        }, indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())