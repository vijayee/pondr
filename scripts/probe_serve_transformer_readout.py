"""Task #39 follow-up: cheap Transformer-readout probe on the captured SERVE
state -- does query-dependent channel attention lift serve top-3 above the task
#38 mean-pool plateau (0.478)?

The task #39 serve-state probe ([[pondr-strm-task39-serve-state-fork-a-
transformer]]) forked FORK A: the mean over the 16 ``d_state`` channels of the
last SSM layer partially CANCELS opposing-sign serve signal (per-channel /
``z_flat_last`` carry 2.4x the mean-pool's across-slot variance). The diagnosed
lever is the STATE-TRAJECTORY TRANSFORMER (attend over the per-channel state, not
the mean-pool). This probe is the cheap confirmation BEFORE the full rewire: it
trains three readouts side-by-side on the SAME captured ``slots_h_raw`` and asks
which one clears the task #38 plateau.

The three readouts all share the SAME bilinear-to-query scoring
``score = (proj_z(slot_rep) . proj_q(query))/sqrt(P) + bias`` (per-slot scalar
logit, BCE with pos_weight, 80/20 split, Wilson gate -- mirroring
``fit_relevance`` so the comparison is fair and (a) reproduces task #38). They
differ ONLY in how the slot representation ``slot_rep [384]`` is built from the
last-layer per-channel state ``[16,384]``:

  (a) MeanPoolLinear  -- ``slot_rep = channels.mean(0)`` (the task #38 z_i).
       Reproduces the 0.478 plateau (sanity / control).
  (b) FlatMLP         -- ``slot_rep = MLP(z_flat_last [6144])``. A learned,
       query-INDEPENDENT channel mix. Tests: does a learned channel-mix already
       beat the mean-pool (a cheaper StateReadout would suffice)?
  (c) ChannelTransformer -- query cross-attends the 16 channel tokens (2 attn
       layers, d=128). A query-DEPENDENT channel mix. Tests: does
       query-conditioned channel selection beat (b)?

The comparison isolates the win:
  (c) >> (b) > (a) -> the Transformer's query-dependent mixing is the win (the
       state-trajectory Transformer is the right lever).
  (b) ~ (c) > (a) -> a learned linear/MLP readout suffices (cheaper than a
       Transformer; the mean-pool was the bottleneck, not the mixing).
  (b) ~ (c) ~ (a) -> the 2.4x per-channel variance is NOT query-relevant; the
       Transformer lever is not validated -- reconsider before the full rewire.

Offline: reads ``traces_serve_identity_hraw.pt`` (carries ``slots_h_raw``
[K,4,16,384] + ``query_emb`` + ``labels`` + ``slots_z`` + ``slots_doc_emb``).
No backbone, no WorkingMemory, no embedder. CPU-fine (114 records, K~14).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.subconscious.training.relevance_training import (  # noqa: E402
    _split_queries,
    _wilson_ci95,
)

LAST_LAYER = -1
D_STATE = 16
D_MODEL = 384
PROJ_DIM = 128


# ── shared bilinear-to-query scorer (mirrors ZRelevanceHead) ──

class _BilinearScore(nn.Module):
    """``score = (proj_z(slot_rep) . proj_q(query))/sqrt(P) + bias`` per slot.

    All three readouts reuse this so the only difference is how ``slot_rep`` is
    built. ``slot_rep`` [K,384], ``query`` [384] (broadcast over K)."""

    def __init__(self, rep_dim: int = D_MODEL, proj_dim: int = PROJ_DIM) -> None:
        super().__init__()
        self.rep_dim = int(rep_dim)
        self.proj_dim = int(proj_dim)
        self.proj_z = nn.Linear(self.rep_dim, self.proj_dim)
        self.proj_q = nn.Linear(D_MODEL, self.proj_dim)
        self.scale = 1.0 / math.sqrt(self.proj_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def score(self, slot_rep: Tensor, query: Tensor) -> Tensor:
        # slot_rep [K, rep_dim], query [384] -> [K]
        zp = self.proj_z(slot_rep)
        qp = self.proj_q(query).unsqueeze(0).expand_as(zp)
        return (zp * qp).sum(-1) * self.scale + self.bias.squeeze(0)


# ── (a) mean-pool baseline (reproduces task #38 z_i) ──

class MeanPoolReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scorer = _BilinearScore(D_MODEL, PROJ_DIM)

    def slot_rep(self, channels: Tensor) -> Tensor:
        # channels [K,16,384] -> mean over the 16 d_state channels -> [K,384]
        return channels.mean(dim=1)

    def logits(self, channels: Tensor, query: Tensor) -> Tensor:
        return self.scorer.score(self.slot_rep(channels), query)


# ── (b) learned flat MLP readout (query-independent channel mix) ──

class FlatMLPReadout(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D_STATE * D_MODEL, hidden), nn.GELU(),
            nn.Linear(hidden, D_MODEL),
        )
        self.scorer = _BilinearScore(D_MODEL, PROJ_DIM)

    def slot_rep(self, channels: Tensor) -> Tensor:
        K = channels.shape[0]
        return self.mlp(channels.reshape(K, D_STATE * D_MODEL))

    def logits(self, channels: Tensor, query: Tensor) -> Tensor:
        return self.scorer.score(self.slot_rep(channels), query)


# ── (c) channel-Transformer readout (query-dependent channel mix) ──

class ChannelTransformerReadout(nn.Module):
    """Query cross-attends the 16 last-layer channel tokens.

    For each of the K slots (batch dim K): the query token [1, d] attends its 16
    channel tokens [16, d] over ``n_layers`` cross-attention blocks; the output
    is projected to [384] as the slot rep. Query-DEPENDENT: different queries
    surface different channels."""

    def __init__(self, d: int = 128, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.d = d
        self.chan_proj = nn.Linear(D_MODEL, d)
        self.query_proj = nn.Linear(D_MODEL, d)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d, D_MODEL)
        self.scorer = _BilinearScore(D_MODEL, PROJ_DIM)

    def slot_rep(self, channels: Tensor, query: Tensor) -> Tensor:
        # channels [K,16,384], query [384]
        K = channels.shape[0]
        kv = self.chan_proj(channels)                          # [K,16,d]
        q = self.query_proj(query).unsqueeze(0).unsqueeze(0).expand(K, 1, self.d)  # [K,1,d]
        x = q
        for attn, norm in zip(self.layers, self.norms):
            a, _ = attn(x, kv, kv, need_weights=False)         # [K,1,d]
            x = norm(x + a)                                    # residual + norm
        return self.out_proj(x.squeeze(1))                     # [K,384]

    def logits(self, channels: Tensor, query: Tensor) -> Tensor:
        return self.scorer.score(self.slot_rep(channels, query), query)


# ── data ──

def load_hraw(traces_path: str) -> list[dict]:
    raw = torch.load(traces_path, map_location="cpu", weights_only=False)
    out = []
    for rec in raw:
        h = rec.get("slots_h_raw")
        if h is None:
            continue
        h = h.float()                                          # [K,4,16,384]
        channels = h[:, LAST_LAYER, :, :]                       # [K,16,384]
        labels = rec["labels"].float()
        if int(labels.sum().item()) == 0 or channels.shape[0] < 3:
            continue
        out.append({
            "channels": channels,            # [K,16,384]
            "query": rec["query_emb"].float(),  # [384]
            "labels": labels,                 # [K]
            "doc": rec["slots_doc_emb"].float(),  # [K,384] (doc-baseline ceiling)
        })
    return out


# ── train + eval (mirrors fit_relevance's gate) ──

def _eval(model, val, device) -> dict:
    """Per-query top-3 recall + Wilson on the val split."""
    model.eval()
    recalls, hits = [], 0
    with torch.no_grad():
        for rec in val:
            ch = rec["channels"].to(device); q = rec["query"].to(device)
            lab = rec["labels"].to(device)
            logits = model.logits(ch, q)
            gold = (lab > 0).nonzero(as_tuple=True)[0].tolist()
            n_gold = len(gold)
            if n_gold == 0:
                continue
            k_top = min(3, logits.shape[0])
            top = set(logits.topk(k_top).indices.tolist())
            n_in = sum(1 for i in gold if i in top)
            recalls.append(n_in / n_gold)
            if n_in == n_gold:
                hits += 1
    if not recalls:
        return {"mean_top3": 0.0, "hit_rate": 0.0, "ci": [0.0, 1.0], "n": 0}
    mean_top3 = sum(recalls) / len(recalls)
    hit_rate = hits / len(recalls)
    ci = _wilson_ci95(hit_rate, len(recalls))
    return {"mean_top3": mean_top3, "hit_rate": hit_rate, "ci": ci, "n": len(recalls)}


def _doc_baseline(val) -> dict:
    """cos(query, slots_doc_emb) top-3 -- the ceiling (label is top-1-cos)."""
    recalls, hits = [], 0
    for rec in val:
        q = rec["query"]; d = rec["doc"]; lab = rec["labels"]
        qn = q / (q.norm() + 1e-9); dn = d / (d.norm(dim=1, keepdim=True) + 1e-9)
        sims = dn @ qn
        gold = (lab > 0).nonzero(as_tuple=True)[0].tolist()
        if not gold:
            continue
        top = set(sims.topk(min(3, sims.shape[0])).indices.tolist())
        n_in = sum(1 for i in gold if i in top)
        recalls.append(n_in / len(gold))
        if n_in == len(gold):
            hits += 1
    if not recalls:
        return {"mean_top3": float("nan"), "hit_rate": float("nan")}
    return {"mean_top3": sum(recalls) / len(recalls),
            "hit_rate": hits / len(recalls), "n": len(recalls)}


def train_one(name: str, model: nn.Module, records: list[dict],
              device, epochs: int = 120, lr: float = 1e-3, wd: float = 1e-2,
              seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_idx, val_idx = _split_queries(len(records), 0.2, seed)
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    # pos_weight = n_neg/n_pos over train slots, capped 14.0 (mirror fit_relevance)
    n_pos = sum(int(r["labels"].sum().item()) for r in train)
    n_neg = sum(int((1 - r["labels"]).sum().item()) for r in train)
    pw = min(n_neg / max(n_pos, 1), 14.0)
    pos_weight = torch.tensor([pw], device=device)
    rng = random.Random(seed)
    best, best_epoch, best_pc = -1.0, -1, None
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train))); rng.shuffle(order)
        tot = 0.0
        for qi in order:
            rec = train[qi]
            ch = rec["channels"].to(device); q = rec["query"].to(device)
            lab = rec["labels"].to(device)
            logits = model.logits(ch, q)
            loss = F.binary_cross_entropy_with_logits(
                logits, lab, pos_weight=pos_weight, reduction="mean")
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        pc = _eval(model, val, device)
        if pc["mean_top3"] >= best:
            best = pc["mean_top3"]; best_epoch = epoch; best_pc = pc
    return {"name": name, "best_epoch": best_epoch, "best_pc": best_pc,
            "n_train": len(train), "n_val": len(val), "pos_weight": pw,
            "n_params": sum(p.numel() for p in model.parameters())}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Task #39 follow-up: cheap Transformer-readout probe on the "
                    "captured SERVE state -- does query-dependent channel "
                    "attention beat the mean-pool plateau (0.478)?")
    p.add_argument("--traces", required=True,
                   help="serve traces with slots_h_raw (probe_strm_selectivity_real.py "
                        "--emit-traces --emit-raw-state).")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=0,
                   help="train/val split seed + init seed (for multi-seed sweeps)")
    p.add_argument("--only", default="",
                   help="comma-separated subset of models to run: a,b,c (default all)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="")
    args = p.parse_args()

    if not Path(args.traces).exists():
        print(f"ERROR: traces not found at {args.traces}", file=sys.stderr)
        return 1
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"device: {device}", flush=True)

    records = load_hraw(args.traces)
    if len(records) < 5:
        print(f"ERROR: only {len(records)} usable records (need >=5)", file=sys.stderr)
        return 1
    print(f"loaded {len(records)} records (K median "
          f"{int(sorted(r['channels'].shape[0] for r in records)[len(records)//2])})",
          flush=True)

    # doc-baseline ceiling (the label is top-1-cos -> cos-ranker trivially ranks
    # gold #1, so this is ~1.0; shown for scale).
    _, val_idx = _split_queries(len(records), 0.2, args.seed)
    doc_base = _doc_baseline([records[i] for i in val_idx])
    print(f"doc-baseline ceiling (val): top-3 = {doc_base['mean_top3']:.3f}  "
          f"(label is top-1-cos, so a cos-ranker hits ~1.0)", flush=True)
    print()

    all_models = (
        ("(a) MeanPoolLinear  (task #38 z_i)", "a", MeanPoolReadout()),
        ("(b) FlatMLP          (learned mix)", "b", FlatMLPReadout()),
        ("(c) ChannelTransformer(q-dep mix)", "c", ChannelTransformerReadout()),
    )
    only = set(args.only.split(",")) if args.only else {"a", "b", "c"}
    results = []
    for name, key, model in all_models:
        if key not in only:
            continue
        print(f"training {name} ...", flush=True)
        r = train_one(name, model, records, device, epochs=args.epochs, seed=args.seed)
        pc = r["best_pc"]
        results.append(r)
        print(f"  {name}: best top3 = {pc['mean_top3']:.3f} @ ep {r['best_epoch']}  "
              f"hit {pc['hit_rate']:.2f} ci[{pc['ci'][0]:.2f},{pc['ci'][1]:.2f}]  "
              f"(n={pc['n']}, params={r['n_params']:,}, pw={r['pos_weight']:.1f})",
              flush=True)
        print(f"    task #38 plateau = 0.478  gate = 0.6  -> "
              f"{'CLEARS plateau' if pc['mean_top3'] > 0.478 else 'at/below plateau'}"
              f"{', CLEARS gate' if pc['mean_top3'] >= 0.6 else ''}", flush=True)
        print()

    # ── verdict ──
    # results preserves the a,b,c order filtered by --only; map back by name tag
    name_to_key = {"(a)": "a", "(b)": "b", "(c)": "c"}
    by_key = {}
    for r in results:
        tag = r["name"].split()[0]  # "(a)" / "(b)" / "(c)"
        by_key[name_to_key.get(tag, tag)] = r["best_pc"]["mean_top3"]
    a = by_key.get("a", float("nan"))
    b = by_key.get("b", float("nan"))
    c = by_key.get("c", float("nan"))
    print("=" * 72)
    print("VERDICT (task #39 Transformer-readout probe)")
    print("=" * 72)
    print(f"  (a) mean-pool   top3 = {a:.3f}  (task #38 plateau 0.478)")
    print(f"  (b) flat MLP    top3 = {b:.3f}  (learned channel mix)")
    print(f"  (c) Transformer top3 = {c:.3f}  (query-dependent channel mix)")
    print(f"  doc ceiling    top3 = {doc_base['mean_top3']:.3f}")
    print()
    import math as _m
    _nan = _m.isnan
    have_all = not (_nan(a) or _nan(b) or _nan(c))
    if have_all and c > max(a, b) + 0.03:
        print("  -> TRANSFORMER WINS: query-dependent channel attention lifts serve")
        print("     top-3 above the mean-pool AND the learned flat mix. The")
        print("     state-trajectory Transformer is the right lever -- the win is")
        print("     the ATTENTION MECHANISM (query-conditioned channel selection),")
        print("     not just more input dims. Proceed to the full rewire.")
    elif have_all and b > a + 0.03 and c <= b + 0.03:
        print("  -> FLAT MLP SUFFICES: a learned channel-mix beats the mean-pool but")
        print("     the Transformer adds nothing over it. The mean-pool was the")
        print("     bottleneck (a cheaper StateReadout, not a full Transformer).")
        print("     Reconsider whether the full state-trajectory Transformer rewire")
        print("     is warranted vs a learned StateReadout on z_flat_last.")
    elif have_all and max(b, c) <= a + 0.03:
        print("  -> NO WIN: neither the flat mix nor the Transformer beats the")
        print("     mean-pool. The 2.4x per-channel variance is NOT query-relevant")
        print("     (a learned readout can't turn it into top-3). The Transformer")
        print("     lever is NOT validated by this probe -- reconsider before the")
        print("     full rewire (e.g. cross-slot trajectory attention, not per-slot")
        print("     channel attention; or backbone-on-serve retrain after all).")
    elif not have_all and not _nan(b) and not _nan(a) and b > a + 0.03:
        print("  -> FLAT MLP > MEAN-POOL (a,b only): a learned channel-mix beats the")
        print("     mean-pool. The mean-pool was the bottleneck.")
    else:
        print("  -> AMBIGUOUS: small differences; rerun with more seeds / epochs")
        print("     before committing.")
    print("=" * 72)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "n_records": len(records), "doc_baseline": doc_base,
            "results": [{"name": r["name"], "best_top3": r["best_pc"]["mean_top3"],
                         "best_epoch": r["best_epoch"], "ci": r["best_pc"]["ci"],
                         "hit_rate": r["best_pc"]["hit_rate"],
                         "n_params": r["n_params"], "pos_weight": r["pos_weight"],
                         "n_train": r["n_train"], "n_val": r["n_val"]}
                        for r in results],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())