"""Train + gate the R2 fill-holes readout (task #35 Stage 2/3).

Stage 2: train ``FillHolesReadout`` (``src/subconscious/fill_holes_readout.py``)
to recover a faded anchor's bge from MEMORY (the degraded SSM-A state + the
recent ring) with the JEPA-fade InfoNCE objective (``jepa_infonce_loss``). Stage
3: the four-metric gate that decides whether R2 wires in (Stage 4) or fails
honestly (keep R3+R4).

## The data

Stream Bible anchors (domain A, John 1) + ERAG technical docs (domain B)
through the REAL ``FadeMemory`` (real bge, real ``VectorCarrySSM``) -- the
cross-domain stream that ``probe_r2_band.py`` showed reaches the R2/R4 band
(cos < cos_gist). At every step, for each EVICTED anchor (lag > ring_capacity),
capture a training triple:

    (state = ssm_a.state(),                    # the degraded bge-vector channel
     ring_bges = blurbs.vector(aid) for aid in ring,   # the recent context
     target = blurbs.vector(anchor))           # the anchor's OWN bge (the address)

Train/eval split by ANCHOR (the readout must generalize to recovering anchors
it was not trained on -- the primary held-out; the ERAG stream is the
interferer, not the target): anchors 0..7 train, anchors 8..15 eval. The
negative pool is all anchor bges in the store EXCEPT the eval anchors (so the
readout never learns to AVOID an anchor it must later recover); sampled to 64
per batch. The positive-in-negatives redundancy (a train anchor's bge is also in
the pool) is the standard in-batch InfoNCE minor redundancy, accepted.

## The gate (Stage 3, all four must pass)

On the eval anchors' R2-band triples (evicted AND cos_raw < cos_gist):

  1. vs raw state   : readout top-1 >= raw-state top-1 + 0.15, AND
                      mean cos(recovered, stored) - cos(raw_state, stored) >= 0.10.
  2. vs ring-only   : readout top-1 >= ring-only-closed-form top-1 + 0.05
                      (else ship the closed-form instead -- a SUCCESS for
                      R2-the-feature, a FAIL for R2-the-Transformer-readout).
  3. no-collapse    : readout top-1 >= 5x the in-batch-negative retrieval rate.
  4. content (9455795): the recovered vector retrieves the anchor's OWN blurb
                      (top-1 == anchor_id), NOT a cross-domain/same-domain
                      sibling. Drift = FAIL.

PASS -> save the checkpoint (Stage 4 wires it in). FAIL -> no save, keep
R3+R4, write the honest-negative doc. A FAIL's checkpoint is NOT uploaded
(the policy).

Standalone, CPU-runnable (bge + a small 2-layer Transformer, minutes on CPU).
Reads the gitignored untracked ERAG parquet locally (public ERAG only). Bible
OEB-US via bible-api.com (public domain).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # repo root

import eval_fade_bible as bib  # noqa: E402
import eval_fade_cross_domain as cd  # noqa: E402

from src.subconscious.fade import FadeConfig, FadeMemory, bge_embedder
from src.subconscious.fill_holes_readout import (
    FillHolesConfig,
    FillHolesReadout,
)
from src.subconscious.jepa_gist import jepa_infonce_loss

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_OUTPUT_DIR = "data/probe/r2_readout"
DEFAULT_BOOK, DEFAULT_CHAPTER = "john", 1
DEFAULT_CKPT = "data/r2_readout/best.pt"


# -------------------------------------------------------------- closed-form
def _ring_only_unfade(state: np.ndarray, ring_bges: np.ndarray,
                      anchor_id: int, ring_ids: list[int], decay: float,
                      write_gate: float, T: int) -> np.ndarray:
    """The linear baseline (``probe_r2_band._closed_form`` ring-only). Returns
    the L2-normalized recovered bge. Used for gate metric 2 (the readout must
    beat the closed-form)."""
    w_anchor = (decay ** (T - anchor_id)) * write_gate
    if w_anchor == 0.0:
        return np.zeros_like(state)
    sub = np.zeros_like(state)
    for k, aid in enumerate(ring_ids):
        if aid == anchor_id:
            continue
        sub += (decay ** (T - aid)) * write_gate * ring_bges[k]
    rec = (state - sub) / w_anchor
    n = float(np.linalg.norm(rec))
    return (rec / n).astype(np.float32) if n else rec.astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _top1_on(triples: list[dict], readout, mem, dev) -> tuple[int, float, float, float]:
    """Readout + raw top-1, and mean cos delta, on a triple list.

    The overfitting-vs-impossibility diagnostic: if the readout recovers TRAIN
    anchors (top-1 > 0) but not EVAL anchors, R2 is learnable but fails to
    generalize (overfitting -- a less damning negative, might improve with more
    anchors/data). If TRAIN top-1 is also ~0, recovery is not learnable at this
    fade depth at all (impossibility -- the definitive negative)."""
    readout.eval()
    t1r = t1raw = 0
    dcos = 0.0
    with torch.no_grad():
        for t in triples:
            target = t["target"]
            q = t["state"].astype(np.float32)
            qn = q / max(float(np.linalg.norm(q)), 1e-12)
            hits_raw = mem.blurbs.retrieve(qn, k=1)
            if bool(hits_raw) and hits_raw[0][0] == t["anchor_id"]:
                t1raw += 1
            st = torch.from_numpy(t["state"]).to(dev).float().unsqueeze(0)
            rg = torch.from_numpy(t["ring_bges"]).to(dev).float().unsqueeze(0)
            pred = readout(st, rg)[0].cpu().numpy().astype(np.float32)
            hits = mem.blurbs.retrieve(pred, k=1)
            if bool(hits) and hits[0][0] == t["anchor_id"]:
                t1r += 1
            dcos += _cos(pred, target) - _cos(qn, target)
    n = len(triples)
    return (n, (t1r / n if n else 0.0), (t1raw / n if n else 0.0),
            (dcos / n if n else 0.0))


# -------------------------------------------------------------- triple capture
def _capture_triples(mem: FadeMemory, raw_bges: dict[int, np.ndarray],
                     anchor_ids: list[int], cos_ring: float,
                     ring_capacity: int) -> list[dict]:
    """Capture an evicted-anchor triple at the CURRENT stream step.

    Called once per stream step (after the step's ingest). For each anchor in
    ``anchor_ids`` that is EVICTED (not in the ring) AND below ``cos_ring``
    (the degraded band -- R3 + R2), capture (state, ring_bges, target, cos_raw,
    anchor_id, lag_N). Returns the list of triples captured at this step."""
    K = mem._next_id
    T = K - 1
    state = mem.ssm_a.state().astype(np.float64)
    ring = list(mem.ring)
    ring_bges = np.stack([mem.blurbs.vector(aid) for aid in ring]).astype(np.float64)
    triples: list[dict] = []
    for a in anchor_ids:
        if a in ring:                     # in-ring -> R1, not degraded
            continue
        cos_raw = mem._recoverability(a)
        if cos_raw is None or cos_raw >= cos_ring:
            continue                      # above cos_ring -> R1-ish, skip
        triples.append({
            "state": state.copy(),
            "ring_bges": ring_bges.copy(),
            "ring_ids": ring,
            "target": mem.blurbs.vector(a).astype(np.float64).copy(),
            "cos_raw": float(cos_raw),
            "anchor_id": int(a),
            "lag_N": int(T - a),
            "T": int(T),
        })
    return triples


# -------------------------------------------------------------- the run
def run(args) -> int:
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Bible anchors (domain A).
    print(f"[bible] fetching {args.book} {args.chapter} ({args.translation})...",
          flush=True)
    verses = bib.fetch_chapter(args.book, args.chapter, args.translation)
    n_anchors = min(args.n_anchors, len(verses))
    anchor_texts = [t for _, t in verses[:n_anchors]]
    print(f"[bible] {len(verses)} verses; using first {n_anchors} as anchors "
          f"(train 0-{n_anchors // 2 - 1}, eval {n_anchors // 2}-{n_anchors - 1})",
          flush=True)

    # 2. ERAG stream (domain B).
    print(f"[erag] loading {args.n_erag} chunks from {args.erag_path}...", flush=True)
    erag = cd.load_erag_chunks(args.erag_path, args.n_erag, args.seed, args.chunk_chars)
    print(f"[erag] {len(erag)} non-empty chunks (domain B)", flush=True)

    # 3. Build the fade memory (real bge). Track raw bge per anchor id (the
    #    closed-form needs the raw vectors the state was built from).
    print(f"[bge] loading bge-small-en-v1.5 (device={args.device})...", flush=True)
    base = bge_embedder()
    if args.device != "cpu":
        try:
            base = base.to(args.device)
        except Exception:
            pass
    # Caching wrapper so we can recover the raw bge per chunk (the state is
    # built from unnormalized bge; blurbs.vector is normalized -- though bge-small
    # outputs unit norms, the wrapper keeps the capture faithful).
    class _Cache:
        def __init__(self, b):
            self.b = b
            self.c: dict[str, np.ndarray] = {}
        def encode(self, texts):
            out = []
            for t in texts:
                if t not in self.c:
                    self.c[t] = np.asarray(self.b.encode([t])[0], dtype=np.float64)
                out.append([float(x) for x in self.c[t].tolist()])
            return out
        def raw(self, t):
            return self.c[t]
    emb = _Cache(base)
    voice = bib.PassthroughVoice()
    cfg = FadeConfig(decay=args.decay, cos_ring=args.cos_ring,
                     cos_gist=args.cos_gist, ring_capacity=args.ring_capacity,
                     regime2_enabled=False, expand_tokens=args.expand_tokens)
    mem = FadeMemory(cfg, emb, voice)
    print(f"[fade] decay={cfg.decay} ring={cfg.ring_capacity} "
          f"cos_ring={cfg.cos_ring} cos_gist={cfg.cos_gist}", flush=True)

    # 4. Ingest the anchors, then stream erag; capture triples at every step.
    raw_bges: dict[int, np.ndarray] = {}
    anchor_ids: list[int] = []
    for text in anchor_texts:
        aid = mem.ingest(text)
        anchor_ids.append(aid)
        raw_bges[aid] = emb.raw(text).copy()
    half = n_anchors // 2
    train_anchors = anchor_ids[:half]
    eval_anchors = anchor_ids[half:]

    train_triples: list[dict] = []
    eval_triples: list[dict] = []
    for step, chunk in enumerate(erag, start=1):
        aid = mem.ingest(chunk)
        raw_bges[aid] = emb.raw(chunk).copy()
        # capture at every step (the ring is full once step > ring_capacity).
        if step <= cfg.ring_capacity:
            continue                       # ring not full -> no evicted anchors yet
        train_triples.extend(_capture_triples(
            mem, raw_bges, train_anchors, cfg.cos_ring, cfg.ring_capacity))
        eval_triples.extend(_capture_triples(
            mem, raw_bges, eval_anchors, cfg.cos_ring, cfg.ring_capacity))
    print(f"[data] {len(train_triples)} train triples (anchors {train_anchors[0]}-"
          f"{train_anchors[-1]}), {len(eval_triples)} eval triples (anchors "
          f"{eval_anchors[0]}-{eval_anchors[-1]})", flush=True)

    # 5. The negative pool: all anchor bges EXCEPT the eval anchors (so the
    #    readout never learns to avoid an anchor it must later recover).
    neg_pool = np.stack([raw_bges[a] for a in anchor_ids if a not in eval_anchors
                        ] + [raw_bges[a] for a in range(len(anchor_ids),
                              mem._next_id)]).astype(np.float32)
    # normalize the pool (raw bge is already unit, but be safe).
    neg_pool = neg_pool / np.maximum(
        np.linalg.norm(neg_pool, axis=1, keepdims=True), 1e-12)
    neg_pool_t = torch.from_numpy(neg_pool)
    print(f"[neg] pool={neg_pool_t.shape}", flush=True)

    # 6. Build the readout + optimizer.
    dev = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available()
                       else "cpu")
    readout = FillHolesReadout(FillHolesConfig(
        dim=cfg.dim if hasattr(cfg, "dim") else 384, max_pos=cfg.ring_capacity + 2,
        dropout=args.dropout)).to(dev)
    opt = torch.optim.AdamW(readout.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in readout.parameters())
    print(f"[readout] {n_params:,} params, d_model={readout.cfg.d_model}, "
          f"max_pos={readout.cfg.max_pos}, dev={dev}", flush=True)

    # 7. Train. Shuffle + batch; jepa_infonce_loss with in-batch + pool negatives.
    def to_batch(trs: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(trs)
        state = torch.from_numpy(np.stack([t["state"] for t in trs])).to(dev).float()
        ring = torch.from_numpy(np.stack([t["ring_bges"] for t in trs])).to(dev).float()
        target = torch.from_numpy(np.stack([t["target"] for t in trs])).to(dev).float()
        return state, ring, target

    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    readout.train()
    for ep in range(args.epochs):
        order = rng.permutation(len(train_triples))
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            trs = [train_triples[j] for j in idx]
            state, ring, target = to_batch(trs)
            # sample 64 negatives from the pool (fresh per batch). Index on CPU
            # (neg_pool_t lives on CPU) then move to dev -- device-safe for cuda.
            nidx = torch.randint(0, neg_pool_t.shape[0], (args.n_negatives,))
            negatives = neg_pool_t[nidx].to(dev).float()       # [n, dim]
            pred = readout(state, ring)                       # [B, dim]
            loss = jepa_infonce_loss(pred, target, negatives,
                                     temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        ep_loss = ep_loss / max(n_batches, 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
        print(f"[train] epoch {ep + 1}/{args.epochs} loss={ep_loss:.4f} "
              f"(best={best_loss:.4f})", flush=True)

    # 8. EVAL over ALL eval triples. The gate uses the R2-band subset (cos_raw <
    #    cos_gist); the lag-bin diagnostic uses ALL of them to show WHERE
    #    recovery breaks down (the information-theoretic "why" for the negative).
    band = [t for t in eval_triples if t["cos_raw"] < cfg.cos_gist]
    print(f"\n[eval] {len(band)} R2-band triples (cos_raw < {cfg.cos_gist}) "
          f"out of {len(eval_triples)} eval triples", flush=True)
    if not band:
        print("[eval] NO band triples -- cannot evaluate the gate. (The eval "
              "anchors did not fade below cos_gist in this stream -- try more "
              "ERAG chunks or older eval anchors.)", flush=True)
        return 1

    readout.eval()
    # Per-triple records. ``has_sd_ring`` = a same-domain (Bible, id < n_anchors)
    # anchor is still in the ring at this step -- i.e. the ring carries SOME
    # anchor-domain signal. When the ring is all-ERAG (has_sd_ring=False), the
    # readout's inputs (state + ring) hold zero anchor-domain signal.
    rows: list[dict] = []
    with torch.no_grad():
        for t in eval_triples:
            target = t["target"]
            # raw-state baseline (the degraded ssm_a query).
            q = t["state"].astype(np.float32)
            qn = q / max(float(np.linalg.norm(q)), 1e-12)
            hits_raw = mem.blurbs.retrieve(qn, k=1)
            t1_raw = bool(hits_raw) and hits_raw[0][0] == t["anchor_id"]
            # ring-only closed-form baseline (the linear un-fade from memory).
            rec_ring = _ring_only_unfade(
                t["state"], t["ring_bges"], t["anchor_id"], t["ring_ids"],
                cfg.decay, cfg.write_gate, t["T"])
            hits_ring = mem.blurbs.retrieve(rec_ring, k=1)
            t1_ring = bool(hits_ring) and hits_ring[0][0] == t["anchor_id"]
            # readout (the Transformer un-fade from memory).
            st = torch.from_numpy(t["state"]).to(dev).float().unsqueeze(0)
            rg = torch.from_numpy(t["ring_bges"]).to(dev).float().unsqueeze(0)
            pred = readout(st, rg)[0].cpu().numpy().astype(np.float32)
            hits = mem.blurbs.retrieve(pred, k=1)
            t1_readout = bool(hits) and hits[0][0] == t["anchor_id"]
            # neg-retrieval (criterion 3): the recovered vector retrieves a
            # NON-anchor (a sibling) -- the collapse / drift signature.
            t1_neg = bool(hits) and hits[0][0] != t["anchor_id"]
            has_sd = any(int(aid) < n_anchors for aid in t["ring_ids"])
            rows.append({
                "lag": t["lag_N"], "in_band": t["cos_raw"] < cfg.cos_gist,
                "has_sd_ring": has_sd, "t1_raw": int(t1_raw),
                "t1_ring": int(t1_ring), "t1_readout": int(t1_readout),
                "t1_neg": int(t1_neg), "cos_raw": _cos(qn, target),
                "cos_rec": _cos(pred, target),
            })

    # Aggregate the gate on the R2-band subset.
    brow = [r for r in rows if r["in_band"]]
    n = len(brow)
    top1_raw_r = sum(r["t1_raw"] for r in brow) / n
    top1_readout_r = sum(r["t1_readout"] for r in brow) / n
    top1_ring_r = sum(r["t1_ring"] for r in brow) / n
    top1_neg_r = sum(r["t1_neg"] for r in brow) / n
    cos_raw_m = sum(r["cos_raw"] for r in brow) / n
    cos_readout_m = sum(r["cos_rec"] for r in brow) / n

    # Train-band top-1 (the overfitting-vs-impossibility diagnostic). If the
    # readout recovers TRAIN-band anchors but not EVAL-band, R2 is learnable but
    # fails to generalize; if TRAIN-band is also ~0, recovery is not learnable
    # at this fade depth (the definitive impossibility negative).
    train_band = [t for t in train_triples if t["cos_raw"] < cfg.cos_gist]
    tr_n, tr_t1r, tr_t1raw, tr_dcos = _top1_on(train_band, readout, mem, dev)

    # Lag-bin diagnostic (all eval triples): top-1 by lag, with the fraction of
    # triples whose ring still carries a same-domain anchor. The transition
    # sd_ring_frac 1->0 marks where the ring goes all-ERAG; readout top-1 should
    # collapse there (the cross-domain-floor negative, criterion #3).
    bins = [(33, 64), (65, 128), (129, 256), (257, 10**9)]
    bin_stats: list[dict | None] = []
    for lo, hi in bins:
        br = [r for r in rows if lo <= r["lag"] <= hi]
        if not br:
            bin_stats.append(None)
            continue
        nb = len(br)
        bin_stats.append({
            "lag": f"{lo}-{hi if hi < 10**9 else 'inf'}", "n": nb,
            "sd_ring_frac": sum(r["has_sd_ring"] for r in br) / nb,
            "top1_raw": sum(r["t1_raw"] for r in br) / nb,
            "top1_readout": sum(r["t1_readout"] for r in br) / nb,
            "in_band_frac": sum(r["in_band"] for r in br) / nb,
        })

    # ---- the four gate metrics
    g1_top1 = top1_readout_r >= top1_raw_r + 0.15
    g1_cos = (cos_readout_m - cos_raw_m) >= 0.10
    g1 = g1_top1 and g1_cos
    g2 = top1_readout_r >= top1_ring_r + 0.05
    g3 = top1_readout_r >= 5 * top1_neg_r if top1_neg_r > 0 else top1_readout_r > 0
    # criterion 4 (content / 9455795): top-1 == anchor_id is exactly top1_readout
    # (every readout top-1 hit IS the anchor's own blurb by construction of the
    # check). Drift = a top-1 hit on a SIBLING -- measured by the readout top-1
    # rate being driven by sibling matches. The honest content check: among
    # readout top-1 HITS, are they ALL the anchor's own blurb? Since the check is
    # ``hits[0][0] == anchor_id``, every counted hit IS the own blurb -- so g4
    # is the readout top-1 rate itself being non-trivial (>= 0.15, same as g1's
    # threshold) AND the sibling-hit rate (top1_neg among readout non-hits) low.
    # The cleanest g4: the readout's top-1-hit rate is the own-blurb rate; require
    # it >= 0.15 (a non-trivial recovery, not all-fail) -- the drift regression
    # would show as top1_readout driven by sibling matches, which the
    # top1_readout==anchor_id check already excludes.
    g4 = top1_readout_r >= 0.15
    passed = g1 and g2 and g3 and g4

    # ---- report
    print("\n" + "=" * 72)
    print("R2 FILL-HOLES READOUT GATE (task #35 Stage 2/3)")
    print("=" * 72)
    print(f"readout params  : {n_params:,} (2-layer Transformer, d_model=384)")
    print(f"train / eval    : {len(train_triples)} train triples (anchors "
          f"{train_anchors[0]}-{train_anchors[-1]}), {n} R2-band eval triples")
    print(f"baselines       : raw-state top-1={top1_raw_r:.3f}  "
          f"ring-only closed-form top-1={top1_ring_r:.3f}")
    print(f"readout         : top-1={top1_readout_r:.3f}  "
          f"mean cos(recovered,stored)={cos_readout_m:.3f}  "
          f"(raw cos={cos_raw_m:.3f}, delta={cos_readout_m - cos_raw_m:+.3f})")
    print(f"neg-retrieval   : {top1_neg_r:.3f} (the collapse signature; "
          f"readout must be >= 5x = {5 * top1_neg_r:.3f})")
    print()
    print("GATE (all four must pass):")
    print(f"  [1] vs raw state   : {'PASS' if g1 else 'FAIL'}  "
          f"(top-1 {top1_readout_r:.3f} >= {top1_raw_r + 0.15:.3f}? {g1_top1}; "
          f"cos delta {cos_readout_m - cos_raw_m:+.3f} >= 0.10? {g1_cos})")
    print(f"  [2] vs ring-only    : {'PASS' if g2 else 'FAIL'}  "
          f"(top-1 {top1_readout_r:.3f} >= {top1_ring_r + 0.05:.3f}?)")
    print(f"  [3] no-collapse     : {'PASS' if g3 else 'FAIL'}  "
          f"(top-1 {top1_readout_r:.3f} >= 5x neg {5 * top1_neg_r:.3f}?)")
    print(f"  [4] content (9455795): {'PASS' if g4 else 'FAIL'}  "
          f"(readout top-1 (own-blurb) {top1_readout_r:.3f} >= 0.15?)")
    print()
    print(f"VERDICT: {'PASS -> wire (Stage 4)' if passed else 'FAIL -> keep R3+R4, honest-negative doc'}")
    print(f"elapsed: {time.time() - t0:.1f}s")

    # Lag-bin diagnostic -- the "why" behind a FAIL (or the robustness check
    # behind a PASS). sd_ring_frac = fraction of triples in the bin whose ring
    # still carries a same-domain (Bible) anchor. When sd_ring_frac -> 0 the
    # ring is all-ERAG and the readout's inputs hold no anchor-domain signal.
    print("\nLAG-BIN DIAGNOSTIC (all eval triples; readout top-1 by lag):")
    print(f"  {'lag':>10}  {'n':>5}  {'sd_ring':>8}  {'in_band':>8}  "
          f"{'raw_t1':>7}  {'readout_t1':>11}")
    for bs in bin_stats:
        if bs is None:
            continue
        print(f"  {bs['lag']:>10}  {bs['n']:>5}  {bs['sd_ring_frac']:>8.3f}  "
              f"{bs['in_band_frac']:>8.3f}  {bs['top1_raw']:>7.3f}  "
              f"{bs['top1_readout']:>11.3f}")
    print("  (sd_ring=1 -> ring has a Bible anchor; 0 -> ring all-ERAG. The R2 "
          "band is the deep-lag all-ERAG region.)")
    print(f"\nOVERFITTING vs IMPOSSIBILITY (R2-band top-1, train vs eval anchors):")
    print(f"  train-band: n={tr_n}  readout top-1={tr_t1r:.3f}  raw top-1={tr_t1raw:.3f}  "
          f"cos delta={tr_dcos:+.3f}")
    print(f"  eval-band : n={n}  readout top-1={top1_readout_r:.3f}  raw top-1={top1_raw_r:.3f}  "
          f"cos delta={cos_readout_m - cos_raw_m:+.3f}")
    if tr_t1r > 0.15 and top1_readout_r < 0.15:
        print("  -> readout recovers TRAIN but not EVAL: OVERFITTING (learnable, "
              "fails to generalize -- more anchors/data might help).")
    elif tr_t1r < 0.15:
        print("  -> readout recovers NEITHER train nor eval at band depth: "
              "IMPOSSIBILITY (the anchor signal is not recoverable from memory "
              "at this fade depth -- the definitive negative).")

    # ---- save checkpoint on PASS only
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {"n_anchors": n_anchors, "n_erag": len(erag),
                   "decay": cfg.decay, "ring_capacity": cfg.ring_capacity,
                   "cos_ring": cfg.cos_ring, "cos_gist": cfg.cos_gist,
                   "lr": args.lr, "weight_decay": args.weight_decay,
                   "epochs": args.epochs, "temperature": args.temperature,
                   "n_negatives": args.n_negatives, "seed": args.seed},
        "data": {"n_train": len(train_triples), "n_eval_band": n,
                 "n_eval_total": len(eval_triples)},
        "baselines": {"top1_raw": top1_raw_r, "top1_ring": top1_ring_r,
                      "cos_raw": cos_raw_m},
        "readout": {"top1": top1_readout_r, "cos_recovered": cos_readout_m,
                    "cos_delta": cos_readout_m - cos_raw_m,
                    "neg_retrieval": top1_neg_r},
        "gate": {"g1_vs_raw": g1, "g2_vs_ring": g2, "g3_no_collapse": g3,
                 "g4_content": g4, "passed": passed},
        "lag_bins": [b for b in bin_stats if b is not None],
        "train_band": {"n": tr_n, "top1_readout": tr_t1r, "top1_raw": tr_t1raw,
                       "cos_delta": tr_dcos},
        "elapsed_s": time.time() - t0,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    if passed:
        ckpt_path = Path(args.ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(readout.checkpoint(step=args.epochs), str(ckpt_path))
        print(f"saved checkpoint -> {ckpt_path}")
    else:
        print("FAIL -> no checkpoint saved (a FAIL's ckpt is NOT uploaded, per policy)")
    print(f"wrote {out / 'run_summary.json'}")
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--erag-path", default=ERAG_PATH)
    p.add_argument("--book", default=DEFAULT_BOOK)
    p.add_argument("--chapter", type=int, default=DEFAULT_CHAPTER)
    p.add_argument("--translation", default="oeb-us")
    p.add_argument("--n-anchors", type=int, default=16)
    p.add_argument("--n-erag", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-chars", type=int, default=600)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--cos-ring", type=float, default=0.95)
    p.add_argument("--cos-gist", type=float, default=0.40)
    p.add_argument("--ring-capacity", type=int, default=32)
    p.add_argument("--expand-tokens", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-negatives", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())