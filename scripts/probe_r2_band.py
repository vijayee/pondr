"""R2 band-emptiness + closed-form baseline probe (FADE Regime 2, task #35).

The cheap, no-training gate for the R2 ("fill the holes") recovery path. R2 sits
BELOW R3 (cos < cos_gist, the band R3 declares "[forgotten]"): a readout
reconstructs the anchor's bge ADDRESS from MEMORY (state + ring) and, if it
matches the stored bge, recalls the anchor's own blurb (anchor-locked, no drift
-- the 9455795 invariant). This probe answers the two questions that gate
whether R2 needs a Transformer at all:

  Stage 0 -- is the R2 band non-empty under shipped defaults?
  Stage 1 -- does the RING-ONLY closed-form already recover it (linear baseline)?

If the band is empty -> R2 is dead under shipped defaults (honest stop). If the
ring-only closed-form already recovers at the target rate -> R2 ships as ~20
lines of arithmetic, no Transformer (skip Stage 2). Only if the band is
non-empty AND the ring-only closed-form FAILS does the Transformer earn its
keep (Stage 2).

## The closed-form (SSM-A is a LINEAR EWMA)

``state_{K-1} = sum_{p=0..K-1} decay^{(K-1)-p} * write_gate * bge_raw(chunk_p)``
(linear recurrence, state_0 = 0). So un-fade is closed-form arithmetic. With
ALL interferers (the full blurb store -- the RECORD) it is EXACT:

    recovered_full = (state - sum_{j != anchor} w_j * bge_raw[j]) / w_anchor
                   = bge_raw[anchor]                                  (exactly)

where ``w_j = decay**(T-j) * write_gate``, ``T = K-1``. This is the oracle
upper bound -- but it uses the RECORD (defeats the fade: always recovers, R4
never fires). With RING-ONLY interferers (the recent ones; evicted older ones
MISSING) it is an approximation -- the linear baseline the Transformer must beat:

    recovered_ring = (state - sum_{j in ring} w_j * bge_raw[j]) / w_anchor
                   = bge_raw[anchor] + sum_{j evicted, j != anchor}
                       decay^{anchor-j} * bge_raw[j]

The evicted interferers NEWER than the anchor (j > anchor) get AMPLIFIED
(``decay^{anchor-j} = (1/decay)^{j-anchor} > 1``) and swamp the signal -> the
ring-only closed-form is a poor approximation when the most-recent evicted
chunk still carries weight (``decay**ring_capacity`` is not small at
decay=0.99, ring_capacity=32 -> 0.725). That is the hole the Transformer would
have to fill -- only it can't see the evicted interferers, so in a
cross-domain stream (where they are unpredictable from the ring) it fails too
(the honest-negative criterion 3). This probe MEASURES that rather than
assuming it.

## Why cross-domain (Bible + ERAG)

The R2/R4 band (cos < cos_gist=0.40) is only reached CROSS-DOMAIN: same-domain
bge cos plateaus at ~0.6-0.8 (the tip-of-tongue floor, never below cos_gist),
while cross-domain floor is ~0.37 < 0.40 (docs/fade-cross-domain-eval-result.md).
So we ingest Bible anchors (domain A) and stream ERAG technical docs (domain B)
past them -- the proven way to drive anchors into the R2/R4 band on real bge.
``--erag-only`` mode checks whether ERAG alone (homogeneous ML/tech) reaches the
band (expected: no -- the floor sits above cos_gist).

## Faithfulness

Drives the REAL ``FadeMemory`` (real bge, real VectorCarrySSM.step, real
BlurbStore, real ring). A caching embedder wrapper captures the RAW bge vector
fed to ``ssm_a.step`` for every chunk (the state is built from UNnormalized
bge; ``SentenceTransformer.encode`` does not normalize, while ``BlurbStore.add``
does -- so the closed-form must use the raw vectors, not the stored normalized
ones). The retrieval/top-1 check uses the real ``blurbs.retrieve`` (normalized
dot product = cosine). No numpy EWMA reimplementation (probe_vector_carry's
``simulate_channel`` is NOT used); the EWMA is the real ``VectorCarrySSM``.

Standalone, CPU-runnable. Reads the gitignored untracked ERAG parquet locally
(public ERAG only -- no onyx, no private transcripts). Bible OEB-US via
bible-api.com (public domain). Writes ``run_summary.json`` + prints the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # repo root

import eval_fade_bible as bib  # noqa: E402  (fetch_chapter, PassthroughVoice)
import eval_fade_cross_domain as cd  # noqa: E402  (load_erag_chunks)

from src.subconscious.fade import (  # noqa: E402
    FadeConfig,
    FadeMemory,
    bge_embedder,
)

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_OUTPUT_DIR = "data/probe/r2_band"
# Bible anchors (domain A). John 1 (NT gospel) is plainly distinct from ERAG
# technical runbooks (domain B) -> cross-domain interference drives anchors
# below cos_gist into the R2/R4 band.
DEFAULT_BOOK, DEFAULT_CHAPTER = "john", 1
# erag-step grid (chunks of domain B streamed AFTER the bible anchors) at which
# to record the closed-form. 0 = just the bible anchors ingested (lag 0).
DEFAULT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
# cos_reconstruct sweep (the threshold on cos(recovered, stored) for "recovery
# succeeded"). 0.40 mirrors the shipped cos_gist; the sweep finds where the
# closed-form's cos lands.
DEFAULT_COS_RECONSTRUCT_SWEEP = [0.30, 0.40, 0.50, 0.60]


# -------------------------------------------------------------- caching embedder
class _CachingEmbedder:
    """Wrap a real embedder and cache the RAW vector for every text seen.

    The fade's ``ingest`` calls ``embedder.encode([text])`` internally and feeds
    the result to ``ssm_a.step``. To compute the closed-form we need the EXACT
    raw vector the state was built from (unnormalized -- ``BlurbStore.add``
    normalizes before storing, so ``blurbs.vector(aid)`` is NOT the vector the
    state was built from). This wrapper caches by text so we can recover
    ``raw_bges[anchor_id]`` after the fact, while the real ``FadeMemory`` does
    the encoding/stepping/storing. Encode is called once per unique text (the
    fade ingests each chunk once), so the cache is a 1:1 record of what the
    state was built from."""

    def __init__(self, base) -> None:
        self.base = base
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if t not in self.cache:
                raw = self.base.encode([t])[0]
                self.cache[t] = np.asarray(raw, dtype=np.float64)
            out.append([float(x) for x in self.cache[t].tolist()])
        return out

    def raw(self, text: str) -> np.ndarray:
        return self.cache[text]


# -------------------------------------------------------------- closed-form
def _closed_form(state: np.ndarray, raw_bges: list[np.ndarray],
                 anchor_id: int, ring_ids: list[int], decay: float,
                 write_gate: float, T: int, full: bool,
                 all_ids: list[int] | None = None) -> np.ndarray:
    """The linear un-fade. Returns the L2-normalized recovered bge.

    ``state``    : the raw SSM-A state after T+1 steps (``ssm_a.state()``).
    ``raw_bges`` : raw bge per anchor_id (what ``ssm_a.step`` was fed).
    ``anchor_id``: the anchor to recover; ``N = T - anchor_id``.
    ``ring_ids``: the ring's anchor_ids (the recent interferers).
    ``T``        : the last step index = K-1 (K = number of ingests so far).
    ``full``     : True -> subtract ALL other interferers (the oracle, uses the
                   record); False -> subtract only the ring (the linear baseline,
                   uses memory).
    ``all_ids``  : 0..K-1 (required when full=True).

    ``recovered = (state - sum_{j != anchor} w_j * raw[j]) / w_anchor``,
    ``w_j = decay**(T-j) * write_gate``. Normalized for the retrieval/cos check.
    """
    w_anchor = (decay ** (T - anchor_id)) * write_gate
    if w_anchor == 0.0:
        return np.zeros_like(state)
    subtractor = np.zeros_like(state)
    if full:
        for j in all_ids:  # type: ignore[union-attr]
            if j == anchor_id:
                continue
            subtractor += (decay ** (T - j)) * write_gate * raw_bges[j]
    else:
        for j in ring_ids:
            if j == anchor_id:
                continue  # anchor is evicted in the band, but guard anyway
            subtractor += (decay ** (T - j)) * write_gate * raw_bges[j]
    recovered = (state - subtractor) / w_anchor
    n = float(np.linalg.norm(recovered))
    if n == 0.0:
        return recovered.astype(np.float32)
    return (recovered / n).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -------------------------------------------------------------- the probe
def run(args) -> int:
    t0 = time.time()
    # 1. Bible anchors (domain A) -- or skip for --erag-only.
    anchor_texts: list[str] = []
    anchor_refs: list[str] = []
    if not args.erag_only:
        print(f"[bible] fetching {args.book} {args.chapter} ({args.translation})...",
              flush=True)
        verses = bib.fetch_chapter(args.book, args.chapter, args.translation)
        k = min(args.n_anchors, len(verses))
        for ref, text in verses[:k]:
            anchor_texts.append(text)
            anchor_refs.append(ref)
        print(f"[bible] {len(verses)} verses; using first {k} as probe anchors",
              flush=True)
    else:
        # --erag-only: the first n_anchors ERAG chunks ARE the anchors (same
        # domain as the stream -> probes whether ERAG-alone reaches the band).
        pass

    # 2. ERAG stream (domain B).
    print(f"[erag] loading {args.n_erag} chunks from {args.erag_path}...", flush=True)
    erag = cd.load_erag_chunks(args.erag_path, args.n_erag, args.seed, args.chunk_chars)
    print(f"[erag] {len(erag)} non-empty chunks (domain B)", flush=True)
    if args.erag_only:
        # first n_anchors chunks are the anchors; the rest are the stream.
        n = min(args.n_anchors, len(erag))
        anchor_texts = erag[:n]
        anchor_refs = [f"erag:{i}" for i in range(n)]
        erag = erag[n:]
    if len(erag) < max(args.steps):
        raise RuntimeError(
            f"only {len(erag)} erag stream chunks -- need >= max step "
            f"{max(args.steps)}; raise --n-erag")

    # 3. Build the fade memory with the caching embedder (real bge).
    print(f"[bge] loading bge-small-en-v1.5 (device={args.device})...", flush=True)
    base = bge_embedder()
    if args.device != "cpu":
        try:
            base = base.to(args.device)
        except Exception:
            pass  # CPU fallback is fine
    emb = _CachingEmbedder(base)
    voice = bib.PassthroughVoice()
    cfg = FadeConfig(decay=args.decay, cos_ring=args.cos_ring,
                     cos_gist=args.cos_gist, ring_capacity=args.ring_capacity,
                     regime2_enabled=False, expand_tokens=args.expand_tokens)
    mem = FadeMemory(cfg, emb, voice)
    print(f"[fade] decay={cfg.decay} ring={cfg.ring_capacity} "
          f"cos_ring={cfg.cos_ring} cos_gist={cfg.cos_gist} (R2 off)", flush=True)

    # 4. Ingest the anchors, then stream erag; probe at the step grid.
    anchor_ids: list[int] = []
    for text in anchor_texts:
        aid = mem.ingest(text)
        anchor_ids.append(aid)
    step_set = set(args.steps)
    # rows: one per (anchor, step) probed.
    rows: list[dict] = []

    def probe(step: int) -> None:
        K = mem._next_id                 # number of ingests so far
        T = K - 1                        # last step index
        state = mem.ssm_a.state().astype(np.float64)
        ring = list(mem.ring)            # the last ring_capacity anchor_ids
        # all_ids for the full closed-form: 0..K-1. _ingested_raw (populated in
        # the ingest loop, including the anchors) holds the raw bge for every
        # ingested chunk id -- the closed-form subtracts from it.
        all_ids = list(range(K))
        for a in anchor_ids:
            N = T - a
            in_ring = a in ring
            cos_raw = mem._recoverability(a)
            if cos_raw is None:
                continue
            # raw-state baseline top-1 (what R3/R4 sees).
            q = mem.ssm_a.query()
            hits_raw = mem.blurbs.retrieve(q, k=1)
            top1_raw = bool(hits_raw) and hits_raw[0][0] == a
            # ring-only closed-form (the linear baseline).
            rec_ring = _closed_form(
                state, _ingested_raw, a, ring, cfg.decay, cfg.write_gate, T,
                full=False)
            cos_ring_cf = _cos(rec_ring, mem.blurbs.vector(a))
            hits_ring = mem.blurbs.retrieve(rec_ring, k=1)
            top1_ring = bool(hits_ring) and hits_ring[0][0] == a
            # full closed-form (the oracle -- uses the record).
            rec_full = _closed_form(
                state, _ingested_raw, a, ring, cfg.decay, cfg.write_gate, T,
                full=True, all_ids=all_ids)
            cos_full = _cos(rec_full, mem.blurbs.vector(a))
            hits_full = mem.blurbs.retrieve(rec_full, k=1)
            top1_full = bool(hits_full) and hits_full[0][0] == a
            rows.append({
                "anchor_id": a, "ref": anchor_refs[a], "step": step,
                "lag_N": N, "in_ring": in_ring, "evicted": not in_ring,
                "cos_raw": cos_raw,
                "top1_raw": top1_raw,
                "cos_ring_cf": cos_ring_cf, "top1_ring": top1_ring,
                "cos_full": cos_full, "top1_full": top1_full,
            })

    # Track the raw bge for EVERY ingested chunk (anchors + stream) by id, so
    # the full closed-form can subtract all interferers. The cache is keyed by
    # text; we record the id->text mapping at ingest time.
    _ingested_raw: dict[int, np.ndarray] = {}

    def ingest(text: str) -> int:
        aid = mem.ingest(text)
        _ingested_raw[aid] = emb.raw(text).copy()
        return aid

    # Populate _ingested_raw for the already-ingested anchors. The anchors were
    # ingested above via `mem.ingest` directly, BEFORE _ingested_raw existed, so
    # their raw bge was never recorded. We must NOT re-ingest them -- that would
    # double the stream (a second step per anchor). Instead, recover each
    # anchor's raw bge from the embedder cache and record it here by id.
    for aid, text in zip(anchor_ids, anchor_texts):
        _ingested_raw[aid] = emb.raw(text).copy()

    if 0 in step_set:
        probe(0)
    for step, chunk in enumerate(erag, start=1):
        ingest(chunk)
        if step in step_set:
            probe(step)

    # ---- raw bge norm spread (validates the closed-form uses raw, not stored)
    raw_norms = [float(np.linalg.norm(v)) for v in _ingested_raw.values()]
    norm_min = min(raw_norms) if raw_norms else 0.0
    norm_max = max(raw_norms) if raw_norms else 0.0
    norm_mean = float(np.mean(raw_norms)) if raw_norms else 0.0

    # ---- the band: evicted AND cos_raw < cos_gist (the R2/R4 band)
    band = [r for r in rows if r["evicted"] and r["cos_raw"] < cfg.cos_gist]
    n_band = len(band)
    n_evicted = sum(1 for r in rows if r["evicted"])

    def rate(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else 0.0

    def mean(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else 0.0

    band_top1_raw = rate(band, "top1_raw")
    band_top1_ring = rate(band, "top1_ring")
    band_top1_full = rate(band, "top1_full")
    band_cos_raw = mean(band, "cos_raw")
    band_cos_ring = mean(band, "cos_ring_cf")
    band_cos_full = mean(band, "cos_full")

    # cos_reconstruct sweep on the ring-only closed-form (does it recover at
    # the target threshold?).
    sweep = {}
    for thr in args.cos_reconstruct_sweep:
        rec = [r for r in band
               if r["cos_ring_cf"] >= thr and r["top1_ring"]]
        sweep[thr] = len(rec)

    # Flaw-1 confirmation: the above-R3 thin slice (cos in [0.85, 0.95]) among
    # evicted anchors -- expected EMPTY (anchors plateau at ~0.8, below 0.85).
    above_r3_thin = [r for r in rows
                     if r["evicted"] and 0.85 <= r["cos_raw"] < 0.95]
    # The R3 band (cos_gist <= cos < cos_ring) among evicted -- where anchors
    # plateau same-domain.
    r3_band = [r for r in rows
               if r["evicted"] and cfg.cos_gist <= r["cos_raw"] < cfg.cos_ring]

    # cos histogram of evicted anchors (the plateau shape).
    hist_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist = {f"[{hist_bins[i]:.1f},{hist_bins[i+1]:.1f})": 0
            for i in range(len(hist_bins) - 1)}
    for r in rows:
        if not r["evicted"]:
            continue
        c = r["cos_raw"]
        for i in range(len(hist_bins) - 1):
            if hist_bins[i] <= c < hist_bins[i + 1]:
                hist[f"[{hist_bins[i]:.1f},{hist_bins[i+1]:.1f})"] += 1
                break

    # ---- the gate
    band_nonempty = n_band > 0
    # ring-only closed-form "already recovers everything": top-1 rate >= 0.9
    # (the threshold above which R2 ships as the closed-form, no Transformer).
    ring_already = band_top1_ring >= 0.9
    # oracle sanity: the full closed-form should be ~exact (top-1 ~1.0) WHEN the
    # band is non-empty -- this validates the raw-bge capture (the closed-form
    # uses the raw vectors the state was built from). When the band is empty
    # (e.g. ERAG-only, same-domain floor above cos_gist) there are no samples to
    # check -> N/A, not a broken capture.
    if n_band:
        oracle_ok = band_top1_full >= 0.9
        oracle_na = False
    else:
        oracle_ok = True      # vacuously: no band samples to recover
        oracle_na = True

    # ---- report
    print("\n" + "=" * 72)
    print("R2 BAND-EMPTINESS + CLOSED-FORM BASELINE (task #35 Stage 0/1)")
    print("=" * 72)
    print(f"stream           : {'ERAG-only' if args.erag_only else 'Bible + ERAG (cross-domain)'}")
    print(f"anchors / steps  : {len(anchor_ids)} anchors, {len(erag)} stream chunks, "
          f"{len(rows)} (anchor,step) samples ({n_evicted} evicted)")
    print(f"decay/ring/gist  : decay={cfg.decay} ring_cap={cfg.ring_capacity} "
          f"cos_ring={cfg.cos_ring} cos_gist={cfg.cos_gist}")
    print(f"raw bge norm     : min={norm_min:.4f} mean={norm_mean:.4f} max={norm_max:.4f} "
          f"(spread {norm_max - norm_min:.4f}; closed-form uses RAW, not stored-normalized)")
    print()
    print(f"R2/R4 band (evicted AND cos_raw < {cfg.cos_gist}): {n_band} samples")
    if n_band:
        print(f"  raw-state   top-1={band_top1_raw:.3f}  mean cos={band_cos_raw:.3f}")
        print(f"  ring-only   top-1={band_top1_ring:.3f}  mean cos={band_cos_ring:.3f}  "
              f"(the linear baseline)")
        print(f"  full (oracle) top-1={band_top1_full:.3f}  mean cos={band_cos_full:.3f}  "
              f"(uses the RECORD)")
        print(f"  cos_reconstruct sweep (ring-only, cos>=thr AND top1==anchor):")
        for thr in args.cos_reconstruct_sweep:
            print(f"    thr={thr:.2f} -> {sweep[thr]}/{n_band} "
                  f"({(sweep[thr]/n_band) if n_band else 0:.3f})")
    print()
    print(f"Flaw-1 check (above-R3 thin slice [0.85,0.95), evicted): "
          f"{len(above_r3_thin)} samples (expected ~0)")
    print(f"R3 band (cos_gist<=cos<cos_ring, evicted): {len(r3_band)} samples "
          f"(the plateau)")
    print(f"evicted cos histogram: {hist}")
    print()
    print("GATE:")
    print(f"  [Stage 0] band non-empty            : "
          f"{'PASS' if band_nonempty else 'FAIL (R2 dead under shipped defaults)'}"
          f"  ({n_band} samples)")
    print(f"  [oracle]  full closed-form ~exact   : "
          f"{'N/A (band empty)' if oracle_na else 'PASS' if oracle_ok else 'FAIL (raw-bge capture broken)'}"
          f"  (top-1={band_top1_full:.3f})")
    print(f"  [Stage 1] ring-only already recovers : "
          f"{'YES -> ship closed-form, skip Transformer' if ring_already else 'NO -> Transformer needed (proceed Stage 2)'}"
          f"  (top-1={band_top1_ring:.3f})")
    print()
    verdict = ("DEAD (band empty)" if not band_nonempty
              else "SHIP CLOSED-FORM (no Transformer)" if ring_already
              else "PROCEED TO STAGE 2 (Transformer)")
    print(f"VERDICT: {verdict}")
    print(f"elapsed: {time.time() - t0:.1f}s")

    # ---- write summary
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "stream": "erag_only" if args.erag_only else "bible+erag",
            "book": args.book, "chapter": args.chapter,
            "n_anchors": len(anchor_ids), "n_erag_stream": len(erag),
            "decay": cfg.decay, "ring_capacity": cfg.ring_capacity,
            "cos_ring": cfg.cos_ring, "cos_gist": cfg.cos_gist,
            "write_gate": cfg.write_gate,
        },
        "raw_bge_norm": {"min": norm_min, "mean": norm_mean, "max": norm_max,
                         "spread": norm_max - norm_min},
        "n_samples": len(rows), "n_evicted": n_evicted, "n_band": n_band,
        "band": {
            "top1_raw": band_top1_raw, "top1_ring": band_top1_ring,
            "top1_full": band_top1_full,
            "cos_raw": band_cos_raw, "cos_ring_cf": band_cos_ring,
            "cos_full": band_cos_full,
            "cos_reconstruct_sweep": {str(t): sweep[t]
                                      for t in args.cos_reconstruct_sweep},
        },
        "flaw1_above_r3_thin_slice": len(above_r3_thin),
        "r3_band_evicted": len(r3_band),
        "evicted_cos_histogram": hist,
        "gate": {
            "band_nonempty": band_nonempty,
            "oracle_ok": oracle_ok,
            "ring_already_recovers": ring_already,
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    # also dump the per-sample rows for plotting/inspection.
    (out / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out / 'run_summary.json'} and {out / 'rows.json'}")

    # exit code: 0 = band non-empty (worth continuing), 1 = band empty (dead),
    # 2 = oracle broken (probe invalid). The verdict string carries the rest.
    if not oracle_ok:
        return 2
    return 0 if band_nonempty else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--erag-path", default=ERAG_PATH)
    p.add_argument("--erag-only", action="store_true",
                   help="skip Bible; use the first n_anchors ERAG chunks as "
                        "anchors (probes whether ERAG-alone reaches the band).")
    p.add_argument("--book", default=DEFAULT_BOOK)
    p.add_argument("--chapter", type=int, default=DEFAULT_CHAPTER)
    p.add_argument("--translation", default="oeb-us")
    p.add_argument("--n-anchors", type=int, default=8)
    p.add_argument("--n-erag", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-chars", type=int, default=600)
    p.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--cos-ring", type=float, default=0.95)
    p.add_argument("--cos-gist", type=float, default=0.40)
    p.add_argument("--ring-capacity", type=int, default=32)
    p.add_argument("--expand-tokens", type=int, default=64)
    p.add_argument("--cos-reconstruct-sweep", type=float, nargs="+",
                   default=DEFAULT_COS_RECONSTRUCT_SWEEP)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())