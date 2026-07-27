"""Cross-domain fade eval -- exercise R4 (forgotten) on REAL bge (task #36 follow-on).

The Bible eval (#36, ``scripts/eval_fade_bible.py``) showed only R1+R3 on a real
319-verse Bible session: Bible is homogeneous, so bge cross-verse cosine plateaus at
~0.6 (above the gist threshold, whether 0.30 or the re-calibrated 0.40) and R4
never triggered -- old anchors plateau in R3 (the tip-of-tongue floor). R4 was
therefore validated ONLY on the SYNTHETIC embedder in the unit tests
(``test_regime4_forgotten_no_confabulation``), never on real bge on a real session.
That is exactly the synthetic-vs-real surface that has bitten this project before
(train/serve OOD), so closing the gap matters.

This eval closes it. Ingest a few Bible verses (domain A) as probe anchors, then
stream many ERAG technical docs (domain B -- confluence/runbooks, a clearly different
domain) past them. Real bge-small has a HIGH cosine floor: same-domain ~0.6,
cross-domain (Bible <-> technical runbook) ~0.37 (measured here -- NOT the ~0.1 one
might expect; bge-small lives in a narrow cone). So the router threshold
``cos_gist`` must sit BETWEEN those floors: 0.40 -> same-domain stays R3 (fuzzy
gist), cross-domain drops below it -> R4 (forgotten) on REAL bge. The probe-#32
default 0.30 (calibrated on the synthetic test embedder whose cross-doc floor is
~0.01) is BELOW real bge's cross-domain floor, so it never reaches R4 on real bge --
this eval re-calibrates it to 0.40. The fade must EMERGE from state compression (the
state drifting to domain B), not a policy switch.

Gates (all must hold):
  1. R4 TRIGGERS ON REAL BGE: at least one Bible anchor reaches regime R4 at some
     erag-step. THE KEY GATE -- the thing the Bible eval could not exercise.
  2. R4 NO-CONFABULATION: every R4 recall content == "[forgotten]" (the voice is
     never called for R4 -- the graceful tip-of-tongue floor, no confabulation).
  3. GRACEFUL: at least one anchor transitions R1 -> R3 -> R4 in order (R3 appears
     before R4; not an R1 -> R4 cliff with no gist window).
  4. R1 EXACT (sanity): at step 0 (just ingested, in ring) the anchors are R1 with
     content == the original verse text (confirms the Bible anchors ingested
     correctly and the ring gives true verbatim).

R2 (fill) is off (the #36 decision); without the Stage-2
``CrossSlotTransformerZHead`` it degrades to R3 in the module.

Standalone: fetches one Bible chapter from bible-api.com (OEB-US, public domain),
loads ERAG chunks from a local parquet (``--erag-path``; the default is gitignored
untracked data -- NOT committed, never re-distributed), streams through
``FadeMemory`` (real bge, passthrough voice), writes ``run_summary.json``, prints the
curves. No orchestrator/runtime/serve changes. CPU-runnable.

If R4 does NOT trigger even cross-domain (real bge Bible-ERAG cos still > cos_gist),
the eval FAILS honestly -- meaning the cross-domain floor is higher than expected and
``cos_gist`` must be raised, or the fade needs a faster decay. Report the observed
cross-domain cos floor either way.
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

import eval_fade_bible as bib  # noqa: E402  (reuse fetch_chapter, voice, REGIME_NAME)
from src.subconscious.fade import (  # noqa: E402
    REGIME_FILL,
    REGIME_FORGOTTEN,
    REGIME_GIST,
    REGIME_VERBATIM,
    FadeConfig,
    FadeMemory,
    bge_embedder,
)

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_OUTPUT_DIR = "data/probe/fade_cross_domain"
# Bible anchors (domain A). John 1 is NT (available OEB-US on bible-api.com); a
# gospel chapter is plainly distinct from ERAG technical runbooks.
DEFAULT_BOOK, DEFAULT_CHAPTER = "john", 1
# erag-step grid (chunks of domain B streamed AFTER the bible anchors) at which to
# record the regime/cos. 0 = just the bible anchors ingested (lag 0).
DEFAULT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
REGIME_NAME = bib.REGIME_NAME


# -------------------------------------------------------------- erag loader
def load_erag_chunks(path: str, n: int, seed: int, chunk_chars: int) -> list[str]:
    """Load ``n`` deterministic, non-empty ERAG content chunks (domain B).

    Truncates each to ``chunk_chars`` so the embedded bge vector and the stored
    blurb are consistent (the fade stores ``chunk_text[:blurb_chars]``). The ERAG
    parquet is gitignored untracked data -- this function only READS it locally."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - dependency check
        raise SystemExit(f"pandas required to read the ERAG parquet: {e}")
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"ERAG parquet not found at {p} (gitignored untracked data; not in "
            f"git). Point --erag-path at a local copy of the ERAG documents "
            f"parquet (cols: doc_id, source_type, title, content).")
    df = pd.read_parquet(p)
    if "content" not in df.columns:
        raise SystemExit(f"ERAG parquet {p} has no 'content' column; cols="
                         f"{list(df.columns)}")
    rng = np.random.default_rng(seed)
    n = min(n, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    chunks: list[str] = []
    for i in idx:
        c = str(df.iloc[int(i)]["content"]).strip()
        if not c:
            continue
        chunks.append(c[:chunk_chars])
    return chunks


# -------------------------------------------------------------- the eval
def run(args) -> int:
    t0 = time.time()
    # 1. Bible anchors (domain A).
    print(f"[bible] fetching {args.book} {args.chapter} ({args.translation})...",
          flush=True)
    verses = bib.fetch_chapter(args.book, args.chapter, args.translation)
    k = min(args.n_anchors, len(verses))
    anchor_verses = verses[:k]
    print(f"[bible] {len(verses)} verses; using first {k} as probe anchors",
          flush=True)

    # 2. ERAG stream (domain B).
    print(f"[erag] loading {args.n_erag} chunks from {args.erag_path}...", flush=True)
    erag = load_erag_chunks(args.erag_path, args.n_erag, args.seed, args.chunk_chars)
    print(f"[erag] {len(erag)} non-empty chunks (domain B)", flush=True)
    if len(erag) < max(args.steps):
        raise RuntimeError(
            f"only {len(erag)} erag chunks -- need >= max step {max(args.steps)}; "
            f"raise --n-erag or point --erag-path at a larger parquet")

    # 3. Build the fade memory.
    print(f"[bge] loading bge-small-en-v1.5 (device={args.device})...", flush=True)
    emb = bge_embedder()
    if args.device != "cpu":
        try:
            emb = emb.to(args.device)
        except Exception:
            pass  # CPU fallback is fine
    voice = bib.PassthroughVoice()
    cfg = FadeConfig(decay=args.decay, cos_ring=args.cos_ring,
                     cos_gist=args.cos_gist, ring_capacity=args.ring_capacity,
                     regime2_enabled=False, expand_tokens=args.expand_tokens)
    mem = FadeMemory(cfg, emb, voice)
    print(f"[fade] decay={cfg.decay} ring={cfg.ring_capacity} "
          f"cos_ring={cfg.cos_ring} cos_gist={cfg.cos_gist} (R2 off)", flush=True)

    # 4. Ingest the bible anchors, then stream erag; probe at the step grid.
    anchor_ids: list[int] = []
    anchor_text: dict[int, str] = {}
    anchor_ref: dict[int, str] = {}
    for ref, text in anchor_verses:
        aid = mem.ingest(text)
        anchor_ids.append(aid)
        anchor_text[aid] = text
        anchor_ref[aid] = ref
    step_set = set(args.steps)
    curves: dict[int, list[dict]] = {a: [] for a in anchor_ids}

    def probe(step: int) -> None:
        for a in anchor_ids:
            r = mem.recall_anchor(a)
            if r is None:
                continue
            curves[a].append({
                "erag_step": step,
                "regime": r.regime,
                "regime_name": REGIME_NAME[r.regime],
                "cos": r.cos,
                "content": r.content,
            })

    if 0 in step_set:
        probe(0)
    for step, chunk in enumerate(erag, start=1):
        mem.ingest(chunk)
        if step in step_set:
            probe(step)

    # ---- gates
    # 1. R4 TRIGGERS ON REAL BGE: >=1 bible anchor reaches R4 at some step.
    r4_triggered = any(
        pt["regime"] == REGIME_FORGOTTEN
        for a in anchor_ids for pt in curves[a]
    )

    # 2. R4 NO-CONFABULATION: every R4 recall content == "[forgotten]".
    r4_no_confab = all(
        pt["content"] == "[forgotten]"
        for a in anchor_ids for pt in curves[a]
        if pt["regime"] == REGIME_FORGOTTEN
    )

    # 3. GRACEFUL: >=1 anchor transitions R1 -> R3 -> R4 in order (R3 before R4).
    graceful = False
    for a in anchor_ids:
        regs = [pt["regime"] for pt in curves[a]]
        if REGIME_GIST in regs and REGIME_FORGOTTEN in regs:
            if regs.index(REGIME_GIST) < regs.index(REGIME_FORGOTTEN):
                graceful = True
                break

    # 4. R1 EXACT (sanity): at step 0, anchors are R1 with content == verse text.
    r1_exact = False
    for a in anchor_ids:
        for pt in curves[a]:
            if pt["erag_step"] == 0 and pt["regime"] == REGIME_VERBATIM:
                if pt["content"].strip() == anchor_text[a].strip():
                    r1_exact = True
                    break
        if r1_exact:
            break

    verdict = "PASS" if (r4_triggered and r4_no_confab and graceful
                        and r1_exact) else "FAIL"

    # Cross-domain cos floor: the lowest cos any anchor reaches (the real-bge
    # cross-domain floor -- if it stays above cos_gist, R4 cannot trigger).
    min_cos = min((pt["cos"] for a in anchor_ids for pt in curves[a]),
                  default=float("nan"))
    regimes_observed = sorted({pt["regime"] for a in anchor_ids for pt in curves[a]})
    regimes_observed_names = [REGIME_NAME[r] for r in regimes_observed]

    # ---- print the curves
    print(f"\n{'='*84}\nFADE CURVES (regime vs erag-step, per Bible anchor; "
          f"domain A={args.book} {args.chapter}, domain B=ERAG)\n{'='*84}",
          flush=True)
    for a in anchor_ids:
        print(f"\n  anchor {a} ({anchor_ref[a]}): "
              f"'{anchor_text[a][:60]}{'...' if len(anchor_text[a]) > 60 else ''}'",
              flush=True)
        print("    step : regime        cos    | content-kind", flush=True)
        for pt in curves[a]:
            kind = ("EXACT verse" if pt["regime"] == REGIME_VERBATIM
                    else "retrieved blurb" if pt["regime"] == REGIME_GIST
                    else "FILL (degraded->R3)" if pt["regime"] == REGIME_FILL
                    else "[forgotten]")
            print(f"    {pt['erag_step']:>4} : {pt['regime_name']:<13} "
                  f"{pt['cos']:.3f} | {kind}", flush=True)

    print(f"\n{'='*84}\nGATES\n{'='*84}", flush=True)
    print(f"  1. R4 triggers on REAL bge             : "
          f"{'PASS' if r4_triggered else 'FAIL'}", flush=True)
    print(f"  2. R4 no-confabulation                 : "
          f"{'PASS' if r4_no_confab else 'FAIL'}", flush=True)
    print(f"  3. graceful (R1->R3->R4)               : "
          f"{'PASS' if graceful else 'FAIL'}", flush=True)
    print(f"  4. R1 exact (sanity, step 0)          : "
          f"{'PASS' if r1_exact else 'FAIL'}", flush=True)
    print(f"\n  regimes observed : {regimes_observed_names}", flush=True)
    print(f"  cross-domain cos floor (min cos) : {min_cos:.3f} "
          f"(cos_gist={cfg.cos_gist})", flush=True)
    if not r4_triggered:
        print(f"  NOTE: R4 did NOT trigger -- the real-bge cross-domain cos floor "
              f"({min_cos:.3f}) stayed above cos_gist={cfg.cos_gist}. Raise "
              f"--cos-gist or --decay (faster fade), or use a more distant domain B.",
              flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)

    summary = {
        "probe": "fade_cross_domain",
        "purpose": "exercise R4 (forgotten) on real bge via a cross-domain session",
        "translation": args.translation,
        "domain_a": f"{args.book} {args.chapter}",
        "domain_b": "ERAG technical docs",
        "n_anchors": k,
        "n_erag_streamed": len(erag),
        "config": {"decay": cfg.decay, "cos_ring": cfg.cos_ring,
                   "cos_gist": cfg.cos_gist, "ring_capacity": cfg.ring_capacity,
                   "regime2_enabled": False, "voice": "passthrough"},
        "anchors": [{"anchor": a, "ref": anchor_ref[a]}
                    for a in anchor_ids],
        "fade_curves": {str(a): curves[a] for a in anchor_ids},
        "gates": {"r4_triggered_on_real_bge": r4_triggered,
                  "r4_no_confab": r4_no_confab, "graceful": graceful,
                  "r1_exact": r1_exact},
        "regimes_observed": regimes_observed_names,
        "cross_domain_cos_floor": min_cos,
        "verdict": verdict,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(f"\n[summary] wrote {out_dir / 'run_summary.json'}  "
          f"({time.time() - t0:.1f}s)", flush=True)
    return 0 if verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-domain fade eval: exercise R4 on real bge (Bible + ERAG).")
    ap.add_argument("--book", default=DEFAULT_BOOK,
                    help="Bible book for the domain-A anchors (default: john)")
    ap.add_argument("--chapter", type=int, default=DEFAULT_CHAPTER,
                    help="Bible chapter (default: 1)")
    ap.add_argument("--translation", default="oeb-us")
    ap.add_argument("--erag-path", default=ERAG_PATH,
                    help="path to the ERAG documents parquet (gitignored untracked "
                         "data; cols doc_id, source_type, title, content)")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-anchors", type=int, default=8,
                    help="Bible verses to ingest as probe anchors (domain A)")
    ap.add_argument("--n-erag", type=int, default=300,
                    help="ERAG chunks to stream (domain B; must be >= max step)")
    ap.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS,
                    help="erag-step grid at which to record the regime/cos")
    ap.add_argument("--chunk-chars", type=int, default=600,
                    help="truncate each ERAG chunk to this many chars")
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--cos-ring", type=float, default=0.95)
    ap.add_argument("--cos-gist", type=float, default=0.40,
                    help="gist/forgotten threshold (default 0.40 -- calibrated for "
                         "real bge by THIS eval: sits between the cross-domain cos "
                         "floor ~0.37 and the same-domain floor ~0.6, so cross-domain "
                         "anchors reach R4 and same-domain stay R3. 0.30 never "
                         "reaches R4 on real bge (floor too high).)")
    ap.add_argument("--ring-capacity", type=int, default=32)
    ap.add_argument("--expand-tokens", type=int, default=64)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda for bge")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    return run(args)


if __name__ == "__main__":
    sys.exit(main())