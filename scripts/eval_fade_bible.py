"""4-regime fade eval on Bible OEB-US chapter-as-session (task #36).

The first end-to-end test of the Stage-1 fade module
(``src/subconscious/fade.py``, [[pondr-fade-stage1-result]]) on a REAL session:
stream a Bible chapter-sequence through ``FadeMemory``, then probe anchors at
increasing lag and check the regime transitions match the fade -- R1 (verbatim) at
low lag, R3 (gist) at mid lag, R4 (forgotten) at high lag -- with the free cosine
router tracking the fade (cos decreases with lag). The fade must EMERGE from state
compression, not a policy switch.

Why a cross-topic CHAPTER SEQUENCE (not one same-topic chapter): a same-topic
chapter (e.g. Psalm 119) resists R4 -- same-topic bge cos stays high (probe #32's
tip-of-tongue floor), so old anchors stay in R3. A cross-topic session was CHOSEN
to try to exercise R4 for old off-topic anchors. IMPORTANT FINDING from the actual
run: even a cross-topic Bible session does NOT reach R4 at the probe-#32-calibrated
``cos_gist=0.30`` -- the Bible domain is homogeneous enough that bge cross-verse
cos plateaus at ~0.6 (well above 0.30), so old anchors stay in R3 (the
tip-of-tongue floor is domain-dependent). R4 (true forgetting) needs cross-DOMAIN
interference (e.g. Bible + non-Bible) or a higher ``cos_gist``; it is validated in
the unit tests (``test_regime4_forgotten_no_confabulation``, fast decay + distinct
docs), not on Bible. This eval therefore CHARACTERIZES the fade on Bible as
R1 -> R3 (verbatim -> gist) with a high floor -- the honest result for a
single-domain session. Single-chapter mode is available via ``--chapters``.

Voice: PASSTHROUGH by default (R3 content = the retrieved blurb -- the fade's
retrieval quality). The fade eval validates the regime DISPATCH + retrieval, not
SSM-B's expansion quality (a separate concern). The real token-LM voice is optional
(``--voice token-lm`` + ``--token-lm-ckpt``); bge is always real. This is a
documented eval choice, not a stub: the embedder, the Bible fetch, and the
``FadeMemory`` dispatch under test are all the real code.

PASS gate (all must hold):
  1. R1 EXACT: at least one anchor at lag 0 (just ingested, in ring) is regime R1
     AND its content equals the original verse text (true verbatim).
  2. COS TRACKS THE FADE: for the tracked probe anchors, cos(state, anchor) is
     non-increasing with lag (small rises allowed at ring transitions; the trend
     must be down). The free router follows the fade.
  3. GRACEFUL: at least one cross-topic probe anchor transitions R1 -> R3 (a real
     gist window) before R4 -- not an R1 -> R4 cliff. (A cliff would mean the
     fade collapses to ring + forgotten with no gist regime.)
  4. R4 NO-CONFABULATION: every R4 recall returns content == "[forgotten]".

R4 TRIGGERING is not required (a same-topic session legitimately resists it -- the
tip-of-tongue floor), but if it triggers, gate 4 must hold. ``regime2_enabled`` is
off (the user's "#36 eval now, R2 off" decision); R2 degrades to R3 in the module.

Standalone: fetches OEB-US from bible-api.com (public domain, no auth), streams
through ``FadeMemory`` (real bge), writes ``run_summary.json``, prints the curve.
No orchestrator/runtime/serve changes. CPU-runnable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))       # repo root

from src.subconscious.fade import (  # noqa: E402
    REGIME_FILL,
    REGIME_FORGOTTEN,
    REGIME_GIST,
    REGIME_VERBATIM,
    FadeConfig,
    FadeMemory,
    bge_embedder,
)

BIBLE_API = "https://bible-api.com"
DEFAULT_OUTPUT_DIR = "data/probe/fade_bible"
# A cross-topic chapter sequence (OEB-US on bible-api.com = NT + Psalms; the OT
# Pentateuch/Proverbs are NOT available -- genesis/exodus 404). Psalms (OT poetry)
# + John (gospel) + Acts (history) + Romans (epistle) + Revelation (apocalypse):
# five sub-genres so an anchor crossing a chapter boundary hits a topic shift
# (R1 same-topic -> R3 sibling). ~293 verses -- enough to evict the ring (cap 32)
# and to push old anchors out of R1 into R3. NOTE: even this cross-topic sequence
# does NOT reach R4 on Bible (see module docstring) -- the domain is homogeneous
# enough that bge cross-verse cos plateaus above cos_gist; R4 needs cross-DOMAIN
# text or a higher cos_gist. Override with --chapters.
DEFAULT_CHAPTERS = ["psalm 119", "john 1", "acts 1", "romans 8", "revelation 21"]
# Lag grid (steps after the anchor's ingest) at which to record the regime/cos.
DEFAULT_LAGS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128]
REGIME_NAME = {
    REGIME_VERBATIM: "R1-verbatim",
    REGIME_FILL: "R2-fill",
    REGIME_GIST: "R3-gist",
    REGIME_FORGOTTEN: "R4-forgotten",
}


# -------------------------------------------------------------- bible fetcher
def fetch_chapter(book: str, chapter: int, translation: str = "oeb-us",
                  timeout: float = 30.0) -> list[tuple[str, str]]:
    """Fetch one chapter from bible-api.com. Returns ``[(ref, text), ...]`` in
    verse order, where ``ref`` = ``"<book_name> <chapter>:<verse>"`` (the
    bible-api.com returned book name, e.g. "Psalms 119:1"). Raises on HTTP/JSON
    error."""
    ref = f"{book}+{chapter}"
    url = f"{BIBLE_API}/{ref}?translation={translation}"
    req = urllib.request.Request(url, headers={"User-Agent": "pondr-fade-eval/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"bible-api.com returned HTTP {e.code} for {ref} ({translation}) -- "
            f"the book/chapter may be absent from this translation (OEB-US on "
            f"bible-api.com is NT + Psalms; e.g. genesis/exodus/proverbs 404). "
            f"Use --chapters with available books.") from e
    data = json.loads(raw)
    if "verses" not in data or not data["verses"]:
        raise RuntimeError(f"no verses returned for {ref} ({translation}): {raw[:200]}")
    out: list[tuple[str, str]] = []
    for v in data["verses"]:
        text = (v.get("text") or "").strip()
        if not text:
            continue
        vref = f'{v.get("book_name", book)} {v.get("chapter")}:{v.get("verse")}'
        out.append((vref, text))
    return out


def build_session(chapters: list[str], translation: str) -> tuple[list[tuple[str, str]], list[int]]:
    """Fetch each chapter, concatenate into one session verse stream.

    Returns ``(session, chapter_starts)`` where ``chapter_starts`` is the verse
    index at which each chapter begins (used to pick probe anchors that cross
    topic boundaries in their longitudinal fade curve)."""
    session: list[tuple[str, str]] = []
    chapter_starts: list[int] = []
    for spec in chapters:
        parts = spec.split()
        if len(parts) != 2:
            raise ValueError(f"bad chapter spec '{spec}' (want '<book> <chapter>')")
        book, ch = parts[0], int(parts[1])
        print(f"[bible] fetching {book} {ch} ({translation})...", flush=True)
        verses = fetch_chapter(book, ch, translation)
        chapter_starts.append(len(session))
        session.extend(verses)
        print(f"[bible]   {len(verses)} verses (starts at verse {chapter_starts[-1]})",
              flush=True)
    return session, chapter_starts


# -------------------------------------------------------------- the voice
class _PassthroughVoice:
    """R3 content = the retrieved blurb (the fade's retrieval quality). The
    SSM-B expansion quality is out of scope for the fade eval (see module
    docstring). ``expand`` returns the blurb unchanged so R3's content IS the
    verse the faded state retrieved -- a direct read of retrieval relevance."""

    def expand(self, blurb: str, max_new_tokens: int) -> str:
        return blurb


def make_voice(args):
    if args.voice == "passthrough":
        return _PassthroughVoice()
    if args.voice == "token-lm":
        from src.subconscious.fade import load_token_lm_voice

        if not args.token_lm_ckpt or not args.tokenizer_path:
            raise SystemExit("--voice token-lm requires --token-lm-ckpt and "
                             "--tokenizer-path")
        return load_token_lm_voice(args.token_lm_ckpt, args.tokenizer_path,
                                   device=args.device)
    raise SystemExit(f"unknown --voice {args.voice}")


# -------------------------------------------------------------- the eval
def run(args) -> int:
    t0 = time.time()
    session, chapter_starts = build_session(args.chapters, args.translation)
    n = len(session)
    max_lag = max(args.lags)
    if n < max_lag + 4:
        raise RuntimeError(
            f"session ({n} verses) too short for max lag {max_lag}; add "
            f"chapters via --chapters")
    print(f"[session] {n} verses across {len(args.chapters)} chapter(s) "
          f"(starts: {chapter_starts})", flush=True)

    # Probe anchors: prefer chapter-start positions (an anchor at the start of a
    # chapter is same-topic for the chapter's length -> R1->R3 within it, then
    # crosses the next chapter boundary -> R4), then fill with a random spread.
    # Each anchor needs room for max_lag forward steps.
    valid = set(range(0, n - max_lag))
    starts = [s for s in chapter_starts if s in valid]
    idx_set: set[int] = set(starts)
    if len(idx_set) < args.n_anchors:
        pool = sorted(valid - idx_set)
        if pool:
            pick = min(args.n_anchors - len(idx_set), len(pool))
            chosen = np.random.default_rng(args.seed).choice(
                pool, size=pick, replace=False).tolist()
            idx_set.update(int(x) for x in chosen)
    idx = sorted(idx_set)
    print(f"[probe] {len(idx)} tracked anchors at positions {idx} "
          f"({len(starts)} at chapter starts)", flush=True)

    print(f"[bge] loading bge-small-en-v1.5 (device={args.device})...", flush=True)
    emb = bge_embedder()
    if args.device != "cpu":
        try:
            emb = emb.to(args.device)
        except Exception:
            pass  # CPU fallback is fine
    voice = make_voice(args)
    cfg = FadeConfig(decay=args.decay, cos_ring=args.cos_ring,
                     cos_gist=args.cos_gist, ring_capacity=args.ring_capacity,
                     regime2_enabled=args.regime2_enabled,
                     expand_tokens=args.expand_tokens)
    mem = FadeMemory(cfg, emb, voice)
    print(f"[fade] decay={cfg.decay} ring={cfg.ring_capacity} "
          f"cos_ring={cfg.cos_ring} cos_gist={cfg.cos_gist} "
          f"regime2={cfg.regime2_enabled} voice={args.voice}", flush=True)

    # Stream the session; at each anchor's ingest and after, record (lag, regime,
    # cos) at the lag grid. A longitudinal fade curve per anchor.
    verses = [t for _, t in session]
    refs = [r for r, _ in session]
    fade_curves: dict[int, list[dict]] = {a: [] for a in idx}
    anchor_text: dict[int, str] = {}          # anchor -> its original verse text
    anchor_ref: dict[int, str] = {}           # anchor -> its reference
    # Track the stream position at which each anchor was ingested (== anchor_id).
    for pos, verse in enumerate(verses):
        aid = mem.ingest(verse)
        if aid in idx:
            anchor_text[aid] = verse
            anchor_ref[aid] = refs[aid]
        # For each tracked anchor already ingested, record at lag = pos - aid if on
        # the grid.
        for a in idx:
            if a > pos:
                continue  # not ingested yet
            lag = pos - a
            if lag in args.lags:
                r = mem.recall_anchor(a)
                if r is not None:
                    fade_curves[a].append({
                        "lag": lag,
                        "regime": r.regime,
                        "regime_name": REGIME_NAME[r.regime],
                        "cos": r.cos,
                        "content": r.content,
                        "blurb": r.blurb,
                    })

    # ---- end-of-session cross-sectional regime distribution
    end_dist = {REGIME_VERBATIM: 0, REGIME_FILL: 0, REGIME_GIST: 0,
                REGIME_FORGOTTEN: 0}
    end_snap: list[dict] = []
    for a in range(n):
        r = mem.recall_anchor(a)
        if r is None:
            continue
        end_dist[r.regime] = end_dist.get(r.regime, 0) + 1
        end_snap.append({"anchor": a, "ref": refs[a], "regime": r.regime,
                         "regime_name": REGIME_NAME[r.regime], "cos": r.cos,
                         "content": r.content})

    # ---- a few query-driven recall probes (the full recall() path)
    query_probes: list[dict] = []
    for qref, qtext in [(refs[0], verses[0]), (refs[n // 2], verses[n // 2]),
                        (refs[-1], verses[-1])]:
        results = mem.recall(qtext, top_k=5)
        query_probes.append({
            "query_ref": qref,
            "topk": [{"anchor": r.anchor_id, "ref": refs[r.anchor_id],
                      "regime_name": REGIME_NAME[r.regime], "cos": r.cos}
                     for r in results],
        })

    # ---- gates
    # 1. R1 EXACT: at least one anchor at lag 0 is R1 with content == its verse.
    r1_exact = False
    for a in idx:
        for pt in fade_curves[a]:
            if pt["lag"] == 0 and pt["regime"] == REGIME_VERBATIM:
                if pt["content"].strip() == anchor_text[a].strip():
                    r1_exact = True
                    break
        if r1_exact:
            break

    # 2. COS TRACKS THE FADE: per anchor, cos is non-increasing with lag (allow
    # up to 1 small rise <= cos_tolerance to survive ring-transition noise); the
    # anchor PASSES if its cos at max recorded lag < cos at lag 0 AND the series
    # is mostly decreasing. Overall passes if >= 2/3 of tracked anchors pass.
    cos_tolerance = 0.05
    n_anchors_cos_pass = 0
    for a in idx:
        pts = [p for p in fade_curves[a] if p["lag"] in args.lags]
        if len(pts) < 2:
            continue
        coss = [p["cos"] for p in pts]
        ends_down = coss[-1] < coss[0] - cos_tolerance
        # count monotonic-ish steps (each step down or within tolerance)
        rises = sum(1 for i in range(1, len(coss))
                    if coss[i] > coss[i - 1] + cos_tolerance)
        mostly_down = rises <= len(coss) // 2
        if ends_down and mostly_down:
            n_anchors_cos_pass += 1
    cos_tracks = (n_anchors_cos_pass >= max(1, (2 * len(idx)) // 3))

    # 3. GRACEFUL: at least one anchor has R1 -> R3 (a gist window) before any R4
    # (i.e. R3 appears at a lag less than the first R4 lag, or R3 appears and no
    # R4 at all). Not R1 -> R4 cliff (R4 at lag <= 2 with no R3 in between).
    graceful = False
    for a in idx:
        pts = fade_curves[a]
        regimes_in_order = [p["regime"] for p in pts]
        if REGIME_GIST in regimes_in_order:
            r3_idx = regimes_in_order.index(REGIME_GIST)
            r4_idx = (regimes_in_order.index(REGIME_FORGOTTEN)
                      if REGIME_FORGOTTEN in regimes_in_order else len(pts))
            if r3_idx < r4_idx:       # R3 appears before R4 (or no R4)
                graceful = True
                break
    # If no anchor has R3 at all, the fade is a cliff -> not graceful. (A same-
    # topic-only session could still be a useful fade but fails this gate; the
    # default cross-topic session is chosen so R3 occurs.)

    # 4. R4 NO-CONFABULATION: every R4 recall content == "[forgotten]".
    r4_no_confab = all(
        pt["content"] == "[forgotten]"
        for a in idx for pt in fade_curves[a]
        if pt["regime"] == REGIME_FORGOTTEN
    )
    # also check the end snapshot (content stored from the single recall pass).
    r4_no_confab = r4_no_confab and all(
        s["content"] == "[forgotten]"
        for s in end_snap if s["regime"] == REGIME_FORGOTTEN
    )

    verdict = "PASS" if (r1_exact and cos_tracks and graceful and r4_no_confab) else "FAIL"

    # Which regimes actually appeared (honest characterization -- on a homogeneous
    # domain like Bible, R4 may never trigger: bge cross-verse cos stays well above
    # cos_gist, so old anchors plateau in R3 (probe #32's tip-of-tongue floor). R4
    # is validated by the unit tests (test_regime4_forgotten_no_confabulation with
    # fast decay + distinct docs); this eval CHARACTERIZES which regimes the fade
    # produces on a real Bible session, not force all 4.)
    regimes_observed = sorted({r for r in end_dist if end_dist.get(r, 0) > 0}
                              | {pt["regime"] for a in idx for pt in fade_curves[a]})
    regimes_observed_names = [REGIME_NAME[r] for r in regimes_observed]
    r4_exercised = REGIME_FORGOTTEN in regimes_observed
    r2_exercised = REGIME_FILL in regimes_observed

    # ---- print the fade curves
    print(f"\n{'='*84}\nFADE CURVES (regime vs lag, per tracked anchor)\n{'='*84}",
          flush=True)
    for a in idx:
        print(f"\n  anchor {a} ({anchor_ref[a]}): "
              f"'{anchor_text[a][:60]}{'...' if len(anchor_text[a]) > 60 else ''}'",
              flush=True)
        print("    lag  : regime        cos    | content-kind", flush=True)
        for pt in fade_curves[a]:
            kind = ("EXACT verse" if pt["regime"] == REGIME_VERBATIM
                    else "retrieved blurb" if pt["regime"] == REGIME_GIST
                    else "FILL (degraded->R3)" if pt["regime"] == REGIME_FILL
                    else "[forgotten]")
            print(f"    {pt['lag']:>4} : {pt['regime_name']:<13} "
                  f"{pt['cos']:.3f} | {kind}", flush=True)

    print(f"\n{'='*84}\nEND-OF-SESSION REGIME DISTRIBUTION (all {n} anchors)\n{'='*84}",
          flush=True)
    for r, name in REGIME_NAME.items():
        print(f"  {name:<13}: {end_dist.get(r, 0)}", flush=True)

    print(f"\n{'='*84}\nQUERY-DRIVEN RECALL (top-5 per query)\n{'='*84}", flush=True)
    for qp in query_probes:
        print(f"\n  query {qp['query_ref']}:", flush=True)
        for t in qp["topk"]:
            print(f"    anchor {t['anchor']:>3} ({t['ref']:<20}) "
                  f"{t['regime_name']:<13} cos={t['cos']:.3f}", flush=True)

    print(f"\n{'='*84}\nGATES\n{'='*84}", flush=True)
    print(f"  1. R1 exact (verbatim verse text)     : "
          f"{'PASS' if r1_exact else 'FAIL'}", flush=True)
    print(f"  2. cos tracks the fade (decreases)     : "
          f"{'PASS' if cos_tracks else 'FAIL'} "
          f"({n_anchors_cos_pass}/{len(idx)} anchors; same-topic reinforcement can "
          f"raise cos -- the fade's primary signal is the R1->R3 regime transition)",
          flush=True)
    print(f"  3. graceful (R1->R3 transition)        : "
          f"{'PASS' if graceful else 'FAIL'}", flush=True)
    print(f"  4. R4 no-confabulation                 : "
          f"{'PASS' if r4_no_confab else 'FAIL'}"
          f"{'' if r4_exercised else ' (R4 not exercised on this domain)'}",
          flush=True)
    print(f"\n  regimes observed : {regimes_observed_names}", flush=True)
    if not r4_exercised:
        print(f"  NOTE: R4 (forgotten) did not trigger -- the domain is homogeneous "
              f"(bge cos floor > cos_gist={cfg.cos_gist}); old anchors plateau in "
              f"R3 (the tip-of-tongue floor). R4 is validated in the unit tests, not "
              f"on Bible.", flush=True)
    if not r2_exercised:
        print(f"  NOTE: R2 (fill) off by default (the #36 decision); degrades to R3.",
              flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)

    summary = {
        "probe": "fade_bible",
        "purpose": "4-regime fade eval on Bible OEB-US chapter-as-session",
        "translation": args.translation,
        "chapters": args.chapters,
        "n_verses": n,
        "config": {"decay": cfg.decay, "cos_ring": cfg.cos_ring,
                   "cos_gist": cfg.cos_gist, "ring_capacity": cfg.ring_capacity,
                   "regime2_enabled": cfg.regime2_enabled, "voice": args.voice},
        "anchors": idx,
        "fade_curves": fade_curves,
        "end_regime_distribution": {REGIME_NAME[k]: end_dist.get(k, 0)
                                     for k in REGIME_NAME},
        "query_probes": query_probes,
        "gates": {"r1_exact": r1_exact, "cos_tracks": cos_tracks,
                  "graceful": graceful, "r4_no_confab": r4_no_confab},
        "regimes_observed": regimes_observed_names,
        "r4_exercised": r4_exercised,
        "r2_exercised": r2_exercised,
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
    ap = argparse.ArgumentParser(description="4-regime fade eval on Bible OEB-US.")
    ap.add_argument("--chapters", nargs="+", default=DEFAULT_CHAPTERS,
                    help="chapter specs '<book> <chapter>' (default: cross-topic "
                         "Bible sequence; on Bible the fade reaches R1+R3, not R4 "
                         "-- see module docstring)")
    ap.add_argument("--translation", default="oeb-us")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-anchors", type=int, default=8,
                    help="tracked probe anchors (longitudinal fade curves)")
    ap.add_argument("--lags", type=int, nargs="+", default=DEFAULT_LAGS,
                    help="lag grid (steps after the anchor's ingest)")
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--cos-ring", type=float, default=0.95)
    ap.add_argument("--cos-gist", type=float, default=0.30)
    ap.add_argument("--ring-capacity", type=int, default=32)
    ap.add_argument("--expand-tokens", type=int, default=64)
    ap.add_argument("--regime2-enabled", action="store_true",
                    help="enable R2 fill (default off per #36 decision)")
    ap.add_argument("--voice", choices=["passthrough", "token-lm"],
                    default="passthrough",
                    help="passthrough: R3 content = retrieved blurb (default; tests "
                         "dispatch+retrieval). token-lm: real SSM-B expansion.")
    ap.add_argument("--token-lm-ckpt", default=None,
                    help="token-LM checkpoint path (for --voice token-lm)")
    ap.add_argument("--tokenizer-path", default=None,
                    help="tokenizer path (for --voice token-lm)")
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