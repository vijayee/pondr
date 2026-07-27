"""Code fade eval -- does the fade work on source code? (task follow-on to #37).

The fade (short-term memory that degrades verbatim -> gist -> forgotten, routed
by the free cosine recoverability signal) is built, gated, and shipped on PROSE
(Bible OEB-US + ERAG technical runbooks, scripts/eval_fade_bible.py /
eval_fade_cross_domain.py). This eval answers the next question: **does it work
on source code?** Three concerns: (a) chunking must be AST-aware (functions, not
fixed-character windows), (b) a "gist" of code is a raw excerpt not a purpose-
summary, (c) code often needs exact recall of old APIs (so the R4 forgetting
floor is the valuable part). This eval tests them empirically on the project's
own code -- no training, no checkpoint, no HF upload (an eval/probe, analogous to
``eval_fade_cross_domain.py``).

The comparison. bge-small-en-v1.5 is the only embedder, and the repo already
knows it mis-ranks raw code (code-doc z_logit -0.801 vs text-doc +3.969; see
``src/ingestion/code_summarizer.py``). The production mitigation is to embed a
prose SUMMARY of a function (the handle that ranks against prose queries) while
recalling the raw SOURCE (the thing the user wants back). The fade's ``ingest``
was extended with an optional ``blurb_text`` override so the embed handle and
the recalled blurb can differ; this eval is the first user of that split. Three
configs, each run on a same-module + cross-module stream:

  A raw        : ingest(src)                 -- fade on code-as-code (direct test)
  B prose-handle: ingest(summary, blurb_text=src) -- the production design
  C prose-gist : ingest(summary)             -- illustrative: R3 content = purpose-summary

A vs B is the comparison the user asked for: same recalled content (raw source),
different embed handle -> different cosine floors / gradient -> does the prose-
handle embedding make the fade work on code where raw-code embedding doesn't? C
reuses B's state (same summary handle -> identical cos curve by construction),
so it is NOT a separate gate run -- it only prints one faded anchor's R3 content
with blurb=summary next to A/B's blurb=source, to show the raw-excerpt-vs-
purpose-summary gist difference (concern b).

Cold-start safe. ``CodeSectionSummarizer.summarize_sections`` never raises; a
down Oracle (local Ollama http://localhost:11434/v1, model deepseek-v4-flash:cloud)
returns all-None and B/C auto-degrade to A (embed raw source). The script prints
"Oracle unavailable -- prose-handle degraded to raw-source" and A's results stand.

Gates (mirror eval_fade_cross_domain.py, applied to each config's CROSS-MODULE run):
  1. R1 EXACT   -- at step 0 the anchor is REGIME_VERBATIM and its content == src[:blurb_chars]
  2. GRACEFUL   -- >=1 anchor transitions R1 -> R3 -> R4 in order (R3 before R4)
  3. R4 NO-CONFAB -- every R4 content == "[forgotten]"
  4. R4 TRIGGERS -- >=1 anchor reaches R4 on the cross-module stream (the key code
                   question: does cross-module streaming drive cos below cos_gist?)

Floor report (diagnostic, not pass/fail): same-module min cos and cross-module
min cos per config. Verdict = PASS iff all four gates hold on the cross-module
run, per config. Exit 0 if A's cross-module gates pass, else 1 (the honest
negative: bge sees all code as same-domain -> R4 never fires -> fade is thin on
code under that embed handle; documented, not a crash).

Standalone: reads the project's own ``src/`` tree (no licensing issues), reuses
``CodeParser`` (stdlib ast, zero deps) + ``CodeSectionSummarizer`` + the REAL
``FadeMemory`` (real bge, passthrough voice). Writes ``run_summary.json`` to
``--output-dir`` (gitignored untracked data, not committed). CPU-runnable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # repo root

from src.subconscious.fade import (  # noqa: E402
    REGIME_FORGOTTEN,
    REGIME_GIST,
    REGIME_NAME,
    REGIME_VERBATIM,
    FadeConfig,
    FadeMemory,
    bge_embedder,
)
from src.ingestion.code_parser import CodeParser  # noqa: E402
from src.ingestion.code_summarizer import CodeSectionSummarizer  # noqa: E402

DEFAULT_ROOT = "src"
DEFAULT_ANCHOR_MODULE = "src/subconscious/fade.py"
DEFAULT_OUTPUT_DIR = "data/probe/fade_code"
DEFAULT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256]
# Code chunks are larger than ERAG prose chunks (whole functions); 1500 lets most
# functions fit whole. bge truncates at 512 tokens internally regardless, so this
# only bounds the stored blurb / the embed input before that truncation.
DEFAULT_CHUNK_CHARS = 1500


# --------------------------------------------------------------------- corpus
@dataclass
class CodeChunk:
    module: str          # repo-relative path (the "domain")
    heading: str         # `def name(args)` / `class Name` (the section signature)
    content: str         # the section source, truncated to chunk_chars


def load_code_chunks(root: str, chunk_chars: int) -> list[CodeChunk]:
    """Parse every ``.py`` file under ``root`` into one CodeChunk per top-level
    function/class def (stdlib ast backend; the ``<module>`` root section is
    skipped -- it is imports/docstrings, not a function, and the summarizer
    skips it too). Files are walked in sorted order for determinism."""
    parser = CodeParser()
    root_path = Path(root)
    if not root_path.exists():
        raise SystemExit(f"code root not found: {root_path}")
    chunks: list[CodeChunk] = []
    for py in sorted(root_path.rglob("*.py")):
        rel = py.as_posix()
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        doc = parser.parse_text(text, source_path=rel, language="python")
        for sec in doc.sections:
            if sec.heading == "<module>":
                continue                       # imports/docstring, not a function
            body = (sec.content or "").strip()
            if not body:
                continue
            chunks.append(CodeChunk(module=rel, heading=sec.heading,
                                    content=body[:chunk_chars]))
    return chunks


# ------------------------------------------------------------- memo embedder
class _MemoEmbedder:
    """Wraps the bge embedder with a per-text cache so configs A/B/C that embed
    the SAME handle string (B and C both embed the prose summaries) reuse one
    bge pass instead of recomputing. The cache key is the exact text. ``to``
    proxies device moves to the inner embedder (the cache is device-agnostic --
    bge returns float lists either way)."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            hit = self._cache.get(t)
            if hit is None:
                hit = self._inner.encode([t])[0]
                self._cache[t] = hit
            out.append(list(hit))
        return out

    def to(self, device):
        self._inner = self._inner.to(device)
        return self


# --------------------------------------------------------------- one config
@dataclass
class ConfigRun:
    name: str
    embed_handle: str           # "raw" or "summary"
    blurb_kind: str             # "raw" or "summary"
    # cross-module curves + gates
    curves: dict[int, list[dict]] = field(default_factory=dict)
    same_min_cos: float = float("nan")
    cross_min_cos: float = float("nan")
    gates: dict = field(default_factory=dict)
    verdict: str = "FAIL"
    # did this config degrade to raw-source (Oracle down)?
    degraded: bool = False


def _ingest_chunk(mem: FadeMemory, chunk: CodeChunk, summary: str | None,
                  handle: str, blurb_kind: str) -> int:
    """Ingest one chunk under a config's handle/blurb policy."""
    src = chunk.content
    if handle == "summary":
        h = summary if summary is not None else src    # degrade to raw on None
    else:
        h = src
    if blurb_kind == "summary":
        b = summary if summary is not None else src
    else:
        b = src
    return mem.ingest(h, blurb_text=b)


def run_config(
    name: str, handle: str, blurb_kind: str,
    anchors: list[CodeChunk], same_stream: list[CodeChunk], cross_stream: list[CodeChunk],
    summaries: dict[int, str | None], anchor_idx: list[int], same_idx: list[int],
    cross_idx: list[int], cfg: FadeConfig, emb, steps: list[int],
    degraded: bool,
) -> ConfigRun:
    """Run one config on same-module + cross-module streams; collect curves,
    floors, and the four gates on the cross-module run.

    ``summaries`` is the chunk-index-keyed dict of prose summaries (or None per
    chunk when the Oracle was down); ``anchor_idx``/``same_idx``/``cross_idx``
    map each stream's chunks to their index in that dict."""
    mem = FadeMemory(cfg, emb, voice=None)
    step_set = set(steps)

    # cross-module stream: ingest anchors first, then cross_stream; probe at grid.
    anchor_ids: list[int] = []
    curves: dict[int, list[dict]] = {}
    for chunk, gi in zip(anchors, anchor_idx):
        aid = _ingest_chunk(mem, chunk, summaries.get(gi), handle, blurb_kind)
        anchor_ids.append(aid)
        curves[aid] = []

    def probe_cross(step: int) -> None:
        for aid in anchor_ids:
            r = mem.recall_anchor(aid)
            if r is None:
                continue
            curves[aid].append({"step": step, "regime": r.regime,
                                "regime_name": REGIME_NAME[r.regime],
                                "cos": r.cos, "content": r.content})

    if 0 in step_set:
        probe_cross(0)
    for step, (chunk, gi) in enumerate(zip(cross_stream, cross_idx), start=1):
        _ingest_chunk(mem, chunk, summaries.get(gi), handle, blurb_kind)
        if step in step_set:
            probe_cross(step)

    # ---- gates on the cross-module run
    r4_triggered = any(pt["regime"] == REGIME_FORGOTTEN
                       for a in anchor_ids for pt in curves[a])
    r4_no_confab = all(pt["content"] == "[forgotten]"
                       for a in anchor_ids for pt in curves[a]
                       if pt["regime"] == REGIME_FORGOTTEN)
    graceful = False
    for a in anchor_ids:
        regs = [pt["regime"] for pt in curves[a]]
        if REGIME_GIST in regs and REGIME_FORGOTTEN in regs:
            if regs.index(REGIME_GIST) < regs.index(REGIME_FORGOTTEN):
                graceful = True
                break
    r1_exact = False
    for a, chunk, gi in zip(anchor_ids, anchors, anchor_idx):
        for pt in curves[a]:
            if pt["step"] == 0 and pt["regime"] == REGIME_VERBATIM:
                expect = (chunk.content if blurb_kind == "raw"
                          else (summaries.get(gi) or chunk.content))[: cfg.blurb_chars]
                if pt["content"].strip() == expect.strip():
                    r1_exact = True
                    break
        if r1_exact:
            break
    verdict = "PASS" if (r4_triggered and r4_no_confab and graceful
                        and r1_exact) else "FAIL"
    cross_min = min((pt["cos"] for a in anchor_ids for pt in curves[a]),
                   default=float("nan"))

    # ---- same-module stream (the same-domain floor): fresh FadeMemory, ingest
    # the same anchors, then the same-module siblings; probe at grid steps that
    # fit, take the min cos. A short same-module stream just means fewer probes.
    mem_s = FadeMemory(cfg, emb, voice=None)
    s_anchor_ids = [_ingest_chunk(mem_s, chunk, summaries.get(gi), handle, blurb_kind)
                    for chunk, gi in zip(anchors, anchor_idx)]
    s_curves: dict[int, list[dict]] = {a: [] for a in s_anchor_ids}

    def probe_same(step: int) -> None:
        for a in s_anchor_ids:
            r = mem_s.recall_anchor(a)
            if r is None:
                continue
            s_curves[a].append({"step": step, "cos": r.cos, "regime": r.regime})

    if 0 in step_set:
        probe_same(0)
    for step, (chunk, gi) in enumerate(zip(same_stream, same_idx), start=1):
        _ingest_chunk(mem_s, chunk, summaries.get(gi), handle, blurb_kind)
        if step in step_set:
            probe_same(step)
    # also take the final-state cos (the deepest fade same-module reaches)
    for a in s_anchor_ids:
        r = mem_s.recall_anchor(a)
        if r is not None:
            s_curves[a].append({"step": len(same_stream), "cos": r.cos,
                                 "regime": r.regime})
    same_min = min((pt["cos"] for a in s_anchor_ids for pt in s_curves[a]),
                   default=float("nan"))

    return ConfigRun(name=name, embed_handle=handle, blurb_kind=blurb_kind,
                     curves=curves, same_min_cos=same_min,
                     cross_min_cos=cross_min,
                     gates={"r1_exact": r1_exact, "graceful": graceful,
                            "r4_no_confab": r4_no_confab,
                            "r4_triggers": r4_triggered},
                     verdict=verdict, degraded=degraded)


# --------------------------------------------------------------------- main
def run(args) -> int:
    t0 = time.time()
    # 1. Corpus.
    print(f"[code] parsing {args.root}/**/*.py ...", flush=True)
    chunks = load_code_chunks(args.root, args.chunk_chars)
    if not chunks:
        raise SystemExit(f"no code chunks parsed from {args.root}")
    by_module: dict[str, list[CodeChunk]] = {}
    for c in chunks:
        by_module.setdefault(c.module, []).append(c)
    print(f"[code] {len(chunks)} chunks across {len(by_module)} modules",
          flush=True)

    anchor_mod = args.anchor_module
    if anchor_mod not in by_module:
        raise SystemExit(
            f"anchor module {anchor_mod!r} not parsed (have: "
            f"{sorted(by_module)[:8]}{'...' if len(by_module) > 8 else ''}). "
            f"Pass --anchor-module <repo-rel .py path>.")
    mod_chunks = by_module[anchor_mod]
    n_anchors = min(args.n_anchors, len(mod_chunks))
    anchors = mod_chunks[:n_anchors]
    same_stream = mod_chunks[n_anchors:]
    print(f"[code] anchor module {anchor_mod}: {len(mod_chunks)} funcs; "
          f"{n_anchors} anchors, {len(same_stream)} same-module siblings",
          flush=True)

    # 2. Cross-module stream (deterministic shuffle of all other modules' chunks).
    cross_pool = [c for m in by_module if m != anchor_mod for c in by_module[m]]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(cross_pool))
    cross_stream = [cross_pool[int(i)] for i in order[:args.n_stream]]
    if len(cross_stream) < max(args.steps):
        raise RuntimeError(
            f"only {len(cross_stream)} cross-module chunks -- need >= max step "
            f"{max(args.steps)}; raise --n-stream or parse a larger --root.")
    print(f"[code] {len(cross_pool)} cross-module chunks; using {len(cross_stream)}",
          flush=True)

    # 3. Prose summaries (cold-start safe). Summarize the union of chunks used
    # across configs (anchors + same + cross) ONCE; reuse for B and C. Index by
    # position in `chunks` so the same summary serves every config.
    idx_of = {id(c): i for i, c in enumerate(chunks)}
    used = anchors + same_stream + cross_stream
    used_idx = [idx_of[id(c)] for c in used]
    # Build a de-duplicated (heading, content) batch in chunk-index order.
    used_idx_sorted = sorted(set(used_idx))
    items = [(chunks[i].heading, chunks[i].content) for i in used_idx_sorted]
    print(f"[summ] requesting {len(items)} prose summaries from Oracle "
          f"(model={args.oracle_model})...", flush=True)
    summarizer = CodeSectionSummarizer(model=args.oracle_model,
                                       timeout=args.oracle_timeout)
    sums_sorted = summarizer.summarize_sections(items)
    summaries_by_idx: dict[int, str | None] = {
        used_idx_sorted[j]: sums_sorted[j] for j in range(len(used_idx_sorted))
    }
    n_have = sum(1 for v in summaries_by_idx.values() if v)
    oracle_up = n_have > 0
    degraded = not oracle_up
    if degraded:
        print("[summ] Oracle unavailable -- prose-handle degraded to raw-source "
              "(B/C fall back to A; A's results stand).", flush=True)
    else:
        print(f"[summ] {n_have}/{len(items)} summaries returned; "
              f"summarizer stats: {summarizer.get_stats()}", flush=True)

    def summ_for(chunk: CodeChunk) -> str | None:
        return summaries_by_idx.get(idx_of[id(chunk)])

    # 4. bge embedder (shared memo wrapper -> B and C reuse the summary embeds).
    print(f"[bge] loading bge-small-en-v1.5 (device={args.device})...", flush=True)
    inner = bge_embedder()
    if args.device != "cpu":
        try:
            inner = inner.to(args.device)
        except Exception:
            pass
    emb = _MemoEmbedder(inner)

    cfg = FadeConfig(decay=args.decay, cos_ring=args.cos_ring,
                     cos_gist=args.cos_gist, ring_capacity=args.ring_capacity,
                     regime2_enabled=False, expand_tokens=args.expand_tokens,
                     blurb_chars=args.blurb_chars)
    print(f"[fade] decay={cfg.decay} ring={cfg.ring_capacity} "
          f"cos_ring={cfg.cos_ring} cos_gist={cfg.cos_gist} "
          f"blurb_chars={cfg.blurb_chars} (R2 off)", flush=True)

    # 5. Run the three configs. A always runs; B/C run but degrade to raw when
    # the Oracle is down (summaries all-None -> handle falls back to src).
    runs: list[ConfigRun] = []
    # Index alignment: each stream's chunks -> their index in `chunks` (so the
    # summaries_by_idx dict can be looked up per chunk).
    a_idx = [idx_of[id(c)] for c in anchors]
    s_idx = [idx_of[id(c)] for c in same_stream]
    c_idx = [idx_of[id(c)] for c in cross_stream]
    for name, handle, blurb_kind in [("A raw", "raw", "raw"),
                                    ("B prose-handle", "summary", "raw"),
                                    ("C prose-gist", "summary", "summary")]:
        # When degraded, force handle to raw so the config is an honest A clone
        # (the summary is None -> _ingest_chunk falls back to src anyway, but
        # label it so the report is honest).
        eff_handle = "raw" if (handle == "summary" and degraded) else handle
        r = run_config(name, eff_handle, blurb_kind, anchors, same_stream,
                       cross_stream, summaries_by_idx,
                       a_idx, s_idx, c_idx, cfg, emb, args.steps,
                       degraded=(degraded and handle == "summary"))
        # C reuses B's cos curve by construction (same summary handle); if
        # degraded, C == A. Mark verdict N/A for C (illustrative, not gated).
        if name.startswith("C"):
            r.verdict = "N/A (illustrative)"
        runs.append(r)

    A, B, C = runs[0], runs[1], runs[2]

    # 6. Print curves + gates per config.
    for r in runs:
        print(f"\n{'='*84}\nCONFIG {r.name}  (handle={r.embed_handle}, "
              f"blurb={r.blurb_kind}"
              f"{', DEGRADED to raw' if r.degraded else ''})\n{'='*84}",
              flush=True)
        print(f"  same-module cos floor : {r.same_min_cos:.3f}", flush=True)
        print(f"  cross-module cos floor: {r.cross_min_cos:.3f} "
              f"(cos_gist={cfg.cos_gist})", flush=True)
        # one representative anchor curve (anchor 0)
        a0 = list(r.curves.keys())[0] if r.curves else None
        if a0 is not None:
            print(f"\n  anchor 0 curve ({anchors[0].heading}):", flush=True)
            print("    step : regime        cos    | content-kind", flush=True)
            for pt in r.curves[a0]:
                kind = ("EXACT src" if pt["regime"] == REGIME_VERBATIM
                        else "retrieved blurb" if pt["regime"] == REGIME_GIST
                        else "[forgotten]")
                print(f"    {pt['step']:>4} : {pt['regime_name']:<13} "
                      f"{pt['cos']:.3f} | {kind}", flush=True)
        if r.verdict != "N/A (illustrative)":
            print(f"\n  gates: r1_exact={r.gates['r1_exact']} "
                  f"graceful={r.gates['graceful']} "
                  f"r4_no_confab={r.gates['r4_no_confab']} "
                  f"r4_triggers={r.gates['r4_triggers']}", flush=True)
            if not r.gates["r4_triggers"]:
                print(f"  NOTE: R4 did NOT trigger -- cross-module cos floor "
                      f"({r.cross_min_cos:.3f}) stayed above cos_gist="
                      f"{cfg.cos_gist}. bge sees all code as same-domain under "
                      f"the {r.embed_handle} handle -> fade is THIN on code "
                      f"under this handle.", flush=True)
        print(f"  VERDICT: {r.verdict}", flush=True)

    # 7. A-vs-C R3-content side-by-side (concern b: raw excerpt vs purpose-summary).
    print(f"\n{'='*84}\nR3 CONTENT SIDE-BY-SIDE (raw excerpt vs purpose-summary gist)"
          f"\n{'='*84}", flush=True)
    # pick the first anchor that reaches R3 in B, at the first R3 lag
    a_pick, lag_pick, found = None, None, False
    for a in B.curves:
        for pt in B.curves[a]:
            if pt["regime"] == REGIME_GIST:
                a_pick, lag_pick, found = a, pt["step"], True
                break
        if found:
            break
    if not found:
        print("  (no anchor reached R3 in B; skipping the side-by-side.)",
              flush=True)
    else:
        # a_pick is an anchor_id; anchor_ids are sequential 0..n-1 in ingest
        # order, so anchors[a_pick] is the matching anchor chunk.
        chunk = anchors[a_pick] if a_pick < len(anchors) else None
        summ = summ_for(chunk) if chunk is not None else None
        print(f"  anchor {a_pick} ({chunk.heading}) at step {lag_pick}:",
              flush=True)
        for r in runs:
            pt = next((p for p in r.curves.get(a_pick, [])
                       if p["step"] == lag_pick), None)
            if pt is None:
                continue
            label = f"{r.name} [regime={pt['regime_name']}, cos={pt['cos']:.3f}]"
            body = pt["content"]
            shown = body if len(body) <= 200 else body[:197] + "..."
            print(f"  {label}\n    content = {shown!r}", flush=True)
        if summ:
            print(f"  (prose summary for this anchor: {summ!r})", flush=True)

    # 8. Floor report + diagnosis.
    print(f"\n{'='*84}\nFLOOR REPORT\n{'='*84}", flush=True)
    for r in runs:
        print(f"  {r.name:<18} handle={r.embed_handle:<7} "
              f"same-floor={r.same_min_cos:.3f} "
              f"cross-floor={r.cross_min_cos:.3f} "
              f"r4_triggers={r.gates.get('r4_triggers')}", flush=True)
    print(f"\n  cos_gist={cfg.cos_gist}", flush=True)
    if A.gates["r4_triggers"]:
        print("  -> A (raw) reaches R4 on code: the fade works on code-as-code.",
              flush=True)
    else:
        print("  -> A (raw) does NOT reach R4 on code: bge sees all code as "
              "same-domain under raw-source embed -> fade is THIN on code-as-"
              "code (matches the known bge-mis-ranks-code finding).", flush=True)
    if not degraded and B.gates["r4_triggers"]:
        print("  -> B (prose-handle) reaches R4 on code: the prose-summary "
              "embed is enough to separate code domains for the fade.",
              flush=True)
    elif not degraded and not B.gates["r4_triggers"] and A.gates["r4_triggers"]:
        print("  -> B (prose-handle) does NOT reach R4 though A does: the prose "
              "handle is WORSE for code-domain separation here (unexpected).",
              flush=True)
    elif not degraded and not B.gates["r4_triggers"]:
        print("  -> B (prose-handle) does NOT reach R4 either: bge sees all "
              "code (even via prose summaries) as same-domain -> fade is thin "
              "on code regardless of handle; the fade is a prose-memory "
              "mechanism (concern c).", flush=True)

    print(f"\n  A VERDICT: {A.verdict}  (exit code reflects A's cross-module gates)",
          flush=True)

    # 9. run_summary.json
    summary = {
        "probe": "fade_code",
        "purpose": "does the fade work on source code? raw vs prose-handle embed",
        "root": args.root,
        "anchor_module": anchor_mod,
        "n_anchors": n_anchors,
        "n_same_module": len(same_stream),
        "n_cross_module": len(cross_stream),
        "n_modules": len(by_module),
        "oracle_up": oracle_up,
        "oracle_model": args.oracle_model,
        "config": {"decay": cfg.decay, "cos_ring": cfg.cos_ring,
                   "cos_gist": cfg.cos_gist, "ring_capacity": cfg.ring_capacity,
                   "blurb_chars": cfg.blurb_chars, "regime2_enabled": False,
                   "voice": "passthrough"},
        "runs": [
            {"name": r.name, "embed_handle": r.embed_handle,
             "blurb_kind": r.blurb_kind, "degraded": r.degraded,
             "same_min_cos": r.same_min_cos, "cross_min_cos": r.cross_min_cos,
             "gates": r.gates, "verdict": r.verdict}
            for r in runs
        ],
        "a_verdict": A.verdict,
        "steps": args.steps,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(f"\n[summary] wrote {out_dir / 'run_summary.json'}  "
          f"({time.time() - t0:.1f}s)", flush=True)
    return 0 if A.verdict == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Code fade eval: does the fade work on source code?")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="repo-relative code root to parse (default: src)")
    ap.add_argument("--anchor-module", default=DEFAULT_ANCHOR_MODULE,
                    help="repo-rel .py path whose first --n-anchors funcs are "
                         "the tracked anchors (default: src/subconscious/fade.py)")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-anchors", type=int, default=8,
                    help="functions of the anchor module to ingest as anchors")
    ap.add_argument("--n-stream", type=int, default=256,
                    help="cross-module chunks to stream (must be >= max step)")
    ap.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS,
                    help="cross-module step grid at which to record regime/cos")
    ap.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS,
                    help="truncate each function's source to this many chars "
                         "(bge truncates at 512 tokens internally regardless)")
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--cos-ring", type=float, default=0.95)
    ap.add_argument("--cos-gist", type=float, default=0.40,
                    help="gist/forgotten threshold (default 0.40 -- calibrated "
                         "for real bge on prose; the code cross-module floor is "
                         "the thing this eval measures against it)")
    ap.add_argument("--ring-capacity", type=int, default=32)
    ap.add_argument("--blurb-chars", type=int, default=600,
                    help="stored blurb length (the recalled content cap)")
    ap.add_argument("--expand-tokens", type=int, default=64)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda for bge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle-model", default="deepseek-v4-flash:cloud",
                    help="Oracle model for prose summaries (local Ollama)")
    ap.add_argument("--oracle-timeout", type=float, default=60.0)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    return run(args)


if __name__ == "__main__":
    sys.exit(main())