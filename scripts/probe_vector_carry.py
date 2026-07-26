"""Vector-carry probe -- the vector-decay curve (FADE architecture, task #32).

The other half of the no-training gate (with ``probe_verbatim_reach.py``). The
decided fade architecture ([[pondr-fade-architecture-router]]) puts the fade on
SSM-A, which carries ``bge(chunk)`` and fades slowly over the stream. This probe
measures THAT curve: inject a chunk's bge vector, let the recurrence compress it
as subsequent chunks stream in, read the decayed state as a retrieval query, and
retrieve the original chunk. = the vector decay curve. Tells us whether the vector
fades GRACEFULLY (Regime 3 viable, the fade is rich) or falls off a cliff.

No-training stand-in for SSM-A (per the design decision: a parallel vector channel
with its own decay, no 388<->256 projection for the probe): a controlled-decay
384-d vector channel -- an exponentially-weighted moving average of chunk vectors,

    state_p = decay * state_{p-1} + write_gate * bge(chunk_p),   state_0 = 0

so the anchor at stream position ``i`` contributes ``decay**N * bge(chunk_i)`` to
``state_{i+N}``. The architecture will TRAIN an SSM to approximate a learnable
version of this; the probe characterizes the curve shape for a few fixed decay
rates, which is what the router's regime boundaries depend on.

The fade is NOT "the vector shrinks" -- cosine retrieval is invariant to a scalar
scale, so a read-only slot that only decays in magnitude would retrieve the anchor
forever (the no-interference CONTROL below confirms this). The fade is the
recurrence OVERWRITING the slot with newer chunk vectors: the anchor's weight
``decay**N`` shrinks while subsequent chunks accumulate, so the query drifts from
the anchor toward the recent-chunk blend. That drift -- exact -> related ->
unrelated -- IS the verbatim -> gist -> forgotten transition in retrieval space.

What the curve tells the architecture:
  - For each decay, the N-window where retrieval is EXACT (Regime 1/3 exact), the
    window where it is a SAME-DOC SIBLING (Regime 3 fuzzy gist -- the vector still
    retrieves a RELATED chunk), and where it is UNRELATED (Regime 4 forgotten).
  - A GRACEFUL transition (a useful exact window -> a sibling/gist window ->
    forgotten) -> Regime 3 is viable, the fade is rich, the architecture holds.
  - A CLIFF (exact -> unrelated in 1-2 steps, no sibling window) -> Regime 3 is
    thin, the fade collapses to ring + forgotten.

Standalone: bge-encodes ERAG chunks (bge-small-en-v1.5, 384-d, frozen open model),
simulates the channel, writes ``run_summary.json``, prints the curve. No
orchestrator/runtime/serve changes. CPU-runnable. ERAG public text only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))       # repo root (src/)
sys.path.insert(0, str(Path(__file__).resolve().parent))               # scripts/

from src.retrieval.vector_search import _sentence_transformers_embedder  # noqa: E402
from probe_verbatim_reach import _iter_erag_content  # noqa: E402

ERAG_PATH = "scripts/_scratch/erag/data/documents/test.parquet"
DEFAULT_OUTPUT_DIR = "data/probe/vector_carry"

# Lags (steps after the anchor's write). N=0 = read immediately after write --
# should be ~exact top-1 (the control confirms the mechanism).
DEFAULT_LAGS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128]
# Decay rates to sweep. Near-1 = slow fade (long verbatim window); lower = fast.
DEFAULT_DECAYS = [0.99, 0.97, 0.95, 0.9, 0.8, 0.5]


# -------------------------------------------------------------- corpus builder
def build_corpus(
    erag_path: str,
    n_docs: int,
    skip_docs: int,
    chunk_words: int,
    min_chunks_per_doc: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream ERAG docs, split each into word chunks, bge-encode them.

    Returns ``(vecs [C,384] fp32 L2-normalized, doc_ids [C] int64)``.
    ``doc_ids[c]`` is the doc index the chunk belongs to (for same-doc/sibling
    analysis). Chunks are in STREAM order (doc 0's chunks, then doc 1's, ...)."""
    print(f"[data] streaming {n_docs} ERAG docs (skip first {skip_docs}) "
          f"chunk_size={chunk_words} words", flush=True)
    chunks: list[str] = []
    doc_ids: list[int] = []
    di = 0
    for content in _iter_erag_content(erag_path, max_docs=n_docs, skip=skip_docs):
        words = content.split()
        if len(words) < chunk_words * min_chunks_per_doc:
            continue  # too short to yield enough chunks
        for j in range(0, len(words), chunk_words):
            piece = " ".join(words[j:j + chunk_words])
            if len(piece.split()) >= max(20, chunk_words // 2):
                chunks.append(piece)
                doc_ids.append(di)
        if doc_ids.count(di) >= min_chunks_per_doc:
            di += 1
        else:
            # drop this doc's chunks if it didn't yield enough
            chunks = [c for c, d in zip(chunks, doc_ids) if d != di]
            doc_ids = [d for d in doc_ids if d != di]
        if di >= n_docs:
            break
    if len(chunks) < 40:
        raise RuntimeError(f"need >=40 chunks; got {len(chunks)} (raise --chunk-words "
                           f"or --n-docs)")
    print(f"[data] {len(chunks)} chunks across {di} docs  "
          f"(avg {len(chunks) / max(1, di):.1f} chunks/doc)", flush=True)

    print(f"[bge] encoding {len(chunks)} chunks (bge-small-en-v1.5, device={device})...",
          flush=True)
    t0 = time.time()
    emb = _sentence_transformers_embedder()
    if device != "cpu":
        try:
            emb = emb.to(device)
        except Exception:
            pass  # CPU fallback is fine
    raw = emb.encode(chunks, show_progress_bar=False)
    vecs = np.asarray(raw, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms  # L2-normalize for cosine retrieval
    print(f"[bge] {vecs.shape}  ({time.time() - t0:.1f}s)", flush=True)
    return vecs, np.array(doc_ids, dtype=np.int64)


# -------------------------------------------------------------- the simulation
def simulate_channel(
    vecs: np.ndarray,
    doc_ids: np.ndarray,
    decays: list[float],
    lags: list[int],
    n_anchors: int,
    write_gate: float,
    interferes: bool,
    rng: np.random.Generator,
) -> dict:
    """Run the controlled-decay vector channel and measure retrieval vs lag.

    For each anchor position ``i`` (sampled across the stream) and lag ``N``:
    build ``state_{i+N}`` per the recurrence, L2-normalize it as the query,
    retrieve the nearest corpus chunk by cosine, and classify the hit as
    exact (== chunk i), sibling (same doc, != chunk i), or unrelated (other doc).

    ``interferes``: if True, subsequent chunks are written into the channel after
    the anchor (the real stream -- the fade). If False, the slot is read-only after
    the anchor (``state_{i+N} = decay**N * bge(chunk_i)``) -- the CONTROL: pure
    scalar decay does not fade cosine retrieval, so this should stay ~100% exact.

    Returns a nested dict ``results[decay][N] = {exact, sibling, unrelated, n,
    cos_to_anchor}`` averaged over anchors."""
    C = vecs.shape[0]
    max_lag = max(lags)
    # Anchor positions must leave room for max_lag subsequent steps in-stream.
    valid = list(range(0, C - max_lag))
    if not valid:
        raise RuntimeError(
            f"corpus ({C} chunks) too small for max lag {max_lag}; "
            f"raise --n-docs/--chunk-words or lower --lags")
    if len(valid) > n_anchors:
        anchor_idx = sorted(rng.choice(valid, size=n_anchors, replace=False).tolist())
    else:
        anchor_idx = valid
    n_a = len(anchor_idx)

    # Corpus matrix for retrieval (normalized already).
    corpus = vecs  # [C,384]
    # Precompute per-anchor baseline: same-doc sibling count (for the chance floor).
    sib_counts = np.array([int((doc_ids == doc_ids[i]).sum()) - 1 for i in anchor_idx],
                          dtype=np.int64)  # siblings per anchor (excl. itself)

    results: dict[float, dict[int, dict]] = {d: {} for d in decays}
    for decay in decays:
        # Precompute decay**N for each lag.
        dN = {N: float(decay ** N) for N in lags}
        for N in lags:
            exact = 0
            sibling = 0
            unrelated = 0
            cos_sum = 0.0
            for ai, i in enumerate(anchor_idx):
                if interferes:
                    # state_{i+N} = sum_{j=i..i+N} decay**(i+N-j) * g * bge(chunk_j)
                    # = g * sum over a window of length N+1 ending at i+N.
                    weights = np.array(
                        [decay ** (N - (j - i)) for j in range(i, i + N + 1)],
                        dtype=np.float32,
                    )  # [N+1], weights[0]=decay**N (anchor), weights[N]=1 (newest)
                    state = write_gate * (weights[:, None] * corpus[i:i + N + 1]).sum(axis=0)
                else:
                    # read-only slot: only the anchor, scalar-decayed.
                    state = write_gate * dN[N] * corpus[i]
                # Normalize the query in float64. At extreme decay**N (e.g. decay=0.5,
                # N>=96) the control state is ~1e-29 * unit_vec and a float32 norm
                # squares-and-sums the components into underflow (-> 0), leaving q
                # un-normalized so the cos_to_anchor diagnostic reads 0.0 even though
                # argmax is still the anchor (scalar decay is scale-invariant -> the
                # control stays 100% exact regardless). float64 holds the norm; the
                # FADE path (state O(1)) is unaffected by this change.
                state64 = state.astype(np.float64)
                nrm = float(np.linalg.norm(state64))
                if nrm == 0.0:
                    nrm = 1.0
                q = (state64 / nrm).astype(np.float32)  # [384]
                sims = corpus @ q  # [C] cosine (corpus normalized)
                top = int(np.argmax(sims))
                cos_sum += float(sims[i])  # cos(query, anchor)
                if top == i:
                    exact += 1
                elif doc_ids[top] == doc_ids[i]:
                    sibling += 1
                else:
                    unrelated += 1
            n = n_a
            results[decay][N] = {
                "exact": exact / n,
                "sibling": sibling / n,
                "unrelated": unrelated / n,
                "n": n,
                "cos_to_anchor": cos_sum / n,
            }
    # Attach chance floors.
    corpus_size = C
    sib_chance = float(sib_counts.mean() / max(1, corpus_size - 1))
    return {
        "results": results,
        "n_anchors": n_a,
        "corpus_size": corpus_size,
        "exact_chance": 1.0 / corpus_size,
        "sibling_chance": sib_chance,
        "anchor_idx": anchor_idx,
    }


# ----------------------------------------------------------------------- driver
def run(args) -> int:
    rng = np.random.default_rng(args.seed)
    vecs, doc_ids = build_corpus(
        args.erag_path, args.n_docs, args.skip_docs, args.chunk_words,
        args.min_chunks_per_doc, args.device,
    )

    print(f"\n[sim] interferes=TRUE (the real stream -- the fade)", flush=True)
    fade = simulate_channel(vecs, doc_ids, args.decays, args.lags, args.n_anchors,
                            args.write_gate, interferes=True, rng=rng)
    print(f"[sim] interferes=FALSE (control: read-only slot, scalar decay only)",
          flush=True)
    control = simulate_channel(vecs, doc_ids, args.decays, args.lags, args.n_anchors,
                               args.write_gate, interferes=False, rng=rng)

    # ---- print the curves
    def print_curve(title: str, sim: dict) -> None:
        print(f"\n{'='*78}\n{title}\n  corpus={sim['corpus_size']}  "
              f"anchors={sim['n_anchors']}  exact_chance={sim['exact_chance']:.4f}  "
              f"sibling_chance={sim['sibling_chance']:.4f}\n{'='*78}", flush=True)
        for decay in args.decays:
            print(f"\n  decay={decay}", flush=True)
            print("    N  : exact  sibling  unrelated | cos(q,anchor)", flush=True)
            for N in args.lags:
                r = sim["results"][decay][N]
                print(f"    {N:>3}: {r['exact']:.3f}   {r['sibling']:.3f}    "
                      f"{r['unrelated']:.3f}    | {r['cos_to_anchor']:.3f}", flush=True)

    print_curve("FADE (with subsequent writes -- the real stream)", fade)
    print_curve("CONTROL (read-only slot -- scalar decay only)", control)

    # ---- shape verdict per decay: does the fade go exact -> sibling -> unrelated
    # GRACEFULLY (a non-trivial sibling window at intermediate N) or CLIFF (exact
    # -> unrelated with ~no sibling)?
    verdicts: dict[float, dict] = {}
    for decay in args.decays:
        r = fade["results"][decay]
        max_sib = max(r[N]["sibling"] for N in args.lags)
        # exact window: largest N with exact >= 0.5
        exact_window = max((N for N in args.lags if r[N]["exact"] >= 0.5), default=0)
        # sibling window: largest N with sibling >= 0.2 (a real gist regime)
        sib_window = max((N for N in args.lags if r[N]["sibling"] >= 0.2), default=0)
        graceful = max_sib >= 0.2  # a non-trivial gist/sibling regime exists
        verdicts[decay] = {
            "max_sibling": max_sib,
            "exact_window_N": exact_window,
            "sibling_window_N": sib_window,
            "graceful": graceful,
        }
    print("\n[verdict] per-decay fade shape (FADE condition):", flush=True)
    for decay in args.decays:
        v = verdicts[decay]
        print(f"  decay={decay}: max_sibling={v['max_sibling']:.3f}  "
              f"exact_window(>=0.5)={v['exact_window_N']}  "
              f"sibling_window(>=0.2)={v['sibling_window_N']}  "
              f"-> {'GRACEFUL' if v['graceful'] else 'CLIFF'}", flush=True)

    # Control sanity: read-only should stay ~exact (scalar decay doesn't fade
    # cosine). Flag if it doesn't (would mean a bug or non-normalized corpus).
    ctrl_ok = all(
        control["results"][d][0]["exact"] > 0.95 and
        control["results"][d][max(args.lags)]["exact"] > 0.9
        for d in args.decays
    )
    print(f"\n[control] read-only slot stays ~exact for all N: "
          f"{'OK (confirms fade needs interference)' if ctrl_ok else 'UNEXPECTED'}",
          flush=True)

    summary = {
        "probe": "vector_carry",
        "purpose": "vector decay curve (FADE Regime 3 viability -- the vector leg)",
        "erag_path": args.erag_path,
        "n_docs": args.n_docs,
        "chunk_words": args.chunk_words,
        "write_gate": args.write_gate,
        "decays": args.decays,
        "lags": args.lags,
        "corpus_size": fade["corpus_size"],
        "n_anchors": fade["n_anchors"],
        "exact_chance": fade["exact_chance"],
        "sibling_chance": fade["sibling_chance"],
        "fade": fade["results"],
        "control": control["results"],
        "verdicts": verdicts,
        "control_ok": ctrl_ok,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(f"\n[summary] wrote {out_dir / 'run_summary.json'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Vector-carry probe (vector decay curve).")
    ap.add_argument("--erag-path", default=ERAG_PATH)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-docs", type=int, default=48,
                    help="number of ERAG docs to chunk into the corpus/stream")
    ap.add_argument("--skip-docs", type=int, default=500,
                    help="skip the first N docs (stay off the trainer's val split)")
    ap.add_argument("--chunk-words", type=int, default=120,
                    help="words per chunk (a bge-sized passage)")
    ap.add_argument("--min-chunks-per-doc", type=int, default=3,
                    help="drop docs with fewer than this many chunks")
    ap.add_argument("--n-anchors", type=int, default=60,
                    help="anchor positions to sample across the stream")
    ap.add_argument("--write-gate", type=float, default=1.0,
                    help="write gain g (only direction matters; magnitude normalized)")
    ap.add_argument("--decays", type=float, nargs="+", default=DEFAULT_DECAYS,
                    help="decay rates to sweep (near-1 = slow fade)")
    ap.add_argument("--lags", type=int, nargs="+", default=DEFAULT_LAGS,
                    help="lag values N = steps after the anchor's write")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) or cuda for bge encoding")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    return run(args)


if __name__ == "__main__":
    sys.exit(main())