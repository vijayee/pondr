"""SCRATCH probe (never committed) -- GLiNER length-scaling + max_len lever.

Two questions, measured (not cited):
1. Does GLiNER extract cost scale super-linearly with text length (the
   quadratic-attention x sliding-window cliff)? Measure at ~150/400/800 words
   (approx 200/550/1100 tokens -- straddling the 512 boundary) on CPU + CUDA,
   with the stable (GLiNER2) vs open (GLiNER-Decoder) breakdown.
2. How much does the max_len / chunk_size lever buy on the long sample?

Run: python scripts/_probe_gliner_length.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# gliner2 prints a brain emoji to stdout; cp1252 console can't encode it.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

import torch

from src.encoding.gliner_extractor import GLiNERExtractor

# Several DISTINCT multi-sentence blocks (cycled to hit a target word count;
# repetition only pads length for the forward-pass cost, the honest target).
_BLOCKS = [
    "User: I'm switching our team's database from MySQL to Postgres for the "
    "JSONB support and full-text search. We keep hitting row-level locking "
    "issues during the nightly aggregation jobs, and MySQL's JSON handling "
    "has been painful for our semi-structured event data. ",
    "Assistant: Postgres MVCC handles concurrent readers and writers far "
    "better, and JSONB with GIN indexes will help a lot there. For full-text "
    "search you'd want a tsvector column plus a GIN index. One decision up "
    "front: managed RDS, or self-host on a dedicated box? We chose DEBOUNCED "
    "WAL sync last time and it bit us on failover. ",
    "User: Probably managed RDS to start with multi-AZ for automated "
    "failover, synchronous_commit on the primary, replicas async. Let's go "
    "with RDS and revisit if costs balloon. Anything else on the migration "
    "plan, like the cutover strategy or the read-replica rollout? ",
    "Assistant: For the cutover I'd freeze writes, take a final snapshot, "
    "pg_dump the schema and data, restore on RDS, then run a parity check "
    "against a shadow read. The read replicas can be promoted to serve "
    "traffic before we cut the app over. Set synchronous_commit to on for "
    "the primary and leave the replicas async. ",
]


def sample_of(target_words: int) -> str:
    parts, n = [], 0
    i = 0
    while n < target_words:
        b = _BLOCKS[i % len(_BLOCKS)]
        parts.append(b)
        n += len(b.split())
        i += 1
    return "".join(parts).strip()


def time_stages(ext: GLiNERExtractor, text: str, iters: int = 3):
    """Return (stable_best, open_best, total_best) seconds (steady-state min)."""
    # warm-up
    ext.extract(text)
    stable_ts, open_ts, tot_ts = [], [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        ext._extract_stable(text)
        stable_ts.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        ext._extract_open(text)
        open_ts.append(time.perf_counter() - t1)
        t2 = time.perf_counter()
        ext.extract(text)
        tot_ts.append(time.perf_counter() - t2)
    return min(stable_ts), min(open_ts), min(tot_ts)


def report_config(label: str, disc) -> None:
    # Introspect the GLiNER-Decoder's chunking knobs (API varies by gliner
    # version); print whatever exists so the lever is grounded.
    print(f"[cfg {label}] type={type(disc).__module__}.{type(disc).__name__}")
    cfg = getattr(disc, "config", None)
    for attr in ("chunk_size", "max_len", "max_width", "max_position_embeddings",
                 "tokens_per_chunk"):
        val = getattr(cfg, attr, None) if cfg is not None else None
        if val is None:
            val = getattr(disc, attr, None)
        if val is not None:
            print(f"  {attr} = {val}")


def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}\n")

    lengths = [150, 400, 800]
    samples = {n: sample_of(n) for n in lengths}
    for n in lengths:
        s = samples[n]
        print(f"  sample {n}w: {len(s.split())} words / {len(s)} chars")

    print("\n== length-scaling (steady-state best of 3) ==")
    print(f"{'dev':<6}{'words':>7}{'tokens~':>9}{'stable':>9}{'open':>9}{'total':>9}")
    results: dict[str, dict[int, tuple]] = {}
    for dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        ext = GLiNERExtractor(device=dev, timing=False)
        results[dev] = {}
        for n in lengths:
            stable, opn, tot = time_stages(ext, samples[n])
            results[dev][n] = (stable, opn, tot)
            toks = int(len(samples[n].split()) * 1.3)
            print(f"{dev:<6}{n:>7}{toks:>9}{stable:>8.3f}s{opn:>8.3f}s{tot:>8.3f}s")
        if dev == "cuda":
            report_config("discoverer(cuda)", ext.discoverer)
        del ext
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- lever: cap chunk size / max_len on the long (800w) sample ----
    print("\n== max_len / chunk_size lever on 800w sample ==")
    long_text = samples[800]
    for dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        # baseline
        ext = GLiNERExtractor(device=dev, timing=False)
        _, _, base_tot = time_stages(ext, long_text)
        report_config(f"discoverer baseline ({dev})", ext.discoverer)
        # apply lever: shrink chunk_size + cap max_len. Try a few attribute
        # shapes the gliner API has used; report which took.
        applied = {}
        cfg = getattr(ext.discoverer, "config", None)
        for attr, val in (("chunk_size", 128), ("max_len", 512)):
            target = cfg if cfg is not None and hasattr(cfg, attr) else ext.discoverer
            if hasattr(target, attr):
                try:
                    setattr(target, attr, val)
                    applied[attr] = val
                except Exception as e:  # noqa: BLE001
                    applied[attr] = f"set-failed:{e!r}"
        # also try the GLiNER2 extractor's chunk knob if present
        cfg2 = getattr(ext.extractor, "config", None)
        for attr, val in (("chunk_size", 128), ("max_len", 512)):
            target = cfg2 if cfg2 is not None and hasattr(cfg2, attr) else ext.extractor
            if hasattr(target, attr):
                try:
                    setattr(target, attr, val)
                    applied[f"gl2.{attr}"] = val
                except Exception as e:  # noqa: BLE001
                    applied[f"gl2.{attr}"] = f"set-failed:{e!r}"
        _, _, lever_tot = time_stages(ext, long_text)
        print(f"  {dev}: baseline={base_tot:.3f}s -> lever={lever_tot:.3f}s "
              f"({base_tot / max(lever_tot, 1e-9):.2f}x) applied={applied}")
        del ext
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())