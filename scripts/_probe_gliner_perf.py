"""SCRATCH probe (never committed) -- measure GLiNER CPU extraction cost.

Answers "why do the GLiNER models run poorly on CPU?" with measured numbers on
THIS box, plus the CUDA-availability reality. Run, record the numbers, discard.

    python scripts/_probe_gliner_perf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# gliner2.from_pretrained prints a brain emoji to stdout; the Windows cp1252
# console can't encode it. Reconfigure to utf-8 before any GLiNER construction.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

import torch

from src.encoding.gliner_extractor import GLiNERExtractor

# A representative multi-turn conversation (~600 tokens). The bottleneck scales
# with text length (GLiNER sliding-window-chunks long text -> multiple forward
# passes), so a real-length conv is the honest measurement, not a one-liner.
SAMPLE = (
    "User: Hey, I'm thinking about switching our team's database from MySQL to Postgres. "
    "What do you think?\n"
    "Assistant: That's a solid move for most workloads. Postgres gives you JSONB, better "
    "concurrency via MVCC, and richer indexing. What's driving the switch?\n"
    "User: Mostly the JSONB support and full-text search. We're storing a lot of semi-"
    "structured event data and MySQL's JSON handling has been painful. Also we keep hitting "
    "row-level locking issues during our nightly aggregation jobs.\n"
    "Assistant: JSONB with GIN indexes will help a lot there, and Postgres MVCC handles "
    "concurrent readers and writers far better. For full-text search you'd want a tsvector "
    "column plus a GIN index. One decision to make up front: do you want to run Postgres "
    "on managed RDS, or self-host on a dedicated box?\n"
    "User: Probably managed RDS to start. We chose DEBOUNCED WAL sync last time and it bit "
    "us on failover, so I'd rather not own the ops surface again. Let's go with RDS and "
    "revisit if costs balloon.\n"
    "Assistant: Agreed. Managed RDS with multi-AZ gives you automated failover without the "
    "WAL tuning headaches. I'd set synchronous_commit to on for the primary and leave the "
    "replicas async. Anything else on the migration plan?"
)


def _time_extract(ext: GLiNERExtractor, iters: int) -> list[float]:
    # Warm-up (first call pays import/kernel-init cost).
    ext.extract(SAMPLE)
    totals = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ext.extract(SAMPLE)
        totals.append(time.perf_counter() - t0)
    return totals


def main() -> int:
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"cuda VRAM: free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB")
    else:
        print("cuda VRAM: N/A (CPU-only torch build -- GLiNER cannot reach the GPU "
              "from Python even though the 5080 has CUDA; Bonsai uses its own CUDA "
              "binary, not PyTorch)")

    nchars = len(SAMPLE)
    nwords = len(SAMPLE.split())
    print(f"sample: {nchars} chars / ~{nwords} words\n")

    # CPU path (the actual runtime path on this CPU-torch box).
    print("== CPU ==")
    cpu = GLiNERExtractor(device="cpu", timing=True)
    totals = _time_extract(cpu, iters=3)
    print(f"CPU extract() per call: {[round(t, 3) for t in totals]}s "
          f"(mean={sum(totals)/len(totals):.3f}s)\n")

    # Attempt the "auto"/CUDA path. With a CUDA torch build this resolves to
    # cuda; with the CPU build it resolves to CPU. Time the extract so we get a
    # real CPU-vs-CUDA comparison either way.
    print('== device="auto" ==')
    auto = GLiNERExtractor(device="auto", timing=True)
    print(f"resolved device_d={auto._device_d} device_e={auto._device_e}")
    auto_totals = _time_extract(auto, iters=3)
    print(f'{auto._device_d}/{auto._device_e} extract() per call: '
          f'{[round(t, 3) for t in auto_totals]}s '
          f'(mean={sum(auto_totals)/len(auto_totals):.3f}s)')
    if torch.cuda.is_available():
        free2, _ = torch.cuda.mem_get_info()
        print(f"cuda VRAM after GLiNER load: free={free2 / 1e9:.2f}GB")

    # Footprint.
    try:
        import os
        import psutil  # type: ignore
        p = psutil.Process()
        print(f"RSS after load: {p.memory_info().rss / 1e9:.2f}GB")
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())