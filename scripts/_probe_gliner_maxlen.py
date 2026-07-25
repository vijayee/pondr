"""SCRATCH probe (never committed) -- max_len cost/quality frontier.

The prior probe found GLiNER's max_len is ALREADY 512 by default (sentences
are truncated to 512). So the only remaining max_len knob is going BELOW 512,
which trades extraction quality (entities in the truncated tail are lost) for
speed. Map that frontier on the 800w sample: cost + entity counts at
max_len in {512, 256, 128}, CPU + CUDA.

Run: python scripts/_probe_gliner_maxlen.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

import torch

from src.encoding.gliner_extractor import GLiNERExtractor

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
    parts, n, i = [], 0, 0
    while n < target_words:
        b = _BLOCKS[i % len(_BLOCKS)]
        parts.append(b)
        n += len(b.split())
        i += 1
    return "".join(parts).strip()


def main() -> int:
    text = sample_of(800)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"sample: {len(text.split())} words / {len(text)} chars\n")

    print(f"{'dev':<6}{'max_len':>8}{'stable':>9}{'open':>9}{'total':>9}"
          f"{'n_stable':>9}{'n_open':>7}")
    for dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        ext = GLiNERExtractor(device=dev, timing=False)
        cfg = ext.discoverer.config
        cfg2 = ext.extractor.config
        for ml in (512, 256, 128):
            # set on BOTH models (max_len caps per-forward token cost). Already
            # 512 at baseline; lower values truncate the tail -> cheaper but
            # fewer entities surfaced.
            cfg.max_len = ml
            cfg2.max_len = ml if hasattr(cfg2, "max_len") else cfg2
            try:
                cfg2.max_len = ml
            except Exception:  # noqa: BLE001
                pass
            ext.extract(text)  # warm
            ts_st, ts_op, ts_tot = [], [], []
            n_stable = n_open = 0
            for _ in range(3):
                t0 = time.perf_counter(); st = ext._extract_stable(text); ts_st.append(time.perf_counter() - t0)
                t1 = time.perf_counter(); op = ext._extract_open(text); ts_op.append(time.perf_counter() - t1)
                t2 = time.perf_counter(); ext.extract(text); ts_tot.append(time.perf_counter() - t2)
                n_stable = len(st.get("entities", [])) + len(st.get("topics", [])) + len(st.get("decisions", []))
                n_open = len(op)
            print(f"{dev:<6}{ml:>8}{min(ts_st):>8.3f}s{min(ts_op):>8.3f}s"
                  f"{min(ts_tot):>8.3f}s{n_stable:>9}{n_open:>7}")
        del ext
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())