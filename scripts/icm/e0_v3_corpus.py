"""E0-v3 corpus -- SEMANTIC values: content correlates with value by law.

Per the value-kind taxonomy: the twin corpus was all-SYMBOLIC (arbitrary
identifiers; dense cannot compete there by definition, and the lens smoke's
negative was correct generalization on noise). This corpus is the well-posed
dense-lens test: each session ingests a UNIQUE row count, and the world's
values follow laws of that count with small noise:

    ingest_rows (unique, 200..3200)  -- the content anchor
    probe_latency = 8 + rows/40 + noise(+-2)     (semantic: correlates)
    disk_percent  = 55 + rows//80 + noise(+-1)   (semantic: correlates)
    prune_images  = rows//16 + noise(+-3)        (semantic: correlates)
    backup_ts     = random                       (SYMBOLIC control: does NOT
                                                  correlate with anything)

The lens question splits by taxon:
    semantic arm:  predict the value of a correlated fact from the question's
                   anchor (rows) -- a trained head SHOULD be able to earn its
                   cost (predict within band).
    symbolic arm:  predict the same-size-distribution fact that has NO law
                   (backup_ts) -- a trained head SHOULD fail (chance).
Baselines: global median (majority), the true law's value, chance band rate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

N_SESSIONS = 300


@dataclass
class Session:
    si: int
    rows: int
    lat: int
    pct: int
    pruned: int
    ts: str  # symbolic


def build(seed: int = 7, n: int = N_SESSIONS) -> list[Session]:
    rng = random.Random(seed)
    rows_pool = rng.sample(range(200, 3201), n)  # unique counts
    ts_slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60)]
    ts_shuffled = rng.sample(ts_slots, n)
    out = []
    for si, rows in enumerate(rows_pool):
        lat = int(round(8 + rows / 40 + rng.uniform(-2, 2)))
        pct = int(round(55 + rows / 80 + rng.uniform(-1, 1)))
        pruned = int(round(rows / 16 + rng.uniform(-3, 3)))
        out.append(Session(si, rows, max(1, lat), max(50, min(100, pct)), max(0, pruned), ts_shuffled[si]))
    return out


def episode_text(s: Session) -> str:
    return (f"[run {s.si:04d}] Ingested a dataset of {s.rows} rows. "
            f"Checkpoint [deploy]: probe returned {s.lat}ms. "
            f"Checkpoint [storage]: disk at {s.pct} percent. "
            f"Checkpoint [ci]: pruned {s.pruned} stale images. "
            f"Checkpoint [backup]: completed at {s.ts}.")


def question_semantic(s: Session) -> str:
    return (f"The run that ingested {s.rows} rows -- what latency did its "
            f"dependency probe return?")


def question_symbolic(s: Session) -> str:
    return (f"The run that ingested {s.rows} rows -- at what time did its "
            f"backup complete?")


def law_value(s: Session) -> int:
    return int(round(8 + s.rows / 40))


def main() -> int:
    import json
    import io

    data = build()
    rows_j = [s.rows for s in data]
    lat_j = [s.lat for s in data]
    # correlation sanity: Pearson via numpy-free formula on the sampled data
    n = len(data)
    mr = sum(rows_j) / n
    ml = sum(lat_j) / n
    cov = sum((r - mr) * (l - ml) for r, l in zip(rows_j, lat_j))
    vr = sum((r - mr) ** 2 for r in rows_j) ** 0.5
    vl = sum((l - ml) ** 2 for l in lat_j) ** 0.5
    corr = cov / (vr * vl)
    out = {"n": n, "rows_lat_pearson": round(corr, 4),
           "sample": {"rows": data[0].rows, "lat": data[0].lat, "pct": data[0].pct,
                      "pruned": data[0].pruned, "ts": data[0].ts},
           "text_sample": episode_text(data[0])}
    path = "data/_scratch/icm/e0_v3_corpus_check.json"
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    main()