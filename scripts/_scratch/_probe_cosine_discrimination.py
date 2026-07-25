"""Does bge-cosine discriminate the RIGHT fact from fillers (the targeting test)?

For each fact in SHIP_FACT_BANK: cosine(query, fact_summary) vs
cosine(query, each filler). If the fact's cosine is reliably the max (or clearly
above fillers), a cosine-based salience trigger would target the right slot.
If fillers tie/beat the fact, cosine targeting fails and we need the learned
bilinear. UNTRACKED scratch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.subconscious.training.routing_training import build_embedder
from scripts.eval_strm_context_coverage import FACT_BANK, FILLER_BANK
from scripts.eval_strm_ship_decision import SHIP_FACT_BANK

import numpy as np


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    emb = build_embedder("on-demand")
    # Use the eval's gold bank (fact_summary, question, gold_answer).
    facts = SHIP_FACT_BANK[:6]
    fact_summaries = [s for s, _, _ in facts]
    questions = [q for _, q, _ in facts]
    # Embed once.
    sum_embs = emb.encode(fact_summaries)
    q_embs = emb.encode(questions)
    fill_embs = emb.encode(list(FILLER_BANK))

    print("Per-fact: cosine(query, own_fact) vs cosine(query, best_filler) "
          "vs mean filler. Targeting works if own >> best_filler.\n")
    n_right_max = 0
    gaps = []
    for i, (s, q, gold) in enumerate(facts):
        c_own = cos(q_embs[i], sum_embs[i])
        c_fillers = [cos(q_embs[i], f) for f in fill_embs]
        c_best_filler = max(c_fillers)
        c_mean_filler = float(np.mean(c_fillers))
        gap = c_own - c_best_filler
        gaps.append(gap)
        right_max = c_own >= c_best_filler
        n_right_max += int(right_max)
        print(f"fact_{i:02d} q={q[:42]!r}")
        print(f"   own={c_own:.4f}  best_filler={c_best_filler:.4f}  "
              f"mean_filler={c_mean_filler:.4f}  gap(own-best)={gap:+.4f}  "
              f"own_is_max={right_max}")
    print(f"\n{n_right_max}/{len(facts)} facts: own cosine is the max over all "
          f"fillers.")
    print(f"gap(own - best_filler): median={float(np.median(gaps)):+.4f} "
          f"min={min(gaps):+.4f} max={max(gaps):+.4f}")
    print("\nINTERPRETATION: median gap >> 0 and own-is-max for most facts => "
          "cosine targeting is sound (no learned head needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())