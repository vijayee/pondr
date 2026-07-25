"""Real-onyx held-out cosine discrimination ship test (Task #120). UNTRACKED
scratch probe -- matches the existing untracked ``_probe_*`` / ``_run_1f*``
siblings per commit-at-will (never committed).

THE SHIP DECISION for the cosine+age STRM salience mode (commit d537f5c). The
synthetic gate passed (coverage 3/3, accuracy 2/3), but synthetic fillers are
topically-DISTINCT (easy cosine competitors). The DeepSeek #5 falsification
predicts that at REAL serve the ring is full of topically-CLOSE recent
conversation turns, so ``bge_cos(query, recent_turn)`` exceeds ``cos_phi`` for
many slots -> the trigger fires indiscriminately. This test falsifies (or
confirms) that on UNSEEN held-out onyx sessions with hand-authored gold.

NON-CIRCULAR: a human picked the target turn each query refers back to (NOT
cosine). The test asks whether ``bge_cos(query, target_user_text)`` beats the
OTHER recent user turns in the window (the topically-close competitors) and
clears ``cos_phi`` with bounded breadth.

FIDELITY: at serve the conversation ring slots carry the USER text
(``orchestrator.py:642``), and the cosine+age trigger scores
``bge_cos(query_user_text, slot_text)`` with the same frozen bge-small-en-v1.5
embedder that ``build_embedder("on-demand")`` returns
(``routing_training.py:328``; ``working_memory.embed`` is a passthrough to
``embedder.encode`` with NO extra normalization, ``working_memory.py:475``).
So this script reproduces the trigger's signal exactly with no orchestrator,
no heads, no retriever, no LLM -- zero confounds on the open question.

Run (no servers needed):
  PYTHONPATH=. python scripts/_probe_heldout_cosine_gap.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scripts/_scratch"
EP_PATH = SCRATCH / "_heldout_episodes.json"
GOLD_PATH = SCRATCH / "_heldout_gold.json"
OUT_PATH = SCRATCH / "_heldout_cosine_result.json"

HELD_IDS = (
    "682afdd9-e8ea-4258-a329-65f67b5d27d5",
    "69e17901-9c6c-4375-a6f1-736e95e1d316",
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return s[idx]


def _median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.subconscious.training.routing_training import build_embedder

    episodes = json.loads(EP_PATH.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    cos_phi = float(gold.get("cos_phi", 0.6))
    age_threshold = int(gold.get("age_threshold", 3))
    window = int(gold.get("window", 16))
    budget = 3  # the salience recall budget (Phase 4 BUDGET=3)

    # User turns per session (the conversation-slot units) + cache their bge
    # embeddings. Only the held-out sessions are loaded.
    user_texts: dict[str, list[str]] = {}
    user_embs: dict[str, list[list[float]]] = {}
    embedder = build_embedder("on-demand")
    for sid in HELD_IDS:
        sess = episodes.get(sid)
        if sess is None:
            print(f"[warn] session {sid} missing from {EP_PATH}", file=sys.stderr)
            continue
        uturns = [t["text"] for t in sess["turns"] if t.get("role") == "user"]
        user_texts[sid] = uturns
        user_embs[sid] = embedder.encode(uturns)
        print(f"[load] {sid} {sess['name'][:40]!r:42} -> {len(uturns)} user turns",
              flush=True)

    pairs = gold["pairs"]
    per_pair: list[dict] = []
    n_in_window = 0
    n_target_top1 = 0
    n_fires_target = 0
    n_competitor_beats = 0
    breadths: list[float] = []
    gaps: list[float] = []
    target_coss: list[float] = []

    for p in pairs:
        sid = p["session_id"]
        q_idx = int(p["query"])
        t_idx = int(p["target"])
        uturns = user_texts.get(sid)
        uembs = user_embs.get(sid)
        if uturns is None or q_idx >= len(uturns) or t_idx >= len(uturns):
            per_pair.append({**p, "error": "index-out-of-range"})
            print(f"[skip] {sid[:8]} q{q_idx} t{t_idx} out of range", file=sys.stderr)
            continue
        age = q_idx - t_idx - 1
        # Window = prior conversation slots (user turns) in the ring at query time.
        lo = max(0, q_idx - window)
        win_idx = list(range(lo, q_idx))  # exclusive of the query itself
        q_emb = uembs[q_idx]
        win_cos = [(j, _cosine(q_emb, uembs[j])) for j in win_idx]
        target_cos = _cosine(q_emb, uembs[t_idx])
        # Rank of the target's cos among the window slots (1 = highest).
        ranked = sorted(win_cos, key=lambda kv: kv[1], reverse=True)
        target_rank = next((r + 1 for r, (j, _) in enumerate(ranked) if j == t_idx),
                           None)
        competitor_cos = [c for j, c in win_cos if j != t_idx]
        max_competitor = max(competitor_cos) if competitor_cos else float("-inf")
        gap = target_cos - max_competitor
        breadth = sum(1 for _, c in win_cos if c > cos_phi)
        fires_on_target = (target_cos > cos_phi) and (age >= age_threshold)
        competitor_beats = any(c >= target_cos for c in competitor_cos)

        in_window = age >= age_threshold and age <= window - 1 and t_idx >= lo
        if in_window:
            n_in_window += 1
            if target_rank == 1:
                n_target_top1 += 1
            if fires_on_target:
                n_fires_target += 1
            if competitor_beats:
                n_competitor_beats += 1
            breadths.append(float(breadth))
            gaps.append(gap)
        target_coss.append(target_cos)

        per_pair.append({
            "session_id": sid, "query": q_idx, "target": t_idx, "age": age,
            "in_window": in_window, "target_cos": round(target_cos, 4),
            "max_competitor_cos": round(max_competitor, 4),
            "gap": round(gap, 4), "target_rank": target_rank,
            "breadth": breadth, "fires_on_target": fires_on_target,
            "competitor_beats": competitor_beats,
            "query_preview": uturns[q_idx][:90],
            "target_preview": uturns[t_idx][:90],
            "note": p.get("note", ""),
        })

    n = n_in_window
    top1_rate = n_target_top1 / n if n else 0.0
    fires_rate = n_fires_target / n if n else 0.0
    comp_rate = n_competitor_beats / n if n else 0.0
    med_breadth = _median(breadths) if breadths else float("nan")
    p90_breadth = _pct(breadths, 0.9) if breadths else float("nan")
    med_gap = _median(gaps) if gaps else float("nan")
    p10_gap = _pct(gaps, 0.1) if gaps else float("nan")

    med_target_cos = _median(target_coss) if target_coss else float("nan")

    ship = (n >= 6 and top1_rate >= 2.0 / 3.0 and fires_rate >= 2.0 / 3.0
            and comp_rate <= 1.0 / 3.0 and med_breadth <= budget)
    verdict = "SHIP STRM salience default-on" if ship else "HOLD -- stay opt-in (cosine under-discriminates at real serve -> Step 4 pure-bilinear 2a retrain lever)"

    agg = {
        "n_pairs_total": len(pairs), "n_in_window": n,
        "cos_phi": cos_phi, "age_threshold": age_threshold, "window": window,
        "budget": budget,
        "target_top1_rate": round(top1_rate, 4),
        "fires_on_target_rate": round(fires_rate, 4),
        "competitor_beats_rate": round(comp_rate, 4),
        "median_breadth": round(med_breadth, 4) if not math.isnan(med_breadth) else None,
        "p90_breadth": round(p90_breadth, 4) if not math.isnan(p90_breadth) else None,
        "median_gap": round(med_gap, 4) if not math.isnan(med_gap) else None,
        "p10_gap": round(p10_gap, 4) if not math.isnan(p10_gap) else None,
        "median_target_cos": (round(med_target_cos, 4)
                              if not math.isnan(med_target_cos) else None),
        "ship": ship, "verdict": verdict,
        "gate": "target_top1>=2/3 AND fires_on_target>=2/3 AND competitor_beats<=1/3 AND median_breadth<=budget AND n>=6",
    }
    report = {"aggregate": agg, "per_pair": per_pair}
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print("\n=== HELD-OUT COSINE DISCRIMINATION (per pair) ===", flush=True)
    print(f"{'sess':>4} {'q':>3} {'t':>3} {'age':>3} {'tgt_cos':>7} "
          f"{'maxcmp':>7} {'gap':>7} {'rank':>4} {'brd':>3} {'fire':>4} {'beat':>4}",
          flush=True)
    for r in per_pair:
        if "error" in r:
            print(f"  [error] {r['error']}", flush=True)
            continue
        print(f"{r['session_id'][:4]:>4} {r['query']:>3} {r['target']:>3} "
              f"{r['age']:>3} {r['target_cos']:>7.3f} {r['max_competitor_cos']:>7.3f} "
              f"{r['gap']:>+7.3f} {str(r['target_rank']):>4} {r['breadth']:>3} "
              f"{str(r['fires_on_target']):>4} {str(r['competitor_beats']):>4}",
              flush=True)
    print(f"\n=== AGGREGATE (n_in_window={n}/{len(pairs)}) ===", flush=True)
    print(f"  target_top1_rate     = {top1_rate:.3f}  (need >= 0.667)", flush=True)
    print(f"  fires_on_target_rate = {fires_rate:.3f}  (need >= 0.667)", flush=True)
    print(f"  competitor_beats_rate = {comp_rate:.3f}  (need <= 0.333)", flush=True)
    print(f"  median_breadth       = {med_breadth}   (need <= {budget})", flush=True)
    print(f"  p90_breadth          = {p90_breadth}", flush=True)
    print(f"  median_gap           = {med_gap}", flush=True)
    print(f"  p10_gap              = {p10_gap}", flush=True)
    print(f"  median_target_cos    = {med_target_cos}  (target IS relevant; the problem is breadth/competitors)", flush=True)
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"wrote {OUT_PATH}", flush=True)
    return 0 if ship else 2


if __name__ == "__main__":
    raise SystemExit(main())