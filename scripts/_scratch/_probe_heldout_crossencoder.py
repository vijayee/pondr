"""Real-onyx held-out cross-encoder reranker ship test (Task #120 follow-on).
UNTRACKED scratch probe -- matches the existing untracked ``_probe_*`` siblings
per commit-at-will (never committed).

THE FIX TEST for the cosine+age HOLD ([[pondr-strm-cosage-realonyx-result]]).
DeepSeek consult #6 ([[pondr-strm-cosage-realonyx-result]] -> scripts/_scratch/
ollama_response_cosage_fix.txt) diagnosed the cosine failure as a RANKER +
REPRESENTATION problem (not a threshold problem): frozen bge-cosine is lexically
myopic and cannot do the anaphora/coreference task the salience decision actually
needs ("which prior turn does THIS query refer back to conceptually"). Ranked fix
#1 = a zero-shot cross-encoder reranker + rank-then-budget: bge_cos pre-filters
the 16-turn window to candidates (cos > 0.4), a MS MARCO cross-encoder re-scores
them, take top-3. breadth = 3 by construction; the only question is whether the
CE ranks the gold referent #1 (and inside top-3). No training, no backbone change.

FIDELITY: the slot text is the same USER-turn text the cosine probe used (the
conversation-slot units the ring holds at serve, orchestrator.py:642) and the
gold is the same 17 non-circular hand-authored pairs. The CE scorer
(``cross-encoder/ms-marco-MiniLM-L-6-v2``) is a NEW signal on top -- it is the
proposed fix, not something serve currently uses. The bge pre-filter reuses
``build_embedder("on-demand")`` (byte-identical to serve's cosine).

Run (downloads the public MS MARCO CE on first run; no onyx data leaves the box):
  PYTHONPATH=. python scripts/_scratch/_probe_heldout_crossencoder.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
EP_PATH = SCRATCH / "_heldout_episodes.json"
GOLD_PATH = SCRATCH / "_heldout_gold.json"
OUT_PATH = SCRATCH / "_heldout_crossencoder_result.json"

HELD_IDS = (
    "682afdd9-e8ea-4258-a329-65f67b5d27d5",
    "69e17901-9c6c-4375-a6f1-736e95e1d316",
)
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_COS_FLOOR = 0.4  # DeepSeek's loose pre-filter (discards clear non-matches)


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


def _median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return s[idx]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.subconscious.training.routing_training import build_embedder
    from sentence_transformers import CrossEncoder
    import torch

    episodes = json.loads(EP_PATH.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    cos_phi = float(gold.get("cos_phi", 0.6))
    age_threshold = int(gold.get("age_threshold", 3))
    window = int(gold.get("window", 16))
    budget = 3  # the salience recall budget (Phase 4 BUDGET=3)

    # User turns per session + their bge embeddings (for the cosine pre-filter).
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ce] loading {CE_MODEL} on {device} ...", flush=True)
    ce = CrossEncoder(CE_MODEL, device=device)

    pairs = gold["pairs"]
    per_pair: list[dict] = []
    n_in_window = 0
    n_target_top1 = 0
    n_target_in_top3 = 0
    n_competitor_beats = 0
    n_target_in_candidates = 0
    breadths: list[float] = []
    target_ces: list[float] = []
    # no-filter diagnostic: rank the target by CE over ALL window slots (no cos floor).
    n_target_top1_nofilter = 0

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
        lo = max(0, q_idx - window)
        win_idx = list(range(lo, q_idx))  # exclusive of the query itself
        q_emb = uembs[q_idx]
        q_text = uturns[q_idx]
        win_cos = [(j, _cosine(q_emb, uembs[j])) for j in win_idx]
        target_cos = _cosine(q_emb, uembs[t_idx])

        # CE-score ALL window slots once (cheap; lets us report both the filtered
        # recipe and the no-filter diagnostic from one forward pass).
        ce_pairs = [(q_text, uturns[j]) for j in win_idx]
        ce_scores_all = ce.predict(ce_pairs).tolist()
        ce_by_idx = {j: float(s) for j, s in zip(win_idx, ce_scores_all)}

        # DeepSeek's recipe: pre-filter by cosine, then rank by CE.
        cand_idx = [j for j, c in win_cos if c > CANDIDATE_COS_FLOOR]
        target_in_candidates = t_idx in cand_idx
        if cand_idx:
            cand_ranked = sorted(cand_idx, key=lambda j: ce_by_idx[j], reverse=True)
            target_ce_rank = cand_ranked.index(t_idx) + 1 if target_in_candidates else None
            target_ce = ce_by_idx.get(t_idx, float("nan"))
            cand_ce_scores = [ce_by_idx[j] for j in cand_idx]
            max_competitor_ce = max(ce_by_idx[j] for j in cand_idx if j != t_idx) \
                if len(cand_idx) > 1 else float("-inf")
            breadth = min(budget, len(cand_idx))
        else:
            target_ce_rank = None
            target_ce = float("nan")
            max_competitor_ce = float("-inf")
            breadth = 0

        target_top1 = target_ce_rank == 1
        target_in_top3 = target_ce_rank is not None and target_ce_rank <= budget
        competitor_beats = (max_competitor_ce >= target_ce
                            if target_in_candidates and not math.isnan(target_ce)
                            else True)

        # No-filter diagnostic: target rank by CE over ALL window slots.
        all_ranked = sorted(win_idx, key=lambda j: ce_by_idx[j], reverse=True)
        target_rank_all = all_ranked.index(t_idx) + 1 if t_idx in all_ranked else None
        target_top1_nofilter = target_rank_all == 1

        in_window = age >= age_threshold and age <= window - 1 and t_idx >= lo
        if in_window:
            n_in_window += 1
            if target_in_candidates:
                n_target_in_candidates += 1
            if target_top1:
                n_target_top1 += 1
            if target_in_top3:
                n_target_in_top3 += 1
            if competitor_beats:
                n_competitor_beats += 1
            if target_top1_nofilter:
                n_target_top1_nofilter += 1
            breadths.append(float(breadth))
        if not math.isnan(target_ce):
            target_ces.append(target_ce)

        per_pair.append({
            "session_id": sid, "query": q_idx, "target": t_idx, "age": age,
            "in_window": in_window,
            "target_cos": round(target_cos, 4),
            "target_in_candidates": target_in_candidates,
            "n_candidates": len(cand_idx),
            "target_ce": round(target_ce, 4) if not math.isnan(target_ce) else None,
            "max_competitor_ce": (round(max_competitor_ce, 4)
                                  if max_competitor_ce != float("-inf") else None),
            "target_ce_rank": target_ce_rank,
            "target_rank_all": target_rank_all,
            "target_top1": target_top1, "target_in_top3": target_in_top3,
            "competitor_beats": competitor_beats, "breadth": breadth,
            "query_preview": uturns[q_idx][:90],
            "target_preview": uturns[t_idx][:90],
            "note": p.get("note", ""),
        })

    n = n_in_window
    top1_rate = n_target_top1 / n if n else 0.0
    in_top3_rate = n_target_in_top3 / n if n else 0.0
    comp_rate = n_competitor_beats / n if n else 0.0
    in_cand_rate = n_target_in_candidates / n if n else 0.0
    nofilter_top1_rate = n_target_top1_nofilter / n if n else 0.0
    med_breadth = _median(breadths) if breadths else float("nan")
    med_target_ce = _median(target_ces) if target_ces else float("nan")

    ship = (n >= 6 and top1_rate >= 2.0 / 3.0 and in_top3_rate >= 2.0 / 3.0
            and comp_rate <= 1.0 / 3.0 and med_breadth <= budget)
    verdict = ("SHIP -- cross-encoder reranker clears the real-onyx gate"
               if ship else
               "HOLD -- zero-shot CE does not clear the gate -> fine-tune CE on gold, "
               "or Step 4 pure-bilinear 2a retrain")

    agg = {
        "n_pairs_total": len(pairs), "n_in_window": n,
        "ce_model": CE_MODEL, "candidate_cos_floor": CANDIDATE_COS_FLOOR,
        "cos_phi": cos_phi, "age_threshold": age_threshold, "window": window,
        "budget": budget,
        "target_in_candidates_rate": round(in_cand_rate, 4),
        "target_top1_rate": round(top1_rate, 4),
        "target_in_top3_rate": round(in_top3_rate, 4),
        "competitor_beats_rate": round(comp_rate, 4),
        "median_breadth": round(med_breadth, 4) if not math.isnan(med_breadth) else None,
        "median_target_ce": (round(med_target_ce, 4)
                             if not math.isnan(med_target_ce) else None),
        "nofilter_target_top1_rate": round(nofilter_top1_rate, 4),
        "ship": ship, "verdict": verdict,
        "gate": "target_top1>=2/3 AND target_in_top3>=2/3 AND competitor_beats<=1/3 AND median_breadth<=budget AND n>=6",
    }
    report = {"aggregate": agg, "per_pair": per_pair}
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print("\n=== HELD-OUT CROSS-ENCODER RERANKER (per pair) ===", flush=True)
    print(f"{'sess':>4} {'q':>3} {'t':>3} {'age':>3} {'tgt_cos':>7} {'ncand':>5} "
          f"{'tgt_ce':>7} {'rank':>4} {'top3':>4} {'beat':>4} {'brd':>3}",
          flush=True)
    for r in per_pair:
        if "error" in r:
            print(f"  [error] {r['error']}", flush=True)
            continue
        print(f"{r['session_id'][:4]:>4} {r['query']:>3} {r['target']:>3} "
              f"{r['age']:>3} {r['target_cos']:>7.3f} {r['n_candidates']:>5} "
              f"{(r['target_ce'] if r['target_ce'] is not None else 0):>7.3f} "
              f"{str(r['target_ce_rank']):>4} {str(r['target_in_top3']):>4} "
              f"{str(r['competitor_beats']):>4} {r['breadth']:>3}", flush=True)
    print(f"\n=== AGGREGATE (n_in_window={n}/{len(pairs)}) ===", flush=True)
    print(f"  target_in_candidates_rate = {in_cand_rate:.3f}  (target survives cos>{CANDIDATE_COS_FLOOR} pre-filter)", flush=True)
    print(f"  target_top1_rate          = {top1_rate:.3f}  (need >= 0.667)", flush=True)
    print(f"  target_in_top3_rate       = {in_top3_rate:.3f}  (need >= 0.667, the recall metric)", flush=True)
    print(f"  competitor_beats_rate     = {comp_rate:.3f}  (need <= 0.333)", flush=True)
    print(f"  median_breadth            = {med_breadth}   (need <= {budget}; =3 by construction)", flush=True)
    print(f"  median_target_ce          = {med_target_ce}", flush=True)
    print(f"  nofilter_target_top1_rate = {nofilter_top1_rate:.3f}  (CE rank over ALL window slots, no cos floor)", flush=True)
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"wrote {OUT_PATH}", flush=True)
    return 0 if ship else 2


if __name__ == "__main__":
    raise SystemExit(main())