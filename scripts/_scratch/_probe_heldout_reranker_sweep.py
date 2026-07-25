"""Zero-shot reranker SWEEP over the 17 held-out gold pairs (Task #121). UNTRACKED
scratch probe -- matches the existing untracked ``_probe_*`` siblings per
commit-at-will (never committed).

Cheap insurance before paying the labeling effort for the CE fine-tune
([[pondr-strm-crossencoder-zeroshot-result]]). DeepSeek consult #7 predicted the
zero-shot alternatives would fail (~0.35) but bge-reranker-v2-m3 is a much
stronger reranker than MS MARCO, so it is worth a ~10-min sweep. The sweep also
PICKS THE BEST BASE MODEL for the fine-tune (Task #123).

Runs each model over the same 17 non-circular gold pairs + 16-turn window with
rank-then-budget (breadth=3 by construction). Reuses ``build_embedder("on-demand")``
for the cos>0.4 pre-filter. Public model downloads only; no onyx data leaves the
box.

Run (downloads public rerankers on first run):
  PYTHONPATH=. python scripts/_scratch/_probe_heldout_reranker_sweep.py
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
OUT_PATH = SCRATCH / "_heldout_reranker_sweep_result.json"

HELD_IDS = (
    "682afdd9-e8ea-4258-a329-65f67b5d27d5",
    "69e17901-9c6c-4375-a6f1-736e95e1d316",
)
CANDIDATE_COS_FLOOR = 0.4
BUDGET = 3

# (model_id, is_nli_3class). ms-marco is the baseline (already 0.294 in the
# single-model probe); re-run here for a clean side-by-side.
MODELS = [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", False),
    ("BAAI/bge-reranker-v2-m3", False),
    ("cross-encoder/nli-deberta-v3-base", True),
]


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


def _score_model(ce, is_nli: bool, pairs_texts: list[tuple[str, str]]) -> dict[int, float]:
    """Return {window_index -> ce_score}. For NLI 3-class models, take the
    entailment logit (column 0) of (query, turn) as the relevance score."""
    scores = ce.predict(pairs_texts)
    arr = scores
    try:
        ndim = arr.ndim
    except AttributeError:
        arr = list(arr)
        ndim = 1 if not isinstance(arr[0], (list, tuple)) else 2
    if is_nli and ndim == 2:
        # column 0 = entailment of (query=premise, turn=hypothesis)
        arr = [float(row[0]) for row in arr]
    else:
        arr = [float(row) if ndim == 1 else float(row[0]) for row in arr]
    return arr


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.subconscious.training.routing_training import build_embedder
    from sentence_transformers import CrossEncoder
    import torch

    episodes = json.loads(EP_PATH.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    window = int(gold.get("window", 16))
    age_threshold = int(gold.get("age_threshold", 3))

    user_texts: dict[str, list[str]] = {}
    user_embs: dict[str, list[list[float]]] = {}
    embedder = build_embedder("on-demand")
    for sid in HELD_IDS:
        sess = episodes.get(sid)
        if sess is None:
            continue
        uturns = [t["text"] for t in sess["turns"] if t.get("role") == "user"]
        user_texts[sid] = uturns
        user_embs[sid] = embedder.encode(uturns)
        print(f"[load] {sid} {sess['name'][:40]!r:42} -> {len(uturns)} user turns",
              flush=True)

    # Precompute the per-pair windows + cosine once (model-independent).
    pair_windows: list[dict] = []
    for p in gold["pairs"]:
        sid = p["session_id"]
        q_idx = int(p["query"])
        t_idx = int(p["target"])
        uturns = user_texts.get(sid)
        uembs = user_embs.get(sid)
        if uturns is None or q_idx >= len(uturns) or t_idx >= len(uturns):
            pair_windows.append({"error": "index-out-of-range"})
            continue
        age = q_idx - t_idx - 1
        lo = max(0, q_idx - window)
        win_idx = list(range(lo, q_idx))
        q_emb = uembs[q_idx]
        win_cos = [(j, _cosine(q_emb, uembs[j])) for j in win_idx]
        target_cos = _cosine(q_emb, uembs[t_idx])
        in_window = (age >= age_threshold and age <= window - 1 and t_idx >= lo)
        pair_windows.append({
            "sid": sid, "q_idx": q_idx, "t_idx": t_idx, "age": age,
            "in_window": in_window, "win_idx": win_idx, "win_cos": win_cos,
            "target_cos": target_cos, "q_text": uturns[q_idx],
            "uturns": uturns,
        })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_model: list[dict] = []
    n_in_window = sum(1 for pw in pair_windows if pw.get("in_window"))

    for model_id, is_nli in MODELS:
        print(f"\n[sweep] {model_id} (nli={is_nli}) on {device} ...", flush=True)
        ce = CrossEncoder(model_id, device=device)
        n_top1 = n_top3 = n_beats = 0
        breadths: list[float] = []
        for pw in pair_windows:
            if "error" in pw:
                continue
            win_idx = pw["win_idx"]
            win_cos = pw["win_cos"]
            t_idx = pw["t_idx"]
            ce_pairs = [(pw["q_text"], pw["uturns"][j]) for j in win_idx]
            ce_scores = _score_model(ce, is_nli, ce_pairs)
            ce_by_idx = {j: s for j, s in zip(win_idx, ce_scores)}
            cand_idx = [j for j, c in win_cos if c > CANDIDATE_COS_FLOOR]
            target_in_cand = t_idx in cand_idx
            if cand_idx:
                cand_ranked = sorted(cand_idx, key=lambda j: ce_by_idx[j], reverse=True)
                rank = cand_ranked.index(t_idx) + 1 if target_in_cand else None
                target_ce = ce_by_idx.get(t_idx, float("nan"))
                max_comp = max((ce_by_idx[j] for j in cand_idx if j != t_idx),
                               default=float("-inf"))
            else:
                rank = None
                target_ce = float("nan")
                max_comp = float("-inf")
            top1 = rank == 1
            in_top3 = rank is not None and rank <= BUDGET
            beats = (max_comp >= target_ce if target_in_cand
                     and not math.isnan(target_ce) else True)
            breadth = min(BUDGET, len(cand_idx))
            if pw["in_window"]:
                if top1:
                    n_top1 += 1
                if in_top3:
                    n_top3 += 1
                if beats:
                    n_beats += 1
                breadths.append(float(breadth))
        n = n_in_window
        top1_rate = n_top1 / n if n else 0.0
        top3_rate = n_top3 / n if n else 0.0
        beats_rate = n_beats / n if n else 0.0
        med_breadth = _median(breadths) if breadths else float("nan")
        ship = (n >= 6 and top1_rate >= 2.0 / 3.0 and top3_rate >= 2.0 / 3.0
                and beats_rate <= 1.0 / 3.0 and med_breadth <= BUDGET)
        row = {
            "model": model_id, "is_nli": is_nli, "n_in_window": n,
            "target_top1_rate": round(top1_rate, 4),
            "target_in_top3_rate": round(top3_rate, 4),
            "competitor_beats_rate": round(beats_rate, 4),
            "median_breadth": round(med_breadth, 4) if not math.isnan(med_breadth) else None,
            "ship": ship,
        }
        per_model.append(row)
        print(f"  top1={top1_rate:.3f}  in_top3={top3_rate:.3f}  "
              f"beats={beats_rate:.3f}  breadth={med_breadth}  "
              f"{'SHIP' if ship else 'HOLD'}", flush=True)
        del ce

    out = {"n_in_window": n_in_window, "budget": BUDGET,
           "candidate_cos_floor": CANDIDATE_COS_FLOOR, "models": per_model}
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print("\n=== RERANKER SWEEP SUMMARY (n_in_window="
          f"{n_in_window}, gate: top1>=0.667 AND in_top3>=0.667 AND beats<=0.333 AND breadth<=3) ===",
          flush=True)
    print(f"{'model':<42} {'top1':>6} {'in_top3':>7} {'beats':>6} {'brdth':>5}  verdict",
          flush=True)
    for r in per_model:
        print(f"{r['model']:<42} {r['target_top1_rate']:>6.3f} "
              f"{r['target_in_top3_rate']:>7.3f} {r['competitor_beats_rate']:>6.3f} "
              f"{str(r['median_breadth']):>5}  {'SHIP' if r['ship'] else 'HOLD'}",
              flush=True)
    print(f"\nwrote {OUT_PATH}", flush=True)
    any_ship = any(r["ship"] for r in per_model)
    return 0 if any_ship else 2


if __name__ == "__main__":
    raise SystemExit(main())