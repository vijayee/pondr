"""C1 -- cosine+age + CE reranker on the fired set (rank-then-budget).

Reusable ranker for the STRM 3-way A/B/C isolation test (Step 1). The cosine+age
tract ([[pondr-strm-cosage-realonyx-result]] HOLD at 0.235/0.588) gets a zero-shot
cross-encoder reranker on top: bge_cos pre-filters the 16-turn window to the fired
set (cos > cos_floor), a MS MARCO cross-encoder re-scores the fired set, take
top-budget. breadth = min(budget, |fired|) by construction; the open question is
whether the gold referent is in the surfaced top-3 (recall@budget) and at rank-1
(pinpoint). No training, no backbone, no engine change.

This is the C1 leg of the 3-way comparison. The reusable ``Reranker.rank`` is
imported by Step 2 (anaphora cross-test) and Step 3 (chat harness). The battery
runner here sweeps ``cos_floor`` (and reranker model) over the anaphora battery:
17-pair held-out + LOO-8 normal (the 8 trained sessions with >=6 gold pairs), and
writes the C1-best numbers.

FIDELITY: same gold files, same ``_user_turns`` 16-turn window, age>=3, and
rank-then-budget=BUDGET=3 metric as the LLM probe
(``scripts/_scratch/_llm_salience_probe.py``) -- apples-to-apples across C1/C2/C3.
``rank`` returns WINDOW-RELATIVE indices (0-based into ``window_turns``), matching
``llm_rank``'s contract.

UNTRACKED scratch. onyx PRIVATE -- nothing leaves the box. No uploads. No engine
edits. Per CLAUDE.md de-wonk at completion.

Run (downloads public rerankers on first run; no server needed):
  PYTHONPATH=. python scripts/_scratch/_c1_reranker.py
  (env: CE_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  default; WHICH=both|heldout|loo)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
EP_HELDOUT = SCRATCH / "_heldout_episodes.json"
GOLD_HELDOUT = SCRATCH / "_heldout_gold.json"
EP_TRAINED = SCRATCH / "_trained_episodes_for_labeling.json"
GOLD_TRAINED = SCRATCH / "_trained_gold.json"
OUT_PATH = SCRATCH / "_c1_reranker_result.json"

BUDGET = 3
LOO_MIN_PAIRS = 6
DEFAULT_COS_FLOOR = 0.4
# Reranker candidates to sweep (the "best out of the tract"). ms-marco is the
# baseline CE; bge-reranker-v2-m3 is a stronger reranker (per the sweep probe).
# is_nli flags 3-class NLI models whose score = entailment logit (column 0).
RERANKERS = [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", False),
    ("BAAI/bge-reranker-v2-m3", False),
]
COS_FLOORS = [0.3, 0.4, 0.5, 0.6]


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


def _user_turns(sess: dict) -> list[str]:
    return [t["text"] for t in sess["turns"] if t.get("role") == "user"]


def _rank_from_arrays(win_cos: list[float], win_ce: list[float],
                      cos_floor: float, budget: int) -> tuple[list[int], list[float]]:
    """rank-then-budget from precomputed per-slot cosine + CE scores. Returns
    (window-relative idxs most-relevant-first, ce_scores) over the fired set,
    capped at budget. Empty if no slot clears cos_floor."""
    n = len(win_cos)
    fired = [i for i in range(n) if win_cos[i] > cos_floor]
    if not fired:
        return [], []
    fired.sort(key=lambda i: win_ce[i], reverse=True)
    top = fired[:budget]
    return top, [win_ce[i] for i in top]


class Reranker:
    """cosine+age + CE reranker. Loads the bge embedder (byte-identical to
    serve's cosine pre-filter) and one cross-encoder. ``rank`` is the reusable
    entry point for the chat harness; the battery runner uses the cheaper
    array path (precompute cos+ce once, sweep cos_floor)."""

    def __init__(self, model_id: str, is_nli: bool = False, device: str = "auto"):
        sys.path.insert(0, str(ROOT))
        from src.subconscious.training.routing_training import build_embedder
        from sentence_transformers import CrossEncoder
        import torch

        self.model_id = model_id
        self.is_nli = is_nli
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.embedder = build_embedder("on-demand")
        self.ce = CrossEncoder(model_id, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts)

    def cos_window(self, query_text: str, window_turns: list[str]) -> list[float]:
        embs = self.embed([query_text] + list(window_turns))
        q = embs[0]
        return [_cosine(q, embs[i + 1]) for i in range(len(window_turns))]

    def ce_window(self, query_text: str, window_turns: list[str]) -> list[float]:
        pairs = [(query_text, t) for t in window_turns]
        scores = self.ce.predict(pairs)
        try:
            ndim = scores.ndim
        except AttributeError:
            scores = list(scores)
            ndim = 1 if not isinstance(scores[0], (list, tuple)) else 2
        if self.is_nli and ndim == 2:
            return [float(row[0]) for row in scores]
        return [float(row) if ndim == 1 else float(row[0]) for row in scores]

    def rank(self, query_text: str, window_turns: list[str],
             cos_floor: float = DEFAULT_COS_FLOOR,
             budget: int = BUDGET) -> tuple[list[int], list[float]]:
        """rank-then-budget over the window. Returns (window-relative idxs
        most-relevant-first, ce_scores), capped at budget. Empty if no slot
        clears cos_floor."""
        win_cos = self.cos_window(query_text, window_turns)
        win_ce = self.ce_window(query_text, window_turns)
        return _rank_from_arrays(win_cos, win_ce, cos_floor, budget)


def _precompute_pair_arrays(rk: Reranker, episodes: dict, pairs: list[dict],
                            window: int) -> list[dict]:
    """For each gold pair, embed the query + its 16-turn window once and cache
    per-slot cosine + CE (model-fixed). cos_floor is swept later from these
    arrays. Skips pairs whose query/target are out of range."""
    # Embed per-session user turns once (windows overlap heavily within a
    # session); index into the cached embeddings for cosine. CE is pair-wise
    # so it is computed per (query, window) -- cached per pair.
    out: list[dict] = []
    sess_cache: dict[str, dict] = {}
    for p in pairs:
        sid = p["session_id"]
        q_idx = int(p["query"])
        t_idx = int(p["target"])
        if sid not in sess_cache:
            sess = episodes.get(sid)
            if sess is None:
                out.append({"error": f"session {sid} missing"})
                continue
            ut = _user_turns(sess)
            sess_cache[sid] = {"uturns": ut, "embs": rk.embed(ut)}
        sc = sess_cache[sid]
        uturns = sc["uturns"]
        uembs = sc["embs"]
        if q_idx >= len(uturns) or t_idx >= len(uturns):
            out.append({"error": "index-out-of-range"})
            continue
        age = q_idx - t_idx - 1
        lo = max(0, q_idx - window)
        win_idx = list(range(lo, q_idx))
        q_emb = uembs[q_idx]
        win_cos = [_cosine(q_emb, uembs[j]) for j in win_idx]
        win_texts = [uturns[j] for j in win_idx]
        win_ce = rk.ce_window(uturns[q_idx], win_texts) if win_texts else []
        out.append({
            "sid": sid, "q_idx": q_idx, "t_idx": t_idx, "age": age,
            "lo": lo, "win_idx": win_idx, "win_cos": win_cos, "win_ce": win_ce,
            "target_in_win": t_idx - lo,
            "in_window": None,  # set by caller who knows age_threshold
            "q_preview": uturns[q_idx][:90], "t_preview": uturns[t_idx][:90],
            "note": p.get("note", ""),
        })
    return out


def _score_arrays(arrays: list[dict], age_threshold: int, window: int,
                  cos_floor: float, budget: int) -> tuple[dict, list[dict]]:
    """Score precomputed pair arrays at one cos_floor. Returns (aggregate,
    per_pair). Mirrors the LLM probe's metrics so C1/C2/C3 are comparable."""
    n_in = 0
    n_top1 = n_top3 = n_beats = n_fires = 0
    breadths: list[float] = []
    per_pair: list[dict] = []
    for a in arrays:
        if "error" in a:
            per_pair.append({"error": a["error"]})
            continue
        in_window = (a["age"] >= age_threshold and a["age"] <= window - 1
                     and a["target_in_win"] >= 0
                     and a["target_in_win"] < len(a["win_cos"]))
        if not in_window:
            per_pair.append({**{k: a[k] for k in ("sid", "q_idx", "t_idx", "age")},
                             "in_window": False})
            continue
        n_in += 1
        picked, scores = _rank_from_arrays(a["win_cos"], a["win_ce"],
                                           cos_floor, budget)
        # rank of the gold target within the picked window-positions
        rank = None
        for pos, wpos in enumerate(picked, 1):
            if wpos == a["target_in_win"]:
                rank = pos
                break
        top1 = rank == 1
        in_top3 = rank is not None and rank <= budget
        # fires_on_target: gold survives the cos pre-filter (cos > cos_floor)
        fires = a["win_cos"][a["target_in_win"]] > cos_floor
        # competitor_beats: a non-target slot outranks the target (or target
        # not surfaced at all). For a recall@budget frame this == not top1.
        beats = rank is None or rank > 1
        breadth = min(budget, len(picked))
        if top1:
            n_top1 += 1
        if in_top3:
            n_top3 += 1
        if beats:
            n_beats += 1
        if fires:
            n_fires += 1
        breadths.append(float(breadth))
        per_pair.append({
            "sid": a["sid"][:8], "q": a["q_idx"], "t": a["t_idx"],
            "age": a["age"], "in_window": True, "picked": picked,
            "rank": rank, "top1": top1, "in_top3": in_top3,
            "fires_on_target": fires, "competitor_beats": beats,
            "breadth": breadth, "target_cos": round(a["win_cos"][a["target_in_win"]], 4),
        })
    n = n_in
    agg = {
        "n_in_window": n,
        "target_top1_rate": round(n_top1 / n, 4) if n else 0.0,
        "target_in_top3_rate": round(n_top3 / n, 4) if n else 0.0,
        "competitor_beats_rate": round(n_beats / n, 4) if n else 0.0,
        "fires_on_target_rate": round(n_fires / n, 4) if n else 0.0,
        "median_breadth": round(_median(breadths), 4) if breadths else None,
    }
    return agg, per_pair


def _loo_pairs(gold: dict) -> list[dict]:
    """The LOO-8 normal set: sessions with >= LOO_MIN_PAIRS gold pairs."""
    by_sid: dict[str, list[dict]] = {}
    for p in gold["pairs"]:
        by_sid.setdefault(p["session_id"], []).append(p)
    loo: list[dict] = []
    for sid, pairs in sorted(by_sid.items()):
        if len(pairs) >= LOO_MIN_PAIRS:
            loo.extend(pairs)
    return loo


def main() -> int:
    which = os.getenv("WHICH", "both")
    sys.path.insert(0, str(ROOT))
    import torch  # noqa: F401 -- used by Reranker for device detect

    # Load gold + episodes for both sets.
    heldout_ep = json.loads(EP_HELDOUT.read_text(encoding="utf-8"))
    heldout_g = json.loads(GOLD_HELDOUT.read_text(encoding="utf-8"))
    trained_ep = json.loads(EP_TRAINED.read_text(encoding="utf-8"))
    trained_g = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))

    sets: dict[str, dict] = {}
    if which in ("heldout", "both"):
        sets["heldout-17"] = {
            "episodes": heldout_ep, "pairs": heldout_g["pairs"],
            "window": int(heldout_g.get("window", 16)),
            "age_threshold": int(heldout_g.get("age_threshold", 3)),
        }
    if which in ("loo", "both"):
        sets["loo-8-normal"] = {
            "episodes": trained_ep, "pairs": _loo_pairs(trained_g),
            "window": int(trained_g.get("window", 16)),
            "age_threshold": int(trained_g.get("age_threshold", 3)),
        }

    out: dict = {"which": which, "budget": BUDGET, "cos_floors": COS_FLOORS,
                 "rerankers": [m for m, _ in RERANKERS], "sets": {}}

    for model_id, is_nli in RERANKERS:
        print(f"\n[reranker] {model_id} (nli={is_nli})", flush=True)
        rk = Reranker(model_id, is_nli=is_nli, device="auto")
        # Precompute per-pair cos+ce arrays once per (model, set).
        arrays_by_set: dict[str, list[dict]] = {}
        for set_name, s in sets.items():
            arr = _precompute_pair_arrays(rk, s["episodes"], s["pairs"],
                                         s["window"])
            arrays_by_set[set_name] = arr
            n_ok = sum(1 for a in arr if "error" not in a)
            print(f"  [{set_name}] precomputed {n_ok} pairs", flush=True)

        for cos_floor in COS_FLOORS:
            for set_name, s in sets.items():
                agg, per_pair = _score_arrays(arrays_by_set[set_name],
                                             s["age_threshold"], s["window"],
                                             cos_floor, BUDGET)
                key = f"{model_id}|cos_floor={cos_floor}|{set_name}"
                out["sets"].setdefault(set_name, {})[key] = {
                    "model": model_id, "is_nli": is_nli,
                    "cos_floor": cos_floor, **agg,
                }
                print(f"  {set_name:<14} cos_floor={cos_floor}  "
                      f"top1={agg['target_top1_rate']:.3f}  "
                      f"in_top3={agg['target_in_top3_rate']:.3f}  "
                      f"fires={agg['fires_on_target_rate']:.3f}  "
                      f"beats={agg['competitor_beats_rate']:.3f}  "
                      f"breadth={agg['median_breadth']}", flush=True)
        del rk

    # Pick C1-best per set: highest in_top3 (the recall metric), tiebreak top1,
    # then lowest competitor_beats. Record the winning config.
    best: dict[str, dict] = {}
    for set_name, configs in out["sets"].items():
        winner = None
        for key, c in configs.items():
            cand = (c["target_in_top3_rate"], c["target_top1_rate"],
                    -c["competitor_beats_rate"])
            if winner is None or cand > winner[0]:
                winner = (cand, key, c)
        best[set_name] = {"config_key": winner[1], **winner[2]}
    out["best"] = best

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print("\n=== C1 BEST (cosine+age + CE reranker, rank-then-budget) ===",
          flush=True)
    for set_name, b in best.items():
        print(f"  {set_name:<14} {b['model']} cos_floor={b['cos_floor']}  "
              f"top1={b['target_top1_rate']:.3f}  "
              f"in_top3={b['target_in_top3_rate']:.3f}  "
              f"fires={b['fires_on_target_rate']:.3f}  "
              f"beats={b['competitor_beats_rate']:.3f}  "
              f"breadth={b['median_breadth']}", flush=True)
    print(f"\nwrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())