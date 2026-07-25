"""Leave-one-session-out generalization probe (consult #8 upstream check).

DeepSeek #8 flagged that the 2-session / 17-pair held-out set is pathologically
hard (both sessions are self-referential meta-engine register, absent from the 51
trained sessions) and may be a test-set outlier rather than a true generalization
gap. The decisive test: does the fine-tuned CE generalize to a *normal* held-out
project session, not just the meta-engine register?

METHOD. For each trained session S with >= LOO_MIN_PAIRS verified gold pairs:
  - train on all 123 gold pairs MINUS S's pairs (and their attached hard-negs)
  - fine-tune a fresh ms-marco CE (same recipe as _finetune_ce.py)
  - eval on S's pairs with rank-then-budget (the ship-gate metric)
  - compare to zero-shot ms-marco on S's pairs (same eval, un-fine-tuned)
  - compare to the EXISTING all-51 fine-tuned CE on S's pairs (circular CEILING --
    it saw S during training; if it does not lift on its own train sessions, the
    rank-then-budget top1 metric itself is the problem, not generalization)

DECISION RULE (per DeepSeek #8):
  - LOO top1 lifts over zero-shot on the normal sessions -> the meta-engine
    17-pair set is the outlier; the generalization gap is NOT confirmed;
    recalibrate the gate.
  - LOO top1 shows zero lift (== zero-shot) on the normal sessions too ->
    generalization gap is real; proceed to Fork A (LLM-as-salience).

The all-51 ceiling is a control: it should lift on its own train sessions (it
memorized them). If it does NOT, the eval metric is too strict and the whole
HOLD story changes.

UNTRACKED scratch. onyx PRIVATE -- nothing leaves the box. No uploads.
Run: PYTHONPATH=. python scripts/_scratch/_loo_generalization_probe.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
EP_TRAINED = SCRATCH / "_trained_episodes_for_labeling.json"
GOLD_TRAINED = SCRATCH / "_trained_gold.json"
ALL51_MODEL = SCRATCH / "_ce_anaphora_finetuned"
OUT_RESULT = SCRATCH / "_loo_generalization_result.json"

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_COS_FLOOR = 0.4
BUDGET = 3
NEG_PER_POS = 4
EPOCHS = 3
LR = 2e-5
BATCH = 16
WARMUP_FRAC = 0.1
SEED = 7
LOO_MIN_PAIRS = 6  # sessions with >= this many pairs get a leave-one-out eval


def _cosine(a, b):
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y; na += x * x; nb += y * y
    return 0.0 if na <= 0 or nb <= 0 else dot / (math.sqrt(na) * math.sqrt(nb))


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _user_turns(sess):
    return [t["text"] for t in sess["turns"] if t.get("role") == "user"]


def build_train_examples(embedder, ep, gold, exclude_sid=None):
    """Same as _finetune_ce.build_train_examples but can drop one session's pairs."""
    window = int(gold.get("window", 16))
    age_threshold = int(gold.get("age_threshold", 3))
    sess_cache: dict[str, list] = {}
    pos, neg = [], []
    for p in gold["pairs"]:
        sid = p["session_id"]
        if exclude_sid is not None and sid == exclude_sid:
            continue
        q = int(p["query"]); t = int(p["target"])
        sess = ep.get(sid)
        if sess is None:
            continue
        ut = _user_turns(sess)
        if q >= len(ut) or t >= len(ut):
            continue
        pos.append((ut[q], ut[t], 1.0))
        lo = max(0, q - window)
        cands = [j for j in range(lo, q) if j != t and (q - j - 1) >= age_threshold]
        if not cands:
            continue
        if sid not in sess_cache:
            sess_cache[sid] = embedder.encode(ut)
        embs = sess_cache[sid]; q_emb = embs[q]
        ranked = sorted(cands, key=lambda j: _cosine(q_emb, embs[j]), reverse=True)
        for j in ranked[:NEG_PER_POS]:
            neg.append((ut[q], ut[j], 0.0))
    return pos + neg


def eval_session(ce, embedder, ep, session_pairs, window, age_threshold):
    """Rank-then-budget over one session's gold pairs. Identical math to the
    ship-gate probe."""
    sid = session_pairs[0]["session_id"]
    sess = ep.get(sid)
    if sess is None:
        return None
    ut = _user_turns(sess)
    embs = embedder.encode(ut)
    n_top1 = n_top3 = n_beats = 0
    breadths = []
    n_in = 0
    per_pair = []
    for p in session_pairs:
        q = int(p["query"]); t = int(p["target"])
        if q >= len(ut) or t >= len(ut):
            continue
        age = q - t - 1
        lo = max(0, q - window)
        win_idx = list(range(lo, q))
        in_window = (age >= age_threshold and age <= window - 1 and t >= lo)
        if not in_window:
            continue
        n_in += 1
        q_emb = embs[q]
        win_cos = [(j, _cosine(q_emb, embs[j])) for j in win_idx]
        cand = [j for j, c in win_cos if c > CANDIDATE_COS_FLOOR]
        t_in = t in cand
        ce_pairs = [(ut[q], ut[j]) for j in win_idx]
        raw = ce.predict(ce_pairs)
        scores = [float(r[0]) if isinstance(r, (list, tuple)) else float(r) for r in raw]
        ce_by = dict(zip(win_idx, scores))
        if cand:
            ranked = sorted(cand, key=lambda j: ce_by[j], reverse=True)
            rank = ranked.index(t) + 1 if t_in else None
            t_ce = ce_by.get(t, float("nan"))
            max_comp = max((ce_by[j] for j in cand if j != t), default=float("-inf"))
        else:
            rank = None; t_ce = float("nan"); max_comp = float("-inf")
        top1 = rank == 1
        in3 = rank is not None and rank <= BUDGET
        beats = (max_comp >= t_ce) if (t_in and not math.isnan(t_ce)) else True
        breadth = min(BUDGET, len(cand))
        if top1: n_top1 += 1
        if in3: n_top3 += 1
        if beats: n_beats += 1
        breadths.append(float(breadth))
        per_pair.append({"q": q, "t": t, "age": age, "rank": rank,
                         "top1": top1, "in_top3": in3, "beats": beats,
                         "breadth": breadth})
    if not n_in:
        return None
    return {
        "sid": sid[:8], "n_in": n_in,
        "top1": round(n_top1 / n_in, 4),
        "in_top3": round(n_top3 / n_in, 4),
        "beats": round(n_beats / n_in, 4),
        "median_breadth": round(_median(breadths), 4),
        "per_pair": per_pair,
    }


def finetune(rows, device):
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    import torch
    random.seed(SEED); torch.manual_seed(SEED)
    samples = [InputExample(texts=[a, b], label=float(lbl)) for a, b, lbl in rows]
    loader = DataLoader(samples, shuffle=True, batch_size=BATCH, drop_last=False)
    ce = CrossEncoder(BASE_MODEL, num_labels=1, device=device)
    n_steps = len(loader) * EPOCHS
    ce.fit(train_dataloader=loader, epochs=EPOCHS,
           optimizer_params={"lr": LR}, warmup_steps=int(n_steps * WARMUP_FRAC),
           show_progress_bar=False, use_amp=(device == "cuda"))
    return ce


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.subconscious.training.routing_training import build_embedder
    from sentence_transformers import CrossEncoder
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} base={BASE_MODEL}", flush=True)

    ep = json.loads(EP_TRAINED.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))
    window = int(gold.get("window", 16))
    age_threshold = int(gold.get("age_threshold", 3))
    embedder = build_embedder("on-demand")

    # group gold pairs by session, pick the leave-one-out set
    by_sid: dict[str, list] = {}
    for p in gold["pairs"]:
        by_sid.setdefault(p["session_id"], []).append(p)
    loo_sids = sorted([s for s, ps in by_sid.items() if len(ps) >= LOO_MIN_PAIRS])
    print(f"[plan] {len(loo_sids)} sessions with >= {LOO_MIN_PAIRS} pairs -> "
          f"leave-one-out: {[' '.join(s[:8] for s in loo_sids)]}", flush=True)

    # zero-shot baseline + all-51 ceiling are loaded ONCE and reused per session
    print("[load] zero-shot ms-marco + existing all-51 fine-tuned CE ...", flush=True)
    ce_zero = CrossEncoder(BASE_MODEL, num_labels=1, device=device)
    ce_all51 = CrossEncoder(str(ALL51_MODEL), device=device) if ALL51_MODEL.exists() else None

    results = []
    for idx, sid in enumerate(loo_sids, 1):
        session_pairs = by_sid[sid]
        print(f"\n=== [{idx}/{len(loo_sids)}] {sid[:8]} ({len(session_pairs)} pairs) ===",
              flush=True)

        # zero-shot on S
        r_zero = eval_session(ce_zero, embedder, ep, session_pairs, window, age_threshold)

        # all-51 ceiling on S (circular -- it saw S in training)
        r_all51 = None
        if ce_all51 is not None:
            r_all51 = eval_session(ce_all51, embedder, ep, session_pairs, window, age_threshold)

        # leave-one-out: train on the OTHER sessions, fresh CE, eval on S
        rows = build_train_examples(embedder, ep, gold, exclude_sid=sid)
        n_pos = sum(1 for r in rows if r[2] == 1.0)
        n_neg = sum(1 for r in rows if r[2] == 0.0)
        print(f"  [loo-train] pos={n_pos} neg={n_neg} (excluded {sid[:8]})", flush=True)
        ce_loo = finetune(rows, device)
        r_loo = eval_session(ce_loo, embedder, ep, session_pairs, window, age_threshold)
        del ce_loo
        if device == "cuda":
            torch.cuda.empty_cache()

        row = {
            "sid": sid[:8], "n": (r_loo["n_in"] if r_loo else 0),
            "zero_shot":  {"top1": r_zero["top1"],  "in_top3": r_zero["in_top3"]}  if r_zero  else None,
            "loo_50":      {"top1": r_loo["top1"],   "in_top3": r_loo["in_top3"]}   if r_loo   else None,
            "all51_ceiling": {"top1": r_all51["top1"], "in_top3": r_all51["in_top3"]} if r_all51 else None,
        }
        results.append(row)
        z = row["zero_shot"]; l = row["loo_50"]; a = row["all51_ceiling"]
        print(f"  zero-shot    top1={z['top1']}  in_top3={z['in_top3']}", flush=True)
        print(f"  LOO(50sess)  top1={l['top1']}  in_top3={l['in_top3']}  "
              f"(lift top1 {l['top1']-z['top1']:+.3f})", flush=True)
        if a:
            print(f"  all51 ceil   top1={a['top1']}  in_top3={a['in_top3']}  "
                  f"(circular ceiling; saw {sid[:8]})", flush=True)

    # aggregate
    def mean(key, src):
        vals = [r[src][key] for r in results if r.get(src)]
        return sum(vals) / len(vals) if vals else float("nan")
    agg = {
        "n_sessions": len(results),
        "mean_zero_shot_top1":   round(mean("top1", "zero_shot"), 4),
        "mean_loo_top1":         round(mean("top1", "loo_50"), 4),
        "mean_all51_top1":       round(mean("top1", "all51_ceiling"), 4),
        "mean_zero_shot_in_top3": round(mean("in_top3", "zero_shot"), 4),
        "mean_loo_in_top3":       round(mean("in_top3", "loo_50"), 4),
        "mean_all51_in_top3":    round(mean("in_top3", "all51_ceiling"), 4),
    }
    lift_top1 = agg["mean_loo_top1"] - agg["mean_zero_shot_top1"]
    agg["mean_lift_top1_loo_vs_zero"] = round(lift_top1, 4)
    agg["results"] = results

    print("\n=== LEAVE-ONE-SESSION-OUT GENERALIZATION (normal-register sessions) ===",
          flush=True)
    print(f"  sessions tested         = {agg['n_sessions']}", flush=True)
    print(f"  mean zero-shot top1     = {agg['mean_zero_shot_top1']}", flush=True)
    print(f"  mean LOO(50sess) top1   = {agg['mean_loo_top1']}  "
          f"(lift vs zero-shot {lift_top1:+.3f})", flush=True)
    print(f"  mean all51 ceiling top1 = {agg['mean_all51_top1']}  (circular)", flush=True)
    print(f"  mean zero-shot in_top3  = {agg['mean_zero_shot_in_top3']}", flush=True)
    print(f"  mean LOO(50sess) in_top3= {agg['mean_loo_in_top3']}", flush=True)
    print(f"  mean all51 in_top3      = {agg['mean_all51_in_top3']}", flush=True)
    if lift_top1 > 0.05:
        print("  -> LOO LIFTS over zero-shot on normal sessions: the meta-engine "
              "17-pair set is an OUTLIER; gap NOT confirmed -> recalibrate gate.",
              flush=True)
    elif abs(lift_top1) <= 0.05:
        print("  -> LOO shows ZERO lift over zero-shot on normal sessions too: "
              "generalization gap CONFIRMED -> proceed to Fork A (LLM-as-salience).",
              flush=True)
    else:
        print("  -> LOO WORSE than zero-shot: fine-tune HURTS out-of-session; "
              "generalization gap CONFIRMED -> proceed to Fork A.", flush=True)

    OUT_RESULT.write_text(json.dumps(agg, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nwrote {OUT_RESULT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())