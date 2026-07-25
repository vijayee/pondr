"""Fine-tune a cross-encoder on the mined gold anaphora pairs + eval on the 17
held-out (Task #123). UNTRACKED scratch. The ship gate.

ARCHITECTURE (DeepSeek consult #7, recalibrated): the zero-shot MS MARCO CE has
the right architecture (bi-encoder cross-attention) but the wrong training
distribution (web query->passage, not conversational anaphora). Fine-tuning the
same small base on in-domain gold fixes the distribution mismatch. Predicted
target_top1=0.71, target_in_top3=0.88 on the 17 held-out.

DATA: _trained_gold.json (verified pairs mined from the 51 TRAINED sessions --
NON-CIRCULAR with the 17 held-out test set). Each gold pair -> one positive
(query_text, target_text, 1.0); K in-window non-target turns -> hard-ish
negatives (query_text, neg_text, 0.0). No back-translation for v1 (deferred to
the top1<0.65 fallback).

TRAIN: cross-encoder/ms-marco-MiniLM-L-6-v2 base (best zero-shot, best fine-tune
base per the sweep), binary head, 3 epochs, lr 2e-5, batch 16, warmup 10%.

EVAL: the 17 held-out with rank-then-budget (bge_cos cos>0.4 pre-filter -> CE
re-score -> top-3), identical to _probe_heldout_reranker_sweep.py so the number
is directly comparable to the zero-shot baseline (0.294/0.588).

SHIP GATE: n>=6 AND target_top1>=2/3 AND target_in_top3>=2/3 AND
competitor_beats<=1/3 AND median_breadth<=3.

Run (GPU preferred; falls back to CPU):
  PYTHONPATH=. python scripts/_scratch/_finetune_ce.py
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
EP_HELDOUT = SCRATCH / "_heldout_episodes.json"
GOLD_HELDOUT = SCRATCH / "_heldout_gold.json"
OUT_MODEL = SCRATCH / "_ce_anaphora_finetuned"
OUT_RESULT = SCRATCH / "_ce_finetune_result.json"

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_COS_FLOOR = 0.4
BUDGET = 3
NEG_PER_POS = 4
EPOCHS = 3
LR = 2e-5
BATCH = 16
WARMUP_FRAC = 0.1
SEED = 7


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


def build_train_examples(embedder):
    """Positives from gold; negatives = in-window non-target user turns."""
    ep = json.loads(EP_TRAINED.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))
    window = int(gold.get("window", 16))
    age_threshold = int(gold.get("age_threshold", 3))
    rng = random.Random(SEED)
    # Precompute embeddings per session lazily for negative sampling by cosine
    # (pick the highest-cos non-target turns = hard negatives).
    sess_cache: dict[str, list] = {}
    pos, neg = [], []
    for p in gold["pairs"]:
        sid = p["session_id"]; q = int(p["query"]); t = int(p["target"])
        sess = ep.get(sid)
        if sess is None:
            continue
        ut = _user_turns(sess)
        if q >= len(ut) or t >= len(ut):
            continue
        q_txt, t_txt = ut[q], ut[t]
        pos.append((q_txt, t_txt, 1.0))
        # candidate negatives = user turns in [q-window, q-1] excluding target,
        # age >= age_threshold
        lo = max(0, q - window)
        cands = [j for j in range(lo, q) if j != t and (q - j - 1) >= age_threshold]
        if not cands:
            continue
        if sid not in sess_cache:
            sess_cache[sid] = embedder.encode(ut)
        embs = sess_cache[sid]
        q_emb = embs[q]
        ranked = sorted(cands, key=lambda j: _cosine(q_emb, embs[j]), reverse=True)
        # take top-NEG_PER_POS by cosine (hardest negatives)
        for j in ranked[:NEG_PER_POS]:
            neg.append((q_txt, ut[j], 0.0))
    print(f"[data] positives={len(pos)} negatives={len(neg)} "
          f"(ratio 1:{len(neg)/max(1,len(pos)):.1f})", flush=True)
    return pos + neg


def eval_heldout(ce):
    """Rank-then-budget over the 17 held-out -- identical to the sweep probe."""
    episodes = json.loads(EP_HELDOUT.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_HELDOUT.read_text(encoding="utf-8"))
    window = int(gold.get("window", 16))
    age_threshold = int(gold.get("age_threshold", 3))
    held_ids = {p["session_id"] for p in gold["pairs"]}
    sys.path.insert(0, str(ROOT))
    from src.subconscious.training.routing_training import build_embedder
    embedder = build_embedder("on-demand")
    texts, embs = {}, {}
    for sid in held_ids:
        sess = episodes.get(sid)
        if sess is None:
            continue
        ut = _user_turns(sess)
        texts[sid] = ut
        embs[sid] = embedder.encode(ut)
    n_top1 = n_top3 = n_beats = 0
    breadths = []
    n_in = 0
    per_pair = []
    for p in gold["pairs"]:
        sid = p["session_id"]; q = int(p["query"]); t = int(p["target"])
        if sid not in texts or q >= len(texts[sid]) or t >= len(texts[sid]):
            continue
        ut = texts[sid]; uembs = embs[sid]
        age = q - t - 1
        lo = max(0, q - window)
        win_idx = list(range(lo, q))
        in_window = (age >= age_threshold and age <= window - 1 and t_idx_ok(t, lo))
        if not in_window:
            continue
        n_in += 1
        q_emb = uembs[q]
        win_cos = [(j, _cosine(q_emb, uembs[j])) for j in win_idx]
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
        per_pair.append({"sid": sid[:8], "q": q, "t": t, "age": age,
                         "rank": rank, "top1": top1, "in_top3": in3,
                         "beats": beats, "breadth": breadth})
    top1 = n_top1 / n_in if n_in else 0.0
    top3 = n_top3 / n_in if n_in else 0.0
    beats = n_beats / n_in if n_in else 0.0
    med = _median(breadths) if breadths else float("nan")
    ship = (n_in >= 6 and top1 >= 2/3 and top3 >= 2/3 and beats <= 1/3 and med <= BUDGET)
    return {"n_in_window": n_in, "target_top1_rate": round(top1, 4),
            "target_in_top3_rate": round(top3, 4),
            "competitor_beats_rate": round(beats, 4),
            "median_breadth": round(med, 4) if not math.isnan(med) else None,
            "ship": ship, "per_pair": per_pair}


def t_idx_ok(t, lo):
    return t >= lo


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.subconscious.training.routing_training import build_embedder
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    import torch

    random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} base={BASE_MODEL}", flush=True)

    embedder = build_embedder("on-demand")
    rows = build_train_examples(embedder)
    if len(rows) < 20:
        print(f"[ERR] too few training rows ({len(rows)}); aborting", flush=True)
        return 1
    train_samples = [InputExample(texts=[a, b], label=float(lbl)) for a, b, lbl in rows]
    loader = DataLoader(train_samples, shuffle=True, batch_size=BATCH, drop_last=False)

    ce = CrossEncoder(BASE_MODEL, num_labels=1, device=device)
    n_steps = len(loader) * EPOCHS
    warmup = int(n_steps * WARMUP_FRAC)
    print(f"[train] rows={len(rows)} epochs={EPOCHS} steps={n_steps} "
          f"warmup={warmup} lr={LR}", flush=True)
    ce.fit(
        train_dataloader=loader,
        epochs=EPOCHS,
        optimizer_params={"lr": LR},
        warmup_steps=warmup,
        show_progress_bar=True,
        use_amp=(device == "cuda"),
    )
    ce.save(str(OUT_MODEL))
    print(f"[save] fine-tuned CE -> {OUT_MODEL}", flush=True)

    print("[eval] held-out 17 rank-then-budget gate ...", flush=True)
    res = eval_heldout(ce)
    res["base_model"] = BASE_MODEL
    res["train_rows"] = len(rows)
    res["epochs"] = EPOCHS
    res["neg_per_pos"] = NEG_PER_POS
    OUT_RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n=== CE FINE-TUNE HELD-OUT GATE (n={res['n_in_window']}) ===", flush=True)
    print(f"  target_top1   = {res['target_top1_rate']}  (need >= 0.667)", flush=True)
    print(f"  target_in_top3= {res['target_in_top3_rate']}  (need >= 0.667)", flush=True)
    print(f"  competitor_beats={res['competitor_beats_rate']}  (need <= 0.333)", flush=True)
    print(f"  median_breadth= {res['median_breadth']}  (need <= 3)", flush=True)
    print(f"  VERDICT: {'SHIP' if res['ship'] else 'HOLD'}", flush=True)
    print(f"\nwrote {OUT_RESULT}", flush=True)
    return 0 if res["ship"] else 2


if __name__ == "__main__":
    raise SystemExit(main())