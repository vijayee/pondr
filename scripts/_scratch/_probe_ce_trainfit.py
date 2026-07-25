"""Quick confirmatory probe: does the fine-tuned CE score training positives
above their hard negatives? If not, the pairwise signal isn't there even on
training data (explaining train_loss=0.70 >> 0.17 constant-floor). UNTRACKED."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
sys.path.insert(0, str(ROOT))
from src.subconscious.training.routing_training import build_embedder
from sentence_transformers import CrossEncoder

EP = json.loads((SCRATCH / "_trained_episodes_for_labeling.json").read_text(encoding="utf-8"))
GOLD = json.loads((SCRATCH / "_trained_gold.json").read_text(encoding="utf-8"))
WINDOW = int(GOLD.get("window", 16)); AGE = int(GOLD.get("age_threshold", 3))
NEG = 4; FLOOR = 0.4

def cos(a, b):
    d = na = nb = 0.0
    for x, y in zip(a, b): d += x*y; na += x*x; nb += y*y
    return 0.0 if na<=0 or nb<=0 else d/(math.sqrt(na)*math.sqrt(nb))

def uturns(s): return [t["text"] for t in s["turns"] if t.get("role")=="user"]

emb = build_embedder("on-demand")
ce = CrossEncoder(str(SCRATCH / "_ce_anaphora_finetuned"), device="cuda")

# sample 8 gold pairs, score pos + top-NEG negatives
pairs = GOLD["pairs"][:8]
pos_win = neg_win = 0; rows = []
cache = {}
for p in pairs:
    sid = p["session_id"]; q = int(p["query"]); t = int(p["target"])
    s = EP.get(sid)
    if not s: continue
    ut = uturns(s)
    if q >= len(ut) or t >= len(ut): continue
    if sid not in cache: cache[sid] = emb.encode(ut)
    em = cache[sid]; qe = em[q]
    lo = max(0, q - WINDOW)
    cands = [j for j in range(lo, q) if j != t and (q-j-1) >= AGE]
    cands = [j for j in cands if cos(qe, em[j]) > FLOOR]
    cands.sort(key=lambda j: cos(qe, em[j]), reverse=True)
    negs = cands[:NEG]
    pos_score = float(ce.predict([(ut[q], ut[t])])[0])
    neg_scores = [float(ce.predict([(ut[q], ut[j])])[0]) for j in negs]
    pos_beats = sum(1 for ns in neg_scores if pos_score > ns)
    pos_win += 1 if pos_score > max(neg_scores, default=-1e9) else 0
    neg_win += len(neg_scores)
    rows.append((sid[:8], q, t, round(pos_score,3), [round(x,3) for x in neg_scores], pos_beats))
    print(f"{sid[:8]} q{q:02d}->t{t:02d}  pos={pos_score:.3f}  negs={[round(x,3) for x in neg_scores]}  pos_beats={pos_beats}/{len(neg_scores)}")

tot = sum(r[5] for r in rows); tot_neg = sum(len(r[4]) for r in rows)
print(f"\npositives that beat ALL their hard negs: {pos_win}/{len(rows)}")
print(f"pos>neg pair-win rate: {tot}/{tot_neg} = {tot/max(1,tot_neg):.3f}")
print("(if ~0.5, CE cannot separate anaphoric target from topically-close competitors even on TRAINING data)")