"""Granularity ablation -- how fine does intent capture need to be?

Design (agreed 2026-10-15): capture events carry declarations at EVERY level
(goal -> step -> retrieval). This script measures each grain's MARGINAL
contribution to predicting retrieval utility:

  for each retrieval event e (memory m retrieved for query q at step s in
  goal g):
    y        = utility label from the outcome record (observed branch; the
               rule-computed label -- never stated)
    feature  = cos(embed(grain-text + " " + q), embed(m))  for each grain
               {goal, step, step+goal (full), query-only}
  the grain's value = AUC predicting y, plus the DELTA vs query-only
  (no-declaration) baseline, and full-vs-step-only (does the hierarchy add
  over the step line?).

THE FEED: two modes.
  --file capture.jsonl  -> real captured sessions (the intended use; the
                           ablation runs when >= 3 captured events exist).
  --synthetic           -> authored synthetic capture (INSTRUMENT TEST ONLY:
                           authored worlds carry compact discriminators by
                           the fixed-point theorem, so a synthetic positive
                           proves only that the ablation MEASURES, never that
                           fine grain WINS on real data).

Run:  python scripts/icm/granularity_ablation.py [--synthetic | --file X.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

OUT = os.path.join("data", "_scratch", "icm", "granularity_ablation_results.json")

_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l > 0.5]
    neg = [s for s, l in zip(scores, labels) if l <= 0.5]
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for n in neg if p > n) + \
        0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def synthetic_feed(n_goals=10, steps_per=3, retrs_per=3, seed=7):
    """Authored synthetic capture: usefulness depends ONLY on the step's
    specific object (the goal text carries none of it). If only the step
    text predicts y, the instrument sees the grain split (that is the ONLY
    claim a synthetic run can make)."""
    rng = random.Random(seed)
    actions = ["build", "verify", "measure", "refactor"]
    objects = ["index", "cascade", "gate", "store", "ledger", "probe",
               "cache", "index-head", "census", "ablation"]
    events = []
    gi = 0
    for _g in range(n_goals):
        gi += 1
        goal_text = f"research cycle {gi}: improve the theme area"
        objs = rng.sample(objects, steps_per)
        for ai, obj in enumerate(objs):
            step_text = f"{actions[ai % len(actions)]} the {obj}"
            for _r in range(retrs_per):
                if rng.random() < 0.5:
                    # useful: shares the STEP's object (not the goal's words)
                    mem = f"notes about the {obj} from an earlier session"
                    y = 1
                else:
                    other = rng.choice([o for o in objects if o != obj])
                    mem = f"notes about the {other} from an earlier session"
                    y = 0
                q = rng.choice(["what do we have on this?",
                                "what's relevant for my next move?",
                                "anything from before that would help?"])
                events.append({"goal_text": goal_text, "step_text": step_text,
                               "query_text": q, "memory_text": mem, "y": y})
    return events


def ablate(events, full_only=False):
    """Per-grain AUC over the events. Utility y is taken as given (the
    capture layer computed it from the outcomes)."""
    q_texts = [e["query_text"] for e in events]
    mem_texts = [e["memory_text"] for e in events]
    mvecs = _emb(mem_texts)
    grains = {
        "query-only": q_texts,
        "goal+q": [e["goal_text"] + " | " + e["query_text"] for e in events],
        "step+q": [e["step_text"] + " | " + e["query_text"] for e in events],
        "full(goal+step+q)": [e["goal_text"] + " | " + e["step_text"] + " | " +
                              e["query_text"] for e in events],
    }
    y = [e["y"] for e in events]
    if sum(y) in (0, len(y)):
        return {"error": "degenerate labels (all same class)", "n": len(y)}
    out = {}
    for name, texts in grains.items():
        gvecs = _emb(texts)
        scores = [float(np.dot(mv, gv)) for mv, gv in zip(mvecs, gvecs)]
        out[name] = {"auc": round(auroc(scores, y), 3)}
    q_base = out["query-only"]["auc"]
    for name in out:
        out[name]["marginal_vs_query_only"] = round(out[name]["auc"] - q_base, 3)
    out["summary"] = {
        "step_vs_goal": f'step {out["step+q"]["auc"]} vs goal {out["goal+q"]["auc"]}',
        "hierarchy_adds": round(out["full(goal+step+q)"]["auc"] -
                                out["step+q"]["auc"], 3),
        "n_events": len(events),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    mode = None
    if args.file:
        events = load_capture_file(args.file)
        mode = f"real capture: {args.file}"
    elif args.synthetic:
        events = synthetic_feed()
        mode = "SYNTHETIC (instrument test only -- authored world, no real claim)"
    else:
        print("need --file or --synthetic")
        return 2
    print(f"[mode] {mode}; {len(events)} retrieval events")
    res = ablate(events)
    print(json.dumps(res, indent=1))
    res = {"mode": mode, "results": res}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2))
    print(f"[done] wrote {OUT}")
    return 0


def load_capture_file(path: str):
    """Reconstruct ablation events from an IntentCapture mirror: goal/step
    declaration texts + retrieval events joined by the capture hierarchy.
    Utility y comes ONLY from outcome records (rule-computed labels)."""
    recs = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
    goals = {r["id"]: r["text"] for r in recs if r["level"] == "goal"}
    steps = {r["id"]: r for r in recs if r["level"] == "step"}
    events = {r["id"]: r for r in recs if r["level"] == "retrieval"}
    outcomes = {r["parent"]: r for r in recs if r["level"] == "outcome"}
    out = []
    for eid, ev in events.items():
        oc = outcomes.get(eid)
        if oc is None:
            continue  # an event with no outcome is not scored (no fabricating)
        s = steps.get(ev["parent"])
        gtext = ""
        if s is not None:
            gtext = goals.get(s["parent"], "")
        else:
            gtext = goals.get(ev["parent"], "")
        y = 1 if oc["observed"] == "desired" else 0
        if s is None:
            continue
        out.append({"goal_text": gtext, "step_text": s["text"],
                    "query_text": ev["query"],
                    "memory_text": " | ".join(ev["retrieved"]), "y": y})
    return out


if __name__ == "__main__":
    sys.exit(main())