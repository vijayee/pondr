"""The granularity ABLATION on human-verified labels (CodeTraceBench slice).

The real answer to 'how fine must intent capture be?' — scored against the
dataset's human-verified step-utility labels (not a judge, not completion):
for each mapped step (label = incorrect/unuseful = 1, else 0), predict
faultiness from each declaration grain:

  action-only     -- the step's own action text (the no-declaration baseline)
  task+action     -- the COARSE grain: the task goal text + action
  plan+action     -- the AGENT'S OWN declared plan (self-declared intent) + action
  stage+action    -- the BATCH grain: stage membership (via the manifest's
                     stage ranges) + action (+the plan's phase text where
                     present)

Each grain's AUC + marginal vs action-only = its measured value.

Usage: python scripts/icm/granularity_ablation_codetrace.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

MANIFEST = os.path.join("data", "_scratch", "codetrace", "bench_manifest.verified.jsonl")
MAPPED = os.path.join("data", "_scratch", "icm", "capture", "codetrace_mapped.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "granularity_ablation_codetrace_results.json")

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
    return round(wins / (len(pos) * len(neg)), 3)


def main() -> int:
    mapped = [json.loads(l) for l in io.open(MAPPED, encoding="utf-8") if l.strip()]
    hid_to_plan = {r["id"]: r["text"] for r in mapped if r["level"] == "plan"}
    hid_to_task = {r["id"]: r["text"] for r in mapped if r["level"] == "goal"}
    manifest = {r["task_name"] + "-" + (r.get("agent") or "?")[:12]:
                r for r in (json.loads(l) for l in
                            io.open(MANIFEST, encoding="utf-8") if l.strip())}

    def stage_of(hid, step_id):
        row = manifest.get(hid.replace("ct-", "CT-MISS-"))
        if row is None:
            # find manifest row by the hid's task + agent substring
            key_task = hid[3:]
            for k, r in manifest.items():
                if hid.startswith("ct-" + r["task_name"]):
                    row = r
                    break
        if row is None:
            return ""
        for st in row.get("stages") or []:
            if st["start_step_id"] <= step_id <= st["end_step_id"]:
                return f"in stage {st['stage_id']}"
        return ""

    events = []
    for r in mapped:
        if r["level"] != "step":
            continue
        hid = r["parent"]
        y = 1 if r.get("human_label") in ("incorrect", "unuseful") else 0
        events.append({"task_text": hid_to_task.get(hid, ""),
                       "plan_text": hid_to_plan.get(hid, ""),
                       "stage_text": stage_of(hid, r.get("step_id", 0)),
                       "action": r["text"][:200], "y": y})
    n_pos = sum(e["y"] for e in events)
    print(f"[data] {len(events)} steps; faulty/unuseful labeled: {n_pos}")
    if n_pos < 10 or n_pos == len(events):
        print("labels too sparse/degenerate; pulling more artifacts needed")
        return 1

    q = [e["action"] for e in events]
    mem_texts = q  # predicting faultiness of the step itself
    mvecs = _emb(q)
    grains = {
        "action-only": q,
        "task+action": [t + " | " + a for t, a in
                        zip((e["task_text"] for e in events), q)],
        "plan+action": [p + " | " + a for p, a in
                        zip((e["plan_text"] for e in events), q)],
        "stage+action": [s + " | " + a for s, a in
                         zip((e["stage_text"] for e in events), q)],
    }
    y = [e["y"] for e in events]
    out = {}
    for name, texts in grains.items():
        gvecs = _emb(texts)
        scores = [float(np.dot(mv, gv)) for mv, gv in zip(mvecs, gvecs)]
        out[name] = {"auc": auroc(scores, y)}
    base = out["action-only"]["auc"]
    for name in out:
        out[name]["marginal_vs_action"] = round(out[name]["auc"] - base, 3)
    out["summary"] = {"n_steps": len(events), "n_faulty": n_pos,
                      "verdict_grain": max(out, key=lambda k: out[k]["auc"]) if base is not None else "?",
                      "note": "faultiness prediction; direction-of-interpretation: "
                              "AUC~0.5 = grain uninformative; high/low = informative "
                              "(sign from marginals)"}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())