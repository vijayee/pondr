"""THE LAST MEASUREMENT -- retrieval-utility of shared file-units (the flip
shaped memory units).

Question (the final unmeasured square of the program's grid): on shared
memory units (files touched by >=2 SWE tasks), does the intent-context
explain WHICH task's touch pays?

  unit:   (task, file) touch — a task's engagement with a shared file.
  y:      touch USEFUL = its task VERIFIER-SOLVED and its touching step(s)
          carry no human faulty-label; touch NOT useful = its touching
          step(s) carry a human incorrect/unuseful label. Touches in
          unsolved tasks with unlabeled steps = ambiguous, EXCLUDED
          (no fabricated class). `solved` is a necessary y-condition, so
          solved itself is never a feature.
  features (intent-context at capture time):
    f1 jaccard(task text, file path tokens)     -- the lexical leg
    f2 touch position in the trajectory (idx/step_count)
    f3 engagement (distinct steps touching the file in that task)
    f4 file-path depth
    f5 the agent's plan text mentions the file's basename? (0/1)
  arms:   singles + a small learned logistic stack (LOO over tasks) vs the
          trivial engagement-only rank.

THE PRE-REGISTERED READING: if the learned stack and/or singles separate
touches (AUC >> 0.5 pooled), the flip is cheap-explainable -> the grid
closes with "the router everywhere". If NO arm separates (all ~0.5), the
shared-file utility is beyond-cheap signal — the one place the intent
machinery could still earn its keep, now measured.
"""

from __future__ import annotations

import collections
import io
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

MAPPED = os.path.join("data", "_scratch", "icm", "capture", "codetrace_mapped.jsonl")
MANIFEST = os.path.join("data", "_scratch", "codetrace", "bench_manifest.verified.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "file_touch_utility_results.json")


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l > 0.5]
    neg = [s for s, l in zip(scores, labels) if l <= 0.5]
    if not pos or not neg:
        return None
    w = sum(1 for p in pos for n in neg if p > n) + \
        0.5 * sum(1 for p in pos for n in neg if p == n)
    return w / (len(pos) * len(neg))


def tokset(v):
    return {x for x in re.findall(r"[a-z0-9]+", str(v).lower()) if len(x) > 2}


def steps_for_task(mapped, hid):
    return [r for r in mapped if r["level"] == "step" and r["parent"] == hid]


def main() -> int:
    mapped = [json.loads(l) for l in io.open(MAPPED, encoding="utf-8") if l.strip()]
    task_txt = {r["id"]: r["text"] for r in mapped if r["level"] == "goal"}
    plan_txt = {r["id"]: r["text"] for r in mapped if r["level"] == "plan"}
    manifest = {}
    for r in (json.loads(l) for l in io.open(MANIFEST, encoding="utf-8") if l.strip()):
        manifest[r["task_name"]] = r
    solved_of = {}
    for hid in task_txt:
        tn = hid[3:]
        row = manifest.get(tn) or next((v for k, v in manifest.items()
                                        if hid.startswith("ct-" + k)), {})
        solved_of[hid] = bool(row.get("solved"))

    # touches: file -> {task -> [touch step records]}
    touches = collections.defaultdict(lambda: collections.defaultdict(list))
    total_steps = collections.Counter()
    for r in mapped:
        if r["level"] == "step":
            total_steps[r["parent"]] += 1
    import tarfile
    import glob as _g
    FILE_EXT = r"\.(py|js|ts|json|md|txt|rst|yaml|yml|toml|cfg|ini|c|cpp|h|hpp|vue|ipynb|html|css)\b"
    for tar in sorted(_g.glob(os.path.join("data", "_scratch", "codetrace", "trajs3", "*.tar.zst"))) + \
               sorted(_g.glob(os.path.join("data", "_scratch", "codetrace", "trajs2", "*.tar.zst"))):
        try:
            t = tarfile.open(tar, "r:*")
        except Exception:
            continue
        call_idx = 0
        for mem in sorted([m for m in t.getmembers()
                           if "tensorblock" in m.name and m.name.endswith(".json")],
                          key=lambda m: m.name):
            call_idx += 1
            parts = mem.name.split("/")
            if len(parts) < 4:
                continue
            tn = parts[2]
            hid = next((h for h in task_txt if hid_task_matches(h, tn)), None)
            if hid is None:
                continue
            try:
                data = json.load(t.extractfile(mem))
            except Exception:
                continue
            txt = json.dumps(data.get("response") or {}, default=str)[:6000]
            lab = None
            # the mapped step record matching this call index
            step_rows = steps_for_task(mapped, hid)
            match = next((s for s in step_rows if s.get("step_id") == call_idx), None)
            if match:
                lab = match.get("human_label")
            for tok in re.findall(r"[\w./\\-]{5,140}", txt):
                if "/" in tok and re.search(FILE_EXT, tok):
                    nt = tok.strip(" ,'\"();").lower()
                    if all(x not in nt for x in ("tensorblock", "swe_raw", "verified")):
                        touches[nt][hid].append(call_idx)
        t.close()

    # assemble the measurement rows
    X, y, tnames = [], [], []
    for fp, tasks in touches.items():
        if len(tasks) < 2:
            continue  # only shared memory units
        for hid, steps in tasks.items():
            labs = [next((s.get("human_label") for s in steps_for_task(mapped, hid)
                          if s.get("step_id") == si), None) for si in steps]
            faulty = any(l in ("incorrect", "unuseful") for l in labs)
            if faulty:
                label = 0
            elif solved_of.get(hid):
                label = 1
            else:
                continue  # ambiguous; excluded, never fabricated
            X.append({"fp": fp, "hid": hid,
                      "overlap": len(tokset(fp) & tokset(task_txt.get(hid, ""))) /
                                 max(1, len(tokset(fp) | tokset(task_txt.get(hid, "")))),
                      "pos": min(steps) / max(1, total_steps.get(hid, 1)),
                      "engage": len(set(steps)),
                      "depth": fp.count("/"),
                      "plan_hit": 1.0 if os.path.basename(fp).rsplit(".", 1)[0][:14] and
                      os.path.basename(fp).rsplit(".")[0].lower() in plan_txt.get(hid, "").lower()
                      else 0.0,
                      "y": label})
            tnames.append(fp)
    n_pos = sum(r["y"] for r in X)
    print(f"[data] shared unit touches measured: {len(X)} (useful {n_pos} / not {len(X) - n_pos});"
          f" shared units: {sum(1 for fp, t in touches.items() if len(t) >= 2)}")
    if not X or n_pos in (0, len(X)):
        print("degenerate; excluded rows too aggressive — report and stop")
        return 1
    # singles + learned stack
    yv = np.asarray([r["y"] for r in X], np.float32)
    arms = {
        "lexical-overlap": np.asarray([r["overlap"] for r in X], np.float32),
        "touch-position": np.asarray([-abs(r["pos"]) for r in X], np.float32),
        "engagement": np.asarray([r["engage"] for r in X], np.float32),
        "path-depth": np.asarray([r["depth"] for r in X], np.float32),
    }
    res = {k: safe_auc(auroc(v, yv)) for k, v in arms.items()}
    # learned stack (LOO over the fp units)
    Xs = np.stack([arms["lexical-overlap"], arms["touch-position"],
                   arms["engagement"], arms["path-depth"]], axis=1)
    res["learned-stack(lo)"] = loo_stack(Xs, yv, tnames)
    res["n_rows"] = len(X)
    res["n_useful"] = int(n_pos)
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    sep = [v for v in res.values() if isinstance(v, float) and (v >= 0.56 or v <= 0.44)]
    print(f"VERDICT: {'CHEAP-EXPLAINABLE (grid closes)' if sep else 'BEYOND CHEAP (the one open square)'}")
    print(f"[done] wrote {OUT}")
    return 0


def hid_task_matches(h, tn):
    # hid = f"ct-{tn}-{agent}?"; match both task and agent-agnostic
    return h == f"ct-{tn}" or h.startswith(f"ct-{tn}-")


def safe_auc(a):
    return round(a, 3) if a is not None else None


def loo_stack(Xs, yv, keys):
    sc = np.full(len(yv), np.nan, np.float32)
    for k in sorted(set(keys)):
        te = [i for i, x in enumerate(keys) if x == k]
        tr = [i for i, x in enumerate(keys) if x != k]
        if yv[tr].sum() < 5:
            sc[te] = 0.5
            continue
        mu, sd = Xs[tr].mean(0), np.where(Xs[tr].std(0) == 0, 1, Xs[tr].std(0))
        w = np.zeros(Xs.shape[1], np.float32)
        b = 0.0
        Xz = (Xs[tr] - mu) / sd
        for _ in range(2500):
            z = Xz @ w + b
            p = 1 / (1 + np.exp(-z))
            d = (p - yv[tr]) / len(tr)
            w -= 0.3 * (Xz.T @ d)
            b -= 0.3 * float(d.sum())
        sc[te] = 1 / (1 + np.exp(-((((Xs[te] - mu) / sd) @ w) + b)))
    return safe_auc(auroc(sc, yv))


if __name__ == "__main__":
    sys.exit(main())