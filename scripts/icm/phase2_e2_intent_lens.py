"""PHASE 2 E2 -- the intent-conditioned lens on INTENT-BEARING data
(the agent-intent corpus), per the lens-corpus guard. Module is now
importable: build_data() / train_fold() / run_cv() are shared with the
levers addendum (phase2_e2_intent_battery.py) -- math identical to the
run that produced addendum 11.

  Arms (per the plan, section 7 E2):
    bind(state, intent) -> q_t = BUNDLE(normalize(e(task_t) + e(plan_t)))
        (parameter-free; plan absent -> q_t = e(task_t); no session-state
         exists in the corpus -> bind(intent) only, recorded as a GAP)
    embed(prompt)       -> o1 = max(cos(task, act), cos(plan, act))
    memory              -> m_i = e(step action text[:300])
    usefulness          -> human-verified step utility INVERTED (faulty=0)
  Splits: 8-fold task-disjoint CV; val carve from the TRAIN side (1-in-5
    stride on the sorted train list) selects the early-stop checkpoint;
    every scored task is held out from both fitting and selection.
  Lens: residual -- score = bundle_cos + <tanh(z_t W), tanh(m_i W)>/D,
    W shared 384x128 sigma-0.01 init (W->0 = the untrained bundle), hinge
    on within-task (useful, faulty) pairs, L2, margin, max 600 ep, ckpt 25.
  Opponents (parameter-free, identical pooled pool): o1 prompt leg,
    o2 bundle cosine (the lens's own init geometry), o3 lexical, o4 loglen.
  Survival: beat ALL FOUR by > 0.02 -> mandatory composition audit; ties
    and losses are reported with direction, verdict OPEN, no dismissals.

  CORRECTION NOTE (2026-10-16): the first run of this instrument computed
    o1 as TASK-cosine alone (a patch silently no-op'd; the offsets line was
    accidentally correct only because task_ids is sorted). Both are FIXED
    here; `--opponents-only` re-measures the corrected o1..o4 without
    retraining. The lens-side numbers of addendum 11 are unaffected (they
    never touched o1's construction).

Run:
  python scripts/icm/phase2_e2_intent_lens.py                    (full, ~17 min)
  python scripts/icm/phase2_e2_intent_lens.py --opponents-only   (seconds)
  python scripts/icm/phase2_e2_intent_lens.py --smoke            (structure check)
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from battery_mode2_remediation import auc  # noqa: E402  (audited scorer)

MAPPED = os.path.join("data", "_scratch", "icm", "capture", "codetrace_mapped.jsonl")
FLIP_TABLE = os.path.join("data", "_scratch", "codetrace", "codetrace_flip_table.json")
ECACHE = os.path.join("data", "_scratch", "icm", "intent_lens_embeds.npz")
OUT = os.path.join("data", "_scratch", "icm", "phase2_e2_intent_lens_results.json")

SEED = 42
N_FOLDS = 8
EPOCHS = 600
CKPT = 25
MARGIN = 0.1
L2 = 1e-4
DIM = 128

FAULTY = ("incorrect", "unuseful")
_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def tokset(v):
    return {w for w in re.findall(r"[a-z0-9]+", str(v).lower()) if len(w) > 2}


def build_data(verbose=True):
    """Assembly + embeddings + per-task structures + opponents + gap report."""
    mapped = [json.loads(l) for l in io.open(MAPPED, encoding="utf-8") if l.strip()]
    goals, plans, steps = {}, collections.defaultdict(list), []
    for r in mapped:
        if r["level"] == "goal":
            goals[r["id"]] = r
        elif r["level"] == "plan":
            plans[r["parent"]].append(r["text"])
        elif r["level"] == "step":
            steps.append(r)

    task_ids = sorted({r["parent"] for r in steps})
    tasks = {}
    for hid in task_ids:
        sts = sorted((r for r in steps if r["parent"] == hid),
                     key=lambda r: int(r.get("step_id", 0)))
        tasks[hid] = {
            "goal": goals[hid]["text"] if hid in goals else "",
            "plan": plans.get(hid, [""])[0] or "",
            "steps": sts,
        }

    n_steps = sum(len(t["steps"]) for t in tasks.values())
    n_faulty = sum(1 for t in tasks.values() for s in t["steps"]
                   if s.get("human_label") in FAULTY)
    tasks_with_faulty = {h for h, t in tasks.items()
                         if any(s.get("human_label") in FAULTY for s in t["steps"])}
    echo_steps = sum(1 for t in tasks.values() for s in t["steps"]
                     if s["text"].lstrip().replace(" ", "").startswith('{"command":"plan"'))
    gaps = {
        "level_census": dict(collections.Counter(r["level"] for r in mapped)),
        "tasks_scored": len(tasks),
        "goals_total": len(goals),
        "goals_without_steps": len(goals) - len(tasks),
        "steps_total": n_steps,
        "verified_faulty": n_faulty,
        "label_coverage": round(n_faulty / n_steps, 4),
        "tasks_with_any_verified_faulty": len(tasks_with_faulty),
        "tasks_without_any": len(tasks) - len(tasks_with_faulty),
        "label_note": ("unverified steps are the ORDINARY class -- the bench "
                       "verifies faulty steps and marks nothing else, so 'ok' "
                       "labels do not exist; the contrast is verified-faulty "
                       "vs unverified-ordinary"),
        "plan_presence": {"tasks_with_declared_plan":
                          sum(1 for t in tasks.values() if t["plan"]),
                          "rate": round(sum(1 for t in tasks.values()
                                            if t["plan"]) / len(tasks), 3),
                          "note": ("no plan -> the intent bundle collapses to "
                                   "the task text alone (bind thins to state)")},
        "plan_echo_check": {"plan_echo_steps": echo_steps,
                            "note": ("steps that re-declare the plan are "
                                     "topically trivial; the PASS audit must "
                                     "check the lens does not feed on them")},
        "prompt_analog_note": ("embed(prompt) leg uses the TASK description as "
                               "the recall-time prompt -- the coarsest honest "
                               "analog; a finer query-time prompt trace does "
                               "not exist in this capture"),
        "state_operand_note": ("no session-state beyond trajectory position; "
                               "this run trains bind(intent) ONLY and claims "
                               "no state conditioning -- the plan's full "
                               "bind(state, intent) stays a GAP"),
        "memory_side_note": ("memory units are the agent's own step ACTIONS "
                             "from a trajectory, not independently-retrieved "
                             "records; the usefulness oracle is human "
                             "step-utility INVERTED -- pay-off labels for "
                             "retrieved memory proper remain scarce "
                             "(organic likes: 11)"),
        "flip_shape": {"shared_file_units": None,
                       "note": ("repo-shared file units exist (the flip table) "
                                "but utility labels on them are thin (16 "
                                "negatives) -- not yet trainable at "
                                "utility-label power")},
        "outcome_axis_note": ("547 rule-computed trajectory outcomes exist but "
                              "add no step-level utility signal this run -- a "
                              "second, coarser label axis left unused"),
        "parse_coverage_note": ("~58 percent of trajectory tars are unparsed "
                                "(mini-swe / terminus layouts); the parsed "
                                "universe is openhands-flavor"),
    }
    try:
        ft = json.load(io.open(FLIP_TABLE, encoding="utf-8"))
        gaps["flip_shape"]["shared_file_units"] = len(ft) if isinstance(ft, list) \
            else len(ft.get("flips", ft))
    except Exception:
        gaps["flip_shape"]["shared_file_units"] = "file missing"

    # ---------------------------------------------------- embeddings (cached)
    atexts = [s["text"][:300] for h in task_ids for s in tasks[h]["steps"]]
    gt = [tasks[h]["goal"] for h in task_ids]
    pt = [tasks[h]["plan"] for h in task_ids]          # "" when absent
    plan_present = np.array([bool(x) for x in pt])
    ekey = json.dumps([len(atexts), gt[0][:50], atexts[0][:50]])
    avec = gvec = pvec = None
    if os.path.exists(ECACHE):
        z = np.load(ECACHE, allow_pickle=True)
        if str(z["ekey"]) == ekey:
            avec, gvec, pvec = z["avec"], z["gvec"], z["pvec"].copy()
            pvec[~plan_present] = 0.0
            if verbose:
                print("[cache] embeds loaded", flush=True)
    if avec is None:
        if verbose:
            print(f"[embed] {len(atexts)} actions + {len(gt)} goals + {len(pt)} plans",
                  flush=True)
        avec, gvec, pvec = _emb(atexts), _emb(gt), _emb(pt)
        pvec[~plan_present] = 0.0
        os.makedirs(os.path.dirname(ECACHE), exist_ok=True)
        np.savez(ECACHE, ekey=ekey, avec=avec, gvec=gvec, pvec=pvec)

    # ------------------------------------------------- per-task structures
    D = {}
    offs, acc = {}, 0
    for h in task_ids:                      # contiguous slices in sorted order
        offs[h] = acc
        acc += len(tasks[h]["steps"])
    for ti, h in enumerate(task_ids):
        t = tasks[h]
        q_raw = gvec[ti] + pvec[ti]         # bundle; p=0 row -> e(task)
        q_raw = q_raw / max(1e-9, float(np.linalg.norm(q_raw)))
        lo, hi = offs[h], offs[h] + len(t["steps"])
        ms = avec[lo:hi]
        D[h] = {"q": q_raw, "M": ms,
                "faulty": {i for i, s in enumerate(t["steps"])
                           if s.get("human_label") in FAULTY},
                "bundle": (q_raw @ ms.T).tolist(),
                "task_cos": (gvec[ti] @ ms.T).tolist(),
                "plan_cos": (pvec[ti] @ ms.T).tolist(),
                "jacc": [], "loglen": []}
        ct = tokset(t["goal"] + " " + t["plan"])
        for i, s in enumerate(t["steps"]):
            a = tokset(s["text"])
            u = ct | a
            D[h]["jacc"].append(len(ct & a) / len(u) if u else 0.0)
            D[h]["loglen"].append(math.log1p(len(s["text"])))

    # ------------------------------------------------- opponents (fixed pool)
    opp_pairs = {"o1": [], "o2": [], "o3": [], "o4": []}
    for h in task_ids:
        d = D[h]
        for i in range(len(d["M"])):
            y = 0.0 if i in d["faulty"] else 1.0
            opp_pairs["o1"].append((y, max(d["task_cos"][i], d["plan_cos"][i])))
            opp_pairs["o2"].append((y, d["bundle"][i]))
            opp_pairs["o3"].append((y, d["jacc"][i]))
            opp_pairs["o4"].append((y, d["loglen"][i]))
    opp_auc = {k: auc(v) for k, v in opp_pairs.items()}
    if verbose:
        print(f"[data] {len(tasks)} tasks / {n_steps} steps / {n_faulty} "
              f"verified-faulty / {len(tasks_with_faulty)} tasks with faulty", flush=True)
        print(f"[opp] " + " ".join(f"{k}={None if v is None else round(v, 4)}"
                                   for k, v in opp_auc.items()), flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(task_ids))
    fold_sets = [[task_ids[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]
    return {"task_ids": task_ids, "tasks": tasks, "D": D, "fold_sets": fold_sets,
            "opp_pairs": opp_pairs, "opp_auc": opp_auc, "gaps": gaps,
            "n_steps": n_steps, "n_faulty": n_faulty}


# ------------------------------------------------------------------ the lens

def lens_step_scores(h, W, D):
    d = D[h]
    if W is None:
        return list(d["bundle"])
    gq = np.tanh(d["q"] @ W)
    proj = np.tanh(d["M"] @ W) @ gq / DIM
    return list(d["bundle"] + np.asarray(proj))


def pooled_auc(task_list, W, D):
    pairs = []
    for h in task_list:
        for y, s in zip((0.0 if i in D[h]["faulty"] else 1.0
                         for i in range(len(D[h]["M"]))), lens_step_scores(h, W, D)):
            pairs.append((y, s))
    return auc(pairs)


def train_fold(cfg, tr_tasks, val_tasks, D, log_tag, epochs=None, ckpt=None):
    """Hinge trainer on the residual lens (math verbatim from the addendum-11 run)."""
    lr = cfg["lr"]
    margin = cfg.get("margin", MARGIN)
    l2 = cfg.get("l2", L2)
    dim = cfg.get("dim", DIM)
    epochs = epochs or EPOCHS
    ckpt = ckpt or CKPT
    rng = np.random.default_rng(cfg.get("seed", SEED))
    W = (0.01 * rng.normal(0.0, 1.0, size=(384, dim))).astype(np.float32)
    best_val, best_W, best_ep, exit_streak = -1.0, None, -1, 0
    t0 = time.time()
    vi = 0
    for ep in range(epochs):
        GW = np.zeros_like(W)
        loss = 0.0
        n_pairs = 0
        for h in tr_tasks:
            d = D[h]
            if not d["faulty"]:
                continue
            xq, M = d["q"], d["M"]
            gq = np.tanh(xq @ W)
            G = np.tanh(M @ W)
            t_scores = G @ gq / dim + np.asarray(d["bundle"])
            A = np.zeros(len(t_scores), dtype=np.float64)
            fl = d["faulty"]
            for f in fl:
                viol = (margin - t_scores + t_scores[f]) > 0
                viol[list(fl)] = False
                c = int(viol.sum())
                if c == 0:
                    continue
                loss += float((margin - t_scores[viol] + t_scores[f]).sum())
                n_pairs += c
                dgq = (viol @ G - c * G[f]) / dim
                GW += np.outer(xq, dgq * (1 - gq ** 2))
                A[viol] -= 1.0
                A[f] += float(c)
            for i in np.nonzero(A)[0]:
                GW += np.outer(M[i], A[i] * gq / dim * (1 - G[i] ** 2))
        GW += l2 * W
        W -= lr * GW / max(1, len(tr_tasks))
        if (ep + 1) % ckpt == 0 or ep == epochs - 1:
            v = pooled_auc(val_tasks, W, D) if val_tasks else None
            if v is not None and v > best_val:
                best_val, best_W, best_ep = v, W.copy(), ep + 1
            if n_pairs and loss / n_pairs < 1e-6:
                exit_streak += 1
                if exit_streak >= 4:
                    break
            else:
                exit_streak = 0
            vi = n_pairs
    print(f"[{log_tag}] ep={ep + 1} pairs={vi} val="
          f"{None if best_val < 0 else round(best_val, 3)} ({time.time() - t0:.0f}s)",
          flush=True)
    return {"W": best_W if best_W is not None else W, "val_best": best_val,
            "best_epoch": best_ep, "early_exit": exit_streak >= 4,
            "n_pairs_last": vi}


def run_cv(cfg, data, tag, trainer=None, epochs=None, ckpt=None):
    """8-fold task-disjoint CV: fit per fold, score each fold's tasks with
    that fold's checkpoint-selected W. Returns pooled pairs + per-fold AUCs."""
    D, task_ids, fold_sets = data["D"], data["task_ids"], data["fold_sets"]
    trainer = trainer or train_fold
    W_of, fold_meta = {}, []
    t0 = time.time()
    for k in range(N_FOLDS):
        hold = fold_sets[k]
        trainval = [h for h in task_ids if h not in set(hold)]
        val_t = set(sorted(trainval)[::5])
        tr_t = [h for h in trainval if h not in val_t]
        tr = trainer(cfg, tr_t, sorted(val_t), D, f"{tag}.f{k}",
                     epochs=epochs, ckpt=ckpt)
        W_of[k] = tr["W"]
        fold_meta.append({"fold": k, "best_epoch": tr["best_epoch"],
                          "val_best": None if tr["val_best"] < 0 else round(tr["val_best"], 4),
                          "early_exit": tr["early_exit"],
                          "n_train": len(tr_t), "n_val": len(val_t),
                          "n_hold": len(hold)})
    pooled, per_fold = [], []
    for k in range(N_FOLDS):
        pp = []
        for h in fold_sets[k]:
            for y, s in zip((0.0 if i in D[h]["faulty"] else 1.0
                             for i in range(len(D[h]["M"]))),
                            lens_step_scores(h, W_of[k], D)):
                pp.append((y, s))
        pooled.extend(pp)
        fa = auc(pp)
        if fa is not None:
            per_fold.append(round(fa, 4))
    pa = auc(pooled)
    opp = data["opp_auc"]
    survived = (pa is not None and all(
        pa > (opp[o] + 0.02) for o in opp if opp[o] is not None))
    return {"pooled_auc": None if pa is None else round(pa, 4),
            "per_fold_auc": sorted(per_fold), "beats_all_opponents": bool(survived),
            "fold_meta": fold_meta, "train_s": round(time.time() - t0),
            "pairs": pooled, "W_of": W_of}


def spearman(a, b):
    n = len(a)
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    d2 = float(((ra - rb) ** 2).sum())
    return 1 - 6 * d2 / (n * (n * n - 1))


def composition_audit(cfg_res, data):
    """Spearman of the winning lens's score order vs the cheap legs."""
    D, fold_sets = data["D"], data["fold_sets"]
    lens_s, jacc_s, len_s = [], [], []
    for k in range(N_FOLDS):
        for h in fold_sets[k]:
            lens_s.extend(lens_step_scores(h, cfg_res["W_of"][k], D))
            jacc_s.extend(D[h]["jacc"])
            len_s.extend(D[h]["loglen"])
    return {"spearman_vs_jaccard": round(spearman(lens_s, jacc_s), 3),
            "spearman_vs_loglen": round(spearman(lens_s, len_s), 3),
            "faulty_head_context": {"stack_0.814": "loglen 0.754 workhorse"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponents-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    data = build_data()
    results = {"opponents": {k: None if v is None else round(v, 4)
                             for k, v in data["opp_auc"].items()},
               "gaps": data["gaps"], "configs": {}}
    if args.opponents_only:
        results["note"] = ("opponents re-measured after the o1/offsets fix; "
                           "lens configs not retrained (addendum-11 lens "
                           "numbers stand)")
        with io.open(OUT, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=2))
        print(json.dumps(results["opponents"], indent=1))
        return 0

    epochs = 60 if args.smoke else None
    ckpt = 20 if args.smoke else None
    configs = [{"lr": 1.5}, {"lr": 0.15}]
    if args.smoke:
        configs = configs[:1]
    for cfg in configs:
        tag = f"lens_lr{cfg['lr']:g}"
        res = run_cv(cfg, data, tag, epochs=epochs, ckpt=ckpt)
        W_of = res.pop("W_of")
        res.pop("pairs", None)
        results["configs"][tag] = res
        print(f"[{tag}] pooled={res['pooled_auc']} BEATS_ALL={res['beats_all_opponents']}",
              flush=True)
        if res["beats_all_opponents"]:
            results["audit"] = composition_audit({**res, "W_of": W_of}, data)
    if "audit" not in results:
        results["audit"] = None
    results["pre_registered_bar"] = ("lens pooled AUC > ALL opponents + 0.02; "
                                     "PASS audit mandatory; ties open")
    results["survivors"] = [k for k, c in results["configs"].items()
                            if c["beats_all_opponents"]]
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
    print(f"[done] survivors={results['survivors']} -> {OUT} "
          f"({time.time() - t_start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
