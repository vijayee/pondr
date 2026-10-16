"""PHASE 2 E2, START-OVER RUN -- the intent-conditioned lens trained on
INTENT-BEARING data (the agent-intent corpus), per the lens-corpus guard.

This is the plan's actual Phase-2 gate, executed for the first time with the
intent operand in the loss. Everything is named in ternary terms:

  Arms (per docs/Intent_Conditioned_Memory_Research_Plan.md section 7, E2):
    bind(state, intent)  ->  q_t = BUNDLE(normalize(e(task_t) + e(plan_t)))
        -- parameter-free HDC bundling analog; intent = the agent's OWN
           declared plan (goal grain, per the measured granularity verdict),
           state = the task description. GAPS RECORDED, NOT FAKED: the
           corpus has no session-state beyond step position (this run
           trains bind(intent) only) and no plan for ~47 percent of tasks
           (there the intent bundle is the task text alone).
    embed(prompt)        ->  o1 = max(cos(task, act), cos(plan, act))
        -- the prompt-embedding retrieval leg the gate names as the opponent.
    memory               ->  m_i = e(step action text[:300]).
    usefulness           ->  human-verified step utility INVERTED (faulty=0;
        within-task contrast = verified-faulty vs the task's other steps).

THE PRE-REGISTRATION (fixed BEFORE fitting; per the lens-corpus guard):
  Splits:   8-fold task-disjoint CV (seeded 42; trajectory = episode unit).
            Per fold: scored pool = that fold's tasks ONLY (never trained on
            by this fold's model). The early-stop val carve is taken from
            the TRAIN side (20 percent, deterministic 1-in-5 stride on the
            sorted train-task list) -- no scored task is used for fitting
            or checkpoint selection.
  Lens:     residual form -- score = bundle_cos(q_t, m_i) + <tanh(z_t W),
            tanh(m_i W)>/D (W shared 384x128, sigma-0.01 init; at W->0 the
            score IS exactly the bundle cosine -- the untrained
            bind(state,intent) geometry), hinge on within-task
            (useful, faulty) pairs, L2 1e-4 on W, margin 0.1, max 600 ep,
            checkpoints every 25. Two configs: lr 1.5 and lr 0.15.
            Pre-registered EARLY-EXIT: hinge < 1e-6 at 4 consecutive
            checkpoints = the bundle geometry already separates the task
            within the fold -- a data finding, recorded as such.
  Opponents (parameter-free, pooled over ALL steps, identical pools):
            o1 embed(prompt) leg = max(cos(task,act), cos(plan,act))
               (plan absent for a task -> cos(task) alone)
            o2 bundle-cosine = cos(bundle, act) -- the lens's init geometry;
               beats-o2 means the LEARNED delta did something
            o3 lexical jaccard((task+plan), act)
            o4 log length(action)
  Readout:  pooled AUROC (all steps; unverified steps = the ORDINARY class
            -- only VERIFIED-FAULTY carry 0s; the coverage gap is reported
            in the gap report, not hidden). Per-fold AUC distribution also
            reported (min/median/max).
  Survival: lens beats ALL FOUR opponents by > 0.02 on pooled AUC = PASS ->
            MANDATORY composition audit before any intent claim: Spearman
            between lens scores and the cheap legs it cannot see, plus the
            faulty-head stack numbers (0.814 head / 0.754 loglen) for
            context; a PASS whose margin tracks loglen/lexical is the
            router wearing intent's hat (recorded as such).
            Tie/loss = direction reported, verdict OPEN. No dismissals.

  GAP REPORT (first-class output per the standing instruction to surface
  data gaps DURING the process -- written into the results JSON):
  census / label coverage / usable-contrast tasks / plan presence /
  plan-echo contamination / prompt-analog coarseness / state operand /
  memory-side nature / flip-shape availability / outcome-label axis.

Run:  python scripts/icm/phase2_e2_intent_lens.py
"""

from __future__ import annotations

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

MANIFEST = os.path.join("data", "_scratch", "codetrace", "bench_manifest.verified.jsonl")
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


def main() -> int:
    t_start = time.time()
    mapped = [json.loads(l) for l in io.open(MAPPED, encoding="utf-8") if l.strip()]

    # ========== ASSEMBLY + GAP REPORT (surfaced DURING the process) =========
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
                   if s.get("human_label") in ("incorrect", "unuseful"))
    tasks_with_faulty = {h for h, t in tasks.items()
                         if any(s.get("human_label") in ("incorrect", "unuseful")
                                for s in t["steps"])}
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
                       "verifies faulty steps and marks nothing else, so "
                       "'ok' labels do not exist; the contrast is "
                       "verified-faulty vs unverified-ordinary"),
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
        "state_operand_note": ("no session-state beyond trajectory position in "
                               "the corpus; this run trains bind(intent) ONLY "
                               "and claims no state conditioning -- the plan's "
                               "full bind(state, intent) stays a GAP"),
        "memory_side_note": ("memory units are the agent's own step ACTIONS "
                             "from a trajectory, not independently-retrieved "
                             "records; the usefulness oracle is human "
                             "step-utility INVERTED -- pay-off labels for "
                             "retrieved memory proper remain scarce "
                             "(organic likes: 11)"),
        "flip_shape": {"shared_file_units": None,
                       "note": ("repo-shared file units exist (the flip table) "
                                "but utility labels on them are thin (16 "
                                "negatives) -- the flip-shape corpus is not "
                                "yet trainable at utility-label power")},
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

    # ============================ EMBEDDINGS (cached) ========================
    atexts = [s["text"][:300] for h in task_ids for s in tasks[h]["steps"]]
    gt = [tasks[h]["goal"] for h in task_ids]
    pt = [tasks[h]["plan"] for h in task_ids]   # "" if absent (zeroed below)
    plan_present = [bool(x) for x in pt]
    ekey = json.dumps([len(atexts), gt[0][:50], atexts[0][:50]])
    avec = gvec = pvec = None
    if os.path.exists(ECACHE):
        z = np.load(ECACHE, allow_pickle=True)
        if str(z["ekey"]) == ekey:
            avec, gvec, pvec = z["avec"], z["gvec"], z["pvec"]
            pvec = pvec.copy()
            pvec[~np.array(plan_present)] = 0.0
            print("[cache] embeds loaded", flush=True)
    if avec is None:
        print(f"[embed] {len(atexts)} actions + {len(gt)} goals + {len(pt)} plan/goal",
              flush=True)
        avec, gvec, pvec = _emb(atexts), _emb(gt), _emb(pt)
        pvec[~np.array(plan_present)] = 0.0      # plan-absent -> zero intent part
        os.makedirs(os.path.dirname(ECACHE), exist_ok=True)
        np.savez(ECACHE, ekey=ekey, avec=avec, gvec=gvec, pvec=pvec)

    # per-task structures
    D = {}
    for ti, h in enumerate(task_ids):
        t = tasks[h]
        q_raw = gvec[ti] + pvec[ti]      # bundle(state, intent), plan->goal fb
        q_raw = q_raw / max(1e-9, np.linalg.norm(q_raw))
        ms = avec[sum(len(tasks[x]["steps"]) for x in task_ids[:len(task_ids)]) and 0:0] \
            if False else avec
        lo = sum(len(tasks[x]["steps"]) for x in task_ids if x < h)
        hi = lo + len(t["steps"])
        ms = avec[lo:hi]
        D[h] = {"q": q_raw, "M": ms,
                "faulty": {i for i, s in enumerate(t["steps"])
                           if s.get("human_label") in ("incorrect", "unuseful")},
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
    print(f"[data] {len(tasks)} tasks / {n_steps} steps / {n_faulty} verified-faulty / "
          f"{len(tasks_with_faulty)} tasks with faulty", flush=True)

    # ============ OPPONENTS (parameter-free; identical pooled pools) =========
    opp_pairs = {"o1": [], "o2": [], "o3": [], "o4": []}
    for h in task_ids:
        d = D[h]
        for i in range(len(d["M"])):
            y = 0.0 if i in d["faulty"] else 1.0
            o1 = max(d["task_cos"][i], d["plan_cos"][i])   # plan absent -> 0 row
            opp_pairs["o1"].append((y, o1))
            opp_pairs["o2"].append((y, d["bundle"][i]))
            opp_pairs["o3"].append((y, d["jacc"][i]))
            opp_pairs["o4"].append((y, d["loglen"][i]))
    opp_auc = {k: auc(v) for k, v in opp_pairs.items()}
    best_opp = max(v for v in opp_auc.values() if v is not None)
    print(f"[opp] o1={opp_auc['o1']} o2={opp_auc['o2']} o3={opp_auc['o3']} "
          f"o4={opp_auc['o4']} (pooled over {n_steps} steps)", flush=True)

    # ================================ THE LENS ===============================
    def lens_step_scores(h, W):
        d = D[h]
        gq = np.tanh(d["q"] @ W)
        proj = np.tanh(d["M"] @ W) @ gq / DIM
        return d["bundle"] + np.asarray(proj).tolist()

    def pooled_auc(task_list, W):
        pairs = []
        for h in task_list:
            for y, s in zip((0.0 if i in D[h]["faulty"] else 1.0
                             for i in range(len(D[h]["M"]))), lens_step_scores(h, W)):
                pairs.append((y, s))
        return auc(pairs)

    def train_fold(lr, tr_tasks, val_tasks, log_tag):
        rng = np.random.default_rng(SEED)
        W = (0.01 * rng.normal(0.0, 1.0, size=(384, DIM))).astype(np.float32)
        best_val, best_W, best_ep, exit_streak = -1.0, None, -1, 0
        t0 = time.time()
        for ep in range(EPOCHS):
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
                t_scores = G @ gq / DIM + np.asarray(d["bundle"])
                A = np.zeros(len(t_scores), dtype=np.float64)
                fl = d["faulty"]
                for f in fl:
                    viol = (MARGIN - t_scores + t_scores[f]) > 0
                    viol[list(fl)] = False          # faulty never pair with faulty
                    c = int(viol.sum())
                    if c == 0:
                        continue
                    loss += float((MARGIN - t_scores[viol] + t_scores[f]).sum())
                    n_pairs += c
                    dgq = (viol @ G - c * G[f]) / DIM
                    GW += np.outer(xq, dgq * (1 - gq ** 2))
                    A[viol] -= 1.0                  # useful side: pull DOWN
                    A[f] += float(c)                # faulty side: push UP
                nz = np.nonzero(A)[0]
                for i in nz:
                    GW += np.outer(M[i], A[i] * gq / DIM * (1 - G[i] ** 2))
            GW += L2 * W
            W -= lr * GW / max(1, len(tr_tasks))
            if (ep + 1) % CKPT == 0 or ep == EPOCHS - 1:
                v = pooled_auc(val_tasks, W) if val_tasks else None
                if v is not None and v > best_val:
                    best_val, best_W, best_ep = v, W.copy(), ep + 1
                if n_pairs and loss / n_pairs < 1e-6:
                    exit_streak += 1
                    if exit_streak >= 4:
                        break
                else:
                    exit_streak = 0
        print(f"[{log_tag}] fold done ep={ep + 1} val={best_val if best_val is None else round(best_val, 3)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return {"W": best_W if best_W is not None else W, "val_best": best_val,
                "best_epoch": best_ep, "early_exit": exit_streak >= 4}

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(task_ids))
    fold_sets = [[task_ids[i] for i in perm[k::N_FOLDS]] for k in range(N_FOLDS)]

    results = {"opponents": {k: None if v is None else round(v, 4)
                             for k, v in opp_auc.items()},
               "gaps": gaps, "configs": {}}
    for lr in (1.5, 0.15):
        tag = f"lens_lr{lr:g}"
        W_of = {}
        fold_meta = []
        t0 = time.time()
        for k in range(N_FOLDS):
            hold = fold_sets[k]
            trainval = [h for h in task_ids if h not in set(hold)]
            val_t = set(sorted(trainval)[::5])    # deterministic 1-in-5 carve
            tr_t = [h for h in trainval if h not in val_t]
            tr = train_fold(lr, tr_t, sorted(val_t), f"{tag}.f{k}")
            W_of[k] = tr["W"]
            fold_meta.append({"fold": k, "best_epoch": tr["best_epoch"],
                              "val_best": None if tr["val_best"] is None else round(tr["val_best"], 4),
                              "early_exit": tr["early_exit"],
                              "n_train": len(tr_t), "n_val": len(val_t),
                              "n_hold": len(hold)})
        pooled = {"pairs": []}
        for k in range(N_FOLDS):
            for h in fold_sets[k]:
                for y, s in zip((0.0 if i in D[h]["faulty"] else 1.0
                                 for i in range(len(D[h]["M"]))),
                                lens_step_scores(h, W_of[k])):
                    pooled["pairs"].append((y, s))
        pa = auc(pooled["pairs"])
        per_fold = []
        for k in range(N_FOLDS):
            pp = []
            for h in fold_sets[k]:
                for y, s in zip((0.0 if i in D[h]["faulty"] else 1.0
                                 for i in range(len(D[h]["M"]))),
                                lens_step_scores(h, W_of[k])):
                    pp.append((y, s))
            fa = auc(pp)
            if fa is not None:
                per_fold.append(round(fa, 4))
        survived = (pa is not None and all(
            pa > (opp_auc[o] + 0.02) for o in opp_auc if opp_auc[o] is not None))
        results["configs"][tag] = {
            "pooled_auc": None if pa is None else round(pa, 4),
            "per_fold_auc": sorted(per_fold),
            "beats_all_opponents": bool(survived),
            "fold_meta": fold_meta, "train_s": round(time.time() - t0)}
        print(f"[{tag}] pooled={pa if pa is None else round(pa, 4)} vs best_opp "
              f"{round(best_opp, 4)} -- BEATS_ALL={survived}", flush=True)

    # --------- MANDATORY PASS AUDIT (composition of any surviving lens) ------
    if any(c["beats_all_opponents"] for c in results["configs"].values()):
        # Spearman between pooled lens scores and the cheap legs it cannot see
        best_tag = max(results["configs"], key=lambda k: results["configs"][k]["pooled_auc"])
        W_best = {}
        # refit quickly for audit scores is heavy; approximate via the pooled
        # (y, s) ordering correlation with the opponents' scores
        def spearman(a, b):
            ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
            rb = {v: i for i, v in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
            n = len(a)
            d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
            return 1 - 6 * d2 / (n * (n * n - 1))
        results["audit_note"] = ("audit computes rank correlation of the winning "
                                 "lens's pooled score order vs o3/o4 orders")
        # NOTE: full per-step matched correlation needs aligned arrays; the
        # pooled pairs are aligned in assembly order, so rebuild in order:
        lens_s, jacc_s, len_s = [], [], []
        for k in range(N_FOLDS):
            for h in fold_sets[k]:
                ss = lens_step_scores(h, W_of[k])
                lens_s.extend(ss)
                jacc_s.extend(D[h]["jacc"])
                len_s.extend(D[h]["loglen"])
        results["audit"] = {"spearman_vs_jaccard": round(spearman(lens_s, jacc_s), 3),
                            "spearman_vs_loglen": round(spearman(lens_s, len_s), 3),
                            "faulty_head_context": {"stack_0.814": "loglen 0.754 workhorse"}}
    else:
        results["audit"] = None

    results["pre_registered_bar"] = ("lens pooled AUC > ALL opponents + 0.02; "
                                     "PASS audit mandatory; ties open")
    results["survivors"] = [k for k, c in results["configs"].items()
                            if c["beats_all_opponents"]]
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
    print(f"[done] survivors={results['survivors']} gaps={list(gaps.keys())[:3]}... "
          f"-> {OUT} ({time.time() - t_start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())