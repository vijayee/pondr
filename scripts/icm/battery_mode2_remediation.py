"""PHASE 2 E2 -- THE STANDARD ML REMEDIATION BATTERY on the mode-2 failure
(the composed-mixture lens that LEARNED and made OOS WORSE).

MODE-2 MEASURED STATE (fb72e73 / phase2_e2_v2_results.json):
  from-scratch tanh projection (D=128), hinge m=0.1, L2 1e-4, 1200 epochs:
    mean-arm OOS AUC  bound 0.565  vs  raw cosine 0.824
    tail-arm usefulness relay: 1/2 wins, n=8, p=1.0 (bound 0.134 vs cosine 0.158)
  Mechanism: the hinge pushes paid->question GLOBALLY and collapses the topic
  geometry (trained geometry worse than the frozen one it started from).

THE PRE-REGISTRATION (fixed BEFORE fitting; per the arc's standing laws):

  Data:     phase2_e2_fullmixture.build_rows() verbatim -- composed mixture:
            own-haystack golds (paid) + difflib-near-miss distractor golds
            (unpaid). NO fabricated rows.
  Splits:   seeded, TASK-disjoint (by question, the tuple unit):
            - the baseline's exact split: SEED=42 permutation, 80% train /
              20% holdout (identical rows -> comparable to the 0.565 baseline)
            - a VAL carve INSIDE train: seed 43, 80/20 by question
              (hyperparameter selection lives on VAL only; holdout is read
              at the val-selected checkpoint and at final, direction always).
  Readout per config:
            - holdout mean-arm AUC (bound) vs this split's raw-cosine AUC
            - tail-arm identification: holdout questions whose cosine argmax
              is unpaid ("tail" set); recovery = fraction whose bound argmax
              IS paid (cosine's own rate there is 0 by construction)
            - delta norm (a survivor whose learned delta is a no-op is
              labeled NO-OP and does not count as a remediation)
  Survival bar (ALL of):
            (1) holdout mean-arm bound AUC >= cosine holdout AUC - 0.01
            (2) tail recovery >= 0.5 of tail questions
            (3) delta norm > 1e-3 (not a no-op)
    -> any survivor gets the E2 usefulness relay (held-out tail questions,
       both arms, graded answers, paired binomial) BEFORE any claim.
  If NOTHING survives: verdict stays OPEN -- this sweep covers 4 remediation
  families; unexhausted ones (curriculum, more organic data, other inits,
  two-stage rankers) stay named, none dismissed as impossible.

  Configs (each toggles ONE remediation off the audited baseline math):
    A   ANCHOR   : original exactly (random init 0.02, D=128, proj-only
                   hinge 0.1, L2 1e-4, 1200 ep) -- must reproduce ~0.565.
    B1  RESIDUAL : item-1 in additive form -- bound = cos_raw + proj delta,
                   W ZERO-init D=128 (the score starts EXACTLY at the base
                   geometry and can only add corrections), hinge on the
                   TOTAL score, L2 on W in {1e-4, 1e-3}.
    B2  IDENTITY : item-1 in multiplicative form -- W0 = I_384 (D=384;
                   at init the projection is a linear pass-through, so the
                   bound ranking IS the raw cosine's), proj-only hinge (the
                   baseline's loss, unchanged), L2 on (W - W0), max 600 ep.
    C   EARLYSTOP/WD : original parametrization unchanged, WD in
                   {1e-4, 1e-3, 1e-2}, checkpoints every 50 ep.
    D   INFONCE  : residual form + in-batch negatives (K=12 candidates
                   sampled per question per epoch from OTHER train
                   questions -- the real distribution, no composed
                   distractor conflicts), temperature in {0.05, 0.1, 0.3}.
    E   MARGIN   : residual hinge at margins {0.02, 0.05, 0.3}, L2 1e-4.

  Exactness: the vectorized gradient collapses the per-pair outer-product
  sum into per-candidate outer products (identical sum, exact algebra);
  a unit check asserts vectorized == the original triple loop for one
  epoch on a small slice BEFORE any training.

Run:
  python scripts/icm/battery_mode2_remediation.py --smoke          (fast)
  python scripts/icm/battery_mode2_remediation.py --only A         (anchor)
  python scripts/icm/battery_mode2_remediation.py                  (full)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from phase2_e2_fullmixture import build_rows  # noqa: E402  (verbatim data)

CACHE = os.path.join("data", "_scratch", "icm", "battery_mode2_embeds.npz")
OUT = os.path.join("data", "_scratch", "icm", "battery_mode2_results.json")

SEED = 42            # the baseline's exact split seed
VAL_SEED = 43
TRAIN_FRAC = 0.8
VAL_FRAC = 0.2

EP_HINGE = 1200      # the baseline's epoch count
EP_IDENTITY = 600    # B2 compute bound (D=384); checkpoints cover the window
CKPT_EVERY = 50
LR = 1.5             # baseline lr
K_INBATCH = 12

_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def ensure_embeds(rows):
    """Embed questions + candidate texts once (npz-cached on exact texts)."""
    ctexts = []
    for r in rows:
        for c in r["cands"]:
            ctexts.append(c["text"])
    texts_q = [r["question"] for r in rows]
    i_ = 0
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["ctexts"]) == ctexts and list(z["qtexts"]) == texts_q:
            cz, qz = z["cz"], z["qz"]
            for ri, r in enumerate(rows):
                for c in r["cands"]:
                    c["_z"] = cz[i_]
                    c["cos"] = round(float(np.dot(cz[i_], qz[ri])), 4)
                    i_ += 1
            return qz
    print(f"[embed] encoding {len(ctexts)} cands + {len(texts_q)} questions", flush=True)
    t0 = time.time()
    cz = _emb(ctexts)
    qz = _emb(texts_q)
    for ri, r in enumerate(rows):
        for c in r["cands"]:
            c["_z"] = cz[i_]
            c["cos"] = round(float(np.dot(cz[i_], qz[ri])), 4)
            i_ += 1
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, ctexts=np.array(ctexts, dtype=object),
             qtexts=np.array(texts_q, dtype=object), cz=cz, qz=qz)
    print(f"[embed] done ({time.time()-t0:.1f}s)", flush=True)
    return qz


def auc(pairs):
    pos = [s for y, s in pairs if y]
    neg = [s for y, s in pairs if not y]
    if not pos or not neg:
        return None
    w = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
        1 for p in pos for n in neg if p == n)
    return w / (len(pos) * len(neg))


def score(xq, zc, cos_raw, Wd):
    """bound score under the config's parametrization; Wd None -> raw cosine."""
    if Wd is None:
        return cos_raw
    if Wd["form"] == "residual":
        return cos_raw + float(np.dot(np.tanh(xq @ Wd["W"]), np.tanh(zc @ Wd["W"]))) / Wd["dim"]
    return float(np.dot(np.tanh(xq @ Wd["W"]), np.tanh(zc @ Wd["W"]))) / Wd["dim"]


# ---------------------------------------------------------------- training --

def train_hinge(cfg, tr_rows, val_rows, qz, epochs, log_tag):
    """Hinge trainer: original (plain), residual, or identity parametrization."""
    form, dim = cfg["W"]["form"], cfg["W"]["dim"]
    margin, l2, lr = cfg["margin"], cfg["l2"], cfg.get("lr", LR)
    on_total = (form == "residual")     # residual's hinge sees raw+proj
    rng = np.random.default_rng(cfg.get("seed", SEED))
    if form == "identity":
        W = np.eye(384, dtype=np.float32)
        W0 = W.copy()
    elif form == "residual":
        W = np.zeros((384, dim), dtype=np.float32)
        W0 = np.zeros_like(W)
    else:                                # anchor: baseline's exact random init
        W = rng.normal(0, 0.02, size=(384, dim)).astype(np.float32)
        W0 = None

    # cache per-question structure once
    C = []
    for r in tr_rows:
        paid = [ci for ci, c in enumerate(r["cands"]) if c["paid"]]
        unpaid = [ci for ci, c in enumerate(r["cands"]) if not c["paid"]]
        C.append({
            "Xq": r["_qz"],
            "Xc": np.stack([c["_z"] for c in r["cands"]]),
            "raw": np.array([c["cos"] for c in r["cands"]], dtype=np.float64),
            "paid": paid, "unpaid": unpaid,
        })
    best_val, best_W, best_ep = -1.0, None, -1
    t0 = time.time()
    for ep in range(epochs):
        GW = np.zeros_like(W)
        loss = 0.0
        n_tr = 0
        for d in C:
            xq = d["Xq"]
            G = np.tanh(d["Xc"] @ W)          # (n_c, D)
            gq = np.tanh(xq @ W)              # (D,)
            s = G @ gq / dim
            t = s + d["raw"] if on_total else s
            paid, unpaid = d["paid"], d["unpaid"]
            if not paid or not unpaid:
                continue
            viol = margin - t[paid][:, None] + t[None, unpaid]
            M = viol > 0
            if not M.any():
                continue
            A = M.sum(axis=1).astype(np.float64)   # per paid: # violating unpaid
            B = M.sum(axis=0).astype(np.float64)   # per unpaid: # violating paid
            loss += float(viol[M].sum())
            n_tr += int(M.sum())
            # exact algebra == the original triple loop (unit-checked):
            Gu, Gp = G[unpaid], G[paid]
            dgq = (B @ Gu - A @ Gp) / dim
            dgp = -A * gq / dim                    # per paid
            dgu = +B * gq / dim                    # per unpaid
            GW += np.outer(xq, dgq * (1 - gq ** 2))
            for k, p in enumerate(paid):
                if A[k] == 0.0:
                    continue
                GW += np.outer(d["Xc"][p], dgp[k] * (1 - Gp[k] ** 2))
            for k, u in enumerate(unpaid):
                if B[k] == 0.0:
                    continue
                GW += np.outer(d["Xc"][u], dgu[k] * (1 - Gu[k] ** 2))
        GW += l2 * ((W - W0) if form == "identity" else W)
        W -= lr * GW / max(1, len(C))
        if (ep + 1) % CKPT_EVERY == 0 or ep == epochs - 1:
            v = val_auc(val_rows, qz, {"W": W, "form": form, "dim": dim})
            if v is not None and v > best_val:
                best_val, best_W, best_ep = v, W.copy(), ep + 1
            if (ep + 1) % (CKPT_EVERY * 4) == 0 or ep == epochs - 1:
                print(f"[{log_tag}] ep{ep+1} hinge={loss/max(1,n_tr):.4f} "
                      f"val={v if v is None else round(v,3)} best={round(best_val,3)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return {"W": best_W, "form": form, "dim": dim, "W0": W0,
            "val_best": best_val, "best_epoch": best_ep}


def train_infonce(cfg, tr_rows, val_rows, qz, epochs, log_tag):
    """Residual form + multi-positive InfoNCE with in-batch negatives."""
    dim, tau, l2, lr = cfg["dim"], cfg["tau"], cfg["l2"], cfg.get("lr", LR)
    K = cfg.get("k_inbatch", K_INBATCH)
    rng = np.random.default_rng(cfg.get("seed", SEED))
    W = np.zeros((384, dim), dtype=np.float32)
    all_cands = np.stack([c["_z"] for r in tr_rows for c in r["cands"]])
    n_all = all_cands.shape[0]
    best_val, best_W, best_ep = -1.0, None, -1
    t0 = time.time()
    for ep in range(epochs):
        GW = np.zeros_like(W)
        loss = 0.0
        n_q = 0
        for qi, r in enumerate(tr_rows):
            paid = [ci for ci, c in enumerate(r["cands"]) if c["paid"]]
            if not paid:
                continue
            Xs = np.stack([c["_z"] for c in r["cands"]])
            raws = np.array([c["cos"] for c in r["cands"]], dtype=np.float64)
            Xn = all_cands[rng.integers(0, n_all, size=K)]
            nr = Xn @ r["_qz"]
            Xall = np.concatenate([Xs, Xn], axis=0)
            raws = np.concatenate([raws, nr])
            ispaid = np.zeros(Xall.shape[0], dtype=bool)
            ispaid[paid] = True
            G = np.tanh(Xall @ W)
            gq = np.tanh(r["_qz"] @ W)
            s = (G @ gq) / dim + raws
            smax = s.max()
            z_all = np.exp((s - smax) / tau).sum()
            z_pos = np.exp((s[ispaid] - smax) / tau).sum()
            w_j = np.exp((s - smax) / tau) / z_all
            w_j[ispaid] -= np.exp((s[ispaid] - smax) / tau) / z_pos
            loss += float(np.log(z_all) - np.log(z_pos))
            n_q += 1
            dgq = (w_j @ G) / dim
            GW += np.outer(r["_qz"], dgq * (1 - gq ** 2))
            for j in range(Xall.shape[0]):
                if w_j[j] == 0.0:
                    continue
                GW += np.outer(Xall[j], (w_j[j] * gq / dim) * (1 - G[j] ** 2))
        W -= lr * GW / max(1, n_q)
        if (ep + 1) % CKPT_EVERY == 0 or ep == epochs - 1:
            v = val_auc(val_rows, qz, {"W": W, "form": "residual", "dim": dim})
            if v is not None and v > best_val:
                best_val, best_W, best_ep = v, W.copy(), ep + 1
            if (ep + 1) % (CKPT_EVERY * 4) == 0 or ep == epochs - 1:
                print(f"[{log_tag}] ep{ep+1} nce={loss/max(1,n_q):.4f} "
                      f"val={v if v is None else round(v,3)} best={round(best_val,3)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return {"W": best_W, "form": "residual", "dim": dim, "W0": np.zeros_like(W),
            "val_best": best_val, "best_epoch": best_ep}


def val_auc(val_rows, qz, Wd):
    pairs = []
    for ri, r in enumerate(val_rows):
        for c in r["cands"]:
            pairs.append((1.0 if c["paid"] else 0.0,
                          score(qz[ri], c["_z"], c["cos"], Wd)))
    return auc(pairs)


def holdout_readout(te_rows, qz, Wd, tag):
    pairs, cpairs, tail_rows = [], [], []
    for ri, r in enumerate(te_rows):
        best_c = max(r["cands"], key=lambda c: c["cos"])
        if not best_c["paid"]:
            tail_rows.append(ri)
        for c in r["cands"]:
            y = 1.0 if c["paid"] else 0.0
            pairs.append((y, score(qz[ri], c["_z"], c["cos"], Wd)))
            cpairs.append((y, c["cos"]))
    h_a = auc(pairs)
    c_a = auc(cpairs)
    rec = rec_total = 0
    for ri in tail_rows:
        r = te_rows[ri]
        arg_b = max(r["cands"], key=lambda c: score(qz[ri], c["_z"], c["cos"], Wd))
        rec_total += 1
        rec += 1 if arg_b["paid"] else 0
    print(f"[{tag}] bound {round(h_a, 3) if h_a else None} vs cosine "
          f"{round(c_a, 3)}; tail {rec}/{rec_total} recovered", flush=True)
    return {"holdout_bound_auc": None if h_a is None else round(h_a, 4),
            "holdout_cosine_auc": None if c_a is None else round(c_a, 4),
            "tail_questions": rec_total, "tail_recovered": rec,
            "tail_recovery_rate": round(rec / rec_total, 3) if rec_total else None}


def runs_unit_check(tr_rows, qvecs_tr):
    """Assert the collapsed vectorized epoch == the ORIGINAL triple-loop epoch
    (verbatim math from phase2_e2_fullmixture.train_W, proj-only loss)."""
    dim, margin, lr = 128, 0.1, LR
    dimf = 128.0
    W = np.random.default_rng(0).normal(0, 0.02, size=(384, dim))
    Wv, Wo = W.copy(), W.copy()
    Q = 30
    qvecs = qvecs_tr[:Q]

    # ---- vectorized: per-candidate collapsed sums ----
    GW = np.zeros_like(Wv)
    for qi, r in enumerate(tr_rows[:Q]):
        xq = qvecs[qi]
        Xs = np.stack([c["_z"] for c in r["cands"]])
        G = np.tanh(Xs @ Wv)
        gq = np.tanh(xq @ Wv)
        s = G @ gq / dimf
        paid = [ci for ci, c in enumerate(r["cands"]) if c["paid"]]
        unpaid = [ci for ci, c in enumerate(r["cands"]) if not c["paid"]]
        if not paid or not unpaid:
            continue
        viol = margin - s[paid][:, None] + s[None, unpaid]
        M = viol > 0
        if not M.any():
            continue
        A = M.sum(axis=1).astype(np.float64)
        B = M.sum(axis=0).astype(np.float64)
        Gu, Gp = G[unpaid], G[paid]
        dgq = (B @ Gu - A @ Gp) / dimf
        dgp = -A * gq / dimf
        dgu = +B * gq / dimf
        GW += np.outer(xq, dgq * (1 - gq ** 2))
        for k, p in enumerate(paid):
            if A[k] == 0.0:
                continue
            GW += np.outer(r["cands"][p]["_z"], dgp[k] * (1 - Gp[k] ** 2))
        for k, u in enumerate(unpaid):
            if B[k] == 0.0:
                continue
            GW += np.outer(r["cands"][u]["_z"], dgu[k] * (1 - Gu[k] ** 2))
    Wv -= lr * GW / Q

    # ---- ORIGINAL triple loop (verbatim from phase2_e2_fullmixture.train_W) ----
    GW = np.zeros_like(Wo)
    for qi, r in enumerate(tr_rows[:Q]):
        xq = qvecs[qi]
        gq = np.tanh(xq @ Wo)
        paid = [ci for ci, c in enumerate(r["cands"]) if c["paid"]]
        unpaid = [ci for ci, c in enumerate(r["cands"]) if not c["paid"]]
        if not paid or not unpaid:
            continue
        for cp in paid:
            gp = np.tanh(r["cands"][cp]["_z"] @ Wo)
            s_pos = float(np.dot(gq, gp)) / dimf
            for cu in unpaid:
                gu = np.tanh(r["cands"][cu]["_z"] @ Wo)
                s_neg = float(np.dot(gq, gu)) / dimf
                viol = margin - s_pos + s_neg
                if viol > 0:
                    # EXACT chain (as audited in fb72e73):
                    # dL/dg = dL/ds * ds/dg; dL/dW = outer(x, dL/dg * (1 - g^2))
                    dL_dgq = (gu - gp) / dimf
                    dL_dgp = -gq / dimf
                    dL_dgu = +gq / dimf
                    GW += np.outer(xq, dL_dgq * (1 - gq ** 2))
                    GW += np.outer(r["cands"][cp]["_z"], dL_dgp * (1 - gp ** 2))
                    GW += np.outer(r["cands"][cu]["_z"], dL_dgu * (1 - gu ** 2))
    Wo -= lr * GW / Q

    ok = np.allclose(Wv, Wo, atol=1e-9)
    print(f"[unitcheck] vectorized == original triple-loop: {ok} "
          f"(max diff {float(np.abs(Wv - Wo).max()):.2e})", flush=True)
    if not ok:
        raise SystemExit("unit check FAILED -- gradient mismatch, aborting")


# ------------------------------------------------------------- main / runs --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    rows = build_rows()
    qz = ensure_embeds(rows)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    n_tr = int(len(rows) * TRAIN_FRAC)
    tr_idx, te_idx = list(order[:n_tr]), list(order[n_tr:])
    rng2 = np.random.default_rng(VAL_SEED)
    vord = rng2.permutation(len(tr_idx))
    n_val = int(len(tr_idx) * VAL_FRAC)
    val_idx = [tr_idx[i] for i in vord[:n_val]]
    tr2_idx = [tr_idx[i] for i in vord[n_val:]]
    eps = {"hinge": EP_HINGE, "identity": EP_IDENTITY}
    if args.smoke:
        tr2_idx, val_idx, te_idx = tr2_idx[:24], val_idx[:8], te_idx[:8]
        eps = {"hinge": 40, "identity": 20}

    tr_rows = [rows[i] for i in tr2_idx]
    val_rows = [rows[i] for i in val_idx]
    te_rows = [rows[i] for i in te_idx]
    print(f"[split] train {len(tr_rows)} / val {len(val_rows)} / holdout {len(te_rows)}",
          flush=True)

    if not args.smoke:
        runs_unit_check(tr_rows, qz)

    raw_h = holdout_readout(te_rows, qz, None, "raw-cosine")
    results = {"raw_cosine_holdout": raw_h, "configs": {}}

    def train_cfg(tag, cfg, epochs):
        fam = tag.split("_")[0]
        if only and fam not in only:
            print(f"[skip] {tag}", flush=True)
            return None
        t0 = time.time()
        tr = (train_hinge if cfg["loss"] == "hinge" else train_infonce)(
            cfg, tr_rows, val_rows, qz, epochs, tag)
        Wd = {"W": tr["W"], "form": tr["form"], "dim": tr["dim"]}
        W0 = tr["W0"] if tr["W0"] is not None else np.zeros_like(tr["W"])
        dnorm = float(np.linalg.norm(tr["W"] - W0))
        if tr["form"] == "plain":
            Wd_final = Wd
        else:
            Wd_final = Wd
        ro = holdout_readout(te_rows, qz, Wd, f"{tag}@bestval(ep{tr['best_epoch']})")
        res = {"val_best_auc": round(tr["val_best"], 4), "best_epoch": tr["best_epoch"],
               "delta_norm": round(dnorm, 4), "holdout_bestval": ro,
               "form": tr["form"], "dim": tr["dim"], "l2": cfg["l2"],
               "margin": cfg.get("margin"), "tau": cfg.get("tau"),
               "epochs": epochs, "train_s": round(time.time() - t0)}
        cb, cc = ro["holdout_bound_auc"], ro["holdout_cosine_auc"]
        rate = ro["tail_recovered"] / ro["tail_questions"] if ro["tail_questions"] else 0.0
        surv = (cb is not None and cc is not None and cb >= cc - 0.01
                and rate >= 0.5 and dnorm > 1e-3)
        res["survives"] = bool(surv)
        print(f"[{tag}] SURVIVES={surv} (delta_norm={dnorm:.3f})", flush=True)
        results["configs"][tag] = res
        return res

    # A  anchor reproduction (original exactly)
    train_cfg("A_anchor", {"loss": "hinge", "W": {"form": "plain", "dim": 128},
                           "margin": 0.1, "l2": 1e-4, "seed": SEED}, eps["hinge"])

    # B1 residual-delta (item 1, additive; hinge on the TOTAL score)
    for l2 in (1e-4, 1e-3):
        train_cfg(f"B1_residual_L{l2:g}",
                  {"loss": "hinge", "W": {"form": "residual", "dim": 128},
                   "margin": 0.1, "l2": l2, "seed": SEED}, eps["hinge"])

    # B2 identity multiplicative (item 1, exact pass-through init, D=384)
    train_cfg("B2_identity",
              {"loss": "hinge", "W": {"form": "identity", "dim": 384},
               "margin": 0.1, "l2": 1e-4, "seed": SEED}, eps["identity"])

    # C early-stop/WD sweep (item 2) -- original parametrization, checkpoints
    for wd in (1e-4, 1e-3, 1e-2):
        train_cfg(f"C_earlystop_WD{wd:g}",
                  {"loss": "hinge", "W": {"form": "plain", "dim": 128},
                   "margin": 0.1, "l2": wd, "seed": SEED}, eps["hinge"])

    # D InfoNCE residual (item 3)
    for tau in (0.05, 0.1, 0.3):
        train_cfg(f"D_infonce_T{tau:g}",
                  {"loss": "infonce", "W": {"form": "residual", "dim": 128},
                   "tau": tau, "l2": 1e-4, "seed": SEED, "k_inbatch": K_INBATCH},
                  eps["hinge"])

    # E margin sweep (item 4) on the residual family
    for mg in (0.02, 0.05, 0.3):
        train_cfg(f"E_margin_{mg:g}",
                  {"loss": "hinge", "W": {"form": "residual", "dim": 128},
                   "margin": mg, "l2": 1e-4, "seed": SEED}, eps["hinge"])

    results["pre_registered_bar"] = ("holdout mean-arm bound AUC >= cosine-0.01 AND "
                                     "tail recovery >= 0.5 AND delta_norm > 1e-3")
    results["survivors"] = [k for k, v in results["configs"].items()
                            if v.get("survives")]
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
    print(f"[done] survivors={results['survivors']} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())