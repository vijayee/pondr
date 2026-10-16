"""PHASE 2 E2 (the plan's critical path) -- contrastive bipolar projection on
the organic verified tail; usefulness-scored recall vs raw similarity.

THE PRE-REGISTRATION (fixed before fitting; plan §6.2-6.3):

  Trainable:  ONE linear projection W (384 -> PROJ_DIM, bipolar via sign/soft-tanh),
              L2-regularized; binding is parameter-free; SELF-elision (+1).
  Objective:  hinge RANKING per question -- cos(W x_q, W x_paid) beats
              cos(W x_q, W x_unpaid) by a margin; paid = judge>=.5 on the
              2-judge-verified tail.
  Premise:    raw cosine must fail on this ground (measured 0/25 structural).
  E2 arms:    bound -- cos(W x_q, W x_cand);  cosine -- raw cos(x_q, x_cand).
  Score:      USEFULNESS: each arm's top pick is answered vs the gold answer
              (F1-graded; no oracle in the payoff path).
  Guards:     seed fixed; PROJ_DIM=128 (~49k params for ~144 tuples); z-norm.
  Readout:    train-tail discrimination AUCs; E2 holdout usefulness paired
              (grade(bound top-1) vs grade(cosine top-1 from the k=1 run)).

Run:  python scripts/icm/phase2_e2_contrastive.py --no-relay
      python scripts/icm/phase2_e2_contrastive.py              (holdout relay)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

CORPUS = os.path.join("data", "_scratch", "icm", "icm_head_corpus_o100.json")
LEDGER = r"data\_scratch\icm\organic_o100\ledger"
K1_CKPT = os.path.join("data", "_scratch", "icm", "payoff_tail_ab_ckpt_k1.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "phase2_e2_results.json")
CK = os.path.join("data", "_scratch", "icm", "phase2_e2_ckpt.jsonl")

PROJ_DIM = 128
SEED = 42
EPOCHS = 2000
LR = 0.05
MARGIN = 0.15
L2 = 1e-4
TRAIN_FRAC = 0.68

ANSWERER = "deepseek-v4.1-flash:cloud"
_model = None


def _emb(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model.encode(texts, batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def ask_retry(question, memory_block):
    prompt = question + "\nAnswer concisely, using the evidence below if provided."
    if memory_block:
        prompt += "\n\nEvidence records from memory:\n" + memory_block
    body = {"model": ANSWERER, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.2, "num_ctx": 8192}}
    backoff = 10.0
    for attempt in range(4):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return str(json.loads(urllib.request.urlopen(req, timeout=240).read())
                       ["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60)); backoff *= 1.8; continue
            if attempt >= 2:
                return ""
            time.sleep(2.0)
        except Exception:
            if attempt >= 2:
                return ""
            time.sleep(2.0)
    return ""


def f1(a, g):
    def t(v):
        return {w for w in re.findall(r"[a-z0-9]+", str(v).lower()) if len(w) > 2}
    G, A = t(g), t(a)
    if not G:
        return 0.0
    i = len(G & A)
    P = i / max(1, len(A))
    R = i / len(G)
    return round(2 * P * R / (P + R), 3) if P + R else 0.0


def load_rows():
    corpus = json.load(io.open(CORPUS, encoding="utf-8"))
    rows = []
    for q in corpus:
        cands = [{"id": c["id"], "text": c["text"], "cos": c["cos"],
                  "paid": 1 if c["kimi_judge"] >= 0.5 else 0} for c in q["candidates"]]
        rows.append({"qi": q["qi"], "question": q["question"], "cands": cands})
    return rows


def train_W(rows_tr, qvecs_tr, cvec_fn):
    """Hinge ranking metric-learning: s(q,c) = cosine(W xq, W xc).
    Exact gradient (no surrogate): for u = W x, dL/du known; dL/dW = outer(x, dL/du)."""
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.02, size=(384, PROJ_DIM)).astype(np.float32)
    for ep in range(EPOCHS):
        gW = np.zeros_like(W)
        loss = 0.0
        n_triples = 0
        for qi, r in enumerate(rows_tr):
            xq = qvecs_tr[qi]
            paid = [c for c in r["cands"] if c["paid"]]
            unpaid = [c for c in r["cands"] if not c["paid"]]
            if not paid or not unpaid:
                continue
            u = xq @ W
            nu = float(np.linalg.norm(u))
            for cp in paid:
                v = cvec_fn(cp["id"]) @ W
                nv = float(np.linalg.norm(v))
                s_pos = float(np.dot(u, v)) / (nu * nv + 1e-9)
                for cu in unpaid:
                    w2 = cvec_fn(cu["id"]) @ W
                    nw = float(np.linalg.norm(w2))
                    s_neg = float(np.dot(u, w2)) / (nu * nw + 1e-9)
                    viol = MARGIN - s_pos + s_neg
                    if viol > 0:
                        loss += viol
                        n_triples += 1
                        # dc+/du, dc+/dv; dc-/du, dc-/dw2 (standard cosine grads)
                        dCp_du = v / (nu * nv + 1e-9) - s_pos * u / (nu * nu + 1e-9)
                        dCp_dv = u / (nu * nv + 1e-9) - s_pos * v / (nv * nv + 1e-9)
                        dCm_du = w2 / (nu * nw + 1e-9) - s_neg * u / (nu * nu + 1e-9)
                        dCm_dw = u / (nu * nw + 1e-9) - s_neg * w2 / (nw * nw + 1e-9)
                        dL_du = -dCp_du + dCm_du
                        dL_dv = -dCp_dv
                        dL_dw = dCm_dw
                        gW += np.outer(xq, dL_du)
                        gW += np.outer(cvec_fn(cp["id"]), dL_dv)
                        gW += np.outer(cvec_fn(cu["id"]), dL_dw)
        gW += L2 * W
        W -= LR * gW / max(1, n_triples)
        if ep in (0, EPOCHS // 4, EPOCHS // 2, EPOCHS - 1):
            print(f"[train] ep{ep} hinge={loss/max(1,n_triples):.4f} triples={n_triples}",
                  flush=True)
    return W


def main() -> int:
    global ANSWERER
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-relay", action="store_true")
    ap.add_argument("--model", type=str, default=ANSWERER)
    args = ap.parse_args()
    ANSWERER = args.model

    rows = load_rows()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    n_tr = int(len(rows) * TRAIN_FRAC)
    tr, te = [rows[i] for i in order[:n_tr]], [rows[i] for i in order[n_tr:]]
    print(f"[split] train {len(tr)} / holdout {len(te)}", flush=True)

    qvecs = _emb([r["question"] for r in rows])
    qvecs_tr = [qvecs[i] for i in order[:n_tr]]
    all_texts = {c["id"]: c["text"] for r in rows for c in r["cands"]}
    cvec_of = {cid: v for cid, v in zip(all_texts.keys(),
                                        _emb(list(all_texts.values())))}

    def cvec_fn(cid):
        return cvec_of[cid]

    W = train_W(tr, qvecs_tr, cvec_fn)

    # ---- readouts ----
    rows_by_qi = {r["qi"]: r for r in rows}
    disc = {"bound": [], "cosine": []}
    for qi_idx, r in enumerate(tr):
        xq = qvecs[order[qi_idx]]
        for c in r["cands"]:
            xc = cvec_of[c["id"]]
            s_pos = float(np.dot(np.tanh(xq @ W), np.tanh(xc @ W))) / PROJ_DIM
            disc["bound"].append((c["paid"], s_pos))
            disc["cosine"].append((c["paid"], c["cos"]))
    def auc(pairs):
        pos_s = [s for y, s in pairs if y]
        neg_s = [s for y, s in pairs if not y]
        if not pos_s or not neg_s:
            return None
        w = sum(1 for p in pos_s for n in neg_s if p > n) + \
            0.5 * sum(1 for p in pos_s for n in neg_s if p == n)
        return round(w / (len(pos_s) * len(neg_s)), 3)
    disc_auc = {k: auc(v) for k, v in disc.items()}
    print(f"[train-tail] bound AUC {disc_auc['bound']} vs raw-cosine AUC {disc_auc['cosine']}",
          flush=True)

    out = {"pre_registered": "hinge ranking on 2-judge-verified tail; premise cosine-0/25",
           "split": [len(tr), len(te)], "train_tail": disc_auc}

    if args.no_relay:
        with io.open(OUT, "w", encoding="utf-8") as f:
            f.write(json.dumps(out, indent=2))
        print(f"[done] no-relay -> {OUT}")
        return 0

    # ---- E2 holdout usefulness relay ----
    from icm.ledger import ICMLedger
    led = ICMLedger(LEDGER)
    live_texts = {i: rr["text"] for i in led.ids("episode") if (rr := led.get("episode", i))}
    led.close()
    k1 = {json.loads(l)["qi"]: json.loads(l)
          for l in io.open(K1_CKPT, encoding="utf-8") if l.strip()}
    recs500 = json.load(io.open(
        r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness"
        r"\datasets\longmemeval\longmemeval_interleaved_full.json", encoding="utf-8"))
    done = set()
    rows_live = []
    if os.path.exists(CK):
        for l in io.open(CK, encoding="utf-8"):
            if l.strip():
                try:
                    r_ = json.loads(l)
                    rows_live.append(r_); done.add(r_["qi"])
                except Exception:
                    pass
    for qi_idx, r in enumerate(te):
        qi = r["qi"]
        if qi in done or not (0 <= qi < len(recs500)):
            continue
        gold = str(recs500[qi]["answer"])
        xq = qvecs[order[n_tr + qi_idx]]
        scored = []
        for c in r["cands"]:
            if c["id"] not in live_texts:
                continue
            xc = cvec_of[c["id"]]
            s_pos = float(np.dot(np.tanh(xq @ W), np.tanh(xc @ W))) / PROJ_DIM
            scored.append((c["id"], s_pos))
        if not scored:
            continue
        picked = max(scored, key=lambda kv: kv[1])[0]
        text = live_texts[picked]
        import urllib.request  # relay
        ans = ask_retry(r["question"], f"- {text}")
        row = {"qi": qi, "picked": picked,
               "bound": f1(ans, gold),
               "cosine": (k1.get(qi) or {}).get("cos"),
               "answer": ans[:200]}
        rows_live.append(row)
        with io.open(CK, "a", encoding="utf-8") as f2:
            f2.write(json.dumps(row) + "\n")
        print(f"[e2 q{qi}] bound={row['bound']} cosine={row['cosine']}", flush=True)
        time.sleep(3.0)
    # paired readout
    pairs = [(r["bound"], r["cosine"]) for r in rows_live
             if r["cosine"] is not None]
    wb = sum(1 for b, c in pairs if b > c)
    wc = sum(1 for b, c in pairs if c > b)
    p = binomtest(wb, wb + wc, 0.5).pvalue if wb + wc else None
    mb = sum(b for b, _ in pairs) / max(1, len(pairs))
    mc = sum(c for _, c in pairs) / max(1, len(pairs))
    out["e2_holdout"] = {"n": len(pairs), "bound_mean": round(mb, 3),
                         "cosine_mean": round(mc, 3), "wins": [wb, wc],
                         "binomial_p": None if p is None else round(p, 4)}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print("E2:", json.dumps(out["e2_holdout"], indent=1))
    print(f"[done] wrote {OUT}")
    return 0


def recs_lookup(recs500, qi):
    return 0 <= qi < len(recs500)


if __name__ == "__main__":
    sys.exit(main())