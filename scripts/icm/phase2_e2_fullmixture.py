"""PHASE 2 E2 attempt #2 -- train the bipolar projection on the FULL curated
gold mixture (~1,900 tuples, judge-free), verdict SPLIT by premise domain.

THE PRE-REGISTRATION:
  Train:  every harvested question's own-haystack sessions, labeled by the
          dataset's curated golds (answer_session_ids membership) — a THIRD
          label basis (human-authored golds, no judge, no valence mining).
          ~450 questions x ~4 sessions each ~= 1,900 tuples.
  Split:  by question, seeded (80/20 -> ~360 train / ~90 holdout).
  Arms:   bound = cosine(W zq, W zc) [soft-bipolar linear projection]
          cosine = raw cos(zq, zc).
  Premise: on the train MIXTURE, cosine partially succeeds (the null's mean
          domain). The verdict is therefore SPLIT:
            mean-arm    = OOS discrimination (reported; expected ~cosine+
            tail-arm    = the scientific question: on holdout questions
                          where the raw-cosine argmax is NOT the gold
                          session (cosine provably failing there), does the
                          BOUND argmax find the gold? — paired wins/binomial.
  Guards:  PROJ_DIM=128, L2, exact grads, fixed seed, no fabricated rows.
Run:
  python scripts/icm/phase2_e2_fullmixture.py --no-relay
  python scripts/icm/phase2_e2_fullmixture.py             (tail-arm usefulness relay)
"""

from __future__ import annotations

import argparse
import collections
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

FULL500 = (r"C:\Users\victor morrow\Git-projects\memory_and_context_eval_harness"
           r"\datasets\longmemeval\longmemeval_interleaved_full.json")
SEGMENTS = [(("data", "_scratch", "icm", "organic500"), 50),
            (("data", "_scratch", "icm", "organic_o100"), 100),
            (("data", "_scratch", "icm", "organic_o300"), 300)]
LEDGER_O100 = r"data\_scratch\icm\organic_o100\ledger"
K1 = os.path.join("data", "_scratch", "icm", "payoff_tail_ab_ckpt_k1.jsonl")
OUT = os.path.join("data", "_scratch", "icm", "phase2_e2_v2_results.json")
CK = os.path.join("data", "_scratch", "icm", "phase2_e2_v2_ckpt.jsonl")

PROJ_DIM = 128
SEED = 42
EPOCHS = 1200
LR = 1.5
MARGIN = 0.1
L2 = 1e-4
TRAIN_FRAC = 0.8

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


def build_rows():
    """Composed mixture (LongMemEval gives only all-gold haystacks): paid =
    the record's own curated sessions; unpaid = real gold sessions of the
    NEAR-MISS records (difflib-similar questions) — the same construction the
    benchmark's own designers used for weaved distractors, at training scale."""
    data = json.load(io.open(FULL500, encoding="utf-8"))
    import difflib
    rows = []
    for qi, rec in enumerate(data):
        gold = set(rec["answer_session_ids"])
        cands = []
        for si, sid in enumerate(rec["haystack_session_ids"]):
            turns = rec["haystack_sessions"][si]
            text = "\n".join(t.get("content", "") for t in turns[-10:])
            cands.append({"id": f"lsess-{qi}-{sid}", "text": text,
                          "paid": 1 if sid in gold else 0})
        # distractors: the 4 most question-similar OTHER records' gold sessions
        sims = sorted((o for o in range(len(data)) if o != qi),
                      key=lambda o: -difflib.SequenceMatcher(
                          None, rec["question"].lower(),
                          data[o]["question"].lower()).ratio())[:4]
        seen_ids = {c["id"] for c in cands}
        n_dist = 0
        for o in sims:
            orec = data[o]
            osids = [s for s in orec["haystack_session_ids"]
                     if s in set(orec["answer_session_ids"])]
            for osi, sid2 in enumerate(orec["haystack_session_ids"]):
                if sid2 in osids:
                    eid = f"lsess-{o}-{sid2}"
                    if eid in seen_ids:
                        continue
                    text2 = "\n".join(t.get("content", "")
                                      for t in orec["haystack_sessions"][osi][-10:])
                    cands.append({"id": eid, "text": text2, "paid": 0})
                    seen_ids.add(eid); n_dist += 1
                    break  # one gold session per distractor record
        if len(cands) < 4 or not any(c["paid"] for c in cands) \
                or n_dist < 2:
            continue
        rows.append({"qi": qi, "question": rec["question"], "cands": cands})
    return rows


def train_W(rows_tr, qvecs_tr):
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.02, size=(384, PROJ_DIM)).astype(np.float32)
    cvecs_tr = {}
    for r in rows_tr:
        for c in r["cands"]:
            cvecs_tr[c["id"]] = c["_z"]

    def soft(x):
        return np.tanh(x @ W)   # differentiable soft-bipolar

    for ep in range(EPOCHS):
        gW = np.zeros_like(W)
        loss = 0.0
        n_tr = 0
        for qi, r in enumerate(rows_tr):
            gq = soft(qvecs_tr[qi])
            paid = [c for c in r["cands"] if c["paid"]]
            unpaid = [c for c in r["cands"] if not c["paid"]]
            for cp in paid:
                gp = soft(cvecs_tr[cp["id"]])
                s_pos = float(np.dot(gq, gp)) / PROJ_DIM
                for cu in unpaid:
                    gu = soft(cvecs_tr[cu["id"]])
                    s_neg = float(np.dot(gq, gu)) / PROJ_DIM
                    viol = MARGIN - s_pos + s_neg
                    if viol > 0:
                        loss += viol; n_tr += 1
                        # EXACT chain: dL/dg = dL/ds * ds/dg (no extra x factors);
                        # dL/dW = outer(x, dL/dg * (1 - g^2))
                        dL_dgq = (gu - gp) / PROJ_DIM          # = -ds_pos/dgq + ds_neg/dgq
                        dL_dgp = -gq / PROJ_DIM
                        dL_dgu = +gq / PROJ_DIM
                        gW += np.outer(qvecs_tr[qi], dL_dgq * (1 - gq ** 2))
                        gW += np.outer(cvecs_tr[cp["id"]], dL_dgp * (1 - gp ** 2))
                        gW += np.outer(cvecs_tr[cu["id"]], dL_dgu * (1 - gu ** 2))
        gW += L2 * W
        W -= LR * gW / max(1, len(rows_tr))
        if ep in (0, EPOCHS // 2, EPOCHS - 1):
            print(f"[train] ep{ep} hinge={loss/max(1,n_tr):.4f} triples={n_tr}", flush=True)
    return W


def main() -> int:
    global ANSWERER
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-relay", action="store_true")
    ap.add_argument("--model", type=str, default=ANSWERER)
    args = ap.parse_args()
    ANSWERER = args.model

    rows = build_rows()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    n_tr = int(len(rows) * TRAIN_FRAC)
    tr, te = [rows[i] for i in order[:n_tr]], [rows[i] for i in order[n_tr:]]
    # batched embeds
    all_ctexts = [c["text"] for r in rows for c in r["cands"]]
    cvecs_all = _emb(all_ctexts)
    i_ = 0
    for r in rows:
        for c in r["cands"]:
            c["_z"] = cvecs_all[i_]; i_ += 1
        r["_qz"] = None
    qvecs_all = _emb([r["question"] for r in rows])
    for ri, r in enumerate(rows):
        r["_qz"] = qvecs_all[ri]
        for c in r["cands"]:
            c["cos"] = round(float(np.dot(c["_z"], r["_qz"])), 4)
    qvecs_tr = [rows[i]["_qz"] for i in order[:n_tr]]
    print(f"[split] train {len(tr)} / holdout {len(te)} questions "
          f"({sum(len(r['cands']) for r in tr)} tuples train)", flush=True)

    W = train_W(tr, qvecs_tr)

    def bound_s(xq, xc):
        return float(np.dot(np.tanh(xq @ W), np.tanh(xc @ W))) / PROJ_DIM

    # ---- readouts: mean-arm + tail-arm ----
    mean_disc = {"bound": [], "cosine": []}
    tail_rows = []
    for r in te:
        xq = r["_qz"]
        coss = [(c, c["cos"]) for c in r["cands"]]
        arg_cos = max(coss, key=lambda kv: kv[1])[0]
        is_tail = (not arg_cos["paid"]) or (arg_cos["id"] != max(
            ((c["paid"], c["id"]) for c in r["cands"]), key=lambda kv: kv[0])[1])
        if not arg_cos["paid"]:
            tail_rows.append(r)   # cosine's top pick = an unpaid session: cosine fails
        for c in r["cands"]:
            mean_disc["bound"].append((c["paid"], bound_s(xq, c["_z"])))
            mean_disc["cosine"].append((c["paid"], c["cos"]))
    def auc(pairs):
        pos_s = [s for y, s in pairs if y]; neg_s = [s for y, s in pairs if not y]
        if not pos_s or not neg_s: return None
        w = sum(1 for p in pos_s for n in neg_s if p > n) + 0.5 * sum(
            1 for p in pos_s for n in neg_s if p == n)
        return round(w / (len(pos_s) * len(neg_s)), 3)
    mean_auc = {k: auc(v) for k, v in mean_disc.items()}
    n_tail = sum(len(r["cands"]) for r in tail_rows)
    print(f"[readouts] mean-arm: bound {mean_auc['bound']} vs cosine {mean_auc['cosine']}; "
          f"tail-arm questions {len(tail_rows)} ({n_tail} tuples)", flush=True)

    out = {"n_questions": len(rows), "mean_arm": mean_auc,
           "tail_arm": {"questions": len(tail_rows)}}
    if args.no_relay:
        with io.open(OUT, "w", encoding="utf-8") as f:
            f.write(json.dumps(out, indent=2))
        print(f"[done] no-relay -> {OUT}")
        return 0

    # ---- tail-arm usefulness relay (the scientific question) ----
    done = set()
    live = []
    if os.path.exists(CK):
        for l in io.open(CK, encoding="utf-8"):
            if l.strip():
                try:
                    r_ = json.loads(l)
                    live.append(r_); done.add(r_["qi"])
                except Exception:
                    pass
    for r in tail_rows:
        if r["qi"] in done:
            continue
        gold = str(json.load(io.open(FULL500, encoding="utf-8"))[r["qi"]]["answer"])
        xq = r["_qz"]
        arg_b = max(r["cands"], key=lambda c: bound_s(xq, c["_z"]))
        arg_c = max(r["cands"], key=lambda c: c["cos"])
        if arg_b["id"] == arg_c["id"]:
            continue  # no disagreement -> not measurable
        row = {"qi": r["qi"], "bound_pick": arg_b["id"]}
        for arm, cid in (("bound", arg_b["id"]), ("cosine", arg_c["id"])):
            ans = ask_retry(r["question"], f"- {arg_b['text'] if arm=='bound' else arg_c['text']}")
            row[arm] = f1(ans, gold)
            time.sleep(3.0)
        live.append(row)
        with io.open(CK, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[tail q{r['qi']}] bound={row['bound']} cosine={row['cosine']}", flush=True)
    pairs = [(x["bound"], x["cosine"]) for x in live if "bound" in x]
    wb = sum(1 for b, c in pairs if b > c); wc = sum(1 for b, c in pairs if c > b)
    p = binomtest(wb, wb + wc, 0.5).pvalue if wb + wc else None
    out["tail_arm_results"] = {"n": len(pairs), "wins": [wb, wc], "p": None if p is None else round(p, 4),
                               "bound_mean": round(sum(b for b, _ in pairs) / max(1, len(pairs)), 3),
                               "cosine_mean": round(sum(c for _, c in pairs) / max(1, len(pairs)), 3)}
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, indent=2))
    print("TAIL-ARM:", json.dumps(out["tail_arm_results"], indent=1))
    print(f"[done] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())