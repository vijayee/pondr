"""Build 3 -- train CrossSlotTransformerZHead on (SSM state, query) -> qwen label.

The distillation trainer. Joins the raw SSM states (Build 2) with the dense qwen
teacher labels (Build 1) by ``(session_id, q)`` and trains the cross-slot
transformer to predict the teacher's per-turn relevance scores FROM THE STATES
-- the crux test of the user's "the backbone encodes all we need to know about
the text" claim. The teacher works over TEXT (which it can read); the student
works over the SSM STATE that produced that text. If the student learns the
teacher's labels from the states alone, the state carries query-relevance (faith
confirmed); if not, the state encodes identity not relevance (faith falsified).

THE JOIN (the crux). For each state record (one per query turn Q) the ring has
K' slots, each with a source_id -> user-turn index N. The teacher, run over Q's
16-turn window, gives a soft score for every window turn N. So::

    label[k] = teacher_score(Q, N)   if N in Q's window
             = 0.0                  otherwise (out-of-window / foreign recall)

Both the conv slot (#msg{N}) and the retrieved-episode slot (__ep{N}) for turn N
get the SAME label (they encode the same turn); the transformer learns "this
state encodes turn N, whose relevance to Q is X" for the mixed ring it sees at
serve. The query's OWN #msg{q} slot is EXCLUDED from the input (matches the
gate's self-exclusion N >= q at serve -- train/serve input parity). Records with
no in-window slot are dropped (nothing to learn); all-negative in-window records
are kept (valid BCE examples).

PRIMITIVE. CrossSlotTransformerZHead (src/subconscious/cross_slot_transformer.py)
-- cross-slot attention, each slot's logit depends on the query AND every other
slot (a RELATIVE score, the mechanism that beat the bilinear 2.614 vs 0.200 in
task #45). n_slot_types=0 = byte-identical to the 2.614 arch. SUPERVISION: BCE-
with-logits on the dense soft labels (pattern from relevance_training.py:379),
AdamW lr=1e-3 wd=0.01, accum=4, ~120 epochs (skeleton from
probe_head_to_head_onyx._train_head:499-892).

CHECKPOINT SELECTION. LOO-8 split into 7 train sessions + 1 proxy-val session
(da0964cd, the most gold pairs = 11) -- a clean leave-1-session-out mini-LOO.
Per-epoch: grade the proxy-val session's REAL anaphora gold pairs (from
_trained_gold.json) by the head's logit (rank turns, top-3, check gold target) --
the SAME metric Build 4's gate uses, on an UNSEEN session. best.pt = best proxy-
val in_top3; final.pt = last epoch. The HELD-OUT gate is Build 4 (heldout-17), NOT
this proxy.

NO engine change. CrossSlotTransformerZHead + load helpers IMPORTED read-only
(never modified). onyx PRIVATE -- nothing leaves scripts/_scratch/. No uploads.
Per CLAUDE.md de-wonk at completion.

Run (GPU; ~5-15 min for 7 sessions):
  PYTHONPATH=. python scripts/_scratch/_train_xslot_distill.py
  (env: SCOPE=loo8 default|trained for Build 5; EPOCHS=120; SEED for the data
  shuffle; PROXY_SESSION overrides the held-out session id.)
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch  # module-level: _build_train_recs/_grade_proxy use torch.stack/tensor
             # before main()'s local import is in scope.

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
STATES_PATH = SCRATCH / (
    "_states_loo8.pt" if os.getenv("SCOPE", "loo8") == "loo8"
    else "_states_trained27.pt")
LABELS_PATH = SCRATCH / (
    "_dense_teacher_labels_loo8.json" if os.getenv("SCOPE", "loo8") == "loo8"
    else "_dense_teacher_labels_trained27.json")
GOLD_TRAINED = SCRATCH / "_trained_gold.json"
OUT_DIR = SCRATCH / (
    "_xslot_distill_s0" if os.getenv("SCOPE", "loo8") == "loo8"
    else "_xslot_distill_trained27_s0")

EPOCHS = int(os.getenv("EPOCHS", "120"))
LR = float(os.getenv("LR", "1e-3"))
WD = float(os.getenv("WD", "0.01"))
ACCUM = int(os.getenv("ACCUM", "4"))
SEED = int(os.getenv("SEED", "0"))
BUDGET = 3
# Proxy-val session: da0964cd has the most gold pairs (11) -> most reliable proxy.
PROXY_SESSION = os.getenv(
    "PROXY_SESSION", "da0964cd-6f2f-4e0d-a156-dadb578a285f")


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _build_teacher_index(labels_doc: dict) -> dict:
    """(session_id, q) -> {N: score} for the join. Keys are ints."""
    out = {}
    for sid, recs in labels_doc.get("sessions", {}).items():
        for r in recs:
            out[(sid, int(r["q"]))] = {int(k): float(v)
                                       for k, v in r.get("scores", {}).items()}
    return out


def _build_train_recs(states: list, teacher: dict, gold: dict):
    """Join states with teacher labels -> list of {sid, q, z[K,6144],
    labels[K], slot_types[K], turns[K], query_emb[384]}.

    LABEL RULE (per slot k, N = slot_turns[k]):
      * FOREIGN slot (slot_sids[k] != sid) -> 0.0. ``replay_and_capture`` reuses
        one store across sessions, so a later session's ring contains retrieved
        ``__ep`` from PRIOR sessions -- serve-faithful cross-session distractors
        (at real serve the retriever recalls the user's full prior history).
        They are NOT the within-session anaphora target the teacher scored, so
        the student learns to SUPPRESS them. Detected by source_id prefix !=
        the record's true session_id (Build 2's ``slot_sids``).
      * query's own #msg{q} (N == q, same session) -> 0.0. The engine scores
        this self-match slot (cos=1.0, typically the highest z_logit); the gate
        excludes it post-hoc in ranking (N>=q), NOT from scoring. Keeping it in
        the input with label 0.0 is serve-faithful AND teaches suppression --
        cross-slot attention means DROPPING it would shift the other slots'
        logits (a train/serve mismatch), so it stays in, label 0.0.
      * same-session in-window turn -> teacher_score(N).
      * same-session out-of-window turn -> 0.0 (tscores.get default).
    Drops records with no same-session in-window slot (nothing to learn)."""
    out = []
    n_drop_no_teacher = n_drop_no_inwindow = 0
    for rec in states:
        sid = rec.get("session_id"); q = rec.get("q")
        if sid is None or q is None:
            continue
        tscores = teacher.get((sid, q))
        if tscores is None:
            n_drop_no_teacher += 1
            continue
        sh = rec["slots_h_raw"]               # [K',4,16,384] fp16
        slot_turns = rec.get("slot_turns")    # [K'] (int | None)
        slot_types = rec.get("slot_types")    # [K'] (int | None)
        slot_sids = rec.get("slot_sids")      # [K'] (str|None) per-slot prefix
        query_emb = rec["query_emb"]          # [384] fp32
        z_list, lab_list, st_list, turn_list = [], [], [], []
        in_window = False
        for k in range(sh.shape[0]):
            N = slot_turns[k]
            if N is None:
                continue
            z6144 = sh[k, -1].float().reshape(-1)   # last layer [16,384] -> 6144
            slot_sid = slot_sids[k] if slot_sids is not None else sid
            if slot_sid != sid:
                label = 0.0                    # foreign cross-session distractor
            elif N == q:
                label = 0.0                    # query's own self-match (#msg{q})
            else:
                label = tscores.get(N, 0.0)    # teacher score if in window else 0
                if N in tscores:
                    in_window = True
            z_list.append(z6144); lab_list.append(label)
            st_list.append(int(slot_types[k]) if slot_types[k] is not None else 0)
            turn_list.append(N)
        if not z_list:
            n_drop_no_inwindow += 1
            continue
        if not in_window:
            # no same-session slot the teacher even considered -> nothing to
            # learn (the ring may be all-foreign or all out-of-window).
            n_drop_no_inwindow += 1
            continue
        out.append({
            "sid": sid, "q": q,
            "z": torch.stack(z_list),               # [K,6144] fp32
            "labels": torch.tensor(lab_list, dtype=torch.float32),  # [K]
            "slot_types": torch.tensor(st_list, dtype=torch.long),   # [K]
            "turns": turn_list,                     # [K] int
            "query_emb": query_emb.float(),         # [384] fp32
        })
    return out, (n_drop_no_teacher, n_drop_no_inwindow)


def _grade_proxy(val_recs: list, head, gold_pairs: list, device) -> dict:
    """Grade a proxy-val session's gold pairs by the head's logit. Mirrors the
    C3 gate: rank slots by logit desc, dedupe by turn (max logit per turn), take
    top-BUDGET turns, check the gold target. Returns {n, top1_rate,
    in_top3_rate, median_breadth}. A gold pair with no state record (K<3 at that
    turn) is an honest miss (counted in the denominator as a miss)."""
    by_rec = {(r["sid"], r["q"]): r for r in val_recs}
    n = n_top1 = n_top3 = 0
    breadths = []
    for p in gold_pairs:
        sid = p["session_id"]; q = int(p["query"]); t = int(p["target"])
        rec = by_rec.get((sid, q))
        n += 1
        if rec is None:
            breadths.append(0.0)
            continue
        z = rec["z"].to(device); qe = rec["query_emb"].to(device)
        turns = rec["turns"]; rec_q = rec["q"]
        with torch.no_grad():
            logits = head.logits(torch.zeros(z.shape[0], 1, device=device),
                                  z, qe).squeeze(-1)   # [K]
        # dedupe by turn (max logit per turn), excluding the query's own turn
        # (N>=rec_q) and foreign recall (N is None) -- the gate's _rank_turns
        # self-exclusion. The q-slot IS in the input for serve parity but is
        # never a valid anaphora candidate, so it is dropped here.
        best: dict[int, float] = {}
        for logit, N in zip(logits.tolist(), turns):
            if N is None or N >= rec_q:
                continue
            if N not in best or logit > best[N]:
                best[N] = logit
        ranked = [N for N, _ in sorted(best.items(), key=lambda kv: kv[1],
                                       reverse=True)]
        top = ranked[:BUDGET]
        breadths.append(float(min(BUDGET, len(top))))
        rank = None
        for pos, N in enumerate(top, 1):
            if N == t:
                rank = pos; break
        if rank == 1:
            n_top1 += 1
        if rank is not None and rank <= BUDGET:
            n_top3 += 1
    return {"n": n, "top1_rate": round(n_top1 / n, 4) if n else 0.0,
            "in_top3_rate": round(n_top3 / n, 4) if n else 0.0,
            "median_breadth": round(_median(breadths), 4) if breadths else None}


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    from src.subconscious.cross_slot_transformer import CrossSlotTransformerZHead

    if not STATES_PATH.exists():
        print(f"ERROR: need {STATES_PATH} (run _capture_states.py first)",
              file=sys.stderr); return 1
    if not LABELS_PATH.exists():
        print(f"ERROR: need {LABELS_PATH} (run _dense_teacher_labels.py first)",
              file=sys.stderr); return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED)
    torch.manual_seed(SEED)

    # map_location='cpu': Build 2 cpu-offloads before save, but force cpu here
    # too so a cuda-saved .pt still loads on a CPU-only host (then .to(device)).
    states = torch.load(STATES_PATH, weights_only=False, map_location="cpu")
    labels_doc = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))
    teacher = _build_teacher_index(labels_doc)
    print(f"[distill] states={len(states)} teacher-labeled turns={len(teacher)} "
          f"device={device}", flush=True)

    recs, (n_no_t, n_no_iw) = _build_train_recs(states, teacher, gold)
    print(f"[distill] built {len(recs)} training records "
          f"(dropped: no_teacher={n_no_t} no_in_window={n_no_iw})", flush=True)
    if not recs:
        print("ERROR: no training records", file=sys.stderr); return 1

    # Split: proxy-val session held out from training; rest train.
    val_recs = [r for r in recs if r["sid"] == PROXY_SESSION]
    train_recs = [r for r in recs if r["sid"] != PROXY_SESSION]
    val_pairs = [p for p in gold["pairs"] if p["session_id"] == PROXY_SESSION]
    print(f"[distill] train={len(train_recs)} recs ({len({r['sid'] for r in train_recs})} "
          f"sessions) | proxy-val={len(val_recs)} recs, {len(val_pairs)} gold pairs "
          f"(session {PROXY_SESSION[:8]})", flush=True)
    if not val_recs or not val_pairs:
        print("ERROR: proxy-val session has no records or no gold pairs",
              file=sys.stderr); return 1

    # pos_weight from the TRAIN labels (soft-label mass balance): pos_weight =
    # neg_mass / pos_mass so the rare relevant slots are not drowned out.
    pos_mass = sum(float(r["labels"].sum()) for r in train_recs)
    neg_mass = sum(float((1.0 - r["labels"]).sum()) for r in train_recs)
    pos_weight = (neg_mass / pos_mass) if pos_mass > 0 else 1.0
    print(f"[distill] pos_weight={pos_weight:.3f} "
          f"(pos_mass={pos_mass:.1f} neg_mass={neg_mass:.1f})", flush=True)
    pos_weight_t = torch.tensor(pos_weight, device=device)

    head = CrossSlotTransformerZHead(dim_in=6144, hidden=128, n_slot_types=0,
                                     learnable_temp=False, dropout=0.0).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _save(path: Path, ep: int, metric: float):
        ckpt = {"head": head.state_dict(), "arch": "transformer",
                "slot_dim": head.slot_dim, "doc_dim": head.doc_dim,
                "query_dim": head.query_dim, "proj_dim": head.proj_dim,
                "hidden": 128, "n_slot_types": 0, "learnable_temp": False,
                "dropout": 0.0, "epoch": ep, "proxy_in_top3": metric}
        torch.save(ckpt, path)

    best_metric = -1.0
    for ep in range(EPOCHS):
        head.train()
        random.shuffle(train_recs)
        opt.zero_grad()
        n_pending = 0
        for r in train_recs:
            z = r["z"].to(device); labels = r["labels"].to(device)
            qe = r["query_emb"].to(device)
            logits = head.logits(torch.zeros(z.shape[0], 1, device=device),
                                 z, qe).squeeze(-1)        # [K]
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight_t)
            (loss / ACCUM).backward()
            n_pending += 1
            if n_pending == ACCUM:
                opt.step(); opt.zero_grad(); n_pending = 0
        if n_pending > 0:                 # flush the trailing partial accum only
            opt.step(); opt.zero_grad()   # (no double-step / spurious wd when
                                          # len % ACCUM == 0 -- n_pending is 0)

        head.eval()
        m = _grade_proxy(val_recs, head, val_pairs, device)
        if m["in_top3_rate"] > best_metric:
            best_metric = m["in_top3_rate"]
            _save(OUT_DIR / "best.pt", ep, best_metric)
        if ep % 10 == 0 or ep == EPOCHS - 1 or m["in_top3_rate"] >= 0.999:
            print(f"  ep{ep:03d} proxy top1={m['top1_rate']:.3f} "
                  f"in_top3={m['in_top3_rate']:.3f} "
                  f"(best={best_metric:.3f}) n={m['n']}", flush=True)
    _save(OUT_DIR / "final.pt", EPOCHS - 1, m["in_top3_rate"])
    print(f"\n[distill] done. best proxy in_top3={best_metric:.3f}", flush=True)
    print(f"[distill] best.pt + final.pt -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())