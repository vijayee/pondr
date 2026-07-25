"""Build 4 -- the gate: grade the distilled cross-slot transformer on held-out
conv-anaphora gold.

THE VERDICT. Replays the 2 held-out sessions (heldout-17) through the real
orchestrator+ring with the TRAINED ``CrossSlotTransformerZHead`` (Build 3's
best.pt) as the z-head -- ``z_head_arch="transformer"`` -- so the live serve
branch reads the raw last-layer SSM state ``h[-1]`` -> [6144] on the fly
(``probe_strm_selectivity_real.py:505-510``) and stamps each ring slot's
``z_logit``. NO ``emit_raw_state`` needed (the live branch builds h_raw from
``slot.h`` directly). Then grades with the EXACT C3 math
(``_rank_turns``/``_grade_pair``/``_score_set``: self-exclusion N>=q, dedupe by
turn, budget=3) -- only the z_logit SOURCE changed (bilinear -> trained
transformer). Also runs the LOO-8 sessions as an in-distribution sanity
(train-set fit).

BAR. heldout-17 ``in_top3`` > 0.647 (beats C1 cosine+age+CE-reranker, the recall
winner). Baselines on heldout-17: C1 0.235/0.647, C2 qwen3:8b 0.294/0.647, C3
text2x CompositeZHead 0.353/0.355. PASS = the SSM state carries query-relevance
at the right primitive (user's faith confirmed, Phase 0b overturned at the
retrieval objective); FAIL = the state encodes identity not relevance (faith
falsified). Either result is decisive -> Build 5 (expand to 27) runs ONLY on a
PASS / clear directional lift.

NO engine change. ``replay_and_capture`` + ``load_cross_slot_transformer``
IMPORTED read-only (never modified); only the global
``_runtime_config.strm_ring_text`` singleton is flipped (same flag C3 flips).
onyx PRIVATE -- nothing leaves scripts/_scratch/. No uploads. Per CLAUDE.md
de-wonk at completion.

Run (GPU; ~10 min):
  PYTHONPATH=. python scripts/_scratch/_xslot_distill_gate.py
  (env: CKPT=_xslot_distill_s0/best.pt default; SCOPE=loo8 picks the Build-3
  dir; FINAL=1 uses final.pt instead of best.pt.)
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scripts/_scratch"
EP_HELDOUT = SCRATCH / "_heldout_episodes.json"
GOLD_HELDOUT = SCRATCH / "_heldout_gold.json"
EP_TRAINED = SCRATCH / "_trained_episodes_for_labeling.json"
GOLD_TRAINED = SCRATCH / "_trained_gold.json"
OUT_PATH = SCRATCH / "_xslot_distill_gate.json"
RAW_PATH = SCRATCH / "_xslot_distill_gate_raw_turns.json"

BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full_finetuned_text2x.pt"
REL_HEAD = "data/training/strm_relevance/best.pt"

BUDGET = 3
LOO_MIN_PAIRS = 6
RING_CAPACITY = 16
SCOPE = os.getenv("SCOPE", "loo8")
DISTILL_DIR = SCRATCH / (
    "_xslot_distill_s0" if SCOPE == "loo8" else "_xslot_distill_trained27_s0")
CKPT_NAME = "final.pt" if os.getenv("FINAL", "0") == "1" else "best.pt"
CKPT_PATH = DISTILL_DIR / CKPT_NAME

# C1/C2/C3 heldout-17 baselines (the bar to beat).
C1_HELDOUT = (0.235, 0.647)
C2_HELDOUT = (0.294, 0.647)
C3_HELDOUT = (0.353, 0.355)
BAR_IN_TOP3 = 0.647

_MSG_RE = re.compile(r"#msg(\d+)$")
_EP_RE = re.compile(r"__ep(\d+)$")


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals); n = len(s); m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _user_turns(sess: dict) -> list[str]:
    return [t["text"] for t in sess["turns"] if t.get("role") == "user"]


def _loo_sessions(gold: dict) -> list[str]:
    by_sid: dict[str, int] = {}
    for p in gold["pairs"]:
        by_sid[p["session_id"]] = by_sid.get(p["session_id"], 0) + 1
    return sorted(sid for sid, n in by_sid.items() if n >= LOO_MIN_PAIRS)


def _write_transcript(sid: str, sess: dict, tmpdir: str) -> str:
    msgs = [{"message_type": t["role"], "message": t["text"]}
            for t in sess["turns"] if t.get("text")]
    path = Path(tmpdir) / f"{sid}.json"
    path.write_text(json.dumps({"chat_session_id": sid, "messages": msgs},
                               ensure_ascii=False), encoding="utf-8")
    return str(path)


def _slot_turn(source_id: str) -> int | None:
    if not source_id:
        return None
    m = _MSG_RE.search(source_id)
    if m:
        return int(m.group(1))
    m = _EP_RE.search(source_id)
    if m:
        return int(m.group(1))
    return None


def _rank_turns(slots: list[dict], exclude_turn: int | None = None) -> list[int]:
    """Rank ring slots by z_logit desc, dedupe by turn (max z_logit per turn),
    exclude the query's own turn N>=q. Verbatim C3 logic (only the z_logit
    source differs: trained transformer here)."""
    best: dict[int, float] = {}
    for s in slots:
        if s.get("z_logit") is None:
            continue
        N = _slot_turn(s.get("source_id"))
        if N is None:
            continue
        if exclude_turn is not None and N >= exclude_turn:
            continue
        z = float(s["z_logit"])
        if N not in best or z > best[N]:
            best[N] = z
    return [N for N, _ in sorted(best.items(), key=lambda kv: kv[1],
                                 reverse=True)]


def _grade_pair(rec: dict, t_idx: int):
    if rec is None:
        return None
    q_turn = rec.get("turn_index")
    ranked = _rank_turns(rec["slots"], exclude_turn=q_turn)
    if not ranked:
        return {"target_in_ring": False, "picked": [], "rank": None,
                "top1": False, "in_top3": False, "competitor_beats": True,
                "breadth": 0}
    top = ranked[:BUDGET]
    rank = None
    for pos, N in enumerate(top, 1):
        if N == t_idx:
            rank = pos; break
    return {"target_in_ring": t_idx in ranked, "picked": top, "rank": rank,
            "top1": rank == 1, "in_top3": rank is not None and rank <= BUDGET,
            "competitor_beats": rank is None or rank > 1,
            "breadth": min(BUDGET, len(top))}


def _score_set(turn_records, episodes, pairs, window, age_threshold, label):
    by_rec = {(r["session_id"], r["turn_index"]): r for r in turn_records}
    n_in = n_top1 = n_top3 = n_beats = n_in_ring = 0
    breadths: list[float] = []
    per_pair = []
    for p in pairs:
        sid = p["session_id"]; q = int(p["query"]); t = int(p["target"])
        sess = episodes.get(sid)
        if sess is None:
            continue
        ut = _user_turns(sess)
        if q >= len(ut) or t >= len(ut):
            continue
        age = q - t - 1
        lo = max(0, q - window)
        if not (age >= age_threshold and age <= window - 1 and t >= lo):
            continue
        rec = by_rec.get((sid, q))
        if rec is None:
            per_pair.append({"sid": sid[:8], "q": q, "t": t, "age": age,
                             "record_missing": True, "top1": False,
                             "in_top3": False, "competitor_beats": True,
                             "breadth": 0})
            n_in += 1; n_beats += 1; breadths.append(0.0)
            continue
        g = _grade_pair(rec, t)
        n_in += 1
        if g["target_in_ring"]:
            n_in_ring += 1
        if g["top1"]:
            n_top1 += 1
        if g["in_top3"]:
            n_top3 += 1
        if g["competitor_beats"]:
            n_beats += 1
        breadths.append(float(g["breadth"]))
        per_pair.append({"sid": sid[:8], "q": q, "t": t, "age": age,
                         "target_in_ring": g["target_in_ring"],
                         "picked": g["picked"], "rank": g["rank"],
                         "top1": g["top1"], "in_top3": g["in_top3"],
                         "competitor_beats": g["competitor_beats"],
                         "breadth": g["breadth"]})
    n = n_in
    agg = {
        "label": label, "n_in_window": n,
        "target_top1_rate": round(n_top1 / n, 4) if n else 0.0,
        "target_in_top3_rate": round(n_top3 / n, 4) if n else 0.0,
        "competitor_beats_rate": round(n_beats / n, 4) if n else 0.0,
        "target_in_ring_rate": round(n_in_ring / n, 4) if n else 0.0,
        "median_breadth": round(_median(breadths), 4) if breadths else None,
    }
    print(f"  {label:<20} n={n}  top1={agg['target_top1_rate']:.3f}  "
          f"in_top3={agg['target_in_top3_rate']:.3f}  "
          f"in_ring={agg['target_in_ring_rate']:.3f}  "
          f"beats={agg['competitor_beats_rate']:.3f}  "
          f"breadth={agg['median_breadth']}", flush=True)
    return {**agg, "per_pair": per_pair}


def _verdict(heldout_in_top3: float) -> str:
    if heldout_in_top3 > BAR_IN_TOP3:
        return (f"PASS -- heldout-17 in_top3={heldout_in_top3:.3f} > "
                f"{BAR_IN_TOP3} (beats C1). State carries query-relevance; "
                "faith CONFIRMED. -> Build 5 (expand to 27).")
    if heldout_in_top3 > C3_HELDOUT[1] + 0.05:
        return (f"LIFT -- heldout-17 in_top3={heldout_in_top3:.3f} clears C3 "
                f"({C3_HELDOUT[1]}) but not the {BAR_IN_TOP3} bar. Directional "
                "signal; Build 5 may push it over.")
    return (f"FAIL -- heldout-17 in_top3={heldout_in_top3:.3f} <= "
            f"{BAR_IN_TOP3}. State does not carry anaphora signal at this "
            "primitive; faith FALSIFIED. Do NOT expand.")


def main() -> int:
    if not CKPT_PATH.exists():
        print(f"ERROR: trained transformer not found: {CKPT_PATH}\n"
              f"       (run _train_xslot_distill.py first)", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT))
    import torch  # noqa: F401
    from src.config import config as _runtime_config
    from scripts.probe_strm_selectivity_real import replay_and_capture

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gate] ckpt={CKPT_PATH} device={device}", flush=True)
    print(f"[gate] backbone={BACKBONE}", flush=True)
    print(f"[gate] bar: heldout-17 in_top3 > {BAR_IN_TOP3} (C1={C1_HELDOUT[1]}, "
          f"C2={C2_HELDOUT[1]}, C3={C3_HELDOUT[1]})", flush=True)

    heldout_ep = json.loads(EP_HELDOUT.read_text(encoding="utf-8"))
    heldout_g = json.loads(GOLD_HELDOUT.read_text(encoding="utf-8"))
    trained_ep = json.loads(EP_TRAINED.read_text(encoding="utf-8"))
    trained_g = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))
    heldout_sids = list(heldout_ep.keys())
    loo_sids = _loo_sessions(trained_g)
    print(f"[gate] heldout sessions={heldout_sids}  loo-8={loo_sids}", flush=True)

    tmpdir = tempfile.mkdtemp(prefix="pondr_xslot_gate_")
    transcripts: list[str] = []
    sid_to_set: dict[str, str] = {}
    for sid in heldout_sids:
        transcripts.append(_write_transcript(sid, heldout_ep[sid], tmpdir))
        sid_to_set[sid] = "heldout-17"
    for sid in loo_sids:
        if sid in trained_ep:
            transcripts.append(_write_transcript(sid, trained_ep[sid], tmpdir))
            sid_to_set[sid] = "loo-8-normal"
    print(f"[gate] wrote {len(transcripts)} transcripts to {tmpdir}", flush=True)

    _runtime_config.strm_ring_text = True

    turn_records, run_stats = replay_and_capture(
        transcripts=transcripts,
        backbone_path=BACKBONE,
        rel_head_path=REL_HEAD,
        z_relevance_head_path=str(CKPT_PATH),
        z_head_arch="transformer",
        identity_instance=True,
        salience="off",
        doc_store=None,
        ring_capacity=RING_CAPACITY,
        max_turns=0,
        device=device,
        user_id="xslot-gate",
        rec_head_path="",
        ld_head_path="",
        ablate_yt=False,
        emit_traces=None,
    )
    print(f"[gate] captured {len(turn_records)} turn-records "
          f"(encoded={run_stats.get('n_encoded')} skipped={run_stats.get('n_skipped')})",
          flush=True)

    # Slim per-turn dump for offline ranking iteration (no re-replay). onyx PRIVATE.
    slim = [{"session_id": r["session_id"], "turn_index": r["turn_index"],
             "slots": [{"source_id": s.get("source_id"),
                        "slot_type": s.get("slot_type"),
                        "z_logit": s.get("z_logit"), "cos": s.get("cos")}
                       for s in r.get("slots", [])]}
            for r in turn_records]
    RAW_PATH.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    print(f"[gate] wrote {RAW_PATH} ({len(slim)} turn-records, slim)", flush=True)

    recs_by_set = {"heldout-17": [], "loo-8-normal": []}
    for rec in turn_records:
        s = sid_to_set.get(rec["session_id"])
        if s:
            recs_by_set[s].append(rec)

    out = {"ckpt": str(CKPT_PATH), "device": device, "backbone": BACKBONE,
           "ring_capacity": RING_CAPACITY, "budget": BUDGET,
           "n_turn_records": len(turn_records),
           "baselines_heldout17": {"C1": C1_HELDOUT, "C2": C2_HELDOUT,
                                   "C3": C3_HELDOUT, "bar_in_top3": BAR_IN_TOP3},
           "sets": {}}
    print("\n=== distilled CrossSlotTransformerZHead (mixed ring) ===", flush=True)
    out["sets"]["heldout-17"] = _score_set(
        recs_by_set["heldout-17"], heldout_ep, heldout_g["pairs"],
        int(heldout_g.get("window", 16)), int(heldout_g.get("age_threshold", 3)),
        "heldout-17")
    loo_pairs = [p for p in trained_g["pairs"] if p["session_id"] in set(loo_sids)]
    out["sets"]["loo-8-normal"] = _score_set(
        recs_by_set["loo-8-normal"], trained_ep, loo_pairs,
        int(trained_g.get("window", 16)), int(trained_g.get("age_threshold", 3)),
        "loo-8-normal")

    h = out["sets"]["heldout-17"]["target_in_top3_rate"]
    out["verdict"] = _verdict(h)
    print(f"\n[gate] VERDICT: {out['verdict']}", flush=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[gate] wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())