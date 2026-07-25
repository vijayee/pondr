"""C3 -- Stage-3 text2x CompositeZHead readout scored on CONV-turn anaphora.

The C3 leg of the STRM 3-way A/B/C isolation test (Step 2). C3's readout
(CompositeZHead = StateReadout flat_last[6144]->384 + ZRelevanceHead bilinear,
backbone ``backbone_v2_full_finetuned_text2x.pt``) PASSED the doc-ring 6-seed
gate (ret_text+ret_code z_logit >= 2.0) -- but that gate grades DOC retrieval,
never CONV-turn anaphora. This probe is the readout's anaphora number it never
had: does a readout trained on doc-ring generalize to "which prior USER TURN
does THIS query refer back to"?

NO engine change. The orchestrator's replay path is IMPORTED read-only from
``scripts/probe_strm_selectivity_real.replay_and_capture`` (never modified);
only the global ``_runtime_config.strm_ring_text`` singleton is flipped (the
same flag the probe's own CLI flips) so the orchestrator emits scoreable conv
slots (slot_type 0, ``source_id = f"{session_id}#msg{turn}"``).

PROVENANCE / TURN MAPPING (the key). Under ``strm_ring_text`` the orchestrator
stamps each conv slot ``source_id = f"{session_id}#msg{N}"`` where N is the
user-turn index (the replay loop's ``turn_index = i`` over ``_pair_turns`` =
i-th user turn). Retrieved conv-pair episodes are stamped ``__ep{N:04d}`` for
the same N. So each ring slot maps cleanly back to a user-turn index by parsing
``#msg`` (slot_type 0) or ``__ep`` (slot_type 1) -- no text matching, no gold
circularity. The anaphora gold's ``query``/``target`` ARE user-turn indices into
``_user_turns(sess)``; the transcript is written FROM the episode JSON's
interleaved user/assistant turns, so ``_pair_turns``[i] == gold user-turn i.

RANKING. The faithful C3 serve behavior: the CompositeZHead scores EVERY
text-bearing ring slot (conv slot_type 0 + retrieved slot_type 1). We rank all
ring slots by ``z_logit`` desc, dedupe by turn index (a turn may appear as BOTH
``#msg{N}`` and ``__ep{N:04d}`` -- take the max z_logit per turn), take top-3
turns, and check the gold target turn. If the target turn is not in the ring at
all (evicted / not recalled) it is an honest miss -- reported as
``target_in_ring=False``, not silently dropped. A type-0-only variant is also
reported as a diagnostic (conv slots cover turns 1..q-1; turn 0 is retrieved-
only, so type-0-only undercounts target=0 pairs -- documented, not hidden).

FIDELITY: same gold files, same ``_user_turns`` 16-turn window, age>=3, and
rank-then-budget=BUDGET=3 metric as C1 (``_c1_reranker.py``) and C2
(``_llm_salience_probe.py``) -- apples-to-apples across C1/C2/C3. One CompositeZHead
seed per run (env SEED, default 0); run SEED=0..5 and combine for the 6-seed
view the 1f-7 doc-ring gate used.

UNTRACKED scratch. onyx PRIVATE -- nothing leaves the box. No uploads. No engine
edits. Per CLAUDE.md de-wonk at completion.

Run (GPU; one seed):
  PYTHONPATH=. SEED=0 python scripts/_scratch/_c3_anaphora_probe.py
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
OUT_PATH = SCRATCH / "_c3_anaphora_result.json"
RAW_PATH = SCRATCH / "_c3_anaphora_raw_turns.json"

BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full_finetuned_text2x.pt"
REL_HEAD = "data/training/strm_relevance/best.pt"
ZHEAD_DIR = "data/training/strm_state_readout/head_to_head_onyx"

BUDGET = 3
LOO_MIN_PAIRS = 6
RING_CAPACITY = 16
SEED = int(os.getenv("SEED", "0"))

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
    """The LOO-8 normal set: sessions with >= LOO_MIN_PAIRS gold pairs."""
    by_sid: dict[str, int] = {}
    for p in gold["pairs"]:
        by_sid[p["session_id"]] = by_sid.get(p["session_id"], 0) + 1
    return sorted(sid for sid, n in by_sid.items() if n >= LOO_MIN_PAIRS)


def _write_transcript(sid: str, sess: dict, tmpdir: str) -> str:
    """Episode JSON ``{turns:[{role,text}]}`` -> Onyx transcript
    ``{chat_session_id, messages:[{message_type,message}]}`` on disk, preserving
    the interleaved user/assistant order so ``_pair_turns``[i] == gold user-turn i."""
    msgs = [{"message_type": t["role"], "message": t["text"]}
            for t in sess["turns"] if t.get("text")]
    path = Path(tmpdir) / f"{sid}.json"
    path.write_text(json.dumps({"chat_session_id": sid, "messages": msgs},
                               ensure_ascii=False), encoding="utf-8")
    return str(path)


def _slot_turn(source_id: str) -> int | None:
    """Map a ring slot's source_id to its user-turn index. ``#msg{N}`` (conv,
    slot_type 0) and ``__ep{N:04d}`` (retrieved episode, slot_type 1) both encode
    the user-turn index N. Returns None if the source_id is neither (e.g. a
    foreign recalled episode from another session -- not rankable here)."""
    if not source_id:
        return None
    m = _MSG_RE.search(source_id)
    if m:
        return int(m.group(1))
    m = _EP_RE.search(source_id)
    if m:
        return int(m.group(1))
    return None


def _rank_turns(slots: list[dict], type_filter: int | None,
                exclude_turn: int | None = None) -> list[int]:
    """Rank ring slots by z_logit desc, dedupe by turn index (max z_logit per
    turn), return the ordered list of turn indices (most-relevant first).
    ``type_filter`` None = all text-bearing slots (the faithful mixed ring);
    0 = conv slots only (diagnostic); 1 = retrieved only. ``exclude_turn`` =
    the current query's turn index (q): the orchestrator adds the query's OWN
    conv slot ``#msg{q}`` to the ring during ``query()`` (cos=1.0 self-match,
    typically the highest z_logit) -- it is the query, not a candidate, so it
    MUST be excluded to match C1/C2's window (q-16..q-1, never q). Slots with
    no z_logit or unmappable source_id are skipped (honest -- not rankable)."""
    best: dict[int, float] = {}
    for s in slots:
        if s.get("z_logit") is None:
            continue
        if type_filter is not None and s.get("slot_type") != type_filter:
            continue
        N = _slot_turn(s.get("source_id"))
        if N is None:
            continue
        if exclude_turn is not None and N >= exclude_turn:
            # Exclude the query's own turn AND any future turn (future turns
            # should never be in the ring, but guard anyway -- they are not
            # valid anaphora candidates).
            continue
        z = float(s["z_logit"])
        if N not in best or z > best[N]:
            best[N] = z
    return [N for N, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]


def _grade_pair(rec: dict, t_idx: int, window: int, age_threshold: int,
                type_filter: int | None):
    """Grade one gold pair against one captured turn record. Returns a per-pair
    dict or None if the pair is out-of-window / record missing."""
    if rec is None:
        return None
    q_turn = rec.get("turn_index")
    ranked = _rank_turns(rec["slots"], type_filter, exclude_turn=q_turn)
    if not ranked:
        return {"in_window": True, "target_in_ring": False, "picked": [],
                "rank": None, "top1": False, "in_top3": False,
                "competitor_beats": True, "breadth": 0}
    top = ranked[:BUDGET]
    rank = None
    for pos, N in enumerate(top, 1):
        if N == t_idx:
            rank = pos
            break
    top1 = rank == 1
    in_top3 = rank is not None and rank <= BUDGET
    beats = rank is None or rank > 1
    return {"in_window": True, "target_in_ring": t_idx in ranked,
            "picked": top, "rank": rank, "top1": top1, "in_top3": in_top3,
            "competitor_beats": beats, "breadth": min(BUDGET, len(top))}


def _score_set(turn_records: list[dict], episodes: dict, pairs: list[dict],
               window: int, age_threshold: int, type_filter: int | None,
               label: str) -> dict:
    """Score all gold pairs for one set. Builds a (session_id, turn_index)->record
    index, then grades each in-window pair. Mirrors C1/C2 metrics."""
    by_rec: dict[tuple[str, int], dict] = {}
    for rec in turn_records:
        by_rec[(rec["session_id"], rec["turn_index"])] = rec
    n_in = n_top1 = n_top3 = n_beats = n_fires = n_in_ring = 0
    breadths: list[float] = []
    per_pair: list[dict] = []
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
        in_window = (age >= age_threshold and age <= window - 1 and t >= lo)
        if not in_window:
            continue
        rec = by_rec.get((sid, q))
        if rec is None:
            # turn_index q not captured (e.g. query-fail / turn 0). Honest miss.
            per_pair.append({"sid": sid[:8], "q": q, "t": t, "age": age,
                             "in_window": True, "record_missing": True,
                             "top1": False, "in_top3": False,
                             "competitor_beats": True, "breadth": 0})
            n_in += 1
            n_beats += 1
            breadths.append(0.0)
            continue
        g = _grade_pair(rec, t, window, age_threshold, type_filter)
        if g is None:
            continue
        n_in += 1
        if g["target_in_ring"]:
            n_in_ring += 1
        # fires_on_target analog: target is rankable in the ring (in_ring).
        if g["target_in_ring"]:
            n_fires += 1
        if g["top1"]:
            n_top1 += 1
        if g["in_top3"]:
            n_top3 += 1
        if g["competitor_beats"]:
            n_beats += 1
        breadths.append(float(g["breadth"]))
        per_pair.append({"sid": sid[:8], "q": q, "t": t, "age": age,
                         "in_window": True, "target_in_ring": g["target_in_ring"],
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
        "fires_on_target_rate": round(n_fires / n, 4) if n else 0.0,
        "median_breadth": round(_median(breadths), 4) if breadths else None,
    }
    print(f"  {label:<22} n={n}  top1={agg['target_top1_rate']:.3f}  "
          f"in_top3={agg['target_in_top3_rate']:.3f}  "
          f"in_ring={agg['target_in_ring_rate']:.3f}  "
          f"beats={agg['competitor_beats_rate']:.3f}  "
          f"breadth={agg['median_breadth']}", flush=True)
    return {**agg, "per_pair": per_pair}


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import torch  # noqa: F401 -- device detect
    from src.config import config as _runtime_config
    from scripts.probe_strm_selectivity_real import replay_and_capture

    device = "cuda" if torch.cuda.is_available() else "cpu"
    zhead_path = f"{ZHEAD_DIR}/bilinear_s{SEED}/final.pt"
    if not Path(zhead_path).exists():
        print(f"ERROR: z_head not found: {zhead_path}", file=sys.stderr)
        return 1
    print(f"[c3] seed={SEED} device={device} z_head={zhead_path}", flush=True)
    print(f"[c3] backbone={BACKBONE}", flush=True)

    heldout_ep = json.loads(EP_HELDOUT.read_text(encoding="utf-8"))
    heldout_g = json.loads(GOLD_HELDOUT.read_text(encoding="utf-8"))
    trained_ep = json.loads(EP_TRAINED.read_text(encoding="utf-8"))
    trained_g = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))

    heldout_sids = list(heldout_ep.keys())
    loo_sids = _loo_sessions(trained_g)
    print(f"[c3] heldout sessions={heldout_sids}  loo-8 sessions={loo_sids}",
          flush=True)

    # Write Onyx transcripts for exactly the sessions we grade (no extra replay).
    tmpdir = tempfile.mkdtemp(prefix="pondr_c3_")
    transcripts: list[str] = []
    sid_to_set: dict[str, str] = {}
    for sid in heldout_sids:
        transcripts.append(_write_transcript(sid, heldout_ep[sid], tmpdir))
        sid_to_set[sid] = "heldout-17"
    for sid in loo_sids:
        if sid in trained_ep:
            transcripts.append(_write_transcript(sid, trained_ep[sid], tmpdir))
            sid_to_set[sid] = "loo-8-normal"
    print(f"[c3] wrote {len(transcripts)} transcripts to {tmpdir}", flush=True)

    # Flip the global singleton so the orchestrator emits scoreable conv slots
    # (the same flag the probe's own --strm-ring-text CLI flips). NO src/ edit.
    _runtime_config.strm_ring_text = True

    turn_records, run_stats = replay_and_capture(
        transcripts=transcripts,
        backbone_path=BACKBONE,
        rel_head_path=REL_HEAD,
        z_relevance_head_path=zhead_path,
        z_head_arch="composite-raw",
        identity_instance=True,
        salience="off",
        doc_store=None,
        ring_capacity=RING_CAPACITY,
        max_turns=0,
        device=device,
        user_id="c3-probe",
        rec_head_path="",
        ld_head_path="",
        ablate_yt=False,
        emit_traces=None,
    )
    print(f"[c3] captured {len(turn_records)} turn-records "
          f"(encoded={run_stats.get('n_encoded')} skipped={run_stats.get('n_skipped')})",
          flush=True)

    # Dump a SLIM copy of the per-turn ring slots (source_id, slot_type, z_logit,
    # cos) so ranking logic (self-exclusion, dedup, type filters) can be iterated
    # OFFLINE without re-replaying the backbone (~10 min/run). The full slot dict
    # is NOT needed to score -- only the rankable fields. onyx PRIVATE: nothing
    # leaves scripts/_scratch/.
    slim = [{"session_id": r["session_id"], "turn_index": r["turn_index"],
             "slots": [{"source_id": s.get("source_id"),
                        "slot_type": s.get("slot_type"),
                        "z_logit": s.get("z_logit"), "cos": s.get("cos")}
                       for s in r.get("slots", [])]}
            for r in turn_records]
    RAW_PATH.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    print(f"[c3] wrote {RAW_PATH} ({len(slim)} turn-records, slim)", flush=True)

    # Partition records by set (a session belongs to exactly one set here).
    recs_by_set: dict[str, list[dict]] = {"heldout-17": [], "loo-8-normal": []}
    for rec in turn_records:
        s = sid_to_set.get(rec["session_id"])
        if s:
            recs_by_set[s].append(rec)

    out = {"seed": SEED, "device": device, "z_head": zhead_path,
           "backbone": BACKBONE, "ring_capacity": RING_CAPACITY,
           "budget": BUDGET, "n_turn_records": len(turn_records),
           "sets": {}}
    print("\n=== C3 (CompositeZHead, mixed ring conv+retrieved) ===", flush=True)
    out["sets"]["heldout-17"] = _score_set(
        recs_by_set["heldout-17"], heldout_ep, heldout_g["pairs"],
        int(heldout_g.get("window", 16)), int(heldout_g.get("age_threshold", 3)),
        None, "heldout-17")
    loo_pairs = [p for p in trained_g["pairs"] if p["session_id"] in set(loo_sids)]
    out["sets"]["loo-8-normal"] = _score_set(
        recs_by_set["loo-8-normal"], trained_ep, loo_pairs,
        int(trained_g.get("window", 16)), int(trained_g.get("age_threshold", 3)),
        None, "loo-8-normal")

    # Diagnostic: conv slots only (type 0). Undercounts target=0 pairs (turn 0
    # is retrieved-only) -- documented, not hidden.
    print("\n=== C3 diagnostic (conv slots only, slot_type 0) ===", flush=True)
    out["sets"]["heldout-17_conv0"] = _score_set(
        recs_by_set["heldout-17"], heldout_ep, heldout_g["pairs"],
        int(heldout_g.get("window", 16)), int(heldout_g.get("age_threshold", 3)),
        0, "heldout-17_conv0")
    out["sets"]["loo-8-normal_conv0"] = _score_set(
        recs_by_set["loo-8-normal"], trained_ep, loo_pairs,
        int(trained_g.get("window", 16)), int(trained_g.get("age_threshold", 3)),
        0, "loo-8-normal_conv0")

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[c3] wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())