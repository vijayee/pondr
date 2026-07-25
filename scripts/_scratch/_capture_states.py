"""Build 2 -- capture raw SSM states + query embeddings for the LOO-8 train set.

The state-capture half of the distillation dataset. Replays each LOO-8 session
through the REAL orchestrator+ring (``replay_and_capture``, IMPORTED read-only)
with ``emit_traces=<path>`` + ``emit_raw_state=True`` so every scoreable turn
emits a trace record carrying the raw per-layer SSM state per slot
(``slots_h_raw [K',4,16,384]`` fp16) + the bge query embedding (``query_emb
[384]``) + the slot ``source_ids``. Build 3 joins these states with the dense
qwen teacher labels (Build 1) by ``(session_id, q)`` and trains the
``CrossSlotTransformerZHead`` to predict the teacher's relevance scores FROM THE
STATES -- the crux test of whether the SSM encodes query-relevance.

PROVENANCE RECOVERY (the key). The trace records are popped from the capture
record BEFORE ``session_id``/``turn_index`` are stamped
(``probe_strm_selectivity_real.py:890`` vs ``:893``), and
``_build_serve_trace`` does not include ``slot_types`` -- so a trace record has
no session_id, no query-turn index, no slot_type. We recover all three from the
slot ``source_ids`` themselves, which ENCODE them:
  * conv slot    ``{session_id}#msg{N}``   -> slot_type 0, turn N
  * retrieved ep ``{session_id}__ep{N:04d}`` -> slot_type 1, turn N
  * session_id   = the prefix before ``#msg`` / ``__ep``.

FOREIGN RECALL (why a naive recovery is wrong). ``replay_and_capture`` reuses
ONE orchestrator+store across all sessions; ``working_memory.reset()`` clears
the ring but NOT the episode store, so a later session's ring contains
retrieved ``__ep`` episodes from PRIOR sessions (foreign session_id prefix).
These are serve-faithful distractors (kept in the ring; Build 3 labels them
0.0), but they defeat majority-vote session_id AND ``q=max(turn over all
slots)``. So both are recovered from the query's OWN ``#msg{q}`` conv slot
instead -- always present, always current-session (the ring is cleared between
sessions), always the highest ``#msg`` turn (FIFO):
  * q_global     = max(turn over slot_type-0 ``#msg`` slots only)
  * session_id   = the prefix of the ``#msg`` slot whose turn == q_global
  * slot_sids    = per-slot prefix (so Build 3 can label foreign slots 0.0)

GLOBAL COUNTER -> PER-SESSION OFFSET (the second wrinkle).
``_strm_ring_text_turn_counter`` is MONOTONIC ACROSS THE ORCHESTRATOR LIFETIME
-- it is NOT reset on ``load_session`` / ``working_memory.reset()``
(orchestrator.py:327), by design (the session_id prefix keeps source_ids
unique across sessions). So ``#msg{N}`` N is a GLOBAL counter, not a per-
session user-turn index: session 1's #msg run 1..n1-1, session 2's run
n1..n1+n2-2, .... The teacher labels per-session q in [1, n_ut). So the raw
global ``q_global`` / #msg slot turns DO NOT join the teacher keys for any
session after the first. The fix is a deterministic offset:
``offset_k = sum(n_ut_j - 1 for prior replayed sessions j)`` (turn 0 is seeded
without a query, so n_ut-1 queries -> n_ut-1 counter increments per session),
applied to BOTH ``q`` and every #msg slot turn:
  * q            = q_global - offset[session_id]   (per-session user-turn idx)
  * slot_turns[k]= #msg turn - offset  (slot_type 0 only; __ep turns are
                  already per-session 0-based for the slot's OWN session and
                  are left untouched -- foreign __ep are labeled 0.0 by Build
                  3 regardless of their turn value).
Recovered q is validated in [1, n_ut); any out-of-range record is dropped
loudly (would mean the offset assumption is wrong -- e.g. a session skipped or
the replay order changed). This is a join by ``source_id``->turn, NOT by row
position -- robust to the skipped-turn gaps (K<3 turns emit no trace, so
trace_records is NOT positionally aligned with turn_records).

NO engine change. ``replay_and_capture`` is IMPORTED read-only (never modified);
only the global ``_runtime_config.strm_ring_text`` singleton is flipped (the
same flag the C3 probe + the probe's own CLI flip) so the orchestrator emits
scoreable conv slots. onyx PRIVATE -- nothing leaves scripts/_scratch/. No
uploads. Per CLAUDE.md de-wonk at completion.

Run (GPU; ~10 min for 8 sessions):
  PYTHONPATH=. python scripts/_scratch/_capture_states.py
  (env: SCOPE=loo8 default|trained for Build 5; SEED picks the bilinear z_head
  used only for the z_logit measurement path -- the emitted STATES are
  z_head-independent.)
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
EP_TRAINED = SCRATCH / "_trained_episodes_for_labeling.json"
GOLD_TRAINED = SCRATCH / "_trained_gold.json"

BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full_finetuned_text2x.pt"
REL_HEAD = "data/training/strm_relevance/best.pt"
ZHEAD_DIR = "data/training/strm_state_readout/head_to_head_onyx"

SCOPE = os.getenv("SCOPE", "loo8")          # loo8 (default) | trained (all 27)
LOO_MIN_PAIRS = 6
RING_CAPACITY = 16
SEED = int(os.getenv("SEED", "0"))

# z_head_arch="composite-raw" matches the C3 probe (a known-working config). The
# emitted states are z_head-INDEPENDENT (emit happens in _build_serve_trace
# regardless of z_head_arch); the z_head only scores z_logit on the measurement
# path, which we do not consume here. Kept only so the call is a proven config.
Z_HEAD_ARCH = "composite-raw"
Z_HEAD_PATH = f"{ZHEAD_DIR}/bilinear_s{SEED}/final.pt"

EMIT_PATH = SCRATCH / (
    "_states_emit_loo8.pt" if SCOPE == "loo8" else "_states_emit_trained27.pt")
OUT_PATH = SCRATCH / (
    "_states_loo8.pt" if SCOPE == "loo8" else "_states_trained27.pt")

_MSG_RE = re.compile(r"#msg(\d+)$")
_EP_RE = re.compile(r"__ep(\d+)$")


def _user_turns(sess: dict) -> list[str]:
    return [t["text"] for t in sess["turns"] if t.get("role") == "user"]


def _n_replay_user_turns(sess: dict) -> int:
    """User turns that actually make it into the transcript (role=user AND
    non-empty text). The orchestrator's ``_strm_ring_text_turn_counter``
    increments once per ``query()`` call, and ``query()`` is called for each
    transcript user turn i in [1, len(pairs)), so the counter advances
    ``len(pairs) - 1`` times per session = this count - 1. Used for the global
    counter -> per-session offset (NOT ``_user_turns``, which counts role=user
    turns that may lack text and never reach the transcript)."""
    return sum(1 for t in sess["turns"]
               if t.get("role") == "user" and t.get("text"))


def _loo_sessions(gold: dict) -> list[str]:
    by_sid: dict[str, int] = {}
    for p in gold["pairs"]:
        by_sid[p["session_id"]] = by_sid.get(p["session_id"], 0) + 1
    return sorted(sid for sid, n in by_sid.items() if n >= LOO_MIN_PAIRS)


def _trained_sessions(gold: dict) -> list[str]:
    return sorted({p["session_id"] for p in gold["pairs"]})


def _write_transcript(sid: str, sess: dict, tmpdir: str) -> str:
    """Episode JSON -> Onyx transcript on disk, preserving interleaved order so
    ``_pair_turns``[i] == gold user-turn i (verbatim from the C3 probe)."""
    msgs = [{"message_type": t["role"], "message": t["text"]}
            for t in sess["turns"] if t.get("text")]
    path = Path(tmpdir) / f"{sid}.json"
    path.write_text(json.dumps({"chat_session_id": sid, "messages": msgs},
                               ensure_ascii=False), encoding="utf-8")
    return str(path)


def _slot_turn(source_id: str) -> int | None:
    """Map a slot source_id to its user-turn index (``#msg{N}`` or ``__ep{N}``)."""
    if not source_id:
        return None
    m = _MSG_RE.search(source_id)
    if m:
        return int(m.group(1))
    m = _EP_RE.search(source_id)
    if m:
        return int(m.group(1))
    return None


def _slot_type(source_id: str) -> int | None:
    """0 = conv (``#msg``), 1 = retrieved episode (``__ep``). None = neither
    (a foreign recalled episode -- not present in ephemeral doc_store=None mode
    but guarded anyway)."""
    if not source_id:
        return None
    if _MSG_RE.search(source_id):
        return 0
    if _EP_RE.search(source_id):
        return 1
    return None


def _session_id_of(source_id: str) -> str | None:
    """Recover the session_id prefix from a slot source_id. ``{sid}#msg{N}`` ->
    sid; ``{sid}__ep{N}`` -> sid. None if neither (foreign recall)."""
    if not source_id:
        return None
    if "#msg" in source_id:
        return source_id.split("#msg", 1)[0]
    if "__ep" in source_id:
        return source_id.split("__ep", 1)[0]
    return None


def _augment(trace_recs: list[dict], offset_by_sid: dict,
             n_ut_by_sid: dict) -> tuple[list[dict], int, int]:
    """Attach the TRUE ``session_id``, ``q`` (per-session query turn), and
    per-slot ``slot_types``/``slot_sids``/``slot_turns`` to each trace record,
    recovered from ``source_ids``. Drops a record only if NO ``#msg``
    (slot_type 0) slot is present (none expected -- the orchestrator always
    adds the query's own ``#msg{q}`` during ``query()``) or if the recovered
    per-session q is out of [1, n_ut) (the offset assumption is wrong).

    FOREIGN RECALL (the wrinkle that defeats a naive recovery).
    ``replay_and_capture`` reuses ONE orchestrator+store across all sessions;
    ``working_memory.reset()`` clears the ring (``_ring.clear()``) but NOT the
    episode store, so a later session's ring contains RETRIEVED episodes from
    PRIOR sessions (``__ep{N}`` with a foreign session_id prefix). These cross-
    session episodes are SERVE-FAITHFUL distractors -- at real serve the
    retriever recalls the user's full prior history -- so they STAY in the ring
    (Build 3 labels them 0.0, the within-session anaphora target they are not).
    But they break two naive recovery rules:

      * majority-vote the session_id prefix -> WRONG when foreign recalled
        episodes outnumber the current session's slots (a record whose ring is
        4/6 prior-session ``__ep`` majority-votes to the PRIOR session,
        mis-attributing it).
      * q = max(turn over ALL slots) -> WRONG when a foreign ``__ep{N}`` has a
        higher turn index than the current query (a prior 180-turn session's
        ``__ep0180`` in a 5-turn query's ring hijacks q).

    THE FIX (foreign recall). The query's OWN ``#msg{q}`` conv slot is ALWAYS
    present (added in ``query()``), ALWAYS from the current session (the ring
    is cleared between sessions so no foreign ``#msg`` survives), and ALWAYS
    the highest ``#msg`` turn in the ring (FIFO -- the query is the newest
    user turn). So:

      * q_global = max(turn over slot_type-0 ``#msg`` slots ONLY) -- foreign
        ``__ep`` are slot_type 1, excluded, so a high foreign turn cannot
        hijack q.
      * session_id = the prefix of the ``#msg`` slot whose turn == q_global.

    THE FIX (global counter). ``_strm_ring_text_turn_counter`` is monotonic
    across the orchestrator lifetime (NOT reset between sessions), so
    ``#msg{N}`` N is a GLOBAL counter, not a per-session user-turn index. The
    teacher labels per-session q. So ``q = q_global - offset[session_id]`` and
    every #msg slot turn is shifted by the same offset. __ep turns are already
    per-session 0-based for the slot's own session and are LEFT UNCHANGED
    (foreign __ep are labeled 0.0 by Build 3 regardless of turn).

    Per-slot ``slot_sids`` (recovered prefix per slot) is stored so Build 3 can
    spot foreign slots (``slot_sids[k] != session_id``) and label them 0.0."""
    out = []
    n_dropped = 0
    n_oob = 0
    for rec in trace_recs:
        sids = rec.get("source_ids", [])
        slot_turns = [_slot_turn(s) for s in sids]      # int | None per slot
        slot_types = [_slot_type(s) for s in sids]      # 0=#msg, 1=__ep, None
        slot_sids = [_session_id_of(s) for s in sids]   # prefix per slot
        # q_global + session_id from the #msg (slot_type 0) slots ONLY -- the
        # query's own #msg{q} is the highest-turn #msg in the ring and always
        # from the current session. Foreign __ep (slot_type 1) are excluded so
        # a high foreign turn index cannot hijack q.
        msg_turns = [t for t, st in zip(slot_turns, slot_types)
                     if st == 0 and t is not None]
        if not msg_turns:
            n_dropped += 1
            continue
        q_global = max(msg_turns)
        # the #msg slot at turn q_global -> its prefix is the true session_id
        q_sid = None
        for t, st, s in zip(slot_turns, slot_types, slot_sids):
            if st == 0 and t == q_global and s is not None:
                q_sid = s
                break
        if q_sid is None:
            n_dropped += 1
            continue
        offset = offset_by_sid.get(q_sid)
        if offset is None:
            n_dropped += 1
            continue
        q = q_global - offset               # global counter -> per-session idx
        n_ut = n_ut_by_sid.get(q_sid, 0)
        if not (1 <= q < n_ut):
            # offset assumption wrong (session skipped / replay order changed)
            n_oob += 1
            n_dropped += 1
            continue
        # #msg slot turns (global) -> per-session; __ep turns are already
        # per-session 0-based for the slot's own session -- leave them.
        slot_turns_ps = []
        for t, st in zip(slot_turns, slot_types):
            if st == 0 and t is not None:
                slot_turns_ps.append(t - offset)
            else:
                slot_turns_ps.append(t)
        rec["session_id"] = q_sid
        rec["q"] = q
        rec["slot_types"] = slot_types
        rec["slot_turns"] = slot_turns_ps
        rec["slot_sids"] = slot_sids
        out.append(rec)
    return out, n_dropped, n_oob


def main() -> int:
    if not EP_TRAINED.exists() or not GOLD_TRAINED.exists():
        print(f"ERROR: need {EP_TRAINED} and {GOLD_TRAINED}", file=sys.stderr)
        return 1
    if not Path(Z_HEAD_PATH).exists():
        print(f"ERROR: z_head not found: {Z_HEAD_PATH}", file=sys.stderr)
        return 1
    if not Path(BACKBONE).exists():
        print(f"ERROR: backbone not found: {BACKBONE}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT))
    import torch
    from src.config import config as _runtime_config
    from scripts.probe_strm_selectivity_real import replay_and_capture

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ep = json.loads(EP_TRAINED.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_TRAINED.read_text(encoding="utf-8"))
    if SCOPE == "loo8":
        sids = _loo_sessions(gold)
    elif SCOPE == "trained":
        sids = _trained_sessions(gold)
    else:
        print(f"ERROR: SCOPE={SCOPE} not in (loo8, trained)", file=sys.stderr)
        return 1
    print(f"[states] scope={SCOPE} sessions={len(sids)} device={device}", flush=True)
    print(f"[states] backbone={BACKBONE}", flush=True)
    print(f"[states] z_head={Z_HEAD_PATH} (z_logit only; states are z_head-independent)",
          flush=True)

    # GLOBAL COUNTER -> per-session offset. ``_strm_ring_text_turn_counter``
    # is monotonic across the orchestrator lifetime (NOT reset between
    # sessions -- orchestrator.py:327), so ``#msg{N}`` N is a GLOBAL counter,
    # not a per-session user-turn index. The teacher labels per-session q.
    # offset_k = sum(n_replay_j - 1 for prior REPLAYED sessions j) -- turn 0
    # is seeded without a query, so n_replay-1 queries -> n_replay-1 counter
    # increments per session. Computed over the sessions that actually have a
    # transcript (the ones replayed), in sorted/replay order.
    offset_by_sid: dict[str, int] = {}
    n_ut_by_sid: dict[str, int] = {}
    cum = 0
    for sid in sids:
        sess = ep.get(sid)
        if sess is None:
            continue                   # not replayed -> no counter advance
        n_ut = _n_replay_user_turns(sess)
        n_ut_by_sid[sid] = n_ut
        offset_by_sid[sid] = cum
        cum += n_ut - 1
    print(f"[states] counter offsets: "
          f"{ {s[:8]: offset_by_sid[s] for s in sorted(offset_by_sid)} }",
          flush=True)

    # Write Onyx transcripts for exactly the sessions we capture.
    tmpdir = tempfile.mkdtemp(prefix="pondr_states_")
    transcripts: list[str] = []
    for sid in sids:
        sess = ep.get(sid)
        if sess is None:
            print(f"  [{sid[:8]}] no transcript; skip", flush=True)
            continue
        transcripts.append(_write_transcript(sid, sess, tmpdir))
    print(f"[states] wrote {len(transcripts)} transcripts to {tmpdir}", flush=True)

    # Flip the global singleton so the orchestrator emits scoreable conv slots
    # (the same flag the C3 probe + the probe's own --strm-ring-text CLI flip).
    _runtime_config.strm_ring_text = True

    _run, run_stats = replay_and_capture(
        transcripts=transcripts,
        backbone_path=BACKBONE,
        rel_head_path=REL_HEAD,
        z_relevance_head_path=Z_HEAD_PATH,
        z_head_arch=Z_HEAD_ARCH,
        identity_instance=True,
        salience="off",
        doc_store=None,
        ring_capacity=RING_CAPACITY,
        max_turns=0,
        device=device,
        user_id="states-capture",
        rec_head_path="",
        ld_head_path="",
        ablate_yt=False,
        emit_traces=str(EMIT_PATH),
        emit_raw_state=True,
    )
    n_emit = run_stats.get("n_trace_records")
    print(f"[states] emitted {n_emit} trace records -> {EMIT_PATH}", flush=True)
    if not EMIT_PATH.exists():
        print("ERROR: emit file not written", file=sys.stderr)
        return 1

    trace_recs = torch.load(EMIT_PATH, weights_only=False)
    print(f"[states] loaded {len(trace_recs)} trace records", flush=True)

    # Structural sanity on the first record (Verification step 2).
    if trace_recs:
        r0 = trace_recs[0]
        sh = r0.get("slots_h_raw")
        qe = r0.get("query_emb")
        sids = r0.get("source_ids", [])
        print(f"[states] sanity rec0: slots_h_raw={tuple(sh.shape)} "
              f"dtype={sh.dtype} query_emb={tuple(qe.shape)} "
              f"n_source_ids={len(sids)}", flush=True)
        # K' >= 3 on most turns; report the distribution.
        ks = [int(r["slots_h_raw"].shape[0]) for r in trace_recs
              if "slots_h_raw" in r]
        print(f"[states] K' distribution: min={min(ks)} max={max(ks)} "
              f"mean={sum(ks)/len(ks):.1f}  (turns with K'<3 were skipped at emit)",
              flush=True)

    aug, n_dropped, n_oob = _augment(trace_recs, offset_by_sid, n_ut_by_sid)
    print(f"[states] augmented {len(aug)} records (dropped {n_dropped} "
          f"unparseable, {n_oob} out-of-range q)", flush=True)
    if n_oob:
        print(f"[states] WARNING: {n_oob} records had per-session q outside "
              f"[1, n_ut) -- the counter-offset assumption is wrong (a session "
              f"skipped or the replay order changed). Investigate before Build 3.",
              file=sys.stderr, flush=True)

    # Coverage: how many (session_id, q) keys did we capture? (Build 3 joins
    # these with the teacher labels.)
    keys = {(r["session_id"], r["q"]) for r in aug}
    print(f"[states] unique (session_id, q) keys: {len(keys)}", flush=True)
    by_sid = {}
    for r in aug:
        by_sid[r["session_id"]] = by_sid.get(r["session_id"], 0) + 1
    for sid in sorted(by_sid):
        print(f"  [{sid[:8]}] {by_sid[sid]} captured turns", flush=True)

    # Offload tensors to CPU before saving: replay ran on GPU so ``slots_h_raw``
    # / ``query_emb`` are cuda tensors; saving them cuda-bound would make the
    # .pt unloadable on a CPU-only host (and pin GPU memory on load). fp16
    # slots_h_raw + fp32 query_emb -> cpu is cheap and keeps Build 3 host-
    # agnostic (it .to(device) on load anyway).
    for r in aug:
        if "slots_h_raw" in r and r["slots_h_raw"].is_cuda:
            r["slots_h_raw"] = r["slots_h_raw"].cpu()
        if "query_emb" in r and r["query_emb"].is_cuda:
            r["query_emb"] = r["query_emb"].cpu()
    torch.save(aug, OUT_PATH)
    print(f"[states] wrote {OUT_PATH} ({len(aug)} records, tensors cpu-offloaded)",
          flush=True)
    # Clean up the raw emit file (the augmented OUT_PATH supersedes it).
    try:
        EMIT_PATH.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())