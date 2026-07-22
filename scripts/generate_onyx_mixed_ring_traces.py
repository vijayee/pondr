"""Phase 1c (task #51): mixed-ring training traces from REAL Onyx sessions.

The decisive user instruction: "should it not also be trained on documents.
The production server will ingest them." At production serve the WM ring is
CONVERSATION messages + RECALLED doc-episodes (the orchestrator's retriever
injects the latter every turn). Head B (``CrossSlotTransformerZHead``, task
#45 winner) was trained ONLY on conversation rings (the existing
``traces_onyx_serve_hraw.pt`` from ``generate_onyx_serve_traces.py`` steps
prior messages via ``wm.step`` with NO retrieval), so it is OOD by
construction at serve -> the live gate fails (task #46/#47, H2 content
shift). D0.4b confirmed it: the fully-trained conversation Head B (final.pt)
gives z_logit median -7.80 on the live retrieved-doc ring (gate 2.0) ->
FAIL, with a wild bimodal distribution (p10=-20, p90=75) = classic OOD.

This script closes that gap by emitting training traces whose ring = the
REAL production mixed ring (conversation slots + retrieved doc-episodes),
so a Phase-1 retrained Head B trains on the distribution it will serve.

How (the cheapest route to the production distribution, per the approved
[[reflective-greeting-beacon]] plan): replay the orchestrator's real
retrieval for the 76 Onyx sessions. For each user turn, the orchestrator
(1) steps the prompt into the ring as a CONVERSATION slot (slot_type=0,
text=prompt, source_id="{session}#msg{turn}" -- enabled by the Phase 1a
``strm_ring_text`` flag) and (2) retrieves + injects doc-episodes as
RETRIEVED slots (slot_type=1, text=episode summary, source_id="{session}
__ep{i}"). The corpus each turn retrieves from = the growing set of PRIOR
message-pairs ingested as episodes (build_episode -> _encode_best_effort),
exactly the live probe's (``probe_strm_selectivity_real._replay``) corpus
construction for the held-out transcripts -- so train/serve corpus
distribution matches. One shared store across all training sessions
matches production (a query in session B may recall a doc ingested in
session A); the 2 held-out live sessions are EXCLUDED entirely (no
train/test corpus leakage).

Each emitted record is byte-identical to ``generate_lmsys_serve_traces.
_build_record`` (so the downstream loader ``probe_serve_composite_zrgate.
_load_serve_traces`` consumes it unchanged) PLUS a new ``slot_types`` field
([K] long, 0=conversation, 1=retrieved) the Phase-1 head's slot-type
embedding consumes. Gold = top-1-cos over the FULL mixed ring (bge cosine
between each slot's text-emb and the query), so the head learns to rank
BOTH slot types, not just one.

PRIVACY: identical to ``generate_onyx_serve_traces`` -- the input
``sessions.jsonl`` is the user's PRIVATE Onyx chat history (local +
gitignored, NEVER uploaded to HF). The OUTPUT records carry NO message
text (``--keep-question`` OFF by default): only bge embeddings + opaque
source_ids (session UUID + index, no PII) + raw SSM state + slot_types.
Standalone (never touches ``DEFAULT_BACKBONE_PATH`` / ``build_ponder`` /
``serve_ponder``); loads the new backbone via ``--backbone``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the committed transcript-replay helpers (the same episode builder +
# encoder the 2d v2 harness + the live probe use -> the ring slots we
# capture are the SAME shape the live deploy produces).
from replay_chat_to_graduation import (  # noqa: E402
    _encode_best_effort,
    _iso,
    _pair_turns,
    build_episode,
)

from src.config import Phase2cConfig, config as _runtime_config  # noqa: E402
from src.orchestrator import PonderOrchestrator  # noqa: E402
from src.retrieval.query_planner import BonsaiQueryPlanner  # noqa: E402
from src.retrieval.retriever import HippocampalRetriever  # noqa: E402
from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.latent_dynamics_head import LatentDynamicsHead  # noqa: E402
from src.subconscious.training.routing_training import (  # noqa: E402
    build_embedder,
    load_backbone,
)

DEFAULT_BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full.pt"
DEFAULT_SESSIONS = "data/training/strm_graduation/sessions.jsonl"
DEFAULT_OUT = "data/training/strm_relevance/traces_onyx_mixed_ring_hraw.pt"
# MUST match probe_head_to_head_onyx.DEFAULT_LIVE_EVAL_SESSION_IDS exactly so
# the held-out live-eval sessions are the SAME ones D0.4a/D0.4b held out --
# never train on them, never ingest their episodes into the corpus.
DEFAULT_LIVE_EVAL_SESSION_IDS = (
    "682afdd9-e8ea-4258-a329-65f67b5d27d5",
    "69e17901-9c6c-4375-a6f1-736e95e1d316",
)
D_STATE = 16
D_MODEL = 384


class _StubModeA:
    """No LLM round-trip (mirror probe_strm_selectivity_real._StubModeA). The
    generator captures the WM ring, not synthesis."""

    def _complete(self, messages, tools=None, tool_choice=None):
        return ("[gen-stub-response]", None)


def _ordered_messages(session: dict) -> list[dict]:
    """A session's messages in conversation order (stable sort on time_sent).

    Robust to regeneration-branch reordering the Onyx API may return; stable
    sort preserves the fetcher's insertion order among equal timestamps =
    the true conversation order for a linear chat. Mirrors
    ``generate_onyx_serve_traces._ordered_messages``.
    """
    msgs = list(session.get("messages", []))

    def _key(m: dict) -> str:
        return str(m.get("time_sent") or "")

    return sorted(msgs, key=_key)


def _msg_text(m: dict) -> str:
    """Onyx message content (``text`` field); None -> "" for system msgs."""
    t = m.get("text", "")
    if t is None:
        return ""
    return str(t)


def _session_turns(session: dict, min_msg_chars: int) -> tuple[str, list[dict]]:
    """Reduce a session to ordered ``{"role","content"}`` turns (user +
    assistant only, non-empty). Mirrors ``load_transcript_threads`` but reads
    a sessions.jsonl record instead of a docs/*.json transcript. Returns
    (session_id, turns)."""
    session_id = str(session.get("session_id") or "")
    msgs = _ordered_messages(session)
    turns = [
        {"role": m.get("role", "?"), "content": _msg_text(m)}
        for m in msgs
        if m.get("role") in ("user", "assistant")
        and len(_msg_text(m).strip()) >= min_msg_chars
    ]
    return session_id, turns


def _infer_slot_type(source_id: str | None, slot_type: int | None) -> int:
    """Slot type from the recorded field, with a source_id-prefix fallback.

    The Phase-1a ``RingSlot.slot_type`` is the source of truth (0=conversation
    via ``update``, 1=retrieved via ``inject``). The prefix fallback (``#`` ->
    0, ``__ep`` -> 1) covers slots recorded before Phase 1a (old checkpoints)
    or any path that leaves the field None -- never silently default, so a
    mis-tagged slot surfaces rather than corrupts the type embedding.
    """
    if slot_type is not None:
        return int(slot_type)
    if not source_id:
        return 0
    # Conversation slots: "{session_id}#msg{turn}". Retrieved: "{...}__ep{NNNN}".
    if "__ep" in source_id:
        return 1
    if "#msg" in source_id:
        return 0
    # Unknown prefix -> raise so the mis-tag is visible (de-wonk: never default
    # silently on a training label).
    raise ValueError(f"cannot infer slot_type for source_id={source_id!r}")


def _build_mixed_record(ring, query_emb, ld_head, slot_doc_embs,
                        slot_source_ids, slot_slot_types,
                        emit_question: bool, question: str) -> dict | None:
    """Build ONE fit_relevance-format record from the current MIXED ring.

    ``slot_doc_embs`` / ``slot_source_ids`` / ``slot_slot_types`` are the
    caller's per-ring-slot parallel lists (aligned with ``ring``) -- the
    caller embeds each slot's text once and threads provenance + slot_type
    through, so this fn does NO embedding (pure tensor assembly). Self-
    contained (does NOT reuse ``_build_record`` then append) so the ``kept``
    filter + the slot_types alignment are computed in ONE pass -- a reuse-
    then-append would risk a kept-filter mismatch (a de-wonk hazard).

    Byte-identical to ``generate_lmsys_serve_traces._build_record`` for the
    shared fields (same shapes/dtypes the downstream loader expects), PLUS
    ``slot_types`` [K'] long. Drops slots whose Phase-A state capture
    (``slot.h``) is missing (a None here would be a Phase-A regression) and
    slots with no text (the scorer's own filter; with ``strm_ring_text`` ON
    every slot has text, but a retrieved episode with an empty summary is
    skipped rather than scored on an empty embedding).
    """
    if len(ring) < 3:
        return None
    kept = [(j, s) for j, s in enumerate(ring)
            if getattr(s, "h", None) is not None
            and s.text is not None and str(s.text).strip()]
    if len(kept) < 3:
        return None
    q = query_emb.to("cpu").to(torch.float32).reshape(-1)
    qn = q / (q.norm() + 1e-9)
    cos_vals = torch.empty(len(kept), dtype=torch.float32)
    for k, (j, _s) in enumerate(kept):
        # ``slot_doc_embs[j]`` arrives on the backbone device (CUDA when
        # device='auto') -- the embedder returns there. CPU-cast BEFORE the
        # dot product (``q`` is already CPU); a cross-device ``torch.dot``
        # raises (the bug the first smoke run caught). Mirrors the lmsys
        # generator's ``doc_embs`` which it pre-stores CPU-side.
        d = slot_doc_embs[j].to("cpu").to(torch.float32).reshape(-1)
        dn = d / (d.norm() + 1e-9)
        cos_vals[k] = float(torch.dot(dn, qn).item())
    labels = torch.zeros(len(kept), dtype=torch.float32)
    labels[int(cos_vals.argmax().item())] = 1.0              # top-1-cos = gold
    slots_z = torch.stack([
        ld_head.project(s.h).squeeze(0).detach().to("cpu").to(torch.float32)
        for _j, s in kept
    ])                                                        # [K', 384]
    slots_y = torch.stack([
        s.y.detach().to("cpu").to(torch.float32).squeeze(0).reshape(-1)
        for _j, s in kept
    ])                                                        # [K', 256]
    slots_doc_emb = torch.stack([
        slot_doc_embs[j].to("cpu").to(torch.float32).squeeze(0).reshape(-1)
        for j, _s in kept
    ])                                                        # [K', 384]
    slots_h_raw = torch.stack([
        torch.stack([
            layer.detach().to("cpu").to(torch.float16).reshape(D_STATE, D_MODEL)
            for layer in s.h
        ])
        for _j, s in kept
    ])                                                        # [K',4,16,384] fp16
    slot_types = torch.tensor(
        [_infer_slot_type(slot_source_ids[j], slot_slot_types[j])
         for j, _s in kept], dtype=torch.long)                 # [K']
    rec = {
        "query_emb": q, "slots_y": slots_y, "slots_z": slots_z,
        "labels": labels,
        "source_ids": [slot_source_ids[j] for j, _s in kept],
        "cos": cos_vals, "slots_doc_emb": slots_doc_emb,
        "slots_h_raw": slots_h_raw,
        "slot_types": slot_types,                              # Phase 1c: NEW
    }
    if emit_question:
        rec["question"] = question
    return rec


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 1c: mixed-ring (conversation + retrieved-doc) "
                    "training traces from REAL Onyx sessions (task #51).")
    p.add_argument("--sessions", default=DEFAULT_SESSIONS,
                   help="Onyx sessions.jsonl (scripts/_scratch/_fetch_onyx_cookie.py)")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="path to the new backbone (default backbone_v2_full.pt).")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--live-eval-sessions", default=",".join(DEFAULT_LIVE_EVAL_SESSION_IDS),
                   help="comma-separated Onyx session UUIDs to HOLD OUT (never "
                        "train, never ingest their episodes). Default = the 2 "
                        "live-transcript sessions matching D0.4a/D0.4b. Empty "
                        "string disables (NOT recommended -- trains on test).")
    p.add_argument("--min-user-turns", type=int, default=3,
                   help="min user turns (ring>=3) to keep a session; sessions "
                        "below this yield zero traces and are skipped.")
    p.add_argument("--min-msg-chars", type=int, default=5,
                   help="skip messages shorter than this (empty replies).")
    p.add_argument("--ring-capacity", type=int, default=16,
                   help="STRM ring capacity; older slots FIFO-evict (matches the "
                        "live gate's ring_capacity=16).")
    p.add_argument("--max-sessions", type=int, default=0,
                   help="cap training sessions emitted (0 = all; dev speed knob).")
    p.add_argument("--device", default="auto")
    p.add_argument("--keep-question", action="store_true",
                   help="retain the 'question' field (message text) on disk. "
                        "OFF by default -- PRIVATE chats; only embeddings + "
                        "source_ids + raw state + slot_types are written.")
    args = p.parse_args()

    sessions_path = Path(args.sessions)
    if not sessions_path.exists():
        print(f"ERROR: sessions.jsonl not found at {sessions_path}\n"
              f"  run: python scripts/_scratch/_fetch_onyx_cookie.py",
              file=sys.stderr)
        return 1
    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"ERROR: backbone not found at {backbone_path}", file=sys.stderr)
        return 1

    live_eval_ids = frozenset(
        s for s in (args.live_eval_sessions or "").split(",") if s.strip())
    print(f"Loading sessions from {sessions_path}", flush=True)
    sessions = [json.loads(l) for l in sessions_path.read_text(encoding="utf-8")
                .splitlines() if l.strip()]
    print(f"  {len(sessions)} sessions total", flush=True)
    n_held = sum(1 for s in sessions
                 if str(s.get("session_id") or "") in live_eval_ids)
    print(f"  holding out {n_held} live-eval sessions (excluded from train + "
          f"corpus): {sorted(live_eval_ids)}", flush=True)

    print(f"Loading backbone from {backbone_path} (device {args.device})",
          flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    print(f"  {sum(p.numel() for p in backbone.parameters()):,} params", flush=True)
    embedder = build_embedder("on-demand")
    ld_head = LatentDynamicsHead()                          # parameter-free project
    emit_question = args.keep_question

    # Phase 1a: turn ON the orchestrator's prompt-step provenance so
    # conversation slots carry text + source_id + slot_type=0 and survive the
    # scorer's ``text is not None`` filter -> the scored ring is the FULL mixed
    # ring (conversation + retrieved docs), the production distribution. The
    # flag is a runtime singleton read by the orchestrator; set it before any
    # query() call. (This script is standalone + standalone-configured; the
    # default-OFF path in build_ponder/serve_ponder is untouched.)
    _runtime_config.strm_ring_text = True
    print(f"  strm_ring_text=True (Phase 1a: conversation slots scoreable)",
          flush=True)

    # One shared store + retriever across ALL training sessions matches the
    # production distribution (the personal corpus grows across sessions; a
    # query in session B may recall a doc ingested in session A). The 2 held-
    # out live sessions are excluded entirely -> no train/test corpus leakage.
    import tempfile  # local import; the store needs a real temp dir
    tmpdir = tempfile.mkdtemp(prefix="pondr_mixed_ring_")
    from src.memory.store import HippocampalStore  # noqa: E402
    db_path = str(Path(tmpdir) / "db")
    store = HippocampalStore(db_path)
    planner = BonsaiQueryPlanner(endpoint=None)  # None -> rule-based fallback
    retriever = HippocampalRetriever(
        store, planner=planner, auto_load_index=True,
        retrieval_gate=None, embedder=embedder,
    )
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(Path(tmpdir) / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone, embedder=embedder,
        mode_a=_StubModeA(), config=cfg, user_id="pondr_mixed_ring",
        encoder=None, relevance_head=None, ring_capacity=args.ring_capacity,
        identity_instance=True,
    )

    records: list[dict] = []
    n_sessions_kept = 0
    n_sessions_skipped = 0
    n_sessions_held = 0
    n_user_turns_total = 0
    n_turns_no_ring = 0
    n_turns_query_fail = 0
    t0 = time.time()

    for s_idx, session in enumerate(sessions):
        if args.max_sessions and n_sessions_kept >= args.max_sessions:
            print(f"  hit max_sessions {args.max_sessions}, stopping", flush=True)
            break
        session_id, turns = _session_turns(session, args.min_msg_chars)
        if not session_id:
            n_sessions_skipped += 1
            continue
        if session_id in live_eval_ids:
            n_sessions_held += 1
            continue  # never train, never ingest
        pairs = _pair_turns(turns)
        user_turn_count = sum(1 for u, _a in pairs if len(u.strip()) >= args.min_msg_chars)
        if user_turn_count < args.min_user_turns:
            n_sessions_skipped += 1
            continue

        # Per-session WM ring reset (the ring is per-conversation); the STORE
        # persists across sessions (the production corpus). Each session's
        # conversation slots get a session-scoped source_id, so the monotonic
        # ``_strm_ring_text_turn_counter`` never collides across sessions.
        orch.working_memory.reset()
        orch.user_id = session_id
        history: list[dict] = []
        # Seed: encode turn 0 so query 1 has memory to recall (the live probe's
        # pattern). ``epoch_base`` advances per session so timestamps are
        # globally ordered + session-scoped (mirrors the live probe).
        epoch_base = float(s_idx) * 1e6
        u0, a0 = pairs[0]
        ep0 = build_episode(
            f"{session_id}__ep0000", u0, a0, timestamp=_iso(epoch_base, 0),
            user_id="pondr_mixed_ring", session_id=session_id, embedder=embedder)
        _encode_best_effort(store, ep0, session_id, 0)
        history.append({"role": "user", "content": u0})
        history.append({"role": "assistant", "content": a0})

        sess_user_turns = 0
        for i in range(1, len(pairs)):
            u, a = pairs[i]
            if len(u.strip()) < args.min_msg_chars:
                continue
            try:
                orch.query(u, conversation_history=list(history),
                           auto_persist=False, signal="routine")
            except Exception as e:  # noqa: BLE001 - one bad turn must not kill the run
                print(f"  [query-fail] session={session_id} turn={i}: {e}",
                      file=sys.stderr)
                n_turns_query_fail += 1
            # Score the ring NOW (after the query step's conversation slot +
            # the retrieved-episode injects populate text-bearing slots).
            # prompt_emb is re-derived from the user text (deterministic --
            # same embed call query() uses internally).
            ring = orch.working_memory.ring_buffer()
            if len(ring) < 3:
                n_turns_no_ring += 1
            else:
                prompt_emb = orch.working_memory.embed([u])[0]
                slot_texts = [s.text if s.text is not None else "" for s in ring]
                # Embed every slot's text once (CPU bge; K is small). Empty-
                # text slots are dropped inside _build_mixed_record.
                try:
                    slot_doc_embs = orch.working_memory.embed(slot_texts)
                except Exception as e:  # noqa: BLE001
                    print(f"  [embed-skip] session={session_id} turn={i}: {e}",
                          file=sys.stderr)
                    slot_doc_embs = [None] * len(ring)
                # Only keep slots whose embed succeeded (aligns with the
                # ring; ``_build_mixed_record`` re-filters on h+text).
                if any(e is None for e in slot_doc_embs):
                    # Drop the None-embedded slots from ring + parallel lists
                    # so the indices stay aligned.
                    keep_idx = [k for k, e in enumerate(slot_doc_embs)
                                if e is not None]
                    ring = [ring[k] for k in keep_idx]
                    slot_doc_embs = [e for e in slot_doc_embs if e is not None]
                    slot_source_ids = [s.source_id for s in ring]
                    slot_slot_types = [s.slot_type for s in ring]
                else:
                    slot_source_ids = [s.source_id for s in ring]
                    slot_slot_types = [s.slot_type for s in ring]
                rec = _build_mixed_record(
                    ring, prompt_emb, ld_head, slot_doc_embs,
                    slot_source_ids, slot_slot_types, emit_question, u)
                if rec is not None:
                    records.append(rec)
                    n_user_turns_total += 1
                    sess_user_turns += 1
            # Ingest this turn's message-pair as an episode for FUTURE turns'
            # retrieval (the growing corpus). Best-effort (WaveDB pool bug).
            ep = build_episode(
                f"{session_id}__ep{i:04d}", u, a,
                timestamp=_iso(epoch_base, i), user_id="pondr_mixed_ring",
                session_id=session_id, embedder=embedder)
            _encode_best_effort(store, ep, session_id, i)
            history.append({"role": "user", "content": u})
            history.append({"role": "assistant", "content": a})

        n_sessions_kept += 1
        if (s_idx + 1) % 10 == 0 or s_idx == len(sessions) - 1:
            elapsed = time.time() - t0
            print(f"  [{s_idx + 1}/{len(sessions)}] kept {n_sessions_kept} "
                  f"sessions, {n_user_turns_total} user-turn records, "
                  f"held {n_sessions_held}, skipped {n_sessions_skipped} "
                  f"(no_ring={n_turns_no_ring} qfail={n_turns_query_fail}) "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print()
    print("=" * 64)
    print(f"sessions total:      {len(sessions)}")
    print(f"sessions held out:   {n_sessions_held}")
    print(f"sessions kept:        {n_sessions_kept}")
    print(f"sessions skipped:     {n_sessions_skipped} (< {args.min_user_turns} "
          f"user turns or no session_id)")
    print(f"user-turn records:    {n_user_turns_total}")
    print(f"  turns with ring<3:  {n_turns_no_ring}")
    print(f"  turns query-failed: {n_turns_query_fail}")
    if records:
        med_k = int(sorted(r['slots_h_raw'].shape[0] for r in records)[len(records)//2])
        # slot_type coverage: how many records have BOTH types (the production
        # mix). A record with only one type = a pure ring (no retrieval fired
        # or no conversation yet) -> log so the mix is visible, never silent.
        mixed = sum(1 for r in records
                    if int((r['slot_types'] == 0).sum()) > 0
                    and int((r['slot_types'] == 1).sum()) > 0)
        only_conv = sum(1 for r in records if int((r['slot_types'] == 1).sum()) == 0)
        only_ret = sum(1 for r in records if int((r['slot_types'] == 0).sum()) == 0)
        print(f"  median ring K:      {med_k}")
        print(f"  mixed (conv+ret):   {mixed}")
        print(f"  conv-only records:  {only_conv}")
        print(f"  retrieved-only:     {only_ret}")
    print(f"elapsed:              {elapsed:.0f}s")
    print("=" * 64)
    if not records:
        print("ERROR: no records built", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {len(records)} records -> {out_path} ({size_mb:.1f} MB)",
          flush=True)
    summary_path = out_path.with_suffix(".yield.json")
    summary_path.write_text(json.dumps({
        "n_sessions": len(sessions), "n_sessions_kept": n_sessions_kept,
        "n_sessions_held": n_sessions_held, "n_sessions_skipped": n_sessions_skipped,
        "n_records": len(records), "n_turns_no_ring": n_turns_no_ring,
        "n_turns_query_fail": n_turns_query_fail,
        "live_eval_session_ids": sorted(live_eval_ids),
        "min_user_turns": args.min_user_turns, "ring_capacity": args.ring_capacity,
        "backbone": str(backbone_path), "sessions_src": str(sessions_path),
        "elapsed_s": round(elapsed, 1), "keep_question": emit_question,
        "strm_ring_text": True,
    }, indent=2), encoding="utf-8")
    print(f"wrote yield summary -> {summary_path}", flush=True)
    # Best-effort tempdir cleanup (de-wonk: the store's WaveDB files live in
    # tmpdir; on Windows they may still be locked at process exit, so ignore
    # errors -- the OS clears its temp dir eventually. Never fail the run on
    # cleanup.
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())