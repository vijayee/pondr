"""Task #45: build SERVE-distribution serve traces from REAL Onyx sessions.

DeepSeek-v4-pro's recommended head-to-head test (task #44 follow-up) needs
500-1000 REAL Onyx serve traces -- not the 114 turns from the 3 hand-exported
transcripts (``traces_serve_identity_hraw.pt``), and not lmsys (the supplement,
which task #44 showed transfers WORSE than BCE under the contrastive loss).
This script bridges the cookie-fetched Onyx sessions
(``data/training/strm_graduation/sessions.jsonl`` -- 76 sessions, 2457
messages, ~1024 user turns with ring>=3) into the SAME ``fit_relevance``-format
serve-trace records ``generate_lmsys_serve_traces.py`` emits, so the
downstream head-to-head probe (Head A bilinear vs Head B cross-slot
Transformer, identical contrastive InfoNCE loss, identical frozen
``backbone_v2_full.pt``) trains + evals on real Onyx in-distribution -- the
option-A path task #41 pointed to ("needs MORE Onyx serve transcripts"), now
armed with the better loss + enough data.

Mapping (one Onyx session = one serve session) -- byte-identical to the lmsys
generator + ``probe_strm_selectivity_real._build_serve_trace``:
  * Each PRIOR message (user OR assistant, non-empty, non-system) is ingested
    as a ring slot with ``source_id = "{session_id}#{msg_idx}"`` via
    ``WorkingMemory.step`` on the new backbone (identity-instance direct-SSM
    path -- the SAME path the existing Onyx serve traces + lmsys traces used).
    The WM ring persists across the session, so a message ingested at index
    ``j`` stays in the ring and is a candidate slot for every LATER user turn
    -> it recurses as a candidate, exactly the per-source z_r/z_logit gap
    structure.
  * At each USER turn ``i`` (ring has ``>= 3`` text-bearing slots), capture
    ONE ``fit_relevance``-format record: ``query_emb`` = bge(user msg ``i``)
    vs the ring of prior messages; ``labels`` = top-1-cos (the prior msg most
    bge-similar to this query); ``cos`` = continuous bge cosine per slot;
    ``slots_h_raw`` flat_last [K,4,16,384] fp16; ``slots_z`` = parameter-free
    ``LatentDynamicsHead.project(slot.h)`` [384]; ``slots_y`` = slot.y [256];
    ``slots_doc_emb`` = bge of each slot's text [384]; ``source_ids`` = the
    ring slots' source_ids.

Thread order: Onyx messages carry ``time_sent`` + a ``parent_message`` tree
link. The fetcher returns messages in insertion order, but to be robust
against any regeneration-branch reordering we sort each session's messages by
``time_sent`` before replaying (a stable sort preserves insertion order among
ties). Regenerated assistant branches just become extra ring slots -- harmless
for the relevance signal.

PRIVACY: the input ``sessions.jsonl`` is the user's PRIVATE Onyx chat history
(local + gitignored, NEVER uploaded to HF per user directive -- Onyx data must
be sanitized first). The OUTPUT records carry NO message text by default
(``--no-question`` is ON by default here, unlike the lmsys generator): only
bge embeddings + source_ids (session_id#index, no PII -- the session_id is an
opaque Onyx UUID) + the raw SSM state. ``--keep-question`` re-enables the debug
text field for local inspection only. This script is standalone (never touches
``DEFAULT_BACKBONE_PATH`` / ``build_ponder`` / ``serve_ponder``); loads the new
backbone via ``--backbone`` (default ``backbone_v2_full.pt``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Reuse the lmsys generator's _build_record so the record format is
# byte-identical by construction (same code path -> same fields, same dtypes,
# same shapes). scripts/ has no __init__.py; import as a top-level module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_lmsys_serve_traces import _build_record  # noqa: E402

from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.latent_dynamics_head import LatentDynamicsHead  # noqa: E402
from src.subconscious.training.routing_training import (  # noqa: E402
    _resolve_device,
    build_embedder,
    load_backbone,
)
from src.subconscious.working_memory import WorkingMemory  # noqa: E402

DEFAULT_BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full.pt"
DEFAULT_SESSIONS = "data/training/strm_graduation/sessions.jsonl"
DEFAULT_OUT = "data/training/strm_relevance/traces_onyx_serve_hraw.pt"


def _ordered_messages(session: dict) -> list[dict]:
    """Return a session's messages in conversation order.

    The fetcher returns messages in insertion order, but a stable sort on
    ``time_sent`` makes us robust to any regeneration-branch reordering the
    Onyx API might return (a regenerated assistant reply shares its
    parent_message with the stale one; insertion order can interleave them).
    Stable sort preserves the fetcher's insertion order among equal
    timestamps, which is the true conversation order for a linear chat.
    """
    msgs = list(session.get("messages", []))

    def _key(m: dict) -> str:
        ts = m.get("time_sent") or ""
        return str(ts)

    return sorted(msgs, key=_key)


def _msg_text(m: dict) -> str:
    """Onyx message content lives in the ``text`` field (the fetcher's schema).

    Onyx tool-call / system messages may carry non-string ``text`` (None for
    system prompts); coerce to str + strip so the ``min_msg_chars`` filter
    drops them.
    """
    t = m.get("text", "")
    if t is None:
        return ""
    return str(t)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build SERVE-distribution serve traces from REAL Onyx "
                    "sessions (task #45; data for the task #44 head-to-head).")
    p.add_argument("--sessions", default=DEFAULT_SESSIONS,
                   help="Onyx sessions.jsonl from scripts/_scratch/_fetch_onyx_cookie.py")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="path to the new backbone (default backbone_v2_full.pt).")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--min-user-turns", type=int, default=3,
                   help="min user turns (ring>=3) to keep a session; sessions "
                        "below this yield zero traces and are skipped.")
    p.add_argument("--min-msg-chars", type=int, default=5,
                   help="skip messages shorter than this (empty replies).")
    p.add_argument("--ring-capacity", type=int, default=32,
                   help="STRM ring capacity; older messages FIFO-evict in long sessions.")
    p.add_argument("--max-sessions", type=int, default=0,
                   help="cap sessions emitted (0 = all; dev speed knob).")
    p.add_argument("--device", default="auto")
    p.add_argument("--keep-question", action="store_true",
                   help="retain the 'question' field (message text) on disk. "
                        "OFF by default -- these are PRIVATE chats; only "
                        "embeddings + source_ids + raw state are written.")
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
    dev = _resolve_device(args.device)

    print(f"Loading sessions from {sessions_path}", flush=True)
    sessions = [json.loads(l) for l in sessions_path.read_text(encoding="utf-8")
                .splitlines() if l.strip()]
    print(f"  {len(sessions)} sessions", flush=True)

    print(f"Loading backbone from {backbone_path} (device {dev})", flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    print(f"  {sum(p.numel() for p in backbone.parameters()):,} params", flush=True)
    embedder = build_embedder("on-demand")
    wm = WorkingMemory(backbone, embedder=embedder,
                       ring_capacity=args.ring_capacity, identity_instance=True)
    ld_head = LatentDynamicsHead()                          # parameter-free project
    emit_question = args.keep_question

    records: list[dict] = []
    n_sessions_kept = 0
    n_sessions_skipped = 0
    n_user_turns_total = 0
    t0 = time.time()

    for s_idx, session in enumerate(sessions):
        if args.max_sessions and n_sessions_kept >= args.max_sessions:
            print(f"  hit max_sessions {args.max_sessions}, stopping", flush=True)
            break
        session_id = str(session.get("session_id") or f"session{s_idx}")
        msgs = _ordered_messages(session)
        # Pre-filter to text-bearing user/assistant messages (drop system +
        # empties); keep original indices for source_ids.
        usable = [(i, m) for i, m in enumerate(msgs)
                  if (m.get("role") in ("user", "assistant")
                      and len(_msg_text(m).strip()) >= args.min_msg_chars)]
        user_turn_count = sum(1 for _i, m in usable if m.get("role") == "user")
        if user_turn_count < args.min_user_turns:
            n_sessions_skipped += 1
            continue

        wm.reset()
        source_ids: list[str] = []
        doc_embs: list[torch.Tensor] = []
        # Pre-embed all usable messages once (cache by position in usable).
        texts = [_msg_text(m) for _i, m in usable]
        try:
            embs = wm.embed(texts)
        except Exception as e:  # noqa: BLE001 - embedding failure -> skip session
            print(f"  [embed-skip] session={session_id}: {e!r}", file=sys.stderr)
            n_sessions_skipped += 1
            continue

        sess_user_turns = 0
        for k, (orig_i, m) in enumerate(usable):
            role = m.get("role", "?")
            text = texts[k]
            # Capture at user turns (query) against the ring so far, BEFORE
            # ingesting this user msg, so the query is not its own slot.
            if role == "user" and len(text.strip()) >= args.min_msg_chars:
                if len(source_ids) >= 3:
                    rec = _build_record(
                        wm.ring_buffer(), embs[k], ld_head,
                        source_ids, doc_embs, text, emit_question)
                    if rec is not None:
                        records.append(rec)
                        n_user_turns_total += 1
                        sess_user_turns += 1
            # Ingest this message as a slot for future turns.
            v = embs[k]
            sid = f"{session_id}#{orig_i}"
            wm.step(v, source_id=sid, text=text)
            source_ids.append(sid)
            doc_embs.append(v.detach().to("cpu"))

        n_sessions_kept += 1
        if (s_idx + 1) % 10 == 0 or s_idx == len(sessions) - 1:
            elapsed = time.time() - t0
            print(f"  [{s_idx + 1}/{len(sessions)}] kept {n_sessions_kept} "
                  f"sessions, {n_user_turns_total} user-turn records "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"sessions total:   {len(sessions)}")
    print(f"sessions kept:    {n_sessions_kept} (skipped {n_sessions_skipped} "
          f"with < {args.min_user_turns} user turns or embed errors)")
    print(f"user-turn recs:   {n_user_turns_total}"
          + (f" (median K "
             f"{int(sorted((r['slots_h_raw'].shape[0] for r in records))[len(records)//2])})"
             if records else ""))
    print(f"elapsed:          {elapsed:.0f}s")
    print("=" * 60)
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
        "n_sessions_skipped": n_sessions_skipped,
        "n_records": len(records),
        "min_user_turns": args.min_user_turns,
        "ring_capacity": args.ring_capacity,
        "backbone": str(backbone_path),
        "sessions_src": str(sessions_path),
        "elapsed_s": round(elapsed, 1),
        "keep_question": emit_question,
    }, indent=2), encoding="utf-8")
    print(f"wrote yield summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())