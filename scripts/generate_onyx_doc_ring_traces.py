"""Phase 1f-2 (task #55): doc-ring training traces from REAL Onyx sessions.

The Phase 1c mixed-ring generator (``generate_onyx_mixed_ring_traces.py``) is
ILL-POSED for the cross-slot Transformer: its retrieved pool is conv-pair
episodes (``__ep{NNNN}``, text ``u_j + a_j``) which content-overlap with the
conversation slots (text ``u_j``) -> single-argmax gold is contradictory and
the transformer overfits the tie-breaking noise (DeepSeek section-2; see
[[pondr-strm-phase1d-self-match-rootcause]]). The decisive 1e sweep confirmed
it: the transformer stays flat (~0 z_logit) on the conv-pair mixed ring in
every config while bilinear recovers.

Phase 1f replaces the content-overlapping retrieved pool with REAL documents
(this repo's .md/.py/.txt, ingested once into a persisted store by
``build_doc_corpus_store.py``). The two slot types are now GENUINELY distinct
(chat messages vs external docs) -> single-argmax gold becomes well-posed ->
the transformer's type-conditioned cross-slot advantage should re-emerge.

Structurally a copy of the 1c generator with TWO changes:

1. PRE-LOOP: open the PERSISTED doc corpus store
   (``data/training/strm_relevance/doc_corpus_store``) instead of an empty
   tempdir store. The store already contains the ingested docs; the generator
   does NOT re-ingest. Attach ``DocumentRetriever`` when ``store_has_documents``
   (mirror production: ``runtime.build_ponder`` wires it then) so multi-section
   doc hits aggregate to ONE slot per doc -- without it the ring would carry
   one slot per section and the acceptance probe (which runs under
   ``build_ponder`` -> with aggregation) would be OOD.

2. PER-TURN LOOP: REMOVE the conv-pair episode seeding. The 1c generator grew
   the corpus each turn by ingesting ``build_episode(u_i, a_i)`` as an episode
   (``__ep{NNNN}``); that is the content-overlapping pool we are eliminating.
   The ONLY type-1 slots in the ring are the pre-ingested docs surfaced by
   ``HippocampalRetriever.retrieve``. WM still resets per session; the doc store
   still persists across sessions (matches production: a query in session B can
   recall a doc ingested once). ``encoder=None`` + ``auto_persist=False`` ensure
   the session's conversation turns are NOT encoded as episodes (which would
   re-introduce the conv-pair pool).

Everything else is identical to 1c: ``strm_ring_text=True``; per-turn
``orch.query(u, auto_persist=False)`` then ``ring = orch.working_memory.
ring_buffer()``; ``_build_mixed_record`` emits the SAME record schema the
downstream loader + ``probe_head_to_head_onyx.py`` consume. Gold = top-1-cos
over the FULL mixed ring (the trainer's ``--drop-self-slot`` removes the
self-slot at train time -- the self-match persists because ``orch.query`` still
adds the current prompt). The 2 held-out live sessions are EXCLUDED entirely.

PRIVACY: identical to 1c -- the input ``sessions.jsonl`` is the user's PRIVATE
Onyx chat history (local + gitignored, NEVER uploaded to HF). The OUTPUT
records carry NO message text (``--keep-question`` OFF by default): only bge
embeddings + opaque source_ids + raw SSM state + slot_types. Standalone (never
touches ``DEFAULT_BACKBONE_PATH`` / ``build_ponder`` / ``serve_ponder``).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Windows cp1252 console cannot encode some emoji GLiNER prints during model
# load -> UnicodeEncodeError -> extractor/embedder load can fail. Reconfigure
# stdio to UTF-8 (no-op on POSIX).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
# Reuse the committed transcript-replay helper (the pair-turns reduction the 1c
# generator uses). build_episode / _encode_best_effort are intentionally NOT
# imported -- the doc ring has NO conv-pair episode seeding.
from replay_chat_to_graduation import _pair_turns  # noqa: E402

from src.config import Phase2cConfig, config as _runtime_config  # noqa: E402
from src.orchestrator import PonderOrchestrator  # noqa: E402
from src.retrieval.document_retriever import (  # noqa: E402
    DocumentRetriever,
    store_has_documents,
)
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
DEFAULT_DOC_STORE = "data/training/strm_relevance/doc_corpus_store"
DEFAULT_OUT = "data/training/strm_relevance/traces_onyx_doc_ring_hraw.pt"
# MUST match probe_head_to_head_onyx.DEFAULT_LIVE_EVAL_SESSION_IDS exactly so the
# held-out live-eval sessions are the SAME ones D0.4a/D0.4b held out -- never
# train on them.
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

    Mirrors ``generate_onyx_serve_traces._ordered_messages``; stable sort
    preserves the fetcher's insertion order among equal timestamps.
    """
    msgs = list(session.get("messages", []))

    def _key(m: dict) -> str:
        return str(m.get("time_sent") or "")

    return sorted(msgs, key=_key)


def _msg_text(m: dict) -> str:
    t = m.get("text", "")
    if t is None:
        return ""
    return str(t)


def _session_turns(session: dict, min_msg_chars: int) -> tuple[str, list[dict]]:
    """Reduce a session to ordered ``{"role","content"}`` turns (user +
    assistant only, non-empty). Returns (session_id, turns)."""
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
    via ``update``, 1=retrieved via ``inject``). The prefix fallback covers
    slots recorded before Phase 1a or any path that leaves the field None --
    never silently default, so a mis-tagged slot surfaces rather than corrupts
    the type embedding. ``doc_`` / ``__ep`` -> 1 (retrieved); ``#msg`` -> 0.
    """
    if slot_type is not None:
        return int(slot_type)
    if not source_id:
        return 0
    if source_id.startswith("doc_") or "__ep" in source_id:
        return 1
    if "#msg" in source_id:
        return 0
    raise ValueError(f"cannot infer slot_type for source_id={source_id!r}")


def _build_mixed_record(ring, query_emb, ld_head, slot_doc_embs,
                        slot_source_ids, slot_slot_types,
                        emit_question: bool, question: str) -> dict | None:
    """Build ONE fit_relevance-format record from the current MIXED ring.

    Pure tensor assembly (no embedding -- the caller embeds each slot's text
    once and threads provenance + slot_type through). Byte-identical to the 1c
    ``_build_mixed_record`` for the shared fields (same shapes/dtypes the
    downstream loader expects) PLUS ``slot_types`` [K'] long. Drops slots whose
    Phase-A state capture (``slot.h``) is missing or whose text is empty.
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
        "slot_types": slot_types,
    }
    if emit_question:
        rec["question"] = question
    return rec


def _load_doc_store_stats(doc_store_path: Path) -> dict:
    """Read the builder's coverage summary if present (n_docs/n_sections).

    The builder (``build_doc_corpus_store.py``) writes ``build_summary.json``;
    reusing it avoids a re-scan. Returns ``{}`` if absent (the generator still
    probes ``store_has_documents`` live for the findability assert).
    """
    summary = doc_store_path / "build_summary.json"
    if not summary.exists():
        return {}
    try:
        return json.loads(summary.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - best-effort; coverage is probed live too
        return {}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 1f-2: doc-ring (conversation + retrieved-doc) "
                    "training traces from REAL Onyx sessions (task #55).")
    p.add_argument("--sessions", default=DEFAULT_SESSIONS,
                   help="Onyx sessions.jsonl (scripts/_scratch/_fetch_onyx_cookie.py)")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="path to the new backbone (default backbone_v2_full.pt).")
    p.add_argument("--doc-store", default=DEFAULT_DOC_STORE,
                   help="persisted doc corpus store (built by "
                        "build_doc_corpus_store.py).")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--live-eval-sessions", default=",".join(DEFAULT_LIVE_EVAL_SESSION_IDS),
                   help="comma-separated Onyx session UUIDs to HOLD OUT (never "
                        "train). Default = the 2 live-transcript sessions matching "
                        "D0.4a/D0.4b. Empty string disables (NOT recommended).")
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
    p.add_argument("--start-session", type=int, default=0,
                   help="skip sessions with index < this (0-indexed). Generic "
                        "resumability/chunking knob -- e.g. run a long job as "
                        "fresh-process chunks and merge the .pt records afterward "
                        "(each chunk gets a clean WaveDB memory_pool). 0 = start "
                        "at the first session.")
    p.add_argument("--end-session", type=int, default=0,
                   help="stop before sessions with index >= this (0-indexed, "
                        "exclusive; 0 = all). Companion to --start-session.")
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
    doc_store_path = Path(args.doc_store)
    if not doc_store_path.exists():
        print(f"ERROR: doc corpus store not found at {doc_store_path}\n"
              f"  run: python scripts/build_doc_corpus_store.py --store "
              f"{args.doc_store}", file=sys.stderr)
        return 1

    live_eval_ids = frozenset(
        s for s in (args.live_eval_sessions or "").split(",") if s.strip())
    print(f"Loading sessions from {sessions_path}", flush=True)
    sessions = [json.loads(l) for l in sessions_path.read_text(encoding="utf-8")
                .splitlines() if l.strip()]
    print(f"  {len(sessions)} sessions total", flush=True)
    n_held = sum(1 for s in sessions
                 if str(s.get("session_id") or "") in live_eval_ids)
    print(f"  holding out {n_held} live-eval sessions (excluded from train): "
          f"{sorted(live_eval_ids)}", flush=True)

    print(f"Loading backbone from {backbone_path} (device {args.device})",
          flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    print(f"  {sum(p.numel() for p in backbone.parameters()):,} params", flush=True)
    embedder = build_embedder("on-demand")
    ld_head = LatentDynamicsHead()                          # parameter-free project
    emit_question = args.keep_question

    _runtime_config.strm_ring_text = True
    print(f"  strm_ring_text=True (Phase 1a: conversation slots scoreable)",
          flush=True)

    # Open the PERSISTED doc corpus store (built once by build_doc_corpus_store).
    # The store contains the ingested docs; the generator does NOT re-ingest and
    # does NOT seed any conv-pair episodes -> the ONLY type-1 slots are docs.
    from src.memory.store import HippocampalStore  # noqa: E402
    print(f"Opening persisted doc corpus store: {doc_store_path}", flush=True)
    store = HippocampalStore(str(doc_store_path))
    if not store_has_documents(store):
        print(f"ERROR: doc store has no document section edges -- "
              f"store_has_documents(store) is False. Re-run "
              f"build_doc_corpus_store.py (the generator would surface zero "
              f"doc slots -> invalid doc ring).", file=sys.stderr)
        store.close()
        return 1
    stats = _load_doc_store_stats(doc_store_path)
    n_docs_in_store = int(stats.get("n_docs_ingested", 0))
    n_sections_in_store = int(stats.get("n_sections_total", 0))
    print(f"  docs in store: {n_docs_in_store} ({n_sections_in_store} sections)",
          flush=True)

    planner = BonsaiQueryPlanner(endpoint=None, force_rule_based=True)  # offline: deterministic rule-based plans, no (flapping) server dependency
    retriever = HippocampalRetriever(
        store, planner=planner, auto_load_index=True,
        retrieval_gate=None, embedder=embedder,
    )
    # Mirror production: attach the document-aware aggregator so multi-section
    # doc hits surface as ONE slot per doc (the acceptance probe runs under
    # build_ponder which wires this -> the generator MUST match or it is OOD).
    retriever.document_retriever = DocumentRetriever(store)
    print(f"  attached DocumentRetriever (multi-section -> one slot per doc)",
          flush=True)

    # Session state dir lives in a tempdir (NOT the doc store) -- with
    # encoder=None it is effectively unused, but keep it off the persisted store
    # so the generator never writes session state into the doc corpus.
    tmpdir = tempfile.mkdtemp(prefix="pondr_doc_ring_")
    cfg = Phase2cConfig()
    cfg.session.state_dir = str(Path(tmpdir) / "sessions")
    orch = PonderOrchestrator(
        store=store, retriever=retriever, backbone=backbone, embedder=embedder,
        mode_a=_StubModeA(), config=cfg, user_id="pondr_doc_ring",
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
    n_turns_with_doc = 0           # retrieval_coverage numerator
    t0 = time.time()

    for s_idx, session in enumerate(sessions):
        if s_idx < args.start_session:
            continue
        if args.end_session and s_idx >= args.end_session:
            print(f"  hit end_session {args.end_session}, stopping", flush=True)
            break
        if args.max_sessions and n_sessions_kept >= args.max_sessions:
            print(f"  hit max_sessions {args.max_sessions}, stopping", flush=True)
            break
        session_id, turns = _session_turns(session, args.min_msg_chars)
        if not session_id:
            n_sessions_skipped += 1
            continue
        if session_id in live_eval_ids:
            n_sessions_held += 1
            continue  # never train
        pairs = _pair_turns(turns)
        user_turn_count = sum(1 for u, _a in pairs if len(u.strip()) >= args.min_msg_chars)
        if user_turn_count < args.min_user_turns:
            n_sessions_skipped += 1
            continue

        # Per-session WM ring reset (the ring is per-conversation); the doc
        # STORE persists across sessions (the production corpus). NO episode
        # seeding -- the retrieved pool is the pre-ingested docs only.
        orch.working_memory.reset()
        orch.user_id = session_id
        history: list[dict] = []
        # Prime history with turn 0 so the planner has context for query 1
        # (pronoun / implicit-reference resolution). Turn 0 itself is NOT
        # queried (no prior ring to score against); query starts at turn 1.
        u0, a0 = pairs[0]
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
            # the retrieved-doc injects populate text-bearing slots).
            ring = orch.working_memory.ring_buffer()
            if len(ring) < 3:
                n_turns_no_ring += 1
            else:
                prompt_emb = orch.working_memory.embed([u])[0]
                slot_texts = [s.text if s.text is not None else "" for s in ring]
                try:
                    slot_doc_embs = orch.working_memory.embed(slot_texts)
                except Exception as e:  # noqa: BLE001
                    print(f"  [embed-skip] session={session_id} turn={i}: {e}",
                          file=sys.stderr)
                    slot_doc_embs = [None] * len(ring)
                if any(e is None for e in slot_doc_embs):
                    keep_idx = [k for k, e in enumerate(slot_doc_embs)
                                if e is not None]
                    ring = [ring[k] for k in keep_idx]
                    slot_doc_embs = [e for e in slot_doc_embs if e is not None]
                    slot_source_ids = [s.source_id for s in ring]
                    slot_slot_types = [s.slot_type for s in ring]
                else:
                    slot_source_ids = [s.source_id for s in ring]
                    slot_slot_types = [s.slot_type for s in ring]
                # retrieval_coverage: did this turn's ring surface >=1 doc slot?
                has_doc_slot = any(
                    _infer_slot_type(sid, st) == 1
                    for sid, st in zip(slot_source_ids, slot_slot_types))
                if has_doc_slot:
                    n_turns_with_doc += 1
                rec = _build_mixed_record(
                    ring, prompt_emb, ld_head, slot_doc_embs,
                    slot_source_ids, slot_slot_types, emit_question, u)
                if rec is not None:
                    records.append(rec)
                    n_user_turns_total += 1
                    sess_user_turns += 1
            # NO episode ingest here (the 1c conv-pair seeding is removed). The
            # retrieved pool is the pre-ingested docs only. History still grows
            # so the planner has context for the next turn.
            history.append({"role": "user", "content": u})
            history.append({"role": "assistant", "content": a})

        n_sessions_kept += 1
        if (s_idx + 1) % 10 == 0 or s_idx == len(sessions) - 1:
            elapsed = time.time() - t0
            print(f"  [{s_idx + 1}/{len(sessions)}] kept {n_sessions_kept} "
                  f"sessions, {n_user_turns_total} user-turn records, "
                  f"held {n_sessions_held}, skipped {n_sessions_skipped} "
                  f"(no_ring={n_turns_no_ring} qfail={n_turns_query_fail} "
                  f"with_doc={n_turns_with_doc}) ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    retrieval_coverage = (
        n_turns_with_doc / n_user_turns_total if n_user_turns_total else 0.0)
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
    print(f"  turns with doc slot:{n_turns_with_doc} "
          f"(retrieval_coverage={retrieval_coverage:.1%})")
    print(f"docs in store:        {n_docs_in_store} ({n_sections_in_store} sections)")
    if records:
        med_k = int(sorted(r['slots_h_raw'].shape[0] for r in records)[len(records)//2])
        mixed = sum(1 for r in records
                    if int((r['slot_types'] == 0).sum()) > 0
                    and int((r['slot_types'] == 1).sum()) > 0)
        only_conv = sum(1 for r in records if int((r['slot_types'] == 1).sum()) == 0)
        only_ret = sum(1 for r in records if int((r['slot_types'] == 0).sum()) == 0)
        print(f"  median ring K:      {med_k}")
        print(f"  mixed (conv+doc):   {mixed}")
        print(f"  conv-only records:  {only_conv}")
        print(f"  doc-only records:   {only_ret}")
    print(f"elapsed:              {elapsed:.0f}s")
    print("=" * 64)
    if not records:
        print("ERROR: no records built", file=sys.stderr)
        store.close()
        return 1
    if retrieval_coverage == 0.0:
        print("ERROR: retrieval_coverage == 0.0 -- no turn surfaced a doc slot. "
              "The retriever is broken or the doc store has no findable docs; "
              "the doc ring is invalid (not a real fail). Fix the store/"
              "retriever and re-run.", file=sys.stderr)
        store.close()
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {len(records)} records -> {out_path} ({size_mb:.1f} MB)",
          flush=True)
    mixed = sum(1 for r in records
                if int((r['slot_types'] == 0).sum()) > 0
                and int((r['slot_types'] == 1).sum()) > 0)
    summary_path = out_path.with_suffix(".yield.json")
    summary_path.write_text(json.dumps({
        "n_sessions": len(sessions), "n_sessions_kept": n_sessions_kept,
        "n_sessions_held": n_sessions_held, "n_sessions_skipped": n_sessions_skipped,
        "n_records": len(records), "n_turns_no_ring": n_turns_no_ring,
        "n_turns_query_fail": n_turns_query_fail,
        "n_turns_with_doc": n_turns_with_doc,
        "retrieval_coverage": round(retrieval_coverage, 4),
        "mixed_records": mixed,
        "n_docs_in_store": n_docs_in_store,
        "n_sections_in_store": n_sections_in_store,
        "live_eval_session_ids": sorted(live_eval_ids),
        "min_user_turns": args.min_user_turns, "ring_capacity": args.ring_capacity,
        "backbone": str(backbone_path), "sessions_src": str(sessions_path),
        "doc_store": str(doc_store_path),
        "elapsed_s": round(elapsed, 1), "keep_question": emit_question,
        "strm_ring_text": True,
    }, indent=2), encoding="utf-8")
    print(f"wrote yield summary -> {summary_path}", flush=True)

    store.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())