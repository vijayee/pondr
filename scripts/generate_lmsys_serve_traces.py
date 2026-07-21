"""Task #42: stream ``lmsys/lmsys-chat-1m`` -> SERVE-distribution serve traces
for z-head regularization (supplement data; user 2026-07-21).

Task #41 ([[pondr-strm-task41-serve-zrgate-saturation]]) found the flat-readout
CompositeZHead FAILS the SERVE z_r gate (saturation) AND does not robustly clear
the z_logit gate held-out -- 934K-2.5M params on ~91 train turns overfits, and
~23 val turns make the per-source z_logit gap median noisy. The cheap decisive
test is MORE serve-like data + regularization, then RE-GATE on real Onyx (the
user's chosen framing: lmsys is a SUPPLEMENT, not a new target distribution).
This script builds the supplement.

Mapping (one conversation = one serve session):
  * Each PRIOR message (user OR assistant) is ingested as a ring slot with
    ``source_id = "{conv_id}#{msg_idx}"`` via ``WorkingMemory.step`` (the SAME
    identity-instance direct-SSM path the Onyx serve traces used
    [[pondr-strm-task33-gate-train-go-serve-fail]]). The WM ring persists across
    the conversation, so a message ingested at index ``j`` stays in the ring and
    is a candidate slot for every LATER user turn -> it recurses as a candidate,
    exactly the structure the per-source z_r/z_logit gap needs.
  * At each USER turn ``i`` (``i >= 1`` and the ring has ``>= 3`` text-bearing
    slots), capture ONE ``fit_relevance``-format record: ``query_emb`` =
    bge(user msg ``i``) scored against the ring of all prior messages; ``labels``
    = top-1-cos (the prior msg most bge-similar to this query -- the SAME probe/
    filler label signal ``_build_serve_trace`` uses); ``cos`` = the continuous
    bge cosine per slot; ``slots_h_raw`` flat_last [6144]; ``slots_z`` =
    parameter-free ``LatentDynamicsHead.project(slot.h)`` [384]; ``slots_y`` =
    slot.y [256]; ``slots_doc_emb`` = the bge of each slot's text [384];
    ``source_ids`` = the ring slots' source_ids. Structurally IDENTICAL to
    ``scripts/probe_strm_selectivity_real.py::_build_serve_trace`` output, so
    train-on-lmsys / eval-Onyx is apples-to-apples.

Filtering: keep only English conversations with ``>= --min-user-turns`` user
turns (the long-conversation tail; the avg is 2.0 turns/conv so most rows are
single-turn and useless for the per-source gap, which needs a source to recur
>= 3 times -> the earliest messages need >= 3 LATER user turns -> min ~4 user
turns). Messages shorter than ``--min-msg-chars`` are skipped (empty assistant
replies embed to garbage). ``--max-convs`` caps the slice for a cheap local
probe; ``--stream-limit`` caps the rows scanned.

Isolation (the binding constraint): standalone script. Loads the new backbone
via ``--backbone`` (default ``backbone_v2_full.pt``); NEVER touches
``DEFAULT_BACKBONE_PATH`` / ``build_ponder`` / ``serve_ponder`` / the live
orchestrator. No HF upload (diagnostic supplement data, gitignored under
``data/``). Output records carry no PII (lmsys already redacts names to
NAME_N; we keep only bge embeddings + source_ids + the raw SSM state, not
message text -- ``question`` is retained for debugging but can be dropped with
``--no-question``).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from src.subconscious.configs import BackboneConfig  # noqa: E402
from src.subconscious.latent_dynamics_head import LatentDynamicsHead  # noqa: E402
from src.subconscious.training.routing_training import (  # noqa: E402
    _resolve_device,
    build_embedder,
    load_backbone,
)
from src.subconscious.working_memory import WorkingMemory  # noqa: E402

D_STATE = 16
D_MODEL = 384
DEFAULT_BACKBONE = "data/training/strm_backbone_relevance/backbone_v2_full.pt"
DEFAULT_OUT = "data/training/strm_relevance/traces_lmsys_serve_hraw.pt"


def _msg_text(m: dict) -> str:
    """lmsys stores content as a str or (sometimes) a list; coerce to str."""
    c = m.get("content", "")
    if isinstance(c, list):
        # list of parts (e.g. OpenAI multimodal) -> join string parts
        parts = []
        for part in c:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", part.get("content", ""))))
            else:
                parts.append(str(part))
        return " ".join(parts)
    return str(c) if c is not None else ""


def _conv_messages(row: dict) -> list[dict]:
    """Extract the OpenAI-format message list from a row (column name varies
    across dataset revisions: ``conversation`` / ``conversations``)."""
    for key in ("conversation", "conversations", "messages"):
        conv = row.get(key)
        if conv is not None:
            return list(conv)
    return []


def _is_lang(row: dict, target_lower: str) -> bool:
    """True if the row's detected language matches ``target_lower`` (e.g.
    'english'). The column name varies across dataset revisions."""
    lang = row.get("language") or row.get("detected_language") or ""
    return isinstance(lang, str) and lang.lower() == target_lower


def _build_record(ring, query_emb, ld_head, source_ids, doc_embs,
                  question: str, emit_question: bool) -> dict | None:
    """Build ONE fit_relevance-format record from the current ring + query.
    Mirrors ``probe_strm_selectivity_real._build_serve_trace`` field-for-field
    so the downstream probe can mix lmsys + Onyx traces freely."""
    K = len(ring)
    if K < 3:
        return None
    # Drop any slot whose Phase-A state capture (slot.h) is missing -- a None
    # here would be a Phase A regression; skip the slot rather than crash.
    kept = [(j, s) for j, s in enumerate(ring) if getattr(s, "h", None) is not None]
    if len(kept) < 3:
        return None
    # bge cosine between each kept slot's text-emb and the query -> label + cos.
    # All on CPU (tiny K x 384 math); query may arrive on the backbone device.
    q = query_emb.to("cpu").to(torch.float32).reshape(-1)
    qn = q / (q.norm() + 1e-9)
    cos_vals = torch.empty(len(kept), dtype=torch.float32)
    for k, (j, _s) in enumerate(kept):
        d = doc_embs[j].to(torch.float32).reshape(-1)
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
        doc_embs[j].to(torch.float32).squeeze(0).reshape(-1)
        for j, _s in kept
    ])                                                        # [K', 384]
    # Raw per-layer per-channel state [K',4,16,384] fp16 -- SAME format as
    # probe_strm_selectivity_real._build_serve_trace emits, so the downstream
    # probe loads lmsys + Onyx traces identically (it flattens the last layer
    # itself). slot.h is a list of 4 per-layer tensors; reshape(16,384) handles
    # both [1,16,384] and [16,384] storage (6144 elements either way).
    slots_h_raw = torch.stack([
        torch.stack([
            layer.detach().to("cpu").to(torch.float16).reshape(D_STATE, D_MODEL)
            for layer in s.h
        ])
        for _j, s in kept
    ])                                                        # [K',4,16,384] fp16
    rec = {
        "query_emb": q, "slots_y": slots_y, "slots_z": slots_z,
        "labels": labels, "source_ids": [source_ids[j] for j, _s in kept],
        "cos": cos_vals, "slots_doc_emb": slots_doc_emb,
        "slots_h_raw": slots_h_raw,
    }
    if emit_question:
        rec["question"] = question
    return rec


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stream lmsys-chat-1m -> SERVE-distribution serve traces "
                    "(supplement data for z-head regularization, task #42).")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="path to the new backbone (default backbone_v2_full.pt).")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--max-convs", type=int, default=2000,
                   help="cap on qualifying conversations emitted (slice).")
    p.add_argument("--stream-limit", type=int, default=200000,
                   help="cap on rows scanned from the stream (safety).")
    p.add_argument("--min-user-turns", type=int, default=4,
                   help="min user turns to keep a conversation (the long-conv "
                        "tail; a source needs >=3 later user turns to recur).")
    p.add_argument("--min-msg-chars", type=int, default=5,
                   help="skip messages shorter than this (empty replies).")
    p.add_argument("--ring-capacity", type=int, default=32,
                   help="STRM ring capacity; older messages FIFO-evict in long convs.")
    p.add_argument("--language", default="English",
                   help="language filter value (case-insensitive).")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-question", action="store_true",
                   help="drop the 'question' field (no message text on disk).")
    args = p.parse_args()

    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"ERROR: backbone not found at {backbone_path}", file=sys.stderr)
        return 1
    dev = _resolve_device(args.device)
    lang_lower = args.language.lower()

    print(f"Loading backbone from {backbone_path} (device {dev})", flush=True)
    backbone = load_backbone(str(backbone_path), BackboneConfig(), device=args.device)
    print(f"  {sum(p.numel() for p in backbone.parameters()):,} params", flush=True)
    embedder = build_embedder("on-demand")
    wm = WorkingMemory(backbone, embedder=embedder,
                       ring_capacity=args.ring_capacity, identity_instance=True)
    ld_head = LatentDynamicsHead()                          # parameter-free project
    emit_question = not args.no_question

    print(f"Streaming lmsys/lmsys-chat-1m (language={args.language}, "
          f"min_user_turns={args.min_user_turns}, max_convs={args.max_convs}, "
          f"stream_limit={args.stream_limit})", flush=True)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    records: list[dict] = []
    n_scanned = 0
    n_kept_conv = 0
    n_user_turns_total = 0
    skip_reasons = {"not_en": 0, "too_short": 0, "no_conv": 0}
    t0 = time.time()
    first_keys = None

    for row in ds:
        if first_keys is None:
            first_keys = list(row.keys())
        n_scanned += 1
        if n_scanned > args.stream_limit:
            print(f"  hit stream_limit {args.stream_limit}, stopping scan",
                  flush=True)
            break
        msgs = _conv_messages(row)
        if not msgs:
            skip_reasons["no_conv"] += 1
            continue
        if not _is_lang(row, lang_lower):
            skip_reasons["not_en"] += 1
            continue
        # count user turns with enough content
        user_idxs = [i for i, m in enumerate(msgs)
                     if (m.get("role") == "user"
                         and len(_msg_text(m).strip()) >= args.min_msg_chars)]
        if len(user_idxs) < args.min_user_turns:
            skip_reasons["too_short"] += 1
            continue

        conv_id = row.get("conversation_id") or f"conv{n_kept_conv}"
        # Ingest messages in order; capture a record at each user turn (query)
        # BEFORE ingesting that user msg, so the query is not its own slot.
        wm.reset()
        source_ids: list[str] = []
        doc_embs: list[torch.Tensor] = []
        # Pre-embed all messages once (cache by index).
        msg_texts = [_msg_text(m) for m in msgs]
        # embed in one batch per conversation (small, <= ~170 msgs).
        embeddable = [t if len(t.strip()) >= args.min_msg_chars else ""
                      for t in msg_texts]
        # embed all (empty strings -> bge returns a vector; we just won't ingest them)
        try:
            embs = wm.embed(embeddable)
        except Exception as e:  # embedding failure on a weird row -> skip conv
            skip_reasons["no_conv"] += 1
            continue

        for i, m in enumerate(msgs):
            role = m.get("role", "?")
            text = msg_texts[i]
            # capture at user turns (query) against the ring so far
            if role == "user" and len(text.strip()) >= args.min_msg_chars:
                if len(source_ids) >= 3:
                    rec = _build_record(
                        wm.ring_buffer(), embs[i], ld_head,
                        source_ids, doc_embs, text, emit_question)
                    if rec is not None:
                        records.append(rec)
                        n_user_turns_total += 1
            # ingest this message as a slot for future turns (skip empties)
            if len(text.strip()) >= args.min_msg_chars:
                v = embs[i]
                sid = f"{conv_id}#{i}"
                wm.step(v, source_id=sid, text=text)
                source_ids.append(sid)
                doc_embs.append(v.detach().to("cpu"))

        n_kept_conv += 1
        if n_kept_conv % 100 == 0:
            elapsed = time.time() - t0
            print(f"  scanned {n_scanned}, kept {n_kept_conv} convs, "
                  f"{n_user_turns_total} user-turn records ({elapsed:.0f}s)",
                  flush=True)
        if n_kept_conv >= args.max_convs:
            print(f"  hit max_convs {args.max_convs}, stopping", flush=True)
            break

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"rows scanned:   {n_scanned}")
    print(f"columns:        {first_keys}")
    print(f"skip reasons:   {skip_reasons}")
    print(f"convs kept:     {n_kept_conv}")
    print(f"user-turn recs: {n_user_turns_total} (median K "
          f"{int(sorted((r['slots_h_raw'].shape[0] for r in records))[len(records)//2]) if records else 0})")
    print(f"elapsed:        {elapsed:.0f}s")
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
    # yield summary for the transfer probe's planning
    summary_path = out_path.with_suffix(".yield.json")
    import json
    summary_path.write_text(json.dumps({
        "n_scanned": n_scanned, "n_kept_conv": n_kept_conv,
        "n_records": len(records),
        "skip_reasons": skip_reasons,
        "min_user_turns": args.min_user_turns,
        "ring_capacity": args.ring_capacity,
        "backbone": str(backbone_path),
        "elapsed_s": round(elapsed, 1),
        "columns": first_keys,
    }, indent=2), encoding="utf-8")
    print(f"wrote yield summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())