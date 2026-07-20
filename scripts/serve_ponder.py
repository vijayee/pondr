"""Serve the Ponder Engine on the TRAINED SSM/JEPA backbone + RetrievalGate.

Thin CLI over ``src.runtime.build_ponder`` -- the runtime entrypoint the
SSM/JEPA consistency audit (2026-07-15) found missing. Loads the trained Phase
2a backbone + trained Phase 2b gate, wires the real bge embedder + the local
8B Bonsai endpoint, and -- by default -- live-encodes each exchange as a
learnable episode (GLiNER on CUDA with an OOM-safe CPU fallback).

Usage (one-shot, against the running 8B Bonsai server):
    python scripts/serve_ponder.py --query "Why did we choose Postgres?"

Usage (interactive REPL; blank line or Ctrl-C to exit):
    python scripts/serve_ponder.py

Flags:
    --backbone PATH        Phase 2a backbone checkpoint
    --gate PATH            Phase 2b RetrievalGate checkpoint (best.pt)
    --db PATH              WaveDB memory DB path
    --embed-source NAME    on-demand (real bge) | stub (shape-only smoke)
    --device NAME          backbone+gate device (auto|cpu|cuda)
    --gliner-device NAME   GLiNER device (auto|cpu|cuda); OOM-safe fallback
    --gliner-timing        log per-stage GLiNER extraction timing (DEFAULT ON;
                          --no-gliner-timing to disable)
    --no-live-encode       do not persist exchanges (skip the encoder + GLiNER)
    --user-id ID           user the encoder attributes episodes to
    --query TEXT           one-shot query; omit for the interactive REPL
    --bonsai-endpoint URL  override the Bonsai LLM endpoint
    --async-distill        background episode distillation (response returns
                          immediately; extraction fills graph edges in the gaps
                          between turns). DEFAULT ON; --no-async-distill to
                          disable (synchronous encode).
    --bonsai-isolation     10-pass isolated per-class Bonsai extractor (has_state
                          0 -> 11/13 zero-shot, ~22.8 s/doc); DEFAULT ON and
                          viable because --async-distill is also on. --no-bonsai-
                          isolation for the V1 single-pass extractor.
    --strm-relevance-head PATH  optional STRM Phase 2a relevance-head checkpoint
                          (best.pt from scripts/train_relevance_head.py). When
                          set, the head is loaded + attached to the orchestrator
                          (it scores each WM ring slot's relevance to the query;
                          Phase 3's context-builder consumes r_i). Default off.
    --strm-relevance-logging   append the raw per-unit rating to the STRM 2a
                          feedback.jsonl tap (data/training/strm_relevance/).
                          DEFAULT OFF; the tap is side-effect-only and the
                          labels only matter once a 2a head is in training.
    --strm-graduation-proxy  attach the STRM Phase 2d v1 graduation proxy (the
                          parameter-free integral(r_i dt) heuristic the v2 head
                          must beat). No checkpoint. DEFAULT OFF (byte-identical
                          to pre-2d). Full graduation -> LTM promotion is Phase 4.
    --strm-graduation-logging append per-turn ring-slot state to the STRM 2d
                          replay.jsonl tap (data/training/strm_graduation/) so
                          the v2 graduation labels accumulate. DEFAULT OFF.
    --strm-ring-capacity N  WM ring buffer capacity K (default 0 = OFF). The
                          STRM relevance head + graduation replay logger need
                          the ring ON to populate per-slot state; pass K>0
                          (e.g. 16) when running --strm-relevance-head or
                          --strm-graduation-logging, or those heads/loggers see
                          an empty ring and produce nothing.

Pre-warm the Bonsai server first (PTX-JIT cold-start ~18s/shape; see memory
``hippo-bonsai-local-server``). Live-encode (default on) loads GLiNER; the
first exchange also pays the GLiNER model-download/warm-up cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import DEFAULT_BACKBONE_PATH, DEFAULT_GATE_PATH, build_ponder  # noqa: E402
from src.config import config as _config  # noqa: E402


def _print_result(res: dict) -> None:
    """Print the query result + the self-chat loop transcript + persistence."""
    end_state = res.get("end_state_plan")
    end_state_name = getattr(end_state, "end_state", "?") if end_state else "?"
    print(f"\n[end-state] {end_state_name}")
    response = res.get("response") or ""
    # Model output may be UTF-8; stdout is reconfigured to utf-8 in main().
    print(f"\n{response}")
    fb = res.get("feedback_collected")
    if fb:
        print(f"\n[feedback_collected] {fb}", file=sys.stderr)
    # Self-chat tool-loop transcript (surfaced only when the loop ran).
    if "loop_tool_messages" in res:
        names = [c.get("name") for c in res.get("loop_collected", [])]
        print(f"[loop] exhausted={res.get('loop_exhausted')} tools={names}",
              file=sys.stderr)
    pid = res.get("persisted_episode_id")
    if pid:
        print(f"[live-encode] persisted episode {pid}", file=sys.stderr)
    # Per-stage GLiNER timing is already logged to stderr by the extractor when
    # --gliner-timing is on (the flag is passed through to build_ponder), so
    # nothing extra is printed here.


def main() -> int:
    # Model output is UTF-8; the Windows console default (cp1252) would crash
    # on non-ASCII. Reconfigure stdout/stderr to UTF-8 with replacement so a
    # live Bonsai response never kills the REPL. Our own strings stay ASCII.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover - non-CPython / fixed stream
        pass

    p = argparse.ArgumentParser(description="Serve the Ponder Engine on the trained models")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE_PATH,
                   help="Phase 2a backbone checkpoint (backbone_final.pt)")
    p.add_argument("--gate", default=DEFAULT_GATE_PATH,
                   help="Phase 2b RetrievalGate checkpoint (best.pt)")
    p.add_argument("--db", default=None,
                   help="WaveDB memory DB path (default: config.db_path)")
    p.add_argument("--embed-source", default="on-demand", choices=["on-demand", "stub"],
                   help="on-demand = real bge-small; stub = shape-only smoke")
    p.add_argument("--device", default="auto", help="backbone+gate device: auto|cpu|cuda")
    p.add_argument("--gliner-device", default="auto",
                   help="GLiNER device: auto|cpu|cuda (OOM-safe CPU fallback)")
    p.add_argument("--gliner-timing", action=argparse.BooleanOptionalAction, default=True,
                   help="log per-stage GLiNER extraction timing to stderr (default on; "
                        "use --no-gliner-timing to disable)")
    p.add_argument("--no-live-encode", action="store_true",
                   help="do not persist exchanges (skip the encoder + GLiNER)")
    p.add_argument("--user-id", default="ponder", help="user the encoder attributes episodes to")
    p.add_argument("--query", default=None, help="one-shot query; omit for the interactive REPL")
    p.add_argument("--bonsai-endpoint", default=None, help="override the Bonsai LLM endpoint")
    p.add_argument("--async-distill", action=argparse.BooleanOptionalAction, default=True,
                   help="background episode distillation: the response returns immediately "
                        "while GLiNER + Bonsai extraction fills the graph edges on a "
                        "single-worker FIFO in the gaps between turns (Phase 3c). "
                        "DEFAULT ON; use --no-async-distill for synchronous encode.")
    p.add_argument("--bonsai-isolation", action=argparse.BooleanOptionalAction, default=True,
                   help="10-pass isolated per-class Bonsai extractor (lifts strict "
                        "has_state 0 -> 11/13 zero-shot) at ~22.8 s/doc. DEFAULT ON; "
                        "viable because --async-distill is also on (async hides the "
                        "22.8 s). Use --no-bonsai-isolation for the V1 single-pass "
                        "extractor. Do NOT pass --no-async-distill without also passing "
                        "--no-bonsai-isolation, or the response blocks ~22s/turn.")
    p.add_argument("--strm-relevance-head", default=None,
                   help="optional STRM Phase 2a relevance-head checkpoint (best.pt). "
                        "When set, the head is loaded + attached to the orchestrator. "
                        "Default off (no relevance scoring at serve).")
    p.add_argument("--strm-relevance-logging", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="append the raw per-unit rating to the STRM 2a feedback.jsonl "
                        "tap. DEFAULT OFF (side-effect-only; labels only matter once a "
                        "2a head is in training).")
    p.add_argument("--strm-graduation-proxy", action="store_true",
                   default=False,
                   help="attach the STRM Phase 2d v1 graduation proxy (the "
                        "parameter-free integral(r_i dt) heuristic the v2 head must "
                        "beat). No checkpoint. DEFAULT OFF (byte-identical to pre-2d).")
    p.add_argument("--strm-graduation-logging", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="append per-turn ring-slot state to the STRM 2d replay.jsonl "
                        "tap so the v2 graduation labels accumulate. DEFAULT OFF.")
    p.add_argument("--strm-ring-capacity", type=int, default=0,
                   help="WM ring buffer capacity K (default 0 = OFF). The STRM "
                        "relevance head + graduation replay logger need the ring "
                        "ON to populate per-slot state; pass K>0 (e.g. 16) when "
                        "running --strm-relevance-head or --strm-graduation-logging.")
    p.add_argument("--strm-context-builder", default=None,
                   help="optional STRM Phase 3 context-builder checkpoint (best.pt, "
                        "the learned PresentationGate selector/reranker). When set, "
                        "the orchestrator attends over the WM ring with the 2a r_i "
                        "as a bias and selects top-m primary context instead of the "
                        "heuristic PresentationGate. Requires --strm-relevance-head "
                        "AND --strm-ring-capacity > 0; otherwise warns + falls back "
                        "to the heuristic. DEFAULT OFF (byte-identical to pre-3).")
    p.add_argument("--strm-recoverability-head", default=None,
                   help="optional STRM Phase 2b recoverability-head checkpoint "
                        "(best.pt from scripts/train_recoverability_head.py). When "
                        "set, the head is loaded + attached; Phase 4's salience "
                        "trigger consumes it as the 'recoverability < theta' term "
                        "(low = likely forgotten = salient). Requires the ring ON. "
                        "DEFAULT OFF (byte-identical to pre-Phase-4).")
    p.add_argument("--strm-latent-dynamics-head", default=None,
                   help="optional STRM Phase 2c latent-dynamics-head checkpoint "
                        "(best.pt from scripts/train_latent_dynamics_head.py -- the "
                        "linear z_{t+1}=A z_t+b next-state predictor). When set, the "
                        "head is loaded + attached; Phase 4's salience trigger "
                        "consumes its surprise() as the 'surprise < surprise_cap' "
                        "term (high surprise -> suppress). Requires the ring ON. "
                        "DEFAULT OFF (byte-identical to pre-Phase-4).")
    p.add_argument("--strm-graduation-head", default=None,
                   help="optional STRM Phase 2d v2 graduation-head checkpoint "
                        "(best.pt from scripts/train_graduation_head.py -- the "
                        "learned later_needed classifier the v1 proxy is the "
                        "baseline for). When set, the head is loaded + attached "
                        "(completes the full serve-wiring of all STRM read-out "
                        "heads). Phase 4's LTM-promotion path consumes the "
                        "decision. DEFAULT OFF (byte-identical to pre-Phase-4).")
    args = p.parse_args()

    # The orchestrator reads these two flags off the global config singleton at
    # __init__ (not the per-instance cfg), so set them BEFORE build_ponder. Both
    # default ON (Phase 1c-3c hardening) -> async distill + the 10-pass extractor
    # are the production path; --no-async-distill / --no-bonsai-isolation opt out.
    _config.async_distill_enabled = args.async_distill
    _config.bonsai_isolation_extraction = args.bonsai_isolation
    _config.strm_relevance_logging = args.strm_relevance_logging
    _config.strm_graduation_logging = args.strm_graduation_logging
    if args.bonsai_isolation and not args.async_distill:
        print("WARNING: --bonsai-isolation without --async-distill will block the "
              "response ~22.8 s/turn (10 Bonsai calls on the sync path). Enable "
              "--async-distill too (or pass --no-bonsai-isolation).", file=sys.stderr)
    if (args.strm_relevance_head or args.strm_relevance_logging
            or args.strm_graduation_logging) and args.strm_ring_capacity <= 0:
        print("WARNING: --strm-relevance-head / --strm-relevance-logging / "
              "--strm-graduation-logging need the WM ring ON to populate per-slot "
              "state, but --strm-ring-capacity is 0 (ring OFF). Pass "
              "--strm-ring-capacity 16 (or similar), or the relevance head scores "
              "an empty ring and the replay logger writes nothing.",
              file=sys.stderr)
    # STRM Phase 4 read-out heads (2b recoverability, 2c latent-dynamics, 2d v2
    # graduation) all score per-slot WM state at serve, so they need the ring
    # ON. Warn on a missing ring (a warning, not a hard error -- the orchestrator
    # stores the head inert when the ring is off, byte-identical to flag-off).
    if (args.strm_recoverability_head or args.strm_latent_dynamics_head
            or args.strm_graduation_head) and args.strm_ring_capacity <= 0:
        print("WARNING: --strm-recoverability-head / --strm-latent-dynamics-head / "
              "--strm-graduation-head need the WM ring ON (--strm-ring-capacity > "
              "0) to score per-slot state, but the ring is OFF. The heads load "
              "but stay inert at serve this round (they are attached only; the "
              "salience trigger that reads them is a later Phase 4 step). Pass "
              "--strm-ring-capacity 16.", file=sys.stderr)
    # STRM Phase 3 context-builder: requires the ring ON + a 2a relevance head
    # (the builder's r_i bias comes from it). Warn on either missing; the
    # orchestrator also falls back to the heuristic PresentationGate at runtime,
    # so this is a warning, not a hard error (matches the 2a/2d flag style).
    if args.strm_context_builder:
        if args.strm_ring_capacity <= 0:
            print("WARNING: --strm-context-builder needs the WM ring ON "
                  "(--strm-ring-capacity > 0) to attend over ring slots, but "
                  "the ring is OFF. The orchestrator will fall back to the "
                  "heuristic PresentationGate. Pass --strm-ring-capacity 16.",
                  file=sys.stderr)
        if not args.strm_relevance_head:
            print("WARNING: --strm-context-builder needs --strm-relevance-head "
                  "(its r_i bias comes from the 2a relevance head), but no "
                  "relevance head is set. The orchestrator will fall back to "
                  "the heuristic PresentationGate. Pass --strm-relevance-head "
                  "PATH.", file=sys.stderr)

    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"ERROR: backbone checkpoint not found at {backbone_path}", file=sys.stderr)
        return 1
    gate_path = Path(args.gate)
    if not gate_path.exists():
        print(f"ERROR: gate checkpoint not found at {gate_path}", file=sys.stderr)
        return 1
    relevance_head_path = None
    if args.strm_relevance_head:
        relevance_head_path = Path(args.strm_relevance_head)
        if not relevance_head_path.exists():
            print(f"ERROR: STRM relevance-head checkpoint not found at "
                  f"{relevance_head_path}", file=sys.stderr)
            return 1
        relevance_head_path = str(relevance_head_path)
    context_builder_path = None
    if args.strm_context_builder:
        context_builder_path = Path(args.strm_context_builder)
        if not context_builder_path.exists():
            print(f"ERROR: STRM context-builder checkpoint not found at "
                  f"{context_builder_path}", file=sys.stderr)
            return 1
        context_builder_path = str(context_builder_path)
    recoverability_head_path = None
    if args.strm_recoverability_head:
        recoverability_head_path = Path(args.strm_recoverability_head)
        if not recoverability_head_path.exists():
            print(f"ERROR: STRM recoverability-head checkpoint not found at "
                  f"{recoverability_head_path}", file=sys.stderr)
            return 1
        recoverability_head_path = str(recoverability_head_path)
    latent_dynamics_head_path = None
    if args.strm_latent_dynamics_head:
        latent_dynamics_head_path = Path(args.strm_latent_dynamics_head)
        if not latent_dynamics_head_path.exists():
            print(f"ERROR: STRM latent-dynamics-head checkpoint not found at "
                  f"{latent_dynamics_head_path}", file=sys.stderr)
            return 1
        latent_dynamics_head_path = str(latent_dynamics_head_path)
    graduation_head_path = None
    if args.strm_graduation_head:
        graduation_head_path = Path(args.strm_graduation_head)
        if not graduation_head_path.exists():
            print(f"ERROR: STRM graduation-head (v2) checkpoint not found at "
                  f"{graduation_head_path}", file=sys.stderr)
            return 1
        graduation_head_path = str(graduation_head_path)

    print(f"[load] backbone={backbone_path}", file=sys.stderr)
    print(f"[load] gate={gate_path}", file=sys.stderr)
    print(f"[load] live_encode={not args.no_live_encode} "
          f"gliner_device={args.gliner_device} gliner_timing={args.gliner_timing}",
          file=sys.stderr)
    print(f"[load] async_distill={args.async_distill} "
          f"bonsai_isolation={args.bonsai_isolation}", file=sys.stderr)
    print(f"[load] strm_relevance_head={relevance_head_path or '(off)'} "
          f"strm_relevance_logging={args.strm_relevance_logging} "
          f"strm_graduation_proxy={args.strm_graduation_proxy} "
          f"strm_graduation_logging={args.strm_graduation_logging} "
          f"strm_ring_capacity={args.strm_ring_capacity} "
          f"strm_context_builder={context_builder_path or '(off)'} "
          f"strm_recoverability_head={recoverability_head_path or '(off)'} "
          f"strm_latent_dynamics_head={latent_dynamics_head_path or '(off)'} "
          f"strm_graduation_head={graduation_head_path or '(off)'}", file=sys.stderr)

    orch = build_ponder(
        args.db,
        backbone_path=str(backbone_path),
        gate_path=str(gate_path),
        embedder_source=args.embed_source,
        bonsai_endpoint=args.bonsai_endpoint,
        device=args.device,
        gliner_device=args.gliner_device,
        gliner_timing=args.gliner_timing,
        live_encode=not args.no_live_encode,
        user_id=args.user_id,
        relevance_head_path=relevance_head_path,
        graduation_proxy=args.strm_graduation_proxy,
        graduation_head_path=graduation_head_path,
        recoverability_head_path=recoverability_head_path,
        latent_dynamics_head_path=latent_dynamics_head_path,
        ring_capacity=args.strm_ring_capacity,
        context_builder_path=context_builder_path,
    )

    try:
        if args.query is not None:
            res = orch.query(args.query)
            _print_result(res)
            return 0

        # Interactive REPL. The orchestrator's Working Memory carries cross-query
        # state; we also thread the raw turn pairs as conversation_history so the
        # planner can resolve pronouns across turns.
        print("Ponder REPL. Blank line to exit.", file=sys.stderr)
        history: list[dict] = []
        while True:
            try:
                line = input("you> ")
            except EOFError:
                break
            if not line.strip():
                break
            res = orch.query(line, conversation_history=list(history))
            _print_result(res)
            response = res.get("response")
            if isinstance(response, str) and response.strip():
                history.append({"role": "user", "content": line})
                history.append({"role": "assistant", "content": response})
        return 0
    finally:
        # Drain the background distill worker before the store closes so any
        # queued stub episodes get their graph edges filled while WaveDB is
        # still writable. No-op when async_distill_enabled is off (drain()
        # returns immediately if there is no worker). Best-effort: a drain
        # failure must not block store close.
        #
        # One-shot (--query): wait long enough for the single in-flight fill to
        # complete so the episode is fully encoded when the process exits (the
        # ~22.8 s isolated fill needs a >22 s budget). REPL: a snappy exit
        # matters more than flushing the last fill -- the stub is already
        # persisted, so a short budget abandons only in-flight extraction.
        drain_timeout = 45.0 if args.query is not None else 8.0
        try:
            if args.query is not None and args.async_distill:
                print(f"[drain] waiting up to {drain_timeout:.0f}s for the "
                      f"background fill to finish...", file=sys.stderr)
            orch.drain(timeout=drain_timeout)
        except Exception as e:  # noqa: BLE001 - never crash on cleanup
            print(f"[drain-fail] {e}", file=sys.stderr)
        try:
            orch.store.close()
        except Exception as e:  # noqa: BLE001 - never crash on cleanup
            print(f"[close-fail] {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())